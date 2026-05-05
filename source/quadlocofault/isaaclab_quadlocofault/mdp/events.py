

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal
import omni.usd
from isaaclab.assets import RigidObject,Articulation, AssetBase
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.actuators import DCMotor
from isaaclab_quadlocofault.actuators import CustomDCMotor
from isaaclab.sensors import RayCasterCamera
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import  ManagerBasedEnv
    from isaaclab.managers import EventTermCfg

####################################
####################################

def set_actuator_faults(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    joint: str | int | tuple,
    ratio: float,    
):
    asset: Articulation = env.scene[asset_cfg.name]
    # assert ((joint in asset.joint_names) or (joint < 12)), "must be a correct joint"
    # ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 
    # 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 
    # 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 
    # 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
    if isinstance(joint, str):
        joint = asset.joint_names.index(joint)
    for actuator in asset.actuators.values():
        # breakpoint()
        asset.faulty_joint_idx[env_ids] = joint
        
        asset.motors_strength[env_ids] = asset.default_motors_strength[env_ids].clone()
        asset.motors_strength[env_ids, joint] = ratio

        actuator.stiffness[env_ids] = (asset.data.default_joint_stiffness * asset.motors_strength)[env_ids].clone()
        actuator.damping[env_ids] = (asset.data.default_joint_damping * asset.motors_strength)[env_ids].clone()

def randomize_actuator_faults(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    ratio: float = 0.3,
    failure_range: tuple[float,float] = (0.3,0.8),
    num_faults: int = 1,
):
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    lb, ub = failure_range
    for actuator in asset.actuators.values():
        # breakpoint()
        size = len(asset.joint_names)
        N = env_ids.shape[0]
        is_severe = torch.bernoulli(ratio * torch.ones((N,))).to(device=asset.device).unsqueeze(1)
        u1 = torch.rand((N,num_faults), device=asset.device) * lb # severe failure
        u2 = torch.rand((N,num_faults), device=asset.device) * (ub - lb) + lb # moderate failure
        failure_coef = is_severe*u1 + (1-is_severe)*u2
        faulty_joint_idx = torch.randint(low=0, high=size, size=(N,num_faults), dtype=torch.long, device=asset.device)
        # if (asset.faulty_joint_idx[env_ids]).sum() > 0:
        #     breakpoint()
        asset.faulty_joint_idx[env_ids] = torch.zeros((env_ids.shape[0],len(asset.joint_names)), dtype=torch.long, device=asset.device)
        asset.faulty_joint_idx[env_ids[:,None], faulty_joint_idx] = 1
        # breakpoint()
        asset.motors_strength[env_ids] = asset.default_motors_strength[env_ids].clone()
        asset.motors_strength[env_ids[:,None],faulty_joint_idx] = failure_coef

        actuator.stiffness[env_ids] = (asset.data.default_joint_stiffness * asset.motors_strength)[env_ids].clone()
        actuator.damping[env_ids] = (asset.data.default_joint_damping * asset.motors_strength)[env_ids].clone()

def reset_actuator_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    motors_strength_range: tuple[float, float] = (0.9, 1.1),
):
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    for actuator in asset.actuators.values():
        # if not hasattr(asset, "default_motors_strength"): # create default config if first call
        low, high = motors_strength_range
        asset.default_motors_strength = torch.rand((env.scene.num_envs, len(asset.joint_names)), device=asset.device) * (high - low) + low
        actuator.stiffness[env_ids] = (asset.data.default_joint_stiffness * asset.default_motors_strength)[env_ids].clone()
        actuator.damping[env_ids] = (asset.data.default_joint_damping * asset.default_motors_strength)[env_ids].clone()
        # breakpoint()
        if hasattr(asset, "motors_strength"): # if created before, only reset the reseted envs
            asset.motors_strength[env_ids] = asset.default_motors_strength[env_ids].clone()
        else:
            asset.motors_strength = asset.default_motors_strength.clone()
        
        # if hasattr(asset, "faulty_joint_idx"): # reset fault idx
        #     asset.faulty_joint_idx[env_ids] = (-1)*torch.ones((env_ids.shape[0],), dtype=torch.long, device=asset.device)
        # else: # initialize fault idx
        #     asset.faulty_joint_idx = (-1)*torch.ones((env.scene.num_envs,), dtype=torch.long, device=asset.device)
        
        if hasattr(asset, "faulty_joint_idx"): # reset fault idx
            asset.faulty_joint_idx[env_ids] = torch.zeros((env_ids.shape[0],len(asset.joint_names)), dtype=torch.long, device=asset.device)
        else: # initialize fault idx
            asset.faulty_joint_idx = torch.zeros_like(asset.default_motors_strength, dtype=torch.long, device=asset.device)

