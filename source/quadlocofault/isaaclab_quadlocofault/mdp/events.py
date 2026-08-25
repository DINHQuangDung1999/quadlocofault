

from __future__ import annotations

from collections.abc import Sequence

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

def randomize_actuator_faults(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    severe_fault_prob: float | Sequence[float] | torch.Tensor = 0.3,
    failure_coef_severe: float | Sequence[float] | torch.Tensor = 0.3,
    failure_coef_moderate: float | Sequence[float] | torch.Tensor = 0.8,
    num_faults: int = 1,
    fixed_joint_idx: int | None = None,
    apply_once_per_episode: bool = False,
):
    asset: Articulation = env.scene[asset_cfg.name]

    def _resolve_env_param(
        value: float | Sequence[float] | torch.Tensor,
        target_env_ids: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            value_tensor = value.to(device=asset.device, dtype=torch.float32)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            value_tensor = torch.tensor(value, device=asset.device, dtype=torch.float32)
        else:
            return torch.full((target_env_ids.numel(),), float(value), device=asset.device, dtype=torch.float32)

        if value_tensor.ndim == 0:
            return torch.full((target_env_ids.numel(),), float(value_tensor.item()), device=asset.device, dtype=torch.float32)
        if value_tensor.numel() == target_env_ids.numel():
            return value_tensor.reshape(-1)
        if value_tensor.numel() == env.scene.num_envs:
            return value_tensor.reshape(-1)[target_env_ids]

        raise ValueError(
            f"Invalid '{name}' size {tuple(value_tensor.shape)} for {target_env_ids.numel()} target envs "
            f"and {env.scene.num_envs} total envs."
        )

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if apply_once_per_episode:
        if not hasattr(asset, "episode_fault_applied"):
            asset.episode_fault_applied = torch.zeros(
                env.scene.num_envs, dtype=torch.bool, device=asset.device
            )
        # Deployment faults persist for the remainder of the episode.
        env_ids = env_ids[~asset.episode_fault_applied[env_ids]]
        if env_ids.numel() == 0:
            return
    # breakpoint()
    for actuator in asset.actuators.values():
        size = len(asset.joint_names)
        N = env_ids.shape[0]
        severe_fault_prob_tensor = _resolve_env_param(severe_fault_prob, env_ids, "severe_fault_prob")
        lb_tensor = _resolve_env_param(failure_coef_severe, env_ids, "failure_coef_severe")
        ub_tensor = _resolve_env_param(failure_coef_moderate, env_ids, "failure_coef_moderate")
        is_severe = torch.bernoulli(severe_fault_prob_tensor).to(device=asset.device).unsqueeze(1)
        u1 = torch.rand((N, num_faults), device=asset.device) * lb_tensor.unsqueeze(1)  # severe failure
        u2 = torch.rand((N, num_faults), device=asset.device) * (ub_tensor - lb_tensor).unsqueeze(1) + lb_tensor.unsqueeze(1)  # moderate failure
        failure_coef = is_severe*u1 + (1-is_severe)*u2
        if fixed_joint_idx is None:
            faulty_joint_idx = torch.randint(
                low=0,
                high=size,
                size=(N, num_faults),
                dtype=torch.long,
                device=asset.device,
            )
        else:
            if num_faults != 1:
                raise ValueError("fixed_joint_idx requires num_faults=1.")
            if not 0 <= fixed_joint_idx < size:
                raise ValueError(
                    f"fixed_joint_idx must be in [0, {size - 1}], got {fixed_joint_idx}."
                )
            faulty_joint_idx = torch.full(
                (N, 1),
                fixed_joint_idx,
                dtype=torch.long,
                device=asset.device,
            )
        # if (asset.faulty_joint_idx[env_ids]).sum() > 0:
        #     breakpoint()
        asset.faulty_joint_idx[env_ids] = torch.zeros((env_ids.shape[0],len(asset.joint_names)), dtype=torch.long, device=asset.device)
        asset.faulty_joint_idx[env_ids[:,None], faulty_joint_idx] = 1
        # breakpoint()
        asset.motors_strength[env_ids] = asset.default_motors_strength[env_ids].clone()
        asset.motors_strength[env_ids[:,None],faulty_joint_idx] = failure_coef

        actuator.stiffness[env_ids] = (asset.data.default_joint_stiffness * asset.motors_strength)[env_ids].clone()
        actuator.damping[env_ids] = (asset.data.default_joint_damping * asset.motors_strength)[env_ids].clone()
    if apply_once_per_episode:
        asset.episode_fault_applied[env_ids] = True

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

        if hasattr(asset, "faulty_joint_idx"): # reset fault idx
            asset.faulty_joint_idx[env_ids] = torch.zeros((env_ids.shape[0],len(asset.joint_names)), dtype=torch.long, device=asset.device)
        else: # initialize fault idx
            asset.faulty_joint_idx = torch.zeros_like(asset.default_motors_strength, dtype=torch.long, device=asset.device)

        # This attribute is created only by deployment/play configurations
        # that opt into apply_once_per_episode.
        if hasattr(asset, "episode_fault_applied"):
            asset.episode_fault_applied[env_ids] = False
