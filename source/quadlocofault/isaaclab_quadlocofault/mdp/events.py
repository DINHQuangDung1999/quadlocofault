

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
    fixed_joint_idx: int | Sequence[int] | torch.Tensor | None = None,
    allowed_joint_indices: Sequence[int] | torch.Tensor | None = None,
    apply_once_per_episode: bool = False,
    track_postfault_rewards: bool = False,
    skip_healthy_episodes: bool = False,
):
    asset: Articulation = env.scene[asset_cfg.name]
    if track_postfault_rewards and not apply_once_per_episode:
        raise ValueError("Postfault scoring requires apply_once_per_episode=True.")
    if skip_healthy_episodes and not apply_once_per_episode:
        raise ValueError("Healthy episodes require apply_once_per_episode=True.")

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
        if skip_healthy_episodes:
            if not hasattr(asset, "episode_healthy"):
                raise RuntimeError("Healthy episode selection must run at reset before the fault event.")
            # Healthy episodes were selected at reset. Mark them as handled so
            # repeated interval callbacks cannot introduce a fault later.
            healthy = asset.episode_healthy[env_ids]
            asset.episode_fault_applied[env_ids[healthy]] = True
            env_ids = env_ids[~healthy]
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
        if fixed_joint_idx is not None and allowed_joint_indices is not None:
            raise ValueError("fixed_joint_idx and allowed_joint_indices are mutually exclusive.")
        if fixed_joint_idx is None:
            if allowed_joint_indices is None:
                candidate_joint_indices = torch.arange(size, device=asset.device)
            else:
                candidate_joint_indices = torch.as_tensor(
                    allowed_joint_indices, dtype=torch.long, device=asset.device
                ).reshape(-1)
                if candidate_joint_indices.numel() == 0:
                    raise ValueError("allowed_joint_indices must not be empty.")
                if torch.any((candidate_joint_indices < 0) | (candidate_joint_indices >= size)):
                    raise ValueError(f"All allowed joint indices must be in [0, {size - 1}].")
                if torch.unique(candidate_joint_indices).numel() != candidate_joint_indices.numel():
                    raise ValueError("allowed_joint_indices must not contain duplicates.")
            sampled_candidate_ids = torch.randint(
                low=0,
                high=candidate_joint_indices.numel(),
                size=(N, num_faults),
                dtype=torch.long,
                device=asset.device,
            )
            faulty_joint_idx = candidate_joint_indices[sampled_candidate_ids]
        elif isinstance(fixed_joint_idx, int):
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
        else:
            if num_faults != 1:
                raise ValueError("Per-environment fixed_joint_idx requires num_faults=1.")
            joint_indices = torch.as_tensor(
                fixed_joint_idx, dtype=torch.long, device=asset.device
            ).reshape(-1)
            if joint_indices.numel() == env.scene.num_envs:
                joint_indices = joint_indices[env_ids]
            elif joint_indices.numel() != N:
                raise ValueError(
                    "Per-environment fixed_joint_idx must contain either "
                    f"{env.scene.num_envs} entries or {N} target entries, got "
                    f"{joint_indices.numel()}."
                )
            if torch.any((joint_indices < 0) | (joint_indices >= size)):
                raise ValueError(f"All fixed joint indices must be in [0, {size - 1}].")
            faulty_joint_idx = joint_indices.unsqueeze(1)
        # if (asset.faulty_joint_idx[env_ids]).sum() > 0:
        #     breakpoint()
        asset.faulty_joint_idx[env_ids] = torch.zeros((env_ids.shape[0],len(asset.joint_names)), dtype=torch.long, device=asset.device)
        asset.faulty_joint_idx[env_ids[:,None], faulty_joint_idx] = 1
        # breakpoint()
        asset.motors_strength[env_ids] = asset.default_motors_strength[env_ids].clone()
        asset.motors_strength[env_ids[:,None],faulty_joint_idx] = failure_coef

        actuator.stiffness[env_ids] = (asset.data.default_joint_stiffness * asset.motors_strength)[env_ids].clone()
        actuator.damping[env_ids] = (asset.data.default_joint_damping * asset.motors_strength)[env_ids].clone()
    if track_postfault_rewards:
        if not hasattr(env, "_postfault_start_step"):
            env._postfault_start_step = torch.full(
                (env.num_envs,), -1, dtype=torch.long, device=env.device
            )
            env._postfault_reward_start = {
                name: torch.zeros_like(total)
                for name, total in env.reward_manager._episode_sums.items()
            }
        # Interval events run after reward computation: next step is the first
        # interval of physics affected by this fault.
        env._postfault_start_step[env_ids] = env.common_step_counter
        for name, total in env.reward_manager._episode_sums.items():
            env._postfault_reward_start[name][env_ids] = total[env_ids]
    if apply_once_per_episode:
        asset.episode_fault_applied[env_ids] = True

def reset_actuator_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    motors_strength_range: tuple[float, float] = (0.9, 1.1),
    healthy_episode_prob: float = 0.0,
):
    asset: Articulation = env.scene[asset_cfg.name]
    if not 0.0 <= healthy_episode_prob <= 1.0:
        raise ValueError("healthy_episode_prob must be in [0, 1].")

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if not hasattr(asset, "default_motors_strength"):
        asset.default_motors_strength = torch.ones(
            (env.scene.num_envs, len(asset.joint_names)), device=asset.device
        )
    low, high = motors_strength_range
    asset.default_motors_strength[env_ids] = (
        torch.rand((len(env_ids), len(asset.joint_names)), device=asset.device) * (high - low) + low
    )
    for actuator in asset.actuators.values():
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
    # The interval fault event may check this mask even when every episode is
    # intentionally faulted (healthy_episode_prob == 0), as in play mode.
    if not hasattr(asset, "episode_healthy"):
        asset.episode_healthy = torch.zeros(env.scene.num_envs, dtype=torch.bool, device=asset.device)
    if healthy_episode_prob:
        asset.episode_healthy[env_ids] = (
            torch.rand(env_ids.numel(), device=asset.device) < healthy_episode_prob
        )
    else:
        asset.episode_healthy[env_ids] = False

    # Curriculum is evaluated before reset events, so clear only after scoring.
    if hasattr(env, "_postfault_start_step"):
        env._postfault_start_step[env_ids] = -1
        for total in env._postfault_reward_start.values():
            total[env_ids] = 0.0
