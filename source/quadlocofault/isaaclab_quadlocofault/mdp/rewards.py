# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_faulty_leg_mask(asset: Articulation, body_names: list[str]) -> torch.Tensor:
    """Aggregate per-joint fault flags into a per-body leg mask using FL/FR/RL/RR prefixes."""
    if not hasattr(asset, "faulty_joint_idx"):
        return torch.zeros((asset.num_instances, len(body_names)), device=asset.device, dtype=torch.float)

    # Go2 joint order is:
    #   [FL, FR, RL, RR] hips,
    #   [FL, FR, RL, RR] thighs,
    #   [FL, FR, RL, RR] calves.
    # Reshape to [env, joint_type, leg], reduce over joint types, and select
    # the corresponding leg for every requested body. Repeated prefixes allow
    # the same mask to be used for feet, calves, thighs, or any combination.
    leg_index_by_prefix = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}
    body_prefixes = [name[:2] for name in body_names]
    invalid_prefixes = sorted(set(body_prefixes) - leg_index_by_prefix.keys())
    if invalid_prefixes:
        raise ValueError(
            "Faulty-leg mask requires FL/FR/RL/RR body-name prefixes, "
            f"but received invalid prefixes {invalid_prefixes} from {body_names}."
        )
    faults_by_type_and_leg = asset.faulty_joint_idx.reshape(asset.num_instances, 3, 4)
    faulty_legs = faults_by_type_and_leg.any(dim=1).float()
    return torch.stack(
        [faulty_legs[:, leg_index_by_prefix[prefix]] for prefix in body_prefixes],
        dim=1,
    )

def power_distribution(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.var(torch.abs(power), dim=1)


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(power), dim=1)


class ActionSmoothnessPenalty(ManagerTermBase):
    """Penalize the squared second-order finite difference of policy actions."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_prev_action = torch.zeros_like(env.action_manager.action)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._prev_prev_action.zero_()
        else:
            self._prev_prev_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        action = env.action_manager.action
        prev_action = env.action_manager.prev_action
        second_difference = action - 2.0 * prev_action + self._prev_prev_action
        self._prev_prev_action.copy_(prev_action)
        return torch.sum(torch.square(second_difference), dim=1)


class DreamWaQActionSmoothnessPenalty(ManagerTermBase):
    """DreamWaQ second-difference penalty on processed joint targets.

    DreamWaQ masks the first two steps of every episode. Keeping that state in
    the reward term also avoids treating a reset discontinuity as an action
    smoothness violation.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_target = torch.zeros_like(env.action_manager.action)
        self._previous_previous_target = torch.zeros_like(env.action_manager.action)
        self._valid_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._previous_target.zero_()
            self._previous_previous_target.zero_()
            self._valid_steps.zero_()
        else:
            self._previous_target[env_ids] = 0.0
            self._previous_previous_target[env_ids] = 0.0
            self._valid_steps[env_ids] = 0

    def __call__(self, env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
        target = env.action_manager.get_term(action_name).processed_actions
        second_difference = (
            target - 2.0 * self._previous_target + self._previous_previous_target
        )
        penalty = torch.sum(torch.square(second_difference), dim=1)
        penalty *= self._valid_steps.ge(2)

        self._previous_previous_target.copy_(self._previous_target)
        self._previous_target.copy_(target)
        self._valid_steps.add_(1)
        return penalty


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, tanh_mult: float, std: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    # foot_velocity_tanh = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    reward = foot_z_target_error * foot_velocity_tanh
    # return torch.sum(reward, dim=1)
    return torch.exp(-torch.sum(reward, dim=1) / std)

def foot_clearance_reward_dreamflex(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
):
    """Penalize healthy swing feet that deviate from the target terrain clearance.

    Ray-caster misses are represented by non-finite hit positions. Ignore those
    rays when estimating terrain height and disable the term for environments
    where the scanner has no valid hits.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    foot_ids, foot_names = asset.find_bodies(".*_foot", preserve_order=True)
    foot_height_w = asset.data.body_pos_w[:, foot_ids, 2]
    ray_hit_height_w = sensor.data.ray_hits_w[..., 2]
    valid_ray_mask = torch.isfinite(ray_hit_height_w)
    valid_ray_count = valid_ray_mask.sum(dim=1, keepdim=True)

    # Invalid values must be removed before the reduction: inf * 0 is NaN.
    valid_ray_height_sum = torch.where(
        valid_ray_mask,
        ray_hit_height_w,
        torch.zeros_like(ray_hit_height_w),
    ).sum(dim=1, keepdim=True)
    terrain_height_w = valid_ray_height_sum / valid_ray_count.clamp_min(1)

    foot_z_error = torch.square(foot_height_w - terrain_height_w - target_height)
    foot_xy_speed = torch.norm(asset.data.body_lin_vel_w[:, foot_ids, :2], dim=-1)
    healthy_leg_mask = _get_faulty_leg_mask(asset, foot_names) < 0.5
    valid_terrain_mask = valid_ray_count > 0

    clearance_penalty = foot_z_error * foot_xy_speed
    clearance_penalty = torch.where(
        healthy_leg_mask & valid_terrain_mask,
        clearance_penalty,
        torch.zeros_like(clearance_penalty),
    )
    return torch.sum(clearance_penalty, dim=1)


def faulty_leg_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_sensor_ids, foot_names = contact_sensor.find_bodies(".*_foot", preserve_order=True)
    faulty_leg_mask = _get_faulty_leg_mask(asset, foot_names)
    foot_contact_force = torch.norm(contact_sensor.data.net_forces_w[:, foot_sensor_ids, :], dim=-1)
    faulty_contacts = (foot_contact_force > threshold).float() * faulty_leg_mask
    return torch.sum(faulty_contacts, dim=1)


def faulty_leg_vertical_load(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the faulty leg's vertical support load normalized by robot weight.

    Averaging the contact sensor's existing short history reduces single-step
    contact noise.  This function returns a non-negative cost and should be
    configured with a negative reward weight.  Integrated over an episode, the
    term is proportional to the vertical support impulse transmitted through
    the faulty leg.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_sensor_ids, foot_names = contact_sensor.find_bodies(
        ".*_foot", preserve_order=True
    )
    vertical_load = torch.abs(
        contact_sensor.data.net_forces_w_history[:, :, foot_sensor_ids, 2]
    ).mean(dim=1)
    faulty_leg_mask = _get_faulty_leg_mask(asset, foot_names)

    robot_mass = asset.root_physx_view.get_masses().to(asset.device).sum(dim=1)
    robot_weight = robot_mass * 9.81
    return torch.sum(vertical_load * faulty_leg_mask, dim=1) / robot_weight.clamp_min(1.0e-6)


def faulty_foot_planar_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
) -> torch.Tensor:
    """Penalize squared world-frame planar velocity of the faulty foot.

    A planted foot is approximately stationary in the world frame, so this
    quantity penalizes both swing and ground sliding without requiring a
    contact-state gate.  Only the leg containing the configured faulty joint
    contributes.  The vertical component is deliberately excluded so the foot
    can still accommodate rough terrain.

    This function returns a non-negative cost and should be configured with a
    negative reward weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        foot_names = asset.body_names[body_ids]
    elif isinstance(body_ids, int):
        foot_names = [asset.body_names[body_ids]]
    else:
        foot_names = [asset.body_names[body_id] for body_id in body_ids]

    foot_planar_speed_l2 = torch.sum(
        torch.square(asset.data.body_lin_vel_w[:, body_ids, :2]), dim=-1
    )
    faulty_leg_mask = _get_faulty_leg_mask(asset, foot_names)
    return torch.sum(foot_planar_speed_l2 * faulty_leg_mask, dim=1)


def hip_fault_thigh_calf_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize thigh and calf velocity only on a hip-faulted leg.

    A one-time compensating bend incurs a short-lived cost, while repeated
    flexion accumulates cost throughout the episode.  The passive faulty hip
    and every joint on healthy legs are excluded.  This function assumes the
    project's hip/thigh/calf x FL/FR/RL/RR joint ordering and returns a
    non-negative cost for use with a negative reward weight.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_velocity = asset.data.joint_vel.reshape(asset.num_instances, 3, 4)
    faults = asset.faulty_joint_idx.reshape(asset.num_instances, 3, 4).bool()

    hip_fault_mask = faults[:, 0, :].unsqueeze(1)
    thigh_calf_velocity_l2 = torch.square(joint_velocity[:, 1:, :])
    return torch.sum(thigh_calf_velocity_l2 * hip_fault_mask, dim=(1, 2))


class FaultyHipFootLateralDeviationL2(ManagerTermBase):
    """Penalize lateral foot displacement from its nominal hip-relative offset.

    The nominal signed offsets are captured once from the initialized robot,
    rather than assuming that each foot should have zero lateral displacement
    from its hip link origin.  Only a leg with a faulty hip
    abduction/adduction joint contributes.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[asset_cfg.name]

        self.foot_ids, self.foot_names = self.asset.find_bodies(
            ".*_foot", preserve_order=True
        )
        hip_body_ids, hip_body_names = self.asset.find_bodies(
            ".*_hip", preserve_order=True
        )
        hip_joint_ids, hip_joint_names = self.asset.find_joints(
            ".*_hip_joint", preserve_order=True
        )

        hip_body_id_by_prefix = {
            name[:2]: body_id for body_id, name in zip(hip_body_ids, hip_body_names)
        }
        hip_joint_id_by_prefix = {
            name[:2]: joint_id for joint_id, name in zip(hip_joint_ids, hip_joint_names)
        }
        foot_prefixes = [name[:2] for name in self.foot_names]
        missing_prefixes = sorted(
            set(foot_prefixes)
            - (hip_body_id_by_prefix.keys() & hip_joint_id_by_prefix.keys())
        )
        if missing_prefixes:
            raise ValueError(
                "Could not pair feet with hip bodies and joints for prefixes "
                f"{missing_prefixes}."
            )

        self.hip_body_ids = [
            hip_body_id_by_prefix[prefix] for prefix in foot_prefixes
        ]
        self.hip_joint_ids = [
            hip_joint_id_by_prefix[prefix] for prefix in foot_prefixes
        ]
        self.num_feet = len(self.foot_ids)
        self.nominal_lateral_offset: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        foot_to_hip_w = (
            self.asset.data.body_pos_w[:, self.foot_ids, :]
            - self.asset.data.body_pos_w[:, self.hip_body_ids, :]
        )
        yaw_w = yaw_quat(self.asset.data.root_quat_w)
        yaw_feet = yaw_w.unsqueeze(1).expand(-1, self.num_feet, -1).reshape(-1, 4)
        foot_to_hip_b = quat_rotate_inverse(
            yaw_feet, foot_to_hip_w.reshape(-1, 3)
        ).view(self.asset.num_instances, self.num_feet, 3)
        lateral_offset = foot_to_hip_b[..., 1]

        if self.nominal_lateral_offset is None:
            self.nominal_lateral_offset = lateral_offset.detach().mean(
                dim=0, keepdim=True
            )

        hip_fault_mask = self.asset.faulty_joint_idx[:, self.hip_joint_ids].float()
        lateral_deviation_l2 = torch.square(
            lateral_offset - self.nominal_lateral_offset
        )
        return torch.sum(lateral_deviation_l2 * hip_fault_mask, dim=1)


def faulty_leg_link_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize thigh and calf contacts belonging to a faulty leg."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    link_sensor_ids, link_names = contact_sensor.find_bodies(
        ".*_(thigh|calf)", preserve_order=True
    )
    faulty_link_mask = _get_faulty_leg_mask(asset, link_names)
    link_contact_force = torch.norm(
        contact_sensor.data.net_forces_w[:, link_sensor_ids, :], dim=-1
    )
    faulty_contacts = (link_contact_force > threshold).float() * faulty_link_mask
    return torch.sum(faulty_contacts, dim=1)


def faulty_foot_lift_reward(
    env: ManagerBasedRLEnv,
    height_sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_clearance: float = 0.08,
    severity_temperature: float = 0.15,
) -> torch.Tensor:
    """Penalize insufficient faulty-foot clearance above the terrain.

    The mean terrain height follows Isaac Lab's ``base_height_l2`` convention.
    The normalized squared cost is one at zero clearance and becomes zero once
    the target clearance is reached; lifting higher incurs no additional cost.
    The penalty is scaled continuously by fault severity using
    ``exp(-motor_strength / severity_temperature)``. Thus, an almost powerless
    motor receives a weight near one and moderate faults rapidly approach zero.

    This function returns a non-negative cost and should therefore be configured
    with a negative reward weight.
    """
    if target_clearance <= 0.0:
        raise ValueError(f"target_clearance must be positive, got {target_clearance}.")
    if severity_temperature <= 0.0:
        raise ValueError(f"severity_temperature must be positive, got {severity_temperature}.")

    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(asset, "motors_strength") or not hasattr(asset, "faulty_joint_idx"):
        return torch.zeros(asset.num_instances, device=asset.device)

    height_sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name]
    foot_ids, foot_names = asset.find_bodies(".*_foot", preserve_order=True)
    foot_height_w = asset.data.body_link_pos_w[:, foot_ids, 2]
    terrain_height_w = torch.mean(height_sensor.data.ray_hits_w[..., 2], dim=1, keepdim=True)
    # Go2 joint order is [joint_type, leg] after reshaping:
    # hip/thigh/calf x FL/FR/RL/RR.
    if tuple(name[:2] for name in foot_names) != ("FL", "FR", "RL", "RR"):
        raise ValueError(f"Expected feet in FL/FR/RL/RR order, received {foot_names}.")
    fault_marked = asset.faulty_joint_idx.reshape(asset.num_instances, 3, 4).bool()
    motor_strength = asset.motors_strength.reshape(asset.num_instances, 3, 4)
    joint_severity = torch.exp(-motor_strength.clamp_min(0.0) / severity_temperature)
    joint_severity *= fault_marked
    leg_severity = joint_severity.amax(dim=1)

    clearance = foot_height_w - terrain_height_w
    normalized_shortfall = torch.clamp(
        (target_clearance - clearance) / target_clearance,
        min=0.0,
        max=1.0,
    )
    lift_cost = torch.square(normalized_shortfall)
    return torch.sum(lift_cost * leg_severity, dim=1)


def fault_compensation_posture_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    thigh_lift_delta: float = -0.20,
    knee_bend_delta: float = -0.20,
    std: float = 0.15,
) -> torch.Tensor:
    """Reward a mild, joint-specific compensating posture on the faulty leg.

    A calf (knee) fault targets the corresponding thigh/hip-pitch joint. A
    thigh/hip-pitch fault targets flexion of the corresponding calf (knee)
    joint. Lateral hip-joint faults are not shaped by this term. Targets are
    offsets from the default pose.
    """
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")

    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(asset, "faulty_joint_idx"):
        return torch.zeros(asset.num_instances, device=asset.device)

    # Go2 ordering: [hip joints (4), thigh joints (4), calf joints (4)].
    joint_pos = asset.data.joint_pos.view(asset.num_instances, 3, 4)
    default_pos = asset.data.default_joint_pos.view(asset.num_instances, 3, 4)
    faults = asset.faulty_joint_idx.view(asset.num_instances, 3, 4).bool()

    knee_fault = faults[:, 2]
    thigh_fault = faults[:, 1]
    thigh_reward = torch.exp(-torch.square(joint_pos[:, 1] - default_pos[:, 1] - thigh_lift_delta) / std**2)
    knee_reward = torch.exp(-torch.square(joint_pos[:, 2] - default_pos[:, 2] - knee_bend_delta) / std**2)

    fault_count = knee_fault.sum(dim=1) + thigh_fault.sum(dim=1)
    reward = (thigh_reward * knee_fault + knee_reward * thigh_fault).sum(dim=1)
    return reward / fault_count.clamp_min(1)


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    asset_cfg: SceneEntityCfg | None = None,
    ignore_faulty_legs: bool = False,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    air_time_reward = (last_air_time - threshold) * first_contact

    # DreamFLEX applies this term only to healthy feet, whereas FT-Net's Table I
    # sums it over all four feet.
    if ignore_faulty_legs:
        if asset_cfg is None:
            raise ValueError("asset_cfg is required when ignore_faulty_legs=True.")
        asset: Articulation = env.scene[asset_cfg.name]
        _, foot_names = asset.find_bodies(".*_foot", preserve_order=True)
        faulty_leg_mask = _get_faulty_leg_mask(asset, foot_names)
        air_time_reward *= 1.0 - faulty_leg_mask

    reward = torch.sum(air_time_reward, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ignore_faulty_legs: bool = False,
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slide_cost = body_vel.norm(dim=-1) * contacts
    if ignore_faulty_legs:
        body_names = [asset.body_names[body_id] for body_id in asset_cfg.body_ids]
        slide_cost *= 1.0 - _get_faulty_leg_mask(asset, body_names)
    reward = torch.sum(slide_cost, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def joint_motion_cosmetic(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    cur_pos = asset.data.joint_pos
    init_pos = asset.data.default_joint_pos
    rew = torch.zeros_like(cur_pos)
    for i, name in enumerate(asset.joint_names):
        if name.startswith('F'):
            rew[:,i] = -0.05 * (cur_pos[:,i] - init_pos[:,i])**2
        elif name.startswith('R'):
            rew[:,i] = -0.2 * (cur_pos[:,i] - init_pos[:,i])**2
        else:
            raise ValueError(f'Must be either front or rear leg instead of {name}.')
    return torch.sum(rew, dim=1)

def vhip_style_reward_ftnet(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
    theta_scale: float = 1.0,
    theta_ddot_scale: float = 1.0,
    support_dist_scale: float = 1.0,
):
    """Compute the three FT-Net VHIP heuristic terms.

    The support-polygon term is evaluated on the convex hull of the feet that
    are currently in contact. This avoids connecting FL/FR/RL/RR in a
    self-crossing order and correctly closes the triangle when one foot is in
    swing. Environments with no valid contact receive zero VHIP angle and
    acceleration instead of measuring against the world origin.
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_asset_ids, foot_names = asset.find_bodies(".*_foot", preserve_order=True)
    foot_sensor_ids, sensor_foot_names = contact_sensor.find_bodies(".*_foot", preserve_order=True)
    if foot_names != sensor_foot_names:
        raise ValueError(
            f"Foot body ordering mismatch between asset and contact sensor: {foot_names} vs {sensor_foot_names}"
        )
    foot_asset_ids = torch.tensor(foot_asset_ids, device=asset.device)
    foot_sensor_ids = torch.tensor(foot_sensor_ids, device=asset.device)

    # ``root_com_pos_w`` is only the root link COM. FT-Net's p_COM denotes the
    # whole robot, so use the mass-weighted COM of every articulation body.
    body_masses = asset.root_physx_view.get_masses().to(asset.device)
    com_w = torch.sum(asset.data.body_com_pos_w * body_masses.unsqueeze(-1), dim=1)
    com_w /= body_masses.sum(dim=1, keepdim=True).clamp_min(1e-6)

    foot_pos_w = asset.data.body_link_pos_w[:, foot_asset_ids, :]
    forces_w = contact_sensor.data.net_forces_w[:, foot_sensor_ids, :]

    fz = torch.clamp(forces_w[..., 2], min=0.0)
    contact_mask = fz > contact_threshold

    masked_fz = fz * contact_mask
    total_fz = masked_fz.sum(dim=1, keepdim=True)
    has_contact = total_fz.squeeze(1) > 1e-6
    cop_w = (foot_pos_w * masked_fz.unsqueeze(-1)).sum(dim=1) / total_fz.clamp_min(1e-6)

    l = com_w - cop_w
    l_norm = torch.norm(l, dim=-1).clamp_min(1e-6)
    theta = torch.acos(torch.clamp(torch.abs(l[:, 2]) / l_norm, 0.0, 1.0))
    theta = torch.where(has_contact, theta, torch.zeros_like(theta))

    g = 9.81
    # FT-Net Table I penalizes squared VHIP angular acceleration.
    theta_ddot = torch.square((g / l_norm) * torch.sin(theta))
    theta_ddot = torch.where(has_contact, theta_ddot, torch.zeros_like(theta_ddot))

    com_xy = com_w[:, :2]
    foot_xy = foot_pos_w[:, :, :2]

    if foot_xy.shape[1] != 4:
        raise ValueError(f"FT-Net VHIP reward expects four feet, got {foot_xy.shape[1]}.")

    # Test all six foot pairs. A pair belongs to the convex hull when all other
    # contacting feet lie on one side of its supporting line. This handles all
    # two-, three-, and four-foot contact combinations without Python loops.
    pair_i = torch.tensor((0, 0, 0, 1, 1, 2), device=asset.device)
    pair_j = torch.tensor((1, 2, 3, 2, 3, 3), device=asset.device)
    ci = foot_xy[:, pair_i]
    cj = foot_xy[:, pair_j]
    edge = cj - ci
    edge_len = torch.linalg.vector_norm(edge, dim=-1)

    point_rel = foot_xy.unsqueeze(1) - ci.unsqueeze(2)
    point_side = (
        edge[..., 0].unsqueeze(2) * point_rel[..., 1]
        - edge[..., 1].unsqueeze(2) * point_rel[..., 0]
    )
    active_points = contact_mask.unsqueeze(1)
    has_positive_side = ((point_side > 1e-6) & active_points).any(dim=2)
    has_negative_side = ((point_side < -1e-6) & active_points).any(dim=2)
    pair_in_contact = contact_mask[:, pair_i] & contact_mask[:, pair_j]
    hull_edge = pair_in_contact & ~(has_positive_side & has_negative_side) & (edge_len > 1e-6)

    com_rel = com_xy.unsqueeze(1) - ci
    com_cross = edge[..., 0] * com_rel[..., 1] - edge[..., 1] * com_rel[..., 0]
    # FT-Net Eq. (6): triangle area divided by edge length. Triangle area
    # contributes the factor 1/2.
    dist = 0.5 * torch.abs(com_cross) / edge_len.clamp_min(1e-6)
    dist = torch.where(hull_edge, dist, torch.zeros_like(dist))
    d_max = dist.max(dim=1).values

    return theta_scale * theta + theta_ddot_scale * theta_ddot + support_dist_scale * d_max


def VHIP_style_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 2.0,
    theta_scale: float = 1.0,
    theta_ddot_scale: float = 1.0,
    support_dist_scale: float = 1.0,
) -> torch.Tensor:
    """Backward-compatible FT-Net-style VHIP heuristic reward entrypoint."""
    return vhip_style_reward_ftnet(
        env=env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        contact_threshold=contact_threshold,
        theta_scale=theta_scale,
        theta_ddot_scale=theta_ddot_scale,
        support_dist_scale=support_dist_scale,
    )

def faulty_joint_motion_reward_dreamflex(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_name: str = "joint_pos",
):
    asset: Articulation = env.scene[asset_cfg.name]

    faulty_mask = asset.faulty_joint_idx.float()
    q = asset.data.joint_pos

    # DreamFLEX uses q_des = q_default + action. For a JointPositionAction,
    # processed_actions is the exact scaled/offset position target sent to the
    # articulation's PD controller.
    action_term = env.action_manager.get_term(action_name)
    q_des = action_term.processed_actions

    if q_des.shape[1] != q.shape[1]:
        raise ValueError(
            f"DreamFLEX faulty-joint reward expected {q.shape[1]} desired joint positions "
            f"from action term '{action_name}', but received {q_des.shape[1]}."
        )

    return torch.sum(((q - q_des) ** 2) * faulty_mask, dim=1)


class RaibertFootPlacementReward(ManagerTermBase):
    """Penalize healthy swing-foot error from the Raibert landing location.

    Following Song et al. (CoRL 2024), the desired planar landing location is

        p_land = p_hip_ground + v_com * stance_time / 2.

    All quantities are expressed in the yaw-aligned body frame.  Contact data
    are used only to restrict this minimal, phase-free version to swing legs;
    no additional sensor or prescribed gait phase is required.
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.stance_time: float = cfg.params["stance_time"]
        self.contact_threshold: float = cfg.params.get("contact_threshold", 1.0)
        if self.stance_time <= 0.0:
            raise ValueError(f"stance_time must be positive, got {self.stance_time}.")

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.foot_ids = list(asset_cfg.body_ids)
        self.foot_names = [self.asset.body_names[body_id] for body_id in self.foot_ids]
        foot_prefixes = [name[:2] for name in self.foot_names]

        hip_ids, hip_names = self.asset.find_bodies(".*_hip", preserve_order=True)
        hip_id_by_prefix = {name[:2]: body_id for body_id, name in zip(hip_ids, hip_names)}
        missing_hips = sorted(set(foot_prefixes) - hip_id_by_prefix.keys())
        if missing_hips:
            raise ValueError(
                "Raibert reward could not pair feet with hip bodies for prefixes "
                f"{missing_hips}. Available hip bodies: {hip_names}."
            )
        self.hip_ids = [hip_id_by_prefix[prefix] for prefix in foot_prefixes]

        sensor_foot_ids, sensor_foot_names = self.contact_sensor.find_bodies(
            ".*_foot", preserve_order=True
        )
        sensor_id_by_prefix = {
            name[:2]: body_id for body_id, name in zip(sensor_foot_ids, sensor_foot_names)
        }
        missing_sensor_feet = sorted(set(foot_prefixes) - sensor_id_by_prefix.keys())
        if missing_sensor_feet:
            raise ValueError(
                "Raibert reward could not pair asset feet with contact-sensor feet for prefixes "
                f"{missing_sensor_feet}. Available sensor feet: {sensor_foot_names}."
            )
        self.sensor_foot_ids = [sensor_id_by_prefix[prefix] for prefix in foot_prefixes]
        self.num_feet = len(self.foot_ids)
        # Filled lazily on the first reward call, after startup mass
        # randomization has completed.
        self.body_masses: torch.Tensor | None = None
        self.total_mass: torch.Tensor | None = None

    def _refresh_body_masses(self) -> None:
        """Cache masses, including any startup mass randomization."""
        self.body_masses = self.asset.root_physx_view.get_masses().to(self.asset.device)
        self.total_mass = self.body_masses.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        stance_time: float,
        contact_threshold: float = 1.0,
    ) -> torch.Tensor:
        if self.body_masses is None:
            self._refresh_body_masses()

        # Express actual feet and the points directly below their hips in the
        # yaw-aligned body frame. Projection to the ground does not change xy.
        root_pos_w = self.asset.data.root_pos_w.unsqueeze(1)
        yaw_w = yaw_quat(self.asset.data.root_quat_w)
        yaw_feet = yaw_w.unsqueeze(1).expand(-1, self.num_feet, -1).reshape(-1, 4)

        foot_pos_rel_w = self.asset.data.body_pos_w[:, self.foot_ids, :] - root_pos_w
        hip_pos_rel_w = self.asset.data.body_pos_w[:, self.hip_ids, :] - root_pos_w
        num_envs = self.asset.num_instances
        foot_xy_b = quat_rotate_inverse(yaw_feet, foot_pos_rel_w.reshape(-1, 3)).view(
            num_envs, self.num_feet, 3
        )[..., :2]
        hip_ground_xy_b = quat_rotate_inverse(yaw_feet, hip_pos_rel_w.reshape(-1, 3)).view(
            num_envs, self.num_feet, 3
        )[..., :2]

        # Use the mass-weighted velocity of the complete robot for v_com.
        com_vel_w = torch.sum(
            self.asset.data.body_com_lin_vel_w * self.body_masses.unsqueeze(-1), dim=1
        )
        com_vel_b = quat_rotate_inverse(yaw_w, com_vel_w / self.total_mass)[..., :2]
        desired_landing_xy_b = hip_ground_xy_b + 0.5 * self.stance_time * com_vel_b.unsqueeze(1)

        contact_force = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self.sensor_foot_ids, :], dim=-1
        )
        swing_mask = contact_force <= self.contact_threshold
        healthy_mask = _get_faulty_leg_mask(self.asset, self.foot_names) < 0.5

        placement_error = torch.sum(torch.square(foot_xy_b - desired_landing_xy_b), dim=-1)
        return torch.sum(placement_error * swing_mask * healthy_mask, dim=1)


def raibert_foot_placement_reward_approximation(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stance_time: float,
    nominal_foot_positions_xy: list[list[float]],
    vel_gain: float = 0.0,
) -> torch.Tensor:
    """Legacy phase-free approximation retained for reference; currently unused.

    The desired foothold is defined in the yaw-aligned body frame as:

        p_des_xy = p_nom_xy + 0.5 * stance_time * v_cmd_xy
                   + vel_gain * (v_body_xy - v_cmd_xy)

    Args:
        command_name: Usually "base_velocity".
        asset_cfg: Robot asset with foot body_ids resolved in the desired foot order.
        stance_time: Approximate stance duration used by the Raibert heuristic.
        nominal_foot_positions_xy: Per-foot nominal xy positions in the yaw/body frame.
            Example for Go2:
                [[ 0.20,  0.13],
                 [ 0.20, -0.13],
                 [-0.20,  0.13],
                 [-0.20, -0.13]]
        vel_gain: Optional feedback term on body velocity tracking error.

    Returns:
        Sum of squared xy placement error for all selected feet.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # commanded and measured planar base velocity in body frame
    cmd = env.command_manager.get_command(command_name)
    v_cmd_xy = cmd[:, :2]                          # (N, 2)
    v_body_xy = asset.data.root_lin_vel_b[:, :2]  # (N, 2)

    # current foot positions in the yaw-aligned body frame
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]               # (N, F, 3)
    root_pos_w = asset.data.root_pos_w.unsqueeze(1)                            # (N, 1, 3)
    foot_pos_rel_w = foot_pos_w - root_pos_w                                   # (N, F, 3)

    num_envs = foot_pos_w.shape[0]
    num_feet = foot_pos_w.shape[1]

    yaw_quat_w = yaw_quat(asset.data.root_quat_w)                              # (N, 4)
    yaw_quat_feet = yaw_quat_w.unsqueeze(1).expand(-1, num_feet, -1)           # (N, F, 4)

    foot_pos_b = quat_rotate_inverse(
        yaw_quat_feet.reshape(-1, 4),
        foot_pos_rel_w.reshape(-1, 3),
    ).view(num_envs, num_feet, 3)

    foot_xy = foot_pos_b[:, :, :2]                                             # (N, F, 2)

    # nominal per-foot stance template in yaw/body frame
    p_nom_xy = torch.tensor(
        nominal_foot_positions_xy, dtype=foot_xy.dtype, device=foot_xy.device
    ).unsqueeze(0).expand(num_envs, -1, -1)                                    # (N, F, 2)

    # Raibert-style desired foothold
    p_des_xy = (
        p_nom_xy
        + 0.5 * stance_time * v_cmd_xy.unsqueeze(1)
        + vel_gain * (v_body_xy - v_cmd_xy).unsqueeze(1)
    )
    foot_ids, foot_names = asset.find_bodies(".*_foot", preserve_order=True)
    faulty_leg_mask = _get_faulty_leg_mask(asset, foot_names)
    normal_leg_mask = (1.0 - faulty_leg_mask).unsqueeze(-1)

    placement_error = torch.square(foot_xy - p_des_xy) * normal_leg_mask
    return torch.sum(placement_error, dim=(1, 2))
