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
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def power_distribution(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.var(power, dim=1)


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(power), dim=1)


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    # foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    foot_velocity_tanh = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.sum(reward, dim=1)
    # return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
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
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
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


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
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
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
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
            rew[:,i] = 0.05 * (cur_pos[:,i] - init_pos[:,i])**2
        elif name.startswith('R'):
            rew[:,i] = 0.2 * (cur_pos[:,i] - init_pos[:,i])**2
        else:
            raise ValueError(f'Must be either front or rear leg instead of {name}.')
    return torch.sum(rew, dim=1)

def VHIP_style_reward(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # COM in world frame
    com_w = asset.data.root_com_pos_w  # (num_envs, 3)

    # COP in world frame
    foot_ids, _ = asset.find_bodies(".*_foot", preserve_order=True)
    foot_ids = torch.tensor(foot_ids, device=asset.device)
    foot_pos_w = asset.data.body_link_pos_w[:, foot_ids, :] 
    # normal contact forces (world)
    forces_w = contact_sensor.data.net_forces_w[:, foot_ids, :] 

    # use vertical component as normal force magnitude
    fz = torch.clamp(forces_w[..., 2], min=0.0)
    # weighted COP (world)
    cop_w = (foot_pos_w * fz.unsqueeze(-1)).sum(dim=1) / (fz.sum(dim=1, keepdim=True) + 1e-6)
    
    # angle between v and +Z
    v = com_w - cop_w 
    length = torch.norm(v, dim=-1).clamp_min(1e-6)
    theta = torch.acos(torch.abs(v[:, 2]) / length)

    # theta acc
    g = 9.81
    theta_ddot = (-(g / length) * torch.sin(theta)**2
)
    # distance to support polygon
    foot_xy = foot_pos_w[:, :, :2]
    com_xy = com_w[:, :2].unsqueeze(1)
    # edges: Ci -> Cj (wrap)
    Ci = foot_xy
    Cj = torch.roll(foot_xy, shifts=1, dims=1)
    # vectors from COMproj to vertices
    oi = com_xy - Ci
    oj = com_xy - Cj
    # 2D cross product magnitude
    cross = oi[..., 0] * oj[..., 1] - oi[..., 1] * oj[..., 0]  # (N, K)
    # edge length
    edge_len = torch.norm(Cj - Ci, dim=-1).clamp_min(1e-6)
    # distance to each edge
    dist = torch.abs(cross) / edge_len 
    d_max = dist.max(dim=1).values
    # breakpoint()
    return 0.015*theta + 0.01*theta_ddot + 0.01*d_max

def raibert_foot_placement_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stance_time: float,
    nominal_foot_positions_xy: list[list[float]],
    vel_gain: float = 0.0,
) -> torch.Tensor:
    """Penalize foot xy placement error against a Raibert-style desired foothold.

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

    return torch.sum(torch.square(foot_xy - p_des_xy), dim=(1, 2))
