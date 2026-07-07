# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_episode_reward_event(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    move_up_reward_threshold: float,
    move_down_reward_threshold: float | None = None,
    reward_term_names: Sequence[str] | None = None,
    successes_per_level: int = 1,
    failures_per_level: int = 1,
) -> dict[str, float]:
    """Curriculum based on accumulated episodic reward instead of traveled distance.

    This avoids promoting agents that merely slide or fall a long distance on sloped terrain.
    Promotion/demotion can also be gated by sustained performance over multiple episodes.

    Args:
        env: The RL environment.
        env_ids: Environment indices to update.
        reward_threshold: Episodic reward threshold above which environments move to harder terrain.
        move_down_threshold: Episodic reward threshold below which environments move to easier terrain.
            If omitted, defaults to half of ``reward_threshold``.
        reward_term_names: Optional subset of reward terms to accumulate. If omitted, all active reward
            terms are summed.
        normalize_by_episode_length: Whether to divide the episodic reward by ``env.max_episode_length_s``
            before thresholding.
        successes_per_level: Number of consecutive successful episodes required before moving an environment
            to a harder terrain level.
        failures_per_level: Number of consecutive failed episodes required before moving an environment
            to an easier terrain level.

    Returns:
        Logging dictionary with the mean terrain level and mean episodic reward of the selected environments.
    """
    terrain: TerrainImporter = env.scene.terrain
    if move_down_reward_threshold is None:
        move_down_reward_threshold = 0.5 * move_up_reward_threshold
    env_ids_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    episode_reward = torch.zeros(env.num_envs, device=env.device)

    if reward_term_names is None:
        reward_term_names = tuple(env.reward_manager._episode_sums.keys())

    for term_name in reward_term_names:
        if term_name not in env.reward_manager._episode_sums:
            raise ValueError(f"Reward term '{term_name}' not found in reward manager episode sums.")
        episode_reward += env.reward_manager._episode_sums[term_name]

    success = episode_reward[env_ids_tensor] >= move_up_reward_threshold
    failure = episode_reward[env_ids_tensor] < move_down_reward_threshold

    if not hasattr(env, "_terrain_curriculum_success_count"):
        env._terrain_curriculum_success_count = torch.zeros(env.num_envs, dtype=torch.long, device=terrain.device)
    if not hasattr(env, "_terrain_curriculum_failure_count"):
        env._terrain_curriculum_failure_count = torch.zeros(env.num_envs, dtype=torch.long, device=terrain.device)

    successes_per_level = max(int(successes_per_level), 1)
    failures_per_level = max(int(failures_per_level), 1)

    env._terrain_curriculum_success_count[env_ids_tensor] = torch.where(
        success,
        env._terrain_curriculum_success_count[env_ids_tensor] + 1,
        torch.zeros_like(env._terrain_curriculum_success_count[env_ids_tensor]),
    )
    env._terrain_curriculum_failure_count[env_ids_tensor] = torch.where(
        failure,
        env._terrain_curriculum_failure_count[env_ids_tensor] + 1,
        torch.zeros_like(env._terrain_curriculum_failure_count[env_ids_tensor]),
    )

    move_up = env._terrain_curriculum_success_count[env_ids_tensor] >= successes_per_level
    move_down = env._terrain_curriculum_failure_count[env_ids_tensor] >= failures_per_level
    move_down *= ~move_up

    changed = move_up | move_down
    changed_env_ids = env_ids_tensor[changed]
    env._terrain_curriculum_success_count[changed_env_ids] = 0
    env._terrain_curriculum_failure_count[changed_env_ids] = 0
    terrain.update_env_origins(env_ids_tensor, move_up, move_down)

    return {
        "mean_achieved_reward": float(torch.mean(episode_reward[env_ids_tensor]).item()),
        "mean_terrain_level": float(torch.mean(terrain.terrain_levels[env_ids_tensor].float()).item()),
        "move_up_rate": float(move_up.float().mean().item()),
        "move_down_rate": float(move_down.float().mean().item()),
    }

def actuator_fault_episode_reward_event(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    start_severe_fault_prob: float = 0.0,
    end_severe_fault_prob: float = 1.0,
    start_failure_range: tuple[float,float] = (0.3,1.0),
    end_failure_range: tuple[float,float] = (0.1,0.6),
    move_up_reward_threshold: float = 15.0,
    reward_term_names: Sequence[str] | None = ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"),
    num_levels: int = 20,
    successes_per_level: int = 1,
    event_name: str = "randomize_actuator_faults",
) -> dict[str, float] | None:
    """Gradually update the actuator fault event parameters when reward targets are met.

    The schedule advances by one stage after the mean episodic reward over the resetting environments
    reaches ``reward_threshold`` enough consecutive times. This avoids racing through stages when
    resets are frequent and the policy is already above threshold.
    """
    event_cfg = getattr(env.cfg.events, event_name, None)
    if event_cfg is None:
        return None

    if reward_term_names is None:
        reward_term_names = tuple(env.reward_manager._episode_sums.keys())

    env_ids_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    episode_reward = torch.zeros(env.num_envs, device=env.device)
    for term_name in reward_term_names:
        if term_name not in env.reward_manager._episode_sums:
            raise ValueError(f"Reward term '{term_name}' not found in reward manager episode sums.")
        episode_reward += env.reward_manager._episode_sums[term_name]

    success = episode_reward[env_ids_tensor] >= move_up_reward_threshold

    if not hasattr(env, "_actuator_fault_curriculum_level"):
        env._actuator_fault_curriculum_level = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if not hasattr(env, "_actuator_fault_curriculum_success_count"):
        env._actuator_fault_curriculum_success_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    successes_per_level = max(int(successes_per_level), 1)

    env._actuator_fault_curriculum_success_count[env_ids_tensor] = torch.where(
        success,
        env._actuator_fault_curriculum_success_count[env_ids_tensor] + 1,
        torch.zeros_like(env._actuator_fault_curriculum_success_count[env_ids_tensor]),
    )
    
    move_up = env._actuator_fault_curriculum_success_count[env_ids_tensor] >= successes_per_level
    env._actuator_fault_curriculum_level[env_ids_tensor] = torch.clamp(
        env._actuator_fault_curriculum_level[env_ids_tensor] + move_up.float(),
        max=float(num_levels),
        )
    progress = env._actuator_fault_curriculum_level / float(max(num_levels, 1))
    # if env._actuator_fault_curriculum_level[env_ids_tensor].sum() > 0:
    #     breakpoint()
    severe_fault_prob = start_severe_fault_prob + (end_severe_fault_prob - start_severe_fault_prob) * progress
    slb, sub = start_failure_range
    elb, eub = end_failure_range
    lb = slb + (elb - slb) * progress
    ub = sub + (eub - sub) * progress
    # breakpoint()
    event_cfg.params["severe_fault_prob"] = severe_fault_prob.tolist()
    event_cfg.params["failure_coef_severe"] = lb.tolist()
    event_cfg.params["failure_coef_moderate"] = ub.tolist()
    
    env._actuator_fault_curriculum_success_count[env_ids_tensor[move_up]] = 0

    return {
        "mean_achieved_reward": float(torch.mean(episode_reward[env_ids_tensor]).item()),
        "mean_fault_level": progress[env_ids_tensor].float().mean(),
        "move_up_rate": float(move_up.float().mean().item()),
    }


def actuator_fault_event_schedule_linear(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    start_severe_fault_prob: float = 0.0,
    end_severe_fault_prob: float = 1.0,
    start_failure_range: tuple[float, float] = (0.3, 1.0),
    end_failure_range: tuple[float, float] = (0.1, 0.6),
    num_epochs: int = 2000,
    steps_per_iteration: int = 24,
    event_name: str = "randomize_actuator_faults",
) -> dict[str, float] | None:
    """Linearly update actuator fault parameters based on training progress."""
    del env_ids

    event_cfg = getattr(env.cfg.events, event_name, None)
    if event_cfg is None:
        return None

    total_steps = max(int(num_epochs) * int(steps_per_iteration), 1)
    progress = min(float(env.common_step_counter) / float(total_steps), 1.0)
    iteration_estimate = float(env.common_step_counter) / float(max(int(steps_per_iteration), 1))

    severe_fault_prob = start_severe_fault_prob + (end_severe_fault_prob - start_severe_fault_prob) * progress
    slb, sub = start_failure_range
    elb, eub = end_failure_range
    lb = slb + (elb - slb) * progress
    ub = sub + (eub - sub) * progress

    event_cfg.params["severe_fault_prob"] = float(severe_fault_prob)
    event_cfg.params["failure_range"] = (float(lb), float(ub))

    return {
        "progress": progress,
        "severe_fault_prob": float(severe_fault_prob),
        "failure_lower": float(lb),
        "failure_upper": float(ub),
        "num_epochs": float(num_epochs),
        "steps_per_iteration": float(steps_per_iteration),
        "iteration_estimate": iteration_estimate,
    }


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_names: str = "track_lin_vel_xy_exp",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_names)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_names][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
