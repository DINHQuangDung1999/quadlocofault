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


def terrain_levels_episode_reward(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_threshold: float,
    move_down_threshold: float | None = None,
    reward_term_names: Sequence[str] | None = None,
    normalize_by_episode_length: bool = False,
) -> dict[str, float]:
    """Curriculum based on accumulated episodic reward instead of traveled distance.

    This avoids promoting agents that merely slide or fall a long distance on sloped terrain.

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

    Returns:
        Logging dictionary with the mean terrain level and mean episodic reward of the selected environments.
    """
    terrain: TerrainImporter = env.scene.terrain
    if move_down_threshold is None:
        move_down_threshold = 0.5 * reward_threshold

    env_ids_tensor = torch.as_tensor(env_ids, device=terrain.device, dtype=torch.long)
    episode_reward = torch.zeros(len(env_ids_tensor), device=terrain.device)

    if reward_term_names is None:
        reward_term_names = tuple(env.reward_manager._episode_sums.keys())

    for term_name in reward_term_names:
        if term_name not in env.reward_manager._episode_sums:
            raise ValueError(f"Reward term '{term_name}' not found in reward manager episode sums.")
        episode_reward += env.reward_manager._episode_sums[term_name][env_ids_tensor]

    if normalize_by_episode_length:
        episode_reward = episode_reward / env.max_episode_length_s

    move_up = episode_reward > reward_threshold
    move_down = episode_reward < move_down_threshold
    move_down *= ~move_up

    terrain.update_env_origins(env_ids_tensor, move_up, move_down)

    return {
        "mean_episode_reward": float(torch.mean(episode_reward).item()),
        "mean_terrain_level": float(torch.mean(terrain.terrain_levels[env_ids_tensor].float()).item()),
        "move_up_ratio": float(move_up.float().mean().item()),
        "move_down_ratio": float(move_down.float().mean().item()),
    }

def actuator_fault_event_schedule(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
    start_failure_range: tuple[float,float] = (0.3,0.8),
    end_failure_range: tuple[float,float] = (0.1,0.6),
    num_epochs: int = 3000,
    steps_per_iteration: int = 24,
    event_name: str = "randomize_actuator_faults",
) -> dict[str, float] | None:
    """Gradually update the actuator fault event parameters over training.

    The schedule is linear in environment steps and is applied whenever curriculum terms are recomputed,
    which in manager-based RL environments happens on reset.
    """
    event_cfg = getattr(env.cfg.events, event_name, None)
    if event_cfg is None:
        return None

    total_steps = max(num_epochs * steps_per_iteration, 1) # 300*24
    progress = min(float(env.common_step_counter) / float(total_steps), 1.0)

    ratio = start_ratio + (end_ratio - start_ratio) * progress
    slb, sub = start_failure_range
    elb, eub = end_failure_range

    ub = (
        sub
        + (eub - sub) * progress
    )

    lb = (
        slb
        + (elb - slb) * progress
    )

    event_cfg.params["ratio"] = float(ratio)
    event_cfg.params["failure_range"] = (float(lb), float(ub))

    return {
        "progress": progress,
        "iteration_estimate": float(env.common_step_counter) / float(steps_per_iteration),
        "ratio": float(ratio),
        "failure_lower": float(lb),
        "failure_upper": float(ub),
    }
