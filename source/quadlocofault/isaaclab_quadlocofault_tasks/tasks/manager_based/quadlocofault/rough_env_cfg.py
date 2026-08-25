# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree Go2 rough-terrain configurations.

Policy variants share robot dynamics, observations, actions, events, and
terminations. Only their training rewards are selected by the parent classes
in :mod:`velocity_env_cfg`. Evaluation uses one policy-independent config.
"""

from isaaclab.utils import configclass

from ..velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    LocomotionVelocityRoughFLEXEnvCfg,
    LocomotionVelocityRoughFTNetEnvCfg,
    LocomotionVelocityRoughGCNEnvCfg,
    LocomotionVelocityRoughOracleEnvCfg,
    LocomotionVelocityRoughPINNEnvCfg,
)


_ZERO_ROOT_VELOCITY = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}


def _configure_go2_rough(cfg) -> None:
    """Apply settings common to every Go2 rough-terrain policy."""
    cfg.actions.joint_pos.scale = 0.25
    cfg.events.push_robot = None
    cfg.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
    cfg.events.add_base_mass.params["asset_cfg"].body_names = "base"
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_base.params = {
        "pose_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "yaw": (-3.14, 3.14),
        },
        "velocity_range": _ZERO_ROOT_VELOCITY.copy(),
    }
    cfg.terminations.base_contact.params["sensor_cfg"].body_names = "base"


def _configure_play(cfg) -> None:
    """Apply lightweight settings shared by visualization configurations."""
    fault_event = cfg.events.randomize_actuator_faults
    fault_event.params.update(
        severe_fault_prob=1.0,
        failure_coef_severe=0.0,
        failure_coef_moderate=0.6,
        num_faults=1,
        apply_once_per_episode=True,
    )
    fault_event.interval_range_s = (2.0, 2.0)

    command = cfg.commands.base_velocity
    command.heading_command = False
    command.rel_heading_envs = 0.0
    command.rel_standing_envs = 0.0
    command.ranges.lin_vel_x = (0.7, 0.7)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)

    cfg.scene.num_envs = 50
    cfg.scene.env_spacing = 2.5
    cfg.scene.terrain.max_init_terrain_level = None
    terrain_generator = cfg.scene.terrain.terrain_generator
    if terrain_generator is not None:
        terrain_generator.num_rows = 5
        terrain_generator.num_cols = 5
        terrain_generator.curriculum = False
        terrain_generator.difficulty_range = (0.5, 0.5)

    cfg.observations.policy.enable_corruption = False
    cfg.observations.history.enable_corruption = False
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.actuator_faults = None
    cfg.terminations.base_contact.params["sensor_cfg"].body_names = [
        "base",
        "Head.*",
    ]


class _UnitreeGo2RoughMixin:
    def __post_init__(self):
        super().__post_init__()
        _configure_go2_rough(self)


@configclass
class UnitreeGo2RoughEnvCfg(_UnitreeGo2RoughMixin, LocomotionVelocityRoughEnvCfg):
    pass


@configclass
class UnitreeGo2RoughFTNetEnvCfg(_UnitreeGo2RoughMixin, LocomotionVelocityRoughFTNetEnvCfg):
    pass


@configclass
class UnitreeGo2RoughPINNEnvCfg(_UnitreeGo2RoughMixin, LocomotionVelocityRoughPINNEnvCfg):
    pass


@configclass
class UnitreeGo2RoughFLEXEnvCfg(_UnitreeGo2RoughMixin, LocomotionVelocityRoughFLEXEnvCfg):
    pass


@configclass
class UnitreeGo2RoughGCNEnvCfg(_UnitreeGo2RoughMixin, LocomotionVelocityRoughGCNEnvCfg):
    pass


@configclass
class UnitreeGo2RoughEquivGCNEnvCfg(UnitreeGo2RoughGCNEnvCfg):
    pass


@configclass
class UnitreeGo2RoughOracleEnvCfg(_UnitreeGo2RoughMixin, LocomotionVelocityRoughOracleEnvCfg):
    pass


class _PlayMixin:
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)


@configclass
class UnitreeGo2RoughEnvCfg_PLAY(_PlayMixin, UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.7)


@configclass
class UnitreeGo2RoughFTNetEnvCfg_PLAY(_PlayMixin, UnitreeGo2RoughFTNetEnvCfg):
    pass

@configclass
class UnitreeGo2RoughPINNEnvCfg_PLAY(_PlayMixin, UnitreeGo2RoughPINNEnvCfg):
    pass

@configclass
class UnitreeGo2RoughFLEXEnvCfg_PLAY(_PlayMixin, UnitreeGo2RoughFLEXEnvCfg):
    pass


@configclass
class UnitreeGo2RoughGCNEnvCfg_PLAY(_PlayMixin, UnitreeGo2RoughGCNEnvCfg):
    pass

@configclass
class UnitreeGo2RoughEquivGCNEnvCfg_PLAY(UnitreeGo2RoughGCNEnvCfg_PLAY):
    pass

@configclass
class UnitreeGo2RoughOracleEnvCfg_PLAY(_PlayMixin, UnitreeGo2RoughOracleEnvCfg):
    pass

class _EvaluationMixin:
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False
        self.observations.history.enable_corruption = False
        self.curriculum.terrain_levels = None
        self.curriculum.actuator_faults = None
        self.events.push_robot = None
        self.events.base_external_force_torque = None


@configclass
class UnitreeGo2EvaluationEnvCfg(_EvaluationMixin, UnitreeGo2RoughEnvCfg):
    """Common 45-D evaluation environment used by non-FTNet policies."""

    pass


@configclass
class UnitreeGo2EvaluationFTNetEnvCfg(_EvaluationMixin, UnitreeGo2RoughFTNetEnvCfg):
    """Evaluation environment retaining FTNet's 49-D and physical inputs."""

    pass
