# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ..velocity_env_cfg import LocomotionVelocityRoughEnvCfg, \
    LocomotionVelocityRoughFTNetEnvCfg, \
    LocomotionVelocityRoughPINNEnvCfg, \
    LocomotionVelocityRoughFLEXEnvCfg, \
    LocomotionVelocityRoughGCNEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        # self.events.randomize_actuator_gains = None
        # self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 5.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["force_range"] = (0.0, 10.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }


        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.events.randomize_actuator_faults.params["severe_fault_prob"] = 1.0
        self.events.randomize_actuator_faults.params["failure_coef_severe"] = 0.0
        self.events.randomize_actuator_faults.params["failure_coef_moderate"] = 0.6
        self.events.randomize_actuator_faults.params["num_faults"] = 1
        self.events.randomize_actuator_faults.interval_range_s=(3.0, 5.0)
        # self.events.randomize_actuator_faults = None
        self.commands.base_velocity.ranges.lin_vel_x = (1.0,1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0,0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0,0.0)
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.difficulty_range = (0.5, 0.5)
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.05, 0.15)
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.05, 0.15)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None

@configclass
class UnitreeGo2RoughFTNetEnvCfg(LocomotionVelocityRoughFTNetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.events.set_actuator_faults = None
        # self.observations.history.history_length = 30
        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        # self.events.randomize_actuator_gains = None
        # self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 5.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        # self.events.base_external_force_torque.params["force_range"] = (0.0, 10.0)
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
        # self.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)

@configclass
class UnitreeGo2RoughFTNetEnvCfg_PLAY(UnitreeGo2RoughFTNetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.events.randomize_actuator_faults.params["severe_fault_prob"] = 1.0
        self.events.randomize_actuator_faults.params["failure_coef_severe"] = 0.0
        self.events.randomize_actuator_faults.params["failure_coef_moderate"] = 0.6
        self.events.randomize_actuator_faults.params["num_faults"] = 1
        self.events.randomize_actuator_faults.interval_range_s=(3.0, 5.0)
        # self.events.randomize_actuator_faults = None
        self.commands.base_velocity.ranges.lin_vel_x = (1.0,1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0,0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0,0.0)
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.difficulty_range = (0.5, 0.5)
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.05, 0.15)
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.05, 0.15)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None


@configclass
class UnitreeGo2RoughPINNEnvCfg(LocomotionVelocityRoughPINNEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.events.set_actuator_faults = None
        # scale down the terrains because the robot is small
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.05)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        # self.events.randomize_actuator_gains = None
        # self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 5.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["force_range"] = (0.0, 10.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

@configclass
class UnitreeGo2RoughPINNEnvCfg_PLAY(UnitreeGo2RoughPINNEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.events.randomize_actuator_faults.params["severe_fault_prob"] = 1.0
        self.events.randomize_actuator_faults.params["failure_coef_severe"] = 0.0
        self.events.randomize_actuator_faults.params["failure_coef_moderate"] = 0.01
        self.events.randomize_actuator_faults.params["num_faults"] = 1
        self.events.randomize_actuator_faults.interval_range_s=(3.0, 5.0)
        # self.events.randomize_actuator_faults = None
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.difficulty_range = (1.0, 1.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None

@configclass
class UnitreeGo2RoughFLEXEnvCfg(LocomotionVelocityRoughFLEXEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.events.set_actuator_faults = None
        # scale down the terrains because the robot is small
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.05)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        # self.events.randomize_actuator_gains = None
        # self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 5.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["force_range"] = (0.0, 10.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

@configclass
class UnitreeGo2RoughFLEXEnvCfg_PLAY(UnitreeGo2RoughFLEXEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.events.randomize_actuator_faults.params["severe_fault_prob"] = 1.0
        self.events.randomize_actuator_faults.params["failure_coef_severe"] = 0.0
        self.events.randomize_actuator_faults.params["failure_coef_moderate"] = 0.6
        self.events.randomize_actuator_faults.params["num_faults"] = 1
        self.events.randomize_actuator_faults.interval_range_s=(3.0, 5.0)
        # self.events.randomize_actuator_faults = None
        self.commands.base_velocity.ranges.lin_vel_x = (1.0,1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0,0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0,0.0)
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.difficulty_range = (0.5, 0.5)
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.05, 0.15)
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.05, 0.15)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None

@configclass
class UnitreeGo2RoughGCNEnvCfg(LocomotionVelocityRoughGCNEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.observations.history.history_length = 5

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        # self.events.randomize_actuator_gains = None
        # self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 5.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["force_range"] = (0.0, 10.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

@configclass
class UnitreeGo2RoughGCNEnvCfg_PLAY(UnitreeGo2RoughGCNEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # post init of parent
        self.events.randomize_actuator_faults.params["severe_fault_prob"] = 1.0
        self.events.randomize_actuator_faults.params["failure_coef_severe"] = 0.0
        self.events.randomize_actuator_faults.params["failure_coef_moderate"] = 0.6
        self.events.randomize_actuator_faults.params["num_faults"] = 1
        self.events.randomize_actuator_faults.interval_range_s=(3.0, 5.0)
        # self.events.randomize_actuator_faults = None
        self.commands.base_velocity.ranges.lin_vel_x = (1.0,1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0,0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0,0.0)
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.difficulty_range = (0.5, 0.5)
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.05, 0.15)
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.05, 0.15)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None
