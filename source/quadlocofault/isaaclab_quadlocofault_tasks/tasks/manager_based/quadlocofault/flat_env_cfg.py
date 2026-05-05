# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

# from .rough_env_cfg import UnitreeGo2RoughEnvCfg, UnitreeGo2RoughPINNEnvCfg, UnitreeGo2RoughFLEXEnvCfg
from ..velocity_env_cfg import LocomotionVelocityRoughEnvCfg, LocomotionVelocityRoughFTNetEnvCfg, LocomotionVelocityRoughPINNEnvCfg, LocomotionVelocityRoughFLEXEnvCfg

@configclass
class UnitreeGo2FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.observations.history.prop_observations.params["num_hist"] = 30

        # self.events.set_actuator_faults = None
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

        # rewards
        # self.rewards.track_lin_vel_xy_exp.weight = 1.0
        # self.rewards.track_ang_vel_z_exp.weight = 0.5
        # self.rewards.lin_vel_z_l2.weight = -2.0
        # self.rewards.ang_vel_xy_l2.weight = -0.01
        # self.rewards.dof_torques_l2.weight = -2e-5
        # self.rewards.dof_power_l2.weight = -2e-5
        # self.rewards.dof_acc_l2.weight = -2.5e-7 
           
        # self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        # self.rewards.feet_air_time.weight = 0.25
        # self.rewards.flat_orientation_l2.weight = -2.5
        # self.rewards.undesired_contacts = None


        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

@configclass
class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        self.events.randomize_actuator_faults = None
        # self.events.randomize_actuator_faults.params["ratio"] = 1.0
        # self.events.randomize_actuator_faults.params["failure_range"] = 0.0
        self.events.set_actuator_faults.params["joint"] = 0
        self.events.set_actuator_faults.params["ratio"] = 0.3
        self.events.set_actuator_faults.interval_range_s=(0.0, 2.0)
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

@configclass
class UnitreeGo2FlatFTNetEnvCfg(LocomotionVelocityRoughFTNetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.observations.history.prop_observations.params["num_hist"] = 30

        # self.events.set_actuator_faults = None
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

        # rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.01
        # self.rewards.dof_torques_l2.weight = -2e-5
        # self.rewards.dof_power_l2.weight = -2e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7 
           
        # self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        # self.rewards.feet_air_time.weight = 0.25
        # self.rewards.flat_orientation_l2.weight = -2.5
        # self.rewards.undesired_contacts = None


        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

@configclass
class UnitreeGo2FlatFTNetEnvCfg_PLAY(UnitreeGo2FlatFTNetEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        self.events.randomize_actuator_faults = None
        # self.events.randomize_actuator_faults.params["ratio"] = 1.0
        # self.events.randomize_actuator_faults.params["failure_range"] = 0.0
        self.events.set_actuator_faults.params["joint"] = 0
        self.events.set_actuator_faults.params["ratio"] = 0.3
        self.events.set_actuator_faults.interval_range_s=(0.0, 2.0)
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

@configclass
class UnitreeGo2FlatPINNEnvCfg(LocomotionVelocityRoughPINNEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # self.events.set_actuator_faults = None
        # reduce action scale
        self.actions.joint_pos.scale = 0.25
        # self.events.randomize_actuator_faults.params["ratio"] = 1.0
        # self.events.randomize_actuator_faults.params["failure_range"] = (0.0, 0.0)
        # self.events.randomize_actuator_faults.params["num_faults"] = 1
        # self.events.randomize_actuator_faults.interval_range_s=(0.0, 0.0)
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

        # rewards
        # self.rewards.track_lin_vel_xy_exp.weight = 1.0
        # self.rewards.track_ang_vel_z_exp.weight = 0.5
        # self.rewards.lin_vel_z_l2.weight = -2.0
        # self.rewards.ang_vel_xy_l2.weight = -0.01
        # self.rewards.dof_torques_l2.weight = -2e-5
        # self.rewards.dof_power_l2.weight = -2e-5
        # self.rewards.dof_acc_l2.weight = -2.5e-7 
           
        # self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        # self.rewards.feet_air_time.weight = 0.25
        # self.rewards.flat_orientation_l2.weight = -2.5
        # self.rewards.undesired_contacts = None


        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # self.events.randomize_actuator_faults.params["ratio"] = 1.0
        # self.events.randomize_actuator_faults.params["failure_range"] = 0.0
        # self.events.randomize_actuator_faults.interval_range_s=(2.0, 5.0)
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

@configclass
class UnitreeGo2FlatPINNEnvCfg_PLAY(UnitreeGo2FlatPINNEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        # self.events.randomize_actuator_faults = None
        self.events.randomize_actuator_faults.params["ratio"] = 1.0
        self.events.randomize_actuator_faults.params["failure_range"] = (0.0, 0.01)
        self.events.randomize_actuator_faults.params["num_faults"] = 1
        self.events.randomize_actuator_faults.interval_range_s=(3.0, 5.0)
        # self.events.set_actuator_faults.params["joint"] = 0
        # self.events.set_actuator_faults.params["ratio"] = 0.3
        # self.events.set_actuator_faults.interval_range_s=(0.0, 2.0)

        self.episode_length_s = 10.0
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

@configclass
class UnitreeGo2FlatFLEXEnvCfg(LocomotionVelocityRoughFLEXEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # self.events.set_actuator_faults = None
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

        # rewards
        # self.rewards.track_lin_vel_xy_exp.weight = 1.0
        # self.rewards.track_ang_vel_z_exp.weight = 0.5
        # self.rewards.lin_vel_z_l2.weight = -2.0
        # self.rewards.ang_vel_xy_l2.weight = -0.01
        # self.rewards.dof_torques_l2.weight = -2e-5
        # self.rewards.dof_power_l2.weight = -2e-5
        # self.rewards.dof_acc_l2.weight = -2.5e-7 
           
        # self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        # self.rewards.feet_air_time.weight = 0.25
        # self.rewards.flat_orientation_l2.weight = -2.5
        # self.rewards.undesired_contacts = None


        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # self.events.randomize_actuator_faults.params["ratio"] = 1.0
        # self.events.randomize_actuator_faults.params["failure_range"] = 0.0
        # self.events.randomize_actuator_faults.interval_range_s=(2.0, 5.0)
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        # self.scene.height_scanner = None
        # self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

@configclass
class UnitreeGo2FlatFLEXEnvCfg_PLAY(UnitreeGo2FlatFLEXEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        self.events.randomize_actuator_faults = None
        # self.events.randomize_actuator_faults.params["ratio"] = 1.0
        # self.events.randomize_actuator_faults.params["failure_range"] = 0.0
        # self.events.randomize_actuator_faults.interval_range_s=(2.0, 5.0)
        self.events.set_actuator_faults.params["joint"] = 0
        self.events.set_actuator_faults.params["ratio"] = 0.3
        self.events.set_actuator_faults.interval_range_s=(0.0, 2.0)
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
