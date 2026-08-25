# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_quadlocofault.mdp as mdp
from isaaclab_quadlocofault.actuators import CustomDCMotorCfg
##
# Pre-defined configs
##
from isaaclab_quadlocofault.terrains import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot", actuators={
        "base_legs": CustomDCMotorCfg(
        # "base_legs": DelayedCustomDCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
            # min_delay=0,
            # max_delay=6,
        ),
    })
    
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", 
                                      history_length=3, 
                                      track_air_time=True, 
                                      debug_vis= False,
                                      force_threshold=1.
                                      )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )
# @configclass
# class CommandsCfg:
#     """Command specifications for the MDP."""

#     base_velocity = mdp.UniformLevelVelocityCommandCfg(
#         asset_name="robot",
#         resampling_time_range=(10.0, 10.0),
#         rel_standing_envs=0.05,
#         rel_heading_envs=1.0,
#         heading_command=True,
#         heading_control_stiffness=0.5,
#         debug_vis=True,
#         ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
#             lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-1, 1), heading=(-math.pi, math.pi)
#         ),
#         limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
#             lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
#         ),
#     )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.FaultClampJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        constraint_limits={
            ".*_hip_joint": (-0.5, 0.5),
            ".*_thigh_joint": (0.3, 1.3),
            ".*_calf_joint": (-2.1, -0.8),
        },
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            clip=(-100, 100), 
                            noise=Unoise(n_min=-0.01, n_max=0.01)
                            )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel,
                            scale=0.05,
                            clip=(-100, 100), 
                            noise=Unoise(n_min=-1.5, n_max=1.5)
                            )
        actions = ObsTerm(func=mdp.last_action, 
                          clip=(-100, 100)
                          )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, 
                               scale=0.2, 
                               clip=(-100, 100), 
                               noise=Unoise(n_min=-0.2, n_max=0.2)
                               )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, 
            clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, 
                                    clip=(-100, 100), 
                                    params={"command_name": "base_velocity"})
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for policy group."""
        # base observation terms
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            clip=(-100, 100))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel,
                            scale=0.05,
                            clip=(-100, 100))
        actions = ObsTerm(func=mdp.last_action,
                            clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, 
                               scale=0.2, 
                               clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, 
                               clip=(-100, 100))
        velocity_commands = ObsTerm(func=mdp.generated_commands, 
                               clip=(-100, 100), 
                               params={"command_name": "base_velocity"})
        # privileged observation terms
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, 
                               clip=(-100, 100))
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )     
        body_mass_params = ObsTerm(func=mdp.body_mass_params)
        friction_coeffs = ObsTerm(func=mdp.friction_coeffs)
        motor_strengths = ObsTerm(func=mdp.motor_strengths)
        faulty_joints = ObsTerm(func=mdp.faulty_joints)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HistoryCfg(ObsGroup):
        """Observations for policy group."""
        history_length = 30
        flatten_history_dim=False
        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            clip=(-100, 100), 
                            noise=Unoise(n_min=-0.01, n_max=0.01)
                            )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel,
                            scale=0.05,
                            clip=(-100, 100), 
                            noise=Unoise(n_min=-1.5, n_max=1.5)
                            )
        actions = ObsTerm(func=mdp.last_action, 
                          clip=(-100, 100)
                          )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, 
                               scale=0.2, 
                               clip=(-100, 100), 
                               noise=Unoise(n_min=-0.2, n_max=0.2)
                               )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, 
            clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, 
                                    clip=(-100, 100), 
                                    params={"command_name": "base_velocity"})
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    history: HistoryCfg = HistoryCfg()


@configclass
class FTNetObservationsCfg(ObservationsCfg):
    """FT-Net observations matching its separate actor/adaptor inputs.

    The policy and history contain the paper's 49-D proprioception, while the
    physical encoder receives only simulator-only environment parameters.
    The critic retains the complete privileged observation used for value
    estimation.
    """

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        foot_contacts = ObsTerm(
            func=mdp.foot_contact_boolean,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                ),
                "threshold": 1.0,
            },
        )

    @configclass
    class HistoryCfg(ObservationsCfg.HistoryCfg):
        foot_contacts = ObsTerm(
            func=mdp.foot_contact_boolean,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                ),
                "threshold": 1.0,
            },
        )

    @configclass
    class PrivilegedPhysicsCfg(ObsGroup):
        """Ground-truth physical parameters supplied only to E_theta."""

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
        )
        body_mass_params = ObsTerm(func=mdp.body_mass_params)
        friction_coeffs = ObsTerm(func=mdp.friction_coeffs)
        # The location and severity of a fault are represented by which entry
        # of this 12-D motor-strength vector is reduced and by how much.
        motor_strengths = ObsTerm(func=mdp.motor_strengths)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: ObservationsCfg.CriticCfg = ObservationsCfg.CriticCfg()
    history: HistoryCfg = HistoryCfg()
    privileged: PrivilegedPhysicsCfg = PrivilegedPhysicsCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    reset_actuator_gains = EventTerm(
        func= mdp.reset_actuator_gains,
        params={
            "asset_cfg" :SceneEntityCfg("robot", joint_names=".*"),
            "motors_strength_range": (0.9, 1.1),
            },
        mode="reset",
    )
    randomize_actuator_faults = EventTerm(
        func= mdp.randomize_actuator_faults,
        params={
            "asset_cfg" :SceneEntityCfg("robot", joint_names=".*"),
            "severe_fault_prob": 0.5,
            "failure_coef_severe": 0.3,
            "failure_coef_moderate": 0.8,
            "num_faults": 1,
            },
        mode="interval",
        interval_range_s=(3.0, 8.0),
    )

@configclass
class RewardsCfg:
    """Shared base/GCN reward terms for the MDP."""
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=1.0, 
        params={
            "command_name": "base_velocity", 
            "std": math.sqrt(0.25)
            })
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=0.5, 
        params={
            "command_name": "base_velocity", 
            "std": math.sqrt(0.25)
            })
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.01)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2, 
        weight=-1.0,
        params={
            "target_height": 0.32,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        })
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.6,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "threshold": 0.25,
            "ignore_faulty_legs": True,
        },
    )
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # hip_position_deviation_l1 = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.05,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", joint_names=".*_hip_joint"
    #         )
    #     },
    # )
    # undesired_leg_link_contact = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-0.2,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces", body_names=".*_(thigh|calf)"
    #         ),
    #         "threshold": 5.0,
    #     },
    # )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.025,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "ignore_faulty_legs": True,
        },
    )
    # VHIP_style = RewTerm(
    #     func=mdp.vhip_style_reward_ftnet,
    #     weight=1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces"),
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "contact_threshold": 1.0,
    #         "theta_scale": -0.015,
    #         "theta_ddot_scale": -0.01,
    #         "support_dist_scale": -0.01,
    #     },
    # )
    faulty_leg_vertical_load = RewTerm(
        func=mdp.faulty_leg_vertical_load,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_foot"
            ),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    faulty_foot_planar_velocity = RewTerm(
        func=mdp.faulty_foot_planar_velocity_l2,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    hip_fault_thigh_calf_velocity = RewTerm(
        func=mdp.hip_fault_thigh_calf_velocity_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # A lateral-position penalty was tested conceptually but is not enabled:
    # complete hip failure makes some passive folding unavoidable, so forcing
    # the foot back to its nominal lateral position can create an infeasible
    # objective. The implementation remains available as
    # mdp.FaultyHipFootLateralDeviationL2 for future comparisons.
    # faulty_hip_foot_lateral_deviation = RewTerm(
    #     func=mdp.FaultyHipFootLateralDeviationL2,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward_dreamflex,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_height": 0.12,
        },
    )
    faulty_leg_link_contact = RewTerm(
        func=mdp.faulty_leg_link_contact_reward,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_(thigh|calf)"
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 1.0,
        },
    )
@configclass
class FTNetRewardsCfg:
    """Standalone reward configuration matching FT-Net Table I."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.6,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
            "ignore_faulty_legs": False,
        },
    )
    VHIP_style = RewTerm(
        func=mdp.vhip_style_reward_ftnet,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_threshold": 1.0,
            "theta_scale": -0.015,
            "theta_ddot_scale": -0.01,
            "support_dist_scale": -0.01,
        })
    joint_motion_cosmetic = RewTerm(
        func=mdp.joint_motion_cosmetic,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        }
    )

@configclass
class DreamFLEXRewardsCfg:
    """Standalone DreamFLEX reward configuration.

    The task/style terms are stated explicitly so future changes to the base
    or FTNet reward configurations cannot silently change DreamFLEX. The final
    five terms implement the fault-tolerant rewards from DreamFLEX Table I.
    """

    # Common task rewards.
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # Common locomotion/style rewards used by this implementation.
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness = RewTerm(
        func=mdp.DreamWaQActionSmoothnessPenalty,
        weight=-0.01,
        params={"action_name": "joint_pos"},
    )
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "target_height": 0.30,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)
    power_distribution = RewTerm(
        func=mdp.power_distribution,
        weight=-1e-5,
    )
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-2e-5,
    )

    # DreamFLEX Table I: apply gait-shaping terms only to healthy legs.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
            "ignore_faulty_legs": True,
        },
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward_dreamflex,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_height": 0.12,
        },
    )
    raibert = RewTerm(
        func=mdp.RaibertFootPlacementReward,
        weight=-1.0e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "stance_time": 0.20,
            "contact_threshold": 1.0,
        },
    )

    # Previous phase-free approximation, retained for reference. It used fixed
    # nominal footholds and a blend of commanded and measured velocity instead
    # of the paper's actual hip position and measured CoM velocity.
    # raibert = RewTerm(
    #     func=mdp.raibert_foot_placement_reward_approximation,
    #     weight=-1.0e-5,
    #     params={
    #         "command_name": "base_velocity",
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    #         ),
    #         "stance_time": 0.20,
    #         "nominal_foot_positions_xy": [
    #             [0.20,  0.13],
    #             [0.20, -0.13],
    #             [-0.20,  0.13],
    #             [-0.20, -0.13],
    #         ],
    #         "vel_gain": 0.05,
    #     },
    # )

    fault_leg_motion = RewTerm(
        func=mdp.faulty_joint_motion_reward_dreamflex,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "action_name": "joint_pos",
        },
    )
    faulty_leg_contact = RewTerm(
        func=mdp.faulty_leg_contact_reward,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 1.0,
        },
    )
    # faulty_leg_link_contact = RewTerm(
    #     func=mdp.faulty_leg_link_contact_reward,
    #     weight=-0.2,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces", body_names=".*_(thigh|calf)"
    #         ),
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "threshold": 1.0,
    #     },
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.pi/2},
    )
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels,
    #     params={
    #         "reward_term_names": ("track_lin_vel_xy_exp")
    #     }
    # )
    
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_episode_reward_event,
        params={
            "move_up_reward_threshold": 24.0,
            "move_down_reward_threshold": 12.0,
            "successes_per_level": 3,
            "failures_per_level": 1,
            "reward_term_names": ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"),
            # "reward_term_names": None,
        },
    )
    # Previous version:
    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    actuator_faults = CurrTerm(
        func=mdp.actuator_fault_episode_reward_event,
        params={
            "start_severe_fault_prob": 0.2,
            "end_severe_fault_prob": 0.6,
            "start_failure_range": (0.3, 1.0),
            "end_failure_range": (0.1, 0.8),
            "move_up_reward_threshold": 24.0,
            "reward_term_names": (
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
            ),
            "num_levels": 10,
            "successes_per_level": 3,
            "event_name": "randomize_actuator_faults",
        },
    )
    # Linear time-based alternative:
    # actuator_faults = CurrTerm(
    #     func=mdp.actuator_fault_event_schedule_linear,
    #     params={
    #         "start_ratio": 0.5,
    #         "end_ratio": 1.0,
    #         "start_failure_range": (0.3, 1.0),
    #         "end_failure_range": (0.1, 0.6),
    #         "num_epochs": 2000,
    #         "steps_per_iteration": 24,
    #         "event_name": "randomize_actuator_faults",
    #     },
    # )


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    only_positive_rewards: bool = False
    # ### Behind and above
    viewer = ViewerCfg(
        eye=(-2.0, -2.0, 1.0),
        lookat=(0.0, 0.0, 0.3),
        asset_name="robot",
        origin_type="asset_root",
    )
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.run_data_collection = False
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
        # https://github.com/CAI23sbP/Isaaclab_Parkour/blob/master/parkour_tasks/parkour_tasks/default_cfg.py
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        # self.scene.robot.actuators['base_legs'] = CustomDCMotorCfg(
        #     joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        #     effort_limit={
        #                 '.*_hip_joint':35.0,
        #                 '.*_thigh_joint':40.0,
        #                 '.*_calf_joint':40.0,
        #                 },
        #     saturation_effort={
        #                 '.*_hip_joint':35.0,
        #                 '.*_thigh_joint':45.0,
        #                 '.*_calf_joint':45.0,
        #                 },
        #     velocity_limit={
        #                 '.*_hip_joint':52.4,
        #                 '.*_thigh_joint':30.1,
        #                 '.*_calf_joint':30.1,
        #                 },
        #     stiffness=40.0,
        #     damping=1.0,
        #     friction=0.0,
        # )
        # actuators={
        #     "base_legs": DCMotorCfg(
        #         joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        #         effort_limit=23.5,
        #         saturation_effort=23.5,
        #         velocity_limit=30.0,
        #         stiffness=25.0,
        #         damping=0.5,
        #         friction=0.0,
        #     ),
        # },


@configclass
class LocomotionVelocityRoughFTNetEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations = FTNetObservationsCfg()
    rewards = FTNetRewardsCfg()
    only_positive_rewards = False

@configclass
class LocomotionVelocityRoughPINNEnvCfg(LocomotionVelocityRoughEnvCfg):
    pass

@configclass
class LocomotionVelocityRoughFLEXEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards = DreamFLEXRewardsCfg()
    only_positive_rewards = False

    def __post_init__(self):
        super().__post_init__()
        # DreamFLEX uses the current observation plus N=5 historical frames.
        self.observations.history.history_length = 5

@configclass
class LocomotionVelocityRoughGCNEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards = RewardsCfg()
    # only_positive_rewards = True

@configclass
class LocomotionVelocityRoughOracleEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy = self.observations.CriticCfg()
        self.observations.critic = None
        self.observations.history = None
