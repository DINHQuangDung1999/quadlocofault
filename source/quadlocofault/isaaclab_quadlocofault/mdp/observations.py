# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.sensors import ContactSensor, RayCaster, RayCasterCamera
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class CustomProprioceptiveObservations(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, 
                 env: ManagerBasedRLEnv, 
                 num_action: int = 12,
                 ):
        super().__init__(cfg, env)
        self.num_action = num_action

        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensors['contact_forces']
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self._prev_action = torch.zeros(self.num_envs, self.num_action, device=self.device)

        self.foot_ids, _ = self.asset.find_bodies(".*_foot", preserve_order=True)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._prev_action[env_ids, :] = 0.

    def __call__(
        self,
        env: ManagerBasedRLEnv,        
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        ) -> torch.Tensor:

        forces_w = self.contact_sensor.data.net_forces_w
        foot_forces_w = forces_w[:, self.foot_ids, :]
        foot_contact_boolean = torch.where(foot_forces_w.norm(dim=-1) > 0.0, 1.0, 0.0)
        commands = env.command_manager.get_command('base_velocity')
        action = env.action_manager.get_term("joint_pos").raw_actions

        prop_obs = torch.cat((
                            self.asset.data.joint_pos - self.asset.data.default_joint_pos,
                            self.asset.data.joint_vel,
                            self.asset.data.root_ang_vel_b, 
                            self.asset.data.projected_gravity_b,
                            commands,
                            self._prev_action,
                            foot_contact_boolean,
                            ),dim=-1)

        self._prev_action = action
        return prop_obs

class CustomPrivilegedObservations(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, 
                 env: ManagerBasedRLEnv, 
                 num_action: int = 12,
                 ):
        super().__init__(cfg, env)
        self.num_action = num_action

        self.contact_sensor: ContactSensor = env.scene.sensors['contact_forces']
        self.ray_sensor: RayCaster = env.scene.sensors['height_scanner']
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self._prev_action = torch.zeros(self.num_envs, self.num_action, device=self.device)
        # self._prev_motors_strength = torch.ones(self.num_envs, self.num_action, device=self.device)
        # self._prev_faulty_joint_idx = torch.zeros(self.num_envs, self.num_action, device=self.device)
        self.motors_strength = torch.ones(self.num_envs, self.num_action, device=self.device)
        self.faulty_joint_idx = torch.zeros(self.num_envs, self.num_action, device=self.device)

        self.body_id = self.asset.find_bodies('base')[0]
        self.foot_ids, _ = self.asset.find_bodies(".*_foot", preserve_order=True)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._prev_action[env_ids, :] = 0.

    def __call__(
        self,
        env: ManagerBasedRLEnv,      
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        ) -> torch.Tensor:

        forces_w = self.contact_sensor.data.net_forces_w
        foot_forces_w = forces_w[:, self.foot_ids, :]
        foot_contact_boolean = torch.where(foot_forces_w.norm(dim=-1) > 0.0, 1.0, 0.0)        
        commands = env.command_manager.get_command('base_velocity')
        action = env.action_manager.get_term("joint_pos").raw_actions
        measured_heights = self._get_heights()
        priv_phys = self._get_priv_phys(env)

        obs_buf = torch.cat((
                            self.asset.data.joint_pos - self.asset.data.default_joint_pos,
                            self.asset.data.joint_vel,
                            self.asset.data.root_ang_vel_b,   #[1,3] 0~2
                            self.asset.data.projected_gravity_b,
                            commands,
                            self._prev_action,
                            foot_contact_boolean,
                            ),dim=-1)

                     
        observations = torch.cat([obs_buf, #49
                                  measured_heights, # 187
                                  priv_phys, #34
                                  ],dim=-1)

        self._prev_action = action
        return observations 
    
    def _get_priv_phys(
        self,
        env
        ):
        body_lin_vel = self.asset.data.root_lin_vel_b
        # ex_force = self.asset._external_force_b[:, 0, :]
        body_mass = self.asset.root_physx_view.get_masses()[:,self.body_id].to(self.device)
        body_com = self.asset.data.com_pos_b[:,self.body_id,:].to(self.device).squeeze(1)
        mass_params_tensor = torch.cat([body_mass, body_com],dim=-1).to(self.device)
        friction_coeffs_tensor = self.asset.root_physx_view.get_material_properties()[:, 0, 0]
        if hasattr(self.asset, "motors_strength"):
            self.motors_strength = self.asset.motors_strength.clone()
        if hasattr(self.asset, "faulty_joint_idx"):
            self.faulty_joint_idx = self.asset.faulty_joint_idx.clone()
        applied_torque = self.asset.data.applied_torque.clone()
        # breakpoint()
        # default_stiffness = self.asset.data.default_joint_stiffness[:,:1]
        # default_damping = self.asset.data.default_joint_damping[:,:1]
        priv_phys = torch.cat((
            body_lin_vel,
            # ex_force,
            mass_params_tensor,
            friction_coeffs_tensor.unsqueeze(1).to(self.device),
            # self._prev_motors_strength,
            # self._prev_faulty_joint_idx,
            applied_torque,
            # default_stiffness,
            # default_damping,
            self.motors_strength,
            self.faulty_joint_idx,
        ), dim=-1).to(self.device)    
        # breakpoint()
        return priv_phys
    
    def _get_heights(self):
        return torch.clip(self.ray_sensor.data.pos_w[:, 2].unsqueeze(1) - self.ray_sensor.data.ray_hits_w[..., 2] - 0.3, -1, 1).to(self.device)

class CustomProprioceptionHistory(ManagerTermBase):

    def __init__(self, cfg: ObservationTermCfg, 
                 env: ManagerBasedRLEnv, 
                 history_length: int = 30, 
                 num_prop: int = 49,
                 num_action: int = 12,
                 ):
        super().__init__(cfg, env)
        self.history_length = history_length
        self.num_prop = num_prop
        self.num_action = num_action

        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensors['contact_forces']
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self._obs_history_buffer = torch.zeros(self.num_envs, self.history_length, self.num_prop, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, self.num_action, device=self.device)

        self.body_id = self.asset.find_bodies('base')[0]
        self.foot_ids, _ = self.asset.find_bodies(".*_foot", preserve_order=True)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._obs_history_buffer[env_ids, :, :] = 0. 
        self._prev_action[env_ids, :] = 0.

    def __call__(
        self,
        env: ManagerBasedRLEnv,      
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        num_hist: int = 5,
        ) -> torch.Tensor:

        forces_w = self.contact_sensor.data.net_forces_w
        foot_forces_w = forces_w[:, self.foot_ids, :]
        foot_contact_boolean = torch.where(foot_forces_w.norm(dim=-1) > 0.0, 1.0, 0.0)
        commands = env.command_manager.get_command('base_velocity')
        action = env.action_manager.get_term("joint_pos").raw_actions

        obs_buf = torch.cat((
                            self.asset.data.joint_pos - self.asset.data.default_joint_pos,
                            self.asset.data.joint_vel,
                            self.asset.data.root_ang_vel_b,   #[1,3] 0~2
                            self.asset.data.projected_gravity_b,
                            commands,
                            self._prev_action,
                            foot_contact_boolean
                            ),dim=-1)

        self._obs_history_buffer = torch.where(
            (env.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.history_length, dim=1),
            torch.cat([
                self._obs_history_buffer[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )
        self._prev_action = action
        return self._obs_history_buffer[:, -num_hist:]
        