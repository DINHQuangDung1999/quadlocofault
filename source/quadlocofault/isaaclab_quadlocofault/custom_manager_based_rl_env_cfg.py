# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

@configclass
class CustomManagerBasedRLEnvCfg(ManagerBasedRLEnvCfg):
    pass