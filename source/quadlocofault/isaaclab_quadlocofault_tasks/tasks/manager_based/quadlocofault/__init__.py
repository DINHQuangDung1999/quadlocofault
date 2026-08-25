# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registrations for Unitree Go2 fault-tolerant locomotion."""

import gymnasium as gym

from . import agents


_ENTRY_POINT = "isaaclab.envs:ManagerBasedRLEnv"
_AGENT_MODULE = f"{agents.__name__}.rsl_rl_ppo_cfg"


def _register(task_id: str, env_cfg: str, runner_cfg: str, terrain: str) -> None:
    """Register one task with the common environment and agent metadata."""
    gym.register(
        id=task_id,
        entry_point=_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{terrain}_env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{_AGENT_MODULE}:{runner_cfg}",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_{terrain}_ppo_cfg.yaml",
        },
    )


# Existing training/play IDs remain unchanged for checkpoint compatibility.
_POLICIES = {
    "Base": {
        "rough_env": "UnitreeGo2RoughEnvCfg",
        "flat_env": "UnitreeGo2FlatEnvCfg",
        "rough_runner": "UnitreeGo2RoughPPORunnerCfg",
        "flat_runner": "UnitreeGo2FlatPPORunnerCfg",
    },
    "FTNet": {
        "rough_env": "UnitreeGo2RoughFTNetEnvCfg",
        "flat_env": "UnitreeGo2FlatFTNetEnvCfg",
        "rough_runner": "UnitreeGo2RoughPPOFTNetRunnerCfg",
        "flat_runner": "UnitreeGo2FlatPPOFTNetRunnerCfg",
    },
    "FLEX": {
        "rough_env": "UnitreeGo2RoughFLEXEnvCfg",
        "flat_env": "UnitreeGo2FlatFLEXEnvCfg",
        "rough_runner": "UnitreeGo2RoughPPOFLEXRunnerCfg",
        "flat_runner": "UnitreeGo2FlatPPOFLEXRunnerCfg",
    },
    "PINN": {
        "rough_env": "UnitreeGo2RoughPINNEnvCfg",
        "flat_env": "UnitreeGo2FlatPINNEnvCfg",
        "rough_runner": "UnitreeGo2RoughPPOPINNRunnerCfg",
        "flat_runner": "UnitreeGo2FlatPPOPINNRunnerCfg",
    },
    "GCN": {
        "rough_env": "UnitreeGo2RoughGCNEnvCfg",
        "flat_env": "UnitreeGo2FlatGCNEnvCfg",
        "rough_runner": "UnitreeGo2RoughPPOGCNRunnerCfg",
        "flat_runner": "UnitreeGo2FlatPPOGCNRunnerCfg",
    },
}

for policy_name, cfg in _POLICIES.items():
    for terrain_name in ("flat", "rough"):
        # Every policy has train and play variants with the same runner.
        for play_suffix, env_suffix in (("", ""), ("-Play", "_PLAY")):
            _register(
                task_id=f"{policy_name}-Isaac-Velocity-{terrain_name.title()}-Unitree-Go2{play_suffix}-v0",
                env_cfg=f"{cfg[f'{terrain_name}_env']}{env_suffix}",
                runner_cfg=cfg[f"{terrain_name}_runner"],
                terrain=terrain_name,
            )


for play_suffix, env_suffix in (("", ""), ("-Play", "_PLAY")):
    _register(
        task_id=f"Oracle-Isaac-Velocity-Rough-Unitree-Go2{play_suffix}-v0",
        env_cfg=f"UnitreeGo2RoughOracleEnvCfg{env_suffix}",
        runner_cfg="UnitreeGo2RoughOraclePPORunnerCfg",
        terrain="rough",
    )
    _register(
        task_id=f"EquivGCN-Isaac-Velocity-Rough-Unitree-Go2{play_suffix}-v0",
        env_cfg=f"UnitreeGo2RoughEquivGCNEnvCfg{env_suffix}",
        runner_cfg="UnitreeGo2RoughPPOEquivGCNRunnerCfg",
        terrain="rough",
    )
    _register(
        task_id=f"EquivGCNMLP-Isaac-Velocity-Rough-Unitree-Go2{play_suffix}-v0",
        env_cfg=f"UnitreeGo2RoughEquivGCNEnvCfg{env_suffix}",
        runner_cfg="UnitreeGo2RoughPPOEquivGCNMLPRunnerCfg",
        terrain="rough",
    )


# Benchmark IDs share the same physical configuration. FTNet retains its
# paper-specific 49-D proprioception and separate privileged-physics group;
# all other architectures use the common observation configuration.
_EVAL_RUNNERS = {
    "GCN": "UnitreeGo2RoughPPOGCNRunnerCfg",
    "EquivGCN": "UnitreeGo2RoughPPOEquivGCNRunnerCfg",
    "EquivGCNMLP": "UnitreeGo2RoughPPOEquivGCNMLPRunnerCfg",
    "FTNet": "UnitreeGo2RoughPPOFTNetRunnerCfg",
    "FLEX": "UnitreeGo2RoughPPOFLEXRunnerCfg",
}
_EVAL_ENVS = {
    "FTNet": "UnitreeGo2EvaluationFTNetEnvCfg",
}
for policy_name, runner_cfg in _EVAL_RUNNERS.items():
    _register(
        task_id=f"{policy_name}-Isaac-Velocity-Eval-Unitree-Go2-v0",
        env_cfg=_EVAL_ENVS.get(policy_name, "UnitreeGo2EvaluationEnvCfg"),
        runner_cfg=runner_cfg,
        terrain="rough",
    )
