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

    _register(
        task_id=f"EquivGCNMLPConcat-Isaac-Velocity-Rough-Unitree-Go2{play_suffix}-v0",
        env_cfg=f"UnitreeGo2RoughEquivGCNEnvCfg{env_suffix}",
        runner_cfg="UnitreeGo2RoughPPOEquivGCNMLPConcatRunnerCfg",
        terrain="rough",
    )

    _register(
        task_id=f"GCNMLP-Isaac-Velocity-Rough-Unitree-Go2{play_suffix}-v0",
        env_cfg=f"UnitreeGo2RoughEquivGCNEnvCfg{env_suffix}",
        runner_cfg="UnitreeGo2RoughPPOGCNMLPRunnerCfg",
        terrain="rough",
    )

    _register(
        task_id=f"FLEXMatched-Isaac-Velocity-Rough-Unitree-Go2{play_suffix}-v0",
        env_cfg=f"UnitreeGo2RoughFLEXMatchedEnvCfg{env_suffix}",
        runner_cfg="UnitreeGo2RoughPPOFLEXRunnerCfg",
        terrain="rough",
    )


# Benchmark IDs share the same physical configuration. FTNet retains its
# paper-specific 49-D proprioception and separate privileged-physics group;
# all other architectures use the common observation configuration.
_EVAL_RUNNERS = {
    "GCNMLP": "UnitreeGo2RoughPPOGCNMLPRunnerCfg",
    "GCN": "UnitreeGo2RoughPPOGCNRunnerCfg",
    "EquivGCN": "UnitreeGo2RoughPPOEquivGCNRunnerCfg",
    "EquivGCNNoFilm": "UnitreeGo2RoughPPOEquivGCNNoFilmRunnerCfg",
    "EquivGCNUngatedFilm": "UnitreeGo2RoughPPOEquivGCNUngatedFilmRunnerCfg",
    "EquivGCNFaultOnly": "UnitreeGo2RoughPPOEquivGCNFaultOnlyRunnerCfg",
    "EquivGCNGCNOnly": "UnitreeGo2RoughPPOEquivGCNGCNOnlyRunnerCfg",
    "EquivGCNRLLatent": "UnitreeGo2RoughPPOEquivGCNRLLatentRunnerCfg",
    "EquivGCN13Node": "UnitreeGo2RoughPPOEquivGCN13NodeRunnerCfg",
    "HistoryMLP": "UnitreeGo2RoughPPOHistoryMLPRunnerCfg",
    "EquivGCNMLP": "UnitreeGo2RoughPPOEquivGCNMLPRunnerCfg",
    "EquivGCNMLPConcat": "UnitreeGo2RoughPPOEquivGCNMLPConcatRunnerCfg",
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


_register(
    task_id="EquivGCNMLP-V83-CompleteFault-RandomRough-Unitree-Go2-v0",
    env_cfg="UnitreeGo2RoughEquivGCNMLPV83CompleteFaultEnvCfg",
    runner_cfg="UnitreeGo2RoughPPOEquivGCNMLPRunnerCfg",
    terrain="rough",
)


_register(
    task_id="DreamFLEX-V83-CompleteFault-RandomRough-Unitree-Go2-v0",
    env_cfg="UnitreeGo2RoughFLEXV83CompleteFaultEnvCfg",
    runner_cfg="UnitreeGo2RoughPPOFLEXRunnerCfg",
    terrain="rough",
)


_register(
    task_id="EquivGCNMLP-V84-CompleteFault-MixedTerrain-Unitree-Go2-v0",
    env_cfg="UnitreeGo2RoughEquivGCNMLPV84CompleteFaultEnvCfg",
    runner_cfg="UnitreeGo2RoughPPOEquivGCNMLPRunnerCfg",
    terrain="rough",
)


_register(
    task_id="DreamFLEX-V84-CompleteFault-MixedTerrain-Unitree-Go2-v0",
    env_cfg="UnitreeGo2RoughFLEXV84CompleteFaultEnvCfg",
    runner_cfg="UnitreeGo2RoughPPOFLEXRunnerCfg",
    terrain="rough",
)


_register(
    task_id="EquivGCNMLP-V84-CompleteFault-MixedTerrain-Unitree-Go2-Play-v0",
    env_cfg="UnitreeGo2RoughEquivGCNMLPV84CompleteFaultEnvCfg_PLAY",
    runner_cfg="UnitreeGo2RoughPPOEquivGCNMLPRunnerCfg",
    terrain="rough",
)


_register(
    task_id="DreamFLEX-V84-CompleteFault-MixedTerrain-Unitree-Go2-Play-v0",
    env_cfg="UnitreeGo2RoughFLEXV84CompleteFaultEnvCfg_PLAY",
    runner_cfg="UnitreeGo2RoughPPOFLEXRunnerCfg",
    terrain="rough",
)


_register(
    task_id="FTNet-V84-CompleteFault-MixedTerrain-Unitree-Go2-v0",
    env_cfg="UnitreeGo2RoughFTNetV84CompleteFaultEnvCfg",
    runner_cfg="UnitreeGo2RoughPPOFTNetRunnerCfg",
    terrain="rough",
)


_register(
    task_id="FTNet-V84-CompleteFault-MixedTerrain-Unitree-Go2-Play-v0",
    env_cfg="UnitreeGo2RoughFTNetV84CompleteFaultEnvCfg_PLAY",
    runner_cfg="UnitreeGo2RoughPPOFTNetRunnerCfg",
    terrain="rough",
)


_register(
    task_id="GCN-V84-CompleteFault-MixedTerrain-Unitree-Go2-v0",
    env_cfg="UnitreeGo2RoughGCNV84CompleteFaultEnvCfg",
    runner_cfg="UnitreeGo2RoughPPOGCNRunnerCfg",
    terrain="rough",
)


_register(
    task_id="GCN-V84-CompleteFault-MixedTerrain-Unitree-Go2-Play-v0",
    env_cfg="UnitreeGo2RoughGCNV84CompleteFaultEnvCfg_PLAY",
    runner_cfg="UnitreeGo2RoughPPOGCNRunnerCfg",
    terrain="rough",
)


_register(
    task_id="EquivGCN-V84-CompleteFault-MixedTerrain-Unitree-Go2-v0",
    env_cfg="UnitreeGo2RoughEquivGCNV84CompleteFaultEnvCfg",
    runner_cfg="UnitreeGo2RoughPPOEquivGCNRunnerCfg",
    terrain="rough",
)


_register(
    task_id="EquivGCN-V84-CompleteFault-MixedTerrain-Unitree-Go2-Play-v0",
    env_cfg="UnitreeGo2RoughEquivGCNV84CompleteFaultEnvCfg_PLAY",
    runner_cfg="UnitreeGo2RoughPPOEquivGCNRunnerCfg",
    terrain="rough",
)


_register(
    task_id="EquivGCNMLPConcat-V84-CompleteFault-MixedTerrain-Unitree-Go2-v0",
    env_cfg="UnitreeGo2RoughEquivGCNMLPConcatV84CompleteFaultEnvCfg",
    runner_cfg="UnitreeGo2RoughPPOEquivGCNMLPConcatRunnerCfg",
    terrain="rough",
)


_register(
    task_id="EquivGCNMLPConcat-V84-CompleteFault-MixedTerrain-Unitree-Go2-Play-v0",
    env_cfg="UnitreeGo2RoughEquivGCNMLPConcatV84CompleteFaultEnvCfg_PLAY",
    runner_cfg="UnitreeGo2RoughPPOEquivGCNMLPConcatRunnerCfg",
    terrain="rough",
)


_register(
    task_id="EquivGCNNoFilm-V84-CompleteFault-MixedTerrain-Unitree-Go2-v0",
    env_cfg="UnitreeGo2RoughEquivGCNV84CompleteFaultEnvCfg",
    runner_cfg="UnitreeGo2RoughPPOEquivGCNNoFilmRunnerCfg",
    terrain="rough",
)


_register(
    task_id="EquivGCNNoFilm-V84-CompleteFault-MixedTerrain-Unitree-Go2-Play-v0",
    env_cfg="UnitreeGo2RoughEquivGCNV84CompleteFaultEnvCfg_PLAY",
    runner_cfg="UnitreeGo2RoughPPOEquivGCNNoFilmRunnerCfg",
    terrain="rough",
)


for ablation, runner_cfg in (
    ("UngatedFilm", "UnitreeGo2RoughPPOEquivGCNUngatedFilmRunnerCfg"),
    ("FaultOnly", "UnitreeGo2RoughPPOEquivGCNFaultOnlyRunnerCfg"),
    ("GCNOnly", "UnitreeGo2RoughPPOEquivGCNGCNOnlyRunnerCfg"),
    ("RLLatent", "UnitreeGo2RoughPPOEquivGCNRLLatentRunnerCfg"),
):
    for play_suffix, env_suffix in (("", ""), ("-Play", "_PLAY")):
        _register(
            task_id=f"EquivGCN{ablation}-V84-CompleteFault-MixedTerrain-Unitree-Go2{play_suffix}-v0",
            env_cfg=f"UnitreeGo2RoughEquivGCNV84CompleteFaultEnvCfg{env_suffix}",
            runner_cfg=runner_cfg,
            terrain="rough",
        )


_register(
    task_id="EquivGCN13Node-V84-CompleteFault-MixedTerrain-Unitree-Go2-v0",
    env_cfg="UnitreeGo2RoughEquivGCNV84CompleteFaultEnvCfg",
    runner_cfg="UnitreeGo2RoughPPOEquivGCN13NodeRunnerCfg",
    terrain="rough",
)


_register(
    task_id="EquivGCN13Node-V84-CompleteFault-MixedTerrain-Unitree-Go2-Play-v0",
    env_cfg="UnitreeGo2RoughEquivGCNV84CompleteFaultEnvCfg_PLAY",
    runner_cfg="UnitreeGo2RoughPPOEquivGCN13NodeRunnerCfg",
    terrain="rough",
)


_register(
    task_id="HistoryMLP-V84-CompleteFault-MixedTerrain-Unitree-Go2-v0",
    env_cfg="UnitreeGo2RoughEquivGCNV84CompleteFaultEnvCfg",
    runner_cfg="UnitreeGo2RoughPPOHistoryMLPRunnerCfg",
    terrain="rough",
)


_register(
    task_id="HistoryMLP-V84-CompleteFault-MixedTerrain-Unitree-Go2-Play-v0",
    env_cfg="UnitreeGo2RoughEquivGCNV84CompleteFaultEnvCfg_PLAY",
    runner_cfg="UnitreeGo2RoughPPOHistoryMLPRunnerCfg",
    terrain="rough",
)
