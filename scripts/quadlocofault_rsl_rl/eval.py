# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate GCN, EquivGCN, FTNet, and DreamFLEX baseline checkpoints.

Two protocols are provided:

* ``rough``: fault timing inherited from the evaluation environment, command
  ``(0.5, 0, 0)``, and mean capped
  locomotion lifetime plus termination-aware distance success over 4000
  environments for six terrain families and fault coefficients 0.0 and 0.1.
* ``flat``: flat ground, command ``(1, 0, 0)``, fault injected at 3 s, and a
  10 s episode. Forward-velocity ATE and a representative foot-contact
  diagram are saved.

Isaac Sim is launched in a fresh worker process for each experiment. The
parent process persists every completed worker immediately and resumes by
skipping matching rows already present in the protocol CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


MODELS = ("GCN", "EquivGCN", "EquivGCNMLP", "FTNet", "FLEX")
FAULT_COEFFICIENTS = (0.0, 0.1)
JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)
TERRAINS = {
    "rough": "random_rough",
    "grid": "grid",
    # Robots spawn on the center platform and travel outwards.  The regular
    # pyramid variants have a high center, so outward travel is downhill;
    # the inverted variants have a low center, so outward travel is uphill.
    "slope_up": "hf_pyramid_slope_inv",
    "slope_down": "hf_pyramid_slope",
    "stairs_up": "pyramid_stairs_inv",
    "stairs_down": "pyramid_stairs",
}
EVAL_TASKS = {
    model: f"{model}-Isaac-Velocity-Eval-Unitree-Go2-v0" for model in MODELS
}
EXPERIMENTS = {
    "GCN": "unitree_go2_rough_gcn",
    "EquivGCN": "unitree_go2_rough_equiv_gcn",
    "EquivGCNMLP": "unitree_go2_rough_equiv_gcn_mlp",
    "FTNet": "unitree_go2_rough_ftnet",
    "FLEX": "unitree_go2_rough_flex",
}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--protocol", choices=("all", "rough", "flat"), default="all")
parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
parser.add_argument(
    "--terrains",
    nargs="+",
    choices=tuple(TERRAINS),
    default=None,
    help=(
        "Rough-terrain families to evaluate (default: all). For example, "
        "--terrains slope_down stairs_up stairs_down resumes from slope_down."
    ),
)
parser.add_argument("--num_envs", type=int, default=4000)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Environment seed for a single worker (normally set through --eval_seeds).",
)
parser.add_argument("--rough_duration", type=float, default=20.0)
parser.add_argument(
    "--success_distance",
    type=float,
    default=None,
    help=(
        "Forward distance in metres required for rough-terrain success. "
        "Defaults to half the terrain-tile length minus --success_distance_margin."
    ),
)
parser.add_argument(
    "--success_distance_margin",
    type=float,
    default=0.25,
    help="Safety margin subtracted from half the terrain-tile length.",
)
parser.add_argument(
    "--success_confirmation_time",
    type=float,
    default=0.5,
    help="Time the robot must remain non-terminated after reaching the target distance.",
)
parser.add_argument("--flat_duration", type=float, default=10.0)
parser.add_argument("--fault_time", type=float, default=3.0)
parser.add_argument(
    "--fault_joint",
    choices=JOINT_NAMES,
    default=None,
    help=(
        "Optional joint that receives the actuator fault. If omitted, each "
        "environment samples a faulty joint randomly."
    ),
)
parser.add_argument(
    "--fault_joints",
    nargs="+",
    choices=JOINT_NAMES,
    default=None,
    help=(
        "Parent-mode sweep over explicit faulty joints. Results are saved "
        "separately for every joint and seed."
    ),
)
parser.add_argument(
    "--batch_fault_joints",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Evaluate --fault_joints concurrently. --num_envs is interpreted as "
        "the number of environments per joint."
    ),
)
parser.add_argument(
    "--eval_seeds",
    nargs="+",
    type=int,
    default=None,
    help="Parent-mode evaluation seeds (default: --seed, or 0 when unset).",
)
parser.add_argument("--terrain_difficulty_min", type=float, default=0.5)
parser.add_argument("--terrain_difficulty_max", type=float, default=0.7)
parser.add_argument(
    "--stair_step_height_max",
    type=float,
    default=None,
    help="Optional maximum step height in metres for both stair terrain directions.",
)
parser.add_argument("--contact_threshold", type=float, default=1.0)
parser.add_argument("--plot_env_id", type=int, default=0)
parser.add_argument(
    "--flex_history_length",
    type=int,
    default=5,
    help=(
        "DreamFLEX history horizon. Use 5 for newly trained models (default), "
        "or 30 only to evaluate legacy checkpoints trained before the fix."
    ),
)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument(
    "--resume-eval",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Skip cases already saved for the same checkpoint and number of environments (default: true).",
)
parser.add_argument(
    "--latest_run_name",
    type=str,
    default="baseline",
    help=(
        "Run directory below logs/rsl_rl/<experiment>/ used for automatic "
        "checkpoint selection (default: baseline)."
    ),
)
parser.add_argument("--equivgcn_checkpoint", type=str, default=None)
parser.add_argument("--equiv_gcn_mlp_checkpoint", type=str, default=None)
parser.add_argument("--gcn_checkpoint", type=str, default=None)
parser.add_argument("--ftnet_checkpoint", type=str, default=None)
parser.add_argument("--flex_checkpoint", type=str, default=None)
parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--model", choices=MODELS, default=None, help=argparse.SUPPRESS)
parser.add_argument("--terrain", choices=tuple(TERRAINS), default=None, help=argparse.SUPPRESS)
parser.add_argument("--fault_coef", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument(
    "--worker_fault_joints",
    nargs="+",
    choices=JOINT_NAMES,
    default=None,
    help=argparse.SUPPRESS,
)
parser.add_argument("--result_json", type=str, default=None, help=argparse.SUPPRESS)
parser.add_argument("--task", type=str, default=None, help=argparse.SUPPRESS)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Gym registry entry point used to load the RSL-RL agent configuration.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _checkpoint_override(model: str) -> str | None:
    argument_names = {
        "GCN": "gcn_checkpoint",
        "EquivGCN": "equivgcn_checkpoint",
        "EquivGCNMLP": "equiv_gcn_mlp_checkpoint",
        "FTNet": "ftnet_checkpoint",
        "FLEX": "flex_checkpoint",
    }
    return getattr(args_cli, argument_names[model])


def _latest_baseline_checkpoint(model: str) -> Path:
    override = _checkpoint_override(model)
    if override:
        checkpoint = Path(override).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{model} checkpoint does not exist: {checkpoint}")
        return checkpoint

    run_dir = (
        _repo_root()
        / "logs"
        / "rsl_rl"
        / EXPERIMENTS[model]
        / args_cli.latest_run_name
    )
    checkpoints = list(run_dir.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in {run_dir}")

    def iteration(path: Path) -> int:
        try:
            return int(path.stem.rsplit("_", 1)[1])
        except ValueError:
            return -1

    return max(checkpoints, key=iteration)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as stream:
            existing_rows = list(csv.DictReader(stream))
        merged = {_case_key(row): row for row in existing_rows}
        for row in rows:
            merged[_case_key(row)] = row
        rows = list(merged.values())
    # Existing evaluation files may predate newly added metrics. Use the union
    # of all row fields so partial reruns can upgrade the CSV incrementally;
    # values unavailable in older rows are left blank until those cases rerun.
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Wrote {path}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read an evaluation CSV, returning an empty list when it does not exist."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _case_key(row: dict) -> tuple[str, str, str, str, str]:
    """Return the fields that uniquely identify one evaluation worker case."""
    return (
        str(row["model"]),
        str(row["terrain"]),
        f"{float(row['fault_coefficient']):g}",
        str(row.get("fault_joint") or ""),
        str(row.get("seed", "")),
    )


def _matches_saved_run(row: dict, checkpoint: Path, num_envs: int) -> bool:
    """Check resume metadata while tolerating older CSVs with missing fields."""
    try:
        saved_num_envs = int(row.get("num_envs", -1))
    except (TypeError, ValueError):
        return False
    saved_checkpoint = row.get("checkpoint")
    if not saved_checkpoint:
        return False
    common_match = (
        Path(saved_checkpoint).expanduser().resolve() == checkpoint.resolve()
        and saved_num_envs == num_envs
    )
    if not common_match:
        return False
    # Do not silently reuse rows produced by an older benchmark protocol.
    required_values = {
        "terrain_difficulty_min": args_cli.terrain_difficulty_min,
        "terrain_difficulty_max": args_cli.terrain_difficulty_max,
        "stair_step_height_max_m": args_cli.stair_step_height_max,
        "success_distance_m": args_cli.success_distance,
        "success_confirmation_time_s": args_cli.success_confirmation_time,
        "horizon_s": args_cli.rough_duration,
        "initial_xy_range_m": 0.5,
        "initial_yaw_range_rad": math.pi,
    }
    for key, expected in required_values.items():
        saved = row.get(key)
        if saved in (None, "") or expected is None:
            if saved not in (None, "") or expected is not None:
                return False
            continue
        if not math.isclose(float(saved), float(expected), rel_tol=0.0, abs_tol=1.0e-9):
            return False
    if row.get("progress_frame") != "initial_heading":
        return False
    return True


def _persist_result(output_dir: Path, protocol: str, row: dict) -> None:
    """Persist one completed worker immediately so interrupted sweeps can resume."""
    filename = "rough_locomotion_lifetime.csv" if protocol == "rough" else "flat_ate.csv"
    _write_csv(output_dir / filename, [row])
    joint_label = row.get("fault_joint") or "random_joint"
    _write_csv(
        output_dir / "by_fault" / joint_label / f"{protocol}_seed_{row['seed']}.csv",
        [row],
    )


def _worker_command(
    *,
    model: str,
    protocol: str,
    fault_coef: float,
    result_json: Path,
    output_dir: Path,
    checkpoint: Path,
    terrain: str | None = None,
    fault_joint: str | None = None,
    fault_joints: list[str] | None = None,
    worker_num_envs: int | None = None,
    seed: int = 0,
) -> list[str]:
    """Build a worker invocation while retaining launcher/device arguments."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--worker",
        "--model",
        model,
        "--protocol",
        protocol,
        "--fault_coef",
        str(fault_coef),
        "--result_json",
        str(result_json),
        "--output_dir",
        str(output_dir),
        "--task",
        EVAL_TASKS[model],
        "--checkpoint",
        str(checkpoint),
        "--seed",
        str(seed),
    ]
    if terrain is not None:
        command.extend(("--terrain", terrain))
    if fault_joint is not None:
        command.extend(("--fault_joint", fault_joint))
    if fault_joints is not None:
        command.extend(("--worker_fault_joints", *fault_joints))
    if worker_num_envs is not None:
        command.extend(("--num_envs", str(worker_num_envs)))
    return command


def _run_parent() -> int:
    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be positive.")
    if args_cli.flex_history_length <= 0:
        raise ValueError("--flex_history_length must be positive.")
    if args_cli.rough_duration <= 0.0 or args_cli.flat_duration <= 0.0:
        raise ValueError("Evaluation durations must be positive.")
    if args_cli.success_distance is not None and args_cli.success_distance <= 0.0:
        raise ValueError("--success_distance must be positive when specified.")
    if args_cli.success_distance_margin < 0.0:
        raise ValueError("--success_distance_margin must be non-negative.")
    if args_cli.success_confirmation_time < 0.0:
        raise ValueError("--success_confirmation_time must be non-negative.")
    if args_cli.stair_step_height_max is not None and args_cli.stair_step_height_max <= 0.05:
        raise ValueError("--stair_step_height_max must be greater than the 0.05 m minimum.")
    if not 0.0 <= args_cli.fault_time < args_cli.flat_duration:
        raise ValueError("--fault_time must be within the flat evaluation episode.")

    output_dir = Path(
        args_cli.output_dir or (_repo_root() / "logs" / "evaluation" / args_cli.latest_run_name)
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_protocols = ("rough", "flat") if args_cli.protocol == "all" else (args_cli.protocol,)
    if args_cli.fault_joints is not None and args_cli.fault_joint is not None:
        raise ValueError("Use either --fault_joint or --fault_joints, not both.")
    selected_joints = (
        args_cli.fault_joints
        if args_cli.fault_joints is not None
        else [args_cli.fault_joint]
    )
    if args_cli.batch_fault_joints and args_cli.fault_joints is None:
        raise ValueError("--batch_fault_joints requires --fault_joints.")
    if args_cli.batch_fault_joints and args_cli.protocol != "rough":
        raise ValueError("--batch_fault_joints currently supports --protocol rough only.")
    joint_batches = (
        [list(selected_joints)]
        if args_cli.batch_fault_joints
        else [[joint] for joint in selected_joints]
    )
    selected_seeds = (
        args_cli.eval_seeds
        if args_cli.eval_seeds is not None
        else [args_cli.seed if args_cli.seed is not None else 0]
    )
    results: dict[str, list[dict]] = {"rough": [], "flat": []}
    saved_rows = {
        "rough": _read_csv(output_dir / "rough_locomotion_lifetime.csv"),
        "flat": _read_csv(output_dir / "flat_ate.csv"),
    }

    with tempfile.TemporaryDirectory(prefix="quadlocofault_eval_") as temp_dir:
        temp_path = Path(temp_dir)
        for model in args_cli.models:
            checkpoint = _latest_baseline_checkpoint(model)
            print(f"[INFO] {model}: {checkpoint}")
            for protocol in selected_protocols:
                terrain_names = (
                    tuple(args_cli.terrains or TERRAINS)
                    if protocol == "rough"
                    else (None,)
                )
                for fault_batch in joint_batches:
                    batched = len(fault_batch) > 1
                    joint_label = (
                        "balanced_joints"
                        if batched
                        else (fault_batch[0] or "random_joint")
                    )
                    for seed in selected_seeds:
                        for terrain in terrain_names:
                            for fault_coef in FAULT_COEFFICIENTS:
                                cases = [
                                    {
                                        "model": model,
                                        "terrain": terrain or "flat",
                                        "fault_coefficient": fault_coef,
                                        "fault_joint": fault_joint,
                                        "seed": seed,
                                    }
                                    for fault_joint in fault_batch
                                ]
                                already_saved = all(
                                    any(
                                        _case_key(row) == _case_key(case)
                                        and _matches_saved_run(row, checkpoint, args_cli.num_envs)
                                        for row in saved_rows[protocol]
                                    )
                                    for case in cases
                                )
                                if args_cli.resume_eval and already_saved:
                                    print(
                                        "[INFO] Skipping completed "
                                        f"{model}_{protocol}_{terrain or 'flat'}_"
                                        f"{joint_label}_seed{seed}_{fault_coef:.1f}"
                                    )
                                    continue
                                label = (
                                    f"{model}_{protocol}_{terrain or 'flat'}_"
                                    f"{joint_label}_seed{seed}_{fault_coef:.1f}"
                                )
                                result_json = temp_path / f"{label}.json"
                                print(f"[INFO] Running {label}")
                                completed = subprocess.run(
                                    _worker_command(
                                        model=model,
                                        protocol=protocol,
                                        terrain=terrain,
                                        fault_coef=fault_coef,
                                        fault_joint=None if batched else fault_batch[0],
                                        fault_joints=fault_batch if batched else None,
                                        worker_num_envs=(
                                            args_cli.num_envs * len(fault_batch)
                                            if batched
                                            else None
                                        ),
                                        seed=seed,
                                        result_json=result_json,
                                        output_dir=output_dir,
                                        checkpoint=checkpoint,
                                    ),
                                    check=False,
                                )
                                if completed.returncode != 0:
                                    return completed.returncode
                                if not result_json.is_file():
                                    print(
                                        f"[ERROR] Worker {label} exited without writing {result_json}.",
                                        file=sys.stderr,
                                    )
                                    return 1
                                worker_result = json.loads(
                                    result_json.read_text(encoding="utf-8")
                                )
                                rows = (
                                    worker_result
                                    if isinstance(worker_result, list)
                                    else [worker_result]
                                )
                                for row in rows:
                                    results[protocol].append(row)
                                    saved_rows[protocol].append(row)
                                    _persist_result(output_dir, protocol, row)

    if results["rough"]:
        _write_csv(output_dir / "rough_locomotion_lifetime.csv", results["rough"])
    if results["flat"]:
        _write_csv(output_dir / "flat_ate.csv", results["flat"])
    for protocol, protocol_rows in results.items():
        for fault_joint in selected_joints:
            for seed in selected_seeds:
                subset = [
                    row
                    for row in protocol_rows
                    if row.get("fault_joint") == fault_joint and row.get("seed") == seed
                ]
                if subset:
                    joint_label = fault_joint or "random_joint"
                    _write_csv(
                        output_dir / "by_fault" / joint_label / f"{protocol}_seed_{seed}.csv",
                        subset,
                    )
    return 0


def _run_worker() -> int:
    if args_cli.model is None or args_cli.fault_coef is None or args_cli.result_json is None:
        raise ValueError("Worker requires --model, --fault_coef, and --result_json.")
    if args_cli.protocol == "rough" and args_cli.terrain is None:
        raise ValueError("A rough worker requires --terrain.")

    sys.argv = [sys.argv[0]] + hydra_args
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import importlib.metadata as metadata

    import gymnasium as gym
    import numpy as np
    import torch
    from packaging import version
    from rsl_rl.runners import DistillationRunner
    from runners import CustomOnPolicyRunner

    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_quadlocofault_rl.rsl_rl import CustomRslRlVecEnvWrapper

    import isaaclab_tasks  # noqa: F401
    import isaaclab_quadlocofault_tasks  # noqa: F401

    installed_version = metadata.version("rsl-rl-lib")

    def configure_common(env_cfg) -> None:
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.curriculum.terrain_levels = None
        env_cfg.curriculum.actuator_faults = None
        env_cfg.events.add_base_mass = None
        env_cfg.events.base_external_force_torque = None
        env_cfg.events.push_robot = None
        env_cfg.events.reset_actuator_gains.params["motors_strength_range"] = (1.0, 1.0)
        env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        env_cfg.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                axis: (0.0, 0.0)
                for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            },
        }

        command = env_cfg.commands.base_velocity
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.rel_standing_envs = 0.0
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)

        # The event implementation samples severe faults in [0, coef]. Setting
        # both moderate bounds to coef and severe probability to zero makes the
        # benchmark coefficient exact rather than random.
        fault_event = env_cfg.events.randomize_actuator_faults
        if args_cli.worker_fault_joints is not None:
            if env_cfg.scene.num_envs % len(args_cli.worker_fault_joints) != 0:
                raise ValueError(
                    "The worker environment count must be divisible by the number "
                    "of batched fault joints."
                )
            envs_per_joint = env_cfg.scene.num_envs // len(args_cli.worker_fault_joints)
            fixed_joint_idx = [
                JOINT_NAMES.index(joint_name)
                for joint_name in args_cli.worker_fault_joints
                for _ in range(envs_per_joint)
            ]
        else:
            fixed_joint_idx = (
                JOINT_NAMES.index(args_cli.fault_joint)
                if args_cli.fault_joint is not None
                else None
            )
        fault_event.params.update(
            severe_fault_prob=0.0,
            failure_coef_severe=float(args_cli.fault_coef),
            failure_coef_moderate=float(args_cli.fault_coef),
            num_faults=1,
            fixed_joint_idx=fixed_joint_idx,
            apply_once_per_episode=False,
        )

    def configure_rough(env_cfg) -> None:
        configure_common(env_cfg)
        env_cfg.episode_length_s = args_cli.rough_duration
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        env_cfg.commands.base_velocity.resampling_time_range = (
            args_cli.rough_duration + 1.0,
            args_cli.rough_duration + 1.0,
        )

        # Apply one persistent fault during reset so the entire measured
        # traversal takes place under the requested fault condition.
        fault_event = env_cfg.events.randomize_actuator_faults
        fault_event.mode = "reset"
        fault_event.interval_range_s = None

        generator = env_cfg.scene.terrain.terrain_generator
        generator.curriculum = False
        generator.difficulty_range = (
            args_cli.terrain_difficulty_min,
            args_cli.terrain_difficulty_max,
        )
        if args_cli.stair_step_height_max is not None:
            generator.sub_terrains["pyramid_stairs"].step_height_range = (
                0.05,
                args_cli.stair_step_height_max,
            )
            generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (
                0.05,
                args_cli.stair_step_height_max,
            )
        env_cfg.scene.terrain.max_init_terrain_level = None
        for subterrain in generator.sub_terrains.values():
            subterrain.proportion = 0.0
        generator.sub_terrains[TERRAINS[args_cli.terrain]].proportion = 1.0

    def configure_flat(env_cfg) -> None:
        configure_common(env_cfg)
        env_cfg.episode_length_s = args_cli.flat_duration
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        env_cfg.commands.base_velocity.resampling_time_range = (
            args_cli.flat_duration + 1.0,
            args_cli.flat_duration + 1.0,
        )
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.max_init_terrain_level = None

        fault_event = env_cfg.events.randomize_actuator_faults
        fault_event.mode = "interval"
        fault_event.interval_range_s = (args_cli.fault_time, args_cli.fault_time)

    def create_runner(env, agent_cfg):
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = CustomOnPolicyRunner(
                env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
            )
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(
                env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
            )
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        if version.parse(installed_version) >= version.parse("4.0.0"):
            policy_module = None
        elif version.parse(installed_version) >= version.parse("2.3.0"):
            policy_module = runner.alg.policy
        else:
            policy_module = runner.alg.actor_critic
        return runner, policy, policy_module

    def reset_policy(policy, policy_module, dones) -> None:
        if version.parse(installed_version) >= version.parse("4.0.0"):
            policy.reset(dones)
        else:
            policy_module.reset(dones)

    def evaluate_rough(env, policy, policy_module) -> dict:
        base_env = env.unwrapped
        dt = base_env.step_dt
        max_steps = round(args_cli.rough_duration / dt)
        terrain_length = float(base_env.cfg.scene.terrain.terrain_generator.size[0])
        success_distance = args_cli.success_distance
        if success_distance is None:
            success_distance = 0.5 * terrain_length - args_cli.success_distance_margin
        if success_distance <= 0.0:
            raise ValueError(
                "The derived success distance must be positive; adjust the terrain size "
                "or --success_distance_margin."
            )
        confirmation_steps = math.ceil(args_cli.success_confirmation_time / dt)

        lifetime = torch.full(
            (env.num_envs,), max_steps * dt, device=base_env.device
        )
        finished = torch.zeros(
            env.num_envs, dtype=torch.bool, device=base_env.device
        )
        collision_finished = torch.zeros_like(finished)
        orientation_finished = torch.zeros_like(finished)
        reached_target = torch.zeros_like(finished)
        success = torch.zeros_like(finished)
        confirmation_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=base_env.device
        )
        robot = base_env.scene["robot"]
        initial_xy = robot.data.root_pos_w[:, :2].clone()
        initial_heading = robot.data.heading_w.clone()
        initial_forward_xy = torch.stack(
            (torch.cos(initial_heading), torch.sin(initial_heading)), dim=1
        )
        max_forward_progress = torch.zeros(
            env.num_envs, dtype=torch.float, device=base_env.device
        )
        abs_vx_error_sum = torch.zeros(env.num_envs, device=base_env.device)
        abs_xy_error_sum = torch.zeros(env.num_envs, device=base_env.device)
        ate_sample_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=base_env.device
        )
        obs = env.get_observations()
        with torch.inference_mode():
            for step in range(max_steps):
                # Save first-episode progress before env.step(), since Isaac Lab
                # automatically resets terminated environments inside that call.
                active_before_step = ~finished
                displacement_xy = robot.data.root_pos_w[:, :2] - initial_xy
                forward_progress = torch.sum(displacement_xy * initial_forward_xy, dim=1)
                max_forward_progress[active_before_step] = torch.maximum(
                    max_forward_progress[active_before_step],
                    forward_progress[active_before_step],
                )

                outputs = policy(obs)
                actions = outputs[0] if isinstance(outputs, tuple) else outputs
                obs, _, dones, _ = env.step(actions)
                dones = dones.bool()
                # A time-out at the requested horizon is censoring, not a
                # locomotion failure. Only physical termination ends lifetime.
                terminated = base_env.reset_terminated.bool()
                timed_out = base_env.reset_time_outs.bool()
                base_contact = base_env.termination_manager.get_term("base_contact").bool()
                bad_orientation = base_env.termination_manager.get_term("bad_orientation").bool()

                # Isaac Lab resets terminated environments during env.step().
                # Exclude those reset observations from first-episode ATE.
                valid_ate = active_before_step & ~terminated & ~timed_out
                if valid_ate.any():
                    command = base_env.command_manager.get_command("base_velocity")
                    measured_xy = base_env.scene["robot"].data.root_lin_vel_b[:, :2]
                    velocity_error = command[:, :2] - measured_xy
                    abs_vx_error_sum[valid_ate] += torch.abs(
                        velocity_error[valid_ate, 0]
                    )
                    abs_xy_error_sum[valid_ate] += torch.linalg.vector_norm(
                        velocity_error[valid_ate], dim=1
                    )
                    ate_sample_count[valid_ate] += 1

                # Reaching the target on a step that ends in termination does
                # not count. A successful traversal must then remain alive for
                # the requested confirmation window, preventing a fall or
                # forward tumble from being classified as success.
                valid_first_episode = active_before_step & ~terminated
                reached_target |= (
                    valid_first_episode & (forward_progress >= success_distance)
                )
                confirming = valid_first_episode & reached_target & ~success
                confirmation_count[confirming] += 1
                success |= confirming & (confirmation_count >= confirmation_steps)

                newly_finished = terminated & ~finished
                collision_finished |= newly_finished & base_contact
                orientation_finished |= newly_finished & bad_orientation
                lifetime[newly_finished] = (step + 1) * dt
                finished |= newly_finished
                reset_policy(policy, policy_module, dones)
                if finished.all():
                    break

        def result_for_group(group_mask: torch.Tensor, fault_joint: str | None) -> dict:
            group_ate_samples = int(ate_sample_count[group_mask].sum().item())
            group_abs_vx_error = abs_vx_error_sum[group_mask].sum()
            group_abs_xy_error = abs_xy_error_sum[group_mask].sum()
            mean_abs_vx_error = float(
                (group_abs_vx_error / max(group_ate_samples, 1)).item()
            )
            return {
                "model": args_cli.model,
                "terrain": args_cli.terrain,
                "terrain_cfg": TERRAINS[args_cli.terrain],
                "fault_coefficient": float(args_cli.fault_coef),
                "fault_joint": fault_joint,
                "seed": int(args_cli.seed),
                "fault_time_s": 0.0,
                "command_vx_mps": 0.5,
                "num_envs": int(group_mask.sum().item()),
                "horizon_s": max_steps * dt,
                "success_distance_m": float(success_distance),
                "success_confirmation_time_s": float(args_cli.success_confirmation_time),
                "terrain_difficulty_min": float(args_cli.terrain_difficulty_min),
                "terrain_difficulty_max": float(args_cli.terrain_difficulty_max),
                "stair_step_height_max_m": args_cli.stair_step_height_max,
                "initial_xy_range_m": 0.5,
                "initial_yaw_range_rad": math.pi,
                "progress_frame": "initial_heading",
                "flex_history_length": (
                    args_cli.flex_history_length if args_cli.model == "FLEX" else None
                ),
                "distance_success_rate": float(success[group_mask].float().mean().item()),
                "num_distance_successes": int(success[group_mask].sum().item()),
                "mean_max_forward_progress_m": float(
                    max_forward_progress[group_mask].mean().item()
                ),
                "median_max_forward_progress_m": float(
                    max_forward_progress[group_mask].median().item()
                ),
                "ate_vx_mps": mean_abs_vx_error,
                "mean_abs_vx_tracking_error_mps": mean_abs_vx_error,
                "ate_xy_mps": float(
                    (group_abs_xy_error / max(group_ate_samples, 1)).item()
                ),
                "num_ate_samples": group_ate_samples,
                "mean_locomotion_time_s": float(lifetime[group_mask].mean().item()),
                "std_locomotion_time_s": float(
                    lifetime[group_mask].std(unbiased=False).item()
                ),
                "median_locomotion_time_s": float(lifetime[group_mask].median().item()),
                "min_locomotion_time_s": float(lifetime[group_mask].min().item()),
                "survival_to_horizon": float((~finished[group_mask]).float().mean().item()),
                "timeout_rate": float((~finished[group_mask]).float().mean().item()),
                "collision_rate": float(
                    collision_finished[group_mask].float().mean().item()
                ),
                "bad_orientation_rate": float(
                    orientation_finished[group_mask].float().mean().item()
                ),
                "num_collision_terminations": int(
                    collision_finished[group_mask].sum().item()
                ),
                "num_bad_orientation_terminations": int(
                    orientation_finished[group_mask].sum().item()
                ),
                "num_resets_before_horizon": int(finished[group_mask].sum().item()),
                "checkpoint": str(Path(args_cli.checkpoint).resolve()),
            }

        if args_cli.worker_fault_joints is None:
            all_envs = torch.ones(env.num_envs, dtype=torch.bool, device=base_env.device)
            return result_for_group(all_envs, args_cli.fault_joint)

        envs_per_joint = env.num_envs // len(args_cli.worker_fault_joints)
        results = []
        for joint_index, joint_name in enumerate(args_cli.worker_fault_joints):
            group_mask = torch.zeros(
                env.num_envs, dtype=torch.bool, device=base_env.device
            )
            start = joint_index * envs_per_joint
            group_mask[start : start + envs_per_joint] = True
            results.append(result_for_group(group_mask, joint_name))
        return results

    def write_contact_plot(
        times: list[float],
        contacts: list[list[float]],
        foot_names: list[str],
    ) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
        contact_array = np.asarray(contacts).T
        for foot_index, foot_name in enumerate(foot_names):
            axis.step(
                times,
                contact_array[foot_index] + foot_index,
                where="post",
                label=foot_name,
            )
        axis.axvline(args_cli.fault_time, color="red", linestyle="--", label="fault")
        axis.set_yticks(range(len(foot_names)), foot_names)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Foot contact")
        axis.set_title(
            f"{args_cli.model}, fault coefficient={args_cli.fault_coef:.1f}, "
            f"environment={args_cli.plot_env_id}"
        )
        axis.set_xlim(0.0, args_cli.flat_duration)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
        output_dir = Path(args_cli.output_dir).expanduser().resolve() / "contact_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (
            f"{args_cli.model.lower()}_{args_cli.fault_joint or 'random_joint'}_"
            f"seed_{args_cli.seed}_fault_{args_cli.fault_coef:.1f}_contacts.png"
        )
        figure.savefig(output_path, dpi=180)
        plt.close(figure)
        return output_path

    def evaluate_flat(env, policy, policy_module) -> dict:
        if not 0 <= args_cli.plot_env_id < env.num_envs:
            raise ValueError(f"--plot_env_id must be in [0, {env.num_envs - 1}].")

        base_env = env.unwrapped
        dt = base_env.step_dt
        max_steps = round(args_cli.flat_duration / dt)
        fault_step = round(args_cli.fault_time / dt)
        sensor = base_env.scene.sensors["contact_forces"]
        foot_ids, foot_names = sensor.find_bodies(".*_foot", preserve_order=True)
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=base_env.device)
        total_abs_vx_error = torch.tensor(0.0, device=base_env.device)
        post_abs_vx_error = torch.tensor(0.0, device=base_env.device)
        total_abs_xy_error = torch.tensor(0.0, device=base_env.device)
        total_samples = 0
        post_samples = 0
        times: list[float] = []
        contact_history: list[list[float]] = []
        plot_env_alive = True
        obs = env.get_observations()

        with torch.inference_mode():
            for step in range(max_steps):
                outputs = policy(obs)
                actions = outputs[0] if isinstance(outputs, tuple) else outputs
                obs, _, dones, _ = env.step(actions)
                dones = dones.bool()
                terminated = base_env.reset_terminated.bool()

                command = base_env.command_manager.get_command("base_velocity")
                measured_xy = base_env.scene["robot"].data.root_lin_vel_b[:, :2]
                abs_vx_error = torch.abs(command[:, 0] - measured_xy[:, 0])
                abs_xy_error = torch.linalg.vector_norm(
                    command[:, :2] - measured_xy, dim=1
                )
                if alive.any():
                    total_abs_vx_error += abs_vx_error[alive].sum()
                    total_abs_xy_error += abs_xy_error[alive].sum()
                    sample_count = int(alive.sum().item())
                    total_samples += sample_count
                    if step >= fault_step:
                        post_abs_vx_error += abs_vx_error[alive].sum()
                        post_samples += sample_count

                foot_forces = torch.linalg.vector_norm(
                    sensor.data.net_forces_w[:, foot_ids, :], dim=-1
                )
                contacts = foot_forces > args_cli.contact_threshold
                times.append((step + 1) * dt)
                if plot_env_alive:
                    contact_history.append(
                        contacts[args_cli.plot_env_id].float().cpu().tolist()
                    )
                else:
                    contact_history.append([float("nan")] * len(foot_names))
                plot_env_alive &= not bool(terminated[args_cli.plot_env_id].item())

                alive &= ~terminated
                reset_policy(policy, policy_module, dones)

        plot_path = write_contact_plot(times, contact_history, foot_names)
        return {
            "model": args_cli.model,
            "terrain": "flat",
            "fault_coefficient": float(args_cli.fault_coef),
            "fault_joint": args_cli.fault_joint,
            "seed": int(args_cli.seed),
            "fault_time_s": args_cli.fault_time,
            "command_vx_mps": 1.0,
            "num_envs": env.num_envs,
            "episode_length_s": max_steps * dt,
            "ate_vx_mps": float((total_abs_vx_error / max(total_samples, 1)).item()),
            "post_fault_ate_vx_mps": float(
                (post_abs_vx_error / max(post_samples, 1)).item()
            ),
            "ate_xy_mps": float((total_abs_xy_error / max(total_samples, 1)).item()),
            "survival_to_10s": float(alive.float().mean().item()),
            "num_resets_before_10s": int((~alive).sum().item()),
            "contact_plot": str(plot_path),
            "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        }

    @hydra_task_config(args_cli.task, args_cli.agent)
    def main(
        env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
        agent_cfg: RslRlBaseRunnerCfg,
    ):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        # Benchmark tasks share all physical environment settings, but each
        # architecture must retain the history horizon it was trained with.
        # DreamFLEX uses N=5; FTNet and the other history-based models use the
        # common 30-frame history.
        if args_cli.model == "FLEX":
            env_cfg.observations.history.history_length = args_cli.flex_history_length
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = (
            args_cli.device if args_cli.device is not None else env_cfg.sim.device
        )
        if args_cli.protocol == "rough":
            configure_rough(env_cfg)
        elif args_cli.protocol == "flat":
            configure_flat(env_cfg)
        else:
            raise ValueError("Worker protocol must be rough or flat.")

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = CustomRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner, policy, policy_module = create_runner(env, agent_cfg)
        try:
            if args_cli.protocol == "rough":
                result = evaluate_rough(env, policy, policy_module)
            else:
                result = evaluate_flat(env, policy, policy_module)
            result_path = Path(args_cli.result_json).expanduser().resolve()
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result, indent=2))
        finally:
            env.close()

    try:
        main()
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    if args_cli.worker:
        raise SystemExit(_run_worker())
    raise SystemExit(_run_parent())
