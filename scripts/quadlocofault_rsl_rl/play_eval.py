# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained RSL-RL checkpoint over a fixed terrain/fault grid."""

import argparse
import copy
import csv
import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


TERRAIN_FAMILIES = {
    "stairs": ("pyramid_stairs", "pyramid_stairs_inv"),
    "slopes": ("hf_pyramid_slope", "hf_pyramid_slope_inv"),
    "uniform_noise": ("random_rough",),
}
FAULT_COEFS = (0.0, 0.1)
DIFFICULTY_RANGE = (0.5, 0.7)


parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL over a fixed grid.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--eval_steps", type=int, default=1000, help="Number of policy steps to average metrics over.")
parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps to skip before metric collection.")
parser.add_argument("--command_x", type=float, default=1.0, help="Fixed forward command for evaluation.")
parser.add_argument("--command_y", type=float, default=0.0, help="Fixed lateral command for evaluation.")
parser.add_argument("--command_yaw", type=float, default=0.0, help="Fixed yaw-rate command for evaluation.")
parser.add_argument("--csv_path", type=str, default=None, help="Optional CSV path for the aggregated results.")
parser.add_argument("--worker", action="store_true", help="Internal flag: run a single evaluation worker.")
parser.add_argument("--terrain_family", type=str, choices=sorted(TERRAIN_FAMILIES.keys()), default=None, help="Single terrain family for a worker run.")
parser.add_argument("--fault_coef", type=float, default=None, help="Single fault coefficient for a worker run.")
parser.add_argument("--result_json", type=str, default=None, help="Worker output JSON path.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def _print_results_table(rows: list[dict[str, float | str]]) -> None:
    headers = [
        "terrain_family",
        "fault_coef",
        "difficulty_range",
        "command",
        "mean_reward",
        "abs_lin_x_err",
        "abs_lin_y_err",
        "abs_lin_xy_err",
        "abs_yaw_err",
    ]
    formatted_rows = []
    for row in rows:
        formatted_rows.append(
            {
                "terrain_family": str(row["terrain_family"]),
                "fault_coef": f"{float(row['fault_coef']):.3f}",
                "difficulty_range": str(row["difficulty_range"]),
                "command": str(row["command"]),
                "mean_reward": f"{float(row['mean_reward']):.4f}",
                "abs_lin_x_err": f"{float(row['abs_lin_x_err']):.4f}",
                "abs_lin_y_err": f"{float(row['abs_lin_y_err']):.4f}",
                "abs_lin_xy_err": f"{float(row['abs_lin_xy_err']):.4f}",
                "abs_yaw_err": f"{float(row['abs_yaw_err']):.4f}",
            }
        )
    widths = {
        header: max(len(header), *(len(row[header]) for row in formatted_rows))
        for header in headers
    }
    separator = " | ".join("-" * widths[header] for header in headers)
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print(separator)
    for row in formatted_rows:
        print(" | ".join(row[header].ljust(widths[header]) for header in headers))


def _write_results_csv(rows: list[dict[str, float | str]], csv_path: str) -> None:
    fieldnames = [
        "terrain_family",
        "fault_coef",
        "difficulty_range",
        "command",
        "mean_reward",
        "abs_lin_x_err",
        "abs_lin_y_err",
        "abs_lin_xy_err",
        "abs_yaw_err",
    ]
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Wrote evaluation table to: {output_path}")


def _run_parent() -> int:
    results = []
    log_dir = None
    with tempfile.TemporaryDirectory(prefix="play_eval_") as temp_dir:
        for terrain_family in TERRAIN_FAMILIES:
            for fault_coef in FAULT_COEFS:
                result_json = Path(temp_dir) / f"{terrain_family}_{fault_coef:.3f}.json"
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                    "--worker",
                    "--terrain_family",
                    terrain_family,
                    "--fault_coef",
                    str(fault_coef),
                    "--result_json",
                    str(result_json),
                ]
                print(
                    "[INFO] Evaluating "
                    f"terrain_family={terrain_family}, fault_prob=1.0, fault_coef={fault_coef}, "
                    f"difficulty_range={DIFFICULTY_RANGE}"
                )
                completed = subprocess.run(cmd, check=False)
                if completed.returncode != 0:
                    return completed.returncode
                result = json.loads(result_json.read_text(encoding="utf-8"))
                log_dir = result.pop("log_dir", log_dir)
                results.append(result)

    _print_results_table(results)
    csv_path = args_cli.csv_path or os.path.join(log_dir or os.getcwd(), "eval_grid_results.csv")
    _write_results_csv(results, csv_path)
    return 0


def _run_worker() -> int:
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import importlib.metadata as metadata

    import gymnasium as gym
    import torch
    from packaging import version
    from rsl_rl.runners import DistillationRunner
    from runners import CustomOnPolicyRunner

    from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, handle_deprecated_rsl_rl_cfg
    from isaaclab_quadlocofault_rl.rsl_rl import CustomRslRlVecEnvWrapper
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config

    import isaaclab_quadlocofault_tasks  # noqa: F401

    installed_version = metadata.version("rsl-rl-lib")

    def _resolve_checkpoint(train_task_name: str, agent_cfg: RslRlBaseRunnerCfg) -> tuple[str, str]:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                raise RuntimeError("A pre-trained checkpoint is currently unavailable for this task.")
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        return resume_path, os.path.dirname(resume_path)

    def _configure_eval_env(
        env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
        terrain_family: str,
        fault_coef: float,
    ) -> None:
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.curriculum.terrain_levels = None
        env_cfg.curriculum.actuator_faults = None
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.command_x, args_cli.command_x)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (args_cli.command_y, args_cli.command_y)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.command_yaw, args_cli.command_yaw)

        terrain_generator = env_cfg.scene.terrain.terrain_generator
        terrain_generator.curriculum = False
        terrain_generator.num_rows = 1
        terrain_generator.difficulty_range = DIFFICULTY_RANGE
        env_cfg.scene.terrain.max_init_terrain_level = 0
        active_subterrains = TERRAIN_FAMILIES[terrain_family]
        for _, sub_cfg in terrain_generator.sub_terrains.items():
            sub_cfg.proportion = 0.0
        active_proportion = 1.0 / float(len(active_subterrains))
        for name in active_subterrains:
            terrain_generator.sub_terrains[name].proportion = active_proportion

        fault_event = env_cfg.events.randomize_actuator_faults
        fault_event.params["severe_fault_prob"] = 1.0
        # fault_event.params["failure_coef_severe"] = fault_coef
        # fault_event.params["failure_coef_moderate"] = fault_coef

    def _create_runner(env, agent_cfg: RslRlBaseRunnerCfg, resume_path: str):
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = CustomOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        if version.parse(installed_version) >= version.parse("4.0.0"):
            return runner, policy, None
        if version.parse(installed_version) >= version.parse("2.3.0"):
            return runner, policy, runner.alg.policy
        return runner, policy, runner.alg.actor_critic

    def _evaluate_one_experiment(
        base_env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
        agent_cfg: RslRlBaseRunnerCfg,
        resume_path: str,
        log_dir: str,
        terrain_family: str,
        fault_coef: float,
    ) -> dict[str, float | str]:
        eval_env_cfg = copy.deepcopy(base_env_cfg)
        _configure_eval_env(eval_env_cfg, terrain_family, fault_coef)
        eval_env_cfg.log_dir = log_dir

        env = gym.make(args_cli.task, cfg=eval_env_cfg, render_mode=None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        env = CustomRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner, policy, policy_nn = _create_runner(env, agent_cfg, resume_path)

        obs = env.get_observations()
        sums = {
            "mean_reward": 0.0,
            "abs_lin_x_err": 0.0,
            "abs_lin_y_err": 0.0,
            "abs_lin_xy_err": 0.0,
            "abs_yaw_err": 0.0,
        }
        collected_steps = 0

        with torch.inference_mode():
            for step_idx in range(args_cli.eval_steps + args_cli.warmup_steps):
                outputs = policy(obs)
                actions = outputs[0] if isinstance(outputs, tuple) else outputs
                obs, rew, dones, _ = env.step(actions)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
                else:
                    policy_nn.reset(dones)

                if step_idx < args_cli.warmup_steps:
                    continue

                base_env = env.unwrapped
                asset = base_env.scene["robot"]
                cmd = base_env.command_manager.get_command("base_velocity")
                lin_vel_body = asset.data.root_lin_vel_b[:, :2]
                yaw_vel = asset.data.root_ang_vel_w[:, 2]
                lin_err = torch.abs(cmd[:, :2] - lin_vel_body)
                xy_err = torch.linalg.vector_norm(cmd[:, :2] - lin_vel_body, dim=1)
                yaw_err = torch.abs(cmd[:, 2] - yaw_vel)

                sums["mean_reward"] += float(rew.mean().item())
                sums["abs_lin_x_err"] += float(lin_err[:, 0].mean().item())
                sums["abs_lin_y_err"] += float(lin_err[:, 1].mean().item())
                sums["abs_lin_xy_err"] += float(xy_err.mean().item())
                sums["abs_yaw_err"] += float(yaw_err.mean().item())
                collected_steps += 1

        env.close()
        del runner, policy, policy_nn, env, obs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if collected_steps == 0:
            raise RuntimeError("No evaluation steps were collected. Check warmup/eval step settings.")

        return {
            "terrain_family": terrain_family,
            "fault_coef": fault_coef,
            "difficulty_range": f"{DIFFICULTY_RANGE[0]:.1f}-{DIFFICULTY_RANGE[1]:.1f}",
            "command": f"({args_cli.command_x:.2f}, {args_cli.command_y:.2f}, {args_cli.command_yaw:.2f})",
            "mean_reward": sums["mean_reward"] / collected_steps,
            "abs_lin_x_err": sums["abs_lin_x_err"] / collected_steps,
            "abs_lin_y_err": sums["abs_lin_y_err"] / collected_steps,
            "abs_lin_xy_err": sums["abs_lin_xy_err"] / collected_steps,
            "abs_yaw_err": sums["abs_yaw_err"] / collected_steps,
            "log_dir": log_dir,
        }

    @hydra_task_config(args_cli.task, args_cli.agent)
    def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        resume_path, log_dir = _resolve_checkpoint(train_task_name, agent_cfg)
        result = _evaluate_one_experiment(
            base_env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            resume_path=resume_path,
            log_dir=log_dir,
            terrain_family=args_cli.terrain_family,
            fault_coef=float(args_cli.fault_coef),
        )
        Path(args_cli.result_json).write_text(json.dumps(result), encoding="utf-8")

    try:
        main()
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    if args_cli.worker:
        raise SystemExit(_run_worker())
    raise SystemExit(_run_parent())
