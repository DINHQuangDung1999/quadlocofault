# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play an RSL-RL policy and collect DreamFLEX/FT-Net evaluation data.

DreamFLEX reports ATE as the absolute error between commanded and measured
linear velocity (m/s), along with foot-contact histories. FT-Net compares the
commanded and measured velocity through time. This evaluator saves all three.
"""

import argparse
import importlib.metadata as metadata
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play an RSL-RL policy and record velocity-tracking ATE.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the evaluation.")
parser.add_argument("--video_length", type=int, default=None, help="Video length in steps (defaults to the evaluation length).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of parallel environments (paper: 100).")
parser.add_argument("--task", type=str, default=None, help="Gym task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use a published checkpoint.")
parser.add_argument("--duration", type=float, default=10.0, help="Metric collection duration in seconds (paper: 10).")
parser.add_argument("--warmup", type=float, default=0.0, help="Warm-up duration excluded from metrics, in seconds.")
parser.add_argument("--num_episodes", type=int, default=1, help="Number of fixed-duration evaluation episodes.")
parser.add_argument(
    "--distance_failure_ratio",
    type=float,
    default=0.5,
    help="Fail an episode when traveled distance is below this fraction of commanded distance.",
)
parser.add_argument("--command_x", type=float, default=1.0, help="Fixed forward velocity command in m/s.")
parser.add_argument("--command_y", type=float, default=0.0, help="Fixed lateral velocity command in m/s.")
parser.add_argument("--command_yaw", type=float, default=0.0, help="Fixed yaw-rate command in rad/s.")
parser.add_argument("--command_name", type=str, default="base_velocity", help="Velocity command term name.")
parser.add_argument("--asset_name", type=str, default="robot", help="Scene articulation used for velocity measurement.")
parser.add_argument("--metrics_path", type=str, default=None, help="Output JSON path (default: checkpoint/eval/ate.json).")
parser.add_argument("--timeseries_npz", type=str, default=None, help="NPZ path (default: checkpoint/eval/timeseries.npz).")
parser.add_argument("--plot_path", type=str, default=None, help="Plot path (default: checkpoint/eval/tracking.png).")
parser.add_argument("--contact_sensor_name", type=str, default="contact_forces", help="Contact sensor scene key.")
parser.add_argument("--foot_body_regex", type=str, default=".*_foot", help="Contact-sensor foot body regex.")
parser.add_argument("--contact_threshold", type=float, default=1.0, help="Foot-contact force threshold in N.")
parser.add_argument("--plot_env_id", type=int, default=0, help="Environment shown in contact/tracking curves.")
parser.add_argument("--real-time", action="store_true", default=False, help="Throttle simulation to wall-clock time.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner
from runners import CustomOnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_quadlocofault_rl.rsl_rl import CustomRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaaclab_quadlocofault_tasks  # noqa: F401


installed_version = metadata.version("rsl-rl-lib")


def _set_fixed_velocity_command(env_cfg) -> None:
    """Make the requested velocity command deterministic for the whole evaluation."""
    try:
        command_cfg = getattr(env_cfg.commands, args_cli.command_name)
    except AttributeError as exc:
        raise ValueError(f"Task has no command term named '{args_cli.command_name}'.") from exc
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_standing_envs = 0.0
    command_cfg.resampling_time_range = (args_cli.duration + args_cli.warmup + 1.0,) * 2
    command_cfg.ranges.lin_vel_x = (args_cli.command_x, args_cli.command_x)
    command_cfg.ranges.lin_vel_y = (args_cli.command_y, args_cli.command_y)
    command_cfg.ranges.ang_vel_z = (args_cli.command_yaw, args_cli.command_yaw)


def _write_npz(
    path: str,
    rows: list[dict[str, float]],
    episode_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    """Store each time-series field as an ``[episode, step]`` array."""
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    episode_ids = sorted({int(row["episode"]) for row in rows})
    steps_per_episode = len(rows) // len(episode_ids)
    arrays = {
        key: np.asarray([row[key] for row in rows]).reshape(len(episode_ids), steps_per_episode)
        for key in rows[0]
        if key != "episode"
    }
    if episode_arrays:
        arrays.update(episode_arrays)
    np.savez_compressed(output_path, episode=np.asarray(episode_ids), **arrays)
    print(f"[INFO] Wrote ATE time-series arrays to: {output_path}")


def _write_plot(path: str, rows: list[dict[str, float]], foot_names: list[str]) -> None:
    """Write DreamFLEX-style contact/error and FT-Net-style velocity curves."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is unavailable; skipping the evaluation plot.")
        return

    time_s = [row["time_s"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for offset, name in enumerate(foot_names):
        axes[0].step(
            time_s,
            [row[f"env_contact_{name}"] + offset for row in rows],
            where="post",
            label=name,
        )
    axes[0].set_yticks(range(len(foot_names)), foot_names)
    axes[0].set_ylabel("Foot contact")
    axes[0].set_title(f"Foot contact — environment {args_cli.plot_env_id}")
    axes[0].grid(alpha=0.25)

    axes[1].plot(time_s, [row["env_cmd_vx_mps"] for row in rows], "--", label="command vx")
    axes[1].plot(time_s, [row["env_measured_vx_mps"] for row in rows], label="measured vx")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_title("Velocity tracking curve (FT-Net-style)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[1].set_xlabel("Time (s)")

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[INFO] Wrote evaluation plot to: {output_path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Load a checkpoint, play it for a finite duration, and save ATE."""
    if (
        args_cli.duration <= 0.0
        or args_cli.warmup < 0.0
        or args_cli.num_episodes <= 0
        or not 0.0 <= args_cli.distance_failure_ratio <= 1.0
    ):
        raise ValueError(
            "--duration and --num_episodes must be positive, --warmup must be non-negative, "
            "and --distance_failure_ratio must be in [0, 1]."
        )

    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _set_fixed_velocity_command(env_cfg)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            raise RuntimeError("A published checkpoint is unavailable for this task.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    dt = env.unwrapped.step_dt
    warmup_steps = round(args_cli.warmup / dt)
    eval_steps = round(args_cli.duration / dt)
    if args_cli.plot_env_id < 0 or args_cli.plot_env_id >= env.unwrapped.num_envs:
        raise ValueError(f"--plot_env_id must be in [0, {env.unwrapped.num_envs - 1}].")
    if args_cli.video:
        video_length = args_cli.video_length or warmup_steps + eval_steps
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "eval"),
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = CustomRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = CustomOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = None
    if version.parse(installed_version) < version.parse("4.0.0"):
        policy_nn = runner.alg.policy if version.parse(installed_version) >= version.parse("2.3.0") else runner.alg.actor_critic

    obs = env.get_observations()
    sum_abs_error = torch.zeros(3, device=env.unwrapped.device)
    sum_xy_norm_error = torch.tensor(0.0, device=env.unwrapped.device)
    per_env_x_error = torch.zeros(env.num_envs, device=env.unwrapped.device)
    base_env = env.unwrapped
    contact_sensor = base_env.scene[args_cli.contact_sensor_name]
    foot_ids, foot_names = contact_sensor.find_bodies(args_cli.foot_body_regex, preserve_order=True)
    if not foot_ids:
        raise RuntimeError(
            f"Foot regex {args_cli.foot_body_regex!r} matched no bodies in sensor "
            f"{args_cli.contact_sensor_name!r}."
        )
    contact_counts = torch.zeros(len(foot_ids), device=env.unwrapped.device)
    rows: list[dict[str, float]] = []
    episode_ates: list[float] = []
    episode_traveled_distances: list[np.ndarray] = []
    episode_expected_distances: list[np.ndarray] = []
    episode_reset_flags: list[np.ndarray] = []
    episode_failure_flags: list[np.ndarray] = []

    with torch.inference_mode():
        for episode in range(args_cli.num_episodes):
            if episode > 0:
                env.reset()
                obs = env.get_observations()
                reset_dones = torch.ones(env.num_envs, dtype=torch.bool, device=env.unwrapped.device)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(reset_dones)
                else:
                    policy_nn.reset(reset_dones)

            episode_x_error = 0.0
            traveled_distance = torch.zeros(env.num_envs, device=env.unwrapped.device)
            expected_distance = torch.zeros(env.num_envs, device=env.unwrapped.device)
            reset_during_measurement = torch.zeros(env.num_envs, dtype=torch.bool, device=env.unwrapped.device)
            asset = base_env.scene[args_cli.asset_name]
            previous_xy = asset.data.root_pos_w[:, :2].clone()
            for step in range(warmup_steps + eval_steps):
                start_time = time.time()
                outputs = policy(obs)
                actions = outputs[0] if isinstance(outputs, tuple) else outputs
                obs, _, dones, _ = env.step(actions)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
                else:
                    policy_nn.reset(dones)

                current_xy = asset.data.root_pos_w[:, :2].clone()
                if step >= warmup_steps:
                    command = base_env.command_manager.get_command(args_cli.command_name)
                    measured = torch.cat((asset.data.root_lin_vel_b[:, :2], asset.data.root_ang_vel_b[:, 2:3]), dim=1)
                    abs_error = torch.abs(command[:, :3] - measured)
                    reset_mask = dones.bool()
                    traveled_distance += torch.linalg.vector_norm(current_xy - previous_xy, dim=1) * (~reset_mask)
                    expected_distance += torch.linalg.vector_norm(command[:, :2], dim=1) * dt
                    reset_during_measurement |= reset_mask
                    foot_forces = torch.linalg.vector_norm(
                        contact_sensor.data.net_forces_w[:, foot_ids, :], dim=-1
                    )
                    foot_contacts = foot_forces > args_cli.contact_threshold
                    step_x_error = float(abs_error[:, 0].mean().item())
                    episode_x_error += step_x_error
                    sum_abs_error += abs_error.mean(dim=0)
                    sum_xy_norm_error += torch.linalg.vector_norm(command[:, :2] - measured[:, :2], dim=1).mean()
                    per_env_x_error += abs_error[:, 0]
                    contact_counts += foot_contacts.float().mean(dim=0)
                    row = {
                        "episode": episode,
                        "time_s": (step - warmup_steps + 1) * dt,
                        "ate_x_mps": step_x_error,
                        "abs_y_error_mps": float(abs_error[:, 1].mean().item()),
                        "abs_yaw_error_radps": float(abs_error[:, 2].mean().item()),
                        "mean_cmd_vx_mps": float(command[:, 0].mean().item()),
                        "mean_measured_vx_mps": float(measured[:, 0].mean().item()),
                        "env_cmd_vx_mps": float(command[args_cli.plot_env_id, 0].item()),
                        "env_measured_vx_mps": float(measured[args_cli.plot_env_id, 0].item()),
                    }
                    for foot_index, foot_name in enumerate(foot_names):
                        row[f"env_contact_force_{foot_name}_N"] = float(
                            foot_forces[args_cli.plot_env_id, foot_index].item()
                        )
                        row[f"env_contact_{foot_name}"] = int(
                            foot_contacts[args_cli.plot_env_id, foot_index].item()
                        )
                    rows.append(row)

                previous_xy = current_xy
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0.0:
                    time.sleep(sleep_time)

            episode_ates.append(episode_x_error / eval_steps)
            distance_failure = traveled_distance < args_cli.distance_failure_ratio * expected_distance
            failure = reset_during_measurement | distance_failure
            episode_traveled_distances.append(traveled_distance.cpu().numpy())
            episode_expected_distances.append(expected_distance.cpu().numpy())
            episode_reset_flags.append(reset_during_measurement.cpu().numpy())
            episode_failure_flags.append(failure.cpu().numpy())
            print(f"[INFO] Completed episode {episode + 1}/{args_cli.num_episodes}")

    total_eval_steps = eval_steps * args_cli.num_episodes
    mean_abs_error = sum_abs_error / total_eval_steps
    traveled_distance_array = np.stack(episode_traveled_distances)
    expected_distance_array = np.stack(episode_expected_distances)
    reset_flag_array = np.stack(episode_reset_flags)
    failure_flag_array = np.stack(episode_failure_flags)
    metrics = {
        "definition": "mean_t,env(abs(commanded_body_vx - measured_body_vx))",
        "ate_mps": float(mean_abs_error[0].item()),
        "abs_lateral_error_mps": float(mean_abs_error[1].item()),
        "abs_xy_vector_error_mps": float((sum_xy_norm_error / total_eval_steps).item()),
        "abs_yaw_rate_error_radps": float(mean_abs_error[2].item()),
        "per_env_ate_mps": (per_env_x_error / total_eval_steps).cpu().tolist(),
        "per_episode_ate_mps": episode_ates,
        "locomotion_failure_definition": (
            "reset during measurement OR traveled_distance < "
            f"{args_cli.distance_failure_ratio:.3f} * commanded_distance"
        ),
        "locomotion_failure_rate": float(failure_flag_array.mean()),
        "num_locomotion_failures": int(failure_flag_array.sum()),
        "num_episode_env_trials": int(failure_flag_array.size),
        "distance_failure_ratio": args_cli.distance_failure_ratio,
        "foot_names": foot_names,
        "contact_threshold_N": args_cli.contact_threshold,
        "mean_foot_contact_ratio": {
            name: float((contact_counts[index] / total_eval_steps).item()) for index, name in enumerate(foot_names)
        },
        "command": {"vx_mps": args_cli.command_x, "vy_mps": args_cli.command_y, "yaw_rate_radps": args_cli.command_yaw},
        "duration_s": eval_steps * dt,
        "warmup_s": warmup_steps * dt,
        "step_dt_s": dt,
        "num_envs": env.num_envs,
        "num_episodes": args_cli.num_episodes,
        "num_samples": total_eval_steps * env.num_envs,
        "task": args_cli.task,
        "checkpoint": os.path.abspath(resume_path),
    }
    metrics_path = Path(args_cli.metrics_path or os.path.join(log_dir, "eval", "ate.json")).expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    timeseries_path = args_cli.timeseries_npz or os.path.join(log_dir, "eval", "timeseries.npz")
    plot_path = args_cli.plot_path or os.path.join(log_dir, "eval", "tracking.png")
    _write_npz(
        timeseries_path,
        rows,
        episode_arrays={
            "traveled_distance_m": traveled_distance_array,
            "expected_distance_m": expected_distance_array,
            "reset_during_measurement": reset_flag_array,
            "locomotion_failure": failure_flag_array,
        },
    )
    if args_cli.num_episodes == 1:
        _write_plot(plot_path, rows, foot_names)

    print("\n[RESULT] DreamFLEX forward-velocity ATE")
    print(f"  Average ATE: {metrics['ate_mps']:.6f} m/s")
    if args_cli.num_episodes > 1:
        print(f"  Failure rate: {metrics['locomotion_failure_rate']:.2%}")
    print(
        f"  samples:     {metrics['num_samples']} "
        f"({args_cli.num_episodes} episodes x {env.num_envs} envs x {eval_steps} steps)"
    )
    print(f"  metrics:   {metrics_path}")
    print(f"  time series: {Path(timeseries_path).expanduser().resolve()}")
    if args_cli.num_episodes == 1:
        print(f"  plot:        {Path(plot_path).expanduser().resolve()}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
