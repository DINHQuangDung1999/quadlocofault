# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect proprioceptive histories and oracle joint-fault labels from a trained policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(
    description="Run a trained RSL-RL policy and collect an offline joint-fault dataset."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the registered task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the published pre-trained checkpoint.",
)
parser.add_argument(
    "--collection_steps",
    type=int,
    default=10_000,
    help="Number of policy steps to collect from every environment.",
)
parser.add_argument(
    "--shard_steps",
    type=int,
    default=250,
    help="Number of policy steps per saved shard. Each step contains all environments.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Dataset directory. Defaults to datasets/prop_fault/<task>/<timestamp>.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim has launched."""

import importlib.metadata as metadata
import os
import time
from typing import Any

import gymnasium as gym
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
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_quadlocofault_rl.rsl_rl import CustomRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_quadlocofault_tasks  # noqa: F401


installed_version = metadata.version("rsl-rl-lib")


class ShardedProprioceptionCollector:
    """Accumulate rollout steps on CPU and save bounded PyTorch shards."""

    _TENSOR_KEYS = (
        "policy",
        "history",
        "fault_target",
        "fault_class",
        "motor_strength",
        "action",
        "reward",
        "done",
        "env_id",
        "episode_id",
        "episode_step",
        "fault_age_steps",
    )

    def __init__(self, output_dir: str, shard_steps: int, metadata_dict: dict[str, Any]) -> None:
        if shard_steps <= 0:
            raise ValueError(f"shard_steps must be positive, got {shard_steps}.")

        self.output_dir = os.path.abspath(output_dir)
        self.shard_steps = shard_steps
        self.metadata = metadata_dict
        self.buffers: dict[str, list[torch.Tensor]] = {key: [] for key in self._TENSOR_KEYS}
        self.shard_index = 0
        self.steps_in_shard = 0
        self.total_steps = 0
        self.total_samples = 0
        self.total_positive_bits = 0
        self.total_faulty_samples = 0

        os.makedirs(self.output_dir, exist_ok=True)
        if os.listdir(self.output_dir):
            raise FileExistsError(
                f"Collection directory is not empty: {self.output_dir}. "
                "Choose a new --output_dir to avoid overwriting data."
            )

    @staticmethod
    def _to_cpu(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(device="cpu").clone()

    def append(self, sample: dict[str, torch.Tensor]) -> None:
        missing_keys = set(self._TENSOR_KEYS).difference(sample)
        if missing_keys:
            raise KeyError(f"Dataset sample is missing fields: {sorted(missing_keys)}")

        for key in self._TENSOR_KEYS:
            self.buffers[key].append(self._to_cpu(sample[key]))

        fault_target = sample["fault_target"]
        self.total_positive_bits += int(fault_target.sum().item())
        self.total_faulty_samples += int(fault_target.bool().any(dim=-1).sum().item())
        self.total_samples += fault_target.shape[0]
        self.total_steps += 1
        self.steps_in_shard += 1

        if self.steps_in_shard >= self.shard_steps:
            self.flush()

    def flush(self) -> None:
        if self.steps_in_shard == 0:
            return

        shard = {key: torch.cat(values, dim=0) for key, values in self.buffers.items()}
        shard["metadata"] = self.metadata
        shard["rollout_steps"] = self.steps_in_shard

        shard_name = f"shard_{self.shard_index:05d}.pt"
        shard_path = os.path.join(self.output_dir, shard_name)
        temporary_path = f"{shard_path}.tmp"
        torch.save(shard, temporary_path)
        os.replace(temporary_path, shard_path)

        print(
            f"[INFO] Saved {shard_name}: {shard['history'].shape[0]} samples, "
            f"history shape {tuple(shard['history'].shape[1:])}"
        )

        self.shard_index += 1
        self.steps_in_shard = 0
        self.buffers = {key: [] for key in self._TENSOR_KEYS}

    def save_summary(self) -> None:
        self.flush()
        num_joints = int(self.metadata["num_joints"])
        total_bits = self.total_samples * num_joints
        summary = {
            "format_version": 1,
            "num_shards": self.shard_index,
            "total_rollout_steps": self.total_steps,
            "total_samples": self.total_samples,
            "faulty_sample_rate": self.total_faulty_samples / max(self.total_samples, 1),
            "target_positive_rate": self.total_positive_bits / max(total_bits, 1),
            "metadata": self.metadata,
        }
        torch.save(summary, os.path.join(self.output_dir, "summary.pt"))
        print(
            "[INFO] Collection complete: "
            f"{summary['total_samples']} samples, "
            f"faulty sample rate={summary['faulty_sample_rate']:.4f}, "
            f"target positive rate={summary['target_positive_rate']:.4f}"
        )
        print(f"[INFO] Dataset directory: {self.output_dir}")


def _default_output_dir(task_name: str) -> str:
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    safe_task_name = task_name.replace(":", "_").replace("/", "_")
    return os.path.join("datasets", "prop_fault", safe_task_name, timestamp)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    """Run the policy and collect observations with simulator-oracle labels."""
    if args_cli.collection_steps <= 0:
        raise ValueError(f"collection_steps must be positive, got {args_cli.collection_steps}.")

    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    )
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            raise FileNotFoundError(f"No published checkpoint is available for {train_task_name}.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    env_cfg.log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = CustomRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
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
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    required_obs = {"policy", "history"}
    missing_obs = required_obs.difference(obs.keys())
    if missing_obs:
        raise KeyError(
            f"Task {args_cli.task} does not provide required observations: {sorted(missing_obs)}"
        )

    asset = env.unwrapped.scene["robot"]
    if not hasattr(asset, "faulty_joint_idx") or not hasattr(asset, "motors_strength"):
        raise AttributeError(
            "The robot must expose faulty_joint_idx and motors_strength oracle tensors."
        )

    output_dir = args_cli.output_dir or _default_output_dir(task_name)
    dataset_metadata = {
        "format_version": 1,
        "task": args_cli.task,
        "checkpoint": os.path.abspath(resume_path),
        "seed": int(agent_cfg.seed),
        "step_dt": float(env.unwrapped.step_dt),
        "num_envs": int(env.num_envs),
        "num_joints": int(asset.faulty_joint_idx.shape[-1]),
        "joint_names": list(asset.joint_names),
        "policy_shape": tuple(obs["policy"].shape[1:]),
        "history_shape": tuple(obs["history"].shape[1:]),
        "fault_class_convention": "0..num_joints-1=faulty joint, num_joints=no fault",
        "fault_age_steps": "consecutive policy steps since the current oracle label appeared",
        "flattening": "step-major; each rollout step contains env_id 0..num_envs-1",
    }
    collector = ShardedProprioceptionCollector(
        output_dir=output_dir,
        shard_steps=args_cli.shard_steps,
        metadata_dict=dataset_metadata,
    )

    env_ids = torch.arange(env.num_envs, device=env.unwrapped.device, dtype=torch.long)
    episode_ids = env_ids.clone()
    next_episode_id = env.num_envs
    previous_fault_target: torch.Tensor | None = None
    fault_age_steps = torch.zeros_like(env_ids)

    try:
        with torch.inference_mode():
            for step_index in range(args_cli.collection_steps):
                if not simulation_app.is_running():
                    print("[INFO] Simulator stopped before the requested collection length.")
                    break

                fault_target = asset.faulty_joint_idx.to(dtype=torch.float32).clone()
                if previous_fault_target is None:
                    fault_age_steps.zero_()
                else:
                    fault_changed = (fault_target != previous_fault_target).any(dim=-1)
                    fault_age_steps = torch.where(
                        fault_changed, torch.zeros_like(fault_age_steps), fault_age_steps + 1
                    )
                previous_fault_target = fault_target.clone()

                has_fault = fault_target.bool().any(dim=-1)
                fault_class = fault_target.argmax(dim=-1).to(dtype=torch.long)
                fault_class = torch.where(
                    has_fault,
                    fault_class,
                    torch.full_like(fault_class, fault_target.shape[-1]),
                )

                outputs = policy(obs)
                actions = outputs[0] if isinstance(outputs, tuple) else outputs

                pre_step_sample = {
                    # Clone before env.step() so every input stays aligned with its
                    # pre-step oracle label even if simulator buffers are reused.
                    "policy": obs["policy"].clone(),
                    "history": obs["history"].clone(),
                    "fault_target": fault_target,
                    "fault_class": fault_class.unsqueeze(-1),
                    "motor_strength": asset.motors_strength.clone(),
                    "action": actions.clone(),
                    "env_id": env_ids.unsqueeze(-1),
                    "episode_id": episode_ids.unsqueeze(-1),
                    "episode_step": env.unwrapped.episode_length_buf.clone().unsqueeze(-1),
                    "fault_age_steps": fault_age_steps.clone().unsqueeze(-1),
                }

                obs, rewards, dones, _ = env.step(actions)
                policy.reset(dones)

                pre_step_sample["reward"] = rewards.unsqueeze(-1)
                pre_step_sample["done"] = dones.unsqueeze(-1)
                collector.append(pre_step_sample)

                done_mask = dones.bool()
                num_dones = int(done_mask.sum().item())
                if num_dones:
                    # The next observation belongs to a new episode. Set age to
                    # -1 so its first collected sample becomes age zero.
                    fault_age_steps[done_mask] = -1
                    previous_fault_target[done_mask] = asset.faulty_joint_idx[
                        done_mask
                    ].to(dtype=previous_fault_target.dtype)
                    episode_ids[done_mask] = torch.arange(
                        next_episode_id,
                        next_episode_id + num_dones,
                        device=episode_ids.device,
                        dtype=episode_ids.dtype,
                    )
                    next_episode_id += num_dones

                if (step_index + 1) % 100 == 0:
                    print(
                        f"[INFO] Collected {step_index + 1}/{args_cli.collection_steps} "
                        f"rollout steps ({collector.total_samples} samples)."
                    )
    finally:
        collector.save_summary()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
