# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import sys
from collections import deque

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import math

FAULT_JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--terrain_type",
    type=str,
    default=None,
    help="Use only the named terrain from the environment's configured terrain mix.",
)
parser.add_argument(
    "--fault_joint",
    choices=FAULT_JOINT_NAMES,
    default=None,
    help="Apply the play-mode actuator fault to this fixed joint instead of sampling one randomly.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--fault_tcn_checkpoint",
    type=str,
    default=None,
    help=(
        "Optional offline FaultResidualTCN checkpoint used to predict faults from observation "
        "history. Fault prediction is disabled when omitted."
    ),
)
parser.add_argument(
    "--fault_threshold",
    type=float,
    default=0.5,
    help="Sigmoid probability threshold used to declare a joint faulty.",
)
parser.add_argument(
    "--fault_print_interval",
    type=int,
    default=15,
    help="Print TCN fault predictions every N simulation steps (zero disables printing).",
)
parser.add_argument(
    "--collect_fused_latent",
    action="store_true",
    default=False,
    help="Collect the EquivGCN actor's fused latent and save a fault-colored t-SNE plot.",
)
parser.add_argument(
    "--latent_collect_step",
    type=int,
    default=50,
    help="Simulation step at which to collect fused latents from all environments.",
)
parser.add_argument(
    "--latent_tsne_output",
    type=str,
    default="fused_latent_tsne.pdf",
    help="Output path for the fused-latent t-SNE PDF.",
)
parser.add_argument(
    "--latent_npz_output",
    type=str,
    default=None,
    help=(
        "Output path for the fused-latent NPZ data. When omitted, the t-SNE "
        "output path is used with an .npz suffix."
    ),
)
parser.add_argument(
    "--latent_tsne_perplexity",
    type=float,
    default=30.0,
    help="Perplexity used by t-SNE.",
)
parser.add_argument(
    "--export",
    choices=("jit", "onnx", "both"),
    default=None,
    help="Export the loaded deterministic policy next to its checkpoint.",
)
parser.add_argument(
    "--export_only",
    action="store_true",
    default=False,
    help="Exit after exporting; requires --export.",
)
parser.add_argument(
    "--log_fault_motion",
    action="store_true",
    default=False,
    help="Log faulted-leg thigh/calf targets and measured motion to CSV.",
)
parser.add_argument(
    "--fault_motion_output",
    type=str,
    default=None,
    help="Fault-motion CSV path; defaults beside the loaded checkpoint.",
)
parser.add_argument(
    "--fault_motion_env",
    type=int,
    default=0,
    help="Environment index used for fault-motion diagnostics.",
)
parser.add_argument(
    "--fault_motion_print_interval",
    type=int,
    default=100,
    help="Print online motion-source statistics every N logged samples.",
)
parser.add_argument(
    "--fault_motion_window",
    type=int,
    default=200,
    help="Rolling sample window used for online motion-source statistics.",
)
parser.add_argument(
    "--fault_motion_steps",
    type=int,
    default=0,
    help="Stop after this many simulation steps when logging; zero runs indefinitely.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner
from runners import CustomOnPolicyRunner
from models.equiv_gcn_actor import FaultResidualTCN

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_quadlocofault_rl.rsl_rl import CustomRslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_quadlocofault_tasks  # noqa: F401
import isaacsim.core.utils.stage as stage_utils
from pxr import UsdPhysics, UsdGeom, Gf, Sdf
if not args_cli.headless:
    import omni.ui as ui 


def load_fault_tcn(checkpoint_path: str, device: torch.device | str) -> tuple[FaultResidualTCN, dict]:
    """Create the offline fault classifier and restore its trained weights."""
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Fault TCN checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_class") != "FaultResidualTCN":
        raise ValueError(
            f"Expected a FaultResidualTCN checkpoint, got {checkpoint.get('model_class')!r}."
        )

    model_kwargs = checkpoint.get("model_kwargs")
    classifier_state_dict = checkpoint.get("classifier_state_dict")
    if not isinstance(model_kwargs, dict) or not isinstance(classifier_state_dict, dict):
        raise KeyError(
            "Fault TCN checkpoint must contain model_kwargs and classifier_state_dict."
        )

    model = FaultResidualTCN(**model_kwargs).to(device)
    incompatible = model.load_state_dict(classifier_state_dict, strict=False)
    expected_missing = {"film_head.weight", "film_head.bias"}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Fault TCN weights do not match the model: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    print(
        f"[INFO] Loaded fault TCN from {checkpoint_path} "
        f"(epoch {checkpoint.get('epoch', 'unknown')})."
    )
    return model, checkpoint


def print_fault_predictions(
    probabilities: torch.Tensor,
    joint_names: list[str],
    threshold: float,
) -> None:
    """Print one readable prediction line per simulated environment."""
    predicted_mask = probabilities >= threshold
    probabilities_cpu = probabilities.detach().cpu()
    predicted_mask_cpu = predicted_mask.detach().cpu()
    for env_id, (env_probabilities, env_mask) in enumerate(
        zip(probabilities_cpu, predicted_mask_cpu)
    ):
        predicted = [
            f"{joint_names[joint_id]}={env_probabilities[joint_id]:.3f}"
            for joint_id in env_mask.nonzero(as_tuple=False).flatten().tolist()
        ]
        if not predicted:
            most_likely_id = int(env_probabilities.argmax())
            prediction = (
                f"healthy (highest: {joint_names[most_likely_id]}="
                f"{env_probabilities[most_likely_id]:.3f})"
            )
        else:
            prediction = ", ".join(predicted)
        print(f"[FAULT TCN] env={env_id}: {prediction}")


class FaultMotionLogger:
    """Record commanded and measured thigh/calf motion on one faulted leg."""

    JOINT_TYPES = ("thigh", "calf")

    def __init__(
        self,
        base_env,
        env_id: int,
        output_path: str,
        dt: float,
        print_interval: int,
        window_size: int,
    ) -> None:
        if not 0 <= env_id < base_env.num_envs:
            raise ValueError(
                f"--fault_motion_env must be in [0, {base_env.num_envs - 1}], got {env_id}."
            )
        if print_interval <= 0:
            raise ValueError("--fault_motion_print_interval must be positive.")
        if window_size < 2:
            raise ValueError("--fault_motion_window must be at least 2.")

        self.asset = base_env.scene["robot"]
        self.action_term = base_env.action_manager.get_term("joint_pos")
        self.env_id = env_id
        self.dt = dt
        self.print_interval = print_interval
        self.window_size = window_size
        self.asset_joint_id = {
            name: joint_id for joint_id, name in enumerate(self.asset.joint_names)
        }
        self.action_joint_id = {
            name: joint_id
            for joint_id, name in enumerate(self.action_term._joint_names)
        }

        output_path = os.path.abspath(os.path.expanduser(output_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.output_path = output_path
        self.file = open(output_path, "w", newline="", encoding="utf-8")
        fieldnames = ["step", "time_s", "env_id", "fault_joint", "motor_strength"]
        for joint_type in self.JOINT_TYPES:
            fieldnames.extend(
                [
                    f"{joint_type}_raw_action",
                    f"{joint_type}_processed_target",
                    f"{joint_type}_applied_target",
                    f"{joint_type}_position",
                    f"{joint_type}_velocity",
                    f"{joint_type}_applied_torque",
                    f"{joint_type}_position_error",
                    f"{joint_type}_target_rate",
                ]
            )
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.current_fault: str | None = None
        self.previous_target: dict[str, float] = {}
        self.target_rate = {
            joint_type: deque(maxlen=window_size) for joint_type in self.JOINT_TYPES
        }
        self.joint_velocity = {
            joint_type: deque(maxlen=window_size) for joint_type in self.JOINT_TYPES
        }
        self.logged_samples = 0

    def _reset_window(self, fault_name: str | None) -> None:
        self.current_fault = fault_name
        self.previous_target.clear()
        for values in self.target_rate.values():
            values.clear()
        for values in self.joint_velocity.values():
            values.clear()

    def _applied_target(self, action_id: int, processed_target: float) -> float:
        constrained_mask = getattr(self.action_term, "_constrained_mask", None)
        lower_limits = getattr(self.action_term, "_constraint_lower_limits", None)
        upper_limits = getattr(self.action_term, "_constraint_upper_limits", None)
        if (
            constrained_mask is None
            or lower_limits is None
            or upper_limits is None
            or not bool(constrained_mask[action_id].item())
        ):
            return processed_target
        lower = float(lower_limits[action_id].item())
        upper = float(upper_limits[action_id].item())
        return min(max(processed_target, lower), upper)

    @staticmethod
    def _rms(values: deque[float]) -> float:
        return math.sqrt(sum(value * value for value in values) / len(values))

    @staticmethod
    def _correlation(first: deque[float], second: deque[float]) -> float:
        first_mean = sum(first) / len(first)
        second_mean = sum(second) / len(second)
        first_centered = [value - first_mean for value in first]
        second_centered = [value - second_mean for value in second]
        covariance = sum(a * b for a, b in zip(first_centered, second_centered))
        first_norm = math.sqrt(sum(value * value for value in first_centered))
        second_norm = math.sqrt(sum(value * value for value in second_centered))
        denominator = first_norm * second_norm
        return covariance / denominator if denominator > 1.0e-8 else 0.0

    def _motion_source(self, joint_type: str) -> tuple[str, float, float, float]:
        target_rate = self.target_rate[joint_type]
        velocity = self.joint_velocity[joint_type]
        target_rate_rms = self._rms(target_rate)
        velocity_rms = self._rms(velocity)
        correlation = self._correlation(target_rate, velocity)

        if velocity_rms < 0.1:
            source = "quiet"
        elif target_rate_rms < 0.25 * velocity_rms and abs(correlation) < 0.25:
            source = "mostly-passive"
        elif target_rate_rms >= 0.25 * velocity_rms and correlation > 0.35:
            source = "target-driven"
        else:
            source = "mixed/unclear"
        return source, target_rate_rms, velocity_rms, correlation

    def record(self, step: int) -> None:
        fault_ids = torch.nonzero(
            self.asset.faulty_joint_idx[self.env_id], as_tuple=False
        ).flatten()
        if fault_ids.numel() == 0:
            if self.current_fault is not None:
                self._reset_window(None)
            return

        fault_id = int(fault_ids[0].item())
        fault_name = self.asset.joint_names[fault_id]
        leg_prefix = fault_name[:2]
        if fault_name != self.current_fault:
            self._reset_window(fault_name)

        row: dict[str, str | int | float] = {
            "step": step,
            "time_s": step * self.dt,
            "env_id": self.env_id,
            "fault_joint": fault_name,
            "motor_strength": float(
                self.asset.motors_strength[self.env_id, fault_id].item()
            ),
        }
        for joint_type in self.JOINT_TYPES:
            joint_name = f"{leg_prefix}_{joint_type}_joint"
            if joint_name not in self.asset_joint_id or joint_name not in self.action_joint_id:
                raise KeyError(
                    f"Could not map diagnostic joint {joint_name!r} into asset/action order."
                )
            asset_id = self.asset_joint_id[joint_name]
            action_id = self.action_joint_id[joint_name]
            raw_action = float(
                self.action_term.raw_actions[self.env_id, action_id].item()
            )
            processed_target = float(
                self.action_term.processed_actions[self.env_id, action_id].item()
            )
            applied_target = self._applied_target(action_id, processed_target)
            position = float(self.asset.data.joint_pos[self.env_id, asset_id].item())
            velocity = float(self.asset.data.joint_vel[self.env_id, asset_id].item())
            torque = float(
                self.asset.data.applied_torque[self.env_id, asset_id].item()
            )
            previous_target = self.previous_target.get(joint_type)
            target_rate = (
                0.0
                if previous_target is None
                else (applied_target - previous_target) / self.dt
            )
            self.previous_target[joint_type] = applied_target
            if previous_target is not None:
                self.target_rate[joint_type].append(target_rate)
                self.joint_velocity[joint_type].append(velocity)

            row.update(
                {
                    f"{joint_type}_raw_action": raw_action,
                    f"{joint_type}_processed_target": processed_target,
                    f"{joint_type}_applied_target": applied_target,
                    f"{joint_type}_position": position,
                    f"{joint_type}_velocity": velocity,
                    f"{joint_type}_applied_torque": torque,
                    f"{joint_type}_position_error": applied_target - position,
                    f"{joint_type}_target_rate": target_rate,
                }
            )

        self.writer.writerow(row)
        self.logged_samples += 1
        if self.logged_samples % self.print_interval == 0:
            self.file.flush()
            summaries = []
            for joint_type in self.JOINT_TYPES:
                if len(self.target_rate[joint_type]) < 2:
                    continue
                source, target_rms, velocity_rms, correlation = self._motion_source(
                    joint_type
                )
                summaries.append(
                    f"{joint_type}: source={source}, target_rate_rms={target_rms:.3f}, "
                    f"velocity_rms={velocity_rms:.3f}, corr={correlation:.3f}"
                )
            print(
                f"[FAULT MOTION] env={self.env_id}, fault={fault_name}, "
                + "; ".join(summaries)
            )

    def close(self) -> None:
        if not self.file.closed:
            self.file.flush()
            self.file.close()
            print(f"[INFO] Saved fault-motion diagnostics to: {self.output_path}")


def compute_fused_latent(actor, history: torch.Tensor) -> torch.Tensor:
    """Reproduce the EquivGCN actor's fused latent for a batch of histories."""
    required_attributes = (
        "obs_hist_normalizer",
        "gcn_encoder",
        "fault_residual_encoder",
    )
    missing = [name for name in required_attributes if not hasattr(actor, name)]
    if missing:
        raise TypeError(
            "Fused-latent collection requires an EquivGCNActor; "
            f"the loaded actor is missing {missing}."
        )

    normalized_history = actor.obs_hist_normalizer(history)
    gcn_latent = actor.gcn_encoder(normalized_history).mean(dim=1)
    fault_logits, gamma, beta = actor.fault_residual_encoder(normalized_history)
    fault_probability = torch.sigmoid(fault_logits)
    fault_gate = fault_probability.amax(dim=1, keepdim=True)
    return (1.0 + fault_gate * gamma) * gcn_latent + fault_gate * beta


def save_fused_latent_tsne(
    fused_latent: torch.Tensor,
    fault_mask: torch.Tensor,
    joint_names: list[str],
    output_path: str,
    npz_output_path: str | None,
    perplexity: float,
    seed: int,
) -> None:
    """Save fused latent data and its fault-colored t-SNE projection."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "Fused-latent plotting requires matplotlib and scikit-learn."
        ) from exc

    latent_numpy = fused_latent.detach().float().cpu().numpy()
    fault_mask_cpu = fault_mask.detach().bool().cpu()
    if latent_numpy.shape[0] < 2:
        raise ValueError("t-SNE requires latent vectors from at least two environments.")
    if not 0.0 < perplexity < latent_numpy.shape[0]:
        raise ValueError(
            f"t-SNE perplexity must be between 0 and the sample count "
            f"({latent_numpy.shape[0]}), got {perplexity}."
        )

    healthy_class = len(joint_names)
    fault_labels = torch.full(
        (fault_mask_cpu.shape[0],),
        healthy_class,
        dtype=torch.long,
    )
    has_fault = fault_mask_cpu.any(dim=1)
    fault_labels[has_fault] = fault_mask_cpu[has_fault].float().argmax(dim=1)
    fault_labels_numpy = fault_labels.numpy()

    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(latent_numpy)

    output_path = os.path.abspath(os.path.expanduser(output_path))
    if npz_output_path is None:
        npz_output_path = os.path.splitext(output_path)[0] + ".npz"
    else:
        npz_output_path = os.path.abspath(os.path.expanduser(npz_output_path))

    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    npz_output_directory = os.path.dirname(npz_output_path)
    if npz_output_directory:
        os.makedirs(npz_output_directory, exist_ok=True)

    np.savez_compressed(
        npz_output_path,
        fused_latent=latent_numpy,
        tsne_embedding=embedding,
        fault_labels=fault_labels_numpy,
        fault_mask=fault_mask_cpu.numpy(),
        joint_names=np.asarray(joint_names),
        healthy_class=np.asarray(healthy_class, dtype=np.int64),
        collect_step=np.asarray(args_cli.latent_collect_step, dtype=np.int64),
        tsne_perplexity=np.asarray(perplexity, dtype=np.float64),
        seed=np.asarray(seed, dtype=np.int64),
    )

    figure, axis = plt.subplots(figsize=(10, 8))
    colors = plt.get_cmap("tab20", healthy_class + 1)
    present_classes = np.unique(fault_labels_numpy)
    for class_id in present_classes:
        class_points = fault_labels_numpy == class_id
        class_name = "healthy" if class_id == healthy_class else joint_names[class_id]
        axis.scatter(
            embedding[class_points, 0],
            embedding[class_points, 1],
            s=10,
            alpha=0.7,
            color=colors(class_id),
            label=f"{class_name} (n={class_points.sum()})",
            rasterized=True,
        )
    axis.set_title(f"EquivGCN fused latent at step {args_cli.latent_collect_step}")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)
    print(
        f"[INFO] Saved {latent_numpy.shape[0]} fused latent vectors "
        f"({latent_numpy.shape[1]} dimensions) to: {output_path} and {npz_output_path}"
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    if args_cli.export_only and args_cli.export is None:
        raise ValueError("--export_only requires --export.")
    if args_cli.fault_motion_steps < 0:
        raise ValueError("--fault_motion_steps must be non-negative.")
    if args_cli.fault_motion_steps > 0 and not args_cli.log_fault_motion:
        raise ValueError("--fault_motion_steps requires --log_fault_motion.")

    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.terrain_type is not None:
        terrain_generator = env_cfg.scene.terrain.terrain_generator
        if terrain_generator is None:
            raise ValueError("--terrain_type requires an environment with a terrain generator.")
        if args_cli.terrain_type not in terrain_generator.sub_terrains:
            available_terrains = ", ".join(sorted(terrain_generator.sub_terrains))
            raise ValueError(
                f"Unknown terrain type {args_cli.terrain_type!r}. "
                f"Available terrain types: {available_terrains}."
            )
        terrain_cfg = terrain_generator.sub_terrains[args_cli.terrain_type]
        terrain_cfg.proportion = 1.0
        terrain_generator.sub_terrains = {args_cli.terrain_type: terrain_cfg}
    if args_cli.fault_joint is not None:
        if not hasattr(env_cfg.events, "randomize_actuator_faults"):
            raise AttributeError("--fault_joint requires the randomize_actuator_faults event.")
        fault_event = env_cfg.events.randomize_actuator_faults
        fault_event.params["fixed_joint_idx"] = FAULT_JOINT_NAMES.index(args_cli.fault_joint)
        print(f"[INFO] Fixed actuator fault joint: {args_cli.fault_joint}")

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.collect_fused_latent:
        if args_cli.latent_collect_step < 0:
            raise ValueError("--latent_collect_step must be non-negative.")
        if not hasattr(env_cfg.events, "randomize_actuator_faults"):
            raise AttributeError(
                "Fused-latent collection requires the randomize_actuator_faults event."
            )
        env_cfg.events.randomize_actuator_faults.interval_range_s = (0.0, 0.0)
        print("[INFO] Fused-latent collection enabled; actuator faults start immediately.")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Fault visualization is a play-only concern, so keep it out of the core
    # environment and create the marker only when a viewport is available.
    if not args_cli.headless:
        marker_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/faulty_joint")
        marker_cfg.markers["hit"].radius = 0.05
        env.unwrapped._fault_marker = VisualizationMarkers(marker_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = CustomRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = CustomOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    actor = runner.alg.actor
    fault_tcn = None
    if args_cli.fault_tcn_checkpoint is not None:
        if not 0.0 < args_cli.fault_threshold < 1.0:
            raise ValueError("--fault_threshold must be between zero and one.")
        if args_cli.fault_print_interval < 0:
            raise ValueError("--fault_print_interval must be non-negative.")
        fault_tcn, _ = load_fault_tcn(
            args_cli.fault_tcn_checkpoint,
            env.unwrapped.device,
        )

    # Export only when explicitly requested. Custom history-based actors expose
    # two tensor inputs through their as_jit()/as_onnx() wrappers.
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if args_cli.export in ("jit", "both"):
        runner.export_policy_to_jit(
            path=export_model_dir, filename="policy.pt"
        )
        print(f"[INFO] Exported TorchScript policy to: {export_model_dir}/policy.pt")
    if args_cli.export in ("onnx", "both"):
        runner.export_policy_to_onnx(
            path=export_model_dir, filename="policy.onnx"
        )
        print(f"[INFO] Exported ONNX policy to: {export_model_dir}/policy.onnx")
    if args_cli.export_only:
        env.close()
        return

    dt = env.unwrapped.step_dt
    fault_motion_logger = None
    if args_cli.log_fault_motion:
        fault_motion_output = args_cli.fault_motion_output
        if fault_motion_output is None:
            fault_motion_output = os.path.join(
                os.path.dirname(resume_path), "fault_leg_motion.csv"
            )
        fault_motion_logger = FaultMotionLogger(
            base_env=env.unwrapped,
            env_id=args_cli.fault_motion_env,
            output_path=fault_motion_output,
            dt=dt,
            print_interval=args_cli.fault_motion_print_interval,
            window_size=args_cli.fault_motion_window,
        )

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        asset = env.unwrapped.scene["robot"]
        # breakpoint()
        # Vis debug joint fault
        if (
            not args_cli.collect_fused_latent
            and hasattr(env.unwrapped, "_fault_marker")
            and (asset.faulty_joint_idx > 0).sum() > 0
        ):
            
            stage = stage_utils.get_current_stage()
            dof_paths = asset.root_physx_view.dof_paths[0]  # joint prim paths
            body_name_to_id = {n: i for i, n in enumerate(asset.body_names)}

            env_idx = asset.faulty_joint_idx.nonzero()[:,0]
            fault_idx = asset.faulty_joint_idx.nonzero()[:,1]
            parent_ids = []
            child_ids = []
            for j in fault_idx:
                joint_prim = UsdPhysics.Joint.Get(stage, dof_paths[j])
                body0 = joint_prim.GetBody0Rel().GetTargets()[0]  # parent
                body1 = joint_prim.GetBody1Rel().GetTargets()[0]  # child
                parent_name = body0.pathString.split("/")[-1]
                child_name = body1.pathString.split("/")[-1]
                parent_ids.append(body_name_to_id[parent_name])
                child_ids.append(body_name_to_id[child_name])
            parent_ids = torch.tensor(parent_ids, device=asset.device)
            child_ids = torch.tensor(child_ids, device=asset.device)
            _link_idx = child_ids
            # breakpoint()
            # try:
            pos = asset.data.body_pos_w[env_idx, _link_idx, :]
            # except:
            #     breakpoint()
            env.unwrapped._fault_marker.visualize(translations=pos)
            # breakpoint()
            # if timestep % 15 == 0:
            #     print(np.array(asset.joint_names)[(asset.faulty_joint_idx[asset.faulty_joint_idx >= 0]).tolist()])
            #     print(asset.motors_strength[_env_idx])        
        
        with torch.inference_mode():
            history = obs["history"]
            if fault_tcn is not None:
                if history.ndim != 3 or history.shape[-1] != 45:
                    raise ValueError(
                        "Fault TCN requires obs['history'] shaped [num_envs, history_length, 45], "
                        f"got {tuple(history.shape)}."
                    )
                fault_logits, _, _ = fault_tcn(history)
                if fault_logits.shape[-1] == 13:
                    # Classes 0..11 are faulty joints and class 12 is healthy.
                    fault_probabilities = torch.softmax(fault_logits, dim=-1)[:, :12]
                else:
                    fault_probabilities = torch.sigmoid(fault_logits)
                if (
                    not args_cli.collect_fused_latent
                    and args_cli.fault_print_interval > 0
                    and timestep % args_cli.fault_print_interval == 0
                ):
                    print_fault_predictions(
                        fault_probabilities,
                        list(asset.joint_names),
                        args_cli.fault_threshold,
                    )

            # agent stepping
            # breakpoint()
            # obs_policy, obs_hist = obs['policy'], obs['history']
            # actions, _ = policy(obs_policy, obs_hist)
            outputs = policy(obs)
            if isinstance(outputs, tuple):
                actions, _ = outputs 
            else:
                actions = outputs

            if args_cli.collect_fused_latent and timestep == args_cli.latent_collect_step:
                fused_latent = compute_fused_latent(actor, history)
                save_fused_latent_tsne(
                    fused_latent=fused_latent,
                    fault_mask=asset.faulty_joint_idx,
                    joint_names=list(asset.joint_names),
                    output_path=args_cli.latent_tsne_output,
                    npz_output_path=args_cli.latent_npz_output,
                    perplexity=args_cli.latent_tsne_perplexity,
                    seed=env_cfg.seed,
                )
                break

            # env stepping
            obs, _, dones, _ = env.step(actions)
            if fault_motion_logger is not None:
                fault_motion_logger.record(timestep)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
            # with torch.no_grad():
            #     fault_logits, gamma, beta = policy.fault_residual_encoder(obs['history'])
            #     targets = obs['critic'][:, -12:].float()
            #     probs = torch.sigmoid(fault_logits - math.log(19))

            #     faulty_sample_rate = (targets.sum(dim=-1) > 0).float().mean()
            #     positive_bit_rate = targets.mean()
            #     predicted_probability = probs.mean()
            #     predicted_positive_rate = (probs > 0.5).float().mean()

            #     true_positive = ((probs > 0.5) & (targets > 0.5)).sum()
            #     recall = true_positive / (targets.sum() + 1e-8)

            #     fault_mask = targets.sum(dim=-1) > 0
            #     localization_accuracy = (
            #         probs[fault_mask].argmax(dim=-1)
            #         == targets[fault_mask].argmax(dim=-1)
            #     ).float().mean()
            #     print("Fault pred prob:", probs)
            #     print("Fault pred pos rate:", predicted_positive_rate)

        timestep += 1
        if (
            fault_motion_logger is not None
            and args_cli.fault_motion_steps > 0
            and timestep >= args_cli.fault_motion_steps
        ):
            break
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if fault_motion_logger is not None:
        fault_motion_logger.close()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
