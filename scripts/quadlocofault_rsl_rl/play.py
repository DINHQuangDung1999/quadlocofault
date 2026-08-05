# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import math
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
    "--latent_tsne_perplexity",
    type=float,
    default=30.0,
    help="Perplexity used by t-SNE.",
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
    perplexity: float,
    seed: int,
) -> None:
    """Project fused latents to two dimensions and save a fault-colored PDF."""
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
    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

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
        f"({latent_numpy.shape[1]} dimensions) to: {output_path}"
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

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

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        # runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        # runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        pass
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        # export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        # export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

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
                    perplexity=args_cli.latent_tsne_perplexity,
                    seed=env_cfg.seed,
                )
                break

            # env stepping
            obs, _, dones, _ = env.step(actions)
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
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
