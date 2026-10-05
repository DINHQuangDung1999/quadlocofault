"""Measure checkpoint reflection errors on recorded observations, using CPU only.

Example::

    python archive/analysis/analyze_actor_reflection.py \
        --checkpoint path/to/model_3999.pt --rollout path/to/rollout.npz \
        --output-dir logs/analysis/reflection

This is a paired, offline function test. Mirrored observations are synthetic;
the comparison does not claim that a mirrored robot was simulated. The 45-D
observation layout is joint position, joint velocity, previous action, angular
velocity, projected gravity, and velocity command, in that order.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

# Resolve the model used by the original diagnostic script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "quadlocofault_rsl_rl"))

import numpy as np
import torch
import yaml
from tensordict import TensorDict

from models.equiv_gcn_actor import EquivGCNActor


JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR"))
JOINT_PERMUTATION = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10])
JOINT_SIGNS = torch.tensor([-1.0] * 4 + [1.0] * 8)
BASE_SIGNS = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
NODE_PERMUTATION = torch.tensor(list(range(7, 14)) + list(range(7)))


def reflect_joints(x: torch.Tensor) -> torch.Tensor:
    return x[..., JOINT_PERMUTATION] * JOINT_SIGNS


def reflect_observations(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != 45:
        raise ValueError(f"Expected 45 observation features, received {x.shape}.")
    return torch.cat((reflect_joints(x[..., :12]), reflect_joints(x[..., 12:24]),
                      reflect_joints(x[..., 24:36]), x[..., 36:45] * BASE_SIGNS), dim=-1)


def _linear_widths(state: dict, prefix: str) -> list[int]:
    weights = [(int(key[len(prefix):].split(".")[0]), value.shape[0])
               for key, value in state.items() if key.startswith(prefix) and key.endswith(".weight")]
    return [width for _, width in sorted(weights)]


def load_actor(checkpoint: Path) -> tuple[EquivGCNActor, int, dict]:
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = data["actor_state_dict"]
    if "fault_residual_encoder.mlp.0.weight" not in state:
        raise ValueError("This analysis requires an EquivGCNMLP checkpoint with an MLP fault encoder.")
    history_length = state["fault_residual_encoder.mlp.0.weight"].shape[1] // 45
    fault_widths = _linear_widths(state, "fault_residual_encoder.mlp.")
    actor_widths = _linear_widths(state, "actor_mlp.")
    config_path = checkpoint.parent / "params" / "agent.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Saved configuration required to verify reflection/FiLM flags: {config_path}")
    config = yaml.load(config_path.read_text(), Loader=yaml.BaseLoader)["actor"]
    if config["class_name"] != "EquivGCNActor":
        raise ValueError(f"Unsupported actor class {config['class_name']!r}.")
    obs = TensorDict({"policy": torch.zeros(1, 45), "history": torch.zeros(1, history_length, 45)}, batch_size=[1])
    distribution_cfg = config.get("distribution_cfg")
    if isinstance(distribution_cfg, dict):
        distribution_cfg = {
            "class_name": distribution_cfg["class_name"],
            "init_std": float(distribution_cfg["init_std"]),
            "std_type": distribution_cfg["std_type"],
        }
    else:
        distribution_cfg = None
    actor = EquivGCNActor(
        obs=obs, activation=config.get("activation", "elu"),
        obs_normalization=config.get("obs_normalization", "false").lower() == "true",
        actor_hidden_dims=actor_widths[:-1], fault_encoder_type="mlp",
        fault_mlp_hidden_dims=fault_widths[:-1], tcn_hidden_dim=fault_widths[-1],
        gcn_hidden_dim=state["gcn_encoder.gcn1.linear.weight"].shape[0],
        gcn_out_dim=state["gcn_encoder.gcn2.linear.weight"].shape[0],
        graph_reflection_equivariant=config.get("graph_reflection_equivariant", "true").lower() == "true",
        use_film=config.get("use_film", "true").lower() == "true",
        distribution_cfg=distribution_cfg,
    ).cpu().eval()
    with torch.inference_mode():
        actor(obs)  # Materialize lazy graph layers before strict loading.
    actor.load_state_dict(state, strict=True)
    return actor, history_length, {"iteration": data.get("iter"), "saved_actor_config": config,
                                    "state_dict_keys": len(state), "strict_load": True}


def _read_rollout(path: Path, history_length: int, dt: float) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if "obs_history" in arrays:
        history = arrays["obs_history"]
        if history.ndim == 3:
            history = history[:, None]
        if history.ndim != 4 or history.shape[2:] != (history_length, 45):
            raise ValueError(f"Invalid obs_history shape {history.shape} in {path}.")
        policy = arrays.get("obs_policy", history[:, :, -1])
        reconstructed = False
    else:
        policy = arrays["obs_policy"]
        if policy.ndim == 2:
            policy = policy[:, None]
        history = None
        reconstructed = True
    if policy.ndim == 2:
        policy = policy[:, None]
    if policy.ndim != 3 or policy.shape[-1] != 45:
        raise ValueError(f"Invalid obs_policy shape {policy.shape} in {path}.")
    steps, envs = policy.shape[:2]
    times = np.asarray(arrays.get("time_s", arrays.get("time", np.arange(steps) * dt)))
    if times.ndim == 1:
        times = np.broadcast_to(times[:, None], (steps, envs))
    if times.shape != (steps, envs):
        raise ValueError(f"Time shape {times.shape} does not match {(steps, envs)}.")
    alive = np.asarray(arrays.get("alive", np.ones((steps, envs))), dtype=bool)
    if alive.ndim == 1:
        alive = alive[:, None]
    alive = np.broadcast_to(alive, (steps, envs)).copy()
    if reconstructed:
        intervals = np.diff(times[:, 0])
        if len(intervals) and not np.allclose(intervals, dt, rtol=1e-3, atol=1e-5):
            raise ValueError("Cannot reconstruct full history from sparse obs_policy. Save obs_history or supply dense observations.")
        indices = np.maximum(np.arange(steps)[:, None] - np.arange(history_length - 1, -1, -1), 0)
        history = policy[indices].transpose(0, 2, 1, 3)
        # Exclude startup padding, whose exact convention may differ by recorder.
        alive[:history_length - 1] = False
    fault = np.asarray(arrays.get("fault_mask", arrays.get("faulty_joint_idx", np.zeros((steps, envs, 12)))), dtype=bool)
    if fault.ndim == 2 and envs == 1:
        fault = fault[:, None]
    if fault.shape != (steps, envs, 12):
        raise ValueError(f"Invalid fault mask shape {fault.shape} in {path}.")
    valid = alive & np.isfinite(policy).all(axis=-1) & np.isfinite(history).all(axis=(-1, -2))
    has_fault = fault.any(axis=-1)
    onset = np.asarray(arrays.get("actual_fault_time_s", np.where(has_fault, times, np.inf).min(axis=0)))
    onset = np.where(np.isfinite(onset), onset, np.inf)
    since_fault = times - onset[None]
    scheduled_onset = float(arrays.get("scheduled_fault_time_s", 5.0))
    rows = {"policy": policy[valid].astype(np.float32), "history": history[valid].astype(np.float32),
            "has_fault": has_fault[valid], "since_fault": since_fault[valid],
            "joint": fault.argmax(axis=-1)[valid], "before_scheduled_fault": times[valid] < scheduled_onset}
    return rows, {"path": str(path.resolve()), "history_reconstructed": reconstructed,
                  "input_shape": list(policy.shape), "valid_samples": int(valid.sum()),
                  "has_alive_mask": "alive" in arrays, "has_fault_mask": "fault_mask" in arrays or "faulty_joint_idx" in arrays,
                  "scheduled_fault_time_s": scheduled_onset, "has_exact_fault_onset": "actual_fault_time_s" in arrays}


def actor_components(actor: EquivGCNActor, policy: torch.Tensor, history: torch.Tensor) -> dict[str, torch.Tensor]:
    obs = TensorDict({"policy": policy, "history": history}, batch_size=[len(policy)])
    action, (_, logits, (gamma, beta)) = actor(obs)
    graph = actor.gcn_encoder(actor.obs_hist_normalizer(history))
    latent = graph.mean(dim=1)
    probability = logits.sigmoid()
    gate = probability.amax(dim=1, keepdim=True)
    fused = (1 + gate * gamma) * latent + gate * beta if actor.use_film else latent
    return {"action": action, "graph_nodes": graph, "graph_latent": latent,
            "fault_logits": logits, "fault_probability": probability,
            "gamma": gamma, "beta": beta, "fused_latent": fused}


def _groups(rows: dict, start: int, stop: int) -> dict[str, np.ndarray]:
    fault = rows["has_fault"][start:stop]
    age = rows["since_fault"][start:stop]
    before = rows["before_scheduled_fault"][start:stop]
    groups = {"all_alive": np.ones(stop - start, dtype=bool), "healthy": ~fault,
              "pre_fault": before & ~fault, "healthy_after_scheduled_fault": ~before & ~fault,
              "post_fault": fault, "post_fault_0_to_0.2s": fault & (age < 0.2),
              "post_fault_0.2_to_1s": fault & (age >= 0.2) & (age < 1),
              "post_fault_1s_onward": fault & (age >= 1)}
    for joint, name in enumerate(JOINT_NAMES):
        groups[f"post_fault/{name}"] = fault & (rows["joint"][start:stop] == joint)
    return groups


def measure(actor: EquivGCNActor, rows: dict, batch_size: int) -> dict:
    stats: dict = {}
    with torch.inference_mode():
        for start in range(0, len(rows["policy"]), batch_size):
            stop = min(start + batch_size, len(rows["policy"]))
            policy = torch.from_numpy(rows["policy"][start:stop])
            history = torch.from_numpy(rows["history"][start:stop])
            original = actor_components(actor, policy, history)
            mirrored = actor_components(actor, reflect_observations(policy), reflect_observations(history))
            expected = dict(original)
            expected["action"] = reflect_joints(original["action"])
            expected["graph_nodes"] = original["graph_nodes"][:, NODE_PERMUTATION]
            for key in ("fault_logits", "fault_probability"):
                expected[key] = original[key][:, JOINT_PERMUTATION]
            for group, mask_np in _groups(rows, start, stop).items():
                if not mask_np.any():
                    continue
                mask = torch.from_numpy(mask_np)
                for component in expected:
                    actual = mirrored[component][mask].double()
                    reference = expected[component][mask].double()
                    error = actual - reference
                    entry = stats.setdefault((group, component), {"sample_count": 0, "element_count": 0,
                        "error_squared_sum": 0., "reference_squared_sum": 0., "absolute_error_sum": 0., "max_abs_error": 0.})
                    entry["sample_count"] += int(mask.sum())
                    entry["element_count"] += error.numel()
                    entry["error_squared_sum"] += error.square().sum().item()
                    entry["reference_squared_sum"] += reference.square().sum().item()
                    entry["absolute_error_sum"] += error.abs().sum().item()
                    entry["max_abs_error"] = max(entry["max_abs_error"], error.abs().max().item())
    return stats


def _finalize(source: str, stats: dict, action_scale: float) -> list[dict]:
    output = []
    for (group, component), entry in sorted(stats.items()):
        count = entry["element_count"]
        rms = (entry["error_squared_sum"] / count) ** 0.5
        reference_rms = (entry["reference_squared_sum"] / count) ** 0.5
        output.append({"source": source, "group": group, "component": component,
                       "sample_count": entry["sample_count"], "rms_error": rms,
                       "mean_abs_error": entry["absolute_error_sum"] / count,
                       "max_abs_error": entry["max_abs_error"], "reference_rms": reference_rms,
                       "relative_rms_error": rms / max(reference_rms, 1e-12),
                       "action_target_rms_rad": rms * action_scale if component == "action" else ""})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.02, help="Policy step duration for dense-history reconstruction.")
    parser.add_argument("--action-scale", type=float, default=0.25)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("Batch size and threads must be positive.")
    torch.set_num_threads(args.threads)
    actor, history_length, metadata = load_actor(args.checkpoint)
    # Reflection must be an involution for all channels, including axial omega.
    probe = torch.randn(2, history_length, 45)
    torch.testing.assert_close(reflect_observations(reflect_observations(probe)), probe, rtol=0, atol=0)
    output = []
    metadata.update(checkpoint=str(args.checkpoint.resolve()), history_length=history_length,
                    device="cpu", action_scale=args.action_scale, rollouts=[],
                    caveats=["Mirrored observations are synthetic; this is not a mirrored dynamics rollout.",
                             "Graph equivariance does not require classifier/FiLM/action equivariance in this architecture.",
                             "Action-target radians are before the ground-truth faulty-leg position clamp.",
                             "Only alive, finite states are analyzed; early termination changes the later-state sample mix."])
    for path in args.rollout:
        rows, description = _read_rollout(path, history_length, args.dt)
        metadata["rollouts"].append(description)
        output.extend(_finalize(str(path.resolve()), measure(actor, rows, args.batch_size), args.action_scale))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not output:
        raise ValueError("No valid recorded states were found.")
    csv_path = args.output_dir / "actor_reflection.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    (args.output_dir / "actor_reflection_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Strictly loaded {metadata['state_dict_keys']} actor tensors on CPU; analyzed {sum(x['valid_samples'] for x in metadata['rollouts'])} states.")
    for row in output:
        if row["group"] in ("healthy", "post_fault") and row["component"] in ("action", "graph_latent", "fault_probability", "fused_latent"):
            print(f"{Path(row['source']).name} {row['group']} {row['component']}: RMS={row['rms_error']:.6g}, relative={row['relative_rms_error']:.6g}, n={row['sample_count']}")
    print(csv_path)


if __name__ == "__main__":
    main()
