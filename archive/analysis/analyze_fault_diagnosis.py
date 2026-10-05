"""Analyze first-episode traces produced by diagnose_faults.py.

This script has no Isaac Sim dependency. All post-fault comparisons condition on
surviving until the observed fault onset, and exclude samples at/after the first
termination. Healthy controls use the scheduled fault time as their reference.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LEGS = ("FL", "FR", "RL", "RR")
COLORS = {"FL": "#2878b5", "FR": "#d44a3a", "RL": "#479b54", "RR": "#9065af", "healthy": "#666666"}


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def finite_quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else float("nan")


def wilson_interval(successes: int, count: int) -> tuple[float, float]:
    """Two-sided 95% Wilson binomial interval, undefined for empty groups."""
    if not count:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / count
    denominator = 1.0 + z * z / count
    midpoint = (p + z * z / (2.0 * count)) / denominator
    halfwidth = z * math.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count**2)) / denominator
    return max(0.0, midpoint - halfwidth), min(1.0, midpoint + halfwidth)


def scalar(data: dict, key: str, default):
    return np.asarray(data[key]).item() if key in data else default


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def first_sustained_time(time: np.ndarray, condition: np.ndarray, duration: float = 0.1) -> float:
    """Return the start of the first contiguous sampled interval of this length."""
    start = None
    for current_time, active in zip(time, condition, strict=True):
        if not active:
            start = None
        elif start is None:
            start = float(current_time)
        elif current_time - start >= duration - 1.0e-6:
            return start
    return float("nan")


def analyze(npz_path: Path, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    time = data["time_s"].astype(float)
    names = data["joint_names"].astype(str)
    assigned = data["assigned_joint"].astype(int)
    done = data["first_done_time_s"].astype(float)
    causes = data["first_done_cause"].astype(str)
    actual_onset = data["actual_fault_time_s"].astype(float)
    healthy_index = len(names)
    scheduled = float(scalar(data, "scheduled_fault_time_s", 5.0))
    # A last sample cannot prove survival beyond its timestamp; prefer the
    # runner's full horizon when supplied (it includes unsampled final steps).
    horizon = float(scalar(data, "horizon_s", time[-1]))
    dt = float(scalar(data, "step_dt_s", 0.02))
    sample_dt = float(scalar(data, "sample_dt_s", np.median(np.diff(time))))
    references = np.where(np.isfinite(actual_onset), actual_onset, scheduled)
    relative_time = time[:, None] - references[None, :]
    valid = data["alive"].astype(bool) & (
        ~np.isfinite(done)[None, :] | (time[:, None] < done[None, :] - 1.0e-7)
    )
    prealive = ~np.isfinite(done) | (done > scheduled)
    eligible = prealive & ((assigned == healthy_index) | np.isfinite(actual_onset))
    eligible &= ~np.isfinite(done) | (done > references)
    remaining = np.where(np.isfinite(done), done, horizon) - references
    observed_duration = horizon - references
    before = valid & (relative_time >= -1.0) & (relative_time < 0.0)
    after = valid & eligible[None, :] & (relative_time >= 0.0) & (relative_time < 2.0)

    def values(name: str) -> np.ndarray:
        array = data[name].astype(float)
        return array[..., 0] if array.ndim == 3 and array.shape[-1] == 1 else array

    velocity = data["root_lin_vel_b"].astype(float)
    command = data.get("command", data.get("command_velocity"))
    if command is None:
        command_vx = np.full(velocity.shape[:2], float(scalar(data, "command_vx_mps", 0.75)))
    else:
        command_vx = np.asarray(command)[..., 0]
    tracking = np.abs(velocity[..., 0] - command_vx)
    quat = data["root_quat_w"].astype(float)
    qw, qx, qy, qz = np.moveaxis(quat, -1, 0)
    roll = np.degrees(np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1)))
    height = values("base_height")
    forces = data["foot_force_w"].astype(float)
    contact = np.linalg.norm(forces, axis=-1) > 1.0
    vertical_load = np.maximum(forces[..., 2], 0.0)
    total_load = vertical_load.sum(axis=-1)
    load_share = np.divide(vertical_load, total_load[..., None], out=np.full_like(vertical_load, np.nan), where=total_load[..., None] > 1.0)
    foot_speed = np.linalg.norm(data["foot_vel_w"][..., :2], axis=-1)
    probabilities = data["fault_probability"].astype(float)
    true_probability = np.full(valid.shape, np.nan)
    false_probability = np.full(valid.shape, np.nan)
    healthy_contacts = contact.sum(axis=-1).astype(float)
    faulty_share = np.full(valid.shape, np.nan)
    healthy_slide = np.full(valid.shape, np.nan)
    fault_applied_torque = np.full(valid.shape, np.nan)
    fault_computed_torque = np.full(valid.shape, np.nan)
    healthy_applied_torque = np.full(valid.shape, np.nan)
    assigned_strength = np.full(valid.shape, np.nan)

    for env, index in enumerate(assigned):
        healthy_feet = np.ones(4, dtype=bool)
        healthy_joints = np.ones(len(names), dtype=bool)
        if index != healthy_index:
            leg = LEGS.index(names[index].split("_")[0])
            healthy_feet[leg] = False
            healthy_joints[index] = False
            true_probability[:, env] = probabilities[:, env, index]
            false_probability[:, env] = probabilities[:, env, healthy_joints].max(axis=-1)
            healthy_contacts[:, env] = contact[:, env, healthy_feet].sum(axis=-1)
            faulty_share[:, env] = load_share[:, env, leg]
            fault_applied_torque[:, env] = np.abs(data["applied_torque"][:, env, index])
            fault_computed_torque[:, env] = np.abs(data["computed_torque"][:, env, index])
            if "motor_strength" in data:
                assigned_strength[:, env] = data["motor_strength"][:, env, index]
        else:
            true_probability[:, env] = probabilities[:, env].max(axis=-1)
            false_probability[:, env] = true_probability[:, env]
        healthy_applied_torque[:, env] = np.abs(data["applied_torque"][:, env, healthy_joints]).mean(axis=-1)
        contacting = contact[:, env, healthy_feet]
        summed_speed = (foot_speed[:, env, healthy_feet] * contacting).sum(axis=-1)
        num_contacting = contacting.sum(axis=-1)
        healthy_slide[:, env] = np.divide(summed_speed, num_contacting, out=np.full_like(summed_speed, np.nan), where=num_contacting > 0)

    traces = {
        "tracking_error_mps": tracking,
        "pitch_deg": pitch,
        "roll_deg": roll,
        "base_height_m": height,
        "healthy_contacts": healthy_contacts,
        "true_fault_probability": true_probability,
        "max_wrong_probability": false_probability,
        "faulty_foot_load_share": faulty_share,
        "healthy_contact_slide_mps": healthy_slide,
        "fault_applied_torque_nm": fault_applied_torque,
        "fault_computed_torque_nm": fault_computed_torque,
        "healthy_applied_torque_nm": healthy_applied_torque,
    }
    episode_rows = []
    for env, index in enumerate(assigned):
        label = "healthy" if index == healthy_index else names[index]
        post = after[:, env]
        pre = before[:, env]
        p = true_probability[:, env]
        detection_condition = valid[:, env] & (relative_time[:, env] >= 0) & (p > 0.5)
        latency = first_sustained_time(relative_time[:, env], detection_condition, 0.1)
        row = {
            "env_id": env,
            "fault_joint": label,
            "prefault_survivor": bool(prealive[env]),
            "postfault_eligible": bool(eligible[env]),
            "actual_fault_time_s": actual_onset[env],
            "first_done_time_s": done[env],
            "first_done_cause": causes[env],
            "postfault_observed_time_s": max(0.0, remaining[env]) if eligible[env] else float("nan"),
            "survival_censored": not bool(np.isfinite(done[env])),
            "prefault_ate_vx_mps": finite_mean(tracking[pre, env]),
            "post2s_ate_vx_mps": finite_mean(tracking[post, env]),
            "post2s_p95_abs_roll_deg": finite_quantile(np.abs(roll[post, env]), 0.95),
            "post2s_p95_abs_pitch_deg": finite_quantile(np.abs(pitch[post, env]), 0.95),
            "post2s_min_base_height_m": finite_quantile(height[post, env], 0.0),
            "post2s_mean_healthy_contacts": finite_mean(healthy_contacts[post, env]),
            "post2s_mean_faulty_load_share": finite_mean(faulty_share[post, env]),
            "post2s_mean_healthy_slide_mps": finite_mean(healthy_slide[post, env]),
            "post2s_mean_true_fault_probability": finite_mean(p[post]) if index != healthy_index else float("nan"),
            "post2s_mean_max_wrong_probability": finite_mean(false_probability[post, env]),
            "prefault_mean_max_fault_probability": finite_mean(probabilities[pre, env].max(axis=-1)),
            "detection_latency_s": latency if eligible[env] and index != healthy_index else float("nan"),
            "post2s_mean_fault_applied_torque_nm": finite_mean(fault_applied_torque[post, env]),
            "post2s_mean_fault_computed_torque_nm": finite_mean(fault_computed_torque[post, env]),
            "post2s_mean_healthy_applied_torque_nm": finite_mean(healthy_applied_torque[post, env]),
            "post2s_mean_assigned_motor_strength": finite_mean(assigned_strength[post, env]),
            "terrain_level": int(data["terrain_levels"][env]) if "terrain_levels" in data else float("nan"),
            "mean_static_friction": finite_mean(data["material_properties"][env, :, 0]) if "material_properties" in data else float("nan"),
            "mean_dynamic_friction": finite_mean(data["material_properties"][env, :, 1]) if "material_properties" in data else float("nan"),
            "total_mass_kg": float(data["masses"][env].sum()) if "masses" in data else float("nan"),
        }
        episode_rows.append(row)
    write_csv(output_dir / "episode_metrics.csv", episode_rows)

    summaries = []
    for index in sorted(np.unique(assigned)):
        group = assigned == index
        eligible_group = group & eligible
        row = {
            "fault_joint": "healthy" if index == healthy_index else names[index],
            "n_episodes": int(group.sum()),
            "n_prefault_failures": int((group & ~prealive).sum()),
            "n_fault_observed": int((group & np.isfinite(actual_onset)).sum()),
            "n_postfault_eligible": int(eligible_group.sum()),
        }
        for offset in (2, 5, 10):
            evaluable = eligible_group & (observed_duration >= offset - dt / 2)
            count = int(evaluable.sum())
            survived = int((evaluable & (~np.isfinite(done) | (remaining > offset + 1.0e-7))).sum())
            low, high = wilson_interval(survived, count)
            row.update({f"n_at_{offset}s": count, f"survival_{offset}s": survived / count if count else float("nan"), f"survival_{offset}s_ci_low": low, f"survival_{offset}s_ci_high": high})
        complete_horizon = eligible_group & (observed_duration >= 10 - dt / 2)
        row["restricted_mean_postfault_survival_10s"] = finite_mean(np.minimum(remaining[complete_horizon], 10.0))
        post_terminated = eligible_group & np.isfinite(done)
        row["n_postfault_terminations"] = int(post_terminated.sum())
        row["n_base_contact"] = int((post_terminated & (np.char.find(causes, "base_contact") >= 0)).sum())
        row["n_bad_orientation"] = int((post_terminated & (np.char.find(causes, "bad_orientation") >= 0)).sum())
        for key in episode_rows[0]:
            if key.startswith(("prefault_ate", "post2s_", "prefault_mean_", "mean_static", "mean_dynamic", "total_mass")):
                row[f"mean_{key}"] = finite_mean(np.asarray([item[key] for item in episode_rows])[eligible_group])
        latency = np.asarray([item["detection_latency_s"] for item in episode_rows])
        row["n_detected_before_termination"] = int((eligible_group & np.isfinite(latency)).sum())
        row["median_detected_latency_s"] = finite_quantile(latency[eligible_group], 0.5)
        summaries.append(row)
    write_csv(output_dir / "joint_summary.csv", summaries)

    # Means are per-episode averages, then averaged across episodes. This gives
    # short lived and long lived episodes equal weight within each metric.
    comparisons = []
    for family in ("hip", "thigh", "calf"):
        left = next((row for row in summaries if row["fault_joint"] == f"FL_{family}_joint"), None)
        right = next((row for row in summaries if row["fault_joint"] == f"FR_{family}_joint"), None)
        if left is None or right is None:
            continue
        row = {"joint_family": family}
        for offset in (2, 5, 10):
            lval, rval = left[f"survival_{offset}s"], right[f"survival_{offset}s"]
            difference = rval - lval
            # Newcombe's independent-proportions interval from Wilson limits.
            low = difference - math.sqrt((rval - right[f"survival_{offset}s_ci_low"]) ** 2 + (left[f"survival_{offset}s_ci_high"] - lval) ** 2)
            high = difference + math.sqrt((right[f"survival_{offset}s_ci_high"] - rval) ** 2 + (lval - left[f"survival_{offset}s_ci_low"]) ** 2)
            row.update({f"fr_minus_fl_survival_{offset}s": difference, f"difference_{offset}s_ci_low": low, f"difference_{offset}s_ci_high": high})
        comparisons.append(row)
    write_csv(output_dir / "front_comparison.csv", comparisons)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True, constrained_layout=True)
    labels = [row["fault_joint"].replace("_joint", "") for row in summaries]
    colors = [COLORS.get(label.split("_")[0], COLORS["healthy"]) for label in labels]
    x = np.arange(len(labels))
    for axis, offset in zip(axes, (2, 5, 10), strict=True):
        fractions = np.asarray([row[f"survival_{offset}s"] for row in summaries])
        lower = np.asarray([row[f"survival_{offset}s_ci_low"] for row in summaries])
        upper = np.asarray([row[f"survival_{offset}s_ci_high"] for row in summaries])
        axis.bar(x, fractions, color=colors, alpha=0.85)
        axis.errorbar(x, fractions, yerr=np.maximum(0, np.stack((fractions - lower, upper - fractions))), fmt="none", color="black", capsize=2, linewidth=1)
        axis.set_xticks(x, labels, rotation=75, ha="right")
        axis.set_ylim(0, 1.04)
        axis.set_title(f"Survival {offset} s after fault")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Fraction of eligible episodes (95% Wilson CI)")
    fig.suptitle("Conditional survival after fault onset; prefault failures excluded")
    fig.savefig(output_dir / "postfault_survival.png", dpi=180)
    plt.close(fig)

    grid = np.arange(-1.0, min(10.0, horizon - scheduled) + sample_dt / 2, sample_dt)
    aligned: dict[str, np.ndarray] = {}
    for key, trace in traces.items():
        target = np.full((len(grid), len(assigned)), np.nan)
        for env in range(len(assigned)):
            mask = valid[:, env] & np.isfinite(trace[:, env])
            if eligible[env] and mask.sum() >= 2:
                target[:, env] = np.interp(grid, relative_time[mask, env], trace[mask, env], left=np.nan, right=np.nan)
        aligned[key] = target

    def plot_traces(filename: str, fields: list[tuple[str, str]], title: str) -> None:
        fig, axes = plt.subplots(len(fields), 3, figsize=(15, 2.0 * len(fields)), sharex=True, squeeze=False, constrained_layout=True)
        for column, family in enumerate(("hip", "thigh", "calf")):
            for leg in (*LEGS, "healthy"):
                label = "healthy" if leg == "healthy" else f"{leg}_{family}_joint"
                matches = np.flatnonzero(names == label)
                index = healthy_index if leg == "healthy" else (int(matches[0]) if len(matches) else -1)
                selected = assigned == index
                if not selected.any():
                    continue
                for row, (field, ylabel) in enumerate(fields):
                    samples = aligned[field][:, selected]
                    count = np.isfinite(samples).sum(axis=1)
                    mean = np.divide(np.nansum(samples, axis=1), count, out=np.full(len(grid), np.nan), where=count > 0)
                    axes[row, column].plot(grid, mean, color=COLORS[leg], label=leg, linewidth=1.7, linestyle="--" if leg == "healthy" else "-")
                    axes[row, column].set_ylabel(ylabel)
            for axis in axes[:, column]:
                axis.axvline(0, color="black", linestyle=":", linewidth=1)
                axis.grid(alpha=0.2)
            axes[0, column].set_title(f"{family.capitalize()} joint faults")
            axes[-1, column].set_xlabel("Time since fault (s)")
        axes[0, 0].legend(ncol=5, fontsize=8)
        fig.suptitle(title + "\nMeans over surviving first episodes; independent randomized groups")
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    plot_traces("posture_tracking_traces.png", [
        ("tracking_error_mps", "Absolute vx error (m/s)"),
        ("pitch_deg", "Pitch (deg)"),
        ("roll_deg", "Roll (deg)"),
        ("base_height_m", "Base height (m)"),
        ("healthy_contacts", "Healthy feet in contact"),
        ("true_fault_probability", "True-joint p\n(control: max p)"),
    ], "Postfault posture, tracking, and detection")
    plot_traces("support_effort_traces.png", [
        ("faulty_foot_load_share", "Faulty leg load share"),
        ("healthy_contact_slide_mps", "Healthy foot slide (m/s)"),
        ("fault_computed_torque_nm", "Fault requested |torque| (Nm)"),
        ("fault_applied_torque_nm", "Fault applied |torque| (Nm)"),
        ("healthy_applied_torque_nm", "Healthy mean |torque| (Nm)"),
        ("max_wrong_probability", "Maximum wrong-joint p"),
    ], "Postfault support and actuator effort")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, constrained_layout=True)
    for axis, family in zip(axes, ("hip", "thigh", "calf"), strict=True):
        for leg in LEGS:
            matches = np.flatnonzero(names == f"{leg}_{family}_joint")
            if not len(matches):
                continue
            selected = eligible & (assigned == matches[0])
            if not selected.any():
                continue
            curve_time = np.linspace(0, 10, 501)
            curve = np.asarray([
                np.mean(~np.isfinite(done[selected]) | (remaining[selected] > t + 1e-7))
                if np.all(observed_duration[selected] >= t - dt / 2) else np.nan
                for t in curve_time
            ])
            axis.step(curve_time, curve, where="post", color=COLORS[leg], label=f"{leg} (n={selected.sum()})")
        axis.set_title(f"{family.capitalize()} joint faults")
        axis.set_xlabel("Time after observed fault (s)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Survival fraction")
    axes[0].set_ylim(0, 1.03)
    fig.savefig(output_dir / "survival_curves.png", dpi=180)
    plt.close(fig)

    def percent(value: float) -> str:
        return "n/a" if not math.isfinite(value) else f"{100 * value:.1f}%"

    checkpoint = str(scalar(data, "checkpoint", "Not stored in archive"))
    lines = [
        "# Delayed-fault diagnosis",
        "",
        f"Input: `{npz_path.resolve()}`",
        f"Checkpoint: `{checkpoint}`",
        "",
        f"Recorded {len(assigned)} first episodes over {horizon:.2f} s; sampled every {sample_dt:.3f} s. "
        f"Fault onset reference is measured per episode (scheduled {scheduled:g} s). "
        "Healthy controls use the scheduled onset. Survival estimates exclude prefault failures and episodes without an observed assigned fault.",
        "",
        "| Fault | Episodes | Prefault failures | Eligible | Survived +2 s | Survived +5 s | Survived +10 s (95% CI) | Mean survival capped at 10 s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(f"| {row['fault_joint']} | {row['n_episodes']} | {row['n_prefault_failures']} | {row['n_postfault_eligible']} | {percent(row['survival_2s'])} | {percent(row['survival_5s'])} | {percent(row['survival_10s'])} ({percent(row['survival_10s_ci_low'])}–{percent(row['survival_10s_ci_high'])}) | {row['restricted_mean_postfault_survival_10s']:.2f} s |")
    lines += ["", "## Front-right versus front-left", "", "Positive differences favor FR. Differences compare independent randomized groups; confidence intervals use Wilson-based Newcombe intervals.", "", "| Joint family | FR − FL survival at +10 s | 95% CI |", "|---|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['joint_family']} | {100 * row['fr_minus_fl_survival_10s']:+.1f} percentage points | [{100 * row['difference_10s_ci_low']:+.1f}, {100 * row['difference_10s_ci_high']:+.1f}] |")
    lines += [
        "", "## Outputs", "",
        "- `joint_summary.csv`: per-joint survival, termination, tracking, posture, support, detection, and effort metrics.",
        "- `episode_metrics.csv`: individual first-episode outcomes and randomization diagnostics.",
        "- `front_comparison.csv`: FR–FL survival differences with intervals.",
        "- `postfault_survival.png` and `survival_curves.png`: conditional survival comparisons.",
        "- `posture_tracking_traces.png`: posture, height, tracking, healthy support count, and classifier outputs.",
        "- `support_effort_traces.png`: support loads, foot sliding, requested/applied effort, and classifier false positives.",
        "", "## Interpretation limits", "",
        "- Each group has independent startup randomization; these are not paired or mirrored initial states. Per-joint friction and mass averages are included to check gross imbalance.",
        "- Mean traces contain only surviving episodes at each time. Later means can look better because difficult episodes have terminated. Read them alongside survival curves.",
        "- The first two seconds' scalar metrics are computed per episode over available pretermination frames, then averaged across episodes. Very early falls have shorter observation windows.",
        "- Healthy contact count excludes the leg containing the assigned faulty joint; control counts all four feet. Contact means instantaneous force magnitude >1 N. Foot sliding is horizontal world speed averaged over contacting healthy feet.",
        "- Base height subtracts mean scanner terrain height; it is approximate on uneven ground. A low height or large tilt alone is not labeled a fall: termination determines survival.",
        "- Requested/applied torque fields follow the simulator actuator implementation. Their ratio is not an independently measured hardware motor-strength estimate.",
        "- Detection means true-joint probability >0.5 for at least 0.1 s of sampled observations. Median latency includes detections only; undetected cases remain visible in `n_detected_before_termination`.",
        "- Frames at/after the first termination are excluded, including automatic reset frames. The archive supports first-fall analysis, not repeated recovery attempts in one uninterrupted physical episode.",
        "- The sample interval may miss short impacts, slip bursts, or rapid classifier transients. Saved proprioceptive observations can support finer investigation if sampling is increased.",
        "- These simulation results identify associations and reproduce failure patterns; they do not establish a hardware cause or a sim-to-real explanation. Small samples cannot reliably rank fault joints.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote analysis of {len(assigned)} episodes to {output_dir}")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_path", type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    analyze(args.npz_path, args.output_dir or args.npz_path.parent / "analysis")
