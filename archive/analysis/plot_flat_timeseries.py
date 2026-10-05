"""Plot flat-ground recovery traces saved by eval.py."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_flat_timeseries(npz_path: str | Path):
    """Return a figure showing velocity tracking and fault probabilities."""
    data = np.load(npz_path)
    time = data["time_s"]
    command = data["command_xy_mps"]
    measured = data["measured_xy_mps"]
    probability = data["fault_probability"]
    joint_names = data["joint_names"].astype(str)
    fault_joint = str(data["fault_joint"])
    fault_time = float(data["fault_time_s"])

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(time, command[:, 0], "k--", label=r"command $v_x$")
    axes[0].plot(time, measured[:, 0], label=r"measured $v_x$")
    axes[0].plot(time, measured[:, 1], label=r"measured $v_y$")
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].legend(ncol=3)

    axes[1].plot(time, data["xy_tracking_error_mps"], label="instantaneous error")
    axes[1].plot(time, data["cumulative_ate_xy_mps"], label="cumulative ATE")
    axes[1].set_ylabel(r"$xy$ error (m/s)")
    axes[1].legend()

    for joint_id, joint_name in enumerate(joint_names):
        is_fault = joint_name == fault_joint
        axes[2].plot(
            time,
            probability[:, joint_id],
            linewidth=2.5 if is_fault else 0.9,
            alpha=1.0 if is_fault else 0.45,
            label=joint_name,
        )
    axes[2].set_ylabel("Fault probability")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].legend(ncol=3, fontsize=8)

    for axis in axes:
        axis.axvline(fault_time, color="red", linestyle="--", label="fault onset")
        axis.grid(alpha=0.25)

    figure.suptitle(f"Flat-ground recovery: {fault_joint}")
    figure.tight_layout()
    return figure, axes


def plot_foot_contacts(npz_path: str | Path):
    """Return a step plot of the per-foot contact states."""
    data = np.load(npz_path)
    time = data["time_s"]
    contacts = data["foot_contact"]
    foot_names = data["foot_names"].astype(str)
    fault_joint = str(data["fault_joint"])
    fault_time = float(data["fault_time_s"])

    figure, axis = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    for foot_id, foot_name in enumerate(foot_names):
        axis.step(
            time,
            contacts[:, foot_id] + foot_id,
            where="post",
            label=foot_name,
        )
    axis.axvline(fault_time, color="red", linestyle="--", label="fault onset")
    axis.set_yticks(range(len(foot_names)), foot_names)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Foot contact")
    axis.set_title(f"Flat-ground recovery contacts: {fault_joint}")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right")
    return figure, axis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    fig, _ = plot_flat_timeseries(args.npz_path)
    if args.output is None:
        plt.show()
    else:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
