"""Plot iterative endpoint optimization and independent physical replays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.run/"optimization.json").read_text())
    verified = json.loads((args.run/"chrono_verification.json").read_text())
    history = json.loads((args.run/"history.json").read_text())
    with np.load(args.run/"trajectories.npz") as packet:
        before, after = packet["before"], packet["after"]
    with np.load(args.run/"chrono_trajectories.npz") as packet:
        physical = packet["states"]
    fig, axes = plt.subplots(2, len(result["cases"]), figsize=(3.3*len(result["cases"]), 6.4),
                             squeeze=False, constrained_layout=True)
    for i, (case, verification) in enumerate(zip(result["cases"], verified["cases"], strict=True)):
        ax, loss_ax = axes[:, i]
        ax.plot(before[i, :, 0], before[i, :, 1], color="0.65", lw=1.5, label="Initial NRD")
        ax.plot(physical[i, 1, :, 0], physical[i, 1, :, 1], color="#2166ac", lw=2, label="Optimized Chrono")
        ax.plot(after[i, :, 0], after[i, :, 1], "--", color="#d95f02", lw=1.5, label="Optimized NRD")
        ax.scatter(*case["xz_m"], marker="*", color="#b2182b", s=95, zorder=5, label="Target at 1.7 s")
        ax.scatter(*verification["optimized"]["endpoint_m"], color="#2166ac", s=30, zorder=6)
        ax.axhline(0, color="0.25", lw=1)
        ax.axvline(5, color="0.25", lw=1)
        ax.set(xlabel="x (m)", ylabel="z (m)", ylim=(-.15, 6.2), xlim=(-.2, 5.2),
               title=f"{case['name']} ({case['xz_m'][0]:.2f}, {case['xz_m'][1]:.2f})")
        ax.grid(alpha=.15)
        distance = verification["optimized"]["target_distance_m"]
        loss_ax.semilogy([row["iteration"] for row in history],
                         np.maximum([row["loss_m2"][i] for row in history], 1e-12), color="#d95f02")
        restart_rows = [row["iteration"] for row in history if row.get("restart_case") == case["name"]]
        if restart_rows:
            loss_ax.axvline(min(restart_rows), color="#2166ac", ls="--", alpha=.6, label="Fixed restart")
        loss_ax.axhline(result["config"]["nrd_tolerance_m"]**2, color=".5", ls=":", label="NRD tolerance")
        loss_ax.set(xlabel="Gradient iteration", ylabel="Squared endpoint distance (m²)")
        loss_ax.set_title(f"NRD {100*case['optimized_nrd_distance_m']:.2f} / Chrono {100*distance:.2f} cm", fontsize=10)
        if restart_rows:
            loss_ax.legend(fontsize=7, loc="upper right")
        loss_ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    axes[1, 0].legend(fontsize=7)
    fig.suptitle("Frozen NRD launch optimization — common initial launch, fixed restarts if stalled", fontsize=14)
    fig.savefig(args.run/"optimization_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
