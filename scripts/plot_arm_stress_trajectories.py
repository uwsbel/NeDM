#!/usr/bin/env python3
"""Selected arm end-effector reaching trajectories from the Chrono stress battery.

Six successful goals from the 100-goal seeded Chrono reach benchmark, chosen for
diverse movement direction and scale (the arm always starts from the same home
pose, so diversity comes from the end point). Each panel is a 3D end-effector path
from its start to the target, in the arm-base frame (drawn z-up). Writes to the
manuscript image archive.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / ("artifacts/rl_runs/"
    "arm_reach_adaptivekl005_lr1e4_tol005_ep150_bonus150_sigma015_8d_rom_20260727/"
    "chrono_reach_benchmark_N100_seed12345")
DEFAULT_OUT = Path("/home/harry/Manuscripts/ImageArchive/journals/2026/"
                   "neural-dynamics-model/arm_stress_trajectories.png")
# Six successes chosen for diverse movement direction/scale from the fixed home
# start: short-near, moderate +y, straight-up, deep -y+up, long far +y, long flat -y.
GOALS = [86, 95, 4, 62, 77, 96]
FLIP = np.array([1.0, -1.0, -1.0])  # arm-base frame is z-down; draw z-up.


def load(idx: int):
    d = np.load(BENCH / f"goal_{idx:03d}" / f"chrono_arm_reach_{idx % 100:02d}.npz")
    return d["ee_base"] * FLIP, d["goal_base"] * FLIP, float(d["error_m"][-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    fig = plt.figure(figsize=(7.2, 5.0))
    for k, idx in enumerate(GOALS):
        # per-goal npz files are named by their in-shard index; fall back to glob.
        npz = sorted((BENCH / f"goal_{idx:03d}").glob("chrono_arm_reach_*.npz"))[0]
        d = np.load(npz)
        ee = d["ee_base"] * FLIP
        goal = d["goal_base"] * FLIP

        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color="C0", lw=1.8)
        ax.scatter(*ee[0], color="black", marker="o", s=26, depthshade=False,
                   label="start")
        ax.scatter(*goal, color="C3", marker="*", s=100, depthshade=False,
                   zorder=6, label="target")
        ax.scatter(*ee[-1], color="C0", marker="X", s=38, depthshade=False,
                   zorder=6, label="final EE")
        ax.tick_params(labelsize=6, pad=-2)
        ax.set_xticks(ax.get_xticks()[::2])
        ax.set_yticks(ax.get_yticks()[::2])
        ax.set_zticks(ax.get_zticks()[::2])
        ax.view_init(elev=18, azim=-60)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
