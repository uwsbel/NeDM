#!/usr/bin/env python3
"""Plot saved neural choices against measured Chrono paths; never selects routes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def measured(folder):
    folder = Path(folder)
    with np.load(folder / "trajectory.npz") as data:
        xy = np.concatenate((data["pose"][:, :2], data["terminal_pose"][None, :2]))
        time = np.arange(len(xy)) * float(data["dt_s"])
    return xy, time, read(folder / "outcome.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.manifest)
    rows = manifest["scenes"]
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4.3*len(rows)), squeeze=False,
                             constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
    for index, record in enumerate(rows):
        image_ax, path_ax, time_ax = axes[index]
        with np.load(record["observation"]) as observation:
            rgb = observation["rgbd"][:3].transpose(1, 2, 0)
            goal = observation["goal_xy"].copy()
        image_ax.imshow(np.clip(rgb, 0., 1.))
        image_ax.set_title(record["title"] + "\nCurrent camera RGB (depth also supplied)")
        image_ax.set_axis_off()
        alternatives = []
        for folder in record["alternatives"]:
            xy, time, outcome = measured(folder)
            color = "#aeb7c2" if outcome["safe_goal_reached"] else "#e5a09b"
            path_ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.2, alpha=.85)
            time_ax.plot(time, np.linalg.norm(xy-goal, axis=1), color=color, lw=1.2, alpha=.85)
            alternatives.append((folder, xy, time, outcome))
        safe = [item for item in alternatives if item[3]["safe_goal_reached"]]
        if safe:
            _, xy, time, outcome = min(safe, key=lambda item: item[3]["goal_time_s"])
            label = f"Fastest safe sampled: {outcome['goal_time_s']:.2f} s"
            path_ax.plot(xy[:, 0], xy[:, 1], color="#169b62", lw=2.2, label=label)
            time_ax.plot(time, np.linalg.norm(xy-goal, axis=1), color="#169b62", lw=2.2, label=label)
        selection = read(record["selection"])
        if selection["abstained"]:
            path_ax.set_title("Learned planner abstained")
        else:
            xy, time, outcome = measured(record["execution"])
            color = "#2469ad" if outcome["safe_goal_reached"] else "#c13d37"
            label = (f"RGB-D + MPPI: {outcome['goal_time_s']:.2f} s" if outcome["goal_reached"]
                     else f"RGB-D + MPPI: timeout at {outcome['elapsed_s']:.1f} s")
            path_ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2.8, label=label)
            time_ax.plot(time, np.linalg.norm(xy-goal, axis=1), color=color, lw=2.8, label=label)
            path_ax.scatter(*xy[0], color="#202630", s=32, zorder=5)
            status = "Safe goal reached" if outcome["safe_goal_reached"] else "Unsafe or incomplete"
            path_ax.set_title(status + " — actual Chrono paths")
        path_ax.scatter(*goal, color="#edaa26", marker="*", s=140, zorder=5, label="Goal")
        path_ax.set_aspect("equal", adjustable="datalim")
        path_ax.set_xlabel("World x (m)"); path_ax.set_ylabel("World y (m)")
        path_ax.legend(fontsize=8, loc="best"); path_ax.grid(alpha=.2)
        time_ax.axhline(record.get("goal_radius_m", 2.5), color="#edaa26", ls=":", lw=1.5)
        time_ax.set_title("Measured goal progress")
        time_ax.set_xlabel("Actual elapsed time (s)"); time_ax.set_ylabel("Distance to goal (m)")
        time_ax.set_ylim(bottom=0); time_ax.grid(alpha=.2); time_ax.legend(fontsize=8, loc="best")
    fig.suptitle(manifest["title"], fontsize=15)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(args.out)


if __name__ == "__main__":
    main()
