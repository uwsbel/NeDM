"""Plot a consecutive-waypoint Chrono transfer rollout as one continuous route figure.

Reads the .npz written by ``eval_tracked_waypoints_chrono.py`` and draws the single
driven path (coloured by time), the intended waypoint polygon, and per-waypoint markers
(green = reached, red = timed out) with their tolerance circles.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

# reuse the shared palette / time colormap
from eval_tracked_rl_goal import INK, SECONDARY, MUTED, GRID, GOOD, CRIT, TIME_CMAP


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Plot a consecutive-waypoint Chrono rollout.")
    p.add_argument("--eval-npz", type=Path, required=True, help="traj.npz from eval_tracked_waypoints_chrono.py")
    p.add_argument("--output", type=Path, default=None, help="PNG path (default: alongside the npz).")
    args = p.parse_args(argv)

    d = np.load(args.eval_npz)
    path = d["poses"]                      # (N, 2) start frame
    waypoints = d["waypoints"]             # (K+1, 2) incl. start (0,0)
    targets = waypoints[1:]                # driven targets
    leg_reached = d["leg_reached"]         # (n_legs,)
    leg_time_s = d["leg_time_s"]
    leg_min_dist = d["leg_min_dist"]
    tol = float(d["tol"])
    capture_radius = float(d["capture_radius"])
    finished = bool(d["finished"])
    total_time_s = float(np.sum(leg_time_s))
    n_legs_run = len(leg_reached)

    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    # intended route polyline (light dashed guide through every waypoint)
    ax.plot(waypoints[:, 0], waypoints[:, 1], ls=(0, (5, 4)), lw=1.1, color=MUTED, alpha=0.8, zorder=1)

    # driven path coloured by time (light -> dark)
    pts = path.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=TIME_CMAP, norm=plt.Normalize(0, max(len(segs), 1)), zorder=3)
    lc.set_array(np.arange(len(segs)))
    lc.set_linewidth(2.4)
    ax.add_collection(lc)

    # start pose + initial heading (+x in the start frame)
    xs_all = np.concatenate([path[:, 0], waypoints[:, 0]])
    ys_all = np.concatenate([path[:, 1], waypoints[:, 1]])
    span = max(xs_all.max() - xs_all.min(), ys_all.max() - ys_all.min(), 1.0)
    ax.scatter([0], [0], marker="o", s=70, color=INK, zorder=6, label="start")
    ax.annotate("", xy=(0.07 * span, 0.0), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))

    # per-waypoint markers + tolerance circles
    for j, (gx, gy) in enumerate(targets):
        reached = bool(leg_reached[j]) if j < n_legs_run else False
        col = GOOD if reached else CRIT
        tol_j = tol if j == len(targets) - 1 else capture_radius
        ax.add_patch(plt.Circle((gx, gy), tol_j, fill=False, ls=(0, (3, 2)), lw=1.0, ec=col, alpha=0.9, zorder=4))
        ax.scatter([gx], [gy], marker="*", s=280, color=col, edgecolor="white", linewidth=0.9, zorder=6)
        if j < n_legs_run:
            tag = f"W{j+1}\n{'✓ ' + format(leg_time_s[j], '.1f') + 's' if reached else '✕ ' + format(leg_min_dist[j], '.2f') + 'm'}"
        else:
            tag = f"W{j+1}\n(not run)"
        ax.annotate(tag, (gx, gy), textcoords="offset points", xytext=(8, 8),
                    fontsize=9, color=SECONDARY, zorder=7)

    m = 0.06 * span + 1.0
    ax.set_xlim(xs_all.min() - m, xs_all.max() + m)
    ax.set_ylim(ys_all.min() - m, ys_all.max() + m)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color=GRID, lw=0.7)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.set_xlabel("x [m]  (start frame)", fontsize=9, color=SECONDARY)
    ax.set_ylabel("y [m]  (start frame)", fontsize=9, color=SECONDARY)

    n_reached = int(np.sum(leg_reached))
    n_targets = len(targets)
    verdict = "FINISHED" if finished else f"FAILED at leg {n_legs_run}"
    vcol = "#006300" if finished else CRIT
    tol_desc = f"tol {tol:.2f} m" + ("" if abs(capture_radius - tol) < 1e-6 else f" (corners {capture_radius:.2f} m)")
    title = (f"Tracked-vehicle consecutive-waypoint route in Chrono  ·  {verdict}\n"
             f"{n_reached}/{n_targets} waypoints reached ({tol_desc})  ·  total {total_time_s:.1f} s  ·  "
             f"path colour = time (light→dark)")
    fig.suptitle(title, fontsize=12, color=INK, y=0.98)
    ax.set_title(f"{verdict}", fontsize=13, color=vcol, pad=6)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output = args.output or args.eval_npz.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {output}  ({n_reached}/{n_targets} reached, {verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
