#!/usr/bin/env python3
"""3D plot of the arm link bounding boxes placed by forward kinematics.

Illustrates Appendix F: for a configuration q, the batched product-of-exponentials
forward kinematics (nedm.rl.arm_kinematics.ArmKinematics) places each link's
axis-aligned box into the world. In world coordinates these become *oriented*
boxes; the geometric safety shield tests them (and their 8 corner + 1 center
sample points {x_p(q)}) for collision. This draws the four arm-chain link boxes
(shoulder, biceps, elbow, wrist) and the end-effector, in the arm-base frame.

Run:  PYTHONPATH=src python scripts/plot_arm_fk_boxes.py --q 0.7 0.5 -0.5 0.3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from nedm.rl.arm_kinematics import ArmKinematics

REPO = Path(__file__).resolve().parents[1]
GEOM = REPO / "artifacts/arm_geometry/arm_geometry_v1.json"
DEFAULT_OUT = Path(
    "/home/harry/Manuscripts/ImageArchive/journals/2026/neural-dynamics-model/arm_fk_boxes.png"
)

# Corner order from ArmKinematics._box_points: k -> bits (ix,iy,iz),
# ix=(k>>2)&1, iy=(k>>1)&1, iz=k&1, sign +1 if bit else -1.
FACES = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
         [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8)
         if bin(a ^ b).count("1") == 1]

ARM_LINKS = ["shoulder", "biceps", "elbow", "wrist"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def draw_box(ax, corners: np.ndarray, color: str) -> None:
    """corners: (8,3) world/base corner points in the standard bit order."""
    faces = [[corners[i] for i in f] for f in FACES]
    ax.add_collection3d(Poly3DCollection(
        faces, facecolor=color, edgecolor="none", alpha=0.16))
    segs = [[corners[a], corners[b]] for a, b in EDGES]
    ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=1.3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, nargs=4, default=[0.7, 0.5, -0.5, 0.3],
                    metavar=("q1", "q2", "q3", "q4"))
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-72.0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    kin = ArmKinematics.from_json(str(GEOM))
    q = torch.tensor([args.q], dtype=torch.float32)
    pts_world = kin.link_points(q)[0]                     # (L, 9, 3)
    pts_base = kin.to_base(pts_world).numpy()             # arm-base frame
    ee_base = kin.to_base(kin.ee_world(q))[0].numpy()

    # The arm-base frame is z-down by convention; draw it z-up for readability
    # (a 180-deg rotation about the base x-axis: y,z -> -y,-z, a proper rotation).
    flip = np.array([1.0, -1.0, -1.0], dtype=pts_base.dtype)
    pts_base = pts_base * flip
    ee_base = ee_base * flip

    fig = plt.figure(figsize=(5.2, 4.2))
    ax = fig.add_subplot(111, projection="3d")

    drawn = []
    for name, color in zip(ARM_LINKS, COLORS):
        i = kin.link_names.index(name)
        corners = pts_base[i, :8]
        center = pts_base[i, 8]
        draw_box(ax, corners, color)
        ax.scatter(*pts_base[i, :9].T, color=color, s=9, depthshade=False)  # {x_p(q)}
        ax.text(*center, f" {name}", color=color, fontsize=8.5,
                fontweight="bold", ha="left", va="center")
        drawn.append(pts_base[i, :9])

    ax.scatter(*ee_base, color="black", marker="*", s=130, depthshade=False,
               zorder=6, label="end-effector $p^{\\mathrm{ee}}$")
    ax.scatter([0], [0], [0], color="black", marker="o", s=30, depthshade=False)
    ax.text(0, 0, 0, "  arm base", fontsize=8, color="black", va="top")

    allp = np.concatenate(drawn + [ee_base[None]], axis=0)
    mid = allp.mean(0)
    span = (allp.max(0) - allp.min(0)).max() * 0.55
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_zlabel("$z$ (m)")
    ax.view_init(elev=args.elev, azim=args.azim)
    # Expand the 3D axes to fill the canvas (title/legend live in the caption).
    ax.set_position([0.0, 0.0, 1.0, 1.0])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out}")
    print(f"EE (base): {[round(v, 3) for v in ee_base.tolist()]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
