#!/usr/bin/env python3
"""Sample points on the underside of the Gator's chassis collision hull (arena_gator E2, belly-clearance diagnostic).

Chrono builds the Gator chassis collision shape (CollisionType HULLS) as the convex hull of every vertex of
data/vehicle/gator/gator_chassis_col.obj (one object in the file, so one hull), in the chassis reference frame.
This script computes that hull once (scipy, run locally) and writes

  * the lower envelope of the hull on a 0.10 m grid in the chassis x-y plane (for every grid point inside the hull's
    footprint: the lowest z of the hull above that point), and
  * every hull vertex,

to scripts/ag_gator_belly.json.  The collectors (ag_vehicle.BellyClearance) only read that file (numpy only, no scipy
on the cluster) and check that the mesh file they run with has the recorded sha256.

usage: /usr/bin/python3.12 scripts/ag_gator_belly_points.py --obj /home/harry/chrono/data/vehicle/gator/gator_chassis_col.obj
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--obj", required=True)
    p.add_argument("--grid-m", type=float, default=0.10)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "ag_gator_belly.json"))
    a = p.parse_args()
    obj = Path(a.obj)
    v = np.array([[float(t) for t in line.split()[1:4]] for line in obj.read_text().splitlines() if line.startswith("v ")])
    groups = sum(1 for line in obj.read_text().splitlines() if line.startswith(("o ", "g ")))
    hull = ConvexHull(v)
    n, d = hull.equations[:, :3], hull.equations[:, 3]          # n.x + d <= 0 inside
    down, up, side = n[:, 2] < -1e-9, n[:, 2] > 1e-9, np.abs(n[:, 2]) <= 1e-9
    xs = np.arange(np.floor(v[:, 0].min() / a.grid_m) * a.grid_m, v[:, 0].max() + 1e-9, a.grid_m)
    ys = np.arange(np.floor(v[:, 1].min() / a.grid_m) * a.grid_m, v[:, 1].max() + 1e-9, a.grid_m)
    gx, gy = [g.ravel() for g in np.meshgrid(xs, ys, indexing="ij")]
    zl = np.max(-(n[down, 0][None] * gx[:, None] + n[down, 1][None] * gy[:, None] + d[down][None]) / n[down, 2][None], axis=1)
    zu = np.min(-(n[up, 0][None] * gx[:, None] + n[up, 1][None] * gy[:, None] + d[up][None]) / n[up, 2][None], axis=1)
    ok = zl <= zu + 1e-9
    if side.any():
        ok &= np.all(n[side, 0][None] * gx[:, None] + n[side, 1][None] * gy[:, None] + d[side][None] <= 1e-9, axis=1)
    grid = np.stack([gx[ok], gy[ok], zl[ok]], 1)
    verts = v[hull.vertices]
    pts = np.concatenate([grid, verts])
    # every sampled point must lie in the hull (within round-off)
    assert float((pts @ n.T + d).max()) <= 1e-6
    out = {"schema": "ag_gator_belly_points_v1", "source_obj": "gator/gator_chassis_col.obj",
           "source_obj_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(), "obj_groups": groups,
           "frame": "Gator chassis reference frame (ChBodyAuxRef REF frame; x forward, y left, z up), the frame of the collision shape",
           "grid_m": a.grid_m, "n_grid_points": int(len(grid)), "n_hull_vertices": int(len(verts)),
           "lowest_point_z_m": float(pts[:, 2].min()),
           "method": "scipy ConvexHull of all mesh vertices (Chrono HULLS = convex hull per object; the file has one object); lower envelope = max over downward facets of the facet plane height, kept where the grid point lies inside the hull footprint; plus all hull vertices",
           "points": np.round(pts, 6).tolist()}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in out if k != "points"}))


if __name__ == "__main__":
    main()
