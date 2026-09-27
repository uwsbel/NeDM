#!/usr/bin/env python3
"""VERIFY_E2: independent checks of the belly-clearance diagnostic.

1. Points file vs the chassis collision OBJ (scipy ConvexHull, own code): the OBJ has one object (Chrono HULLS =
   one hull per object); every sampled point lies on the hull surface; every grid point is on the LOWER envelope
   (1 mm below it is outside the hull); the 78 hull vertices equal the hull's vertex set; the lowest sampled point
   equals the lowest hull vertex; the grid covers the hull footprint (points of a finer 0.05 m probe grid whose lower
   envelope is lower than every sampled point by more than the facet slope allows are counted).
2. Per-frame recomputation from a Gator rigid run: pose from rich_telemetry (chassis reference position +
   quaternion), own quaternion rotation, own bilinear BMP lookup (TerrainMap) with Chrono's node-on-edge scale
   (pixels-1)/pixels and, as a control, without it.  Compared with vehicle_extra.npz.
3. Surface-convention control: hub z - surface(hub xy) vs the TMEASY loaded radius (tire_*_radius_m) for both
   conventions (the right one gives the smaller mismatch).
usage (system python with scipy + PYTHONPATH=src): check_belly.py RUN_DIR [RUN_DIR ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from nedm.traverse.terrain import TerrainMap  # noqa: E402

OBJ = Path("/home/harry/chrono/data/vehicle/gator/gator_chassis_col.obj")
spec = json.loads((ROOT / "scripts/ag_gator_belly.json").read_text())
P = np.asarray(spec["points"], float)
res = {}

# ---------------------------------------------------------------- 1. points vs hull
lines = OBJ.read_text().splitlines()
V = np.array([[float(t) for t in l.split()[1:4]] for l in lines if l.startswith("v ")])
objects = sum(1 for l in lines if l.startswith(("o ", "g ")))
hull = ConvexHull(V)
eq = hull.equations                       # n.x + d <= 0 inside
dist = P @ eq[:, :3].T + eq[:, 3]         # signed distance to every facet plane
on_surface = np.abs(dist.max(1)) < 1e-5   # max over facets = 0 on the boundary
tri = Delaunay(V[hull.vertices])
below = P - np.array([0, 0, 1e-3])
grid = P[:spec["n_grid_points"]]
grid_below_outside = tri.find_simplex(grid - np.array([0, 0, 1e-3])) < 0
grid_above_inside = tri.find_simplex(grid + np.array([0, 0, 1e-3])) >= 0
hv = V[hull.vertices]
file_hv = P[spec["n_grid_points"]:]
same_vertex_set = (len(hv) == len(file_hv) and
                   np.abs(np.sort(hv.round(6).view([("", float)] * 3), axis=0).view(float).reshape(-1, 3)
                          - np.sort(file_hv.round(6).view([("", float)] * 3), axis=0).view(float).reshape(-1, 3)).max() < 1e-6)
# footprint coverage: finer probe grid, lower envelope by bisection with the Delaunay inside test
xs = np.arange(V[:, 0].min(), V[:, 0].max(), 0.05)
ys = np.arange(V[:, 1].min(), V[:, 1].max(), 0.05)
gx, gy = [g.ravel() for g in np.meshgrid(xs, ys, indexing="ij")]
lo, hi = np.full(gx.shape, V[:, 2].min() - 0.01), np.full(gx.shape, V[:, 2].max() + 0.01)
inside_any = np.zeros(gx.shape, bool)
zs = np.linspace(V[:, 2].min(), V[:, 2].max(), 200)
for z in zs:
    inside_any |= tri.find_simplex(np.stack([gx, gy, np.full(gx.shape, z)], 1)) >= 0
gx, gy = gx[inside_any], gy[inside_any]
# bisection per probe point for the lowest inside z: lo = outside (below the hull), hi = first inside level
lo = np.full(gx.shape, V[:, 2].min() - 0.01)
hi = np.array([next(z for z in zs if tri.find_simplex([[x, y, z]])[0] >= 0) for x, y in zip(gx, gy)])
for _ in range(30):
    mid = (lo + hi) / 2
    ins = tri.find_simplex(np.stack([gx, gy, mid], 1)) >= 0
    hi = np.where(ins, mid, hi)
    lo = np.where(ins, lo, mid)
probe_min = float(hi.min())
res["points_vs_hull"] = {
    "obj_sha256_matches_spec": bool(__import__("hashlib").sha256(OBJ.read_bytes()).hexdigest() == spec["source_obj_sha256"]),
    "obj_objects": objects, "obj_vertices": int(len(V)), "hull_vertices": int(len(hv)),
    "n_points": int(len(P)), "n_grid": int(spec["n_grid_points"]),
    "all_points_on_hull_surface": bool(on_surface.all()), "max_abs_facet_distance": float(np.abs(dist.max(1)).max()),
    "grid_points_on_lower_envelope (1 mm below is outside)": int(grid_below_outside.sum()),
    "grid_points_with_hull_above (1 mm above is inside)": int(grid_above_inside.sum()),
    "hull_vertex_set_equal": bool(same_vertex_set),
    "lowest_sample_z": float(P[:, 2].min()), "lowest_hull_vertex_z": float(hv[:, 2].min()),
    "probe_grid_0.05m_points_in_footprint": int(len(gx)), "probe_lowest_envelope_z": probe_min,
    "hull_x_range": [float(V[:, 0].min()), float(V[:, 0].max())], "hull_y_range": [float(V[:, 1].min()), float(V[:, 1].max())],
}


# ---------------------------------------------------------------- 2./3. recompute from runs
def rot(q):
    w, x, y, z = q
    return np.array([[w*w + x*x - y*y - z*z, 2*(x*y - w*z), 2*(x*z + w*y)],
                     [2*(x*y + w*z), w*w - x*x + y*y - z*z, 2*(y*z - w*x)],
                     [2*(x*z - w*y), 2*(y*z + w*x), w*w - x*x - y*y + z*z]])


for run in sys.argv[1:]:
    run = Path(run).resolve()
    case = json.loads((run / "case.json").read_text())
    tmap = TerrainMap.from_dir((ROOT / case["arena"]).resolve())
    s = (tmap.pixels - 1) / tmap.pixels
    r = np.load(run / "rich_telemetry.npz")
    ve = np.load(run / "vehicle_extra.npz")
    n = len(ve["belly_clearance_min_m"])
    frames = r["frame"].astype(int)
    keep = np.isin(frames, ve["frame"])
    idx = {f: i for i, f in enumerate(frames)}
    mine, mine_noscale = [], []
    for f in ve["frame"]:
        i = idx[int(f)]
        R = rot((r["quat_e0"][i], r["quat_e1"][i], r["quat_e2"][i], r["quat_e3"][i]))
        w = P @ R.T + np.array([r["pos_world_x_m"][i], r["pos_world_y_m"][i], r["pos_world_z_m"][i]])
        mine.append((w[:, 2] - tmap.height(w[:, 0] * s, w[:, 1] * s)).min())
        mine_noscale.append((w[:, 2] - tmap.height(w[:, 0], w[:, 1])).min())
    mine, mine_noscale = np.asarray(mine), np.asarray(mine_noscale)
    rec = ve["belly_clearance_min_m"].astype(float)
    # surface convention control on the hubs
    conv = {}
    for name, scale in (("node_on_edge_511_512", s), ("plain", 1.0)):
        errs = []
        for t in ("fl", "fr", "rl", "rr"):
            hx, hy, hz = r[f"tire_{t}_hub_world_x_m"], r[f"tire_{t}_hub_world_y_m"], r[f"tire_{t}_hub_world_z_m"]
            rad = r[f"tire_{t}_radius_m"]
            nz = r[f"tire_{t}_terrain_normal_under_hub_world_z_unit"]
            # on a slope the contact point is not under the hub: vertical hub height = radius / n_z (to first order)
            errs.append((hz - tmap.height(hx * scale, hy * scale)) - rad / np.clip(nz, 0.5, 1))
        e = np.concatenate(errs)
        e = e[np.isfinite(e)]
        conv[name] = {"median_abs_m": float(np.median(np.abs(e))), "p95_abs_m": float(np.percentile(np.abs(e), 95))}
    vz_equal = bool(np.array_equal(ve["chassis_vz_mps"], r["vel_world_z_mps"][np.isin(frames, ve["frame"])].astype(np.float32)))
    res[str(run.relative_to(ROOT))] = {
        "frames": int(n), "max_abs_diff_recomputed_m": float(np.abs(mine - rec).max()),
        "max_abs_diff_without_511_512_scale_m": float(np.abs(mine_noscale - rec).max()),
        "recorded_anchor_m": float(rec[0]), "recorded_min_m": float(rec.min()),
        "outcome_belly_min_equals_npz_min": bool(abs(json.loads((run / "outcome.json").read_text())["vehicle"]["belly"]["min_clearance_m"] - rec.min()) < 1e-6),
        "chassis_vz_equals_rich_telemetry_vel_world_z": vz_equal,
        "hub_height_minus_loaded_radius": conv}
print(json.dumps(res, indent=1))
