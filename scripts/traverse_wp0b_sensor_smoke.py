#!/usr/bin/env python
"""WP0b sensor smoke: alignment, depth-at-edges, rendering FPS (plan §14, G0b).

For N smoke layouts, renders the settled t=0 overhead RGB-D pair at collection
resolution and measures:

1. ALIGNMENT — project GT vehicle-marker / roof / canopy / rock centers
   through the CameraModel and compare to color-blob centroids.
   Bars (plan §6.4): median <= 2 px, p95 <= 4 px at 256^2.
2. DEPTH->ELEVATION — back-project the depth image under both "ray" and
   "planar" conventions and compare to the calibrated heightmap, split into
   image-center vs image-edge pixels (plan §3.3: edges checked explicitly).
3. RENDER FPS — wall-clock frames/s of the 20 Hz sim+render collection loop
   (first layout only).

Usage (from repo root, nedm env):
  PYTHONPATH=src python scripts/traverse_wp0b_sensor_smoke.py --layouts 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pychrono as chrono  # noqa: E402
import pychrono.vehicle as veh  # noqa: E402

from nedm.traverse.camera import CameraModel  # noqa: E402
from nedm.traverse.layout import sample_episode  # noqa: E402
from nedm.traverse.scene import (  # noqa: E402
    CANOPY_RGB,
    HOUSE_ROOF_RGB,
    RenderSpec,
    ROCK_RGB,
    VEHICLE_MARKER_RGB,
    build_config,
    build_scene,
)
from nedm.traverse.terrain import TerrainMap  # noqa: E402

SETTLE_S = 0.5
COLOR_TOL = 70.0  # euclidean RGB distance for blob membership
BLOB_WINDOW_PX = 14
MIN_BLOB_PX = 3


def blob_centroid(rgb: np.ndarray, u: float, v: float, ref: tuple[float, float, float]) -> tuple[float, float] | None:
    h, w, _ = rgb.shape
    x0, x1 = max(0, int(u) - BLOB_WINDOW_PX), min(w, int(u) + BLOB_WINDOW_PX + 1)
    y0, y1 = max(0, int(v) - BLOB_WINDOW_PX), min(h, int(v) + BLOB_WINDOW_PX + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    win = rgb[y0:y1, x0:x1].astype(np.float64)
    dist = np.linalg.norm(win - np.array(ref) * 255.0, axis=-1)
    mask = dist < COLOR_TOL
    if mask.sum() < MIN_BLOB_PX:
        return None
    vs, us = np.nonzero(mask)
    return x0 + float(us.mean()), y0 + float(vs.mean())


def alignment_targets(scene, layout, tmap) -> list[tuple[str, float, float, float, tuple]]:
    """(class, x, y, z_of_visible_blob_center, ref_color) for every target."""
    targets = []
    chassis = scene.hmmwv.GetChassisBody()
    marker_world = chassis.GetFrameRefToAbs().TransformPointLocalToParent(
        chrono.ChVector3d(0.1, 0.0, 0.95)
    )
    targets.append(("marker", marker_world.x, marker_world.y, marker_world.z, VEHICLE_MARKER_RGB))
    for asset in layout.assets:
        ground = float(tmap.height(asset.x_m, asset.y_m))
        if asset.kind == "house":
            z = ground + asset.dims["wall_height_m"] + 0.225 - 0.1
            targets.append(("roof", asset.x_m, asset.y_m, z, HOUSE_ROOF_RGB))
        elif asset.kind == "tree":
            z = ground + asset.dims["trunk_height_m"] + 0.55 * asset.dims["canopy_radius_m"] - 0.1
            targets.append(("canopy", asset.x_m, asset.y_m, z, CANOPY_RGB))
        elif asset.kind == "rock":
            z = ground + asset.dims["height_m"] - 0.15
            targets.append(("rock", asset.x_m, asset.y_m, z, ROCK_RGB))
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", default="assets/traverse/arena_v1")
    parser.add_argument("--layouts", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=20269000)
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--fps-frames", type=int, default=100)
    parser.add_argument("--out", default="artifacts/traverse/wp0b_sensor_smoke")
    args = parser.parse_args()

    arena_dir = (REPO_ROOT / args.arena).resolve()
    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmap = TerrainMap.from_dir(arena_dir)
    cam = CameraModel(width=args.res, height=args.res)

    errors: dict[str, list[float]] = {"marker": [], "roof": [], "canopy": [], "rock": []}
    misses: dict[str, int] = {k: 0 for k in errors}
    offsets: dict[str, list[tuple[float, float]]] = {}
    depth_stats: list[dict] = []
    fps_report = None

    for li in range(args.layouts):
        seed = args.seed0 + li
        layout, _plan = sample_episode(tmap, f"smoke_{li:02d}", seed)
        start_z = float(tmap.height(*layout.start_xy)) + 0.75
        config = build_config(arena_dir, (*layout.start_xy, start_z), layout.start_yaw)
        render = RenderSpec(width=args.res, height=args.res, cam_height_m=cam.cam_height_m,
                            hfov_rad=cam.hfov_rad, plan_markers=False)
        scene = build_scene(config, layout, tmap, arena_dir, plan=None, render=render)
        hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain
        dt = float(config["simulation"]["step_size_s"])

        # settle the vehicle, then trigger exactly one frame (manual-trigger pattern)
        inputs = veh.DriverInputs()
        inputs.m_steering, inputs.m_throttle, inputs.m_braking = 0.0, 0.0, 1.0
        while system.GetChTime() < SETTLE_S:
            t = float(system.GetChTime())
            terrain.Synchronize(t)
            hmmwv.Synchronize(t, inputs, terrain)
            terrain.Advance(dt)
            hmmwv.Advance(dt)
        scene.manager.Update()
        rgb = scene.rgb_tap.take()
        depth = scene.depth_tap.take()

        # 1) alignment
        overlay = Image.fromarray(rgb.copy()) if li == 0 else None
        draw = ImageDraw.Draw(overlay) if overlay else None
        for kind, x, y, z, ref in alignment_targets(scene, layout, tmap):
            u, v = cam.world_to_pixel(x, y, z)
            u, v = float(u), float(v)
            if not (2 <= u < args.res - 2 and 2 <= v < args.res - 2):
                continue
            found = blob_centroid(rgb, u, v, ref)
            if found is None:
                misses[kind] += 1
                iu, iv = int(round(u)), int(round(v))
                patch = rgb[max(0, iv - 2):iv + 3, max(0, iu - 2):iu + 3].reshape(-1, 3)
                print(f"  MISS {kind} at ({u:.1f},{v:.1f}) expected {tuple(int(255*c) for c in ref)} "
                      f"rendered mean {patch.mean(axis=0).round(0)}", flush=True)
                continue
            offsets.setdefault(kind, []).append((found[0] - u, found[1] - v))
            errors[kind].append(math.hypot(found[0] - u, found[1] - v))
            if draw:
                draw.line([(u - 3, v), (u + 3, v)], fill=(0, 255, 255))
                draw.line([(u, v - 3), (u, v + 3)], fill=(0, 255, 255))
                draw.ellipse([found[0] - 3, found[1] - 3, found[0] + 3, found[1] + 3], outline=(255, 0, 255))
        if overlay:
            overlay.resize((512, 512), Image.NEAREST).save(out_dir / "alignment_overlay.png")

        # depth image orientation self-test: which flip combo matches the map?
        if li == 0:
            for name, dimg in (("as-is", depth), ("flipud", depth[::-1]),
                               ("fliplr", depth[:, ::-1]), ("rot180", depth[::-1, ::-1])):
                wx, wy, wz = cam.depth_to_world(dimg, convention="ray")
                m = np.isfinite(dimg) & (np.abs(wx) < 0.45 * tmap.size_m) & (np.abs(wy) < 0.45 * tmap.size_m)
                e = float(np.median(np.abs(wz[m] - tmap.height(wx[m], wy[m]))))
                print(f"  depth orientation {name}: median |err| {e:.3f} m", flush=True)

        # 2) depth -> elevation vs calibrated heightmap (terrain pixels only)
        row = {"layout": li}
        for conv in ("ray", "planar"):
            wx, wy, wz = cam.depth_to_world(depth, convention=conv)
            valid = np.isfinite(depth) & (np.abs(wx) < 0.47 * tmap.size_m) & (np.abs(wy) < 0.47 * tmap.size_m)
            for asset in layout.assets:  # exclude asset + vehicle pixels
                valid &= (wx - asset.x_m) ** 2 + (wy - asset.y_m) ** 2 > (asset.footprint_radius_m + 1.5) ** 2
            ref = scene.hmmwv.GetChassis().GetBody().GetFrameRefToAbs().GetPos()
            valid &= (wx - ref.x) ** 2 + (wy - ref.y) ** 2 > 5.0**2
            err = np.abs(wz[valid] - tmap.height(wx[valid], wy[valid]))
            uu, vv = np.meshgrid(np.arange(args.res), np.arange(args.res))
            r = np.hypot(uu - cam.cx, vv - cam.cy)[valid]
            row[conv] = {
                "median_all_m": float(np.median(err)),
                "p95_all_m": float(np.percentile(err, 95)),
                "median_center_m": float(np.median(err[r < 0.25 * args.res])),
                "median_edge_m": float(np.median(err[r > 0.45 * args.res])),
                "p95_edge_m": float(np.percentile(err[r > 0.45 * args.res], 95)),
            }
        depth_stats.append(row)

        # 3) render FPS probe on the first layout
        if li == 0:
            substeps = max(1, int(round(1.0 / (20.0 * dt))))
            t0 = time.time()
            for _ in range(args.fps_frames):
                for _ in range(substeps):
                    t = float(system.GetChTime())
                    terrain.Synchronize(t)
                    hmmwv.Synchronize(t, inputs, terrain)
                    terrain.Advance(dt)
                    hmmwv.Advance(dt)
                scene.manager.Update()
                scene.rgb_tap.take()
                scene.depth_tap.take()
            wall = time.time() - t0
            fps_report = {
                "frames": args.fps_frames,
                "res": args.res,
                "wall_s": wall,
                "frames_per_s": args.fps_frames / wall,
                "sim_seconds_per_wall_second": args.fps_frames * 0.05 / wall,
            }
        print(f"layout {li}: targets ok", flush=True)

    all_err = [e for v in errors.values() for e in v]
    ray_med = float(np.median([d["ray"]["median_all_m"] for d in depth_stats]))
    planar_med = float(np.median([d["planar"]["median_all_m"] for d in depth_stats]))
    winner = "ray" if ray_med <= planar_med else "planar"
    summary = {
        "layouts": args.layouts,
        "res": args.res,
        "alignment_px": {
            "median": float(np.median(all_err)),
            "p95": float(np.percentile(all_err, 95)),
            "max": float(np.max(all_err)),
            "n": len(all_err),
            "per_class_median": {k: float(np.median(v)) for k, v in errors.items() if v},
            "per_class_mean_offset_uv": {
                k: [float(np.mean([o[0] for o in v])), float(np.mean([o[1] for o in v]))]
                for k, v in offsets.items()
            },
            "per_class_n": {k: len(v) for k, v in errors.items()},
            "misses": misses,
            "pass_2px_4px": bool(np.median(all_err) <= 2.0 and np.percentile(all_err, 95) <= 4.0),
        },
        "depth_convention_winner": winner,
        "depth_median_all_m": {"ray": ray_med, "planar": planar_med},
        "depth_edge_m": {
            conv: {
                "median": float(np.median([d[conv]["median_edge_m"] for d in depth_stats])),
                "p95": float(np.median([d[conv]["p95_edge_m"] for d in depth_stats])),
            }
            for conv in ("ray", "planar")
        },
        "fps": fps_report,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "per_layout_depth": depth_stats}, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0 if summary["alignment_px"]["pass_2px_4px"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
