"""Read-only check of saved depth geometry; no rendering, training, or simulation.

Run with /home/harry/miniconda3/envs/nedm/bin/python audit_geometry.py.
BMP heights are evaluation references only, never sensor inputs.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/harry/NeDM-traverse_mppi")
sys.path.insert(0, str(ROOT / "src"))
from nedm.traverse.terrain import TerrainMap


def main():
    maps = ROOT / "artifacts/traverse/fdm_f104_50h_20260909/sensor_v1/maps"
    results = []
    for directory in sorted(maps.iterdir()):
        camera = json.loads((directory / "observation.json").read_text())["camera"]
        with np.load(directory / "observation.npz") as observation:
            depth = observation["depth_m"]
        height, width = depth.shape
        camera_height = camera["cam_height_m"]
        focal = width / 2 / np.tan(camera["hfov_rad"] / 2)
        v, u = np.mgrid[0:height:8, 0:width:8]
        ray_x = (u - (width - 1) / 2) / focal
        ray_y = -(v - (height - 1) / 2) / focal
        distance = depth[::8, ::8]
        axial_distance = distance / np.sqrt(1 + ray_x**2 + ray_y**2)
        z = camera_height - axial_distance
        flat_x, flat_y = ray_x * camera_height, ray_y * camera_height
        true_x, true_y = ray_x * axial_distance, ray_y * axial_distance
        terrain = TerrainMap.from_dir(ROOT / "assets/traverse" / directory.name)
        valid = (
            np.isfinite(distance) & (distance > 0) & (distance < camera["max_depth_m"])
            & (np.abs(flat_x) < 38) & (np.abs(flat_y) < 38)
            & (np.abs(true_x) < 38) & (np.abs(true_y) < 38)
        )
        flat_error = np.abs(z[valid] - terrain.height(flat_x[valid], flat_y[valid]))
        projected_error = np.abs(z[valid] - terrain.height(true_x[valid], true_y[valid]))
        shift = np.hypot(flat_x - true_x, flat_y - true_y)[valid]
        results.append({
            "arena": directory.name,
            "sample_count": int(valid.sum()),
            "flat_grid_height_mae_m": float(flat_error.mean()),
            "backprojected_height_mae_m": float(projected_error.mean()),
            "flat_grid_height_p95_m": float(np.quantile(flat_error, .95)),
            "backprojected_height_p95_m": float(np.quantile(projected_error, .95)),
            "xy_shift_p95_m": float(np.quantile(shift, .95)),
            "xy_shift_max_m": float(shift.max()),
            "observation_sha256": hashlib.sha256((directory / "observation.npz").read_bytes()).hexdigest(),
        })
    print(json.dumps({
        "method": "Every eighth raw sensor pixel; both world-coordinate assignments inside +/-38 m; BMP bilinear height reference.",
        "limitation": "Checks raw point geometry, not a corrected rasterizer, route ranking, or closed-loop performance.",
        "arenas": results,
    }, indent=2))


if __name__ == "__main__":
    main()
