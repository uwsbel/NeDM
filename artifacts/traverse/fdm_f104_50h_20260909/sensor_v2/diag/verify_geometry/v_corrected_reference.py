"""Corrected evaluation reference: Chrono's RigidTerrain heightmap patch spans the 80 m patch with the
512 BMP samples at its *edges* (spacing 80/511), while TerrainMap assumes sample centres at spacing 80/512
with a half-cell inset.  The exact relation is  x_terrainmap = (511/512) * x_chrono  (pure scale, no shift).

The overhead depth camera rendered Chrono's mesh, so the sensor is in the Chrono frame.  Evaluating the
adapter against the un-corrected TerrainMap therefore charges it a systematic frame error.
Corroboration that the mismatch is TerrainMap's, not the camera's: arena_meta.json's own orientation
calibration against RigidTerrain.GetHeight records rmse_m 0.00812 -- the same magnitude.

  /home/harry/miniconda3/envs/nedm/bin/python <this file>
"""
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/harry/NeDM-traverse_mppi")
sys.path.insert(0, str(ROOT / "src"))
from nedm.traverse.terrain import TerrainMap

MAPS = ROOT / "artifacts/traverse/fdm_f104_50h_20260909/sensor_v1/maps"
GRIDS = ROOT / "artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/grids"
OUT = Path(__file__).parent
ARENAS = ["arena_f104_50h_v1", "arena_g203", "arena_g216", "arena_g217", "arena_g228", "arena_g231"]
S = 511.0 / 512.0


def stats(e):
    e = np.abs(np.asarray(e, float))
    return dict(mae=float(e.mean()), p95=float(np.quantile(e, .95)), max=float(e.max()),
                rmse=float(np.sqrt((e ** 2).mean())), n=int(e.size))


def main():
    res = {}
    for arena in ARENAS:
        cam = json.load(open(MAPS / arena / "observation.json"))["camera"]
        with np.load(MAPS / arena / "observation.npz") as o:
            depth = o["depth_m"].astype(np.float64); rgbd = o["rgbd"].astype(np.float64)
        h, w = depth.shape
        H = float(cam["cam_height_m"]); f = (w / 2) / math.tan(float(cam["hfov_rad"]) / 2)
        tm = TerrainMap.from_dir(ROOT / "assets/traverse" / arena)

        with np.load(GRIDS / arena / "grid.npz") as g:
            z2 = g["z"].astype(np.float64)
        n, half, mpp = 512, 40.0, 80.0 / 512
        c = -half + (np.arange(n) + 0.5) * mpp
        X, Y = np.meshgrid(c, c)
        ref_plain = tm.height(X, Y)
        ref_chrono = tm.height(X * S, Y * S)

        # v1 flat-ground sampling of the encoded 512-px elevation channel at the same locations
        npx = rgbd.shape[1]
        ground = 2 * H * math.tan(float(cam["hfov_rad"]) / 2)
        m1 = ground / npx; ctr = (npx - 1) / 2.0
        row = ctr - Y / m1; col = ctr + X / m1
        r0 = np.clip(np.floor(row).astype(int), 0, npx - 2); c0 = np.clip(np.floor(col).astype(int), 0, npx - 2)
        fr = np.clip(row - r0, 0, 1); fc = np.clip(col - c0, 0, 1)
        A = rgbd[3]
        p = (A[r0, c0] * (1 - fr) * (1 - fc) + A[r0, c0 + 1] * (1 - fr) * fc
             + A[r0 + 1, c0] * fr * (1 - fc) + A[r0 + 1, c0 + 1] * fr * fc)
        z1 = p * float(cam["elevation_scale_m"])
        v1valid = np.minimum(np.minimum(A[r0, c0], A[r0, c0 + 1]), np.minimum(A[r0 + 1, c0], A[r0 + 1, c0 + 1])) > -1.999

        ok = np.isfinite(z2) & v1valid
        res[arena] = dict(
            v2_vs_terrainmap_asis=stats((z2 - ref_plain)[ok]),
            v2_vs_chrono_frame=stats((z2 - ref_chrono)[ok]),
            v1_vs_terrainmap_asis=stats((z1 - ref_plain)[ok]),
            v1_vs_chrono_frame=stats((z1 - ref_chrono)[ok]),
            ratio_mae_asis=float(stats((z1 - ref_plain)[ok])["mae"] / stats((z2 - ref_plain)[ok])["mae"]),
            ratio_mae_chrono=float(stats((z1 - ref_chrono)[ok])["mae"] / stats((z2 - ref_chrono)[ok])["mae"]),
            terrainmap_vs_chrono_frame=stats((ref_plain - ref_chrono)[ok]),
            quant_step_m=float(json.load(open(ROOT / "assets/traverse" / arena / "arena_meta.json"))["quantization_step_m"]),
        )
        r = res[arena]
        print(f"{arena}: v2 MAE  as-is {r['v2_vs_terrainmap_asis']['mae']:.5f} -> chrono-frame {r['v2_vs_chrono_frame']['mae']:.5f}"
              f"  | v1 MAE as-is {r['v1_vs_terrainmap_asis']['mae']:.5f} -> {r['v1_vs_chrono_frame']['mae']:.5f}"
              f"  | ratio {r['ratio_mae_asis']:.2f}x -> {r['ratio_mae_chrono']:.2f}x"
              f"  | TerrainMap-vs-Chrono {r['terrainmap_vs_chrono_frame']['mae']:.5f}", flush=True)
    json.dump(dict(scale=S, note="ref_chrono(x,y) = TerrainMap.height(511/512*x, 511/512*y)", arenas=res),
              open(OUT / "v_corrected_reference.json", "w"), indent=1)


if __name__ == "__main__":
    main()
