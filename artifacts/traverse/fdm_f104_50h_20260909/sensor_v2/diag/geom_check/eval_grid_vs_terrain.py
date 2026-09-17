"""Task A2: v2 grid heights vs TerrainMap, and vs the v1 flat-ground sampling of the same locations.

Read-only. TerrainMap is an evaluation reference only; nothing here feeds a pipeline.
  /home/harry/miniconda3/envs/nedm/bin/python <this file>
"""
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/harry/NeDM-traverse_mppi")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nedm.traverse.terrain import TerrainMap
import sensor_dataset as SD

GRIDS = ROOT / "artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/grids"
MAPS = ROOT / "artifacts/traverse/fdm_f104_50h_20260909/sensor_v1/maps"
ARENAS = ["arena_f104_50h_v1", "arena_g203", "arena_g216", "arena_g217", "arena_g228", "arena_g231"]


def v1_elev(gx, gy):
    """Exactly scripts/sensor_dataset.tensor10 channel 0 before the -e0 offset (flat-ground sampling of rgbd[3])."""
    row = SD.G['ctr'] - gy / SD.G['mpp']; col = SD.G['ctr'] + gx / SD.G['mpp']
    n = SD.G['npx']; R = SD.G['rgbd']
    r0 = np.clip(np.floor(row).astype(int), 0, n - 2); c0 = np.clip(np.floor(col).astype(int), 0, n - 2)
    fr = np.clip(row - r0, 0, 1)[None]; fc = np.clip(col - c0, 0, 1)[None]
    p00, p01, p10, p11 = R[:, r0, c0], R[:, r0, c0 + 1], R[:, r0 + 1, c0], R[:, r0 + 1, c0 + 1]
    valid = np.min(np.stack([p00[3], p01[3], p10[3], p11[3]]), axis=0) > -1.999
    patch = p00 * (1 - fr) * (1 - fc) + p01 * (1 - fr) * fc + p10 * fr * (1 - fc) + p11 * fr * fc
    return patch[3] * SD.G['elev_scale'], valid


def stats(e):
    e = np.abs(e)
    return dict(mae=float(e.mean()), p95=float(np.quantile(e, .95)), p99=float(np.quantile(e, .99)),
                max=float(e.max()), rmse=float(np.sqrt((e ** 2).mean())), n=int(e.size))


def binned(err_v2, err_v1, key, edges, label):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (key >= lo) & (key < hi)
        if m.sum() < 50:
            continue
        out.append({label: [float(lo), float(hi)], "cells": int(m.sum()),
                    "v2_mae": float(np.abs(err_v2[m]).mean()), "v2_p95": float(np.quantile(np.abs(err_v2[m]), .95)),
                    "v2_max": float(np.abs(err_v2[m]).max()),
                    "v1_mae": float(np.abs(err_v1[m]).mean()), "v1_p95": float(np.quantile(np.abs(err_v1[m]), .95)),
                    "v1_max": float(np.abs(err_v1[m]).max())})
    return out


def main():
    res = {}
    for arena in ARENAS:
        with np.load(GRIDS / arena / "grid.npz") as g:
            z = g['z'].astype(np.float64); cover = g['cover']; rng = g['range_m'].astype(np.float64)
            sec = g['sec'].astype(np.float64)
        meta = json.load(open(GRIDS / arena / "grid.json"))
        mpp, half, n = meta['mpp'], meta['half_extent_m'], meta['n']
        c = -half + (np.arange(n) + 0.5) * mpp
        X, Y = np.meshgrid(c, c)                     # X[iy,ix], Y[iy,ix]; row 0 = -y (grid yi = (y+half)/mpp)
        terrain = TerrainMap.from_dir(ROOT / "assets/traverse" / arena)
        ref = terrain.height(X, Y)
        slope = terrain.slope(X, Y)
        SD.init_map(str(MAPS / arena))
        e1, v1valid = v1_elev(X, Y)

        ok = (cover > 0) & np.isfinite(z) & v1valid
        err2 = (z - ref)[ok]; err1 = (e1 - ref)[ok]
        r = np.hypot(X, Y)[ok]; sl = slope[ok]
        # bias-removed variants (the model input is height relative to the route start)
        d2 = err2 - err2.mean(); d1 = err1 - err1.mean()
        # self-consistency of the stored range/sec against the stored z
        zrec = meta['camera_height_m'] - rng / sec
        rec_err = (zrec - z)[ok]

        res[arena] = {
            "cells_compared": int(ok.sum()), "cells_total": int(n * n),
            "empty_cells": int((cover == 0).sum()),
            "cover_min": float(cover[cover > 0].min()), "cover_max": float(cover.max()),
            "cover_mean": float(cover.mean()),
            "v2_abs": stats(err2), "v1_abs": stats(err1),
            "v2_bias_m": float(err2.mean()), "v1_bias_m": float(err1.mean()),
            "v2_demeaned": stats(d2), "v1_demeaned": stats(d1),
            "v2_corr_with_terrain": float(np.corrcoef(z[ok], ref[ok])[0, 1]),
            "v1_corr_with_terrain": float(np.corrcoef(e1[ok], ref[ok])[0, 1]),
            "terrain_height_std_m": float(ref[ok].std()),
            "z_from_stored_range_over_sec_minus_z": stats(rec_err),
            "by_radius_m": binned(err2, err1, r, [0, 5, 10, 15, 20, 25, 30, 35, 40, 60], "radius_m"),
            "by_slope_tan": binned(err2, err1, sl, [0, .05, .10, .15, .20, .25, .30, .40, 10.], "slope_tan"),
        }
        s = res[arena]
        print(f"{arena}: v2 MAE {s['v2_abs']['mae']:.4f} p95 {s['v2_abs']['p95']:.4f} max {s['v2_abs']['max']:.3f} | "
              f"v1 MAE {s['v1_abs']['mae']:.4f} p95 {s['v1_abs']['p95']:.4f} max {s['v1_abs']['max']:.3f} | "
              f"empty {s['empty_cells']} cover[{s['cover_min']:.0f},{s['cover_max']:.0f}]", flush=True)

    out = GRIDS.parent / "diag/geom_check/eval_grid_vs_terrain.json"
    json.dump({"note": "TerrainMap heights are an evaluation reference only. Cell centres of the v2 grid "
                       "coincide exactly with TerrainMap BMP pixel centres (both 0.15625 m over +-40 m).",
               "arenas": res}, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
