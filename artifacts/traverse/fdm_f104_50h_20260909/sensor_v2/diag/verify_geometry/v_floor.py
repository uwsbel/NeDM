"""Is the v2 residual really at the 8-bit quantization floor?

The depth camera renders a mesh built from the SAME quantized BMP that TerrainMap reads, so the quantization
is common-mode and should largely cancel -- it cannot set a floor the way independent noise would.
Test: replace the cell-centre reference by the mean of the reference evaluated AT the contributing pixels'
own back-projected (x, y).  If the residual collapses, the 0.007 m is within-cell sampling, not sensor error.

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
HALF, N = 40.0, 512
MPP = 2 * HALF / N


def main():
    res = {}
    for arena in ARENAS:
        cam = json.load(open(MAPS / arena / "observation.json"))["camera"]
        with np.load(MAPS / arena / "observation.npz") as o:
            depth = o["depth_m"].astype(np.float64)
        h, w = depth.shape
        H = float(cam["cam_height_m"]); f = (w / 2) / math.tan(float(cam["hfov_rad"]) / 2)
        col = np.arange(w, dtype=float)[None, :]; row = np.arange(h, dtype=float)[:, None]
        rx = (col - (w - 1) / 2) / f + 0 * row; ry = -(row - (h - 1) / 2) / f + 0 * col
        sec = np.sqrt(1 + rx ** 2 + ry ** 2)
        ok = np.isfinite(depth) & (depth > 0) & (depth < float(cam["max_depth_m"]) - 1e-6)
        ax = np.where(ok, depth / sec, np.nan)
        x = rx * ax; y = ry * ax; z = H - ax
        ins = ok & (np.abs(x) < HALF) & (np.abs(y) < HALF)
        xi = np.clip(np.floor((x[ins] + HALF) / MPP).astype(int), 0, N - 1)
        yi = np.clip(np.floor((y[ins] + HALF) / MPP).astype(int), 0, N - 1)
        idx = yi * N + xi

        tm = TerrainMap.from_dir(ROOT / "assets/traverse" / arena)
        # per-pixel reference at the pixel's own back-projected position
        ref_px = tm.height(x[ins], y[ins])
        cnt = np.bincount(idx, minlength=N * N).astype(float)
        ref_cellmean = (np.bincount(idx, weights=ref_px, minlength=N * N) / cnt).reshape(N, N)
        zg = (np.bincount(idx, weights=z[ins], minlength=N * N) / cnt).reshape(N, N)

        c = -HALF + (np.arange(N) + 0.5) * MPP
        X, Y = np.meshgrid(c, c)
        ref_centre = tm.height(X, Y)

        e_centre = zg - ref_centre           # the shipped comparison
        e_pixel = zg - ref_cellmean          # sensor error with the within-cell offset removed
        e_geom = ref_cellmean - ref_centre   # the pure within-cell sampling offset

        # per-pixel sensor error (no cell averaging at all)
        e_perpixel = z[ins] - ref_px

        am = json.load(open(ROOT / "assets/traverse" / arena / "arena_meta.json"))
        step = float(am["quantization_step_m"])

        res[arena] = dict(
            quant_step_m=step, ideal_quant_mae=step / 4,
            mae_vs_cell_centre=float(np.abs(e_centre).mean()),
            mae_vs_cellmean_of_ref_at_pixels=float(np.abs(e_pixel).mean()),
            mae_of_pure_within_cell_offset=float(np.abs(e_geom).mean()),
            rmse_vs_cell_centre=float(np.sqrt((e_centre ** 2).mean())),
            rmse_vs_cellmean=float(np.sqrt((e_pixel ** 2).mean())),
            per_pixel_sensor_mae=float(np.abs(e_perpixel).mean()),
            per_pixel_sensor_p95=float(np.quantile(np.abs(e_perpixel), .95)),
            per_pixel_sensor_max=float(np.abs(e_perpixel).max()),
            variance_explained_by_within_cell_offset=float(1 - (e_pixel ** 2).mean() / (e_centre ** 2).mean()),
            corr_e_centre_e_geom=float(np.corrcoef(e_centre.ravel(), e_geom.ravel())[0, 1]),
        )
        r = res[arena]
        print(f"{arena}: vs centre {r['mae_vs_cell_centre']:.5f} | vs cell-mean-of-ref {r['mae_vs_cellmean_of_ref_at_pixels']:.5f}"
              f" | pure offset {r['mae_of_pure_within_cell_offset']:.5f} | per-pixel sensor {r['per_pixel_sensor_mae']:.5f}"
              f" | quant floor {step/4:.5f}", flush=True)
    json.dump(res, open(OUT / "v_floor.json", "w"), indent=1)


if __name__ == "__main__":
    main()
