"""Independent corridor check (Task A4 verification), plus the same numbers against the Chrono-consistent frame.

Corridor geometry is re-derived here (not imported from scripts/sensor_dataset.py) from the documented
96 stations x 32 lateral (+-6 m) construction, and cross-checked against it once.
  /home/harry/miniconda3/envs/nedm/bin/python <this file>
"""
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/harry/NeDM-traverse_mppi")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from nedm.traverse.terrain import TerrainMap
import sensor_dataset as SD

BASE = ROOT / "artifacts/traverse/fdm_f104_50h_20260909/sensor_v1"
GRIDS = ROOT / "artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/grids"
OUT = Path(__file__).parent
ARENAS = {"f104": "arena_f104_50h_v1", "g203": "arena_g203", "g216": "arena_g216",
          "g217": "arena_g217", "g228": "arena_g228", "g231": "arena_g231"}
NR = 34
MPP, HALF = 80.0 / 512, 40.0
S = 511.0 / 512.0


def my_corridor(wp, st):
    """96 stations x 32 lateral offsets in +-6 m, re-derived."""
    s = np.asarray(st, float)
    if s.size != len(wp) or not np.all(np.diff(s) > 0):
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
    grid = np.linspace(s[0], s[-1], 96)
    pts = np.stack([np.interp(grid, s, wp[:, 0]), np.interp(grid, s, wp[:, 1])], 1)
    d = np.gradient(pts, axis=0); tn = np.linalg.norm(d, axis=1, keepdims=True); tn[tn < 1e-9] = 1e-9
    t = d / tn; nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    off = np.linspace(-6.0, 6.0, 32)
    return pts[:, 0:1] + nrm[:, 0:1] * off[None, :], pts[:, 1:2] + nrm[:, 1:2] * off[None, :]


def bil(img, row, col):
    n0, n1 = img.shape[-2], img.shape[-1]
    r0 = np.clip(np.floor(row).astype(int), 0, n0 - 2); c0 = np.clip(np.floor(col).astype(int), 0, n1 - 2)
    fr = np.clip(row - r0, 0, 1); fc = np.clip(col - c0, 0, 1)
    return (img[..., r0, c0] * (1 - fr) * (1 - fc) + img[..., r0, c0 + 1] * (1 - fr) * fc
            + img[..., r0 + 1, c0] * fr * (1 - fc) + img[..., r0 + 1, c0 + 1] * fr * fc)


def q(e):
    e = np.asarray(e, float)
    return dict(mae=float(np.abs(e).mean()), p95=float(np.quantile(np.abs(e), .95)),
                p99=float(np.quantile(np.abs(e), .99)), max=float(np.abs(e).max()), n=int(e.size))


def main():
    A = {k: [] for k in ("v1", "v2", "v1flat_native", "v1_minus_v2", "pred", "disp", "slope",
                         "v1c", "v2c")}
    nroutes = 0; corridor_check = None
    for tag, arena in ARENAS.items():
        files = sorted((BASE / f"test2_{tag}/routes").glob("*.json"))[:NR]
        SD.init_map(str(BASE / f"maps/{arena}"))
        with np.load(GRIDS / arena / "grid.npz") as g:
            zg = g["z"].astype(np.float64)
        tm = TerrainMap.from_dir(ROOT / "assets/traverse" / arena)
        H = SD.G["cam_h"]
        for fp in files:
            r = json.load(open(fp))
            wp = np.asarray(r["waypoints"], float); st = np.asarray(r["stations"], float)
            gx, gy = my_corridor(wp, st)
            if corridor_check is None:
                ax, ay, _ = SD.corridor(wp, st)
                corridor_check = [float(np.abs(gx - ax).max()), float(np.abs(gy - ay).max())]
            # v1 elevation: flat lookup into the 512-px encoded height
            row = SD.G["ctr"] - gy / SD.G["mpp"]; col = SD.G["ctr"] + gx / SD.G["mpp"]
            R = SD.G["rgbd"]; n = SD.G["npx"]
            r0 = np.clip(np.floor(row).astype(int), 0, n - 2); c0 = np.clip(np.floor(col).astype(int), 0, n - 2)
            fr = np.clip(row - r0, 0, 1); fc = np.clip(col - c0, 0, 1)
            a = R[3]
            e1 = (a[r0, c0] * (1 - fr) * (1 - fc) + a[r0, c0 + 1] * (1 - fr) * fc
                  + a[r0 + 1, c0] * fr * (1 - fc) + a[r0 + 1, c0 + 1] * fr * fc) * SD.G["elev_scale"]
            valid = np.minimum(np.minimum(a[r0, c0], a[r0, c0 + 1]),
                               np.minimum(a[r0 + 1, c0], a[r0 + 1, c0 + 1])) > -1.999
            # v1 flat lookup at native 1024 px (isolates the downsample)
            rw = SD.G["ctrW"] - gy / SD.G["mppW"]; cw = SD.G["ctrW"] + gx / SD.G["mppW"]
            d = bil(SD.G["depth"].astype(np.float64), rw, cw)
            sec = np.sqrt(1 + ((cw - SD.G["ctrW"]) / SD.G["f"]) ** 2 + ((rw - SD.G["ctrW"]) / SD.G["f"]) ** 2)
            e1f = H - d / sec
            # v2 grid sample
            e2 = bil(zg, (gy + HALF) / MPP - 0.5, (gx + HALF) / MPP - 0.5)
            ref = tm.height(gx, gy)
            refc = tm.height(gx * S, gy * S)
            ggx, ggy = tm.gradient(gx, gy)
            m = valid & (np.abs(gx) < HALF - 0.2) & (np.abs(gy) < HALF - 0.2) & np.isfinite(e2)
            A["v1"].append((e1 - ref)[m]); A["v2"].append((e2 - ref)[m])
            A["v1c"].append((e1 - refc)[m]); A["v2c"].append((e2 - refc)[m])
            A["v1flat_native"].append((e1f - ref)[m]); A["v1_minus_v2"].append((e1 - e2)[m])
            A["pred"].append((-(ggx * gx + ggy * gy) * ref / H)[m])
            A["disp"].append((np.hypot(gx, gy) * np.abs(ref) / H)[m])
            A["slope"].append(tm.slope(gx, gy)[m])
            nroutes += 1
    A = {k: np.concatenate(v) for k, v in A.items()}
    out = dict(
        routes=nroutes, corridor_points=int(A["v1"].size),
        my_corridor_vs_sensor_dataset_max_abs_m=corridor_check,
        v1_vs_terrainmap=q(A["v1"]), v2_vs_terrainmap=q(A["v2"]),
        v1_vs_chrono_frame=q(A["v1c"]), v2_vs_chrono_frame=q(A["v2c"]),
        v1_native_res_flat_vs_terrainmap=q(A["v1flat_native"]),
        v1_minus_v2=q(A["v1_minus_v2"]),
        predicted_displacement_m=q(A["disp"]),
        corr_v1err_pred=float(np.corrcoef(A["v1"], A["pred"])[0, 1]),
        corr_v1minusv2_pred=float(np.corrcoef(A["v1_minus_v2"], A["pred"])[0, 1]),
        regression_slope=float(np.polyfit(A["pred"], A["v1"], 1)[0]),
        r2_of_pred_for_v1err=float(1 - ((A["v1"] - A["pred"]) ** 2).sum() / ((A["v1"] - A["v1"].mean()) ** 2).sum()),
        residual_after_pred=q(A["v1"] - A["pred"]),
        frac_v1minusv2_over_10cm=float((np.abs(A["v1_minus_v2"]) > .10).mean()),
        frac_v1minusv2_over_25cm=float((np.abs(A["v1_minus_v2"]) > .25).mean()),
        frac_v1_over_5cm=float((np.abs(A["v1"]) > .05).mean()),
        frac_v2_over_5cm=float((np.abs(A["v2"]) > .05).mean()),
        by_slope=[dict(band=[lo, hi], n=int(((A["slope"] >= lo) & (A["slope"] < hi)).sum()),
                       v1=float(np.abs(A["v1"][(A["slope"] >= lo) & (A["slope"] < hi)]).mean()),
                       v2=float(np.abs(A["v2"][(A["slope"] >= lo) & (A["slope"] < hi)]).mean()))
                  for lo, hi in [(0, .10), (.10, .20), (.20, .30), (.30, 10.)]],
    )
    json.dump(out, open(OUT / "v_corridor.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
