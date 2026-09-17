"""Task A4: 200 real candidate routes -- corridor heights sampled from the v1 map vs the v2 grid vs TerrainMap.

Also checks that the v1-vs-v2 difference is the size the flat-ground displacement predicts:
a pixel whose *flat* coordinate is (x,y) really sits at (x,y)*(H-z)/H, so to first order
    v1(x,y) - terrain(x,y) ~= -(grad_x*x + grad_y*y) * z / H .
Read-only. TerrainMap is an evaluation reference only.
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

BASE = ROOT / "artifacts/traverse/fdm_f104_50h_20260909/sensor_v1"
GRIDS = ROOT / "artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/grids"
ARENAS = {"f104": "arena_f104_50h_v1", "g203": "arena_g203", "g216": "arena_g216",
          "g217": "arena_g217", "g228": "arena_g228", "g231": "arena_g231"}
N_ROUTES_PER_ARENA = 34          # 6 x 34 = 204 routes, truncated to 200
MPP, HALF, N = 80.0 / 512, 40.0, 512


def bilin(img, row, col):
    n0, n1 = img.shape[-2], img.shape[-1]
    r0 = np.clip(np.floor(row).astype(int), 0, n0 - 2); c0 = np.clip(np.floor(col).astype(int), 0, n1 - 2)
    fr = np.clip(row - r0, 0, 1); fc = np.clip(col - c0, 0, 1)
    return (img[..., r0, c0] * (1 - fr) * (1 - fc) + img[..., r0, c0 + 1] * (1 - fr) * fc
            + img[..., r0 + 1, c0] * fr * (1 - fc) + img[..., r0 + 1, c0 + 1] * fr * fc)


def v1_elev(gx, gy):
    """scripts/sensor_dataset.tensor10 channel 0 (absolute, before -e0): 512-px rgbd elevation, flat lookup."""
    row = SD.G['ctr'] - gy / SD.G['mpp']; col = SD.G['ctr'] + gx / SD.G['mpp']
    n = SD.G['npx']; R = SD.G['rgbd']
    r0 = np.clip(np.floor(row).astype(int), 0, n - 2); c0 = np.clip(np.floor(col).astype(int), 0, n - 2)
    fr = np.clip(row - r0, 0, 1)[None]; fc = np.clip(col - c0, 0, 1)[None]
    p00, p01, p10, p11 = R[:, r0, c0], R[:, r0, c0 + 1], R[:, r0 + 1, c0], R[:, r0 + 1, c0 + 1]
    valid = np.min(np.stack([p00[3], p01[3], p10[3], p11[3]]), axis=0) > -1.999
    patch = p00 * (1 - fr) * (1 - fc) + p01 * (1 - fr) * fc + p10 * fr * (1 - fc) + p11 * fr * fc
    return patch[3] * SD.G['elev_scale'], valid


def v1_flat_fullres(gx, gy):
    """Same flat-ground lookup but on the native 1024-px depth (isolates geometry from the 512 downsample)."""
    row = SD.G['ctrW'] - gy / SD.G['mppW']; col = SD.G['ctrW'] + gx / SD.G['mppW']
    d = bilin(SD.G['depth'], row, col)
    sec = np.sqrt(1 + ((col - SD.G['ctrW']) / SD.G['f']) ** 2 + ((row - SD.G['ctrW']) / SD.G['f']) ** 2)
    return SD.G['cam_h'] - d / sec


def v2_sample(zg, gx, gy):
    col = (gx + HALF) / MPP - 0.5; row = (gy + HALF) / MPP - 0.5      # grid row index runs with +y
    return bilin(zg, row, col)


def q(e):
    e = np.asarray(e, float)
    return dict(mean=float(e.mean()), mae=float(np.abs(e).mean()), p50=float(np.quantile(np.abs(e), .50)),
                p95=float(np.quantile(np.abs(e), .95)), p99=float(np.quantile(np.abs(e), .99)),
                max=float(np.abs(e).max()), rmse=float(np.sqrt((e ** 2).mean())), n=int(e.size))


def main():
    rows = {"centreline": {}, "corridor": {}}
    acc = {k: {n: [] for n in ("v1_err", "v2_err", "v1flat_err", "v1_minus_v2", "pred_v1_err",
                               "disp_m", "radius_m", "slope")} for k in rows}
    per_arena = {}
    used = []
    for tag, arena in ARENAS.items():
        rdir = BASE / f"test2_{tag}/routes"
        files = sorted(rdir.glob("*.json"))[:N_ROUTES_PER_ARENA]
        SD.init_map(str(BASE / f"maps/{arena}"))
        with np.load(GRIDS / arena / "grid.npz") as g:
            zg = g['z'].astype(np.float64)
        terrain = TerrainMap.from_dir(ROOT / "assets/traverse" / arena)
        H = SD.G['cam_h']
        loc = {k: {n: [] for n in acc[k]} for k in rows}
        for fp in files:
            r = json.load(open(fp))
            wp = np.asarray(r['waypoints'], float); st = np.asarray(r['stations'], float)
            gx, gy, _ = SD.corridor(wp, st)
            e1, valid = v1_elev(gx, gy)
            e1f = v1_flat_fullres(gx, gy)
            e2 = v2_sample(zg, gx, gy)
            ref = terrain.height(gx, gy)
            ggx, ggy = terrain.gradient(gx, gy)
            pred = -(ggx * gx + ggy * gy) * ref / H
            disp = np.hypot(gx, gy) * np.abs(ref) / H
            sl = terrain.slope(gx, gy)
            rad = np.hypot(gx, gy)
            sel = {"centreline": (slice(None), 16), "corridor": (slice(None), slice(None))}
            for k, s in sel.items():
                m = valid[s] & (np.abs(gx[s]) < HALF - 0.2) & (np.abs(gy[s]) < HALF - 0.2)
                loc[k]["v1_err"].append((e1[s] - ref[s])[m]); loc[k]["v2_err"].append((e2[s] - ref[s])[m])
                loc[k]["v1flat_err"].append((e1f[s] - ref[s])[m]); loc[k]["v1_minus_v2"].append((e1[s] - e2[s])[m])
                loc[k]["pred_v1_err"].append(pred[s][m]); loc[k]["disp_m"].append(disp[s][m])
                loc[k]["radius_m"].append(rad[s][m]); loc[k]["slope"].append(sl[s][m])
            used.append(fp.name)
        for k in rows:
            for n in acc[k]:
                a = np.concatenate(loc[k][n]); acc[k][n].append(a)
        per_arena[arena] = {k: dict(v1=q(np.concatenate(loc[k]['v1_err'])), v2=q(np.concatenate(loc[k]['v2_err'])),
                                    v1_minus_v2=q(np.concatenate(loc[k]['v1_minus_v2'])),
                                    v1_flat_fullres=q(np.concatenate(loc[k]['v1flat_err'])))
                            for k in rows}
        print(f"{arena}: centreline  v1 MAE {per_arena[arena]['centreline']['v1']['mae']:.4f}  "
              f"v2 MAE {per_arena[arena]['centreline']['v2']['mae']:.4f}  "
              f"|v1-v2| p95 {per_arena[arena]['centreline']['v1_minus_v2']['p95']:.4f}", flush=True)

    out = {"routes_used": len(used), "route_files_head": used[:5],
           "routes_per_arena": N_ROUTES_PER_ARENA, "stations": 96, "lateral": 32, "per_arena": per_arena}
    for k in rows:
        A = {n: np.concatenate(acc[k][n]) for n in acc[k]}
        pooled = dict(v1_vs_terrain=q(A['v1_err']), v2_vs_terrain=q(A['v2_err']),
                      v1_fullres_flat_vs_terrain=q(A['v1flat_err']), v1_minus_v2=q(A['v1_minus_v2']),
                      predicted_v1_err=q(A['pred_v1_err']),
                      predicted_displacement_m=q(A['disp_m']),
                      corr_v1err_vs_prediction=float(np.corrcoef(A['v1_err'], A['pred_v1_err'])[0, 1]),
                      corr_v1minusv2_vs_prediction=float(np.corrcoef(A['v1_minus_v2'], A['pred_v1_err'])[0, 1]),
                      slope_of_v1err_on_prediction=float(np.polyfit(A['pred_v1_err'], A['v1_err'], 1)[0]),
                      residual_after_prediction=q(A['v1_err'] - A['pred_v1_err']),
                      frac_v1_err_over_5cm=float((np.abs(A['v1_err']) > .05).mean()),
                      frac_v1_err_over_10cm=float((np.abs(A['v1_err']) > .10).mean()),
                      frac_v2_err_over_5cm=float((np.abs(A['v2_err']) > .05).mean()),
                      frac_v1minusv2_over_10cm=float((np.abs(A['v1_minus_v2']) > .10).mean()),
                      frac_v1minusv2_over_25cm=float((np.abs(A['v1_minus_v2']) > .25).mean()))
        by = []
        for lo, hi in [(0, .10), (.10, .20), (.20, .30), (.30, 10.)]:
            m = (A['slope'] >= lo) & (A['slope'] < hi)
            if m.sum() > 100:
                by.append(dict(slope_tan=[lo, hi], n=int(m.sum()), v1_mae=float(np.abs(A['v1_err'][m]).mean()),
                               v2_mae=float(np.abs(A['v2_err'][m]).mean()),
                               v1_minus_v2_p95=float(np.quantile(np.abs(A['v1_minus_v2'][m]), .95))))
        pooled['by_slope'] = by
        out[k] = pooled
        print(f"[{k}] v1 MAE {pooled['v1_vs_terrain']['mae']:.4f} p95 {pooled['v1_vs_terrain']['p95']:.4f} "
              f"max {pooled['v1_vs_terrain']['max']:.3f} | v2 MAE {pooled['v2_vs_terrain']['mae']:.4f} "
              f"p95 {pooled['v2_vs_terrain']['p95']:.4f} max {pooled['v2_vs_terrain']['max']:.3f} | "
              f"corr(v1 err, predicted) {pooled['corr_v1err_vs_prediction']:.3f} "
              f"slope {pooled['slope_of_v1err_on_prediction']:.3f}")
    p = GRIDS.parent / "diag/geom_check/corridor_compare.json"
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
