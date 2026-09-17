"""Independent check of TASK A / TASK C's geometry headline: v2 grid heights vs the authored heightmap,
and the v1 flat-ground lookup at the same cells, on all six arenas. TerrainMap is an evaluation reference only.

Also checks the two orientation conventions explicitly (grid row index vs +y) and the shipped grid.json metadata.
"""
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path('/home/harry/NeDM-traverse_mppi')
EXP = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from nedm.traverse.terrain import TerrainMap

ARENAS = ['arena_f104_50h_v1', 'arena_g203', 'arena_g216', 'arena_g217', 'arena_g228', 'arena_g231']
out = {}
for a in ARENAS:
    g = np.load(EXP / f'sensor_v2/grids/{a}/grid.npz')
    meta = json.load(open(EXP / f'sensor_v2/grids/{a}/grid.json'))
    z = g['z'].astype(np.float64); cover = g['cover']
    n = int(meta['n']); mpp = float(meta['mpp']); half = float(meta['half_extent_m'])
    tm = TerrainMap.from_dir(ROOT / f'assets/traverse/{a}')
    xs = -half + (np.arange(n) + 0.5) * mpp
    X, Y = np.meshgrid(xs, xs)                      # X varies along columns, Y along rows
    T = tm._bilinear(tm.height_grid, X, Y)
    err = z - T
    # v1 flat-ground lookup of the same world points, from the raw capture
    cam = json.load(open(EXP / f'sensor_v1/maps/{a}/observation.json'))['camera']
    with np.load(EXP / f'sensor_v1/maps/{a}/observation.npz') as o:
        depth = o['depth_m'].astype(np.float64)
    H = float(cam['cam_height_m']); w = depth.shape[1]
    f = (w / 2) / math.tan(float(cam['hfov_rad']) / 2)
    ground_mpp = 2 * H * math.tan(float(cam['hfov_rad']) / 2) / w
    ctr = (w - 1) / 2.0
    row = ctr - Y / ground_mpp; col = ctr + X / ground_mpp
    r0 = np.clip(np.floor(row).astype(int), 0, w - 2); c0 = np.clip(np.floor(col).astype(int), 0, w - 2)
    fr = np.clip(row - r0, 0, 1); fc = np.clip(col - c0, 0, 1)
    bl = lambda I: (I[r0, c0] * (1 - fr) * (1 - fc) + I[r0, c0 + 1] * (1 - fr) * fc
                    + I[r0 + 1, c0] * fr * (1 - fc) + I[r0 + 1, c0 + 1] * fr * fc)
    secpix = np.sqrt(1 + ((col - ctr) / f) ** 2 + ((row - ctr) / f) ** 2)
    z_v1 = H - bl(depth) / secpix
    e1 = z_v1 - T
    # orientation controls
    flips = dict(identity=z, flipud=np.flipud(z), fliplr=np.fliplr(z), transpose=z.T, rot180=z[::-1, ::-1])
    orient = {k: float(np.nanmean(np.abs(v - T))) for k, v in flips.items()}
    out[a] = dict(
        n_cells=int(z.size), empty_cells=int((cover == 0).sum()),
        v2=dict(mae=float(np.abs(err).mean()), p95=float(np.percentile(np.abs(err), 95)),
                max=float(np.abs(err).max()), bias=float(err.mean()),
                corr=float(np.corrcoef(z.ravel(), T.ravel())[0, 1])),
        v1_flat=dict(mae=float(np.abs(e1).mean()), p95=float(np.percentile(np.abs(e1), 95)),
                     max=float(np.abs(e1).max()), bias=float(e1.mean()),
                     corr=float(np.corrcoef(z_v1.ravel(), T.ravel())[0, 1])),
        bmp_quant_step_m=float((tm.meta['height_max_m'] - tm.meta['height_min_m']) / 255.0),
        orientation_mae=orient,
        shipped_grid_json_keys=sorted(meta.keys()))
    r = out[a]
    print(f"{a:20s} v2 MAE {r['v2']['mae']:.4f} p95 {r['v2']['p95']:.4f} max {r['v2']['max']:.4f} "
          f"bias {r['v2']['bias']:+.4f} | v1-flat MAE {r['v1_flat']['mae']:.4f} p95 {r['v1_flat']['p95']:.4f} "
          f"| quant step {r['bmp_quant_step_m']:.4f} (step/4 = {r['bmp_quant_step_m']/4:.4f}) "
          f"| empty {r['empty_cells']} | flipud MAE {orient['flipud']:.3f}")

print('\nshipped grid.json keys:', out[ARENAS[0]]['shipped_grid_json_keys'])
print("shipped grid.json['row0'] =", json.load(open(EXP / f'sensor_v2/grids/{ARENAS[0]}/grid.json')).get('row0'))
json.dump(out, open(HERE / 'verify_geometry.json', 'w'), indent=1)
print('wrote', HERE / 'verify_geometry.json')
