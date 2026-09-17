"""TASK C sanity check: how accurate are the v1 and v2 height read-outs at a world point?

Simulator heights (nedm.traverse.terrain.TerrainMap) are used ONLY as an evaluation reference here; neither
sampler sees them. Two point sets: a dense lattice over the arena, and the corridor points of a sample of the
actual test-2 candidate routes.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path('/home/harry/NeDM-traverse_mppi')
EXP = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(HERE))
from nedm.traverse.terrain import TerrainMap
from corridor_v2 import V2Map

ARENA_DIR = {'f104': 'arena_f104_50h_v1', 'g203': 'arena_g203', 'g216': 'arena_g216',
             'g217': 'arena_g217', 'g228': 'arena_g228', 'g231': 'arena_g231'}


def v1_height(mapdir, x, y):
    """Exactly f104_n2_dataset.sample_map channel 3 (what the deployed corridor calls 'elevation at (x, y)')."""
    cam = json.load(open(Path(mapdir) / 'observation.json'))['camera']
    R = np.load(Path(mapdir) / 'observation.npz')['rgbd'].astype(np.float32)
    npx = R.shape[1]; mpp = (2 * cam['cam_height_m'] * np.tan(cam['hfov_rad'] / 2)) / npx; ctr = (npx - 1) / 2.0
    row = ctr - np.asarray(y) / mpp; col = ctr + np.asarray(x) / mpp
    r0 = np.clip(np.floor(row).astype(int), 0, npx - 2); c0 = np.clip(np.floor(col).astype(int), 0, npx - 2)
    fr = np.clip(row - r0, 0, 1); fc = np.clip(col - c0, 0, 1)
    a = R[3]
    p00, p01, p10, p11 = a[r0, c0], a[r0, c0 + 1], a[r0 + 1, c0], a[r0 + 1, c0 + 1]
    valid = np.minimum(np.minimum(p00, p01), np.minimum(p10, p11)) > -1.999
    h = (p00 * (1 - fr) * (1 - fc) + p01 * (1 - fr) * fc + p10 * fr * (1 - fc) + p11 * fr * fc) * float(cam['elevation_scale_m'])
    return h, valid


def main():
    rows = {}
    for arena, ad in ARENA_DIR.items():
        tm = TerrainMap.from_dir(ROOT / 'assets/traverse' / ad)
        vm = V2Map(EXP / 'sensor_v2/grids' / ad)
        c = np.linspace(-38, 38, 400)
        X, Y = np.meshgrid(c, c)
        x, y = X.ravel(), Y.ravel()
        ref = tm.height(x, y)
        h1, ok1 = v1_height(EXP / 'sensor_v1/maps' / ad, x, y)
        s = vm.sample(x, y); h2, ok2 = s['z'], s['valid']
        m = ok1 & ok2
        e1 = np.abs(h1[m] - ref[m]); e2 = np.abs(h2[m] - ref[m])
        slope = np.degrees(np.arctan(np.hypot(*tm.gradient(x, y))))[m] if hasattr(tm, 'gradient') else None
        rad = np.hypot(x, y)[m]
        rows[arena] = dict(n=int(m.sum()),
                           v1_mae=float(e1.mean()), v1_p95=float(np.percentile(e1, 95)), v1_max=float(e1.max()),
                           v2_mae=float(e2.mean()), v2_p95=float(np.percentile(e2, 95)), v2_max=float(e2.max()),
                           v1_mae_outer=float(e1[rad > 25].mean()), v2_mae_outer=float(e2[rad > 25].mean()),
                           v1_mae_inner=float(e1[rad < 10].mean()), v2_mae_inner=float(e2[rad < 10].mean()))
        if slope is not None:
            hi = slope > np.percentile(slope, 90)
            rows[arena].update(v1_mae_steep=float(e1[hi].mean()), v2_mae_steep=float(e2[hi].mean()),
                               steep_slope_deg_thresh=float(np.percentile(slope, 90)))
        print(arena, json.dumps(rows[arena]), flush=True)
    json.dump(rows, open(HERE / 'geometry_check.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
