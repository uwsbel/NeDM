"""Control grid: identical to scripts/sensor_map_v2.grid_from_capture EXCEPT that each pixel keeps the v1
flat-ground placement x = ray_x * H, y = ray_y * H instead of x = ray_x * axial, y = ray_y * axial.

Purpose: v1 -> v2 changes two things at once (the geometry fix AND a re-rasterisation onto a 0.15625 m world grid
that also drops channels 5-9 from their native 1024-px resolution). Scoring this control isolates them:
  v1 (image lookup)  vs  flat grid  =  resampling/pipeline only
  flat grid          vs  v2 grid    =  the geometry correction only
Written to this diagnostic directory only.
"""
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path('/home/harry/NeDM-traverse_mppi')
EXP = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909'
HERE = Path(__file__).resolve().parent
MPP = 80.0 / 512
HALF = 40.0
N = int(round(2 * HALF / MPP))


def flat_grid_from_capture(mapdir):
    cam = json.load(open(Path(mapdir) / 'observation.json'))['camera']
    with np.load(Path(mapdir) / 'observation.npz') as o:
        depth = o['depth_m'].astype(np.float64); rgb = o['rgb'].astype(np.float64) / 255.
    h, w = depth.shape
    H = float(cam['cam_height_m']); f = (w / 2) / math.tan(float(cam['hfov_rad']) / 2)
    v, u = np.mgrid[0:h, 0:w]
    ray_x = (u - (w - 1) / 2) / f
    ray_y = -(v - (h - 1) / 2) / f
    sec = np.sqrt(1 + ray_x ** 2 + ray_y ** 2)
    ok = np.isfinite(depth) & (depth > 0) & (depth < float(cam['max_depth_m']) - 1e-6)
    axial = np.where(ok, depth / sec, np.nan)
    x = ray_x * H; y = ray_y * H              # <-- the v1 flat-ground assumption, everything else unchanged
    z = H - axial
    inside = ok & (np.abs(x) < HALF) & (np.abs(y) < HALF)
    xi = np.clip(((x[inside] + HALF) / MPP).astype(int), 0, N - 1)
    yi = np.clip(((y[inside] + HALF) / MPP).astype(int), 0, N - 1)
    flat = yi * N + xi
    cnt = np.bincount(flat, minlength=N * N).astype(np.float32)
    acc = lambda q: np.bincount(flat, weights=q[inside], minlength=N * N)
    with np.errstate(invalid='ignore', divide='ignore'):
        zg = (acc(z) / cnt).reshape(N, N)
        rng = (acc(depth) / cnt).reshape(N, N)
        secg = (acc(sec) / cnt).reshape(N, N)
        col = np.stack([(acc(rgb[..., c]) / cnt).reshape(N, N) for c in range(3)])
    return dict(z=zg.astype(np.float32), range_m=rng.astype(np.float32), sec=secg.astype(np.float32),
                rgb=col.astype(np.float32), cover=cnt.reshape(N, N), camera=cam,
                meta=dict(mpp=MPP, half_extent_m=HALF, n=N, camera_height_m=H, focal_px=f,
                          row0='grid row index increases with +y',
                          fill='none: empty cells are invalid',
                          source='CONTROL: v1 flat-ground pixel placement rasterised on the v2 grid'))


def main():
    outroot = HERE / 'flat_grids'
    for d in sorted((EXP / 'sensor_v1/maps').iterdir()):
        if not (d / 'observation.npz').exists():
            continue
        g = flat_grid_from_capture(d)
        out = outroot / d.name; out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'grid.npz', z=g['z'], range_m=g['range_m'], sec=g['sec'], rgb=g['rgb'], cover=g['cover'])
        json.dump({'camera': g['camera'], **g['meta'],
                   'coverage_fraction': float((g['cover'] > 0).mean()),
                   'mean_pixels_per_cell': float(g['cover'][g['cover'] > 0].mean())}, open(out / 'grid.json', 'w'), indent=1)
        print(f"{d.name}: coverage {100 * (g['cover'] > 0).mean():.2f}%  mean px/cell {g['cover'][g['cover'] > 0].mean():.2f}")


if __name__ == '__main__':
    main()
