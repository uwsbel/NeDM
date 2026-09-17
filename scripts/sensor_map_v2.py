"""v2 sensor adapter: depth image -> 3D points (camera intrinsics) -> metric world grid. No authored heights.

Fixes the flat-ground correspondence of the v1 sampler (scripts/sensor_dataset.py, f104_n2_dataset.py), which placed a
pixel at ray_dir * camera_height instead of ray_dir * axial_range, displacing points by up to ~1.5 m on slopes
(review artifacts/reviews/sensor_input_20260915). Everything here comes from the capture and the camera model:

    ray  = ((u - cx)/f, -(v - cy)/f, -1)          pixel ray in world axes (image row 0 is +y)
    axial = range / sqrt(1 + ray_x^2 + ray_y^2)   vertical component of the measured ray range
    x, y  = ray_x * axial, ray_y * axial          <-- v1 used ray * camera_height here
    z     = camera_height - axial

Points are rasterised into a fixed metric grid (0.15625 m, +-40 m, i.e. the arena BMP's own resolution) holding, per
cell: mean height z, mean measured range, mean ray secant, mean colour, and the number of contributing pixels. A cell
with no pixel is marked invalid and left empty (no fallback). Absolute range and secant are kept as provenance and for
depth-side model variants: z = H - range/sec holds per pixel, and per cell to ~1e-5 m here (both are cell means, so it
is an approximation, not an identity). NOTE: this is NOT the reason v1's relative-depth arm struggled - an audit
(sensor_v2/diag/task_b_depth_information) showed v1 also carried enough information (the route-start secant is in the
same tensor, height recoverable to ~4 mm); the v2 case rests on correct placement, locality and conditioning.

  python scripts/sensor_map_v2.py --maps <dir of captures> --out <dir>
"""
import argparse, json, math
from pathlib import Path
import numpy as np

MPP = 80.0 / 512
HALF = 40.0
N = int(round(2 * HALF / MPP))


def grid_from_arrays(depth, rgb, cam, arena_size_m=2 * HALF):
    """Back-project one RGB-D frame into the metric world grid. Used both offline (grid_from_capture) and
    online, once per planning decision, by scripts/nav_online.py."""
    if abs(float(cam.get('depth_ray_scale', 1.0)) - 1.0) > 1e-9 or cam.get('depth_measurement') != 'Euclidean_ray_range_m':
        raise ValueError(f"unsupported depth convention: {cam.get('depth_measurement')} scale {cam.get('depth_ray_scale')}")
    if abs(float(arena_size_m) - 2 * HALF) > 1e-6:
        raise ValueError(f'capture covers a {arena_size_m} m arena but this grid is fixed to +-{HALF} m; set MPP/HALF/N')
    depth = np.asarray(depth, np.float64)
    rgb = np.asarray(rgb, np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.
    h, w = depth.shape
    H = float(cam['cam_height_m']); f = (w / 2) / math.tan(float(cam['hfov_rad']) / 2)
    v, u = np.mgrid[0:h, 0:w]
    ray_x = (u - (w - 1) / 2) / f
    ray_y = -(v - (h - 1) / 2) / f          # image row 0 is +y, as in the capture encoding
    sec = np.sqrt(1 + ray_x ** 2 + ray_y ** 2)
    ok = np.isfinite(depth) & (depth > 0) & (depth < float(cam['max_depth_m']) - 1e-6)
    axial = np.where(ok, depth / sec, np.nan)
    x = ray_x * axial; y = ray_y * axial; z = H - axial
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
    cover = cnt.reshape(N, N)
    return dict(z=zg.astype(np.float32), range_m=rng.astype(np.float32), sec=secg.astype(np.float32),
                rgb=col.astype(np.float32), cover=cover, camera=cam,
                meta=dict(mpp=MPP, half_extent_m=HALF, n=N, camera_height_m=H, focal_px=f,
                          source_image_row0='+y (capture convention)',
                          grid_index_order='grid[row, col] with row = y bin (increasing +y), col = x bin (increasing +x)',
                          fill='none: empty cells are invalid',
                          source='back-projected sensor pixels only; no authored heights'))


def grid_from_capture(mapdir):
    obs = json.load(open(Path(mapdir) / 'observation.json'))
    with np.load(Path(mapdir) / 'observation.npz') as o:
        depth, rgb = o['depth_m'], o['rgb']
    return grid_from_arrays(depth, rgb, obs['camera'],
                            float(obs.get('native_geometry', {}).get('length_m', 2 * HALF)))


def save(grid, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'grid.npz', z=grid['z'], range_m=grid['range_m'], sec=grid['sec'],
                        rgb=grid['rgb'], cover=grid['cover'])
    cov = grid['cover']
    json.dump({'camera': grid['camera'], **grid['meta'],
               'coverage_fraction': float((cov > 0).mean()),
               'pixels_per_cell': {'mean': float(cov[cov > 0].mean()), 'min': float(cov.min()),
                                   'p5': float(np.percentile(cov, 5)), 'p50': float(np.percentile(cov, 50)),
                                   'frac_single_pixel': float((cov == 1).mean())},
               'empty_cell_policy': 'z/range/sec are NaN and cover is 0; consumers must use the cover mask '
                                    '(scripts/sensor_dataset_v2.tensor12 marks such points invalid)'},
              open(out / 'grid.json', 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    for d in sorted(Path(a.maps).iterdir()):
        if not (d / 'observation.npz').exists():
            continue
        g = grid_from_capture(d); save(g, Path(a.out) / d.name)
        cov = (g['cover'] > 0).mean()
        print(f"{d.name}: coverage {100*cov:.2f}%  mean pixels/cell {g['cover'][g['cover']>0].mean():.2f}  "
              f"z range [{np.nanmin(g['z']):.2f}, {np.nanmax(g['z']):.2f}]", flush=True)


if __name__ == '__main__':
    main()
