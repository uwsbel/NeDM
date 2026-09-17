"""How long does the planner's back-projection (scripts/sensor_map_v2.grid_from_arrays: 1024x1024 depth frame -> 512x512
metric grid) take on this CPU, and where inside it does the time go?

  PYTHONPATH=/home/harry/chrono/build/bin /usr/bin/python3.12 time_backprojection.py --out backprojection_timing.json
"""
import argparse, json, math, os, platform, sys, time
from pathlib import Path
import numpy as np

REPO = Path('/home/harry/NeDM-traverse_mppi')
sys.path.insert(0, str(REPO / 'scripts'))
import sensor_map_v2 as M  # noqa: E402

HERE = Path(__file__).resolve().parent
AMD = REPO / 'artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/vehicle_frames/g216_v2_group_0000'


def cpu_model():
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def breakdown(depth, rgb, cam):
    """grid_from_arrays with timers between its stages (same operations, same order)."""
    T = {}; t = time.perf_counter()
    def lap(k):
        nonlocal t
        n = time.perf_counter(); T[k] = n - t; t = n
    depth = np.asarray(depth, np.float64); rgb = np.asarray(rgb, np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.
    lap('to_float64_and_rgb_scale')
    h, w = depth.shape
    H = float(cam['cam_height_m']); f = (w / 2) / math.tan(float(cam['hfov_rad']) / 2)
    v, u = np.mgrid[0:h, 0:w]
    ray_x = (u - (w - 1) / 2) / f; ray_y = -(v - (h - 1) / 2) / f
    sec = np.sqrt(1 + ray_x ** 2 + ray_y ** 2)
    lap('pixel_rays_mgrid_secant')
    ok = np.isfinite(depth) & (depth > 0) & (depth < float(cam['max_depth_m']) - 1e-6)
    axial = np.where(ok, depth / sec, np.nan)
    x = ray_x * axial; y = ray_y * axial; z = H - axial
    inside = ok & (np.abs(x) < M.HALF) & (np.abs(y) < M.HALF)
    lap('points_xyz_and_masks')
    xi = np.clip(((x[inside] + M.HALF) / M.MPP).astype(int), 0, M.N - 1)
    yi = np.clip(((y[inside] + M.HALF) / M.MPP).astype(int), 0, M.N - 1)
    flat = yi * M.N + xi
    cnt = np.bincount(flat, minlength=M.N * M.N).astype(np.float32)
    lap('cell_index_and_count')
    acc = lambda q: np.bincount(flat, weights=q[inside], minlength=M.N * M.N)
    with np.errstate(invalid='ignore', divide='ignore'):
        zg = (acc(z) / cnt).reshape(M.N, M.N); rng = (acc(depth) / cnt).reshape(M.N, M.N)
        secg = (acc(sec) / cnt).reshape(M.N, M.N)
        lap('accumulate_z_range_sec')
        col = np.stack([(acc(rgb[..., c]) / cnt).reshape(M.N, M.N) for c in range(3)])
        lap('accumulate_rgb_3ch')
    _ = (zg.astype(np.float32), rng.astype(np.float32), secg.astype(np.float32), col.astype(np.float32))
    lap('cast_float32')
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=30)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    obs = json.loads((AMD / 'observation.json').read_text()); cam = obs['camera']
    with np.load(AMD / 'observation.npz') as o:
        d_amd = o['depth_m']; rgb_amd = o['rgb']
    with np.load(HERE / 'optix5090_rgbd_1024_frame0.npz') as o:
        d_opx = o['depth_m']; rgb_opx = o['rgb']
    blank = np.zeros_like(rgb_amd)
    cases = {'amd_frame_with_rgb': (d_amd, rgb_amd), 'optix_frame_with_rgb': (d_opx, rgb_opx),
             'optix_frame_blank_rgb_(depth_only_runner)': (d_opx, blank)}
    res = {'host': platform.node(), 'cpu': cpu_model(), 'cpus': os.cpu_count(), 'numpy': np.__version__,
           'python': sys.version.split()[0], 'repeats': a.repeats, 'input_dtypes': {'depth': str(d_amd.dtype), 'rgb': str(rgb_amd.dtype)},
           'grid_from_arrays': {}}
    for name, (d, rgb) in cases.items():
        for _ in range(2):
            M.grid_from_arrays(d, rgb, cam)
        ts = []
        for _ in range(a.repeats):
            t0 = time.perf_counter(); M.grid_from_arrays(d, rgb, cam); ts.append(time.perf_counter() - t0)
        ts = np.asarray(ts)
        res['grid_from_arrays'][name] = {'median_s': float(np.median(ts)), 'min_s': float(ts.min()), 'max_s': float(ts.max()),
                                         'p90_s': float(np.percentile(ts, 90)), 'all_s': ts.tolist()}
        print(name, 'median %.4f s  min %.4f  max %.4f' % (np.median(ts), ts.min(), ts.max()), flush=True)
    bd = [breakdown(d_amd, rgb_amd, cam) for _ in range(a.repeats)]
    res['stage_breakdown_median_s_amd_frame_with_rgb'] = {k: float(np.median([b[k] for b in bd])) for k in bd[0]}
    print(json.dumps(res['stage_breakdown_median_s_amd_frame_with_rgb'], indent=1))
    Path(a.out).write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
