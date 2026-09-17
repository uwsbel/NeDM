"""Do the local Vulkan-RT depth frames (RTX 5090 via the NVIDIA Vulkan driver, and lavapipe on this CPU) match the AMD
lavapipe frame of the same case?

Same method as ../optix_local/compare_depth_vs_amd.py: 1024x1024 frame0 from the harness (after the 0.8 s settle) vs the
stored AMD frame; terrain pixels = valid in both frames and farther than 6 m (in XY, after back-projection with the
correct pinhole focal length) from BOTH vehicle poses. Every pair among the given frames is compared, plus the
back-projected planner grid (scripts/sensor_map_v2.grid_from_arrays) against the AMD grid.

  python compare_vulkan_depth.py --frame vk5090=vk5090_depth_1024_frame0.npz --frame lvp16=... --out compare.json
"""
import argparse, itertools, json, math, sys
from pathlib import Path
import numpy as np

REPO = Path('/home/harry/NeDM-traverse_mppi')
sys.path.insert(0, str(REPO / 'src')); sys.path.insert(0, str(REPO / 'scripts'))
import sensor_map_v2 as M  # noqa: E402

AMD = REPO / 'artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/vehicle_frames/g216_v2_group_0000'
EXCL_M = 6.0


def stats(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if a.size == 0:
        return {'n': 0}
    b = np.abs(a)
    return {'n': int(a.size), 'mean_signed': float(a.mean()), 'median_abs': float(np.median(b)),
            'p99_abs': float(np.percentile(b, 99)), 'max_abs': float(b.max()),
            'n_gt_0.05m': int((b > .05).sum()), 'n_gt_0.5m': int((b > .5).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', action='append', required=True, help='name=path to a harness frame0 npz')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    obs = json.loads((AMD / 'observation.json').read_text()); camcfg = obs['camera']
    frames = {}
    with np.load(AMD / 'observation.npz') as o:
        frames['amd_lavapipe'] = (o['depth_m'].astype(np.float64), list(obs['measured_pose_xy_yaw']))
    for spec in a.frame:
        name, path = spec.split('=', 1)
        with np.load(path) as o:
            frames[name] = (o['depth_m'].astype(np.float64), o['pose'].tolist())
    h, w = frames['amd_lavapipe'][0].shape
    f = (w / 2) / math.tan(float(camcfg['hfov_rad']) / 2)
    H, maxd = float(camcfg['cam_height_m']), float(camcfg['max_depth_m'])
    v, u = np.mgrid[0:h, 0:w]
    rx = (u - (w - 1) / 2) / f; ry = -(v - (h - 1) / 2) / f; sec = np.sqrt(1 + rx ** 2 + ry ** 2)

    def valid(d):
        return np.isfinite(d) & (d > 0) & (d < maxd - 1e-6)

    def xy(d):
        ax = np.where(valid(d), d / sec, np.nan)
        return np.nan_to_num(rx * ax, nan=1e9), np.nan_to_num(ry * ax, nan=1e9)

    rep = {'amd_frame': str(AMD / 'observation.npz'), 'exclusion_radius_m': EXCL_M, 'focal_px': f,
           'frames': {k: {'shape': list(d.shape), 'pose_xy_yaw': p, 'valid_fraction': float(valid(d).mean())}
                      for k, (d, p) in frames.items()},
           'pairs': {}, 'grid_vs_amd': {}}
    for (na, (da, pa)), (nb, (db, pb)) in itertools.combinations(frames.items(), 2):
        xa, ya = xy(da); xb, yb = xy(db)
        near = np.zeros_like(da, bool)
        for x, y in ((xa, ya), (xb, yb)):
            for p in (pa, pb):
                near |= np.hypot(x - p[0], y - p[1]) < EXCL_M
        terr = valid(da) & valid(db) & ~near
        rep['pairs'][f'{nb}_minus_{na}'] = {
            'pose_xy_difference_m': float(math.hypot(pa[0] - pb[0], pa[1] - pb[1])),
            'pose_yaw_difference_rad': float(pb[2] - pa[2]),
            'valid_mask_disagreement_pixels': int((valid(da) != valid(db)).sum()),
            'terrain_pixels': int(terr.sum()), 'excluded_near_vehicle_pixels': int((valid(da) & valid(db) & near).sum()),
            'depth_terrain_m': stats((db - da)[terr]),
            'bit_identical_terrain_fraction': float((db[terr] == da[terr]).mean()),
            'depth_all_pixels_identical': bool(np.array_equal(da, db))}
    # planner grid
    n = M.N; cc = -M.HALF + (np.arange(n) + .5) * M.MPP; gx, gy = np.meshgrid(cc, cc)
    da, pa = frames['amd_lavapipe']
    ga = M.grid_from_arrays(da, np.zeros((h, w, 3), np.uint8), camcfg)
    for name, (d, p) in frames.items():
        if name == 'amd_lavapipe':
            continue
        g = M.grid_from_arrays(d, np.zeros((h, w, 3), np.uint8), camcfg)
        gex = (np.hypot(gx - pa[0], gy - pa[1]) < EXCL_M) | (np.hypot(gx - p[0], gy - p[1]) < EXCL_M)
        c1, c0 = g['cover'] > 0, ga['cover'] > 0
        both = c1 & c0 & ~gex
        rep['grid_vs_amd'][name] = {'coverage': float(c1.mean()), 'coverage_amd': float(c0.mean()),
                                    'coverage_disagreement_cells': int((c1 != c0).sum()),
                                    'z_minus_amd_m': stats((g['z'] - ga['z'])[both])}
    Path(a.out).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
