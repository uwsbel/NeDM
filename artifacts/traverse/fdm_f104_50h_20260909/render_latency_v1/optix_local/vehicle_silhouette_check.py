"""RGB-vs-depth registration of the vehicle silhouette at 1024 (OptiX --with-rgb frame0 and the AMD lavapipe frame).

Depth silhouette: pixels near the vehicle whose back-projected height is > 0.5 m above the heightmap.
RGB silhouette: pixels near the vehicle whose colour is far from the local terrain colour (Otsu threshold on the
distance to the median colour of a ring 5-6 m from the vehicle) AND brighter than it (excludes the cast shadow). Reports IoU and the integer shift maximising overlap.
"""
import json, math, sys
from pathlib import Path
import numpy as np
REPO = Path('/home/harry/NeDM-traverse_mppi'); sys.path.insert(0, str(REPO / 'src'))
from nedm.traverse.terrain import TerrainMap
HERE = Path(__file__).resolve().parent
AMD = REPO / 'artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/vehicle_frames/g216_v2_group_0000'
obs = json.loads((AMD / 'observation.json').read_text()); cam = obs['camera']
tmap = TerrainMap.from_dir(REPO / obs['arena'])
W = 1024; f = (W / 2) / math.tan(cam['hfov_rad'] / 2); c0 = (W - 1) / 2
v, u = np.mgrid[0:W, 0:W]; rx = (u - c0) / f; ry = -(v - c0) / f; sec = np.sqrt(1 + rx ** 2 + ry ** 2)

def otsu(x):
    h, e = np.histogram(x, 256); m = (e[:-1] + e[1:]) / 2; best = (-1, 0)
    for i in range(1, 256):
        w0, w1 = h[:i].sum(), h[i:].sum()
        if w0 == 0 or w1 == 0: continue
        m0 = (h[:i] * m[:i]).sum() / w0; m1 = (h[i:] * m[i:]).sum() / w1
        s = w0 * w1 * (m0 - m1) ** 2
        if s > best[0]: best = (s, m[i])
    return best[1]

def check(depth, rgb, pose):
    ax = depth / sec; x = rx * ax; y = ry * ax; z = cam['cam_height_m'] - ax
    dist = np.hypot(x - pose[0], y - pose[1])
    above = z - tmap.height(np.clip(x, -40, 40) * 511 / 512, np.clip(y, -40, 40) * 511 / 512)
    near = dist < 4.5
    dmask = near & (above > 0.5)
    ring = (dist > 5) & (dist < 6)
    ref = np.median(rgb[ring].astype(float), 0)
    cd = np.sqrt(((rgb.astype(float) - ref) ** 2).sum(-1))
    t = otsu(cd[near]); rmask = near & (cd > t) & (rgb.astype(float).sum(-1) > ref.sum())  # brighter than terrain: drops the shadow
    iou = (dmask & rmask).sum() / (dmask | rmask).sum()
    best = None
    for dv in range(-4, 5):
        for du in range(-4, 5):
            s = (np.roll(np.roll(dmask, dv, 0), du, 1) & rmask).sum() / (np.roll(np.roll(dmask, dv, 0), du, 1) | rmask).sum()
            if best is None or s > best[0]: best = (float(s), du, dv)
    cen = lambda m: [float(u[m].mean()), float(v[m].mean())]
    return {'depth_silhouette_px': int(dmask.sum()), 'rgb_silhouette_px': int(rmask.sum()), 'otsu_colour_threshold': float(t),
            'iou_at_zero_shift': float(iou), 'best_shift_du_dv_px': [best[1], best[2]], 'iou_at_best_shift': best[0],
            'centroid_depth_uv': cen(dmask), 'centroid_rgb_uv': cen(rmask),
            'centroid_rgb_minus_depth_px': list(np.subtract(cen(rmask), cen(dmask)).round(3))}

o = np.load(HERE / 'optix5090_rgbd_1024_frame0.npz'); a = np.load(AMD / 'observation.npz')
rep = {'optix_rgbd': check(o['depth_m'].astype(float), o['rgb'], o['pose'].tolist()),
       'amd_lavapipe': check(a['depth_m'].astype(float), a['rgb'], obs['measured_pose_xy_yaw'])}
(HERE / 'vehicle_silhouette_check.json').write_text(json.dumps(rep, indent=1) + '\n')
print(json.dumps(rep, indent=1))
