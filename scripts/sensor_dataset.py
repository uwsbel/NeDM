"""Sensor-channel corridor tensors for every route, sampled from one overhead RGB-D capture per arena.

Per route, the same 96 stations x 32 lateral samples (+-6 m) as the deployed model, ten channels:
  0 elev_rel   height from the depth image relative to the route start   (the current model input,
  1 grade      along-path grade of that height                             reproduced exactly as a check)
  2 cross      cross-slope of that height
  3 speed      commanded speed at the station
  4 valid      corridor point inside the arena image
  5 depth_rel  RAW sensor depth (metres along the camera ray) minus the depth at the route start
  6 ray_sec1   known camera geometry at that pixel: sec(ray angle) - 1  (depth = (camera height - z) * sec)
  7-9 R, G, B  sensor colour in [0, 1]
Channels 5-9 never pass through a depth->height conversion or a finite-difference slope.
Numpy only; runs on the cluster next to the raw runs.

  f104 mode:   --ids ids.json --run-dirs run_dirs.txt --map maps/arena_f104_50h_v1 --labels labels.npz
  arena mode:  --runs 'glob' --map maps/<arena>   (labels computed with the corrected throttle column)
"""
import argparse, glob, json, math, os, sys
from multiprocessing import Pool
import numpy as np

N_STATION, N_LATERAL, HALF_WIDTH_M = 96, 32, 6.0
G = {}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def init_map(mapdir):
    cam = json.load(open(mapdir + '/observation.json'))['camera']
    z = np.load(mapdir + '/observation.npz')
    G['rgbd'] = z['rgbd'].astype(np.float32); G['depth'] = z['depth_m'].astype(np.float32); G['rgb'] = z['rgb'].astype(np.float32) / 255.
    G['max_depth'] = float(cam['max_depth_m']); G['cam_h'] = float(cam['cam_height_m'])
    G['elev_scale'] = float(cam['elevation_scale_m'])
    ground = 2 * cam['cam_height_m'] * np.tan(cam['hfov_rad'] / 2)
    G['npx'] = G['rgbd'].shape[1]; G['mpp'] = ground / G['npx']; G['ctr'] = (G['npx'] - 1) / 2.0
    W = G['depth'].shape[1]; G['W'] = W; G['mppW'] = ground / W; G['ctrW'] = (W - 1) / 2.0
    G['f'] = (W / 2) / math.tan(float(cam['hfov_rad']) / 2)


def bilinear(img, row, col):
    n0, n1 = img.shape[-2], img.shape[-1]
    r0 = np.clip(np.floor(row).astype(int), 0, n0 - 2); c0 = np.clip(np.floor(col).astype(int), 0, n1 - 2)
    fr = np.clip(row - r0, 0, 1); fc = np.clip(col - c0, 0, 1)
    p00, p01, p10, p11 = img[..., r0, c0], img[..., r0, c0 + 1], img[..., r0 + 1, c0], img[..., r0 + 1, c0 + 1]
    return p00 * (1 - fr) * (1 - fc) + p01 * (1 - fr) * fc + p10 * fr * (1 - fc) + p11 * fr * fc


def corridor(wp, stations):
    s = np.asarray(stations, float)
    if s.size != len(wp) or not np.all(np.diff(s) > 0):
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
    grid = np.linspace(s[0], s[-1], N_STATION)
    pts = np.stack([np.interp(grid, s, wp[:, 0]), np.interp(grid, s, wp[:, 1])], 1)
    d = np.gradient(pts, axis=0); tn = np.linalg.norm(d, axis=1, keepdims=True); tn[tn < 1e-9] = 1e-9
    tang = d / tn; norm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    off = np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, N_LATERAL)
    gx = pts[:, 0:1] + norm[:, 0:1] * off[None, :]; gy = pts[:, 1:2] + norm[:, 1:2] * off[None, :]
    return gx, gy, grid


def tensor10(wp, sp, st):
    wp = np.asarray(wp, float); sp = np.asarray(sp, float)
    gx, gy, grid = corridor(wp, st)
    # --- channels 0-4: exactly f104_n2_dataset.station_tensor (512 px encoded elevation) ---
    row = G['ctr'] - gy / G['mpp']; col = G['ctr'] + gx / G['mpp']
    n = G['npx']; R = G['rgbd']
    r0 = np.clip(np.floor(row).astype(int), 0, n - 2); c0 = np.clip(np.floor(col).astype(int), 0, n - 2)
    fr = np.clip(row - r0, 0, 1)[None]; fc = np.clip(col - c0, 0, 1)[None]
    p00, p01, p10, p11 = R[:, r0, c0], R[:, r0, c0 + 1], R[:, r0 + 1, c0], R[:, r0 + 1, c0 + 1]
    valid = np.min(np.stack([p00[3], p01[3], p10[3], p11[3]]), axis=0) > -1.999
    patch = p00 * (1 - fr) * (1 - fc) + p01 * (1 - fr) * fc + p10 * fr * (1 - fc) + p11 * fr * fc
    elev = np.where(valid, patch[3] * G['elev_scale'], np.nan)
    e0 = elev[0, N_LATERAL // 2] if np.isfinite(elev[0, N_LATERAL // 2]) else np.nanmean(elev[0])
    fill = np.where(np.isfinite(elev), elev, e0)
    ds = max(float(grid[-1] - grid[0]) / (N_STATION - 1), 1e-3); dl = 2 * HALF_WIDTH_M / (N_LATERAL - 1)
    ga = np.clip(np.gradient(fill, ds, axis=0), -2, 2); gc = np.clip(np.gradient(fill, dl, axis=1), -2, 2)
    s_ref = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
    v = np.interp(np.linspace(0, s_ref[-1], N_STATION), s_ref, sp)
    # --- channels 5-9: raw sensor, full resolution ---
    rowW = G['ctrW'] - gy / G['mppW']; colW = G['ctrW'] + gx / G['mppW']
    depth = bilinear(G['depth'], rowW, colW)
    dvalid = valid & (depth < G['max_depth'] - 1e-3)
    d0 = depth[0, N_LATERAL // 2] if dvalid[0, N_LATERAL // 2] else (np.nanmean(np.where(dvalid[0], depth[0], np.nan)) if dvalid[0].any() else G['cam_h'])
    depth_rel = np.where(dvalid, depth - d0, 0.0)
    sec1 = np.sqrt(1 + ((colW - G['ctrW']) / G['f']) ** 2 + ((rowW - G['ctrW']) / G['f']) ** 2) - 1
    rgb = np.stack([bilinear(G['rgb'][..., c], rowW, colW) for c in range(3)])
    X = np.stack([fill - e0, ga, gc, np.repeat(v[:, None], N_LATERAL, 1), valid.astype(float),
                  depth_rel, sec1, rgb[0], rgb[1], rgb[2]])
    return X.astype(np.float32), float(grid[-1] - grid[0])


def load_route(d):
    z = np.load(d + '/command_reference.npz', allow_pickle=True)
    return (np.asarray(z['reference_waypoints'], float), np.asarray(z['reference_speeds'], float),
            np.asarray(z['reference_stations'], float))


def f104_one(d):
    try:
        wp, sp, st = load_route(d)
        X, L = tensor10(wp, sp, st)
        return X.astype(np.float16), L
    except Exception as e:
        return None, str(e)


def arena_one(d):
    import f104_n2_dataset as DS
    DS.G.update(rgbd=G['rgbd'], elev_scale=G['elev_scale'], npx=G['npx'], mpp=G['mpp'], ctr=G['ctr'])
    r = DS.one((d, None, 'designed' if '/data/' in d else 'gen_test_arm'))
    if r is None:
        return None
    wp, sp, st = load_route(d)
    X, L = tensor10(wp, sp, st)
    r['X'] = X.astype(np.float16); r['arena'] = os.path.basename(d).split('_')[0]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--ids'); ap.add_argument('--run-dirs'); ap.add_argument('--labels')
    ap.add_argument('--runs', nargs='*', default=[]); ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    init_map(a.map)
    if a.ids:
        ids = json.load(open(a.ids))['id']
        by = {os.path.basename(l.strip()): l.strip() for l in open(a.run_dirs)}
        dirs = [os.path.join(os.path.dirname(os.path.abspath(a.run_dirs)), '..', by[i]) for i in ids]
        with Pool(a.workers, initializer=init_map, initargs=(a.map,)) as pool:
            res = pool.map(f104_one, dirs, chunksize=64)
        bad = [(i, r[1]) for i, r in zip(ids, res) if r[0] is None]
        if bad:
            raise RuntimeError(f'{len(bad)} routes failed, e.g. {bad[:3]}')
        X = np.stack([r[0] for r in res]); L = np.array([r[1] for r in res], np.float32)
        lab = dict(np.load(a.labels, allow_pickle=True))
        assert list(lab['id'].astype(str)) == ids
        lab.pop('X', None)
        np.savez(a.out, X10=X, route_len10=L, **lab)
        print('wrote', a.out, X.shape)
    else:
        dirs = sorted(d for pat in a.runs for d in glob.glob(pat) if os.path.exists(d + '/outcome.json'))
        with Pool(a.workers, initializer=init_map, initargs=(a.map,)) as pool:
            rows = [r for r in pool.map(arena_one, dirs, chunksize=32) if r is not None]
        keys = ['ctx', 'id', 'group', 'split', 'source', 'profile', 'fail', 'unsafe', 'status', 'event_idx', 'route_len', 'min_vx', 'back_s', 'arena']
        out = {k: np.array([r[k] for r in rows]) for k in keys}
        out['ctx'] = np.stack([r['ctx'] for r in rows]); out['X10'] = np.stack([r['X'] for r in rows])
        np.savez(a.out, **out)
        print('wrote', a.out, out['X10'].shape, 'from', len(dirs), 'run dirs')


if __name__ == '__main__':
    main()
