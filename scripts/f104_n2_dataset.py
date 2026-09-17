"""Night-2: label + per-station tensors for newly collected runs, runnable on the AMD cluster.

Self-contained (numpy only): copies the map sampling from build_f104_planner_dataset.py, the station tensor
from f104_night_dataset.py and the event labelling from f104_night_index.py -- with the CORRECTED throttle
column (action = [steering, throttle, braking]; night 1 read column 0).
Output keys match night_v1/station_ds.npz so the trainer is unchanged (old-model features are not rebuilt).
"""
import argparse, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

N_STATION, N_LATERAL, HALF_WIDTH_M = 96, 32, 6.0
DT, SETTLE_S = 0.05, 1.0
G = {}


def init_map(root):
    cam = json.load(open(root + '/static_map_v1/observation.json'))['camera']
    obs = np.load(root + '/static_map_v1/observation.npz')
    G['rgbd'] = obs['rgbd'].astype(np.float32)
    G['elev_scale'] = float(cam['elevation_scale_m'])
    G['npx'] = G['rgbd'].shape[1]
    G['mpp'] = (2 * cam['cam_height_m'] * np.tan(cam['hfov_rad'] / 2)) / G['npx']
    G['ctr'] = (G['npx'] - 1) / 2.0


def sample_map(x, y):
    row = G['ctr'] - np.asarray(y) / G['mpp']; col = G['ctr'] + np.asarray(x) / G['mpp']
    n = G['npx']; R = G['rgbd']
    r0 = np.clip(np.floor(row).astype(int), 0, n - 2); c0 = np.clip(np.floor(col).astype(int), 0, n - 2)
    fr = np.clip(row - r0, 0, 1)[None]; fc = np.clip(col - c0, 0, 1)[None]
    p00, p01, p10, p11 = R[:, r0, c0], R[:, r0, c0 + 1], R[:, r0 + 1, c0], R[:, r0 + 1, c0 + 1]
    valid = np.min(np.stack([p00[3], p01[3], p10[3], p11[3]]), axis=0) > -1.999
    return p00 * (1 - fr) * (1 - fc) + p01 * (1 - fr) * fc + p10 * fr * (1 - fc) + p11 * fr * fc, valid


def resample_route(wp, stations, n):
    s = np.asarray(stations, float)
    if s.size != len(wp) or not np.all(np.diff(s) > 0):
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
    grid = np.linspace(s[0], s[-1], n)
    return np.stack([np.interp(grid, s, wp[:, 0]), np.interp(grid, s, wp[:, 1])], 1), grid


def station_tensor(wp, sp, st):
    pts, grid = resample_route(wp, st, N_STATION)
    d = np.gradient(pts, axis=0); tn = np.linalg.norm(d, axis=1, keepdims=True); tn[tn < 1e-9] = 1e-9
    tang = d / tn; norm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    off = np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, N_LATERAL)
    gx = pts[:, 0:1] + norm[:, 0:1] * off[None, :]; gy = pts[:, 1:2] + norm[:, 1:2] * off[None, :]
    patch, valid = sample_map(gx, gy)
    elev = np.where(valid, patch[3] * G['elev_scale'], np.nan)
    e0 = elev[0, N_LATERAL // 2] if np.isfinite(elev[0, N_LATERAL // 2]) else np.nanmean(elev[0])
    fill = np.where(np.isfinite(elev), elev, e0)
    ds = max(float(grid[-1] - grid[0]) / (N_STATION - 1), 1e-3); dl = 2 * HALF_WIDTH_M / (N_LATERAL - 1)
    ga = np.clip(np.gradient(fill, ds, axis=0), -2, 2); gc = np.clip(np.gradient(fill, dl, axis=1), -2, 2)
    s_ref = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
    v = np.interp(np.linspace(0, s_ref[-1], N_STATION), s_ref, sp)
    X = np.stack([fill - e0, ga, gc, np.repeat(v[:, None], N_LATERAL, 1), valid.astype(float)])
    return X.astype(np.float32), float(grid[-1] - grid[0])


def first_run(mask, need, start):
    c = 0
    for i in range(start, len(mask)):
        c = c + 1 if mask[i] else 0
        if c >= need: return i - need + 1
    return None


def project(pose, wp):
    seg = np.diff(wp, axis=0); L = np.linalg.norm(seg, axis=1); L[L < 1e-9] = 1e-9
    s0 = np.r_[0.0, np.cumsum(L)]
    rel = pose[:, None, :] - wp[None, :-1, :]
    t = np.clip((rel * seg[None]).sum(-1) / (L ** 2)[None], 0, 1)
    proj = wp[None, :-1, :] + t[..., None] * seg[None]
    d = np.linalg.norm(pose[:, None, :] - proj, axis=-1)
    k = np.argmin(d, 1)
    return s0[k] + t[np.arange(len(pose)), k] * L[k], d[np.arange(len(pose)), k], float(s0[-1])


def load_route(src):
    if src.endswith('.npz'):
        z = np.load(src, allow_pickle=True)
        return np.asarray(z['reference_waypoints'], float), np.asarray(z['reference_speeds'], float), \
               np.asarray(z['reference_stations'], float)
    r = json.load(open(src))
    return np.asarray(r['waypoints'], float), np.asarray(r['speeds'], float), np.asarray(r['stations'], float)


def one(item):
    d, root, source = item
    try:
        z = np.load(d + '/trajectory.npz'); o = json.load(open(d + '/outcome.json'))
        wp, sp, st = load_route(d + '/command_reference.npz')
        c = json.load(open(d + '/case.json'))
        anc = np.asarray(np.load(d + '/anchor_state.npz', allow_pickle=True)['state'], np.float32)
    except Exception as e:
        return None
    vx = z['state'][:, 0].astype(float); thr = z['action'][:, 1].astype(float)   # column 1 = throttle
    pose = z['pose'][:, :2].astype(float); n = len(vx); s0 = int(SETTLE_S / DT)
    fail = o['status'] != 'goal_reached'
    back = (vx < -0.10) & (thr > 0.3)
    rb = first_run(back | (vx < -0.30), 1, s0)
    ns = first_run((np.abs(vx) < 0.3) & (thr > 0.3), 20, s0)
    cands = [x for x in (rb, ns) if x is not None]
    ev = min(cands) if cands else (n - 1 if fail else None)
    back_s = float(back[s0:].sum() * DT); min_vx = float(vx[s0:].min()) if n > s0 else float(vx.min())
    clean = (not fail) and back_s < 0.05 and min_vx > -0.30
    if clean: ev = None
    s, dev, L = project(pose, wp)
    hwm = float(s.max()) if ev is None else float(s[:ev + 1].max())
    frac = hwm / max(L, 1e-6)
    X, route_len = station_tensor(wp, sp, st)
    gxy = np.asarray(c['goal_xy'], np.float32); sxy = np.asarray(c['layout']['start_xy'], np.float32)
    rel = gxy - sxy
    ctx = np.concatenate([anc, [rel[0], rel[1], float(np.linalg.norm(rel)), float(c['layout']['start_yaw']), route_len]]).astype(np.float32)
    rid = os.path.basename(d)
    prof = int(rid.split('_route_')[1]) % 4 if '_route_' in rid else -1
    return dict(X=X, ctx=ctx, id=rid, group=c['id'], split=c['split'], source=source, profile=prof,
                fail=int(fail), unsafe=int(not clean), status=o['status'],
                event_idx=(-1 if ev is None else int(np.clip(round(frac * (N_STATION - 1)), 0, N_STATION - 1))),
                route_len=route_len, min_vx=min_vx, back_s=back_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--runs', nargs='+', required=True, help='glob:source pairs, e.g. /path/production_v3/runs/*:designed')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    init_map(a.root)
    items = []
    for spec in a.runs:
        pat, source = spec.rsplit(':', 1)
        ds = sorted(glob.glob(pat))
        print(f'{len(ds)} runs from {pat} as {source}', flush=True)
        items += [(d, a.root, source) for d in ds]
    res = []
    with ProcessPoolExecutor(a.workers, initializer=init_map, initargs=(a.root,)) as ex:
        for k, r in enumerate(ex.map(one, items, chunksize=16)):
            if r is not None: res.append(r)
            if (k + 1) % 2000 == 0: print(f'  {k+1}/{len(items)}', flush=True)
    print(f'labelled {len(res)}/{len(items)}', flush=True)
    A = lambda k, dt: np.array([r[k] for r in res], dtype=dt)
    np.savez(a.out, X=np.stack([r['X'] for r in res]).astype(np.float16), ctx=np.stack([r['ctx'] for r in res]),
             id=A('id', object), group=A('group', object), split=A('split', object), source=A('source', object),
             profile=A('profile', np.int8), fail=A('fail', np.int8), unsafe=A('unsafe', np.int8),
             status=A('status', object), event_idx=A('event_idx', np.int16), route_len=A('route_len', np.float32),
             min_vx=A('min_vx', np.float32), back_s=A('back_s', np.float32))
    print('wrote', a.out, round(os.path.getsize(a.out) / 1e6), 'MB')
    print('unsafe rate %.3f   fail rate %.3f' % (A('unsafe', float).mean(), A('fail', float).mean()))


if __name__ == '__main__':
    main()
