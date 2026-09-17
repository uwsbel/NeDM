"""v2 corridor tensors sampled from the back-projected world grid (scripts/sensor_map_v2.py).

Same corridor geometry as v1 (96 stations x 32 lateral samples over +-6 m), but every channel is read from the metric
grid, so a corridor point at world (x, y) reads the sensor measurement that actually belongs to (x, y). Channels:
  0 z_rel      back-projected height minus height at the route start   (height arm)
  1 grade      along-path gradient of z
  2 cross      cross-path gradient of z
  3 speed      commanded speed at the station
  4 valid      grid cell covered by >= 1 sensor pixel
  5 range_abs  ABSOLUTE measured ray range at that cell (the v1 arm subtracted the route-start range and lost it)
  6 range_rel  range minus the route-start range (v1 depth arm, kept as the control)
  7 sec1       ray secant - 1 at that cell; with range_abs and sec, z = camera_height - range/sec exactly
  8-10 R,G,B   sensor colour
  11 cover1    pixels contributing to the cell / 8, clipped (sampling density)
Modes: --ids/--run-dirs/--labels (f104 rows, labels reused) or --runs glob (labels computed as in f104_n2_dataset).
"""
import argparse, glob, json, os, sys
from multiprocessing import Pool
import numpy as np

N_STATION, N_LATERAL, HALF_WIDTH_M = 96, 32, 6.0
CHANNELS = ['z_rel', 'grade', 'cross', 'speed', 'valid', 'range_abs', 'range_rel', 'sec1', 'R', 'G', 'B', 'cover1']
G = {}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def init_grid(griddir):
    z = np.load(os.path.join(griddir, 'grid.npz'))
    meta = json.load(open(os.path.join(griddir, 'grid.json')))
    G.update(z=z['z'], range_m=z['range_m'], sec=z['sec'], rgb=z['rgb'], cover=z['cover'],
             mpp=float(meta['mpp']), half=float(meta['half_extent_m']), n=int(meta['n']),
             cam_h=float(meta['camera_height_m']))


def set_grid(grid):
    """Same state as init_grid, from an in-memory grid (scripts/sensor_map_v2.grid_from_arrays): used online,
    once per planning decision, when the frame is rendered inside the running simulation."""
    m = grid['meta']
    G.update(z=grid['z'], range_m=grid['range_m'], sec=grid['sec'], rgb=grid['rgb'], cover=grid['cover'],
             mpp=float(m['mpp']), half=float(m['half_extent_m']), n=int(m['n']), cam_h=float(m['camera_height_m']))


def sample(img, x, y):
    """Bilinear sample of a world grid (cell centres at -half + (i+0.5)*mpp; row index is y, column index is x)."""
    fx = (x + G['half']) / G['mpp'] - 0.5; fy = (y + G['half']) / G['mpp'] - 0.5
    n = G['n']
    i0 = np.clip(np.floor(fy).astype(int), 0, n - 2); j0 = np.clip(np.floor(fx).astype(int), 0, n - 2)
    ty = np.clip(fy - i0, 0, 1); tx = np.clip(fx - j0, 0, 1)
    a = img[..., i0, j0]; b = img[..., i0, j0 + 1]; c = img[..., i0 + 1, j0]; d = img[..., i0 + 1, j0 + 1]
    return a * (1 - ty) * (1 - tx) + b * (1 - ty) * tx + c * ty * (1 - tx) + d * ty * tx


def corridor_points(wp, stations):
    s = np.asarray(stations, float)
    if s.size != len(wp) or not np.all(np.diff(s) > 0):
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
    grid = np.linspace(s[0], s[-1], N_STATION)
    pts = np.stack([np.interp(grid, s, wp[:, 0]), np.interp(grid, s, wp[:, 1])], 1)
    d = np.gradient(pts, axis=0); tn = np.linalg.norm(d, axis=1, keepdims=True); tn[tn < 1e-9] = 1e-9
    tang = d / tn; norm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    off = np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, N_LATERAL)
    return pts[:, 0:1] + norm[:, 0:1] * off[None, :], pts[:, 1:2] + norm[:, 1:2] * off[None, :], grid


def tensor12(wp, sp, st):
    wp = np.asarray(wp, float); sp = np.asarray(sp, float)
    gx, gy, grid = corridor_points(wp, st)
    inside = (np.abs(gx) < G['half'] - 1e-6) & (np.abs(gy) < G['half'] - 1e-6)
    cover = sample(G['cover'], gx, gy)
    zs = sample(G['z'], gx, gy)
    # a corridor point is valid only if every grid cell it interpolates from is covered; an empty cell yields NaN in
    # z/range/sec, and bilinear interpolation would spread that NaN into neighbours (coverage is 100% on the current
    # captures, so this branch has never fired - it is here for the first sparse capture).
    valid = inside & (cover > 0.999) & np.isfinite(zs)
    z = np.where(valid, zs, np.nan)
    z0 = z[0, N_LATERAL // 2] if np.isfinite(z[0, N_LATERAL // 2]) else np.nanmean(z[0])
    if not np.isfinite(z0):
        z0 = float(np.nanmean(z)) if np.isfinite(z).any() else 0.0
    zf = np.where(np.isfinite(z), z, z0)
    ds = max(float(grid[-1] - grid[0]) / (N_STATION - 1), 1e-3); dl = 2 * HALF_WIDTH_M / (N_LATERAL - 1)
    ga = np.clip(np.gradient(zf, ds, axis=0), -2, 2); gc = np.clip(np.gradient(zf, dl, axis=1), -2, 2)
    rng = sample(G['range_m'], gx, gy); sec = sample(G['sec'], gx, gy)
    c = N_LATERAL // 2
    # route-start reference taken from a VALID cell (previously read after masking, so an invalid start silently
    # became the camera height)
    if valid[0, c]:
        r0 = float(rng[0, c])
    elif valid[0].any():
        r0 = float(np.nanmean(np.where(valid[0], rng[0], np.nan)))
    else:
        r0 = float(np.nanmean(np.where(valid, rng, np.nan))) if valid.any() else G['cam_h']
    rng = np.where(valid, rng, G['cam_h']); sec = np.where(valid, sec, 1.0)
    rgb = sample(G['rgb'], gx, gy)
    s_ref = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
    v = np.interp(np.linspace(0, s_ref[-1], N_STATION), s_ref, sp)
    X = np.stack([zf - z0, ga, gc, np.repeat(v[:, None], N_LATERAL, 1), valid.astype(float),
                  rng, rng - r0, sec - 1.0, rgb[0], rgb[1], rgb[2], np.clip(cover, 0, 8) / 8.0])
    return X.astype(np.float32), float(grid[-1] - grid[0])


def load_route(d):
    z = np.load(d + '/command_reference.npz', allow_pickle=True)
    return (np.asarray(z['reference_waypoints'], float), np.asarray(z['reference_speeds'], float),
            np.asarray(z['reference_stations'], float))


def f104_one(d):
    try:
        X, L = tensor12(*load_route(d)); return X.astype(np.float16), L
    except Exception as e:
        return None, str(e)


def labels_only(d):
    """Episode labels/context exactly as f104_n2_dataset.one(), without its (v1) corridor tensor."""
    import json as _json
    z = np.load(d + '/trajectory.npz'); o = _json.load(open(d + '/outcome.json'))
    c = _json.load(open(d + '/case.json'))
    anc = np.asarray(np.load(d + '/anchor_state.npz', allow_pickle=True)['state'], np.float32)
    wp, sp, st = load_route(d)
    DT, S0 = 0.05, 20
    vx = z['state'][:, 0].astype(float); thr = z['action'][:, 1].astype(float)   # column 1 = throttle
    pose = z['pose'][:, :2].astype(float); n = len(vx)
    fail = o['status'] != 'goal_reached'
    back = (vx < -0.10) & (thr > 0.3)

    def first_run(mask, need, start):
        cnt = 0
        for i in range(start, len(mask)):
            cnt = cnt + 1 if mask[i] else 0
            if cnt >= need: return i - need + 1
        return None
    rb = first_run(back | (vx < -0.30), 1, S0); ns = first_run((np.abs(vx) < 0.3) & (thr > 0.3), 20, S0)
    cands = [x for x in (rb, ns) if x is not None]
    ev = min(cands) if cands else (n - 1 if fail else None)
    back_s = float(back[S0:].sum() * DT); min_vx = float(vx[S0:].min()) if n > S0 else float(vx.min())
    clean = (not fail) and back_s < 0.05 and min_vx > -0.30
    if clean: ev = None
    seg = np.diff(wp, axis=0); L = np.linalg.norm(seg, axis=1); L[L < 1e-9] = 1e-9
    s0 = np.r_[0.0, np.cumsum(L)]
    rel = pose[:, None, :] - wp[None, :-1, :]
    t = np.clip((rel * seg[None]).sum(-1) / (L ** 2)[None], 0, 1)
    proj = wp[None, :-1, :] + t[..., None] * seg[None]
    dist = np.linalg.norm(pose[:, None, :] - proj, axis=-1); k = np.argmin(dist, 1)
    sdist = s0[k] + t[np.arange(len(pose)), k] * L[k]
    hwm = float(sdist.max()) if ev is None else float(sdist[:ev + 1].max())
    frac = hwm / max(float(s0[-1]), 1e-6)
    gxy = np.asarray(c['goal_xy'], np.float32); sxy = np.asarray(c['layout']['start_xy'], np.float32); d_ = gxy - sxy
    ctx = np.concatenate([anc, [d_[0], d_[1], float(np.linalg.norm(d_)), float(c['layout']['start_yaw']), 0.0]]).astype(np.float32)
    rid = os.path.basename(d)
    prof = int(rid.split('_route_')[1]) % 4 if '_route_' in rid else -1
    return dict(ctx=ctx, id=rid, group=c['id'], split=c['split'], profile=prof, fail=int(fail), unsafe=int(not clean),
                status=o['status'], event_idx=(-1 if ev is None else int(np.clip(round(frac * (N_STATION - 1)), 0, N_STATION - 1))),
                min_vx=min_vx, back_s=back_s)


def arena_one(d):
    try:
        r = labels_only(d)
        X, L = tensor12(*load_route(d))
    except Exception:
        return None
    r['X'] = X.astype(np.float16); r['arena'] = os.path.basename(d).split('_')[0]
    r['route_len'] = L; r['ctx'][21] = L
    r['source'] = 'designed' if '/data/' in d else 'gen_test_arm'
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--ids'); ap.add_argument('--run-dirs'); ap.add_argument('--labels')
    ap.add_argument('--runs', nargs='*', default=[]); ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    init_grid(a.grid)
    if a.ids:
        ids = json.load(open(a.ids))['id']
        by = {os.path.basename(l.strip()): l.strip() for l in open(a.run_dirs)}
        with Pool(a.workers, initializer=init_grid, initargs=(a.grid,)) as pool:
            res = pool.map(f104_one, [by[i] for i in ids], chunksize=64)
        bad = [(i, r[1]) for i, r in zip(ids, res) if r[0] is None]
        if bad:
            raise RuntimeError(f'{len(bad)} routes failed, e.g. {bad[:3]}')
        lab = dict(np.load(a.labels, allow_pickle=True)); lab.pop('X', None)
        assert list(lab['id'].astype(str)) == ids
        np.savez(a.out, X12=np.stack([r[0] for r in res]), route_len12=np.array([r[1] for r in res], np.float32),
                 arena=np.array(['f104'] * len(ids)), channels=np.array(CHANNELS), **lab)
    else:
        dirs = sorted(d for pat in a.runs for d in glob.glob(pat) if os.path.exists(d + '/outcome.json'))
        with Pool(a.workers, initializer=init_grid, initargs=(a.grid,)) as pool:
            rows = [r for r in pool.map(arena_one, dirs, chunksize=32) if r is not None]
        keys = ['ctx', 'id', 'group', 'split', 'source', 'profile', 'fail', 'unsafe', 'status', 'event_idx', 'route_len', 'min_vx', 'back_s', 'arena']
        out = {k: np.array([r[k] for r in rows]) for k in keys}
        out['ctx'] = np.stack([r['ctx'] for r in rows]); out['X12'] = np.stack([r['X'] for r in rows])
        np.savez(a.out, channels=np.array(CHANNELS), **out)
    print('wrote', a.out, flush=True)


if __name__ == '__main__':
    main()
