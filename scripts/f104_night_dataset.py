"""Night session step 2: per-station tensors for every driven route.

Channels on a route-aligned 96 (station) x 32 (lateral, +-6 m) grid:
  0 elevation relative to the route start [m]
  1 along-path grade  d(elev)/ds          [m/m]
  2 cross-slope        d(elev)/dlateral   [m/m]
  3 commanded speed at that station       [m/s]
  4 in-arena mask
Old-model features (4ch patch + 18 scalars) are computed for the same routes so the
current production model can be scored on exactly the same set.
"""
import json, os, sys, glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, 'scripts')
from build_f104_planner_dataset import sample_map, resample_route, N_STATION, N_LATERAL, HALF_WIDTH_M, ELEV_SCALE
from score_f104_prospective import feats

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
OUT = ROOT + '/night_v1'


def load_route(src):
    if src.endswith('.npz'):
        z = np.load(src)
        return (np.asarray(z['reference_waypoints'], float), np.asarray(z['reference_speeds'], float),
                np.asarray(z['reference_stations'], float), np.asarray(z['reference_headings'], float))
    r = json.load(open(src))
    return (np.asarray(r['waypoints'], float), np.asarray(r['speeds'], float),
            np.asarray(r['stations'], float), np.asarray(r['headings'], float))


def station_tensor(wp, sp, st):
    pts, grid = resample_route(wp, st, N_STATION)
    d = np.gradient(pts, axis=0); tn = np.linalg.norm(d, axis=1, keepdims=True); tn[tn < 1e-9] = 1e-9
    tang = d / tn; norm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    off = np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, N_LATERAL)
    gx = pts[:, 0:1] + norm[:, 0:1] * off[None, :]; gy = pts[:, 1:2] + norm[:, 1:2] * off[None, :]
    patch, valid = sample_map(gx, gy)
    elev = np.where(valid, patch[3] * ELEV_SCALE, np.nan)
    e0 = elev[0, N_LATERAL // 2] if np.isfinite(elev[0, N_LATERAL // 2]) else np.nanmean(elev[0])
    fill = np.where(np.isfinite(elev), elev, e0)
    ds = max(float(grid[-1] - grid[0]) / (N_STATION - 1), 1e-3); dl = 2 * HALF_WIDTH_M / (N_LATERAL - 1)
    ga = np.clip(np.gradient(fill, ds, axis=0), -2, 2); gc = np.clip(np.gradient(fill, dl, axis=1), -2, 2)
    s_ref = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
    v = np.interp(np.linspace(0, s_ref[-1], N_STATION), s_ref, sp)
    X = np.stack([fill - e0, ga, gc, np.repeat(v[:, None], N_LATERAL, 1), valid.astype(float)])
    return X.astype(np.float32), float(grid[-1] - grid[0])


def one(r):
    try:
        wp, sp, st, hd = load_route(r['route'])
        X, L = station_tensor(wp, sp, st)
        op, os_ = feats(wp, sp, st, hd)
    except Exception as e:
        return None, str(e)
    ev = -1 if r['event_frame'] is None else int(np.clip(round(r['event_frac'] * (N_STATION - 1)), 0, N_STATION - 1))
    return (X, op.astype(np.float32), os_.astype(np.float32), ev, L), None


def main():
    R = json.load(open(OUT + '/episodes.json'))
    anc, gxy, sxy, yaw = {}, {}, {}, {}
    for d in glob.glob(ROOT + '/production_v2/runs/*'):
        g = os.path.basename(d).split('_route_')[0]
        if g in anc: continue
        try:
            anc[g] = np.asarray(np.load(d + '/anchor_state.npz', allow_pickle=True)['state'], np.float32)
            c = json.load(open(d + '/case.json'))
            gxy[g] = np.asarray(c['goal_xy'], np.float32); sxy[g] = np.asarray(c['layout']['start_xy'], np.float32)
            yaw[g] = float(c['layout']['start_yaw'])
        except Exception: pass
    R = [r for r in R if r['group'] in anc]
    res = list(ProcessPoolExecutor(14).map(one, R, chunksize=32))
    keep = [i for i, (x, e) in enumerate(res) if x is not None]
    print(f'built {len(keep)}/{len(R)}  (failures {len(R)-len(keep)})', flush=True)
    R = [R[i] for i in keep]; res = [res[i][0] for i in keep]
    ctx = []
    for r, (X, op, os_, ev, L) in zip(R, res):
        g = r['group']; rel = gxy[g] - sxy[g]
        ctx.append(np.concatenate([anc[g], [rel[0], rel[1], np.linalg.norm(rel), yaw[g], L]]).astype(np.float32))
    A = lambda k, dt: np.array([r[k] for r in R], dtype=dt)
    np.savez(OUT + '/station_ds.npz',
             X=np.stack([x[0] for x in res]).astype(np.float16),
             old_patch=np.stack([x[1] for x in res]).astype(np.float16),
             old_scal=np.stack([x[2] for x in res]), ctx=np.stack(ctx),
             event_idx=np.array([x[3] for x in res], np.int16),
             route_len=np.array([x[4] for x in res], np.float32),
             id=A('id', object), source=A('source', object), group=A('group', object), split=A('split', object),
             profile=A('profile', np.int8), geom=A('geom', object), fail=A('fail', np.int8),
             unsafe=A('unsafe', np.int8), max_dev_pre=A('max_dev_pre_event', np.float32),
             mean_cmd_speed=A('mean_cmd_speed', np.float32))
    print('wrote', OUT + '/station_ds.npz', round(os.path.getsize(OUT + '/station_ds.npz') / 1e6), 'MB')


if __name__ == '__main__':
    main()
