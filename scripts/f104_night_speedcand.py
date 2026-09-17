"""Robustness test, part A: MPPI candidates that perturb BOTH lateral offset and speed along the route
(the regime that broke the old model: 90% failure in the group_0200 demo). Smooth knots for both;
speed deltas follow the same sin^2 envelope and the deform_reference acceleration projection."""
import json, sys, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from nedm.traverse.fdm_mppi import MPPIConfig, validate_reference
from f104_night_paired import bases
from f104_night_dataset import station_tensor
from score_f104_prospective import feats
OUT = 'artifacts/traverse/fdm_f104_50h_20260909/night_v1/speedmppi/cand'
N = 256


def smooth_interp(f, k, vals):
    kx = np.linspace(0.0, 1.0, k + 2); kv = np.r_[0.0, vals, 0.0]
    seg = np.clip(np.searchsorted(kx, f, side='right') - 1, 0, k)
    u = (f - kx[seg]) / (kx[seg + 1] - kx[seg]); w = u * u * (3 - 2 * u)
    return kv[seg] + (kv[seg + 1] - kv[seg]) * w


def deform_smooth_speed(base, pars, cfg):
    xy = np.asarray(base['waypoints'], float); station = np.asarray(base['stations'], float)
    speed = np.asarray(base['speeds'], float); k = cfg.knots
    f = np.clip((station - station[0]) / max(station[-1] - station[0], 1e-6), 0, 1); env = np.sin(np.pi * f) ** 2
    lat = smooth_interp(f, k, pars[:k]) * env; dv = smooth_interp(f, k, pars[k:]) * env
    t = np.gradient(xy, axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    pts = xy + lat[:, None] * np.stack([-t[:, 1], t[:, 0]], 1)
    st = np.r_[0.0, np.linalg.norm(np.diff(pts, axis=0), axis=1).cumsum()]
    v = np.clip(speed + dv, 0.5, cfg.max_speed_mps); v[speed <= 1e-6] = 0.0   # floor while moving; keep the goal stop
    for j in range(1, len(v)):                                        # acceleration limit
        v[j] = min(v[j], np.sqrt(v[j-1] ** 2 + 2 * cfg.max_accel_mps2 * (st[j] - st[j-1])))
    for j in range(len(v) - 2, -1, -1):                               # deceleration limit
        v[j] = min(v[j], np.sqrt(v[j+1] ** 2 + 2 * cfg.max_decel_mps2 * (st[j+1] - st[j])))
    hd = np.arctan2(np.gradient(pts[:, 1]), np.gradient(pts[:, 0]))
    return {'waypoints': pts, 'speeds': v, 'stations': st, 'headings': hd, 'meta': {'candidate': 'smooth_speed'}}


def one(item):
    split, seed, g, b = item
    base = {'waypoints': b['wp'], 'speeds': b['sp'], 'stations': b['st'], 'headings': b['hd'], 'meta': {}}
    lay = b['case']['layout']; anchor = np.array([lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']])
    state0 = np.asarray(np.load(b['dirname'] + '/anchor_state.npz', allow_pickle=True)['state'], np.float32)
    gxy = np.asarray(b['case']['goal_xy'], np.float32); sxy = np.asarray(lay['start_xy'], np.float32); rel = gxy - sxy
    extra = np.array([rel[0], rel[1], np.linalg.norm(rel), float(lay['start_yaw'])], np.float32)
    cfg = MPPIConfig(samples=N, knots=3, lateral_sigma_m=2.6, speed_sigma_mps=1.2, max_lateral_m=6.,
                     max_speed_delta_mps=4., min_speed_mps=0.0, max_speed_mps=6., max_curvature_inv_m=.125,
                     arena_half_extent_m=40.)
    rng = np.random.default_rng(seed); cand = []
    for _ in range(N * 4):
        if len(cand) >= N: break
        p = np.r_[np.clip(rng.normal(0, 2.6, 3), -6, 6), np.clip(rng.normal(0, 1.2, 3), -4, 4)]
        r = deform_smooth_speed(base, p, cfg)
        if validate_reference(r, [], cfg, anchor)['valid']: cand.append(r)
    P, V, XS, C = [], [], [], []
    for r in cand:
        patch, scal = feats(r['waypoints'], r['speeds'], r['stations'], r['headings'])
        P.append(patch); V.append(np.concatenate([scal, state0, extra]))
        X, L = station_tensor(np.asarray(r['waypoints']), np.asarray(r['speeds']), np.asarray(r['stations']))
        XS.append(X); C.append(np.concatenate([state0, [rel[0], rel[1], np.linalg.norm(rel), float(lay['start_yaw']), L]]))
    sp = np.array([np.asarray(r['speeds']) for r in cand])
    np.savez(f'{OUT}/{g}.npz', split=split, old_patch=np.stack(P).astype(np.float16), old_vec=np.stack(V).astype(np.float32),
             X=np.stack(XS).astype(np.float16), ctx=np.stack(C).astype(np.float32),
             wp=np.stack([np.asarray(r['waypoints']) for r in cand]), sp=sp,
             st=np.stack([np.asarray(r['stations']) for r in cand]), hd=np.stack([np.asarray(r['headings']) for r in cand]))
    return g, len(cand), float(sp[:, 10:-10].mean()), float(sp[:, 10:-10].max())


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True); items = []
    for split, s0 in (('test', 7000), ('val', 9000)):
        for gi, (g, b) in enumerate(sorted(bases(split).items())):
            items.append((split, s0 + gi, g, b))
    res = list(ProcessPoolExecutor(12).map(one, items))
    print(f'cached {len(res)} groups; mean candidates {np.mean([r[1] for r in res]):.0f}; '
          f'mid-route commanded speed mean {np.mean([r[2] for r in res]):.2f} m/s, max {max(r[3] for r in res):.2f} m/s')
