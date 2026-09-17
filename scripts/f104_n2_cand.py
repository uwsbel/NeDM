"""Night-2 closed loop, step 1: candidate sets for the frozen fresh test groups, for both proposals.

night1 proposal: scripts/f104_night_speedcand.deform_smooth_speed (3 knots, lateral sigma 2.6 clip 6,
                 speed sigma 1.2 with a sin^2 envelope that is zero at both ends)
night2 proposal: scripts/f104_n2_sampler.propose (curvature-safe sine basis, lateral to +-10 m, free end speed,
                 plus the 9 designed-style anchors)
Both are scored later with the same models, so the comparison isolates the proposal.
"""
import hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from nedm.traverse.fdm_mppi import MPPIConfig, validate_reference
from f104_night_dataset import station_tensor
from f104_night_speedcand import deform_smooth_speed
import f104_n2_sampler as S

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
CASES = ROOT + '/cases_test_final'
OUT = ROOT + '/night2_v1/testcand'
N = 256
CFG1 = MPPIConfig(samples=N, knots=3, lateral_sigma_m=2.6, speed_sigma_mps=1.2, max_lateral_m=6.,
                  max_speed_delta_mps=4., min_speed_mps=0.0, max_speed_mps=6., max_curvature_inv_m=.125,
                  arena_half_extent_m=40.)
CFG2 = MPPIConfig(max_speed_mps=6., min_speed_mps=0.0, max_curvature_inv_m=.125, arena_half_extent_m=40.)


def pack(routes):
    X, ctxL, wp, sp, st, hd, meta = [], [], [], [], [], [], []
    for r in routes:
        w = np.asarray(r['waypoints'], float); s = np.asarray(r['speeds'], float)
        t = np.asarray(r['stations'], float); h = np.asarray(r['headings'], float)
        x, L = station_tensor(w, s, t)
        X.append(x); ctxL.append(L); wp.append(w); sp.append(s); st.append(t); hd.append(h)
        meta.append(r.get('meta', {}))
    return np.stack(X).astype(np.float16), np.array(ctxL, np.float32), wp, sp, st, hd, meta


def one(g):
    case = json.load(open(f'{CASES}/{g}.json'))
    lay = case['layout']; anchor = np.array([lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']])
    base = {k: np.asarray(v, float) for k, v in json.load(open(f'{CASES}/routes/{g}/route_00.json')).items()
            if k in ('waypoints', 'speeds', 'stations', 'headings')}
    base['meta'] = {}
    out = {}
    seed = int(hashlib.md5(g.encode()).hexdigest()[:8], 16)      # stable across processes (hash() is salted)
    rng = np.random.default_rng(seed)
    c1 = []
    for _ in range(N * 4):
        if len(c1) >= N: break
        p = np.r_[np.clip(rng.normal(0, 2.6, 3), -6, 6), np.clip(rng.normal(0, 1.2, 3), -4, 4)]
        r = deform_smooth_speed(base, p, CFG1)
        if validate_reference(r, [], CFG1, anchor)['valid']: c1.append(r)
    rng2 = np.random.default_rng(int(hashlib.md5((g + 'n2').encode()).hexdigest()[:8], 16))
    c2, _ = S.propose(base, anchor, rng2, n=N, validate=validate_reference, cfg=CFG2)
    # fixed-speed set: same lateral family, speed pinned to the 2 m/s base (the constrained regime)
    rng3 = np.random.default_rng(int(hashlib.md5((g + 'f2').encode()).hexdigest()[:8], 16))
    c3 = []
    while len(c3) < N and len(c3) < 4 * N:
        r = S.sample_one(base, rng3, lat_sigma=5.0, sp_sigma=0.0, base_speed=2.0)
        if validate_reference(r, [], CFG2, anchor)['valid']: c3.append(r)
        if len(c3) == 0 and rng3.random() < 1e-9: break
    gxy = np.asarray(case['goal_xy'], float); sxy = np.asarray(lay['start_xy'], float); rel = gxy - sxy
    for name, cand in (('night1', c1), ('night2', c2), ('fixed2', c3)):
        X, L, wp, sp, st, hd, meta = pack(cand)
        ctx = np.stack([np.concatenate([[rel[0], rel[1], np.linalg.norm(rel), float(lay['start_yaw']), l]])
                        for l in L]).astype(np.float32)
        np.savez(f'{OUT}/{g}__{name}.npz', X=X, geom_ctx=ctx, route_len=L,
                 wp=np.stack([w for w in wp]), sp=np.stack(sp), st=np.stack(st), hd=np.stack(hd),
                 anchor=np.array([m.get('candidate') == 'n2_anchor' for m in meta]),
                 cruise=np.array([m.get('cruise_speed_mps', np.nan) for m in meta], np.float32),
                 offset=np.array([m.get('lateral_offset_m', np.nan) for m in meta], np.float32),
                 max_lateral=np.array([m.get('max_lateral_m', np.nan) for m in meta], np.float32),
                 mean_speed=np.array([m.get('mean_speed_mps', np.nan) for m in meta], np.float32),
                 split=case['split'])
        out[name] = len(cand)
    return g, out


def main():
    os.makedirs(OUT, exist_ok=True)
    groups = json.load(open(ROOT + '/night2_v1/fresh_test_groups.json'))['groups']
    gs = [x['group'] for x in groups]
    res = []
    with ProcessPoolExecutor(14) as ex:
        for g, o in ex.map(one, gs):
            res.append((g, o))
            if len(res) % 40 == 0: print(f'  {len(res)}/{len(gs)}', flush=True)
    n1 = np.mean([o['night1'] for _, o in res]); n2 = np.mean([o['night2'] for _, o in res])
    print(f'{len(res)} groups; mean candidates night1 {n1:.0f}, night2 {n2:.0f} -> {OUT}')


if __name__ == '__main__':
    main()
