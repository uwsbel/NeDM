"""Hazard test: candidate sets for 300 hill/crater groups, three proposals (night1, night2, fixed 2 m/s)."""
import hashlib, json, os, shutil, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from nedm.traverse.fdm_mppi import MPPIConfig, validate_reference
from f104_night_dataset import station_tensor
from f104_night_speedcand import deform_smooth_speed
import f104_n2_sampler as S

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
N2 = ROOT + '/night2_v1'
OUT = N2 + '/testcand_haz'
CASES = ROOT + '/cases_haz_final'
N = 256
CFG1 = MPPIConfig(samples=N, knots=3, lateral_sigma_m=2.6, speed_sigma_mps=1.2, max_lateral_m=6.,
                  max_speed_delta_mps=4., min_speed_mps=0.0, max_speed_mps=6., max_curvature_inv_m=.125,
                  arena_half_extent_m=40.)
CFG2 = MPPIConfig(max_speed_mps=6., min_speed_mps=0.0, max_curvature_inv_m=.125, arena_half_extent_m=40.)


def stage():
    sel = json.load(open(N2 + '/haz_test_groups.json'))['groups']
    os.makedirs(CASES + '/routes', exist_ok=True)
    for s in sel:
        g = s['group']; src = f"{ROOT}/{s['dir']}/cases"
        shutil.copy(f'{src}/{g}.json', f'{CASES}/{g}.json')
        os.makedirs(f'{CASES}/routes/{g}', exist_ok=True)
        shutil.copy(f'{src}/routes/{g}/route_00.json', f'{CASES}/routes/{g}/route_00.json')
    return [s['group'] for s in sel]


def pack(cand, case, path, anchor_flags=None):
    X, L = [], []
    for r in cand:
        x, l = station_tensor(np.asarray(r['waypoints']), np.asarray(r['speeds']), np.asarray(r['stations']))
        X.append(x); L.append(l)
    lay = case['layout']
    gxy = np.asarray(case['goal_xy'], float); sxy = np.asarray(lay['start_xy'], float); rel = gxy - sxy
    ctx = np.stack([[rel[0], rel[1], np.linalg.norm(rel), float(lay['start_yaw']), l] for l in L]).astype(np.float32)
    meta = [r.get('meta', {}) for r in cand]
    np.savez(path, X=np.stack(X).astype(np.float16), geom_ctx=ctx,
             wp=np.stack([np.asarray(r['waypoints']) for r in cand]),
             sp=np.stack([np.asarray(r['speeds']) for r in cand]),
             st=np.stack([np.asarray(r['stations']) for r in cand]),
             hd=np.stack([np.asarray(r['headings']) for r in cand]),
             anchor=np.array([m.get('candidate') == 'n2_anchor' for m in meta]),
             cruise=np.array([m.get('cruise_speed_mps', np.nan) for m in meta], np.float32),
             offset=np.array([m.get('lateral_offset_m', np.nan) for m in meta], np.float32))


def one(g):
    case = json.load(open(f'{CASES}/{g}.json')); lay = case['layout']
    anchor = np.array([lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']])
    base = {k: np.asarray(v, float) for k, v in json.load(open(f'{CASES}/routes/{g}/route_00.json')).items()
            if k in ('waypoints', 'speeds', 'stations', 'headings')}
    base['meta'] = {}
    seed = lambda tag: int(hashlib.md5((g + tag).encode()).hexdigest()[:8], 16)
    r1 = np.random.default_rng(seed('h1')); c1 = []
    for _ in range(40 * N):          # generous budget so the control proposal also reaches 256 candidates
        if len(c1) >= N: break
        p = np.r_[np.clip(r1.normal(0, 2.6, 3), -6, 6), np.clip(r1.normal(0, 1.2, 3), -4, 4)]
        r = deform_smooth_speed(base, p, CFG1)
        if validate_reference(r, [], CFG1, anchor)['valid']: c1.append(r)
    r2 = np.random.default_rng(seed('h2'))
    c2, _ = S.propose(base, anchor, r2, n=N, validate=validate_reference, cfg=CFG2)
    r3 = np.random.default_rng(seed('h3')); c3 = []
    for _ in range(6 * N):
        if len(c3) >= N: break
        r = S.sample_one(base, r3, lat_sigma=5.0, sp_sigma=0.0, base_speed=2.0)
        if validate_reference(r, [], CFG2, anchor)['valid']: c3.append(r)
    for name, cand in (('night1', c1), ('night2', c2), ('fixed2', c3)):
        pack(cand, case, f'{OUT}/{g}__{name}.npz')
    return g, (len(c1), len(c2), len(c3))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    gs = stage()
    print(f'{len(gs)} hazard groups staged', flush=True)
    res = []
    with ProcessPoolExecutor(14) as ex:
        for g, c in ex.map(one, gs):
            res.append(c)
            if len(res) % 60 == 0: print(f'  {len(res)}/{len(gs)}', flush=True)
    a = np.array(res)
    print(f'mean candidates: night1 {a[:,0].mean():.0f}, night2 {a[:,1].mean():.0f}, fixed2 {a[:,2].mean():.0f}')
