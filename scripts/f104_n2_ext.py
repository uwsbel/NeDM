"""Extension: larger paired 2 m/s test — night-1 model vs night-2 model on identical fixed-speed candidate sets."""
import hashlib, json, os, shutil, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from nedm.traverse.fdm_mppi import MPPIConfig, validate_reference
from f104_night_dataset import station_tensor
import f104_n2_sampler as S

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
N2 = ROOT + '/night2_v1'
OUT = N2 + '/testcand_ext'
CASES = ROOT + '/cases_ext_final'
CFG = MPPIConfig(max_speed_mps=6., min_speed_mps=0.0, max_curvature_inv_m=.125, arena_half_extent_m=40.)
N = 192


def stage():
    sel = json.load(open(N2 + '/ext_test_groups.json'))['groups']
    os.makedirs(CASES + '/routes', exist_ok=True)
    for s in sel:
        g = s['group']; src = f"{ROOT}/{s['dir']}/cases"
        shutil.copy(f'{src}/{g}.json', f'{CASES}/{g}.json')
        os.makedirs(f'{CASES}/routes/{g}', exist_ok=True)
        shutil.copy(f'{src}/routes/{g}/route_00.json', f'{CASES}/routes/{g}/route_00.json')
    return [s['group'] for s in sel]


def one(g):
    case = json.load(open(f'{CASES}/{g}.json')); lay = case['layout']
    anchor = np.array([lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']])
    base = {k: np.asarray(v, float) for k, v in json.load(open(f'{CASES}/routes/{g}/route_00.json')).items()
            if k in ('waypoints', 'speeds', 'stations', 'headings')}
    rng = np.random.default_rng(int(hashlib.md5((g + 'ext').encode()).hexdigest()[:8], 16))
    cand = []
    while len(cand) < N:
        r = S.sample_one(base, rng, lat_sigma=5.0, sp_sigma=0.0, base_speed=2.0)
        if validate_reference(r, [], CFG, anchor)['valid']: cand.append(r)
    X, L = [], []
    for r in cand:
        x, l = station_tensor(np.asarray(r['waypoints']), np.asarray(r['speeds']), np.asarray(r['stations']))
        X.append(x); L.append(l)
    gxy = np.asarray(case['goal_xy'], float); sxy = np.asarray(lay['start_xy'], float); rel = gxy - sxy
    ctx = np.stack([[rel[0], rel[1], np.linalg.norm(rel), float(lay['start_yaw']), l] for l in L]).astype(np.float32)
    np.savez(f'{OUT}/{g}.npz', X=np.stack(X).astype(np.float16), geom_ctx=ctx,
             wp=np.stack([np.asarray(r['waypoints']) for r in cand]),
             sp=np.stack([np.asarray(r['speeds']) for r in cand]),
             st=np.stack([np.asarray(r['stations']) for r in cand]),
             hd=np.stack([np.asarray(r['headings']) for r in cand]))
    return g, len(cand)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    gs = stage()
    print(f'{len(gs)} groups staged', flush=True)
    n = 0
    with ProcessPoolExecutor(14) as ex:
        for g, c in ex.map(one, gs):
            n += 1
            if n % 100 == 0: print(f'  {n}/{len(gs)}', flush=True)
    print('candidate sets written to', OUT)
