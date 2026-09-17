"""How different is the route v2 would have driven from the one v1 drove?

Rebuilds each group's pools (same seeds) and, for every model/pool where the argmin moved, measures the
geometric separation of the two picked routes: mean and max distance between the two paths resampled to 96
stations, plus their mean-speed and length difference. No scoring here: the argmins come from scores/*_all.npz.
"""
import hashlib, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/harry/NeDM-traverse_mppi')
EXP = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909'
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(HERE))
ARENA_DIR = {'f104': 'arena_f104_50h_v1', 'g203': 'arena_g203', 'g216': 'arena_g216',
             'g217': 'arena_g217', 'g228': 'arena_g228', 'g231': 'arena_g231'}
MODELS = ['n2', 'e0', 'd']
POOLS = ['proposal', 'fixed2']
_S = {}


def _seed(g, tag):
    return int(hashlib.md5((g + tag).encode()).hexdigest()[:8], 16)


def _init(arena):
    import gen_planner as P
    P.set_sensor_map(str(EXP / 'sensor_v1/maps' / ARENA_DIR[arena]))
    _S['P'] = P


def resample(r):
    wp = np.asarray(r['waypoints'], float); st = np.asarray(r['stations'], float)
    if st.size != len(wp) or not np.all(np.diff(st) > 0):
        st = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
    g = np.linspace(st[0], st[-1], 96)
    return np.stack([np.interp(g, st, wp[:, 0]), np.interp(g, st, wp[:, 1])], 1)


def build(args):
    case_path, arena, want = args
    P = _S['P']
    case = json.load(open(case_path)); g = case['id']; lay = case['layout']
    pose = [lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']]
    base = {k: np.asarray(v, float) for k, v in
            json.load(open(Path(case_path).parent / 'routes' / g / 'route_00.json')).items()
            if k in ('waypoints', 'speeds', 'stations', 'headings')}
    base['meta'] = {}
    c1, _ = P.proposal_pool(base, pose, np.random.default_rng(_seed(g, 's1_proposal')))
    c2 = P.fixed2_pool(base, pose, np.random.default_rng(_seed(g, 's1_fixed2')))
    pools = {'proposal': c1, 'fixed2': c2}
    res = {}
    for key, (pool, i1, i2) in want.items():
        a = resample(pools[pool][i1]); b = resample(pools[pool][i2])
        d = np.linalg.norm(a - b, axis=1)
        sa = np.asarray(pools[pool][i1]['speeds'])[1:-1].mean(); sb = np.asarray(pools[pool][i2]['speeds'])[1:-1].mean()
        la = np.asarray(pools[pool][i1]['stations'])[-1]; lb = np.asarray(pools[pool][i2]['stations'])[-1]
        res[key] = (float(d.mean()), float(d.max()), float(sb - sa), float(lb - la))
    return g, res


def main():
    out = {}
    for arena in ARENA_DIR:
        t0 = time.time()
        z = np.load(HERE / f'scores/{arena}_all.npz')
        groups = z['group'].astype(str)
        cases = {Path(p).stem: str(p) for p in (EXP / f'sensor_v1/cases2_{arena}/cases').glob('*.json')}
        tasks = []
        for i, g in enumerate(groups):
            want = {}
            for p in POOLS:
                for m in MODELS:
                    i1 = int(z[f'{p}_{m}_v1'][i].argmin()); i2 = int(z[f'{p}_{m}_v2'][i].argmin())
                    if i1 != i2:
                        want[f'{p}/{m}'] = (p, i1, i2)
            if want:
                tasks.append((cases[g], arena, want))
        with ProcessPoolExecutor(10, initializer=_init, initargs=(arena,)) as ex:
            for g, res in ex.map(build, tasks, chunksize=4):
                for k, v in res.items():
                    out.setdefault(k, []).append(v)
        print(f'{arena}: {len(tasks)} groups with at least one flip, {time.time() - t0:.0f}s', flush=True)
    summ = {}
    print(f'\n{"pool/model":16s} {"flips":>6s} {"mean sep (m)":>13s} {"max sep (m)":>12s} {"d speed":>9s} {"d length":>9s}')
    for k, v in sorted(out.items()):
        a = np.array(v)
        summ[k] = dict(n=len(a), median_mean_sep_m=float(np.median(a[:, 0])), p90_mean_sep_m=float(np.percentile(a[:, 0], 90)),
                       median_max_sep_m=float(np.median(a[:, 1])), p90_max_sep_m=float(np.percentile(a[:, 1], 90)),
                       frac_mean_sep_under_1m=float((a[:, 0] < 1.0).mean()),
                       frac_max_sep_under_2m=float((a[:, 1] < 2.0).mean()),
                       median_abs_dspeed=float(np.median(np.abs(a[:, 2]))), median_abs_dlen=float(np.median(np.abs(a[:, 3]))))
        print(f'{k:16s} {len(a):6d} {np.median(a[:, 0]):13.2f} {np.median(a[:, 1]):12.2f} '
              f'{np.median(np.abs(a[:, 2])):9.3f} {np.median(np.abs(a[:, 3])):9.2f}   '
              f'(mean sep < 1 m in {100 * (a[:, 0] < 1.0).mean():.0f}%)')
    json.dump(summ, open(HERE / 'pick_separation.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
