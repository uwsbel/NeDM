"""Control pass: score the same regenerated pools under THREE corridor sources on a subset of groups.

  v1   scripts/sensor_dataset.tensor10 via gen_planner.corridors10   (deployed: flat-ground image lookup)
  fg   the flat-placement control grid (build_flat_grid.py)          (same rasterisation as v2, v1's geometry)
  v2   the back-projected metric grid                                (sensor_map_v2.py)

v1 -> fg is the resampling step, fg -> v2 is the geometry correction.
"""
import argparse, hashlib, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/harry/NeDM-traverse_mppi')
EXP = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909'
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(HERE))

ARENA_DIR = {'f104': 'arena_f104_50h_v1', 'g203': 'arena_g203', 'g216': 'arena_g216',
             'g217': 'arena_g217', 'g228': 'arena_g228', 'g231': 'arena_g231'}
_S = {}


def _seed(g, tag):
    return int(hashlib.md5((g + tag).encode()).hexdigest()[:8], 16)


def _init(arena):
    import gen_planner as P
    from corridor_v2 import V2Map, tensor10_v2
    P.set_sensor_map(str(EXP / 'sensor_v1/maps' / ARENA_DIR[arena]))
    _S.update(P=P, t10=tensor10_v2,
              v2=V2Map(EXP / 'sensor_v2/grids' / ARENA_DIR[arena]),
              fg=V2Map(HERE / 'flat_grids' / ARENA_DIR[arena]))


def build(args):
    case_path, arena = args
    P, t10 = _S['P'], _S['t10']
    case = json.load(open(case_path)); g = case['id']; lay = case['layout']
    pose = [lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']]
    base = {k: np.asarray(v, float) for k, v in
            json.load(open(Path(case_path).parent / 'routes' / g / 'route_00.json')).items()
            if k in ('waypoints', 'speeds', 'stations', 'headings')}
    base['meta'] = {}
    c1, _ = P.proposal_pool(base, pose, np.random.default_rng(_seed(g, 's1_proposal')))
    c2 = P.fixed2_pool(base, pose, np.random.default_rng(_seed(g, 's1_fixed2')))
    out = {'group': g}
    for name, cands in (('proposal', c1), ('fixed2', c2)):
        X1, L = P.corridors10(cands)
        Xf = np.stack([t10(_S['fg'], np.asarray(r['waypoints']), np.asarray(r['speeds']), np.asarray(r['stations']))[0] for r in cands])
        X2 = np.stack([t10(_S['v2'], np.asarray(r['waypoints']), np.asarray(r['speeds']), np.asarray(r['stations']))[0] for r in cands])
        out[name] = dict(v1=X1.astype(np.float16), fg=Xf.astype(np.float16), X2=X2.astype(np.float16),
                         ctx=P.geom_ctx(lay['start_xy'], case['goal_xy'], lay['start_yaw'], L))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    import gen_planner as P
    models = {'n2': P.RiskModel(),
              'e0': P.SensorRiskModel(str(EXP / 'sensor_v1/final/E0_s*.pt')),
              'd': P.SensorRiskModel(str(EXP / 'sensor_v1/final/D_s*.pt'))}
    (HERE / 'scores').mkdir(exist_ok=True)
    for arena in ARENA_DIR:
        t0 = time.time()
        cases = sorted(str(p) for p in (EXP / f'sensor_v1/cases2_{arena}/cases').glob('*.json') if p.name != 'cases.json')
        sel = cases[::a.stride]
        acc = {f'{p}_{k}_{v}': [] for p in ('proposal', 'fixed2') for k in models for v in ('v1', 'fg', 'v2')}
        groups = []
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(arena,)) as ex:
            for r in ex.map(build, [(c, arena) for c in sel], chunksize=1):
                groups.append(r['group'])
                for pool in ('proposal', 'fixed2'):
                    d = r[pool]; ctx = d['ctx']
                    for k, m in models.items():
                        for ver, key in (('v1', 'v1'), ('fg', 'fg'), ('v2', 'X2')):
                            X = d[key].astype(np.float32)
                            z, _ = m.score(X[:, :5] if k == 'n2' else X, ctx)
                            acc[f'{pool}_{k}_{ver}'].append(z.astype(np.float64))
        np.savez_compressed(HERE / f'scores/{arena}_ctl.npz', group=np.array(groups),
                            **{k: np.stack(v) for k, v in acc.items()})
        print(f'{arena}: {len(groups)} groups in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
