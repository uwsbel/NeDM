"""Chrono pilot picks for the matched v2 models (height vs depth), corridors from the back-projected world grid.

Arms: for each model NAME given in --models, NAME (speed free) and NAME_fixed2 (geometry-only pool at 2 m/s),
plus straight6. Everything else (pools, seeds, dedup, task file) matches scripts/sensor_pools.py.
"""
import argparse, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
_S = {}


def _init(griddir):
    import gen_planner as P
    P.set_grid_map(griddir); _S['P'] = P


def _seed(g, tag):
    return int(hashlib.md5((g + tag).encode()).hexdigest()[:8], 16)


def build(case_path):
    P = _S['P']
    case = json.load(open(case_path)); g = case['id']; lay = case['layout']
    pose = [lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']]
    base = {k: np.asarray(v, float) for k, v in json.load(open(Path(case_path).parent / 'routes' / g / 'route_00.json')).items()
            if k in ('waypoints', 'speeds', 'stations', 'headings')}
    base['meta'] = {}
    c1, tries = P.proposal_pool(base, pose, np.random.default_rng(_seed(g, 'v2_proposal')))
    c2 = P.fixed2_pool(base, pose, np.random.default_rng(_seed(g, 'v2_fixed2')))
    pools = {}
    for name, cands in (('proposal', c1), ('fixed2', c2)):
        X, L = P.corridors12(cands)
        pools[name] = dict(cands=cands, X=X.astype(np.float16), ctx=P.geom_ctx(lay['start_xy'], case['goal_xy'], lay['start_yaw'], L))
    return g, case_path, tries, pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True); ap.add_argument('--grid', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--arena-tag', required=True); ap.add_argument('--models', required=True)
    ap.add_argument('--case-prefix', required=True); ap.add_argument('--route-prefix', required=True)
    ap.add_argument('--workers', type=int, default=12); ap.add_argument('--shards', type=int, default=8)
    a = ap.parse_args()
    import gen_planner as P
    models = {k: P.GridRiskModel(v) for k, v in (kv.split('=', 1) for kv in a.models.split(','))}
    arms = ['straight6'] + list(models) + [f'{k}_fixed2' for k in models]
    out = Path(a.out); (out / 'routes').mkdir(parents=True, exist_ok=True); (out / 'picks').mkdir(exist_ok=True)
    cases = sorted(str(p) for p in Path(a.cases).glob('*.json') if p.name != 'cases.json')
    tasks = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.grid,)) as ex:
        for gi, (g, case_path, tries, pools) in enumerate(ex.map(build, cases, chunksize=2)):
            sc = {n: {k: m.score(pl['X'].astype(np.float32), pl['ctx']) for k, m in models.items()} for n, pl in pools.items()}
            c1, c2 = pools['proposal']['cands'], pools['fixed2']['cands']
            pick = {'straight6': ('proposal', P.anchor_index(c1, 0.0, 6.0))}
            for k in models:
                pick[k] = ('proposal', int(np.argmin(sc['proposal'][k][0])))
                pick[f'{k}_fixed2'] = ('fixed2', int(np.argmin(sc['fixed2'][k][0]))) if c2 else None
            shard = int(hashlib.md5(g.encode()).hexdigest(), 16) % a.shards
            seen = {}; summ = dict(group=g, arena=a.arena_tag, n_proposal=len(c1), n_fixed2=len(c2), tries=tries, arms={})
            for arm in arms:
                if pick.get(arm) is None or pick[arm][1] is None:
                    summ['arms'][arm] = None; continue
                pool, idx = pick[arm]; cands = c1 if pool == 'proposal' else c2
                first = (pool, idx) not in seen; seen.setdefault((pool, idx), arm); rid = f'{g}__{seen[(pool, idx)]}'
                r = cands[idx]
                if first:
                    json.dump({'waypoints': np.asarray(r['waypoints']).tolist(), 'speeds': np.asarray(r['speeds']).tolist(),
                               'stations': np.asarray(r['stations']).tolist(), 'headings': np.asarray(r['headings']).tolist(),
                               'meta': {'candidate': f'v2_{arm}', 'scene_id': g, 'pool': pool, 'cand_index': idx}},
                              open(out / 'routes' / f'{rid}.json', 'w'))
                    tasks.append(dict(id=rid, group=g, arena=a.arena_tag, case=f"{a.case_prefix}/{os.path.basename(case_path)}",
                                      route=f"{a.route_prefix}/{rid}.json", shard=shard, run=True))
                summ['arms'][arm] = dict(route_id=rid, pool=pool, index=idx, mean_speed=float(np.asarray(r['speeds'])[1:-1].mean()),
                                         **{f'risk_{k}': float(v[1][idx]) for k, v in sc[pool].items()})
            json.dump(summ, open(out / 'picks' / f'{g}.json', 'w'), indent=1)
    json.dump(tasks, open(out / 'tasks.json', 'w'), indent=1)
    print(f'{a.arena_tag}: {len(cases)} groups, {len(tasks)} episodes')


if __name__ == '__main__':
    main()
