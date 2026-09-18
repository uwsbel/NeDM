"""Held-out single-goal CRM missions: one candidate pool per start/goal, every model picks from the SAME pool.

Adapted from scripts/sensor_pools_v2.py. Differences: corridors come from the N2 sampler on ONE overhead reference
depth image (f104_n2_dataset.init_map(<root>) -> <root>/static_map_v1, here the OptiX render made on luffy), models are
N2-format ensembles (gen_planner.RiskModel), and the model-free arms straight6 / straight2 are always included.

Arms: straight6, straight2, and for each NAME=glob in --models: NAME (speed free, 256-candidate proposal pool),
NAME_fixed2 (geometry-only pool at 2 m/s), NAME_pess (speed free, max over ensemble members instead of the mean).
Identical (pool, index) picks share one route file and are driven once. Pools are rebuilt bit-for-bit from the
md5 seeds (tags 'crm_proposal' / 'crm_fixed2').
"""
import argparse, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
_S = {}


def _init(map_root):
    import gen_planner as P
    P.DS.init_map(map_root); _S['P'] = P


def _seed(g, tag):
    return int(hashlib.md5((g + tag).encode()).hexdigest()[:8], 16)


def build(case_path):
    P = _S['P']
    case = json.load(open(case_path)); g = case['id']; lay = case['layout']
    pose = [lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']]
    base = {k: np.asarray(v, float) for k, v in json.load(open(Path(case_path).parent / 'routes' / g / 'route_00.json')).items()
            if k in ('waypoints', 'speeds', 'stations', 'headings')}
    base['meta'] = {}
    c1, tries = P.proposal_pool(base, pose, np.random.default_rng(_seed(g, 'crm_proposal')))
    c2 = P.fixed2_pool(base, pose, np.random.default_rng(_seed(g, 'crm_fixed2')))
    pools = {}
    for name, cands in (('proposal', c1), ('fixed2', c2)):
        if not cands:
            pools[name] = dict(cands=[], X=np.zeros((0, 5, 96, 32), np.float16), ctx=np.zeros((0, 5), np.float32)); continue
        X, L = P.corridors(cands)
        pools[name] = dict(cands=cands, X=X.astype(np.float16), ctx=P.geom_ctx(lay['start_xy'], case['goal_xy'], lay['start_yaw'], L))
    return g, case_path, tries, pools


def member_logits(model, X, ctx):
    """(members, n) route logits; the ensemble mean of these is what RiskModel.score returns."""
    import gen_planner as P
    out = []
    for m, ck in model.members:
        one = P.RiskModel.__new__(P.RiskModel)
        one.torch, one.route_logit, one.dev, one.members = model.torch, model.route_logit, model.dev, [(m, ck)]
        out.append(one.score(X, ctx)[0])
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True); ap.add_argument('--map-root', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--models', required=True, help='NAME=glob,NAME=glob')
    ap.add_argument('--case-prefix', required=True); ap.add_argument('--route-prefix', required=True)
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    import gen_planner as P
    models = {k: P.RiskModel(v) for k, v in (kv.split('=', 1) for kv in a.models.split(','))}
    arms = ['straight6', 'straight2'] + list(models) + [f'{k}_fixed2' for k in models] + [f'{k}_pess' for k in models]
    out = Path(a.out); (out / 'routes').mkdir(parents=True, exist_ok=True); (out / 'picks').mkdir(exist_ok=True)
    cases = sorted(str(p) for p in Path(a.cases).glob('*.json') if p.name != 'cases.json')
    tasks = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.map_root,)) as ex:
        for gi, (g, case_path, tries, pools) in enumerate(ex.map(build, cases, chunksize=2)):
            sc = {n: {k: member_logits(m, pl['X'].astype(np.float32), pl['ctx']) for k, m in models.items()} if len(pl['cands']) else None
                  for n, pl in pools.items()}
            c1, c2 = pools['proposal']['cands'], pools['fixed2']['cands']
            pick = {'straight6': ('proposal', P.anchor_index(c1, 0.0, 6.0)), 'straight2': ('proposal', P.anchor_index(c1, 0.0, 2.0))}
            for k in models:
                pick[k] = ('proposal', int(np.argmin(sc['proposal'][k].mean(0))))
                pick[f'{k}_pess'] = ('proposal', int(np.argmin(sc['proposal'][k].max(0))))
                pick[f'{k}_fixed2'] = ('fixed2', int(np.argmin(sc['fixed2'][k].mean(0)))) if c2 else None
            seen = {}; summ = dict(group=g, n_proposal=len(c1), n_fixed2=len(c2), tries=tries, arms={})
            for arm in arms:
                if pick.get(arm) is None or pick[arm][1] is None:
                    summ['arms'][arm] = None; continue
                pool, idx = pick[arm]; cands = c1 if pool == 'proposal' else c2
                first = (pool, idx) not in seen; seen.setdefault((pool, idx), arm); rid = f'{g}__{seen[(pool, idx)]}'
                r = cands[idx]
                if first:
                    json.dump({'waypoints': np.asarray(r['waypoints']).tolist(), 'speeds': np.asarray(r['speeds']).tolist(),
                               'stations': np.asarray(r['stations']).tolist(), 'headings': np.asarray(r['headings']).tolist(),
                               'meta': {'candidate': f'crm_eval_{arm}', 'scene_id': g, 'pool': pool, 'cand_index': idx}},
                              open(out / 'routes' / f'{rid}.json', 'w'))
                    tasks.append(dict(id=rid, group=g, case=f"{a.case_prefix}/{os.path.basename(case_path)}",
                                      route=f"{a.route_prefix}/{rid}.json", run=True, tier=gi,  # group by group
                                      episode_seed=int(hashlib.md5(rid.encode()).hexdigest()[:8], 16)))
                z = {k: sc[pool][k].mean(0) for k in models}
                summ['arms'][arm] = dict(route_id=rid, pool=pool, index=idx, mean_speed=float(np.asarray(r['speeds'])[1:-1].mean()),
                                         length_m=float(np.asarray(r['stations'])[-1]),
                                         max_lateral_m=float(r.get('meta', {}).get('max_lateral_m', 0.0)),
                                         **{f'logit_{k}': float(z[k][idx]) for k in models},
                                         **{f'rank_{k}': int((z[k] < z[k][idx]).sum()) for k in models})
            json.dump(summ, open(out / 'picks' / f'{g}.json', 'w'), indent=1)
    json.dump(tasks, open(out / 'tasks.json', 'w'), indent=1)
    print(f'{len(cases)} groups, {len(tasks)} distinct drives, arms {arms}')


if __name__ == '__main__':
    main()
