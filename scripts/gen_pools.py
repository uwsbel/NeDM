"""Generalisation test, step 1: candidate pools, scores and the six arms' picks for every test start/goal.

For each group (arena heightmap corridors, deployed model, hand rule):
  proposal pool (256, night-2 sampler, anchors included) and fixed-2 m/s geometry pool (256)
  arms  n2            argmin model risk over the proposal pool          (the deployed planner)
        rule          argmin hand-rule score over the SAME proposal pool (non-learned, same candidates)
        straight6     the 6 m/s straight anchor                          (no model, no search)
        n2_fixed2     argmin model risk over the fixed-2 pool            (speed removed)
        rule_fixed2   argmin hand-rule score over the fixed-2 pool
        straight2     the 2 m/s straight anchor
Identical picks are driven once. Seeds are md5(group) so the pools are reproducible.
Writes <out>/picks/<group>.json (scores summary), <out>/routes/<group>__<arm>.json, <out>/tasks_cluster.json.
"""
import argparse, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

ARMS = ('n2', 'rule', 'straight6', 'n2_fixed2', 'rule_fixed2', 'straight2')
_STATE = {}


def _init(arena):
    import gen_planner as P
    P.set_map(arena)
    _STATE['P'] = P; _STATE['rule'] = P.HandRule()


def _seed(g, tag):
    return int(hashlib.md5((g + tag).encode()).hexdigest()[:8], 16)


def build(case_path):
    P = _STATE['P']; rule = _STATE['rule']
    case = json.load(open(case_path)); g = case['id']; lay = case['layout']
    pose = [lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']]
    rp = Path(case_path).parent / 'routes' / g / 'route_00.json'
    base = {k: np.asarray(v, float) for k, v in json.load(open(rp)).items() if k in ('waypoints', 'speeds', 'stations', 'headings')}
    base['meta'] = {}
    pools = {}
    c1, tries = P.proposal_pool(base, pose, np.random.default_rng(_seed(g, 'gen_night2')))
    c2 = P.fixed2_pool(base, pose, np.random.default_rng(_seed(g, 'gen_fixed2')))
    for name, cands in (('proposal', c1), ('fixed2', c2)):
        X, L = P.corridors(cands)
        F = rule.features(X, L)
        pools[name] = dict(cands=cands, X=X.astype(np.float16), L=L, F=F,
                           ctx=P.geom_ctx(lay['start_xy'], case['goal_xy'], lay['start_yaw'], L))
    return g, case_path, pose, tries, pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True, help='directory with <group>.json and routes/')
    ap.add_argument('--arena', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--arena-tag', required=True)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--shards', type=int, default=24)
    a = ap.parse_args()
    import gen_planner as P
    model = P.RiskModel(); rule = P.HandRule()
    out = Path(a.out); (out / 'routes').mkdir(parents=True, exist_ok=True); (out / 'picks').mkdir(exist_ok=True)
    cases = sorted(str(p) for p in Path(a.cases).glob('*.json') if p.name != 'cases.json')
    tasks = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.arena,)) as ex:
        for k, (g, case_path, pose, tries, pools) in enumerate(ex.map(build, cases, chunksize=2)):
            sc = {}
            for name, pl in pools.items():
                z, p = model.score(pl['X'].astype(np.float32), pl['ctx'])
                s = rule.score(pl['F'])
                sc[name] = (z, p, s)
            c1, c2 = pools['proposal']['cands'], pools['fixed2']['cands']
            i6 = P.anchor_index(c1, 0.0, 6.0); i2 = P.anchor_index(c1, 0.0, 2.0)
            pick = {'n2': ('proposal', int(np.argmin(sc['proposal'][0]))),
                    'rule': ('proposal', int(np.argmin(sc['proposal'][2]))),
                    'straight6': ('proposal', i6), 'straight2': ('proposal', i2),
                    'n2_fixed2': ('fixed2', int(np.argmin(sc['fixed2'][0]))) if c2 else None,
                    'rule_fixed2': ('fixed2', int(np.argmin(sc['fixed2'][2]))) if c2 else None}
            shard = int(hashlib.md5(g.encode()).hexdigest(), 16) % a.shards
            seen = {}; summary = dict(group=g, arena=a.arena_tag, n_proposal=len(c1), n_fixed2=len(c2), tries=tries, arms={})
            for arm in ARMS:
                if pick[arm] is None or pick[arm][1] is None:
                    summary['arms'][arm] = None; continue
                pool, idx = pick[arm]
                cands = c1 if pool == 'proposal' else c2
                first = (pool, idx) not in seen
                seen.setdefault((pool, idx), arm)
                rid = f'{g}__{seen[(pool, idx)]}'
                r = cands[idx]
                if first:
                    json.dump({'waypoints': np.asarray(r['waypoints']).tolist(), 'speeds': np.asarray(r['speeds']).tolist(),
                               'stations': np.asarray(r['stations']).tolist(), 'headings': np.asarray(r['headings']).tolist(),
                               'meta': {'candidate': f'gen_{arm}', 'scene_id': g, 'pool': pool, 'cand_index': idx}},
                              open(out / 'routes' / f'{rid}.json', 'w'))
                z, p, s = sc[pool]
                summary['arms'][arm] = dict(route_id=rid, pool=pool, index=idx, risk=float(p[idx]), logit=float(z[idx]),
                                            rule_score=float(s[idx]), rule_rank=int((s < s[idx]).sum()),
                                            model_rank=int((z < z[idx]).sum()),
                                            mean_speed=float(np.asarray(r['speeds'])[1:-1].mean()),
                                            length_m=float(np.asarray(r['stations'])[-1]))
                tasks.append(dict(id=rid, group_id=g, arena=a.arena_tag, arm=arm, case=os.path.relpath(case_path, a.cases),
                                  shard=shard, run=first))
            for name in ('proposal', 'fixed2'):
                z, p, s = sc[name]
                summary[f'{name}_logit_rule_spearman'] = float(np.corrcoef(np.argsort(np.argsort(z)), np.argsort(np.argsort(s)))[0, 1]) if len(z) > 2 else None
                summary[f'{name}_risk_quantiles'] = np.quantile(p, [0, .1, .5, .9, 1]).tolist() if len(p) else None
            json.dump(summary, open(out / 'picks' / f'{g}.json', 'w'), indent=1)
            if (k + 1) % 25 == 0:
                print(f'  {a.arena_tag}: {k + 1}/{len(cases)}', flush=True)
    json.dump(tasks, open(out / 'tasks_cluster.json', 'w'), indent=1)
    print(f'{a.arena_tag}: {len(cases)} groups, {sum(t["run"] for t in tasks)} episodes to drive')


if __name__ == '__main__':
    main()
