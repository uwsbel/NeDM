#!/usr/bin/env python3
"""Dev-arena headroom check, pick side (arena_gator_20260925, E3a; PLAN 2.2 "Dev (g217)").

Arms per g217 dev group (standing start at the case pose):
  Scrm:B     frozen K1 soil specialist (crm_f104_v1/train_v1/deploy/CRM_N2_s*.pt, the ensemble behind the 800-group
             standing-start suite, generalist_20260921/A_adapt/suite/picks_crm_Scrm), CEM 4 x 64 (ga_planner.py
             --world crm --arms B, run separately; its picks dir is --picks)
  straight6  the designed anchor with lateral offset 0 at 6 m/s from the night-2 proposal pool, built exactly as
             crm_pools.py does (gen_planner.proposal_pool with rng md5(group + 'crm_proposal'), anchor_index(0, 6))
Identical routes (same content sha256) are driven once (row 'arms' lists both). Before anything is written the map
root is checked against the cases' arena (ag_map_check). Outputs in --out: straight6/routes/<g>__straight6.json,
straight6/picks.json, tasks_dev.json (soil task rows, tier --tier, paths relative to G3), DEV_PICKS_LOCKED.sha256
(one sha256 over name + content of every route file of both arms, sorted; then one line per file).
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ag_map_check
import ag_tasklib as L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True, help='local case dir (K3/cases/dev_g217/cases)')
    ap.add_argument('--map-root', required=True)
    ap.add_argument('--picks', required=True, help='ga_planner.py output dir (arm B)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--arena', default='g217')
    ap.add_argument('--tier', type=int, default=-1)
    ap.add_argument('--model-tag', default='Scrm', help="run id = <group>__<tag>_B (ga_planner's own id <group>__B would clash "
                    "with later models' picks in the shared soil output folder)")
    a = ap.parse_args(argv)
    chk = ag_map_check.check(a.map_root, [a.cases], L.ROOT)
    assert chk['ok'], chk['problems']
    import gen_planner as P
    import f104_n2_iter as IT
    P.DS.init_map(a.map_root)
    out = Path(a.out); (out / 'straight6/routes').mkdir(parents=True, exist_ok=True)
    picks = Path(a.picks)
    ps = json.load(open(picks / 'summary.json'))
    assert ps['arms'] == ['B'] and ps['tags']['B'] == 'n2iter_cem4x64' and not ps['fixed2'] and ps['world'] == 'crm', ps['arms']
    cases = sorted(p for p in Path(a.cases).glob('*.json') if p.name != 'cases.json')
    rows, s6 = [], {}
    case_rel = lambda g: f'cases/dev_{a.arena}/cases/{g}.json'
    k3_rel = lambda p: str(Path(p).resolve().relative_to(L.K3.resolve()))
    for cp in cases:
        case = json.load(open(cp)); g = case['id']; lay = case['layout']
        pose = [lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']]
        base = {k: np.asarray(v, float) for k, v in json.load(open(cp.parent / 'routes' / g / 'route_00.json')).items()
                if k in ('waypoints', 'speeds', 'stations', 'headings')}
        base['meta'] = {}
        c1, tries = P.proposal_pool(base, pose, np.random.default_rng(int(hashlib.md5((g + 'crm_proposal').encode()).hexdigest()[:8], 16)))
        i6 = P.anchor_index(c1, 0.0, 6.0)
        pb = json.load(open(picks / 'picks' / f'{g}.json'))['arms']['B']
        bid = pb['route_id']; run_id = f'{g}__{a.model_tag}_B'
        rows.append(dict(id=run_id, group=g, arena=a.arena, case=case_rel(g), route=k3_rel(picks / 'routes' / f'{bid}.json'),
                         tier=a.tier, episode_seed=L.md5_int(run_id), run=True, kind='dev_headroom', arms=[f'{a.model_tag}:B'],
                         sha256=pb['route_sha256']))
        if i6 is None:
            s6[g] = None; continue
        r = c1[i6]; h = IT.route_sha256(r)
        if h == pb['route_sha256']:
            rows[-1]['arms'].append('straight6'); s6[g] = dict(route_id=run_id, index=i6, route_sha256=h, same_as=f'{a.model_tag}:B'); continue
        rid = f'{g}__straight6'
        json.dump({'waypoints': np.asarray(r['waypoints']).tolist(), 'speeds': np.asarray(r['speeds']).tolist(),
                   'stations': np.asarray(r['stations']).tolist(), 'headings': np.asarray(r['headings']).tolist(),
                   'meta': {'candidate': 'crm_eval_straight6', 'scene_id': g, 'pool': 'proposal', 'cand_index': i6}},
                  open(out / 'straight6/routes' / f'{rid}.json', 'w'))
        s6[g] = dict(route_id=rid, index=i6, route_sha256=h, mean_speed=float(np.asarray(r['speeds'])[1:-1].mean()),
                     length_m=float(np.asarray(r['stations'])[-1]))
        rows.append(dict(id=rid, group=g, arena=a.arena, case=case_rel(g), route=k3_rel(out / 'straight6/routes' / f'{rid}.json'),
                         tier=a.tier, episode_seed=L.md5_int(rid), run=True, kind='dev_headroom', arms=['straight6'], sha256=h))
    json.dump(dict(arena=a.arena, map_root=a.map_root, map_check=chk, picks=s6), open(out / 'straight6/picks.json', 'w'), indent=1)
    ids = [r['id'] for r in rows]; assert len(set(ids)) == len(ids)
    json.dump(rows, open(out / 'tasks_dev.json', 'w'), indent=1)
    files = sorted(list((picks / 'routes').glob('*.json')) + list((out / 'straight6/routes').glob('*.json')), key=lambda p: p.name)
    lock = hashlib.sha256(); lines = []
    for p in files:
        lock.update(p.name.encode()); lock.update(hashlib.sha256(p.read_bytes()).digest())
        lines.append(f'{L.sha256_file(p)}  {k3_rel(p)}')
    (out / 'DEV_PICKS_LOCKED.sha256').write_text(lock.hexdigest() + '  (name + content of every route file of both arms, sorted by name)\n'
                                                 + '\n'.join(lines) + '\n')
    n_same = sum(1 for v in s6.values() if v and v.get('same_as'))
    print(json.dumps(dict(groups=len(cases), rows=len(rows), straight6_missing=sum(v is None for v in s6.values()),
                          straight6_same_as_B=n_same, lock=lock.hexdigest(), map_check_ok=chk['ok']), indent=1))


if __name__ == '__main__':
    main()
