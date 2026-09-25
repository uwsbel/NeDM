#!/usr/bin/env python3
"""Rigid task file v2 for arena_gator_20260925 (module E3b2): every row of rigid_hmmwv_v1.json unchanged (shards 0-13,
run by job 436075 from that file) plus

  shards 1000-1299  the Gator rigid pool: the 24,000 f104 night-2 pool routes behind every current rigid model
                    (production_v3 = 14,400 designed route_NN, production_v4 = 9,600 on-policy op_NN; case / route =
                    the absolute fdm_f104_50h_20260909 paths of those task files), id 'gator__<id>', extra
                    ['--vehicle', 'gator', '--runtime-fingerprint', G3/runtime/gator_runtime_fingerprint.json] (the
                    pilot's rows), tier = the crm_tasks.py tier of the same id (the collect_v1 task file), split = the
                    case's split. 4 groups (80 routes) per shard: the 1,200 groups sorted by md5(group id), consecutive
                    blocks of 4.
  shards 2000-2124  the 12 designed routes of every spread test group (g258 g268 g263 g241), the rows of
                    e3/tasks/rigid_spread_designed_rows.json with only 'shard' changed (8 groups = 96 routes per shard,
                    groups sorted by md5(group id)). HMMWV rows. They must run from the second source tree G3/r2/source
                    (its gen_arenas.json lists the spread arenas; G3/source's frozen copy does not).
Every route of a group is in one shard, and a shard runs on one node (scripts/ag_rigid_pool.py), so rows of one group
share a node. Small shards so that pool steps inside the soil allocations can share the work.
Checks: superset of rigid_hmmwv_v1.json, unique ids, one shard per group, the Gator ids = 'gator__' + the 24,000 pool
ids, case/route sha256 of the production task files recorded in the rows (case_sha256 / route_sha256), relative paths
exist in K3.
  PYTHONPATH=src:scripts python scripts/ag_rigid_tasks_v2.py --out artifacts/traverse/arena_gator_20260925/e3/tasks/rigid_v2.json
"""
import argparse, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ag_tasklib as L

F104 = '/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909'
F104_LOCAL = L.ROOT / 'artifacts/traverse/fdm_f104_50h_20260909'
FINGERPRINT = f'{L.G3}/runtime/gator_runtime_fingerprint.json'
V1 = L.K3 / 'e3/tasks/rigid_hmmwv_v1.json'
V1_SHA = '168044497dd338de'
SPREAD = L.K3 / 'e3/tasks/rigid_spread_designed_rows.json'
SPREAD_SHA = '99b50c56'
F104_TASKS = L.K3 / 'e3/tasks/ref/crm_f104_tasks_train.json'
PILOT = L.K3 / 'e3/tasks/pilot_gator_rigid.json'
GATOR_BASE, GATOR_PER = 1000, 4
SPREAD_BASE, SPREAD_PER = 2000, 8


def md5hex(s):
    return hashlib.md5(s.encode()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--prod-v3', default='/tmp/ag_e3b2/prod_v3_tasks.json', help='scp of fdm_f104_50h_20260909/production_v3/tasks.json')
    ap.add_argument('--prod-v4', default='/tmp/ag_e3b2/prod_v4_tasks.json', help='scp of fdm_f104_50h_20260909/production_v4/tasks.json')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    assert L.sha256_file(V1).startswith(V1_SHA) and L.sha256_file(SPREAD).startswith(SPREAD_SHA)
    v1 = json.load(open(V1))
    tier = {r['id']: r['tier'] for r in json.load(open(F104_TASKS))}
    stratum = {r['scene_id']: r['evaluation_stratum'] for r in json.load(open(F104_LOCAL / 'cases_night2/cases/cases.json'))['records']}
    prod = [(r, 'production_v3') for r in json.load(open(a.prod_v3))['tasks']] + [(r, 'production_v4') for r in json.load(open(a.prod_v4))['tasks']]
    assert len(prod) == 24000 and len({r['id'] for r, _ in prod}) == 24000
    groups = sorted({r['group_id'] for r, _ in prod}, key=md5hex)
    assert len(groups) == 1200
    gshard = {g: GATOR_BASE + i // GATOR_PER for i, g in enumerate(groups)}
    gator = []
    for r, src in sorted(prod, key=lambda x: x[0]['id']):
        rid, g = r['id'], r['group_id']
        op = '_op_' in rid
        k = int(rid.rsplit('_', 1)[1])
        assert r['case'] == f'{F104}/cases_night2_v1/{g}.json'
        assert r['route'] == (f'{F104}/cases_night2_onpolicy_v1/routes/{g}/op_{k:02d}.json' if op else f'{F104}/cases_night2_v1/routes/{g}/route_{k:02d}.json')
        row = dict(id='gator__' + rid, pair_id=rid, group=g, stratum=stratum[g], arena='f104', case=r['case'], route=r['route'],
                   shard=gshard[g], run=True, tier=tier[rid], episode_seed=L.md5_int(rid),
                   kind='on_policy' if op else 'designed', split=r['split'], vehicle='gator',
                   extra=['--vehicle', 'gator', '--runtime-fingerprint', FINGERPRINT],
                   ref_runs=f'{F104}/{src}/runs', case_sha256=r['case_sha256'], route_sha256=r['route_sha256'])
        if not op:
            row['profile'] = k % 4
        gator.append(row)
    # the pilot's 288 rows are the same routes (only shard and tier differ: pilot shard 0/1, pilot tier = route index)
    pil = {r['id']: r for r in json.load(open(PILOT))}
    gm = {r['id']: r for r in gator}
    for i, p in pil.items():
        q = gm[i]
        assert all(p[k] == q[k] for k in ('pair_id', 'group', 'arena', 'case', 'route', 'extra', 'kind', 'vehicle', 'split')), i
    spread_rows = json.load(open(SPREAD))
    sgroups = sorted({r['group'] for r in spread_rows}, key=md5hex)
    assert len(sgroups) == 1000
    sshard = {g: SPREAD_BASE + i // SPREAD_PER for i, g in enumerate(sgroups)}
    spread = []
    for r in spread_rows:
        q = dict(r)
        q['shard'] = sshard[r['group']]
        spread.append(q)
    rows = v1 + gator + spread
    ids = [r['id'] for r in rows]
    assert len(set(ids)) == len(ids), 'duplicate ids'
    new = {r['id']: r for r in rows}
    changed = [r['id'] for r in v1 if new.get(r['id']) != r]
    assert not changed, changed[:3]
    gs = defaultdict(set)
    for r in rows:
        gs[(r['group'], r['id'].startswith('gator__'), r['id'].startswith('drift__'))].add(r['shard'])
    assert all(len(v) == 1 for v in gs.values()), 'a group spans shards'
    assert not ({r['shard'] for r in v1} & {r['shard'] for r in gator + spread})
    missing = [(r['id'], p) for r in rows for p in (r['case'], r['route']) if L.local_path(p) is not None and not L.local_path(p).exists()]
    assert not missing, missing[:2]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, 'w'))
    sh = Counter(r['shard'] for r in gator + spread)
    meta = dict(tool='scripts/ag_rigid_tasks_v2.py', tool_sha256=L.sha256_file(__file__), n_rows=len(rows),
                file_sha256=L.sha256_file(out), superset_of=dict(rigid_hmmwv_v1=L.sha256_file(V1)),
                inputs=dict(prod_v3=L.sha256_file(a.prod_v3), prod_v4=L.sha256_file(a.prod_v4), spread=L.sha256_file(SPREAD),
                            f104_tasks=L.sha256_file(F104_TASKS)),
                blocks=dict(v1=dict(rows=len(v1), shards='0-13'),
                            gator=dict(rows=len(gator), shards=f'{GATOR_BASE}-{max(gshard.values())}', groups_per_shard=GATOR_PER,
                                       by_kind=dict(Counter(r['kind'] for r in gator)), by_split=dict(Counter(r['split'] for r in gator)),
                                       by_tier=dict(sorted(Counter(r['tier'] for r in gator).items()))),
                            spread=dict(rows=len(spread), shards=f'{SPREAD_BASE}-{max(sshard.values())}', groups_per_shard=SPREAD_PER,
                                        by_arena=dict(Counter(r['arena'] for r in spread)))),
                rows_per_new_shard=dict(min=min(sh.values()), max=max(sh.values())),
                pilot_rows_same_routes=len(pil))
    json.dump(meta, open(str(out) + '.meta.json', 'w'), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
