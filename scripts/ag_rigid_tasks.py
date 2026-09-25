#!/usr/bin/env python3
"""Rigid (HMMWV) collection tasks for arena_gator_20260925 (E3a), rows for gen_runner_g.py / gen_array_g.sbatch.

  training pools : every group of cases/train_<a>/cases x 20 routes (12 designed + 8 on-policy) for each --arenas,
                   ids <group>_route_NN / <group>_op_NN (= the soil twin ids), tier = the soil tier of that route
                   (ag_tasklib.group_routes, the crm_tasks.py order), episode_seed = md5(id)[:8] (provenance only; the
                   rigid collector does not read it), shard = md5(group) % --train-shards (all routes of a group on one
                   node).
  test designed  : (REVIEW_R1 amendment 5c) the 12 designed routes of every group of cases/test_<a>/cases for each
                   --test-arenas, ids <group>_route_NN (blacklisted by the <a>_test_group_* pattern), shards after the
                   training shards (--test-shards), so they run after the training shards and can be cancelled alone.
  drift          : (REVIEW_R1 amendment 7b) --drift N designed f104 night-2 ids re-driven on tonight's tree, ids
                   drift__<orig id>, case/route = the absolute cluster paths of production_v3's own inputs, all in
                   shard 0 (one node); compare with production_v3 outcomes (scripts/ag_drift_check.py).
Rows: {id, group, arena, case, route, shard, run, tier, episode_seed, kind, split[, ref_id]}; paths relative to G3
(= GEN_ROOT) unless absolute. Every referenced file is checked to exist in the local K3 copy of the staged folders.

  PYTHONPATH=src:scripts python scripts/ag_rigid_tasks.py --out artifacts/traverse/arena_gator_20260925/e3/tasks/rigid_hmmwv_v1.json
"""
import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ag_tasklib as L

F104_CAMPAIGN = '/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909'
F104_LOCAL = L.ROOT / 'artifacts/traverse/fdm_f104_50h_20260909'


def shard_of(g, k, offset=0):
    return offset + int(hashlib.md5(g.encode()).hexdigest(), 16) % k


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--arenas', nargs='+', default=['g203', 'g228'])
    ap.add_argument('--groups', type=int, default=0, help='first N groups per training arena (0 = all)')
    ap.add_argument('--train-shards', type=int, default=12)
    ap.add_argument('--test-arenas', nargs='*', default=['g260', 'g271', 'g251', 'g247'])
    ap.add_argument('--test-shards', type=int, default=2)
    ap.add_argument('--drift', type=int, default=200, help='f104 night-2 designed ids re-driven in shard 0')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)

    rows = []
    for arena in a.arenas:
        des, onp = L.train_dirs(arena)
        opi = L.onpolicy_index(arena)
        recs = L.records(des)
        for g in L.group_ids(des, a.groups or None):
            case = f'{des}/{g}.json'
            for tier, rid, route, kind in L.group_routes(g, des, onp, opi):
                rows.append(dict(id=rid, group=g, arena=arena, case=case, route=route, shard=shard_of(g, a.train_shards),
                                 run=True, tier=tier, episode_seed=L.md5_int(rid), kind=kind, split=recs[g]['split']))
    for arena in a.test_arenas:
        des = f'cases/test_{arena}/cases'
        recs = L.records(des)
        for g in L.group_ids(des):
            for k in range(12):
                rid = f'{g}_route_{k:02d}'
                rows.append(dict(id=rid, group=g, arena=arena, case=f'{des}/{g}.json', route=f'{des}/routes/{g}/route_{k:02d}.json',
                                 shard=shard_of(g, a.test_shards, a.train_shards), run=True, tier=k, episode_seed=L.md5_int(rid),
                                 kind='test_designed', split=recs[g]['split']))
    if a.drift:
        runs = sorted(p.name for p in (F104_LOCAL / 'production_v3/runs').iterdir() if '_route_' in p.name
                      and (p / 'outcome.json').exists())
        pick = sorted(runs, key=lambda r: hashlib.sha256(r.encode()).hexdigest())[:a.drift]
        for rid in sorted(pick):
            g, k = rid.split('_route_')
            rows.append(dict(id=f'drift__{rid}', group=g, arena='f104', case=f'{F104_CAMPAIGN}/cases_night2_v1/{g}.json',
                             route=f'{F104_CAMPAIGN}/cases_night2_v1/routes/{g}/route_{k}.json', shard=0, run=True, tier=-1,
                             episode_seed=L.md5_int(rid), kind='drift', split=json.load(open(F104_LOCAL / 'production_v3/runs' / rid / 'case.json'))['split'],
                             ref_id=rid, ref_runs=f'{F104_CAMPAIGN}/production_v3/runs'))

    ids = [r['id'] for r in rows]
    assert len(set(ids)) == len(ids), 'duplicate ids'
    missing = [r for r in rows for p in (r['case'], r['route']) if L.local_path(p) is not None and not L.local_path(p).exists()]
    assert not missing, f'{len(missing)} referenced files missing locally, e.g. {missing[:2]}'
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, 'w'))
    meta = dict(tool='scripts/ag_rigid_tasks.py', tool_sha256=L.sha256_file(__file__), argv=sys.argv[1:] if argv is None else argv,
                n_rows=len(rows), file_sha256=L.sha256_file(out), by_kind=dict(Counter(r['kind'] for r in rows)),
                by_arena=dict(Counter(r['arena'] for r in rows)), by_shard=dict(sorted(Counter(r['shard'] for r in rows).items())),
                n_shards=a.train_shards + a.test_shards, by_split=dict(Counter(r['split'] for r in rows)))
    json.dump(meta, open(str(out) + '.meta.json', 'w'), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
