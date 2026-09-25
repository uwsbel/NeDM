#!/usr/bin/env python3
"""Soil (CRM) task file for arena_gator_20260925 (E3a): a parameterised copy of scripts/crm_tasks.py.

Row order and tiers (crm_worker.py sorts each worker's shuffled rows by tier, so lower tiers always run first):
  --head FILE ...   pre-built priority rows, placed first, must carry tier < 0 (the g217 dev headroom rows from
                    ag_dev_headroom.py: tier -1)
  --drift N         (REVIEW_R1 amendment 7b) N f104 soil ids of collect_v1 re-driven on tonight's tree, ids
                    drift__<orig id>, case/route = the absolute crm_f104_20260916 paths of the original rows, the original
                    episode_seed, tier -1; compare with collect_v1 (scripts/ag_drift_check.py)
  training rows     for each --arenas: the first --groups groups of cases/train_<a>/cases (per-arena override
                    --groups-for g228=634), 20 routes per group (12 designed + 8 on-policy) in the crm_tasks.py per-group
                    shuffled order (ag_tasklib.group_routes, seeded by md5 of the group), kept up to --max-tier (tier 12 =
                    13 routes per group; nothing above it is written); ids <group>_route_NN / <group>_op_NN (= the rigid
                    twin ids), episode_seed = md5(id)[:8] as in crm_tasks.py. Written tier by tier, and within a tier
                    group index by group index alternating the arenas, so tier k of every arena precedes tier k+1.
  --append FILE ... rows added later (e.g. Gator rows for the relaunch; their ids must carry a prefix such as
                    'gator__' so they never equal an HMMWV id), kept as given after the training rows
  --check-superset OLD  assert that every row of an earlier task file is present unchanged (relaunch safety: the
                    worker skips completed ids, so ids and their case/route/seed must never change)
Paths are relative to G3 (= CRM_ROOT) unless absolute. Every referenced file must exist in the local K3 copy of the
staged folders (absolute paths outside G3 are not checked here). Writes <out> and <out>.meta.json (counts, sha256).
"""
import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ag_tasklib as L

CRM_F104 = '/work1/dannegrut/harry/experiments/crm_f104_20260916'
CRM_F104_LOCAL = L.ROOT / 'artifacts/traverse/crm_f104_v1/collect_v1'
F104_TASKS = '/tmp/ag_e3a/crm_f104_tasks_train.json'   # scp of $CRM_F104/tasks_train.json


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--arenas', nargs='+', default=['g203', 'g228'])
    ap.add_argument('--groups', type=int, default=606)
    ap.add_argument('--groups-for', nargs='*', default=[], help='per-arena override, e.g. g228=634')
    ap.add_argument('--max-tier', type=int, default=12)
    ap.add_argument('--head', nargs='*', default=[])
    ap.add_argument('--drift', type=int, default=10)
    ap.add_argument('--f104-tasks', default=F104_TASKS)
    ap.add_argument('--append', nargs='*', default=[])
    ap.add_argument('--check-superset')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    n_for = {k: int(v) for k, v in (s.split('=') for s in a.groups_for)}

    rows = []
    for f in a.head:
        head = json.load(open(f))
        assert all(r['tier'] < 0 for r in head), f'{f}: head rows must have tier < 0'
        rows += head
    if a.drift:
        ref = {r['id']: r for r in json.load(open(a.f104_tasks))}
        qa = json.load(open(CRM_F104_LOCAL / 'qa.json'))
        flagged = {f['id'] if isinstance(f, dict) else f for f in qa.get('flagged_ids', [])}
        done = sorted(p.name for p in (CRM_F104_LOCAL / 'runs').iterdir() if p.name not in flagged and (p / 'outcome.json').exists())
        for rid in sorted(sorted(done, key=lambda r: hashlib.sha256(r.encode()).hexdigest())[:a.drift]):
            r = ref[rid]
            rows.append(dict(id=f'drift__{rid}', group=r['group'], arena='f104', case=f"{CRM_F104}/{r['case']}",
                             route=f"{CRM_F104}/{r['route']}", tier=-1, episode_seed=r['episode_seed'], run=True, kind='drift',
                             ref_id=rid, ref_runs=f'{CRM_F104}/collect_v1/runs', vehicle='hmmwv'))
    per_arena = {}
    for arena in a.arenas:
        des, onp = L.train_dirs(arena)
        opi = L.onpolicy_index(arena)
        recs = L.records(des)
        groups = L.group_ids(des, n_for.get(arena, a.groups))
        per_arena[arena] = [(g, recs[g]['split'], L.group_routes(g, des, onp, opi)) for g in groups]
    width = max(len(v) for v in per_arena.values())
    for tier in range(a.max_tier + 1):
        for i in range(width):
            for arena in a.arenas:
                if i >= len(per_arena[arena]):
                    continue
                g, split, routes = per_arena[arena][i]
                t, rid, route, kind = routes[tier]
                assert t == tier
                rows.append(dict(id=rid, group=g, arena=arena, case=f'{L.train_dirs(arena)[0]}/{g}.json', route=route, tier=tier,
                                 episode_seed=L.md5_int(rid), run=True, kind=kind, split=split, vehicle='hmmwv'))
    for f in a.append:
        rows += json.load(open(f))

    ids = [r['id'] for r in rows]
    assert len(set(ids)) == len(ids), 'duplicate ids'
    assert len({r['episode_seed'] for r in rows}) == len(rows), 'duplicate episode seeds'
    assert max(r['tier'] for r in rows if r.get('kind') in ('designed', 'on_policy')) <= a.max_tier
    missing = [(r['id'], p) for r in rows for p in (r['case'], r['route']) if L.local_path(p) is not None and not L.local_path(p).exists()]
    assert not missing, f'{len(missing)} referenced files missing locally, e.g. {missing[:2]}'
    if a.check_superset:
        new = {r['id']: r for r in rows}
        old = json.load(open(a.check_superset))
        changed = [r['id'] for r in old if new.get(r['id']) != r]
        assert not changed, f'{len(changed)} rows of {a.check_superset} missing or changed, e.g. {changed[:3]}'
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, 'w'))
    by_tier = Counter((r.get('arena'), r['tier']) for r in rows)
    meta = dict(tool='scripts/ag_soil_tasks.py', tool_sha256=L.sha256_file(__file__), argv=sys.argv[1:] if argv is None else argv,
                n_rows=len(rows), file_sha256=L.sha256_file(out), by_kind=dict(Counter(r.get('kind') for r in rows)),
                by_arena=dict(Counter(r.get('arena') for r in rows)),
                groups={k: len(v) for k, v in per_arena.items()},
                train_groups={k: sum(1 for _, s, _ in v if s == 'train') for k, v in per_arena.items()},
                by_arena_tier={f'{k[0]}:{k[1]}': n for k, n in sorted(by_tier.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))},
                check_superset=a.check_superset)
    json.dump(meta, open(str(out) + '.meta.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != 'by_arena_tier'}, indent=1))


if __name__ == '__main__':
    main()
