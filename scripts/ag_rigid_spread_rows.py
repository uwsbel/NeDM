#!/usr/bin/env python3
"""Rigid rows for the 12 designed routes of every spread-arena test group (arena_gator_20260925, E1b; PLAN 7.1/7.6).

The same rows as the near arenas' 'test_designed' block of rigid_hmmwv_v1.json (scripts/ag_rigid_tasks.py): ids
<group>_route_NN (blacklist pattern <arena>_test_group_*), case/route relative to G3, tier = route index,
episode_seed = md5(id)[:8], kind 'test_designed', split copied from cases.json, shard = --shard-offset +
md5(group) % --shards (all routes of a group on one node). The default offset 14 puts them after the 14 shards of
rigid_hmmwv_v1.json, so the rows can be appended to that file (superset) or run alone (--array=14-15). Ids are
asserted unique against --against.

  PYTHONPATH=src:scripts python scripts/ag_rigid_spread_rows.py
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ag_tasklib as L
from ag_rigid_tasks import shard_of


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-arenas', nargs='+', default=['g258', 'g268', 'g263', 'g241'])
    ap.add_argument('--shards', type=int, default=2)
    ap.add_argument('--shard-offset', type=int, default=14)
    ap.add_argument('--against', nargs='*', default=[str(L.K3 / 'e3/tasks/rigid_hmmwv_v1.json')])
    ap.add_argument('--out', default=str(L.K3 / 'e3/tasks/rigid_spread_designed_rows.json'))
    a = ap.parse_args(argv)
    rows = []
    for arena in a.test_arenas:
        des = f'cases/test_{arena}/cases'
        recs = L.records(des)
        for g in L.group_ids(des):
            for k in range(12):
                rid = f'{g}_route_{k:02d}'
                rows.append(dict(id=rid, group=g, arena=arena, case=f'{des}/{g}.json', route=f'{des}/routes/{g}/route_{k:02d}.json',
                                 shard=shard_of(g, a.shards, a.shard_offset), run=True, tier=k, episode_seed=L.md5_int(rid),
                                 kind='test_designed', split=recs[g]['split']))
    ids = [r['id'] for r in rows]
    assert len(set(ids)) == len(ids), 'duplicate ids'
    for f in a.against:
        old = json.load(open(f))
        assert not set(ids) & {r['id'] for r in old}, f'id clash with {f}'
        assert not {r['shard'] for r in rows} & {r['shard'] for r in old}, f'shard clash with {f}'
    missing = [r for r in rows for p in (r['case'], r['route']) if not L.local_path(p).exists()]
    assert not missing, f'{len(missing)} referenced files missing locally, e.g. {missing[:2]}'
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, 'w'))
    meta = dict(tool='scripts/ag_rigid_spread_rows.py', tool_sha256=L.sha256_file(__file__),
                argv=sys.argv[1:] if argv is None else argv, n_rows=len(rows), file_sha256=L.sha256_file(out),
                by_arena=dict(Counter(r['arena'] for r in rows)), by_shard=dict(sorted(Counter(r['shard'] for r in rows).items())),
                by_split=dict(Counter(r['split'] for r in rows)))
    json.dump(meta, open(str(out) + '.meta.json', 'w'), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
