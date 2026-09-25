#!/usr/bin/env python3
"""Progress of an arena_gator_20260925 collection (soil or rigid), for the monitor. Plain python3 (3.9), json only;
run on the cluster login node:

  python3 ag_collect_status.py --tasks $G3/tasks/soil_v1.json --out $G3/soil_v1 [--json summary.json]

Per kind (dev_headroom / drift / designed / on_policy / test_designed ...) and arena: rows, complete runs, failed ids
(attempts), status mix; per training tier: complete / rows (the tier the collection is in); simulated seconds and
worker wall seconds (from episode_complete.json / outcome.json wall_s) -> wall per simulated second; count of
collection_failure.json files; claims without a complete run (in flight or stale).
"""
import argparse, json, os, time
from collections import Counter, defaultdict


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--json')
    a = ap.parse_args()
    rows = [r for r in load(a.tasks) if r.get('run', True)]
    runs = os.path.join(a.out, 'runs')
    have = set(os.listdir(runs)) if os.path.isdir(runs) else set()
    failed = set(n[:-5] for n in os.listdir(os.path.join(a.out, 'failed'))) if os.path.isdir(os.path.join(a.out, 'failed')) else set()
    claims = set(os.listdir(os.path.join(a.out, 'claims'))) if os.path.isdir(os.path.join(a.out, 'claims')) else set()
    by = defaultdict(lambda: dict(rows=0, complete=0, failed_ids=0, status=Counter(), sim_s=0.0, wall_s=0.0))
    tiers = defaultdict(lambda: [0, 0])
    coll_fail = 0
    for r in rows:
        k = f"{r.get('kind', '?')}:{r.get('arena', '?')}"
        b = by[k]; b['rows'] += 1
        tid = r['id']
        if r.get('kind') in ('designed', 'on_policy'):
            tiers[r['tier']][0] += 1
        done = tid in have and os.path.exists(os.path.join(runs, tid, 'episode_complete.json'))
        if tid in have and os.path.exists(os.path.join(runs, tid, 'collection_failure.json')):
            coll_fail += 1
        if done:
            b['complete'] += 1
            if r.get('kind') in ('designed', 'on_policy'):
                tiers[r['tier']][1] += 1
            o = load(os.path.join(runs, tid, 'outcome.json')) or {}
            b['status'][o.get('status', '?')] += 1
            b['sim_s'] += float(o.get('elapsed_s', 0) or 0)
            b['wall_s'] += float(o.get('wall_s', 0) or 0)
        elif tid in failed:
            b['failed_ids'] += 1
    in_flight = len([c for c in claims if not os.path.exists(os.path.join(runs, c, 'episode_complete.json'))])
    tot = dict(rows=len(rows), complete=sum(b['complete'] for b in by.values()), failed_ids=len(failed),
               collection_failure_files=coll_fail, claims_without_complete_run=in_flight,
               sim_h=round(sum(b['sim_s'] for b in by.values()) / 3600, 2),
               stop_claims=os.path.exists(os.path.join(a.out, 'STOP_CLAIMS')), time=time.strftime('%F %T'))
    print(json.dumps(tot))
    for k in sorted(by):
        b = by[k]
        wps = b['wall_s'] / b['sim_s'] if b['sim_s'] else None
        print(f"{k:28s} {b['complete']:6d}/{b['rows']:<6d} failed {b['failed_ids']:4d}  sim {b['sim_s'] / 3600:7.2f} h  "
              f"wall/sim {wps if wps is None else round(wps, 2)}  " + ' '.join(f'{s}={n}' for s, n in b['status'].most_common()))
    if tiers:
        print('training tiers (complete/rows):', ' '.join(f'{t}:{c}/{n}' for t, (n, c) in sorted(tiers.items())))
    if a.json:
        json.dump(dict(total=tot, by={k: dict(v, status=dict(v['status'])) for k, v in by.items()},
                       tiers={str(t): dict(rows=n, complete=c) for t, (n, c) in sorted(tiers.items())}), open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
