#!/usr/bin/env python3
"""Per-drive extras of the final E6b rigid index (arena_gator_20260925, 09-26) from the login-node record
(scripts/ag_e6b_host_extract.py -> e6/index/rigid_run_hosts.json) instead of local copies of simulation_provenance.json
and rich_intervals.npz. Writes the same schema as scripts/ag_e6b_extras.py (summary / runs / multi_host_groups, keyed by
the index's run_dir), so scripts/ag_e6b_results.py reads it unchanged, and adds the pool-record checks.

  PYTHONPATH=src:scripts python scripts/ag_e6b_final_extras.py --index $K3/e6/index/rigid_eval_v1.json \
      --hosts $K3/e6/index/rigid_run_hosts.json --pool-done $K3/e6/pool/done --out $K3/e6/index/rigid_eval_v1_extras.json

Checks (VERIFY_E6a point 2: Chrono rigid is deterministic per node only, so every drive of a group must come from one
host): per group, the set of hosts over every driven arm (both vehicles); per shard (pool done records), complete ==
rows, no incomplete rows, no host mismatch, no takeover; every run of the index has a host.
"""
import argparse, json, os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True); ap.add_argument('--hosts', required=True)
    ap.add_argument('--pool-done', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    idx = json.load(open(a.index))
    rows = [r for r in idx['rows'] if not r.get('missing')]
    H = json.load(open(a.hosts))['runs']
    ex_ = {}
    for r in rows:
        rid = Path(r['run_dir']).name
        h = H.get(rid)
        ex_[r['run_dir']] = dict(host=None if h is None else h['host'], chassis_contact_n=None if h is None else h['chassis_contact_n'],
                                 belly_min_m=None if h is None else h['belly_min_m'], belly_frames_below=None if h is None else h['belly_frames_below'],
                                 complete_on_cluster=None if h is None else h['complete'])
    dirs = sorted(ex_)
    hosts = defaultdict(set)
    for r in rows:
        hosts[r['group']].add(ex_[r['run_dir']]['host'])
    multi = {g: sorted(str(h) for h in v) for g, v in hosts.items() if len(v) > 1}
    nohost = sum(1 for d in dirs if ex_[d]['host'] is None)
    contact = defaultdict(list)
    for r in rows:
        c = ex_[r['run_dir']]['chassis_contact_n']
        if c is not None:
            contact[f"{r['vehicle']}|{r['arm']}|{r['set']}"].append(c > 0)
    # pool records
    recs = [json.load(open(os.path.join(a.pool_done, f))) for f in sorted(os.listdir(a.pool_done)) if f.endswith('.json')]
    pool = dict(shards=len(recs), rows=sum(x['rows'] for x in recs), complete=sum(x['complete'] for x in recs),
                shards_incomplete=[x['shard'] for x in recs if x['incomplete']], shards_host_mismatch=[x['shard'] for x in recs if x['host_mismatch']],
                shards_takeover=[x['shard'] for x in recs if x['takeover']], shards_multi_run_host=[x['shard'] for x in recs if len(x['run_hosts']) > 1],
                shards_second_attempt=[x['shard'] for x in recs if len(x['runner_rc']) > 1], runner_rc_nonzero=[x['shard'] for x in recs if any(x['runner_rc'])],
                shards_by_task_file=dict(Counter(os.path.basename(x['tasks']) for x in recs)), nodes=len({x['host'] for x in recs}),
                wall_h_sum=sum(x['wall_s'] for x in recs) / 3600)
    # the shard host (pool record) equals every run's recorded host
    shard_host = {}
    for x in recs:
        tf = x['tasks']
        shard_host[(os.path.basename(tf), x['shard'])] = x['host'].split('.')[0]
    summ = dict(driven_rows=len(rows), distinct_runs=len(dirs), runs_without_host=nohost, groups=len(hosts), groups_multi_host=len(multi),
                multi_host_examples=dict(list(multi.items())[:20]), hosts=dict(Counter(ex_[d]['host'] for d in dirs)),
                runs_not_complete_on_cluster=sum(1 for d in dirs if ex_[d]['complete_on_cluster'] is False),
                chassis_contact_share={k: dict(n=len(v), share=100 * float(np.mean(v))) for k, v in sorted(contact.items())},
                pool=pool, source='login-node record scripts/ag_e6b_host_extract.py (simulation_provenance.json host, rich_intervals.npz max chassis contact)')
    json.dump(dict(summary=summ, runs=ex_, multi_host_groups=multi), open(a.out, 'w'), indent=None, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k not in ('chassis_contact_share', 'hosts')}, indent=1, default=float))


if __name__ == '__main__':
    main()
