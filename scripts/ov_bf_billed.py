#!/usr/bin/env python3
"""Billed node-hours since a start time (offroad_vehicles_20260927, module M3; PLAN section 5: soft cap 150, hard 180).

A copy of ag_bf_billed.py with a weight for every partition of the cluster: billed = elapsed node-hours x 10 x the
partition's TRESBillingWeights NODE value in /etc/slurm/slurm.conf (read 2026-09-27 23:50):
  mi2101x 0.1, devel 0.1, mi2104x 0.4, mi2508x 0.8, mi3001x 0.125, mi3501x 0.125, mi3008x 1.0, mi3258x 1.2,
  mi3508x 1.2.
sacct -X (allocations only, running jobs up to now). Every job of the user counts (this study's and any other
session's, e.g. the rg_* jobs): the budget is shared. The split by job-name prefix shows what this study used (its job
names start with ov_). --check-weights re-reads slurm.conf on the cluster and stops if a weight changed.
Run locally (calls ssh amd):
  python3 scripts/ov_bf_billed.py [--since 2026-09-27T22:00] [--jobs 441600,441601] [--json out.json] [--check-weights]
"""
import argparse, json, re, subprocess, time
from collections import defaultdict

W = dict(mi2101x=0.1, devel=0.1, mi2104x=0.4, mi2508x=0.8, mi3001x=0.125, mi3501x=0.125, mi3008x=1.0, mi3258x=1.2,
         mi3508x=1.2)
SOFT, HARD = 150.0, 180.0


def ssh(cmd):
    return subprocess.run(['ssh', 'amd', cmd], capture_output=True, text=True, check=True).stdout


def check_weights():
    conf = ssh('cat /etc/slurm/slurm.conf')
    found = {}
    for m in re.finditer(r'^PartitionName=(\S+)[^\n]*(?:\\\n[^\n]*)*?TRESBillingWeights="NODE=([0-9.]+)"', conf, re.M):
        found[m.group(1)] = round(10 * float(m.group(2)), 6)
    bad = {p: (W.get(p), w) for p, w in found.items() if p in W and abs(W[p] - w) > 1e-9}
    missing = sorted(p for p in found if p not in W and p not in ('staff',))
    return found, bad, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', default='2026-09-27T22:00')
    ap.add_argument('--jobs', default=None, help='comma-separated job ids: bill only these (array jobs: the master id)')
    ap.add_argument('--json', default=None)
    ap.add_argument('--check-weights', action='store_true')
    a = ap.parse_args()
    if a.check_weights:
        found, bad, missing = check_weights()
        print('slurm.conf weights x10:', found)
        if bad or missing:
            raise SystemExit(f'weights changed {bad} / partitions without a weight here {missing}')
    q = f"sacct -u harry -X -S {a.since} -n -P --format=JobID,JobName,Partition,ElapsedRaw,NNodes,State"
    out = ssh(q)
    tot, per, jobs, names, unknown = 0.0, defaultdict(float), defaultdict(float), defaultdict(float), set()
    want = set(a.jobs.split(',')) if a.jobs else None
    for line in out.splitlines():
        if not line.strip():
            continue
        jid, name, part, el, nn, state = line.split('|')
        master = jid.split('_')[0].split('.')[0]
        if want is not None and master not in want:
            continue
        if part not in W:
            unknown.add(part); continue
        b = int(el or 0) / 3600 * int(nn or 1) * W[part]
        tot += b; per[part] += b; jobs[master] += b
        names[name.split('_')[0] + '_' if '_' in name else name] += b
    res = dict(time=time.strftime('%F %T'), since=a.since, billed=round(tot, 3), soft_cap=SOFT, hard_cap=HARD,
               by_partition={k: round(v, 3) for k, v in sorted(per.items())},
               by_name_prefix={k: round(v, 3) for k, v in sorted(names.items())},
               this_study_ov=round(sum(v for k, v in names.items() if k.startswith('ov_')), 3),
               unknown_partitions=sorted(unknown))
    if want:
        res['by_job'] = {k: round(v, 3) for k, v in sorted(jobs.items())}
    print(f'billed node-hours since {a.since}' + (f' for jobs {a.jobs}' if want else '') + f': {tot:.2f} '
          f'(soft cap {SOFT:.0f}, hard cap {HARD:.0f}; ov_* jobs {res["this_study_ov"]:.2f})')
    print('by partition:', res['by_partition'])
    print('by job-name prefix:', res['by_name_prefix'])
    if want:
        print('by job:', res['by_job'])
    if unknown:
        print('partitions without a weight (NOT counted):', sorted(unknown))
    if tot >= HARD:
        print('HARD CAP REACHED')
    elif tot >= SOFT:
        print('soft cap reached')
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
