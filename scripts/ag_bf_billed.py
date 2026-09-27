#!/usr/bin/env python3
"""Billed node-hours of this session from sacct (arena_gator_20260925, task B stage 2 record-keeping).

Billed = sum over harry's jobs (sacct -X, allocations only) of elapsed node-hours x the partition weight measured in
NOTES_S1 section 2 (the ledger moves by these amounts per node-hour): mi2101x 0.1, mi2104x 0.4, mi3501x 0.125,
mi2508x 0.8, devel 0.1 (MI210). With start 2026-09-25T00:00 this reproduces the 108.2 recorded in LOG.md at 19:18 on 09-26.
Run locally (calls ssh amd):
  python3 scripts/ag_bf_billed.py [--since 2026-09-25T00:00] [--jobs 439361,439362,...]
"""
import argparse, subprocess
from collections import defaultdict

W = dict(mi2101x=0.1, mi2104x=0.4, mi3501x=0.125, mi2508x=0.8, devel=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', default='2026-09-25T00:00')
    ap.add_argument('--jobs', default=None, help='comma-separated job ids: bill only these (array jobs: the master id)')
    a = ap.parse_args()
    q = f"sacct -u harry -X -S {a.since} -n -P --format=JobID,Partition,ElapsedRaw,NNodes,State"
    out = subprocess.run(['ssh', 'amd', q], capture_output=True, text=True, check=True).stdout
    tot, per, jobs, unknown = 0.0, defaultdict(float), defaultdict(float), set()
    want = set(a.jobs.split(',')) if a.jobs else None
    for line in out.splitlines():
        jid, part, el, nn, state = line.split('|')
        master = jid.split('_')[0].split('.')[0]
        if want is not None and master not in want:
            continue
        if part not in W:
            unknown.add(part); continue
        b = int(el or 0) / 3600 * int(nn or 1) * W[part]
        tot += b; per[part] += b; jobs[master] += b
    print(f'billed node-hours since {a.since}' + (f' for jobs {a.jobs}' if want else '') + f': {tot:.2f}')
    print('by partition:', {k: round(v, 2) for k, v in sorted(per.items())})
    if want:
        print('by job:', {k: round(v, 3) for k, v in sorted(jobs.items())})
    if unknown:
        print('partitions without a weight (not counted):', sorted(unknown))


if __name__ == '__main__':
    main()
