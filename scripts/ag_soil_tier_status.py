#!/usr/bin/env python3
"""arena_gator_20260925 soil stage 1: per (arena, vehicle) and tier, how many training-pool rows of a soil task file
have a complete run (episode_complete.json) in the output folder. Plain python3 (json only), for the login node:

  python3 ag_soil_tier_status.py --tasks $G3/tasks/soil_v2.json --out $G3/soil_v1 [--tiers 0-6] [--json f]

Rows counted: kind designed / on_policy (tier >= 0). Key = <arena>:<vehicle> (Gator rows: f104:gator).
'complete up to tier K' = every row of tiers 0..K complete. Also prints claims in flight per key and the
simulated seconds of the complete rows (outcome.json elapsed_s) and of the remaining rows (estimated from the
mean of the complete rows of the same key).
"""
import argparse, json, os, time
from collections import defaultdict


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--tiers', default=None, help='A-B: also print the rows left inside this tier range')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    rows = [r for r in load(a.tasks) if r.get('run', True) and r.get('kind') in ('designed', 'on_policy') and int(r.get('tier', -1)) >= 0]
    runs = os.path.join(a.out, 'runs'); claims_dir = os.path.join(a.out, 'claims')
    have = set(os.listdir(runs)) if os.path.isdir(runs) else set()
    claims = set(os.listdir(claims_dir)) if os.path.isdir(claims_dir) else set()
    T = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))   # key -> tier -> [rows, complete, in flight]
    sim = defaultdict(lambda: [0.0, 0])
    for r in rows:
        key = f"{r.get('arena')}:{r.get('vehicle', 'hmmwv')}"
        t = int(r['tier']); c = T[key][t]; c[0] += 1
        rid = r['id']
        if rid in have and os.path.exists(os.path.join(runs, rid, 'episode_complete.json')):
            c[1] += 1
            o = load(os.path.join(runs, rid, 'outcome.json')) or {}
            sim[key][0] += float(o.get('elapsed_s', 0) or 0); sim[key][1] += 1
        elif rid in claims:
            c[2] += 1
    lo, hi = (None, None)
    if a.tiers:
        lo, _, hi = a.tiers.partition('-'); lo, hi = int(lo), int(hi or lo)
    res = dict(time=time.strftime('%F %T'), tasks=a.tasks, out=a.out, keys={})
    print(res['time'])
    for key in sorted(T):
        tiers = T[key]; upto = -1
        for t in sorted(tiers):
            if tiers[t][1] == tiers[t][0] and upto == t - 1:
                upto = t
        mean_sim = sim[key][0] / sim[key][1] if sim[key][1] else None
        line = ' '.join(f'{t}:{tiers[t][1]}/{tiers[t][0]}' + (f'(+{tiers[t][2]})' if tiers[t][2] else '') for t in sorted(tiers))
        blk = dict(tiers={str(t): dict(rows=v[0], complete=v[1], in_flight=v[2]) for t, v in sorted(tiers.items())},
                   complete_up_to_tier=upto, sim_h_complete=round(sim[key][0] / 3600, 2), mean_sim_s=round(mean_sim, 1) if mean_sim else None)
        extra = ''
        if lo is not None:
            left = sum(v[0] - v[1] for t, v in tiers.items() if lo <= t <= hi)
            blk['rows_left_in_range'] = left
            blk['sim_h_left_in_range_est'] = round(left * (mean_sim or 0) / 3600, 2)
            extra = f'  left in {a.tiers}: {left} rows (~{blk["sim_h_left_in_range_est"]} sim h)'
        print(f'{key:12s} complete up to tier {upto:2d}  mean {blk["mean_sim_s"]} s{extra}\n   {line}')
        res['keys'][key] = blk
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
