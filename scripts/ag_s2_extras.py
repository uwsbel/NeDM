#!/usr/bin/env python3
"""Soil evaluation drive checks for arena_gator_20260925 (soil track step 2; PLAN 7.7 criteria applied to the
evaluation drives): per (vehicle, arm) of the soil index: planned / driven share, the crm_qa.py check of every driven
run (unreadable, shape, non-finite, launch-check failure, explosion, breakthrough without a stall), statuses, simulated
hours, and for Gator drives the belly-in-soil flag (lowest hull point > 0.05 m under the undisturbed surface for > 1 s in
a row, vehicle_extra.npz belly_clearance_min_m at 0.05 s frames; the ag_s1_gator_soil_qa.py rule). Also the fail rate
with QA-flagged drives left out (sensitivity; the declared analysis keeps every driven outcome).
  PYTHONPATH=src:scripts python scripts/ag_s2_extras.py --index $K3/e6/index/soil_eval_v1.json --out $K3/e6/analysis/soil_extras_v1.json
"""
import argparse, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crm_qa  # noqa: E402

DT = 0.05


def longest_run(m):
    best = cur = 0
    for v in m:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    idx = json.load(open(a.index))
    rows = [r for r in idx['rows'] if r['world'] == 'crm']
    per_run = {}
    for r in rows:
        if r.get('missing') or r['run_id'] in per_run:
            continue
        d = r['run_dir']
        q = crm_qa.check(d)
        x = dict(qa_ok=bool(q['ok']), qa_flag=q.get('flag'), status=r['status'], sim_s=float(r['elapsed']), vehicle=r['vehicle'])
        iv = json.load(open(os.path.join(d, 'initial_state_validation.json'))) if os.path.exists(os.path.join(d, 'initial_state_validation.json')) else {}
        x['launch_ok'] = bool(iv.get('passed', False))
        if r['vehicle'] == 'gator':
            try:
                b = np.load(os.path.join(d, 'vehicle_extra.npz'))['belly_clearance_min_m'].astype(float)
                deep = b < -0.05
                x['belly_min_m'] = float(np.nanmin(b)); x['belly_run_s'] = longest_run(deep) * DT
                x['belly_flag'] = x['belly_run_s'] > 1.0; x['belly_total_s'] = float(deep.sum() * DT)
            except Exception as e:  # noqa: BLE001
                x['belly_error'] = type(e).__name__
        per_run[r['run_id']] = x
    arms = defaultdict(list)
    for r in rows:
        arms[(r['vehicle'], r['arm'], r['set'])].append(r)
    out = {}
    for (veh, arm, s), rs in sorted(arms.items()):
        drv = [r for r in rs if not r.get('missing')]
        xs = [per_run[r['run_id']] for r in drv]
        ok = [(r, x) for r, x in zip(drv, xs) if x['qa_ok']]
        e = dict(vehicle=veh, arm=arm, set=s, planned=len(rs), driven=len(drv), driven_share=len(drv) / len(rs) if rs else None,
                 qa_flags=dict(Counter(x['qa_flag'] for x in xs if not x['qa_ok'])), launch_failures=sum(1 for x in xs if not x['launch_ok']),
                 statuses=dict(Counter(x['status'] for x in xs)), sim_h=sum(x['sim_s'] for x in xs) / 3600,
                 fail_all=float(np.mean([r['fail'] for r in drv])) if drv else None,
                 fail_qa_ok_only=float(np.mean([r['fail'] for r, _ in ok])) if ok else None)
        if veh == 'gator':
            fl = [x for x in xs if 'belly_flag' in x]
            e.update(belly_flag_share=float(np.mean([x['belly_flag'] for x in fl])) if fl else None,
                     belly_flag_n=len(fl), belly_errors=sum(1 for x in xs if 'belly_error' in x),
                     belly_flag_on_goal=sum(1 for r, x in zip(drv, xs) if x.get('belly_flag') and not r['fail']))
        out[f'{veh}|{arm}|{s}'] = e
    tot = dict(runs=len(per_run), qa_flagged=sum(1 for x in per_run.values() if not x['qa_ok']),
               qa_flags=dict(Counter(x['qa_flag'] for x in per_run.values() if not x['qa_ok'])),
               launch_failures=sum(1 for x in per_run.values() if not x['launch_ok']),
               nonfinite=sum(1 for x in per_run.values() if x['qa_flag'] == 'nonfinite'),
               sim_h=dict((v, sum(x['sim_s'] for x in per_run.values() if x['vehicle'] == v) / 3600) for v in ('hmmwv', 'gator')),
               gator_belly_flag_share=float(np.mean([x['belly_flag'] for x in per_run.values() if 'belly_flag' in x])) if any('belly_flag' in x for x in per_run.values()) else None)
    json.dump(dict(index=a.index, totals=tot, by_arm_set=out, qa_flagged_runs=sorted(k for k, x in per_run.items() if not x['qa_ok'])),
              open(a.out, 'w'), indent=1, default=float)
    print(json.dumps(tot, indent=1, default=float))


if __name__ == '__main__':
    main()
