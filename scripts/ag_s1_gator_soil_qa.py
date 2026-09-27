#!/usr/bin/env python3
"""arena_gator_20260925 soil stage 1: the PLAN 7.7 collection read-outs for the Gator soil rows of a tier range, against
the HMMWV collect_v1 runs of the identical ids. numpy only (login node: the nrd_pychrono python).

  python ag_s1_gator_soil_qa.py --tasks <gator task rows json> --runs $G3/soil_v1/runs \
      --ref-runs /work1/dannegrut/harry/experiments/crm_f104_20260916/collect_v1/runs --out <summary.json>

Per Gator row: complete (episode_complete.json), validated (crm_qa.check), launch check, status, simulated seconds,
belly-in-soil flag (lowest hull point > 0.05 m under the undisturbed surface for > 1 s in a row, vehicle_extra.npz
belly_clearance_min_m at 0.05 s frames, as scripts/ag_pilot_eval.py), and the HMMWV twin's status (fail = not goal).
Reports: validated share, crashed / non-finite (crm_qa flags), launch failures, belly flag share, failure Gator vs
HMMWV (overall, by tier, by designed profile = route index % 4, on-policy), discordant pairs, simulated hours.
"""
import argparse, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import crm_qa  # noqa: E402

DT = 0.05
PROFILES = {0: 'constant_2', 1: 'constant_4', 2: 'constant_6', 3: 'smooth_2_6_2', -1: 'planner_proposal'}


def longest_run(m):
    best = cur = 0
    for v in m:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', required=True); ap.add_argument('--runs', required=True); ap.add_argument('--ref-runs', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = [r for r in json.load(open(a.tasks)) if r['id'].startswith('gator__')]
    recs = []
    for r in rows:
        d = os.path.join(a.runs, r['id']); pid = r['id'][len('gator__'):]
        x = dict(id=r['id'], tier=int(r['tier']), kind=r.get('kind'), split=r.get('split'),
                 profile=PROFILES[int(pid.split('_route_')[1]) % 4 if '_route_' in pid else -1])
        x['complete'] = os.path.isfile(os.path.join(d, 'episode_complete.json'))
        if x['complete']:
            q = crm_qa.check(d); x['validated'] = bool(q['ok']); x['qa_flag'] = q.get('flag')
            iv = load(os.path.join(d, 'initial_state_validation.json')) or {}
            x['launch_ok'] = bool(iv.get('passed', False))
            o = load(os.path.join(d, 'outcome.json')) or {}
            x['status'] = o.get('status'); x['sim_s'] = float(o.get('elapsed_s') or 0)
            x['vehicle'] = (o.get('vehicle') or {}).get('name') if isinstance(o.get('vehicle'), dict) else o.get('vehicle')
            try:
                b = np.load(os.path.join(d, 'vehicle_extra.npz'))['belly_clearance_min_m'].astype(float)
                x['belly_min_m'] = float(np.nanmin(b)); deep = b < -0.05
                x['belly_run_s'] = longest_run(deep) * DT; x['belly_flag'] = x['belly_run_s'] > 1.0
                x['belly_total_s'] = float(deep.sum() * DT)
            except Exception as e:  # noqa: BLE001
                x['belly_error'] = type(e).__name__
        ro = load(os.path.join(a.ref_runs, pid, 'outcome.json'))
        x['h_status'] = ro.get('status') if ro else None; x['h_sim_s'] = float(ro.get('elapsed_s') or 0) if ro else None
        recs.append(x)
    comp = [x for x in recs if x['complete']]
    val = [x for x in comp if x.get('validated')]
    both = [x for x in val if x['h_status'] is not None]
    fail = lambda s: s != 'goal_reached'
    def rate(xs, f):
        xs = list(xs); return (round(sum(map(f, xs)) / len(xs), 4), len(xs)) if xs else (None, 0)
    res = dict(tasks=os.path.abspath(a.tasks), runs=os.path.abspath(a.runs), ref_runs=os.path.abspath(a.ref_runs),
               rows=len(recs), complete=len(comp), validated=len(val), validated_share_of_rows=round(len(val) / max(len(recs), 1), 4),
               qa_flags=dict(Counter(x.get('qa_flag') for x in comp if not x.get('validated'))),
               launch_failures=sum(1 for x in comp if not x.get('launch_ok')), vehicle_blocks=dict(Counter(str(x.get('vehicle')) for x in comp)),
               belly_flag=rate(val, lambda x: bool(x.get('belly_flag'))), belly_flag_total_gt1s=rate(val, lambda x: x.get('belly_total_s', 0) > 1.0),
               belly_errors=sum(1 for x in val if 'belly_error' in x),
               status_gator=dict(Counter(x['status'] for x in val)), status_hmmwv_twins=dict(Counter(x['h_status'] for x in both)),
               fail_gator=rate(both, lambda x: fail(x['status'])), fail_hmmwv=rate(both, lambda x: fail(x['h_status'])),
               gator_only_fail=sum(1 for x in both if fail(x['status']) and not fail(x['h_status'])),
               hmmwv_only_fail=sum(1 for x in both if fail(x['h_status']) and not fail(x['status'])),
               fail_gator_flag_as_fail=rate(both, lambda x: fail(x['status']) or bool(x.get('belly_flag'))),
               sim_h_gator=round(sum(x['sim_s'] for x in both) / 3600, 2), sim_h_hmmwv=round(sum(x['h_sim_s'] for x in both) / 3600, 2),
               by_tier={}, by_profile={}, by_split={})
    for key, name in (('tier', 'by_tier'), ('profile', 'by_profile'), ('split', 'by_split')):
        g = defaultdict(list)
        for x in both:
            g[x[key]].append(x)
        res[name] = {str(k): dict(n=len(v), fail_gator=rate(v, lambda x: fail(x['status']))[0], fail_hmmwv=rate(v, lambda x: fail(x['h_status']))[0],
                                  belly_flag=rate(v, lambda x: bool(x.get('belly_flag')))[0]) for k, v in sorted(g.items())}
    json.dump(dict(summary=res, rows=recs), open(a.out, 'w'), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
