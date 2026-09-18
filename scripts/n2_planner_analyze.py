"""Paired analysis of planner arms produced by planner_arms.py / planner_grad_arms.py (picks/<group>.json with arms{...route_id}).
  python scripts/n2_planner_analyze.py --picks <dir> --runs <runs dir> --arms A,B,C,D,E,F --primary B:A --contrasts C:A,D:A,E:B,F:B [--label fail|unsafe] [--ref-runs <eval_v1 runs> --ref-arm crm]"""
import argparse, json, os, sys
from math import comb
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from f104_n2_analyze import labels


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def safe_labels(d):
    z = np.load(d + '/trajectory.npz')
    if len(z['state']) < 22:
        o = json.load(open(d + '/outcome.json'))
        return dict(fail=int(o['status'] != 'goal_reached'), unsafe=1, status=o['status'], elapsed=float(o['elapsed_s']), back_s=0.0, min_vx=0.0, max_tilt=0.0)
    return labels(d)


ap = argparse.ArgumentParser()
ap.add_argument('--picks', required=True); ap.add_argument('--runs', required=True); ap.add_argument('--arms', required=True)
ap.add_argument('--primary', default='B:A'); ap.add_argument('--contrasts', default='C:A,D:A,E:B,F:B'); ap.add_argument('--label', default='fail')
ap.add_argument('--ref-runs', default=None); ap.add_argument('--ref-arm', default=None); ap.add_argument('--out', default=None)
a = ap.parse_args(); arms = a.arms.split(',')
by = {}; missing = 0
for p in sorted(Path(a.picks).glob('*.json')):
    s = json.load(open(p)); row = {}
    for arm in arms:
        info = s['arms'].get(arm)
        if not info: continue
        d = f"{a.runs}/{info['route_id']}"
        if os.path.exists(d + '/episode_complete.json'):
            o = json.load(open(d + '/outcome.json')); L = safe_labels(d); L['tilt30'] = int(L['max_tilt'] > 30.0)
            L['work_kj'] = o.get('positive_work_kj'); L['route_id'] = info['route_id']; L['P'] = info.get('P'); L['z'] = info.get('z_mean'); L['T_cmd'] = info.get('T')
            row[arm] = L
        else:
            missing += 1
    by[s['group']] = row
G = [g for g in by if all(x in by[g] for x in arms)]
rng = np.random.default_rng(0); key = a.label
out = {'groups': len(by), 'complete': len(G), 'missing_drives': missing, 'arms': arms, 'label': key}


def cmp(x, y):
    xa = np.array([by[g][x][key] for g in G], float); ya = np.array([by[g][y][key] for g in G], float)
    idx = rng.integers(0, len(G), (4000, len(G))); boots = 100 * (xa[idx].mean(1) - ya[idx].mean(1))
    b = int(((xa == 1) & (ya == 0)).sum()); c = int(((xa == 0) & (ya == 1)).sum())
    same = sum(by[g][x]['route_id'] == by[g][y]['route_id'] for g in G)
    return dict(n=len(G), rate_a=100 * xa.mean(), rate_b=100 * ya.mean(), diff=100 * (xa.mean() - ya.mean()), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                a_worse=b, b_worse=c, p=mcnemar(b, c), identical_picks=same)


if G:
    x, y = a.primary.split(':'); out['PRIMARY_' + a.primary] = cmp(x, y)
    out['contrasts'] = {c: cmp(*c.split(':')) for c in a.contrasts.split(',') if c}
    rates = {}
    for arm in arms:
        v = [by[g][arm] for g in G]; t = [r['elapsed'] for r in v if not r['fail']]; w = [r['work_kj'] for r in v if not r['fail'] and r['work_kj'] is not None]
        rates[arm] = dict(goal_reached=100 * (1 - np.mean([r['fail'] for r in v])), unsafe=100 * np.mean([r['unsafe'] for r in v]), tilt30=100 * np.mean([r['tilt30'] for r in v]),
                          median_time_s=float(np.median(t)) if t else None, median_work_kj=float(np.median(w)) if w else None,
                          mean_pred_P=float(np.mean([r['P'] for r in v if r['P'] is not None])) if any(r['P'] is not None for r in v) else None,
                          pred_below_1pct_fail=int(sum(1 for r in v if r['P'] is not None and r['P'] < 0.01 and r['fail'])))
    out['rates'] = rates
    if a.ref_runs and a.ref_arm:   # A/A reproducibility against an earlier campaign that drove the same routes
        agree = tot = 0
        for g in G:
            rid = by[g][arms[0]]['route_id']; ref = f"{a.ref_runs}/{g}__{a.ref_arm}"
            if os.path.exists(ref + '/outcome.json'):
                tot += 1; agree += int((json.load(open(ref + '/outcome.json'))['status'] != 'goal_reached') == bool(by[g][arms[0]]['fail']))
        out['AA_reproduction'] = dict(agree=agree, total=tot)
if a.out: json.dump(dict(summary=out, per_group=by), open(a.out, 'w'), indent=1, default=float)
print(f"{out['complete']}/{out['groups']} groups with every arm; missing drives {missing}")
if G:
    for k, r in [('PRIMARY_' + a.primary, out['PRIMARY_' + a.primary])] + list(out['contrasts'].items()):
        print(f"  {k:14s} {key} {r['rate_a']:5.1f}% vs {r['rate_b']:5.1f}%  diff {r['diff']:+.1f} [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}]  worse {r['a_worse']} vs {r['b_worse']}  p={r['p']:.4f}  identical picks {r['identical_picks']}")
    print('  arm: goal reached / unsafe / tilt30 / median time / median work / mean predicted P / confident failures')
    for arm, v in out['rates'].items():
        print(f"   {arm}: {v['goal_reached']:5.1f}%  {v['unsafe']:5.1f}%  {v['tilt30']:4.1f}%  {v['median_time_s'] or 0:5.1f} s  {v['median_work_kj'] or 0:6.0f} kJ  P {v['mean_pred_P'] if v['mean_pred_P'] is not None else float('nan'):.3f}  {v['pred_below_1pct_fail']}")
    if 'AA_reproduction' in out: print('  A/A reproduction vs reference drives:', out['AA_reproduction'])
