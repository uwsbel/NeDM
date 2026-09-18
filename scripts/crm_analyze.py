"""Held-out single-goal CRM missions: paired analysis of the CRM-trained planner vs the frozen rigid-trained planner.

Adapted from scripts/sensor_analyze_v2.py (same label function, exact McNemar, paired group bootstrap). Pre-registered
in artifacts/traverse/crm_f104_v1/PLAN.md section 6 + amendment 2: single PRIMARY = goal not reached, crm vs rigid,
speed free (unsafe is secondary: on CRM it duplicates fail). Analysed set = longest prefix of groups, by index, with
every arm driven.

  python scripts/crm_analyze.py --eval artifacts/traverse/crm_f104_v1/eval_v1 [--runs <dir>] [--first-n N]
"""
import argparse, hashlib, json, os, sys
from math import comb
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from f104_n2_analyze import labels

ARMS = ['crm', 'rigid', 'crm_fixed2', 'rigid_fixed2', 'crm_pess', 'rigid_pess', 'straight6', 'straight2']


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def safe_labels(d):
    z = np.load(d + '/trajectory.npz')
    if len(z['state']) < 22:  # ended inside the 1 s label settle: goal cannot have been reached in < 1 s
        o = json.load(open(d + '/outcome.json'))
        return dict(fail=int(o['status'] != 'goal_reached'), unsafe=1, status=o['status'], elapsed=float(o['elapsed_s']),
                    back_s=0.0, min_vx=float(z['state'][:, 0].min()), max_tilt=float(np.degrees(np.abs(z['state'][:, 2:4])).max()))
    return labels(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', required=True); ap.add_argument('--runs', default=None)
    ap.add_argument('--first-n', type=int, default=None, help='analyse only the first N groups by index (declared before driving)')
    a = ap.parse_args()
    runs = a.runs or a.eval + '/runs'
    by = {}; stale = []; configs = set()
    picks = sorted(Path(a.eval, 'picks').glob('*.json'))
    if a.first_n:
        picks = picks[:a.first_n]
    for p in picks:
        s = json.load(open(p)); row = {}
        for arm in ARMS:
            info = s['arms'].get(arm)
            if not info: continue
            d = f"{runs}/{info['route_id']}"
            if os.path.exists(d + '/episode_complete.json'):
                req = json.load(open(d + '/collection_request.json'))
                want = hashlib.sha256(open(f"{a.eval}/routes/{info['route_id']}.json", 'rb').read()).hexdigest()
                if req['route_sha256'] != want:   # a drive of an older pick with the same id must never be reused
                    stale.append(info['route_id']); continue
                configs.add(json.dumps(req['crm_config'], sort_keys=True))
                L = safe_labels(d); L['tilt30'] = int(L['max_tilt'] > 30.0); L['same_as'] = info['route_id'].split('__')[1]
                L['mean_speed'] = info['mean_speed']; L['length_m'] = info['length_m']
                row[arm] = L
        by[s['group']] = row
    order = [json.load(open(p))['group'] for p in picks]
    complete = [all(x in by[g] for x in ARMS) for g in order]
    missing_arm = [g for g in order if any(json.load(open(f"{a.eval}/picks/{g}.json"))['arms'].get(x) is None for x in ARMS)]
    G = []
    for g, ok in zip(order, complete):  # longest prefix; groups where an arm does not EXIST (no valid anchor / empty pool) are skipped, not a stop
        if ok: G.append(g)
        elif g in missing_arm: continue
        else: break
    rng = np.random.default_rng(0)
    out = {'groups_with_picks': len(by), 'complete_groups': len(G), 'arms': ARMS, 'groups_without_some_arm': missing_arm, 'stale_drives_ignored': stale,
           'distinct_crm_configs': len(configs)}
    assert len(configs) <= 1, 'evaluation drives mix CRM physics configurations'

    def cmp(x, y, key):
        xa = np.array([by[g][x][key] for g in G], float); ya = np.array([by[g][y][key] for g in G], float)
        idx = rng.integers(0, len(G), (4000, len(G)))
        boots = 100 * (xa[idx].mean(1) - ya[idx].mean(1))
        b = int(((xa == 1) & (ya == 0)).sum()); c = int(((xa == 0) & (ya == 1)).sum())
        return dict(n=len(G), rate_a=100 * xa.mean(), rate_b=100 * ya.mean(), diff=100 * (xa.mean() - ya.mean()),
                    ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], a_worse=b, b_worse=c, p=mcnemar(b, c))

    if G:
        out['PRIMARY_fail_crm_vs_rigid'] = cmp('crm', 'rigid', 'fail')
        out['secondary_unsafe_crm_vs_rigid'] = cmp('crm', 'rigid', 'unsafe')
        sec = {}
        for x, y in (('crm_fixed2', 'rigid_fixed2'), ('crm_pess', 'rigid_pess'), ('crm', 'straight6'), ('rigid', 'straight6'),
                     ('crm', 'crm_pess'), ('crm_fixed2', 'straight2'), ('rigid_fixed2', 'straight2')):
            for key in ('fail', 'unsafe', 'tilt30'):
                sec[f'{key}_{x}_vs_{y}'] = cmp(x, y, key)
        out['secondary'] = sec
        rates = {}
        for arm in ARMS:
            v = [by[g][arm] for g in G]; t = [x['elapsed'] for x in v if not x['fail']]
            from collections import Counter
            rates[arm] = dict(n=len(v), goal_reached=100 * (1 - np.mean([x['fail'] for x in v])), fail=100 * np.mean([x['fail'] for x in v]),
                              unsafe=100 * np.mean([x['unsafe'] for x in v]), tilt30=100 * np.mean([x['tilt30'] for x in v]),
                              median_time_to_goal_s=float(np.median(t)) if t else None, mean_cmd_speed=float(np.mean([x['mean_speed'] for x in v])),
                              median_max_tilt=float(np.median([x['max_tilt'] for x in v])), status=dict(Counter(x['status'] for x in v)))
        out['rates'] = rates
        out['identical_pick_crm_rigid'] = sum(1 for g in G if by[g]['crm']['same_as'] == by[g]['rigid']['same_as'])
        out['identical_pick_fixed2'] = sum(1 for g in G if by[g]['crm_fixed2']['same_as'] == by[g]['rigid_fixed2']['same_as'])
    json.dump(dict(summary=out, per_group=by), open(a.eval + '/results.json', 'w'), indent=1, default=float)
    print(f"{out['complete_groups']}/{out['groups_with_picks']} groups with every arm driven")
    if not G:
        return
    for k in ('PRIMARY_fail_crm_vs_rigid', 'secondary_unsafe_crm_vs_rigid'):
        r = out[k]
        print(f"  {k:34s} crm {r['rate_a']:5.1f}% vs rigid {r['rate_b']:5.1f}%  diff {r['diff']:+.1f} [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}]  "
              f"crm worse {r['a_worse']} / rigid worse {r['b_worse']}  p={r['p']:.4f}")
    for k, r in out['secondary'].items():
        print(f"  {k:40s} {r['rate_a']:5.1f}% vs {r['rate_b']:5.1f}%  diff {r['diff']:+.1f} [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}]  {r['a_worse']} vs {r['b_worse']}  p={r['p']:.4f}")
    print('\n  arm: goal reached / unsafe / tilt>30 / median time to goal / mean commanded speed')
    for arm, v in out['rates'].items():
        print(f"   {arm:13s} {v['goal_reached']:5.1f}%  {v['unsafe']:5.1f}%  {v['tilt30']:5.1f}%  {v['median_time_to_goal_s'] or 0:5.1f} s  {v['mean_cmd_speed']:.2f} m/s  {v['status']}")
    print('  identical picks crm==rigid:', out['identical_pick_crm_rigid'], ' fixed2:', out['identical_pick_fixed2'])


if __name__ == '__main__':
    main()
