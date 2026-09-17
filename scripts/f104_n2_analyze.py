"""Night-2 closed loop, step 3: paired analysis of the five arms on the fresh test groups.

unsafe = NOT(goal reached AND < 0.05 s rolling backwards under throttle AND min vx > -0.30), after the 1 s settle,
with the corrected throttle column (action = [steering, throttle, braking]).
"""
import json, os, sys
from math import comb
import numpy as np

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
OUT = ROOT + '/night2_v1/closed'
ARMS = ('control', 'model', 'sampler', 'abstain', 'anchor6', 'pessimist', 'fixed2_control', 'fixed2_new')
S0 = 20


def labels(d):
    o = json.load(open(d + '/outcome.json')); z = np.load(d + '/trajectory.npz')
    vx = z['state'][:, 0]; thr = z['action'][:, 1]
    back = ((vx[S0:] < -0.10) & (thr[S0:] > 0.3)).sum() * 0.05
    fail = o['status'] != 'goal_reached'
    clean = (not fail) and back < 0.05 and vx[S0:].min() > -0.30
    return dict(fail=int(fail), unsafe=int(not clean), status=o['status'], elapsed=float(o['elapsed_s']),
                back_s=float(back), min_vx=float(vx[S0:].min()),
                max_tilt=float(np.degrees(np.abs(z['state'][S0:, 2:4])).max()))


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    runs = sys.argv[1] if len(sys.argv) > 1 else OUT + '/runs'
    tasks = json.load(open(OUT + '/tasks_cluster.json'))
    lab = {}
    for t in tasks:
        d = f"{runs}/{t['id']}"
        if t['id'] not in lab and os.path.exists(d + '/outcome.json') and os.path.exists(d + '/trajectory.npz'):
            lab[t['id']] = labels(d)
    by = {}
    for t in tasks:
        if t['id'] in lab:
            by.setdefault(t['group_id'], {})[t['arm']] = dict(lab[t['id']], risk=t['risk'])
    G = [g for g, v in by.items() if all(a in v for a in ARMS)]
    print(f'{len(G)} groups with all arms  (of {len(by)} with any)\n')
    rng = np.random.default_rng(0)
    for key in ('fail', 'unsafe'):
        print(f'== {key.upper()} ==')
        r = {a: np.array([by[g][a][key] for g in G]) for a in ARMS}
        for a in ARMS:
            print(f'  {a:8s} {100*r[a].mean():5.1f}%  ({int(r[a].sum())}/{len(G)})')
        for a, b, what in (('control', 'model', 'model gain (same candidates)'),
                           ('model', 'sampler', 'sampler gain (same model)'),
                           ('sampler', 'abstain', 'abstain rule'),
                           ('control', 'abstain', 'total: night-1 planner -> night-2 planner'),
                           ('anchor6', 'abstain', 'night-2 planner vs always 6 m/s straight'),
                           ('control', 'anchor6', 'night-1 planner vs always 6 m/s straight'),
                           ('sampler', 'pessimist', 'pessimistic ensemble vs mean'),
                           ('fixed2_control', 'fixed2_new', 'SECONDARY, 2 m/s fixed: night-1 -> night-2 model')):
            x, y = r[a], r[b]
            bb = int(((x == 1) & (y == 0)).sum()); cc = int(((x == 0) & (y == 1)).sum())
            d = [x[i].mean() - y[i].mean() for i in (rng.integers(0, len(G), len(G)) for _ in range(4000))]
            print(f'    {what:44s} {100*(x.mean()-y.mean()):+5.1f} pts [{100*np.percentile(d,2.5):+.1f}, '
                  f'{100*np.percentile(d,97.5):+.1f}]  {bb} vs {cc}  p={mcnemar(bb, cc):.4f}')
        print()
    print('== time to goal (successful runs only) ==')
    for a in ARMS:
        t = [by[g][a]['elapsed'] for g in G if by[g][a]['fail'] == 0]
        print(f'  {a:8s} median {np.median(t):5.1f} s  (n={len(t)})')
    print('\n== body tilt (max |roll| or |pitch|, deg) ==')
    for a in ARMS:
        t = [by[g][a]['max_tilt'] for g in G]
        print(f'  {a:8s} median {np.median(t):4.1f}  p90 {np.percentile(t,90):4.1f}')
    json.dump({g: by[g] for g in G}, open(OUT + '/results.json', 'w'), indent=1)
    print('\nwrote', OUT + '/results.json')


if __name__ == '__main__':
    main()
