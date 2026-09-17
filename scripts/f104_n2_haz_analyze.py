"""Hazard-test analysis: six arms on 300 hill/crater groups, three metrics reported together.

metrics: failed | unsafe (failed or slid back) | unsafe_or_tilt (also bad if |roll| or |pitch| > 35 deg)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, 'scripts')
from f104_n2_analyze import labels, mcnemar

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909/night2_v1/closed_haz'
ARMS = ('control', 'sampler', 'sampler_noanchor', 'anchor6', 'fixed2_old', 'fixed2_new')
PAIRS = (('control', 'sampler', 'night-1 planner -> night-2 planner'),
         ('control', 'sampler_noanchor', 'night-2 planner WITHOUT the designed-route anchors'),
         ('sampler_noanchor', 'sampler', 'what the injected anchors add on top of the wider sampler'),
         ('anchor6', 'sampler', 'night-2 planner vs always 6 m/s straight'),
         ('control', 'anchor6', 'night-1 planner vs always 6 m/s straight'),
         ('fixed2_old', 'fixed2_new', 'at 2 m/s, night-1 model -> night-2 model'))


def main():
    runs = sys.argv[1] if len(sys.argv) > 1 else ROOT + '/runs'
    tasks = json.load(open(ROOT + '/tasks_cluster.json'))
    lab = {}
    for t in tasks:
        d = f"{runs}/{t['id']}"
        if t['id'] not in lab and os.path.exists(d + '/outcome.json') and os.path.exists(d + '/trajectory.npz'):
            L = labels(d); L['unsafe_or_tilt'] = int(L['unsafe'] or L['max_tilt'] > 35.0)
            lab[t['id']] = L
    by = {}
    for t in tasks:
        if t['id'] in lab:
            by.setdefault(t['group_id'], {})[t['arm']] = dict(lab[t['id']], risk=t['risk'])
    G = [g for g, v in by.items() if all(a in v for a in ARMS)]
    picks = {r['group']: r for r in json.load(open(ROOT + '/picks.json'))}
    print(f'{len(G)} hazard groups with all {len(ARMS)} arms '
          f'(night-2 pick was a designed anchor in {sum(picks[g]["pick_is_anchor"] for g in G)})\n')
    rng = np.random.default_rng(0)
    for key in ('fail', 'unsafe', 'unsafe_or_tilt'):
        print(f'== {key.upper()} ==')
        r = {a: np.array([by[g][a][key] for g in G]) for a in ARMS}
        for a in ARMS:
            print(f'  {a:17s} {100*r[a].mean():5.1f}%  ({int(r[a].sum())}/{len(G)})')
        for a, b, what in PAIRS:
            x, y = r[a], r[b]
            bb = int(((x == 1) & (y == 0)).sum()); cc = int(((x == 0) & (y == 1)).sum())
            d = [x[i].mean() - y[i].mean() for i in (rng.integers(0, len(G), len(G)) for _ in range(4000))]
            print(f'    {what:55s} {100*(x.mean()-y.mean()):+5.1f} pts [{100*np.percentile(d,2.5):+.1f}, '
                  f'{100*np.percentile(d,97.5):+.1f}]  {bb} vs {cc}  p={mcnemar(bb, cc):.4f}')
        print()
    print('== median time to goal (successful) / median max tilt / runs above 35 deg ==')
    for a in ARMS:
        t = [by[g][a]['elapsed'] for g in G if by[g][a]['fail'] == 0]
        ti = [by[g][a]['max_tilt'] for g in G]
        print(f'  {a:17s} {np.median(t):5.1f} s   {np.median(ti):4.1f} deg   {sum(x > 35 for x in ti):3d}/{len(G)}')
    json.dump({g: by[g] for g in G}, open(ROOT + '/results.json', 'w'), indent=1)
    print('\nwrote', ROOT + '/results.json')


if __name__ == '__main__':
    main()
