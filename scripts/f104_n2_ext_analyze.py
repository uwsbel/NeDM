"""Extension analysis: paired night-1 vs night-2 model at a fixed 2 m/s on 523 fresh groups."""
import json, os, sys
from math import comb
import numpy as np
sys.path.insert(0, 'scripts')
from f104_n2_analyze import labels, mcnemar

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909/night2_v1/closed_ext'


def main():
    runs = sys.argv[1] if len(sys.argv) > 1 else ROOT + '/runs'
    tasks = json.load(open(ROOT + '/tasks_cluster.json'))
    lab = {}
    for t in tasks:
        d = f"{runs}/{t['id']}"
        if t['id'] not in lab and os.path.exists(d + '/outcome.json') and os.path.exists(d + '/trajectory.npz'):
            lab[t['id']] = labels(d)
    by = {}
    for t in tasks:
        if t['id'] in lab:
            by.setdefault(t['group_id'], {})[t['arm']] = dict(lab[t['id']], risk=t['risk'])
    G = [g for g, v in by.items() if 'old' in v and 'new' in v]
    same = sum(1 for g in G if by[g]['old'] == by[g]['new'])
    print(f'{len(G)} groups with both arms (identical picks in {same})\n')
    rng = np.random.default_rng(0)
    for key in ('unsafe', 'fail'):
        o = np.array([by[g]['old'][key] for g in G]); n = np.array([by[g]['new'][key] for g in G])
        b = int(((o == 1) & (n == 0)).sum()); c = int(((o == 0) & (n == 1)).sum())
        d = [o[i].mean() - n[i].mean() for i in (rng.integers(0, len(G), len(G)) for _ in range(4000))]
        print(f'{key.upper():7s} night-1 model {100*o.mean():5.2f}%  night-2 model {100*n.mean():5.2f}%  '
              f'diff {100*(o.mean()-n.mean()):+5.2f} pts [{100*np.percentile(d,2.5):+.2f}, {100*np.percentile(d,97.5):+.2f}]  '
              f'{b} vs {c}  McNemar p={mcnemar(b, c):.4f}')
    t_old = [by[g]['old']['elapsed'] for g in G if by[g]['old']['fail'] == 0]
    t_new = [by[g]['new']['elapsed'] for g in G if by[g]['new']['fail'] == 0]
    print(f'\ntime to goal (successful): night-1 median {np.median(t_old):.1f}s   night-2 median {np.median(t_new):.1f}s')
    print(f'max body tilt: night-1 median {np.median([by[g]["old"]["max_tilt"] for g in G]):.1f} deg  '
          f'night-2 median {np.median([by[g]["new"]["max_tilt"] for g in G]):.1f} deg')
    json.dump({g: by[g] for g in G}, open(ROOT + '/results.json', 'w'), indent=1)
    print('wrote', ROOT + '/results.json')


if __name__ == '__main__':
    main()
