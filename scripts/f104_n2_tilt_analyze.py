"""Tilt experiment: paired comparison of the current night-2 model and the tilt-aware model."""
import json, os, sys
import numpy as np
sys.path.insert(0, 'scripts')
from f104_n2_analyze import labels, mcnemar

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909/night2_v1/closed_tilt'


def main():
    runs = ROOT + '/runs'
    tasks = json.load(open(ROOT + '/tasks_cluster.json'))
    lab = {}
    for t in tasks:
        d = f"{runs}/{t['id']}"
        if t['id'] not in lab and os.path.exists(d + '/outcome.json'):
            L = labels(d); L['tilt35'] = int(L['max_tilt'] > 35); L['tilt30'] = int(L['max_tilt'] > 30)
            L['unsafe_or_tilt'] = int(L['unsafe'] or L['max_tilt'] > 35)
            lab[t['id']] = L
    by = {}
    for t in tasks:
        if t['id'] in lab: by.setdefault(t['group_id'], {})[t['arm']] = lab[t['id']]
    G = [g for g, v in by.items() if 'base' in v and 'tilt' in v]
    print(f'{len(G)} groups with both arms\n')
    rng = np.random.default_rng(0)
    for key in ('max_tilt',):
        b = np.array([by[g]['base'][key] for g in G]); t = np.array([by[g]['tilt'][key] for g in G])
        d = b - t
        boot = [np.median(d[i]) for i in (rng.integers(0, len(G), len(G)) for _ in range(4000))]
        print(f'max body tilt (deg): current model median {np.median(b):.1f}  tilt-aware {np.median(t):.1f}  '
              f'paired median difference {np.median(d):+.2f} [{np.percentile(boot,2.5):+.2f}, {np.percentile(boot,97.5):+.2f}]')
        print(f'  mean {np.mean(b):.1f} vs {np.mean(t):.1f}; p90 {np.percentile(b,90):.1f} vs {np.percentile(t,90):.1f}; '
              f'max {b.max():.1f} vs {t.max():.1f}')
    for key in ('tilt35', 'tilt30', 'unsafe', 'unsafe_or_tilt', 'fail'):
        b = np.array([by[g]['base'][key] for g in G]); t = np.array([by[g]['tilt'][key] for g in G])
        bb = int(((b == 1) & (t == 0)).sum()); cc = int(((b == 0) & (t == 1)).sum())
        print(f'{key:15s} current {100*b.mean():5.2f}% ({int(b.sum())})  tilt-aware {100*t.mean():5.2f}% ({int(t.sum())})  '
              f'{bb} vs {cc}  p={mcnemar(bb, cc):.4f}')
    tb = [by[g]['base']['elapsed'] for g in G if by[g]['base']['fail'] == 0]
    tt = [by[g]['tilt']['elapsed'] for g in G if by[g]['tilt']['fail'] == 0]
    print(f'\ntime to goal: current median {np.median(tb):.1f} s   tilt-aware {np.median(tt):.1f} s')
    json.dump({g: by[g] for g in G}, open(ROOT + '/results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
