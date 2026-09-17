"""Chrono pilot analysis for the matched v2 models: route choice at fixed speed, then safety and time with speed free."""
import argparse, json, os, sys
from math import comb
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from f104_n2_analyze import labels

V = 'artifacts/traverse/fdm_f104_50h_20260909/sensor_v2'


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='?', default=V + '/pilot_runs/runs')
    ap.add_argument('--arenas', default='g216,g231'); ap.add_argument('--models', default='H,Dabs,Drel')
    ap.add_argument('--out', default=V + '/pilot_results.json')
    a = ap.parse_args()
    models = a.models.split(','); arms = ['straight6'] + models + [f'{m}_fixed2' for m in models]
    by = {}
    for ar in a.arenas.split(','):
        for p in sorted(Path(f'{V}/pilot_{ar}/picks').glob('*.json')):
            s = json.load(open(p)); row = {}
            for arm in arms:
                info = s['arms'].get(arm)
                if not info: continue
                d = f"{a.runs}/{info['route_id']}"
                if os.path.exists(d + '/outcome.json') and os.path.exists(d + '/trajectory.npz'):
                    L = labels(d); L['tilt30'] = int(L['max_tilt'] > 30.0); L['same_as'] = info['route_id'].split('__')[1]
                    row[arm] = L
            by[s['group']] = dict(arena=ar, arms=row)
    G = [g for g in by if all(x in by[g]['arms'] for x in arms)]
    rng = np.random.default_rng(0)
    out = {'groups': len(by), 'complete': len(G), 'per_arena': {ar: sum(by[g]['arena'] == ar for g in G) for ar in a.arenas.split(',')}}

    def cmp(x, y, key, groups=None):
        gs = groups or G
        xa = np.array([by[g]['arms'][x][key] for g in gs], float); ya = np.array([by[g]['arms'][y][key] for g in gs], float)
        ar = np.array([by[g]['arena'] for g in gs]); idx = {k: np.where(ar == k)[0] for k in set(ar)}
        boots = [100 * (xa[j].mean() - ya[j].mean()) for j in
                 (np.concatenate([rng.choice(ix, len(ix)) for ix in idx.values()]) for _ in range(4000))]
        b = int(((xa == 1) & (ya == 0)).sum()); c = int(((xa == 0) & (ya == 1)).sum())
        return dict(n=len(gs), rate_a=100 * xa.mean(), rate_b=100 * ya.mean(), diff=100 * (xa.mean() - ya.mean()),
                    ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    a_worse=b, b_worse=c, p=mcnemar(b, c))

    for m in models:
        if m == 'H': continue
        out[f'fixed2_unsafe_{m}_vs_H'] = cmp(f'{m}_fixed2', 'H_fixed2', 'unsafe')
        out[f'speedfree_unsafe_{m}_vs_H'] = cmp(m, 'H', 'unsafe')
        out[f'speedfree_fail_{m}_vs_H'] = cmp(m, 'H', 'fail')
        out[f'speedfree_tilt30_{m}_vs_H'] = cmp(m, 'H', 'tilt30')
    rates = {}
    for scope in ['all'] + a.arenas.split(','):
        gs = G if scope == 'all' else [g for g in G if by[g]['arena'] == scope]
        rates[scope] = {}
        for arm in arms:
            v = [by[g]['arms'][arm] for g in gs]
            t = [x['elapsed'] for x in v if not x['fail']]
            rates[scope][arm] = dict(n=len(v), unsafe=100 * np.mean([x['unsafe'] for x in v]), fail=100 * np.mean([x['fail'] for x in v]),
                                     tilt30=100 * np.mean([x['tilt30'] for x in v]), median_time_s=float(np.median(t)) if t else None,
                                     median_max_tilt=float(np.median([x['max_tilt'] for x in v])))
    out['rates'] = rates
    out['identical_to_H'] = {m: sum(1 for g in G if by[g]['arms'][m]['same_as'] == 'H') for m in models if m != 'H'}
    json.dump(out, open(a.out, 'w'), indent=1, default=float)
    print(f"{out['complete']}/{out['groups']} groups with every arm  {out['per_arena']}  identical picks {out['identical_to_H']}")
    for k, r in out.items():
        if isinstance(r, dict) and 'diff' in r:
            print(f"  {k:34s} {r['rate_a']:5.2f}% vs H {r['rate_b']:5.2f}%  diff {r['diff']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]  "
                  f"{r['a_worse']} vs {r['b_worse']}  p={r['p']:.3f}")
    print('\n  arm rates (unsafe / fail / tilt30 / median time / median tilt)')
    for scope, d in rates.items():
        print('   ' + scope.ljust(6) + '  '.join(f"{arm}:{v['unsafe']:.1f}/{v['fail']:.1f}/{v['tilt30']:.1f}/{v['median_time_s'] or 0:.0f}s/{v['median_max_tilt']:.0f}d" for arm, v in d.items()))


if __name__ == '__main__':
    main()
