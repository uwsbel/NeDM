"""sensor_v1 closed-loop analysis, exactly as pre-registered in sensor_v1/PLAN.md.

Primary: pooled over six arenas, fixed 2 m/s, unsafe(s_fixed2) - unsafe(n2_fixed2); non-inferior if the upper end of
the 95% paired bootstrap CI (groups resampled within arena, 4,000 draws) <= +2.0 points.
Secondary: speed free with margin +1.0; exact McNemar; f104 / new arenas; fail; tilt30; median time.
"""
import json, os, sys
from math import comb
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from f104_n2_analyze import labels

S = 'artifacts/traverse/fdm_f104_50h_20260909/sensor_v1'
ARENAS = ['f104', 'g228', 'g203', 'g217', 'g216', 'g231']
ARMS = ['n2', 's', 'straight6', 'n2_fixed2', 's_fixed2']
# follow-up test: python sensor_analyze_test.py <runs> --tag test2 --models e0,d  (first model is the primary)


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('runs', nargs='?', default=S + '/test/runs')
    ap.add_argument('--tag', default='test'); ap.add_argument('--models', default='s')
    a_ = ap.parse_args(); runs = a_.runs; models = a_.models.split(',')
    global ARMS
    ARMS = ['n2', 'straight6', 'n2_fixed2'] + models + [f'{m}_fixed2' for m in models]
    by = {}
    for a in ARENAS:
        for p in sorted(Path(f'{S}/{a_.tag}_{a}/picks').glob('*.json')):
            s = json.load(open(p)); row = {}
            for arm in ARMS:
                info = s['arms'].get(arm)
                if not info: continue
                d = f"{runs}/{info['route_id']}"
                if os.path.exists(d + '/outcome.json') and os.path.exists(d + '/trajectory.npz'):
                    L = labels(d); L['tilt30'] = int(L['max_tilt'] > 30.0); L['same_route_as'] = info['route_id'].split('__')[1]
                    row[arm] = L
            by[s['group']] = dict(arena=a, arms=row)
    rng = np.random.default_rng(0)

    def compare(a, b, key, groups, margin):
        G_ = [g for g in groups if a in by[g]['arms'] and b in by[g]['arms']]
        x = np.array([by[g]['arms'][a][key] for g in G_], float); y = np.array([by[g]['arms'][b][key] for g in G_], float)
        arena_of = np.array([by[g]['arena'] for g in G_]); idx = {ar: np.where(arena_of == ar)[0] for ar in set(arena_of)}
        diffs = []
        for _ in range(4000):
            j = np.concatenate([rng.choice(ix, len(ix)) for ix in idx.values()])
            diffs.append(100 * (x[j].mean() - y[j].mean()))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        bb = int(((x == 1) & (y == 0)).sum()); cc = int(((x == 0) & (y == 1)).sum())
        return dict(n=len(G_), rate_a=100 * x.mean(), rate_b=100 * y.mean(), diff=100 * (x.mean() - y.mean()), ci95=[lo, hi],
                    margin=margin, non_inferior=bool(hi <= margin) if margin is not None else None,
                    a_worse=bb, b_worse=cc, p_mcnemar=mcnemar(bb, cc))

    allg = list(by); f104 = [g for g in by if by[g]['arena'] == 'f104']; new = [g for g in by if by[g]['arena'] != 'f104']
    out = {'n_groups': {a: sum(by[g]['arena'] == a for g in by) for a in ARENAS}}
    for k, m in enumerate(models):
        pre = 'PRIMARY_' if k == 0 else f'{m}_'
        out[f'{pre}fixed2_unsafe_all'] = compare(f'{m}_fixed2', 'n2_fixed2', 'unsafe', allg, 2.0)
        out[f'{m}_speedfree_unsafe_all'] = compare(m, 'n2', 'unsafe', allg, 1.0)
        for tag, gs in (('f104', f104), ('new', new)):
            out[f'{m}_fixed2_unsafe_{tag}'] = compare(f'{m}_fixed2', 'n2_fixed2', 'unsafe', gs, None)
            out[f'{m}_speedfree_unsafe_{tag}'] = compare(m, 'n2', 'unsafe', gs, None)
        for key in ('fail', 'tilt30'):
            out[f'{m}_fixed2_{key}_all'] = compare(f'{m}_fixed2', 'n2_fixed2', key, allg, None)
            out[f'{m}_speedfree_{key}_all'] = compare(m, 'n2', key, allg, None)
        out[f'{m}_vs_straight6_unsafe_all'] = compare(m, 'straight6', 'unsafe', allg, None)
    rates = {}
    for a in ARENAS + ['all']:
        gs = allg if a == 'all' else [g for g in by if by[g]['arena'] == a]
        rates[a] = {}
        for arm in ARMS:
            v = [by[g]['arms'][arm] for g in gs if arm in by[g]['arms']]
            if v:
                t = [x['elapsed'] for x in v if not x['fail']]
                rates[a][arm] = dict(n=len(v), unsafe=100 * np.mean([x['unsafe'] for x in v]), fail=100 * np.mean([x['fail'] for x in v]),
                                     tilt30=100 * np.mean([x['tilt30'] for x in v]), median_time_s=float(np.median(t)) if t else None)
    out['rates'] = rates
    out['identical_picks'] = {k: sum(1 for g in by if k[0] in by[g]['arms'] and by[g]['arms'][k[0]]['same_route_as'] == k[1])
                              for m in models for k in ((m, 'n2'), (f'{m}_fixed2', 'n2_fixed2'))}
    out['identical_picks'] = {f'{a}=={b}': v for (a, b), v in out['identical_picks'].items()}
    json.dump(out, open(S + f'/{a_.tag}_results.json', 'w'), indent=1, default=float)
    print(json.dumps(out['n_groups']), out['identical_picks'])
    for k, r in out.items():
        if isinstance(r, dict) and 'diff' in r:
            ni = '' if r['non_inferior'] is None else ('  NON-INFERIOR' if r['non_inferior'] else '  NOT shown non-inferior')
            print(f"{k:32s} n={r['n']:4d}  new input {r['rate_a']:5.2f}% vs current {r['rate_b']:5.2f}%  diff {r['diff']:+.2f} "
                  f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]  discordant {r['a_worse']} vs {r['b_worse']}  p={r['p_mcnemar']:.3f}{ni}")
    for a, d in rates.items():
        print('  ' + a.ljust(6) + '  '.join(f"{arm}:{v['unsafe']:.1f}/{v['fail']:.1f}/{v['tilt30']:.1f}/{v['median_time_s'] or 0:.0f}s" for arm, v in d.items()))


if __name__ == '__main__':
    main()
