"""gen_v1 section A analysis: six arms on 200 hill/crater start/goals per arena (f104 + five new arenas).

Exactly the pre-registered read-out (gen_v1/PLAN.md): primary P1 = pooled new arenas, unsafe, n2 vs rule, exact
two-sided McNemar; Holm over the six declared unsafe tests; secondary failure / tilt30 / time / generalisation gap /
per-arena / arena-level sign test. Writes gen_v1/test_results.json and prints a table.
"""
import json, os, sys
from math import comb
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from f104_n2_analyze import labels

G = 'artifacts/traverse/fdm_f104_50h_20260909/gen_v1'
ARENAS = ['f104', 'g228', 'g203', 'g217', 'g216', 'g231']
NEW = ARENAS[1:]
ARMS = ['n2', 'rule', 'straight6', 'n2_fixed2', 'rule_fixed2', 'straight2']


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load(runs):
    by = {}
    for a in ARENAS:
        for p in sorted(Path(f'{G}/test_{a}/picks').glob('*.json')):
            s = json.load(open(p)); g = s['group']; row = {}
            for arm in ARMS:
                info = s['arms'].get(arm)
                if not info: continue
                d = f"{runs}/{info['route_id']}"
                if os.path.exists(d + '/outcome.json') and os.path.exists(d + '/trajectory.npz'):
                    L = labels(d); L['tilt30'] = int(L['max_tilt'] > 30.0)
                    L.update(risk=info['risk'], mean_speed=info['mean_speed'], same_as=info['route_id'].split('__')[1])
                    row[arm] = L
            by[g] = dict(arena=a, arms=row)
    return by


def paired(by, groups, a, b, key):
    G_ = [g for g in groups if a in by[g]['arms'] and b in by[g]['arms']]
    x = np.array([by[g]['arms'][a][key] for g in G_]); y = np.array([by[g]['arms'][b][key] for g in G_])
    bb = int(((x == 1) & (y == 0)).sum()); cc = int(((x == 0) & (y == 1)).sum())
    return dict(n=len(G_), rate_a=float(x.mean()) if len(G_) else None, rate_b=float(y.mean()) if len(G_) else None,
                a_worse=bb, b_worse=cc, p=mcnemar(bb, cc))


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i])); adj[i] = run
    return adj.tolist()


def main():
    runs = sys.argv[1] if len(sys.argv) > 1 else G + '/test/runs'
    by = load(runs)
    new = [g for g in by if by[g]['arena'] in NEW]; f104 = [g for g in by if by[g]['arena'] == 'f104']
    out = {'n_groups': {a: sum(by[g]['arena'] == a for g in by) for a in ARENAS},
           'n_complete': {a: sum(by[g]['arena'] == a and len(by[g]['arms']) >= 5 for g in by) for a in ARENAS}}
    family = [('P1 new arenas: n2 vs rule', new, 'n2', 'rule'),
              ('new arenas: n2 vs straight6', new, 'n2', 'straight6'),
              ('new arenas at 2 m/s: n2 vs rule', new, 'n2_fixed2', 'rule_fixed2'),
              ('new arenas at 2 m/s: n2 vs straight line', new, 'n2_fixed2', 'straight2'),
              ('f104: n2 vs rule', f104, 'n2', 'rule'),
              ('f104: n2 vs straight6', f104, 'n2', 'straight6')]
    res = [dict(test=t, **paired(by, gs, a, b, 'unsafe')) for t, gs, a, b in family]
    for r, adj in zip(res, holm([r['p'] for r in res])): r['p_holm'] = adj
    out['unsafe_family'] = res
    out['secondary'] = {key: [dict(test=t, **paired(by, gs, a, b, key)) for t, gs, a, b in family] for key in ('fail', 'tilt30')}
    rates = {}
    for a in ARENAS + ['new_pooled']:
        gs = new if a == 'new_pooled' else [g for g in by if by[g]['arena'] == a]
        rates[a] = {}
        for arm in ARMS:
            v = [by[g]['arms'][arm] for g in gs if arm in by[g]['arms']]
            if not v: continue
            t = [x['elapsed'] for x in v if not x['fail']]
            rates[a][arm] = dict(n=len(v), unsafe=float(np.mean([x['unsafe'] for x in v])), fail=float(np.mean([x['fail'] for x in v])),
                                 tilt30=float(np.mean([x['tilt30'] for x in v])), median_time_s=float(np.median(t)) if t else None,
                                 median_max_tilt=float(np.median([x['max_tilt'] for x in v])))
    out['rates'] = rates
    # generalisation gap: n2 unsafe on new arenas minus f104, bootstrap resampling groups within arena
    rng = np.random.default_rng(0)
    def arm_rate(gs):
        v = [by[g]['arms']['n2']['unsafe'] for g in gs if 'n2' in by[g]['arms']]; return np.mean(v) if v else np.nan
    per = {a: [g for g in by if by[g]['arena'] == a and 'n2' in by[g]['arms']] for a in ARENAS}
    boots = []
    for _ in range(4000):
        nb = np.concatenate([rng.choice(per[a], len(per[a])) for a in NEW]); fb = rng.choice(per['f104'], len(per['f104']))
        boots.append(arm_rate(nb) - arm_rate(fb))
    out['gap_n2_unsafe_new_minus_f104'] = dict(estimate=float(arm_rate(new) - arm_rate(f104)),
                                               ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])
    # arena-level sign test for P1
    signs = []
    for a in NEW:
        r = paired(by, [g for g in by if by[g]['arena'] == a], 'n2', 'rule', 'unsafe'); signs.append(np.sign(r['b_worse'] - r['a_worse']))
    wins, losses = int(sum(s > 0 for s in signs)), int(sum(s < 0 for s in signs))
    out['P1_arena_sign_test'] = dict(n2_better_arenas=wins, rule_better_arenas=losses, ties=len(NEW) - wins - losses, p=mcnemar(losses, wins))
    out['identical_picks'] = {f'{x}=={y}': sum(1 for g in by if x in by[g]['arms'] and by[g]['arms'][x]['same_as'] == y) for x, y in
                              (('rule', 'n2'), ('straight6', 'n2'), ('straight6', 'rule'), ('rule_fixed2', 'n2_fixed2'), ('straight2', 'n2'))}
    json.dump(out, open(G + '/test_results.json', 'w'), indent=1)
    print(json.dumps(out['n_complete']))
    print('\nUNSAFE (fail or slide) — declared family, exact McNemar, Holm')
    for r in res:
        print(f"  {r['test']:42s} n={r['n']:4d}  {100*r['rate_a']:5.1f}% vs {100*r['rate_b']:5.1f}%  "
              f"discordant {r['a_worse']} vs {r['b_worse']}  p={r['p']:.4f}  Holm {r['p_holm']:.4f}")
    for key in ('fail', 'tilt30'):
        print(f'\n{key.upper()}')
        for r in out['secondary'][key]:
            print(f"  {r['test']:42s} {100*r['rate_a']:5.1f}% vs {100*r['rate_b']:5.1f}%  {r['a_worse']} vs {r['b_worse']}  p={r['p']:.4f}")
    print('\nRATES by arena (unsafe / fail / tilt30 / median time)')
    for a, d in rates.items():
        print('  ' + a.ljust(10) + '  '.join(f"{arm}:{100*v['unsafe']:.1f}/{100*v['fail']:.1f}/{100*v['tilt30']:.1f}/{v['median_time_s'] or 0:.0f}s" for arm, v in d.items()))
    print('\ngap', out['gap_n2_unsafe_new_minus_f104'], '\narena sign test', out['P1_arena_sign_test'], '\nidentical picks', out['identical_picks'])


if __name__ == '__main__':
    main()
