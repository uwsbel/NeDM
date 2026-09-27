#!/usr/bin/env python3
"""Tables and post-hoc robustness checks for RESULTS_rigid.md (arena_gator_20260925, module E6b final analysis, 09-26).

Reads (all written earlier): e6/analysis/results_rigid_v1.json (frozen spec), results_rigid_v1_B2.json (task B fixed
2 m/s addendum), results_rigid_v1_addons.json (post-hoc add-on spec written before the index), family_final_E6b.json
(Holm family of four with the soil results), e6/index/rigid_eval_v1.json (+ _extras.json), e6/offline/offline_unseen.json,
e6/collection/*.json, the designed-route records of the unseen test groups (G3/tools/e6b/out/test_designed_records.json,
local copy e6/collection/test_designed_records.json).

Writes e6/analysis/rigid_tables.md (every table of RESULTS_rigid.md, generated, so no number is typed by hand) and
e6/analysis/results_rigid_v1_posthoc.json. POST-HOC (decided after the frozen-spec results were seen; descriptive):
  * the primary rigid contrasts with single ensembles (M3a / M3b / A3 vs M1a / M1b), each with an exact McNemar test
    (the composite arms of the family are 0 / 0.5 / 1 per group, so McNemar is not defined for them);
  * the primary contrasts split into their two parts: goal not reached, and backward-only (reached the goal but rolled
    backwards under throttle / below -0.30 m/s);
  * the fixed 2 m/s outcomes conditioned on the designed-route feasibility of each unseen group (does any of the 3
    designed constant 2 m/s routes reach the goal / stay safe).
  PYTHONPATH=src:scripts python scripts/ag_e6b_final_tables.py
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ag_analyze as AA  # noqa: E402
import ga_analyze as GAN  # noqa: E402

K3 = Path('artifacts/traverse/arena_gator_20260925')
AN = K3 / 'e6/analysis'
NEAR = ('g260', 'g271', 'g251', 'g247'); SPREAD = ('g258', 'g268', 'g263', 'g241')
PLAIN = {
    'M1a': 'f104 only (ensemble a)', 'M1b': 'f104 only (ensemble b)', 'M1': 'f104 only (mean of ensembles a and b)',
    'M2': 'two arenas, same total data (f104 + g203)', 'M3a': 'three arenas, same total data (ensemble a)',
    'M3b': 'three arenas, same total data (ensemble b)', 'M3': 'three arenas, same total data (mean of a and b)',
    'A3': 'three arenas, all data', 'straight6': 'straight route at 6 m/s', 'straight2': 'straight route at 2 m/s',
    'G': 'Gator-trained', 'H': 'HMMWV-trained (= M1a)'}


def plain(arm):
    base = arm.replace('_free', '').replace('_fx2', '').replace('_gator', '')
    return PLAIN.get(base, base)


def f1(x):
    return '-' if x is None else f'{x:.1f}'


def ci(c, d=1):
    return f'[{c[0]:+.{d}f}, {c[1]:+.{d}f}]'


def pv(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def main():
    res = json.load(open(AN / 'results_rigid_v1.json'))
    b2 = json.load(open(AN / 'results_rigid_v1_B2.json'))
    add = json.load(open(AN / 'results_rigid_v1_addons.json'))
    fam = json.load(open(AN / 'family_final_E6b.json'))
    idx = json.load(open(K3 / 'e6/index/rigid_eval_v1.json'))
    ext = json.load(open(K3 / 'e6/index/rigid_eval_v1_extras.json'))
    off = json.load(open(K3 / 'e6/offline/offline_unseen.json'))
    col = json.load(open(K3 / 'e6/collection/rigid_f104_gator_vs_hmmwv.json'))
    fea = json.load(open(K3 / 'e6/collection/test_designed_feasibility.json'))
    rec = json.load(open(K3 / 'e6/collection/test_designed_records.json'))
    L, post = [], {}
    P = L.append
    rates = res['rates']['rigid|hmmwv']
    rates_add = add['rates']['rigid|hmmwv']

    # ---------------------------------------------------------------- T1 rates
    P('## T1. Rates on the 8 unseen arenas (rigid, HMMWV, 2,000 groups; near = 4 arenas closest to f104, spread = 4 farther)\n')
    P('| arm (code) | plain label | mode | goal not reached: all / near / spread | unsafe: all / near / spread | of which backward-only (all) | median time, s (all) |')
    P('|---|---|---|---|---|---|---|')
    order = ['M1a_free', 'M1b_free', 'M1_free', 'M2_free', 'M3a_free', 'M3b_free', 'M3_free', 'A3_free', 'straight6',
             'M1a_fx2', 'M1b_fx2', 'M1_fx2', 'M2_fx2', 'M3a_fx2', 'M3b_fx2', 'M3_fx2', 'A3_fx2', 'straight2']
    for a in order:
        u, n_, s_ = rates['unseen'][a], rates['near'][a], rates['spread'][a]
        mode = 'speed free' if (a.endswith('_free') or a == 'straight6') else 'fixed 2 m/s'
        P(f"| {a} | {plain(a)} | {mode} | {u['fail']:.2f} / {n_['fail']:.2f} / {s_['fail']:.2f} | {u['unsafe']:.2f} / {n_['unsafe']:.2f} / {s_['unsafe']:.2f} | "
          f"{u['backward_only']:.2f} | {f1(u['median_time_s'])} |")
    P('\nComposite rows (M1, M3) are per-group means of the two ensembles (0, 0.5 or 1 per group); they have no times.\n')

    # ---------------------------------------------------------------- T2 per arena rates, fixed 2
    P('## T2. Fixed 2 m/s unsafe per unseen arena (%, 250 groups each), with the designed-route feasibility of the arena\n')
    ar_list = list(NEAR) + list(SPREAD)
    P('| arena | role | distance to nearest training arena | map error rmse, m | M1a | M1b | M2 | M3a | M3b | A3 | straight 2 m/s | groups with no safe designed 2 m/s route | groups with no goal-reaching designed 2 m/s route |')
    P('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    gi = {}
    for r in idx['rows']:
        gi.setdefault(r['arena'], (r['role'], r['dist_nearest_training'], r['map_err_rmse_m']))
    for ar in ar_list:
        t = res['rates']['rigid|hmmwv'][f'arena:{ar}']
        c2 = fea['arenas'][ar]['constant_2']
        P(f"| {ar} | {gi[ar][0]} | {gi[ar][1]:.2f} | {gi[ar][2]:.3f} | " + ' | '.join(f"{t[a]['unsafe']:.1f}" for a in
          ('M1a_fx2', 'M1b_fx2', 'M2_fx2', 'M3a_fx2', 'M3b_fx2', 'A3_fx2', 'straight2')) +
          f" | {100 - c2['any_safe_in_group_pct']:.1f} | {100 - c2['any_goal_in_group_pct']:.1f} |")
    P('\nSame table for goal not reached at fixed 2 m/s:\n')
    P('| arena | M1a | M1b | M2 | M3a | M3b | A3 | straight 2 m/s |')
    P('|---|---|---|---|---|---|---|---|')
    for ar in ar_list:
        t = res['rates']['rigid|hmmwv'][f'arena:{ar}']
        P(f"| {ar} | " + ' | '.join(f"{t[a]['fail']:.1f}" for a in ('M1a_fx2', 'M1b_fx2', 'M2_fx2', 'M3a_fx2', 'M3b_fx2', 'A3_fx2', 'straight2')) + ' |')
    P('\nSpeed free, goal not reached / unsafe per arena (%):\n')
    P('| arena | M1a | M1b | M2 | M3a | M3b | A3 | straight 6 m/s |')
    P('|---|---|---|---|---|---|---|---|')
    for ar in ar_list:
        t = res['rates']['rigid|hmmwv'][f'arena:{ar}']
        P(f"| {ar} | " + ' | '.join(f"{t[a]['fail']:.1f} / {t[a]['unsafe']:.1f}" for a in ('M1a_free', 'M1b_free', 'M2_free', 'M3a_free', 'M3b_free', 'A3_free', 'straight6')) + ' |')

    # ---------------------------------------------------------------- T3 family
    P('\n## T3. The declared family of four (PLAN 7.3; Holm at 0.05 over one-sided cluster-bootstrap p; 4,000 resamples)\n')
    P('| test | comparison (plain) | groups | rate test / reference, % | difference, points | 90 % cluster interval | 95 % group interval | one-sided p (cluster) | Holm-adjusted p | decision |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    for x in fam['tests']:
        r = x['result']
        lab = 'goal not reached' if x['label'] == 'fail' else 'unsafe'
        P(f"| {x['name']} | {plain(x['test'])} vs {plain(x['ref'])}, {'soil speed free' if x['world'] == 'crm' else 'rigid fixed 2 m/s'}, {lab} | {r['n']} | "
          f"{r['rate_test']:.2f} / {r['rate_ref']:.2f} | {r['diff_pts']:+.2f} | {ci(r['cluster']['ci90'])} | {ci(r['group']['ci95'])} | {pv(x['p_one_sided_cluster'])} | {x['p_holm']:.4f} | {x['decision']} |")
    rf = {f['name']: f for f in res['family']}
    P('\nRigid-only run of the frozen spec (soil entered at p = 1, so these adjusted p are upper bounds): ' +
      '; '.join(f"{n} Holm {rf[n]['holm']['p_adjusted']:.4f} -> {rf[n]['decision']}" for n in ('P3_rigid_fx2_M3_vs_M1', 'P4_rigid_fx2_A3_vs_M1')) + '.\n')

    # ---------------------------------------------------------------- T4 per arena of the primary rigid tests
    P('## T4. Rigid primary tests per unseen arena (difference in unsafe points, 95 % group-bootstrap interval, groups test worse / better)\n')
    P('| arena | role | three arenas same total (M3) vs f104 only (M1) | three arenas all data (A3) vs f104 only (M1) |')
    P('|---|---|---|---|')
    p3, p4 = rf['P3_rigid_fx2_M3_vs_M1']['result'], rf['P4_rigid_fx2_A3_vs_M1']['result']
    for ar in ar_list:
        a, b = p3['per_arena'][ar], p4['per_arena'][ar]
        P(f"| {ar} | {a['role']} | {a['diff_pts']:+.1f} {ci(a['ci95'])} ({a['test_worse']} / {a['test_better']}) | {b['diff_pts']:+.1f} {ci(b['ci95'])} ({b['test_worse']} / {b['test_better']}) |")
    for nm, r in (('M3 vs M1', p3), ('A3 vs M1', p4)):
        re_ = r['random_effects']; ns = r['near_spread']; nm_s = r['near_minus_spread']
        P(f"\n{nm}: arenas better {r['arena_signs']['better']} / worse {r['arena_signs']['worse']}; random-effects pooled {re_['pooled_pts']:+.2f} {ci(re_['ci95'], 2)} "
          f"(tau {re_['tau_pts']:.2f} points, I2 {100 * re_['I2']:.0f} %); near {ns['near']['diff_pts']:+.2f} {ci(ns['near']['cluster_ci95'], 2)}, spread {ns['spread']['diff_pts']:+.2f} "
          f"{ci(ns['spread']['cluster_ci95'], 2)}, near minus spread {nm_s['diff_pts']:+.2f} {ci(nm_s['ci95'], 2)}; per-arena effect vs distance to the nearest training arena: "
          f"slope {r['vs_distance_nearest_training']['ols_slope']:+.2f} points per unit distance, Spearman {r['vs_distance_nearest_training']['spearman']:+.2f} (8 arenas); "
          f"vs map error: Spearman {r['vs_map_error']['spearman']:+.2f}; cluster wins / losses {r['cluster']['cluster_wins']} / {r['cluster']['cluster_losses']} of "
          f"{r['cluster']['clusters']} clusters; design-feature clustering 90 % {ci(r['cluster_alt']['ci90'])}, p {pv(r['cluster_alt']['p_one_sided'])}.")

    # ---------------------------------------------------------------- post-hoc: single ensembles + McNemar, parts of unsafe
    by, ginfo = AA.build_by(idx['rows'], res['spec']['combine'])
    byw = by[('rigid', 'hmmwv')]
    G = AA.set_groups(ginfo, 'unseen', list(byw))
    rng = np.random.default_rng(1)
    AA.CLUSTER_KEY[0] = 'cluster'
    ph = {}
    P('\n## T5. Post-hoc robustness of the rigid primary tests (decided after seeing T3; descriptive, unadjusted)\n')
    P('| comparison (fixed 2 m/s, 8 unseen arenas) | label | rate test / reference, % | difference | 90 % cluster interval | groups test worse / better | exact McNemar two-sided |')
    P('|---|---|---|---|---|---|---|')
    pairs = [('M3a_fx2', 'M1a_fx2'), ('M3a_fx2', 'M1b_fx2'), ('M3b_fx2', 'M1a_fx2'), ('M3b_fx2', 'M1b_fx2'), ('A3_fx2', 'M1a_fx2'), ('A3_fx2', 'M1b_fx2'),
             ('M2_fx2', 'M1a_fx2'), ('M1a_fx2', 'M1b_fx2'), ('M3a_fx2', 'M3b_fx2')]
    for t, rf_ in pairs:
        for lab in ('unsafe', 'fail', 'backward_only'):
            c = AA.contrast(byw, ginfo, G, t, rf_, lab, rng, 4000, 2.0, detail=False)
            d = c['discordant']
            ph[f'{t}_vs_{rf_}_{lab}'] = c
            P(f"| {t} ({plain(t)}) vs {rf_} ({plain(rf_)}) | {lab} | {c['rate_test']:.2f} / {c['rate_ref']:.2f} | {c['diff_pts']:+.2f} | {ci(c['cluster']['ci90'])} | "
              f"{d['test_worse']} / {d['test_better']} | {d['mcnemar_two_sided']:.2g} |")
    for nm in ('M3_fx2', 'A3_fx2'):
        for lab in ('fail', 'backward_only'):
            c = AA.contrast(byw, ginfo, G, nm, 'M1_fx2', lab, rng, 4000, 2.0, detail=False)
            ph[f'{nm}_vs_M1_fx2_{lab}'] = c
    P('\nThe family contrasts split into their two parts (composite arms): ' + '; '.join(
        f"{nm} vs M1 goal not reached {ph[f'{nm}_vs_M1_fx2_fail']['diff_pts']:+.2f} {ci(ph[f'{nm}_vs_M1_fx2_fail']['cluster']['ci90'])}, backward-only "
        f"{ph[f'{nm}_vs_M1_fx2_backward_only']['diff_pts']:+.2f} {ci(ph[f'{nm}_vs_M1_fx2_backward_only']['cluster']['ci90'])}" for nm in ('M3_fx2', 'A3_fx2')) +
      ' (unsafe = goal not reached + backward-only, exactly).\n')

    # ---------------------------------------------------------------- post-hoc: feasibility-conditioned
    recs = [x for x in rec['records'] if x.get('ok') and x.get('complete')]
    fg = defaultdict(lambda: defaultdict(list))
    for x in recs:
        g = x['id'].rsplit('_route_', 1)[0]
        fg[g][x['profile']].append(x)
    cond = {}
    for key, pred in (('some designed 2 m/s route safe', lambda g: any(not x['unsafe'] for x in fg[g]['constant_2'])),
                      ('no designed 2 m/s route safe, some reaches the goal', lambda g: (not any(not x['unsafe'] for x in fg[g]['constant_2'])) and any(not x['fail'] for x in fg[g]['constant_2'])),
                      ('no designed 2 m/s route reaches the goal', lambda g: not any(not x['fail'] for x in fg[g]['constant_2']))):
        Gs = [g for g in G if g in fg and len(fg[g]['constant_2']) == 3 and pred(g)]
        cond[key] = dict(n=len(Gs), **{a: dict(unsafe=100 * float(np.mean([byw[g][a]['unsafe'] for g in Gs])), fail=100 * float(np.mean([byw[g][a]['fail'] for g in Gs])))
                                        for a in ('M1_fx2', 'M2_fx2', 'M3_fx2', 'A3_fx2', 'straight2')})
    P('## T6. Fixed 2 m/s outcomes by designed-route feasibility of the unseen group (post-hoc; the 3 designed constant 2 m/s routes of each group, HMMWV, driven in the rigid collection)\n')
    P('| groups | n | f104 only (M1): unsafe / not reached | two arenas (M2) | three arenas same total (M3) | three arenas all data (A3) | straight 2 m/s |')
    P('|---|---|---|---|---|---|---|')
    for k, v in cond.items():
        P(f"| {k} | {v['n']} | " + ' | '.join(f"{v[a]['unsafe']:.1f} / {v[a]['fail']:.1f}" for a in ('M1_fx2', 'M2_fx2', 'M3_fx2', 'A3_fx2', 'straight2')) + ' |')
    post['feasibility_conditioned'] = cond
    post['designed_records'] = dict(file='e6/collection/test_designed_records.json', n=len(recs), groups=len(fg))

    # ---------------------------------------------------------------- T7 secondary list
    P('\n## T7. Secondary contrasts (frozen spec, then the add-on spec; unadjusted). Tool decision: "improves" if the one-sided p <= 0.05 and the difference is negative, else "no meaningful difference" if the 90 % interval lies inside +-2 points, else "inconclusive". The column "within +-2" shows the interval test on its own: a small effect can be both detectable and inside the 2-point band.\n')
    P('| name | comparison (plain) | set | label | groups | rates, % | difference | 90 % cluster interval | p (one-sided) | within +-2 points (90 %) | tool decision |')
    P('|---|---|---|---|---|---|---|---|---|---|---|')
    for c in res['contrasts'] + [dict(x, _addon=True) for x in add['contrasts']]:
        r = c['result']
        if not r.get('n'):
            continue
        P(f"| {c['name']}{' (add-on)' if c.get('_addon') else ''} | {plain(c['test'])} vs {plain(c['ref'])} | {c['set']} | {c['label']} | {r['n']} | {r['rate_test']:.2f} / {r['rate_ref']:.2f} | "
          f"{r['diff_pts']:+.2f} | {ci(r['cluster']['ci90'])} | {pv(r['cluster']['p_one_sided'])} | {'yes' if r['within_margin_90'] else 'no'} | {c['decision']} |")

    # ---------------------------------------------------------------- T8 gaps
    P('\n## T8. Generalisation gap: rate on the 8 unseen arenas minus rate in distribution (95 % intervals; independent bootstraps)\n')
    P('In distribution = the 200 declared f104 hill/crater groups (frozen spec) or the g203 / g228 held-out groups (add-on spec; in distribution for M3 and A3, '
      'an arena never seen by M1; g203 was seen by M2). "vs straight" = [model - straight](unseen) - [model - straight](in distribution).\n')
    P('| model | label | f104 in distribution: unseen / in-dist, % | gap | vs straight | held-out g203+g228: in-dist, % | gap | vs straight |')
    P('|---|---|---|---|---|---|---|---|')
    ga = {(g['model'], g['indist']): g['result'] for g in res['gaps'] + add['gaps']}
    for m in ('M1_fx2', 'M2_fx2', 'M3_fx2', 'A3_fx2', 'M1_free', 'M2_free', 'M3_free', 'A3_free'):
        for lab in (('unsafe', 'fail') if m.endswith('fx2') else ('fail', 'unsafe')):
            a, b = ga[(m, 'indist_f104')][lab], ga[(m, 'heldout')][lab]
            P(f"| {m} ({plain(m)}) | {lab} | {a['raw']['rate_unseen']:.2f} / {a['raw']['rate_indist']:.2f} | {a['raw']['gap_pts']:+.2f} {ci(a['raw']['ci95'])} | "
              f"{a['vs_straight']['did_pts']:+.1f} {ci(a['vs_straight']['ci95'])} | {b['raw']['rate_indist']:.2f} | {b['raw']['gap_pts']:+.2f} {ci(b['raw']['ci95'])} | "
              f"{b['vs_straight']['did_pts']:+.1f} {ci(b['vs_straight']['ci95'])} |")
    P('\nHeld-out groups per arena, fixed 2 m/s unsafe (%): ' + '; '.join(
        f"{ar}: " + ', '.join(f"{a} {rates_add[f'heldout_{ar}'][a]['unsafe']:.1f}" for a in ('M1a_fx2', 'M1b_fx2', 'M2_fx2', 'M3a_fx2', 'M3b_fx2', 'A3_fx2', 'straight2'))
        for ar in ('g203', 'g228')) + '.')
    P('\nIn-distribution rates (f104 200 groups; held-out g203 / g228 150 each; dev g217 150), goal not reached / unsafe (%):\n')
    P('| arm | f104 in distribution | g203 held-out | g228 held-out | g217 dev | 8 unseen arenas |')
    P('|---|---|---|---|---|---|')
    for a in order:
        cells = []
        for s, src in (('indist_f104', rates), ('heldout_g203', rates_add), ('heldout_g228', rates_add), ('dev', rates), ('unseen', rates)):
            v = src.get(s, {}).get(a)
            cells.append('-' if v is None else f"{v['fail']:.1f} / {v['unsafe']:.1f}")
        P(f"| {a} ({plain(a)}) | " + ' | '.join(cells) + ' |')

    # ---------------------------------------------------------------- T9 dose
    P('\n## T9. Dose response f104 only -> two arenas -> three arenas (same total data)\n')
    P('| set | mode, label | rates M1 / M2 / M3, % | slope, points per added arena (95 %) | steps M2-M1, M3-M2 |')
    P('|---|---|---|---|---|')
    for d in res['dose'] + add['dose']:
        r = d['result']
        st = list(r['steps'].values())
        P(f"| {d['set']} | {'fixed 2 m/s' if 'fx2' in d['arms'][0] else 'speed free'}, {d['label']} | {' / '.join(f'{v:.2f}' for v in r['rates'].values())} | "
          f"{r['slope_pts_per_arena']:+.2f} {ci(r['slope_ci95'], 2)} | {st[0]['diff_pts']:+.2f} {ci(st[0]['cluster']['ci90'])}, {st[1]['diff_pts']:+.2f} {ci(st[1]['cluster']['ci90'])} |")

    # ---------------------------------------------------------------- T10 time ratios
    P('\n## T10. Median time ratio on joint successes (test / reference)\n')
    for t in res['time_ratio'] + b2['time_ratio'] + add['time_ratio']:
        r = t['result']
        if r.get('n'):
            P(f"- {t['test']} ({plain(t['test'])}) / {t['ref']} ({plain(t['ref'])}), {t['set']}: {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, (list, dict))})}"
              + (f" ci95 {[round(x, 3) for x in r['ci95']]}" if isinstance(r.get('ci95'), list) else ''))

    # ---------------------------------------------------------------- T11 task B
    P('\n## T11. Task B rigid (800 f104 groups: 600 f104_pair + 200 f104_crm_eval), rates\n')
    P('| arm (code) | plain label | vehicle driven | mode | goal reached, % | unsafe, % | unsafe without the backward clause (= not reached), % | backward-only, % | tilt > 30 deg, % | median time, s | chassis ground contact, % of drives |')
    P('|---|---|---|---|---|---|---|---|---|---|---|')
    cc = ext['summary']['chassis_contact_share']

    def contact(veh, arm):
        v = [cc.get(f'{veh}|{arm}|{s}') for s in ('f104_suite', 'indist_f104')]
        v = [x for x in v if x]
        return sum(x['n'] * x['share'] for x in v) / max(1, sum(x['n'] for x in v)) if v else None
    for veh, key, arms in (('gator', 'rigid|gator', ('G_free_gator', 'H_free_gator', 'straight6_gator', 'G_fx2_gator', 'H_fx2_gator', 'straight2_gator')),
                           ('hmmwv', 'rigid|hmmwv', ('M1a_free', 'straight6'))):
        src = b2['rates'][key]['f104_800'] if veh == 'gator' and 'G_fx2_gator' in b2['rates'].get(key, {}).get('f104_800', {}) else None
        for a in arms:
            t = (b2['rates'][key]['f104_800'] if a in b2['rates'].get(key, {}).get('f104_800', {}) else res['rates'][key]['f104_800'])[a]
            lab = {'G_free_gator': 'Gator-trained on the Gator', 'H_free_gator': 'HMMWV-trained on the Gator', 'straight6_gator': 'straight 6 m/s on the Gator',
                   'G_fx2_gator': 'Gator-trained on the Gator', 'H_fx2_gator': 'HMMWV-trained on the Gator', 'straight2_gator': 'straight 2 m/s on the Gator',
                   'M1a_free': 'HMMWV-trained (H = M1a) on the HMMWV', 'straight6': 'straight 6 m/s on the HMMWV'}[a]
            mode = 'fixed 2 m/s' if ('fx2' in a or 'straight2' in a) else 'speed free'
            P(f"| {a} | {lab} | {veh} | {mode} | {t['goal_reached']:.2f} | {t['unsafe']:.2f} | {t['unsafe_noback']:.2f} | {t['backward_only']:.2f} | {t['tilt30']:.1f} | {f1(t['median_time_s'])} | {f1(contact(veh, a))} |")
    P('\nTask B contrasts (difference test - reference in points; 90 % cluster interval; one-sided p; non-inferiority = one-sided upper 95 % bound < +2 points):\n')
    P('| name | comparison | label | rates, % | difference | 90 % cluster interval | p | upper 95 % bound | within 2 points (non-inferior) | groups test worse / better | identical picks |')
    P('|---|---|---|---|---|---|---|---|---|---|---|')
    for c in [x for x in res['contrasts'] if x['name'].startswith('B_')] + b2['contrasts']:
        r = c['result']
        P(f"| {c['name']} | {c['test']} vs {c['ref']} | {c['label']} | {r['rate_test']:.2f} / {r['rate_ref']:.2f} | {r['diff_pts']:+.2f} | {ci(r['cluster']['ci90'])} | {pv(r['cluster']['p_one_sided'])} | "
          f"{r['noninferiority']['upper95_one_sided_cluster']:+.2f} | {'yes' if r['noninferiority']['pass_cluster'] else 'no'} | {r['discordant']['test_worse']} / {r['discordant']['test_better']} | {r['identical_picks']} |")
    P('\nHeadroom closed = (straight - model) / straight on the failure (or unsafe) rate:\n')
    for h in res['headroom'] + b2['headroom']:
        r = h['result']
        P(f"- {h['model']} vs {h['straight']} ({h['label']}): {r['rate_model']:.2f} % vs {r['rate_straight']:.2f} %, closed {100 * r['headroom_closed']:.0f} %"
          + (f" [{100 * r['ci95'][0]:.0f}, {100 * r['ci95'][1]:.0f}]" if r.get('ci95') else '') + f", n {r['n']}")

    # ---------------------------------------------------------------- T12 collection
    P('\n## T12. Task B collection: the same 24,000 rigid f104 routes driven by both vehicles (1,200 groups x 20 routes)\n')
    P('| speed profile | routes | goal not reached: Gator / HMMWV, % | only Gator / only HMMWV fails | unsafe: Gator / HMMWV, % | backward-only (rolled back but reached the goal): Gator / HMMWV, % | tilt > 30 deg: Gator / HMMWV, % | simulated h: Gator / HMMWV | median time ratio Gator / HMMWV (joint goals) |')
    P('|---|---|---|---|---|---|---|---|---|')
    for k, v in col['by_profile'].items():
        P(f"| {k} | {v['n']} | {v['fail']['gator']:.1f} / {v['fail']['hmmwv']:.1f} | {v['fail']['gator_only']} / {v['fail']['hmmwv_only']} | {v['unsafe']['gator']:.1f} / {v['unsafe']['hmmwv']:.1f} | "
          f"{v['backward_only']['gator']:.1f} / {v['backward_only']['hmmwv']:.1f} | {v['tilt30']['gator']:.1f} / {v['tilt30']['hmmwv']:.1f} | {v['sim_hours']['gator']:.1f} / {v['sim_hours']['hmmwv']:.1f} | {v['time_ratio_joint_goals_median']:.3f} |")
    gc, hc = col['gator_checks'], col['hmmwv_checks']
    P(f"\nStatuses Gator {col['status']['gator']}; HMMWV {col['status']['hmmwv']}. Gator checks: launch failed {gc['launch_failed']}, native height failed {gc['native_height_failed']}, "
      f"non-finite {gc['not_finite']}, rollovers {gc['rollover']}, chassis ground contact on {gc['chassis_contact_routes']} of {gc['chassis_contact_recorded']} routes "
      f"({100 * gc['chassis_contact_routes'] / gc['chassis_contact_recorded']:.1f} %; max {gc['chassis_contact_max_n'] / 1000:.0f} kN), lowest hull point below the surface on "
      f"{gc['belly_below_surface_routes']} routes (min {gc['belly_min_m']:.2f} m). HMMWV: chassis contact not recorded in that collection (the rich telemetry was not kept), rollovers {hc['rollover']}.")

    # ---------------------------------------------------------------- T13 offline
    P('\n## T13. Offline ranking on the 8 unseen arenas (standing start, the 12 designed routes of each of the 2,000 groups; within-group AUC)\n')
    P('| model (code) | plain label | AUC unsafe: all / near / spread | AUC not reached | lowest-risk route unsafe, % (random route) |')
    P('|---|---|---|---|---|')
    tab = off['table']
    for t in ('M1a_deploy', 'M1b_deploy', 'M2_deploy', 'M3a_deploy', 'M3b_deploy', 'A3_deploy', 'G_deploy', 'M1_holdout', 'M2_holdout', 'M3_holdout', 'A3_holdout',
              'LC545_holdout', 'LC272_holdout', 'LOAO1_g203_holdout', 'LOAO1_g228_holdout', 'LOAO2_f104_g203_holdout', 'LOAO2_f104_g228_holdout', 'LOAO2_g203_g228_holdout'):
        v = tab.get(t)
        if not v:
            continue
        lab = {'M1_holdout': 'f104 only, holdout mode (865 fitted groups)', 'M2_holdout': 'two arenas, holdout mode', 'M3_holdout': 'three arenas same total, holdout mode',
               'A3_holdout': 'three arenas all data, holdout mode', 'LC545_holdout': 'f104 only, 431 fitted groups', 'LC272_holdout': 'f104 only, 213 fitted groups',
               'LOAO1_g203_holdout': 'g203 only (~433 groups)', 'LOAO1_g228_holdout': 'g228 only (~436 groups)', 'LOAO2_f104_g203_holdout': 'f104 + g203 (~435 groups)',
               'LOAO2_f104_g228_holdout': 'f104 + g228 (~441 groups)', 'LOAO2_g203_g228_holdout': 'g203 + g228 (~449 groups)', 'G_deploy': 'Gator-trained (scored on HMMWV drives)'}.get(t, plain(t.replace('_deploy', '')))
        P(f"| {t} | {lab} | {v['all']['W_unsafe']:.3f} / {v['near']['W_unsafe']:.3f} / {v['spread']['W_unsafe']:.3f} | {v['all']['W_fail']:.3f} | {v['all']['pick_unsafe']:.1f} ({v['all']['random_unsafe']:.1f}) |")
    P('\nOffline paired contrasts (group bootstrap, 2,000 resamples; AUC difference): ' + '; '.join(
        f"{k}: unsafe {v['all']['W_unsafe']['diff']:+.4f} [{v['all']['W_unsafe']['ci95'][0]:+.4f}, {v['all']['W_unsafe']['ci95'][1]:+.4f}]" for k, v in off['contrasts'].items()) + '.')

    # ---------------------------------------------------------------- T14 drives
    s = ext['summary']
    P('\n## T14. Drives and integrity\n')
    P(f"- Index rows {idx['summary']['rows']} (arm x group), all driven, 0 missing; distinct drives {s['distinct_runs']} (identical routes of two arms of a group driven once); "
      f"pool shards {s['pool']['shards']}, rows {s['pool']['rows']}, complete {s['pool']['complete']}; shards with incomplete rows {len(s['pool']['shards_incomplete'])}, "
      f"host mismatch {len(s['pool']['shards_host_mismatch'])}, takeovers {len(s['pool']['shards_takeover'])}, second attempts {len(s['pool']['shards_second_attempt'])}, "
      f"non-zero runner exit {len(s['pool']['runner_rc_nonzero'])}; {s['pool']['nodes']} nodes, {s['pool']['wall_h_sum']:.1f} shard wall hours.")
    P(f"- Hosts: every run has a recorded host ({s['runs_without_host']} without); every group's drives within one task file come from one host; "
      f"{s['groups_multi_host']} f104 groups have their task B fixed 2 m/s drives (added at 14:10, separate shards) on a different node than their speed-free drives: "
      "no contrast pairs arms across those two task files.")
    Path(AN / 'rigid_tables.md').write_text('\n'.join(L) + '\n')
    post['single_ensemble_and_parts'] = {k: {kk: vv for kk, vv in v.items() if kk in ('test', 'ref', 'label', 'n', 'rate_test', 'rate_ref', 'diff_pts', 'group', 'cluster', 'discordant')}
                                         for k, v in ph.items()}
    post['note'] = ('POST-HOC, decided after the frozen-spec results were seen; descriptive only. Bootstrap seed 1, 4,000 resamples, cluster = arena : nearest terrain feature.')
    json.dump(post, open(AN / 'results_rigid_v1_posthoc.json', 'w'), indent=1, default=float)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
