#!/usr/bin/env python3
"""Assemble the machine-readable rigid results of tasks A and B (arena_gator_20260925, module E6b) into
K3/RESULTS_rigid.json, and print a digest used for RESULTS_rigid.md.

Inputs (all written earlier by E6b): e6/analysis/results_rigid_v1.json (+ _B2), e6/offline/offline_unseen.json,
e6/collection/{rigid_f104_gator_vs_hmmwv,test_designed_feasibility}.json, e6/index/rigid_eval_v1_extras.json,
e6/index/rigid_eval_v1.json, e5/offline/auc_summary.json (E5a, in-arena offline scores).
Holm (PLAN 7.3): the declared family includes the two soil tests, which have no drives yet (p = 1 in the rigid run). Holm
adjusted p-values do not decrease when another p-value increases, so the rigid adjusted p here is an UPPER bound; the
lower bound is Holm with both soil p-values at 0. A rigid rejection is final when the upper bound is <= alpha; no
rejection is possible when the lower bound is > alpha; in between it waits for the soil tests.
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

K3 = Path('artifacts/traverse/arena_gator_20260925')
E6 = K3 / 'e6'
PLAIN = {
    'M1a_free': 'f104 only (ensemble a), speed free', 'M1b_free': 'f104 only (ensemble b), speed free', 'M1_free': 'f104 only (both ensembles), speed free',
    'M2_free': 'f104 + g203, same total data, speed free', 'M3a_free': 'three arenas, same total data (a), speed free',
    'M3b_free': 'three arenas, same total data (b), speed free', 'M3_free': 'three arenas, same total data (both), speed free',
    'A3_free': 'three arenas, all their data, speed free', 'straight6': 'straight route at 6 m/s',
    'M1a_fx2': 'f104 only (a), fixed 2 m/s', 'M1b_fx2': 'f104 only (b), fixed 2 m/s', 'M1_fx2': 'f104 only (both), fixed 2 m/s',
    'M2_fx2': 'f104 + g203, fixed 2 m/s', 'M3a_fx2': 'three arenas same total (a), fixed 2 m/s', 'M3b_fx2': 'three arenas same total (b), fixed 2 m/s',
    'M3_fx2': 'three arenas same total (both), fixed 2 m/s', 'A3_fx2': 'three arenas all data, fixed 2 m/s', 'straight2': 'straight route at 2 m/s',
    'G_free_gator': 'Gator-trained planner on the Gator, speed free', 'H_free_gator': 'HMMWV-trained planner on the Gator, speed free',
    'straight6_gator': 'straight 6 m/s on the Gator', 'G_fx2_gator': 'Gator-trained planner on the Gator, fixed 2 m/s',
    'H_fx2_gator': 'HMMWV-trained planner on the Gator, fixed 2 m/s', 'straight2_gator': 'straight 2 m/s on the Gator'}


def holm_bounds(fam, alpha=0.05):
    """(upper, lower) Holm-adjusted p for each rigid family member; soil p = 1 (upper) or 0 (lower)."""
    def holm(p):
        m = len(p); o = np.argsort(p, kind='stable'); adj = np.empty(m); run = 0.0
        for r, i in enumerate(o):
            run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
        return adj
    p = [f['holm']['p_one_sided_cluster'] for f in fam]
    soil = [i for i, f in enumerate(fam) if f['world'] == 'crm']
    up = holm([1.0 if i in soil else v for i, v in enumerate(p)])
    lo = holm([0.0 if i in soil else v for i, v in enumerate(p)])
    out = {}
    for i, f in enumerate(fam):
        if i in soil:
            continue
        status = 'rejected (final: soil results can only lower this bound)' if up[i] <= alpha else \
            ('not rejected (final: even with soil p = 0 the bound exceeds alpha)' if lo[i] > alpha else 'provisional: depends on the soil tests')
        out[f['name']] = dict(p_one_sided_cluster=p[i], holm_adjusted_upper=float(up[i]), holm_adjusted_lower=float(lo[i]), status=status)
    return out


def slim(c):
    r = c['result']
    if not r.get('n'):
        return dict(name=c['name'], n=0)
    d = dict(name=c['name'], world=c['world'], vehicle=c['vehicle'], set=c['set'], label=c['label'], test=c['test'], ref=c['ref'],
             test_plain=PLAIN.get(c['test'], c['test']), ref_plain=PLAIN.get(c['ref'], c['ref']), n=r['n'],
             rate_test_pct=r['rate_test'], rate_ref_pct=r['rate_ref'], diff_pts=r['diff_pts'],
             group_ci95=r['group']['ci95'], cluster_ci90=r['cluster']['ci90'], cluster_ci95=r['cluster']['ci95'],
             p_one_sided_cluster=r['cluster']['p_one_sided'], clusters=r['cluster']['clusters'],
             cluster_alt_ci90=r['cluster_alt']['ci90'], mcnemar_two_sided=r['discordant']['mcnemar_two_sided'],
             test_worse=r['discordant']['test_worse'], test_better=r['discordant']['test_better'], identical_picks=r['identical_picks'],
             noninferiority_upper95=r['noninferiority']['upper95_one_sided_cluster'], noninferior_2pts=r['noninferiority']['pass_cluster'],
             decision=c.get('decision'))
    if 'per_arena' in r:
        d['per_arena'] = {a: dict(n=v['n'], test=v['rate_test'], ref=v['rate_ref'], diff=v['diff_pts'], ci95=v['ci95'], role=v['role'])
                          for a, v in r['per_arena'].items()}
        d['arena_signs'] = r.get('arena_signs'); d['random_effects'] = r.get('random_effects')
        d['near_spread'] = r.get('near_spread'); d['near_minus_spread'] = r.get('near_minus_spread')
        d['vs_distance_nearest_training'] = r.get('vs_distance_nearest_training'); d['vs_map_error'] = r.get('vs_map_error')
    if 'time_ratio' in r:
        d['time_ratio'] = r['time_ratio']
    if c.get('holm'):
        d['holm_rigid_only_run'] = c['holm']
    return d


def main():
    res = json.load(open(E6 / 'analysis/results_rigid_v1.json'))
    b2 = json.load(open(E6 / 'analysis/results_rigid_v1_B2.json')) if (E6 / 'analysis/results_rigid_v1_B2.json').exists() else None
    off = json.load(open(E6 / 'offline/offline_unseen.json'))
    col = json.load(open(E6 / 'collection/rigid_f104_gator_vs_hmmwv.json'))
    fea = json.load(open(E6 / 'collection/test_designed_feasibility.json'))
    ext = json.load(open(E6 / 'index/rigid_eval_v1_extras.json'))
    e5 = json.load(open(K3 / 'e5/offline/auc_summary.json')) if (K3 / 'e5/offline/auc_summary.json').exists() else None
    idx = json.load(open(E6 / 'index/rigid_eval_v1.json'))
    fam = [f for f in res['family']]
    hb = holm_bounds(fam)
    out = dict(schema='ag_results_rigid_v1', created=__import__('time').strftime('%Y-%m-%d %H:%M:%S'),
               spec=dict(main='e6/analysis/spec_rigid_v1.json', addendum_B2='e6/analysis/spec_rigid_v1_B2.json'),
               labels=dict(fail='goal not reached', unsafe='not (goal reached and < 0.05 s rolling backwards under throttle and min forward speed > -0.30 m/s), after the 1 s settle',
                           backward_only='unsafe but goal reached (only the backward-motion clauses)', plain_names=PLAIN),
               drives=dict(index_rows=idx['summary']['rows'], driven=idx['summary']['driven'], missing=idx['summary']['missing'],
                           distinct_runs=ext['summary']['distinct_runs'], groups_multi_host=ext['summary']['groups_multi_host'],
                           hosts=len(ext['summary']['hosts'])),
               task_A=dict(primary_family=[slim(f) for f in fam if f['world'] == 'rigid'], holm_provisional=hb,
                           soil_family_members='P1_soil_M3_vs_M1, P2_soil_A3_vs_M1: no soil drives yet (p = 1 placeholder)'),
               task_B=dict(), offline_unseen=off, collection_B=col, feasibility_unseen=fea)
    sec = {c['name']: slim(c) for c in res['contrasts']}
    out['task_A']['secondary'] = {k: v for k, v in sec.items() if k.startswith('A_')}
    out['task_B']['speed_free'] = {k: v for k, v in sec.items() if k.startswith('B_')}
    out['task_A']['gaps'] = res['gaps']; out['task_A']['dose'] = res['dose']
    out['time_ratio'] = res['time_ratio']; out['headroom'] = res['headroom']
    out['rates'] = res['rates']
    if b2:
        out['task_B']['fixed2'] = {c['name']: slim(c) for c in b2['contrasts']}
        out['task_B']['fixed2_headroom'] = b2['headroom']; out['task_B']['fixed2_rates'] = b2['rates']; out['task_B']['fixed2_time_ratio'] = b2['time_ratio']
    out['ground_contact_eval'] = ext['summary']['chassis_contact_share']
    if e5:
        out['offline_training_arenas_E5a'] = e5
    json.dump(out, open(K3 / 'RESULTS_rigid.json', 'w'), indent=1, default=float)
    # digest
    P = print
    P('HOLM', json.dumps(hb, indent=1))
    for f in out['task_A']['primary_family']:
        P('PRIMARY', f['name'], f"{f['test']} {f['rate_test_pct']:.2f} vs {f['ref']} {f['rate_ref_pct']:.2f} diff {f['diff_pts']:+.2f} ci90 {f['cluster_ci90']} "
          f"p1 {f['p_one_sided_cluster']:.4f} mcn {f['mcnemar_two_sided']} worse/better {f['test_worse']}/{f['test_better']} n {f['n']}")
        P('   per arena', {a: round(v['diff'], 1) for a, v in f['per_arena'].items()}, 'signs', f['arena_signs'])
        P('   RE', f['random_effects'] and {k: round(v, 2) if isinstance(v, float) else v for k, v in f['random_effects'].items()})
        P('   near/spread', f['near_spread'], f['near_minus_spread'])
        P('   vs distance', f['vs_distance_nearest_training'], 'vs map err', f['vs_map_error'])
    for k, v in sec.items():
        if v.get('n'):
            P('SEC', k, f"{v['rate_test_pct']:.2f} vs {v['rate_ref_pct']:.2f} diff {v['diff_pts']:+.2f} ci90 {[round(x, 2) for x in v['cluster_ci90']]} p1 {v['p_one_sided_cluster']:.4f} "
              f"NI {v['noninferior_2pts']} ({v['noninferiority_upper95']:.2f}) n {v['n']} -> {v['decision']}")
    if b2:
        for k, v in out['task_B']['fixed2'].items():
            P('B2', k, f"{v['rate_test_pct']:.2f} vs {v['rate_ref_pct']:.2f} diff {v['diff_pts']:+.2f} ci90 {[round(x, 2) for x in v['cluster_ci90']]} p1 {v['p_one_sided_cluster']:.4f} "
              f"NI {v['noninferior_2pts']} ({v['noninferiority_upper95']:.2f}) worse/better {v['test_worse']}/{v['test_better']} n {v['n']} -> {v['decision']}")


if __name__ == '__main__':
    main()
