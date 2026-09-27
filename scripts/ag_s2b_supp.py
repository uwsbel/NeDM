#!/usr/bin/env python3
"""Supplementary soil read-outs (arena_gator_20260925, S2 finish), written 2026-09-26 AFTER the soil outcomes were seen.
Nothing here changes the frozen analysis (e6/analysis/spec_soil_v1.json -> results_soil_v1.json); every block is a
declared post-hoc addition, computed with the unchanged statistics of scripts/ag_analyze.py (same bootstrap sizes, seed,
cluster key, margin):

  1. ensemble-pair robustness of P1 / P2: every single-ensemble pair (M3a|M3b vs M1a|M1b, A3 vs M1a|M1b), unseen, fail;
  2. training-noise sensitivity of P1 / P2: the two same-data ensemble contrasts (M1a vs M1b, M3a vs M3b) give a rough
     training-noise variance per ensemble (sigma_b^2 = mean(d^2) / 2); it is added to the cluster-bootstrap variance of
     P1 (composite vs composite: sigma_b^2) and P2 (single vs composite: 1.5 sigma_b^2); normal one-sided p, Holm with the
     rigid P3 / P4 p-values as declared;
  3. QA sensitivity: P1 / P2 without the groups where any drive of the contrast carries a crm_qa flag
     (e6/analysis/s2b/soil_extras_v1.json qa_flagged_runs);
  4. generalisation gap against the g203 / g228 held-out groups (in distribution for M3a / A3, unseen sibling arenas for
     M1a), raw and against straight 6 m/s (ag_analyze.gap with indist = 'heldout');
  5. in-arena effect of M3a per held-out arena (the frozen spec has A3 per arena only);
  6. per-arena descriptive regressions (8 points): M1 failure and M1 minus straight 6 against the distance to f104; the P1 /
     P2 effects against the distance to f104 and against how much closer the nearest of f104 / g203 / g228 is than f104;
  7. task B split: G vs H on the Gator on the 200 in-distribution and the other 600 f104 suite groups.
  PYTHONPATH=src:scripts python scripts/ag_s2b_supp.py --index $K3/e6/index/soil_eval_v1.json --extras $K3/e6/analysis/s2b/soil_extras_v1.json \
      --rigid $K3/e6/analysis/results_rigid_v1.json --out $K3/e6/analysis/results_soil_v1_supp.json
"""
import argparse, hashlib, json, sys, time
from math import erf, sqrt
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ag_analyze as AA  # noqa: E402

K3 = HERE.parent / 'artifacts/traverse/arena_gator_20260925'
MAPERR = K3 / 'arenas/map_lookup_error.json'
FROZEN = K3 / 'e6/analysis/spec_soil_v1.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm_sf(z):
    return 0.5 * (1 - erf(z / sqrt(2)))


def slim(r):
    keep = ('test', 'ref', 'label', 'n', 'rate_test', 'rate_ref', 'diff_pts', 'group', 'cluster', 'cluster_alt', 'discordant', 'identical_picks',
            'within_margin_90', 'noninferiority', 'per_arena', 'arena_signs', 'random_effects', 'near_spread', 'near_minus_spread', 'time_ratio')
    return {k: r[k] for k in keep if k in r}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index', required=True); ap.add_argument('--extras', required=True); ap.add_argument('--rigid')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    spec = json.load(open(FROZEN))
    boot, margin, alpha = int(spec['boot']), float(spec['margin_pts']), float(spec['alpha'])
    AA.CLUSTER_KEY[0] = spec.get('cluster_key', 'cluster')
    rng = np.random.default_rng(int(spec['seed']) + 1)
    rows, meta = AA.load_index([a.index])
    by, ginfo = AA.build_by(rows, spec['combine'])
    H = by[('crm', 'hmmwv')]; Gt = by[('crm', 'gator')]
    U = AA.set_groups(ginfo, 'unseen', list(H))
    out = dict(schema='ag_s2b_supp_v1', tool='scripts/ag_s2b_supp.py', tool_sha256=sha(__file__), created=time.strftime('%Y-%m-%d %H:%M:%S'),
               note='post-hoc supplementary read-outs written after the soil outcomes were seen; the declared analysis is results_soil_v1.json',
               frozen_spec=dict(file=str(FROZEN), sha256=sha(FROZEN)), index=meta, analyze_sha256=sha(HERE / 'ag_analyze.py'))

    # 1. ensemble pairs
    pairs = {}
    for t in ('M3a_free', 'M3b_free', 'A3_free'):
        for r_ in ('M1a_free', 'M1b_free'):
            c = AA.contrast(H, ginfo, U, t, r_, 'fail', rng, boot, margin)
            pairs[f'{t}_vs_{r_}'] = slim(c)
    for t, r_ in (('M1a_free', 'M1b_free'), ('M3a_free', 'M3b_free')):
        pairs[f'seed_{t}_vs_{r_}'] = slim(AA.contrast(H, ginfo, U, t, r_, 'fail', rng, boot, margin))
    out['ensemble_pairs'] = pairs

    # 2. training-noise sensitivity
    d1 = pairs['seed_M1a_free_vs_M1b_free']['diff_pts']; d2 = pairs['seed_M3a_free_vs_M3b_free']['diff_pts']
    sb2 = (d1 ** 2 + d2 ** 2) / 2 / 2
    fam = {}
    for name, t, var_mult in (('P1_soil_M3_vs_M1', 'M3', 1.0), ('P2_soil_A3_vs_M1', 'A3_free', 1.5)):
        c = AA.contrast(H, ginfo, U, t, 'M1', 'fail', np.random.default_rng(int(spec['seed'])), boot, margin, detail=False)
        lo, hi = c['cluster']['ci90']
        sd_boot = (hi - lo) / (2 * 1.6449)
        for label, s2 in (('estimate', sb2), ('double', 2 * sb2), ('pessimistic_d1_as_sd', d1 ** 2)):
            sd = sqrt(sd_boot ** 2 + var_mult * s2)
            fam.setdefault(name, {})[label] = dict(diff_pts=c['diff_pts'], sd_cluster_boot=sd_boot, sigma_b2_per_ensemble=s2, sd_total=sd,
                                                   z=c['diff_pts'] / sd, p_one_sided=norm_sf(-c['diff_pts'] / sd),
                                                   ci90=[c['diff_pts'] - 1.6449 * sd, c['diff_pts'] + 1.6449 * sd])
    rig = {}
    if a.rigid and Path(a.rigid).exists():
        rr = json.load(open(a.rigid))
        for f in rr['family']:
            if f['name'].startswith('P3') or f['name'].startswith('P4'):
                rig[f['name']] = f['result']['cluster']['p_one_sided'] if f['result'].get('n') else 1.0
    holm_sens = {}
    for label in ('estimate', 'double', 'pessimistic_d1_as_sd'):
        names = ['P1_soil_M3_vs_M1', 'P2_soil_A3_vs_M1', 'P3_rigid_fx2_M3_vs_M1', 'P4_rigid_fx2_A3_vs_M1']
        ps = [fam['P1_soil_M3_vs_M1'][label]['p_one_sided'], fam['P2_soil_A3_vs_M1'][label]['p_one_sided'],
              rig.get('P3_rigid_fx2_M3_vs_M1', 1.0), rig.get('P4_rigid_fx2_A3_vs_M1', 1.0)]
        adj, rej = AA.holm(ps, alpha)
        holm_sens[label] = {n: dict(p=p, p_holm=q, reject=r) for n, p, q, r in zip(names, ps, adj, rej)}
    out['training_noise'] = dict(seed_diffs_pts=dict(M1a_vs_M1b=d1, M3a_vs_M3b=d2), sigma_b_per_ensemble_pts=sqrt(sb2), P=fam, holm=holm_sens,
                                 rigid_p_from=a.rigid if rig else None,
                                 note='rigid P3 / P4 enter with their cluster-bootstrap p unchanged (no rigid training-noise term here)')

    # 3. QA sensitivity
    ex = json.load(open(a.extras))
    flagged = set(ex['qa_flagged_runs'])
    qa = {}
    for name, t, members in (('P1_soil_M3_vs_M1', 'M3', ['M1a_free', 'M1b_free', 'M3a_free', 'M3b_free']),
                             ('P2_soil_A3_vs_M1', 'A3_free', ['M1a_free', 'M1b_free', 'A3_free'])):
        Gq = [g for g in U if not any(H[g][m]['run_id'] in flagged for m in members)]
        c = AA.contrast(H, ginfo, Gq, t, 'M1', 'fail', rng, boot, margin, detail=False)
        qa[name] = dict(groups_dropped=len(U) - len(Gq), **slim(c))
    out['qa_sensitivity'] = dict(flagged_runs=len(flagged), **qa)

    # 4. gap against the held-out groups of g203 / g228
    gaps = []
    for m in ('M1a_free', 'M3a_free', 'A3_free'):
        g = dict(world='crm', vehicle='hmmwv', model=m, straight='straight6', labels=['fail'], unseen='unseen', indist='heldout')
        gaps.append(dict(g, result=AA.gap(H, ginfo, g, rng, boot)))
    out['gaps_vs_heldout'] = gaps

    # 5. in-arena M3a per held-out arena
    out['inarena_M3a'] = {ar: slim(AA.contrast(H, ginfo, AA.set_groups(ginfo, f'heldout_{ar}', list(H)), 'M3a_free', 'M1a_free', 'fail', rng, boot, margin, detail=False))
                          for ar in ('g203', 'g228')}

    # 6. per-arena regressions
    me = json.load(open(MAPERR))['arenas']
    arenas = sorted({ginfo[g]['arena'] for g in U})
    per = {}
    for ar in arenas:
        Ga = [g for g in U if ginfo[g]['arena'] == ar]
        rate = lambda arm: 100 * float(np.mean([H[g][arm]['fail'] for g in Ga]))
        per[ar] = dict(role=ginfo[Ga[0]]['role'], dist_f104=me[ar]['distance_f104'], dist_nearest_training=me[ar]['distance_nearest_training'],
                       nearest_training_arena=me[ar]['nearest_training_arena'], closer_by=me[ar]['distance_f104'] - me[ar]['distance_nearest_training'],
                       map_err_rmse_m=me[ar]['all']['rmse'], M1=rate('M1'), M3=rate('M3'), A3=rate('A3_free'), M2=rate('M2_free'), straight6=rate('straight6'))
        per[ar].update(P1=per[ar]['M3'] - per[ar]['M1'], P2=per[ar]['A3'] - per[ar]['M1'], M1_minus_straight6=per[ar]['M1'] - per[ar]['straight6'],
                       M1_headroom_closed=(per[ar]['straight6'] - per[ar]['M1']) / per[ar]['straight6'] if per[ar]['straight6'] > 0 else None)
    reg = {}
    col = lambda k: [per[ar][k] for ar in arenas]
    for y in ('M1', 'M1_minus_straight6', 'M1_headroom_closed', 'P1', 'P2'):
        for x in ('dist_f104', 'closer_by', 'dist_nearest_training', 'map_err_rmse_m', 'straight6'):
            reg[f'{y}~{x}'] = AA.corr(col(x), col(y))
    out['per_arena'] = per
    out['per_arena_regressions'] = reg

    # 7. task B split
    tb = {}
    for s in ('indist_f104', 'f104_800'):
        Gs = AA.set_groups(ginfo, s, list(Gt))
        if s == 'f104_800':
            ind = set(AA.set_groups(ginfo, 'indist_f104', list(Gt)))
            Gs = [g for g in Gs if g not in ind]; s = 'f104_800_minus_indist'
        tb[s] = dict(G_vs_H=slim(AA.contrast(Gt, ginfo, Gs, 'G_free_gator', 'H_free_gator', 'fail', rng, boot, margin, detail=False)),
                     G_vs_straight6=slim(AA.contrast(Gt, ginfo, Gs, 'G_free_gator', 'straight6_gator', 'fail', rng, boot, margin, detail=False)))
    out['taskB_split'] = tb
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))

    # short report
    L = []
    for k, v in pairs.items():
        L.append(f"pair {k:28s} {v['rate_test']:.1f} vs {v['rate_ref']:.1f} diff {v['diff_pts']:+.1f} cluster90 {AA.fmt_ci(v['cluster']['ci90'])} p1 {v['cluster']['p_one_sided']:.4f}")
    L.append(f"training noise: seed diffs {d1:+.2f} / {d2:+.2f} pts -> sigma_b {sqrt(sb2):.2f} pts per ensemble")
    for n, dd in fam.items():
        for lab, v in dd.items():
            L.append(f"  {n} [{lab}] diff {v['diff_pts']:+.2f} sd boot {v['sd_cluster_boot']:.2f} total {v['sd_total']:.2f} p1 {v['p_one_sided']:.4f} "
                     f"90% {AA.fmt_ci(v['ci90'])} Holm {holm_sens[lab][n]['p_holm']:.4f} {'reject' if holm_sens[lab][n]['reject'] else 'no'}")
    for n, v in qa.items():
        L.append(f"QA-clean {n}: dropped {v['groups_dropped']} groups, diff {v['diff_pts']:+.2f} cluster90 {AA.fmt_ci(v['cluster']['ci90'])} p1 {v['cluster']['p_one_sided']:.4f}")
    for g in gaps:
        r = g['result']['fail']
        L.append(f"gap vs heldout {g['model']}: unseen {r['raw']['rate_unseen']:.1f} - heldout {r['raw']['rate_indist']:.1f} = {r['raw']['gap_pts']:+.1f} {AA.fmt_ci(r['raw']['ci95'])}; "
                 f"vs straight6 DiD {r['vs_straight']['did_pts']:+.1f} {AA.fmt_ci(r['vs_straight']['ci95'])}")
    for ar, v in out['inarena_M3a'].items():
        L.append(f"in-arena M3a vs M1a {ar}: {v['rate_test']:.1f} vs {v['rate_ref']:.1f} diff {v['diff_pts']:+.1f} cluster90 {AA.fmt_ci(v['cluster']['ci90'])} p1 {v['cluster']['p_one_sided']:.4f}")
    for k, v in reg.items():
        if v:
            L.append(f"reg {k:34s} slope {v['ols_slope']:+.2f} pearson {v['pearson']:+.2f} spearman {v['spearman']:+.2f}")
    for s, v in tb.items():
        L.append(f"task B {s}: G vs H {v['G_vs_H']['rate_test']:.1f} vs {v['G_vs_H']['rate_ref']:.1f} diff {v['G_vs_H']['diff_pts']:+.1f} n {v['G_vs_H']['n']}; "
                 f"G vs straight6 {v['G_vs_straight6']['diff_pts']:+.1f}")
    txt = '\n'.join(L)
    Path(str(a.out).rsplit('.', 1)[0] + '.txt').write_text(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
