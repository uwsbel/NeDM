#!/usr/bin/env python3
"""Paired closed-loop statistics of the evaluation drives (arena_gator_20260925, module E6a; PLAN 2.3, 3, 7.1-7.4).

Input: one or more outcome indexes from scripts/ag_eval_index.py (one row per world, vehicle, group, arm) and a spec
(json; --write-template writes the declared default). Every contrast names its world and vehicle; vehicle 'any' pools
both vehicles of a world (arm names are unique per world), e.g. task B's H on the HMMWV against H on the Gator. Outcomes per group, per arm: fail (goal not reached), unsafe,
unsafe_noback (= fail), backward_only, tilt30, elapsed time. Composite arms ('combine': e.g. M1 = mean of M1a and M1b,
0 / 0.5 / 1 per group, REVIEW_R1 4) are formed before any statistic.

Per contrast (TEST vs REF on one label and one group set; diff = rate(TEST) - rate(REF) in points, negative = TEST
fails less; 'improvement' = -diff):
  - paired group bootstrap (--boot resamples of the groups): 95 % and 90 % intervals, one-sided p = (1 + #{diff_b >= 0})
    / (B + 1) (H1: TEST better); exact McNemar on the discordant groups, two-sided (ga_analyze.mcnemar) and one-sided;
  - cluster bootstrap over (arena, nearest terrain feature) clusters (ag_eval_index 'cluster'; the n2_cluster_ci rule;
    spec 'cluster_key' = 'cluster_design' uses the case design feature instead): 95 % / 90 % intervals and the one-sided
    p the same way; cluster wins / losses with their sign test; the other clustering is reported as 'cluster_alt';
  - per arena: difference, 95 % group-bootstrap interval, discordant counts; arena sign count; a random-effects pooled
    estimate over arenas (DerSimonian-Laird, per-arena variance of the paired difference) with tau and I^2;
  - near vs spread: the contrast on each and the difference of the two effects (independent cluster bootstraps);
  - effect against distance to the nearest training arena and against the map-lookup error (per-arena OLS slope,
    Pearson and Spearman correlation; 8 points at most, descriptive);
  - identical picks (same route sha256) and the median time ratio on joint successes (ga_analyze.time_ratio).
Decision (declared family, PLAN 7.3): Holm at alpha over the family's one-sided CLUSTER p-values; 'improves' if Holm
rejects; otherwise 'no meaningful difference' if the 90 % cluster interval lies within +-margin points, else
'inconclusive' (fewer than spec 'min_groups' (50) paired groups: 'too few groups'). Contrasts outside the family get
the same wording with their unadjusted p ('secondary').
Generalisation gap (per model and label): rate(unseen) - rate(in distribution) with independent bootstraps of the two
group sets, and the difference in differences against the straight arm, [model - straight](unseen) - [model -
straight](in distribution), pooled and per arena. Dose response (e.g. M1 -> M2 -> M3): rates, consecutive paired
contrasts, and the bootstrap OLS slope of the rate over the number of arenas (1, 2, 3) on groups with every arm.
Every contrast also carries a non-inferiority line (one-sided upper 95 % bound of the diff < margin: 'not worse by more
than the margin'; PLAN 2.3 no-harm on f104, task B criterion 2). 'headroom' blocks: (straight - model) / straight on the
failure rate (task B criterion 4). Rates tables per world, vehicle, set and arm.

  PYTHONPATH=src:scripts python scripts/ag_analyze.py --index <index.json> [...] --spec <spec.json> --out <results.json>
  python scripts/ag_analyze.py --write-template <spec.json>          # the declared family (PLAN 7.3) with placeholder arm names
  python scripts/ag_analyze.py --synthetic-selftest <dir>            # planted effects: checks the statistics end to end
"""
import argparse, copy, hashlib, json, os, sys, time
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ga_analyze as GAN                   # noqa: E402  (mcnemar, time_ratio: reused unchanged)

LABELS = ('fail', 'unsafe', 'unsafe_noback', 'backward_only', 'tilt30')
NEAR = ('g260', 'g271', 'g251', 'g247')
SPREAD = ('g258', 'g268', 'g263', 'g241')

TEMPLATE = dict(
    schema='ag_analyze_spec_v1', margin_pts=2.0, alpha=0.05, boot=4000, seed=0,
    note=('PLAN 7.3 family: soil M3 vs M1 and A3 vs M1 on "fail", rigid fixed 2 m/s M3 vs M1 and A3 vs M1 on "unsafe", pooled over '
          'the 8 unseen test arenas; Holm over the four one-sided cluster p-values. Replace the arm names by the index arm names '
          '(ag_eval_tasks --arm NAME); with second ensembles use "combine" (REVIEW_R1 4).'),
    combine=[dict(world='crm', vehicle='hmmwv', name='M1', members=['M1a_free', 'M1b_free']),
             dict(world='crm', vehicle='hmmwv', name='M3', members=['M3a_free', 'M3b_free']),
             dict(world='rigid', vehicle='hmmwv', name='M1_fx2', members=['M1a_fx2', 'M1b_fx2']),
             dict(world='rigid', vehicle='hmmwv', name='M3_fx2', members=['M3a_fx2', 'M3b_fx2'])],
    family=[dict(name='P1_soil_M3_vs_M1', world='crm', vehicle='hmmwv', test='M3', ref='M1', label='fail', set='unseen'),
            dict(name='P2_soil_A3_vs_M1', world='crm', vehicle='hmmwv', test='A3_free', ref='M1', label='fail', set='unseen'),
            dict(name='P3_rigid_fx2_M3_vs_M1', world='rigid', vehicle='hmmwv', test='M3_fx2', ref='M1_fx2', label='unsafe', set='unseen'),
            dict(name='P4_rigid_fx2_A3_vs_M1', world='rigid', vehicle='hmmwv', test='A3_fx2', ref='M1_fx2', label='unsafe', set='unseen')],
    contrasts=[dict(name='soil_M2_vs_M1', world='crm', vehicle='hmmwv', test='M2_free', ref='M1', label='fail', set='unseen'),
               dict(name='seed_floor_soil_M1a_vs_M1b', world='crm', vehicle='hmmwv', test='M1a_free', ref='M1b_free', label='fail', set='unseen'),
               dict(name='rigid_free_noharm_M3_vs_M1', world='rigid', vehicle='hmmwv', test='M3a_free', ref='M1a_free', label='unsafe', set='unseen'),
               dict(name='noharm_f104_soil_M3_vs_M1', world='crm', vehicle='hmmwv', test='M3', ref='M1', label='fail', set='indist_f104'),
               dict(name='inarena_soil_M3_vs_M1', world='crm', vehicle='hmmwv', test='M3', ref='M1', label='fail', set='heldout'),
               # task B (PLAN 3; declared primary B = G vs H on the Gator, soil goal reached, 800 groups; not part of the Holm family)
               dict(name='PRIMARY_B_soil_G_vs_H_on_gator', world='crm', vehicle='gator', test='G_free', ref='H_free_gator', label='fail', set='f104_800', declared_primary=True),
               dict(name='B_works1_soil_G_vs_straight6_on_gator', world='crm', vehicle='gator', test='G_free', ref='straight6_gator', label='fail', set='f104_800'),
               dict(name='B_anchor_soil_H_gator_vs_H_hmmwv', world='crm', vehicle='any', test='H_free_gator', ref='H_free_hmmwv', label='fail', set='f104_800'),
               dict(name='B_rigid_G_vs_H_on_gator', world='rigid', vehicle='gator', test='G_free_r', ref='H_free_gator_r', label='unsafe', set='f104_800')],
    headroom=[dict(world='crm', vehicle='gator', model='G_free', straight='straight6_gator', label='fail', set='f104_800'),
              dict(world='crm', vehicle='hmmwv', model='H_free_hmmwv', straight='straight6_hmmwv', label='fail', set='f104_800')],
    gaps=[dict(world='crm', vehicle='hmmwv', model=m, straight='straight6', labels=['fail'], unseen='unseen', indist='indist_f104')
          for m in ('M1', 'M3', 'A3_free')] +
         [dict(world='rigid', vehicle='hmmwv', model=m, straight='straight2', labels=['unsafe', 'fail'], unseen='unseen', indist='indist_f104')
          for m in ('M1_fx2', 'M3_fx2', 'A3_fx2')],
    dose=[dict(world='crm', vehicle='hmmwv', arms=['M1', 'M2_free', 'M3'], label='fail', set='unseen'),
          dict(world='rigid', vehicle='hmmwv', arms=['M1_fx2', 'M2_fx2', 'M3_fx2'], label='unsafe', set='unseen')],
    time_ratio=[dict(world='crm', vehicle='hmmwv', test='M3a_free', ref='M1a_free', set='unseen')],
    rates=dict(sets=['unseen', 'near', 'spread', 'indist_f104', 'heldout', 'dev', 'f104_800', 'per_arena']))


# ------------------------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------------------------
def load_index(paths):
    rows, meta = [], []
    for p in paths:
        d = json.load(open(p))
        rows += d['rows']; meta.append(dict(path=str(p), sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(), summary=d.get('summary')))
    return rows, meta


def build_by(rows, combine):
    """by[(world, vehicle)][group][arm] = outcome dict; ginfo[group] = arena, role, set, cluster, covariates."""
    by, ginfo = defaultdict(lambda: defaultdict(dict)), {}
    for r in rows:
        if r.get('missing'):
            continue
        for key in ((r['world'], r['vehicle']), (r['world'], 'any')):      # 'any': both vehicles (cross-vehicle contrasts)
            assert r['arm'] not in by[key][r['group']], f"duplicate outcome {key} {r['group']} {r['arm']} (arm names must be unique per world)"
            by[key][r['group']][r['arm']] = {k: r.get(k) for k in LABELS + ('elapsed', 'status', 'route_sha256', 'run_id', 'P', 'work_kj')}
        gi = dict(arena=r['arena'], role=r['role'], set=r['set'], cluster=r['cluster'], cluster_design=r.get('cluster_design', r['cluster']), stratum=r.get('stratum'),
                  dist_nearest_training=r.get('dist_nearest_training'), map_err_rmse_m=r.get('map_err_rmse_m'))
        if r['group'] in ginfo:
            assert ginfo[r['group']]['cluster'] == gi['cluster'], r['group']
        ginfo[r['group']] = gi
    for c in combine or []:
        key = (c['world'], c['vehicle'])
        for g, arms in by.get(key, {}).items():
            if all(m in arms for m in c['members']):
                v = {k: float(np.mean([arms[m][k] for m in c['members']])) for k in LABELS}
                v.update(elapsed=None, status='composite', route_sha256=None, run_id=None, P=None, work_kj=None, members=list(c['members']))
                arms[c['name']] = v
    return by, ginfo


def set_groups(ginfo, name, groups):
    """Groups of a named set: unseen | near | spread | indist_f104 | heldout | heldout_<a> | dev | f104_800 | arena:<a> | all."""
    def ok(g):
        i = ginfo[g]
        if name == 'all':
            return True
        if name == 'unseen':
            return i['set'] == 'unseen'
        if name in ('near', 'spread'):
            return i['set'] == 'unseen' and i['role'] == name
        if name in ('indist_f104', 'heldout', 'dev'):
            return i['set'] == name
        if name.startswith('heldout_'):
            return i['set'] == 'heldout' and i['arena'] == name[len('heldout_'):]
        if name == 'f104_800':
            return i['arena'] == 'f104' and (g.startswith('f104_pair_group_') or g.startswith('f104_crm_eval_group_'))
        if name.startswith('arena:'):
            return i['arena'] == name[len('arena:'):]
        raise ValueError(f'unknown set {name!r}')
    return sorted(g for g in groups if ok(g))


# ------------------------------------------------------------------------------------------------------------------
# statistics
# ------------------------------------------------------------------------------------------------------------------
def mcnemar_one_sided(better, worse):
    """P(X >= better) for X ~ Bin(better + worse, 1/2): exact one-sided McNemar (H1: the test arm fails less)."""
    n = better + worse
    return 1.0 if n == 0 else float(sum(comb(n, i) for i in range(better, n + 1)) / 2 ** n)


def holm(pvals, alpha):
    """Holm step-down: adjusted p-values (same order as given) and reject flags."""
    m = len(pvals); order = np.argsort(pvals, kind='stable'); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i])); adj[i] = run
    return adj.tolist(), [bool(x <= alpha) for x in adj]


def boot_p(d):
    """One-sided bootstrap p for H1 'diff < 0' (test fails less): (1 + #{d_b >= 0}) / (B + 1)."""
    return float((1 + np.sum(d >= 0)) / (len(d) + 1))


def pct(d, lo, hi):
    return [float(np.percentile(d, lo)), float(np.percentile(d, hi))]


def paired_arrays(byw, G, x, y, label):
    return np.array([byw[g][x][label] for g in G], float), np.array([byw[g][y][label] for g in G], float)


def cluster_boot(xt, xr, cl, rng, boot):
    ids, inv = np.unique(cl, return_inverse=True); K = len(ids)
    st = np.bincount(inv, xt, K); sr = np.bincount(inv, xr, K); n = np.bincount(inv, None, K).astype(float)
    pick = rng.integers(0, K, (boot, K))
    d = 100 * (st[pick].sum(1) - sr[pick].sum(1)) / n[pick].sum(1)
    mt, mr = st / n, sr / n
    wins, losses = int((mt < mr).sum()), int((mt > mr).sum())
    return d, dict(clusters=int(K), cluster_wins=wins, cluster_losses=losses, sign_p=GAN.mcnemar(wins, losses))


def random_effects(per):
    """DerSimonian-Laird over arenas: per = [(diff_pts, var_pts2, n)]; zero variances get the floor of one discordant group."""
    if len(per) < 2:
        return None
    d = np.array([p[0] for p in per], float); n = np.array([p[2] for p in per], float)
    v = np.array([max(p[1], 1e4 / p[2] ** 2) for p in per], float)
    w = 1 / v; mu_f = float((w * d).sum() / w.sum()); Q = float((w * (d - mu_f) ** 2).sum()); k = len(d)
    C = float(w.sum() - (w ** 2).sum() / w.sum()); tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    ws = 1 / (v + tau2); mu = float((ws * d).sum() / ws.sum()); se = float(np.sqrt(1 / ws.sum()))
    return dict(k=k, pooled_pts=mu, ci95=[mu - 1.96 * se, mu + 1.96 * se], se=se, tau_pts=float(np.sqrt(tau2)), Q=Q,
                I2=float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0, fixed_effect_pts=mu_f, spread_sd_pts=float(d.std(ddof=1)))


def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    rank = lambda a: np.argsort(np.argsort(a, kind='stable'), kind='stable').astype(float)
    slope = float(np.polyfit(x, y, 1)[0])
    return dict(n=int(len(x)), ols_slope=slope, pearson=float(np.corrcoef(x, y)[0, 1]), spearman=float(np.corrcoef(rank(x), rank(y))[0, 1]))


def contrast(byw, ginfo, G, x, y, label, rng, boot, margin, detail=True):
    G = [g for g in G if x in byw[g] and y in byw[g]]
    out = dict(test=x, ref=y, label=label, n=len(G))
    if not G:
        return out
    xt, xr = paired_arrays(byw, G, x, y, label)
    diff = 100 * (xt.mean() - xr.mean())
    idx = rng.integers(0, len(G), (boot, len(G)))
    db = 100 * (xt[idx].mean(1) - xr[idx].mean(1))
    binary = bool(np.isin(xt, (0, 1)).all() and np.isin(xr, (0, 1)).all())
    worse, better = int(((xt > xr)).sum()), int(((xt < xr)).sum())
    cl = np.array([ginfo[g][CLUSTER_KEY[0]] for g in G])
    dc, cinfo = cluster_boot(xt, xr, cl, rng, boot)
    alt_key = 'cluster_design' if CLUSTER_KEY[0] == 'cluster' else 'cluster'
    da, ainfo = cluster_boot(xt, xr, np.array([ginfo[g][alt_key] for g in G]), rng, boot)
    out.update(rate_test=100 * xt.mean(), rate_ref=100 * xr.mean(), diff_pts=diff, improvement_pts=-diff,
               group=dict(ci95=pct(db, 2.5, 97.5), ci90=pct(db, 5, 95), p_one_sided=boot_p(db)),
               cluster=dict(key=CLUSTER_KEY[0], ci95=pct(dc, 2.5, 97.5), ci90=pct(dc, 5, 95), p_one_sided=boot_p(dc), **cinfo),
               cluster_alt=dict(key=alt_key, ci95=pct(da, 2.5, 97.5), ci90=pct(da, 5, 95), p_one_sided=boot_p(da), **ainfo),
               discordant=dict(test_worse=worse, test_better=better, binary=binary,
                               mcnemar_two_sided=GAN.mcnemar(worse, better) if binary else None,
                               mcnemar_one_sided=mcnemar_one_sided(better, worse) if binary else None),
               identical_picks=int(sum(1 for g in G if byw[g][x].get('route_sha256') and byw[g][x]['route_sha256'] == byw[g][y].get('route_sha256'))),
               within_margin_90=bool(-margin <= pct(dc, 5, 95)[0] and pct(dc, 5, 95)[1] <= margin), margin_pts=margin,
               noninferiority=dict(upper95_one_sided_cluster=float(np.percentile(dc, 95)), upper95_one_sided_group=float(np.percentile(db, 95)),
                                   margin_pts=margin, pass_cluster=bool(np.percentile(dc, 95) < margin),
                                   note=f'{x} is within {margin} points of {y} (not worse by more) if the one-sided upper 95 % bound of the diff < margin'),
               note=f'diff = {label}({x}) - {label}({y}) in points; negative = {x} better')
    if not detail:
        return out
    arenas = sorted({ginfo[g]['arena'] for g in G})
    per, per_re = {}, []
    for ar in arenas:
        Ga = [g for g in G if ginfo[g]['arena'] == ar]
        a_t, a_r = paired_arrays(byw, Ga, x, y, label)
        ia = rng.integers(0, len(Ga), (boot, len(Ga))); da = 100 * (a_t[ia].mean(1) - a_r[ia].mean(1))
        dd = 100 * (a_t - a_r)
        per[ar] = dict(n=len(Ga), rate_test=100 * a_t.mean(), rate_ref=100 * a_r.mean(), diff_pts=float(dd.mean()), ci95=pct(da, 2.5, 97.5),
                       test_worse=int((a_t > a_r).sum()), test_better=int((a_t < a_r).sum()), role=ginfo[Ga[0]]['role'],
                       dist_nearest_training=ginfo[Ga[0]]['dist_nearest_training'], map_err_rmse_m=ginfo[Ga[0]]['map_err_rmse_m'])
        per_re.append((float(dd.mean()), float(dd.var(ddof=1) / len(Ga)) if len(Ga) > 1 else 1e4, len(Ga)))
    out['per_arena'] = per
    out['arena_signs'] = dict(better=sum(1 for v in per.values() if v['diff_pts'] < 0), worse=sum(1 for v in per.values() if v['diff_pts'] > 0),
                              equal=sum(1 for v in per.values() if v['diff_pts'] == 0))
    out['random_effects'] = random_effects(per_re)
    ds = [v['diff_pts'] for v in per.values()]
    out['vs_distance_nearest_training'] = corr([v['dist_nearest_training'] for v in per.values()], ds) if all(v['dist_nearest_training'] is not None for v in per.values()) else None
    out['vs_map_error'] = corr([v['map_err_rmse_m'] for v in per.values()], ds) if all(v['map_err_rmse_m'] is not None for v in per.values()) else None
    roles = {}
    for role in ('near', 'spread'):
        Gr = [g for g in G if ginfo[g]['role'] == role and ginfo[g]['set'] == 'unseen']
        if Gr:
            r_t, r_r = paired_arrays(byw, Gr, x, y, label)
            drb, ci = cluster_boot(r_t, r_r, np.array([ginfo[g][CLUSTER_KEY[0]] for g in Gr]), rng, boot)
            roles[role] = dict(n=len(Gr), diff_pts=100 * (r_t.mean() - r_r.mean()), cluster_ci95=pct(drb, 2.5, 97.5), boot=drb)
    if len(roles) == 2:
        dd = roles['near']['boot'] - roles['spread']['boot']
        out['near_minus_spread'] = dict(diff_pts=roles['near']['diff_pts'] - roles['spread']['diff_pts'], ci95=pct(dd, 2.5, 97.5))
    out['near_spread'] = {k: {kk: vv for kk, vv in v.items() if kk != 'boot'} for k, v in roles.items()}
    if label == 'fail' and binary:
        byt = {g: {x: byw[g][x], y: byw[g][y]} for g in G}
        if all(byw[g][x].get('elapsed') is not None and byw[g][y].get('elapsed') is not None for g in G):
            out['time_ratio'] = GAN.time_ratio(byt, G, x, y, rng, boot, 1.10)
    return out


MIN_GROUPS = 50
CLUSTER_KEY = ['cluster']          # spec 'cluster_key': 'cluster' (nearest feature to the start-goal midpoint) | 'cluster_design'


def decide(c, p_used, reject, margin, min_groups=MIN_GROUPS):
    if not c.get('n'):
        return 'no data'
    if c['n'] < min_groups:
        return f'too few groups (n {c["n"]} < {min_groups})'
    if reject and c['diff_pts'] < 0:
        return 'improves'
    if c['within_margin_90']:
        return 'no meaningful difference'
    return 'inconclusive'


def set_boot(byw, G, arm, label, rng, boot):
    v = np.array([byw[g][arm][label] for g in G], float)
    idx = rng.integers(0, len(G), (boot, len(G)))
    return 100 * v.mean(), 100 * v[idx].mean(1)


def gap(byw, ginfo, spec, rng, boot):
    out = {}
    groups = list(byw)
    for label in spec['labels']:
        m, s = spec['model'], spec['straight']
        Gu = [g for g in set_groups(ginfo, spec['unseen'], groups) if m in byw[g]]
        Gi = [g for g in set_groups(ginfo, spec['indist'], groups) if m in byw[g]]
        res = dict(model=m, straight=s, label=label, n_unseen=len(Gu), n_indist=len(Gi))
        if Gu and Gi:
            ru, bu = set_boot(byw, Gu, m, label, rng, boot); ri, bi = set_boot(byw, Gi, m, label, rng, boot)
            res['raw'] = dict(rate_unseen=ru, rate_indist=ri, gap_pts=ru - ri, ci95=pct(bu - bi, 2.5, 97.5))
        Gu2 = [g for g in Gu if s in byw[g]]; Gi2 = [g for g in Gi if s in byw[g]]
        if Gu2 and Gi2:
            def dd(G):
                a, b = paired_arrays(byw, G, m, s, label)
                idx = rng.integers(0, len(G), (boot, len(G)))
                return 100 * (a.mean() - b.mean()), 100 * (a[idx].mean(1) - b[idx].mean(1))
            du, dbu = dd(Gu2); di, dbi = dd(Gi2)
            res['vs_straight'] = dict(model_minus_straight_unseen=du, model_minus_straight_indist=di, did_pts=du - di,
                                      ci95=pct(dbu - dbi, 2.5, 97.5), note='[model - straight](unseen) - [model - straight](in distribution)')
            per = {}
            for ar in sorted({ginfo[g]['arena'] for g in Gu2}):
                Ga = [g for g in Gu2 if ginfo[g]['arena'] == ar]
                da, dba = dd(Ga)
                per[ar] = dict(n=len(Ga), model_minus_straight=da, ci95=pct(dba, 2.5, 97.5),
                               headroom_closed=(None if np.mean([byw[g][s][label] for g in Ga]) == 0 else
                                                float(-da / (100 * np.mean([byw[g][s][label] for g in Ga])))))
            res['per_arena_vs_straight'] = per
        out[label] = res
    return out


def dose(byw, ginfo, spec, rng, boot, margin):
    arms = spec['arms']
    G = [g for g in set_groups(ginfo, spec['set'], list(byw)) if all(a in byw[g] for a in arms)]
    out = dict(arms=arms, label=spec['label'], set=spec['set'], n=len(G))
    if not G:
        return out
    V = np.stack([[byw[g][a][spec['label']] for g in G] for a in arms])            # (k, n)
    k = np.arange(1, len(arms) + 1, dtype=float); kc = k - k.mean()
    rates = 100 * V.mean(1)
    idx = rng.integers(0, len(G), (boot, len(G)))
    rb = 100 * V[:, idx].mean(2)                                                    # (k, boot)
    slope_b = (kc[:, None] * rb).sum(0) / (kc ** 2).sum()
    out.update(rates=dict(zip(arms, rates.tolist())), slope_pts_per_arena=float((kc * rates).sum() / (kc ** 2).sum()),
               slope_ci95=pct(slope_b, 2.5, 97.5), slope_p_one_sided=boot_p(slope_b),
               monotone_decreasing=bool(np.all(np.diff(rates) <= 0)),
               steps={f'{arms[i + 1]}:{arms[i]}': contrast(byw, ginfo, G, arms[i + 1], arms[i], spec['label'], rng, boot, margin, detail=False)
                      for i in range(len(arms) - 1)})
    return out


def rates_table(byw, G, arms):
    out = {}
    for a in arms:
        v = [byw[g][a] for g in G if a in byw[g]]
        if not v:
            continue
        t = [r['elapsed'] for r in v if r.get('elapsed') is not None and r['fail'] == 0]
        out[a] = dict(n=len(v), goal_reached=100 * (1 - np.mean([r['fail'] for r in v])), **{k: 100 * float(np.mean([r[k] for r in v])) for k in LABELS},
                      median_time_s=float(np.median(t)) if t else None,
                      status_counts=dict(Counter(r['status'] for r in v)))
    return out


def analyze(rows, spec, index_meta=None):
    boot, margin, alpha = int(spec.get('boot', 4000)), float(spec.get('margin_pts', 2.0)), float(spec.get('alpha', 0.05))
    rng = np.random.default_rng(int(spec.get('seed', 0)))
    CLUSTER_KEY[0] = spec.get('cluster_key', 'cluster'); assert CLUSTER_KEY[0] in ('cluster', 'cluster_design')
    by, ginfo = build_by(rows, spec.get('combine'))
    res = dict(schema='ag_analyze_results_v1', tool='scripts/ag_analyze.py', tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               created=time.strftime('%Y-%m-%d %H:%M:%S'), spec=spec, index=index_meta,
               worlds={f'{w}|{v}': dict(groups=len(byw), arms=sorted({a for g in byw.values() for a in g})) for (w, v), byw in by.items()})

    def run(c):
        byw = by.get((c['world'], c['vehicle']), {})
        G = set_groups(ginfo, c['set'], list(byw))
        return contrast(byw, ginfo, G, c['test'], c['ref'], c['label'], rng, boot, margin)

    fam = [dict(c, result=run(c)) for c in spec.get('family', [])]
    pv = [f['result']['cluster']['p_one_sided'] if f['result'].get('n') else 1.0 for f in fam]
    adj, rej = holm(pv, alpha) if fam else ([], [])
    for f, p, a_, r in zip(fam, pv, adj, rej):
        f['holm'] = dict(p_one_sided_cluster=p, p_adjusted=a_, reject=r, alpha=alpha, family_size=len(fam))
        f['decision'] = decide(f['result'], a_, r, margin, int(spec.get('min_groups', MIN_GROUPS)))
    res['family'] = fam
    sec = []
    for c in spec.get('contrasts', []):
        r = run(c)
        p = r['cluster']['p_one_sided'] if r.get('n') else 1.0
        tag = ' (declared primary, outside the Holm family, unadjusted)' if c.get('declared_primary') else ' (secondary, unadjusted)'
        sec.append(dict(c, result=r, decision=decide(r, p, p <= alpha, margin, int(spec.get('min_groups', MIN_GROUPS))) + tag))
    res['contrasts'] = sec
    res['gaps'] = [dict(g, result=gap(by.get((g['world'], g['vehicle']), {}), ginfo, g, rng, boot)) for g in spec.get('gaps', [])]
    res['dose'] = [dict(d, result=dose(by.get((d['world'], d['vehicle']), {}), ginfo, d, rng, boot, margin)) for d in spec.get('dose', [])]
    tr = []
    for t in spec.get('time_ratio', []):
        byw = by.get((t['world'], t['vehicle']), {})
        G = [g for g in set_groups(ginfo, t['set'], list(byw)) if t['test'] in byw[g] and t['ref'] in byw[g]]
        tr.append(dict(t, result=GAN.time_ratio(byw, G, t['test'], t['ref'], rng, boot, 1.10) if G else dict(n=0)))
    res['time_ratio'] = tr
    hr = []
    for h in spec.get('headroom', []):
        byw = by.get((h['world'], h['vehicle']), {})
        G = [g for g in set_groups(ginfo, h['set'], list(byw)) if h['model'] in byw[g] and h['straight'] in byw[g]]
        if not G:
            hr.append(dict(h, result=dict(n=0))); continue
        m_, s_ = paired_arrays(byw, G, h['model'], h['straight'], h['label'])
        idx = rng.integers(0, len(G), (boot, len(G)))
        fs = s_[idx].mean(1); closed_b = np.where(fs > 0, (fs - m_[idx].mean(1)) / np.maximum(fs, 1e-12), np.nan)
        hr.append(dict(h, result=dict(n=len(G), rate_model=100 * m_.mean(), rate_straight=100 * s_.mean(),
                                      headroom_closed=float((s_.mean() - m_.mean()) / s_.mean()) if s_.mean() > 0 else None,
                                      ci95=pct(closed_b[np.isfinite(closed_b)], 2.5, 97.5) if np.isfinite(closed_b).any() else None,
                                      note='(straight - model) / straight on the failure rate = the share of the straight route\'s failures the model removes; '
                                           'normalising by feasibility instead of 1 (REVIEW_R1 3d) needs the collection\'s any-route-reaches-the-goal share')))
    res['headroom'] = hr
    rt = {}
    sets = spec.get('rates', {}).get('sets', ['all'])
    for (w, v), byw in by.items():
        if v == 'any':
            continue
        arms = sorted({a for g in byw.values() for a in g})
        d = {}
        for s in sets:
            if s == 'per_arena':
                for ar in sorted({ginfo[g]['arena'] for g in byw}):
                    d[f'arena:{ar}'] = rates_table(byw, set_groups(ginfo, f'arena:{ar}', list(byw)), arms)
            else:
                G = set_groups(ginfo, s, list(byw))
                if G:
                    d[s] = rates_table(byw, G, arms)
        rt[f'{w}|{v}'] = d
    res['rates'] = rt
    return res


def fmt_ci(c):
    return f'[{c[0]:+.1f}, {c[1]:+.1f}]'


def report(res):
    L = []
    for f in res['family']:
        r = f['result']
        if not r.get('n'):
            L.append(f"FAMILY {f['name']}: no data"); continue
        L.append(f"FAMILY {f['name']} ({f['world']}/{f['vehicle']}, {f['label']}, {f['set']}): {f['test']} {r['rate_test']:.1f} % vs {f['ref']} {r['rate_ref']:.1f} %, "
                 f"diff {r['diff_pts']:+.1f} pts, cluster 90 % {fmt_ci(r['cluster']['ci90'])} ({r['cluster']['clusters']} clusters), p1 {r['cluster']['p_one_sided']:.4f}, "
                 f"Holm adj {f['holm']['p_adjusted']:.4f} -> {f['decision']}; group 95 % {fmt_ci(r['group']['ci95'])}; McNemar worse/better "
                 f"{r['discordant']['test_worse']}/{r['discordant']['test_better']}; n {r['n']}")
    for c in res['contrasts']:
        r = c['result']
        if not r.get('n'):
            L.append(f"contrast {c['name']}: no data"); continue
        L.append(f"contrast {c['name']} ({c['world']}/{c['vehicle']}, {c['label']}, {c['set']}): {c['test']} {r['rate_test']:.1f} % vs {c['ref']} {r['rate_ref']:.1f} %, "
                 f"diff {r['diff_pts']:+.1f}, cluster 90 % {fmt_ci(r['cluster']['ci90'])}, p1 {r['cluster']['p_one_sided']:.4f} -> {c['decision']}; n {r['n']}")
    for g in res['gaps']:
        for lab, r in g['result'].items():
            if 'raw' in r:
                L.append(f"gap {g['world']} {r['model']} {lab}: unseen {r['raw']['rate_unseen']:.1f} % - indist {r['raw']['rate_indist']:.1f} % = {r['raw']['gap_pts']:+.1f} {fmt_ci(r['raw']['ci95'])}"
                         + (f"; vs {r['straight']}: DiD {r['vs_straight']['did_pts']:+.1f} {fmt_ci(r['vs_straight']['ci95'])}" if 'vs_straight' in r else ''))
            else:
                L.append(f"gap {g['world']} {r['model']} {lab}: n unseen {r['n_unseen']}, indist {r['n_indist']} (not computed)")
    for d in res['dose']:
        r = d['result']
        if r.get('n'):
            L.append(f"dose {d['world']} {' -> '.join(r['arms'])} ({r['label']}): rates {', '.join(f'{v:.1f}' for v in r['rates'].values())}; slope {r['slope_pts_per_arena']:+.2f} pts/arena {fmt_ci(r['slope_ci95'])}; n {r['n']}")
        else:
            L.append(f"dose {d['world']} {' -> '.join(d['arms'])}: no groups with every arm")
    for h in res.get('headroom', []):
        r = h['result']
        if r.get('n') and r.get('headroom_closed') is not None:
            L.append(f"headroom {h['world']}/{h['vehicle']} {h['model']} vs {h['straight']} ({h['label']}, {h['set']}): {r['rate_model']:.1f} % vs {r['rate_straight']:.1f} %, "
                     f"closed {100 * r['headroom_closed']:.0f} %" + (f" [{100 * r['ci95'][0]:.0f}, {100 * r['ci95'][1]:.0f}]" if r.get('ci95') else '') + f"; n {r['n']}")
        else:
            L.append(f"headroom {h['world']}/{h['vehicle']} {h['model']} vs {h['straight']}: n {r.get('n', 0)} (not computed)")
    for key, sets in res['rates'].items():
        for s, arms in sets.items():
            for a, v in arms.items():
                L.append(f"rates {key} {s:14s} {a:14s} n {v['n']:4d} goal {v['goal_reached']:5.1f} % unsafe {v['unsafe']:5.1f} % (no-back {v['unsafe_noback']:5.1f}) tilt30 {v['tilt30']:4.1f} % "
                         f"median time {v['median_time_s'] if v['median_time_s'] is not None else float('nan'):.1f} s")
    return '\n'.join(L)


# ------------------------------------------------------------------------------------------------------------------
# synthetic self-test: planted effects with known answers
# ------------------------------------------------------------------------------------------------------------------
def synthetic_rows(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    arenas = [(a, 'near') for a in NEAR] + [(a, 'spread') for a in SPREAD]

    def add(world, arm, g, info, fail, unsafe=None):
        unsafe = fail if unsafe is None else unsafe
        rows.append(dict(world=world, vehicle='hmmwv', arena=info['arena'], role=info['role'], set=info['set'], group=g, arm=arm, cluster=info['cluster'],
                         stratum='hill_cross_slope', dist_nearest_training=info['dist'], map_err_rmse_m=info['err'], missing=False,
                         fail=int(fail), unsafe=int(unsafe), unsafe_noback=int(fail), backward_only=int(unsafe and not fail), tilt30=0,
                         elapsed=20.0 + 3 * rng.random() + (0 if not fail else 20), status='goal_reached' if not fail else 'blockage', route_sha256=f'{g}:{arm}:{fail}'))
    groups = []
    for ai, (a, role) in enumerate(arenas):
        for k in range(125):
            groups.append((f'{a}_test_group_{k:04d}', dict(arena=a, role=role, set='unseen', cluster=f'{a}:{rng.integers(0, 10)}', dist=0.6 + 0.1 * ai, err=0.06)))
    for k in range(200):
        groups.append((f'f104_pair_group_{k:04d}', dict(arena='f104', role='training', set='indist_f104', cluster=f'f104:{rng.integers(0, 9)}', dist=0.81, err=0.05)))
    cp = {}
    for g, info in groups:
        c = info['cluster']
        cp.setdefault(c, rng.beta(1.2, 8.0))
        p = cp[c] * (0.5 if info['set'] == 'indist_f104' else 1.0)
        m1a = rng.random() < p; m1b = m1a          # second ensemble identical here, so the composite M1 equals M1a exactly
        # soil: M3 fixes 45 % of M1a's failures (a planted improvement); A3 = M1a exactly (a planted null); M2 half way
        m3 = m1a and rng.random() > 0.45; m2 = m1a and rng.random() > 0.2
        s6 = m1a or rng.random() < 0.15
        for arm, f in (('M1a_free', m1a), ('M1b_free', m1b), ('M3a_free', m3), ('M3b_free', m3), ('A3_free', m1a), ('M2_free', m2), ('straight6', s6)):
            add('crm', arm, g, info, f)
        # rigid fixed 2 m/s: M3 = M1 exactly (null), A3 much better
        r1 = rng.random() < p * 1.5
        for arm, f in (('M1a_fx2', r1), ('M1b_fx2', r1), ('M3a_fx2', r1), ('M3b_fx2', r1), ('A3_fx2', r1 and rng.random() > 0.6),
                       ('M2_fx2', r1), ('straight2', r1 or rng.random() < 0.1)):
            add('rigid', arm, g, info, f)
    return rows


def synthetic_selftest(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    checks = {}
    adj, rej = holm([0.01, 0.04, 0.03, 0.2], 0.05)
    checks['holm_known'] = bool(np.allclose(adj, [0.04, 0.09, 0.09, 0.2]) and rej == [True, False, False, False])
    checks['mcnemar_known'] = bool(abs(GAN.mcnemar(0, 10) - 2 / 1024) < 1e-15 and abs(mcnemar_one_sided(10, 0) - 1 / 1024) < 1e-15
                                   and mcnemar_one_sided(0, 10) == 1.0 and abs(mcnemar_one_sided(3, 3) - 42 / 64) < 1e-15)
    rows = synthetic_rows()
    spec = copy.deepcopy(TEMPLATE); spec['boot'] = 2000
    res = analyze(rows, spec, dict(synthetic=True))
    fam = {f['name']: f for f in res['family']}
    checks['P1_improves'] = fam['P1_soil_M3_vs_M1']['decision'] == 'improves'
    checks['P2_null_is_no_meaningful_difference'] = bool(fam['P2_soil_A3_vs_M1']['decision'] == 'no meaningful difference' and
                                                         fam['P2_soil_A3_vs_M1']['result']['diff_pts'] == 0.0 and fam['P2_soil_A3_vs_M1']['holm']['p_one_sided_cluster'] == 1.0)
    checks['P3_null_is_no_meaningful_difference'] = fam['P3_rigid_fx2_M3_vs_M1']['decision'] == 'no meaningful difference'
    checks['P4_improves'] = fam['P4_rigid_fx2_A3_vs_M1']['decision'] == 'improves'
    # the composite M1 is the per-group mean of M1a and M1b
    by, gi = build_by(rows, spec['combine'])
    b = by[('crm', 'hmmwv')]
    checks['composite_mean'] = bool(all(b[g]['M1']['fail'] == (b[g]['M1a_free']['fail'] + b[g]['M1b_free']['fail']) / 2 for g in b))
    # cluster bootstrap on identical arms: every replicate 0 -> p 1; group count per arena 125
    r = fam['P1_soil_M3_vs_M1']['result']
    checks['per_arena_n'] = bool(all(v['n'] == 125 for v in r['per_arena'].values()) and len(r['per_arena']) == 8)
    checks['near_spread_present'] = bool(set(r['near_spread']) == {'near', 'spread'} and 'near_minus_spread' in r)
    # the planted gap: M1 fails ~2x as often on unseen as in distribution
    g0 = [g for g in res['gaps'] if g['model'] == 'M1'][0]['result']['fail']
    checks['gap_positive'] = bool(g0['raw']['gap_pts'] > 0 and g0['raw']['ci95'][0] > 0)
    d0 = res['dose'][0]['result']
    checks['dose_monotone'] = bool(d0['monotone_decreasing'] and d0['slope_ci95'][1] < 0)
    # coverage of the cluster interval for a planted null with noise: 40 re-draws of a random-flip null, 90 % interval
    cover = []
    for s in range(40):
        rr = synthetic_rows(seed=100 + s)
        rng = np.random.default_rng(s)
        twins = []
        for x in rr:
            if x['world'] == 'crm' and x['arm'] == 'M1a_free':
                y = dict(x); y['arm'] = 'NULL_twin'
                if rng.random() < 0.1:
                    y['fail'] = 1 - x['fail']; y['unsafe'] = y['fail']; y['unsafe_noback'] = y['fail']
                twins.append(y)
        rr += twins
        byx, gix = build_by(rr, [])
        bw = byx[('crm', 'hmmwv')]
        G = set_groups(gix, 'unseen', list(bw))
        c = contrast(bw, gix, G, 'NULL_twin', 'M1a_free', 'fail', np.random.default_rng(s), 1000, 2.0, detail=False)
        # the true difference of a symmetric flip is E[1 - 2 fail] * 0.1 * 100 points; coverage of that value
        truth = 100 * 0.1 * (1 - 2 * np.mean([bw[g]['M1a_free']['fail'] for g in G]))
        cover.append(c['cluster']['ci90'][0] <= truth <= c['cluster']['ci90'][1])
    checks['cluster_ci90_coverage'] = float(np.mean(cover))
    checks['cluster_ci90_coverage_ok'] = bool(0.75 <= float(np.mean(cover)) <= 1.0)
    ok = all(v for k, v in checks.items() if isinstance(v, bool)) and all(isinstance(checks[k], bool) for k in checks if k != 'cluster_ci90_coverage')
    json.dump(dict(checks=checks, passed=ok, family=[dict(name=f['name'], decision=f['decision'], diff=f['result']['diff_pts'],
                                                         ci90=f['result']['cluster']['ci90'], p=f['holm']['p_one_sided_cluster'],
                                                         p_adj=f['holm']['p_adjusted']) for f in res['family']]),
              open(out / 'synthetic_selftest.json', 'w'), indent=1, default=float)
    (out / 'synthetic_report.txt').write_text(report(res) + '\n')
    print(json.dumps(checks, indent=1, default=float))
    print('SYNTHETIC SELFTEST', 'PASSED' if ok else 'FAILED')
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index', nargs='*', default=[])
    ap.add_argument('--spec')
    ap.add_argument('--out')
    ap.add_argument('--write-template')
    ap.add_argument('--synthetic-selftest')
    a = ap.parse_args(argv)
    if a.write_template:
        json.dump(TEMPLATE, open(a.write_template, 'w'), indent=1); print(f'wrote {a.write_template}'); return
    if a.synthetic_selftest:
        ok = synthetic_selftest(a.synthetic_selftest); sys.exit(0 if ok else 1)
    assert a.index and a.spec and a.out, '--index, --spec and --out are required'
    rows, meta = load_index(a.index)
    spec = json.load(open(a.spec))
    res = analyze(rows, spec, meta)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    txt = report(res)
    Path(str(a.out).rsplit('.', 1)[0] + '.txt').write_text(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
