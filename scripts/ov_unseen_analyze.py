#!/usr/bin/env python3
"""Read-out of the Polaris unseen-arena evaluation (offroad_vehicles_20260927, PLAN amendment 9.2).

The question: does the f104-trained Polaris planner (all tiers, ensemble polaris_full_soil) reach the goal SAFELY on 90 %
or more of the 1,000 declared soil pairs of the 8 test arenas it never trained on (K3 suites/soil_unseen_subset.json:
125 pairs per arena; near g260 g271 g251 g247, spread g258 g268 g263 g241), planning from a standing start with each
arena's own overhead map, driven by the Polaris?

  goal reached safely = NOT unsafe_belly, i.e. goal reached AND not unsafe (no roll-back under throttle, min forward
                        speed > -0.30 m/s, after the 1 s settle) AND no belly-in-soil flag (lowest hull point > 0.05 m
                        under the undisturbed surface for > 1 s); labels from scripts/ov_eval_index.py.
  Arms (Polaris):  polaris_u_grad (declared default planner: own model + CEM 4 x 64 + gradient refinement),
                   polaris_u_cem (CEM 4 x 64 only), straight6_polaris_u (straight 6 m/s route = K3's straight6 picks).

Read-out (all numbers from the frozen spec written by --write-spec BEFORE the first drive):
  pooled rate over the 1,000 pairs with four 95 % intervals: Wilson (pairs), pair bootstrap, cluster bootstrap over
  (arena, nearest terrain feature) clusters, and the bootstrap over the 8 ARENAS (arenas resampled with replacement,
  pooled rate of the resampled pairs); a t interval over the 8 arena rates is reported beside it.
  Bar 90 %: 'meets the bar' = pooled point estimate >= 90.0; 'clearly above the bar' = also the lower end of the
  two-sided 95 % arena-bootstrap interval >= 90.0; else 'below the bar'. 'not physically trustworthy' if the primary
  arm's drives fail validity (a crash / non-finite / collector failure, >= 5 % launch-check failures, belly flag
  > 10 %); 'incomplete' while a declared pair has no drive of the primary arm (interim numbers on the driven pairs).
  A pair whose drive failed on the cluster twice (--failed-ids) counts as NOT reached safely and as a crash.
  Per arena: rate, Wilson and pair-bootstrap 95 % intervals, and its position against 90 % ('above' if the Wilson
  lower bound >= 90, 'below' if the upper bound < 90, else 'not distinguishable from 90').
  Paired comparisons (declared family, Holm at 0.05 over the one-sided cluster-bootstrap p-values, as K3 and
  scripts/ov_analyze.py): polaris_u_grad vs straight6_polaris_u and polaris_u_grad vs polaris_u_cem on the declared
  label, each with exact McNemar (two- and one-sided), the paired pair bootstrap and the cluster bootstrap over
  (arena, nearest feature) (scripts/ag_analyze.contrast, unchanged). Decision words of ag_analyze.decide: 'improves' /
  'no meaningful difference' (90 % cluster interval within +-2 points) / 'inconclusive'. Secondary (unadjusted): the
  same on 'fail' (goal only) and polaris_u_cem vs straight6_polaris_u.
  Context (not tests): the HMMWV planners of K3 on the same 1,000 pairs (K3 e6/index/soil_eval_v1.json: M1a, M1b, M3a,
  A3, composites M1 = mean(M1a, M1b) and M3 = mean(M3a, M3b), the HMMWV straight 6 m/s route), their rates with the same
  intervals, and the paired difference Polaris planner minus each (the HMMWV has no belly record: safe = not unsafe).

  PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts
  $PY scripts/ov_unseen_analyze.py --write-spec $K4/e6/analysis/spec_unseen_v1.json
  $PY scripts/ov_unseen_analyze.py --index <ov_eval_index output> --spec $K4/e6/analysis/spec_unseen_v1.json \
      [--failed-ids <ids that failed twice>] --out $K4/e6/analysis/results_unseen_v1.json
  $PY scripts/ov_unseen_analyze.py --selftest $K4/e6/unseen/selftest
"""
import argparse, copy, hashlib, json, math, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ov_analyze as OA                    # noqa: E402  (patches ag_analyze: labels fail_belly / unsafe_belly; load_rows; validity)
import ag_analyze as AA                    # noqa: E402  (frozen: build_by, contrast, holm, decide, pct)

K3 = ROOT / 'artifacts/traverse/arena_gator_20260925'
K4 = ROOT / 'artifacts/traverse/offroad_vehicles_20260927'
NEAR, SPREAD = ('g260', 'g271', 'g251', 'g247'), ('g258', 'g268', 'g263', 'g241')
ARENAS = NEAR + SPREAD
BAR = 90.0
Z95 = 1.959963984540054
T975 = {1: 12.7062047361747, 2: 4.302652729911275, 3: 3.182446305284263, 4: 2.7764451051977987, 5: 2.5705818366147395,
        6: 2.4469118487916806, 7: 2.3646242510102993, 8: 2.306004135033371, 9: 2.2621571627409915, 10: 2.2281388519649385}


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------------------
# spec
# ------------------------------------------------------------------------------------------------------------------
def declared_spec():
    subset = K3 / 'suites/soil_unseen_subset.json'
    sub = json.load(open(subset))
    groups = {a: sorted(sub['arenas'][a]['groups']) for a in ARENAS}
    assert all(len(v) == 125 for v in groups.values()) and len({g for v in groups.values() for g in v}) == 1000
    ctx_index = K3 / 'e6/index/soil_eval_v1.json'
    deploy = K4 / 'e5/deploy/polaris_full_soil'
    sums = {l.split()[1]: l.split()[0] for l in open(deploy / 'SHA256SUMS') if l.strip()}
    rel = lambda p: str(Path(p).relative_to(ROOT))
    lab = 'unsafe_belly'
    fam = [dict(name='U1_grad_vs_straight6', world='crm', vehicle='polaris', test='polaris_u_grad', ref='straight6_polaris_u', label=lab, set='unseen'),
           dict(name='U2_grad_vs_cem', world='crm', vehicle='polaris', test='polaris_u_grad', ref='polaris_u_cem', label=lab, set='unseen')]
    sec = [dict(name='S1_grad_vs_straight6_goal_only', world='crm', vehicle='polaris', test='polaris_u_grad', ref='straight6_polaris_u', label='fail', set='unseen'),
           dict(name='S2_grad_vs_cem_goal_only', world='crm', vehicle='polaris', test='polaris_u_grad', ref='polaris_u_cem', label='fail', set='unseen'),
           dict(name='S3_cem_vs_straight6', world='crm', vehicle='polaris', test='polaris_u_cem', ref='straight6_polaris_u', label=lab, set='unseen')]
    ctx_arms = ['M1a_free', 'M1b_free', 'M3a_free', 'A3_free', 'M1', 'M3', 'straight6']
    return dict(
        schema='ov_unseen_spec_v1', created=time.strftime('%Y-%m-%d %H:%M:%S'),
        plan=dict(file=rel(K4 / 'PLAN.md'), sha256=sha256_file(K4 / 'PLAN.md'), amendment='9.2 (2026-09-28 13:55)'),
        written_before='any drive of polaris_u_grad, polaris_u_cem or straight6_polaris_u (none existed; the 16-row pilot comes after this spec)',
        question=('Does the f104-trained Polaris planner (all tiers, polaris_full_soil) reach the goal safely on >= 90 % of the 1,000 declared '
                  'soil pairs of the 8 unseen test arenas, planning from a standing start with each arena\'s own overhead map?'),
        vehicle='polaris',
        arenas=dict(near=list(NEAR), spread=list(SPREAD)),
        subset=dict(file=rel(subset), sha256=sha256_file(subset), n_per_arena=125, n_total=1000,
                    groups_sha256=hashlib.sha256('\n'.join(g for a in ARENAS for g in groups[a]).encode()).hexdigest()),
        model=dict(glob=rel(deploy) + '/polaris_full_soil_deploy_s*.pt', sha256sums=sums, trained_on='f104 soil only, Polaris tiers 0-12'),
        arms=dict(primary='polaris_u_grad', cem='polaris_u_cem', straight='straight6_polaris_u',
                  descriptions=dict(polaris_u_grad='own model + CEM 4 x 64 + gradient refinement (ov_grad_picks.py = ci_grad defaults), standing start',
                                    polaris_u_cem='own model + CEM 4 x 64 (ag_picks.py --mode free), standing start',
                                    straight6_polaris_u="the straight 6 m/s route (K3 e6/picks/crm/<arena>/straight6), driven by the Polaris"),
                  pick_dirs=dict(polaris_u_grad='e6/picks/<arena>/polaris_u_grad', polaris_u_cem='e6/picks/<arena>/polaris_u_free',
                                 straight6_polaris_u='K3 e6/picks/crm/<arena>/straight6')),
        label=lab, labels_reported=['unsafe_belly', 'fail', 'unsafe', 'fail_belly'],
        label_meaning=dict(unsafe_belly='not reached safely: goal not reached, OR unsafe (roll-back under throttle >= 0.05 s or min forward speed <= -0.30 m/s after the 1 s settle), OR belly flag',
                           fail='goal not reached (goal-only rate, reported beside)'),
        bar=BAR, boot=4000, seed=0, alpha=0.05, margin_pts=2.0, min_groups=50, cluster_key='cluster',
        readout=dict(
            pooled='rate = 100 x share of the 1,000 pairs reached safely; 95 % intervals: Wilson (pairs), pair bootstrap, cluster bootstrap over (arena, nearest feature), '
                   'arena bootstrap (8 arenas resampled with replacement, pooled rate of the resampled pairs); t interval over the 8 arena rates reported beside',
            meets='pooled point estimate >= 90.0',
            clearly_above='meets AND lower end of the two-sided 95 % arena-bootstrap interval >= 90.0',
            verdicts=['clearly above the bar', 'meets the bar', 'below the bar', 'not physically trustworthy', 'incomplete'],
            per_arena='rate, Wilson 95 %, pair bootstrap 95 %; vs 90: above if Wilson lower >= 90, below if Wilson upper < 90, else not distinguishable from 90',
            failed_drives='a declared pair whose drive failed twice on the cluster (no completed run) counts as NOT reached safely and as a crash (validity)',
            missing_drives='any other declared pair without a drive of the primary arm -> verdict incomplete (interim numbers on the driven pairs)'),
        validity=dict(arms=['polaris_u_grad', 'polaris_u_cem', 'straight6_polaris_u'], crashed_max=0, launch_share_lt=0.05, belly_share_le=0.10,
                      note='PLAN 4.3(c) / 9.1 item 6; only the primary arm gates the verdict'),
        family=fam, contrasts=sec,
        context=dict(index=rel(ctx_index), index_sha256=sha256_file(ctx_index), vehicle='hmmwv', arms=ctx_arms,
                     combine=[dict(world='crm', vehicle='hmmwv', name='M1', members=['M1a_free', 'M1b_free']),
                              dict(world='crm', vehicle='hmmwv', name='M3', members=['M3a_free', 'M3b_free'])],
                     expected_goal_rates=dict(M1=87.95, M3=90.3, M1a_free=88.5, M3a_free=90.3, A3_free=90.6, straight6=60.0),
                     note='context only, not tests: the HMMWV planners of the Gator study (f104 only = M1, three arenas = M3, A3) on the same 1,000 pairs; '
                          'the HMMWV has no belly record, so safe = not unsafe'),
        cross_vehicle_context=[dict(test='polaris_u_grad', ref=r, label=lab) for r in ('M1a_free', 'M3a_free', 'A3_free', 'straight6')])


# ------------------------------------------------------------------------------------------------------------------
# one-sample read-out
# ------------------------------------------------------------------------------------------------------------------
def wilson(k, n, z=Z95):
    if n == 0:
        return None
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [100 * (c - h), 100 * (c + h)]


def one_rate(bad, arenas, clusters, rng, boot, bar=BAR):
    """bad: per pair 1 = not reached safely (fractional for composite arms). Rates are 'reached safely' in percent."""
    bad = np.asarray(bad, float); n = len(bad)
    if n == 0:
        return dict(n=0)
    good = 1 - bad; k = float(good.sum()); rate = 100 * k / n
    binary = bool(np.isin(bad, (0, 1)).all())
    idx = rng.integers(0, n, (boot, n)); pb = 100 * good[idx].mean(1)
    ids, inv = np.unique(clusters, return_inverse=True); K = len(ids)
    s = np.bincount(inv, good, K); c = np.bincount(inv, None, K).astype(float)
    pick = rng.integers(0, K, (boot, K)); cb = 100 * s[pick].sum(1) / c[pick].sum(1)
    aid, ainv = np.unique(arenas, return_inverse=True); A = len(aid)
    sa = np.bincount(ainv, good, A); ca = np.bincount(ainv, None, A).astype(float)
    apick = rng.integers(0, A, (boot, A)); ab = 100 * sa[apick].sum(1) / ca[apick].sum(1)
    ar = 100 * sa / ca
    t = None
    if A >= 2:
        q = T975.get(A - 1, Z95); m = float(ar.mean()); se = float(ar.std(ddof=1) / math.sqrt(A))
        t = dict(mean_of_arena_rates=m, ci95=[m - q * se, m + q * se], df=A - 1, t975=q)
    per = {}
    for j, a in enumerate(aid):
        v = good[ainv == j]; na = len(v); ka = float(v.sum())
        ia = rng.integers(0, na, (boot, na)); pa = 100 * v[ia].mean(1)
        w = wilson(ka, na) if binary else None
        pos = None if w is None else ('above' if w[0] >= bar else 'below' if w[1] < bar else 'not distinguishable from 90')
        per[str(a)] = dict(n=na, reached_safely=int(ka) if binary else ka, rate=100 * ka / na, wilson95=w, pair_boot95=AA.pct(pa, 2.5, 97.5),
                           point_ge_bar=bool(100 * ka / na >= bar), vs_bar=pos, role='near' if a in NEAR else 'spread' if a in SPREAD else 'other')
    return dict(n=n, reached_safely=int(k) if binary else k, rate=rate, binary=binary, wilson95=wilson(k, n) if binary else None,
                pair_boot95=AA.pct(pb, 2.5, 97.5), cluster_boot95=AA.pct(cb, 2.5, 97.5), clusters=int(K),
                arena_boot95=AA.pct(ab, 2.5, 97.5), arenas=int(A), arena_t95=t,
                arena_rate_min=float(ar.min()), arena_rate_max=float(ar.max()),
                arenas_point_ge_bar=int(sum(1 for v in per.values() if v['point_ge_bar'])), per_arena=per)


def near_spread(bad, arenas, clusters, rng, boot):
    out = {}
    for role, names in (('near', NEAR), ('spread', SPREAD)):
        m = np.isin(arenas, names)
        if m.any():
            r = one_rate(bad[m], arenas[m], clusters[m], rng, boot)
            out[role] = {k: r[k] for k in ('n', 'rate', 'wilson95', 'pair_boot95', 'cluster_boot95', 'arena_boot95')}
    return out


# ------------------------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------------------------
def load_polaris(index_paths, spec, failed_ids):
    rows, meta = OA.load_rows(index_paths, [])
    groups = {g for a in ARENAS for g in json.load(open(ROOT / spec['subset']['file']))['arenas'][a]['groups']}
    arms = set(spec['validity']['arms'])
    rows = [r for r in rows if r['world'] == 'crm' and r['vehicle'] == spec['vehicle'] and r['arm'] in arms and r['group'] in groups]
    seen = Counter((r['group'], r['arm']) for r in rows)
    dup = [k for k, v in seen.items() if v > 1]
    assert not dup, f'duplicate (group, arm) rows: {dup[:5]}'
    failed = set(failed_ids or [])
    n_failed = Counter()
    for r in rows:
        if r.get('missing') and r['run_id'] in failed:
            r.update(missing=False, collector_failure=True, status='collector_failure', fail=1, unsafe=1, fail_belly=1, unsafe_belly=1,
                     unsafe_noback=1, backward_only=0, tilt30=0, elapsed=None, qa_flag='collector_failure', launch_ok=None, belly_flag=None)
            n_failed[r['arm']] += 1
    return rows, meta, dict(n_failed), groups


def load_context(spec, groups):
    c = spec['context']
    p = ROOT / c['index']
    assert sha256_file(p) == c['index_sha256'], f'{p} changed since the spec was frozen'
    d = json.load(open(p))
    want = set(c['arms']) | {m for x in c['combine'] for m in x['members']}
    rows = [dict(r) for r in d['rows'] if r['world'] == 'crm' and r['vehicle'] == c['vehicle'] and r['arm'] in want and r['group'] in groups]
    for r in rows:
        if not r.get('missing'):
            r.setdefault('belly_flag', None)
            r['fail_belly'] = int(r['fail']); r['unsafe_belly'] = int(r['unsafe'])
    return rows, dict(path=str(p), sha256=c['index_sha256'], rows=len(rows))


# ------------------------------------------------------------------------------------------------------------------
# analysis
# ------------------------------------------------------------------------------------------------------------------
def validity(rows, arm, spec, vehicle):
    v = OA.validity_block(rows, dict(vehicle=vehicle, arm=arm, set='unseen',
                                     limits=dict(crashed=spec['validity']['crashed_max'], launch_share=spec['validity']['launch_share_lt'],
                                                 belly_share=spec['validity']['belly_share_le'])))
    cf = [r['run_id'] for r in rows if r['vehicle'] == vehicle and r['arm'] == arm and r.get('collector_failure')]
    v['collector_failures'] = len(cf); v['collector_failure_ids'] = cf[:20]
    v['checks']['no_crash'] = bool(v['checks']['no_crash'] and not cf)
    # collector failures have no QA files: judge 'qa_checked_all' on the completed drives only
    done = [r for r in rows if r['vehicle'] == vehicle and r['arm'] == arm and not r.get('missing') and not r.get('collector_failure')]
    v['checks']['qa_checked_all'] = bool(done) and all(r.get('qa_flag') not in (None, 'not_synced') or r.get('qa_ok') is not None for r in done)
    v['valid'] = bool(all(v['checks'].values()))
    return v


def arm_block(byw, ginfo, arm, groups_all, label, rng, boot, bar):
    G = [g for g in groups_all if g in byw and arm in byw[g]]
    bad = np.array([byw[g][arm][label] for g in G], float)
    ar = np.array([ginfo[g]['arena'] for g in G]); cl = np.array([ginfo[g]['cluster'] for g in G])
    r = one_rate(bad, ar, cl, rng, boot, bar)
    r['declared_pairs'] = len(groups_all); r['pairs_with_drive'] = len(G)
    if len(G):
        r['near_spread'] = near_spread(bad, ar, cl, rng, boot)
    return r


def verdict(r, valid, complete, bar):
    if not complete:
        return 'incomplete'
    if not valid:
        return 'not physically trustworthy'
    if r['rate'] >= bar and r['arena_boot95'][0] >= bar:
        return 'clearly above the bar'
    if r['rate'] >= bar:
        return 'meets the bar'
    return 'below the bar'


def analyze(prow, pmeta, ctx_rows, ctx_meta, spec, n_failed, groups):
    boot, bar, alpha, margin = int(spec['boot']), float(spec['bar']), float(spec['alpha']), float(spec['margin_pts'])
    rng = np.random.default_rng(int(spec['seed']))
    AA.CLUSTER_KEY[0] = spec.get('cluster_key', 'cluster')
    allrows = prow + ctx_rows
    by, ginfo = AA.build_by(allrows, spec['context']['combine'])
    byp = by.get(('crm', spec['vehicle']), {})
    G_all = sorted(groups)
    arms = spec['validity']['arms']; prim = spec['arms']['primary']
    res = dict(schema='ov_unseen_results_v1', tool='scripts/ov_unseen_analyze.py', tool_sha256=sha256_file(__file__),
               created=time.strftime('%Y-%m-%d %H:%M:%S'), spec_sha256=spec.get('_sha256'), index=pmeta, context_index=ctx_meta,
               collector_failures_counted_as_failures=n_failed)
    res['arms'] = {}
    for arm in arms:
        res['arms'][arm] = {lab: arm_block(byp, ginfo, arm, G_all, lab, rng, boot, bar) for lab in spec['labels_reported']}
    res['validity'] = {arm: validity(prow, arm, spec, spec['vehicle']) for arm in arms}
    pr = res['arms'][prim][spec['label']]
    complete = pr.get('pairs_with_drive', 0) == len(G_all) == spec['subset']['n_total']
    res['verdict'] = dict(arm=prim, label=spec['label'], bar=bar, complete=complete, pairs_with_drive=pr.get('pairs_with_drive', 0),
                          declared_pairs=len(G_all), valid=res['validity'][prim]['valid'],
                          rate=pr.get('rate'), arena_boot95=pr.get('arena_boot95'), wilson95=pr.get('wilson95'),
                          meets=bool(pr.get('n') and pr['rate'] >= bar), clearly_above=bool(pr.get('n') and pr['rate'] >= bar and pr['arena_boot95'][0] >= bar),
                          verdict=verdict(pr, res['validity'][prim]['valid'], complete, bar) if pr.get('n') else 'no data',
                          # interim label on the driven pairs: validity without the 'all driven' check (missing drives are what 'incomplete' says)
                          interim_verdict_on_driven_pairs=(None if complete or not pr.get('n') else
                                                           verdict(pr, all(v for k, v in res['validity'][prim]['checks'].items() if k != 'all_driven'), True, bar)),
                          goal_only=dict(rate=res['arms'][prim]['fail'].get('rate'), arena_boot95=res['arms'][prim]['fail'].get('arena_boot95'),
                                         verdict=(verdict(res['arms'][prim]['fail'], res['validity'][prim]['valid'], complete, bar)
                                                  if res['arms'][prim]['fail'].get('n') else 'no data')),
                          arenas_point_ge_bar=pr.get('arenas_point_ge_bar'),
                          per_arena={a: dict(rate=v['rate'], wilson95=v['wilson95'], vs_bar=v['vs_bar']) for a, v in (pr.get('per_arena') or {}).items()})

    def run(c):
        byw = by.get((c['world'], c['vehicle']), {})
        G = [g for g in AA.set_groups(ginfo, c['set'], list(byw)) if g in groups]
        return AA.contrast(byw, ginfo, G, c['test'], c['ref'], c['label'], rng, boot, margin)

    fam = [dict(c, result=run(c)) for c in spec['family']]
    pv = [f['result']['cluster']['p_one_sided'] if f['result'].get('n') else 1.0 for f in fam]
    adj, rej = AA.holm(pv, alpha) if fam else ([], [])
    for f, p, a_, r in zip(fam, pv, adj, rej):
        f['holm'] = dict(p_one_sided_cluster=p, p_adjusted=a_, reject=r, alpha=alpha, family_size=len(fam))
        f['decision'] = AA.decide(f['result'], a_, r, margin, int(spec['min_groups']))
    res['family'] = fam
    sec = []
    for c in spec['contrasts']:
        r = run(c)
        p = r['cluster']['p_one_sided'] if r.get('n') else 1.0
        sec.append(dict(c, result=r, decision=AA.decide(r, p, p <= alpha, margin, int(spec['min_groups'])) + ' (secondary, unadjusted)'))
    res['contrasts'] = sec
    # context: HMMWV planners of K3 on the same pairs (rates + paired cross-vehicle differences; not tests)
    byh = by.get(('crm', spec['context']['vehicle']), {})
    ctx = {}
    for arm in spec['context']['arms']:
        ctx[arm] = {lab: arm_block(byh, ginfo, arm, G_all, lab, rng, boot, bar) for lab in ('unsafe_belly', 'fail')}
    exp = spec['context'].get('expected_goal_rates', {})
    res['context'] = dict(arms=ctx, reproduces_k3_goal_rates={a: dict(expected=e, got=ctx[a]['fail'].get('rate'),
                                                                        equal=bool(ctx[a]['fail'].get('n') == 1000 and abs(ctx[a]['fail']['rate'] - e) < 1e-6))
                                                              for a, e in exp.items() if a in ctx})
    cv = []
    bya = by.get(('crm', 'any'), {})
    for c in spec['cross_vehicle_context']:
        G = [g for g in G_all if g in bya]
        r = AA.contrast(bya, ginfo, G, c['test'], c['ref'], c['label'], rng, boot, margin, detail=False)
        cv.append(dict(c, note='context, not a test (different vehicles)', result=r))
    res['cross_vehicle_context'] = cv
    return res


# ------------------------------------------------------------------------------------------------------------------
def f_ci(c):
    return 'n/a' if not c else f'[{c[0]:.1f}, {c[1]:.1f}]'


def report(res, spec):
    L = []
    v = res['verdict']
    L.append(f"VERDICT {v['arm']} ({spec['label']} = goal reached safely), bar {v['bar']:.0f} %: {v['verdict']}"
             + (f" (interim on {v['pairs_with_drive']}/{v['declared_pairs']} pairs: {v['interim_verdict_on_driven_pairs']})" if v['interim_verdict_on_driven_pairs'] else ''))
    if v['rate'] is not None:
        L.append(f"  pooled {v['rate']:.1f} % reached safely on {v['pairs_with_drive']} pairs; 95 % over arenas {f_ci(v['arena_boot95'])}; Wilson {f_ci(v['wilson95'])}; "
                 f"goal only {v['goal_only']['rate']:.1f} % over arenas {f_ci(v['goal_only']['arena_boot95'])} -> {v['goal_only']['verdict']}; valid {v['valid']}; "
                 f"arenas at or above 90: {v['arenas_point_ge_bar']}/8")
    for arm, labs in res['arms'].items():
        for lab in spec['labels_reported']:
            r = labs[lab]
            if not r.get('n'):
                L.append(f"arm {arm} {lab}: no drives"); continue
            L.append(f"arm {arm:20s} {lab:13s} reached {r['rate']:5.1f} % n {r['n']:4d}  Wilson {f_ci(r['wilson95'])} pairs {f_ci(r['pair_boot95'])} "
                     f"clusters({r['clusters']}) {f_ci(r['cluster_boot95'])} arenas {f_ci(r['arena_boot95'])} t {f_ci((r['arena_t95'] or {}).get('ci95'))}")
            if lab == spec['label']:
                L.append('   per arena: ' + '; '.join(f"{a} {x['rate']:.1f} {f_ci(x['wilson95'])} {x['vs_bar']}" for a, x in r['per_arena'].items()))
                ns = r.get('near_spread', {})
                L.append('   near/spread: ' + '; '.join(f"{k} {x['rate']:.1f} % {f_ci(x['cluster_boot95'])}" for k, x in ns.items()))
    for arm, x in res['validity'].items():
        L.append(f"validity {arm}: driven {x['driven']}/{x['planned']}, crashed {x['crashed_or_nonfinite']}, collector failures {x['collector_failures']}, "
                 f"launch fails {x['launch_failures']}/{x['launch_checked']}, belly {x['belly_flagged']}/{x['belly_checked']} -> valid {x['valid']} {x['checks']}")
    for f in res['family']:
        r = f['result']
        if not r.get('n'):
            L.append(f"FAMILY {f['name']}: no data"); continue
        d = r['discordant']
        L.append(f"FAMILY {f['name']} ({f['label']}): {f['test']} {100 - r['rate_test']:.1f} % vs {f['ref']} {100 - r['rate_ref']:.1f} % reached; "
                 f"failure diff {r['diff_pts']:+.1f} pts, cluster 95 % [{r['cluster']['ci95'][0]:+.1f}, {r['cluster']['ci95'][1]:+.1f}] ({r['cluster']['clusters']} clusters), "
                 f"p1 {r['cluster']['p_one_sided']:.4f}, Holm {f['holm']['p_adjusted']:.4f} -> {f['decision']}; McNemar only-test-fails / only-ref-fails "
                 f"{d['test_worse']}/{d['test_better']} (two-sided p {d['mcnemar_two_sided']}, one-sided {d['mcnemar_one_sided']}); n {r['n']}")
    for c in res['contrasts']:
        r = c['result']
        if not r.get('n'):
            L.append(f"contrast {c['name']}: no data"); continue
        L.append(f"contrast {c['name']} ({c['label']}): failure diff {r['diff_pts']:+.1f}, cluster 95 % [{r['cluster']['ci95'][0]:+.1f}, {r['cluster']['ci95'][1]:+.1f}], "
                 f"McNemar {r['discordant']['test_worse']}/{r['discordant']['test_better']} -> {c['decision']}")
    for arm, labs in res['context']['arms'].items():
        r, g = labs['unsafe_belly'], labs['fail']
        if r.get('n'):
            L.append(f"context HMMWV {arm:9s}: reached safely {r['rate']:.1f} % arenas {f_ci(r['arena_boot95'])}; goal {g['rate']:.2f} % (n {r['n']})")
    L.append(f"context reproduces K3 goal rates: {res['context']['reproduces_k3_goal_rates']}")
    for c in res['cross_vehicle_context']:
        r = c['result']
        if r.get('n'):
            L.append(f"context {c['test']} (Polaris) - {c['ref']} (HMMWV), {c['label']}: failure diff {r['diff_pts']:+.1f} pts, cluster 95 % "
                     f"[{r['cluster']['ci95'][0]:+.1f}, {r['cluster']['ci95'][1]:+.1f}], only-Polaris / only-HMMWV fails {r['discordant']['test_worse']}/{r['discordant']['test_better']}")
    return '\n'.join(L)


# ------------------------------------------------------------------------------------------------------------------
# self-test
# ------------------------------------------------------------------------------------------------------------------
def mock_rows_from_k3(spec, relabel):
    """K3 HMMWV drives relabelled as Polaris arms (plumbing: rates and paired numbers must equal the K3 ones)."""
    d = json.load(open(ROOT / spec['context']['index']))
    groups = {g for a in ARENAS for g in json.load(open(ROOT / spec['subset']['file']))['arenas'][a]['groups']}
    out = []
    for r in d['rows']:
        if r['world'] == 'crm' and r['vehicle'] == 'hmmwv' and r['arm'] in relabel and r['group'] in groups:
            x = dict(r, arm=relabel[r['arm']], vehicle='polaris', qa_ok=True, qa_flag=None, launch_ok=True, belly_flag=0)
            x['fail_belly'] = int(x['fail']); x['unsafe_belly'] = int(x['unsafe'])
            out.append(x)
    return out


def synthetic_rows(p_by_arena, seed, arm='polaris_u_grad', drop=0):
    rng = np.random.default_rng(seed)
    rows = []
    for a in ARENAS:
        n_bad = int(round((1 - p_by_arena[a]) * 125))          # exact planted count per arena, random positions
        bad_k = set(rng.permutation(125)[:n_bad].tolist())
        for k in range(125):
            g = f'{a}_test_group_{k:04d}'
            bad = int(k in bad_k)
            rows.append(dict(world='crm', vehicle='polaris', arena=a, role='near' if a in NEAR else 'spread', set='unseen', group=g, arm=arm,
                             run_id=f'polaris__{g}__{arm}', cluster=f'{a}:{k % 9}', cluster_design=f'{a}:{k % 7}', missing=False,
                             fail=bad, unsafe=bad, unsafe_noback=bad, backward_only=0, tilt30=0, elapsed=20.0, status='x', route_sha256=f'{g}{arm}',
                             qa_ok=True, qa_flag=None, launch_ok=True, belly_flag=0, fail_belly=bad, unsafe_belly=bad,
                             dist_nearest_training=0.5, map_err_rmse_m=0.05))
    for r in rows[:drop]:
        for k in ('fail', 'unsafe', 'fail_belly', 'unsafe_belly', 'status'):
            r.pop(k, None)
        r['missing'] = True
    return rows


def selftest(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    spec = declared_spec(); spec['boot'] = 2000
    groups = {g for a in ARENAS for g in json.load(open(ROOT / spec['subset']['file']))['arenas'][a]['groups']}
    ck = {}
    w = wilson(90, 100)
    ck['wilson_90_of_100'] = bool(abs(w[0] - 82.5629) < 1e-3 and abs(w[1] - 94.4776) < 1e-3)
    w0 = wilson(0, 10); w1 = wilson(10, 10)
    ck['wilson_edges'] = bool(abs(w0[0]) < 1e-9 and abs(w1[1] - 100) < 1e-9 and w0[1] > 0 and w1[0] < 100)
    ctx_rows, ctx_meta = load_context(spec, groups)
    # 1. mock on K3's stored drives: M3a -> polaris_u_grad, A3 -> polaris_u_cem, straight6 -> straight6_polaris_u
    mock = mock_rows_from_k3(spec, {'M3a_free': 'polaris_u_grad', 'A3_free': 'polaris_u_cem', 'straight6': 'straight6_polaris_u'})
    res = analyze(copy.deepcopy(mock), [dict(mock=True)], copy.deepcopy(ctx_rows), ctx_meta, spec, {}, groups)
    txt = report(res, spec)
    g = res['arms']['polaris_u_grad']['fail']
    ck['mock_goal_rate_equals_K3_M3a_90p3'] = bool(g['n'] == 1000 and abs(g['rate'] - 90.3) < 1e-9)
    ck['mock_safe_rate_equals_K3_M3a'] = bool(abs(res['arms']['polaris_u_grad']['unsafe_belly']['rate'] - 90.3) < 1e-9)
    ck['mock_per_arena_125'] = bool(all(v['n'] == 125 for v in g['per_arena'].values()) and len(g['per_arena']) == 8)
    ck['mock_arena_ci_contains_point'] = bool(g['arena_boot95'][0] <= g['rate'] <= g['arena_boot95'][1])
    ck['mock_arena_ci_within_min_max'] = bool(g['arena_rate_min'] - 1e-9 <= g['arena_boot95'][0] and g['arena_boot95'][1] <= g['arena_rate_max'] + 1e-9)
    # the paired numbers equal ag_analyze on the original names
    by, gi = AA.build_by(ctx_rows, spec['context']['combine'])
    bw = by[('crm', 'hmmwv')]
    G = sorted(g_ for g_ in groups if 'M3a_free' in bw[g_] and 'straight6' in bw[g_])
    ref = AA.contrast(bw, gi, G, 'M3a_free', 'straight6', 'unsafe_belly', np.random.default_rng(5), 2000, 2.0, detail=False)
    f1 = res['family'][0]['result']
    ck['mock_paired_equals_ag_analyze'] = bool(abs(f1['diff_pts'] - ref['diff_pts']) < 1e-9 and f1['discordant'] == ref['discordant'])
    ck['mock_context_reproduces_K3'] = bool(all(v['equal'] for v in res['context']['reproduces_k3_goal_rates'].values()))
    ck['mock_verdict'] = res['verdict']['verdict']
    ck['mock_verdict_ok'] = bool(res['verdict']['verdict'] in ('meets the bar', 'below the bar', 'clearly above the bar'))
    (out / 'mock_k3_report.txt').write_text(txt + '\n')
    json.dump(res, open(out / 'mock_k3_results.json', 'w'), indent=1, default=float)
    # 2. planted (exact counts): every arena 121/125 = 96.8 % -> clearly above; near 124/125, spread 100/125 -> pooled 89.6 ->
    #    below; near 124/125, spread 110/125 -> pooled 93.6 but the 2.5 % arena-bootstrap point is 7 low arenas of 8 = 89.4 -> meets
    cases = dict(all97=({a: 0.968 for a in ARENAS}, 'clearly above the bar'),
                 split80=({a: (0.992 if a in NEAR else 0.80) for a in ARENAS}, 'below the bar'),
                 split88=({a: (0.992 if a in NEAR else 0.88) for a in ARENAS}, 'meets the bar'))
    for name, (p, want) in cases.items():
        rows = synthetic_rows(p, 11)
        for arm in ('polaris_u_cem', 'straight6_polaris_u'):
            rows += synthetic_rows(p, 12 if arm == 'polaris_u_cem' else 13, arm=arm)
        sg = {r['group'] for r in rows}
        r = analyze(rows, [dict(synthetic=name)], [], dict(), dict(spec, cross_vehicle_context=[]), {}, sg)
        ck[f'planted_{name}'] = r['verdict']['verdict']
        ck[f'planted_{name}_ok'] = bool(r['verdict']['verdict'] == want)
    # 3. a missing drive -> incomplete; the same id listed as failed twice -> counted as a failure and as a crash
    rows = synthetic_rows({a: 0.97 for a in ARENAS}, 11, drop=1)
    sg = {r['group'] for r in rows}
    r = analyze(copy.deepcopy(rows), [dict(synthetic='drop1')], [], dict(), dict(spec, cross_vehicle_context=[], family=[], contrasts=[]), {}, sg)
    ck['missing_is_incomplete'] = bool(r['verdict']['verdict'] == 'incomplete' and r['verdict']['interim_verdict_on_driven_pairs'] is not None)
    fid = rows[0]['run_id']
    rr = copy.deepcopy(rows)
    for x in rr:
        if x['run_id'] == fid:
            x['missing'] = True
            if x.get('collector_failure'):
                x.pop('collector_failure')
    nf = Counter()
    for x in rr:
        if x.get('missing') and x['run_id'] == fid:
            x.update(missing=False, collector_failure=True, status='collector_failure', fail=1, unsafe=1, fail_belly=1, unsafe_belly=1,
                     unsafe_noback=1, backward_only=0, tilt30=0, qa_flag='collector_failure', launch_ok=None, belly_flag=None)
            nf[x['arm']] += 1
    r2 = analyze(rr, [dict(synthetic='failed1')], [], dict(), dict(spec, cross_vehicle_context=[], family=[], contrasts=[]), dict(nf), sg)
    ck['failed_counts_as_failure_and_crash'] = bool(r2['verdict']['complete'] and r2['verdict']['verdict'] == 'not physically trustworthy'
                                                    and r2['validity']['polaris_u_grad']['collector_failures'] == 1)
    ok = all(v for k, v in ck.items() if k.endswith('_ok') or isinstance(v, bool))
    json.dump(dict(checks=ck, passed=ok), open(out / 'selftest.json', 'w'), indent=1, default=str)
    print(json.dumps(ck, indent=1, default=str)); print('OV_UNSEEN_ANALYZE SELFTEST', 'PASSED' if ok else 'FAILED')
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write-spec'); ap.add_argument('--selftest')
    ap.add_argument('--index', nargs='*', default=[]); ap.add_argument('--spec'); ap.add_argument('--out')
    ap.add_argument('--failed-ids', help='file: one run id per line whose drive failed twice on the cluster (no completed run)')
    a = ap.parse_args(argv)
    if a.selftest:
        sys.exit(0 if selftest(a.selftest) else 1)
    if a.write_spec:
        p = Path(a.write_spec)
        assert not p.exists(), f'{p} exists: a frozen spec is never rewritten'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(declared_spec(), indent=1) + '\n')
        print(p, sha256_file(p)); return
    assert a.index and a.spec and a.out, '--index, --spec and --out are required'
    spec = json.load(open(a.spec)); spec['_sha256'] = sha256_file(a.spec)
    failed = [l.strip() for l in open(a.failed_ids) if l.strip()] if a.failed_ids else []
    prow, pmeta, n_failed, groups = load_polaris(a.index, spec, failed)
    ctx_rows, ctx_meta = load_context(spec, groups)
    res = analyze(prow, pmeta, ctx_rows, ctx_meta, spec, n_failed, groups)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    txt = report(res, spec)
    Path(str(a.out).rsplit('.', 1)[0] + '.txt').write_text(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
