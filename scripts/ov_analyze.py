#!/usr/bin/env python3
"""Declared analysis of the soil planner evaluation for any vehicle (offroad_vehicles_20260927, module M4; PLAN 4.3-4.4).
A thin layer over scripts/ag_analyze.py (frozen; every paired statistic is its code: paired group bootstrap, cluster
bootstrap over the nearest terrain feature, exact McNemar, Holm over the family's one-sided CLUSTER p-values, the +-2
point 'no meaningful difference' rule, headroom, time ratio, rates tables). What this file adds:

  sets      'f104_600' (the 600 fresh f104_pair_group pairs; CEM 4 x 64 was tuned on the other 200) and 'f104_200'
            (f104_crm_eval_group) beside ag_analyze's 'f104_800'.
  labels    'fail_belly' / 'unsafe_belly' (fail / unsafe OR the belly-in-soil flag of scripts/ov_eval_index.py; equal to
            fail / unsafe for drives without a belly record) beside ag_analyze's labels.
  bar       the 90 % bar of PLAN 4.3(b), one sample per (vehicle, arm, label, set): goal-reached rate = 100 - rate(label);
            'meets the bar' = point estimate >= 90.0 on EVERY listed set (800 and 600 fresh); 'clearly above' = also
            the lower end of the two-sided 95 % paired-group bootstrap interval >= 90.0 on every set (the one-sided 95 %
            bound and the cluster-bootstrap bounds are reported beside it).
  validity  PLAN 4.3(c) per (vehicle, arm): missing drives, crashed / non-finite / exploded runs (crm_qa flags nonfinite,
            explosion, shape, unreadable: must be 0), launch-check failures (< 5 %), belly flag share (<= 10 %), and
            whether the bar still holds with belly-flagged drives counted as failures (label fail_belly).
  criteria  per declared vehicle, PLAN 4.3 "the planner works": (a) its <v>_grad vs straight6 family test decides
            'improves' (Holm), (b) the bar is met on 800 and on 600 fresh, (c) valid drives (all of the above). Verdict:
            'works' | 'helps but below the bar' ((a), (c) hold, (b) not) | 'not physically trustworthy' ((c) fails) |
            'does not beat the straight route'. The 'clearly above' flag is reported with it.
  reuse     --reuse INDEX=ARM[,ARM...]: take only these arms from another index (e.g. stored Gator CEM / straight drives).

  PYTHONPATH=src:scripts python scripts/ov_analyze.py --write-spec <spec.json> --vehicle polaris   # the PLAN 4.4 family
  python scripts/ov_analyze.py --index <ov index> [--reuse <index>=Gfull_free_gator,straight6_gator] --spec <spec> --out <res.json>
  python scripts/ov_analyze.py --selftest <dir>      # reproduces arena_gator's Bfull numbers (G_full 67.4 % on the 800)
"""
import argparse, copy, hashlib, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ag_analyze as AA                    # noqa: E402  (frozen)

EXTRA_LABELS = ('fail_belly', 'unsafe_belly')
_ORIG_SET_GROUPS = AA.set_groups
CRASH_FLAGS = ('nonfinite', 'explosion', 'shape', 'unreadable')
BAR = 90.0


SAMPLE_B = ROOT / 'artifacts/traverse/offroad_vehicles_20260927/scratch/S3/sample_B.json'
_B_GROUPS = []


def sample_b_groups():
    if not _B_GROUPS:
        _B_GROUPS.extend(json.load(open(SAMPLE_B))['groups'])
        assert len(_B_GROUPS) == 96
    return set(_B_GROUPS)


def set_groups(ginfo, name, groups):
    if name == 'f104_800_notB':        # REVIEW_R1 N3: the 704 suite pairs whose outcomes were not seen in the smoke (sample B)
        B = sample_b_groups()
        return sorted(g for g in _ORIG_SET_GROUPS(ginfo, 'f104_800', groups) if g not in B)
    if name == 'f104_600':
        return sorted(g for g in groups if ginfo[g]['arena'] == 'f104' and g.startswith('f104_pair_group_'))
    if name == 'f104_200':
        return sorted(g for g in groups if ginfo[g]['arena'] == 'f104' and g.startswith('f104_crm_eval_group_'))
    return _ORIG_SET_GROUPS(ginfo, name, groups)


AA.set_groups = set_groups
AA.LABELS = tuple(AA.LABELS) + EXTRA_LABELS


def load_rows(index_paths, reuse):
    rows, meta = AA.load_index(index_paths) if index_paths else ([], [])
    for spec in reuse or []:
        p, arms = spec.rsplit('=', 1)
        want = set(arms.split(','))
        d = json.load(open(p))
        got = [r for r in d['rows'] if r['arm'] in want]
        assert {r['arm'] for r in got} == want, f'{p}: arms {sorted(want - {r["arm"] for r in got})} not in the index'
        rows += got
        meta.append(dict(path=str(p), sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(), reused_arms=sorted(want), rows=len(got)))
    for r in rows:
        if r.get('missing'):
            continue
        bf = r.get('belly_flag')
        r.setdefault('belly_known', bf is not None)
        r.setdefault('fail_belly', int(r['fail'] or (bf or 0)))
        r.setdefault('unsafe_belly', int(r['unsafe'] or (bf or 0)))
    return rows, meta


def one_sample(v, cl, rng, boot):
    """Goal-reached rate (100 - label rate) with group- and cluster-bootstrap bounds."""
    n = len(v); rate = 100 * (1 - v.mean())
    idx = rng.integers(0, n, (boot, n)); gb = 100 * (1 - v[idx].mean(1))
    ids, inv = np.unique(cl, return_inverse=True); K = len(ids)
    s = np.bincount(inv, v, K); c = np.bincount(inv, None, K).astype(float)
    pick = rng.integers(0, K, (boot, K)); cb = 100 * (1 - s[pick].sum(1) / c[pick].sum(1))
    return dict(n=n, goal_rate=float(rate), group_ci95=AA.pct(gb, 2.5, 97.5), group_lower95_one_sided=float(np.percentile(gb, 5)),
                cluster_ci95=AA.pct(cb, 2.5, 97.5), cluster_lower95_one_sided=float(np.percentile(cb, 5)), clusters=int(K))


def bar_block(rows_by, ginfo, b, rng, boot):
    byw = rows_by.get(('crm', b['vehicle']), {})
    out = dict(b, per_set={})
    for s in b['sets']:
        G = [g for g in set_groups(ginfo, s, list(byw)) if b['arm'] in byw[g]]
        if not G:
            out['per_set'][s] = dict(n=0); continue
        v = np.array([byw[g][b['arm']][b['label']] for g in G], float)
        cl = np.array([ginfo[g]['cluster'] for g in G])
        out['per_set'][s] = one_sample(v, cl, rng, boot)
    ps = [out['per_set'][s] for s in b['sets']]
    have = all(p.get('n') for p in ps)
    out['meets_bar'] = bool(have and all(p['goal_rate'] >= b.get('bar', BAR) for p in ps))
    # 'clearly above': PLAN 4.3(b) = lower end of the 95 % group-bootstrap interval; REVIEW_R1 S6 = the one-sided 95 %
    # cluster-bootstrap bound (ag_analyze's 9 terrain clusters). Both are computed; the spec's 'clearly_above_bound' decides.
    out['clearly_above_group95'] = bool(have and all(p['group_ci95'][0] >= b.get('bar', BAR) for p in ps))
    out['clearly_above_cluster95_one_sided'] = bool(have and all(p['cluster_lower95_one_sided'] >= b.get('bar', BAR) for p in ps))
    rule = b.get('clearly_above_bound', 'group95')
    out['clearly_above_rule'] = rule
    out['clearly_above'] = out['clearly_above_group95'] if rule == 'group95' else out['clearly_above_cluster95_one_sided']
    out['gap_to_bar_pts'] = {s: (None if not p.get('n') else p['goal_rate'] - b.get('bar', BAR)) for s, p in zip(b['sets'], ps)}
    return out


def validity_block(rows, v):
    rs = [r for r in rows if r['world'] == 'crm' and r['vehicle'] == v['vehicle'] and r['arm'] == v['arm'] and
          (v.get('set') is None or set_groups({r['group']: dict(arena=r['arena'], role=r['role'], set=r['set'], cluster=r['cluster'])}, v['set'], [r['group']]))]
    drv = [r for r in rs if not r.get('missing')]
    qa_known = [r for r in drv if r.get('qa_flag') not in (None, 'not_synced') or r.get('qa_ok') is not None]
    crashed = [r['run_id'] for r in drv if r.get('qa_flag') in CRASH_FLAGS]
    launch_known = [r for r in drv if r.get('launch_ok') is not None]
    launch_fail = [r['run_id'] for r in launch_known if not r['launch_ok']]
    belly_known = [r for r in drv if r.get('belly_flag') is not None]
    belly = [r['run_id'] for r in belly_known if r['belly_flag']]
    lim = v.get('limits', dict(crashed=0, launch_share=0.05, belly_share=0.10))
    out = dict(v, planned=len(rs), driven=len(drv), missing=len(rs) - len(drv), qa_checked=len(qa_known), crashed_or_nonfinite=len(crashed),
               crashed_ids=crashed[:20], launch_checked=len(launch_known), launch_failures=len(launch_fail),
               launch_share=(len(launch_fail) / len(launch_known)) if launch_known else None,
               belly_checked=len(belly_known), belly_flagged=len(belly), belly_share=(len(belly) / len(belly_known)) if belly_known else None,
               qa_flags={f: sum(1 for r in drv if r.get('qa_flag') == f) for f in sorted({r.get('qa_flag') for r in drv if r.get('qa_flag')})})
    checks = dict(all_driven=out['missing'] == 0, qa_checked_all=len(qa_known) == len(drv) and len(drv) > 0,
                  no_crash=len(crashed) <= lim['crashed'],
                  launch_ok=out['launch_share'] is not None and out['launch_share'] < lim['launch_share'],
                  belly_ok=(out['belly_share'] is None and v.get('belly_optional', False)) or (out['belly_share'] is not None and out['belly_share'] <= lim['belly_share']))
    out['checks'] = checks
    out['valid'] = bool(all(checks.values()))
    return out


def analyze(rows, spec, meta):
    res = AA.analyze(rows, spec, meta)
    rng = np.random.default_rng(int(spec.get('seed', 0)) + 1)
    by, ginfo = AA.build_by(rows, spec.get('combine'))
    boot = int(spec.get('boot', 4000))
    res['bar'] = [bar_block(by, ginfo, b, rng, boot) for b in spec.get('bar', [])]
    res['validity'] = [validity_block(rows, v) for v in spec.get('validity', [])]
    fam = {f['name']: f for f in res['family']}
    bars = {(b['vehicle'], b['arm'], b['label']): b for b in res['bar'] if not b.get('key_suffix')}
    vals = {(v['vehicle'], v['arm']): v for v in res['validity']}
    crit = []
    for c in spec.get('criteria', []):
        f = fam.get(c['beats_straight_test'])
        a_ok = bool(f and f['decision'] == 'improves')
        b = bars.get((c['vehicle'], c['arm'], c.get('bar_label', 'fail')))
        bb = bars.get((c['vehicle'], c['arm'], c.get('belly_label', 'fail_belly')))
        b_ok = bool(b and b['meets_bar'])
        v = vals.get((c['vehicle'], c['arm']))
        c_ok = bool(v and v['valid'] and bb and bb['meets_bar'])
        # PLAN 4.3: works = (a) and (b) and (c).  REVIEW_R1 S3 (if adopted): Q3 = (b) and (c); (a) reported on its own.
        v_plan = ('works' if a_ok and b_ok and c_ok else 'not physically trustworthy' if not c_ok else
                  'helps but below the bar' if a_ok else 'does not beat the straight route')
        v_r1 = ('meets the bar' if b_ok and c_ok else 'not physically trustworthy' if not c_ok else 'below the bar')
        rule = c.get('rule', 'plan')
        cem = bars.get((c['vehicle'], c.get('cem_arm'), c.get('bar_label', 'fail'))) if c.get('cem_arm') else None
        st = bars.get((c['vehicle'], c.get('straight_arm'), c.get('bar_label', 'fail'))) if c.get('straight_arm') else None
        st800 = (st['per_set'].get('f104_800') or {}).get('goal_rate') if st else None
        crit.append(dict(c, a_beats_straight=a_ok, a_decision=f['decision'] if f else None,
                         a_readout=('yes' if a_ok else 'no' if (f and f['result'].get('n') and f['result']['diff_pts'] >= 0) else 'inconclusive'),
                         b_meets_bar=b_ok, b_goal_rates={s: p.get('goal_rate') for s, p in (b['per_set'].items() if b else [])},
                         b_clearly_above=bool(b and b['clearly_above']), c_valid=c_ok, c_validity=v['checks'] if v else None,
                         c_bar_with_belly_as_failure=bool(bb and bb['meets_bar']), verdict_plan=v_plan, verdict_r1_s3=v_r1,
                         verdict=v_plan if rule == 'plan' else v_r1, rule=rule,
                         cem_meets_bar_but_grad_not=bool(cem and cem['meets_bar'] and not b_ok),
                         straight_goal_rate_800=st800,
                         ceiling_note=(None if st800 is None or st800 < 88.0 else
                                       f'the straight route already reaches {st800:.1f} % of the 800 pairs: the bar is met by the straight route '
                                       f'if the planner arm is not better; report the planner gain with its interval')))
    res['criteria'] = crit
    res['tool_ov'] = dict(tool='scripts/ov_analyze.py', sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return res


def report(res):
    L = [AA.report(res)]
    for b in res.get('bar', []):
        parts = [f"{s} {p['goal_rate']:.1f} % [{p['group_ci95'][0]:.1f}, {p['group_ci95'][1]:.1f}] n {p['n']}" if p.get('n') else f'{s} no data'
                 for s, p in b['per_set'].items()]
        L.append(f"bar {b['vehicle']}/{b['arm']} ({b['label']}): " + '; '.join(parts) + f" -> meets {b['meets_bar']}, clearly above {b['clearly_above']}")
    for v in res.get('validity', []):
        L.append(f"validity {v['vehicle']}/{v['arm']}: driven {v['driven']}/{v['planned']}, crashed {v['crashed_or_nonfinite']}, launch fails "
                 f"{v['launch_failures']}/{v['launch_checked']}, belly {v['belly_flagged']}/{v['belly_checked']} -> valid {v['valid']} {v['checks']}")
    for c in res.get('criteria', []):
        L.append(f"CRITERIA {c['vehicle']}/{c['arm']} ({c['bar_label']}): (a) beats straight {c['a_beats_straight']} ({c['a_decision']}), (b) bar {c['b_meets_bar']} "
                 f"{ {k: round(x, 1) for k, x in c['b_goal_rates'].items() if x is not None} } clearly above {c['b_clearly_above']}, (c) valid {c['c_valid']} "
                 f"-> {c['verdict']} (rule {c['rule']}; PLAN: {c['verdict_plan']}; R1-S3: {c['verdict_r1_s3']})"
                 + (f"; NOTE cem meets the bar, grad does not" if c['cem_meets_bar_but_grad_not'] else '') + (f"; {c['ceiling_note']}" if c['ceiling_note'] else ''))
    return '\n'.join(L)


# ------------------------------------------------------------------------------------------------------------------
def declared_spec(vehicles, gator_cem='Gfull_free_gator', gator_straight='straight6_gator', gator_grad='Gfull_grad_gator',
                  h_grad='Hfull_grad_hmmwv', h_cem='Hfull_free', h_straight='straight6', bar_label='fail', rule='plan',
                  clearly_above_bound='group95', extra_bars=()):
    """PLAN 4.3-4.4 family and criteria. Arm names: <v>_grad, <v>_cem, straight6_<v> per vehicle; Gator and HMMWV names
    as given (defaults: the stored arena_gator CEM / straight arms reused, new gradient arms). Options for the REVIEW_R1
    amendments (only if the orchestrator adopts them): bar_label 'unsafe_belly' (S4: reached safely, belly = failure),
    rule 'r1' (S3: Q3 = (b) and (c)), clearly_above_bound 'cluster95' (S6); extra_bars = [(vehicle, arm)] reported
    bars (S1: the polaris_grad picks driven with polaris_pc)."""
    S, S6 = 'f104_800', 'f104_600'
    c = lambda name, test, ref, vehicle, label='fail', set_=S, **kw: dict(name=name, world='crm', vehicle=vehicle, test=test, ref=ref,
                                                                          label=label, set=set_, **kw)
    fam, sec, head, tr, bar, val, crit = [], [], [], [], [], [], []
    for v in vehicles:
        g, m, s = f'{v}_grad', f'{v}_cem', f'straight6_{v}'
        fam += [c(f'F_{v}_grad_vs_straight6', g, s, v), c(f'F_{v}_grad_vs_gator_grad', g, gator_grad, 'any'), c(f'F_{v}_grad_vs_cem', g, m, v)]
        sec += [c(f'S_{v}_cem_vs_gator_cem', m, gator_cem, 'any'), c(f'S_{v}_grad_vs_hmmwv_grad', g, h_grad, 'any'),
                c(f'S_{v}_grad_vs_straight6_unsafe', g, s, v, 'unsafe'), c(f'S_{v}_grad_vs_gator_grad_unsafe', g, gator_grad, 'any', 'unsafe'),
                c(f'S_{v}_grad_vs_straight6_fresh600', g, s, v, set_=S6), c(f'S_{v}_grad_vs_gator_grad_fresh600', g, gator_grad, 'any', set_=S6),
                c(f'S_{v}_grad_vs_cem_fresh600', g, m, v, set_=S6), c(f'S_{v}_grad_vs_gator_grad_belly', g, gator_grad, 'any', 'fail_belly')]
        head.append(dict(world='crm', vehicle=v, model=g, straight=s, label='fail', set=S))
        head.append(dict(world='crm', vehicle=v, model=m, straight=s, label='fail', set=S))
        tr.append(dict(world='crm', vehicle=v, test=g, ref=m, set=S))
        belly_label = 'unsafe_belly' if bar_label in ('unsafe', 'unsafe_belly') else 'fail_belly'
        for lab in ('fail', 'unsafe', 'fail_belly', 'unsafe_belly'):
            bar.append(dict(vehicle=v, arm=g, label=lab, sets=[S, S6], bar=BAR, declared=(lab == bar_label), clearly_above_bound=clearly_above_bound))
            bar.append(dict(vehicle=v, arm=g, label=lab, sets=['f104_800_notB'], bar=BAR, declared=False, clearly_above_bound=clearly_above_bound,
                            note='REVIEW_R1 N3: the 704 pairs outside sample B', key_suffix='notB'))
        for arm in (m, s):
            bar.append(dict(vehicle=v, arm=arm, label=bar_label, sets=[S, S6], bar=BAR, declared=False, clearly_above_bound=clearly_above_bound))
        for arm in (g, m, s):
            val.append(dict(vehicle=v, arm=arm, set=S))
        crit.append(dict(vehicle=v, arm=g, beats_straight_test=f'F_{v}_grad_vs_straight6', bar_label=bar_label, belly_label=belly_label,
                         cem_arm=m, straight_arm=s, rule=rule))
    fam.append(c('F_gator_grad_vs_gator_cem', gator_grad, gator_cem, 'gator'))
    sec += [c('S_gator_grad_vs_straight6', gator_grad, gator_straight, 'gator'), c('S_hmmwv_grad_vs_cem_bridge', h_grad, h_cem, 'hmmwv'),
            c('S_hmmwv_grad_vs_straight6', h_grad, h_straight, 'hmmwv'), c('S_gator_grad_vs_gator_cem_fresh600', gator_grad, gator_cem, 'gator', set_=S6)]
    head += [dict(world='crm', vehicle='gator', model=gator_grad, straight=gator_straight, label='fail', set=S),
             dict(world='crm', vehicle='hmmwv', model=h_grad, straight=h_straight, label='fail', set=S)]
    tr += [dict(world='crm', vehicle='gator', test=gator_grad, ref=gator_cem, set=S), dict(world='crm', vehicle='hmmwv', test=h_grad, ref=h_cem, set=S)]
    for veh, arm in (('gator', gator_grad), ('gator', gator_cem), ('hmmwv', h_grad), ('hmmwv', h_cem)) + tuple(extra_bars):
        bar.append(dict(vehicle=veh, arm=arm, label=bar_label, sets=[S, S6], bar=BAR, declared=False, clearly_above_bound=clearly_above_bound))
    for veh, arm in extra_bars:
        val.append(dict(vehicle=veh, arm=arm, set=S))
    val += [dict(vehicle='gator', arm=gator_grad, set=S), dict(vehicle='hmmwv', arm=h_grad, set=S, belly_optional=True)]
    return dict(schema='ov_analyze_spec_v1', margin_pts=2.0, alpha=0.05, boot=4000, seed=0, cluster_key='cluster', min_groups=50,
                options=dict(bar_label=bar_label, rule=rule, clearly_above_bound=clearly_above_bound, extra_bars=[list(x) for x in extra_bars]),
                note=('offroad_vehicles_20260927 PLAN 4.3-4.4, written before any outcome of these arms exists. Soil, f104 800-pair suite, '
                      'standing start, fail = goal not reached. Family (Holm at 0.05 over the one-sided cluster-bootstrap p; group bootstrap and '
                      'exact McNemar beside each): per vehicle <v>_grad vs straight6_<v>, <v>_grad vs the Gator with its best planner, '
                      '<v>_grad vs <v>_cem; and the Gator gradient vs its stored CEM arm. Decision: improves if Holm rejects; no meaningful '
                      'difference if the 90 % cluster interval lies within +-2 points; else inconclusive. The 90 % bar on <v>_grad (goal reached, '
                      '800 and 600 fresh; clearly above if the lower end of the 95 % group-bootstrap interval >= 90 on both), validity (0 crashed, '
                      '< 5 % launch failures, belly flag <= 10 %, the bar with belly-flagged drives counted as failures). HMMWV arms: validity '
                      'without a belly record.'),
                family=fam, contrasts=sec, headroom=head, time_ratio=tr, bar=bar, validity=val, criteria=crit,
                rates=dict(sets=[S, S6, 'f104_200']))


def selftest(out):
    """Reproduce the stored arena_gator Bfull analysis with this layer: every family / contrast number and the rates of the
    stored results, from the stored index (and from ov_eval_index's rebuild of it when given)."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    K3 = ROOT / 'artifacts/traverse/arena_gator_20260925'
    idx = K3 / 'e6/index/soil_eval_bfull.json'
    spec = json.load(open(K3 / 'e6/analysis/spec_soil_v1_Bfull.json'))
    ref = json.load(open(K3 / 'e6/analysis/results_soil_v1_Bfull.json'))
    rows, meta = load_rows([idx], [])
    res = analyze(rows, spec, meta)
    checks = {}
    for f, g in zip(res['family'], ref['family']):
        checks[f['name']] = bool(f['name'] == g['name'] and f['decision'] == g['decision'] and
                                 all(np.isclose(f['result'][k], g['result'][k]) for k in ('diff_pts', 'rate_test', 'rate_ref')) and
                                 np.allclose(f['result']['cluster']['ci90'], g['result']['cluster']['ci90']) and
                                 np.isclose(f['holm']['p_adjusted'], g['holm']['p_adjusted']))
    for f, g in zip(res['contrasts'], ref['contrasts']):
        checks[f['name']] = bool(f['decision'] == g['decision'] and np.isclose(f['result']['diff_pts'], g['result']['diff_pts'])
                                 and np.allclose(f['result']['group']['ci95'], g['result']['group']['ci95']))
    r800 = res['rates']['crm|gator']['f104_800']['Gfull_free_gator']
    checks['G_full_goal_67p4_on_800'] = bool(abs(r800['goal_reached'] - 67.375) < 1e-9 and r800['n'] == 800)
    checks['rates_equal_stored'] = bool(all(np.isclose(res['rates'][k][s][a]['goal_reached'], ref['rates'][k][s][a]['goal_reached'])
                                            for k in ref['rates'] for s in ref['rates'][k] for a in ref['rates'][k][s]))
    # the added blocks on the stored data: the Gator's G_full CEM arm against the bar, validity from the rebuild
    bspec = dict(vehicle='gator', arm='Gfull_free_gator', label='fail', sets=['f104_800', 'f104_600'], bar=BAR)
    by, ginfo = AA.build_by(rows, None)
    bb = bar_block(by, ginfo, bspec, np.random.default_rng(1), 4000)
    checks['bar_block_800_rate'] = bool(abs(bb['per_set']['f104_800']['goal_rate'] - 67.375) < 1e-9 and not bb['meets_bar'])
    fresh = [g for g in by[('crm', 'gator')] if g.startswith('f104_pair_group_')]
    checks['f104_600_has_600'] = bool(bb['per_set']['f104_600']['n'] == 600 == len(fresh))
    ok = all(checks.values())
    json.dump(dict(checks=checks, passed=ok, bar_example=bb, G_full_rate_800=r800), open(out / 'ov_analyze_selftest.json', 'w'), indent=1, default=float)
    (out / 'ov_analyze_selftest_report.txt').write_text(report(res) + '\n')
    print(json.dumps(checks, indent=1)); print('OV_ANALYZE SELFTEST', 'PASSED' if ok else 'FAILED')
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index', nargs='*', default=[]); ap.add_argument('--reuse', nargs='*', default=[], help='INDEX=ARM[,ARM]')
    ap.add_argument('--spec'); ap.add_argument('--out')
    ap.add_argument('--write-spec'); ap.add_argument('--vehicle', action='append', default=[])
    ap.add_argument('--gator-cem', default='Gfull_free_gator'); ap.add_argument('--gator-straight', default='straight6_gator')
    ap.add_argument('--gator-grad', default='Gfull_grad_gator'); ap.add_argument('--hmmwv-grad', default='Hfull_grad_hmmwv')
    ap.add_argument('--hmmwv-cem', default='Hfull_free'); ap.add_argument('--hmmwv-straight', default='straight6')
    ap.add_argument('--bar-label', choices=['fail', 'unsafe', 'fail_belly', 'unsafe_belly'], default='fail',
                    help="declared bar label: 'fail' = PLAN 4.3(b) goal reached; 'unsafe_belly' = REVIEW_R1 S4 reached safely, belly = failure")
    ap.add_argument('--rule', choices=['plan', 'r1'], default='plan', help="'plan' = (a)+(b)+(c); 'r1' = REVIEW_R1 S3: (b)+(c), (a) separate")
    ap.add_argument('--clearly-above-bound', choices=['group95', 'cluster95'], default='group95')
    ap.add_argument('--extra-bar', action='append', default=[], help='VEHICLE:ARM reported bar + validity (e.g. polaris_pc:polaris_grad_pc)')
    ap.add_argument('--selftest')
    a = ap.parse_args(argv)
    if a.selftest:
        sys.exit(0 if selftest(a.selftest) else 1)
    if a.write_spec:
        assert a.vehicle, '--vehicle required'
        p = Path(a.write_spec)
        assert not p.exists(), f'{p} exists: a frozen spec is never rewritten'
        sp = declared_spec(a.vehicle, a.gator_cem, a.gator_straight, a.gator_grad, a.hmmwv_grad, a.hmmwv_cem, a.hmmwv_straight,
                           a.bar_label, a.rule, a.clearly_above_bound, [tuple(x.split(':')) for x in a.extra_bar])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(sp, indent=1) + '\n')
        print(p, hashlib.sha256(p.read_bytes()).hexdigest()); return
    assert (a.index or a.reuse) and a.spec and a.out, '--index/--reuse, --spec and --out are required'
    rows, meta = load_rows(a.index, a.reuse)
    spec = json.load(open(a.spec))
    res = analyze(rows, spec, meta)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    txt = report(res)
    Path(str(a.out).rsplit('.', 1)[0] + '.txt').write_text(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
