#!/usr/bin/env python3
"""arena_gator_20260925, skeptical check of the soil results (VERIFY_soil_results.md): independent recount.

Written from scratch; imports nothing from ag_eval_index / ag_analyze / ga_analyze. Reads only:
  - the 5 soil mapping files e6/tasks/soil_eval_p{1..5}.json.mapping.json (the pre-drive plan: group -> arm -> run id),
  - the raw run folders e6/runs_soil/<run id>/{outcome.json, episode_complete.json, case.json, trajectory.npz},
  - the declared group lists (suites/*.json, case folders) and TerrainMap features (for the cluster key),
  - the rigid raw run folders e6/runs_rigid (for my own P3 / P4) and results_rigid_v1.json (their P3 / P4).
fail = outcome.json status != 'goal_reached'. unsafe (rigid) = not (goal and backward-under-throttle time < 0.05 s and
min forward speed > -0.30 m/s after frame 20), own implementation.

  PYTHONPATH=src:scripts python scripts/ag_vsr_recount.py --out artifacts/traverse/arena_gator_20260925/verify_soil_results
"""
import argparse, hashlib, json, os, sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
K3 = ROOT / 'artifacts/traverse/arena_gator_20260925'
RUNS = K3 / 'e6/runs_soil'
RRUNS = K3 / 'e6/runs_rigid'
NEAR = ['g260', 'g271', 'g251', 'g247']
SPREAD = ['g258', 'g268', 'g263', 'g241']
UNSEEN = NEAR + SPREAD


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


def exact_mcnemar_two(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(ps):
    names = sorted(ps, key=lambda k: ps[k]); m = len(names); out = {}; run = 0.0
    for i, k in enumerate(names):
        run = max(run, min(1.0, (m - i) * ps[k])); out[k] = run
    return out


# ------------------------------------------------------------------ data
def load_mappings():
    plan = {}   # (vehicle, group, arm) -> entry
    src = {}
    for i in range(1, 6):
        mp = K3 / f'e6/tasks/soil_eval_p{i}.json.mapping.json'
        m = json.load(open(mp))
        assert m['world'] == 'crm'
        for g, arms in m['groups'].items():
            for arm, e in arms.items():
                key = (e['vehicle'], g, arm)
                assert key not in plan, f'duplicate plan entry {key} ({src.get(key)} and p{i})'
                plan[key] = e; src[key] = f'p{i}'
    return plan, src


def read_run(rid):
    d = RUNS / rid
    r = dict(run_id=rid, exists=d.is_dir())
    if not r['exists']:
        return r
    o = json.load(open(d / 'outcome.json'))
    ec = json.load(open(d / 'episode_complete.json')) if (d / 'episode_complete.json').exists() else None
    c = json.load(open(d / 'case.json'))
    r.update(status=o['status'], goal=bool(o.get('goal_reached')), elapsed=float(o['elapsed_s']), case_id=o.get('case_id'),
             case_json_id=c.get('id'), vehicle=(o.get('vehicle') or {}).get('name') if isinstance(o.get('vehicle'), dict) else o.get('vehicle'),
             complete=ec is not None)
    if ec is not None:
        hs = ec.get('artifacts_sha256', {})
        r['sha_ok'] = {f: (sha(d / f) == hs[f]) for f in ('outcome.json', 'trajectory.npz', 'case.json') if f in hs and (d / f).exists()}
        r['ec_status_ok'] = ec.get('status') == o['status']
    return r


def cluster_of(case_path, cache={}):
    from nedm.traverse.terrain import TerrainMap
    c = json.load(open(case_path))
    a = c['arena']
    if a not in cache:
        cache[a] = np.array([[f['x_m'], f['y_m']] for f in TerrainMap.from_dir(ROOT / a).features], float)
    mid = (np.asarray(c['layout']['start_xy'], float) + np.asarray(c['goal_xy'], float)) / 2
    return int(np.argmin(((cache[a] - mid) ** 2).sum(1)))


def local_case(p):
    return ROOT / p[4:] if p.startswith('ext/') else K3 / p


# ------------------------------------------------------------------ statistics (own implementation)
class Stats:
    def __init__(self, seed, B):
        self.rng = np.random.default_rng(seed); self.B = B

    def paired(self, xt, xr, cl):
        xt = np.asarray(xt, float); xr = np.asarray(xr, float); n = len(xt)
        diff = 100 * (xt.mean() - xr.mean())
        dg = 100 * (xt - xr)
        gi = self.rng.integers(0, n, (self.B, n))
        gb = dg[gi].mean(1)
        labs, inv = np.unique(np.asarray(cl), return_inverse=True); K = len(labs)
        sd = np.zeros(K); cn = np.zeros(K)
        np.add.at(sd, inv, dg); np.add.at(cn, inv, 1)
        ci = self.rng.integers(0, K, (self.B, K))
        cb = sd[ci].sum(1) / cn[ci].sum(1)
        b = int(((xt < xr)).sum()); w = int(((xt > xr)).sum())
        return dict(n=n, rate_test=100 * xt.mean(), rate_ref=100 * xr.mean(), diff=diff,
                    group90=[float(np.quantile(gb, .05)), float(np.quantile(gb, .95))],
                    group95=[float(np.quantile(gb, .025)), float(np.quantile(gb, .975))],
                    cl90=[float(np.quantile(cb, .05)), float(np.quantile(cb, .95))],
                    cl95=[float(np.quantile(cb, .025)), float(np.quantile(cb, .975))],
                    p_cl=float((1 + (cb >= 0).sum()) / (self.B + 1)), p_group=float((1 + (gb >= 0).sum()) / (self.B + 1)),
                    clusters=int(K), better=b, worse=w, mcnemar2=exact_mcnemar_two(b, w) if set(np.unique(xt)) <= {0, 1} and set(np.unique(xr)) <= {0, 1} else None,
                    upper95_cl=float(np.quantile(cb, .95)), upper95_group=float(np.quantile(gb, .95)), _cb=cb)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--boot', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=20260926)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    R = dict(inputs={})
    plan, src = load_mappings()
    R['plan_entries'] = len(plan)
    R['plan_by_arm'] = dict(Counter(f'{v}|{arm}' for (v, g, arm) in plan))
    # ---------------- run folders
    rids = sorted({e['run_id'] for e in plan.values()})
    runs = {r: read_run(r) for r in rids}
    miss = [r for r, x in runs.items() if not x['exists']]
    incomplete = [r for r, x in runs.items() if x['exists'] and not x['complete']]
    badsha = [r for r, x in runs.items() if x.get('sha_ok') and not all(x['sha_ok'].values())]
    nosha = [r for r, x in runs.items() if x['exists'] and x['complete'] and len(x.get('sha_ok', {})) < 3]
    ecst = [r for r, x in runs.items() if x['exists'] and x['complete'] and not x.get('ec_status_ok')]
    goal_mismatch = [r for r, x in runs.items() if x['exists'] and x['goal'] != (x['status'] == 'goal_reached')]
    R['runs'] = dict(distinct=len(rids), missing=len(miss), incomplete=len(incomplete), sha_mismatch=len(badsha), sha_not_listed=len(nosha),
                     episode_complete_status_mismatch=len(ecst), goal_flag_vs_status_mismatch=len(goal_mismatch),
                     local_folders=len(os.listdir(RUNS)), statuses=dict(Counter(x.get('status') for x in runs.values())))
    # plan vs run consistency: case id, vehicle block, case file content
    wrong_case, wrong_vehicle, case_bytes = [], [], []
    for (v, g, arm), e in plan.items():
        x = runs[e['run_id']]
        if x['case_id'] != g or x['case_json_id'] != g:
            wrong_case.append((g, arm, e['run_id']))
        if (x['vehicle'] == 'gator') != (v == 'gator'):
            wrong_vehicle.append((g, arm, e['run_id'], x['vehicle']))
    # case.json bytes in the run == the planned case file (content equality of the json objects)
    seen = set()
    for (v, g, arm), e in plan.items():
        rid = e['run_id']
        if rid in seen:
            continue
        seen.add(rid)
        cj = json.load(open(RUNS / rid / 'case.json')); pj = json.load(open(local_case(e['case'])))
        if cj != pj:
            case_bytes.append(rid)
    R['plan_vs_run'] = dict(case_id_mismatch=len(wrong_case), vehicle_mismatch=len(wrong_vehicle), case_content_mismatch=len(case_bytes),
                            examples=dict(case=wrong_case[:5], vehicle=wrong_vehicle[:5], content=case_bytes[:5]))
    # arms sharing a drive: identical route content
    by_rid = defaultdict(list)
    for key, e in plan.items():
        by_rid[e['run_id']].append((key, e))
    shared = {r: v for r, v in by_rid.items() if len(v) > 1}
    shared_bad = [r for r, v in shared.items() if len({e['route_sha256'] for _, e in v}) != 1 or len({k[0] for k, _ in v}) != 1 or len({k[1] for k, _ in v}) != 1]
    n_reused = sum(1 for e in plan.values() if e.get('reused_from'))
    R['sharing'] = dict(arm_results=len(plan), distinct_drives=len(rids), drives_shared_by_several_arms=len(shared),
                        arm_results_in_shared_drives=sum(len(v) for v in shared.values()), shared_with_different_route_or_group_or_vehicle=len(shared_bad),
                        arm_results_reusing_soil_v2=n_reused, reused_from=dict(Counter(e.get('reused_from') for e in plan.values() if e.get('reused_from'))))
    # ---------------- declared group sets (recomputed from the case folders)
    groups_by = defaultdict(set)
    for (v, g, arm) in plan:
        groups_by[arm + '@' + v].add(g)
    unseen_decl = {}
    for ar in UNSEEN:
        ids = sorted(p.stem for p in (K3 / f'cases/test_{ar}/cases').glob(f'{ar}_test_group_*.json'))
        assert len(ids) == 250, (ar, len(ids))
        unseen_decl[ar] = sorted(ids, key=md5)[:125]
    sub = json.load(open(K3 / 'suites/soil_unseen_subset.json'))
    R['declared_sets'] = dict(unseen_subset_file_equals_recomputed=sorted(sum(unseen_decl.values(), [])) == sorted(sub['groups']))
    pair_dir = ROOT / 'artifacts/traverse/generalist_20260921/cases/pair_v1/cases'
    pool = []
    for p in sorted(pair_dir.glob('f104_pair_group_*.json')):
        c = json.load(open(p))
        if c.get('evaluation_stratum') in ('hill_cross_slope', 'hill_entry_cross_exit', 'crater_cross_slope', 'crater_entry_cross_exit'):
            pool.append(p.stem)
    indist = sorted(pool, key=md5)[:200]
    R['declared_sets']['indist_pool'] = len(pool)
    R['declared_sets']['indist_file_equals_recomputed'] = sorted(indist) == sorted(json.load(open(K3 / 'suites/f104_indist_200.json'))['groups'])
    suite800 = sorted(p.stem for p in (ROOT / 'artifacts/traverse/generalist_20260921/A_adapt/suite/cases').glob('f104_*_group_*.json'))
    held = {ar: sorted(p.stem for p in (K3 / f'cases/heldout_{ar}/cases').glob(f'{ar}_heldout_group_*.json')) for ar in ('g203', 'g228')}
    U = sorted(sum(unseen_decl.values(), []))
    want = {  # arm@vehicle -> declared group set
        **{f'{m}@hmmwv': set(U) for m in ('M1b_free', 'M3b_free', 'M2_free')},
        'M1a_free@hmmwv': set(U) | set(suite800) | set(held['g203']) | set(held['g228']),
        'M3a_free@hmmwv': set(U) | set(indist) | set(held['g203']) | set(held['g228']),
        'A3_free@hmmwv': set(U) | set(indist) | set(held['g203']) | set(held['g228']),
        'straight6@hmmwv': set(U) | set(suite800) | set(held['g203']) | set(held['g228']),
        'G_free_gator@gator': set(suite800), 'H_free_gator@gator': set(suite800), 'straight6_gator@gator': set(suite800)}
    R['declared_sets']['arms'] = {k: dict(planned=len(groups_by.get(k, ())), declared=len(w), equal=groups_by.get(k, set()) == w) for k, w in want.items()}
    R['declared_sets']['extra_arms'] = sorted(set(groups_by) - set(want))
    R['declared_sets']['suite800'] = len(suite800); R['declared_sets']['heldout'] = {k: len(v) for k, v in held.items()}
    # ---------------- outcomes table
    fail = {}
    for (v, g, arm), e in plan.items():
        fail[(v, g, arm)] = int(runs[e['run_id']]['status'] != 'goal_reached')
    el = {(v, g, arm): runs[e['run_id']]['elapsed'] for (v, g, arm), e in plan.items()}
    arena_of = {g: e['arena'] for (v, g, arm), e in plan.items()}
    # clusters (own nearest-feature computation) + compare with the index
    cl = {}
    for (v, g, arm), e in plan.items():
        if g not in cl:
            cl[g] = f"{e['arena']}:{cluster_of(local_case(e['case']))}"
    idx = json.load(open(K3 / 'e6/index/soil_eval_v1.json'))
    idx_cl = {r['group']: r['cluster'] for r in idx['rows']}
    idx_fail = {(r['vehicle'], r['group'], r['arm']): r.get('fail') for r in idx['rows']}
    R['index_compare'] = dict(index_rows=len(idx['rows']), index_missing=sum(1 for r in idx['rows'] if r.get('missing')),
                              cluster_mismatch=sum(1 for g in cl if idx_cl.get(g) != cl[g]),
                              fail_mismatch=sum(1 for k in fail if idx_fail.get(k) != fail[k]), keys_only_in_index=len(set(idx_fail) - set(fail)),
                              keys_only_in_plan=len(set(fail) - set(idx_fail)))

    def F(v, g, arm):
        if arm == 'M1':
            return (fail[(v, g, 'M1a_free')] + fail[(v, g, 'M1b_free')]) / 2
        if arm == 'M3':
            return (fail[(v, g, 'M3a_free')] + fail[(v, g, 'M3b_free')]) / 2
        return fail[(v, g, arm)]

    def rate(v, G, arm):
        return 100 * float(np.mean([F(v, g, arm) for g in G]))
    arms_u = ['M1a_free', 'M1b_free', 'M1', 'M2_free', 'M3a_free', 'M3b_free', 'M3', 'A3_free', 'straight6']
    rates = {}
    for arm in arms_u:
        rates[arm] = dict(unseen=rate('hmmwv', U, arm), near=rate('hmmwv', sum((unseen_decl[x] for x in NEAR), []), arm),
                          spread=rate('hmmwv', sum((unseen_decl[x] for x in SPREAD), []), arm),
                          per_arena={ar: rate('hmmwv', unseen_decl[ar], arm) for ar in UNSEEN})
        rates[arm]['goal_reached_unseen'] = 100 - rates[arm]['unseen']
    for arm in ('M1a_free', 'M3a_free', 'A3_free', 'straight6'):
        rates[arm]['indist_f104'] = rate('hmmwv', indist, arm)
        rates[arm]['heldout_g203'] = rate('hmmwv', held['g203'], arm); rates[arm]['heldout_g228'] = rate('hmmwv', held['g228'], arm)
        rates[arm]['heldout'] = rate('hmmwv', held['g203'] + held['g228'], arm)
    for arm in ('M1a_free', 'straight6'):
        rates[arm]['f104_800_hmmwv'] = rate('hmmwv', suite800, arm)
    for arm in ('G_free_gator', 'H_free_gator', 'straight6_gator'):
        rates[arm] = dict(f104_800_gator=rate('gator', suite800, arm), indist_gator=rate('gator', indist, arm),
                          status=dict(Counter(runs[plan[('gator', g, arm)]['run_id']]['status'] for g in suite800)))
    R['rates_fail_pct'] = rates
    # ---------------- contrasts
    S = Stats(a.seed, a.boot)
    C = {}

    def con(name, vt, xt, vr, xr, G):
        G = list(G)
        r = S.paired([F(vt, g, xt) for g in G], [F(vr, g, xr) for g in G], [cl[g] for g in G])
        C[name] = r
        return r
    P1 = con('P1_M3_vs_M1', 'hmmwv', 'M3', 'hmmwv', 'M1', U)
    P2 = con('P2_A3_vs_M1', 'hmmwv', 'A3_free', 'hmmwv', 'M1', U)
    for t, rf in (('M3a_free', 'M1a_free'), ('M3a_free', 'M1b_free'), ('M3b_free', 'M1a_free'), ('M3b_free', 'M1b_free'),
                  ('A3_free', 'M1a_free'), ('A3_free', 'M1b_free'), ('M1a_free', 'M1b_free'), ('M3a_free', 'M3b_free'),
                  ('M2_free', 'M1'), ('A3_free', 'M3'), ('M3', 'M2_free'), ('M1', 'straight6'), ('A3_free', 'straight6')):
        con(f'{t}_vs_{rf}', 'hmmwv', t, 'hmmwv', rf, U)
    for role, ars in (('near', NEAR), ('spread', SPREAD)):
        Gr = sum((unseen_decl[x] for x in ars), [])
        con(f'{role}_M3_vs_M1', 'hmmwv', 'M3', 'hmmwv', 'M1', Gr); con(f'{role}_A3_vs_M1', 'hmmwv', 'A3_free', 'hmmwv', 'M1', Gr)
    for p in ('M3', 'A3'):
        dn = C[f'near_{p}_vs_M1']['_cb']; ds = C[f'spread_{p}_vs_M1']['_cb']
        C[f'near_minus_spread_{p}'] = dict(diff=C[f'near_{p}_vs_M1']['diff'] - C[f'spread_{p}_vs_M1']['diff'],
                                           ci95=[float(np.quantile(dn - ds, .025)), float(np.quantile(dn - ds, .975))])
    per = {}
    for ar in UNSEEN:
        Ga = unseen_decl[ar]
        per[ar] = dict(P1=S.paired([F('hmmwv', g, 'M3') for g in Ga], [F('hmmwv', g, 'M1') for g in Ga], [cl[g] for g in Ga]),
                       P2=S.paired([F('hmmwv', g, 'A3_free') for g in Ga], [F('hmmwv', g, 'M1') for g in Ga], [cl[g] for g in Ga]),
                       seed_M1=100 * (np.mean([F('hmmwv', g, 'M1a_free') for g in Ga]) - np.mean([F('hmmwv', g, 'M1b_free') for g in Ga])))
    R['per_arena_signs'] = dict(P1_better=sum(per[x]['P1']['diff'] < 0 for x in UNSEEN), P2_better=sum(per[x]['P2']['diff'] < 0 for x in UNSEEN),
                                P1_ci95_excl0=[x for x in UNSEEN if per[x]['P1']['group95'][1] < 0], P2_ci95_excl0=[x for x in UNSEEN if per[x]['P2']['group95'][1] < 0],
                                seed_M1_range=[min(per[x]['seed_M1'] for x in UNSEEN), max(per[x]['seed_M1'] for x in UNSEEN)])
    R['per_arena'] = {ar: dict(P1=round(per[ar]['P1']['diff'], 2), P1_g95=[round(v, 1) for v in per[ar]['P1']['group95']],
                               P2=round(per[ar]['P2']['diff'], 2), P2_g95=[round(v, 1) for v in per[ar]['P2']['group95']], seed_M1=round(per[ar]['seed_M1'], 2)) for ar in UNSEEN}
    # distance correlation (per-arena effect vs distance to the nearest training arena, map error)
    me = json.load(open(K3 / 'arenas/map_lookup_error.json'))['arenas']
    dist = [me[ar]['distance_nearest_training'] for ar in UNSEEN]; merr = [me[ar]['all']['rmse'] for ar in UNSEEN]
    R['per_arena_corr'] = dict(P1_vs_dist=float(np.corrcoef(dist, [per[x]['P1']['diff'] for x in UNSEEN])[0, 1]),
                               P2_vs_dist=float(np.corrcoef(dist, [per[x]['P2']['diff'] for x in UNSEEN])[0, 1]),
                               P1_vs_maperr=float(np.corrcoef(merr, [per[x]['P1']['diff'] for x in UNSEEN])[0, 1]),
                               P2_vs_maperr=float(np.corrcoef(merr, [per[x]['P2']['diff'] for x in UNSEEN])[0, 1]))
    # in-distribution / held-out / no-harm
    con('noharm_M3a_vs_M1a_f104', 'hmmwv', 'M3a_free', 'hmmwv', 'M1a_free', indist)
    con('noharm_A3_vs_M1a_f104', 'hmmwv', 'A3_free', 'hmmwv', 'M1a_free', indist)
    con('inarena_A3_vs_M1a', 'hmmwv', 'A3_free', 'hmmwv', 'M1a_free', held['g203'] + held['g228'])
    con('inarena_M3a_vs_M1a', 'hmmwv', 'M3a_free', 'hmmwv', 'M1a_free', held['g203'] + held['g228'])
    # task B
    con('B_G_vs_H_gator', 'gator', 'G_free_gator', 'gator', 'H_free_gator', suite800)
    con('B_G_vs_straight6_gator', 'gator', 'G_free_gator', 'gator', 'straight6_gator', suite800)
    con('B_H_vs_straight6_gator', 'gator', 'H_free_gator', 'gator', 'straight6_gator', suite800)
    con('B_Hgator_vs_Hhmmwv', 'gator', 'H_free_gator', 'hmmwv', 'M1a_free', suite800)
    con('B_S6gator_vs_S6hmmwv', 'gator', 'straight6_gator', 'hmmwv', 'straight6', suite800)
    con('B_H_vs_straight6_hmmwv', 'hmmwv', 'M1a_free', 'hmmwv', 'straight6', suite800)
    B = dict(identical_picks_H_gator_vs_H_hmmwv=sum(plan[('gator', g, 'H_free_gator')]['route_sha256'] == plan[('hmmwv', g, 'M1a_free')]['route_sha256'] for g in suite800),
             identical_straight=sum(plan[('gator', g, 'straight6_gator')]['route_sha256'] == plan[('hmmwv', g, 'straight6')]['route_sha256'] for g in suite800))
    fr = lambda arm, v, G: np.mean([F(v, g, arm) for g in G])
    B['headroom'] = dict(G_gator=1 - fr('G_free_gator', 'gator', suite800) / fr('straight6_gator', 'gator', suite800),
                         H_gator=1 - fr('H_free_gator', 'gator', suite800) / fr('straight6_gator', 'gator', suite800),
                         H_hmmwv=1 - fr('M1a_free', 'hmmwv', suite800) / fr('straight6', 'hmmwv', suite800))
    rest = sorted(set(suite800) - set(indist))
    B['split'] = dict(indist_G=100 * fr('G_free_gator', 'gator', indist), indist_H=100 * fr('H_free_gator', 'gator', indist),
                      rest_G=100 * fr('G_free_gator', 'gator', rest), rest_H=100 * fr('H_free_gator', 'gator', rest))
    js = [g for g in suite800 if not F('gator', g, 'G_free_gator') and not F('gator', g, 'H_free_gator')]
    B['time_ratio_joint'] = dict(n=len(js), median_ratio=float(np.median([el[('gator', g, 'G_free_gator')] / el[('gator', g, 'H_free_gator')] for g in js])),
                                 median_G=float(np.median([el[('gator', g, 'G_free_gator')] for g in js])), median_H=float(np.median([el[('gator', g, 'H_free_gator')] for g in js])))
    R['taskB'] = B
    # gaps (unseen - in distribution), raw
    R['gaps_raw'] = {m: dict(unseen=rates[m]['unseen'], indist=rates[m]['indist_f104'], gap=rates[m]['unseen'] - rates[m]['indist_f104'],
                             vs_heldout=rates[m]['unseen'] - rates[m]['heldout']) for m in ('M1a_free', 'M3a_free', 'A3_free', 'straight6')}
    # dose slope M1 -> M2 -> M3 (OLS on 1, 2, 3 of the three rates)
    r3 = [rates['M1']['unseen'], rates['M2_free']['unseen'], rates['M3']['unseen']]
    R['dose_slope_pts_per_arena'] = float(np.polyfit([1, 2, 3], r3, 1)[0])
    # ---------------- Holm family: their rigid p + my own rigid P3 / P4 from the raw rigid runs
    rr = json.load(open(K3 / 'e6/analysis/results_rigid_v1.json'))
    rp = {f['name']: f['result'].get('cluster', {}).get('p_one_sided') for f in rr['family'] if f['world'] == 'rigid'}
    R['rigid_results_sha256'] = sha(K3 / 'e6/analysis/results_rigid_v1.json')
    rig = rigid_family(S)
    R['rigid_own'] = {k: {kk: vv for kk, vv in v.items() if kk != '_cb'} for k, v in rig.items()}
    fam_theirs = {'P1': P1['p_cl'], 'P2': P2['p_cl'], 'P3': rp['P3_rigid_fx2_M3_vs_M1'], 'P4': rp['P4_rigid_fx2_A3_vs_M1']}
    fam_own = {'P1': P1['p_cl'], 'P2': P2['p_cl'], 'P3': rig['P3']['p_cl'], 'P4': rig['P4']['p_cl']}
    fam_soilonly = {'P1': P1['p_cl'], 'P2': P2['p_cl'], 'P3': 1.0, 'P4': 1.0}
    R['holm'] = dict(p_with_their_rigid=fam_theirs, holm_with_their_rigid=holm(fam_theirs), p_with_own_rigid=fam_own, holm_with_own_rigid=holm(fam_own),
                     holm_soil_only_rigid_p1=holm(fam_soilonly))
    fam_file = json.load(open(K3 / 'e6/analysis/family_v1_S2.json'))
    R['family_file'] = dict(sha256=sha(K3 / 'e6/analysis/family_v1_S2.json'), same_as_E6b=sha(K3 / 'e6/analysis/family_v1_S2.json') == sha(K3 / 'e6/analysis/family_final_E6b.json'),
                            content=json.dumps(fam_file, default=str)[:1500])
    # training-noise sensitivity arithmetic (normal approximation, as the supplement)
    from math import erf, sqrt
    Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
    d1 = C['M1a_free_vs_M1b_free']['diff']; d3 = C['M3a_free_vs_M3b_free']['diff']
    sig = sqrt((d1 ** 2 + d3 ** 2) / 2) / sqrt(2)
    sens = {}
    for tag, s2 in (('estimate', sig ** 2), ('variance_doubled', 2 * sig ** 2), ('sd_doubled_1.1', (2 * sig) ** 2)):
        o = {}
        for nm, P, k in (('P1', P1, 1.0), ('P2', P2, 1.5)):
            sdb = float(np.std(P['_cb'], ddof=1)); tot = sqrt(sdb ** 2 + k * s2)
            o[nm] = dict(sd_boot=sdb, sd_total=tot, p=1 - Phi(-P['diff'] / tot))
        o['holm'] = holm({'P1': o['P1']['p'], 'P2': o['P2']['p'], 'P3': fam_theirs['P3'], 'P4': fam_theirs['P4']})
        o['sigma_per_ensemble'] = sqrt(s2)
        sens[tag] = o
    R['noise_sensitivity'] = sens
    R['contrasts'] = {k: {kk: (round(vv, 4) if isinstance(vv, float) else ([round(x, 3) for x in vv] if isinstance(vv, list) else vv)) for kk, vv in v.items() if kk != '_cb'} for k, v in C.items()}
    R['bootstrap'] = dict(B=a.boot, seed=a.seed, estimator='cluster: resample (arena, nearest feature) clusters with replacement, diff = sum of paired differences / groups drawn')
    json.dump(R, open(out / 'recount.json', 'w'), indent=1, default=float)
    # short text
    L = []
    L.append(f"runs: {R['runs']}")
    L.append(f"plan_vs_run: {R['plan_vs_run']}")
    L.append(f"sharing: {R['sharing']}")
    L.append(f"declared sets: {json.dumps(R['declared_sets'])}")
    L.append(f"index compare: {R['index_compare']}")
    for arm in arms_u:
        L.append(f"rate {arm:10s} unseen {rates[arm]['unseen']:.2f} goal {rates[arm]['goal_reached_unseen']:.2f} near {rates[arm]['near']:.1f} spread {rates[arm]['spread']:.1f} " +
                 ' '.join(f"{x} {rates[arm]['per_arena'][x]:.1f}" for x in UNSEEN) + (f" indist {rates[arm].get('indist_f104', float('nan')):.1f} held {rates[arm].get('heldout_g203', float('nan')):.1f}/{rates[arm].get('heldout_g228', float('nan')):.1f}" if 'indist_f104' in rates[arm] else ''))
    for arm in ('G_free_gator', 'H_free_gator', 'straight6_gator'):
        L.append(f"rate {arm} gator f104_800 {rates[arm]['f104_800_gator']:.2f} statuses {rates[arm]['status']}")
    L.append(f"rate M1a_free hmmwv f104_800 {rates['M1a_free']['f104_800_hmmwv']:.2f}; straight6 hmmwv {rates['straight6']['f104_800_hmmwv']:.2f}")
    for k, v in C.items():
        if 'diff' in v and 'cl90' in v:
            L.append(f"{k:28s} {v['rate_test']:.2f} vs {v['rate_ref']:.2f} diff {v['diff']:+.2f} cl90 [{v['cl90'][0]:+.2f}, {v['cl90'][1]:+.2f}] cl95 [{v['cl95'][0]:+.2f}, {v['cl95'][1]:+.2f}] g95 [{v['group95'][0]:+.2f}, {v['group95'][1]:+.2f}] p_cl {v['p_cl']:.4f} K {v['clusters']} better/worse {v['better']}/{v['worse']} mcn {v['mcnemar2']} up95cl {v['upper95_cl']:+.2f} up95g {v['upper95_group']:+.2f}")
        else:
            L.append(f"{k}: {v}")
    L.append(f"per arena: {json.dumps(R['per_arena'])}")
    L.append(f"per arena signs: {R['per_arena_signs']}; corr {R['per_arena_corr']}")
    L.append(f"task B: {json.dumps(B)}")
    L.append(f"gaps: {json.dumps(R['gaps_raw'])}; dose slope {R['dose_slope_pts_per_arena']:.3f}")
    L.append(f"rigid own: {json.dumps(R['rigid_own'])}")
    L.append(f"holm: {json.dumps(R['holm'])}")
    L.append(f"family file: same as E6b {R['family_file']['same_as_E6b']}")
    L.append(f"noise sensitivity: {json.dumps(sens)}")
    open(out / 'recount.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


def rigid_family(S):
    """P3 / P4 from the raw rigid run folders: unsafe on the 2,000 unseen groups, M1 = mean(M1a, M1b), M3 = mean(M3a, M3b)."""
    m = json.load(open(K3 / 'e6/tasks/rigid_eval_unseen.json.mapping.json'))
    arms = ('M1a_fx2', 'M1b_fx2', 'M3a_fx2', 'M3b_fx2', 'A3_fx2')
    lab, clg = {}, {}
    for g, aa in m['groups'].items():
        for arm in arms:
            e = aa[arm]; d = RRUNS / e['run_id']
            o = json.load(open(d / 'outcome.json')); z = np.load(d / 'trajectory.npz')
            vx = z['state'][20:, 0]; thr = z['action'][20:, 1]
            goal = o['status'] == 'goal_reached'
            if len(z['state']) < 22:
                uns = 1
            else:
                back = float(((vx < -0.10) & (thr > 0.3)).sum()) * 0.05
                uns = int(not (goal and back < 0.05 and vx.min() > -0.30))
            lab[(g, arm)] = uns
        clg[g] = f"{aa['M1a_fx2']['arena']}:{cluster_of(local_case(aa['M1a_fx2']['case']))}"
    G = sorted(m['groups'])
    M1 = [(lab[(g, 'M1a_fx2')] + lab[(g, 'M1b_fx2')]) / 2 for g in G]
    M3 = [(lab[(g, 'M3a_fx2')] + lab[(g, 'M3b_fx2')]) / 2 for g in G]
    A3 = [lab[(g, 'A3_fx2')] for g in G]
    return dict(P3=S.paired(M3, M1, [clg[g] for g in G]), P4=S.paired(A3, M1, [clg[g] for g in G]))


if __name__ == '__main__':
    main()
