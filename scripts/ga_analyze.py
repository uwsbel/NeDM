"""Milestone A statistics: paired closed-loop comparison of planner arms on the frozen suite (PLAN Milestone A).

Arms are (picks dir, planner_arms arm letter, runs dir(s)) triples; outcomes per group come from the driven route of
that arm (trajectory.npz + outcome.json, labels = scripts/f104_n2_analyze.labels as in n2_planner_analyze.py).
  python scripts/ga_analyze.py --arm S=<picks dir>:B:<runs dir>[,<runs dir2>] --arm H=<picks dir>:B:<runs dir> \
      --primary H:S [--contrasts X:Y,...] [--label fail|unsafe] [--suite <suite.json>] [--run-index <run_index.json>] \
      [--margin-pts 3.0] [--time-bound 1.10] [--agreement H:S_own:S_other] [--routes-dir <dir>] [--cluster-ci --cases <dir> [--arena <dir>]] --out results.json
Contrast 'X:Y' follows n2_planner_analyze: rate_a is X, rate_b is Y, diff = label(X) - label(Y) in points (with --label fail this is
goal-reached(Y) - goal-reached(X)). For the decision rule write the TEST arm first and the REFERENCE second: diff = reference minus
test goal-reached, and `p95_one_sided_pts` (95th percentile of the group bootstrap of diff) must be < --margin-pts.
Added over n2_planner_analyze: the one-sided bound per stratum (fresh / reused / all, from --suite) and per evaluation_stratum,
the paired elapsed-time ratio (median over groups of t_X / t_Y on groups both reached; 95th percentile < --time-bound), pick
agreement by route GEOMETRY (fix round 1: d(X, Y) = symmetrised mean nearest-waypoint distance between the two picked routes,
read from <picks dir>/../routes/<route_id>.json or --routes-dir; with --agreement H:OWN:OTHER the paired group bootstrap of
d(H, OTHER) - d(H, OWN) must have its 5th percentile > 0, and the per-group sign fraction d(H, OWN) < d(H, OTHER) is reported;
the sha256-identical pick counts stay as an extra - CEM picks of two models almost never coincide exactly), exact McNemar,
and the terrain-feature-clustered CI of scripts/n2_cluster_ci.py as a robustness line. Run resolution: run_index[route_id].ref_run_local (reused reference drive)
else <runs dir>/<run_index driven_as> else <runs dir>/<route_id>, first runs dir that has episode_complete.json wins.
results.json = {summary: {..., 'PRIMARY_X:Y', contrasts, rates, strata, time_ratio, agreement, cluster_ci}, per_group: {g: {arm: {...}}}}.
"""
import argparse, json, os, subprocess, sys
from math import comb
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from f104_n2_analyze import labels


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def safe_labels(d):
    z = np.load(d + '/trajectory.npz')
    if len(z['state']) < 22:
        o = json.load(open(d + '/outcome.json'))
        return dict(fail=int(o['status'] != 'goal_reached'), unsafe=1, status=o['status'], elapsed=float(o['elapsed_s']), back_s=0.0, min_vx=0.0, max_tilt=0.0)
    return labels(d)


def parse_arm(spec):
    name, rest = spec.split('=', 1)
    picks, letter, runs = rest.split(':', 2)
    picks = Path(picks)
    if (picks / 'picks').is_dir():          # a planner_arms / ga_suite pass directory was given: use its picks/ subdir
        picks = picks / 'picks'
    return name, dict(picks=picks, letter=letter, runs=[Path(r) for r in runs.split(',') if r])


def resolve_run(info, arm, run_index):
    rid = info['route_id']; cands = []
    if run_index and rid in run_index:
        e = run_index[rid]
        if e.get('ref_run_local'):
            cands.append(ROOT / e['ref_run_local'])
        cands += [r / e['driven_as'] for r in arm['runs']]
    cands += [r / rid for r in arm['runs']]
    for d in cands:
        if (d / 'episode_complete.json').exists() or ((d / 'outcome.json').exists() and (d / 'trajectory.npz').exists()):
            return d
    return None


def collect(arms, run_index):
    by, missing = {}, {a: 0 for a in arms}
    for name, arm in arms.items():
        for p in sorted(arm['picks'].glob('*.json')):
            s = json.load(open(p)); info = s['arms'].get(arm['letter'])
            if not info:
                continue
            d = resolve_run(info, arm, run_index)
            if d is None:
                missing[name] += 1; continue
            o = json.load(open(d / 'outcome.json')); L = safe_labels(str(d)); L['tilt30'] = int(L['max_tilt'] > 30.0)
            L.update(work_kj=o.get('positive_work_kj'), route_id=info['route_id'], route_sha256=info.get('route_sha256'), P=info.get('P'), z=info.get('z_mean'),
                     T_cmd=info.get('T'), run_dir=os.path.relpath(d, ROOT) if str(d).startswith(str(ROOT)) else str(d))
            by.setdefault(s['group'], {})[name] = L
    return by, missing


def cmp(by, G, x, y, key, rng, boot, margin, p_seed_style=True):
    xa = np.array([by[g][x][key] for g in G], float); ya = np.array([by[g][y][key] for g in G], float)
    idx = rng.integers(0, len(G), (boot, len(G))); boots = 100 * (xa[idx].mean(1) - ya[idx].mean(1))
    b = int(((xa == 1) & (ya == 0)).sum()); c = int(((xa == 0) & (ya == 1)).sum())
    same = int(sum((by[g][x].get('route_sha256') or by[g][x]['route_id']) == (by[g][y].get('route_sha256') or by[g][y]['route_id']) for g in G))
    p95 = float(np.percentile(boots, 95)); p05 = float(np.percentile(boots, 5))
    return dict(n=len(G), rate_a=100 * xa.mean(), rate_b=100 * ya.mean(), diff=100 * (xa.mean() - ya.mean()),
                ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], a_worse=b, b_worse=c, p=mcnemar(b, c), identical_picks=same,
                p95_one_sided_pts=p95, p05_one_sided_pts=p05, margin_pts=margin, pass_noninferior=bool(p95 < margin),
                note=f'diff = {key}({x}) - {key}({y}) in points' + (f' = goal_reached({y}) - goal_reached({x})' if key == 'fail' else ''))


def time_ratio(by, G, x, y, rng, boot, bound):
    """Paired elapsed-time ratio on groups where both arms reached the goal: median over groups of t_x/t_y, and the ratio of medians."""
    both = [g for g in G if not by[g][x]['fail'] and not by[g][y]['fail']]
    if len(both) < 5:
        return dict(n=len(both), note='fewer than 5 groups with both arms at the goal')
    tx = np.array([by[g][x]['elapsed'] for g in both]); ty = np.array([by[g][y]['elapsed'] for g in both]); r = tx / ty
    idx = rng.integers(0, len(both), (boot, len(both)))
    med_boot = np.median(r[idx], 1); rom_boot = np.median(tx[idx], 1) / np.median(ty[idx], 1)
    return dict(n=len(both), median_paired_ratio=float(np.median(r)), median_paired_ratio_p95=float(np.percentile(med_boot, 95)),
                median_paired_ratio_ci95=[float(np.percentile(med_boot, 2.5)), float(np.percentile(med_boot, 97.5))],
                ratio_of_medians=float(np.median(tx) / np.median(ty)), ratio_of_medians_p95=float(np.percentile(rom_boot, 95)),
                mean_ratio=float(r.mean()), median_time_x=float(np.median(tx)), median_time_y=float(np.median(ty)), bound=bound,
                pass_time=bool(np.percentile(med_boot, 95) < bound), note=f't({x}) / t({y}); 95th percentile of the group bootstrap of the median paired ratio must be < {bound}')


def rates(by, G, arms):
    out = {}
    for arm in arms:
        v = [by[g][arm] for g in G]; t = [r['elapsed'] for r in v if not r['fail']]; w = [r['work_kj'] for r in v if not r['fail'] and r['work_kj'] is not None]
        out[arm] = dict(n=len(v), goal_reached=100 * (1 - np.mean([r['fail'] for r in v])), unsafe=100 * np.mean([r['unsafe'] for r in v]), tilt30=100 * np.mean([r['tilt30'] for r in v]),
                        median_time_s=float(np.median(t)) if t else None, median_work_kj=float(np.median(w)) if w else None,
                        mean_pred_P=float(np.mean([r['P'] for r in v if r['P'] is not None])) if any(r['P'] is not None for r in v) else None,
                        pred_below_1pct_fail=int(sum(1 for r in v if r['P'] is not None and r['P'] < 0.01 and r['fail'])),
                        status_counts={s: int(sum(1 for r in v if r['status'] == s)) for s in sorted({r['status'] for r in v})})
    return out


def block(by, G, arms, primary, contrasts, key, rng, boot, margin, bound):
    out = dict(n=len(G))
    if not G:
        return out
    x, y = primary.split(':'); out['PRIMARY_' + primary] = cmp(by, G, x, y, key, rng, boot, margin)
    out['contrasts'] = {c: cmp(by, G, *c.split(':'), key, rng, boot, margin) for c in contrasts if c}
    out['rates'] = rates(by, G, arms)
    out['time_ratio'] = {c: time_ratio(by, G, *c.split(':'), rng, boot, bound) for c in [primary] + [c for c in contrasts if c]}
    return out


def route_dist(P, Q):
    """Symmetrised mean nearest-waypoint distance between two routes (m): 0.5 * (mean_p min_q |p - q| + mean_q min_p |p - q|); 0 = same path."""
    d = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1)
    return float(0.5 * (d.min(1).mean() + d.min(0).mean()))


def pick_geometry(arm, info, routes_dirs=()):
    """Identity (sha256) and geometry (waypoints xy) of a picked route. The route file is <picks dir>/../routes/<route_id>.json
    (ga_suite / planner_arms layout) or <routes-dir>/<route_id>.json; wp is None when no file is found."""
    rid = info['route_id']; wp = None
    for c in [arm['picks'].parent / 'routes' / f'{rid}.json'] + [Path(d) / f'{rid}.json' for d in routes_dirs]:
        if c.exists():
            wp = np.asarray(json.load(open(c))['waypoints'], float)[:, :2]; break
    return dict(sha=info.get('route_sha256') or rid, route_id=rid, wp=wp)


def agreement(picks_by, arms, spec, rng, boot):
    """Pick agreement by route geometry over the groups that have picks for every arm involved. d(X, Y) = route_dist of the two
    picked routes (0 = same path). Per arm pair: median / mean / p90 of d and the number of sha256-identical picks (extra).
    With spec H:OWN:OTHER: per group delta = d(H, OTHER) - d(H, OWN); pass = the 5th percentile of the paired group bootstrap of
    mean(delta) is > 0 (H's picks lie closer to its own-domain specialist's than to the other specialist's); the sign fraction
    d(H, OWN) < d(H, OTHER) with its exact two-sided sign test, and the sha256-identical counts, are reported next to it."""
    names = list(arms); out = {'metric': 'd(X, Y) = symmetrised mean nearest-waypoint distance between the picked routes (m); identical = equal route sha256', 'pairs': {}}
    Gp = [g for g in picks_by if all(a in picks_by[g] for a in names)]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            Gg = [g for g in Gp if picks_by[g][a]['wp'] is not None and picks_by[g][b]['wp'] is not None]
            dd = np.array([route_dist(picks_by[g][a]['wp'], picks_by[g][b]['wp']) for g in Gg], float)
            out['pairs'][f'{a}={b}'] = dict(n=len(Gp), identical=int(sum(picks_by[g][a]['sha'] == picks_by[g][b]['sha'] for g in Gp)), n_geometry=len(Gg),
                                            median_dist_m=float(np.median(dd)) if dd.size else None, mean_dist_m=float(dd.mean()) if dd.size else None,
                                            p90_dist_m=float(np.percentile(dd, 90)) if dd.size else None)
    if spec:
        h, own, other = spec.split(':')
        G3 = [g for g in picks_by if all(a in picks_by[g] and picks_by[g][a]['wp'] is not None for a in (h, own, other))]
        if not G3:
            out['own_vs_other'] = dict(spec=spec, n=0, note='no group with route geometry for all three arms'); return out
        d_own = np.array([route_dist(picks_by[g][h]['wp'], picks_by[g][own]['wp']) for g in G3]); d_oth = np.array([route_dist(picks_by[g][h]['wp'], picks_by[g][other]['wp']) for g in G3])
        delta = d_oth - d_own
        idx = rng.integers(0, len(G3), (boot, len(G3))); bm = delta[idx].mean(1); p05 = float(np.percentile(bm, 5))
        closer, farther = int((delta > 0).sum()), int((delta < 0).sum())
        so = np.array([picks_by[g][h]['sha'] == picks_by[g][own]['sha'] for g in G3]); st = np.array([picks_by[g][h]['sha'] == picks_by[g][other]['sha'] for g in G3])
        out['own_vs_other'] = dict(spec=spec, n=len(G3), mean_dist_own_m=float(d_own.mean()), mean_dist_other_m=float(d_oth.mean()),
                                   median_dist_own_m=float(np.median(d_own)), median_dist_other_m=float(np.median(d_oth)),
                                   diff_m=float(delta.mean()), p05_one_sided_m=p05, ci95_m=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                                   pass_agreement=bool(p05 > 0), closer_to_own=closer, closer_to_other=farther, ties=int((delta == 0).sum()),
                                   closer_to_own_frac=float(closer / len(G3)), sign_p=mcnemar(closer, farther),
                                   identical_own=int(so.sum()), identical_other=int(st.sum()), identical_diff_pts=100 * float(so.mean() - st.mean()),
                                   note=f'diff = mean over groups of d({h}, {other}) - d({h}, {own}) in m; pass = 5th percentile of its group bootstrap > 0; sha256-identical counts are an extra')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', action='append', required=True, help='NAME=<picks dir>:<arm letter>:<runs dir>[,<runs dir>...]')
    ap.add_argument('--primary', required=True, help='TEST:REFERENCE (n2 style X:Y; diff = label(X) - label(Y))')
    ap.add_argument('--contrasts', default=''); ap.add_argument('--label', default='fail', choices=['fail', 'unsafe'])
    ap.add_argument('--suite', help='suite.json from ga_suite.py (strata fresh/reused, evaluation_stratum)')
    ap.add_argument('--run-index', help='run_index_<world>.json from ga_suite.py merge (route_id -> driven id / reference drive)')
    ap.add_argument('--margin-pts', type=float, default=3.0); ap.add_argument('--time-bound', type=float, default=1.10)
    ap.add_argument('--agreement', help='H:OWN:OTHER pick-agreement statistic (route geometry)')
    ap.add_argument('--routes-dir', action='append', default=[], help='extra dir(s) holding <route_id>.json for the geometric pick agreement (default: <picks dir>/../routes)')
    ap.add_argument('--cluster-ci', action='store_true'); ap.add_argument('--cases', help='case dir for n2_cluster_ci.py (needs <g>.json)')
    ap.add_argument('--arena', default='assets/traverse/arena_f104_50h_v1')
    ap.add_argument('--boot', type=int, default=4000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--world', default=None); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    arms = dict(parse_arm(s) for s in a.arm); names = list(arms)
    run_index = json.load(open(a.run_index)) if a.run_index else None
    by, missing = collect(arms, run_index)
    picks_by = {}
    for name, arm in arms.items():
        for p in arm['picks'].glob('*.json'):
            s = json.load(open(p)); info = s['arms'].get(arm['letter'])
            if info:
                picks_by.setdefault(s['group'], {})[name] = pick_geometry(arm, info, a.routes_dir)
    G = [g for g in by if all(x in by[g] for x in names)]
    rng = np.random.default_rng(a.seed); key = a.label
    contrasts = [c for c in a.contrasts.split(',') if c]
    out = dict(world=a.world, groups=len(by), complete=len(G), missing_drives=missing, arms=names, arm_specs={n: dict(picks=str(v['picks']), letter=v['letter'], runs=[str(r) for r in v['runs']]) for n, v in arms.items()},
               label=key, margin_pts=a.margin_pts, time_bound=a.time_bound, boot=a.boot, seed=a.seed)
    out.update(block(by, G, names, a.primary, contrasts, key, rng, a.boot, a.margin_pts, a.time_bound))
    out['strata'] = {}
    if a.suite:
        suite = json.load(open(a.suite)); strat = {r['group']: r for r in suite['groups']}
        out['suite'] = a.suite
        for s in sorted({r['stratum'] for r in strat.values()}):
            Gs = [g for g in G if g in strat and strat[g]['stratum'] == s]
            out['strata'][s] = block(by, Gs, names, a.primary, contrasts, key, rng, a.boot, a.margin_pts, a.time_bound)
        out['by_evaluation_stratum'] = {}
        for s in sorted({r['evaluation_stratum'] for r in strat.values() if r.get('evaluation_stratum')}):
            Gs = [g for g in G if g in strat and strat[g]['evaluation_stratum'] == s]
            if Gs:
                x, y = a.primary.split(':'); out['by_evaluation_stratum'][s] = cmp(by, Gs, x, y, key, rng, a.boot, a.margin_pts)
        out['groups_not_in_suite'] = int(sum(1 for g in G if g not in strat))
    out['agreement'] = agreement(picks_by, names, a.agreement, rng, a.boot)
    json.dump(dict(summary=out, per_group={g: by[g] for g in G}), open(a.out, 'w'), indent=1, default=float)
    if a.cluster_ci and G:
        if not a.cases:
            print('cluster CI skipped: --cases required')
        else:
            pairs = ','.join([a.primary] + contrasts)
            cmd = [sys.executable, str(HERE / 'n2_cluster_ci.py'), '--results', str(Path(a.out).resolve()), '--cases', a.cases, '--arena', a.arena, '--pairs', pairs, '--label', key]
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            print(r.stdout.strip())
            cc = Path(a.out).resolve().parent / f'cluster_ci_{key}.json'
            out['cluster_ci'] = json.load(open(cc)) if r.returncode == 0 and cc.exists() else dict(error=r.stderr[-2000:])
            json.dump(dict(summary=out, per_group={g: by[g] for g in G}), open(a.out, 'w'), indent=1, default=float)
    # report
    print(f"{out['complete']}/{out['groups']} groups with every arm; missing drives {missing}")
    def show(b, title):
        if not b.get('n'):
            print(f'  [{title}] no complete groups'); return
        print(f'  [{title}] n = {b["n"]}')
        for k, r in [('PRIMARY_' + a.primary, b['PRIMARY_' + a.primary])] + list(b['contrasts'].items()):
            print(f"    {k:16s} {key} {r['rate_a']:5.1f}% vs {r['rate_b']:5.1f}%  diff {r['diff']:+.1f} [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}]  "
                  f"one-sided p95 {r['p95_one_sided_pts']:+.1f} (margin {r['margin_pts']}: {'PASS' if r['pass_noninferior'] else 'FAIL'})  "
                  f"worse {r['a_worse']} vs {r['b_worse']}  p={r['p']:.4f}  identical picks {r['identical_picks']}")
        for k, t in b['time_ratio'].items():
            if 'median_paired_ratio' in t:
                print(f"    time {k:12s} median paired ratio {t['median_paired_ratio']:.3f} (p95 {t['median_paired_ratio_p95']:.3f}, bound {t['bound']}: "
                      f"{'PASS' if t['pass_time'] else 'FAIL'}); ratio of medians {t['ratio_of_medians']:.3f} (p95 {t['ratio_of_medians_p95']:.3f}); n {t['n']}")
        print('    arm: goal reached / unsafe / tilt30 / median time / median work / mean predicted P / confident failures')
        for arm, v in b['rates'].items():
            print(f"     {arm:12s} {v['goal_reached']:5.1f}%  {v['unsafe']:5.1f}%  {v['tilt30']:4.1f}%  {v['median_time_s'] or 0:5.1f} s  {v['median_work_kj'] or 0:6.0f} kJ  "
                  f"P {v['mean_pred_P'] if v['mean_pred_P'] is not None else float('nan'):.3f}  {v['pred_below_1pct_fail']}")
    show(out, 'all')
    for s, b in out['strata'].items():
        show(b, f'stratum {s}')
    if out['agreement']['pairs']:
        print('  pick agreement (median route distance / sha256-identical):', {k: (f"{v['median_dist_m']:.2f} m" if v['median_dist_m'] is not None else '-') + f" / {v['identical']}/{v['n']}" for k, v in out['agreement']['pairs'].items()})
    if out['agreement'].get('own_vs_other', {}).get('n'):
        o = out['agreement']['own_vs_other']
        print(f"  agreement {o['spec']}: d(H, own) mean {o['mean_dist_own_m']:.3f} m (median {o['median_dist_own_m']:.3f}) vs d(H, other) mean {o['mean_dist_other_m']:.3f} m (median {o['median_dist_other_m']:.3f}): "
              f"diff {o['diff_m']:+.3f} m, one-sided p05 {o['p05_one_sided_m']:+.3f} ({'PASS' if o['pass_agreement'] else 'FAIL'}); closer to own {o['closer_to_own']}/{o['n']} "
              f"({100 * o['closer_to_own_frac']:.1f} %, sign p {o['sign_p']:.4f}); sha256-identical picks own {o['identical_own']} / other {o['identical_other']}")
    if 'cluster_ci' in out and 'error' not in out['cluster_ci']:
        for k, v in out['cluster_ci'].items():
            print(f"  feature-cluster CI {k}: diff {v['diff_pts']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] ({v['clusters']} clusters), wins/losses {v['cluster_wins']}/{v['cluster_losses']}, sign p {v['sign_p']:.4f}")


if __name__ == '__main__':
    main()
