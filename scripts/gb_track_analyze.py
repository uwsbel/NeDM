"""Milestone B metrics and decision statistics for the tracking suite (PLAN B0 / B5).

Per run (<runs dir>/<route id>/trajectory.npz + command_reference.npz + outcome.json):
  station cross-track  for every reference station (command_reference.reference_waypoints) the distance to the driven
                       trajectory POLYLINE (point-to-segment projection over every sample-to-sample segment, vectorised;
                       fix round 1: the earlier nearest-SAMPLE distance grew with speed because samples are v * 0.05 s
                       apart), unreached stations capped at --cap-m (6 m, the off-route abort); per route the Winsorised
                       mean (5 % each side), mean, median, p95, max, and the fraction of stations within the cap (station
                       completion)
  heading error        |wrap(yaw at the nearest trajectory SAMPLE - reference heading)| over reached stations (deg)
  speed error          |vx - desired_speed_mps| per frame from frame --s0 (20 = the 1 s settle skipped by the labels)
  completion           outcome status == goal_reached (the decision metric) and station completion
  unsafe event         outcome status contains rollover / breakthrough / blockage / off_route / bounds_exit
  positive work (kJ), mean |da| per channel and L1 mean, elapsed time (s)
Arms: --arm NAME=<runs dir>[,<runs dir>...] (native_pid, held_pid, policy); routes and strata from --suite (ga_suite.py
--suite tracking) or, without it, every route present in every arm (stratum 'all'). Paired statistics per test arm
(--tests, default all but --reference) vs the reference, route-level paired bootstrap (4,000 resamples) plus a
group-clustered bootstrap as a robustness line:
  cross-track ratio  mean_routes(test) / mean_routes(reference) on the feasible stratum, one-sided 95th percentile < --xtrack-bound (0.90); median ratio reported
  completion diff    test - reference in points on every stratum, one-sided 5th percentile > --completion-margin (-3)
  unsafe-rate diff   test - reference, one-sided 95th percentile <= --unsafe-margin (+1)
  speed-error ratio  mean(test) / mean(reference), one-sided 95th percentile < --speed-bound (1.10)
Action bounds (outcome.json keys action_bounds / action_squash / policy_meta.action_*) are asserted equal across arms
whenever at least two arms of a route carry them. --truncate FRAC --truncate-arm NAME cuts that arm's trajectories at
FRAC of their frames before scoring (self-test of the cap).
  python scripts/gb_track_analyze.py --suite <tracking_suite.json> --arm native_pid=<dir> --arm held_pid=<dir> --arm policy=<dir> --reference native_pid --out results.json
"""
import argparse, json, os, sys
from math import comb
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNSAFE_SUBSTR = ('rollover', 'breakthrough', 'blockage', 'off_route', 'bounds_exit')
# every terminal status the two unmodified collectors write today (gen_collect.py:75, crm_collect.py:291-307); any other
# string is reported as unknown so a new status from the external-control collectors cannot silently count as safe
KNOWN_STATUSES = ('goal_reached', 'timeout', 'rollover', 'terrain_bounds_exit', 'prolonged_blockage_terminated', 'soil_breakthrough_terminated')
BOUND_KEYS = ('action_bounds', 'action_squash', 'action_affine', 'policy_meta')


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def winsor_mean(x, frac=0.05):
    x = np.asarray(x, float)
    if x.size == 0:
        return float('nan')
    lo, hi = np.percentile(x, 100 * frac), np.percentile(x, 100 * (1 - frac))
    return float(np.clip(x, lo, hi).mean())


def station_xtrack(stations, traj_xy, cap_m):
    """Distance from every reference station to the driven trajectory POLYLINE.
    Returns (capped distance (m,), uncapped distance (m,), index of the nearest trajectory SAMPLE (m,)). The distance is
    the point-to-segment distance minimised over all sample-to-sample segments (t = clip(((p - a) . ab) / |ab|^2, 0, 1),
    |p - (a + t ab)|), vectorised as an (m, T-1, 2) array; it is never larger than the nearest-sample distance and does
    not depend on how densely the trajectory is sampled (speed). The nearest-sample index is kept only for the heading
    lookup. With fewer than two samples the polyline is the single point."""
    stations = np.asarray(stations, float); traj_xy = np.asarray(traj_xy, float)
    dpt = np.linalg.norm(stations[:, None, :] - traj_xy[None, :, :], axis=-1)   # (m, T) station-to-sample distances
    j = dpt.argmin(1)
    if len(traj_xy) < 2:
        return np.minimum(dpt[np.arange(len(stations)), j], cap_m), dpt[np.arange(len(stations)), j], j
    a = traj_xy[:-1]; ab = traj_xy[1:] - traj_xy[:-1]                            # (T-1, 2) segment starts and vectors
    ab2 = (ab ** 2).sum(1)                                                       # (T-1,) squared segment lengths
    pa = stations[:, None, :] - a[None, :, :]                                    # (m, T-1, 2)
    t = (pa * ab[None, :, :]).sum(-1) / np.where(ab2 > 0, ab2, 1.0)             # projection parameter along each segment
    t = np.clip(np.where(ab2[None, :] > 0, t, 0.0), 0.0, 1.0)                    # zero-length segments: the start point
    dmin = np.linalg.norm(pa - t[..., None] * ab[None, :, :], axis=-1).min(1)   # (m,) min over segments
    return np.minimum(dmin, cap_m), dmin, j


def action_bounds(o):
    found = {k: o[k] for k in BOUND_KEYS if k in o}
    drv = o.get('driver') or {}
    for k in BOUND_KEYS:
        if isinstance(drv, dict) and k in drv:
            found['driver.' + k] = drv[k]
    if 'policy_meta' in found and isinstance(found['policy_meta'], dict):
        found = {**{k: v for k, v in found.items() if k != 'policy_meta'}, **{f'policy_meta.{k}': v for k, v in found['policy_meta'].items() if k.startswith('action')}}
    return found or None


def route_metrics(run_dir, cap_m=6.0, s0=20, truncate=None, winsor=0.05):
    run_dir = Path(run_dir)
    z = np.load(run_dir / 'trajectory.npz'); c = np.load(run_dir / 'command_reference.npz'); o = json.load(open(run_dir / 'outcome.json'))
    pose = np.asarray(z['pose'], float); state = np.asarray(z['state'], float); act = np.asarray(z['action'], float)
    n = len(pose)
    if truncate is not None:
        n = max(2, int(round(truncate * n))); pose, state, act = pose[:n], state[:n], act[:n]
    st = np.asarray(c['reference_waypoints'], float); hd = np.asarray(c['reference_headings'], float)
    vdes = np.asarray(c['desired_speed_mps'], float)[:n]
    xt, dmin, j = station_xtrack(st, pose[:, :2], cap_m)
    reached = dmin < cap_m
    herr = np.degrees(np.abs(wrap(pose[j, 2] - hd)))[reached]
    k0 = min(s0, max(n - 1, 0)); vx = state[:n, 0]
    m = min(len(vx), len(vdes)); serr = np.abs(vx[k0:m] - vdes[k0:m]) if m > k0 else np.zeros(0)
    da = np.abs(np.diff(act, axis=0)) if len(act) > 1 else np.zeros((0, 3))
    status = o['status']
    return dict(status=status, goal_reached=int(status == 'goal_reached'), unsafe=int(any(s in status for s in UNSAFE_SUBSTR)),
                elapsed_s=float(o['elapsed_s']), frames=int(n), positive_work_kj=o.get('positive_work_kj'),
                xtrack=dict(winsor_mean=winsor_mean(xt, winsor), mean=float(xt.mean()), median=float(np.median(xt)), p95=float(np.percentile(xt, 95)),
                            max=float(xt.max()), n_stations=int(len(st)), n_reached=int(reached.sum()), station_completion=float(reached.mean()),
                            n_capped=int((dmin >= cap_m).sum()), cap_m=cap_m),
                heading_err_deg=dict(mean=float(herr.mean()) if herr.size else float('nan'), median=float(np.median(herr)) if herr.size else float('nan')),
                speed_err_mps=dict(mean=float(serr.mean()) if serr.size else float('nan'), rms=float(np.sqrt((serr ** 2).mean())) if serr.size else float('nan'), n_frames=int(serr.size)),
                mean_abs_da=dict(steer=float(da[:, 0].mean()) if len(da) else float('nan'), throttle=float(da[:, 1].mean()) if len(da) else float('nan'),
                                 brake=float(da[:, 2].mean()) if len(da) else float('nan'), l1=float(da.mean()) if len(da) else float('nan')),
                action_bounds=action_bounds(o), truncated_frac=truncate)


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def get(M, path):
    for k in path.split('.'):
        M = M[k]
    return M


def paired(M, ids, groups, test, ref, rng, boot, bounds):
    """M[arm][id] -> metrics; ids = routes of this stratum present in both arms."""
    n = len(ids)
    if n == 0:
        return dict(n=0)
    idx = rng.integers(0, n, (boot, n))
    gid = np.array([groups[i] for i in ids]); ug = np.unique(gid); members = [np.flatnonzero(gid == g) for g in ug]
    gsel = rng.integers(0, len(ug), (boot, len(ug)))

    def arr(arm, path):
        return np.array([get(M[arm][i], path) for i in ids], float)

    def ratio(path, name, bound, upper=True):
        x, r = arr(test, path), arr(ref, path)
        ok = np.isfinite(x) & np.isfinite(r)
        if ok.sum() < 1 or r[ok].mean() == 0:
            return dict(n=int(ok.sum()))
        x, r = x[ok], r[ok]; ii = idx if ok.all() else rng.integers(0, ok.sum(), (boot, int(ok.sum())))
        b = x[ii].mean(1) / r[ii].mean(1)
        gb = [np.concatenate([members[g] for g in row]) for row in gsel] if len(ug) > 1 else None
        if gb is not None:
            okidx = np.flatnonzero(ok); pos = {v: k for k, v in enumerate(okidx)}
            gboot = []
            for row in gb:
                sel = np.array([pos[v] for v in row if v in pos])
                gboot.append(x[sel].mean() / r[sel].mean() if len(sel) else np.nan)
            gboot = np.array(gboot, float)
        p95 = float(np.percentile(b, 95)); p05 = float(np.percentile(b, 5))
        return dict(n=int(ok.sum()), point=float(x.mean() / r.mean()), median_ratio=float(np.median(x) / np.median(r)) if np.median(r) > 0 else None,
                    p95_one_sided=p95, p05_one_sided=p05, ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], bound=bound,
                    passed=bool(p95 < bound) if upper else bool(p05 > bound),
                    group_clustered_p95=float(np.nanpercentile(gboot, 95)) if gb is not None else None,
                    group_clustered_ci95=[float(np.nanpercentile(gboot, 2.5)), float(np.nanpercentile(gboot, 97.5))] if gb is not None else None,
                    mean_test=float(x.mean()), mean_ref=float(r.mean()), degenerate=bool(np.all(b == b[0])))

    def diff_pts(path, bound, lower=True):
        x, r = arr(test, path), arr(ref, path)
        d = 100 * (x[idx].mean(1) - r[idx].mean(1))
        gboot = np.array([100 * (x[np.concatenate([members[g] for g in row])].mean() - r[np.concatenate([members[g] for g in row])].mean()) for row in gsel]) if len(ug) > 1 else None
        b = int(((x == 1) & (r == 0)).sum()); c = int(((x == 0) & (r == 1)).sum())
        p95 = float(np.percentile(d, 95)); p05 = float(np.percentile(d, 5))
        return dict(n=n, rate_test=100 * float(x.mean()), rate_ref=100 * float(r.mean()), diff_pts=100 * float(x.mean() - r.mean()),
                    p05_one_sided=p05, p95_one_sided=p95, ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], bound=bound,
                    passed=bool(p05 > bound) if lower else bool(p95 <= bound), test_only=b, ref_only=c, mcnemar_p=mcnemar(b, c),
                    group_clustered_ci95=[float(np.percentile(gboot, 2.5)), float(np.percentile(gboot, 97.5))] if gboot is not None else None,
                    degenerate=bool(np.all(d == d[0])))

    return dict(n=n, n_groups=int(len(ug)),
                xtrack_ratio=ratio('xtrack.winsor_mean', 'xtrack', bounds['xtrack']),
                xtrack_median_ratio=ratio('xtrack.median', 'xtrack_median', bounds['xtrack']),
                completion_diff=diff_pts('goal_reached', bounds['completion'], lower=True),
                station_completion_ratio=ratio('xtrack.station_completion', 'station_completion', 1.0, upper=False),
                unsafe_diff=diff_pts('unsafe', bounds['unsafe'], lower=False),
                speed_err_ratio=ratio('speed_err_mps.mean', 'speed', bounds['speed']),
                heading_err_ratio=ratio('heading_err_deg.mean', 'heading', float('inf')),
                work_ratio=ratio('positive_work_kj', 'work', float('inf')),
                time_ratio=ratio('elapsed_s', 'time', float('inf')),
                mean_abs_da_ratio=ratio('mean_abs_da.l1', 'da', float('inf')))


def summarise(M, ids, arms):
    out = {}
    for arm in arms:
        v = [M[arm][i] for i in ids if i in M[arm]]
        if not v:
            out[arm] = dict(n=0); continue
        f = lambda path: np.array([get(r, path) for r in v], float)
        out[arm] = dict(n=len(v), goal_reached_pct=100 * float(f('goal_reached').mean()), unsafe_pct=100 * float(f('unsafe').mean()),
                        xtrack_winsor_mean=float(np.nanmean(f('xtrack.winsor_mean'))), xtrack_median_of_medians=float(np.nanmedian(f('xtrack.median'))),
                        xtrack_p95_mean=float(np.nanmean(f('xtrack.p95'))), station_completion=float(np.nanmean(f('xtrack.station_completion'))),
                        heading_err_deg=float(np.nanmean(f('heading_err_deg.mean'))), speed_err_mps=float(np.nanmean(f('speed_err_mps.mean'))),
                        positive_work_kj_median=float(np.nanmedian(f('positive_work_kj'))), mean_abs_da_l1=float(np.nanmean(f('mean_abs_da.l1'))),
                        elapsed_s_median=float(np.median(f('elapsed_s'))), status_counts={s: int(sum(1 for r in v if r['status'] == s)) for s in sorted({r['status'] for r in v})})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--suite', help='tracking_suite.json (routes, groups, strata)')
    ap.add_argument('--arm', action='append', required=True, help='NAME=<runs dir>[,<runs dir>...]')
    ap.add_argument('--reference', default='native_pid'); ap.add_argument('--tests', default=None, help='comma list, default: every other arm')
    ap.add_argument('--cap-m', type=float, default=6.0); ap.add_argument('--s0', type=int, default=20); ap.add_argument('--winsor', type=float, default=0.05)
    ap.add_argument('--xtrack-bound', type=float, default=0.90); ap.add_argument('--completion-margin', type=float, default=-3.0)
    ap.add_argument('--unsafe-margin', type=float, default=1.0); ap.add_argument('--speed-bound', type=float, default=1.10)
    ap.add_argument('--boot', type=int, default=4000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--truncate', type=float, default=None); ap.add_argument('--truncate-arm', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    arms = {}
    for s in a.arm:
        name, dirs = s.split('=', 1); arms[name] = [Path(d) for d in dirs.split(',') if d]
    assert a.reference in arms, a.reference
    tests = [t for t in (a.tests.split(',') if a.tests else arms) if t != a.reference]
    if a.suite:
        suite = json.load(open(a.suite)); routes = suite['routes']
        ids = [r['id'] for r in routes]; stratum = {r['id']: r['stratum'] for r in routes}; groups = {r['id']: r['group'] for r in routes}
    else:
        ids = sorted(set.intersection(*[set(p.name for d in dirs for p in d.iterdir() if p.is_dir()) for dirs in arms.values()]))
        stratum = {i: 'all' for i in ids}; groups = {i: i.rsplit('_route_', 1)[0] if '_route_' in i else i for i in ids}
    M = {arm: {} for arm in arms}; missing = {arm: [] for arm in arms}
    for arm, dirs in arms.items():
        for i in ids:
            d = next((d / i for d in dirs if (d / i / 'trajectory.npz').exists() and (d / i / 'command_reference.npz').exists() and (d / i / 'outcome.json').exists()), None)
            if d is None:
                missing[arm].append(i); continue
            M[arm][i] = route_metrics(d, a.cap_m, a.s0, a.truncate if arm == a.truncate_arm else None, a.winsor)
    # action bounds equal across arms whenever present
    checked = 0
    for i in ids:
        present = {arm: M[arm][i]['action_bounds'] for arm in arms if i in M[arm] and M[arm][i]['action_bounds']}
        if len(present) >= 2:
            checked += 1; vals = list(present.values())
            assert all(json.dumps(v, sort_keys=True) == json.dumps(vals[0], sort_keys=True) for v in vals), f'action bounds differ across arms on {i}: {present}'
    rng = np.random.default_rng(a.seed)
    bounds = dict(xtrack=a.xtrack_bound, completion=a.completion_margin, unsafe=a.unsafe_margin, speed=a.speed_bound)
    strata = sorted(set(stratum.values()))
    unknown = {arm: sorted({m['status'] for m in M[arm].values() if m['status'] not in KNOWN_STATUSES}) for arm in arms}
    unknown = {k: v for k, v in unknown.items() if v}
    if unknown:
        print(f'WARNING: status strings outside the known set {KNOWN_STATUSES}: {unknown} (counted as unsafe only if they contain one of {UNSAFE_SUBSTR})')
    out = dict(suite=a.suite, reference=a.reference, tests=tests, arms={k: [str(d) for d in v] for k, v in arms.items()}, n_routes=len(ids),
               missing={k: len(v) for k, v in missing.items()}, cap_m=a.cap_m, s0=a.s0, winsor=a.winsor, bounds=bounds, boot=a.boot, seed=a.seed,
               action_bounds_checked_routes=checked, unknown_statuses=unknown, unsafe_substrings=list(UNSAFE_SUBSTR),
               truncate=dict(frac=a.truncate, arm=a.truncate_arm) if a.truncate else None, strata={})
    for s in strata + (['all'] if len(strata) > 1 else []):
        sid = [i for i in ids if (s == 'all' or stratum[i] == s)]
        blk = dict(n_routes=len(sid), per_arm=summarise(M, sid, list(arms)), paired={})
        for t in tests:
            both = [i for i in sid if i in M[t] and i in M[a.reference]]
            blk['paired'][f'{t}:{a.reference}'] = paired(M, both, groups, t, a.reference, rng, a.boot, bounds)
        out['strata'][s] = blk
    # decision summary
    dec = {}
    for t in tests:
        k = f'{t}:{a.reference}'; feas = out['strata'].get('feasible', out['strata'].get('all', {})).get('paired', {}).get(k, {})
        dec[k] = dict(xtrack_ratio_feasible=feas.get('xtrack_ratio'),
                      completion={s: out['strata'][s]['paired'][k].get('completion_diff') for s in out['strata'] if k in out['strata'][s]['paired']},
                      unsafe={s: out['strata'][s]['paired'][k].get('unsafe_diff') for s in out['strata'] if k in out['strata'][s]['paired']},
                      speed_err_ratio_feasible=feas.get('speed_err_ratio'))
        xr, sp = dec[k]['xtrack_ratio_feasible'] or {}, dec[k]['speed_err_ratio_feasible'] or {}
        comp = [v for v in dec[k]['completion'].values() if v and v.get('n')]; uns = [v for v in dec[k]['unsafe'].values() if v and v.get('n')]
        dec[k]['pass_all'] = bool(xr.get('passed') and sp.get('passed') and comp and all(v['passed'] for v in comp) and uns and all(v['passed'] for v in uns))
    out['decision'] = dec
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(summary=out, per_route={arm: M[arm] for arm in arms}), open(a.out, 'w'), indent=1, default=float)
    print(f"{len(ids)} routes; missing {out['missing']}; action bounds checked on {checked} routes")
    for s, blk in out['strata'].items():
        print(f"  [{s}] n = {blk['n_routes']}")
        for arm, v in blk['per_arm'].items():
            if v.get('n'):
                print(f"    {arm:12s} n {v['n']:4d}  goal {v['goal_reached_pct']:5.1f}%  unsafe {v['unsafe_pct']:5.1f}%  xtrack {v['xtrack_winsor_mean']:.3f} m (med {v['xtrack_median_of_medians']:.3f}, p95 {v['xtrack_p95_mean']:.2f})  "
                      f"stations {100 * v['station_completion']:5.1f}%  heading {v['heading_err_deg']:5.1f} deg  speed err {v['speed_err_mps']:.3f} m/s  |da| {v['mean_abs_da_l1']:.4f}  W {v['positive_work_kj_median']:.0f} kJ  t {v['elapsed_s_median']:.1f} s")
        for k, p in blk['paired'].items():
            if not p.get('n'):
                continue
            xr, cd, ud, sr = p['xtrack_ratio'], p['completion_diff'], p['unsafe_diff'], p['speed_err_ratio']
            print(f"    {k}: xtrack ratio {xr.get('point', float('nan')):.4f} (p95 {xr.get('p95_one_sided', float('nan')):.4f} < {xr.get('bound')}: {'PASS' if xr.get('passed') else 'FAIL'}"
                  f"{'; degenerate' if xr.get('degenerate') else ''}; clustered p95 {xr.get('group_clustered_p95')}) median ratio {xr.get('median_ratio')}; "
                  f"completion {cd['diff_pts']:+.1f} pts (p05 {cd['p05_one_sided']:+.1f} > {cd['bound']}: {'PASS' if cd['passed'] else 'FAIL'}); "
                  f"unsafe {ud['diff_pts']:+.1f} pts (p95 {ud['p95_one_sided']:+.1f} <= {ud['bound']}: {'PASS' if ud['passed'] else 'FAIL'}); "
                  f"speed err ratio {sr.get('point', float('nan')):.4f} (p95 {sr.get('p95_one_sided', float('nan')):.4f} < {sr.get('bound')}: {'PASS' if sr.get('passed') else 'FAIL'})")
    for k, d in dec.items():
        print(f"  decision {k}: {'PASS' if d['pass_all'] else 'FAIL'}")


if __name__ == '__main__':
    main()
