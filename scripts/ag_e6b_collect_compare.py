#!/usr/bin/env python3
"""Per-route outcome records of a rigid collection, and the Gator vs HMMWV comparison on identical routes
(arena_gator_20260925, module E6b; PLAN 3 "also from the collection itself", 7.7).

  records  --runs <runs dir> [--prefix gator__] --regex <id regex> --out <records.json>
           one record per completed run: status, fail (goal not reached), unsafe (f104_n2_analyze.labels: not (goal and
           < 0.05 s rolling backwards under throttle and min forward speed > -0.30 m/s, after the 1 s settle)),
           backward_only (unsafe and not fail), back_s, min_vx, max tilt after the settle, elapsed, chassis ground contact
           (max chassis contact force in rich_intervals.npz, when the run kept it), belly (outcome vehicle.belly, Gator),
           launch / native-height checks, finite states. numpy only (runs on the cluster login node).
  compare  --gator <records> --hmmwv <records> --out <compare.json>
           pairs by route id (gator__<id> <-> <id>): rates and paired discordance per speed profile (designed route index
           % 4: constant 2 / 4 / 6 m/s, smooth 2-6-2; on-policy = the 8 planner proposals per group), McNemar, statuses,
           simulated hours, ground contact.
"""
import argparse, json, math, os, re, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np

S0, DT = 20, 0.05
PROFILES = {0: 'constant_2', 1: 'constant_4', 2: 'constant_6', 3: 'smooth_2_6_2'}


def profile(rid):
    m = re.search(r'_route_(\d{2})$', rid)
    if m:
        return PROFILES[int(m.group(1)) % 4]
    return 'on_policy' if re.search(r'_op_\d{2}$', rid) else 'other'


def record(d):
    rid = os.path.basename(d)
    try:
        o = json.load(open(os.path.join(d, 'outcome.json')))
        z = np.load(os.path.join(d, 'trajectory.npz'))
    except Exception as e:                                   # noqa: BLE001
        return dict(id=rid, ok=False, reason=f'unreadable: {e}')
    st = z['state']; vx = st[:, 0].astype(float); thr = z['action'][:, 1].astype(float)
    fail = o['status'] != 'goal_reached'
    if len(st) < 22:
        back, minvx, tilt = 0.0, 0.0, 0.0
        unsafe = True
    else:
        back = float(((vx[S0:] < -0.10) & (thr[S0:] > 0.3)).sum() * DT)
        minvx = float(vx[S0:].min())
        tilt = float(np.degrees(np.abs(st[S0:, 2:4])).max())
        unsafe = not ((not fail) and back < 0.05 and minvx > -0.30)
    fin = bool(np.isfinite(st).all() and ('terminal_state' not in z.files or np.isfinite(z['terminal_state']).all()))
    chassis = None
    ri = os.path.join(d, 'rich_intervals.npz')
    if os.path.exists(ri):
        r = np.load(ri)
        ks = [k for k in r.files if 'chassis_contact' in k and 'max' in k]
        chassis = float(r[ks[0]].max()) if ks else None
    veh = o.get('vehicle') if isinstance(o.get('vehicle'), dict) else None
    belly = (veh or {}).get('belly') or {}

    def jload(n):
        p = os.path.join(d, n)
        return json.load(open(p)) if os.path.exists(p) else {}
    return dict(id=rid, ok=True, complete=os.path.exists(os.path.join(d, 'episode_complete.json')), profile=profile(rid),
                status=o['status'], fail=int(fail), unsafe=int(unsafe), backward_only=int(unsafe and not fail), back_s=back, min_vx=minvx,
                max_tilt_deg=tilt, elapsed_s=float(o['elapsed_s']), wall_s=float(o.get('wall_s') or 0.0), finite=fin,
                chassis_contact_n=chassis, belly_min_m=belly.get('min_clearance_m'), belly_frames_below=belly.get('frames_below_surface'),
                launch_passed=jload('initial_state_validation.json').get('passed'), native_height_passed=jload('native_height_check.json').get('passed'),
                vehicle=(veh or {}).get('name') if veh else o.get('vehicle'))


def cmd_records(a):
    pat = re.compile(a.regex)
    dirs = []
    for r in a.runs:
        for e in os.scandir(r):
            if e.is_dir() and e.name.startswith(a.prefix) and pat.match(e.name[len(a.prefix):]):
                dirs.append(e.path)
    with ProcessPoolExecutor(a.workers) as ex:
        recs = list(ex.map(record, sorted(dirs), chunksize=64))
    json.dump(dict(tool='scripts/ag_e6b_collect_compare.py records', argv=sys.argv[1:], created=time.strftime('%Y-%m-%d %H:%M:%S'),
                   host=os.uname().nodename, n=len(recs), records=recs), open(a.out, 'w'))
    print(json.dumps(dict(n=len(recs), ok=sum(r['ok'] for r in recs), status=dict(Counter(r.get('status') for r in recs)))))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def cmd_compare(a):
    G = {r['id'][len('gator__'):]: r for r in json.load(open(a.gator))['records'] if r['ok']}
    H = {r['id']: r for r in json.load(open(a.hmmwv))['records'] if r['ok']}
    ids = sorted(set(G) & set(H))
    out = dict(n_gator=len(G), n_hmmwv=len(H), n_paired=len(ids), only_gator=len(set(G) - set(H)), only_hmmwv=len(set(H) - set(G)))
    res = {}
    for prof in ['all', 'constant_2', 'constant_4', 'constant_6', 'smooth_2_6_2', 'designed', 'on_policy']:
        sel = [i for i in ids if prof == 'all' or G[i]['profile'] == prof or (prof == 'designed' and G[i]['profile'] != 'on_policy')]
        if not sel:
            continue
        d = dict(n=len(sel))
        for lab in ('fail', 'unsafe', 'backward_only'):
            g = np.array([G[i][lab] for i in sel]); h = np.array([H[i][lab] for i in sel])
            gw, hw = int(((g == 1) & (h == 0)).sum()), int(((g == 0) & (h == 1)).sum())
            d[lab] = dict(gator=100 * g.mean(), hmmwv=100 * h.mean(), gator_only=gw, hmmwv_only=hw, both=int(((g == 1) & (h == 1)).sum()),
                          mcnemar_two_sided=mcnemar(gw, hw))
        d['tilt30'] = dict(gator=100 * np.mean([G[i]['max_tilt_deg'] > 30 for i in sel]), hmmwv=100 * np.mean([H[i]['max_tilt_deg'] > 30 for i in sel]))
        d['sim_hours'] = dict(gator=sum(G[i]['elapsed_s'] for i in sel) / 3600, hmmwv=sum(H[i]['elapsed_s'] for i in sel) / 3600)
        jg = [i for i in sel if G[i]['fail'] == 0 and H[i]['fail'] == 0]
        d['time_ratio_joint_goals_median'] = float(np.median([G[i]['elapsed_s'] / H[i]['elapsed_s'] for i in jg])) if jg else None
        res[prof] = d
    out['by_profile'] = res
    out['status'] = dict(gator=dict(Counter(G[i]['status'] for i in ids)), hmmwv=dict(Counter(H[i]['status'] for i in ids)))
    for name, R in (('gator', G), ('hmmwv', H)):
        rec = [R[i] for i in ids]
        ch = [r['chassis_contact_n'] for r in rec if r['chassis_contact_n'] is not None]
        out[f'{name}_checks'] = dict(launch_failed=sum(r['launch_passed'] is False for r in rec), native_height_failed=sum(r['native_height_passed'] is False for r in rec),
                                     not_finite=sum(not r['finite'] for r in rec), rollover=sum(r['status'] == 'rollover' for r in rec),
                                     chassis_contact_recorded=len(ch), chassis_contact_routes=sum(c > 0 for c in ch),
                                     chassis_contact_max_n=max(ch) if ch else None,
                                     belly_min_m=min([r['belly_min_m'] for r in rec if r['belly_min_m'] is not None], default=None),
                                     belly_below_surface_routes=sum((r['belly_frames_below'] or 0) > 0 for r in rec))
    json.dump(out, open(a.out, 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:4000])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('records'); r.add_argument('--runs', nargs='+', required=True); r.add_argument('--prefix', default='')
    r.add_argument('--regex', required=True); r.add_argument('--out', required=True); r.add_argument('--workers', type=int, default=8)
    c = sub.add_parser('compare'); c.add_argument('--gator', required=True); c.add_argument('--hmmwv', required=True); c.add_argument('--out', required=True)
    a = ap.parse_args()
    dict(records=cmd_records, compare=cmd_compare)[a.cmd](a)


if __name__ == '__main__':
    main()
