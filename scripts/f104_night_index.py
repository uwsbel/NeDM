"""Night session step 1: index every driven f104 route, locate the first danger event on
the route (the user's 'clip at first negative vx'), and run data-quality diagnostics.

Event = first of (rollback under throttle, sustained near-stop) after the 1 s settle,
or the terminal frame for failures with neither. Its route location is the high-water
mark of progress reached before the event, projected onto the planned reference.
unsafe = NOT (goal_reached and no rollback and min vx > -0.30).
Action columns are [steering, throttle, braking]. Before 2026-09-11 this script read column 0 (steering) as
throttle; episodes_v0_steering_col.json is that version (the deployed H1/A1 models were trained from it).
"""
import json, glob, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
OUT = ROOT + '/night_v1'
DT = 0.05
SETTLE_S = 1.0


def load_route(src):
    if src.endswith('.npz'):
        z = np.load(src)
        return np.asarray(z['reference_waypoints'], float), np.asarray(z['reference_speeds'], float)
    r = json.load(open(src))
    return np.asarray(r['waypoints'], float), np.asarray(r['speeds'], float)


def project(pts, wp):
    """arc length of the closest point on the polyline wp for each pt, and lateral distance."""
    seg = wp[1:] - wp[:-1]; L2 = np.maximum((seg ** 2).sum(1), 1e-12)
    cum = np.r_[0.0, np.cumsum(np.sqrt(L2))]
    s_out = np.empty(len(pts)); d_out = np.empty(len(pts))
    for i, p in enumerate(pts):
        t = np.clip(((p - wp[:-1]) * seg).sum(1) / L2, 0, 1)
        proj = wp[:-1] + t[:, None] * seg
        d = np.linalg.norm(proj - p, axis=1); j = int(np.argmin(d))
        s_out[i] = cum[j] + t[j] * np.sqrt(L2[j]); d_out[i] = d[j]
    return s_out, d_out, cum[-1]


def first_run(mask, minlen, start):
    run = 0
    for i in range(start, len(mask)):
        run = run + 1 if mask[i] else 0
        if run >= minlen:
            return i - minlen + 1
    return None


def one(rec):
    try:
        z = np.load(rec['traj'])
        o = json.load(open(rec['outcome']))
        wp, sp = load_route(rec['route'])
    except Exception as e:
        return dict(rec, error=str(e))
    vx = z['state'][:, 0].astype(float); thr = z['action'][:, 1].astype(float)
    pose = z['pose'][:, :2].astype(float); n = len(vx); s0 = int(SETTLE_S / DT)
    status = o['status']; fail = status != 'goal_reached'
    back = (vx < -0.10) & (thr > 0.3)
    rb = first_run(back | (vx < -0.30), 1, s0)                    # first backward frame: identical to the unsafe test
    ns = first_run((np.abs(vx) < 0.3) & (thr > 0.3), 20, s0)      # >=1 s effortful near-stop
    cands = [x for x in (rb, ns) if x is not None]
    ev = min(cands) if cands else (n - 1 if fail else None)
    back_s = float(back[s0:].sum() * DT); min_vx = float(vx[s0:].min()) if n > s0 else float(vx.min())
    clean = (not fail) and back_s < 0.05 and min_vx > -0.30
    unsafe = not clean
    if clean:
        ev = None                     # clean runs are censored even if they crawled slowly
    s, d, L = project(pose, wp)
    if ev is None:
        hwm = float(s.max()); dev_pre = float(d.max())
    else:
        hwm = float(s[:ev + 1].max()); dev_pre = float(d[:ev + 1].max())
    return dict(rec, status=status, fail=int(fail), unsafe=int(unsafe), clean=int(clean),
                event_frame=(None if ev is None else int(ev)),
                event_kind=('rollback' if ev is not None and ev == rb else
                            'near_stop' if ev is not None and ev == ns else
                            ('terminal' if ev is not None else None)),
                event_s=hwm, route_len=float(L), event_frac=float(hwm / max(L, 1e-6)),
                back_s=back_s, min_vx=min_vx,
                settle_neg=float(vx[:s0].min()) if n > s0 else float(vx.min()),
                max_dev_pre_event=dev_pre, max_dev=float(d.max()), frames=int(n),
                mean_cmd_speed=float(sp[sp > 0.05].mean()) if (sp > 0.05).any() else 0.0)


def build_index():
    recs = []
    # --- designed routes, production ---
    for d in sorted(glob.glob(ROOT + '/production_v2/runs/*')):
        n = os.path.basename(d)
        if not os.path.exists(d + '/trajectory.npz'):
            continue
        ridx = int(n.split('_route_')[1])
        recs.append(dict(id=n, source='designed', traj=d + '/trajectory.npz', outcome=d + '/outcome.json',
                         route=d + '/command_reference.npz', group=n.split('_route_')[0],
                         profile=ridx % 4, geom=f'off{ridx // 4}', caseref=d + '/case.json'))
    # --- designed routes, prospective (driven later) ---
    tasks = {t['id']: t for t in json.load(open(ROOT + '/prospective/f104_undriven_test.json'))}
    os.makedirs(OUT + '/routes_prospective', exist_ok=True)
    for f in sorted(glob.glob(ROOT + '/traj_pull/prospective/*/trajectory.npz')):
        n = f.split('/')[-2]; t = tasks[n]; ridx = int(n.split('_route_')[1])
        rp = OUT + f'/routes_prospective/{n}.json'
        if not os.path.exists(rp):
            json.dump({k: t[k] for k in ('waypoints', 'speeds', 'stations', 'headings')}, open(rp, 'w'))
        recs.append(dict(id=n, source='designed', traj=f, outcome=ROOT + f'/prospective/outcomes/{n}/outcome.json',
                         route=rp, group=t['group_id'], profile=ridx % 4, geom=f'off{ridx // 4}'))
    # --- Chrono MPPI-variant routes (fixed speed within an arm) ---
    arms = [('fixed_speed_v1', 'fixed_speed_geom_v1', 0), ('wide_v1', 'wide_arm_v1', 0),
            ('band_v1', 'band_arm_v1', 0), ('speed_c4', 'speed_arm_c4', 1), ('speed_c6', 'speed_arm_c6', 2)]
    for pulled, local, prof in arms:
        for f in sorted(glob.glob(ROOT + f'/traj_pull/{pulled}/*/trajectory.npz')):
            n = f.split('/')[-2]; g = n.split('__')[0]
            rp = ROOT + f'/{local}/routes/{n}.json'
            if not os.path.exists(rp):
                continue
            oc = ROOT + f'/traj_pull/{pulled}/{n}/outcome.json'
            if not os.path.exists(oc):
                oc = ROOT + f'/{local}/outcomes/{n}/outcome.json'
            recs.append(dict(id=n, source=pulled, traj=f, outcome=oc, route=rp, group=g,
                             profile=prof, geom=n.split('__')[1]))
    for f in sorted(glob.glob(ROOT + '/showcase_f104_v1_group_0063/outcomes/*/trajectory.npz')):
        n = f.split('/')[-2]
        recs.append(dict(id=n, source='showcase', traj=f, outcome=os.path.dirname(f) + '/outcome.json',
                         route=ROOT + f'/showcase_f104_v1_group_0063/routes/{n}.json',
                         group='f104_v1_group_0063', profile=0, geom=n.split('__')[1]))
    return recs


def main():
    os.makedirs(OUT, exist_ok=True)
    recs = build_index()
    splits = {}
    for d in glob.glob(ROOT + '/production_v2/runs/*'):
        g = os.path.basename(d).split('_route_')[0]
        if g in splits: continue
        try: splits[g] = json.load(open(d + '/case.json'))['split']
        except Exception: pass
    for r in recs:
        r['split'] = splits.get(r['group'], '?')
    print(f'indexed {len(recs)} driven routes', flush=True)
    rows = list(ProcessPoolExecutor(14).map(one, recs, chunksize=32))
    bad = [r for r in rows if 'error' in r]
    rows = [r for r in rows if 'error' not in r]
    json.dump(rows, open(OUT + '/episodes.json', 'w'))
    print(f'labelled {len(rows)}  (errors {len(bad)})')
    if bad: print('  first error:', bad[0]['id'], bad[0]['error'])


if __name__ == '__main__':
    main()
