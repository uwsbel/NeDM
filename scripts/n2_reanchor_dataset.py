"""Re-anchored (moving-start) samples from recorded episodes -> station dataset with velocity context + energy/time targets.

Per episode (run dir with trajectory.npz, command_reference.npz, case.json, outcome.json): the event frame by the
labeller's rule (f104_n2_dataset.one), anchors k = 0 plus up to --anchors-1 frames chosen every 2 s before the event
(remaining route >= --min-remaining m, lateral deviation < 1 m, not parked), stratified over the anchor speed. For each
anchor: remaining route from the projection point (profile speeds), corridor over the remaining length (96 stations),
event station on the remaining part, ctx = [state[k] (17), goal - pose_k, |.|, yaw_k, L_rem], energy/time targets =
first-arrival cumulative W+ / time after the anchor (NaN beyond the furthest station reached).
  python scripts/n2_reanchor_dataset.py --root <map root> --ids <npz with id> --runs <dir> [<dir> ...] --out X.npz
"""
import argparse, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f104_n2_dataset as DS

DT, S0, NS = 0.05, 20, 96
ARGS = {}


def init(root, a):
    DS.init_map(root); ARGS.update(a)


def one(d):
    try:
        z = np.load(d + '/trajectory.npz'); o = json.load(open(d + '/outcome.json')); c = json.load(open(d + '/case.json'))
        r = np.load(d + '/command_reference.npz')
    except Exception:
        return []
    wp, sp, st = np.asarray(r['reference_waypoints'], float), np.asarray(r['reference_speeds'], float), np.asarray(r['reference_stations'], float)
    state, pose_full, act = z['state'].astype(np.float32), z['pose'].astype(float), z['action'].astype(float)
    n = len(state); vx = state[:, 0].astype(float); thr = act[:, 1]; parked = np.asarray(z['parked'], bool)
    fail = o['status'] != 'goal_reached'
    back = (vx < -0.10) & (thr > 0.3)
    rb = DS.first_run(back | (vx < -0.30), 1, S0); ns = DS.first_run((np.abs(vx) < 0.3) & (thr > 0.3), 20, S0)
    cands = [x for x in (rb, ns) if x is not None]; ev = min(cands) if cands else (n - 1 if fail else None)
    back_s = float(back[S0:].sum() * DT); min_vx = float(vx[S0:].min()) if n > S0 else float(vx.min())
    clean = (not fail) and back_s < 0.05 and min_vx > -0.30
    if clean: ev = None
    s, dev, L = DS.project(pose_full[:, :2], wp); smax = np.maximum.accumulate(s)
    cumE = np.concatenate([[0.0], np.cumsum(np.asarray(z['positive_work_kj_per_interval'], float))])   # cumE[k] = work before frame k
    goal = np.asarray(c['goal_xy'], float); rid = os.path.basename(d)
    last = (ev if ev is not None else n) - 1
    pool = [k for k in range(40, max(last, 0), 40) if (L - smax[k]) >= ARGS['min_rem'] and dev[k] < 1.0 and not parked[k]]
    want = ARGS['anchors'] - 1
    if len(pool) > want:  # stratify over anchor speed
        pool = [pool[i] for i in np.round(np.linspace(0, len(pool) - 1, want)).astype(int)] if want > 0 else []
        pool = sorted(set(pool), key=lambda k: vx[k])
    rows = []
    for k in [0] + sorted(pool):
        sk = float(smax[k]) if k > 0 else 0.0; rem = L - sk
        if k > 0:
            i = int(np.searchsorted(st, sk, side='right'))
            p0 = np.array([np.interp(sk, st, wp[:, 0]), np.interp(sk, st, wp[:, 1])]); v0 = float(np.interp(sk, st, sp))
            wp2 = np.vstack([p0[None], wp[i:]]); sp2 = np.r_[v0, sp[i:]]
            if len(wp2) < 3: continue
            st2 = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp2, axis=0), axis=1))]
        else:
            wp2, sp2, st2 = wp, sp, st
        X, L2 = DS.station_tensor(wp2, sp2, st2)
        grid = np.linspace(0.0, L2, NS); E = np.full(NS, np.nan, np.float32); T = np.full(NS, np.nan, np.float32)
        reach = float(smax[-1]) - sk
        for j, gj in enumerate(grid):
            if gj > reach + 1e-9: break
            a = max(k, min(int(np.searchsorted(smax, sk + gj, side='left')), n - 1))
            E[j] = cumE[a + 1] - cumE[k]; T[j] = (a + 1 - k) * DT
        if ev is None: eidx = -1
        else:
            frac = (float(smax[ev]) - sk) / max(rem, 1e-6); eidx = int(np.clip(round(frac * (NS - 1)), 0, NS - 1))
        pk = pose_full[k]; rel = goal - pk[:2]
        ctx = np.concatenate([state[k], [rel[0], rel[1], np.linalg.norm(rel), pk[2], L2]]).astype(np.float32)
        prof = int(rid.split('_route_')[1]) % 4 if '_route_' in rid else -1
        rows.append(dict(X=X.astype(np.float16), ctx=ctx, E=E, T=T, id=f'{rid}@{k}', episode=rid, group=c['id'], split=c['split'],
                         source='designed' if '_route_' in rid else 'on_policy', profile=prof, fail=int(fail), unsafe=int(not clean),
                         event_idx=eidx, route_len=np.float32(L2), anchor_frame=k, vx_anchor=np.float32(vx[k]), rem_m=np.float32(rem),
                         time_to_event_s=np.float32((ev - k) * DT if ev is not None else -1.0), status=o['status']))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--ids', required=True, help='npz whose id array selects the episodes')
    ap.add_argument('--runs', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--anchors', type=int, default=4); ap.add_argument('--min-remaining', type=float, default=12.0); ap.add_argument('--workers', type=int, default=14)
    a = ap.parse_args()
    want = set(np.load(a.ids, allow_pickle=True)['id'].astype(str))
    dirs = [d for pat in a.runs for d in glob.glob(pat + '/*') if os.path.basename(d) in want]
    print(f'{len(dirs)} episodes selected of {len(want)} ids', flush=True)
    res = []
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.root, dict(anchors=a.anchors, min_rem=a.min_remaining))) as ex:
        for rows in ex.map(one, sorted(dirs), chunksize=32): res += rows
    A = lambda k, dt=None: np.array([r[k] for r in res], dtype=dt)
    np.savez_compressed(a.out, X=np.stack([r['X'] for r in res]), ctx=np.stack([r['ctx'] for r in res]), E=np.stack([r['E'] for r in res]), T=np.stack([r['T'] for r in res]),
                        id=A('id', object), episode=A('episode', object), group=A('group', object), split=A('split', object), source=A('source', object), status=A('status', object),
                        profile=A('profile', np.int8), fail=A('fail', np.int8), unsafe=A('unsafe', np.int8), event_idx=A('event_idx', np.int16), route_len=A('route_len', np.float32),
                        anchor_frame=A('anchor_frame', np.int32), vx_anchor=A('vx_anchor', np.float32), rem_m=A('rem_m', np.float32), time_to_event_s=A('time_to_event_s', np.float32))
    v = A('vx_anchor', float); k0 = A('anchor_frame', int) == 0
    print(f'{len(res)} rows from {len(set(A("episode")))} episodes ({k0.sum()} start-anchored); unsafe {A("unsafe", float).mean():.3f}; '
          f'vx at moving anchors p10/50/90 {np.percentile(v[~k0], [10, 50, 90]).round(2).tolist()}; wrote {a.out} {os.path.getsize(a.out) / 1e9:.2f} GB', flush=True)


if __name__ == '__main__':
    main()
