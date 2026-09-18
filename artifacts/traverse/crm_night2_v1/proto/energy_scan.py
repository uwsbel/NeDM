"""Read-only scan of local rigid + CRM episodes: energy signals, stall energy, per-station cumulative energy, moving anchors."""
import sys, os, glob, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, '/home/harry/NeDM-traverse_mppi/scripts')
import f104_n2_dataset as DS
ROOT = '/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_f104_50h_20260909'
DT, S0, NS = 0.05, 20, 96
G = {}

def init():
    DS.init_map(ROOT)

def one(d):
    try:
        z = np.load(d + '/trajectory.npz'); o = json.load(open(d + '/outcome.json'))
        wp, sp, st = DS.load_route(d + '/command_reference.npz')
    except Exception as e:
        return None
    vx = z['state'][:, 0].astype(float); vy = z['state'][:, 1].astype(float); yr = z['state'][:, 6].astype(float)
    pitch = z['state'][:, 3].astype(float)
    es = z['state'][:, 15].astype(float); tq = z['state'][:, 16].astype(float)
    thr = z['action'][:, 1].astype(float); pw = z['power_kw'].astype(float); iw = z['positive_work_kj_per_interval'].astype(float)
    parked = z['parked'].astype(bool); pose = z['pose'][:, :2].astype(float); n = len(vx)
    if n < 2: return None
    Wp = float(o['positive_work_kj']); Wp_int = float(iw.sum())
    Ps = es * tq / 1000.
    E_state_signed = float(Ps.sum() * DT); E_state_pos = float(np.maximum(Ps, 0).sum() * DT)
    E_pw_signed = float(pw.sum() * DT); E_pw_pos = float(np.maximum(pw, 0).sum() * DT); E_pw_neg = float(np.minimum(pw, 0).sum() * DT)
    fail = o['status'] != 'goal_reached'
    back = (vx < -0.10) & (thr > 0.3)
    rb = DS.first_run(back | (vx < -0.30), 1, S0); ns = DS.first_run((np.abs(vx) < 0.3) & (thr > 0.3), 20, S0)
    cands = [x for x in (rb, ns) if x is not None]
    ev = min(cands) if cands else (n - 1 if fail else None)
    back_s = float(back[S0:].sum() * DT); min_vx = float(vx[S0:].min()) if n > S0 else float(vx.min())
    clean = (not fail) and back_s < 0.05 and min_vx > -0.30
    if clean: ev = None
    stalled = (np.abs(vx) < 0.3) & (thr > 0.3) & (~parked)
    slow = (np.abs(vx) < 1.0) & (thr > 0.3) & (~parked)
    E_stalled = float(iw[stalled].sum()); E_slow = float(iw[slow].sum()); E_parked = float(iw[parked].sum())
    E_before_ev = float(iw[:ev].sum()) if ev is not None else Wp_int
    s, dev, L = DS.project(pose, wp)
    smax = np.maximum.accumulate(s)
    hwm = float(smax[-1]) if ev is None else float(smax[ev])
    cumE = np.cumsum(iw)
    grid = np.linspace(0, L, NS)
    idx = np.clip(np.searchsorted(smax, grid, side='left'), 0, n - 1)
    E_station = cumE[idx]; reached = smax[-1] >= grid - 1e-6
    # first-arrival time at each station (s)
    T_station = idx * DT
    patch, valid = DS.sample_map(pose[:, 0], pose[:, 1]); zz = patch[3] * DS.G['elev_scale']
    dz = np.diff(zz); climb = float(np.maximum(dz, 0).sum()); descent = float(-np.minimum(dz, 0).sum())
    # climb along the commanded route (map, not pose): route's total positive elevation gain
    pr, valid_r = DS.sample_map(wp[:, 0], wp[:, 1]); zr = pr[3] * DS.G['elev_scale']
    climb_route = float(np.maximum(np.diff(zr), 0).sum())
    path_len = float(np.linalg.norm(np.diff(pose, axis=0), axis=1).sum())
    mean_cmd = float(sp[1:-1].mean()) if len(sp) > 2 else float(sp.mean())
    vmax_cmd = float(sp.max())
    rid = os.path.basename(d)
    if '_route_' in rid:
        k = int(rid.split('_route_')[1]); profile, offset, src = k % 4, k // 4, 'designed'
    elif '_op_' in rid:
        profile, offset, src = -1, -1, 'on_policy'
    else:
        profile, offset, src = -1, -1, 'other'
    # moving anchors every 2 s (40 frames), only before the event
    anchors = []
    for k in range(40, n, 40):
        if ev is not None and k >= ev: break
        rem = L - smax[k]
        if rem < 5.0: break
        ev_rel = -1.0 if ev is None else (smax[ev] - smax[k]) / max(rem, 1e-6)
        anchors.append([k, vx[k], vy[k], yr[k], pitch[k], smax[k], rem, 0 if ev is None else 1, ev_rel,
                        cumE[k], Wp_int - cumE[k], float(dev[k]), sp[np.searchsorted(st, smax[k])] if len(st) == len(sp) and smax[k] <= st[-1] else np.nan])
    return dict(id=rid, status=o['status'], fail=int(fail), unsafe=int(not clean), n=n, elapsed=n * DT,
                goal_time=o.get('goal_time_s'), Wp=Wp, Wp_int=Wp_int, E_state_signed=E_state_signed, E_state_pos=E_state_pos,
                E_pw_signed=E_pw_signed, E_pw_pos=E_pw_pos, E_pw_neg=E_pw_neg, E_stalled=E_stalled, E_slow=E_slow,
                E_parked=E_parked, E_before_ev=E_before_ev, ev=(-1 if ev is None else int(ev)), hwm=hwm, L=float(L),
                path_len=path_len, climb=climb, descent=descent, climb_route=climb_route, mean_cmd=mean_cmd, vmax_cmd=vmax_cmd,
                profile=profile, offset=offset, src=src, stalled_s=float(stalled.sum() * DT), slow_s=float(slow.sum() * DT),
                mean_pitch_moving=float(pitch[np.abs(vx) > 0.5].mean()) if (np.abs(vx) > 0.5).any() else np.nan,
                peak_kw=float(pw.max()), mean_kw_moving=float(pw[np.abs(vx) > 0.5].mean()) if (np.abs(vx) > 0.5).any() else np.nan,
                mean_kw_stalled=float(pw[stalled].mean()) if stalled.any() else np.nan,
                E_station=E_station.astype(np.float32), reached=reached, T_station=T_station.astype(np.float32),
                anchors=np.asarray(anchors, np.float32).reshape(-1, 13))

def scan(pattern, tag, limit=None, workers=16):
    ds = sorted(glob.glob(pattern))
    if limit: 
        rng = np.random.default_rng(0); ds = [ds[i] for i in sorted(rng.choice(len(ds), min(limit, len(ds)), replace=False))]
    t0 = time.time(); res = []
    with ProcessPoolExecutor(workers, initializer=init) as ex:
        for r in ex.map(one, ds, chunksize=32):
            if r is not None: res.append(r)
    print(tag, len(res), 'of', len(ds), 'in', round(time.time() - t0, 1), 's', flush=True)
    return res

if __name__ == '__main__':
    out = '/home/harry/NeDM-traverse_mppi/artifacts/traverse/crm_night2_v1/scout/energy_scan'
    os.makedirs(out, exist_ok=True)
    jobs = {'rigid_v2': ROOT + '/production_v2/runs/*', 'rigid_v4_onpolicy': ROOT + '/production_v4/runs/*',
            'crm_v1': '/home/harry/NeDM-traverse_mppi/artifacts/traverse/crm_f104_v1/collect_v1/runs/*'}
    lim = {'rigid_v2': 6000, 'rigid_v4_onpolicy': 4000, 'crm_v1': 8000}
    for tag, pat in jobs.items():
        res = scan(pat, tag, limit=lim[tag])
        keys = [k for k in res[0] if k not in ('E_station', 'reached', 'T_station', 'anchors')]
        tab = {k: np.array([r[k] for r in res], dtype=(object if isinstance(res[0][k], str) else float)) for k in keys}
        np.savez(f'{out}/{tag}.npz', **tab, E_station=np.stack([r['E_station'] for r in res]),
                 reached=np.stack([r['reached'] for r in res]), T_station=np.stack([r['T_station'] for r in res]),
                 anchors=np.concatenate([np.c_[np.full(len(r['anchors']), i), r['anchors']] for i, r in enumerate(res) if len(r['anchors'])]))
        print('wrote', tag, flush=True)
