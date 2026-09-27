#!/usr/bin/env python3
"""VERIFY E4 (independent): (a) twin rows of the rebuilt f104 file vs generalist mixed_reanchor.npz, exact, on a
random sample of rows (all arrays incl. X / ctx / labels / hist / privileged) and on every row for the small arrays;
(b) row counts and row contents for a random sample of runs, recomputed from the raw run folders:
    - own anchor count (own projection + own event rule, written from the n2_reanchor_dataset docstring/defaults),
    - the unchanged tools' per-episode functions (n2_reanchor_dataset.one, ga_build_mixed.cut_episode) called in
      process on the raw run folder and compared array by array with the file's rows of that episode.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, '/home/harry/NeDM-traverse_mppi/scripts'); sys.path.insert(0, '/home/harry/NeDM-traverse_mppi/src')
A = '/home/harry/NeDM-traverse_mppi/artifacts/traverse'
K3 = A + '/arena_gator_20260925'
NEW = K3 + '/e4/f104_hmmwv/ci_f104_hmmwv_both.npz'
REF = A + '/generalist_20260921/A_adapt/datasets/mixed_reanchor.npz'
MAP = A + '/crm_f104_v1/map_root'
ROOTS = dict(rigid=[A + '/fdm_f104_50h_20260909/production_v3/runs', A + '/fdm_f104_50h_20260909/production_v4/runs'],
             crm=[A + '/crm_f104_v1/collect_v1/runs'])
rng = np.random.default_rng(20260925)
res = {}
t0 = time.time()
Zn = np.load(NEW, allow_pickle=True); Zr = np.load(REF, allow_pickle=True)
small = ['id', 'episode', 'group', 'split', 'domain', 'anchor_frame', 'fail', 'unsafe', 'event_idx', 'status', 'source', 'profile',
         'route_len', 'vx_anchor', 'rem_m', 'time_to_event_s']
N = {k: Zn[k] for k in small}; R = {k: Zr[k] for k in small}
idn = N['id'].astype(str); idr = R['id'].astype(str)
pos_n = {s: i for i, s in enumerate(idn)}
common = [s for s in idr if s in pos_n]
res['twin'] = dict(ref_rows=len(idr), ref_rows_in_new=len(common))
ix_r = np.arange(len(idr)); ix_n = np.array([pos_n[s] for s in idr])
# all rows, small arrays
eq_small = {}
for k in small:
    a, b = N[k][ix_n], R[k][ix_r]
    eq_small[k] = bool(np.array_equal(a.astype(str), b.astype(str))) if (a.dtype == object or b.dtype == object) else bool(np.array_equal(a, b, equal_nan=True))
res['twin']['all_rows_small_arrays_equal'] = eq_small
# the new file's rows on twin episodes = exactly the ref rows (no extra rows on those episodes)
ref_eps = set(zip(R['domain'].astype(int).tolist(), R['episode'].astype(str)))
on_ref = np.array([(int(d), e) in ref_eps for d, e in zip(N['domain'], N['episode'].astype(str))])
res['twin']['new_rows_on_ref_episodes'] = int(on_ref.sum())
# sample of rows, heavy arrays
samp = np.sort(rng.choice(len(idr), 4000, replace=False))
heavy = {}
for k in ['X', 'ctx', 'E', 'T', 'hist', 'hmask', 'privileged']:
    a = Zn[k]; b = Zr[k]
    heavy[k] = dict(sample_equal=bool(np.array_equal(a[ix_n[samp]], b[ix_r[samp]], equal_nan=True)),
                    all_rows_equal=bool(np.array_equal(a[ix_n], b[ix_r], equal_nan=True)), dtype=str(a.dtype), shape=list(a.shape))
    if k == 'X':
        Xn = a
    del b
    if k != 'X':
        del a
res['twin']['heavy'] = heavy
res['twin']['secs'] = round(time.time() - t0, 1)
print(json.dumps(res['twin'], indent=1), flush=True)

# ---------- (b) sample of runs, recomputed from the raw folders
import n2_reanchor_dataset as RA
import ga_build_mixed as GBM
import f104_n2_dataset as DS
RA.init(MAP, dict(anchors=4, min_rem=12.0))
ctx_all = Zn['ctx']; E_all = Zn['E']; T_all = Zn['T']; H_all = Zn['hist']; M_all = Zn['hmask']; P_all = Zn['privileged']
dom = N['domain'].astype(int); ep = N['episode'].astype(str); af = N['anchor_frame'].astype(int)
ref_eps_by = {w: {e for d, e in ref_eps if d == c} for w, c in (('rigid', 0), ('crm', 1))}


def own_project(pose, wp):
    """own point-to-polyline projection: arc length of the closest point, distance, total length"""
    seg = wp[1:] - wp[:-1]; ln = np.hypot(seg[:, 0], seg[:, 1]); ln = np.where(ln < 1e-9, 1e-9, ln)
    cum = np.concatenate([[0.0], np.cumsum(ln)])
    out_s = np.empty(len(pose)); out_d = np.empty(len(pose))
    for i, p in enumerate(pose):
        t = np.clip(((p - wp[:-1]) * seg).sum(1) / ln ** 2, 0, 1)
        q = wp[:-1] + t[:, None] * seg; dd = np.hypot(*(p - q).T); j = int(np.argmin(dd))
        out_s[i] = cum[j] + t[j] * ln[j]; out_d[i] = dd[j]
    return out_s, out_d, float(cum[-1])


def own_count(d):
    """rows the re-anchoring rule gives for one run: 1 (k = 0) + moving anchors (multiples of 40 frames before the
    event, >= 12 m of route left, < 1 m off the route, not parked; at most 3, spread over the list)"""
    z = np.load(d + '/trajectory.npz'); o = json.load(open(d + '/outcome.json')); r = np.load(d + '/command_reference.npz')
    vx = z['state'][:, 0].astype(float); thr = z['action'][:, 1].astype(float); n = len(vx); parked = np.asarray(z['parked'], bool)
    fail = o['status'] != 'goal_reached'
    back = (vx < -0.10) & (thr > 0.3)
    m1 = back | (vx < -0.30); i1 = next((i for i in range(20, n) if m1[i]), None)
    m2 = (np.abs(vx) < 0.3) & (thr > 0.3); run = 0; i2 = None
    for i in range(20, n):
        run = run + 1 if m2[i] else 0
        if run >= 20: i2 = i - 19; break
    cs = [x for x in (i1, i2) if x is not None]
    ev = min(cs) if cs else (n - 1 if fail else None)
    clean = (not fail) and back[20:].sum() * 0.05 < 0.05 and (vx[20:].min() if n > 20 else vx.min()) > -0.30
    if clean: ev = None
    wp = np.asarray(r['reference_waypoints'], float)
    s, dv, L = own_project(z['pose'][:, :2].astype(float), wp); smax = np.maximum.accumulate(s)
    last = (ev if ev is not None else n) - 1
    pool = [k for k in range(40, max(last, 0), 40) if L - smax[k] >= 12.0 and dv[k] < 1.0 and not parked[k]]
    if len(pool) > 3:
        pool = sorted(set(pool[i] for i in np.round(np.linspace(0, len(pool) - 1, 3)).astype(int)))
    return 1 + len(pool), sorted(pool)


def find(w, e):
    for r in ROOTS[w]:
        if os.path.isdir(os.path.join(r, e)):
            return os.path.join(r, e)


samples = {}
for w, code in (('rigid', 0), ('crm', 1)):
    eps_w = sorted(set(ep[dom == code]))
    twin = [e for e in eps_w if e in ref_eps_by[w]]; new = [e for e in eps_w if e not in ref_eps_by[w]]
    pick = list(rng.choice(new, min(len(new), 150 if w == 'rigid' else 80), replace=False)) + list(rng.choice(twin, 50, replace=False))
    samples[w] = dict(n_twin_eps=len(twin), n_new_eps=len(new), picked=len(pick), picked_new=min(len(new), 150 if w == 'rigid' else 80))
    cnt_ok = frames_ok = rows_ok = hist_ok = 0; bad = []
    for e in pick:
        d = find(w, e)
        rows_f = np.flatnonzero((dom == code) & (ep == e))
        nc, pool = own_count(d)
        rr = RA.one(d)
        eh, cut = GBM.cut_episode((e, [int(af[i]) for i in rows_f], ROOTS[w], w == 'crm'))
        ok_cnt = (nc == len(rows_f) == len(rr)); cnt_ok += ok_cnt
        ok_fr = sorted(af[rows_f].tolist()) == [0] + pool == sorted(r_['anchor_frame'] for r_ in rr); frames_ok += ok_fr
        byk = {r_['anchor_frame']: r_ for r_ in rr}
        ok_rows = True
        for i in rows_f:
            r_ = byk.get(int(af[i]))
            if r_ is None: ok_rows = False; break
            ok_rows &= (np.array_equal(r_['X'], Xn[i]) and np.array_equal(r_['ctx'], ctx_all[i]) and np.array_equal(r_['E'], E_all[i], equal_nan=True)
                        and np.array_equal(r_['T'], T_all[i], equal_nan=True) and r_['fail'] == int(N['fail'][i]) and r_['unsafe'] == int(N['unsafe'][i])
                        and r_['event_idx'] == int(N['event_idx'][i]) and r_['split'] == str(N['split'][i]) and r_['group'] == str(N['group'][i])
                        and r_['status'] == str(N['status'][i]) and f"{r_['id']}@{w}" == idn[i])
        rows_ok += ok_rows
        h, m, p = cut
        ok_h = np.array_equal(h, H_all[rows_f]) and np.array_equal(m, M_all[rows_f]) and np.array_equal(p, P_all[rows_f])
        hist_ok += ok_h
        if not (ok_cnt and ok_fr and ok_rows and ok_h):
            bad.append(dict(ep=e, own=nc, file=len(rows_f), tool=len(rr), frames_file=sorted(af[rows_f].tolist()), own_pool=pool, rows=ok_rows, hist=ok_h))
    samples[w].update(count_equal=cnt_ok, anchor_frames_equal=frames_ok, rows_identical=rows_ok, hist_priv_identical=hist_ok, mismatches=bad[:20])
    print(w, json.dumps(samples[w]), flush=True)
res['runs_sample'] = samples
res['secs'] = round(time.time() - t0, 1)
json.dump(res, open(K3 + '/verify_e4/v2_twin_rows.json', 'w'), indent=1, default=str)
print('done', res['secs'])
