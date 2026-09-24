"""Short-anchor (early moving-start) training rows for both worlds, in the exact format of mixed_reanchor.npz (PLAN S0-W / S2).

For every episode that has a row in the mixed re-anchored file (both worlds, 15,024 twin episodes each) and every decision
frame k in --ks (default 10 / 20 / 30 = 0.5 / 1.0 / 1.5 s after the standing start) one row is built when the anchor is
admitted. Row construction is copied line for line from scripts/n2_reanchor_dataset.py (k > 0 branch):
  event frame by the labeller's rule (reverse / no-progress runs from frame 20; failed episodes without a run -> last frame;
  clean episodes -> no event), admission of k: k < event - 1 (n2's range(40, last, 40) bound), remaining route
  L - smax[k] >= --min-remaining (12 m), lateral deviation dev[k] < 1 m, not parked[k], >= 3 waypoints left;
  remaining route from the projection point (profile speeds), corridor via f104_n2_dataset.station_tensor on the depth map
  (--root), E / T = first-arrival cumulative positive work / time after the anchor, event station on the remaining route,
  ctx (22) = [state[k] (17), goal - pose_k, |.|, yaw_k, L_rem].
History and privileged context come from scripts/ga_build_mixed.cut_episode (imported, not copied): hist (40, 15) f16 cut
at k (hist[t] = [state[k-39+t][cols 0-6, 11-15], action[k-40+t]]), hmask = action frame k-40+t >= 0 (k = 10 -> 10 valid
steps), privileged (8) = means over state frames max(0, k-39)..k.
ids are <episode>@<k>@<world>; domain 0 rigid / 1 crm; rigid rows first, then CRM. Keys, dtypes and trailing shapes are
asserted against the npz headers of the reference file; uniqueness, blacklist, split / group agreement with the reference
and the ga_build_mixed invariants (hmask count, last window step == anchor state in ctx) are asserted too.

Check modes
  --verify-mixed N   rebuild N random episodes per world at exactly the anchors n2_reanchor_dataset.py selects (k = 0 plus
                     the speed-stratified multiples of 40, using this file's admission function) and compare every key
                     byte for byte with the stored rows of the reference file (also asserts the anchor sets are equal).
  --compare-mixed    after a build, compare every built row whose id also exists in the reference file (only possible
                     for k that are multiples of 40, e.g. --ks 40 60 80).

  PYTHONPATH=src:scripts python scripts/ci_short_anchors.py --out artifacts/traverse/crm_improve_20260922/datasets/short_anchor.npz
  PYTHONPATH=src:scripts python scripts/ci_short_anchors.py --verify-mixed 300 --out artifacts/traverse/crm_improve_20260922/datasets/verify_mixed.json
"""
import argparse, fnmatch, json, os, resource, sys, time, zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f104_n2_dataset as DS
import ga_build_mixed as GB

DT, S0, NS = 0.05, 20, 96                      # as scripts/n2_reanchor_dataset.py
ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'artifacts', 'traverse')
DEFAULTS = dict(ref=ART + '/generalist_20260921/A_adapt/datasets/mixed_reanchor.npz', root=ART + '/crm_f104_v1/map_root',
                rigid_runs=GB.DEFAULTS['rigid_runs'], crm_runs=GB.DEFAULTS['crm_runs'])
WORLDS = [('rigid', 0), ('crm', 1)]
REASONS = ['admitted', 'load_error', 'after_event', 'beyond_end', 'remaining_short', 'off_route', 'parked', 'short_route']
ARGS = {}


def init(root, a):
    DS.init_map(root); ARGS.update(a)


def npz_header(path):
    """{key: (shape, dtype)} of every member of an npz, read from the headers only (no decompression of the data)."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            with zf.open(info) as f:
                v = np.lib.format.read_magic(f)
                rd = np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dt = rd(f)
            out[info.filename[:-4]] = (tuple(shape), np.dtype(dt))
    return out


def analyse(d):
    """Per-episode quantities of n2_reanchor_dataset.one (verbatim), or None if a file is missing / unreadable."""
    try:
        z = np.load(d + '/trajectory.npz'); o = json.load(open(d + '/outcome.json')); c = json.load(open(d + '/case.json'))
        r = np.load(d + '/command_reference.npz')
    except Exception:
        return None
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
    return dict(wp=wp, sp=sp, st=st, state=state, pose_full=pose_full, n=n, vx=vx, parked=parked, fail=fail, clean=clean, ev=ev,
                dev=dev, L=L, smax=smax, cumE=cumE, goal=goal, rid=rid, last=last, c=c, o=o)


def admit(e, k, min_rem):
    """Reason string; 'admitted' iff n2_reanchor_dataset.py would put k in its anchor pool (k > 0). The >= 3 waypoint rule is
    applied in make_row (n2 checks it while building)."""
    if k >= e['n']: return 'beyond_end'
    if not k < max(e['last'], 0): return 'after_event'
    if (e['L'] - e['smax'][k]) < min_rem: return 'remaining_short'
    if not e['dev'][k] < 1.0: return 'off_route'
    if e['parked'][k]: return 'parked'
    return 'admitted'


def n2_anchor_set(e, anchors, min_rem):
    """The anchors n2_reanchor_dataset.py selects for this episode: [0] + speed-stratified subset of the multiples of 40."""
    vx = e['vx']
    pool = [k for k in range(40, max(e['last'], 0), 40) if admit(e, k, min_rem) == 'admitted']
    want = anchors - 1
    if len(pool) > want:
        pool = [pool[i] for i in np.round(np.linspace(0, len(pool) - 1, want)).astype(int)] if want > 0 else []
        pool = sorted(set(pool), key=lambda k: vx[k])
    return [0] + sorted(pool)


def make_row(e, k, world):
    """One row, verbatim from the anchor loop of n2_reanchor_dataset.one; None when fewer than 3 waypoints remain."""
    wp, sp, st, smax, L, n = e['wp'], e['sp'], e['st'], e['smax'], e['L'], e['n']
    state, pose_full, vx, cumE, goal, rid, c, o, ev = e['state'], e['pose_full'], e['vx'], e['cumE'], e['goal'], e['rid'], e['c'], e['o'], e['ev']
    sk = float(smax[k]) if k > 0 else 0.0; rem = L - sk
    if k > 0:
        i = int(np.searchsorted(st, sk, side='right'))
        p0 = np.array([np.interp(sk, st, wp[:, 0]), np.interp(sk, st, wp[:, 1])]); v0 = float(np.interp(sk, st, sp))
        wp2 = np.vstack([p0[None], wp[i:]]); sp2 = np.r_[v0, sp[i:]]
        if len(wp2) < 3: return None
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
    return dict(X=X.astype(np.float16), ctx=ctx, E=E, T=T, id=f'{rid}@{k}@{world}', episode=rid, group=c['id'], split=c['split'],
                source='designed' if '_route_' in rid else 'on_policy', profile=prof, fail=int(e['fail']), unsafe=int(not e['clean']),
                event_idx=eidx, route_len=np.float32(L2), anchor_frame=k, vx_anchor=np.float32(vx[k]), rem_m=np.float32(rem),
                time_to_event_s=np.float32((ev - k) * DT if ev is not None else -1.0), status=o['status'])


def work(item):
    """(episode, world, ks, roots, force) -> (episode, world, rows, reasons). force=True skips admission (verify mode: ks are
    already n2's anchors, including k = 0)."""
    episode, world, ks, roots, force = item
    d = GB.find_run(episode, roots)
    if d is None:
        return episode, world, [], {k: 'load_error' for k in ks}
    e = analyse(d)
    if e is None:
        return episode, world, [], {k: 'load_error' for k in ks}
    reasons, rows = {}, []
    for k in ks:
        why = 'admitted' if force else admit(e, int(k), ARGS['min_rem'])
        if why == 'admitted':
            r = make_row(e, int(k), world)
            if r is None: why = 'short_route'
            else: rows.append(r)
        reasons[int(k)] = why
    if rows:
        _, res = GB.cut_episode((episode, [r['anchor_frame'] for r in rows], roots, world == 'crm'))
        assert res is not None
        h, m, p = res
        for j, r in enumerate(rows):
            r['hist'], r['hmask'], r['privileged'] = h[j], m[j], p[j]
    extra = dict(n2_anchors=n2_anchor_set(e, ARGS['anchors'], ARGS['min_rem'])) if ARGS.get('want_n2') else {}
    return episode, world, rows, dict(reasons=reasons, **extra)


def reference_meta(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in ['id', 'episode', 'group', 'split', 'domain', 'anchor_frame']}


def compare(built, ref_path, ref_idx, keys):
    """Byte-level comparison of built rows (dict of arrays, aligned) with reference rows ref_idx. Returns {key: n_mismatch}."""
    z = np.load(ref_path, allow_pickle=True); out = {}
    for k in keys:
        a = built[k]; b = z[k][ref_idx]
        if a.dtype.kind in 'fc':
            same = (a == b) | (np.isnan(a.astype(np.float32)) & np.isnan(b.astype(np.float32)))
        else:
            same = (a.astype(str) == b.astype(str)) if a.dtype == object else (a == b)
        same = same.reshape(len(a), -1).all(1)
        out[k] = int((~same).sum())
        del b
    return out


def stack_rows(rows, header):
    out = {}
    for k, (shape, dt) in header.items():
        if k in ('hist_cols', 'priv_names', 'domain'): continue
        if len(shape) > 1:
            out[k] = np.stack([r[k] for r in rows]).astype(dt)
        else:
            out[k] = np.array([r[k] for r in rows], dtype=dt)
    out['domain'] = np.array([0 if r['id'].endswith('@rigid') else 1 for r in rows], np.int8)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref', default=DEFAULTS['ref'], help='mixed_reanchor.npz: episode list, split/group reference, key/dtype contract')
    ap.add_argument('--root', default=DEFAULTS['root'], help='depth-map root for the corridor (f104_n2_dataset.init_map)')
    ap.add_argument('--rigid-runs', nargs='+', default=DEFAULTS['rigid_runs']); ap.add_argument('--crm-runs', nargs='+', default=DEFAULTS['crm_runs'])
    ap.add_argument('--ks', type=int, nargs='+', default=[10, 20, 30])
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-remaining', type=float, default=12.0); ap.add_argument('--anchors', type=int, default=4, help='n2 anchor count (verify mode)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit-episodes', type=int, default=0, help='self-test: first N episodes (sorted) per world')
    ap.add_argument('--verify-mixed', type=int, default=0, help='rebuild N random episodes per world at n2 anchors and compare with the reference')
    ap.add_argument('--compare-mixed', action='store_true', help='compare built rows whose id exists in the reference file')
    ap.add_argument('--no-compress', action='store_true')
    a = ap.parse_args()
    a.workers = max(1, min(a.workers, 8))
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    header = npz_header(a.ref)
    ref = reference_meta(a.ref)
    ref_ep = ref['episode'].astype(str); ref_dom = ref['domain'].astype(int)
    ep_split = {}; ep_group = {}
    for e, dm, sp, g in zip(ref_ep, ref_dom, ref['split'].astype(str), ref['group'].astype(str)):
        assert ep_split.setdefault((e, dm), sp) == sp and ep_group.setdefault((e, dm), g) == g
    roots = dict(rigid=a.rigid_runs, crm=a.crm_runs)
    episodes = {w: sorted(set(ref_ep[ref_dom == dm])) for w, dm in WORLDS}
    if a.limit_episodes:
        episodes = {w: v[:a.limit_episodes] for w, v in episodes.items()}
    print(f'reference {a.ref}: {len(ref_ep)} rows; episodes rigid {len(episodes["rigid"])} crm {len(episodes["crm"])}', flush=True)

    # ------------------------------------------------------------------ verify mode
    if a.verify_mixed:
        rng = np.random.default_rng(0)
        ref_af = ref['anchor_frame'].astype(int)
        stored = defaultdict(list)
        for i, (e, dm, k) in enumerate(zip(ref_ep, ref_dom, ref_af)):
            stored[(e, dm)].append((int(k), i))
        tasks = []
        for w, dm in WORLDS:
            for e in sorted(rng.choice(episodes[w], size=min(a.verify_mixed, len(episodes[w])), replace=False)):
                ks = sorted(k for k, _ in stored[(e, dm)])
                tasks.append((e, w, ks, roots[w], True))
        ARGS_INIT = dict(anchors=a.anchors, min_rem=a.min_remaining, want_n2=True)
        rows, ref_idx, anchor_mismatch = [], [], []
        with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.root, ARGS_INIT)) as ex:
            for (e, w, ks, _, _), (e2, w2, rr, info) in zip(tasks, ex.map(work, tasks, chunksize=8)):
                dm = dict(WORLDS)[w]
                if sorted(info['n2_anchors']) != ks: anchor_mismatch.append((e, w, ks, info['n2_anchors']))
                idx = dict(stored[(e, dm)])
                for r in rr:
                    rows.append(r); ref_idx.append(idx[r['anchor_frame']])
        built = stack_rows(rows, header); ref_idx = np.asarray(ref_idx)
        keys = [k for k in header if k not in ('hist_cols', 'priv_names')]
        mism = compare(built, a.ref, ref_idx, keys)
        res = dict(mode='verify_mixed', episodes_per_world=a.verify_mixed, rows_compared=len(rows), anchor_set_mismatches=len(anchor_mismatch),
                   anchor_mismatch_examples=[list(map(str, m)) for m in anchor_mismatch[:10]], key_mismatch_rows=mism,
                   anchor_frames=dict(Counter(int(k) for k in built['anchor_frame'])), elapsed_s=round(time.time() - t0, 1))
        with open(a.out, 'w') as f: json.dump(res, f, indent=1)
        print(json.dumps(res, indent=1), flush=True)
        ok = not anchor_mismatch and all(v == 0 for v in mism.values())
        print('VERIFY ' + ('PASSED: every key byte-identical and anchor sets equal' if ok else 'FAILED'), flush=True)
        sys.exit(0 if ok else 3)

    # ------------------------------------------------------------------ build
    tasks = [(e, w, list(a.ks), roots[w], False) for w, _ in WORLDS for e in episodes[w]]
    n_max = len(tasks) * len(a.ks)
    X = np.zeros((n_max, 5, 96, 32), np.float16); hist = np.zeros((n_max, 40, 15), np.float16)
    hmask = np.zeros((n_max, 40), bool); priv = np.zeros((n_max, 8), np.float32)
    small = []; reasons = defaultdict(Counter); o = 0
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.root, dict(anchors=a.anchors, min_rem=a.min_remaining))) as ex:
        for it, (e, w, rr, info) in enumerate(ex.map(work, tasks, chunksize=16)):
            for k, why in info['reasons'].items(): reasons[(w, k)][why] += 1
            for r in rr:
                X[o] = r.pop('X'); hist[o] = r.pop('hist'); hmask[o] = r.pop('hmask'); priv[o] = r.pop('privileged'); small.append(r); o += 1
            if (it + 1) % 5000 == 0: print(f'  {it + 1}/{len(tasks)} episodes, {o} rows, {time.time() - t0:.0f} s', flush=True)
    n = o
    print(f'{n} rows from {len(tasks)} episode tasks in {time.time() - t0:.0f} s', flush=True)
    out = {}
    for k, (shape, dt) in header.items():
        if k in ('X', 'hist', 'hmask', 'privileged', 'hist_cols', 'priv_names', 'domain'): continue
        out[k] = (np.stack([r[k] for r in small]) if len(shape) > 1 else np.array([r[k] for r in small], dtype=object if dt == object else None)).astype(dt)
    out['domain'] = np.array([0 if r['id'].endswith('@rigid') else 1 for r in small], np.int8)
    out['hist'] = hist[:n]; out['hmask'] = hmask[:n]; out['privileged'] = priv[:n]
    out['hist_cols'] = np.asarray(GB.HIST_STATE_COLS, np.int16); out['priv_names'] = np.asarray(GB.PRIV_NAMES, object)
    out['X'] = X[:n]
    out = {k: out[k] for k in header}                      # reference key order

    # ---- contracts
    assert list(out) == list(header), (list(out), list(header))
    for k, (shape, dt) in header.items():
        assert out[k].dtype == dt, (k, out[k].dtype, dt)
        assert out[k].shape[1:] == shape[1:], (k, out[k].shape, shape)
        if k not in ('hist_cols', 'priv_names'): assert len(out[k]) == n, (k, len(out[k]), n)
    assert np.array_equal(out['hist_cols'], np.asarray(GB.HIST_STATE_COLS)), 'hist_cols differ'
    episode = out['episode'].astype(str); dom = out['domain'].astype(int); af = out['anchor_frame'].astype(int)
    split = out['split'].astype(str); group = out['group'].astype(str); ids = out['id'].astype(str)
    trip = list(zip(episode, af.tolist(), dom.tolist()))
    assert len(set(trip)) == n and len(set(ids)) == n, 'duplicate (episode, k, domain) or id'
    assert all(i == f'{e}@{k}@{"crm" if d else "rigid"}' for i, e, k, d in zip(ids, episode, af, dom)), 'id format'
    bad = [s for s in set(ids) | set(group) | set(episode) if GB.blacklisted(s)]
    assert not bad, f'blacklisted ids/groups present: {bad[:10]}'
    assert all(ep_split[(e, d)] == s and ep_group[(e, d)] == g for e, d, s, g in zip(episode, dom, split, group)), 'split/group differ from the reference'
    assert np.all(out['hmask'].sum(1) == np.minimum(af, 40)), 'hmask count != min(k, 40)'
    refc = out['ctx'][:, GB.HIST_STATE_COLS].astype(np.float32); got = out['hist'][:, -1, :12].astype(np.float32)
    assert np.all(np.abs(refc - got) <= 2e-3 * np.maximum(np.abs(refc), 1.0)), 'hist last step != anchor state in ctx'
    assert np.all(out['privileged'][dom == 1, 7] == 1) and np.all(out['privileged'][dom == 0, 5:8] == 0)
    print(f'contracts passed (keys/dtypes/shapes vs {os.path.basename(a.ref)} headers, uniqueness, blacklist, split/group, hmask, hist-ctx)', flush=True)

    # ---- optional byte comparison with the reference rows (multiples of 40 only)
    cmp_res = None
    if a.compare_mixed:
        ref_id_idx = {s: i for i, s in enumerate(ref['id'].astype(str))}
        mine = [i for i, s in enumerate(ids) if s in ref_id_idx]
        if mine:
            mine = np.asarray(mine); ridx = np.asarray([ref_id_idx[ids[i]] for i in mine])
            keys = [k for k in header if k not in ('hist_cols', 'priv_names')]
            cmp_res = dict(rows_compared=int(len(mine)), key_mismatch_rows=compare({k: out[k][mine] for k in keys}, a.ref, ridx, keys))
        else:
            cmp_res = dict(rows_compared=0)
        print(f'comparison with the reference rows: {cmp_res}', flush=True)

    # ---- report
    wname = {0: 'rigid', 1: 'crm'}
    counts = {}; label = {}; pairs = {}
    for dm in (0, 1):
        for k in a.ks:
            for sp in ('train', 'val', 'test'):
                m = (dom == dm) & (af == k) & (split == sp)
                counts[f'{wname[dm]}|k{k}|{sp}'] = int(m.sum())
                label[f'{wname[dm]}|k{k}|{sp}'] = dict(fail=round(float(out['fail'][m].mean()), 4) if m.any() else None,
                                                     unsafe=round(float(out['unsafe'][m].mean()), 4) if m.any() else None)
    have = defaultdict(set)
    for e, k, d in trip: have[(e, k)].add(d)
    for k in a.ks:
        for sp in ('train', 'val', 'test'):
            pairs[f'k{k}|{sp}'] = sum(1 for (e, kk), s in have.items() if kk == k and len(s) == 2 and ep_split[(e, 0)] == sp)
    adm = {f'{w}|k{k}': {r: reasons[(w, k)].get(r, 0) for r in REASONS} for w, _ in WORLDS for k in a.ks}
    vxp = {f'{wname[dm]}|k{k}': np.percentile(out['vx_anchor'][(dom == dm) & (af == k)], [10, 50, 90]).round(3).tolist()
           for dm in (0, 1) for k in a.ks if ((dom == dm) & (af == k)).any()}
    t_save = time.time()
    (np.savez if a.no_compress else np.savez_compressed)(a.out, **out)
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6; rc = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    summary = dict(out=os.path.abspath(a.out), rows=n, ks=a.ks, episodes=dict(rigid=len(episodes['rigid']), crm=len(episodes['crm'])),
                   counts_world_k_split=counts, label_rates_world_k_split=label, pairs_both_worlds_k_split=pairs,
                   admission_world_k=adm, admission_reject_rate_world_k={kk: round(1 - v['admitted'] / max(sum(v.values()), 1), 4) for kk, v in adm.items()},
                   vx_anchor_p10_50_90=vxp, compare_mixed=cmp_res,
                   rules=dict(min_remaining_m=a.min_remaining, max_dev_m=1.0, event='n2 labeller (reverse / no-progress runs from frame 20)',
                              admit='k < event_frame - 1, L - smax[k] >= min_remaining, dev[k] < 1, not parked, >= 3 waypoints left',
                              hist='ga_build_mixed.cut_episode', id='<episode>@<k>@<world>'),
                   sources=dict(ref=a.ref, root=a.root, rigid_runs=a.rigid_runs, crm_runs=a.crm_runs), workers=a.workers,
                   limit_episodes=a.limit_episodes, elapsed_s=round(time.time() - t0, 1), save_s=round(time.time() - t_save, 1),
                   peak_rss_gb_main=round(ru, 2), peak_rss_gb_child_max=round(rc, 2), size_gb=round(os.path.getsize(a.out) / 1e9, 3))
    base = os.path.splitext(a.out)[0]
    with open(base + '_build.json', 'w') as f: json.dump(summary, f, indent=1)
    lines = [f'# Build report: {os.path.basename(a.out)}', '', f'Command: `PYTHONPATH=src:scripts python scripts/ci_short_anchors.py {" ".join(sys.argv[1:])}`', '',
             f'{n} rows from {len(episodes["rigid"])} rigid + {len(episodes["crm"])} CRM episodes; {summary["size_gb"]} GB; {summary["elapsed_s"]} s; peak RSS main {ru:.1f} GB, child max {rc:.1f} GB.', '',
             '## Rows per world x decision frame x split (goal-not-reached rate / unsafe rate of the rows)', '',
             '| world | k | train | val | test | fail rate train/val/test | unsafe rate train/val/test |', '|---|---|---|---|---|---|---|']
    for dm in (0, 1):
        for k in a.ks:
            c3 = [counts[f'{wname[dm]}|k{k}|{sp}'] for sp in ('train', 'val', 'test')]
            fr = '/'.join('-' if label[f'{wname[dm]}|k{k}|{sp}']['fail'] is None else f'{label[f"{wname[dm]}|k{k}|{sp}"]["fail"]:.3f}' for sp in ('train', 'val', 'test'))
            ur = '/'.join('-' if label[f'{wname[dm]}|k{k}|{sp}']['unsafe'] is None else f'{label[f"{wname[dm]}|k{k}|{sp}"]["unsafe"]:.3f}' for sp in ('train', 'val', 'test'))
            lines.append(f'| {wname[dm]} | {k} | {c3[0]} | {c3[1]} | {c3[2]} | {fr} | {ur} |')
    lines += ['', '## Anchor admission (episodes per world x k)', '', '| world | k | ' + ' | '.join(REASONS) + ' | rejected |', '|---|---|' + '---|' * (len(REASONS) + 1)]
    for w, _ in WORLDS:
        for k in a.ks:
            v = adm[f'{w}|k{k}']
            lines.append(f'| {w} | {k} | ' + ' | '.join(str(v[r]) for r in REASONS) + f' | {summary["admission_reject_rate_world_k"][f"{w}|k{k}"]:.3f} |')
    lines += ['', '## (episode, k) pairs present in both worlds', '', '| k | train | val | test |', '|---|---|---|---|']
    for k in a.ks:
        lines.append(f'| {k} | ' + ' | '.join(str(pairs[f"k{k}|{sp}"]) for sp in ('train', 'val', 'test')) + ' |')
    lines += ['', f'Anchor speed vx p10/p50/p90 (m/s): ' + '; '.join(f'{kk} {v}' for kk, v in vxp.items())]
    if cmp_res is not None: lines += ['', f'Byte comparison with the reference rows of the same id: {cmp_res}']
    with open(base + '_build.md', 'w') as f: f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    print(f'wrote {a.out} ({summary["size_gb"]} GB) in {summary["elapsed_s"]} s', flush=True)


if __name__ == '__main__':
    main()
