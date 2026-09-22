"""Branch-continuation rows for the shared risk model (PLAN v2 A4 'Labels'; review finding 16).

One row per driven continuation (rigid: <anchor episode>__a4__c<j> under gen_collect_ext.py --mode branch_auto; CRM:
<episode>__c<j> under crm_collect_ext.py --mode branch). The row describes the drive AFTER the follower swap only:
  - the prefix (frames 0..F-1, the recorded episode re-driven to the branch frame F) is used for nothing but the history
    window: hist (40,15) = [state[F-39+t][12 observable cols], action[F-40+t]], hmask = action frame >= 0 (the
    collector's branch_hist / branch_hmask, re-derived here from the trajectory and asserted equal);
  - the post-branch poses (frames F..n-1) are projected onto the BRANCH route only (command_reference.npz branch_*
    arrays, hash-checked against outcome.json); the corridor X is station_tensor of the branch route (96 stations from
    the branch pose); ctx = [state[F] (17), goal - pose_F (2), |.|, yaw_F, L_branch] (22, mixed_reanchor layout);
  - the event clock starts after a grace window: the first post-branch frame >= 1 m from the branch pose, or 3 s
    (60 frames), whichever comes first; the night-2 rules run on the post-grace frames: rollback rb = first frame with
    (vx < -0.10 and throttle > 0.3) or vx < -0.30; near-stop ns = first run of 20 frames with |vx| < 0.3 and
    throttle > 0.3;
  - labels: fail = outcome status != goal_reached (goal reached from the branch); unsafe = fail or any post-grace
    rb/ns event (NOTE: night-2 ignored a near-stop when the goal was reached without rollback; here it counts, the
    report gives the number of rows that differ); event_idx = branch-route station of the first event (the furthest
    station reached when a failed drive has no explicit event), -1 if clean;
  - E / T = first-arrival cumulative positive work (kJ) and time (s) after the branch per station, NaN beyond reach;
  - privileged (8) as ga_build_mixed: window means of tyre Fz x4, torque, and for CRM slip ratio + spindle height
    above the BMP ground from crm_extra.npz (zeros on rigid), is_crm.
Row metadata: id = <run dir name>@<world>, episode = anchor episode, group / split / cls from the anchors file,
anchor_frame = F, vx_anchor = vx at F, rem_m = route_len = branch route length, source 'branch', profile -1, status.
Output: npz with EXACTLY the mixed_reanchor.npz keys (checked against --schema-ref without loading its arrays) plus
cls and branch_run, and <out stem>_report.json (counts per world x class x split x label, label discordance among the
continuations of one anchor, grace statistics, dropped counts).

  PYTHONPATH=src:scripts python scripts/ga_branch_dataset.py --runs <dir> [--runs <dir> ...] --anchors <anchors json> \
      --world rigid|crm --map-root artifacts/traverse/crm_f104_v1/map_root --out <npz> \
      [--merge <mixed npz> --merged-out <npz>]      # append the branch rows to the mixed file for retraining
"""
import argparse, glob, json, os, sys, time, zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f104_n2_dataset as DS
from ga_build_mixed import HIST_STATE_COLS, PRIV_NAMES, DOMAIN_CODE, blacklisted, T as HT
from gc_control import route_sha256

DT, NS = 0.05, 96
GRACE_M, GRACE_FRAMES, NS_RUN = 1.0, 60, 20
EXTRA_KEYS = ('cls', 'branch_run')
ROW_DTYPES = dict(profile=np.int8, fail=np.int8, unsafe=np.int8, event_idx=np.int16, route_len=np.float32, anchor_frame=np.int32,
                  vx_anchor=np.float32, rem_m=np.float32, time_to_event_s=np.float32, domain=np.int8)
STR_KEYS = ('id', 'episode', 'group', 'split', 'source', 'status') + EXTRA_KEYS
G = {}


def init(map_root, anchors, world):
    DS.init_map(map_root); G['anchors'] = anchors; G['world'] = world


def anchor_episode(rid):
    """'<episode>__a4__c2' (rigid) or '<episode>__c2' (CRM) -> '<episode>'."""
    ep = rid.rsplit('__c', 1)[0]
    return ep[:-4] if ep.endswith('__a4') else ep


def branch_route(d, r, o):
    """(waypoints, speeds, stations, headings, meta, content sha) of the route driven after the swap."""
    for pre in ('branch_', 'branch_reference_'):
        if pre + 'waypoints' in r.files:
            wp, sp, st, hd = (np.asarray(r[pre + k], float) for k in ('waypoints', 'speeds', 'stations', 'headings'))
            break
    else:
        f = next((d + '/' + n for n in ('branch_route.json', 'branch_reference.json') if os.path.isfile(d + '/' + n)), None)
        if f is None: return None
        j = json.load(open(f))
        wp, sp, st, hd = (np.asarray(j[k], float) for k in ('waypoints', 'speeds', 'stations', 'headings'))
    meta = {}
    for n in ('branch_route.json', 'branch_reference.json'):
        if os.path.isfile(d + '/' + n):
            meta = json.load(open(d + '/' + n)).get('meta', {}) or {}; break
    exp = o.get('branch_route_sha256') or (o.get('branch') or {}).get('branch_route_content_sha256')
    return wp, sp, st, hd, meta, route_sha256(dict(waypoints=wp, speeds=sp, stations=st, headings=hd)), exp


def prefix_hist(state, act, F):
    t = np.arange(HT); si, ai = F - (HT - 1) + t, F - HT + t; ok = ai >= 0
    hist = np.zeros((HT, 15), np.float32); hist[ok] = np.concatenate([state[si[ok]][:, HIST_STATE_COLS], act[ai[ok]]], 1)
    return hist, ok


def one(d):
    rid = os.path.basename(d); world = G['world']
    drop = lambda why, **kw: dict(drop=why, run=rid, episode=anchor_episode(rid), **kw)
    if os.path.isfile(d + '/skipped.json'):
        return drop('skipped: ' + str(json.load(open(d + '/skipped.json')).get('reason', ''))[:60])
    try:
        o = json.load(open(d + '/outcome.json')); z = np.load(d + '/trajectory.npz'); r = np.load(d + '/command_reference.npz'); c = json.load(open(d + '/case.json'))
    except Exception as e:
        return drop('incomplete', detail=type(e).__name__)
    if 'branch_frame' not in z.files: return drop('no_branch_info')
    if 'branch_reached' in z.files and not bool(z['branch_reached']): return drop('branch_not_reached', status=o['status'])
    F = int(z['branch_frame']); bpose = np.asarray(z['branch_pose'], float)
    if not np.all(np.isfinite(bpose)): return drop('branch_pose_not_finite')
    an = G['anchors'].get(anchor_episode(rid))
    if an is None: return drop('anchor_not_found')
    if int(an['F']) != F: return drop('branch_frame_mismatch', F=F, anchor_F=int(an['F']))
    state = z['state'].astype(np.float32); pose = z['pose'].astype(float); act = z['action'].astype(float); n = len(state)
    if n <= F + 1: return drop('no_post_branch_frames', frames=n, F=F, status=o['status'])
    br = branch_route(d, r, o)
    if br is None: return drop('no_branch_route')
    wp, sp, st, hd, meta, sha, exp = br
    if exp is not None and sha != exp: return drop('route_hash_mismatch')
    assert act.shape == (n, 3) and state.shape[1] == 17, (rid, state.shape, act.shape)
    # ---- history window from the prefix (asserted equal to the collector's copy)
    hist, hmask = prefix_hist(state, act, F)
    if 'branch_hist' in z.files:
        assert np.allclose(hist, z['branch_hist'], atol=1e-6) and np.array_equal(hmask, z['branch_hmask']), f'{rid}: branch_hist differs from the trajectory prefix'
    # ---- post-branch projection onto the branch route, grace window, events
    pF = pose[F]; vx = state[:, 0].astype(float); thr = act[:, 1]
    s, dev, L = DS.project(pose[F:, :2], wp); smax = np.maximum.accumulate(s)
    dist = np.linalg.norm(pose[F:, :2] - pF[:2], axis=1)
    far = np.flatnonzero(dist >= GRACE_M)
    g_rel = min(int(far[0]) if len(far) else n, GRACE_FRAMES)
    grace_by = '1m' if len(far) and far[0] <= GRACE_FRAMES else ('3s' if n - F > GRACE_FRAMES else 'end')
    g = F + g_rel
    back = (vx < -0.10) & (thr > 0.3)
    rb = DS.first_run(back | (vx < -0.30), 1, g); ns = DS.first_run((np.abs(vx) < 0.3) & (thr > 0.3), NS_RUN, g)
    fail = o['status'] != 'goal_reached'
    cands = [x for x in (rb, ns) if x is not None]; ev = min(cands) if cands else (n - 1 if fail else None)
    unsafe = fail or bool(cands)
    if ev is None: eidx = -1
    else:
        eidx = int(np.clip(round(float(smax[ev - F]) / max(L, 1e-6) * (NS - 1)), 0, NS - 1))
    # ---- corridor, energy / time targets
    X, L2 = DS.station_tensor(wp, sp, st)
    cumE = np.concatenate([[0.0], np.cumsum(np.asarray(z['positive_work_kj_per_interval'], float))])
    grid = np.linspace(0.0, L2, NS); E = np.full(NS, np.nan, np.float32); Tt = np.full(NS, np.nan, np.float32); reach = float(smax[-1])
    for j, gj in enumerate(grid):
        if gj > reach + 1e-9: break
        a = F + min(int(np.searchsorted(smax, gj, side='left')), n - F - 1)
        E[j] = cumE[a + 1] - cumE[F]; Tt[j] = (a + 1 - F) * DT
    goal = np.asarray(c['goal_xy'], float); rel = goal - pF[:2]
    ctx = np.concatenate([state[F], [rel[0], rel[1], np.linalg.norm(rel), pF[2], L2]]).astype(np.float32)
    # ---- privileged teacher context
    w = slice(max(0, F - (HT - 1)), F + 1); priv = np.zeros(8, np.float32)
    priv[0:4] = state[w, 7:11].mean(0); priv[4] = state[w, 16].mean()
    if world == 'crm':
        e = np.load(d + '/crm_extra.npz')
        slip = e['slip_ratio'].astype(np.float32).mean(1); sink = (e['spindle_z_m'].astype(np.float32) - e['bmp_ground_z_m'].astype(np.float32)[:, None]).mean(1)
        assert len(slip) == n, (rid, n, len(slip))
        priv[5] = slip[w].mean(); priv[6] = sink[w].mean(); priv[7] = 1.0
    row = dict(X=X.astype(np.float16), ctx=ctx, E=E, T=Tt, id=f'{rid}@{world}', episode=an['episode'], group=an['group'], split=an['split'],
               source='branch', status=o['status'], profile=-1, fail=int(fail), unsafe=int(unsafe), event_idx=eidx, route_len=np.float32(L2),
               anchor_frame=F, vx_anchor=np.float32(vx[F]), rem_m=np.float32(L2), time_to_event_s=np.float32((ev - F) * DT if ev is not None else -1.0),
               domain=DOMAIN_CODE[world], hist=hist.astype(np.float16), hmask=hmask, privileged=priv, cls=an['cls'], branch_run=rid)
    diag = dict(run=rid, episode=an['episode'], cls=an['cls'], split=an['split'], cont=int(rid.rsplit('__c', 1)[1]), status=o['status'], fail=int(fail), unsafe=int(unsafe),
                grace_s=g_rel * DT, grace_by=grace_by, post_frames=n - F, rb_s=(rb - F) * DT if rb is not None else None, ns_s=(ns - F) * DT if ns is not None else None,
                first_event='rb' if (rb is not None and (ns is None or rb <= ns)) else ('ns' if ns is not None else ('fail_end' if fail else None)),
                event_idx=eidx, ns_only_goal=int((not fail) and rb is None and ns is not None), L=float(L2), reach_frac=float(reach / max(L2, 1e-6)),
                dev_at_branch=float(dev[0]), start_offset=float(np.linalg.norm(wp[0] - pF[:2])), bpose_vs_poseF=float(np.linalg.norm(bpose[:2] - pF[:2])),
                vx_F=float(vx[F]), vx_F_recorded=float(an.get('vx_F', np.nan)), pose_F_vs_recorded=float(np.linalg.norm(np.asarray(an['pose_F'][:2]) - pF[:2])),
                base_speed=meta.get('base_speed_mps'), hash_checked=exp is not None, route_len_vs_anchor_rem=float(L2 - float(an.get('remaining_m', np.nan))))
    return dict(row=row, diag=diag)


def npz_schema(path):
    """{key: (shape, dtype)} from the npz member headers (nothing decompressed)."""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            with zf.open(name) as f:
                v = np.lib.format.read_magic(f)
                shape, _, dt = (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(f)
            out[name[:-4]] = (shape, dt)
    return out


def pct(v, q=(10, 50, 90)):
    v = np.asarray([x for x in v if x is not None], float)
    return np.percentile(v, q).round(3).tolist() if len(v) else None


def build(a):
    anchors = {r['episode']: r for r in json.load(open(a.anchors))}
    assert all(r['world'] == a.world for r in anchors.values()), 'anchors file is for another world'
    dirs = sorted(d for pat in a.runs for d in glob.glob(pat + '/*') if os.path.isdir(d) and '__c' in os.path.basename(d) and os.path.basename(d).rsplit('__c', 1)[1].isdigit())
    if a.limit: dirs = dirs[:a.limit]
    print(f'{len(dirs)} continuation dirs under {a.runs}; {len(anchors)} anchors ({a.world})', flush=True)
    # anchors with no summary yet (rigid: anchor dir without branch_auto.json / skipped.json) -> still running or not synced
    anchor_dirs = [d for pat in a.runs for d in glob.glob(pat + '/*__a4') if os.path.isdir(d)]
    pending = sorted(os.path.basename(d) for d in anchor_dirs if not (os.path.isfile(d + '/branch_auto.json') or os.path.isfile(d + '/skipped.json')))
    anchor_skips = Counter(json.load(open(d + '/skipped.json')).get('reason', '')[:50] for d in anchor_dirs if os.path.isfile(d + '/skipped.json'))
    rows, diags, drops = [], [], []
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.map_root, anchors, a.world)) as ex:
        for res in ex.map(one, dirs, chunksize=8):
            (drops if 'drop' in res else rows).append(res if 'drop' in res else res['row'])
            if 'diag' in res: diags.append(res['diag'])
    print(f'{len(rows)} rows, {len(drops)} dropped in {time.time() - t0:.0f} s', flush=True)
    assert rows, 'no rows'
    A = lambda k, dt=None: np.array([r[k] for r in rows], dtype=dt)
    out = {k: np.stack([r[k] for r in rows]) for k in ('X', 'ctx', 'E', 'T', 'hist', 'hmask', 'privileged')}
    for k in STR_KEYS: out[k] = A(k, object)
    for k, dt in ROW_DTYPES.items(): out[k] = A(k, dt)
    out['hist_cols'] = np.asarray(HIST_STATE_COLS, np.int16); out['priv_names'] = np.asarray(PRIV_NAMES, object)
    ids = out['id'].astype(str); assert len(set(ids)) == len(ids), 'ids not unique'
    bad = [s for s in set(ids) | set(out['group'].astype(str)) | set(out['episode'].astype(str)) if blacklisted(s)]
    assert not bad, f'blacklisted ids/groups present: {bad[:10]}'
    if a.schema_ref and os.path.isfile(a.schema_ref):
        ref = npz_schema(a.schema_ref); mine = {k: (v.shape, v.dtype) for k, v in out.items()}
        assert set(mine) - set(EXTRA_KEYS) == set(ref), f'key set differs from {a.schema_ref}: missing {set(ref) - set(mine)}, extra {set(mine) - set(ref) - set(EXTRA_KEYS)}'
        for k, (shp, dt) in ref.items():
            assert mine[k][1] == dt, f'{k}: dtype {mine[k][1]} != {dt}'
            assert (mine[k][0] == shp) if k in ('hist_cols', 'priv_names') else (mine[k][0][1:] == shp[1:]), f'{k}: shape {mine[k][0]} vs {shp}'
        print(f'schema matches {a.schema_ref} (+ {EXTRA_KEYS})', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    # ---- report
    cls_v = out['cls'].astype(str); sp_v = out['split'].astype(str); fail = out['fail'].astype(int); uns = out['unsafe'].astype(int)
    counts = {}
    for cl in sorted(set(cls_v)):
        for sp in ('train', 'val', 'test'):
            m = (cls_v == cl) & (sp_v == sp)
            if m.any(): counts[f'{a.world}|{cl}|{sp}'] = dict(rows=int(m.sum()), fail=int(fail[m].sum()), unsafe=int(uns[m].sum()), anchors=len(set(out['episode'][m].astype(str))))
    per_anchor = defaultdict(list)
    for dg in diags: per_anchor[dg['episode']].append(dg)
    disc = {}
    for cl in sorted(set(cls_v)):
        full = [v for v in per_anchor.values() if v[0]['cls'] == cl and len(v) == 3]; part = [v for v in per_anchor.values() if v[0]['cls'] == cl and len(v) >= 2]
        disc[cl] = dict(anchors_with_3=len(full), fail_discordant_of_3=sum(len({x['fail'] for x in v}) > 1 for v in full),
                        unsafe_discordant_of_3=sum(len({x['unsafe'] for x in v}) > 1 for v in full),
                        fail_discordance_share_3=round(np.mean([len({x['fail'] for x in v}) > 1 for v in full]), 4) if full else None,
                        unsafe_discordance_share_3=round(np.mean([len({x['unsafe'] for x in v}) > 1 for v in full]), 4) if full else None,
                        anchors_with_2plus=len(part), fail_discordance_share_2plus=round(np.mean([len({x['fail'] for x in v}) > 1 for v in part]), 4) if part else None,
                        all_fail_of_3=sum(all(x['fail'] for x in v) for v in full), all_clean_fail_of_3=sum(not any(x['fail'] for x in v) for v in full),
                        fail_rate=round(float(fail[cls_v == cl].mean()), 4), unsafe_rate=round(float(uns[cls_v == cl].mean()), 4))
    grace = {}
    for cl in sorted(set(cls_v)):
        v = [dg for dg in diags if dg['cls'] == cl]
        grace[cl] = dict(n=len(v), grace_s_p10_50_90=pct([x['grace_s'] for x in v]), grace_s_mean=round(float(np.mean([x['grace_s'] for x in v])), 3),
                         ended_by=dict(Counter(x['grace_by'] for x in v)), post_frames_p10_50_90=pct([x['post_frames'] for x in v]),
                         first_event=dict(Counter(str(x['first_event']) for x in v)), rb_s_p10_50_90=pct([x['rb_s'] for x in v]), ns_s_p10_50_90=pct([x['ns_s'] for x in v]),
                         ns_only_goal_reached_rows=int(sum(x['ns_only_goal'] for x in v)), event_idx_p10_50_90=pct([x['event_idx'] for x in v if x['event_idx'] >= 0]),
                         event_within_1s_of_grace_end=int(sum(1 for x in v if x['first_event'] in ('rb', 'ns') and min(t for t in (x['rb_s'], x['ns_s']) if t is not None) - x['grace_s'] < 1.0)),
                         vx_F_p10_50_90=pct([x['vx_F'] for x in v]), base_speed=dict(Counter(str(x['base_speed']) for x in v)))
    checks = dict(start_offset_m_max=round(max(x['start_offset'] for x in diags), 4), dev_at_branch_m_p50_max=[round(float(np.median([x['dev_at_branch'] for x in diags])), 4), round(max(x['dev_at_branch'] for x in diags), 4)],
                  branch_pose_vs_pose_F_m_max=round(max(x['bpose_vs_poseF'] for x in diags), 6), pose_F_vs_recorded_m_p50_max=[round(float(np.median([x['pose_F_vs_recorded'] for x in diags])), 4), round(max(x['pose_F_vs_recorded'] for x in diags), 4)],
                  vx_F_vs_recorded_abs_max=round(max(abs(x['vx_F'] - x['vx_F_recorded']) for x in diags if np.isfinite(x['vx_F_recorded'])), 4),
                  route_hash_checked_rows=int(sum(x['hash_checked'] for x in diags)), route_len_minus_anchor_remaining_m_p10_50_90=pct([x['route_len_vs_anchor_rem'] for x in diags if np.isfinite(x['route_len_vs_anchor_rem'])]),
                  reach_frac_p10_50_90=pct([x['reach_frac'] for x in diags]), hmask_full_rows=int(out['hmask'].all(1).sum()), hmask_partial_rows=int((~out['hmask'].all(1)).sum()),
                  statuses=dict(Counter(out['status'].astype(str))))
    rep = dict(out=os.path.abspath(a.out), world=a.world, runs=a.runs, anchors=a.anchors, rows=len(rows), anchors_with_rows=len(per_anchor), anchors_in_file=len(anchors),
               groups=len(set(out['group'].astype(str))), counts=counts, label_discordance=disc, grace=grace, checks=checks,
               dropped=dict(Counter(d['drop'] for d in drops)), dropped_runs=[d['run'] for d in drops][:100], anchors_skipped_by_collector=dict(anchor_skips),
               anchors_skipped_total=sum(anchor_skips.values()), anchors_pending_no_summary=len(pending), anchors_pending_list=pending[:50],
               rules=dict(grace='first post-branch frame >= 1 m from pose_F or 60 frames (3 s), whichever first; clock starts at that frame',
                          rb='first frame >= grace with (vx < -0.10 and throttle > 0.3) or vx < -0.30', ns='first run of 20 frames >= grace with |vx| < 0.3 and throttle > 0.3',
                          fail='outcome status != goal_reached', unsafe='fail or rb or ns (night-2 would ignore ns when the goal is reached without rollback)',
                          event_idx='station of the first event on the branch route via cumulative-max projection; furthest station for a failed drive without explicit event; -1 if clean'),
               size_gb=round(os.path.getsize(a.out) / 1e9, 3), elapsed_s=round(time.time() - t0, 1))
    rp = os.path.splitext(a.out)[0] + '_report.json'
    json.dump(rep, open(rp, 'w'), indent=1)
    print(json.dumps({k: rep[k] for k in ('rows', 'anchors_with_rows', 'counts', 'label_discordance', 'dropped', 'anchors_skipped_total', 'anchors_pending_no_summary')}, indent=1))
    print(f'wrote {a.out} ({rep["size_gb"]} GB) and {rp}', flush=True)
    return out


def merge(a, branch=None):
    t0 = time.time()
    b = branch if branch is not None else dict(np.load(a.out, allow_pickle=True))
    m = np.load(a.merge, allow_pickle=True); mk = set(m.files)
    assert set(b) - set(EXTRA_KEYS) == mk, f'key sets differ: mixed-only {mk - set(b)}, branch-only {set(b) - set(EXTRA_KEYS) - mk}'
    nb, nm = len(b['id']), len(m['id']); out = {}
    for k in m.files:
        mv = m[k]; bv = b[k]
        if k in ('hist_cols', 'priv_names'):
            assert np.array_equal(mv.astype(str) if mv.dtype == object else mv, bv.astype(str) if bv.dtype == object else bv), f'{k} differs'
            out[k] = mv; continue
        assert mv.dtype == bv.dtype and mv.shape[1:] == bv.shape[1:] and len(mv) == nm and len(bv) == nb, (k, mv.dtype, bv.dtype, mv.shape, bv.shape)
        out[k] = np.concatenate([mv, bv]); del mv
    for k in EXTRA_KEYS:
        out[k] = np.concatenate([np.array([''] * nm, object), b[k].astype(object)])
    ids = out['id'].astype(str); assert len(set(ids)) == nm + nb, 'ids not unique after the merge'
    os.makedirs(os.path.dirname(os.path.abspath(a.merged_out)), exist_ok=True)
    (np.savez if a.no_compress else np.savez_compressed)(a.merged_out, **out)
    info = dict(merged_out=os.path.abspath(a.merged_out), mixed=a.merge, branch=a.out, rows_mixed=nm, rows_branch=nb, rows=nm + nb, keys=sorted(out),
                size_gb=round(os.path.getsize(a.merged_out) / 1e9, 3), elapsed_s=round(time.time() - t0, 1))
    json.dump(info, open(os.path.splitext(a.merged_out)[0] + '_merge.json', 'w'), indent=1)
    print(json.dumps(info, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', action='append', default=[], help='dir holding the continuation run dirs (repeatable)')
    ap.add_argument('--anchors', help='anchors_<world>.json from ga_branch_anchors.py'); ap.add_argument('--world', choices=('crm', 'rigid'))
    ap.add_argument('--map-root', default='artifacts/traverse/crm_f104_v1/map_root'); ap.add_argument('--out', required=True)
    ap.add_argument('--schema-ref', default='artifacts/traverse/generalist_20260921/A_adapt/datasets/mixed_reanchor.npz', help='npz whose key set / dtypes the output must match ("" to skip)')
    ap.add_argument('--merge', help='mixed npz to append the branch rows to'); ap.add_argument('--merged-out'); ap.add_argument('--no-compress', action='store_true')
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--limit', type=int, default=0, help='self-test: first N continuation dirs')
    a = ap.parse_args()
    built = None
    if a.runs:
        assert a.anchors and a.world, '--anchors and --world are needed to build'
        built = build(a)
    else:
        assert os.path.isfile(a.out), '--runs missing and --out does not exist: nothing to build or merge'
    if a.merge:
        assert a.merged_out, '--merge needs --merged-out'
        merge(a, built)


if __name__ == '__main__':
    main()
