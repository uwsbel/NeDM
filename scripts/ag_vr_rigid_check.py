#!/usr/bin/env python3
"""Independent check of the rigid evaluation results (arena_gator_20260925, E6b final; verifier's own code).

Reads only the raw per-drive files (outcome.json, trajectory.npz, case.json in e6/runs_rigid), the five mapping and
task files, the pick directories and locks, the pool shard records (e6/pool/done) and a login-node extraction of every
run's simulation_provenance host, collection_request.json mtime (drive start) and reference.json sha256
(artifacts/traverse/arena_gator_20260925/verify_rigid/cluster_runs.json, made by scripts/ag_vr_rigid_cluster_extract.py over ssh, read-only). It does NOT import
ag_eval_index / ag_analyze / ga_analyze / f104_n2_analyze: labels, clusters, bootstraps, Holm and McNemar are
re-implemented here.

  PYTHONPATH=src:scripts python scripts/ag_vr_rigid_check.py --out artifacts/traverse/arena_gator_20260925/verify_rigid/check.json
"""
import argparse, glob, hashlib, json, os, sys, time
from math import comb
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K3 = ROOT / 'artifacts/traverse/arena_gator_20260925'
SETS = ['unseen', 'f104', 'f104B2', 'heldout', 'dev']
NEAR = ['g260', 'g271', 'g251', 'g247']
SPREAD = ['g258', 'g268', 'g263', 'g241']
SETTLE = 20  # 1 s at 0.05 s per recorded interval


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


# ---------------------------------------------------------------- labels (own implementation of the declared rule)
def my_labels(run):
    o = json.load(open(run / 'outcome.json'))
    z = np.load(run / 'trajectory.npz')
    st, act = z['state'], z['action']
    fail = o['status'] != 'goal_reached'
    assert bool(o['goal_reached']) == (not fail), run
    if len(st) <= SETTLE + 1:
        return dict(fail=int(fail), unsafe=1, short=True, status=o['status'], t=float(o['elapsed_s']), back_n=0,
                    back_any=0.0, vmin=0.0, route=o.get('route_sha256'))
    vx = st[SETTLE:, 0]
    thr = act[SETTLE:, 1]
    back_thr = int(np.count_nonzero((vx < -0.10) & (thr > 0.3)))  # intervals rolling back under throttle
    vmin = float(vx.min())
    clean = (not fail) and back_thr == 0 and vmin > -0.30
    return dict(fail=int(fail), unsafe=int(not clean), short=False, status=o['status'], t=float(o['elapsed_s']),
                back_n=back_thr, back_any=float(np.count_nonzero(vx < -0.10) * 0.05), vmin=vmin,
                route=o.get('route_sha256'), vx_only=int((not fail) and vmin <= -0.30 and back_thr == 0),
                thr_only=int((not fail) and back_thr > 0 and vmin > -0.30))


# ---------------------------------------------------------------- clusters (own nearest-feature rule)
_F = {}


def feats(arena_dir):
    if arena_dir not in _F:
        from nedm.traverse.terrain import TerrainMap
        fs = TerrainMap.from_dir(ROOT / arena_dir).features
        _F[arena_dir] = np.array([[f['x_m'], f['y_m']] for f in fs], float)
    return _F[arena_dir]


def my_cluster(case_path):
    c = json.load(open(case_path))
    mid = (np.asarray(c['layout']['start_xy'], float) + np.asarray(c['goal_xy'], float)) / 2
    xy = feats(c['arena'])
    return int(np.argmin(np.hypot(xy[:, 0] - mid[0], xy[:, 1] - mid[1])))


# ---------------------------------------------------------------- statistics (own implementation)
def boot_group(d, B=4000, seed=11):
    rng = np.random.default_rng(seed)
    n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)]) * 100
    return bs


def boot_cluster(d, cl, B=4000, seed=13):
    rng = np.random.default_rng(seed)
    keys = sorted(set(cl))
    idx = {k: np.where(cl == k)[0] for k in keys}
    s = np.array([d[idx[k]].sum() for k in keys]); c = np.array([len(idx[k]) for k in keys], float)
    K = len(keys)
    out = np.empty(B)
    for b in range(B):
        j = rng.integers(0, K, K)
        out[b] = s[j].sum() / c[j].sum()
    return out * 100, K


def mcnemar2(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def contrast(T, ref, test, label, groups, clusters=None, B=4000):
    x = np.array([T[g][test][label] for g in groups], float)
    y = np.array([T[g][ref][label] for g in groups], float)
    d = x - y
    gb = boot_group(d, B)
    out = dict(n=len(groups), rate_test=100 * x.mean(), rate_ref=100 * y.mean(), diff=100 * d.mean(),
               g95=[float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))],
               g90=[float(np.percentile(gb, 5)), float(np.percentile(gb, 95))],
               better=int((d < 0).sum()), worse=int((d > 0).sum()))
    if clusters is not None:
        cl = np.array([clusters[g] for g in groups])
        cb, K = boot_cluster(d, cl, B)
        out.update(c90=[float(np.percentile(cb, 5)), float(np.percentile(cb, 95))],
                   c95=[float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))],
                   p1=float((1 + (cb >= 0).sum()) / (B + 1)), K=K,
                   ub95_one_sided=float(np.percentile(cb, 95)))
    if set(np.unique(x)) <= {0, 1} and set(np.unique(y)) <= {0, 1}:
        out['mcnemar2'] = mcnemar2(int((d > 0).sum()), int((d < 0).sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=str(K3 / 'e6/runs_rigid'))
    ap.add_argument('--cluster-extract', default='artifacts/traverse/arena_gator_20260925/verify_rigid/cluster_runs.json')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    runs = Path(a.runs)
    rep = dict(created=time.strftime('%Y-%m-%d %H:%M'), script_sha256=sha(__file__))
    cx = json.load(open(a.cluster_extract))

    # ---------- mappings, task files, shards
    M, TASK, shard_of, taskfile_of = {}, {}, {}, {}
    for s in SETS:
        tf = K3 / f'e6/tasks/rigid_eval_{s}.json'
        M[s] = json.load(open(str(tf) + '.mapping.json'))
        assert M[s]['tasks_sha256'] == sha(tf), s
        TASK[s] = json.load(open(tf))
        for r in TASK[s]:
            assert r['id'] not in shard_of, r['id']
            shard_of[r['id']] = r['shard']; taskfile_of[r['id']] = s
    rep['n_task_rows'] = len(shard_of)
    rep['n_run_dirs'] = len(os.listdir(runs))
    rep['task_rows_without_run'] = sorted(set(shard_of) - set(os.listdir(runs)))[:20]
    rep['run_dirs_not_in_tasks'] = sorted(set(os.listdir(runs)) - set(shard_of))[:20]

    # ---------- labels for every drive
    cache = Path(a.out).with_suffix('.labels.pkl')
    if cache.exists():
        import pickle; L = pickle.load(open(cache, 'rb'))
    else:
        L = {}
        for rid in sorted(shard_of):
            L[rid] = my_labels(runs / rid)
        import pickle; pickle.dump(L, open(cache, 'wb'))
    rep['n_labelled'] = len(L)
    rep['short_trajectories'] = sum(v['short'] for v in L.values())
    rep['route_sha_outcome_vs_reference_json'] = dict(
        equal=sum(L[r]['route'] == cx[r]['ref'] for r in L), total=len(L))

    # ---------- per (set, group, arm) table + route/pick checks
    T = {s: {} for s in SETS}
    info = {s: {} for s in SETS}
    pick_mismatch, map_vs_outcome, share_bad, merge_missed = [], [], [], []
    phys_bad, n_meta_only = [], 0
    route_file_of_dir = {}
    KEYS = ('waypoints', 'speeds', 'stations', 'headings')

    def phys(f):
        r = json.load(open(f))
        return {k: np.asarray(r[k], float) for k in KEYS}

    def content_sha(f):
        r = json.load(open(f))
        return hashlib.sha256(json.dumps({k: np.asarray(r[k], float).tolist() for k in KEYS}).encode()).hexdigest()
    for s in SETS:
        armdirs = {ar['name']: ar['dirs'] for ar in M[s]['arms']}
        armveh = {ar['name']: ar['vehicle'] for ar in M[s]['arms']}
        for g, arms in M[s]['groups'].items():
            T[s][g] = {}
            arena = g.split('_')[0]
            by_run, by_sha, files = {}, {}, {}
            for arm, e in arms.items():
                rid = e['run_id']
                assert e['vehicle'] == armveh[arm]
                assert rid.startswith('gator__') == (e['vehicle'] == 'gator'), (rid, e['vehicle'])
                dirs = [d for d in armdirs[arm] if f'/{arena}/' in d]
                assert len(dirs) == 1, (s, g, arm, dirs)
                fs = glob.glob(str(ROOT / dirs[0] / 'routes' / f'{g}__*.json'))
                assert len(fs) == 1, (s, g, arm, fs)
                files[arm] = fs[0]
                cs = content_sha(fs[0])
                if cs != e['route_sha256']:
                    pick_mismatch.append((s, g, arm))
                by_run.setdefault(rid, []).append(arm)
                by_sha.setdefault((cs, e['vehicle']), set()).add(rid)
                T[s][g][arm] = L[rid]
                info[s][g] = dict(arena=arena, case=e['case'])
            for rid, arms_ in by_run.items():
                # the file actually driven: its bytes hash = reference.json on the cluster = outcome route_sha256
                drv = [a_ for a_ in arms_ if sha(files[a_]) == cx[rid]['ref'] == L[rid]['route']]
                if not drv:
                    map_vs_outcome.append((s, g, rid)); continue
                P0 = phys(files[drv[0]])
                for a_ in arms_:
                    Pa = phys(files[a_])
                    if not all(np.array_equal(P0[k], Pa[k]) for k in KEYS):
                        phys_bad.append((s, g, a_, rid))
                    elif sha(files[a_]) != cx[rid]['ref']:
                        n_meta_only += 1
            merge_missed += [(s, g, k[0][:8]) for k, v in by_sha.items() if len(v) > 1]
    rep['n_arm_rows_whose_driven_route_differs_physically'] = len(phys_bad)
    rep['arm_rows_whose_driven_route_differs_physically'] = phys_bad[:20]
    rep['n_arm_rows_sharing_a_drive_with_route_differing_only_in_meta'] = n_meta_only
    rep['pick_route_vs_mapping_mismatch'] = pick_mismatch[:20]
    rep['n_pick_route_vs_mapping_mismatch'] = len(pick_mismatch)
    rep['n_driven_route_vs_pick_mismatch'] = len(map_vs_outcome)
    rep['driven_route_vs_pick_mismatch'] = map_vs_outcome[:20]
    rep['identical_routes_driven_twice'] = merge_missed[:20]
    n_rows = sum(len(v) for s in SETS for v in M[s]['groups'].values())
    rep['n_arm_group_rows'] = n_rows
    shared = {}
    for s in SETS:
        for g, arms in M[s]['groups'].items():
            for arm, e in arms.items():
                shared.setdefault((s, g, e['run_id']), []).append(arm)
    rep['n_merged_rows'] = sum(len(v) - 1 for v in shared.values())

    # ---------- locks: recompute each pick dir's lock, compare, time vs first drive
    locks = {}
    for s, lk in [('unseen', 'LOCK_unseen'), ('f104', 'LOCK_f104'), ('heldout', 'LOCK_heldout'), ('dev', 'LOCK_dev'),
                  ('f104B2', 'LOCK_f104B2')]:
        J = json.load(open(K3 / f'e6/picks/{lk}.json'))
        lt = os.stat(K3 / f'e6/picks/{lk}.json').st_mtime
        bad, newest_route = [], 0.0
        for d in J['dirs']:
            h = hashlib.sha256()
            for p in sorted((ROOT / d['dir'] / 'routes').glob('*.json')):
                h.update(p.name.encode()); h.update(hashlib.sha256(p.read_bytes()).digest())
                newest_route = max(newest_route, p.stat().st_mtime)
            own = (ROOT / d['dir'] / 'PICKS_LOCKED.sha256').read_text().split()[0]
            if not (h.hexdigest() == d['picks_locked_sha256'] == own):
                bad.append(d['dir'])
        used_dirs = {x for ar in M[s]['arms'] for x in ar['dirs']}
        listed = {d['dir'] for d in J['dirs']}
        ids = [r['id'] for r in TASK[s]]
        t0_req = min(cx[i]['t0'] for i in ids)
        t0_case = min(os.stat(runs / i / 'case.json').st_mtime for i in ids)
        shards = {r['shard'] for r in TASK[s]}
        dn = [json.load(open(K3 / f'e6/pool/done/{k}.json')) for k in sorted(shards)]
        t0_shard = min(x['start'] for x in dn)
        locks[s] = dict(lock=lk, n_dirs=len(J['dirs']), dirs_bad=bad, used_dirs_not_in_lock=sorted(used_dirs - listed),
                        lock_mtime=time.strftime('%m-%d %H:%M:%S', time.localtime(lt)),
                        newest_route_file=time.strftime('%m-%d %H:%M:%S', time.localtime(newest_route)),
                        first_drive_request=time.strftime('%m-%d %H:%M:%S', time.localtime(t0_req)),
                        first_case_copy=time.strftime('%m-%d %H:%M:%S', time.localtime(t0_case)),
                        first_shard_start=time.strftime('%m-%d %H:%M:%S', time.localtime(t0_shard)),
                        lock_before_first_drive=bool(lt < min(t0_req, t0_case, t0_shard)),
                        routes_before_lock=bool(newest_route <= lt),
                        last_drive_end=time.strftime('%m-%d %H:%M:%S', time.localtime(max(cx[i]['t1'] for i in ids))))
    rep['locks'] = locks
    spec_t = {p: time.strftime('%m-%d %H:%M:%S', time.localtime(os.stat(K3 / 'e6/analysis' / p).st_mtime))
              for p in ['spec_rigid_v1.json', 'spec_rigid_v1_B2.json', 'spec_rigid_v1_addons.json']}
    rep['spec_mtimes'] = spec_t
    rep['spec_sha256'] = {p: sha(K3 / 'e6/analysis' / p)[:16] for p in spec_t}
    rep['first_rigid_eval_drive_any'] = time.strftime('%m-%d %H:%M:%S', time.localtime(min(v['t0'] for v in cx.values())))

    # ---------- hosts: one host per shard, one shard (host) per group within a task file
    host_bad, shard_multi, group_hosts_multi, done_bad = [], [], [], []
    shard_host = {}
    for rid, sh in shard_of.items():
        shard_host.setdefault(sh, set()).add(cx[rid]['h'])
    for sh, hs in shard_host.items():
        dn = json.load(open(K3 / f'e6/pool/done/{sh}.json'))
        if len(hs) != 1:
            host_bad.append(sh)
        if dn['host'].split('.')[0] not in hs or dn['incomplete'] or dn['takeover'] or dn['host_mismatch'] \
                or any(rc != 0 for rc in dn['runner_rc']) or dn['complete'] != dn['rows']:
            done_bad.append(sh)
    multi_across = 0
    for s in SETS:
        g2sh = {}
        for r in TASK[s]:
            g2sh.setdefault(r['group'], set()).add(r['shard'])
        shard_multi += [(s, g) for g, v in g2sh.items() if len(v) > 1]
    g_all = {}
    for s in SETS:
        for r in TASK[s]:
            g_all.setdefault(r['group'], {}).setdefault(s, set()).add(cx[r['id']]['h'])
    for g, v in g_all.items():
        hs = set().union(*v.values())
        if any(len(x) > 1 for x in v.values()):
            group_hosts_multi.append(g)
        elif len(hs) > 1:
            multi_across += 1
    rep['hosts'] = dict(n_shards=len(shard_host), shards_with_more_than_one_host=host_bad, done_record_problems=done_bad,
                        groups_split_over_shards_within_a_task_file=shard_multi[:20],
                        groups_with_two_hosts_within_a_task_file=group_hosts_multi[:20],
                        groups_with_two_hosts_across_task_files=multi_across,
                        n_hosts=len({h for v in cx.values() for h in [v['h']]}))

    # ---------- composites and clusters
    for s in SETS:
        for g, arms in T[s].items():
            for comp, (m1, m2) in {'M1_fx2': ('M1a_fx2', 'M1b_fx2'), 'M3_fx2': ('M3a_fx2', 'M3b_fx2'),
                                   'M1_free': ('M1a_free', 'M1b_free'), 'M3_free': ('M3a_free', 'M3b_free')}.items():
                if m1 in arms and m2 in arms:
                    arms[comp] = {k: (arms[m1][k] + arms[m2][k]) / 2 for k in ('fail', 'unsafe')}
            for arm, v in arms.items():
                v['backward_only'] = v['unsafe'] - v['fail']
    CL = {}
    for s in ['unseen']:
        for g, e in info[s].items():
            CL[g] = f"{e['arena']}:{my_cluster(K3 / e['case'])}"
    idx = json.load(open(K3 / 'e6/index/rigid_eval_v1.json'))
    idx_cl, idx_lab_bad = {}, 0
    for r in idx['rows']:
        if r['set'] == 'unseen':
            idx_cl[r['group']] = r['cluster']
        s_ = {'unseen': 'unseen', 'indist_f104': 'f104', 'f104_suite': 'f104', 'f104_800': 'f104', 'heldout': 'heldout', 'dev': 'dev'}.get(r['set'])
        lab = L.get(r['run_id'])
        if lab is not None and (lab['fail'] != r['fail'] or lab['unsafe'] != r['unsafe']):
            idx_lab_bad += 1
    rep['index_rows'] = len(idx['rows'])
    rep['index_label_disagreements_with_mine'] = idx_lab_bad
    rep['clusters'] = dict(n=len(set(CL.values())), agree_with_index=sum(idx_cl.get(g) == c for g, c in CL.items()),
                           n_groups=len(CL))

    # ---------- rates
    U = sorted(T['unseen'])
    arena_of = {g: info['unseen'][g]['arena'] for g in U}
    rates = {}
    for arm in ['M1a_fx2', 'M1b_fx2', 'M1_fx2', 'M2_fx2', 'M3a_fx2', 'M3b_fx2', 'M3_fx2', 'A3_fx2', 'straight2',
                'M1a_free', 'M1b_free', 'M1_free', 'M2_free', 'M3a_free', 'M3b_free', 'M3_free', 'A3_free', 'straight6']:
        r = {}
        for lab in ('fail', 'unsafe', 'backward_only'):
            r[lab] = 100 * np.mean([T['unseen'][g][arm][lab] for g in U])
            for ar in NEAR + SPREAD:
                r[f'{lab}_{ar}'] = 100 * np.mean([T['unseen'][g][arm][lab] for g in U if arena_of[g] == ar])
            r[f'{lab}_near'] = 100 * np.mean([T['unseen'][g][arm][lab] for g in U if arena_of[g] in NEAR])
            r[f'{lab}_spread'] = 100 * np.mean([T['unseen'][g][arm][lab] for g in U if arena_of[g] in SPREAD])
        r['goal_reached'] = 100 - r['fail']
        rates[arm] = r
    rep['rates_unseen'] = rates

    # ---------- family (rigid P3/P4) + declared secondaries on unseen
    C = {}
    for name, test, ref, lab in [('P3', 'M3_fx2', 'M1_fx2', 'unsafe'), ('P4', 'A3_fx2', 'M1_fx2', 'unsafe'),
                                 ('M3a_vs_M1a_fx2_unsafe', 'M3a_fx2', 'M1a_fx2', 'unsafe'),
                                 ('A3_vs_M1a_fx2_unsafe', 'A3_fx2', 'M1a_fx2', 'unsafe'),
                                 ('A3_vs_M3a_fx2_unsafe', 'A3_fx2', 'M3a_fx2', 'unsafe'),
                                 ('M3b_vs_M1b_fx2_unsafe', 'M3b_fx2', 'M1b_fx2', 'unsafe'),
                                 ('M3a_vs_M1b_fx2_unsafe', 'M3a_fx2', 'M1b_fx2', 'unsafe'),
                                 ('M3b_vs_M1a_fx2_unsafe', 'M3b_fx2', 'M1a_fx2', 'unsafe'),
                                 ('A3_vs_M1b_fx2_unsafe', 'A3_fx2', 'M1b_fx2', 'unsafe'),
                                 ('M2_vs_M1_fx2_unsafe', 'M2_fx2', 'M1_fx2', 'unsafe'),
                                 ('M3_vs_M2_fx2_unsafe', 'M3_fx2', 'M2_fx2', 'unsafe'),
                                 ('A3_vs_M3_fx2_unsafe', 'A3_fx2', 'M3_fx2', 'unsafe'),
                                 ('M3_vs_M1_fx2_fail', 'M3_fx2', 'M1_fx2', 'fail'),
                                 ('A3_vs_M1_fx2_fail', 'A3_fx2', 'M1_fx2', 'fail'),
                                 ('M3_vs_M1_fx2_backonly', 'M3_fx2', 'M1_fx2', 'backward_only'),
                                 ('A3_vs_M1_fx2_backonly', 'A3_fx2', 'M1_fx2', 'backward_only'),
                                 ('seed_M1a_vs_M1b_fx2', 'M1a_fx2', 'M1b_fx2', 'unsafe'),
                                 ('seed_M3a_vs_M3b_fx2', 'M3a_fx2', 'M3b_fx2', 'unsafe'),
                                 ('free_M3_vs_M1_fail', 'M3_free', 'M1_free', 'fail'),
                                 ('free_A3_vs_M1_fail', 'A3_free', 'M1_free', 'fail'),
                                 ('free_M3_vs_M1_unsafe', 'M3_free', 'M1_free', 'unsafe'),
                                 ('free_A3_vs_M1_unsafe', 'A3_free', 'M1_free', 'unsafe')]:
        C[name] = contrast(T['unseen'], ref, test, lab, U, CL)
        per = {}
        for ar in NEAR + SPREAD:
            gg = [g for g in U if arena_of[g] == ar]
            x = np.array([T['unseen'][g][test][lab] - T['unseen'][g][ref][lab] for g in gg])
            gb = boot_group(x)
            per[ar] = [100 * x.mean(), float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))]
        C[name]['per_arena'] = per
        C[name]['per_arena_ci_excludes_0'] = sum(1 for v in per.values() if v[2] < 0)
        C[name]['per_arena_negative'] = sum(1 for v in per.values() if v[0] < 0)
    rep['contrasts_unseen'] = C

    # soil p-values from the soil results file (not recomputed here) + my rigid p -> Holm
    soil = json.load(open(K3 / 'e6/analysis/results_soil_v1.json'))
    sp = {f['name']: f['result']['cluster']['p_one_sided'] for f in soil['family'] if f['result'].get('n')}
    rigid_res = json.load(open(K3 / 'e6/analysis/results_rigid_v1.json'))
    rp = {f['name']: f['result']['cluster']['p_one_sided'] for f in rigid_res['family'] if f['result'].get('n')}
    fam = json.load(open(K3 / 'e6/analysis/family_final_E6b.json'))
    names = ['P1_soil_M3_vs_M1', 'P2_soil_A3_vs_M1', 'P3_rigid_fx2_M3_vs_M1', 'P4_rigid_fx2_A3_vs_M1']
    ptool = np.array([sp.get(names[0], np.nan), sp.get(names[1], np.nan), rp[names[2]], rp[names[3]]])
    pmine = np.array([ptool[0], ptool[1], C['P3']['p1'], C['P4']['p1']])
    rep['holm'] = dict(p_tool=ptool.tolist(), holm_of_tool_p=holm(ptool).tolist(), p_mine_rigid=pmine[2:].tolist(),
                       holm_with_my_rigid_p=holm(pmine).tolist(),
                       rigid_only_soil_at_1=holm(np.array([1, 1, ptool[2], ptool[3]])).tolist(),
                       family_file=fam)

    # ---------- backward-only character
    bk = {}
    for arm in ['M1a_fx2', 'M1b_fx2', 'M2_fx2', 'M3a_fx2', 'M3b_fx2', 'A3_fx2']:
        v = [T['unseen'][g][arm] for g in U if T['unseen'][g][arm]['unsafe'] and not T['unseen'][g][arm]['fail']]
        bk[arm] = dict(n=len(v), median_back_s=float(np.median([x['back_any'] for x in v])),
                       median_peak_back=float(np.median([-x['vmin'] for x in v])),
                       frac_peak_gt1=float(np.mean([-x['vmin'] > 1.0 for x in v])),
                       throttle_clause_only=int(sum(x['thr_only'] for x in v)),
                       speed_clause_only=int(sum(x['vx_only'] for x in v)))
    rep['backward_only_unseen_fx2'] = bk

    # ---------- other sets
    F200 = [g for g in T['f104'] if 'M3a_fx2' in T['f104'][g]]
    indist = json.load(open(K3 / 'suites/f104_indist_200.json'))
    indist_ids = set(indist['groups'])
    rep['f104_task_A_groups_equal_declared_200'] = (set(F200) == indist_ids, len(F200))
    O = {}
    O['f104_noharm_M3_fx2'] = contrast(T['f104'], 'M1_fx2', 'M3_fx2', 'unsafe', F200)
    O['f104_noharm_A3_fx2'] = contrast(T['f104'], 'M1_fx2', 'A3_fx2', 'unsafe', F200)
    O['f104_M1_fx2_unsafe'] = 100 * np.mean([T['f104'][g]['M1_fx2']['unsafe'] for g in F200])
    O['f104_straight2_unsafe'] = 100 * np.mean([T['f104'][g]['straight2']['unsafe'] for g in F200])
    HO = sorted(T['heldout'])
    O['heldout_M3_fx2'] = contrast(T['heldout'], 'M1_fx2', 'M3_fx2', 'unsafe', HO)
    O['heldout_A3_fx2'] = contrast(T['heldout'], 'M1_fx2', 'A3_fx2', 'unsafe', HO)
    DV = sorted(T['dev'])
    O['dev_M3_fx2'] = contrast(T['dev'], 'M1_fx2', 'M3_fx2', 'unsafe', DV)
    O['dev_A3_fx2'] = contrast(T['dev'], 'M1_fx2', 'A3_fx2', 'unsafe', DV)
    # task B
    F8 = sorted(T['f104'])
    B2 = sorted(T['f104B2'])
    assert set(F8) == set(B2) and len(F8) == 800
    tb = {}
    for arm in ['G_free_gator', 'H_free_gator', 'straight6_gator', 'M1a_free', 'straight6']:
        tb[arm] = {lab: 100 * np.mean([T['f104'][g][arm][lab] for g in F8]) for lab in ('fail', 'unsafe', 'backward_only')}
    for arm in ['G_fx2_gator', 'H_fx2_gator', 'straight2_gator']:
        tb[arm] = {lab: 100 * np.mean([T['f104B2'][g][arm][lab] for g in B2]) for lab in ('fail', 'unsafe', 'backward_only')}
    rep['taskB_rates'] = tb
    rep['taskB_fx2_G_vs_H_unsafe'] = contrast(T['f104B2'], 'H_fx2_gator', 'G_fx2_gator', 'unsafe', B2)
    rep['taskB_fx2_G_vs_H_fail'] = contrast(T['f104B2'], 'H_fx2_gator', 'G_fx2_gator', 'fail', B2)
    rep['taskB_fx2_G_vs_straight2_unsafe'] = contrast(T['f104B2'], 'straight2_gator', 'G_fx2_gator', 'unsafe', B2)
    rep['taskB_free_G_vs_straight6_fail'] = contrast(T['f104'], 'straight6_gator', 'G_free_gator', 'fail', F8)
    # H picks on the Gator = M1a picks on the HMMWV (same model, same planner)
    same = sum(M['f104']['groups'][g]['H_free_gator']['route_sha256'] == M['f104']['groups'][g]['M1a_free']['route_sha256'] for g in F8)
    same2 = sum(M['f104B2']['groups'][g]['H_fx2_gator']['route_sha256'] == M['f104']['groups'][g]['M1a_fx2']['route_sha256'] for g in F200)
    rep['H_gator_picks_equal_M1a'] = dict(free=same, free_of=len(F8), fx2_on_200=same2)
    rep['other_sets'] = O
    json.dump(rep, open(a.out, 'w'), indent=1, default=float)
    print(json.dumps({k: rep[k] for k in rep if k not in ('contrasts_unseen', 'rates_unseen', 'other_sets')}, indent=1,
                     default=float)[:20000])


if __name__ == '__main__':
    main()
