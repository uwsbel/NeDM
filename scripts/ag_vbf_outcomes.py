#!/usr/bin/env python3
"""VERIFY_gator_full (independent checker of task B stage 2): closed-loop outcomes recomputed from the raw run records.

No study tool is imported. Inputs: the locked pick directories (route files), the run records extracted read-only on the
cluster (scripts/ag_vbf_cluster.py runs -> runs_suite.jsonl: every run folder of an f104 suite group), the local copies of
outcome.json in e6/runs_soil (compared with the cluster hashes), the suite cases (start/goal, stratum) and the f104
TerrainMap features (for the terrain clusters), the task files soil_v3 / soil_v4.

Steps: (1) re-derive every pick-lock hash with my own code; (2) find each arm's drive per group by route CONTENT
(sha256 of the locked route file == sha256 of the run's reference.json, same group, same vehicle), independently of the
study's mapping files, then compare with the study index; (3) provenance of each used drive (collector, wrapper, config,
case, vehicle block, completion) and timing (spec / lock / first new drive); (4) rates, strata, in-distribution set,
paired contrasts with exact McNemar, my own group and cluster bootstraps (seed 12345, 4,000 draws), Holm over F1-F4,
cluster win / loss counts, headroom and time ratios, pick overlap, mean predicted risk.
  PYTHONPATH=src:scripts python scripts/ag_vbf_outcomes.py <K3> <out json>"""
import sys, os, json, hashlib, glob, datetime
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

K3 = Path(sys.argv[1]); OUT = sys.argv[2]
ROOT = K3.parents[2]
VB = K3 / 'verify_gator_full'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


ROUTE_FIELDS = ('waypoints', 'speeds', 'stations', 'headings')


def content_sha(path):
    """route content = the four driven fields (meta excluded), canonical JSON (same rule as ag_vbf_cluster.py)"""
    r = json.load(open(path))
    return hashlib.sha256(json.dumps({k: r.get(k) for k in ROUTE_FIELDS}, sort_keys=True).encode()).hexdigest()


def ts(t):
    return datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')


res = {}
# ---------------------------------------------------------------- (1) locks
ARMS = [  # name, pick dir, vehicle
    ('Gfull_free_gator', 'e6/picks/crm_bfull/f104/G_full_free', 'gator'),
    ('Hfull_free_gator', 'e6/picks/crm_bfull/f104/H_full_free', 'gator'),
    ('G_free_gator', 'e6/picks/crm/f104/G_free', 'gator'),
    ('H_free_gator', 'e6/picks/crm/f104/M1a_free', 'gator'),
    ('straight6_gator', 'e6/picks/crm/f104/straight6', 'gator'),
    ('Hfull_free', 'e6/picks/crm_bfull/f104/H_full_free', 'hmmwv'),
    ('M1a_free', 'e6/picks/crm/f104/M1a_free', 'hmmwv'),
    ('straight6', 'e6/picks/crm/f104/straight6', 'hmmwv'),
]
locks = {}
for d in sorted({a[1] for a in ARMS}):
    p = K3 / d
    h = hashlib.sha256()
    files = sorted((p / 'routes').glob('*.json'))
    for f in files:
        h.update(f.name.encode()); h.update(hashlib.sha256(f.read_bytes()).digest())
    mine = h.hexdigest()
    stored = (p / 'PICKS_LOCKED.sha256').read_text().split()[0]
    locks[d] = dict(mine=mine, stored=stored, equal=mine == stored, n_route_files=len(files), manifest_sha256=sha(p / 'ag_picks.json'))
# set locks: LOCK_crm_bfull (2 dirs) and LOCK_crm (stage-1 dirs)
for lf in ('LOCK_crm_bfull', 'LOCK_crm'):
    lines = (K3 / 'e6/picks' / f'{lf}.sha256').read_text().splitlines()
    body, last = lines[:-1], lines[-1]
    ok = []
    for l in body:
        lk, ms, d = l.split()
        rel = str(Path(d).relative_to('artifacts/traverse/arena_gator_20260925'))
        if rel in locks:
            ok.append(dict(dir=rel, lock_equal=lk == locks[rel]['mine'], manifest_equal=ms == locks[rel]['manifest_sha256']))
    allh = hashlib.sha256(('\n'.join(body) + '\n').encode()).hexdigest()
    res.setdefault('locks', {})[lf] = dict(all_mine=allh, all_stored=last.split()[0], all_equal=allh == last.split()[0], dirs_checked=ok,
                                          mtime=ts(os.path.getmtime(K3 / 'e6/picks' / f'{lf}.sha256')))
res['pick_dirs'] = locks
spec = K3 / 'e6/analysis/spec_soil_v1_Bfull.json'
res['spec'] = dict(sha256=sha(spec), stored=(K3 / 'e6/analysis/spec_soil_v1_Bfull.sha256').read_text().split()[0], mtime=ts(os.path.getmtime(spec)))
res['spec']['equal'] = res['spec']['sha256'] == res['spec']['stored']

# ---------------------------------------------------------------- picks per arm
picks = {}
for name, d, veh in ARMS:
    m = json.load(open(K3 / d / 'ag_picks.json'))
    pk = {}
    for g, e in m['picks'].items():
        f = K3 / d / e['route_file']
        pk[g] = dict(bytes=sha(f), sha=content_sha(f), P=e.get('P'))
    picks[name] = pk
groups = sorted(picks['Gfull_free_gator'])
assert len(groups) == 800 and all(sorted(p) == groups for p in picks.values())
gtxt = (K3 / 'e6/picks/crm_bfull/f104/G_full_free/groups.txt').read_text().split()
res['groups'] = dict(n=len(groups), equal_groups_txt=sorted(gtxt) == groups)

# ---------------------------------------------------------------- (2) runs by route content
runs = [json.loads(l) for l in open(VB / 'cluster_out/runs_suite.jsonl')]
by = defaultdict(list)
for r in runs:
    if r.get('reference_content_sha256'):
        by[(r.get('scene'), r.get('vehicle') or 'hmmwv', r['reference_content_sha256'])].append(r)
idx = json.load(open(K3 / 'e6/index/soil_eval_bfull.json'))['rows']
irow = {(r['arm'], r['group']): r for r in idx}
v3 = json.load(open(K3 / 'e3/tasks/soil_v3.json')); v4 = json.load(open(K3 / 'e3/tasks/soil_v4.json'))
v3ids = {r['id'] for r in v3}
new_ids = [r['id'] for r in v4 if r['id'] not in v3ids]
res['tasks'] = dict(soil_v3_sha256=sha(K3 / 'e3/tasks/soil_v3.json'), soil_v4_sha256=sha(K3 / 'e3/tasks/soil_v4.json'), v3_rows=len(v3), v4_rows=len(v4),
                    v4_prefix_equal=[r['id'] for r in v4[:len(v3)]] == [r['id'] for r in v3] and v4[:len(v3)] == v3, new_rows=len(new_ids),
                    new_by_tier=dict(Counter(r['tier'] for r in v4 if r['id'] not in v3ids)),
                    new_by_vehicle=dict(Counter(r.get('vehicle', 'hmmwv') for r in v4 if r['id'] not in v3ids)))
newset = set(new_ids)
chosen, amb, missing, disagree_idx = {}, [], [], []
byte_equal = Counter()
for name, d, veh in ARMS:
    for g in groups:
        c = by.get((g, veh, picks[name][g]['sha']), [])
        if not c:
            missing.append((name, g)); continue
        st = {x.get('status') for x in c}
        if len(c) > 1:
            amb.append(dict(arm=name, group=g, runs=[x['run'] for x in c], statuses=sorted(st)))
        c = sorted(c, key=lambda x: x.get('req_mtime') or 0)
        r0 = c[0]
        byte_equal[name] += r0.get('reference_sha256') == picks[name][g]['bytes']
        chosen[(name, g)] = r0
        ir = irow.get((name, g))
        if ir is None or ir['run_id'] != r0['run']:
            # a different run with identical route + vehicle + group is acceptable only if its status agrees
            disagree_idx.append(dict(arm=name, group=g, mine=r0['run'], index=None if ir is None else ir['run_id'],
                                     same_status=None if ir is None else (ir['status'] == r0['status'])))
res['matching'] = dict(arm_groups=len(ARMS) * 800, matched=len(chosen), missing=len(missing), missing_sample=missing[:10],
                       reference_bytes_equal_pick_file=dict(byte_equal),
                       several_runs_same_route=len(amb), several_status_disagree=sum(len(a['statuses']) > 1 for a in amb),
                       several_sample=amb[:5], index_run_differs=len(disagree_idx), index_run_differs_sample=disagree_idx[:10])
used = {r['run']: r for r in chosen.values()}
res['matching']['distinct_runs'] = len(used)
res['matching']['new_ids_used'] = len(newset & set(used))
res['matching']['new_ids_total'] = len(newset)
res['matching']['new_ids_not_used'] = sorted(newset - set(used))[:10]

# ---------------------------------------------------------------- (3) provenance + timing
cases = {}
case_dir = ROOT / 'artifacts/traverse/generalist_20260921/A_adapt/suite/cases'
for g in groups:
    cases[g] = json.load(open(case_dir / f'{g}.json'))
case_sha = {g: sha(case_dir / f'{g}.json') for g in groups}
prov = Counter()
local_equal = 0; local_missing = 0
for rid, r in used.items():
    veh = r.get('vehicle') or 'hmmwv'
    prov['complete' if r.get('complete') else 'NOT_complete'] += 1
    prov['launch check passed + finite' if (r.get('launch_passed') and r.get('launch_finite')) else 'launch check NOT passed'] += 1
    prov[f'collector {str(r.get("collector_sha256"))[:8]}'] += 1
    prov[f'config {r.get("crm_config")}'] += 1
    prov[f'vehicle {veh} wrapper {str(r.get("wrapper_sha256"))[:8]} ag_vehicle {str(r.get("ag_vehicle_sha256"))[:8]}'] += 1
    prov['case sha equal' if r.get('case_sha256') == case_sha.get(r.get('scene')) else 'case sha DIFFERS'] += 1
    prov['request route sha = reference.json' if r.get('route_sha256_req') == r.get('reference_sha256') else 'request route sha DIFFERS'] += 1
    lo = K3 / 'e6/runs_soil' / rid / 'outcome.json'
    if lo.exists():
        local_equal += sha(lo) == r.get('outcome_sha256')
    else:
        local_missing += 1
res['provenance'] = dict(counts=dict(prov), local_outcome_equal_cluster=local_equal, local_outcome_missing=local_missing)
newr = [r for r in runs if r['run'] in newset]
res['timing'] = dict(spec_mtime=res['spec']['mtime'], lock_bfull_mtime=res['locks']['LOCK_crm_bfull']['mtime'],
                     new_runs_found=len(newr), new_runs_complete=sum(bool(r.get('complete')) for r in newr),
                     first_new_request=ts(min(r['req_mtime'] for r in newr)), last_new_outcome=ts(max(r['out_mtime'] for r in newr if r.get('out_mtime'))),
                     first_request_any_Gfull_Hfull_route=ts(min(chosen[(a, g)]['req_mtime'] for a in ('Gfull_free_gator', 'Hfull_free_gator', 'Hfull_free') for g in groups)))
res['timing']['note'] = 'reused drives (a G_full / H_full pick equal to an earlier route) started before the lock by construction; see reuse'
reuse = {}
for a in ('Gfull_free_gator', 'Hfull_free_gator', 'Hfull_free'):
    rr = [chosen[(a, g)] for g in groups]
    reuse[a] = dict(new=sum(r['run'] in newset for r in rr), reused=sum(r['run'] not in newset for r in rr),
                    reused_from=dict(Counter(r['run'].split('__')[-1] for r in rr if r['run'] not in newset)))
res['reuse'] = reuse

# ---------------------------------------------------------------- (4) statistics
feat = None
sys.path.insert(0, str(ROOT / 'src'))
from nedm.traverse.terrain import TerrainMap  # data source only (arena features)
arena_dir = {c['arena'] for c in cases.values()}
assert len(arena_dir) == 1
tm = TerrainMap.from_dir(ROOT / list(arena_dir)[0])
fxy = np.array([[f['x_m'], f['y_m']] for f in tm.features], float)
clus, strat = {}, {}
for g, c in cases.items():
    mid = (np.asarray(c['layout']['start_xy'], float) + np.asarray(c['goal_xy'], float)) / 2
    clus[g] = int(np.argmin(np.linalg.norm(fxy - mid[None], axis=1)))
    s = c.get('evaluation_stratum')
    strat[g] = 'hill' if s.startswith('hill') else 'crater' if s.startswith('crater') else 'other'
res['clusters'] = dict(n=len(set(clus.values())), sizes=dict(Counter(clus.values())),
                       equal_to_index=all(f"f104:{clus[g]}" == irow[('Gfull_free_gator', g)]['cluster'] for g in groups))
indist = set(json.load(open(K3 / 'suites/f104_indist_200.json'))['groups'])
assert indist <= set(groups)
G = np.array(groups)
F = {a: np.array([0 if chosen[(a, g)]['goal_reached'] else 1 for g in groups]) for a, _, _ in ARMS}
# goal_reached flag vs status string, and vs the index's fail label
res['labels'] = dict(status_vs_flag=sum((chosen[(a, g)]['status'] == 'goal_reached') != bool(chosen[(a, g)]['goal_reached']) for a, _, _ in ARMS for g in groups),
                     fail_vs_index=sum(int(F[a][i]) != irow[(a, g)]['fail'] for a, _, _ in ARMS for i, g in enumerate(groups)))
T = {a: np.array([chosen[(a, g)].get('elapsed_s') or np.nan for g in groups], float) for a, _, _ in ARMS}
res['rates'] = {}
for a, _, _ in ARMS:
    st = Counter(chosen[(a, g)]['status'] for g in groups)
    res['rates'][a] = dict(fail_pct=100 * F[a].mean(), fails=int(F[a].sum()), statuses=dict(st),
                           strata={s: dict(n=int(sum(strat[g] == s for g in groups)), fail_pct=100 * np.mean([F[a][i] for i, g in enumerate(groups) if strat[g] == s]))
                                   for s in ('hill', 'crater', 'other')},
                           indist_fail_pct=100 * np.mean([F[a][i] for i, g in enumerate(groups) if g in indist]),
                           median_time_goal=float(np.median(T[a][F[a] == 0])),
                           mean_pick_P=(float(np.mean([picks[a][g]['P'] for g in groups])) if picks[a][groups[0]]['P'] is not None else None))

rng = np.random.default_rng(12345)
B = 4000
CL = np.array([clus[g] for g in groups]); ucl = np.unique(CL)
gi = rng.integers(0, 800, size=(B, 800))
ci_idx = [np.flatnonzero(CL == c) for c in ucl]
cdraw = rng.integers(0, len(ucl), size=(B, len(ucl)))


def binom_two_sided(b, c):
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def contrast(t, r):
    d = F[t] - F[r]
    diff = 100 * d.mean()
    gb = 100 * d[gi].mean(1)
    cb = []
    for row in cdraw:
        ii = np.concatenate([ci_idx[j] for j in row])
        cb.append(100 * d[ii].mean())
    cb = np.array(cb)
    better = int(((F[t] == 0) & (F[r] == 1)).sum()); worse = int(((F[t] == 1) & (F[r] == 0)).sum())
    percl = {int(c): 100 * d[CL == c].mean() for c in ucl}
    return dict(test_pct=100 * F[t].mean(), ref_pct=100 * F[r].mean(), diff=diff,
                group95=[float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))], group90=[float(np.percentile(gb, 5)), float(np.percentile(gb, 95))],
                group_p1=float((np.sum(gb >= 0) + 1) / (B + 1)),
                cluster90=[float(np.percentile(cb, 5)), float(np.percentile(cb, 95))], cluster95=[float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))],
                cluster_p1=float((np.sum(cb >= 0) + 1) / (B + 1)),
                test_better=better, test_worse=worse, mcnemar_two_sided=binom_two_sided(better, worse),
                clusters_better=int(sum(v < 0 for v in percl.values())), clusters_worse=int(sum(v > 0 for v in percl.values())),
                clusters_equal=int(sum(v == 0 for v in percl.values())), per_cluster=percl,
                identical_picks=int(sum(picks[t][g]['sha'] == picks[r][g]['sha'] for g in groups)) if t in picks and r in picks else None)


FAM = [('F1', 'Gfull_free_gator', 'Hfull_free_gator'), ('F2', 'Gfull_free_gator', 'straight6_gator'),
       ('F3', 'Hfull_free_gator', 'straight6_gator'), ('F4', 'Gfull_free_gator', 'G_free_gator')]
SEC = [('H_full vs H, Gator', 'Hfull_free_gator', 'H_free_gator'), ('H_full vs H, HMMWV', 'Hfull_free', 'M1a_free'),
       ('H_full vs straight6, HMMWV', 'Hfull_free', 'straight6'), ('G_full vs H stage1, Gator', 'Gfull_free_gator', 'H_free_gator'),
       ('G vs H stage1, Gator', 'G_free_gator', 'H_free_gator'), ('G_full Gator vs H_full HMMWV', 'Gfull_free_gator', 'Hfull_free'),
       ('H_full Gator vs H_full HMMWV', 'Hfull_free_gator', 'Hfull_free')]
res['family'] = {k: contrast(t, r) | dict(test=t, ref=r) for k, t, r in FAM}
ps = sorted(((res['family'][k]['cluster_p1'], k) for k, _, _ in FAM))
m = len(ps); run = 0.0
for i, (p, k) in enumerate(ps):
    run = max(run, min(1.0, (m - i) * p)); res['family'][k]['holm_adj'] = run
    res['family'][k]['rejects'] = run < 0.05
res['secondary'] = {k: contrast(t, r) | dict(test=t, ref=r) for k, t, r in SEC}
# unsafe for F1 needs the backward clause (trajectory): taken from the index, checked only for consistency with fail
res['headroom'] = {}
for a, s in (('Gfull_free_gator', 'straight6_gator'), ('G_free_gator', 'straight6_gator'), ('Hfull_free_gator', 'straight6_gator'),
             ('H_free_gator', 'straight6_gator'), ('Hfull_free', 'straight6'), ('M1a_free', 'straight6')):
    hb = []
    for row in gi[:2000]:
        fs, fm = F[s][row].mean(), F[a][row].mean()
        hb.append((fs - fm) / fs)
    res['headroom'][a] = dict(closed=(F[s].mean() - F[a].mean()) / F[s].mean(), group95=[float(np.percentile(hb, 2.5)), float(np.percentile(hb, 97.5))])
res['time_ratio'] = {}
for t, r in (('Gfull_free_gator', 'Hfull_free_gator'), ('Gfull_free_gator', 'G_free_gator'), ('Hfull_free', 'M1a_free')):
    j = (F[t] == 0) & (F[r] == 0)
    ratio = T[t][j] / T[r][j]
    res['time_ratio'][f'{t}/{r}'] = dict(pairs=int(j.sum()), median=float(np.median(ratio)))
res['identical_picks'] = {f'{a}={b}': int(sum(picks[a][g]['sha'] == picks[b][g]['sha'] for g in groups))
                          for a, b in (('Gfull_free_gator', 'G_free_gator'), ('Hfull_free_gator', 'H_free_gator'), ('Gfull_free_gator', 'Hfull_free_gator'),
                                       ('Gfull_free_gator', 'straight6_gator'), ('Hfull_free_gator', 'straight6_gator'), ('G_free_gator', 'straight6_gator'))}


def conv(o):
    if isinstance(o, dict):
        return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [conv(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


res['chosen_runs'] = {a: {g: chosen[(a, g)]['run'] for g in groups} for a, _, _ in ARMS}
json.dump(conv(res), open(OUT, 'w'), indent=1)
print(json.dumps(conv({k: res[k] for k in ('locks', 'spec', 'groups', 'tasks', 'matching', 'provenance', 'timing', 'reuse', 'clusters', 'labels')}), indent=1)[:12000])
for k in ('family', 'secondary'):
    for n, v in res[k].items():
        print(f"{k} {n}: {v['test_pct']:.3f} vs {v['ref_pct']:.3f} diff {v['diff']:+.3f} cl90 [{v['cluster90'][0]:+.2f},{v['cluster90'][1]:+.2f}] cl95 [{v['cluster95'][0]:+.2f},{v['cluster95'][1]:+.2f}] "
              f"g95 [{v['group95'][0]:+.2f},{v['group95'][1]:+.2f}] g90 [{v['group90'][0]:+.2f},{v['group90'][1]:+.2f}] p1 cl {v['cluster_p1']:.4f} g {v['group_p1']:.4f} "
              f"won/lost {v['test_better']}/{v['test_worse']} McN {v['mcnemar_two_sided']:.3g} clusters b/w/eq {v['clusters_better']}/{v['clusters_worse']}/{v['clusters_equal']} "
              f"ident {v['identical_picks']} holm {v.get('holm_adj')}")
for a, v in res['rates'].items():
    print('rate', a, round(v['fail_pct'], 3), v['statuses'], {s: round(x['fail_pct'], 1) for s, x in v['strata'].items()}, 'indist', v['indist_fail_pct'], 'med', round(v['median_time_goal'], 2), 'P', v['mean_pick_P'])
print('headroom', {a: (round(v['closed'], 3), [round(x, 3) for x in v['group95']]) for a, v in res['headroom'].items()})
print('time', res['time_ratio'])
print('ident', res['identical_picks'])
