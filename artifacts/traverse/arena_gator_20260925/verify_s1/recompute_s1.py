#!/usr/bin/env python3
"""VERIFY_S1 (independent verifier of soil stage 1): recompute every soil stage-1 subset and evaluation file from the
per-arena built files with my own code (nothing imported from ag_subset.py / ag_build_ds.py / ag_e5a_ids.py), and check
tiers, vehicles, H ids vs the Gator-validated ids, suite / blacklist ids, and run records.
  python recompute_s1.py <cluster_out dir> <K3> <out json>"""
import sys, json, re, hashlib, fnmatch, glob, os
from collections import Counter, defaultdict
import numpy as np

LD, K3, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
REPO = os.path.abspath(os.path.join(K3, '../../..'))
md5 = lambda s: hashlib.md5(s.encode()).hexdigest()
dev = lambda g: int(md5(g), 16) % 5 == 0
L = lambda n: {k: v for k, v in np.load(f'{LD}/{n}_light.npz').items() if k != '_keys'}
strip = lambda e: e[len('gator__'):] if e.startswith('gator__') else e
res = dict()

# ---------- per-arena files (soil rows only; f104 HMMWV file holds both worlds) ----------
raw = dict(f104=L('ci_f104_hmmwv_both'), g203=L('ci_g203_hmmwv_crm'), g228=L('ci_g228_hmmwv_crm'), gator=L('ci_f104_gator_crm'))
base, full = {}, {}
for k, z in raw.items():
    m = z['domain'] == 1
    assert all(i.endswith('@crm') for i in z['id'][m]) and not any(i.endswith('@crm') for i in z['id'][~m]), k
    full[k] = {kk: v[m] for kk, v in z.items()}                       # every soil row of the file
    t = full[k]['tier']
    base[k] = {kk: v[(t >= 0) & (t <= 6)] for kk, v in full[k].items()}  # my tier cut
per_arena = {}
for k in raw:
    z = full[k]
    eps_all = {e: int(t) for e, t in zip(z['episode'], z['tier'])}
    per_arena[k] = dict(rows_soil=len(z['id']), rows_other_world=int((raw[k]['domain'] != 1).sum()), tiers=sorted({int(t) for t in z['tier']}),
                        vehicles=sorted(set(z['vehicle'])), arenas=sorted(set(z['arena'])), episodes=len(eps_all),
                        rows_tier0_6=len(base[k]['id']), episodes_tier0_6=len({e for e, t in eps_all.items() if t <= 6}),
                        groups_tier0_6=len(set(base[k]['group'])),
                        split_groups_tier0_6={s: len(set(base[k]['group'][base[k]['split'] == s])) for s in ('train', 'val', 'test')},
                        episodes_per_tier_0_6={t: sum(1 for v in eps_all.values() if v == t) for t in range(7)},
                        standing_start_rows_tier0_6=int((base[k]['anchor_frame'] == 0).sum()))
print('per arena', json.dumps(per_arena, default=str), flush=True)
res['per_arena_files'] = per_arena

# ---------- tiers vs the soil task file; coverage ----------
TS = json.load(open(f'{K3}/e3/tasks/soil_v2.json'))
T2 = {r['id']: r for r in TS}
cov = {}
for k, arena, pre in (('g203', 'g203', ''), ('g228', 'g228', ''), ('gator', 'f104', 'gator__')):
    z = full[k]
    eps = {e: int(t) for e, t in zip(z['episode'], z['tier'])}
    rows = [r for r in TS if r['arena'] == arena and r['id'].startswith(pre) and (pre or not r['id'].startswith(('gator', 'bitid', 'drift')))
            and 0 <= r['tier'] <= 6 and r.get('vehicle', 'hmmwv') == ('gator' if pre else 'hmmwv')]
    want = {r['id']: r['tier'] for r in rows}
    miss = sorted(set(want) - set(eps)); extra = sorted(set(eps) - set(want))
    bad = [e for e in eps if e in want and want[e] != eps[e]]
    cov[k] = dict(task_rows_tier0_6=len(want), file_episodes=len(eps), missing_from_file=miss, not_in_task_rows=extra[:5], n_not_in_task_rows=len(extra),
                  tier_mismatch=len(bad), task_groups=len({T2[i]['group'] for i in want}),
                  task_rows_per_tier={t: sum(1 for v in want.values() if v == t) for t in range(7)})
    print('coverage', k, {kk: v for kk, v in cov[k].items()}, flush=True)
# f104 HMMWV twins of the Gator rows: same tier in the f104 file (collect_v1 task tier) as in soil_v2
f104eps = {e: int(t) for e, t in zip(full['f104']['episode'], full['f104']['tier'])}
gat_rows = {r['id']: r for r in TS if r['id'].startswith('gator__')}
tw_bad = [i for i, r in gat_rows.items() if strip(i) in f104eps and f104eps[strip(i)] != r['tier']]
tw_miss = [i for i in gat_rows if strip(i) not in f104eps]
f104_t06 = {e for e, t in f104eps.items() if t <= 6}
gat_t06 = {strip(i) for i, r in gat_rows.items() if 0 <= r['tier'] <= 6}
cov['gator_vs_f104_twins'] = dict(gator_task_rows=len(gat_rows), twin_tier_mismatch=len(tw_bad), gator_rows_without_f104_twin=len(tw_miss),
                                  f104_episodes_tier0_6=len(f104_t06), gator_task_ids_tier0_6=len(gat_t06), same_set=f104_t06 == gat_t06,
                                  f104_groups=len({re.sub(r'_(route|op)_\d+$', '', e) for e in f104eps}),
                                  f104_groups_with_fewer_than_7_in_tier0_6=sorted(g for g, c in Counter(re.sub(r'_(route|op)_\d+$', '', e) for e in f104_t06).items() if c != 7))
print('twins', cov['gator_vs_f104_twins'], flush=True)
res['coverage'] = cov

# ---------- subsets and evaluation files recomputed ----------
DESIGNS = {  # name: ([(source, n groups | 'all' | 0)], keep dev fold)
    'M1_f104_hmmwv_soil': ([('f104', 1089)], False), 'H_f104_hmmwv_soil': ([('f104', 'all')], False),
    'M2_hmmwv_soil': ([('f104', 545), ('g203', 545)], False), 'M3_hmmwv_soil': ([('f104', 363), ('g203', 363), ('g228', 363)], False),
    'A3_hmmwv_soil': ([('f104', 1089), ('g203', 'all'), ('g228', 'all')], False),
    'LC272_f104_hmmwv_soil': ([('f104', 272)], True), 'LC545_f104_hmmwv_soil': ([('f104', 545)], True),
    'LOAO1_g203_hmmwv_soil': ([('g203', 545)], False), 'LOAO1_g228_hmmwv_soil': ([('g228', 'all')], False),
    'LOAO2_f104_g203_hmmwv_soil': ([('f104', 273), ('g203', 272)], False), 'LOAO2_f104_g228_hmmwv_soil': ([('f104', 273), ('g228', 272)], False),
    'LOAO2_g203_g228_hmmwv_soil': ([('g203', 273), ('g228', 272)], False), 'G_f104_gator_soil': ([('gator', 'all')], False),
    'EV_f104_hmmwv_crm': ([('f104', 0)], True), 'EV_f104_hmmwv_gatorids_crm': ([('f104', 0)], True), 'EV_g203_hmmwv_crm': ([('g203', 0)], True),
    'EV_g228_hmmwv_crm': ([('g228', 0)], True), 'EV_f104_gator_crm': ([('gator', 0)], True)}
validated = [l.strip() for l in open(f'{K3}/e5/ids_soil/gator_soil_validated_t0-6.txt') if l.strip()]
res['designs'] = {}
for name, (want, keepdev) in DESIGNS.items():
    exp_ids, sel = [], {}
    for src, n in want:
        z = base[src]
        tr = z['split'] == 'train'
        ranked = sorted(set(z['group'][tr]), key=md5)
        ranked_before_cut = sorted(set(full[src]['group'][full[src]['split'] == 'train']), key=md5)
        k = len(ranked) if n == 'all' else n
        assert k <= len(ranked), (name, src, k, len(ranked))
        s = set(ranked[:k]) | ({g for g in ranked if dev(g)} if keepdev else set())
        sel[src] = dict(available=len(ranked), selected=k, dev_kept=len({g for g in ranked if dev(g)}) if keepdev else 0, groups=s,
                        same_ranking_before_cut=ranked_before_cut[:k] == ranked[:k])
        keep = (tr & np.isin(z['group'], sorted(s))) | np.isin(z['split'], ['val', 'test'])
        exp_ids += list(z['id'][keep])
    got = L(name)
    gid = list(got['id'])
    sp, grp, ar, af = got['split'], got['group'], got['arena'], got['anchor_frame']
    fit_h = (sp == 'train') & ~np.array([dev(g) for g in grp])
    r = dict(rows=len(gid), rows_expected=len(exp_ids), same_id_set=set(gid) == set(exp_ids), same_order=gid == exp_ids,
             ids_unique=len(set(gid)) == len(gid), domains=sorted({int(x) for x in got['domain']}),
             tiers=sorted({int(t) for t in got['tier']}), vehicles=sorted(set(got['vehicle'])),
             fit_deploy=int((sp == 'train').sum()), fit_holdout=int(fit_h.sum()), fit_holdout_groups=len(set(grp[fit_h])),
             train_drives=len(set(got['episode'][sp == 'train'])), standing_start_train_rows=int(((sp == 'train') & (af == 0)).sum()),
             val_rows=int((sp == 'val').sum()), test_rows=int((sp == 'test').sum()),
             per_arena={a: dict(train_groups=len(set(grp[(sp == 'train') & (ar == a)])), val_groups=len(set(grp[(sp == 'val') & (ar == a)])),
                                test_groups=len(set(grp[(sp == 'test') & (ar == a)])), dev_fold_groups=len({g for g in set(grp[(sp == 'train') & (ar == a)]) if dev(g)}),
                                rows=int((ar == a).sum())) for a in sorted(set(ar))},
             selection={k: {kk: vv for kk, vv in v.items() if kk != 'groups'} for k, v in sel.items()})
    if name.startswith(('G_', 'H_', 'EV_f104_gator', 'EV_f104_hmmwv_gatorids')):
        V = set(validated) if 'gator' in name and not name.startswith('EV_f104_hmmwv') else {strip(v) for v in validated}
        E = set(got['episode'])
        r['episodes_equal_validated_ids'] = E == V if not name.startswith('EV') else E <= V
        r['episodes'] = len(E)
    res['designs'][name] = r
    print(name, {k: r[k] for k in ('rows', 'rows_expected', 'same_id_set', 'same_order', 'tiers', 'vehicles', 'fit_deploy', 'fit_holdout', 'train_drives')},
          {a: (v['train_groups'], v['val_groups'], v['test_groups']) for a, v in r['per_arena'].items()}, r.get('episodes_equal_validated_ids'), flush=True)

# ---------- H == validated Gator ids, exactly ----------
H = L('H_f104_hmmwv_soil'); M1 = L('M1_f104_hmmwv_soil'); G = L('G_f104_gator_soil')
Hep = set(H['episode']); Gep = set(G['episode']); Vs = {strip(v) for v in validated}
res['H_vs_validated'] = dict(validated=len(validated), validated_unique=len(set(validated)), all_prefixed=all(v.startswith('gator__') for v in validated),
                             H_episodes=len(Hep), H_eq_validated=Hep == Vs, H_minus_V=sorted(Hep - Vs)[:5], V_minus_H=sorted(Vs - Hep)[:5],
                             G_episodes=len(Gep), G_eq_validated=Gep == set(validated),
                             H_rows_eq_M1_rows=list(H['id']) == list(M1['id']),
                             G_rows_eq_gator_file=list(G['id']) == list(full['gator']['id']),
                             H_split_eq_G_split=({strip(e): x for e, x in zip(G['episode'], G['split'])} == dict(zip(H['episode'], H['split']))))
print('H vs validated', res['H_vs_validated'], flush=True)

# ---------- run records: vehicle block, completion, launch check, status ----------
runs = {}
for fn in ('runs_soil_v1.jsonl', 'runs_collect_v1.jsonl'):
    for l in open(f'{LD}/{fn}'):
        r = json.loads(l); runs[(fn, r['id'])] = r
S1 = {i: r for (f, i), r in runs.items() if f == 'runs_soil_v1.jsonl'}
C1 = {i: r for (f, i), r in runs.items() if f == 'runs_collect_v1.jsonl'}
gat = {i: r for i, r in S1.items() if i.startswith('gator')}
hm = {i: r for i, r in S1.items() if not i.startswith('gator')}
vr = dict(soil_v1_runs=len(S1), gator_runs=len(gat), gator_prefixes=dict(Counter(i.split('__')[0] for i in gat)),
          gator_with_gator_block=sum(1 for r in gat.values() if r.get('vehicle_name') == 'gator'),
          gator_rear_radius=dict(Counter(str(r.get('wheel')) for r in gat.values())),
          hmmwv_runs=len(hm), hmmwv_with_vehicle_key=sum(1 for r in hm.values() if r.get('has_vehicle_key')),
          hmmwv_without_outcome=sum(1 for r in hm.values() if not r.get('has_outcome')), gator_without_outcome=sum(1 for r in gat.values() if not r.get('has_outcome')),
          collect_v1_runs=len(C1), collect_v1_with_vehicle_key=sum(1 for r in C1.values() if r.get('has_vehicle_key')),
          collect_v1_without_outcome=sum(1 for r in C1.values() if not r.get('has_outcome')))
# every drive of every per-arena training file -> its run record
tf = {}
for k, src in (('g203', S1), ('g228', S1), ('gator', S1), ('f104', C1)):
    eps = sorted(set(base[k]['episode']))
    rr = [src.get(e) for e in eps]
    tf[k] = dict(drives_tier0_6=len(eps), run_found=sum(r is not None for r in rr), complete=sum(bool(r and r['complete']) for r in rr),
                 launch_passed=sum(bool(r and r.get('launch_passed') is True) for r in rr), launch_missing=sum(bool(r and r.get('launch_passed') is None) for r in rr),
                 vehicle_blocks=dict(Counter(str(r.get('vehicle_name')) if r and r.get('has_vehicle_key') else 'none' for r in rr)),
                 case_arena=dict(Counter(str(r.get('case_arena')) for r in rr if r)),
                 statuses=dict(Counter(str(r.get('status')) for r in rr if r)))
vr['training_file_drives'] = tf
# the 2 g203 rejects and any Gator ids of tiers 0-6 not in the file
vr['g203_task_ids_not_in_file'] = {i: (S1.get(i) or {}).get('status') for i in cov['g203']['missing_from_file']}
# Gator vs HMMWV outcomes on the same ids (goal not reached), tiers 0-6
gid06 = sorted(set(base['gator']['episode']))
gf = [not S1[e]['goal_reached'] for e in gid06]; hf = [not C1[strip(e)]['goal_reached'] for e in gid06]
vr['gator_vs_hmmwv_tier0_6'] = dict(n=len(gid06), gator_fail=round(float(np.mean(gf)), 4), hmmwv_fail=round(float(np.mean(hf)), 4),
                                    gator_only=int(sum(a and not b for a, b in zip(gf, hf))), hmmwv_only=int(sum(b and not a for a, b in zip(gf, hf))),
                                    gator_sim_h=round(sum(S1[e]['elapsed_s'] for e in gid06) / 3600, 1), hmmwv_sim_h=round(sum(C1[strip(e)]['elapsed_s'] for e in gid06) / 3600, 1),
                                    gator_status=dict(Counter(S1[e]['status'] for e in gid06)), hmmwv_status=dict(Counter(C1[strip(e)]['status'] for e in gid06)))
print('vehicle records', json.dumps(vr, default=str)[:3000], flush=True)
res['vehicle_records'] = vr

# ---------- suite / blacklist scan of every file ----------
suite = set()
for d in glob.glob(f'{K3}/cases/test_*/cases') + glob.glob(f'{K3}/cases/heldout_*/cases') + glob.glob(f'{K3}/cases/dev_*/cases') + \
        [f'{REPO}/artifacts/traverse/generalist_20260921/A_adapt/suite/cases']:
    suite |= {os.path.basename(p)[:-5] for p in glob.glob(d + '/*.json') if not p.endswith('cases.json')}
suite |= set(json.load(open(f'{K3}/suites/f104_indist_200.json'))['groups'])
n_suite_files = len(suite)
PATS = ['*_test_group_*', '*_heldout_group_*', '*_dev_group_*', '*pair_group_*', '*crm_eval_group_*', '*g1_test_group*', 'bitid__*', 'drift__*',
        'gatorR8__*', '*_dev_*', '*_test_*', '*_heldout_*']
def suite_hit(s):
    s0 = s.split('@')[0]
    s1 = s0[len('gator__'):] if s0.startswith('gator__') else s0
    g = re.sub(r'_(route|op)_\d+$', '', s1)
    return g in suite or s1 in suite or any(fnmatch.fnmatch(s0, p) or fnmatch.fnmatch(s1, p) for p in PATS)
# positive controls
ctrl = ['g260_test_group_0001_route_00@0@crm', 'gator__f104_pair_group_0003_route_01@0@crm', 'f104_crm_eval_group_0010', 'g217_dev_group_0000',
        'g203_heldout_group_0002_route_03', 'bitid__g203_v2_group_0479_route_03', 'g241_test_group_0100']
neg = ['g203_v2_group_0000_op_02@0@crm', 'gator__f104_v2_group_0000_route_02@40@crm']
res['suite_scan'] = dict(n_suite_ids=n_suite_files, patterns=PATS, positive_controls={c: suite_hit(c) for c in ctrl}, negative_controls={c: suite_hit(c) for c in neg}, files={})
for n in sorted(glob.glob(f'{LD}/*_light.npz')):
    name = os.path.basename(n)[:-10]; z = np.load(n)
    dm = z['domain'] == 1 if name == 'ci_f104_hmmwv_both' else np.ones(len(z['id']), bool)
    strs = set(z['id'][dm]) | set(z['group'][dm]) | set(z['episode'][dm])
    hits = sorted(s for s in strs if suite_hit(s))
    res['suite_scan']['files'][name] = dict(strings=len(strs), hits=len(hits), examples=hits[:5])
print('suite scan', res['suite_scan']['positive_controls'], res['suite_scan']['negative_controls'], {k: v['hits'] for k, v in res['suite_scan']['files'].items()}, flush=True)
json.dump(res, open(OUT, 'w'), indent=1, default=str)
