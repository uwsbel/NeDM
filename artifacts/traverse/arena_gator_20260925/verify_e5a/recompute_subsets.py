#!/usr/bin/env python3
"""VERIFY_E5a (independent): recompute every E5a rigid subset from the per-arena built files with my own code (nothing
imported from ag_subset.py / ag_build_ds.py) and compare with the files on the cluster (light columns pulled by
ve5a_extract_light.py). Also: suite / blacklist scan of every file, tier structure, tiers vs the task files.
  python recompute_subsets.py <light dir> <K3> <out json>"""
import sys, json, re, hashlib, fnmatch, glob, os
from collections import Counter, defaultdict
import numpy as np

LD, K3, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
REPO = os.path.abspath(os.path.join(K3, '../../..'))
md5 = lambda s: hashlib.md5(s.encode()).hexdigest()
dev = lambda g: int(md5(g), 16) % 5 == 0
L = lambda n: {k: v for k, v in np.load(f'{LD}/{n}_light.npz').items()}
base = dict(f104=L('ci_f104_hmmwv_both'), g203=L('ci_g203_hmmwv_rigid'), g228=L('ci_g228_hmmwv_rigid'), gator=L('ci_f104_gator_rigid'))
for k, z in base.items():                      # rigid rows only (domain 0 = rigid: ids end in @rigid)
    m = z['domain'] == 0
    assert all(i.endswith('@rigid') for i in z['id'][m]) and not any(i.endswith('@rigid') for i in z['id'][~m])
    base[k] = {kk: v[m] for kk, v in z.items() if kk != '_keys'}

DESIGNS = {  # name: ({source: n groups | 'all'}, keep dev fold)
    'M1_f104_hmmwv_rigid': ({'f104': 1089}, False), 'H_f104_hmmwv_rigid': ({'f104': 'all'}, False),
    'M2_hmmwv_rigid': ({'f104': 545, 'g203': 545}, False), 'M3_hmmwv_rigid': ({'f104': 363, 'g203': 363, 'g228': 363}, False),
    'A3_hmmwv_rigid': ({'f104': 'all', 'g203': 'all', 'g228': 'all'}, False),
    'LC272_f104_hmmwv_rigid': ({'f104': 272}, True), 'LC545_f104_hmmwv_rigid': ({'f104': 545}, True),
    'LOAO1_g203_hmmwv_rigid': ({'g203': 545}, False), 'LOAO1_g228_hmmwv_rigid': ({'g228': 545}, False),
    'LOAO2_f104_g203_hmmwv_rigid': ({'f104': 273, 'g203': 272}, False), 'LOAO2_f104_g228_hmmwv_rigid': ({'f104': 273, 'g228': 272}, False),
    'LOAO2_g203_g228_hmmwv_rigid': ({'g203': 273, 'g228': 272}, False), 'G_f104_gator_rigid': ({'gator': 'all'}, False),
    'EV_f104_hmmwv_rigid': ({'f104': 0}, True), 'EV_g203_hmmwv_rigid': ({'g203': 0}, True), 'EV_g228_hmmwv_rigid': ({'g228': 0}, True),
    'EV_f104_gator_rigid': ({'gator': 0}, True)}

# suite ids: every case file of every suite of this study + the f104 800-group suite + gen_v1 test groups; plus patterns
suite = set()
for d in glob.glob(f'{K3}/cases/test_*/cases') + glob.glob(f'{K3}/cases/heldout_*/cases') + glob.glob(f'{K3}/cases/dev_*/cases') + \
        [f'{REPO}/artifacts/traverse/generalist_20260921/A_adapt/suite/cases']:
    suite |= {os.path.basename(p)[:-5] for p in glob.glob(d + '/*.json') if not p.endswith('cases.json')}
suite |= set(json.load(open(f'{K3}/suites/f104_indist_200.json'))['groups'])
PATS = ['*_test_group_*', '*_heldout_group_*', '*_dev_group_*', '*pair_group_*', '*crm_eval_group_*', '*g1_test_group*']
def suite_hit(s):
    s0 = s.split('@')[0]
    s0 = s0[len('gator__'):] if s0.startswith('gator__') else s0
    g = re.sub(r'_(route|op)_\d+$', '', s0)
    return g in suite or s0 in suite or any(fnmatch.fnmatch(s0, p) for p in PATS)

res = dict(n_suite_ids=len(suite), designs={}, suite_scan={}, tiers={})
for name, (want, keepdev) in DESIGNS.items():
    exp_ids, sel = [], {}
    for src, n in want.items():
        z = base[src]
        tr = z['split'] == 'train'
        ranked = sorted(set(z['group'][tr]), key=md5)
        k = len(ranked) if n == 'all' else n
        assert k <= len(ranked), (name, src, k, len(ranked))
        s = set(ranked[:k]) | ({g for g in ranked if dev(g)} if keepdev else set())
        sel[src] = dict(available=len(ranked), selected=k, dev_kept=len({g for g in ranked if dev(g)}) if keepdev else 0, groups=s)
        keep = (tr & np.isin(z['group'], sorted(s))) | np.isin(z['split'], ['val', 'test'])
        exp_ids += list(z['id'][keep])
    got = L(name)
    gid = list(got['id'])
    sp, grp, ar = got['split'], got['group'], got['arena']
    fit_h = (sp == 'train') & ~np.array([dev(g) for g in grp])
    r = dict(rows=len(gid), rows_expected=len(exp_ids), same_id_set=set(gid) == set(exp_ids), same_order=gid == exp_ids,
             ids_unique=len(set(gid)) == len(gid), fit_deploy=int((sp == 'train').sum()), fit_holdout=int(fit_h.sum()),
             fit_holdout_groups=len(set(grp[fit_h])), val_rows=int((sp == 'val').sum()), test_rows=int((sp == 'test').sum()),
             per_arena={a: dict(train_groups=len(set(grp[(sp == 'train') & (ar == a)])), val_groups=len(set(grp[(sp == 'val') & (ar == a)])),
                                test_groups=len(set(grp[(sp == 'test') & (ar == a)])), rows=int((ar == a).sum()),
                                train_rows=int(((sp == 'train') & (ar == a)).sum())) for a in sorted(set(ar))},
             vehicles=sorted(set(got['vehicle'])), selection={k: {kk: vv for kk, vv in v.items() if kk != 'groups'} for k, v in sel.items()},
             train_groups_equal_selection=all(set(grp[(sp == 'train') & (ar == (('f104') if src == 'gator' else src))]) == sel[src]['groups'] for src in sel),
             tier_min=int(got['tier'].min()), tier_max=int(got['tier'].max()))
    res['designs'][name] = r
    print(name, {k: r[k] for k in ('rows', 'rows_expected', 'same_id_set', 'same_order', 'fit_deploy', 'fit_holdout', 'fit_holdout_groups', 'val_rows', 'train_groups_equal_selection')},
          {a: v['train_groups'] for a, v in r['per_arena'].items()}, flush=True)

for n in sorted(glob.glob(f'{LD}/*_light.npz')):
    name = os.path.basename(n)[:-10]; z = np.load(n)
    strs = set(z['id']) | set(z['group']) | set(z['episode'])
    hits = sorted(s for s in strs if suite_hit(s))
    res['suite_scan'][name] = dict(strings=len(strs), hits=len(hits), examples=hits[:5])
print('suite scan:', {k: v['hits'] for k, v in res['suite_scan'].items()})

# tier structure of the per-arena files: every group's designed + on-policy episodes carry tiers 0..19 once each
for k, z in base.items():
    ep_tier = {}
    for e, t, g in zip(z['episode'], z['tier'], z['group']):
        ep_tier.setdefault(e, set()).add(int(t))
    assert all(len(v) == 1 for v in ep_tier.values()), k
    per_g = defaultdict(list)
    for e, v in ep_tier.items():
        per_g[re.sub(r'_(route|op)_\d+$', '', e.replace('gator__', ''))].append(next(iter(v)))
    full = sum(1 for v in per_g.values() if sorted(v) == list(range(20)))
    res['tiers'][k] = dict(groups=len(per_g), episodes=len(ep_tier), groups_with_tiers_0_19_once=full,
                           episodes_per_group=dict(Counter(len(v) for v in per_g.values())))
# tiers vs the task files
T1 = {r['id']: r['tier'] for r in json.load(open(f'{K3}/e3/tasks/rigid_hmmwv_v1.json'))}
T2 = {r['id']: r['tier'] for r in json.load(open(f'{K3}/e3/tasks/rigid_v2.json'))}
for k, T in (('g203', T1), ('g228', T1), ('gator', T2)):
    z = base[k]; eps = {e: int(t) for e, t in zip(z['episode'], z['tier'])}
    miss = [e for e in eps if e not in T]; bad = [e for e in eps if e in T and T[e] != eps[e]]
    res['tiers'][k].update(task_file_missing=len(miss), task_file_tier_mismatch=len(bad))
print('tiers:', res['tiers'])
json.dump(res, open(OUT, 'w'), indent=1, default=str)
