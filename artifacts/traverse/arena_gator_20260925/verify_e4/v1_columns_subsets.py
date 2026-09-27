#!/usr/bin/env python3
"""VERIFY E4 (independent): light-column checks of the f104 HMMWV training file and the three subset files.

- tier column == tier of the id in the f104 soil task file (crm_f104_20260916/tasks_train.json), every row, both worlds
- arena / vehicle columns == the case.json arena and the outcome.json vehicle block of every run (all 39,235 runs)
- blacklist: no id / group / episode matches any suite pattern (ag_blacklist.ALL, ci_train.SUITE_GROUPS,
  ga_build_mixed.BLACKLIST, generic patterns); no training-file group equals any group name found in the suite case
  folders; every group matches the f104 training-pool pattern
- subsets: recompute the md5 rule (own code, not ag_subset) and compare id sets / row counts / manifest numbers
Own code throughout; the builder's helpers are not imported (only the builder lists, to test them).
"""
import fnmatch, glob, hashlib, json, os, re, sys
from collections import Counter
import numpy as np

A = '/home/harry/NeDM-traverse_mppi/artifacts/traverse'
K3 = A + '/arena_gator_20260925'
BOTH = K3 + '/e4/f104_hmmwv/ci_f104_hmmwv_both.npz'
SUB = {n: K3 + f'/e4/subsets/{n}_f104_hmmwv_rigid.npz' for n in ('M1', 'LC545', 'LC272')}
LIGHT = ['id', 'group', 'episode', 'split', 'domain', 'anchor_frame', 'tier', 'arena', 'vehicle', 'status', 'fail', 'unsafe', 'source']
out = {}


def light(p):
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in LIGHT if k in z.files}, z.files


def md5h(s):
    return hashlib.md5(s.encode()).hexdigest()


D, keys = light(BOTH)
ids = D['id'].astype(str); ep = D['episode'].astype(str); grp = D['group'].astype(str); dom = D['domain'].astype(int)
sp = D['split'].astype(str); af = D['anchor_frame'].astype(int); tier = D['tier'].astype(int)
n = len(ids)
out['rows'] = n; out['rows_by_domain'] = {int(k): int(v) for k, v in Counter(dom.tolist()).items()}

# ---------- tier vs the f104 task file (the order the soil collection ran in; rigid twins share the ids)
T = {r['id']: int(r['tier']) for r in json.load(open('/tmp/ve4_crm_f104_tasks_train.json'))}
tt = np.array([T.get(e, -99) for e in ep])
out['tier'] = dict(rows_without_task_row=int((tt == -99).sum()), rows_tier_mismatch=int((tt != tier).sum()),
                   soil_k0_tier_counts={int(k): int(v) for k, v in sorted(Counter(tier[(dom == 1) & (af == 0)].tolist()).items())},
                   rigid_k0_tier_counts={int(k): int(v) for k, v in sorted(Counter(tier[(dom == 0) & (af == 0)].tolist()).items())})
# tier constant within an episode
et = {}
bad_const = 0
for e, t, d in zip(ep, tier, dom):
    if et.setdefault((d, e), t) != t:
        bad_const += 1
out['tier']['rows_tier_not_constant_in_episode'] = bad_const
# soil tier 12 groups
g12 = {g for g, t, d, a in zip(grp, tier, dom, af) if d == 1 and t == 12 and a == 0}
out['tier']['soil_groups_with_tier12'] = len(g12)
soil_groups_t0 = {g for g, t, d, a in zip(grp, tier, dom, af) if d == 1 and t == 0 and a == 0}
out['tier']['soil_groups_without_tier0'] = sorted({g for g, d in zip(grp, dom) if d == 1} - soil_groups_t0)

# ---------- arena / vehicle columns vs the runs' own records
roots = dict(rigid=[A + '/fdm_f104_50h_20260909/production_v3/runs', A + '/fdm_f104_50h_20260909/production_v4/runs'],
             crm=[A + '/crm_f104_v1/collect_v1/runs'])
col_arena = set(D['arena'].astype(str)); col_veh = set(D['vehicle'].astype(str))
case_arena, veh_blocks, case_group_mismatch, case_split_mismatch, n_runs = Counter(), Counter(), 0, 0, Counter()
ep_split = {}
for e, g, s, d in zip(ep, grp, sp, dom):
    ep_split[(d, e)] = (g, s)
for w, code in (('rigid', 0), ('crm', 1)):
    eps = sorted({e for e, d in zip(ep, dom) if d == code})
    for e in eps:
        dd = [os.path.join(r, e) for r in roots[w] if os.path.isdir(os.path.join(r, e))]
        assert len(dd) == 1, (w, e, dd)
        c = json.load(open(dd[0] + '/case.json')); o = json.load(open(dd[0] + '/outcome.json'))
        case_arena[(w, c['arena'])] += 1; veh_blocks[(w, str(o.get('vehicle')))] += 1; n_runs[w] += 1
        g, s = ep_split[(code, e)]
        case_group_mismatch += int(c['id'] != g); case_split_mismatch += int(c['split'] != s)
    # every run folder id (pattern) is in the file, except the excluded / absent ones
    on_disk = {os.path.basename(p) for r in roots[w] for p in glob.glob(r + '/*') if os.path.isdir(p)}
    out.setdefault('runs_on_disk_not_in_file', {})[w] = sorted(on_disk - set(eps))[:20]
    out.setdefault('file_episodes_not_on_disk', {})[w] = len(set(eps) - on_disk)
out['arena_vehicle'] = dict(column_arena=sorted(col_arena), column_vehicle=sorted(col_veh),
                            case_arena={f'{w}|{a}': v for (w, a), v in case_arena.items()},
                            outcome_vehicle_block={f'{w}|{a}': v for (w, a), v in veh_blocks.items()},
                            case_group_mismatch=case_group_mismatch, case_split_mismatch=case_split_mismatch, runs=dict(n_runs))

# ---------- blacklist
sys.path.insert(0, '/home/harry/NeDM-traverse_mppi/scripts'); sys.path.insert(0, '/home/harry/NeDM-traverse_mppi/src')
import ag_blacklist
pats = set(ag_blacklist.ALL)
try:
    import ga_build_mixed; pats |= set(ga_build_mixed.BLACKLIST)
except Exception as ex:
    print('ga_build_mixed import', ex)
generic = ['*_test_group_*', '*_heldout_group_*', '*_dev_group_*', '*_eval_group_*', '*_pair_group_*', 'drift__*', '*_t2_group_*', '*test*', '*eval*', '*heldout*', '*dev_group*']
pats |= set(generic)
suite_groups = set()
for d in [A + '/generalist_20260921/A_adapt/suite/cases', A + '/generalist_20260921/cases/pair_v1/cases', A + '/crm_f104_v1/cases_eval/cases',
          A + '/fdm_f104_50h_20260909/cases_test_final'] + sorted(glob.glob(K3 + '/cases/test_g*/cases')) + \
         [K3 + '/cases/heldout_g203/cases', K3 + '/cases/heldout_g228/cases', K3 + '/cases/dev_g217/cases']:
    for p in glob.glob(d + '/*.json'):
        b = os.path.basename(p)[:-5]
        if b != 'cases':
            suite_groups.add(b)
out['blacklist'] = dict(n_patterns=len(pats), n_suite_group_names=len(suite_groups))


def bl_check(name, ids_, grp_, ep_):
    strings = set(i.split('@')[0] for i in ids_) | set(grp_) | set(ep_)
    hits = sorted(s for s in strings if any(fnmatch.fnmatch(s, p) for p in pats))
    same = sorted(set(grp_) & suite_groups)
    gpat = re.compile(r'^f104_v2_group_\d{4}$')
    notpool = sorted(g for g in set(grp_) if not gpat.match(g))
    epat = re.compile(r'^f104_v2_group_\d{4}_(route|op)_\d{2}$')
    notep = sorted(e for e in set(ep_) if not epat.match(e))
    out['blacklist'][name] = dict(strings=len(strings), pattern_hits=hits[:5], n_pattern_hits=len(hits), group_equal_to_suite=same[:5],
                                  groups_not_training_pool=notpool[:5], episodes_not_training_ids=notep[:5])


bl_check('ci_f104_hmmwv_both', ids, grp, ep)

# ---------- subsets: own recomputation of the rule
r_mask = dom == 0
rtrain_groups = sorted(set(grp[r_mask & (sp == 'train')]))
rank = sorted(rtrain_groups, key=md5h)
devf = {g for g in rtrain_groups if int(md5h(g), 16) % 5 == 0}
exp = {}
for name, N, keep_dev in (('M1', 1089, False), ('LC545', 545, True), ('LC272', 272, True)):
    sel = set(rank[:N]) | (devf if keep_dev else set())
    m = r_mask & (((sp == 'train') & np.isin(grp, list(sel))) | np.isin(sp, ['val', 'test']))
    exp[name] = dict(ids=set(ids[m]), sel=set(rank[:N]), rows=int(m.sum()), fit_deploy=int((m & (sp == 'train')).sum()),
                     fit_holdout=int((m & (sp == 'train') & ~np.isin(grp, list(devf))).sum()),
                     fit_groups_holdout=len(set(rank[:N]) - devf), dev_rows=int((m & (sp == 'train') & np.isin(grp, list(devf))).sum()),
                     val_rows=int((m & (sp == 'val')).sum()))
out['subsets'] = dict(rigid_train_groups=len(rtrain_groups), dev_fold_groups=len(devf))
for name, p in SUB.items():
    S, skeys = light(p)
    man = json.load(open(p[:-4] + '.manifest.json'))
    sids = S['id'].astype(str); e = exp[name]
    bl_check(name, sids, S['group'].astype(str), S['episode'].astype(str))
    same_order = None
    # rows in the subset keep input order: positions in the both file increase
    pos = {s: i for i, s in enumerate(ids)}
    ix = np.array([pos[s] for s in sids])
    same_order = bool(np.all(np.diff(ix) > 0))
    # every light column equals the source rows
    col_eq = {k: bool(np.array_equal(S[k].astype(str) if S[k].dtype == object else S[k],
                                     D[k][ix].astype(str) if D[k].dtype == object else D[k][ix])) for k in S}
    out['subsets'][name] = dict(rows=len(sids), expected_rows=e['rows'], id_set_equal=set(sids) == e['ids'], input_order=same_order,
                                light_columns_equal_source=col_eq,
                                manifest_rows=man['rows'], manifest_fit_deploy=man['fit_rows_deploy'], expected_fit_deploy=e['fit_deploy'],
                                manifest_fit_holdout=man['fit_rows_holdout'], expected_fit_holdout=e['fit_holdout'],
                                manifest_fit_groups_holdout=man['per_arena']['f104']['fit_groups_holdout'], expected_fit_groups_holdout=e['fit_groups_holdout'],
                                manifest_selected_equal=set(man['selected_groups']['f104']) == e['sel'], dev_rows=e['dev_rows'], val_rows=e['val_rows'],
                                manifest_use=man['use'], keys_equal_source=sorted(skeys) == sorted(keys),
                                domains=sorted(set(S['domain'].astype(int).tolist())), vehicle=sorted(set(S['vehicle'].astype(str))), arena=sorted(set(S['arena'].astype(str))))
# nesting
out['subsets']['nested_272_in_545_in_1089'] = exp['LC272']['sel'] <= exp['LC545']['sel'] <= exp['M1']['sel']
json.dump(out, open(K3 + '/verify_e4/v1_columns_subsets.json', 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
