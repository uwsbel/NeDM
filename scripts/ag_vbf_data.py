#!/usr/bin/env python3
"""VERIFY_gator_full (independent checker of task B stage 2): training files and ids, from the light columns extracted
read-only on the cluster (scripts/ag_vbf_cluster.py light) and the local task files. Own code, no study tool imported.

Checks: Gator training ids = every Gator task row of soil_v3 (tiers 0-12) = the validated-id list = the drives in the
Gator per-arena file = the drives in G_full; H_full drives = the same ids without the gator__ prefix = every soil drive of
the E4 f104 HMMWV file; H_full = F104all row for row; vehicle, domain, arena columns; tiers per drive = task tier; group
splits equal across vehicles; fitted rows (deploy: split train; holdout: train minus md5(group) % 5 == 0); standing-start
rows; no suite group in any training file; statuses; stage-1 G = the tiers 0-6 part of G_full; share of groups in which
every route fails (per vehicle, all tiers).
  python scripts/ag_vbf_data.py <K3> <out json>"""
import sys, json, hashlib
from collections import Counter
from pathlib import Path
import numpy as np
K3 = Path(sys.argv[1]); OUT = sys.argv[2]; C = K3 / 'verify_gator_full/cluster_out'
md5 = lambda s: hashlib.md5(s.encode()).hexdigest()
isdev = lambda g: int(md5(g), 16) % 5 == 0
L = {n: dict(np.load(C / f'{n}_light.npz', allow_pickle=True)) for n in
     ('G_full_f104_gator_soil', 'H_full_f104_hmmwv_soil', 'F104all_f104_hmmwv_soil', 'ci_f104_gator_crm', 'ci_f104_hmmwv_both', 'G_f104_gator_soil')}
SH = {Path(p).name: h for h, p in (l.split() for l in open(C / 'sha256.txt'))}
res = dict(sha256={k: v[:8] for k, v in SH.items()})
v3 = json.load(open(K3 / 'e3/tasks/soil_v3.json'))
gt = {r['id']: r for r in v3 if r['id'].startswith('gator__') and r['tier'] >= 0}
val = [l.strip() for l in open(K3 / 'e5/ids_bf/gator_soil_validated_all.txt') if l.strip()]
res['validated_list'] = dict(n=len(val), unique=len(set(val)), sha256=hashlib.sha256(open(K3 / 'e5/ids_bf/gator_soil_validated_all.txt', 'rb').read()).hexdigest()[:8],
                             equal_cluster=hashlib.sha256(open(K3 / 'e5/ids_bf/gator_soil_validated_all.txt', 'rb').read()).hexdigest() == SH['gator_soil_validated_all.txt'])
ep = lambda n: L[n]['episode'].astype(str)
Gf, Hf, Fa, Gc, E4, G1 = (L[n] for n in L)
e4soil = E4['domain'] == 1
res['ids'] = dict(
    gator_task_rows=len(gt), gator_task_tiers=dict(sorted(Counter(r['tier'] for r in gt.values()).items())),
    validated_eq_tasks=set(val) == set(gt),
    G_full_eq_validated=set(ep('G_full_f104_gator_soil')) == set(val),
    gator_file_eq_validated=set(ep('ci_f104_gator_crm')) == set(val),
    G_full_rows_eq_gator_file=bool((Gf['id'] == Gc['id']).all()) if len(Gf['id']) == len(Gc['id']) else False,
    H_full_eq_validated_stripped=set(ep('H_full_f104_hmmwv_soil')) == {v[len('gator__'):] for v in val},
    H_full_eq_E4_soil=set(ep('H_full_f104_hmmwv_soil')) == set(E4['episode'][e4soil].astype(str)),
    H_full_rows_eq_F104all=bool(len(Hf['id']) == len(Fa['id']) and (Hf['id'] == Fa['id']).all()),
    H_full_rows_eq_E4_soil_rows=bool(len(Hf['id']) == int(e4soil.sum()) and (Hf['id'] == E4['id'][e4soil]).all()),
    n_drives=dict(G_full=len(set(ep('G_full_f104_gator_soil'))), H_full=len(set(ep('H_full_f104_hmmwv_soil'))), E4_soil=len(set(E4['episode'][e4soil]))))
for n, want in (('G_full_f104_gator_soil', 'gator'), ('H_full_f104_hmmwv_soil', 'hmmwv')):
    z = L[n]
    res.setdefault('columns', {})[n] = dict(vehicle=dict(Counter(z['vehicle'].astype(str))), domain=dict(Counter(z['domain'].tolist())),
                                            arena=dict(Counter(z['arena'].astype(str))), rows=len(z['id']),
                                            split_rows=dict(Counter(z['split'].astype(str))),
                                            fit_deploy=int((z['split'] == 'train').sum()),
                                            fit_holdout=int(((z['split'] == 'train') & ~np.array([isdev(g) for g in z['group'].astype(str)])).sum()),
                                            startup_train_rows=int(((z['split'] == 'train') & (z['anchor_frame'] == 0)).sum()),
                                            groups_by_split={s: len(set(z['group'][z['split'] == s].astype(str))) for s in ('train', 'val', 'test')},
                                            group_prefixes=dict(Counter(g.rsplit('_', 1)[0] for g in set(z['group'].astype(str)))),
                                            vehicle_ok=bool((z['vehicle'].astype(str) == want).all()))
# tier per drive vs task tier, and vs the HMMWV twin
tier_g = {e: int(t) for e, t in zip(ep('G_full_f104_gator_soil'), Gf['tier'])}
tier_h = {e: int(t) for e, t in zip(ep('H_full_f104_hmmwv_soil'), Hf['tier'])}
res['tiers'] = dict(G_full_vs_task_mismatch=sum(tier_g[e] != gt[e]['tier'] for e in tier_g),
                    H_vs_G_twin_mismatch=sum(tier_h[e[7:]] != t for e, t in tier_g.items()),
                    drives_per_tier=dict(sorted(Counter(tier_g.values()).items())))
sg = {g: s for g, s in zip(Gf['group'].astype(str), Gf['split'].astype(str))}
sh = {g: s for g, s in zip(Hf['group'].astype(str), Hf['split'].astype(str))}
res['splits_equal_across_vehicles'] = sg == sh
# suite groups
suite = set((K3 / 'e6/picks/crm_bfull/f104/G_full_free/groups.txt').read_text().split())
pat = ('pair_group', 'crm_eval_group', '_test_group_', '_heldout_group_', '_dev_group_')
res['suite_check'] = {n: dict(suite_groups=len(suite & set(L[n]['group'].astype(str))),
                              pattern_hits=sum(any(p in g for p in pat) for g in set(L[n]['group'].astype(str))))
                      for n in ('G_full_f104_gator_soil', 'H_full_f104_hmmwv_soil')}
# statuses per drive and all-fail groups (standing-start row = one per drive)
for n in ('G_full_f104_gator_soil', 'H_full_f104_hmmwv_soil'):
    z = L[n]; st = z['anchor_frame'] == 0
    res.setdefault('statuses', {})[n] = dict(Counter(z['status'][st].astype(str)))
    g = z['group'][st].astype(str); f = z['fail'][st].astype(int)
    per = {}
    for gg, ff in zip(g, f):
        per.setdefault(gg, []).append(ff)
    res.setdefault('groups_all_fail', {})[n] = dict(groups=len(per), all_fail=sum(all(v) for v in per.values()), none_fail=sum(not any(v) for v in per.values()),
                                                   drive_fail_pct=100 * float(f.mean()), drives=int(st.sum()))
# stage-1 G = tiers 0-6 part of G_full
g1 = set(G1['episode'].astype(str))
res['stage1_G'] = dict(drives=len(g1), subset_of_G_full=g1 <= set(tier_g), eq_tiers0_6=g1 == {e for e, t in tier_g.items() if t <= 6},
                       rows=len(G1['id']), fit_deploy=int((G1['split'] == 'train').sum()))
json.dump(res, open(OUT, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
