#!/usr/bin/env python3
"""VERIFY_gator_full (independent checker of task B stage 2): checkpoints and offline ranking.

(1) every G_full / H_full deploy and holdout checkpoint: sha256 against SHA256SUMS, the pick lock, the model manifest and the
    cluster copy; loads with ci_train.load_ci_model; stored recipe fields (arch, cond, ctx, mode, domain filter, epochs,
    seed, data path, fitted rows); scores the val rows of its own vehicle's evaluation file and is compared with the
    trainer's stored member / ensemble logits (local 5090, TF32 off).
(2) my own within-group AUC of "unsafe" from a standing start (pairs (unsafe, safe) inside each group, pooled; ties 1/2),
    my own "lowest-risk pick is unsafe" share, random / best-possible pick; holdout ensembles on the dev fold + val
    (md5(group) % 5 == 0 among training groups, minus any group the holdout run fitted), deploy ensembles on val; also the
    count of groups that contribute pairs. Compared with e5/offline/soil_bf_B_auc.json.
  PYTHONPATH=src:scripts python scripts/ag_vbf_rescore.py <K3> <cluster sha256_train.txt> <out json>"""
import sys, os, glob, json, hashlib
import numpy as np, torch
K3, CLSHA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, 'scripts')
import ci_train as CT
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
md5 = lambda s: hashlib.md5(s.encode()).hexdigest()
isdev = lambda g: int(md5(g), 16) % 5 == 0


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def wauc(y, s, g):
    num = den = 0.0; ng = 0
    for c in np.unique(g):
        k = g == c; p, n = s[k][y[k] == 1], s[k][y[k] == 0]
        if len(p) and len(n):
            d = np.sign(p[:, None] - n[None, :]); num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size; ng += 1
    return (num / den if den else float('nan')), int(den), ng


def picks(y, s, g):
    lo, rnd, best = [], [], []
    for c in np.unique(g):
        k = np.flatnonzero(g == c); lo.append(y[k[np.argmin(s[k])]]); rnd.append(y[k].mean()); best.append(y[k].min())
    return float(np.mean(lo)), float(np.mean(rnd)), float(np.mean(best))


EVF = dict(gator=f'{K3}/e5/eval_soil_bf/EV_f104_gator_full_crm.npz', hmmwv=f'{K3}/e5/eval_soil_bf/EV_f104_hmmwv_gatorids_full_crm.npz')
EXPECT_EV = dict(gator='01bbbee3', hmmwv='375a3887')
CL = {}
for l in open(CLSHA):
    h, p = l.split(); CL[p] = h
res = dict(ev={}, checkpoints={}, auc={})
EV = {}
for k, p in EVF.items():
    h = sha(p); assert h.startswith(EXPECT_EV[k]), (p, h)
    z = np.load(p, allow_pickle=True)
    sp = z['split'].astype(str); grp = z['group'].astype(str)
    dv = (sp == 'train') & np.array([isdev(g) for g in grp])
    m = (sp == 'val') | dv
    idx = np.flatnonzero(m)
    EV[k] = dict(X=np.asarray(z['X'][idx], np.float32), ctx=z['ctx'][idx], id=z['id'][idx].astype(str), group=grp[idx],
                 split=sp[idx], af=z['anchor_frame'][idx].astype(int), unsafe=z['unsafe'][idx].astype(int),
                 domain=z['domain'][idx].astype(int), vehicle=z['vehicle'][idx].astype(str), tier=z['tier'][idx].astype(int))
    assert (EV[k]['domain'] == 1).all()
    res['ev'][k] = dict(sha256=h, rows_dev_val=int(len(idx)), groups=len(set(grp[idx])), vehicles=sorted(set(EV[k]['vehicle'])),
                        splits={s: int((sp[idx] == s).sum()) for s in ('train', 'val')}, tiers=sorted(set(EV[k]['tier'].tolist())),
                        other_splits_in_file={s: int((sp == s).sum()) for s in sorted(set(sp))})
    print(k, res['ev'][k], flush=True)


def sub(D, keep):
    return {k: (x[keep] if isinstance(x, np.ndarray) and len(x) == len(keep) else x) for k, x in D.items()}


def ens(mem, D):
    return np.stack([CT.score(m, ck, D['X'], np.asarray(D['ctx'][:, list(ck['geom_cols'])], np.float32)) for m, ck in mem])


lock = json.load(open(f'{K3}/e6/picks/LOCK_crm_bfull.json'))
lock_sha = {os.path.basename(x['path']): x['sha256'] for d in lock['dirs'] for x in d['models']}
man = json.load(open(f'{K3}/e5/deploy/soil_bf_models.json'))['models']
OFF = json.load(open(f'{K3}/e5/offline/soil_bf_B_auc.json'))['results']
MODELS = dict(
    G_full_deploy=(sorted(glob.glob(f'{K3}/e5/deploy/G_full_soil/G_full_soil_deploy_s*.pt')), f'{K3}/e5/deploy/G_full_soil/G_full_soil_deploy_logits.npz', 'gator', 'deploy', 'G_full_soil'),
    H_full_deploy=(sorted(glob.glob(f'{K3}/e5/deploy/H_full_soil/H_full_soil_deploy_s*.pt')), f'{K3}/e5/deploy/H_full_soil/H_full_soil_deploy_logits.npz', 'hmmwv', 'deploy', 'H_full_soil'),
    G_full_holdout=(sorted(glob.glob(f'{K3}/e5/train/soil_bf/offline_soil_bf/G_full_soil_holdout_s*.pt')), f'{K3}/e5/train/soil_bf/offline_soil_bf/G_full_soil_holdout_logits.npz', 'gator', 'holdout', None),
    H_full_holdout=(sorted(glob.glob(f'{K3}/e5/train/soil_bf/offline_soil_bf/H_full_soil_holdout_s*.pt')), f'{K3}/e5/train/soil_bf/offline_soil_bf/H_full_soil_holdout_logits.npz', 'hmmwv', 'holdout', None),
    G_holdout=(sorted(glob.glob(f'{K3}/e5/train/soil_s1/offline_soil/G_soil_holdout_s*.pt')), f'{K3}/e5/train/soil_s1/offline_soil/G_soil_holdout_logits.npz', 'gator', 'holdout', None),
    H_holdout=(sorted(glob.glob(f'{K3}/e5/train/soil_s1/offline_soil/M1_soil_holdout_s*.pt')), f'{K3}/e5/train/soil_s1/offline_soil/M1_soil_holdout_logits.npz', 'hmmwv', 'holdout', None),
)
for name, (paths, lpath, own, mode, mkey) in MODELS.items():
    assert len(paths) == 5, (name, paths)
    rec = dict(files=[os.path.relpath(p, K3) for p in paths], sha={})
    sums = {}
    sf = os.path.join(os.path.dirname(paths[0]), 'SHA256SUMS')
    if os.path.exists(sf):
        sums = dict(l.split()[::-1] for l in open(sf))
    for p in paths:
        b = os.path.basename(p); h = sha(p)
        clk = [v for kk, v in CL.items() if kk.endswith('/' + b)]
        rec['sha'][b] = dict(sha256=h, SHA256SUMS=(sums.get(b) == h) if sums else None, lock=(lock_sha.get(b) == h) if b in lock_sha else None,
                             manifest=(man[mkey]['sha256'].get(b) == h) if mkey else None, cluster=(clk == [h]) if clk else None)
    mem = [CT.load_ci_model(p, device='cuda') for p in paths]
    ck0 = mem[0][1]
    rec['recipe'] = [dict(seed=ck.get('seed'), mode=ck.get('mode'), arch=ck.get('arch'), cond=ck.get('cond'), ctx=ck.get('ctx_mode'),
                          domain_filter=ck.get('domain_filter'), epochs=ck.get('epochs'), split_eval=ck.get('split_eval'),
                          train_rows=ck.get('train_rows'), ds=ck.get('ds'), split_hash=ck.get('split_hash'), lr=ck.get('lr'), wd=ck.get('wd'))
                     for _, ck in mem]
    L = np.load(lpath, allow_pickle=True)
    pos = {i: j for j, i in enumerate(L['id'].astype(str))}
    D = EV[own]; v = D['split'] == 'val'; S = sub(D, v)
    Z = ens(mem, S)
    common = [(a, pos[i]) for a, i in enumerate(S['id']) if i in pos]
    a_, b_ = np.array([c[0] for c in common]), np.array([c[1] for c in common])
    rec['logits_vs_trainer'] = dict(val_rows=int(v.sum()), matched=len(common), max_abs_member=float(np.abs(Z[:, a_] - L['member_logits'][:, b_]).max()),
                                    max_abs_ensemble=float(np.abs(Z.mean(0)[a_] - L['ensemble_logit'][b_]).max()),
                                    labels_equal=bool((L['unsafe'][b_] == S['unsafe'][a_]).all()))
    fitted = set(L['group'][L['fit'].astype(bool)].astype(str))
    res['checkpoints'][name] = rec
    print(name, json.dumps(rec['logits_vs_trainer']), {b: all(x is not False for x in r.values()) for b, r in rec['sha'].items()}, flush=True)
    for ev in ('gator', 'hmmwv'):
        D = EV[ev]; st = D['af'] == 0
        keep = st & ~np.isin(D['group'], sorted(fitted))
        if mode == 'deploy':
            keep = keep & (D['split'] == 'val')
        S = sub(D, keep)
        s = ens(mem, S).mean(0).astype(np.float64)
        y, g, spl = S['unsafe'], S['group'], S['split']
        a1, n1, g1 = wauc(y, s, g)
        av, nv, gv = wauc(y[spl == 'val'], s[spl == 'val'], g[spl == 'val'])
        pk, rnd, best = picks(y, s, g)
        okey = f'f104_{ev}_all'
        th = OFF[name][okey]
        blk = 'val' if mode == 'deploy' else 'dev+val'
        theirs = th[blk]['ensemble']['crm']['startup']
        r = dict(rows=int(keep.sum()), groups=len(set(g)), groups_with_pairs=g1, pairs=n1, auc=a1, auc_val=av, pairs_val=nv, groups_val_with_pairs=gv,
                 pick_unsafe=pk, random_pick=rnd, best_pick=best, groups_all_unsafe=int(sum(y[g == c].all() for c in np.unique(g))),
                 rows_dropped_fitted=int((st & np.isin(D['group'], sorted(fitted))).sum()),
                 theirs_auc=theirs['W_unsafe'], theirs_pairs=theirs['Wn_unsafe'], theirs_pick=theirs['pick_fail_unsafe'],
                 theirs_random=theirs.get('random_fail_unsafe'), theirs_best=theirs.get('oracle_fail_unsafe'),
                 theirs_auc_val=th['val']['ensemble']['crm']['startup']['W_unsafe'])
        res['auc'][f'{name}@{ev}'] = r
        print(f'  {name} on {ev} ({blk}): mine {a1:.4f} ({n1} pairs, {g1}/{len(set(g))} groups with pairs) theirs {r["theirs_auc"]:.4f} ({r["theirs_pairs"]}) | '
              f'val mine {av:.4f} theirs {r["theirs_auc_val"]:.4f} | pick {pk:.3f}/{r["theirs_pick"]:.3f} random {rnd:.3f}/{r["theirs_random"]} best {best:.3f}/{r["theirs_best"]}', flush=True)
    del mem; torch.cuda.empty_cache()
json.dump(res, open(OUT, 'w'), indent=1)
