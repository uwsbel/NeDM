#!/usr/bin/env python3
"""VERIFY_S1: (1) every synced soil deploy checkpoint loads (ci_train.load_ci_model), its sha256 equals SHA256SUMS, and it
reproduces its trainer's stored member logits on the val rows of its own arenas (local 5090, TF32 off);
(2) my own within-group AUC (pair-pooled, my code, my row selection) of several holdout ensembles on the standing-start
dev-fold + val rows, compared with e5/offline/soil_s1_{A,B}_auc.json; my own fitted-group guard from the logits files.
  PYTHONPATH=src:scripts python rescore_s1.py <K3> <out json>"""
import sys, os, glob, json, hashlib
import numpy as np, torch
K3 = sys.argv[1]; OUT = sys.argv[2]
sys.path.insert(0, 'scripts')
import ci_train as CT
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
md5 = lambda s: hashlib.md5(s.encode()).hexdigest()
isdev = lambda g: int(md5(g), 16) % 5 == 0
EVF = dict(f104=f'{K3}/e5/eval_soil/EV_f104_hmmwv_crm.npz', g203=f'{K3}/e5/eval_soil/EV_g203_hmmwv_crm.npz',
           g228=f'{K3}/e5/eval_soil/EV_g228_hmmwv_crm.npz', gator=f'{K3}/e5/eval_soil/EV_f104_gator_crm.npz')
A = json.load(open(f'{K3}/e5/offline/soil_s1_A_auc.json'))['results']
B = json.load(open(f'{K3}/e5/offline/soil_s1_B_auc.json'))['results']
SH = {}
for l in open(f'{K3}/verify_s1/cluster_out/sha256_npz.txt'):
    h, p = l.split(); SH[os.path.basename(p)] = h

def wauc(y, s, g):
    """pairs (unsafe, safe) inside each group, pooled over groups; ties count 1/2."""
    num = den = 0.0
    for c in np.unique(g):
        k = g == c; p, n = s[k][y[k] == 1], s[k][y[k] == 0]
        if len(p) and len(n):
            d = np.sign(p[:, None] - n[None, :]); num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return num / den, int(den)

def pick_unsafe(y, s, g):
    """share of groups whose lowest-risk route is unsafe (ties: first in file order, as argmin)."""
    out = []
    for c in np.unique(g):
        k = np.flatnonzero(g == c); out.append(y[k[np.argmin(s[k])]])
    return float(np.mean(out))

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()

EV = {}
for k, p in EVF.items():
    assert sha(p) == SH[os.path.basename(p)], f'{p} differs from the cluster copy'
    z = np.load(p, allow_pickle=True)
    sp = z['split'].astype(str); grp = z['group'].astype(str)
    dv = (sp == 'train') & np.array([isdev(g) for g in grp])
    m = (sp == 'val') | dv
    idx = np.flatnonzero(m)
    EV[k] = dict(X=np.asarray(z['X'][idx], np.float32), ctx=z['ctx'][idx], id=z['id'][idx].astype(str), group=grp[idx],
                 split=sp[idx], af=z['anchor_frame'][idx].astype(int), unsafe=z['unsafe'][idx].astype(int),
                 domain=z['domain'][idx].astype(int), vehicle=z['vehicle'][idx].astype(str))
    assert (EV[k]['domain'] == 1).all()
    print(k, len(idx), 'dev+val rows', len(set(grp[idx])), 'groups', flush=True)

def members(paths):
    return [CT.load_ci_model(p, device='cuda') for p in paths]

def ens_score(mem, D):
    return np.stack([CT.score(m, ck, D['X'], np.asarray(D['ctx'][:, list(ck['geom_cols'])], np.float32)) for m, ck in mem])

res = dict(checkpoints={}, auc={})
# (1) deploy checkpoints: sha256 + reproduce trainer logits on val rows of their own arenas
DEP = dict(M1a=('M1a_soil', 'M1a_soil_deploy', ['f104']), M1b=('M1b_soil', 'M1b_soil_deploy', ['f104']),
           M2=('M2_soil', 'M2_soil_deploy', ['f104', 'g203']), M3a=('M3a_soil', 'M3a_soil_deploy', ['f104', 'g203', 'g228']),
           M3b=('M3b_soil', 'M3b_soil_deploy', ['f104', 'g203', 'g228']), A3=('A3_soil', 'A3_soil_deploy', ['f104', 'g203', 'g228']),
           G=('G_soil', 'G_soil_deploy', ['gator']))
for name, (d, tag, arenas) in DEP.items():
    paths = sorted(glob.glob(f'{K3}/e5/deploy/{d}/{tag}_s*.pt'))
    sums = dict(l.split()[::-1] for l in open(f'{K3}/e5/deploy/{d}/SHA256SUMS'))
    sha_ok = all(sha(p) == sums[os.path.basename(p)] for p in paths)
    mem = members(paths)
    L = np.load(f'{K3}/e5/deploy/{d}/{tag}_logits.npz', allow_pickle=True)
    pos = {i: j for j, i in enumerate(L['id'].astype(str))}
    r = dict(n_members=len(paths), sha_ok=sha_ok, modes=sorted({ck['mode'] for _, ck in mem}), per_arena={})
    for ar in arenas:
        D = EV[ar]; v = D['split'] == 'val'
        sub = {k: (x[v] if isinstance(x, np.ndarray) and len(x) == len(v) else x) for k, x in D.items()}
        Z = ens_score(mem, sub)
        common = [(a, pos[i]) for a, i in enumerate(sub['id']) if i in pos]
        a_, b_ = np.array([c[0] for c in common]), np.array([c[1] for c in common])
        r['per_arena'][ar] = dict(val_rows=int(v.sum()), matched=len(common), max_abs_member=float(np.abs(Z[:, a_] - L['member_logits'][:, b_]).max()),
                                  max_abs_ensemble=float(np.abs(Z.mean(0)[a_] - L['ensemble_logit'][b_]).max()),
                                  labels_equal=bool((L['unsafe'][b_] == sub['unsafe'][a_]).all()), val_rows_fitted=int(L['fit'][b_].sum()))
    res['checkpoints'][name] = r
    print(name, r, flush=True)
    del mem; torch.cuda.empty_cache()

# (2) my own AUC of holdout ensembles on standing-start dev+val rows (and val only for G / H on the Gator rows)
HO = f'{K3}/e5/train/soil_s1/offline_soil'
TODO = [('M1', 'f104'), ('M1', 'g203'), ('M1', 'g228'), ('M3', 'f104'), ('M3', 'g203'), ('M3', 'g228'), ('A3', 'f104'), ('A3', 'g203'), ('A3', 'g228'),
        ('LC272', 'f104'), ('LC545', 'f104'), ('G', 'gator'), ('G', 'f104'), ('M1', 'gator')]
cache = {}
for mname, ar in TODO:
    if mname not in cache:
        paths = sorted(glob.glob(f'{HO}/{mname}_soil_holdout_s*.pt'))
        cache[mname] = (members(paths), np.load(f'{HO}/{mname}_soil_holdout_logits.npz', allow_pickle=True))
    mem, L = cache[mname]
    assert len(mem) == 5 and all(ck['mode'] == 'holdout' for _, ck in mem)
    fitted = set(L['group'][L['fit'].astype(bool)].astype(str))
    D = EV[ar]; st = D['af'] == 0
    keep = st & ~np.isin(D['group'], sorted(fitted))
    s = ens_score(mem, {k: (x[keep] if isinstance(x, np.ndarray) and len(x) == len(keep) else x) for k, x in D.items()}).mean(0).astype(np.float64)
    y, g, spl = D['unsafe'][keep], D['group'][keep], D['split'][keep]
    mine, pairs = wauc(y, s, g); mv, _ = wauc(y[spl == 'val'], s[spl == 'val'], g[spl == 'val'])
    key = {'M1': 'M1_holdout', 'M3': 'M3_holdout', 'A3': 'A3_holdout', 'LC272': 'LC272_holdout', 'LC545': 'LC545_holdout', 'G': 'G_holdout'}[mname]
    if ar in ('f104', 'g203', 'g228') and mname != 'G':
        th = A[key][ar]; theirs = th['dev+val']['ensemble']['crm']['startup']['W_unsafe']; theirs_val = th['val']['ensemble']['crm']['startup']['W_unsafe']
        theirs_pick = th['dev+val']['ensemble']['crm']['startup']['pick_fail_unsafe']
    else:
        bk = 'H_holdout' if mname == 'M1' else 'G_holdout'
        ek = 'f104_gator_rows' if ar == 'gator' else 'f104_hmmwv_rows'
        th = B[bk][ek]; theirs = th['dev+val']['ensemble']['crm']['startup']['W_unsafe']; theirs_val = th['val']['ensemble']['crm']['startup']['W_unsafe']
        theirs_pick = th['dev+val']['ensemble']['crm']['startup']['pick_fail_unsafe']
    pk = pick_unsafe(y, s, g)
    res['auc'][f'{mname}_holdout@{ar}'] = dict(groups=len(set(g)), rows=int(keep.sum()), rows_dropped_fitted=int((st & ~keep).sum()), pairs=pairs,
                                               mine=mine, theirs=theirs, diff=mine - theirs, mine_val=mv, theirs_val=theirs_val,
                                               mine_pick_unsafe=pk, theirs_pick_unsafe=theirs_pick,
                                               groups_all_unsafe=int(sum(y[g == c].all() for c in np.unique(g))),
                                               rows_in_mixed_groups=int(sum(len(np.unique(y[g == c])) == 2 and (g == c).sum() for c in np.unique(g))))
    print(f'{mname}_holdout on {ar}: mine {mine:.4f} theirs {theirs:.4f} | val mine {mv:.4f} theirs {theirs_val:.4f} | pick mine {pk:.4f} theirs {theirs_pick:.4f} '
          f'({len(set(g))} groups, {int(keep.sum())} rows, dropped {int((st & ~keep).sum())})', flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
