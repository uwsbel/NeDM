#!/usr/bin/env python3
"""VERIFY_E5a: (1) every synced deploy checkpoint loads (ci_train.load_ci_model) and reproduces its trainer's stored
member logits on the val rows of its arenas (local 5090, TF32 off); (2) my own within-group AUC (pair-pooled, my code)
of several ensembles on the standing-start val rows, compared with e5/offline/auc_summary.json; (3) unsafe rates."""
import sys, os, glob, json, hashlib
import numpy as np, torch
K3 = sys.argv[1]; OUT = sys.argv[2]
sys.path.insert(0, 'scripts')
import ci_train as CT
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
EV = dict(f104=f'{K3}/e5/eval/EV_f104_hmmwv_rigid.npz', g203=f'{K3}/e5/eval/EV_g203_hmmwv_rigid.npz',
          g228=f'{K3}/e5/eval/EV_g228_hmmwv_rigid.npz', f104_gator=f'{K3}/e5/eval/EV_f104_gator_rigid.npz')
MODELS = dict(M1a=('M1a', 'M1a_rigid_deploy'), M1b=('M1b', 'M1b_rigid_deploy'), M2=('M2', 'M2_rigid_deploy'), M3a=('M3a', 'M3a_rigid_deploy'),
              M3b=('M3b', 'M3b_rigid_deploy'), A3=('A3', 'A3_rigid_deploy'), G=('G', 'G_rigid_deploy'))
summ = json.load(open(f'{K3}/e5/offline/auc_summary.json'))['models']

def my_within_auc(y, s, g):
    """pairs (unsafe, safe) inside each group, pooled over groups; ties count 1/2."""
    num = den = 0.0; per = []
    for c in np.unique(g):
        k = g == c; yy, ss = y[k], s[k]
        pos, neg = ss[yy == 1], ss[yy == 0]
        if len(pos) == 0 or len(neg) == 0: continue
        cmp = np.sign(pos[:, None] - neg[None, :])
        num += (cmp > 0).sum() + 0.5 * (cmp == 0).sum(); den += cmp.size; per.append(((cmp > 0).sum() + 0.5 * (cmp == 0).sum()) / cmp.size)
    return num / den, float(np.mean(per)), int(den)

data = {}
for k, p in EV.items():
    z = np.load(p, allow_pickle=True)
    m = (z['split'].astype(str) == 'val') & (z['domain'].astype(int) == 0)
    idx = np.flatnonzero(m)
    data[k] = dict(idx=idx, X=z['X'][idx], ctx=z['ctx'][idx], id=z['id'][idx].astype(str), group=z['group'][idx].astype(str),
                   af=z['anchor_frame'][idx].astype(int), unsafe=z['unsafe'][idx].astype(int), fail=z['fail'][idx].astype(int),
                   sha=hashlib.sha256(open(p, 'rb').read()).hexdigest())
    print(k, len(idx), 'val rows', len(set(data[k]['group'])), 'groups', flush=True)
res = dict(roundtrip={}, auc={}, rates={})
for name, (d, tag) in MODELS.items():
    paths = sorted(glob.glob(f'{K3}/e5/deploy/{d}/{tag}_s*.pt'))
    sums = dict(l.split()[::-1] for l in open(f'{K3}/e5/deploy/{d}/SHA256SUMS'))
    assert all(hashlib.sha256(open(p, 'rb').read()).hexdigest() == sums[os.path.basename(p)] for p in paths)
    mem = [CT.load_ci_model(p, device='cuda') for p in paths]
    L = np.load(f'{K3}/e5/deploy/{d}/{tag}_logits.npz', allow_pickle=True)
    pos = {i: j for j, i in enumerate(L['id'].astype(str))}
    res['roundtrip'][name] = {}; res['auc'][name] = {}
    for ek, D in data.items():
        Z = np.stack([CT.score(m, ck, np.asarray(D['X'], np.float32), np.asarray(D['ctx'][:, list(ck['geom_cols'])], np.float32)) for m, ck in mem])
        ens = Z.mean(0)
        common = [(a, pos[i]) for a, i in enumerate(D['id']) if i in pos]
        if common and ek != 'f104_gator' or (common and name == 'G'):
            a_, b_ = np.array([c[0] for c in common]), np.array([c[1] for c in common])
            res['roundtrip'][name][ek] = dict(rows=len(common), max_abs_member=float(np.abs(Z[:, a_] - L['member_logits'][:, b_]).max()),
                                              max_abs_ensemble=float(np.abs(ens[a_] - L['ensemble_logit'][b_]).max()),
                                              labels_equal=bool((L['unsafe'][b_] == D['unsafe'][a_]).all()))
        st = D['af'] == 0
        w, wmean, npairs = my_within_auc(D['unsafe'][st], ens[st].astype(np.float64), D['group'][st])
        wm, _, _ = my_within_auc(D['unsafe'][~st], ens[~st].astype(np.float64), D['group'][~st])
        theirs = summ[name]['evals'][ek]['val']
        res['auc'][name][ek] = dict(mine_start=w, mine_start_group_mean=wmean, pairs=npairs, mine_moving=wm,
                                    theirs_start=theirs['W_unsafe_start'], theirs_moving=theirs['W_unsafe_moving'],
                                    diff_start=w - theirs['W_unsafe_start'], diff_moving=wm - theirs['W_unsafe_moving'])
        print(f"{name:4s} {ek:10s} start mine {w:.4f} theirs {theirs['W_unsafe_start']:.4f} | moving mine {wm:.4f} theirs {theirs['W_unsafe_moving']:.4f}"
              + (f" | roundtrip {res['roundtrip'][name][ek]}" if ek in res['roundtrip'][name] else ''), flush=True)
for ek, D in data.items():
    st = D['af'] == 0
    res['rates'][ek] = dict(startup_rows=int(st.sum()), unsafe_start=float(D['unsafe'][st].mean()), fail_start=float(D['fail'][st].mean()))
print('rates', res['rates'])
# the Gator and HMMWV f104 val rows describe the same routes? (episode ids without the prefix)
a = sorted(i.replace('gator__', '') for i, s in zip(data['f104_gator']['id'], data['f104_gator']['af']) if s == 0)
b = sorted(i for i, s in zip(data['f104']['id'], data['f104']['af']) if s == 0)
res['same_startup_routes_gator_vs_hmmwv'] = a == b
print('same standing-start routes on the Gator and HMMWV val rows:', a == b)
json.dump(res, open(OUT, 'w'), indent=1)
