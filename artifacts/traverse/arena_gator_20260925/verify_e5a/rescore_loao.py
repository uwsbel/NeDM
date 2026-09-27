#!/usr/bin/env python3
"""VERIFY_E5a: my own dev+val standing-start within-group AUC for the holdout runs behind the leave-one-arena-out claim
(two arenas vs the better single arena on the unseen third), compared with auc_summary.json."""
import sys, glob, json, hashlib
import numpy as np, torch
K3 = sys.argv[1]; OUT = sys.argv[2]
sys.path.insert(0, 'scripts')
import ci_train as CT
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
md5 = lambda s: hashlib.md5(s.encode()).hexdigest()
summ = json.load(open(f'{K3}/e5/offline/auc_summary.json'))['models']
def wauc(y, s, g):
    num = den = 0.0
    for c in np.unique(g):
        k = g == c; p, n = s[k][y[k] == 1], s[k][y[k] == 0]
        if len(p) and len(n):
            d = np.sign(p[:, None] - n[None, :]); num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return num / den
TODO = {'g228': ['LC545_holdout', 'LOAO1_g203_holdout', 'LOAO2_f104_g203_holdout', 'LOAO1_g228_holdout'],
        'g203': ['LC545_holdout', 'LOAO1_g228_holdout', 'LOAO2_f104_g228_holdout', 'LOAO1_g203_holdout'],
        'f104': ['LOAO1_g203_holdout', 'LOAO1_g228_holdout', 'LOAO2_g203_g228_holdout', 'LC545_holdout']}
res = {}
for ar, models in TODO.items():
    z = np.load(f'{K3}/e5/eval/EV_{ar}_hmmwv_rigid.npz', allow_pickle=True)
    sp = z['split'].astype(str); grp = z['group'].astype(str)
    devf = np.array([int(md5(g), 16) % 5 == 0 for g in grp])
    m = ((sp == 'val') | ((sp == 'train') & devf)) & (z['anchor_frame'] == 0) & (z['domain'] == 0)
    idx = np.flatnonzero(m); X = np.asarray(z['X'][idx], np.float32); ctx = z['ctx'][idx]; y = z['unsafe'][idx].astype(int); g = grp[idx]
    for name in models:
        tag = name.replace('_holdout', '_rigid_holdout')
        mem = [CT.load_ci_model(p, device='cuda') for p in sorted(glob.glob(f'{K3}/e5/train/offline_rigid/{tag}_s*.pt'))]
        assert len(mem) == 5 and all(ck['mode'] == 'holdout' for _, ck in mem)
        s = np.mean([CT.score(mm, ck, X, np.asarray(ctx[:, list(ck['geom_cols'])], np.float32)) for mm, ck in mem], 0).astype(np.float64)
        mine = wauc(y, s, g); theirs = summ[name]['evals'][ar]['dev+val']['W_unsafe_start'] if 'dev+val' in summ[name]['evals'][ar] else None
        res[f'{name}@{ar}'] = dict(mine=mine, theirs=theirs, groups=len(set(g)), rows=len(idx))
        print(f'{name:26s} on {ar}: mine {mine:.4f} theirs {theirs} ({len(set(g))} groups, {len(idx)} rows)', flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
