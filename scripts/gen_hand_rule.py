"""Fit the non-learned hand rule on the f104 TRAINING routes (same rows the risk model trained on).

Pairwise logistic on within-group pairs (unsafe vs clean route of the same start/goal), 22 z-scored features,
L2 1e-3, LBFGS. Output gen_v1/hand_rule.json. Higher score = riskier.
"""
import collections, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_terrain_features import FEATURES, route_features
from f104_night_train import cell_auc

R = 'artifacts/traverse/fdm_f104_50h_20260909'
d = np.load(R + '/night2_v1/station_ds_all.npz', allow_pickle=True)
split = d['split'].astype(str); grp = d['group'].astype(str); y = d['unsafe'].astype(int); L = d['route_len'].astype(float)
idx = np.where(split == 'train')[0]
XA = d['X']
F = np.stack([route_features(XA[i].astype(np.float32), float(L[i])) for i in idx])
mu = F.mean(0); sd = F.std(0) + 1e-9; Z = (F - mu) / sd
by = collections.defaultdict(list)
for j, g in enumerate(grp[idx]): by[g].append(j)
P, N = [], []
for jj in by.values():
    jj = np.array(jj); yy = y[idx][jj]; p, n = jj[yy == 1], jj[yy == 0]
    if len(p) and len(n):
        pp, nn = np.meshgrid(p, n, indexing='ij'); P.append(pp.ravel()); N.append(nn.ravel())
p = np.concatenate(P); n = np.concatenate(N)
D = torch.tensor(Z[p] - Z[n], dtype=torch.float64)
w = torch.zeros(Z.shape[1], dtype=torch.float64, requires_grad=True)
opt = torch.optim.LBFGS([w], max_iter=500, line_search_fn='strong_wolfe')
def closure():
    opt.zero_grad(); l = torch.nn.functional.softplus(-(D @ w)).mean() + 1e-3 * (w ** 2).sum(); l.backward(); return l
opt.step(closure)
w = w.detach().numpy()
s = Z @ w
W, wn = cell_auc(y[idx].astype(float), s, grp[idx])
out = dict(features=FEATURES, mu=mu.tolist(), sd=sd.tolist(), w=w.tolist(), n_train_routes=int(len(idx)),
           n_pairs=int(len(p)), train_within_group_auc=float(W),
           scope='fit on night2_v1/station_ds_all.npz split==train, label unsafe; higher score = riskier')
Path(R + '/gen_v1').mkdir(parents=True, exist_ok=True)
json.dump(out, open(R + '/gen_v1/hand_rule.json', 'w'), indent=1)
print('pairs', len(p), 'train within-group AUC', round(float(W), 3))
print(dict(zip(FEATURES, np.round(w, 2))))
