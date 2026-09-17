"""Task B part 4: cross-arena distribution shift of each depth-side representation (the leading alternative
explanation for the raw-depth arm's closed-loop deficit), plus the part-2 d0 histogram in numbers.

Per channel and arena: mean/std/percentiles; versus f104: standardised mean difference, 1-Wasserstein distance
(metres and in f104-sigma), Kolmogorov-Smirnov statistic.
Per representation (multivariate): a two-sample classifier test (small MLP, held-out AUC of telling f104 from the
other arena) and two support measures -- the fraction of the other arena's points outside f104's per-channel box,
and outside f104's occupied cells on a 32-per-dimension grid.
"""
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TAGS = ['f104', 'g203', 'g216', 'g217', 'g228', 'g231']
H = 110.0
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RNG = np.random.default_rng(1)
N = 250_000

REPS = {'a_cell_v1_relative_depth+sec': ['dd', 'sec1'],
        'a_plus_v1_relative_depth+sec+start_sec': ['dd', 'sec1', 'sec1_0'],
        'b_absolute_range+sec': ['dep', 'sec'],
        'c_absolute_range+ray_dir': ['dep', 'ray_x', 'ray_y'],
        'd_backprojected_height': ['zpix'],
        'd_rel_backprojected_height_relative': ['zrel']}


def load(tag):
    z = np.load(HERE / 'points' / f'{tag}.npz', allow_pickle=True)
    d = {k: z[k].astype(np.float64) for k in ('dep', 'sec', 'd0', 's0', 'zpix', 'gx', 'gy', 'ray_x', 'ray_y')}
    keep = (np.abs(d['gx']) < 39.5) & (np.abs(d['gy']) < 39.5)
    d = {k: v[keep] for k, v in d.items()}
    d['dd'] = d['dep'] - d['d0']
    d['sec1'] = d['sec'] - 1.0
    d['sec1_0'] = d['s0'] - 1.0
    d['z0'] = H - d['d0'] / d['s0']
    d['zrel'] = d['zpix'] - d['z0']
    sel = RNG.choice(len(d['dep']), min(N, len(d['dep'])), replace=False)
    return {k: v[sel] for k, v in d.items()}


def w1(a, b):
    q = np.linspace(0, 1, 2001)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def ks(a, b):
    x = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), x, 'right') / len(a)
    fb = np.searchsorted(np.sort(b), x, 'right') / len(b)
    return float(np.abs(fa - fb).max())


def c2st(Xa, Xb, seed=0):
    """Held-out AUC of a small MLP trained to tell sample A from sample B. 0.5 = indistinguishable."""
    torch.manual_seed(seed)
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-12
    X = np.r_[(Xa - mu) / sd, (Xb - mu) / sd]
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    p = np.random.default_rng(seed).permutation(len(X))
    X, y = X[p], y[p]
    cut = len(X) // 2
    xt = torch.tensor(X[:cut], dtype=torch.float32, device=DEV); yt = torch.tensor(y[:cut], dtype=torch.float32, device=DEV)[:, None]
    xv = torch.tensor(X[cut:], dtype=torch.float32, device=DEV); yv = y[cut:]
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.GELU(),
                              torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    for _ in range(12):
        perm = torch.randperm(len(xt), device=DEV)
        for i in range(0, len(xt), 8192):
            j = perm[i:i + 8192]
            opt.zero_grad(set_to_none=True)
            torch.nn.functional.binary_cross_entropy_with_logits(net(xt[j]), yt[j]).backward()
            opt.step()
    with torch.no_grad():
        s = np.concatenate([net(xv[i:i + 200000]).cpu().numpy()[:, 0] for i in range(0, len(xv), 200000)])
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = yv.sum(); n0 = len(yv) - n1
    return float((r[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def support(Xa, Xb, bins=32):
    lo, hi = Xa.min(0), Xa.max(0)
    outside_box = float(np.mean(np.any((Xb < lo) | (Xb > hi), axis=1)))
    idx = lambda X: np.clip(((X - lo) / np.maximum(hi - lo, 1e-12) * bins).astype(np.int64), -1, bins)
    ia, ib = idx(Xa), idx(Xb)
    key = lambda I: (I + 1).astype(np.int64) @ (bins + 3) ** np.arange(I.shape[1])
    occ = np.unique(key(ia))
    kb = key(ib)
    return dict(outside_f104_box=outside_box,
                outside_f104_occupied_cells=float(np.mean(~np.isin(kb, occ))),
                f104_occupied_cells=int(len(occ)))


def main():
    per = {t: load(t) for t in TAGS}
    chans = sorted({c for v in REPS.values() for c in v})
    out = {'n_points_per_arena': {t: int(len(per[t]['dep'])) for t in TAGS},
           'channels': {}, 'representations': {}, 'd0_histogram': {}}
    for c in chans:
        ref = per['f104'][c]
        out['channels'][c] = {}
        for t in TAGS:
            v = per[t][c]
            row = {'mean': float(v.mean()), 'std': float(v.std()),
                   'p1': float(np.quantile(v, .01)), 'p50': float(np.median(v)), 'p99': float(np.quantile(v, .99)),
                   'min': float(v.min()), 'max': float(v.max())}
            if t != 'f104':
                pooled = np.sqrt((ref.var() + v.var()) / 2)
                row |= {'cohens_d_vs_f104': float((v.mean() - ref.mean()) / pooled),
                        'w1_vs_f104': w1(ref, v), 'w1_vs_f104_in_f104_sigma': w1(ref, v) / float(ref.std()),
                        'ks_vs_f104': ks(ref, v)}
            out['channels'][c][t] = row
    for rep, cs in REPS.items():
        Xa = np.stack([per['f104'][c] for c in cs], 1)
        out['representations'][rep] = {'channels': cs, 'per_arena': {}}
        for t in TAGS[1:]:
            Xb = np.stack([per[t][c] for c in cs], 1)
            out['representations'][rep]['per_arena'][t] = {
                'two_sample_classifier_auc': c2st(Xa, Xb)} | support(Xa, Xb)
        aucs = [out['representations'][rep]['per_arena'][t]['two_sample_classifier_auc'] for t in TAGS[1:]]
        cell = [out['representations'][rep]['per_arena'][t]['outside_f104_occupied_cells'] for t in TAGS[1:]]
        out['representations'][rep]['mean_auc'] = float(np.mean(aucs))
        out['representations'][rep]['mean_outside_f104_occupied_cells'] = float(np.mean(cell))
        print(f'{rep:42s} mean C2ST AUC {np.mean(aucs):.3f}  mean unseen-cell rate {np.mean(cell):.3f}', flush=True)

    # part 2 histogram of the omitted d0, in numbers
    edges = np.arange(104, 132.5, 0.5)
    for t in TAGS:
        z = np.load(HERE / 'points' / f'{t}.npz', allow_pickle=True)
        d0 = z['d0_all'].astype(np.float64)
        g = z['group_all'].astype(str)
        _, first = np.unique(g, return_index=True)
        h, _ = np.histogram(d0[first], edges)
        out['d0_histogram'][t] = {'bin_edges_m': edges.tolist(), 'counts_per_group': h.tolist(),
                                  'n_groups': int(len(first)),
                                  'min': float(d0[first].min()), 'max': float(d0[first].max()),
                                  'mean': float(d0[first].mean()), 'std': float(d0[first].std())}
    json.dump(out, open(HERE / 'shift.json', 'w'), indent=1)
    print('wrote', HERE / 'shift.json')


if __name__ == '__main__':
    main()
