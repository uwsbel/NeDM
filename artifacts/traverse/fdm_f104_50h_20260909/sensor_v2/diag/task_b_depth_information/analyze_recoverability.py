"""Task B part 3: feature-to-height recoverability of four depth-side representations, fit on f104 only.

Same corridor points for every representation (points/*.npz). Two targets:
  abs  z_i        absolute height of the observed point   (= H - d_i/s_i, agrees with the simulator to ~0.01 m)
  rel  z_i - z_0  the quantity the deployed height channel carries (elev_rel)

Representations, each given exactly what its own corridor tensor would carry (the route-start cell [0,16] is part of
the same tensor, so a representation that stores absolute range also stores d0, and every representation stores s0):
  a_cell  v1 raw-depth arm, one cell in isolation   [dd, s-1]
  a_plus  v1 raw-depth arm as delivered             [dd, s-1, s0-1]
  b       absolute range + sec                      [d, s]            (+ [d0, s0] for rel)
  c       absolute range + ray direction            [d, rx, ry]       (+ [d0, s0] for rel)
  d       back-projected height (the height arm)    [z]               (+ [z0] for rel)

Estimators: exact closed form where one exists; ridge; degree-3 polynomial ridge; a 2x128 MLP; and a local binned
mean (nonparametric ceiling) for representations of at most three dimensions, with the extrapolation rate reported.
Training uses f104 groups only; evaluation is f104 held-out groups and each of the five unseen arenas.
"""
import itertools, json, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TAGS = ['f104', 'g203', 'g216', 'g217', 'g228', 'g231']
H = 110.0
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RNG = np.random.default_rng(0)
N_TRAIN, N_EVAL = 400_000, 300_000


def load(tag):
    z = np.load(HERE / 'points' / f'{tag}.npz', allow_pickle=True)
    d = {k: z[k].astype(np.float64) for k in ('dep', 'sec', 'd0', 's0', 'zpix', 'gx', 'gy', 'ray_x', 'ray_y', 'route')}
    keep = (np.abs(d['gx']) < 39.5) & (np.abs(d['gy']) < 39.5)
    d = {k: v[keep] for k, v in d.items()}
    d['dd'] = d['dep'] - d['d0']
    d['z0'] = H - d['d0'] / d['s0']
    d['zrel'] = d['zpix'] - d['z0']
    return d


def feats(d, rep, target):
    dd, s, s0, dep, d0 = d['dd'], d['sec'], d['s0'], d['dep'], d['d0']
    rx, ry, z, z0 = d['ray_x'], d['ray_y'], d['zpix'], d['z0']
    base = {'a_cell': [dd, s - 1], 'a_plus': [dd, s - 1, s0 - 1],
            'b': [dep, s], 'c': [dep, rx, ry], 'd': [z]}[rep]
    if target == 'rel':
        base = {'a_cell': [dd, s - 1], 'a_plus': [dd, s - 1, s0 - 1],
                'b': [dep, s, d0, s0], 'c': [dep, rx, ry, d0, s0], 'd': [z, z0]}[rep]
    return np.stack(base, 1)


def closed_form(d, rep, target):
    """Exact inversion from the camera model where the representation admits one; None otherwise."""
    z = H - d['dep'] / d['sec']
    if rep in ('b', 'c', 'd'):
        return z if target == 'abs' else z - (H - d['d0'] / d['s0'])
    if rep == 'a_plus' and target == 'rel':
        return None       # needs the plug-in constant fitted on f104; handled separately
    return None


def ridge_fit(X, y, lam=1e-6):
    X = np.c_[X, np.ones(len(X))]
    A = X.T @ X + lam * np.eye(X.shape[1]) * max(1.0, np.trace(X.T @ X) / X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def ridge_pred(w, X):
    return np.c_[X, np.ones(len(X))] @ w


def poly(X, deg=3):
    n, k = X.shape
    out = [np.ones(n)]
    for o in range(1, deg + 1):
        for c in itertools.combinations_with_replacement(range(k), o):
            v = np.ones(n)
            for i in c:
                v = v * X[:, i]
            out.append(v)
    return np.stack(out, 1)


def mlp_fit(X, y, epochs=40, width=128, seed=0):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    ym, ys = y.mean(), y.std() + 1e-9
    xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=DEV)
    yt = torch.tensor((y - ym) / ys, dtype=torch.float32, device=DEV)[:, None]
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], width), torch.nn.GELU(),
                              torch.nn.Linear(width, width), torch.nn.GELU(),
                              torch.nn.Linear(width, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = 8192
    for _ in range(epochs):
        perm = torch.randperm(len(xt), device=DEV)
        for i in range(0, len(xt), bs):
            j = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(net(xt[j]), yt[j]).backward()
            opt.step()
        sch.step()
    return (net, mu, sd, ym, ys)


def mlp_pred(m, X):
    net, mu, sd, ym, ys = m
    with torch.no_grad():
        p = []
        for i in range(0, len(X), 200_000):
            xt = torch.tensor((X[i:i + 200_000] - mu) / sd, dtype=torch.float32, device=DEV)
            p.append(net(xt).cpu().numpy()[:, 0])
    return np.concatenate(p) * ys + ym


class Binned:
    """Local mean on a regular grid of the (standardised) feature space; reports extrapolation rate."""
    def __init__(self, X, y, bins=48):
        self.lo, self.hi = X.min(0), X.max(0)
        self.bins = bins
        idx = self._idx(X)
        self.table = {}
        order = np.lexsort(tuple(idx.T[::-1]))
        k = idx[order]; ys = y[order]
        new = np.r_[True, np.any(k[1:] != k[:-1], axis=1)]
        st = np.flatnonzero(new); en = np.r_[st[1:], len(ys)]
        cs = np.r_[0.0, np.cumsum(ys)]
        means = (cs[en] - cs[st]) / (en - st)
        self.table = {tuple(kk): m for kk, m in zip(k[st], means)}
        self.global_mean = float(y.mean())

    def _idx(self, X):
        u = (X - self.lo) / np.maximum(self.hi - self.lo, 1e-12)
        return np.clip((u * self.bins).astype(np.int32), -1, self.bins)

    def predict(self, X):
        idx = self._idx(X)
        out = np.full(len(X), np.nan)
        miss = 0
        for i, kk in enumerate(map(tuple, idx)):
            v = self.table.get(kk)
            if v is None:
                out[i] = self.global_mean; miss += 1
            else:
                out[i] = v
        return out, miss / len(X)


def score(yhat, y):
    e = yhat - y
    return dict(rmse_m=float(np.sqrt(np.mean(e ** 2))), mae_m=float(np.mean(np.abs(e))),
                p95_abs_m=float(np.quantile(np.abs(e), .95)), max_abs_m=float(np.abs(e).max()),
                r2=float(1 - np.mean(e ** 2) / np.var(y)))


def main():
    t0 = time.time()
    per = {t: load(t) for t in TAGS}
    f = per['f104']
    grp = (f['route'].astype(int) // 1)          # routes; groups are contiguous pairs of routes in the file order
    tr_mask = (f['route'].astype(int) % 5) < 3
    idx_tr = np.flatnonzero(tr_mask); idx_te = np.flatnonzero(~tr_mask)
    idx_tr = RNG.choice(idx_tr, min(N_TRAIN, len(idx_tr)), replace=False)
    idx_te = RNG.choice(idx_te, min(N_EVAL, len(idx_te)), replace=False)
    evalsets = {'f104_heldout': {k: v[idx_te] for k, v in f.items()}}
    for t in TAGS[1:]:
        sel = RNG.choice(len(per[t]['dd']), min(N_EVAL, len(per[t]['dd'])), replace=False)
        evalsets[t] = {k: v[sel] for k, v in per[t].items()}
    train = {k: v[idx_tr] for k, v in f.items()}

    res = {'setup': {'train': 'f104 routes with index%%5<3, %d points' % len(idx_tr),
                     'eval_points_per_set': {k: int(len(v['dd'])) for k, v in evalsets.items()},
                     'target_abs': 'z = H - d/sec', 'target_rel': 'z - z0'},
           'target_variance_m2': {}, 'results': {}}
    for name, ev in evalsets.items():
        res['target_variance_m2'][name] = {'abs': float(np.var(ev['zpix'])), 'rel': float(np.var(ev['zrel']))}

    # plug-in closed form for the delivered raw-depth arm: zrel = Qbar*(1 - s0/s) - dd/s, Qbar from f104
    Qbar = float(np.mean(H - train['z0']))
    res['plugin_Qbar_m'] = Qbar

    for target in ('abs', 'rel'):
        ykey = 'zpix' if target == 'abs' else 'zrel'
        for rep in ('a_cell', 'a_plus', 'b', 'c', 'd'):
            Xtr = feats(train, rep, target); ytr = train[ykey]
            models = {'ridge': ridge_fit(Xtr, ytr),
                      'poly3': ridge_fit(poly(Xtr), ytr, 1e-8),
                      'mlp': mlp_fit(Xtr, ytr)}
            binned = Binned(Xtr, ytr) if Xtr.shape[1] <= 3 else None
            for name, ev in evalsets.items():
                X = feats(ev, rep, target); y = ev[ykey]
                row = {'ridge': score(ridge_pred(models['ridge'], X), y),
                       'poly3': score(ridge_pred(models['poly3'], poly(X)), y),
                       'mlp': score(mlp_pred(models['mlp'], X), y)}
                if binned is not None:
                    p, miss = binned.predict(X)
                    row['binned_local_mean'] = score(p, y) | {'extrapolation_rate': float(miss)}
                cf = closed_form(ev, rep, target)
                if cf is not None:
                    row['closed_form'] = score(cf, y)
                if rep == 'a_plus' and target == 'rel':
                    zp = Qbar * (1 - ev['s0'] / ev['sec']) - ev['dd'] / ev['sec']
                    row['closed_form_plugin_Qbar'] = score(zp, y)
                res['results'].setdefault(target, {}).setdefault(rep, {})[name] = row
            print(f'{target} {rep}: ' + '  '.join(
                f"{n}={res['results'][target][rep][n]['mlp']['rmse_m']:.4f}" for n in evalsets), flush=True)

    json.dump(res, open(HERE / 'recoverability.json', 'w'), indent=1)
    print('wrote', HERE / 'recoverability.json', f'{time.time()-t0:.0f}s on {DEV}')


if __name__ == '__main__':
    main()
