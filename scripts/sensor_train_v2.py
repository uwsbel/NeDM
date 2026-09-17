"""Matched height-vs-depth training on v2 corridors, split by whole arena.

Identical rows, labels, seeds, schedule and budget across variants; only the input channels differ. Network, loss and
optimiser are the deployed model's (gen_riskmodel.Net, survival loss, AdamW 2e-3 OneCycle, batch 256, 30 epochs).

Primary offline metric (per the 2026-09-15 review: route choice at matched speed, not pooled AUC): on held-out arenas,
within each start/goal and speed profile the designed routes differ only in lateral offset; the model picks the lowest
risk of them. Reported: how often that pick is unsafe, how often it was AVOIDABLE (a safe same-speed alternative
existed), the oracle (all alternatives unsafe) and a random-pick reference.
"""
import argparse, gc, hashlib, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_riskmodel import Net, route_logit
from sensor_train import auc, cell_auc, survival_nll, metrics

CH = ['z_rel', 'grade', 'cross', 'speed', 'valid', 'range_abs', 'range_rel', 'sec1', 'R', 'G', 'B', 'cover1']
VARIANTS = {
    'H':     ['z_rel', 'grade', 'cross', 'speed', 'valid'],          # current input, corrected geometry
    'H0':    ['z_rel', 'speed', 'valid'],                            # height only
    'Drel':  ['range_rel', 'sec1', 'speed', 'valid'],                # v1 depth arm (information-losing control)
    'Dabs':  ['range_abs', 'sec1', 'speed', 'valid'],                # absolute range + ray geometry (recoverable)
    'DabsC': ['range_abs', 'sec1', 'cover1', 'speed', 'valid'],
    'RGBDabs': ['R', 'G', 'B', 'range_abs', 'sec1', 'speed', 'valid'],
}
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
GEOM = [17, 18, 19, 20, 21]


def route_choice(scores, d, mask):
    """Pick the lowest-risk route among same start/goal + same speed profile; report the outcome of that choice."""
    grp = d['group'][mask].astype(str); prof = d['profile'][mask].astype(int); y = d['unsafe'][mask].astype(int)
    s = scores[mask]
    cells = {}
    for i, (g, p) in enumerate(zip(grp, prof)):
        cells.setdefault((g, int(p)), []).append(i)
    picked = avoid = oracle = n = 0; rnd = 0.0
    for idx in cells.values():
        if len(idx) < 2:
            continue
        idx = np.asarray(idx); yy = y[idx]
        k = idx[int(np.argmin(s[idx]))]
        n += 1; picked += int(y[k]); rnd += float(yy.mean())
        if yy.min() == 0:
            avoid += int(y[k])
        else:
            oracle += 1
    return dict(cells=n, picked_unsafe=picked / max(n, 1), avoidable_unsafe=avoid / max(n, 1),
                unavoidable=oracle / max(n, 1), random_pick=rnd / max(n, 1))


class Data:
    def __init__(self, files, variant, train_arenas, eval_arenas, sources=('designed',)):
        parts = [dict(np.load(f, allow_pickle=True)) for f in files]
        key = 'X12' if 'X12' in parts[0] else 'X'
        sel = [CH.index(c) for c in VARIANTS[variant]]
        X = np.concatenate([p[key][:, sel].astype(np.float32) for p in parts])
        self.d = {k: np.concatenate([p[k] for p in parts]) for k in
                  ('id', 'group', 'split', 'source', 'profile', 'fail', 'unsafe', 'event_idx', 'route_len', 'arena', 'ctx')}
        self.variant, self.channels = variant, VARIANTS[variant]
        ar = self.d['arena'].astype(str)
        self.fit = np.isin(ar, list(train_arenas))
        self.eval = np.isin(ar, list(eval_arenas))
        self.eval_designed = self.eval & np.isin(self.d['source'].astype(str), list(sources))
        cont = [i for i, c in enumerate(self.channels) if c != 'valid']
        mu = X[self.fit][:, cont].mean((0, 2, 3)); sd = X[self.fit][:, cont].std((0, 2, 3)) + 1e-6
        X[:, cont] = (X[:, cont] - mu[None, :, None, None]) / sd[None, :, None, None]
        self.norm = dict(channels=self.channels, cont_index=cont, mu=mu, sd=sd)
        self.X = np.concatenate([X, np.ones((len(X), 1, X.shape[2], X.shape[3]), np.float32)], 1)
        ctx = self.d['ctx'].astype(np.float32)[:, GEOM]
        self.ctx_mu, self.ctx_sd = ctx[self.fit].mean(0), ctx[self.fit].std(0) + 1e-6
        self.ctx = (ctx - self.ctx_mu) / self.ctx_sd
        self.n = len(X)


def predict(model, D, bs=1024):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, D.n, bs):
            out.append(route_logit(model(torch.tensor(D.X[i:i + bs], device=DEV),
                                         torch.tensor(D.ctx[i:i + bs], device=DEV))).float().cpu().numpy())
    return np.concatenate(out)


def fit_buffers(D, fit_index):
    """Build the (pinned) training buffers once per variant; re-pinning them per seed exhausts host memory."""
    fi = fit_index if fit_index is not None else np.where(D.fit)[0]
    Xc = torch.from_numpy(D.X[fi]); Xc = Xc.pin_memory() if DEV == 'cuda' else Xc
    return fi, Xc, torch.tensor(D.ctx[fi], device=DEV), torch.tensor(D.d['event_idx'][fi].astype(np.int64), device=DEV)


def train_one(D, seed, epochs=30, lr=2e-3, wd=1e-4, max_steps=None, save=None, fit_index=None, buffers=None):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Net(D.X.shape[1], D.ctx.shape[1], arch='gru', layers=2).to(DEV)
    fi, Xc, Cg, Eg = buffers if buffers is not None else fit_buffers(D, fit_index)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    bs = 256; n = len(fi); steps = max(epochs * (n // bs), 1)
    if max_steps: steps = min(steps, max_steps)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    t0 = time.time(); step = 0
    while step < steps:
        model.train(); perm = torch.randperm(n)
        for i in range(0, n - bs + 1, bs):
            if step >= steps: break
            k = perm[i:i + bs]
            loss = survival_nll(model(Xc[k].to(DEV, non_blocking=True), Cg[k.to(DEV)]), Eg[k.to(DEV)]).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            if step < steps - 1: sch.step()
            step += 1
    del opt, sch
    gc.collect()
    if DEV == 'cuda':
        torch.cuda.empty_cache()
    s = predict(model, D)
    m = dict(variant=D.variant, channels=D.channels, seed=seed, steps=step, secs=round(time.time() - t0, 1),
             n_fit=len(fi), rank=metrics(s[D.eval_designed], D.d, D.eval_designed),
             choice=route_choice(s, D.d, D.eval_designed),
             choice_all=route_choice(s, D.d, D.eval))
    for a in sorted(set(D.d['arena'][D.eval].astype(str))):
        m_ = D.eval_designed & (D.d['arena'].astype(str) == a)
        m[f'choice_{a}'] = route_choice(s, D.d, m_)
    if save:
        torch.save(dict(state=model.state_dict(), arch='gru', layers=2, channels=D.channels, norm=D.norm,
                        ctx_cols=GEOM, ctx_mu=D.ctx_mu, ctx_sd=D.ctx_sd, cin=D.X.shape[1], nctx=D.ctx.shape[1],
                        variant=D.variant, version='v2'), save)
    return model, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--variants', default='H,H0,Drel,Dabs'); ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--train-arenas', default='f104,g228,g203,g217'); ap.add_argument('--eval-arenas', default='g216,g231')
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--save', action='store_true'); ap.add_argument('--tag', default='matched')
    ap.add_argument('--seed-list', default='', help='comma list of seeds, e.g. 2 (default: range(--seeds))')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print('device', DEV, torch.cuda.get_device_name(0) if DEV == 'cuda' else '', flush=True)
    rows = []
    fit_index = None
    for v in a.variants.split(','):
        D = Data(a.files, v, a.train_arenas.split(','), a.eval_arenas.split(','))
        if fit_index is None:
            fit_index = np.where(D.fit)[0]           # identical rows for every variant
        print(f'=== {v} {D.channels}  fit {len(fit_index)}  eval {int(D.eval.sum())} '
              f'({int(D.eval_designed.sum())} designed)', flush=True)
        buffers = fit_buffers(D, fit_index)
        for s in ([int(x) for x in a.seed_list.split(',')] if a.seed_list else range(a.seeds)):
            save = f'{a.out}/{a.tag}_{v}_s{s}.pt' if a.save else None
            _, m = train_one(D, s, epochs=a.epochs, max_steps=a.max_steps, save=save, buffers=buffers)
            rows.append(m)
            c = m['choice']
            print(f"  {v:7s} s{s}  choice: picked-unsafe {100*c['picked_unsafe']:5.2f}%  avoidable {100*c['avoidable_unsafe']:5.2f}%  "
                  f"(random {100*c['random_pick']:5.2f}%, unavoidable {100*c['unavoidable']:4.2f}%)  "
                  f"rank G {m['rank']['G_unsafe']:.3f} W {m['rank']['W_unsafe']:.3f}  {m['secs']}s", flush=True)
            json.dump(rows, open(f'{a.out}/{a.tag}_{a.variants.replace(",", "_")}.json', 'w'), indent=1, default=float)
    print('exit: 0', flush=True)


if __name__ == '__main__':
    main()
