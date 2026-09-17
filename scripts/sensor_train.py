"""Train the risk network on sensor channels instead of the height map (sensor_v1).

Same network (scripts/gen_riskmodel.py Net, BiGRU), loss (discrete-time survival), optimiser and schedule as the
deployed night-2 model (f104_n2_train.train_one: AdamW 2e-3, OneCycle, batch 256, 30 epochs, geometry-only context).
Only the corridor channels differ. Input file: sensor_ds_f104.npz from scripts/sensor_dataset.py (X10 layout below).

  --mode sweep   fit on training groups outside the dev fold, report dev + test-split metrics, 3 seeds per variant
  --mode deploy  fit on every training group, 5 seeds, save checkpoints for the planner
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_riskmodel import Net, route_logit

X10 = ['elev_rel', 'grade', 'cross', 'speed', 'valid', 'depth_rel', 'ray_sec1', 'R', 'G', 'B']
VARIANTS = {
    'E':    ['elev_rel', 'grade', 'cross', 'speed', 'valid'],          # current model input (control)
    'E0':   ['elev_rel', 'speed', 'valid'],                            # height only, no derived slopes
    'D':    ['depth_rel', 'ray_sec1', 'speed', 'valid'],               # raw depth sensor
    'RGBD': ['R', 'G', 'B', 'depth_rel', 'ray_sec1', 'speed', 'valid'],
    'RGB':  ['R', 'G', 'B', 'ray_sec1', 'speed', 'valid'],             # colour only, no depth
}
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
GEOM = [17, 18, 19, 20, 21]


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if y.min() == y.max(): return float('nan')
    r = np.argsort(np.argsort(s)) + 1.0; n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def cell_auc(y, s, cells):
    ok = tot = 0.0
    for c in np.unique(cells):
        m = cells == c; yy, ss = y[m], s[m]
        if yy.min() == yy.max(): continue
        p, n = ss[yy == 1], ss[yy == 0]
        cmp = p[:, None] - n[None, :]
        ok += (cmp > 0).sum() + 0.5 * (cmp == 0).sum(); tot += cmp.size
    return (ok / tot if tot else float('nan')), int(tot)


def metrics(s, d, m):
    out = {}
    grp = d['group'][m].astype(str); prof = d['profile'][m].astype(int)
    gcell = np.array([f'{g}|{p}' for g, p in zip(grp, prof)])
    for lab in ('unsafe', 'fail'):
        y = d[lab][m].astype(float)
        out[f'P_{lab}'] = auc(y, s)
        out[f'W_{lab}'], out[f'Wn_{lab}'] = cell_auc(y, s, grp)
        out[f'G_{lab}'], out[f'Gn_{lab}'] = cell_auc(y, s, gcell)
    return out


def survival_nll(haz, ev):
    S = haz.shape[1]
    idx = torch.arange(S, device=haz.device)[None, :]
    evc = torch.where(ev >= 0, ev, torch.full_like(ev, S - 1))[:, None]
    surv_mask = (idx < evc) | ((ev < 0)[:, None] & (idx <= evc))
    nll = (F.softplus(haz) * surv_mask).sum(1)
    hit = (ev >= 0)
    return nll + torch.where(hit, F.softplus(-haz.gather(1, evc).squeeze(1)), torch.zeros_like(nll))


def dev_group(g):
    return int(hashlib.md5(g.encode()).hexdigest(), 16) % 5 == 0


class Data:
    def __init__(self, path, variant):
        d = np.load(path, allow_pickle=True)
        self.d = {k: d[k] for k in d.files if k != 'X10'}
        chans = VARIANTS[variant]; sel = [X10.index(c) for c in chans]
        self.variant, self.channels = variant, chans
        X = d['X10'][:, sel].astype(np.float32)
        src, sp = self.d['source'].astype(str), self.d['split'].astype(str); grp = self.d['group'].astype(str)
        isdev = np.array([dev_group(g) for g in grp])
        self.fit = (sp == 'train') & ~isdev
        self.dev = (sp == 'train') & isdev & (src == 'designed')
        self.test = (sp == 'test') & (src == 'designed')
        cont = [i for i, c in enumerate(chans) if c != 'valid']
        mu = X[self.fit][:, cont].mean((0, 2, 3)); sd = X[self.fit][:, cont].std((0, 2, 3)) + 1e-6
        X[:, cont] = (X[:, cont] - mu[None, :, None, None]) / sd[None, :, None, None]
        self.norm = dict(channels=chans, cont_index=cont, mu=mu, sd=sd)
        self.X = np.concatenate([X, np.ones((len(X), 1, X.shape[2], X.shape[3]), np.float32)], 1)
        ctx = self.d['ctx'].astype(np.float32)[:, GEOM]
        self.ctx_mu, self.ctx_sd = ctx[self.fit].mean(0), ctx[self.fit].std(0) + 1e-6
        self.ctx = (ctx - self.ctx_mu) / self.ctx_sd
        self.n = len(X)


def predict(model, D, idx, bs=1024):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            out.append(route_logit(model(torch.tensor(D.X[j], device=DEV), torch.tensor(D.ctx[j], device=DEV))).float().cpu().numpy())
    return np.concatenate(out)


def train_one(D, seed, fit, epochs=30, lr=2e-3, wd=1e-4, max_steps=None, save=None):
    torch.manual_seed(seed); np.random.seed(seed)
    fi = np.where(fit)[0]
    model = Net(D.X.shape[1], D.ctx.shape[1], arch='gru', layers=2).to(DEV)
    Xc = torch.from_numpy(D.X[fi]); Xc = Xc.pin_memory() if DEV == 'cuda' else Xc
    Cg = torch.tensor(D.ctx[fi], device=DEV); Eg = torch.tensor(D.d['event_idx'][fi].astype(np.int64), device=DEV)
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
            xb = Xc[k].to(DEV, non_blocking=True); kk = k.to(DEV)
            loss = survival_nll(model(xb, Cg[kk]), Eg[kk]).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            if step < steps - 1: sch.step()
            step += 1
    took = time.time() - t0
    s = predict(model, D, np.arange(D.n))
    m = dict(variant=D.variant, channels=D.channels, seed=seed, steps=step, secs=round(took, 1), n_fit=int(fit.sum()),
             dev=metrics(s[D.dev], D.d, D.dev), test=metrics(s[D.test], D.d, D.test), final_loss=float(loss))
    if save:
        torch.save(dict(state=model.state_dict(), arch='gru', layers=2, channels=D.channels, norm=D.norm,
                        ctx_cols=GEOM, ctx_mu=D.ctx_mu, ctx_sd=D.ctx_sd, cin=D.X.shape[1], nctx=D.ctx.shape[1],
                        variant=D.variant), save)
    return model, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=['sweep', 'deploy'], required=True)
    ap.add_argument('--variants', default='E,E0,D,RGBD,RGB'); ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--max-steps', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print('device', DEV, torch.cuda.get_device_name(0) if DEV == 'cuda' else '', flush=True)
    rows = []
    for v in a.variants.split(','):
        D = Data(a.ds, v)
        fit = D.fit if a.mode == 'sweep' else (D.fit | D.dev)   # deploy: exactly the night-2 deployment rows
        print(f'=== {v} {D.channels}  fit {int(fit.sum())}  dev {int(D.dev.sum())}  test {int(D.test.sum())}', flush=True)
        for s in range(a.seeds):
            save = f'{a.out}/{v}_s{s}.pt'   # sweep checkpoints too: used for zero-shot scoring on the new arenas
            _, m = train_one(D, s, fit, epochs=a.epochs, max_steps=a.max_steps, save=save)
            rows.append(m)
            print(f"  {v:5s} s{s}  dev G {m['dev']['G_unsafe']:.3f} W {m['dev']['W_unsafe']:.3f} P {m['dev']['P_unsafe']:.3f} | "
                  f"test G {m['test']['G_unsafe']:.3f} W {m['test']['W_unsafe']:.3f} | {m['secs']}s", flush=True)
            json.dump(rows, open(f'{a.out}/{a.mode}_{a.variants.replace(",", "_")}.json', 'w'), indent=1, default=float)
    print('exit: 0', flush=True)


if __name__ == '__main__':
    main()
