"""Train the deployed risk network FROM SCRATCH on CRM episodes (same architecture, loss, optimiser, schedule as N2).

Copy of scripts/sensor_train.py (itself the cluster twin of f104_n2_train.train_one: gen_riskmodel.Net BiGRU, discrete-
time survival loss, AdamW 2e-3, OneCycle, batch 256, 30 epochs, geometry-only context) with three changes:
  * reads the 5-channel N2 corridor tensor `X` written by scripts/f104_n2_dataset.py (elev_rel, grade, cross, speed, valid);
  * --mode deploy fits on EVERY row of the training-split groups (all route sources); --mode holdout leaves the md5
    dev fold out. Both report on the dev fold (holdout only) and on the held-out val/test start-goal groups;
  * read-out metrics tolerate empty / single-class cells and cover all route sources.
Checkpoints load with gen_planner.RiskModel exactly like N2_s*.pt.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_riskmodel import Net, route_logit

CHANNELS = ['elev_rel', 'grade', 'cross', 'speed', 'valid']   # the N2 input
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
GEOM = [17, 18, 19, 20, 21]


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if len(y) == 0 or y.min() == y.max(): return float('nan')
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
    des = prof >= 0   # same-speed-profile cells exist only for designed routes (planner-proposal rows have profile -1)
    for lab in ('unsafe', 'fail'):
        y = d[lab][m].astype(float)
        out[f'P_{lab}'] = auc(y, s)
        out[f'W_{lab}'], out[f'Wn_{lab}'] = cell_auc(y, s, grp)
        out[f'G_{lab}'], out[f'Gn_{lab}'] = cell_auc(y[des], s[des], gcell[des]) if des.any() else (float('nan'), 0)
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
    def __init__(self, path, mode):
        d = np.load(path, allow_pickle=True)
        self.d = {k: d[k] for k in d.files if k != 'X'}
        chans = CHANNELS
        self.variant, self.channels = 'CRM_N2', chans
        X = d['X'].astype(np.float32)
        sp = self.d['split'].astype(str); grp = self.d['group'].astype(str)
        isdev = np.array([dev_group(g) for g in grp])
        self.fit = (sp == 'train') & (~isdev if mode == 'holdout' else True)
        self.dev = (sp == 'train') & isdev                       # in-sample under --mode deploy
        self.test = (sp == 'test') | (sp == 'val')               # held-out start-goal groups, never fitted
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
    n = len(fi); bs = min(256, n); steps = max(epochs * (n // bs), 1)
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
    ap.add_argument('--mode', choices=['holdout', 'deploy'], required=True)
    ap.add_argument('--tag', default='CRM_N2'); ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--max-steps', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print('device', DEV, torch.cuda.get_device_name(0) if DEV == 'cuda' else '', flush=True)
    rows = []
    D = Data(a.ds, a.mode)
    print(f'=== {a.tag} {D.channels}  fit {int(D.fit.sum())}  dev {int(D.dev.sum())}  heldout {int(D.test.sum())}  '
          f'unsafe(fit) {D.d["unsafe"][D.fit].mean():.3f} fail(fit) {D.d["fail"][D.fit].mean():.3f}', flush=True)
    logits = []
    for s in range(a.seeds):
        name = f'{a.tag}_s{s}.pt' if a.mode == 'deploy' else f'{a.tag}_{a.mode}_s{s}.pt'  # planner glob = deploy only
        model, m = train_one(D, s, D.fit, epochs=a.epochs, max_steps=a.max_steps, save=f'{a.out}/{name}')
        logits.append(predict(model, D, np.arange(D.n)))
        rows.append(m)
        print(f"  {a.tag} s{s}  dev G {m['dev']['G_unsafe']:.3f} W {m['dev']['W_unsafe']:.3f} P {m['dev']['P_unsafe']:.3f} | "
              f"heldout G {m['test']['G_unsafe']:.3f} W {m['test']['W_unsafe']:.3f} P {m['test']['P_unsafe']:.3f} | {m['secs']}s", flush=True)
        ens = np.mean(logits, 0)
        summary = dict(mode=a.mode, tag=a.tag, ds=a.ds, n_fit=int(D.fit.sum()), n_dev=int(D.dev.sum()), n_heldout=int(D.test.sum()),
                       members=rows, ensemble=dict(n_members=len(logits), dev=metrics(ens[D.dev], D.d, D.dev),
                                                   heldout=metrics(ens[D.test], D.d, D.test)))
        json.dump(summary, open(f'{a.out}/{a.tag}_{a.mode}.json', 'w'), indent=1, default=float)
    np.savez_compressed(f'{a.out}/{a.tag}_{a.mode}_logits.npz', id=D.d['id'], ensemble_logit=np.mean(logits, 0),
                        member_logits=np.stack(logits), fit=D.fit, dev=D.dev, heldout=D.test)
    print('ensemble heldout', json.dumps(summary['ensemble']['heldout'], default=float), flush=True)
    print('exit: 0', flush=True)


if __name__ == '__main__':
    main()
