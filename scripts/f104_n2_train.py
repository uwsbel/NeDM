"""Night-2: architecture (GRU vs transformer vs no-sequence) x vehicle-state inputs, same dev protocol as night 1.

Shared front end (unchanged from the deployed model): station-preserving CNN over the 96x32 route corridor ->
lateral mean+max -> linear to 96 features per station -> mixer -> per-station hazard logit -> survival loss.
  arch gru : Conv1d(k=5) + BiGRU          (the deployed H1 architecture)
  arch tx  : Conv1d(k=5) + TransformerEncoder(d=96, 4 heads, N layers, pre-norm)
  arch mlp : Conv1d(k=1) + per-station MLP  (no cross-station mixing: how much does sequence modelling matter?)
Vehicle state variants (ctx = [state0(17), goal dx, dy, |d|, start yaw, route length]):
  full    : all 22 (current)
  chassis : vx, vy, roll, pitch, roll rate, pitch rate, yaw rate + the 5 route/goal terms (12)
  none    : the 5 route/goal terms only (no vehicle state at all)
Selection rule (same as night 1, fixed before results): maximise same-group same-speed-profile AUC on unsafe
(G_unsafe) on the dev fold; ties within 0.01 broken by G_fail.
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, 'scripts')
import f104_night_train as N1
from f104_night_train import route_logit, survival_nll, metrics, dev_group

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
CHASSIS = [0, 1, 2, 3, 4, 5, 6]          # vx, vy, roll, pitch, roll rate, pitch rate, yaw rate
GEOM = [17, 18, 19, 20, 21]              # goal dx, dy, |d|, start yaw, route length
CTX_COLS = {'full': list(range(22)), 'chassis': CHASSIS + GEOM, 'none': GEOM}


class Net(nn.Module):
    def __init__(self, cin, nctx, arch='gru', width=64, layers=2, heads=4):
        super().__init__()
        c = [cin, 32, 64, 64, 96]
        L = []
        for i in range(4):
            L += [nn.Conv2d(c[i], c[i+1], 3, stride=(1, 1 if i == 0 else 2), padding=1),
                  nn.BatchNorm2d(c[i+1]), nn.GELU()]
        self.cnn = nn.Sequential(*L)
        self.lat = nn.Linear(2 * c[-1], 96)
        self.ctx = nn.Sequential(nn.Linear(nctx, 32), nn.GELU())
        self.arch = arch
        k = 1 if arch == 'mlp' else 5
        self.tconv = nn.Sequential(nn.Conv1d(96 + 32 + 1, 96, k, padding=k // 2), nn.GELU(), nn.Dropout(0.1))
        if arch == 'gru':
            self.mix = nn.GRU(96, width, batch_first=True, bidirectional=True)
            self.head = nn.Linear(2 * width, 1)
        elif arch == 'tx':
            self.pos = nn.Parameter(torch.zeros(1, 96, 96)); nn.init.normal_(self.pos, std=0.02)
            enc = nn.TransformerEncoderLayer(96, heads, 2 * 96, dropout=0.1, batch_first=True,
                                             norm_first=True, activation='gelu')
            self.mix = nn.TransformerEncoder(enc, layers)
            self.head = nn.Linear(96, 1)
        else:
            self.mix = nn.Sequential(nn.Linear(96, 2 * width), nn.GELU(), nn.Linear(2 * width, 2 * width), nn.GELU())
            self.head = nn.Linear(2 * width, 1)

    def forward(self, x, ctx):
        f = self.cnn(x)
        f = torch.cat([f.mean(-1), f.amax(-1)], 1).transpose(1, 2)
        f = F.gelu(self.lat(f))
        B, S, _ = f.shape
        pos = torch.linspace(0, 1, S, device=x.device)[None, :, None].expand(B, S, 1)
        c = self.ctx(ctx)[:, None, :].expand(B, S, 32)
        h = torch.cat([f, c, pos], -1).transpose(1, 2)
        h = self.tconv(h).transpose(1, 2)
        if self.arch == 'gru':
            h, _ = self.mix(h)
        elif self.arch == 'tx':
            h = self.mix(h + self.pos)
        else:
            h = self.mix(h)
        return self.head(h).squeeze(-1)


class Data:
    def __init__(self, ds, ctx_variant='full', extra_train=None):
        d = np.load(ds, allow_pickle=True)
        self.d = {k: d[k] for k in d.files}
        self.n = len(self.d['id'])
        src, sp = self.d['source'].astype(str), self.d['split'].astype(str)
        grp = self.d['group'].astype(str)
        isdev = np.array([dev_group(g) for g in grp])
        train = (sp == 'train')
        self.fit = train & ~isdev
        self.dev = train & isdev & (src == 'designed')   # dev fold is designed routes only, so every
                                                         # point of the scaling curve is scored on the same set
        X = self.d['X'].astype(np.float32)
        mu = X[self.fit][:, :4].mean((0, 2, 3)); sd = X[self.fit][:, :4].std((0, 2, 3)) + 1e-6
        X[:, :4] = (X[:, :4] - mu[None, :, None, None]) / sd[None, :, None, None]
        self.X = np.concatenate([X, np.ones((len(X), 1, X.shape[2], X.shape[3]), np.float32)], 1)
        self.norm = dict(mu=mu, sd=sd)
        ctx = self.d['ctx'].astype(np.float32)[:, CTX_COLS[ctx_variant]]
        self.ctx_mu, self.ctx_sd = ctx[self.fit].mean(0), ctx[self.fit].std(0) + 1e-6
        self.ctx = (ctx - self.ctx_mu) / self.ctx_sd
        self.ctx_variant = ctx_variant


def train_one(D, arch='gru', seed=0, epochs=30, lr=2e-3, wd=1e-4, layers=2, fit_mask=None, save=None, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    fit = D.fit if fit_mask is None else fit_mask
    fi = np.where(fit)[0]
    model = Net(D.X.shape[1], D.ctx.shape[1], arch=arch, layers=layers).to(DEV)
    # keep the corridor tensors in (pinned) CPU memory and move one batch at a time: the merged night-2
    # dataset is ~27 GB as float32 on device, which does not fit.
    Xc = torch.from_numpy(D.X[fi]).pin_memory()
    Cg = torch.tensor(D.ctx[fi], device=DEV)
    Eg = torch.tensor(D.d['event_idx'][fi].astype(np.int64), device=DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    bs = 256; n = len(fi); steps = max(epochs * (n // bs), 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    t0 = time.time(); step = 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n - bs + 1, bs):
            k = perm[i:i+bs]
            xb = Xc[k].to(DEV, non_blocking=True); kk = k.to(DEV)
            loss = survival_nll(model(xb, Cg[kk]), Eg[kk]).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            if step < steps - 1: sch.step()
            step += 1
    took = time.time() - t0
    s = predict(model, D, np.arange(D.n))
    m = metrics(s[D.dev], D.d, D.dev)
    m.update(arch=arch, ctx=D.ctx_variant, seed=seed, layers=layers, epochs=epochs, secs=round(took, 1),
             params=sum(p.numel() for p in model.parameters()), n_fit=int(fit.sum()))
    if save:
        torch.save(dict(state=model.state_dict(), arch=arch, layers=layers, ctx_variant=D.ctx_variant,
                        ctx_cols=CTX_COLS[D.ctx_variant], norm=D.norm, ctx_mu=D.ctx_mu, ctx_sd=D.ctx_sd,
                        cin=D.X.shape[1], nctx=D.ctx.shape[1]), save)
    return model, m


def predict(model, D, idx, bs=1024):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i+bs]
            out.append(route_logit(model(torch.tensor(D.X[j], device=DEV),
                                         torch.tensor(D.ctx[j], device=DEV))).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default='artifacts/traverse/fdm_f104_50h_20260909/night2_v1/station_ds_fix.npz')
    ap.add_argument('--archs', default='gru,tx,mlp')
    ap.add_argument('--ctxs', default='full,chassis,none')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--out', default='artifacts/traverse/fdm_f104_50h_20260909/night2_v1/sweep.json')
    a = ap.parse_args()
    rows = []
    for ctxv in a.ctxs.split(','):
        D = Data(a.ds, ctxv)
        print(f'=== ctx={ctxv} ({D.ctx.shape[1]} dims)  fit {int(D.fit.sum())}  dev {int(D.dev.sum())}', flush=True)
        for arch in a.archs.split(','):
            for seed in range(a.seeds):
                _, m = train_one(D, arch=arch, seed=seed, epochs=a.epochs, layers=a.layers)
                rows.append(m)
                print(f"  {arch:4s} ctx={ctxv:7s} s{seed}  G_unsafe {m['G_unsafe']:.3f}  G_fail {m['G_fail']:.3f}  "
                      f"W_unsafe {m['W_unsafe']:.3f}  P_unsafe {m['P_unsafe']:.3f}  {m['secs']}s  {m['params']/1e3:.0f}k", flush=True)
                json.dump(rows, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
