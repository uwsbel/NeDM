"""Shared rigid/CRM route-risk trainer (milestone A2): CNN-GRU risk model with optional domain tag or history conditioning.

Fork of scripts/n2_arch_train.py keeping only the CNN-GRU (GRUNet) path, the discrete-time survival loss, the metric
helpers and the one-JSON-row-per-run format; energy/time heads dropped.  Adds:
  --cond none|tag|hist|hist_aux|hist_rma   none: geometry ctx only (pooled model); tag: + one-hot domain in ctx (oracle);
                                           hist: + causal GRU over the 2 s observable history -> z (16) into the ctx MLP;
                                           hist_aux: hist + BCE domain head on z (weight 0.5, rows with a visible window);
                                           hist_rma: stage 1 teacher (geometry ctx + tanh(Linear([privileged 8, domain 2]))
                                           embedding, 16-d), stage 2 history encoder fitted to the teacher embedding
                                           (MSE, backbone frozen), stage 3 joint fine-tune with the risk loss (5 epochs).
  --domain-filter crm|rigid|both, --hist-drop P (random full-window masking during training), --split-eval val|test
  (held-out rows = that split only; train = split=='train'; md5 dev fold of train groups stays for --mode holdout),
  --startup-only (anchor_frame == 0 rows only, the twin-equivalent arms), balanced batches (half rigid / half CRM).
Data: npz with X (n,5,96,32) f16, ctx (n,22), id, group, split, profile, fail, unsafe, event_idx, route_len, anchor_frame,
domain (0 rigid, 1 crm) [, hist (n,T,15) f16, hmask (n,T) bool, privileged (n,8)].
Metrics per domain x {startup, established, all}: pooled / within-group (cells 'group|domain') / same-speed AUC against
unsafe and fail, lowest-risk pick failure per cell, Brier and 10-bin ECE of P(unsafe) = 1 - exp(-exp z) (secondary vs fail,
with the unsafe-not-fail rate).
Checkpoint (per the A2 contract): model_kind 'ga_train', cond, state, cin, nctx, zdim, hist_cols, hist_T, norm, ctx_mu,
ctx_sd, hist_mu, hist_sd, train_rows, split_hash.  Importable API: load_ga_model(path, device) -> (model, ck) and
score(model, ck, X, geom5, hist, hmask, domain_onehot=None) -> route logits (numpy f32).
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_riskmodel import route_logit

DEV = os.environ.get('GA_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
GEOM_COLS = [17, 18, 19, 20, 21]                      # goal dx, goal dy, |goal|, start yaw, route_len
HIST_STATE_COLS = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]   # deployable state columns (PLAN conventions)
HIST_ACTION_COLS = [0, 1, 2]                          # steer, throttle, brake
HIST_COLS = HIST_STATE_COLS + HIST_ACTION_COLS        # 15 channels
CONDS = ('none', 'tag', 'hist', 'hist_aux', 'hist_rma')
DOMAIN_NAME = {0: 'rigid', 1: 'crm'}
ZDIM, WIDTH, PRIV_DIM = 16, 64, 8


# ----------------------------------------------------------------------------- model
class CNNFront(nn.Module):
    """Station-preserving CNN of the deployed net: (B,cin,96,32) -> (B,96,192) [mean,max over the 4 lateral cells]."""
    def __init__(self, cin):
        super().__init__()
        c = [cin, 32, 64, 64, 96]; L = []
        for i in range(4):
            L += [nn.Conv2d(c[i], c[i + 1], 3, stride=(1, 1 if i == 0 else 2), padding=1), nn.BatchNorm2d(c[i + 1]), nn.GELU()]
        self.cnn = nn.Sequential(*L)

    def forward(self, x):
        f = self.cnn(x)
        return torch.cat([f.mean(-1), f.amax(-1)], 1).transpose(1, 2)


class GAModel(nn.Module):
    """CNN-GRU risk net.  ctx MLP input = [ctx (nctx)] (+ z (zdim) for the history / privileged conditions).
    forward(x (B,cin,96,32), ctx (B,nctx), hist (B,T,15) standardised, hmask (B,T), priv (B,10)) -> {'haz': (B,96) [, 'z', 'dom']}."""
    def __init__(self, cond, cin, nctx, zdim=ZDIM, hist_dh=len(HIST_COLS), width=WIDTH, priv_dim=PRIV_DIM):
        super().__init__()
        self.cond, self.zdim = cond, zdim
        self.use_hist = cond in ('hist', 'hist_aux', 'hist_rma'); self.use_priv = cond == 'rma_teacher'
        zin = zdim if (self.use_hist or self.use_priv) else 0
        self.front = CNNFront(cin); self.lat = nn.Linear(192, 96); self.ctx = nn.Sequential(nn.Linear(nctx + zin, 32), nn.GELU())
        self.tconv = nn.Sequential(nn.Conv1d(129, 96, 5, padding=2), nn.GELU(), nn.Dropout(0.1))
        self.mix = nn.GRU(96, width, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * width, 1)
        if self.use_hist:
            self.henc = nn.GRU(hist_dh + 1, 32, batch_first=True); self.hz = nn.Linear(32, zdim)   # causal, mask as channel 16
        if self.use_priv:
            self.penc = nn.Linear(priv_dim + 2, zdim)      # privileged teacher context + domain one-hot
        if cond == 'hist_aux':
            self.dom = nn.Linear(zdim, 1)

    def encode(self, hist, hmask):
        """z (B,zdim) from the standardised history; masked steps zeroed, mask appended as the last channel, final hidden state."""
        m = hmask.to(hist.dtype)[..., None]
        _, hn = self.henc(torch.cat([hist * m, m], -1))
        return torch.tanh(self.hz(hn[-1]))

    def encode_priv(self, priv):
        return torch.tanh(self.penc(priv))

    def backbone(self, x, ctx_full):
        f = F.gelu(self.lat(self.front(x))); B, S, _ = f.shape
        pos = torch.linspace(0, 1, S, device=x.device)[None, :, None].expand(B, S, 1)
        c = self.ctx(ctx_full)[:, None, :].expand(B, S, 32)
        h = self.tconv(torch.cat([f, c, pos], -1).transpose(1, 2)).transpose(1, 2)
        return self.head(self.mix(h)[0]).squeeze(-1)

    def forward(self, x, ctx, hist=None, hmask=None, priv=None, z=None):
        out = {}
        if self.use_hist:
            if z is None: z = self.encode(hist, hmask)
            ctx = torch.cat([ctx, z], -1); out['z'] = z
        elif self.use_priv:
            if z is None: z = self.encode_priv(priv)
            ctx = torch.cat([ctx, z], -1); out['z'] = z
        out['haz'] = self.backbone(x, ctx)
        if self.cond == 'hist_aux':
            out['dom'] = self.dom(z).squeeze(-1)
        return out


# ----------------------------------------------------------------------------- metrics
def dev_group(g): return int(hashlib.md5(g.encode()).hexdigest(), 16) % 5 == 0


def rank_avg(s):
    """1-based ranks with ties given their average rank (so identical scores give AUC 0.5, not array order)."""
    _, inv, cnt = np.unique(np.asarray(s, float), return_inverse=True, return_counts=True)
    first = np.cumsum(cnt) - cnt + 1.0
    return first[inv] + (cnt[inv] - 1) / 2.0


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if len(y) == 0 or y.min() == y.max(): return float('nan')
    r = rank_avg(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def cell_auc(y, s, cells):
    ok = tot = 0.0
    for c in np.unique(cells):
        m = cells == c; yy, ss = y[m], s[m]
        if len(yy) == 0 or yy.min() == yy.max(): continue
        d = ss[yy == 1][:, None] - ss[yy == 0][None, :]; ok += (d > 0).sum() + 0.5 * (d == 0).sum(); tot += d.size
    return (ok / tot if tot else float('nan')), int(tot)


def calibration(p, y, nbin=10):
    """Brier and 10-bin (equal width) expected calibration error of probabilities p against binary y."""
    p = np.asarray(p, float); y = np.asarray(y, float)
    if len(p) == 0: return float('nan'), float('nan')
    brier = float(np.mean((p - y) ** 2)); b = np.minimum((p * nbin).astype(int), nbin - 1); ece = 0.0
    for i in range(nbin):
        m = b == i
        if m.any(): ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return brier, float(ece)


def regime_metrics(s, d, m):
    """Metrics on rows m (bool mask): ranking AUCs (within-group cells keyed group|domain), lowest-risk pick failure per
    cell, Brier / ECE of P(unsafe) = 1 - exp(-exp z) against unsafe (fitted event) and fail (secondary)."""
    out = {'n': int(m.sum())}
    if out['n'] == 0: return out
    grp = np.array([f'{g}|{DOMAIN_NAME.get(int(x), x)}' for g, x in zip(d['group'][m].astype(str), d['domain'][m])])
    prof = d['profile'][m].astype(int) if 'profile' in d else np.full(out['n'], -1); des = prof >= 0
    gcell = np.array([f'{g}|{p}' for g, p in zip(grp, prof)])
    p = 1.0 - np.exp(-np.exp(np.asarray(s, float)))
    for lab in ('unsafe', 'fail'):
        y = d[lab][m].astype(float)
        out[f'P_{lab}'] = auc(y, s); out[f'W_{lab}'], out[f'Wn_{lab}'] = cell_auc(y, s, grp)
        out[f'G_{lab}'], out[f'Gn_{lab}'] = cell_auc(y[des], s[des], gcell[des]) if des.any() else (float('nan'), 0)
        out[f'brier_{lab}'], out[f'ece_{lab}'] = calibration(p, y)
        out[f'rate_{lab}'] = float(y.mean())
    out['mean_p'] = float(p.mean()); out['unsafe_not_fail_rate'] = float(((d['unsafe'][m] == 1) & (d['fail'][m] == 0)).mean())
    for lab in ('fail', 'unsafe'):   # route choice: lowest-risk route per group|domain cell vs random / oracle
        y = d[lab][m].astype(float); picks, rand, orac = [], [], []
        for g in np.unique(grp):
            k = np.where(grp == g)[0]
            if len(k) < 2: continue
            picks.append(y[k[np.argmin(s[k])]]); rand.append(y[k].mean()); orac.append(y[k].min())
        suf = '' if lab == 'fail' else '_unsafe'
        out.update({f'pick_fail{suf}': float(np.mean(picks)) if picks else float('nan'), f'random_fail{suf}': float(np.mean(rand)) if rand else float('nan'),
                    f'oracle_fail{suf}': float(np.mean(orac)) if orac else float('nan')})
        if lab == 'fail': out['n_groups'] = len(picks)
    return out


def metrics(s, d, m):
    """{domain: {startup, established, all: regime_metrics}} for domains present in rows m (s = logits of ALL rows), plus
    'both' (pooled, cells still keyed group|domain) when more than one domain is present."""
    dom = d['domain'].astype(int); startup = d['anchor_frame'].astype(int) == 0 if 'anchor_frame' in d else np.ones(len(dom), bool)
    assert len(s) == len(dom) == len(m), 'metrics() takes the full logit vector plus a row mask'
    out = {}; doms = sorted(np.unique(dom[m]).tolist())
    for key, dm in [(DOMAIN_NAME.get(x, str(x)), m & (dom == x)) for x in doms] + ([('both', m)] if len(doms) > 1 else []):
        out[key] = {name: regime_metrics(s[mm], d, mm) for name, mm in (('startup', dm & startup), ('established', dm & ~startup), ('all', dm))}
    return out


def domain_head_metrics(dl, d, m):
    """Domain-head read-out (hist_aux): AUC/accuracy of the domain logit on established rows (startup is 0.5 by definition)."""
    startup = d['anchor_frame'].astype(int) == 0; out = {}
    for name, mm in (('established', m & ~startup), ('startup', m & startup)):
        if mm.sum() == 0: continue
        y = d['domain'][mm].astype(float); out[f'{name}_auc'] = auc(y, dl[mm]); out[f'{name}_acc'] = float(((dl[mm] > 0) == (y > 0.5)).mean()); out[f'{name}_n'] = int(mm.sum())
    return out


# ----------------------------------------------------------------------------- input preparation (shared by Data and score)
def prep_x(x_raw, norm):
    """Raw corridor (b,5,96,32) f32 torch -> standardised channels + ones plane (b,6,96,32) f32."""
    mu = torch.as_tensor(np.asarray(norm['mu'], np.float32), device=x_raw.device)[None, :, None, None]
    sd = torch.as_tensor(np.asarray(norm['sd'], np.float32), device=x_raw.device)[None, :, None, None]
    cont = list(norm['cont_index']); y = torch.ones((x_raw.shape[0], x_raw.shape[1] + 1, 96, 32), dtype=torch.float32, device=x_raw.device)
    y[:, :x_raw.shape[1]] = x_raw; y[:, cont] = (y[:, cont] - mu) / sd
    return y


def prep_hist(hist, hmask, mu, sd):
    """Raw history (b,T,15) -> standardised f32 with masked steps zeroed (NaN-safe)."""
    h = torch.as_tensor(hist).to(torch.float32); m = torch.as_tensor(hmask).to(torch.bool).to(h.device)
    mu = torch.as_tensor(np.asarray(mu, np.float32), device=h.device); sd = torch.as_tensor(np.asarray(sd, np.float32), device=h.device)
    h = torch.where(m[..., None], (torch.nan_to_num(h) - mu) / sd, torch.zeros_like(h))
    return h, m


def chan_stats(X16, rows, cont, chunk=4096):
    """Per-channel mean / sd over rows (float64 accumulation, chunked so the f16 array is never copied whole)."""
    s = np.zeros(len(cont)); s2 = np.zeros(len(cont)); n = 0
    for i in range(0, len(rows), chunk):
        x = X16[rows[i:i + chunk]][:, cont].astype(np.float64); s += x.sum((0, 2, 3)); s2 += (x ** 2).sum((0, 2, 3)); n += x.shape[0] * x.shape[2] * x.shape[3]
    mu = s / n; sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0)) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, path, a):
        d = np.load(path, allow_pickle=True); self.d = {k: d[k] for k in d.files if k not in ('X', 'hist', 'hmask', 'E', 'T')}
        n_all = len(self.d['id'])
        if 'domain' not in self.d:   # derive from the id suffix written by the mixed builder
            ids = self.d['id'].astype(str); assert all(i.endswith(('@crm', '@rigid')) for i in ids[:100]), 'no domain array and ids are not suffixed @crm/@rigid'
            self.d['domain'] = np.array([1 if i.endswith('@crm') else 0 for i in ids], np.int8)
        if 'anchor_frame' not in self.d: self.d['anchor_frame'] = np.zeros(n_all, np.int32)
        if 'profile' not in self.d: self.d['profile'] = np.full(n_all, -1, np.int8)
        dom = self.d['domain'].astype(int); keep = np.ones(n_all, bool)
        if a.domain_filter != 'both': keep &= dom == (1 if a.domain_filter == 'crm' else 0)
        if a.startup_only: keep &= self.d['anchor_frame'].astype(int) == 0
        rows = np.where(keep)[0]; sub = len(rows) < n_all
        if sub: self.d = {k: (v[rows] if getattr(v, 'ndim', 0) >= 1 and v.shape[0] == n_all else v) for k, v in self.d.items()}   # per-row arrays only
        self.n = len(rows); self.need_hist = a.cond in ('hist', 'hist_aux', 'hist_rma'); self.need_priv = a.cond == 'hist_rma'
        sp = self.d['split'].astype(str); grp = self.d['group'].astype(str); isdev = np.array([dev_group(g) for g in grp])
        self.fit = (sp == 'train') & (~isdev if a.mode == 'holdout' else True); self.dev = (sp == 'train') & isdev; self.test = sp == a.split_eval
        assert self.fit.sum() > 0, 'no training rows'; assert self.test.sum() > 0, f'no rows with split == {a.split_eval}'
        if a.subsample:  # keep a fraction of the TRAINING GROUPS
            rng = np.random.default_rng(a.seed0); gs = np.unique(grp[self.fit]); kp = set(rng.choice(gs, max(1, int(len(gs) * a.subsample)), replace=False))
            self.fit = self.fit & np.array([g in kp for g in grp])
        fi = np.where(self.fit)[0]; self.split_hash = hashlib.md5('\n'.join(sorted(self.d['id'][self.fit].astype(str))).encode()).hexdigest()
        # corridor: standardise channels 0-3 on fit rows, ones plane appended; standardisation done on device in chunks
        X16 = d['X']; assert X16.shape[1:] == (5, 96, 32), X16.shape
        if sub: X16 = X16[rows]
        mu, sd = chan_stats(X16, fi, [0, 1, 2, 3])
        self.norm = dict(mu=mu, sd=sd, cont_index=[0, 1, 2, 3], channels=['elev_rel', 'grade', 'cross', 'speed', 'valid'])
        self.X = torch.empty((self.n, 6, 96, 32), dtype=torch.float16 if a.x_half else torch.float32, device=DEV)
        for i in range(0, self.n, 4096):
            self.X[i:i + 4096] = prep_x(torch.from_numpy(X16[i:i + 4096].astype(np.float32)).to(DEV), self.norm).to(self.X.dtype)
        del X16
        # context: 5 geometry columns standardised on fit rows [+ one-hot domain for the tag condition]
        geom = self.d['ctx'].astype(np.float32)[:, GEOM_COLS]; self.ctx_mu, self.ctx_sd = geom[self.fit].mean(0), geom[self.fit].std(0) + 1e-6
        ctx = (geom - self.ctx_mu) / self.ctx_sd; self.onehot = np.eye(2, dtype=np.float32)[dom[rows]]
        if a.cond == 'tag': ctx = np.concatenate([ctx, self.onehot], 1)
        self.ctx = torch.from_numpy(ctx.astype(np.float32)).to(DEV); self.nctx = ctx.shape[1]
        self.ev = torch.from_numpy(self.d['event_idx'].astype(np.int64)).to(DEV); self.dom = torch.from_numpy(dom[rows]).to(DEV)
        # history: standardise per channel on the valid steps of fit rows; masked steps zeroed; mask kept separately
        hc = [int(c) for c in d['hist_cols']] if 'hist_cols' in d.files else HIST_COLS
        if len(hc) == 12: hc = hc + HIST_ACTION_COLS      # builder stores the 12 state columns; the 3 action channels follow
        assert len(hc) == 15, f'hist_cols must list 12 state (+3 action) columns, got {hc}'
        self.hist_cols = hc
        if self.need_hist:
            assert 'hist' in d.files and 'hmask' in d.files, f'--cond {a.cond} needs hist/hmask arrays in {path}'
            H = d['hist']; M = d['hmask'].astype(bool)
            if sub: H, M = H[rows], M[rows]
            assert H.shape[0] == self.n and H.shape[2] == len(self.hist_cols) and M.shape == H.shape[:2], (H.shape, M.shape)
            Hf = np.nan_to_num(H[self.fit].astype(np.float32)); Mf = M[self.fit]; cnt = Mf.sum()
            if cnt > 0:
                self.hist_mu = (Hf * Mf[..., None]).sum((0, 1)) / cnt; self.hist_sd = np.sqrt(((Hf - self.hist_mu) ** 2 * Mf[..., None]).sum((0, 1)) / cnt) + 1e-6
            else:
                self.hist_mu = np.zeros(H.shape[2], np.float32); self.hist_sd = np.ones(H.shape[2], np.float32)
            del Hf, Mf
            self.hist_T = int(H.shape[1]); self.hist = torch.empty((self.n, H.shape[1], H.shape[2]), dtype=torch.float32, device=DEV)
            self.hmask = torch.from_numpy(M).to(DEV)
            for i in range(0, self.n, 8192):
                self.hist[i:i + 8192], _ = prep_hist(torch.from_numpy(H[i:i + 8192].astype(np.float32)).to(DEV), self.hmask[i:i + 8192], self.hist_mu, self.hist_sd)
            del H, M
        else:
            self.hist = self.hmask = None; self.hist_T = int(d['hist'].shape[1]) if 'hist' in d.files else 40
            self.hist_mu = np.zeros(len(self.hist_cols), np.float32); self.hist_sd = np.ones(len(self.hist_cols), np.float32)
        if self.need_priv:
            assert 'privileged' in d.files, f'--cond hist_rma needs the privileged array in {path}'
            P = d['privileged'].astype(np.float32)
            if sub: P = P[rows]
            assert P.shape == (self.n, PRIV_DIM), P.shape
            self.priv_mu, self.priv_sd = P[self.fit].mean(0), P[self.fit].std(0) + 1e-6
            self.priv = torch.from_numpy(np.concatenate([(P - self.priv_mu) / self.priv_sd, self.onehot], 1).astype(np.float32)).to(DEV)
        self.fit_idx = fi; self.dom_np = dom[rows]


def survival_nll(haz, ev):
    S = haz.shape[1]; idx = torch.arange(S, device=haz.device)[None, :]
    evc = torch.where(ev >= 0, ev, torch.full_like(ev, S - 1))[:, None]
    surv = (idx < evc) | ((ev < 0)[:, None] & (idx <= evc))
    nll = (F.softplus(haz) * surv).sum(1)
    return nll + torch.where(ev >= 0, F.softplus(-haz.gather(1, evc).squeeze(1)), torch.zeros_like(nll))


class Batches:
    """Balanced batch stream: when both domains are present in the fit rows, every batch is bs/2 rigid + bs/2 CRM drawn
    from two independent per-domain permutations (each re-permuted when exhausted); otherwise one permutation."""
    def __init__(self, fit_idx, dom_np, bs, balanced, gen):
        self.bs, self.gen = bs, gen; self.streams = []
        parts = [fit_idx[dom_np[fit_idx] == x] for x in (0, 1)] if balanced else [fit_idx]
        parts = [p for p in parts if len(p) > 0]
        self.q = [bs // len(parts)] * len(parts); self.q[-1] = bs - sum(self.q[:-1])
        for p in parts: self.streams.append([torch.from_numpy(p).to(DEV), None, 0])

    def next(self):
        out = []
        for st, q in zip(self.streams, self.q):
            idx, perm, cur = st
            if perm is None or cur + q > len(perm):
                st[1] = perm = idx[torch.randperm(len(idx), generator=self.gen, device='cpu').to(DEV)]; st[2] = cur = 0
            out.append(perm[cur:cur + q]); st[2] = cur + q
        return torch.cat(out)


def model_inputs(D, k, hm=None):
    """Batch tensors for rows k: X f32, ctx, hist, hmask (hm overrides the stored mask, e.g. after --hist-drop)."""
    x = D.X[k].to(torch.float32) if D.X.dtype != torch.float32 else D.X[k]
    if D.hist is None: return x, D.ctx[k], None, None
    return x, D.ctx[k], D.hist[k], (D.hmask[k] if hm is None else hm)


def predict(model, D, bs=1024):
    """Route logits for every row (eval mode; stored history / privileged inputs). Returns (route_logit, domain_logit or None)."""
    model.eval(); z, dl = [], []
    with torch.no_grad():
        for i in range(0, D.n, bs):
            k = torch.arange(i, min(i + bs, D.n), device=DEV); x, c, h, m = model_inputs(D, k)
            o = model(x, c, h, m, priv=(D.priv[k] if model.use_priv else None))
            z.append(route_logit(o['haz']).float().cpu().numpy())
            if 'dom' in o: dl.append(o['dom'].float().cpu().numpy())
    return np.concatenate(z), (np.concatenate(dl) if dl else None)


def run_epochs(model, D, a, params, epochs, lr, gen, loss_fn, balanced=True, hist_drop=0.0, wd=1e-4, log=None):
    """Generic optimisation loop: AdamW + OneCycle over epochs * (n_fit // bs) steps of balanced batches."""
    n = len(D.fit_idx); bs = min(a.bs, n); steps = max(epochs * (n // bs), 1)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd); sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    batches = Batches(D.fit_idx, D.dom_np, bs, balanced and a.domain_filter == 'both', gen); losses = []
    for step in range(steps):
        model.train(); k = batches.next(); hm = None
        if D.hmask is not None:
            hm = D.hmask[k]
            if hist_drop > 0:
                drop = torch.rand(len(k), generator=gen, device='cpu').to(DEV) < hist_drop; hm = hm & ~drop[:, None]
        loss = loss_fn(k, hm)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        if step < steps - 1: sch.step()
        losses.append(loss.item())
        assert np.isfinite(losses[-1]), f'non-finite loss at step {step}'
    if log is not None: log.update(steps=steps, first_loss=losses[0], final_loss=losses[-1], mean_loss_last50=float(np.mean(losses[-50:])))
    return losses


def risk_loss_fn(model, D, a):
    def fn(k, hm):
        x, c, h, m = model_inputs(D, k, hm); out = model(x, c, h, m, priv=(D.priv[k] if model.use_priv else None))
        loss = survival_nll(out['haz'], D.ev[k]).mean()
        if model.cond == 'hist_aux':
            vis = m.any(1)   # domain BCE only on rows whose window is (at least partly) visible
            if vis.any(): loss = loss + a.aux_weight * F.binary_cross_entropy_with_logits(out['dom'][vis], D.dom[k][vis].to(torch.float32))
        return loss
    return fn


def train_one(D, a, seed):
    torch.manual_seed(seed); np.random.seed(seed); gen = torch.Generator().manual_seed(seed)
    cin, nctx = D.X.shape[1], D.nctx; t0 = time.time(); stages = {}
    if a.cond == 'hist_rma':
        # stage 1: privileged teacher (geometry ctx + tanh(Linear([privileged 8, one-hot 2])) -> 16) with the risk loss
        teacher = GAModel('rma_teacher', cin, nctx).to(DEV); s1 = {}
        run_epochs(teacher, D, a, list(teacher.parameters()), a.epochs, a.lr, gen, risk_loss_fn(teacher, D, a), wd=a.wd, log=s1)
        zt, _ = predict(teacher, D); s1['heldout'] = metrics(zt, D.d, D.test); stages['stage1_teacher'] = s1
        # stage 2: student = same backbone (copied, frozen); history encoder fitted to the teacher embedding by MSE
        model = GAModel(a.cond, cin, nctx).to(DEV)
        missing = model.load_state_dict({k: v for k, v in teacher.state_dict().items() if not k.startswith('penc.')}, strict=False)
        assert all(k.startswith(('henc.', 'hz.')) for k in missing.missing_keys), missing
        with torch.no_grad(): teacher.eval(); E = torch.cat([teacher.encode_priv(D.priv[i:i + 4096]) for i in range(0, D.n, 4096)])
        enc_params = list(model.henc.parameters()) + list(model.hz.parameters())
        for p in model.parameters(): p.requires_grad_(False)
        for p in enc_params: p.requires_grad_(True)
        def mse_fn(k, hm):
            return F.mse_loss(model.encode(D.hist[k], hm), E[k])
        s2 = {}; run_epochs(model, D, a, enc_params, a.rma_epochs2, a.lr, gen, mse_fn, hist_drop=0.0, wd=a.wd, log=s2)
        with torch.no_grad():
            model.eval(); Z = torch.cat([model.encode(D.hist[i:i + 4096], D.hmask[i:i + 4096]) for i in range(0, D.n, 4096)])
            est = D.hmask.any(1); r2 = lambda mk: float(1 - ((Z[mk] - E[mk]) ** 2).mean() / E[mk].var(0, unbiased=False).mean().clamp_min(1e-9))
            s2['z_r2_established_heldout'] = r2(torch.from_numpy(D.test).to(DEV) & est); s2['z_mse_heldout'] = float(((Z - E) ** 2)[torch.from_numpy(D.test).to(DEV)].mean())
        stages['stage2_encoder_fit'] = s2
        # stage 3: joint fine-tune with the risk loss (history drop active so masked windows are in-distribution)
        for p in model.parameters(): p.requires_grad_(True)
        s3 = {}; run_epochs(model, D, a, list(model.parameters()), a.rma_epochs3, a.rma_lr3 or a.lr / 4, gen, risk_loss_fn(model, D, a), hist_drop=a.hist_drop, wd=a.wd, log=s3)
        stages['stage3_finetune'] = s3; final_loss = s3['final_loss']; steps = s1['steps'] + s2['steps'] + s3['steps']
    else:
        teacher = None
        model = GAModel(a.cond, cin, nctx).to(DEV); s1 = {}
        run_epochs(model, D, a, list(model.parameters()), a.epochs, a.lr, gen, risk_loss_fn(model, D, a), hist_drop=(a.hist_drop if model.use_hist else 0.0), wd=a.wd, log=s1)
        stages['train'] = s1; final_loss = s1['final_loss']; steps = s1['steps']
    took = time.time() - t0; nparam = sum(p.numel() for p in model.parameters())
    z, dl = predict(model, D)
    row = dict(arch='gru', cond=a.cond, domain_filter=a.domain_filter, split_eval=a.split_eval, mode=a.mode, startup_only=a.startup_only, hist_drop=a.hist_drop,
               lr=a.lr, wd=a.wd, epochs=a.epochs, seed=seed, params=nparam, secs=round(took, 1), n_fit=int(D.fit.sum()), steps=steps, final_loss=final_loss,
               stages=stages, dev=metrics(z, D.d, D.dev), heldout=metrics(z, D.d, D.test))
    if dl is not None: row['domain_head'] = dict(dev=domain_head_metrics(dl, D.d, D.dev), heldout=domain_head_metrics(dl, D.d, D.test))
    if DEV.startswith('cuda'): row['gpu_peak_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    return model, row, z, teacher


# ----------------------------------------------------------------------------- checkpoint contract
def checkpoint_dict(model, D, a, seed, tag):
    ck = dict(model_kind='ga_train', cond=model.cond, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
              cin=int(D.X.shape[1]), nctx=int(D.nctx), zdim=int(model.zdim if (model.use_hist or model.use_priv) else 0), width=WIDTH,
              hist_cols=list(D.hist_cols), hist_cols_state=list(D.hist_cols[:12]), hist_cols_action=list(D.hist_cols[12:]), hist_T=int(D.hist_T),
              hist_layout='hist[:, t, :12] = state[k-T+1+t, hist_cols_state]; hist[:, t, 12:] = action[k-T+t, hist_cols_action]; hmask True where the frame exists',
              norm=dict(mu=np.asarray(D.norm['mu']), sd=np.asarray(D.norm['sd']), cont_index=list(D.norm['cont_index']), channels=list(D.norm['channels'])),
              ctx_mu=np.asarray(D.ctx_mu, np.float32), ctx_sd=np.asarray(D.ctx_sd, np.float32), geom_cols=GEOM_COLS,
              hist_mu=np.asarray(D.hist_mu, np.float32), hist_sd=np.asarray(D.hist_sd, np.float32),
              train_rows=int(D.fit.sum()), split_hash=D.split_hash, split_eval=a.split_eval, mode=a.mode, domain_filter=a.domain_filter,
              startup_only=bool(a.startup_only), hist_drop=float(a.hist_drop), seed=int(seed), tag=tag, ds=os.path.abspath(a.ds), args=vars(a))
    if model.use_priv: ck.update(priv_mu=np.asarray(D.priv_mu, np.float32), priv_sd=np.asarray(D.priv_sd, np.float32))
    return ck


def load_ga_model(path, device=None):
    """Rebuild a ga_train checkpoint: returns (model in eval mode on device, checkpoint dict)."""
    device = device or DEV; ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck.get('model_kind') == 'ga_train', f'{path}: model_kind {ck.get("model_kind")!r} is not ga_train'
    model = GAModel(ck['cond'], ck['cin'], ck['nctx'], zdim=ck['zdim'] or ZDIM, hist_dh=len(ck['hist_cols']), width=ck.get('width', WIDTH))
    model.load_state_dict(ck['state']); model.to(device).eval()
    return model, ck


def encode_history(model, ck, hist, hmask, device=None):
    """z (n,zdim) numpy for raw history windows (n,T,15) with validity mask (n,T); hist=None -> all-masked window (startup)."""
    device = device or next(model.parameters()).device
    if hist is None:
        hist = np.zeros((1, ck['hist_T'], len(ck['hist_cols'])), np.float32); hmask = np.zeros((1, ck['hist_T']), bool)
    h, m = prep_hist(torch.as_tensor(np.asarray(hist, np.float32)).to(device), torch.as_tensor(np.asarray(hmask)).to(device), ck['hist_mu'], ck['hist_sd'])
    with torch.no_grad(): return model.encode(h, m).float().cpu().numpy()


def score(model, ck, X, geom5, hist=None, hmask=None, domain_onehot=None, z=None, bs=1024):
    """Route logits (n,) f32 for raw inputs: X (n,5,96,32) f32 corridor, geom5 (n,5) raw geometry ctx (ctx cols 17-21),
    hist (n or 1, T, 15) raw history + hmask (n or 1, T) (None = all-masked startup window; a single window is shared by
    every candidate), domain_onehot (n or 1, 2) for cond == 'tag'; z (n or 1, zdim) skips the history encoder."""
    device = next(model.parameters()).device; n = len(X); model.eval()
    ctx = (np.asarray(geom5, np.float32) - ck['ctx_mu']) / ck['ctx_sd']
    if ck['cond'] == 'tag':
        assert domain_onehot is not None, 'cond=tag needs domain_onehot'
        oh = np.asarray(domain_onehot, np.float32); ctx = np.concatenate([ctx, np.broadcast_to(oh, (n, 2))], 1)
    assert ck['cond'] != 'rma_teacher', 'the RMA teacher needs privileged inputs and is not a deployable model'
    ctx_t = torch.from_numpy(np.ascontiguousarray(ctx, dtype=np.float32)).to(device); out = []
    if model.use_hist:
        if z is None:
            z = encode_history(model, ck, hist, hmask, device)
        z_t = torch.as_tensor(np.asarray(z, np.float32)).to(device)
        if z_t.shape[0] == 1 and n > 1: z_t = z_t.expand(n, -1)
    with torch.no_grad():
        for i in range(0, n, bs):
            x = prep_x(torch.as_tensor(np.asarray(X[i:i + bs], np.float32)).to(device), ck['norm'])
            o = model(x, ctx_t[i:i + bs], z=(z_t[i:i + bs] if model.use_hist else None))
            out.append(route_logit(o['haz']).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ----------------------------------------------------------------------------- main
def fmt(m, key):
    r = m.get(key, {}); return ' '.join(f"{k}={r[k]:.3f}" for k in ('W_unsafe', 'P_unsafe', 'pick_fail', 'brier_unsafe', 'ece_unsafe') if k in r and np.isfinite(r[k])) + f" n={r.get('n', 0)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--cond', choices=CONDS, default='none'); ap.add_argument('--domain-filter', choices=['crm', 'rigid', 'both'], default='both')
    ap.add_argument('--hist-drop', type=float, default=0.2, help='probability of masking the whole history window of a training row')
    ap.add_argument('--split-eval', choices=['val', 'test'], default='val'); ap.add_argument('--startup-only', action='store_true')
    ap.add_argument('--mode', choices=['holdout', 'deploy'], default='holdout'); ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed0', type=int, default=0); ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--bs', type=int, default=256); ap.add_argument('--subsample', type=float, default=None)
    ap.add_argument('--aux-weight', type=float, default=0.5); ap.add_argument('--rma-epochs2', type=int, default=None, help='stage-2 encoder-fit epochs (default = --epochs)')
    ap.add_argument('--rma-epochs3', type=int, default=5); ap.add_argument('--rma-lr3', type=float, default=None, help='stage-3 lr (default lr/4)')
    ap.add_argument('--x-half', action='store_true', help='store the standardised corridor on the device as float16 (halves memory; predict differs from score at ~1e-3)')
    ap.add_argument('--tag', default=None); ap.add_argument('--no-save', action='store_true')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.rma_epochs2 is None: a.rma_epochs2 = a.epochs
    tag = a.tag or f"ga_{a.cond}_{a.domain_filter}{'_startup' if a.startup_only else ''}_{a.mode}_{a.split_eval}{'_sub%g' % a.subsample if a.subsample else ''}"
    D = Data(a.ds, a); dom_counts = {DOMAIN_NAME[x]: int((D.dom_np[D.fit] == x).sum()) for x in (0, 1)}
    print(f'=== {tag}: rows {D.n} fit {int(D.fit.sum())} {dom_counts} dev {int(D.dev.sum())} heldout({a.split_eval}) {int(D.test.sum())} cin {D.X.shape[1]} nctx {D.nctx} hist_T {D.hist_T} split_hash {D.split_hash[:8]}', flush=True)
    rows, logits = [], []
    for s in range(a.seed0, a.seed0 + a.seeds):
        model, row, z, teacher = train_one(D, a, s); rows.append(row); logits.append(z)
        h = row['heldout']
        for dk in [k for k in ('rigid', 'crm') if k in h]:
            print(f"  s{s} {dk:5s} startup[{fmt(h[dk], 'startup')}] established[{fmt(h[dk], 'established')}]", flush=True)
        extra = f"  domain_head {row['domain_head']['heldout']}" if 'domain_head' in row else ''
        print(f"  s{s} loss {row['final_loss']:.4f} {row['secs']}s {row['params']}p gpu_peak {row.get('gpu_peak_gb', 'n/a')} GB{extra}", flush=True)
        if not a.no_save:
            torch.save(checkpoint_dict(model, D, a, s, tag), f'{a.out}/{tag}_s{s}.pt')
            if teacher is not None: torch.save(checkpoint_dict(teacher, D, a, s, tag + '_teacher'), f'{a.out}/{tag}_s{s}_teacher.pt')
    ens = np.mean(logits, 0)
    np.savez_compressed(f'{a.out}/{tag}_logits.npz', id=D.d['id'], ensemble_logit=ens, member_logits=np.stack(logits), fit=D.fit, dev=D.dev, heldout=D.test,
                        domain=D.d['domain'], anchor_frame=D.d['anchor_frame'], group=D.d['group'], unsafe=D.d['unsafe'], fail=D.d['fail'])
    summary = dict(tag=tag, args=vars(a), n_rows=int(D.n), n_fit=int(D.fit.sum()), fit_by_domain=dom_counts, split_hash=D.split_hash, members=rows,
                   ensemble=dict(dev=metrics(ens, D.d, D.dev), heldout=metrics(ens, D.d, D.test)))
    json.dump(summary, open(f'{a.out}/{tag}.json', 'w'), indent=1, default=float)
    h = summary['ensemble']['heldout']
    for dk in [k for k in ('rigid', 'crm') if k in h]:
        print(f"ENSEMBLE {tag} {dk:5s} startup[{fmt(h[dk], 'startup')}] established[{fmt(h[dk], 'established')}] all[{fmt(h[dk], 'all')}]", flush=True)


if __name__ == '__main__':
    main()
