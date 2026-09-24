"""Route-risk trainer for the soil-improvement effort (crm_improve_20260922, stages S0 B/A, S2, S3): a fork of ga_train.py
with several datasets, four network families, a choice of history encoder and window, a velocity context, batch / data
balance controls and per-source row weights.  ga_train.py is imported (never edited): CNN front, metrics, input
preparation, survival loss.

Networks (--arch):
  gru      ga_train's CNN-GRU (same modules, construction order and state-dict keys as ga_train.GAModel, so the
           same seed gives the same initial weights); the history z (16) joins the context MLP input.
  tx96_2 / tx128_4 (any txD_L)   night-2 tokenised transformer (n2_arch_train.TokenTx): per-station CNN column -> token
           + sinusoidal station position, the context vector (+ z) as a [CTX] token, pre-LN encoder, final LN.
  txjoint (txjointD_L, default d 96, 2 layers)   one transformer over [CTX] + 96 station tokens + the history frames
           as tokens (Linear([frame, mask]) + learned time-of-frame embedding + a type embedding per token kind);
           attention runs jointly over stations and time, invalid frames are never attended to.
--hist-enc gru|tx (gru / tx arches): ga_train's causal GRU (16 -> 32 -> z) or a small causal transformer (d 64,
  2 layers, 4 heads; frame tokens + age sinusoid, a learned read-out token after the newest frame, each token attends
  to itself and to valid frames at or before it; z = tanh(Linear(read-out))).  txjoint has no separate encoder.
--hist-T N  only the newest N of the stored 40 frames: --hist-window mask (default) keeps the 40-frame window and masks
  the older frames; cut feeds N frames only (same for the transformer encoders, not for the GRU, which otherwise steps
  through the masked frames).  score() applies the same rule to caller windows.
--ctx geom|geom_vel  geometry ctx cols 17-21 [+ vx and yaw rate of the last valid history frame, zero for an empty
  window], standardised on fit rows.
--cond none|tag|hist|hist_aux as ga_train (hist_rma not carried over).
Data: --ds repeatable (npz files concatenated in memory; per-row keys missing from some file are dropped and listed;
ids must be unique); --data-frac-crm / --data-frac-rigid keep a fraction of the TRAIN groups per domain (one seeded
group permutation shared by both domains, so fractions are nested and equal fractions keep the same twin groups);
--crm-batch-frac F = CRM share of every batch; --row-weight 'source=w,...' weights the survival loss per row.
Only the rows that are fitted or evaluated are put on the device (fit | dev fold | --split-eval split); --x-half /
--x-host shrink the device footprint.  Suite groups (f104_pair_group_*, f104_crm_eval_group_*, f104_g1_test_group_*)
may never be fit rows (hard error).
Metrics: ga_train's per-domain x startup/established/all blocks, plus decision-state cells (routes from one identical
start state: episode|anchor_frame|domain; branch rows), and the same blocks on rows by 'source' and by anchor-frame
bucket (k0, k10_30, k40p [, k_other]).
Checkpoint: model_kind 'ci_train' (fields in checkpoint_dict); API load_ci_model(path, device) -> (model, ck),
encode_history(model, ck, hist, hmask) (once per decision), score(model, ck, X, geom5, hist, hmask, vel=None,
domain_onehot=None, z=None) -> route logits (n,) f32.  Notes: artifacts/traverse/crm_improve_20260922/NOTES_ci_train.md.
"""
import argparse, fnmatch, hashlib, json, math, os, re, sys, time
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # less allocator fragmentation (read at the first CUDA allocation)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ga_train as GA
from ga_train import CNNFront, dev_group, prep_x, prep_hist, survival_nll, GEOM_COLS, HIST_ACTION_COLS, DOMAIN_NAME, ZDIM, WIDTH
from n2_arch_train import TokenTx, sinusoid
from gen_riskmodel import route_logit

DEV = os.environ.get('CI_DEVICE', os.environ.get('GA_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
CONDS = ('none', 'tag', 'hist', 'hist_aux')
CTX_MODES = ('geom', 'geom_vel')
DOMAIN_VOCAB = {'rigid': 0, 'crm': 1}
VEL_CH = [0, 6]                                    # history channels vx (state col 0) and yaw rate (state col 6)
HEAVY = ('X', 'hist', 'hmask', 'E', 'T')           # never kept whole in the row dict
META_KEYS = ('hist_cols', 'priv_names')
REQUIRED = ('X', 'ctx', 'id', 'group', 'split', 'fail', 'unsafe', 'event_idx')
SUITE_GROUPS = ['f104_crm_eval_group_*', 'f104_g1_test_group_*', 'f104_pair_group_*']   # ga_build_mixed.BLACKLIST
ANCHOR_BUCKETS = [('k0', 0, 0), ('k10_30', 10, 30), ('k40p', 40, 10 ** 9)]
HTX_D, HTX_LAYERS, HTX_HEADS = 64, 2, 4
PRED_BS = 512                                      # predict() batch; the round-trip check scores whole batches of this size


def parse_arch(arch):
    """'gru' -> ('gru', 0, 0); 'txD_L' -> ('tx', D, L); 'txjoint' / 'txjointD_L' -> ('txjoint', D, L) (default 96, 2)."""
    if arch == 'gru': return 'gru', 0, 0
    if arch == 'txjoint': return 'txjoint', 96, 2
    m = re.fullmatch(r'(txjoint|tx)(\d+)_(\d+)', arch)
    if not m: raise ValueError(f'unknown --arch {arch!r} (gru | tx96_2 | tx128_4 | txD_L | txjoint | txjointD_L)')
    return m.group(1), int(m.group(2)), int(m.group(3))


# ----------------------------------------------------------------------------- networks
class Block(nn.Module):
    """Pre-LN transformer layer (the arithmetic of nn.TransformerEncoderLayer(norm_first=True, gelu, ff 4d, dropout 0.1))
    with an explicit boolean attention mask allow (B,1,Lq|1,L), True = may attend; every query must keep >= 1 key."""
    def __init__(self, d, heads, drop=0.1):
        super().__init__()
        assert d % heads == 0, (d, heads)
        self.h, self.drop = heads, drop
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d); self.d1 = nn.Dropout(drop)
        self.n2 = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d)); self.d2 = nn.Dropout(drop)

    def forward(self, x, allow):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=allow, dropout_p=self.drop if self.training else 0.0)
        x = x + self.d1(self.proj(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.d2(self.ff(self.n2(x)))


class HistTx(nn.Module):
    """Causal history transformer: tokens Linear([frame * mask, mask]) + sinusoid of the frame age (0 = newest), a learned
    read-out token after the newest frame; token i attends to itself and to valid frames j <= i; z = tanh(Linear(LN(read-out)))."""
    def __init__(self, din, zdim, T, d=HTX_D, layers=HTX_LAYERS, heads=HTX_HEADS):
        super().__init__()
        self.inp = nn.Linear(din + 1, d); self.register_buffer('age_pe', sinusoid(T, d)); self.readout = nn.Parameter(torch.randn(d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)]); self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, zdim)

    def forward(self, hist, hmask):
        B, T, _ = hist.shape; m = hmask.to(hist.dtype)[..., None]
        age = torch.arange(T - 1, -1, -1, device=hist.device)
        tok = torch.cat([self.inp(torch.cat([hist * m, m], -1)) + self.age_pe[age][None], self.readout.expand(B, 1, -1)], 1)
        L = T + 1; valid = torch.cat([hmask.bool(), torch.ones(B, 1, dtype=torch.bool, device=hist.device)], 1)
        causal = torch.ones(L, L, dtype=torch.bool, device=hist.device).tril()
        allow = (causal[None] & valid[:, None, :]) | torch.eye(L, dtype=torch.bool, device=hist.device)[None]
        h = tok
        for b in self.blocks: h = b(h, allow[:, None])
        return torch.tanh(self.out(self.ln(h[:, -1])))


class TxJoint(nn.Module):
    """[CTX] + 96 station tokens (CNN column + station sinusoid) + T history-frame tokens (Linear([frame, mask]) + learned
    time-of-frame embedding by age); a learned type embedding per kind (0 ctx, 1 station, 2 history); invalid frames are
    masked as keys.  embed_hist is the once-per-decision part (the history tokens do not depend on the candidate)."""
    def __init__(self, cin, nctx, T, use_hist, d=96, layers=2, din=15):
        super().__init__()
        self.front = CNNFront(cin); self.embed = nn.Linear(192, d); self.register_buffer('pe', sinusoid(96, d)); self.ctx = nn.Linear(nctx, d)
        self.tok_type = nn.Parameter(torch.randn(3, d) * 0.02); self.use_hist = use_hist; self.d = d
        if use_hist:
            self.hin = nn.Linear(din + 1, d); self.time = nn.Parameter(torch.randn(T, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, max(1, d // 32)) for _ in range(layers)]); self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, 1)

    def embed_hist(self, hist, hmask):
        m = hmask.to(hist.dtype)[..., None]; T = hist.shape[1]; age = torch.arange(T - 1, -1, -1, device=hist.device)
        return self.hin(torch.cat([hist * m, m], -1)) + self.time[age][None] + self.tok_type[2]

    def forward(self, x, ctx, htok=None, hmask=None):
        B = x.shape[0]; st = self.embed(self.front(x)) + self.pe[None] + self.tok_type[1]
        parts = [(self.ctx(ctx) + self.tok_type[0])[:, None], st]; valid = [torch.ones(B, 97, dtype=torch.bool, device=x.device)]
        if htok is not None:
            parts.append(htok); valid.append(hmask.bool())
        h = torch.cat(parts, 1); allow = torch.cat(valid, 1)[:, None, None, :]
        for b in self.blocks: h = b(h, allow)
        h = self.ln(h)
        return {'haz': self.head(h[:, 1:97]).squeeze(-1), 'hout': h[:, 97:]}


class CIModel(nn.Module):
    """forward(x (B,6,96,32) standardised, ctx (B,nctx), hist (B,T,15) standardised (masked steps zeroed), hmask (B,T),
    z=None) -> {'haz': (B,96)[, 'z', 'dom']}.  z (precomputed by encode) = (B,zdim) for gru/tx arches, (tokens, mask) for txjoint."""
    def __init__(self, arch, hist_enc, cond, cin, nctx, hist_T, zdim=ZDIM, hist_dh=15, width=WIDTH):
        super().__init__()
        self.arch, self.cond, self.hist_T = arch, cond, int(hist_T)
        self.kind, d, L = parse_arch(arch)
        self.hist_enc = 'joint' if self.kind == 'txjoint' else hist_enc
        self.use_hist = cond in ('hist', 'hist_aux'); self.use_z = self.use_hist and self.kind != 'txjoint'; self.use_priv = False
        self.zdim = int(zdim) if self.use_z else 0; zin = self.zdim
        if self.kind == 'gru':   # ga_train.GAModel layout, same construction order -> same initial weights for a seed
            self.front = CNNFront(cin); self.lat = nn.Linear(192, 96); self.ctx = nn.Sequential(nn.Linear(nctx + zin, 32), nn.GELU())
            self.tconv = nn.Sequential(nn.Conv1d(129, 96, 5, padding=2), nn.GELU(), nn.Dropout(0.1))
            self.mix = nn.GRU(96, width, num_layers=1, batch_first=True, bidirectional=True)
            self.head = nn.Linear(2 * width, 1)
        elif self.kind == 'tx':
            self.bb = TokenTx(cin, nctx + zin, d, L)
        else:
            self.bb = TxJoint(cin, nctx, self.hist_T, self.use_hist, d, L, hist_dh)
        if self.use_z:
            if hist_enc == 'gru':
                self.henc = nn.GRU(hist_dh + 1, 32, batch_first=True); self.hz = nn.Linear(32, self.zdim)
            elif hist_enc == 'tx':
                self.htx = HistTx(hist_dh, self.zdim, self.hist_T)
            else:
                raise ValueError(hist_enc)
        if cond == 'hist_aux':
            self.dom = nn.Linear(self.zdim if self.use_z else d, 1)

    def encode(self, hist, hmask):
        """z (B,zdim) for gru/tx arches; (tokens (B,T,d), mask) for txjoint."""
        if self.kind == 'txjoint': return self.bb.embed_hist(hist, hmask), hmask
        if self.hist_enc == 'tx': return self.htx(hist, hmask)
        m = hmask.to(hist.dtype)[..., None]
        _, hn = self.henc(torch.cat([hist * m, m], -1))
        return torch.tanh(self.hz(hn[-1]))

    def backbone_gru(self, x, ctx_full):
        f = F.gelu(self.lat(self.front(x))); B, S, _ = f.shape
        pos = torch.linspace(0, 1, S, device=x.device)[None, :, None].expand(B, S, 1)
        c = self.ctx(ctx_full)[:, None, :].expand(B, S, 32)
        h = self.tconv(torch.cat([f, c, pos], -1).transpose(1, 2)).transpose(1, 2)
        return self.head(self.mix(h)[0]).squeeze(-1)

    def forward(self, x, ctx, hist=None, hmask=None, z=None):
        out = {}
        if self.kind == 'txjoint':
            htok = hm = None
            if self.use_hist:
                htok, hm = self.encode(hist, hmask) if z is None else z
            o = self.bb(x, ctx, htok, hm); out['haz'] = o['haz']
            if self.cond == 'hist_aux':
                w = hm.to(o['hout'].dtype)[..., None]
                out['dom'] = self.dom((o['hout'] * w).sum(1) / w.sum(1).clamp_min(1.0)).squeeze(-1)
            return out
        if self.use_z:
            if z is None: z = self.encode(hist, hmask)
            ctx = torch.cat([ctx, z], -1); out['z'] = z
        out['haz'] = self.backbone_gru(x, ctx) if self.kind == 'gru' else self.bb(x, ctx)['haz']
        if self.cond == 'hist_aux':
            out['dom'] = self.dom(z).squeeze(-1)
        return out


# ----------------------------------------------------------------------------- windows and context
def cut_window(hist, hmask, T):
    """Raw window (T0,15) or (m,T0,15) + mask -> (m,T,15) f32, (m,T) bool: the newest T frames (left-padded masked if T0 < T)."""
    H = np.asarray(hist, np.float32); M = np.asarray(hmask, bool)
    if H.ndim == 2: H, M = H[None], M[None]
    assert H.shape[:2] == M.shape and H.shape[2] == 15, (H.shape, M.shape)
    if H.shape[1] >= T: return H[:, H.shape[1] - T:], M[:, M.shape[1] - T:]
    pad = T - H.shape[1]
    return (np.concatenate([np.zeros((len(H), pad, 15), np.float32), H], 1), np.concatenate([np.zeros((len(M), pad), bool), M], 1))


def apply_window(H, M, T, valid_T):
    """Newest T frames of (m,T0,15) windows (cut), frames older than the newest valid_T masked (mask mode: T > valid_T)."""
    H, M = H[:, H.shape[1] - T:], M[:, M.shape[1] - T:]
    if valid_T < T:
        M = M.copy(); M[:, :T - valid_T] = False
    return H, M


def window(ck, hist, hmask):
    """Caller window -> the model's window (checkpoint rule: hist_T frames consumed, the newest hist_valid_T visible)."""
    H, M = cut_window(hist, hmask, ck['hist_T'])
    return apply_window(H, M, ck['hist_T'], ck.get('hist_valid_T', ck['hist_T']))


def vel_from_history(H, M):
    """(m,2) raw [vx, yaw rate] of the newest valid frame of each window; zero when the window is empty."""
    H = np.asarray(H, np.float32); M = np.asarray(M, bool)
    if H.ndim == 2: H, M = H[None], M[None]
    T = M.shape[1]; last = T - 1 - np.argmax(M[:, ::-1], axis=1); has = M.any(1)
    v = np.nan_to_num(H[np.arange(len(H)), last][:, VEL_CH]).astype(np.float32)
    v[~has] = 0.0
    return v


def blacklisted(g): return any(fnmatch.fnmatch(g, p) for p in SUITE_GROUPS)


# ----------------------------------------------------------------------------- data
def chan_stats_multi(parts, rows_per_part, cont, chunk=4096):
    """ga_train.chan_stats over the fit rows of several arrays (same chunking, so one array gives identical numbers)."""
    s = np.zeros(len(cont)); s2 = np.zeros(len(cont)); n = 0
    for X16, rows in zip(parts, rows_per_part):
        for i in range(0, len(rows), chunk):
            x = X16[rows[i:i + chunk]][:, cont].astype(np.float64); s += x.sum((0, 2, 3)); s2 += (x ** 2).sum((0, 2, 3)); n += x.shape[0] * x.shape[2] * x.shape[3]
    mu = s / n; sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0)) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def parse_weights(s):
    if not s: return {}
    out = {}
    for kv in s.split(','):
        k, v = kv.split('='); out[k.strip()] = float(v)
    return out


class CIData:
    def __init__(self, paths, a):
        t0 = time.time(); self.need_hist = a.cond in ('hist', 'hist_aux'); need_window = self.need_hist or a.ctx == 'geom_vel'
        zs = [np.load(p, allow_pickle=True) for p in paths]; ns = [len(z['id']) for z in zs]
        keysets = [set(z.files) for z in zs]
        for p, ks in zip(paths, keysets):
            miss = [k for k in REQUIRED + (('hist', 'hmask') if need_window else ()) if k not in ks]
            assert not miss, f'{p}: missing keys {miss}'
        common = set.intersection(*keysets); self.dropped_keys = sorted(set.union(*keysets) - common)
        assert not (a.strict_keys and self.dropped_keys), f'--strict-keys: key sets differ ({self.dropped_keys})'
        self.meta = {}
        for k in META_KEYS:
            if k in common:
                vals = [np.asarray(z[k]) for z in zs]
                assert all(v.shape == vals[0].shape and (v == vals[0]).all() for v in vals[1:]), f'{k} differs between --ds files'
                self.meta[k] = vals[0]
        d = {}
        for k in sorted(common - set(HEAVY) - set(META_KEYS)):
            parts = [z[k] for z in zs]
            if all(getattr(v, 'ndim', 0) >= 1 and v.shape[0] == n for v, n in zip(parts, ns)):
                d[k] = np.concatenate(parts) if len(parts) > 1 else parts[0]
            else:
                self.meta[k] = parts[0]
        fidx = np.repeat(np.arange(len(zs)), ns); loc = np.concatenate([np.arange(n) for n in ns]); n_all = len(fidx)
        ids = d['id'].astype(str); u, c = np.unique(ids, return_counts=True)
        assert (c == 1).all(), f'{int((c > 1).sum())} duplicate ids across --ds files, e.g. {u[c > 1][:5].tolist()}'
        if 'domain' not in d:
            assert all(i.endswith(('@crm', '@rigid')) for i in ids), 'no domain array and ids are not suffixed @crm/@rigid'
            d['domain'] = np.array([1 if i.endswith('@crm') else 0 for i in ids], np.int8)
        if 'anchor_frame' not in d: d['anchor_frame'] = np.zeros(n_all, np.int32)
        if 'profile' not in d: d['profile'] = np.full(n_all, -1, np.int8)
        if 'source' not in d: d['source'] = np.array(['unknown'] * n_all, object)
        d['file_index'] = fidx.astype(np.int16)
        # row filters (ga_train order): domain, startup-only
        dom = d['domain'].astype(int); keep = np.ones(n_all, bool)
        if a.domain_filter != 'both': keep &= dom == (1 if a.domain_filter == 'crm' else 0)
        if a.startup_only: keep &= d['anchor_frame'].astype(int) == 0
        K = np.where(keep)[0]; d = {k: v[K] for k, v in d.items()}; fidx, loc = fidx[K], loc[K]
        sp = d['split'].astype(str); grp = d['group'].astype(str); isdev = np.array([dev_group(g) for g in grp])
        fit = (sp == 'train') & (~isdev if a.mode == 'holdout' else True); dev = (sp == 'train') & isdev; test = sp == a.split_eval
        assert fit.sum() > 0, 'no training rows'; assert test.sum() > 0, f'no rows with split == {a.split_eval}'
        if a.subsample:   # ga_train semantics: a fraction of the training groups (both domains)
            rng = np.random.default_rng(a.seed0); gs = np.unique(grp[fit]); kp = set(rng.choice(gs, max(1, int(len(gs) * a.subsample)), replace=False))
            fit = fit & np.array([g in kp for g in grp])
        self.data_frac = {}
        if a.data_frac_crm < 1.0 or a.data_frac_rigid < 1.0:   # one seeded permutation of the fit groups, shared by the domains
            gs = np.unique(grp[fit]); perm = np.random.default_rng([a.data_seed, 7919]).permutation(len(gs)); rank = dict(zip(gs[perm], range(len(gs))))
            r = np.array([rank.get(g, -1) for g in grp]); dom = d['domain'].astype(int)
            for code, frac in ((0, a.data_frac_rigid), (1, a.data_frac_crm)):
                if frac >= 1.0: continue
                kmax = max(1, int(len(gs) * frac)); before = int((fit & (dom == code)).sum())
                fit = fit & ~((dom == code) & (r >= kmax))
                self.data_frac[DOMAIN_NAME[code]] = dict(frac=frac, groups_kept=kmax, groups_total=int(len(gs)), rows_before=before, rows_after=int((fit & (dom == code)).sum()))
        bad = sorted({g for g in np.unique(grp[fit]) if blacklisted(g)})
        assert not bad, f'suite groups among the fit rows (evaluation data, never trained on): {bad[:5]}'
        bad_eval = sorted({g for g in np.unique(grp[dev | test]) if blacklisted(g)})
        if bad_eval: print(f'WARNING: {len(bad_eval)} suite groups among the evaluated rows, e.g. {bad_eval[:3]}', flush=True)
        used = np.ones(len(fit), bool) if a.keep_all_rows else (fit | dev | test)
        U = np.where(used)[0]; self.d = {k: v[U] for k, v in d.items()}; fidx, loc = fidx[U], loc[U]
        self.fit, self.dev, self.test = fit[U], dev[U], test[U]; self.n = len(U)
        self.split_hash = hashlib.md5('\n'.join(sorted(self.d['id'][self.fit].astype(str))).encode()).hexdigest()
        self.files = [dict(path=os.path.abspath(p), rows=int(n), rows_used=int((fidx == i).sum()), fit=int(self.fit[fidx == i].sum())) for i, (p, n) in enumerate(zip(paths, ns))]
        # heavy arrays, per file, used rows only
        Xp, Hp, Mp, offs = [], [], [], []
        for i, z in enumerate(zs):
            sel = loc[fidx == i]
            if len(sel) == 0: continue
            offs.append(int(np.flatnonzero(fidx == i)[0])); full = len(sel) == ns[i] and (sel == np.arange(ns[i])).all()
            Xf = z['X']; assert Xf.shape[1:] == (5, 96, 32), Xf.shape
            Xp.append(Xf if full else Xf[sel]); del Xf
            if need_window:
                Hf, Mf = z['hist'], z['hmask'].astype(bool); assert Hf.shape[2] == 15 and Mf.shape == Hf.shape[:2], (Hf.shape, Mf.shape)
                Hp.append(Hf if full else Hf[sel]); Mp.append(Mf if full else Mf[sel]); del Hf, Mf
        self.fit_idx = np.where(self.fit)[0]; dom = self.d['domain'].astype(int); self.dom_np = dom
        fit_rows = [np.where(self.fit[o:o + len(x)])[0] for o, x in zip(offs, Xp)]
        mu, sd = chan_stats_multi(Xp, fit_rows, [0, 1, 2, 3])
        self.norm = dict(mu=mu, sd=sd, cont_index=[0, 1, 2, 3], channels=['elev_rel', 'grade', 'cross', 'speed', 'valid'])
        self.x_host = a.x_host; xdt = torch.float16 if a.x_half else torch.float32
        self.X = torch.empty((self.n, 6, 96, 32), dtype=xdt, device='cpu', pin_memory=torch.cuda.is_available()) if a.x_host else torch.empty((self.n, 6, 96, 32), dtype=xdt, device=DEV)
        for o, x in zip(offs, Xp):
            for i in range(0, len(x), 4096):
                v = prep_x(torch.from_numpy(x[i:i + 4096].astype(np.float32)).to(DEV), self.norm).to(xdt)
                self.X[o + i:o + i + len(v)] = v.cpu() if a.x_host else v
        # raw sample for the checkpoint round trip (raw corridor, geometry, full stored window)
        rs = np.zeros(0, int)   # whole predict() batches of PRED_BS rows (random blocks, sorted) so the comparison is batch-aligned
        if a.roundtrip_check:
            nb = (self.n + PRED_BS - 1) // PRED_BS; blocks = np.sort(np.random.default_rng(0).permutation(nb)[:max(1, int(math.ceil(a.roundtrip_n / PRED_BS)))])
            rs = np.concatenate([np.arange(b * PRED_BS, min((b + 1) * PRED_BS, self.n)) for b in blocks])
        H_all = np.concatenate(Hp) if (need_window and len(Hp) > 1) else (Hp[0] if need_window else None)
        M_all = np.concatenate(Mp) if (need_window and len(Mp) > 1) else (Mp[0] if need_window else None)
        if len(rs):
            part_of = np.searchsorted(np.array(offs), rs, side='right') - 1
            self.raw = dict(idx=rs, X=np.stack([Xp[p][r - offs[p]].astype(np.float32) for p, r in zip(part_of, rs)]),
                            geom=self.d['ctx'][rs][:, GEOM_COLS].astype(np.float32), onehot=np.eye(2, dtype=np.float32)[dom[rs]],
                            hist=(H_all[rs].astype(np.float32) if need_window else None), hmask=(M_all[rs] if need_window else None))
        else:
            self.raw = None
        del Xp
        # window: newest N frames
        self.hist_T_stored = int(H_all.shape[1]) if need_window else 40
        self.hist_valid_T = int(a.hist_T or self.hist_T_stored); assert 1 <= self.hist_valid_T <= self.hist_T_stored, (self.hist_valid_T, self.hist_T_stored)
        self.hist_window = a.hist_window; self.hist_T = self.hist_valid_T if a.hist_window == 'cut' else self.hist_T_stored   # frames the model consumes
        if need_window:
            H_all, M_all = apply_window(H_all, M_all, self.hist_T, self.hist_valid_T)
        hc = [int(c) for c in self.meta['hist_cols']] if 'hist_cols' in self.meta else list(GA.HIST_COLS)
        if len(hc) == 12: hc = hc + HIST_ACTION_COLS
        assert len(hc) == 15, hc; self.hist_cols = hc
        # context: geometry [+ velocity], standardised on fit rows [+ one-hot domain for tag]
        geom = self.d['ctx'].astype(np.float32)[:, GEOM_COLS]
        if a.ctx == 'geom_vel':
            self.vel = vel_from_history(H_all, M_all); geom = np.concatenate([geom, self.vel], 1)
        self.ctx_mu, self.ctx_sd = geom[self.fit].mean(0), geom[self.fit].std(0) + 1e-6
        ctx = (geom - self.ctx_mu) / self.ctx_sd; self.onehot = np.eye(2, dtype=np.float32)[dom]
        if a.cond == 'tag': ctx = np.concatenate([ctx, self.onehot], 1)
        self.ctx = torch.from_numpy(ctx.astype(np.float32)).to(DEV); self.nctx = ctx.shape[1]
        self.ev = torch.from_numpy(self.d['event_idx'].astype(np.int64)).to(DEV); self.dom = torch.from_numpy(dom).to(DEV)
        # history (ga_train arithmetic on the cut window)
        if self.need_hist:
            H, M = H_all, M_all
            Hf = np.nan_to_num(H[self.fit].astype(np.float32)); Mf = M[self.fit]; cnt = Mf.sum()
            if cnt > 0:
                self.hist_mu = (Hf * Mf[..., None]).sum((0, 1)) / cnt; self.hist_sd = np.sqrt(((Hf - self.hist_mu) ** 2 * Mf[..., None]).sum((0, 1)) / cnt) + 1e-6
            else:
                self.hist_mu = np.zeros(15, np.float32); self.hist_sd = np.ones(15, np.float32)
            del Hf, Mf
            self.hist = torch.empty((self.n, H.shape[1], 15), dtype=torch.float32, device=DEV); self.hmask = torch.from_numpy(np.ascontiguousarray(M)).to(DEV)
            for i in range(0, self.n, 8192):
                self.hist[i:i + 8192], _ = prep_hist(torch.from_numpy(H[i:i + 8192].astype(np.float32)).to(DEV), self.hmask[i:i + 8192], self.hist_mu, self.hist_sd)
        else:
            self.hist = self.hmask = None
            self.hist_mu = np.zeros(15, np.float32); self.hist_sd = np.ones(15, np.float32)
        del H_all, M_all
        # per-row loss weights by source
        self.weights = parse_weights(a.row_weight); src = self.d['source'].astype(str)
        for s_ in self.weights:
            if not (src[self.fit] == s_).any(): print(f'WARNING: --row-weight source {s_!r} has no fit rows', flush=True)
        self.w = torch.from_numpy(np.array([self.weights.get(s_, 1.0) for s_ in src], np.float32)).to(DEV) if self.weights else None
        if DEV.startswith('cuda'): torch.cuda.empty_cache()   # release the upload temporaries (device footprint = data + training)
        self.load_secs = round(time.time() - t0, 1)


class CIBatches:
    """ga_train.Batches with a CRM share: when both domains are fitted each batch = floor(bs*(1-F)) rigid + the rest CRM,
    from two independent per-domain permutations (re-drawn when exhausted); F = 0.5 reproduces ga_train's halves."""
    def __init__(self, fit_idx, dom_np, bs, crm_frac, balanced, gen):
        self.bs, self.gen = bs, gen; self.streams, self.q = [], []
        if balanced:
            q_r = int(math.floor(bs * (1.0 - crm_frac) + 1e-9)); parts = [(fit_idx[dom_np[fit_idx] == 0], q_r), (fit_idx[dom_np[fit_idx] == 1], bs - q_r)]
            parts = [(p, q) for p, q in parts if len(p) > 0]
            if len(parts) == 1: parts = [(parts[0][0], bs)]
            parts = [(p, q) for p, q in parts if q > 0]
        else:
            parts = [(fit_idx, bs)]
        for p, q in parts: self.streams.append([torch.from_numpy(p).to(DEV), None, 0]); self.q.append(q)

    def next(self):
        out = []
        for st, q in zip(self.streams, self.q):
            idx, perm, cur = st
            if perm is None or cur + q > len(perm):
                st[1] = perm = idx[torch.randperm(len(idx), generator=self.gen, device='cpu').to(DEV)]; st[2] = cur = 0
            out.append(perm[cur:cur + q]); st[2] = cur + q
        return torch.cat(out)


def model_inputs(D, k, hm=None):
    x = D.X[k.cpu()].to(DEV, non_blocking=True) if D.x_host else D.X[k]
    if x.dtype != torch.float32: x = x.to(torch.float32)
    if D.hist is None: return x, D.ctx[k], None, None
    return x, D.ctx[k], D.hist[k], (D.hmask[k] if hm is None else hm)


def predict(model, D, bs=PRED_BS):
    model.eval(); z, dl = [], []
    with torch.no_grad():
        for i in range(0, D.n, bs):
            k = torch.arange(i, min(i + bs, D.n), device=DEV); x, c, h, m = model_inputs(D, k)
            o = model(x, c, h, m)
            z.append(route_logit(o['haz']).float().cpu().numpy())
            if 'dom' in o: dl.append(o['dom'].float().cpu().numpy())
    return np.concatenate(z), (np.concatenate(dl) if dl else None)


def run_epochs(model, D, a, params, epochs, lr, gen, loss_fn, hist_drop=0.0, wd=1e-4, log=None):
    """ga_train.run_epochs with CIBatches (same RNG call order: batch permutations, then the history-drop draw)."""
    n = len(D.fit_idx); bs = min(a.bs, n); steps = max(epochs * (n // bs), 1)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd); sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    batches = CIBatches(D.fit_idx, D.dom_np, bs, a.crm_batch_frac, a.domain_filter == 'both', gen); losses = []
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
    if log is not None: log.update(steps=steps, first_loss=losses[0], final_loss=losses[-1], mean_loss_last50=float(np.mean(losses[-50:])),
                                   batch_quota=dict(zip([DOMAIN_NAME.get(int(D.dom_np[s[0][0].item()]), '?') for s in batches.streams], batches.q)))
    return losses


def risk_loss_fn(model, D, a):
    def fn(k, hm):
        x, c, h, m = model_inputs(D, k, hm); out = model(x, c, h, m)
        nll = survival_nll(out['haz'], D.ev[k])
        loss = nll.mean() if D.w is None else (nll * D.w[k]).sum() / D.w[k].sum()
        if model.cond == 'hist_aux':
            vis = m.any(1)
            if vis.any(): loss = loss + a.aux_weight * F.binary_cross_entropy_with_logits(out['dom'][vis], D.dom[k][vis].to(torch.float32))
        return loss
    return fn


# ----------------------------------------------------------------------------- metrics
def cell_index(cells):
    """Row-index arrays of the cells with >= 2 rows."""
    _, inv, cnt = np.unique(cells, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind='stable'); starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    return [order[s:s + c] for s, c in zip(starts, cnt) if c >= 2]


def state_metrics(s, d, m):
    """Decision-state cells episode|anchor_frame|domain (several routes driven from one identical start state, i.e. branch
    rows; singleton cells skipped): within-state AUC S_* and lowest-risk pick failure per state."""
    if 'episode' not in d or m.sum() == 0: return {'S_cells': 0}
    cells = np.array([f'{e}|{k}|{x}' for e, k, x in zip(d['episode'][m].astype(str), d['anchor_frame'][m].astype(int), d['domain'][m].astype(int))])
    idx = cell_index(cells); out = {'S_cells': len(idx), 'S_rows': int(sum(len(i) for i in idx))}
    for lab in ('unsafe', 'fail'):
        y = d[lab][m].astype(float); ok = tot = 0.0; picks, rand, orac = [], [], []
        for k in idx:
            yy, ss = y[k], s[k]; picks.append(yy[np.argmin(ss)]); rand.append(yy.mean()); orac.append(yy.min())
            if yy.min() == yy.max(): continue
            dd = ss[yy == 1][:, None] - ss[yy == 0][None, :]; ok += (dd > 0).sum() + 0.5 * (dd == 0).sum(); tot += dd.size
        suf = '' if lab == 'fail' else '_unsafe'
        out[f'S_{lab}'], out[f'Sn_{lab}'] = (ok / tot if tot else float('nan')), int(tot)
        out.update({f'S_pick_fail{suf}': float(np.mean(picks)) if picks else float('nan'), f'S_random_fail{suf}': float(np.mean(rand)) if rand else float('nan'),
                    f'S_oracle_fail{suf}': float(np.mean(orac)) if orac else float('nan')})
    return out


def regime_plus(s, d, m):
    out = GA.regime_metrics(s, d, m)
    if out['n']: out.update(state_metrics(s, d, m))
    return out


def metrics(s, d, m):
    """ga_train.metrics (domain x startup/established/all, 'both' pooled) with the decision-state cells added."""
    dom = d['domain'].astype(int); startup = d['anchor_frame'].astype(int) == 0
    assert len(s) == len(dom) == len(m), 'metrics() takes the full logit vector plus a row mask'
    out = {}; doms = sorted(np.unique(dom[m]).tolist())
    for key, dm in [(DOMAIN_NAME.get(x, str(x)), m & (dom == x)) for x in doms] + ([('both', m)] if len(doms) > 1 else []):
        out[key] = {name: regime_plus(s[mm], d, mm) for name, mm in (('startup', dm & startup), ('established', dm & ~startup), ('all', dm))}
    return out


def anchor_masks(d):
    k = d['anchor_frame'].astype(int); out = [(name, (k >= lo) & (k <= hi)) for name, lo, hi in ANCHOR_BUCKETS]
    other = ~np.any([mm for _, mm in out], 0)
    if other.any(): out.append(('k_other', other))
    return out


def all_metrics(s, d, m):
    """{'all': metrics, 'by_source': {source: metrics}, 'by_anchor': {bucket: metrics}} on rows m."""
    src = d['source'].astype(str)
    return dict(all=metrics(s, d, m), by_source={x: metrics(s, d, m & (src == x)) for x in sorted(np.unique(src[m]))},
                by_anchor={name: metrics(s, d, m & mm) for name, mm in anchor_masks(d) if (m & mm).any()})


# ----------------------------------------------------------------------------- training
def default_lr_wd(a, kind):
    lr = a.lr if a.lr is not None else (2e-3 if kind == 'gru' else 1e-3)
    wd = a.wd if a.wd is not None else (1e-4 if kind == 'gru' else 0.05)
    return lr, wd


def train_one(D, a, seed):
    torch.manual_seed(seed); np.random.seed(seed); gen = torch.Generator().manual_seed(seed)
    if DEV.startswith('cuda'): torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = CIModel(a.arch, a.hist_enc, a.cond, D.X.shape[1], D.nctx, D.hist_T).to(DEV); lr, wd = default_lr_wd(a, model.kind)
    t0 = time.time(); s1 = {}
    run_epochs(model, D, a, list(model.parameters()), a.epochs, lr, gen, risk_loss_fn(model, D, a), hist_drop=(a.hist_drop if model.use_hist else 0.0), wd=wd, log=s1)
    if DEV.startswith('cuda'): torch.cuda.synchronize()
    took = time.time() - t0
    if DEV.startswith('cuda'): torch.cuda.empty_cache()
    t1 = time.time(); z, dl = predict(model, D); tp = time.time() - t1
    row = dict(arch=a.arch, hist_enc=model.hist_enc, hist_T=D.hist_T, hist_valid_T=D.hist_valid_T, hist_window=D.hist_window, ctx=a.ctx, cond=a.cond, domain_filter=a.domain_filter, split_eval=a.split_eval, mode=a.mode,
               startup_only=a.startup_only, hist_drop=a.hist_drop, crm_batch_frac=a.crm_batch_frac, data_frac_crm=a.data_frac_crm, data_frac_rigid=a.data_frac_rigid,
               row_weight=D.weights, lr=lr, wd=wd, epochs=a.epochs, seed=seed, params=sum(p.numel() for p in model.parameters()), secs=round(took, 1),
               secs_predict=round(tp, 1), sec_per_step=round(took / max(s1['steps'], 1), 4), n_fit=int(D.fit.sum()), steps=s1['steps'], final_loss=s1['final_loss'],
               stages=dict(train=s1), dev=all_metrics(z, D.d, D.dev), heldout=all_metrics(z, D.d, D.test))
    if dl is not None: row['domain_head'] = dict(dev=GA.domain_head_metrics(dl, D.d, D.dev), heldout=GA.domain_head_metrics(dl, D.d, D.test))
    if DEV.startswith('cuda'):
        row['gpu_peak_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2); row['gpu_reserved_peak_gb'] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
    return model, row, z


# ----------------------------------------------------------------------------- checkpoint contract and API
def checkpoint_dict(model, D, a, seed, tag):
    lr, wd = default_lr_wd(a, model.kind); kind, d, L = parse_arch(a.arch)
    ctx_names = ['goal_dx', 'goal_dy', 'goal_dist', 'start_yaw', 'route_len'] + (['vx', 'yaw_rate'] if a.ctx == 'geom_vel' else []) + (['is_rigid', 'is_crm'] if a.cond == 'tag' else [])
    ck = dict(model_kind='ci_train', arch=a.arch, arch_kind=kind, tx_d=d, tx_layers=L, hist_enc=model.hist_enc,
              hist_tx=dict(d=HTX_D, layers=HTX_LAYERS, heads=HTX_HEADS) if model.hist_enc == 'tx' else None,
              cond=model.cond, ctx_mode=a.ctx, ctx_names=ctx_names, geom_cols=GEOM_COLS, vel_hist_channels=VEL_CH, domain_vocab=DOMAIN_VOCAB,
              state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
              cin=int(D.X.shape[1]), nctx=int(D.nctx), zdim=int(model.zdim), width=WIDTH,
              hist_cols=list(D.hist_cols), hist_cols_state=list(D.hist_cols[:12]), hist_cols_action=list(D.hist_cols[12:]),
              hist_T=int(D.hist_T), hist_valid_T=int(D.hist_valid_T), hist_window=D.hist_window, hist_T_stored=int(D.hist_T_stored),
              hist_layout='hist[:, t, :12] = state[k-T+1+t, hist_cols_state]; hist[:, t, 12:] = action[k-T+t, hist_cols_action]; hmask True where the frame exists; the model consumes the newest hist_T frames of which the newest hist_valid_T are visible (older ones masked)',
              norm=dict(mu=np.asarray(D.norm['mu']), sd=np.asarray(D.norm['sd']), cont_index=list(D.norm['cont_index']), channels=list(D.norm['channels'])),
              ctx_mu=np.asarray(D.ctx_mu, np.float32), ctx_sd=np.asarray(D.ctx_sd, np.float32),
              hist_mu=np.asarray(D.hist_mu, np.float32), hist_sd=np.asarray(D.hist_sd, np.float32),
              train_rows=int(D.fit.sum()), train_rows_by_domain={DOMAIN_NAME[x]: int((D.dom_np[D.fit] == x).sum()) for x in (0, 1)},
              split_hash=D.split_hash, split_eval=a.split_eval, mode=a.mode, domain_filter=a.domain_filter, startup_only=bool(a.startup_only),
              hist_drop=float(a.hist_drop), crm_batch_frac=float(a.crm_batch_frac), data_frac_crm=float(a.data_frac_crm), data_frac_rigid=float(a.data_frac_rigid),
              data_seed=int(a.data_seed), row_weight=dict(D.weights), lr=lr, wd=wd, epochs=a.epochs, seed=int(seed), tag=tag, ds=[f['path'] for f in D.files], args=vars(a))
    return ck


def load_ci_model(path, device=None):
    """Rebuild a ci_train checkpoint: (model in eval mode on device, checkpoint dict)."""
    device = device or DEV; ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck.get('model_kind') == 'ci_train', f'{path}: model_kind {ck.get("model_kind")!r} is not ci_train'
    model = CIModel(ck['arch'], ck['hist_enc'], ck['cond'], ck['cin'], ck['nctx'], ck['hist_T'], zdim=ck['zdim'] or ZDIM, hist_dh=len(ck['hist_cols']), width=ck.get('width', WIDTH))
    model.load_state_dict(ck['state']); model.to(device).eval()
    return model, ck


def encode_history(model, ck, hist, hmask, device=None):
    """Once-per-decision history context for raw windows hist (T0,15) | (m,T0,15) + hmask; hist=None -> all-masked window.
    gru/tx arches: z (m,zdim) f32; txjoint: {'tok': (m,hist_T,d) f32, 'mask': (m,hist_T) bool}; no-history models: None."""
    if not model.use_hist: return None
    device = device or next(model.parameters()).device; T = ck['hist_T']
    if hist is None: H, M = np.zeros((1, T, 15), np.float32), np.zeros((1, T), bool)
    else: H, M = window(ck, hist, hmask)
    h, m = prep_hist(torch.from_numpy(H).to(device), torch.from_numpy(M).to(device), ck['hist_mu'], ck['hist_sd'])
    with torch.no_grad():
        z = model.encode(h, m)
    if model.kind == 'txjoint': return dict(tok=z[0].float().cpu().numpy(), mask=M)
    return z.float().cpu().numpy()


def score(model, ck, X, geom5, hist=None, hmask=None, vel=None, domain_onehot=None, z=None, bs=1024):
    """Route logits (n,) f32 for raw inputs: X (n,5,96,32) f32 corridor, geom5 (n|1,5) raw geometry (ctx cols 17-21),
    hist (T0,15) | (n|1,T0,15) raw window + hmask (None = all-masked startup window; one window is shared by every
    candidate; the newest hist_T frames are used), vel (n|1,2) raw [vx, yaw rate] for ctx_mode geom_vel (default: from
    the newest valid frame of hist, zero for the startup window; required when only z is given), domain_onehot (n|1,2)
    for cond tag, z = encode_history(...) output to skip the history encoder."""
    device = next(model.parameters()).device; n = len(X); model.eval()
    geom = np.broadcast_to(np.asarray(geom5, np.float32).reshape(-1, 5), (n, 5))
    H = M = None
    if hist is not None: H, M = window(ck, hist, hmask)
    ctx_raw = geom
    if ck['ctx_mode'] == 'geom_vel':
        if vel is None:
            if H is not None: vel = vel_from_history(H, M)
            elif z is None: vel = np.zeros((1, 2), np.float32)
            else: raise ValueError('ctx_mode geom_vel: pass vel (or hist) together with a precomputed z')
        ctx_raw = np.concatenate([geom, np.broadcast_to(np.asarray(vel, np.float32).reshape(-1, 2), (n, 2))], 1)
    ctx = (ctx_raw - ck['ctx_mu']) / ck['ctx_sd']
    if ck['cond'] == 'tag':
        assert domain_onehot is not None, 'cond=tag needs domain_onehot'
        ctx = np.concatenate([ctx, np.broadcast_to(np.asarray(domain_onehot, np.float32).reshape(-1, 2), (n, 2))], 1)
    ctx_t = torch.from_numpy(np.ascontiguousarray(ctx, dtype=np.float32)).to(device)
    zt = None; per_row = model.use_hist and z is None and H is not None and len(H) > 1   # per-row windows: encoded chunk by chunk
    if model.use_hist and not per_row:
        if z is None: z = encode_history(model, ck, H, M, device)
        if model.kind == 'txjoint':
            tok = torch.as_tensor(np.asarray(z['tok'], np.float32)).to(device); mk = torch.as_tensor(np.asarray(z['mask'], bool)).to(device)
            if tok.shape[0] == 1 and n > 1: tok, mk = tok.expand(n, -1, -1), mk.expand(n, -1)
            zt = (tok, mk)
        else:
            zt = torch.as_tensor(np.asarray(z, np.float32)).to(device)
            if zt.shape[0] == 1 and n > 1: zt = zt.expand(n, -1)
    out = []
    with torch.no_grad():
        for i in range(0, n, bs):
            x = prep_x(torch.as_tensor(np.asarray(X[i:i + bs], np.float32)).to(device), ck['norm'])
            if per_row:   # same chunking as the trainer's predict(), so equal rows in equal batches give identical numbers
                h, m = prep_hist(torch.from_numpy(np.ascontiguousarray(H[i:i + bs])).to(device), torch.from_numpy(np.ascontiguousarray(M[i:i + bs])).to(device), ck['hist_mu'], ck['hist_sd'])
                zb = model.encode(h, m)
            else:
                zb = None if zt is None else ((zt[0][i:i + bs], zt[1][i:i + bs]) if model.kind == 'txjoint' else zt[i:i + bs])
            o = model(x, ctx_t[i:i + bs], z=zb)
            out.append(route_logit(o['haz']).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def roundtrip(path, D, zref, x_half):
    """load_ci_model + score on the raw sample == trainer predict (< 1e-5, float32 corridor); shared-window, startup and
    precomputed-z paths agree."""
    model, ck = load_ci_model(path, DEV); r = D.raw
    s = score(model, ck, r['X'], r['geom'], r['hist'], r['hmask'], domain_onehot=r['onehot'], bs=PRED_BS)
    out = dict(n=int(len(s)), max_abs_diff=float(np.abs(s - zref[r['idx']]).max()))
    u = np.sort(np.random.default_rng(1).choice(len(s), min(len(s), 300), replace=False))   # unaligned rows: batch-composition rounding only
    su = score(model, ck, r['X'][u], r['geom'][u], None if r['hist'] is None else r['hist'][u], None if r['hmask'] is None else r['hmask'][u], domain_onehot=r['onehot'][u])
    loose = dict(unaligned_diff=float(np.abs(su - zref[r['idx'][u]]).max()))
    if r['hist'] is not None:
        k = min(8, len(s)); X8, g8, oh8 = r['X'][:k], r['geom'][:k], r['onehot'][:1]
        a1 = score(model, ck, X8, g8, r['hist'][:1], r['hmask'][:1], domain_onehot=oh8)
        a2 = score(model, ck, X8, g8, np.repeat(r['hist'][:1], k, 0), np.repeat(r['hmask'][:1], k, 0), domain_onehot=oh8)
        zz = encode_history(model, ck, r['hist'][:1], r['hmask'][:1])
        vv = vel_from_history(*window(ck, r['hist'][:1], r['hmask'][:1])) if ck['ctx_mode'] == 'geom_vel' else None
        a3 = score(model, ck, X8, g8, z=zz, vel=vv, domain_onehot=oh8) if model.use_hist else a1
        s0 = score(model, ck, X8, g8, None, None, domain_onehot=oh8)
        s0b = score(model, ck, X8, g8, np.zeros((k, ck['hist_T_stored'], 15), np.float32), np.zeros((k, ck['hist_T_stored']), bool), domain_onehot=oh8)
        s0c = score(model, ck, X8, g8, z=encode_history(model, ck, None, None), vel=(np.zeros((1, 2)) if ck['ctx_mode'] == 'geom_vel' else None), domain_onehot=oh8) if model.use_hist else s0
        out.update(precomputed_z_diff=float(np.abs(a1 - a3).max()), startup_z_diff=float(np.abs(s0 - s0c).max()))
        loose.update(shared_window_diff=float(np.abs(a1 - a2).max()), startup_zero_window_diff=float(np.abs(s0 - s0b).max()))
    tol = 1e-3 if x_half else 1e-5
    bad = {k: v for k, v in out.items() if k != 'n' and v > tol}; bad.update({k: v for k, v in loose.items() if v > 1e-3})
    assert not bad, f'round trip failed (aligned {tol}, batch-shape checks 1e-3): {bad}'
    out.update(loose)
    return out


# ----------------------------------------------------------------------------- main
def fmt(m, key):
    r = m.get(key, {}); return ' '.join(f"{k}={r[k]:.3f}" for k in ('W_unsafe', 'P_unsafe', 'S_unsafe', 'pick_fail', 'brier_unsafe') if k in r and np.isfinite(r[k])) + f" n={r.get('n', 0)}"


def print_block(prefix, h):
    for dk in [k for k in ('rigid', 'crm') if k in h['all']]:
        print(f"{prefix} {dk:5s} startup[{fmt(h['all'][dk], 'startup')}] established[{fmt(h['all'][dk], 'established')}]", flush=True)
    for name, b in list(h['by_anchor'].items()) + [(f'src:{k}', v) for k, v in h['by_source'].items()]:
        print(f"{prefix}   {name:14s} " + ' | '.join(f"{dk} {fmt(b[dk], 'all')}" for dk in ('rigid', 'crm') if dk in b), flush=True)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ds', action='append', required=True, help='npz dataset; repeat to concatenate several files'); ap.add_argument('--out', required=True)
    ap.add_argument('--arch', default='gru', help='gru | tx96_2 | tx128_4 | txD_L | txjoint | txjointD_L'); ap.add_argument('--hist-enc', choices=['gru', 'tx'], default='gru')
    ap.add_argument('--hist-T', type=int, default=None, help='use the newest N of the stored frames (default all)')
    ap.add_argument('--hist-window', choices=['mask', 'cut'], default='mask', help='older frames masked in the stored-length window (default) or cut off'); ap.add_argument('--ctx', choices=CTX_MODES, default='geom')
    ap.add_argument('--cond', choices=CONDS, default='none'); ap.add_argument('--domain-filter', choices=['crm', 'rigid', 'both'], default='both')
    ap.add_argument('--crm-batch-frac', type=float, default=0.5); ap.add_argument('--data-frac-crm', type=float, default=1.0); ap.add_argument('--data-frac-rigid', type=float, default=1.0)
    ap.add_argument('--data-seed', type=int, default=0, help='seed of the group permutation behind --data-frac-*'); ap.add_argument('--row-weight', default='', help="'source=w,...' loss weights, e.g. branch=2")
    ap.add_argument('--hist-drop', type=float, default=0.2); ap.add_argument('--aux-weight', type=float, default=0.5)
    ap.add_argument('--split-eval', choices=['val', 'test'], default='val'); ap.add_argument('--startup-only', action='store_true')
    ap.add_argument('--mode', choices=['holdout', 'deploy'], default='holdout'); ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--lr', type=float, default=None, help='default 2e-3 gru, 1e-3 transformers'); ap.add_argument('--wd', type=float, default=None, help='default 1e-4 gru, 0.05 transformers')
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--bs', type=int, default=256); ap.add_argument('--subsample', type=float, default=None)
    ap.add_argument('--x-half', action='store_true', help='standardised corridor stored as float16 (halves memory, ~1e-5 logit cost)')
    ap.add_argument('--x-host', action='store_true', help='keep the corridor tensor in pinned host memory, move each batch to the device')
    ap.add_argument('--keep-all-rows', action='store_true', help='put every row on the device, as ga_train does (reproduction checks)')
    ap.add_argument('--strict-keys', action='store_true', help='require identical key sets across --ds files'); ap.add_argument('--deterministic', action='store_true', help='cudnn deterministic')
    ap.add_argument('--roundtrip-check', action='store_true', help='reload each checkpoint and compare score() with predict()'); ap.add_argument('--roundtrip-n', type=int, default=2048)
    ap.add_argument('--tag', default=None); ap.add_argument('--no-save', action='store_true')
    return ap


def main():
    a = build_parser().parse_args(); os.makedirs(a.out, exist_ok=True)
    kind, _, _ = parse_arch(a.arch); assert 0.0 <= a.crm_batch_frac <= 1.0 and 0.0 < a.data_frac_crm <= 1.0 and 0.0 < a.data_frac_rigid <= 1.0
    assert not (a.roundtrip_check and a.no_save), '--roundtrip-check needs the saved checkpoint'
    if a.deterministic: torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if kind == 'txjoint' and a.cond in ('none', 'tag'): print('NOTE: txjoint with cond none/tag has no history tokens (station transformer with own blocks)', flush=True)
    henc = 'joint' if kind == 'txjoint' else a.hist_enc
    tag = a.tag or (f"ci_{a.arch}_{henc}_{a.cond}_{a.ctx}_T{a.hist_T or 'all'}{'cut' if (a.hist_T and a.hist_window == 'cut') else ''}_{a.domain_filter}{'_startup' if a.startup_only else ''}_{a.mode}_{a.split_eval}"
                    + (f'_cb{a.crm_batch_frac:g}' if a.crm_batch_frac != 0.5 else '') + (f'_fc{a.data_frac_crm:g}' if a.data_frac_crm < 1 else '')
                    + (f'_fr{a.data_frac_rigid:g}' if a.data_frac_rigid < 1 else '') + (f"_rw{a.row_weight.replace('=', '').replace(',', '-')}" if a.row_weight else '')
                    + (f'_sub{a.subsample:g}' if a.subsample else ''))
    D = CIData(a.ds, a); dom_counts = {DOMAIN_NAME[x]: int((D.dom_np[D.fit] == x).sum()) for x in (0, 1)}
    src = D.d['source'].astype(str); src_counts = {str(x): int((src[D.fit] == x).sum()) for x in sorted(np.unique(src[D.fit]))}
    gpu_data = round(torch.cuda.memory_allocated() / 2 ** 30, 2) if DEV.startswith('cuda') else None
    print(f'=== {tag}: files {len(D.files)} rows used {D.n} fit {int(D.fit.sum())} {dom_counts} {src_counts} dev {int(D.dev.sum())} heldout({a.split_eval}) {int(D.test.sum())} '
          f'nctx {D.nctx} hist frames {D.hist_valid_T} visible / {D.hist_T} consumed / {D.hist_T_stored} stored split_hash {D.split_hash[:8]} load {D.load_secs}s data on gpu {gpu_data} GB'
          + (f' dropped keys {D.dropped_keys}' if D.dropped_keys else '') + (f' data_frac {D.data_frac}' if D.data_frac else ''), flush=True)
    rows, logits = [], []
    for s in range(a.seed0, a.seed0 + a.seeds):
        model, row, z = train_one(D, a, s); rows.append(row); logits.append(z)
        print_block(f'  s{s}', row['heldout'])
        extra = f"  domain_head {row['domain_head']['heldout']}" if 'domain_head' in row else ''
        print(f"  s{s} loss {row['final_loss']:.4f} {row['secs']}s ({row['sec_per_step']} s/step, {row['steps']} steps) {row['params']}p gpu_peak {row.get('gpu_peak_gb', 'n/a')} GB "
              f"(reserved {row.get('gpu_reserved_peak_gb', 'n/a')}){extra}", flush=True)
        if not a.no_save:
            p = f'{a.out}/{tag}_s{s}.pt'; torch.save(checkpoint_dict(model, D, a, s, tag), p)
            if a.roundtrip_check:
                row['roundtrip'] = roundtrip(p, D, z, a.x_half); print(f"  s{s} roundtrip {row['roundtrip']}", flush=True)
    ens = np.mean(logits, 0)
    np.savez_compressed(f'{a.out}/{tag}_logits.npz', id=D.d['id'], ensemble_logit=ens, member_logits=np.stack(logits), fit=D.fit, dev=D.dev, heldout=D.test,
                        domain=D.d['domain'], anchor_frame=D.d['anchor_frame'], group=D.d['group'], source=D.d['source'], unsafe=D.d['unsafe'], fail=D.d['fail'],
                        file_index=D.d['file_index'])
    summary = dict(tag=tag, args=vars(a), files=D.files, dropped_keys=D.dropped_keys, n_rows_used=int(D.n), n_fit=int(D.fit.sum()), fit_by_domain=dom_counts,
                   fit_by_source=src_counts, data_frac=D.data_frac, split_hash=D.split_hash, load_secs=D.load_secs, gpu_data_gb=gpu_data, members=rows,
                   ensemble=dict(dev=all_metrics(ens, D.d, D.dev), heldout=all_metrics(ens, D.d, D.test)))
    json.dump(summary, open(f'{a.out}/{tag}.json', 'w'), indent=1, default=float)
    print_block(f'ENSEMBLE {tag}', summary['ensemble']['heldout'])


if __name__ == '__main__':
    main()
