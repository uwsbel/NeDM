"""Route-risk network of the traversing planners: a CNN-GRU that turns a route corridor into one hazard per station.

A station-preserving CNN reads the 6 x 96 x 32 route corridor (standardised elevation, grade, cross slope, speed,
valid mask and a ones plane) and keeps one feature column per station (mean and max over the lateral cells). Each
column is joined with a 32-number context embedding (route geometry, plus the 16-number history code z for
``cond='hist_aux'``) and the station position, mixed by Conv1d(k=5) and a bidirectional GRU, and read out as one
hazard logit per station. Route risk = 1 - exp(-sum softplus(hazard)); ``route_logit`` returns log(sum softplus).

The history encoder is a causal GRU over the last 2 s of observable state and applied controls (40 frames x 15
channels, invalid frames zeroed, the mask appended as a 16th channel) -> tanh(Linear) -> z. With ``cond='hist_aux'``
a linear head on z predicts the ground type (rigid 0 / soil 1) as an auxiliary loss.

Checkpoints are the ``model_kind='ci_train'`` files written by train_risk_model.py (and by the experiment trainer,
so the released ``deploy_a1_haux_gru_s*.pt`` and the rigid/soil/vehicle ``*_deploy_s*.pt`` ensembles load as is):

    model, ck = load_risk_model('deploy_a1_haux_gru_s0.pt', 'cuda')
    z = encode_history(model, ck, hist, hmask)               # once per decision (None for cond 'none')
    logits = score(model, ck, X, geom5, z=z)                  # (n,) route logits for n candidate corridors
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GEOM_COLS = [17, 18, 19, 20, 21]                               # ctx: goal dx, goal dy, |goal|, start yaw, route length
HIST_STATE_COLS = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]    # observable vehicle-state columns of the history
HIST_ACTION_COLS = [0, 1, 2]                                   # steer, throttle, brake
HIST_COLS = HIST_STATE_COLS + HIST_ACTION_COLS                 # 15 history channels
DOMAIN_NAME = {0: 'rigid', 1: 'crm'}
DOMAIN_VOCAB = {'rigid': 0, 'crm': 1}
CONDS = ('none', 'hist_aux')
ZDIM, WIDTH = 16, 64


def route_logit(haz: torch.Tensor) -> torch.Tensor:
    """log cumulative hazard == cloglog(P(event anywhere on the route))."""
    return torch.log(F.softplus(haz).sum(1) + 1e-6)


def survival_nll(haz: torch.Tensor, ev: torch.Tensor) -> torch.Tensor:
    """Discrete-time survival loss per row: stations before the event (all stations if ev < 0) survive, the event
    station fails."""
    S = haz.shape[1]
    idx = torch.arange(S, device=haz.device)[None, :]
    evc = torch.where(ev >= 0, ev, torch.full_like(ev, S - 1))[:, None]
    surv = (idx < evc) | ((ev < 0)[:, None] & (idx <= evc))
    nll = (F.softplus(haz) * surv).sum(1)
    return nll + torch.where(ev >= 0, F.softplus(-haz.gather(1, evc).squeeze(1)), torch.zeros_like(nll))


class CNNFront(nn.Module):
    """Station-preserving CNN: (B, cin, 96, 32) -> (B, 96, 192) [mean, max over the 4 lateral cells]."""

    def __init__(self, cin: int):
        super().__init__()
        c = [cin, 32, 64, 64, 96]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(c[i], c[i + 1], 3, stride=(1, 1 if i == 0 else 2), padding=1),
                       nn.BatchNorm2d(c[i + 1]), nn.GELU()]
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        f = self.cnn(x)
        return torch.cat([f.mean(-1), f.amax(-1)], 1).transpose(1, 2)


class RiskModel(nn.Module):
    """forward(x (B,6,96,32) standardised, ctx (B,nctx), hist (B,T,15) standardised, hmask (B,T), z=None)
    -> {'haz': (B,96)[, 'z', 'dom']}.  A precomputed z (B,zdim) skips the history encoder.

    Module names, shapes and construction order are those of the experiment trainer, so a seed gives the same
    initial weights and the state-dict keys match the released checkpoints."""

    def __init__(self, cond: str, cin: int, nctx: int, zdim: int = ZDIM, hist_dh: int = len(HIST_COLS),
                 width: int = WIDTH):
        super().__init__()
        assert cond in CONDS, cond
        self.cond = cond
        self.use_hist = cond == 'hist_aux'
        self.zdim = int(zdim) if self.use_hist else 0
        self.front = CNNFront(cin)
        self.lat = nn.Linear(192, 96)
        self.ctx = nn.Sequential(nn.Linear(nctx + self.zdim, 32), nn.GELU())
        self.tconv = nn.Sequential(nn.Conv1d(129, 96, 5, padding=2), nn.GELU(), nn.Dropout(0.1))
        self.mix = nn.GRU(96, width, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * width, 1)
        if self.use_hist:
            self.henc = nn.GRU(hist_dh + 1, 32, batch_first=True)   # causal; the mask is the last input channel
            self.hz = nn.Linear(32, self.zdim)
            self.dom = nn.Linear(self.zdim, 1)                       # auxiliary ground-type head on z

    def encode(self, hist, hmask):
        """z (B, zdim): masked frames zeroed, mask appended as a channel, tanh of the final hidden state."""
        m = hmask.to(hist.dtype)[..., None]
        _, hn = self.henc(torch.cat([hist * m, m], -1))
        return torch.tanh(self.hz(hn[-1]))

    def backbone(self, x, ctx_full):
        f = F.gelu(self.lat(self.front(x)))
        B, S, _ = f.shape
        pos = torch.linspace(0, 1, S, device=x.device)[None, :, None].expand(B, S, 1)
        c = self.ctx(ctx_full)[:, None, :].expand(B, S, 32)
        h = self.tconv(torch.cat([f, c, pos], -1).transpose(1, 2)).transpose(1, 2)
        return self.head(self.mix(h)[0]).squeeze(-1)

    def forward(self, x, ctx, hist=None, hmask=None, z=None):
        out = {}
        if self.use_hist:
            if z is None:
                z = self.encode(hist, hmask)
            ctx = torch.cat([ctx, z], -1)
            out['z'] = z
        out['haz'] = self.backbone(x, ctx)
        if self.use_hist:
            out['dom'] = self.dom(z).squeeze(-1)
        return out


# ----------------------------------------------------------------------------- input preparation
def prep_x(x_raw, norm):
    """Raw corridor (b,5,96,32) f32 tensor -> standardised channels + ones plane (b,6,96,32) f32."""
    mu = torch.as_tensor(np.asarray(norm['mu'], np.float32), device=x_raw.device)[None, :, None, None]
    sd = torch.as_tensor(np.asarray(norm['sd'], np.float32), device=x_raw.device)[None, :, None, None]
    cont = list(norm['cont_index'])
    y = torch.ones((x_raw.shape[0], x_raw.shape[1] + 1, 96, 32), dtype=torch.float32, device=x_raw.device)
    y[:, :x_raw.shape[1]] = x_raw
    y[:, cont] = (y[:, cont] - mu) / sd
    return y


def prep_hist(hist, hmask, mu, sd):
    """Raw history (b,T,15) -> standardised f32 with masked frames zeroed (NaN-safe), and the bool mask."""
    h = torch.as_tensor(hist).to(torch.float32)
    m = torch.as_tensor(hmask).to(torch.bool).to(h.device)
    mu = torch.as_tensor(np.asarray(mu, np.float32), device=h.device)
    sd = torch.as_tensor(np.asarray(sd, np.float32), device=h.device)
    h = torch.where(m[..., None], (torch.nan_to_num(h) - mu) / sd, torch.zeros_like(h))
    return h, m


def cut_window(hist, hmask, T):
    """Raw window (T0,15) or (m,T0,15) + mask -> the newest T frames (m,T,15) f32, (m,T) bool (left-padded, masked)."""
    H = np.asarray(hist, np.float32)
    M = np.asarray(hmask, bool)
    if H.ndim == 2:
        H, M = H[None], M[None]
    assert H.shape[:2] == M.shape and H.shape[2] == 15, (H.shape, M.shape)
    if H.shape[1] >= T:
        return H[:, H.shape[1] - T:], M[:, M.shape[1] - T:]
    pad = T - H.shape[1]
    return (np.concatenate([np.zeros((len(H), pad, 15), np.float32), H], 1),
            np.concatenate([np.zeros((len(M), pad), bool), M], 1))


# ----------------------------------------------------------------------------- checkpoint API
def load_risk_model(path, device='cpu'):
    """Rebuild a checkpoint: (model in eval mode on device, checkpoint dict)."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck.get('model_kind') == 'ci_train', f'{path}: model_kind {ck.get("model_kind")!r} is not ci_train'
    assert ck['arch'] == 'gru' and ck['hist_enc'] == 'gru' and ck['ctx_mode'] == 'geom' and ck['cond'] in CONDS, \
        f'{path}: only the CNN-GRU with ctx geom and cond none | hist_aux is supported'
    assert ck.get('hist_valid_T', ck['hist_T']) == ck['hist_T'], f'{path}: masked-window checkpoints are not supported'
    model = RiskModel(ck['cond'], ck['cin'], ck['nctx'], zdim=ck['zdim'] or ZDIM, hist_dh=len(ck['hist_cols']),
                      width=ck.get('width', WIDTH))
    model.load_state_dict(ck['state'])
    model.to(device).eval()
    return model, ck


def encode_history(model, ck, hist, hmask, device=None):
    """z (m, zdim) f32 for raw windows hist (T0,15) | (m,T0,15) + hmask; hist=None -> the all-masked startup window.
    Returns None for models without history."""
    if not model.use_hist:
        return None
    device = device or next(model.parameters()).device
    T = ck['hist_T']
    if hist is None:
        H, M = np.zeros((1, T, 15), np.float32), np.zeros((1, T), bool)
    else:
        H, M = cut_window(hist, hmask, T)
    h, m = prep_hist(torch.from_numpy(H).to(device), torch.from_numpy(M).to(device), ck['hist_mu'], ck['hist_sd'])
    with torch.no_grad():
        z = model.encode(h, m)
    return z.float().cpu().numpy()


def score(model, ck, X, geom5, hist=None, hmask=None, z=None, bs=1024):
    """Route logits (n,) f32 for raw inputs: X (n,5,96,32) corridors, geom5 (n|1,5) raw geometry (ctx cols 17-21),
    hist (T0,15) | (n|1,T0,15) raw window + hmask (None = all-masked startup window; one window is shared by every
    candidate; the newest hist_T frames are used), z = encode_history(...) output to skip the history encoder."""
    device = next(model.parameters()).device
    n = len(X)
    model.eval()
    geom = np.broadcast_to(np.asarray(geom5, np.float32).reshape(-1, 5), (n, 5))
    H = M = None
    if hist is not None:
        H, M = cut_window(hist, hmask, ck['hist_T'])
    ctx = (geom - ck['ctx_mu']) / ck['ctx_sd']
    ctx_t = torch.from_numpy(np.ascontiguousarray(ctx, dtype=np.float32)).to(device)
    zt = None
    per_row = model.use_hist and z is None and H is not None and len(H) > 1   # per-row windows: encoded per chunk
    if model.use_hist and not per_row:
        if z is None:
            z = encode_history(model, ck, H, M, device)
        zt = torch.as_tensor(np.asarray(z, np.float32)).to(device)
        if zt.shape[0] == 1 and n > 1:
            zt = zt.expand(n, -1)
    out = []
    with torch.no_grad():
        for i in range(0, n, bs):
            x = prep_x(torch.as_tensor(np.asarray(X[i:i + bs], np.float32)).to(device), ck['norm'])
            if per_row:   # same chunking as the trainer's predict(): equal rows in equal batches give equal numbers
                h, m = prep_hist(torch.from_numpy(np.ascontiguousarray(H[i:i + bs])).to(device),
                                 torch.from_numpy(np.ascontiguousarray(M[i:i + bs])).to(device),
                                 ck['hist_mu'], ck['hist_sd'])
                zb = model.encode(h, m)
            else:
                zb = None if zt is None else zt[i:i + bs]
            o = model(x, ctx_t[i:i + bs], z=zb)
            out.append(route_logit(o['haz']).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)
