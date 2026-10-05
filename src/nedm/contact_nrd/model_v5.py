"""Unified contact NRD, version 5: one network design for the core, the collision network and the contact network.

All three networks are the same module (TokenNet) with the same hyperparameters for every system; only the
data differs. Inputs are state-only: the moving bodies' last K states (the current state x(t) and the step
differences d_k = x(t-k+1) - x(t-k), k = 1..K-1) plus learned identity codes for the query (the core, or one of
the pairs) and for a fixed partner body. No geometry, gravity, radii, pair frames or relative features.

Role-ordered positions of one evaluation:
    core               [self]
    collision/contact  [self (the pair's first body), moving partner (absent for a fixed partner), other moving bodies]
Missing history (episode start, history dropout) is masked out of attention, never filled with copies.

Mixers (the only architectural choice):
    mlp        all positions and lags flattened -> pre-LN residual MLP (no attention; the control)
    time       one token per lag (all positions concatenated) + fixed + query tokens -> Transformer encoder
    entity     one token per position (its K lags through a small MLP) + fixed + query tokens -> Transformer encoder
    spacetime  one token per (position, lag) + fixed + query tokens -> Transformer encoder
The query token's output goes through a zero-initialised linear head. A per-query ridge-fitted linear map of the
current states (FP64) is added, then the per-query output scale and mean. The trunk can run in float32; state
bookkeeping stays float64.

    s' = s + core(own history) + sum over pairs of 1[collision >= 0] * contact(...)
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

STATE = 9
ROLES = 3   # self, partner, other


class Encoder(nn.Module):
    """Pre-LN Transformer encoder with a key mask (no causal mask)."""

    def __init__(self, d, heads, ffn, layers):
        super().__init__()
        self.heads = heads
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict(dict(ln1=nn.LayerNorm(d), qkv=nn.Linear(d, 3 * d), proj=nn.Linear(d, d),
                                                  ln2=nn.LayerNorm(d), fc1=nn.Linear(d, ffn), fc2=nn.Linear(ffn, d))))
        self.ln = nn.LayerNorm(d)
        for blk in self.blocks:
            for name in ("qkv", "proj", "fc1", "fc2"):
                nn.init.normal_(blk[name].weight, std=0.02 / (math.sqrt(2 * layers) if name in ("proj", "fc2") else 1.0))
                nn.init.zeros_(blk[name].bias)

    def forward(self, x, ok):
        """x [M, T, d], ok [M, T] (True = may be attended to) -> [M, T, d]."""
        m, t, d = x.shape
        mask = ok[:, None, None, :]
        for blk in self.blocks:
            q, k, v = blk["qkv"](blk["ln1"](x)).split(d, -1)
            q, k, v = (z.view(m, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            x = x + blk["proj"](y.transpose(1, 2).reshape(m, t, d))
            x = x + blk["fc2"](F.gelu(blk["fc1"](blk["ln2"](x))))
        return self.ln(x)


class ResMLP(nn.Module):
    """Pre-LN residual MLP on one vector per evaluation."""

    def __init__(self, d, ffn, layers):
        super().__init__()
        self.blocks = nn.ModuleList([nn.ModuleDict(dict(ln=nn.LayerNorm(d), fc1=nn.Linear(d, ffn), fc2=nn.Linear(ffn, d)))
                                     for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        for blk in self.blocks:
            nn.init.normal_(blk["fc1"].weight, std=0.02)
            nn.init.normal_(blk["fc2"].weight, std=0.02 / math.sqrt(2 * layers))
            nn.init.zeros_(blk["fc1"].bias)
            nn.init.zeros_(blk["fc2"].bias)

    def forward(self, x):
        for blk in self.blocks:
            x = x + blk["fc2"](F.gelu(blk["fc1"](blk["ln"](x))))
        return self.ln(x)


class TokenNet(nn.Module):
    """One network: role-ordered state histories + identity codes -> one output vector per evaluation."""

    def __init__(self, spec, positions, history, queries, fixed, out_dim, linear_path, trunk_dtype):
        super().__init__()
        d, self.mixer = int(spec["width"]), spec["mixer"]
        self.P, self.K, self.Q, self.out_dim, self.linear_path = positions, history, queries, out_dim, linear_path
        self.trunk_dtype = trunk_dtype
        self.e_query = nn.Embedding(queries, d)
        self.e_fixed = nn.Embedding(max(1, fixed), d)
        self.e_role = nn.Embedding(ROLES, d)
        self.e_lag = nn.Embedding(history, d)
        heads, ffn, layers = int(spec["heads"]), int(spec["ffn"]), int(spec["layers"])
        if self.mixer == "mlp":
            self.inp = nn.Linear(positions * history * (STATE + 1), d)
            self.trunk = ResMLP(d, ffn, layers)
        else:
            if self.mixer == "spacetime":
                self.cur_in, self.diff_in = nn.Linear(STATE, d), nn.Linear(STATE, d)
            elif self.mixer == "time":
                self.cur_in, self.diff_in = nn.Linear(positions * (STATE + 1), d), nn.Linear(positions * (STATE + 1), d)
            elif self.mixer == "entity":
                self.ent_in = nn.Sequential(nn.Linear(history * (STATE + 1), d), nn.GELU(), nn.Linear(d, d))
            else:
                raise ValueError(f"unknown mixer {self.mixer}")
            self.trunk = Encoder(d, heads, ffn, layers)
        for emb in (self.e_query, self.e_fixed, self.e_role, self.e_lag):
            nn.init.normal_(emb.weight, std=0.02)
        self.head = nn.Linear(d, out_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Data-fitted maps (set by the trainer, stored in the checkpoint).
        self.register_buffer("in_mean", torch.zeros(queries, positions, STATE, dtype=torch.float64))
        self.register_buffer("in_std", torch.ones(queries, positions, STATE, dtype=torch.float64))
        self.register_buffer("diff_std", torch.ones(queries, positions, STATE, dtype=torch.float64))
        self.register_buffer("out_mean", torch.zeros(queries, out_dim, dtype=torch.float64))
        self.register_buffer("out_std", torch.ones(queries, out_dim, dtype=torch.float64))
        self.lin = nn.Parameter(torch.zeros(queries, positions * STATE + 1, out_dim, dtype=torch.float64), requires_grad=linear_path)
        self.trunk.to(trunk_dtype)
        for mod in (self.e_query, self.e_fixed, self.e_role, self.e_lag, self.head):
            mod.to(trunk_dtype)
        for name in ("inp", "cur_in", "diff_in", "ent_in"):
            if hasattr(self, name):
                getattr(self, name).to(trunk_dtype)

    def scaled(self, cur, diff, present, lag_ok, query, mask):
        """Raw role-ordered inputs -> standardised inputs (float64), absent positions and missing lags zeroed."""
        zc = (cur - self.in_mean[query]) / self.in_std[query] * mask
        zd = diff / self.diff_std[query][:, None] * mask
        zc = zc * present[..., None]
        zd = zd * present[:, None, :, None] * lag_ok[:, 1:, None, None]
        return zc, zd

    def forward(self, cur, diff, present, lag_ok, fixed, query, mask):
        """cur [M, P, 9], diff [M, K-1, P, 9] (index k-1 = lag k), present [M, P], lag_ok [M, K] (lag 0 first),
        fixed [M] (-1 = none), query [M] -> [M, out] in physical units (float64)."""
        zc, zd = self.scaled(cur, diff, present, lag_ok, query, mask)
        td = self.trunk_dtype
        m = len(cur)
        pf = present.to(td)
        if self.mixer == "mlp":
            lags = torch.cat((zc[:, None], zd), 1).to(td)                                        # [M, K, P, 9]
            flags = (present[:, None, :] & lag_ok[:, :, None]).to(td)                          # [M, K, P]
            x = self.inp(torch.cat((lags, flags[..., None]), -1).flatten(1)) + self.e_query(query)
            x = x + self.e_fixed(fixed.clamp_min(0)) * (fixed >= 0)[:, None].to(td)
            h = self.trunk(x)
        else:
            q_tok = self.e_query(query)[:, None]
            f_tok = (self.e_fixed(fixed.clamp_min(0)) + self.e_role.weight[1])[:, None]
            ok_qf = torch.stack((torch.ones_like(fixed, dtype=torch.bool), fixed >= 0), 1)
            role = torch.arange(self.P, device=cur.device).clamp_max(ROLES - 1)
            if self.mixer == "spacetime":
                toks = torch.cat((self.cur_in(zc.to(td))[:, None], self.diff_in(zd.to(td))), 1)          # [M, K, P, d]
                toks = toks + self.e_role(role)[None, None] + self.e_lag.weight[None, :, None]
                ok = (lag_ok[:, :, None] & present[:, None, :]).flatten(1)
                toks = toks.flatten(1, 2)
            elif self.mixer == "time":
                cur_t = torch.cat((zc.to(td), pf[..., None]), -1).flatten(1)                             # [M, P*10]
                dif_t = torch.cat((zd.to(td), pf[:, None, :, None].expand(-1, self.K - 1, -1, -1)), -1).flatten(2)
                toks = torch.cat((self.cur_in(cur_t)[:, None], self.diff_in(dif_t)), 1) + self.e_lag.weight[None]
                ok = lag_ok
            else:   # entity
                lags = torch.cat((zc[:, None], zd), 1).to(td)                                            # [M, K, P, 9]
                x = torch.cat((lags, lag_ok[:, :, None, None].expand(-1, -1, self.P, 1).to(td)), -1)     # [M, K, P, 10]
                toks = self.ent_in(x.permute(0, 2, 1, 3).flatten(2)) + self.e_role(role)[None]
                ok = present
            seq = torch.cat((q_tok, f_tok, toks), 1)
            h = self.trunk(seq, torch.cat((ok_qf, ok), 1))[:, 0]
        out = self.head(h).to(torch.float64)
        if self.linear_path:
            design = torch.cat((zc.flatten(1), torch.ones(m, 1, dtype=zc.dtype, device=zc.device)), -1)
            out = out + torch.einsum("mf,mfo->mo", design, self.lin[query])
        return out * self.out_std[query] + self.out_mean[query]


class UnifiedNRD(nn.Module):
    def __init__(self, config, system, mask):
        super().__init__()
        self.config, self.system = dict(config), system
        self.dt = float(config["dt_s"])
        self.k = int(config["history"])
        bodies = system["bodies"]
        self.moving = [k for k, b in enumerate(bodies) if b["moving"]]
        fixed_bodies = [k for k, b in enumerate(bodies) if not b["moving"]]
        slot = {k: n for n, k in enumerate(self.moving)}
        fidx = {k: n for n, k in enumerate(fixed_bodies)}
        pairs = system["pairs"]
        self.n_bodies, self.n_pairs, self.n_fixed = len(self.moving), len(pairs), len(fixed_bodies)
        positions = self.n_bodies + 1
        pos = torch.full((self.n_pairs, positions), -1, dtype=torch.long)
        pfix = torch.full((self.n_pairs,), -1, dtype=torch.long)
        for p, (i, j) in enumerate(pairs):
            pos[p, 0] = slot[i]
            if j in slot:
                pos[p, 1] = slot[j]
            else:
                pfix[p] = fidx[j]
            others = [s for s in range(self.n_bodies) if s not in (slot[i], slot.get(j, -1))]
            for n, s in enumerate(others):
                pos[p, 2 + n] = s
        self.register_buffer("pair_pos", pos)
        self.register_buffer("pair_fixed", pfix)
        self.register_buffer("pair_mm", pos[:, 1] >= 0)
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float64))
        spec = config["network"]
        td = torch.float32 if config.get("trunk_dtype", "float32") == "float32" else torch.float64
        self.core = TokenNet(spec, 1, self.k, 1, self.n_fixed, STATE, True, td)
        self.collision = TokenNet(spec, positions, self.k, self.n_pairs, self.n_fixed, 1, False, td)
        self.contact = TokenNet(spec, positions, self.k, self.n_pairs, self.n_fixed, 2 * STATE, True, td)

    # ---- history ---------------------------------------------------------------
    def code(self, hist, ok, noise=None):
        """hist [B, K, D, 9] (newest last), ok [B, K] -> cur [B, D, 9], diff [B, K-1, D, 9] (index k-1 = lag k),
        lag_ok [B, K] (lag 0 first). `noise` [9] adds i.i.d. Gaussian noise to every difference (a random walk
        on the past states)."""
        cur = hist[:, -1]
        diff = (hist[:, 1:] - hist[:, :-1]).flip(1)
        lag_ok = ok.flip(1)
        if noise is not None and self.k > 1:
            diff = diff + torch.randn_like(diff) * noise
        return cur, diff, lag_ok

    @staticmethod
    def take(x, pos):
        """x [B, ..., D, 9], pos [Q, P] (slot or -1) -> [B, Q, ..., P, 9] with zeros where pos < 0."""
        pad = torch.cat((x, torch.zeros_like(x[..., :1, :])), -2)
        d = x.shape[-2]
        g = pad[..., torch.where(pos < 0, d, pos), :]          # [B, ..., Q, P, 9]
        return g.movedim(-3, 1)

    def run_core(self, cur, diff, lag_ok, bodies=None):
        """Smooth change of every moving body (or the given slots) -> [B, D, 9]."""
        b, d = cur.shape[:2]
        pos = torch.arange(d, device=cur.device)[:, None]
        c = self.take(cur, pos).reshape(b * d, 1, STATE)
        df = self.take(diff, pos).reshape(b * d, self.k - 1, 1, STATE)
        lo = lag_ok.repeat_interleave(d, 0)
        present = torch.ones(b * d, 1, dtype=torch.bool, device=cur.device)
        fixed = torch.full((b * d,), -1, dtype=torch.long, device=cur.device)
        query = torch.zeros(b * d, dtype=torch.long, device=cur.device)
        return self.core(c, df, present, lo, fixed, query, self.mask).reshape(b, d, STATE) * self.mask

    def pair_inputs(self, cur, diff, lag_ok, rows, pairs):
        """Inputs of the pair networks for (sample, pair) entries `rows`, `pairs` (both [M])."""
        pos = self.pair_pos[pairs]                                                    # [M, P]
        padc = torch.cat((cur, torch.zeros_like(cur[:, :1])), 1)
        padd = torch.cat((diff, torch.zeros_like(diff[:, :, :1])), 2)
        idx = torch.where(pos < 0, cur.shape[1], pos)
        c = padc[rows[:, None], idx]                                                  # [M, P, 9]
        df = padd[rows[:, None, None], torch.arange(self.k - 1, device=cur.device)[None, :, None], idx[:, None, :]]
        return c, df, pos >= 0, lag_ok[rows], self.pair_fixed[pairs], pairs

    def logits(self, cur, diff, lag_ok, rows=None, pairs=None):
        """Collision logits: all pairs -> [B, P_pairs], or the given (row, pair) entries -> [M]."""
        b = cur.shape[0]
        full = rows is None
        if full:
            rows = torch.arange(b, device=cur.device).repeat_interleave(self.n_pairs)
            pairs = torch.arange(self.n_pairs, device=cur.device).repeat(b)
        out = self.collision(*self.pair_inputs(cur, diff, lag_ok, rows, pairs), self.mask)[:, 0]
        return out.reshape(b, self.n_pairs) if full else out

    def contact_sum(self, cur, diff, lag_ok, gates):
        """Summed contact change [B, D, 9]; only (sample, pair) entries with gate != 0 are evaluated."""
        total = torch.zeros_like(cur)
        idx = torch.nonzero(gates != 0)
        if len(idx):
            rows, pairs = idx[:, 0], idx[:, 1]
            out = self.contact(*self.pair_inputs(cur, diff, lag_ok, rows, pairs), self.mask).reshape(-1, 2, STATE)
            g = gates[rows, pairs][:, None]
            total = total.index_put((rows, self.pair_pos[pairs, 0]), g * out[:, 0], accumulate=True)
            mm = self.pair_mm[pairs]
            if mm.any():
                total = total.index_put((rows[mm], self.pair_pos[pairs[mm], 1]), g[mm] * out[mm, 1], accumulate=True)
        return total * self.mask

    def details(self, hist, ok, gates=None, noise=None):
        cur, diff, lag_ok = self.code(hist, ok, noise)
        d_core = self.run_core(cur, diff, lag_ok)
        logits = self.logits(cur, diff, lag_ok)
        g = (logits >= 0).to(cur.dtype) if gates is None else gates.to(cur.dtype)
        nxt = cur + d_core + self.contact_sum(cur, diff, lag_ok, g)
        return nxt, {"logits": logits, "gates": g, "core": d_core}

    def step_with_gates(self, hist, ok, gates, noise=None, core=None):
        """Next state with given switches (training); the core change is computed without gradient unless given."""
        cur, diff, lag_ok = self.code(hist, ok, noise)
        if core is None:
            with torch.no_grad():
                core = self.run_core(cur, diff, lag_ok)
        return cur + core + self.contact_sum(cur, diff, lag_ok, gates.to(cur.dtype))

    @staticmethod
    def advance(hist, ok, state):
        return torch.cat((hist[:, 1:], state[:, None]), 1), torch.cat((ok[:, 1:], torch.ones_like(ok[:, :1])), 1)

    def start(self, state):
        hist = torch.zeros(state.shape[0], self.k, *state.shape[1:], dtype=state.dtype, device=state.device)
        hist[:, -1] = state
        ok = torch.zeros(state.shape[0], self.k, dtype=torch.bool, device=state.device)
        ok[:, -1] = True
        return hist, ok

    def forward(self, hist, ok):
        return self.details(hist, ok)[0]

    def rollout(self, initial, steps, return_gates=False):
        state = initial.to(torch.float64)
        hist, ok = self.start(state)
        trajectory, gate_trace = [state], []
        for _ in range(steps):
            state, info = self.details(hist, ok)
            hist, ok = self.advance(hist, ok, state)
            trajectory.append(state)
            gate_trace.append(info["gates"])
        out = torch.stack(trajectory, -3)
        return (out, torch.stack(gate_trace, -2)) if return_gates else out


def load_v5(path, device="cpu"):
    packet = torch.load(path, map_location=device, weights_only=False)
    model = UnifiedNRD(packet["model_config"], packet["system"], packet["mask"]).to(device)
    model.load_state_dict(packet["model_state_dict"])
    return model.eval(), packet
