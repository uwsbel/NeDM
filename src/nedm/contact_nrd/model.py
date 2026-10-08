"""Contact NRD: a core network, a collision network and a contact network, all the same Transformer design.

Every moving body has its own state size (the ball and a pool ball [position, velocity, angular velocity] = 9,
the arm [joint angles, joint rates] = 10, the T [position, quaternion, velocity, angular velocity] = 13). States
are padded to the largest size, with a per-body channel mask. A body can have an action (the arm's joint position
command), which enters as one extra token of that body. Each body kind has its own input maps, and a learned
body-kind code is added to its tokens; the core has one output head per body kind. Fixed bodies (floor, wall, cushions) are not inputs; a fixed
partner enters only as a learned identity code.

Tokens of one evaluation: [query, fixed partner, (position p, lag k) for the role-ordered bodies, action tokens].
Roles: self (the pair's first body / the core's body), partner (moving partner), other (the remaining moving
bodies). Lag 0 is the current state, lag k the step difference x(t-k+1) - x(t-k). Missing history is masked out
of attention.

    s' = s + core(own history, own action) + sum over pairs of 1[collision >= 0] * contact(...)

The pose of each integrated body kind follows its rates: pose' = pose + step x rate' (the rates are step-average
rates, see data.step_average_rates); quaternions are re-normalised after each step.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

ROLES = 3   # self, partner, other


def model_config(dt_s, history, network, integrate_kinds):
    """The model_config stored in a checkpoint. Only the step, the history length, the network size and the
    integrated body kinds can change; the other entries are the design this code implements, written in the
    format of the released checkpoints ("contact_graph_nrd_v6" is that format's stored name: keep it)."""
    return {"architecture": "contact_graph_nrd_v6", "dt_s": dt_s, "history": history,
            "network": {"mixer": "spacetime", **network}, "trunk_dtype": "float32", "gate_mode": "learned",
            "integrate": {"kinds": list(integrate_kinds), "rule": "new", "position_correction": False, "noise": "velocity"},
            "core_no_pose": None}


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


class TokenNet(nn.Module):
    """One network over role-ordered, kind-typed body histories and actions -> one output per evaluation.

    The query token's output goes through zero-initialised linear heads (one per output part). With
    ``linear_path`` a per-query ridge-fitted linear map of the standardised current states and actions (float64)
    is added; then the per-query output scale and mean. The trunk runs in float32, the bookkeeping in float64.
    """

    def __init__(self, spec, positions, history, queries, fixed, kinds, smax, amax, out_dims, linear_path):
        super().__init__()
        d = int(spec["width"])
        self.P, self.linear_path = positions, linear_path
        self.e_query = nn.Embedding(queries, d)
        self.e_fixed = nn.Embedding(max(1, fixed), d)
        self.e_role = nn.Embedding(ROLES, d)
        self.e_lag = nn.Embedding(history, d)
        self.e_kind = nn.Embedding(kinds, d)
        self.e_action = nn.Parameter(torch.zeros(d))
        self.cur_in = nn.ModuleList([nn.Linear(smax, d) for _ in range(kinds)])
        self.diff_in = nn.ModuleList([nn.Linear(smax, d) for _ in range(kinds)])
        self.act_in = nn.ModuleList([nn.Linear(max(1, amax), d) for _ in range(kinds)])
        self.trunk = Encoder(d, int(spec["heads"]), int(spec["ffn"]), int(spec["layers"]))
        self.heads = nn.ModuleList([nn.Linear(d, n) for n in out_dims])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for emb in (self.e_query, self.e_fixed, self.e_role, self.e_lag, self.e_kind):
            nn.init.normal_(emb.weight, std=0.02)
        out = sum(out_dims)
        f = positions * smax + positions * max(1, amax) + 1
        # Data-fitted maps (set by the trainer, stored in the checkpoint).
        self.register_buffer("in_mean", torch.zeros(queries, positions, smax, dtype=torch.float64))
        self.register_buffer("in_std", torch.ones(queries, positions, smax, dtype=torch.float64))
        self.register_buffer("diff_std", torch.ones(queries, positions, smax, dtype=torch.float64))
        self.register_buffer("act_mean", torch.zeros(queries, positions, max(1, amax), dtype=torch.float64))
        self.register_buffer("act_std", torch.ones(queries, positions, max(1, amax), dtype=torch.float64))
        self.register_buffer("out_mean", torch.zeros(queries, out, dtype=torch.float64))
        self.register_buffer("out_std", torch.ones(queries, out, dtype=torch.float64))
        self.lin = nn.Parameter(torch.zeros(queries, f, out, dtype=torch.float64), requires_grad=linear_path)

    def scaled(self, cur, diff, act, present, has_act, lag_ok, query, smask, amask):
        """Standardised inputs (float64); absent positions, missing lags and masked channels zeroed."""
        zc = (cur - self.in_mean[query]) / self.in_std[query] * smask
        zd = diff / self.diff_std[query][:, None] * smask[:, None]
        za = (act - self.act_mean[query]) / self.act_std[query] * amask
        zc = zc * present[..., None]
        zd = zd * present[:, None, :, None] * lag_ok[:, 1:, None, None]
        za = za * (present & has_act)[..., None]
        return zc, zd, za

    def by_kind(self, maps, x, kind):
        """Apply the kind-specific linear map to x [M, ..., P, n] with kind [M, P] -> [M, ..., P, d]."""
        out = None
        for k, lin in enumerate(maps):
            y = lin(x) * (kind == k).to(x.dtype).view(kind.shape[0], *([1] * (x.dim() - 3)), kind.shape[1], 1)
            out = y if out is None else out + y
        return out

    def forward(self, cur, diff, act, present, has_act, kind, lag_ok, fixed, query, smask, amask):
        """cur [M, P, S], diff [M, K-1, P, S], act [M, P, A], present/has_act [M, P], kind [M, P], lag_ok [M, K],
        fixed [M] (-1 = none), query [M], smask [M, P, S], amask [M, P, A] -> [M, sum(out_dims)] in physical units."""
        zc, zd, za = self.scaled(cur, diff, act, present, has_act, lag_ok, query, smask, amask)
        td = torch.float32
        m = len(cur)
        role = torch.arange(self.P, device=cur.device).clamp_max(ROLES - 1)
        kemb = self.e_kind(kind)                                                                   # [M, P, d]
        toks = torch.cat((self.by_kind(self.cur_in, zc.to(td), kind)[:, None],
                          self.by_kind(self.diff_in, zd.to(td), kind)), 1)                        # [M, K, P, d]
        toks = toks + self.e_role(role)[None, None] + self.e_lag.weight[None, :, None] + kemb[:, None]
        ok = (lag_ok[:, :, None] & present[:, None, :]).flatten(1)
        toks = toks.flatten(1, 2)
        a_tok = self.by_kind(self.act_in, za.to(td), kind) + self.e_role(role)[None] + kemb + self.e_action
        a_ok = present & has_act
        q_tok = self.e_query(query)[:, None]
        f_tok = (self.e_fixed(fixed.clamp_min(0)) + self.e_role.weight[1])[:, None]
        ok_qf = torch.stack((torch.ones_like(fixed, dtype=torch.bool), fixed >= 0), 1)
        seq = torch.cat((q_tok, f_tok, toks, a_tok), 1)
        h = self.trunk(seq, torch.cat((ok_qf, ok, a_ok), 1))[:, 0]
        out = torch.cat([head(h) for head in self.heads], -1).to(torch.float64)
        if self.linear_path:
            design = torch.cat((zc.flatten(1), za.flatten(1), torch.ones(m, 1, dtype=zc.dtype, device=zc.device)), -1)
            out = out + torch.einsum("mf,mfo->mo", design, self.lin[query])
        return out * self.out_std[query] + self.out_mean[query]


class ContactNRD(nn.Module):
    """Bodies of different kinds and sizes, with actions. States are [.., D, smax] (padded), actions [.., D, amax].

    config: a model_config (see model_config()); system: system.json of the data; masks [D, smax]: the active
    channels of each moving body.
    """

    def __init__(self, config, system, masks):
        super().__init__()
        network = {k: v for k, v in config["network"].items() if k != "mixer"}
        if config != model_config(config["dt_s"], config["history"], network, config["integrate"]["kinds"]):
            raise ValueError(f"unsupported model_config {config}")
        self.config, self.system = dict(config), system
        self.dt = float(config["dt_s"])
        self.k = int(config["history"])
        bodies = system["bodies"]
        self.moving = [k for k, b in enumerate(bodies) if b["moving"]]
        fixed_bodies = [k for k, b in enumerate(bodies) if not b["moving"]]
        slot = {k: n for n, k in enumerate(self.moving)}
        fidx = {k: n for n, k in enumerate(fixed_bodies)}
        mov = [bodies[k] for k in self.moving]
        self.n_bodies, self.n_fixed = len(mov), len(fixed_bodies)
        self.state_dims = [int(b.get("state_dim", 9)) for b in mov]
        self.action_dims = [int(b.get("action_dim", 0)) for b in mov]
        self.smax, self.amax = max(self.state_dims), max(1, max(self.action_dims))
        kind_names = []
        for b in mov:
            name = b.get("kind", "body")
            if name not in kind_names:
                kind_names.append(name)
        self.kind_names = kind_names
        self.register_buffer("body_kind", torch.tensor([kind_names.index(b.get("kind", "body")) for b in mov]))
        self.register_buffer("has_action", torch.tensor([a > 0 for a in self.action_dims]))
        smask = torch.tensor(masks, dtype=torch.float64)                                          # [D, smax]
        self.register_buffer("smask", smask)
        amask = torch.zeros(self.n_bodies, self.amax, dtype=torch.float64)
        for n, a in enumerate(self.action_dims):
            amask[n, :a] = 1.0
        self.register_buffer("amask", amask)
        quat = torch.zeros(self.n_bodies, self.smax, dtype=torch.bool)
        for n, b in enumerate(mov):
            for c, t in enumerate(b.get("channel_types", [])):
                quat[n, c] = t == "quaternion"
        self.register_buffer("quat", quat)
        # Pose integration of the body kinds in config["integrate"]["kinds"]: their position (joint angles) follows
        # their velocity (joint rates) channel by channel in order, their quaternion the angular velocity (world
        # frame); the networks' pose outputs are not used.
        self.integ_kinds = set(config["integrate"]["kinds"])
        types = {"position": 0, "quaternion": 1, "velocity": 2, "angular_velocity": 3, "angle": 0, "angular_rate": 2}
        chan = torch.full((self.n_bodies, 4, self.smax), -1, dtype=torch.long)   # [body, type, channel indices]
        integ_body = torch.zeros(self.n_bodies, dtype=torch.bool)
        for n, b in enumerate(mov):
            if b.get("kind", "body") in self.integ_kinds:
                integ_body[n] = True
            for k in range(4):
                cols = [c for c, tt in enumerate(b.get("channel_types", [])) if types.get(tt) == k]
                chan[n, k, : len(cols)] = torch.tensor(cols, dtype=torch.long) if cols else chan[n, k, :0]
            pc, vc = (chan[n, k][chan[n, k] >= 0] for k in (0, 2))
            if integ_body[n] and len(pc) != len(vc):
                raise ValueError(f"body {b.get('name')}: {len(pc)} position channels but {len(vc)} rate channels")
        self.register_buffer("integ_body", integ_body, persistent=False)   # derived from config and system
        self.register_buffer("chan", chan, persistent=False)
        pairs = system["pairs"]
        self.n_pairs = len(pairs)
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
        spec, nk = config["network"], len(kind_names)
        # Core: one query per body kind; one output head per kind (the body's own kind is used).
        self.core = TokenNet(spec, 1, self.k, nk, self.n_fixed, nk, self.smax, self.amax, [self.smax] * nk, True)
        self.collision = TokenNet(spec, positions, self.k, self.n_pairs, self.n_fixed, nk, self.smax, self.amax, [1], False)
        # Contact: changes of the pair's first body and of its moving partner (two padded state vectors).
        self.contact = TokenNet(spec, positions, self.k, self.n_pairs, self.n_fixed, nk, self.smax, self.amax,
                                [self.smax, self.smax], True)

    # ---- history -----------------------------------------------------------------
    def code(self, hist, ok, noise=None):
        """hist [B, K, D, S] (newest last), ok [B, K] -> cur [B, D, S], diff [B, K-1, D, S] (index k-1 = lag k),
        lag_ok [B, K] (lag 0 first). `noise` [D, S] adds i.i.d. Gaussian noise to every difference."""
        cur = hist[:, -1]
        diff = (hist[:, 1:] - hist[:, :-1]).flip(1)
        lag_ok = ok.flip(1)
        if noise is not None and self.k > 1:
            diff = diff + torch.randn_like(diff) * noise
        return cur, diff, lag_ok

    def run_core(self, cur, diff, act, lag_ok):
        """Smooth change of every moving body (its own history and action) -> [B, D, S]."""
        b, d = cur.shape[:2]
        c = cur.reshape(b * d, 1, -1)
        df = diff.transpose(1, 2).reshape(b * d, self.k - 1, 1, self.smax)
        a = act.reshape(b * d, 1, act.shape[-1])
        kind = self.body_kind.repeat(b)[:, None]
        present = torch.ones(b * d, 1, dtype=torch.bool, device=cur.device)
        has = self.has_action.repeat(b)[:, None]
        lo = lag_ok.repeat_interleave(d, 0)
        fixed = torch.full((b * d,), -1, dtype=torch.long, device=cur.device)
        query = self.body_kind.repeat(b)
        smask = self.smask.repeat(b, 1)[:, None]
        amask = self.amask.repeat(b, 1)[:, None]
        out = self.core(c, df, a, present, has, kind, lo, fixed, query, smask, amask)            # [B*D, nk*S]
        out = out.reshape(b * d, len(self.kind_names), self.smax)[torch.arange(b * d, device=cur.device), query]
        return out.reshape(b, d, self.smax) * self.smask

    def pair_inputs(self, cur, diff, act, lag_ok, rows, pairs):
        """Inputs of the pair networks for (sample, pair) entries `rows`, `pairs` (both [M])."""
        pos = self.pair_pos[pairs]                                                     # [M, P]
        idx = torch.where(pos < 0, cur.shape[1], pos)
        padc = torch.cat((cur, torch.zeros_like(cur[:, :1])), 1)
        padd = torch.cat((diff, torch.zeros_like(diff[:, :, :1])), 2)
        pada = torch.cat((act, torch.zeros_like(act[:, :1])), 1)
        c = padc[rows[:, None], idx]
        df = padd[rows[:, None, None], torch.arange(self.k - 1, device=cur.device)[None, :, None], idx[:, None, :]]
        a = pada[rows[:, None], idx]
        present = pos >= 0
        kind = torch.cat((self.body_kind, self.body_kind[:1]))[idx]
        has = torch.cat((self.has_action, self.has_action[:1] & False))[idx]
        smask = torch.cat((self.smask, torch.zeros_like(self.smask[:1])))[idx]
        amask = torch.cat((self.amask, torch.zeros_like(self.amask[:1])))[idx]
        return c, df, a, present, has, kind, lag_ok[rows], self.pair_fixed[pairs], pairs, smask, amask

    def logits(self, cur, diff, act, lag_ok, rows=None, pairs=None):
        """Collision logits: all pairs -> [B, P_pairs], or the given (row, pair) entries -> [M]."""
        b = cur.shape[0]
        full = rows is None
        if full:
            rows = torch.arange(b, device=cur.device).repeat_interleave(self.n_pairs)
            pairs = torch.arange(self.n_pairs, device=cur.device).repeat(b)
        out = self.collision(*self.pair_inputs(cur, diff, act, lag_ok, rows, pairs))[:, 0]
        return out.reshape(b, self.n_pairs) if full else out

    def contact_sum(self, cur, diff, act, lag_ok, gates):
        """Summed contact change [B, D, S]; only (sample, pair) entries with gate != 0 are evaluated."""
        total = torch.zeros_like(cur)
        idx = torch.nonzero(gates != 0)
        if len(idx):
            rows, pairs = idx[:, 0], idx[:, 1]
            out = self.contact(*self.pair_inputs(cur, diff, act, lag_ok, rows, pairs)).reshape(-1, 2, self.smax)
            g = gates[rows, pairs][:, None]
            total = total.index_put((rows, self.pair_pos[pairs, 0]), g * out[:, 0], accumulate=True)
            mm = self.pair_mm[pairs]
            if mm.any():
                total = total.index_put((rows[mm], self.pair_pos[pairs[mm], 1]), g[mm] * out[mm, 1], accumulate=True)
        return total * self.smask

    @staticmethod
    def _qmul(a, b):
        w1, x1, y1, z1 = a.unbind(-1)
        w2, x2, y2, z2 = b.unbind(-1)
        return torch.stack((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), -1)

    def assemble(self, cur, d_core, change):
        """Next state from the core change and the summed contact change; integrated poses follow the new rates."""
        nxt = cur + d_core + change
        out = nxt.clone()
        for n in torch.nonzero(self.integ_body).flatten().tolist():
            pc, qc, vc, wc = (self.chan[n, k][self.chan[n, k] >= 0] for k in range(4))
            m = self.smask[n]
            if len(pc) and len(vc):
                out[:, n, pc] = cur[:, n, pc] + self.dt * nxt[:, n, vc] * m[pc]
            if len(qc) == 4 and len(wc) == 3:
                half = 0.5 * self.dt * (nxt[:, n, wc] * m[wc])
                ang = half.norm(dim=-1, keepdim=True)
                dq = torch.cat((torch.cos(ang), torch.where(ang > 1e-12, torch.sin(ang) / ang.clamp_min(1e-12), torch.ones_like(ang)) * half), -1)
                q = self._qmul(dq, cur[:, n, qc])
                out[:, n, qc] = torch.where(m[qc] > 0, q, cur[:, n, qc])
        return self.finish(out)

    def finish(self, nxt):
        """Re-normalise quaternion channels (unit length; sign kept)."""
        if not self.quat.any():
            return nxt
        out = nxt.clone()
        for n in range(self.n_bodies):
            cols = torch.nonzero(self.quat[n]).flatten()
            if len(cols):
                q = nxt[:, n, cols]
                out[:, n, cols] = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return out

    def details(self, hist, ok, act):
        """One step with the collision network's switches -> next state, {logits, gates, core}."""
        cur, diff, lag_ok = self.code(hist, ok)
        d_core = self.run_core(cur, diff, act, lag_ok)
        logits = self.logits(cur, diff, act, lag_ok)
        g = (logits >= 0).to(cur.dtype)
        nxt = self.assemble(cur, d_core, self.contact_sum(cur, diff, act, lag_ok, g))
        return nxt, {"logits": logits, "gates": g, "core": d_core}

    def step_with_gates(self, hist, ok, act, gates, noise=None, core_grad=False):
        """Next state with given switches (training); the core change has no gradient unless core_grad."""
        cur, diff, lag_ok = self.code(hist, ok, noise)
        with torch.set_grad_enabled(core_grad and torch.is_grad_enabled()):
            core = self.run_core(cur, diff, act, lag_ok)
        return self.assemble(cur, core, self.contact_sum(cur, diff, act, lag_ok, gates.to(cur.dtype)))

    @staticmethod
    def advance(hist, ok, state):
        return torch.cat((hist[:, 1:], state[:, None]), 1), torch.cat((ok[:, 1:], torch.ones_like(ok[:, :1])), 1)

    def start(self, state):
        hist = torch.zeros(state.shape[0], self.k, *state.shape[1:], dtype=state.dtype, device=state.device)
        hist[:, -1] = state
        ok = torch.zeros(state.shape[0], self.k, dtype=torch.bool, device=state.device)
        ok[:, -1] = True
        return hist, ok

    def forward(self, hist, ok, act):
        return self.details(hist, ok, act)[0]

    def rollout(self, initial, actions, return_gates=False):
        """initial [B, D, S]; actions [B, steps, D, A] (the command of each step), or an int number of steps for a
        system without actions -> [B, steps+1, D, S] (and the switches [B, steps, P] with return_gates)."""
        state = initial.to(torch.float64)
        if isinstance(actions, int):
            actions = torch.zeros(state.shape[0], actions, self.n_bodies, self.amax, dtype=state.dtype, device=state.device)
        hist, ok = self.start(state)
        trajectory, gate_trace = [state], []
        for t in range(actions.shape[1]):
            state, info = self.details(hist, ok, actions[:, t].to(torch.float64))
            hist, ok = self.advance(hist, ok, state)
            trajectory.append(state)
            gate_trace.append(info["gates"])
        out = torch.stack(trajectory, 1)
        return (out, torch.stack(gate_trace, 1)) if return_gates else out


def load_model(path, device="cpu"):
    """Checkpoint (best.pt of a run, or a released seedNN.pt) -> (model in eval mode, checkpoint dict)."""
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    packet = torch.load(path, map_location=device, weights_only=True)
    model = ContactNRD(packet["model_config"], packet["system"], packet["masks"]).to(device)
    model.load_state_dict(packet["model_state_dict"])
    return model.eval(), packet
