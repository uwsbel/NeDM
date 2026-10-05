"""Unified contact NRD, version 3: the collision and contact networks see only the NRD state.

The Transformer core is unchanged and frozen (loaded from a version-2 core).
The two contact networks receive only what the model carries from step to step:
the moving bodies' states, optionally a history of the last K steps on the 10 ms
grid. No gravity, plane geometry, relative positions, radii or pair frames; the
networks are system-specific and learn the fixed geometry from data.

Rows. Each network is one set of weights run on rows; a row is a slice of the state:
    routing "pair"  one row per (affected body, partner): [state of the affected body,
                    state of the partner if it moves else nothing]; a moving-moving
                    pair gets both orders. Group = the partner (A-B's two orders share
                    one group). The row's output changes only the affected body.
    routing "joint" one row per pair, each the whole state of every moving body; group =
                    the pair. The row's output changes the pair's moving members.
History coding: [x(t), x(t)-x(t-1), ..., x(t-K+2)-x(t-K+1)] (a linear re-coding of
the last K states). Each network then applies a per-group data-fitted affine map:
centring and whitening (standardise, then rotate and rescale the principal axes,
with an eigenvalue floor; or per-channel standardisation only), fitted on that
group's near-contact windows (collision) or contact windows (contact), followed by
a fixed soft limit c*tanh(z/c). Whitening lets a small relative quantity such as the
A-B separation stand out from large absolute positions; it adds no information.

    collision(row, one-hot(group)) -> logit; pair logit = mean over the pair's rows; on if >= 0
    contact(row; group scale/shift) + group linear path -> change, de-normalised per group:
        routed rows: 9 numbers, the affected body's change (A and B rows of a group share it)
        joint rows:  [D, 9], masked to the pair's moving members
    s' = s + core(s) + sum over rows of gate(pair of row) * change(row)

Trunks: "mlp" on the flattened history, or "time_tf" (a causal Transformer over the
K time tokens of the row).
"""
from __future__ import annotations

import torch
from torch import nn

from nedm.contact_nrd.model import STATE, ContactGraphNRD, mlp
from nedm.core.training.model_transformer import ContinuousTransformer, TransformerConfig


def row_layout(system, routing):
    """Static description of the rows: list of dicts with bodies (slots), affects (slots), pair, group."""
    bodies = system["bodies"]
    moving = [k for k, b in enumerate(bodies) if b["moving"]]
    slot = {k: n for n, k in enumerate(moving)}
    pairs = system["pairs"]
    rows = []
    if routing == "joint":
        for p, (i, j) in enumerate(pairs):
            members = [slot[i]] + ([slot[j]] if j in slot else [])
            rows.append(dict(inputs=list(range(len(moving))), affects=members, pair=p, group=p))
        return rows, len(pairs), len(moving)
    has_moving_partner = any(j in slot for _, j in pairs)
    groups = {}
    for p, (i, j) in enumerate(pairs):
        key = "moving" if j in slot else j
        groups.setdefault(key, len(groups))
        partner = [slot[j]] if j in slot else ([None] if has_moving_partner else [])
        rows.append(dict(inputs=[slot[i]] + partner, affects=[slot[i]], pair=p, group=groups[key]))
        if j in slot:
            rows.append(dict(inputs=[slot[j], slot[i]], affects=[slot[j]], pair=p, group=groups[key]))
    width = 2 if has_moving_partner else 1
    return rows, len(groups), width


class FilmMLP(nn.Module):
    """MLP whose hidden layers get a per-group scale and shift (zero-initialised)."""

    def __init__(self, inputs, hidden, outputs, layers, activation, groups):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
        self.layers = nn.ModuleList([nn.Linear(inputs if k == 0 else hidden, hidden) for k in range(layers)])
        self.act = act()
        self.gamma = nn.Parameter(torch.zeros(layers, groups, hidden))
        self.beta = nn.Parameter(torch.zeros(layers, groups, hidden))
        self.out = nn.Linear(hidden, outputs)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, group):
        h = x
        for k, layer in enumerate(self.layers):
            h = self.act(layer(h))
            h = h * (1 + self.gamma[k][group]) + self.beta[k][group]
        return self.out(h)


class TimeTF(nn.Module):
    """Causal Transformer over K time tokens [x_k, one-hot(group)] -> readout of the last token."""

    def __init__(self, token, k, groups, outputs, spec, with_current=False):
        super().__init__()
        width = int(spec.get("embedding", 128))
        self.k, self.token, self.groups, self.with_current = k, token, groups, with_current
        self.tf = ContinuousTransformer(TransformerConfig(token + groups, k, int(spec.get("layers", 2)), int(spec.get("heads", 4)),
                                                          width, 0.0, True))
        self.tf.final_norm = nn.Identity()
        nn.init.zeros_(self.tf.position_embedding.weight)
        for block in self.tf.blocks:
            for proj in (block.attn.c_proj, block.mlp.c_proj):
                nn.init.normal_(proj.weight, std=float(spec.get("block_init_std", 1e-3)))
                nn.init.zeros_(proj.bias)
        read = width + (token if with_current else 0)
        self.head = mlp(read, int(spec.get("head_width", 32)), outputs, int(spec.get("head_layers", 1)),
                        spec.get("activation", "relu")) if spec.get("head_layers", 1) else nn.Linear(read, outputs)
        last = self.head[-1] if isinstance(self.head, nn.Sequential) else self.head
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x, group):
        """x [M, K*token] coded as [x(t), d(t), d(t-1), ...]; tokens are fed oldest first, x(t) last."""
        seq = x.reshape(len(x), self.k, self.token).flip(1)
        onehot = nn.functional.one_hot(group, self.groups).to(x.dtype)[:, None, :].expand(-1, self.k, -1)
        h = self.tf(torch.cat((seq, onehot), -1))[:, -1]
        if self.with_current:
            h = torch.cat((h, seq[:, -1]), -1)
        return self.head(h)


class StateContactNRD(ContactGraphNRD):
    def __init__(self, config, system, normalization):
        super().__init__({**config, "frame": "world"}, system, _v1_placeholders(normalization))
        dtype = self.state_mean.dtype
        self.k = int(config.get("history", 1))
        self.routing = config.get("routing", "pair")
        rows, n_groups, width = row_layout(system, self.routing)
        self.rows, self.n_groups = rows, n_groups
        d, p, r = len(self.moving), len(system["pairs"]), len(rows)
        self.n_bodies, self.n_pairs, self.n_rows = d, p, r
        self.row_width = width * STATE                       # per time step
        self.features = self.k * self.row_width
        gather = torch.full((r, width), -1, dtype=torch.long)
        for n, row in enumerate(rows):
            for m, s in enumerate(row["inputs"]):
                gather[n, m] = -1 if s is None else s
        affects = torch.zeros(r, d)
        for n, row in enumerate(rows):
            affects[n, row["affects"]] = 1.0
        row_pair = torch.tensor([row["pair"] for row in rows])
        pair_rows = torch.zeros(p, r)
        for n, row in enumerate(rows):
            pair_rows[row["pair"], n] = 1.0
        pair_rows = pair_rows / pair_rows.sum(-1, keepdim=True)
        self.register_buffer("row_gather", gather)
        self.register_buffer("row_affects", affects.to(dtype))
        self.register_buffer("row_pair", row_pair)
        self.register_buffer("row_group", torch.tensor([row["group"] for row in rows]))
        self.register_buffer("pair_rows", pair_rows.to(dtype))              # mean of a pair's row logits
        self.out_dim = STATE if self.routing == "pair" else d * STATE
        cs, qs = config["collision"], config["contact"]
        # Which inputs each network reads: state channels (0-2 position, 3-5 velocity, 6-8 spin) and
        # history blocks (0 = current state, 1.. = step-to-step differences). Default: everything.
        # Diagnostic option: append the current difference (second slot - first slot) of position and
        # velocity for two-body rows, derived from the state (6 numbers, after all history blocks).
        self.derived = bool(config.get("derived_relative", False)) and width == 2
        self.n_derived = 6 if self.derived else 0

        def columns(spec):
            chans = spec.get("channels", list(range(STATE)))
            blocks = spec.get("blocks", list(range(self.k)))
            cols = [b * self.row_width + m * STATE + c for b in blocks for m in range(width) for c in chans]
            return torch.tensor(cols + list(range(self.features, self.features + self.n_derived)))
        self.register_buffer("cols_gate", columns(cs), persistent=False)   # derived from the config
        self.register_buffer("cols_cont", columns(qs), persistent=False)
        self.f_gate, self.f_cont = len(self.cols_gate), len(self.cols_cont)
        for key, f in (("gate", self.f_gate), ("cont", self.f_cont)):
            for name, shape in ((f"{key}_mean3", (n_groups, f)), (f"{key}_proj3", (n_groups, f, f))):
                default = torch.eye(f).expand(n_groups, -1, -1).clone() if "proj" in name else torch.zeros(shape)
                value = torch.tensor(normalization[name]) if name in normalization else default
                self.register_buffer(name, value.to(dtype))
        for name in ("out_mean3", "out_std3"):
            default = torch.ones(n_groups, self.out_dim) if "std" in name else torch.zeros(n_groups, self.out_dim)
            value = torch.tensor(normalization[name]) if name in normalization else default
            self.register_buffer(name, value.to(dtype))
        # Integrated positions (minimal-state option): the networks change velocity and spin only;
        # position follows from the trapezoid of the old and new velocity, plus (optionally) a
        # contact-network position correction for an impact part-way through the step.
        self.integrate = bool(config.get("integrate_positions", False))
        self.position_correction = bool(config.get("position_correction", True))
        # The simulator's own step: with semi-implicit Euler sub-steps of length h under a constant
        # acceleration, the exact position change over the model step is 0.5*dt*(v+v') + 0.5*h*(v'-v).
        self.sub_dt = float(system.get("physics_config", {}).get("simulation", {}).get("step_s", 0.0))
        del self.collision, self.contact, self.contact_skip
        if cs.get("trunk", "mlp") == "mlp":
            act = cs.get("activation", "relu")
            self.collision = nn.Sequential(mlp(self.f_gate + n_groups, int(cs.get("width", 256)), int(cs.get("latent", 32)), 2, act),
                                           {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[act](), nn.Linear(int(cs.get("latent", 32)), 1))
        else:
            assert self.f_gate == self.features, "time Transformer reads every input"
            self.collision = TimeTF(self.row_width, self.k, n_groups, 1, cs, with_current=True)
        if qs.get("trunk", "mlp") == "mlp":
            self.contact = FilmMLP(self.f_cont, int(qs.get("width", 1024)), self.out_dim, int(qs.get("layers", 3)),
                                   qs.get("activation", "tanh"), n_groups)
        else:
            assert self.f_cont == self.features, "time Transformer reads every input"
            self.contact = TimeTF(self.row_width, self.k, n_groups, self.out_dim, {**qs, "head_layers": 0})
        self.contact_skip = nn.Parameter(torch.zeros(n_groups, self.f_cont + 1, self.out_dim))   # per-group linear path
        self.to(dtype)

    # ---- history ----------------------------------------------------------------
    def initial_history(self, state):
        n = max(self.k, self.context)
        return state.unsqueeze(-3).expand(*state.shape[:-2], n, *state.shape[-2:]).clone()

    def core_delta(self, state, history=None):
        history = self.initial_history(state) if history is None else history
        return super().core_delta(state, history[..., -self.context:, :, :])

    def row_inputs(self, history):
        """history [.., >=K, D, 9] -> raw coded rows [.., R, K*width*9] (time-major: x(t), x(t)-x(t-1), ...)."""
        h = history[..., -self.k:, :, :]
        padded = torch.cat((h, torch.zeros_like(h[..., :1, :])), -2)                    # body slot D = zeros
        rows = padded[..., torch.where(self.row_gather < 0, self.n_bodies, self.row_gather), :]   # [.., K, R, width, 9]
        rows = rows.flatten(-2)                                                                            # [.., K, R, width*9]
        cur = rows[..., -1, :, :]
        diffs = [rows[..., k, :, :] - rows[..., k - 1, :, :] for k in range(self.k - 1, 0, -1)]
        extra = [cur[..., STATE:STATE + 6] - cur[..., 0:6]] if getattr(self, "derived", False) else []
        return torch.cat([cur] + diffs + extra, -1)

    def scaled_rows(self, history, which="contact"):
        """Coded rows through the per-group data-fitted affine map (centre, then a whitening matrix)."""
        x = self.row_inputs(history)
        if which == "gate":
            x, mean, proj = x[..., self.cols_gate], self.gate_mean3, self.gate_proj3
        else:
            x, mean, proj = x[..., self.cols_cont], self.cont_mean3, self.cont_proj3
        return self.squash(torch.einsum("...rf,rfg->...rg", x - mean[self.row_group], proj[self.row_group]))

    def squash(self, z):
        """Fixed soft limit c*tanh(z/c) on scaled inputs: about the identity inside +-c/2, bounded far from
        the data the map was fitted on (e.g. a ball 1 m from the cushion whose contacts set the scale)."""
        c = float(self.config.get("input_clip", 0) or 0)
        return c * torch.tanh(z / c) if c > 0 else z

    # ---- networks ---------------------------------------------------------------
    def row_logits(self, z):
        """z [.., R, F] -> row logits [.., R]."""
        lead = z.shape[:-2]
        flat = z.reshape(-1, self.f_gate)
        group = self.row_group.expand(*lead, self.n_rows).reshape(-1)
        if isinstance(self.collision, TimeTF):
            out = self.collision(flat, group)
        else:
            onehot = nn.functional.one_hot(group, self.n_groups).to(z.dtype)
            out = self.collision(torch.cat((flat, onehot), -1))
        return out.reshape(*lead, self.n_rows)

    def pair_logits(self, z):
        return self.row_logits(z) @ self.pair_rows.T

    def row_changes(self, z_rows, group, affects):
        """Selected rows: z [M, F], group [M], affects [M, D] -> world changes [M, D, 9]."""
        raw = self.contact(z_rows, group)
        skip = torch.einsum("mf,mfo->mo", torch.cat((z_rows, torch.ones_like(z_rows[:, :1])), -1), self.contact_skip[group])
        out = (raw + skip) * self.out_std3[group] + self.out_mean3[group]
        if self.routing == "pair":   # one 9-vector: the affected body's change, placed in its slot
            return out[:, None, :] * affects[..., None] * self.mask
        return out.reshape(len(z_rows), self.n_bodies, STATE) * affects[..., None] * self.mask

    def contact_sum(self, z, row_gates):
        """z [.., R, F], row_gates [.., R] -> summed change [.., D, 9]; only rows with gate != 0 are evaluated."""
        lead = z.shape[:-2]
        zf, gf = z.reshape(-1, self.n_rows, self.f_cont), row_gates.reshape(-1, self.n_rows)
        total = torch.zeros(len(zf), self.n_bodies, STATE, dtype=z.dtype, device=z.device)
        idx = torch.nonzero(gf != 0)
        if len(idx):
            ch = self.row_changes(zf[idx[:, 0], idx[:, 1]], self.row_group[idx[:, 1]], self.row_affects[idx[:, 1]])
            total = total.index_add(0, idx[:, 0], gf[idx[:, 0], idx[:, 1], None, None] * ch)
        return total.reshape(*lead, self.n_bodies, STATE)

    def integrated_dp(self, v0, v1):
        """Position change over one model step from the velocities at its start and end."""
        return 0.5 * self.dt * (v0 + v1) + 0.5 * self.sub_dt * (v1 - v0)

    def next_state(self, state, d_core, change):
        """Assemble the next state from the core change and the summed contact change."""
        if not self.integrate:
            return state + d_core + change
        vw = state[..., 3:9] + d_core[..., 3:9] + change[..., 3:9]
        dp = self.integrated_dp(state[..., 3:6], vw[..., 0:3])
        if self.position_correction:
            dp = dp + change[..., 0:3]
        return torch.cat((state[..., 0:3] + dp * self.mask[0:3], vw), -1)

    def contact_target(self, state, nxt, d_core):
        """What the contact network must supply so that next_state(state, d_core, change) == nxt."""
        if not self.integrate:
            return nxt - state - d_core
        dvw = nxt[..., 3:9] - state[..., 3:9] - d_core[..., 3:9]
        if self.position_correction:
            dp = nxt[..., 0:3] - state[..., 0:3] - self.integrated_dp(state[..., 3:6], nxt[..., 3:6])
        else:
            dp = torch.zeros_like(dvw[..., 0:3])
        return torch.cat((dp * self.mask[0:3], dvw), -1)

    def details(self, state, history=None, gates=None):
        history = self.initial_history(state) if history is None else history
        d_core = self.core_delta(state, history)
        logits = self.pair_logits(self.scaled_rows(history, "gate"))
        g = self.gates(logits) if gates is None else gates.to(state.dtype)
        nxt = self.next_state(state, d_core, self.contact_sum(self.scaled_rows(history, "contact"), g[..., self.row_pair]))
        return nxt, {"logits": logits, "gates": g, "core": d_core}

    def forward(self, state, history=None):
        return self.details(state, history)[0]


def _v1_placeholders(normalization):
    from nedm.contact_nrd.model import PAIR_FEATURES
    return {**normalization, "gate_mean": [0.0] * PAIR_FEATURES, "gate_std": [1.0] * PAIR_FEATURES,
            "in_mean": [0.0] * PAIR_FEATURES, "in_std": [1.0] * PAIR_FEATURES, "out_mean": [0.0] * STATE, "out_std": [1.0] * STATE}
