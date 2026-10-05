"""Unified contact NRD: Transformer core + collision network + contact network.

Moving body state (world frame): s = [p(3), v(3), w(3)]. Every body (moving
or fixed) has a token of one layout:
    [is_sphere, is_plane, radius, position(3), normal(3), velocity(3), spin(3)]
Fixed planes take position = a point on the plane and their normal from the
system description; moving spheres take their predicted state.

One step (model_step, e.g. 10 ms), fed back:
    core:      d_core_i = Transformer(history of body i)                 (shared over bodies)
    pairs:     q_ij = features(token_i, token_j, p_j - p_i, v_j - v_i)   for every candidate pair
    collision: g_ij = 1[collision(q_ij) >= 0]   (two moving bodies: mean of both orders' logits)
    contact:   d_ij = contact(q_ij)  = change of body i caused by j    (and d_ji = contact(q_ji))
    s_i' = s_i + mask * (d_core_i + sum_j g_ij d_ij)
Exactly two contact networks, shared over all pairs and systems. Optional
pair frame: vectors are expressed in a frame whose first axis is the
partner plane's normal or the direction to the partner sphere (second axis
from the system's up axis), and the contact output is rotated back.
Nothing in inference computes distances or impact times.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from nedm.core.training.model_transformer import ContinuousTransformer, TransformerConfig

STATE, TOKEN = 9, 15
PAIR_FEATURES = 2 * TOKEN + 6


def mlp(inputs, hidden, outputs, layers, activation):
    act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
    parts, width = [], inputs
    for _ in range(layers):
        parts += [nn.Linear(width, hidden), act()]
        width = hidden
    return nn.Sequential(*parts, nn.Linear(width, outputs))


class ContactGraphNRD(nn.Module):
    def __init__(self, config, system, normalization):
        super().__init__()
        self.config = dict(config)
        self.system = system
        self.dt = float(config["dt_s"])
        self.context = int(config.get("context", 1))
        self.frame = config.get("frame", "world")
        self.anchor_rest = bool(config.get("anchor_rest", False))
        # Exploratory: the core never sees absolute position (free motion does not depend on it).
        self.core_position_free = bool(config.get("core_position_free", False))
        dtype = torch.float64 if config.get("dtype", "float64") == "float64" else torch.float32
        bodies = system["bodies"]
        self.moving = [k for k, b in enumerate(bodies) if b["moving"]]
        slot = {k: n for n, k in enumerate(self.moving)}
        pairs = torch.tensor(system["pairs"], dtype=torch.long)
        if not all(bodies[i]["moving"] for i, _ in system["pairs"]):
            raise ValueError("first body of every pair must be moving")
        self.register_buffer("pair_i", torch.tensor([slot[int(i)] for i in pairs[:, 0]]))
        self.register_buffer("pair_j_body", pairs[:, 1].clone())
        self.register_buffer("pair_j_moving", torch.tensor([bodies[int(j)]["moving"] for j in pairs[:, 1]]))
        self.register_buffer("pair_j_slot", torch.tensor([slot.get(int(j), -1) for j in pairs[:, 1]]))
        tokens = torch.zeros(len(bodies), TOKEN)
        for k, b in enumerate(bodies):
            if b["kind"] == "sphere":
                tokens[k, 0], tokens[k, 2] = 1.0, b["radius"]
            else:
                tokens[k, 1] = 1.0
                tokens[k, 3:6] = torch.tensor(b["point"])
                tokens[k, 6:9] = torch.tensor(b["normal"])
        self.register_buffer("body_tokens", tokens)
        self.register_buffer("up", torch.tensor(system.get("gravity_axis", [0.0, 0.0, 1.0])))
        for key in ("state_mean", "state_std", "delta_scale", "mask", "gate_mean", "gate_std",
                    "in_mean", "in_std", "out_mean", "out_std"):
            self.register_buffer(key, torch.tensor(normalization[key]))
        width = int(config.get("embedding", 128))
        self.backbone = ContinuousTransformer(TransformerConfig(STATE, self.context, int(config.get("layers", 4)),
                                                                int(config.get("heads", 4)), width, 0.0, True))
        self.backbone.final_norm = nn.Identity()
        nn.init.zeros_(self.backbone.position_embedding.weight)
        std = float(config.get("block_init_std", 1e-3))
        for block in self.backbone.blocks:
            for proj in (block.attn.c_proj, block.mlp.c_proj):
                nn.init.normal_(proj.weight, std=std)
                nn.init.zeros_(proj.bias)
        self.core_head = nn.Linear(width, STATE)
        nn.init.zeros_(self.core_head.weight)
        nn.init.zeros_(self.core_head.bias)
        gate_hidden, latent = int(config.get("gate_hidden", 256)), int(config.get("gate_latent", 32))
        act = config.get("gate_activation", "relu")
        self.collision = nn.Sequential(mlp(PAIR_FEATURES, gate_hidden, latent, 2, act),
                                       {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[act](), nn.Linear(latent, 1))
        self.contact = mlp(PAIR_FEATURES, int(config.get("hidden", 512)), STATE, int(config.get("contact_layers", 3)),
                           config.get("contact_activation", "tanh"))
        self.contact_skip = nn.Linear(PAIR_FEATURES, STATE)
        nn.init.zeros_(self.contact[-1].weight)
        nn.init.zeros_(self.contact[-1].bias)
        self.to(dtype=dtype)

    # ---- core --------------------------------------------------------------
    def initial_history(self, state):
        return state.unsqueeze(-3).expand(*state.shape[:-2], self.context, *state.shape[-2:]).clone()

    def advance_history(self, history, state):
        return torch.cat((history[..., 1:, :, :], state.unsqueeze(-3)), -3)

    def body_delta(self, body_history):
        """[M, K, 9] histories of single bodies -> [M, 9] smooth change."""
        def embed(h):
            if self.core_position_free:
                h = torch.cat((self.state_mean[:3].expand_as(h[..., :3]), h[..., 3:]), -1)
            return self.backbone((h - self.state_mean) / self.state_std)[..., -1, :]
        out = self.core_head(embed(body_history))
        if self.anchor_rest:
            rest = torch.cat((body_history[..., :3], torch.zeros_like(body_history[..., 3:])), -1)
            out = out - self.core_head(embed(rest))
        return out * self.delta_scale * self.mask

    def core_delta(self, state, history=None):
        """state [.., D, 9]; history [.., K, D, 9]."""
        history = self.initial_history(state) if history is None else history
        lead, k, d = history.shape[:-3], history.shape[-3], history.shape[-2]
        flat = history.reshape(-1, k, d, STATE).transpose(1, 2).reshape(-1, k, STATE)
        return self.body_delta(flat).reshape(*lead, d, STATE)

    # ---- pairs -------------------------------------------------------------
    def tokens(self, state):
        """[.., D, 9] -> [.., bodies, 15]."""
        lead = state.shape[:-2]
        tokens = self.body_tokens.expand(*lead, *self.body_tokens.shape).clone()
        idx = torch.tensor(self.moving, device=state.device)
        moving = tokens[..., idx, :]
        moving = torch.cat((moving[..., :3], state[..., :3], torch.zeros_like(state[..., :3]), state[..., 3:]), -1)
        tokens[..., idx, :] = moving
        return tokens

    def pair_frames(self, ti, tj):
        """Rotation [.., 3, 3] whose columns are the pair-frame axes in world coordinates."""
        d = tj[..., 3:6] - ti[..., 3:6]
        e1 = torch.where(tj[..., 1:2] > 0.5, tj[..., 6:9], d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9))
        up = self.up.expand_as(e1)
        x_axis = torch.zeros_like(e1); x_axis[..., 0] = 1.0
        y_axis = torch.zeros_like(e1); y_axis[..., 1] = 1.0
        side = torch.where((e1[..., 0:1].abs() <= e1[..., 1:2].abs()), x_axis, y_axis)
        ref = torch.where((e1 * up).sum(-1, keepdim=True).abs() > 0.9, side, up)
        e2 = ref - (ref * e1).sum(-1, keepdim=True) * e1
        e2 = e2 / e2.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        e3 = torch.cross(e1, e2, dim=-1)
        return torch.stack((e1, e2, e3), -1)

    def pair_features(self, ti, tj):
        """Tokens of the pair -> [.., 36] features (world or pair frame) and the frame (or None)."""
        dp, dv = tj[..., 3:6] - ti[..., 3:6], tj[..., 9:12] - ti[..., 9:12]
        if self.frame != "pair":
            return torch.cat((ti, tj, dp, dv), -1), None
        rot = self.pair_frames(ti, tj)

        def to_frame(v):
            return (v.unsqueeze(-2) @ rot).squeeze(-2)

        def tok(t):
            return torch.cat((t[..., :3], to_frame(t[..., 3:6]), to_frame(t[..., 6:9]), to_frame(t[..., 9:12]), to_frame(t[..., 12:15])), -1)
        return torch.cat((tok(ti), tok(tj), to_frame(dp), to_frame(dv)), -1), rot

    def all_pair_features(self, state):
        """Features for every candidate pair (i -> j) and, for moving partners, the reverse (j -> i)."""
        tokens = self.tokens(state)
        ti = tokens[..., self.moving, :][..., self.pair_i, :]
        tj = tokens[..., self.pair_j_body, :]
        forward, rot_f = self.pair_features(ti, tj)
        mm = torch.nonzero(self.pair_j_moving).flatten()
        if len(mm):
            backward, rot_b = self.pair_features(tj[..., mm, :], ti[..., mm, :])
        else:
            backward, rot_b = forward[..., :0, :], None
        return forward, rot_f, backward, rot_b, mm

    def collision_logits(self, forward, backward, mm):
        logits = self.collision((forward - self.gate_mean) / self.gate_std)[..., 0]
        if len(mm):
            back = self.collision((backward - self.gate_mean) / self.gate_std)[..., 0]
            logits = logits.clone()
            logits[..., mm] = 0.5 * (logits[..., mm] + back)
        return logits

    def contact_change(self, features, rot):
        f = (features - self.in_mean) / self.in_std
        out = (self.contact(f) + self.contact_skip(f)) * self.out_std + self.out_mean
        if rot is not None:
            # Output channels [dp, dv, dw] are pair-frame vectors: rotate back to world.
            out = torch.cat([(rot @ out[..., k:k + 3].unsqueeze(-1)).squeeze(-1) for k in (0, 3, 6)], -1)
        return out * self.mask

    def contact_terms(self, state):
        """Per-pair changes for the first body [.., P, 9] and, for moving partners, for the second [.., M, 9]."""
        forward, rot_f, backward, rot_b, mm = self.all_pair_features(state)
        return self.collision_logits(forward, backward, mm), self.contact_change(forward, rot_f), \
            (self.contact_change(backward, rot_b) if len(mm) else None), mm

    def gates(self, logits):
        if self.config.get("gate", "hard") == "soft":
            return logits.sigmoid()
        return (logits >= 0).to(logits.dtype)

    def scatter(self, state, gates, d_first, d_second, mm):
        total = torch.zeros_like(state)
        total = total.index_add(-2, self.pair_i, gates[..., None] * d_first)
        if d_second is not None:
            total = total.index_add(-2, self.pair_j_slot[mm], gates[..., mm, None] * d_second)
        return total

    def details(self, state, history=None, gates=None):
        d_core = self.core_delta(state, history)
        logits, d_first, d_second, mm = self.contact_terms(state)
        g = self.gates(logits) if gates is None else gates.to(state.dtype)
        nxt = state + d_core + self.scatter(state, g, d_first, d_second, mm)
        return nxt, {"logits": logits, "gates": g, "core": d_core}

    def forward(self, state, history=None):
        return self.details(state, history)[0]

    def rollout(self, initial, steps, return_gates=False):
        state = initial.to(self.state_mean.dtype)
        history = self.initial_history(state)
        trajectory, gate_trace = [state], []
        for _ in range(steps):
            state, info = self.details(state, history)
            history = self.advance_history(history, state)
            trajectory.append(state)
            gate_trace.append(info["gates"])
        out = torch.stack(trajectory, -3)
        return (out, torch.stack(gate_trace, -2)) if return_gates else out


def load(path, device="cpu"):
    packet = torch.load(path, map_location=device, weights_only=False)
    model = ContactGraphNRD(packet["model_config"], packet["system"], packet["normalization"]).to(device)
    model.load_state_dict(packet["model_state_dict"])
    return model.eval(), packet
