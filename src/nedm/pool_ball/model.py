"""Pool NRD: Transformer backbone for smooth motion plus learned contact residuals.

State s [.., 14] = ball A then ball B, each [x, y, vx, vy, wx, wy, wz].
One call advances one model step (10 ms by default):

    s_next = s + d_smooth(history) + sum_k gate_k(s) * d_contact_k(s)

d_smooth comes from the shared ContinuousTransformer over the causal history of
predicted states. Contact layouts ("contact" in the config):

  scalar_shared  one contact MLP (14 -> p(any impulsive contact in the next
                 step)) and one shared bounce NN (14 -> 14). Literal port of the
                 bouncing-ball winner.
  structured     ball-ball module: contact MLP 14 -> p(A-B contact), bounce NN
                 14 -> 14; cushion module shared by both balls: contact MLP on
                 that ball's own 7 numbers -> p(cushion contact), bounce NN
                 7 -> 7 applied to that ball.
  none           Transformer only (control).

No geometry, time-to-impact, friction law or true contact label is used in
inference. Normalisations and data-fitted linear initialisations come from
training transitions only. History is left-padded by repeating the first state.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from nedm.core.training.model_transformer import ContinuousTransformer, TransformerConfig

DIM, BALL = 14, 7
CUSHION_NAMES = ("xp", "xm", "yp", "ym")


def mlp(inputs, hidden, outputs, layers, activation):
    act = nn.ReLU if activation == "relu" else nn.Tanh if activation == "tanh" else nn.GELU
    parts, width = [], inputs
    for _ in range(layers):
        parts += [nn.Linear(width, hidden), act()]
        width = hidden
    return nn.Sequential(*parts, nn.Linear(width, outputs))


class ContactModule(nn.Module):
    """Gate MLP (-> latent -> logit) and bounce NN with a learned linear skip."""

    def __init__(self, inputs, outputs, cfg, dtype):
        super().__init__()
        gate_hidden = int(cfg.get("gate_hidden", 256))
        self.gate = nn.Sequential(mlp(inputs, gate_hidden, int(cfg.get("gate_latent", 32)), 2, cfg.get("gate_activation", "relu")),
                                  nn.ReLU() if cfg.get("gate_activation", "relu") == "relu" else nn.Tanh(),
                                  nn.Linear(int(cfg.get("gate_latent", 32)), 1))
        self.bounce = mlp(inputs, int(cfg.get("hidden", 512)), outputs, int(cfg.get("bounce_layers", 3)), cfg.get("bounce_activation", "tanh"))
        self.skip = nn.Linear(inputs, outputs)
        nn.init.zeros_(self.bounce[-1].weight)
        nn.init.zeros_(self.bounce[-1].bias)
        for name, size in (("in_mean", inputs), ("out_mean", outputs)):
            self.register_buffer(name, torch.zeros(size, dtype=dtype))
        for name, size in (("in_std", inputs), ("out_std", outputs), ("gate_std", inputs)):
            self.register_buffer(name, torch.ones(size, dtype=dtype))
        self.register_buffer("gate_mean", torch.zeros(inputs, dtype=dtype))

    def logit(self, x):
        return self.gate((x - self.gate_mean) / self.gate_std)[..., 0]

    def response(self, x):
        f = (x - self.in_mean) / self.in_std
        return (self.bounce(f) + self.skip(f)) * self.out_std + self.out_mean


class PoolNRD(nn.Module):
    def __init__(self, config, normalization):
        super().__init__()
        self.config = dict(config)
        self.dt = float(config["dt_s"])
        self.context = int(config.get("context", 1))
        self.contact = config.get("contact", "scalar_shared")
        # "per_ball": one shared Transformer runs on each ball's own 7 numbers
        # (no path between balls; smooth cloth motion does not couple them) and
        # its output is anchored at rest, h(s) - h(s with v = w = 0), so a ball
        # at rest stays exactly at rest. "joint": one 14-number token (literal
        # bouncing-ball port).
        self.core_mode = config.get("core", "per_ball")
        dtype = torch.float64 if config.get("dtype", "float64") == "float64" else torch.float32
        self.register_buffer("state_mean", torch.tensor(normalization["mean"], dtype=dtype))
        self.register_buffer("state_std", torch.tensor(normalization["std"], dtype=dtype))
        self.register_buffer("delta_mean", torch.tensor(normalization["delta_mean"], dtype=dtype))
        self.register_buffer("delta_scale", torch.tensor(normalization["delta_scale"], dtype=dtype))
        width = int(config.get("embedding", 64))
        core_dim = BALL if self.core_mode == "per_ball" else DIM
        if self.core_mode == "per_ball":
            for key in ("ball_mean", "ball_std", "ball_delta_scale"):
                self.register_buffer(key, torch.tensor(normalization[key], dtype=dtype))
        self.backbone = ContinuousTransformer(TransformerConfig(core_dim, self.context, int(config.get("layers", 2)),
                                                                int(config.get("heads", 4)), width, 0.0, True))
        self.affine_preserving = bool(config.get("affine_preserving", True))
        if self.affine_preserving:
            self.backbone.final_norm = nn.Identity()
            nn.init.zeros_(self.backbone.position_embedding.weight)
            std = float(config.get("block_init_std", 0.0))
            for block in self.backbone.blocks:
                for proj in (block.attn.c_proj, block.mlp.c_proj):
                    nn.init.normal_(proj.weight, std=std) if std > 0 else nn.init.zeros_(proj.weight)
                    nn.init.zeros_(proj.bias)
            self.core_head = nn.Linear(width, core_dim)
            last = self.core_head
        else:
            self.core_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, core_dim))
            last = self.core_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        if self.contact == "scalar_shared":
            self.modules_ = nn.ModuleDict({"any": ContactModule(DIM, DIM, config, dtype)})
        elif self.contact == "structured":
            self.modules_ = nn.ModuleDict({"ab": ContactModule(DIM, DIM, config, dtype),
                                           "cushion": ContactModule(BALL, BALL, config, dtype)})
        elif self.contact == "structured4":
            # One cushion module per cushion (xp, xm, yp, ym), each shared by A and B.
            self.modules_ = nn.ModuleDict({"ab": ContactModule(DIM, DIM, config, dtype),
                                           **{f"cushion_{c}": ContactModule(BALL, BALL, config, dtype) for c in CUSHION_NAMES}})
        elif self.contact == "none":
            self.modules_ = nn.ModuleDict()
        else:
            raise ValueError(f"unknown contact layout {self.contact}")
        self.threshold_logit = math.log(float(config.get("gate_threshold", 0.5)) / (1 - float(config.get("gate_threshold", 0.5))))
        self.to(dtype=dtype)

    def ball_embedding(self, ball_history):
        """Last-token embedding of per-ball histories [M, K, 7]."""
        return self.backbone((ball_history - self.ball_mean) / self.ball_std)[..., -1, :]

    def ball_delta(self, ball_history):
        """Anchored smooth change of one ball over one step: [M, K, 7] -> [M, 7]."""
        rest = torch.cat((ball_history[..., :2], torch.zeros_like(ball_history[..., 2:])), -1)
        return (self.core_head(self.ball_embedding(ball_history)) - self.core_head(self.ball_embedding(rest))) * self.ball_delta_scale

    # --- history -------------------------------------------------------
    def initial_history(self, state):
        return state.to(self.state_mean.dtype).unsqueeze(-2).expand(*state.shape[:-1], self.context, DIM).clone()

    def advance_history(self, history, state):
        return torch.cat((history[..., 1:, :], state.to(self.state_mean.dtype).unsqueeze(-2)), -2)

    # --- branches ------------------------------------------------------
    def core_delta(self, state, history=None):
        history = self.initial_history(state) if history is None else history.to(self.state_mean.dtype)
        if self.core_mode == "per_ball":
            lead, k = history.shape[:-2], history.shape[-2]
            balls = history.reshape(*lead, k, 2, BALL).transpose(-3, -2).reshape(-1, k, BALL)
            return self.ball_delta(balls).reshape(*lead, DIM)
        embedded = self.backbone((history - self.state_mean) / self.state_std)[..., -1, :]
        return self.core_head(embedded) * self.delta_scale + self.delta_mean

    def logits(self, state):
        """Gate logits: scalar_shared -> [.., 1] (any); structured -> [.., 3] (A-B, A cushion, B cushion);
        structured4 -> [.., 9] (A-B, A at xp/xm/yp/ym, B at xp/xm/yp/ym)."""
        s = state.to(self.state_mean.dtype)
        if self.contact == "scalar_shared":
            return self.modules_["any"].logit(s)[..., None]
        if self.contact == "structured":
            cushion = self.modules_["cushion"]
            return torch.stack((self.modules_["ab"].logit(s), cushion.logit(s[..., :BALL]), cushion.logit(s[..., BALL:])), -1)
        if self.contact == "structured4":
            out = [self.modules_["ab"].logit(s)]
            for ball in (s[..., :BALL], s[..., BALL:]):
                out += [self.modules_[f"cushion_{c}"].logit(ball) for c in CUSHION_NAMES]
            return torch.stack(out, -1)
        return s[..., :0]

    def responses(self, state):
        """Per-gate corrections [.., G, 14], aligned with logits()."""
        s = state.to(self.state_mean.dtype)
        if self.contact == "scalar_shared":
            return self.modules_["any"].response(s)[..., None, :]
        if self.contact == "structured":
            cushion = self.modules_["cushion"]
            zeros = torch.zeros_like(s[..., :BALL])
            a = torch.cat((cushion.response(s[..., :BALL]), zeros), -1)
            b = torch.cat((zeros, cushion.response(s[..., BALL:])), -1)
            return torch.stack((self.modules_["ab"].response(s), a, b), -2)
        if self.contact == "structured4":
            zeros = torch.zeros_like(s[..., :BALL])
            out = [self.modules_["ab"].response(s)]
            out += [torch.cat((self.modules_[f"cushion_{c}"].response(s[..., :BALL]), zeros), -1) for c in CUSHION_NAMES]
            out += [torch.cat((zeros, self.modules_[f"cushion_{c}"].response(s[..., BALL:])), -1) for c in CUSHION_NAMES]
            return torch.stack(out, -2)
        return s.new_zeros(*s.shape[:-1], 0, DIM)

    def gates(self, logits):
        if self.config.get("gate", "hard") == "soft":
            return logits.sigmoid()
        return (logits >= self.threshold_logit).to(logits.dtype)

    def details(self, state, history=None, gates=None):
        s = state.to(self.state_mean.dtype)
        d1 = self.core_delta(s, history)
        logits = self.logits(s)
        g = self.gates(logits) if gates is None else gates.to(s.dtype)
        responses = self.responses(s)
        contact = (g[..., None] * responses).sum(-2) if responses.shape[-2] else torch.zeros_like(s)
        return s + d1 + contact, {"delta1": d1, "logits": logits, "gates": g, "responses": responses}

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
        out = torch.stack(trajectory, -2)
        return (out, torch.stack(gate_trace, -2) if gate_trace and gate_trace[0].shape[-1] else None) if return_gates else out


def initial_state(config_scene, velocity):
    """[.., 14] launch state; keeps the autograd link to `velocity` [.., 2]."""
    lead = velocity.shape[:-1]
    a = velocity.new_tensor(config_scene["ball_a_xy_m"]).expand(*lead, 2)
    b = velocity.new_tensor(config_scene["ball_b_xy_m"]).expand(*lead, 2)
    zeros3 = velocity.new_zeros(*lead, 3)
    return torch.cat((a, velocity, zeros3, b, velocity.new_zeros(*lead, 2), zeros3), -1)


def load_pool(path, device="cpu"):
    packet = torch.load(path, map_location=device, weights_only=False)
    model = PoolNRD(packet["model_config"], packet["normalization"]).to(device)
    model.load_state_dict(packet["model_state_dict"])
    return model.eval(), packet
