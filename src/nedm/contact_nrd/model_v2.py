"""Unified contact NRD, version 2 (after the design review of 2026-10-02).

Same structure as version 1: a Transformer core (shared over moving bodies)
plus exactly two contact networks shared over all pairs: a collision network
(on/off per pair) and a contact network (state change of each body).

Changes:
* Pair frame, right-handed and tied to gravity. e1 points from body i toward
  the partner: -normal for a plane, the line of centres for a sphere (or, as
  an option for sphere pairs, the horizontal relative-velocity direction, the
  "approach" frame, in which the impact offset is a coordinate). The
  reference axis is world up unless e1 is within 45 degrees of vertical, then
  world +x; e3 = normalise(a - (a.e1) e1), e2 = e3 x e1.
* Relative geometry only: the plane partner is its foot point, so a plane
  pair's relative position is (distance, 0, 0); no absolute positions enter
  the contact networks. Gravity's direction in the pair frame is an input.
* Relative position is compressed as s*tanh(x/s) with s from the training
  data (per pair type), so far pairs map to the edge of the trained band.
* Pair types (sphere-sphere, sphere-plane) get their own input/output
  normalisation, a ridge-fitted linear skip and a learned scale-and-shift of
  every hidden layer inside the one contact network; the collision network
  gets per-type normalisation and a type one-hot.
"""
from __future__ import annotations

import torch
from torch import nn

from nedm.contact_nrd.model import STATE, ContactGraphNRD

FEATURES = 20   # dp(3), v_i(3), w_i(3), v_j(3), w_j(3), g(3), R_i, R_j
TYPES = 2       # 0 sphere-sphere, 1 sphere-plane


class FilmMLP(nn.Module):
    """MLP whose hidden layers get a per-type scale and shift (zero-initialised)."""

    def __init__(self, inputs, hidden, outputs, layers, activation, types):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]
        self.layers = nn.ModuleList([nn.Linear(inputs if k == 0 else hidden, hidden) for k in range(layers)])
        self.act = act()
        self.gamma = nn.Parameter(torch.zeros(layers, types, hidden))
        self.beta = nn.Parameter(torch.zeros(layers, types, hidden))
        self.out = nn.Linear(hidden, outputs)

    def forward(self, x, kind):
        h = x
        for k, layer in enumerate(self.layers):
            h = self.act(layer(h))
            h = h * (1 + self.gamma[k][kind]) + self.beta[k][kind]
        return self.out(h)


class ContactGraphNRDv2(ContactGraphNRD):
    def __init__(self, config, system, normalization):
        super().__init__({**config, "frame": "pair"}, system, {**normalization, **_v1_placeholders()})
        dtype = self.state_mean.dtype
        bodies = system["bodies"]
        kinds = [0 if bodies[j]["kind"] == "sphere" else 1 for _, j in system["pairs"]]
        self.register_buffer("pair_kind", torch.tensor(kinds))
        self.sphere_frame = config.get("sphere_frame", "centerline")
        for key, shape in (("compress", (TYPES,)), ("gate_mean2", (TYPES, FEATURES)), ("gate_std2", (TYPES, FEATURES)),
                           ("in_mean2", (TYPES, FEATURES)), ("in_std2", (TYPES, FEATURES)),
                           ("out_mean2", (TYPES, STATE)), ("out_std2", (TYPES, STATE))):
            default = torch.ones(shape) if ("std" in key or key == "compress") else torch.zeros(shape)
            value = torch.tensor(normalization[key]) if key in normalization else default
            self.register_buffer(key, value.to(dtype))
        gate_hidden, latent = int(config.get("gate_hidden", 256)), int(config.get("gate_latent", 32))
        act = config.get("gate_activation", "relu")
        a = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[act]
        self.collision = nn.Sequential(nn.Linear(FEATURES + TYPES, gate_hidden), a(), nn.Linear(gate_hidden, gate_hidden), a(),
                                       nn.Linear(gate_hidden, latent), a(), nn.Linear(latent, 1)).to(dtype)
        self.contact = FilmMLP(FEATURES, int(config.get("hidden", 1024)), STATE, int(config.get("contact_layers", 3)),
                               config.get("contact_activation", "tanh"), TYPES).to(dtype)
        nn.init.zeros_(self.contact.out.weight)
        nn.init.zeros_(self.contact.out.bias)
        self.contact_skip = nn.ModuleList([nn.Linear(FEATURES, STATE) for _ in range(TYPES)]).to(dtype)

    # ---- frames and features ------------------------------------------------
    def frames_v2(self, ti, tj, kind):
        plane = tj[..., 1:2] > 0.5
        d = tj[..., 3:6] - ti[..., 3:6]
        centre = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        e1 = torch.where(plane, -tj[..., 6:9], centre)
        if self.sphere_frame == "approach":
            rel = ti[..., 9:12] - tj[..., 9:12]
            up = self.up.expand_as(rel)
            horiz = rel - (rel * up).sum(-1, keepdim=True) * up
            norm = horiz.norm(dim=-1, keepdim=True)
            use = (~plane) & (norm > 1e-6)
            e1 = torch.where(use, horiz / norm.clamp_min(1e-12), e1)
        up = self.up.expand_as(e1)
        x_axis = torch.zeros_like(e1)
        x_axis[..., 0] = 1.0
        a = torch.where((e1 * up).sum(-1, keepdim=True).abs() < 0.7071067811865476, up, x_axis)
        e3 = a - (a * e1).sum(-1, keepdim=True) * e1
        e3 = e3 / e3.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        e2 = torch.cross(e3, e1, dim=-1)
        return torch.stack((e1, e2, e3), -1)

    def pair_features_v2(self, ti, tj, kind):
        rot = self.frames_v2(ti, tj, kind)

        def to_frame(v):
            return (v.unsqueeze(-2) @ rot).squeeze(-2)
        plane = tj[..., 1:2] > 0.5
        # Plane partner: its foot point, so the relative position is (distance, 0, 0).
        dist_plane = ((tj[..., 3:6] - ti[..., 3:6]) * -tj[..., 6:9]).sum(-1, keepdim=True)
        dp_plane = torch.cat((dist_plane, torch.zeros_like(dist_plane), torch.zeros_like(dist_plane)), -1)
        dp = torch.where(plane, dp_plane, to_frame(tj[..., 3:6] - ti[..., 3:6]))
        s = self.compress[kind][..., None]
        dp = s * torch.tanh(dp / s)
        g = to_frame(-self.up.expand_as(ti[..., 3:6]))
        feats = torch.cat((dp, to_frame(ti[..., 9:12]), to_frame(ti[..., 12:15]), to_frame(tj[..., 9:12]),
                           to_frame(tj[..., 12:15]), g, ti[..., 2:3], tj[..., 2:3]), -1)
        return feats, rot

    def all_pair_features(self, state):
        tokens = self.tokens(state)
        ti = tokens[..., self.moving, :][..., self.pair_i, :]
        tj = tokens[..., self.pair_j_body, :]
        kind = self.pair_kind
        forward, rot_f = self.pair_features_v2(ti, tj, kind)
        mm = torch.nonzero(self.pair_j_moving).flatten()
        if len(mm):
            backward, rot_b = self.pair_features_v2(tj[..., mm, :], ti[..., mm, :], kind[mm])
        else:
            backward, rot_b = forward[..., :0, :], None
        return forward, rot_f, backward, rot_b, mm

    # ---- networks ----------------------------------------------------------
    def collision_logits(self, forward, backward, mm):
        kind = self.pair_kind

        def run(x, k):
            z = (x - self.gate_mean2[k]) / self.gate_std2[k]
            onehot = torch.nn.functional.one_hot(k, TYPES).to(x.dtype).expand(*x.shape[:-1], TYPES)
            return self.collision(torch.cat((z, onehot), -1))[..., 0]
        logits = run(forward, kind)
        if len(mm):
            logits = logits.clone()
            logits[..., mm] = 0.5 * (logits[..., mm] + run(backward, kind[mm]))
        return logits

    def contact_change_kind(self, features, rot, kind):
        f = (features - self.in_mean2[kind]) / self.in_std2[kind]
        skip = torch.stack([self.contact_skip[t](f) for t in range(TYPES)], -2)
        skip = torch.gather(skip, -2, kind.expand(*f.shape[:-1])[..., None, None].expand(*f.shape[:-1], 1, STATE))[..., 0, :]
        out = (self.contact(f, kind) + skip) * self.out_std2[kind] + self.out_mean2[kind]
        out = torch.cat([(rot @ out[..., k:k + 3].unsqueeze(-1)).squeeze(-1) for k in (0, 3, 6)], -1)
        return out * self.mask

    def contact_terms(self, state):
        forward, rot_f, backward, rot_b, mm = self.all_pair_features(state)
        logits = self.collision_logits(forward, backward, mm)
        d_first = self.contact_change_kind(forward, rot_f, self.pair_kind)
        d_second = self.contact_change_kind(backward, rot_b, self.pair_kind[mm]) if len(mm) else None
        return logits, d_first, d_second, mm


def _v1_placeholders():
    from nedm.contact_nrd.model import PAIR_FEATURES
    return {"gate_mean": [0.0] * PAIR_FEATURES, "gate_std": [1.0] * PAIR_FEATURES, "in_mean": [0.0] * PAIR_FEATURES,
            "in_std": [1.0] * PAIR_FEATURES, "out_mean": [0.0] * STATE, "out_std": [1.0] * STATE}


def load_any(path, device="cpu"):
    packet = torch.load(path, map_location=device, weights_only=False)
    arch = packet["model_config"].get("architecture")
    if arch == "contact_graph_nrd_v5":
        from nedm.contact_nrd.model_v5 import load_v5
        return load_v5(path, device)
    if arch in ("contact_graph_nrd_v3", "contact_graph_nrd_v4"):
        from nedm.contact_nrd.model_v3 import StateContactNRD as cls
    else:
        cls = ContactGraphNRDv2 if arch == "contact_graph_nrd_v2" else ContactGraphNRD
    model = cls(packet["model_config"], packet["system"], packet["normalization"]).to(device)
    model.load_state_dict(packet["model_state_dict"])
    return model.eval(), packet


def retarget(model, system):
    """Give a loaded model a new scene: fixed bodies' descriptions (and the
    physics config used by launch helpers) from `system`. Bodies and pairs must
    match the training system's layout; only fixed-body geometry may differ."""
    import torch as _torch
    if [b["name"] for b in system["bodies"]] != [b["name"] for b in model.system["bodies"]] or system["pairs"] != model.system["pairs"]:
        raise ValueError("retarget needs the same bodies and pairs")
    with _torch.no_grad():
        for k, b in enumerate(system["bodies"]):
            if b["kind"] == "plane":
                model.body_tokens[k, 3:6] = _torch.tensor(b["point"], dtype=model.body_tokens.dtype)
                model.body_tokens[k, 6:9] = _torch.tensor(b["normal"], dtype=model.body_tokens.dtype)
    model.system = system
    return model
