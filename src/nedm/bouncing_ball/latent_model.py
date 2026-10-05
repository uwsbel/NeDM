"""State-only learned contact models; no analytical contact detector.

Public inference consumes only [x,z,vx,vz,omega_y]. Fixed gravity integration
is retained as a controlled flight prior. Geometry, true contacts, future
states and time-to-impact features are deliberately absent from forward.
"""
from __future__ import annotations

import copy
import torch
from torch import nn


class ContactLatent(nn.Module):
    """State -> 16-dimensional representation -> occurrence and phase heads."""
    def __init__(self, config, normalization):
        super().__init__()
        self.input_dim = int(config.get("contact_input_dim", 5))
        self.latent_dim = int(config.get("latent_dim", 16))
        width = int(config.get("hidden", 128))
        self.register_buffer("mean", torch.tensor(normalization["mean"], dtype=torch.float64))
        self.register_buffer("std", torch.tensor(normalization["std"], dtype=torch.float64))
        self.encoder = nn.Sequential(nn.Linear(self.input_dim, width), nn.Tanh(),
                                     nn.Linear(width, width), nn.Tanh(),
                                     nn.Linear(width, self.latent_dim), nn.Tanh())
        self.contact_head = nn.Linear(self.latent_dim, 2)
        self.phase_head = nn.Sequential(nn.Linear(self.latent_dim, 64), nn.Tanh(), nn.Linear(64, 2))
        nn.init.zeros_(self.phase_head[-1].weight)
        nn.init.constant_(self.phase_head[-1].bias, .5)
        self.double()

    def forward(self, state):
        state = state.to(dtype=self.mean.dtype)
        normalized = (state-self.mean)/self.std
        feature = normalized[..., :2] if self.input_dim == 2 else normalized
        latent = self.encoder(feature)
        return latent, self.contact_head(latent), self.phase_head(latent)


class FrozenBounceResponses(nn.Module):
    """Only the existing learned response weights, with no event geometry."""
    def __init__(self, config):
        super().__init__()
        width = int(config["response_hidden"])
        self.register_buffer("feature_mean", torch.zeros(2, 3, dtype=torch.float64))
        self.register_buffer("feature_std", torch.ones(2, 3, dtype=torch.float64))
        self.register_buffer("response_scale", torch.ones(2, 3, dtype=torch.float64))
        self.linear = nn.ModuleList([nn.Linear(3, 3) for _ in range(2)])
        self.residual = nn.ModuleList([nn.Sequential(nn.Linear(3, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 3)) for _ in range(2)])
        self.double()
        for p in self.parameters():
            p.requires_grad_(False)

    def initialize(self, precision):
        assert precision.feature_dim == 3 and precision.config["activation"] == "tanh"
        with torch.no_grad():
            for name in ("feature_mean", "feature_std", "response_scale"):
                getattr(self, name).copy_(getattr(precision, name))
        self.linear.load_state_dict(precision.linear.state_dict())
        self.residual.load_state_dict(precision.residual.state_dict())

    def forward(self, velocity, kind):
        feature = (velocity-self.feature_mean[kind])/self.feature_std[kind]
        return (self.linear[kind](feature)+self.residual[kind](feature))*self.response_scale[kind]


class LatentContactNRD(nn.Module):
    """Stage 1: learned contact occurrence/phase + frozen response laws.

    Stage 2: the same state encoder, but one shared five-output decoder replaces
    both response networks. The only fixed dynamics are free flight.
    """
    def __init__(self, config, normalization):
        super().__init__()
        self.config = config
        self.dt = float(config["dt_s"])
        self.gravity = float(config["gravity_mps2"])
        self.physics_dt = float(config["physics_step_s"])
        self.contact = ContactLatent(config, normalization)
        self.register_buffer("correction_scale", torch.tensor(config.get("output_scale",[.5, 1., 15., 25., 100.]), dtype=torch.float64))
        if config["stage"] == "response":
            self.responses = FrozenBounceResponses(config)
        else:
            width = int(config.get("hidden", 128))
            # One decoder, no surface-specific experts or analytical timing.
            n = 5+config.get("latent_dim", 16)+2
            self.decoder = nn.Sequential(nn.Linear(n, width), nn.Tanh(),
                nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 5))
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)
            if config["stage"] == "unified_gated":
                self.linear_core = nn.Linear(n, 5)
                if config.get("freeze_affine",True):
                    for parameter in self.linear_core.parameters(): parameter.requires_grad_(False)
        if config.get("freeze_contact",False):
            for parameter in self.contact.parameters(): parameter.requires_grad_(False)
        self.double()

    def flight(self, state, duration=None):
        t = self.dt if duration is None else duration
        t = torch.as_tensor(t, dtype=state.dtype, device=state.device)
        zeros = torch.zeros_like(t)
        return torch.stack((state[..., 0]+state[..., 2]*t,
            state[..., 1]+state[..., 3]*t-.5*self.gravity*t*(t+self.physics_dt),
            state[..., 2]+zeros, state[..., 3]-self.gravity*t, state[..., 4]+zeros), -1)

    def details(self, state):
        state = state.to(dtype=self.contact.mean.dtype)
        latent, logits, raw_phase = self.contact(state)
        probabilities = logits.sigmoid()
        if self.config.get("gate", "hard") == "hard":
            gates = (logits >= 0).to(state.dtype)
        else:
            gates = probabilities
        phase = (raw_phase.clamp(0., 1.)*self.dt if self.config.get("learned_phase", True)
                 else torch.full_like(raw_phase, .5*self.dt))
        free = self.flight(state)
        if self.config["stage"] == "response":
            correction = torch.zeros_like(state)
            for kind in (0, 1):
                at_velocity = state[..., 2:].clone()
                at_velocity = at_velocity+torch.stack((torch.zeros_like(phase[..., kind]),
                    -self.gravity*phase[..., kind], torch.zeros_like(phase[..., kind])), -1)
                impulse = self.responses(at_velocity, kind)
                change = torch.cat((impulse[..., :2]*(self.dt-phase[..., kind, None]), impulse), -1)
                correction = correction+gates[..., kind, None]*change
        else:
            normalized = (state-self.contact.mean)/self.contact.std
            features = torch.cat((normalized, latent, probabilities), -1)
            if self.config["stage"] == "unified_gated":
                correction = (self.linear_core(features)+self.decoder(features))*self.correction_scale
                correction = correction*gates.sum(-1,keepdim=True).clamp_max(1.)
            else:
                correction = self.decoder(features)*self.correction_scale
        return free+correction, {"latent":latent, "logits":logits, "probabilities":probabilities,
                                 "phase_s":phase, "raw_phase":raw_phase, "gates":gates}

    def forward(self, state):
        return self.details(state)[0]

    def rollout(self, initial, steps):
        state = initial.to(dtype=self.contact.mean.dtype)
        trajectory = [state]
        for _ in range(steps):
            state = self(state)
            trajectory.append(state)
        return torch.stack(trajectory, -2)


def checkpoint(model, runtime, update, validation):
    return {"model_config":model.config, "normalization":{"mean":model.contact.mean.cpu().tolist(),
        "std":model.contact.std.cpu().tolist()}, "model_state_dict":model.state_dict(),
        "runtime":runtime, "update":update, "validation":validation}


def load_latent(path, device="cpu"):
    packet = torch.load(path, map_location=device, weights_only=False)
    model = LatentContactNRD(packet["model_config"], packet["normalization"]).to(device)
    model.load_state_dict(packet["model_state_dict"])
    return model.eval(), packet
