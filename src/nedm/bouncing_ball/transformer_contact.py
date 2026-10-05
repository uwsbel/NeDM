"""Transformer state residual plus a state-inferred neural contact residual.

The primary model follows the requested split exactly: state[5] -> existing
ContinuousTransformer -> delta1[5]; x,z[2] -> latent[16] -> contact[1];
vx,vz[2] -> bounce MLP -> delta2[5]. No flight equation, geometry, event time,
episode clock, true contact or future state is consumed by public inference.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from nedm.core.training.model_transformer import ContinuousTransformer, TransformerConfig


class TransformerContactNRD(nn.Module):
    def __init__(self, config, normalization):
        super().__init__()
        self.config = dict(config)
        self.dt = float(config["dt_s"])
        dtype = torch.float64 if config.get("dtype", "float64") == "float64" else torch.float32
        self.register_buffer("state_mean", torch.tensor(normalization["mean"], dtype=dtype))
        self.register_buffer("state_std", torch.tensor(normalization["std"], dtype=dtype))
        self.register_buffer("delta_mean", torch.tensor(normalization["delta_mean"], dtype=dtype))
        self.register_buffer("delta_scale", torch.tensor(config.get("delta_scale", [.05,.05,2.,4.,10.]), dtype=dtype))
        self.backbone = ContinuousTransformer(TransformerConfig(
            input_dim=5, block_size=1, n_layer=config.get("layers", 2), n_head=4,
            n_embd=config.get("embedding", 64), dropout=0., bias=True))
        self.core_head = nn.Sequential(nn.Linear(config.get("embedding",64),128), nn.GELU(), nn.Linear(128,5))
        nn.init.zeros_(self.core_head[-1].weight)
        nn.init.zeros_(self.core_head[-1].bias)
        self.contact_enabled = bool(config.get("contact_enabled",True))
        self.gate_input_dim = int(config.get("gate_input_dim",2))
        self.modes = int(config.get("contact_modes",1))
        if self.contact_enabled:
            self.contact_encoder = nn.Sequential(nn.Linear(self.gate_input_dim,64),nn.Tanh(),
                nn.Linear(64,64),nn.Tanh(),nn.Linear(64,16),nn.Tanh())
            self.contact_head = nn.Linear(16,self.modes)
            self.bounce = nn.ModuleList([nn.Sequential(nn.Linear(2,128),nn.Tanh(),
                nn.Linear(128,128),nn.Tanh(),nn.Linear(128,5)) for _ in range(self.modes)])
            for net in self.bounce:
                nn.init.zeros_(net[-1].weight)
                nn.init.zeros_(net[-1].bias)
        self.to(dtype=dtype)

    def core_delta(self, state):
        state = state.to(dtype=self.state_mean.dtype)
        normalized = (state-self.state_mean)/self.state_std
        embedding = self.backbone(normalized.unsqueeze(-2))[...,0,:]
        return self.core_head(embedding)*self.delta_scale+self.delta_mean

    def contact_features(self, state):
        normalized = (state.to(dtype=self.state_mean.dtype)-self.state_mean)/self.state_std
        return normalized[...,:2] if self.gate_input_dim==2 else normalized

    def bounce_features(self, state):
        return ((state.to(dtype=self.state_mean.dtype)-self.state_mean)/self.state_std)[...,2:4]

    def components(self, state):
        """Training can supervise these outputs without altering public routing."""
        state = state.to(dtype=self.state_mean.dtype)
        delta1 = self.core_delta(state)
        if self.contact_enabled:
            latent = self.contact_encoder(self.contact_features(state))
            logits = self.contact_head(latent)
            probabilities = logits.sigmoid()
            deltas = torch.stack([net(self.bounce_features(state))*self.delta_scale for net in self.bounce],-2)
        else:
            latent = torch.zeros((*state.shape[:-1],16),device=state.device,dtype=state.dtype)
            logits = torch.full((*state.shape[:-1],1),-100.,device=state.device,dtype=state.dtype)
            probabilities = torch.zeros_like(logits)
            deltas = torch.zeros((*state.shape[:-1],1,5),device=state.device,dtype=state.dtype)
        threshold=float(self.config.get("gate_threshold",.5))
        logit_threshold=math.log(threshold/(1-threshold))
        gates = (logits>=logit_threshold).to(state.dtype) if self.config.get("gate","hard")=="hard" else probabilities
        return {"delta1":delta1,"delta2_by_mode":deltas,"latent":latent,
                "logits":logits,"probabilities":probabilities,"gates":gates}

    def details(self, state):
        state = state.to(dtype=self.state_mean.dtype)
        info = self.components(state)
        delta2 = (info["gates"].unsqueeze(-1)*info["delta2_by_mode"]).sum(-2)
        return state+info["delta1"]+delta2, info

    def forward(self, state):
        return self.details(state)[0]

    def rollout(self, initial, steps):
        current = initial.to(dtype=self.state_mean.dtype)
        trajectory = [current]
        for _ in range(steps):
            current = self(current)
            trajectory.append(current)
        return torch.stack(trajectory,-2)


def load_transformer_contact(path, device="cpu"):
    packet = torch.load(path,map_location=device,weights_only=False)
    model = TransformerContactNRD(packet["model_config"],packet["normalization"]).to(device)
    model.load_state_dict(packet["model_state_dict"])
    return model.eval(),packet
