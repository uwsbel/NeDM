"""Five-state differentiable NRD with learned floor and wall response.

The shared NeDM transformer processes the Markov state [x,z,vx,vz,omega_y].
Known geometry and gravity provide a free-flight prior and smooth contact
localization. The impulse architecture learns three velocity/spin changes
per contact and two gate calibrations, then integrates the impulses to update
position. The earlier independent five-state corrections remain loadable.
Restitution and friction are not supplied to the network. There are no true
contacts, future states, launch labels, targets, state clamps, or simulator
calls in inference.
"""
from __future__ import annotations

import torch
from torch import nn

from nedm.core.training.model_transformer import ContinuousTransformer, TransformerConfig


class BouncingBallNRD(nn.Module):
    def __init__(self, config: dict, normalization: dict):
        super().__init__()
        self.config = config
        self.dt = float(config["dt_s"])
        self.physics_dt = float(config["physics_step_s"])
        self.gravity = float(config["gravity_mps2"])
        self.floor_center_z = float(config["floor_center_z_m"])
        self.wall_center_x = float(config["wall_center_x_m"])
        self.width = float(config["gate_width_m"])
        self.impulse_model = config.get("architecture") == "contact_impulse_v2"
        self.register_buffer("state_mean", torch.tensor(normalization["mean"], dtype=torch.float32))
        self.register_buffer("state_std", torch.tensor(normalization["std"], dtype=torch.float32))
        self.register_buffer("correction_scale", torch.tensor([0.05, 0.1, 2, 4, 10], dtype=torch.float32))
        self.backbone = ContinuousTransformer(TransformerConfig(
            input_dim=7 if self.impulse_model else 5, block_size=1, n_layer=config["layers"], n_head=4,
            n_embd=config["embedding"], dropout=0.0, bias=True))
        self.head = nn.Sequential(nn.Linear(config["embedding"], 128), nn.GELU(), nn.Linear(128, 8 if self.impulse_model else 12))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def step_with_logits(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Chrono uses semi-implicit substeps. The extra physics_dt term matches
        # its gravity integration over a model interval, including raw timing.
        flight_delta = torch.stack([
            state[..., 2]*self.dt,
            state[..., 3]*self.dt - 0.5*self.gravity*self.dt*(self.dt+self.physics_dt),
            torch.zeros_like(state[..., 2]),
            torch.full_like(state[..., 3], -self.gravity*self.dt),
            torch.zeros_like(state[..., 4]),
        ], dim=-1)
        flight = state + flight_delta
        normalized = (state-self.state_mean)/self.state_std
        if self.impulse_model:
            # Integrate a learned instantaneous velocity/spin impulse at the
            # estimated impact time. This ties position to velocity and gives
            # launch gradients the correct time-of-impact dependence.
            discriminant = state[...,3].square() + 2*self.gravity*(state[...,1]-self.floor_center_z)
            ground_time = (state[...,3]+discriminant.clamp_min(1e-6).sqrt())/self.gravity
            denominator = torch.where(state[...,2].abs()>1e-3, state[...,2], torch.full_like(state[...,2],1e-3))
            wall_time = (self.wall_center_x-state[...,0])/denominator
            delay = 1.5*self.physics_dt
            time = torch.stack([ground_time+delay, wall_time+delay],dim=-1)
            normalized = torch.cat([normalized, torch.tanh((self.dt-time)/self.dt)],dim=-1)
        features = self.backbone(normalized.unsqueeze(-2))[..., 0, :]
        raw = self.head(features)
        if self.impulse_model:
            closing = torch.stack([-state[...,3],state[...,2]],dim=-1)
            prior = torch.stack([-(flight[...,1]-self.floor_center_z+closing[...,0]*delay)/self.width,
                                 (flight[...,0]-self.wall_center_x-closing[...,1]*delay)/self.width],dim=-1)
            logits = prior + 10*torch.tanh(raw[...,6:]/10) + torch.nn.functional.logsigmoid(closing/0.03)
            impulse = raw[...,:6].reshape(*state.shape[:-1],2,3)*self.correction_scale[2:]
            remaining = (self.dt-time).clamp(0,self.dt)
            position = impulse[...,:2]*remaining.unsqueeze(-1)
            correction = torch.cat([position,impulse],dim=-1)
            return flight + (correction*logits.sigmoid().unsqueeze(-1)).sum(-2), logits
        prior = torch.stack([-(flight[..., 1]-self.floor_center_z)/self.width,
                             (flight[..., 0]-self.wall_center_x)/self.width], dim=-1)
        logits = prior + 10*torch.tanh(raw[..., 10:]/10)
        correction = raw[..., :10].reshape(*state.shape[:-1], 2, 5)*self.correction_scale
        prediction = flight + (correction * logits.sigmoid().unsqueeze(-1)).sum(-2)
        return prediction, logits

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.step_with_logits(state)[0]

    def rollout(self, initial: torch.Tensor, steps: int) -> torch.Tensor:
        state, trajectory = initial, [initial]
        for _ in range(steps):
            state = self(state)
            trajectory.append(state)
        return torch.stack(trajectory, dim=-2)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["model_config"].get("architecture") == "transformer_temporal_contact_v2":
        from nedm.bouncing_ball.transformer_temporal import TemporalContactNRD
        model = TemporalContactNRD(checkpoint["model_config"], checkpoint["normalization"]).to(device)
    elif checkpoint["model_config"].get("architecture") == "transformer_contact_residual_v1":
        from nedm.bouncing_ball.transformer_contact import TransformerContactNRD
        model = TransformerContactNRD(checkpoint["model_config"], checkpoint["normalization"]).to(device)
    elif checkpoint["model_config"].get("architecture") == "learned_contact_latent_v1":
        from nedm.bouncing_ball.latent_model import LatentContactNRD
        model = LatentContactNRD(checkpoint["model_config"], checkpoint["normalization"]).to(device)
    elif checkpoint["model_config"].get("architecture") == "event_nrd_v3":
        from nedm.bouncing_ball.precision import EventNRD
        model = EventNRD(checkpoint["model_config"], checkpoint["normalization"]).to(device)
    else:
        model = BouncingBallNRD(checkpoint["model_config"], checkpoint["normalization"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
