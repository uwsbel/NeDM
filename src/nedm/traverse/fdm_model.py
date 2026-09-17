"""Direct, controller-conditioned finite-horizon trajectory outcome predictor.

Inputs are causal history and a *proposed* path/speed profile, never future
executed controls. Outputs are physical ego poses, cumulative positive shaft
work in kJ, and prior-corrected event logits. This model does not feed its predictions
back into itself. The pilot uses privileged terrain profile features; it is
not a camera deployment model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


EVENT_NAMES = ("contact", "rollover", "low_progress")
ARMS = ("profile", "no_history", "history", "no_terrain")


@dataclass
class FDMConfig:
    history_dim: int = 24
    candidate_dim: int = 13
    global_dim: int = 4
    horizon: int = 20
    dt: float = 0.2
    arm: str = "history"
    terrain_indices: tuple[int, ...] = (5, 6, 7, 8, 9, 10)
    hidden_dim: int = 64
    profile_bins: int = 8
    work_unit: str = "kJ"

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"Unknown arm {self.arm!r}; expected {ARMS}")
        if self.horizon < 1 or self.hidden_dim < 1 or self.profile_bins < 1:
            raise ValueError("Horizon, hidden dimension, and profile bins must be positive")
        if any(i < 0 or i >= self.candidate_dim for i in self.terrain_indices):
            raise ValueError("Terrain ablation indices lie outside candidate features")


class FiniteHorizonFDM(nn.Module):
    """Batched predictor with all normalization stored in checkpoint buffers.

    ``forward`` takes ``history[B,L,Dh]``, ``candidate[B,K,Dc]``,
    ``global_features[B,Dg]``, and ``nominal_pose[B,H,4]``. The pose encoding
    is ``x,y,sin(yaw),cos(yaw)`` in the anchor vehicle frame. All tensors are
    physical, unnormalized inputs. Outputs use keys ``trajectory``, ``work``,
    and ``event_logits``; sigmoid of the last gives per-step event probability.

    ``profile`` is a small direct MLP baseline retaining current state and an
    ordered path profile. ``no_history`` retains precisely the same last
    state row as the full model. ``no_terrain`` removes only declared terrain
    columns, retaining route geometry, clearance, speed, state, and history.
    """

    def __init__(self, config: FDMConfig, normalization: dict[str, Any] | None = None):
        super().__init__()
        self.config = config
        norm = normalization or {}
        for key, dim in (("history", config.history_dim), ("candidate", config.candidate_dim),
                         ("global_features", config.global_dim)):
            values = norm.get(key, {})
            mean = torch.as_tensor(values.get("mean", [0.0] * dim), dtype=torch.float32)
            std = torch.as_tensor(values.get("std", [1.0] * dim), dtype=torch.float32)
            if mean.shape != (dim,) or std.shape != (dim,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                raise ValueError(f"Invalid {key} normalization; expected {dim} finite means/stds")
            self.register_buffer(f"{key}_mean", mean)
            self.register_buffer(f"{key}_std", std.clamp_min(1e-6))
        self.register_buffer("xy_scale", torch.tensor(float(norm.get("xy_scale", 1.0))).clamp_min(0.1))
        self.register_buffer("work_scale", torch.tensor(float(norm.get("work_scale", 1.0))).clamp_min(1.0))
        # With weighted BCE, training uses logits + log(pos_weight). Reporting
        # these unshifted logits removes the known class-weight prior shift.
        pos_weight = torch.as_tensor(norm.get("event_pos_weight", [1.0] * len(EVENT_NAMES)), dtype=torch.float32)
        if pos_weight.shape != (3,) or not torch.isfinite(pos_weight).all() or (pos_weight <= 0).any():
            raise ValueError("event_pos_weight must contain three positive finite numbers")
        self.register_buffer("event_pos_weight", pos_weight)
        supported = torch.as_tensor(norm.get("supported_events", [True, True, True]), dtype=torch.bool)
        if supported.shape != (3,):
            raise ValueError("supported_events must identify the three event heads")
        self.register_buffer("supported_events", supported)
        h = config.hidden_dim
        if config.arm == "profile":
            input_dim = config.history_dim + config.profile_bins * config.candidate_dim + config.global_dim
            self.backbone = nn.Sequential(nn.Linear(input_dim, h * 2), nn.SiLU(), nn.Linear(h * 2, h), nn.SiLU())
            decoder_dim = h
        else:
            self.history_encoder = nn.GRU(config.history_dim, h, batch_first=True)
            self.path_encoder = nn.Sequential(
                nn.Conv1d(config.candidate_dim, h, 5, padding=2), nn.SiLU(),
                nn.Conv1d(h, h, 5, padding=2), nn.SiLU(),
                nn.AdaptiveAvgPool1d(config.profile_bins), nn.Flatten(),
                nn.Linear(h * config.profile_bins, h * 2), nn.SiLU(),
            )
            self.backbone = nn.Sequential(
                nn.Linear(h * 3 + config.global_dim, h * 4), nn.SiLU(),
                nn.Linear(h * 4, h * 2), nn.SiLU(),
            )
            decoder_dim = h * 2
        self.output_head = nn.Linear(decoder_dim, config.horizon * 8)
        nn.init.normal_(self.output_head.weight, std=0.001)
        nn.init.zeros_(self.output_head.bias)
        with torch.no_grad():
            self.output_head.bias.reshape(config.horizon, 8)[:, 4] = -1.0

    def normalize_inputs(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history = (batch["history"] - self.history_mean) / self.history_std
        candidate = (batch["candidate"] - self.candidate_mean) / self.candidate_std
        if self.config.candidate_dim == 13:
            # The valid flag is part of the public data contract. Padding must
            # not turn a constant feature's small standard deviation into a
            # large spurious signal near the route endpoint.
            valid = batch["candidate"][..., 12:13] > 0
            candidate = torch.where(valid, candidate, torch.zeros_like(candidate))
            candidate = torch.cat((candidate[..., :12], valid.to(candidate.dtype)), dim=-1)
        global_features = (batch["global_features"] - self.global_features_mean) / self.global_features_std
        if self.config.arm == "no_history":
            history = history[:, -1:, :]
        if self.config.arm == "no_terrain":
            candidate = candidate.clone()
            candidate[..., list(self.config.terrain_indices)] = 0.0
        return history, candidate, global_features

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        history, candidate, global_features = self.normalize_inputs(batch)
        if self.config.arm == "profile":
            profile = F.adaptive_avg_pool1d(candidate.transpose(1, 2), self.config.profile_bins).flatten(1)
            context = torch.cat((history[:, -1], profile, global_features), dim=-1)
        else:
            _, hidden = self.history_encoder(history)
            profile = self.path_encoder(candidate.transpose(1, 2))
            context = torch.cat((hidden[-1], profile, global_features), dim=-1)
        raw = self.output_head(self.backbone(context)).reshape(-1, self.config.horizon, 8)
        nominal = batch["nominal_pose"]
        xy = nominal[..., :2] + raw[..., :2] * self.xy_scale
        yaw_vector = nominal[..., 2:4] + raw[..., 2:4]
        # nominal is a valid unit direction; zero fallback avoids undefined yaw.
        yaw_norm = torch.linalg.vector_norm(yaw_vector, dim=-1, keepdim=True)
        heading = torch.where(yaw_norm > 1e-6, yaw_vector / yaw_norm.clamp_min(1e-6), nominal[..., 2:4])
        work_increment = F.softplus(raw[..., 4:5]) * (self.work_scale / self.config.horizon)
        return {
            "trajectory": torch.cat((xy, heading), dim=-1),
            "work": torch.cumsum(work_increment, dim=1),
            "event_logits": raw[..., 5:8],
        }

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    mask = mask.expand_as(value).to(value.dtype)
    if sample_weight is not None:
        mask = mask * sample_weight.reshape(-1, *([1] * (value.ndim - 1)))
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def fdm_loss(model: FiniteHorizonFDM, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
             weights: dict[str, float] | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Masked regression and event supervision, with optional sampling correction."""
    weights = weights or {"xy": 1.0, "yaw": 0.5, "work": 0.2, "events": 1.0}
    mask = batch["trajectory_mask"]
    sample_weight = batch.get("sample_weight")
    xy = F.smooth_l1_loss(output["trajectory"][..., :2] / model.xy_scale,
                          batch["trajectory"][..., :2] / model.xy_scale, reduction="none")
    yaw = (output["trajectory"][..., 2:4] - batch["trajectory"][..., 2:4]).square()
    work = F.smooth_l1_loss(output["work"] / model.work_scale, batch["work"] / model.work_scale, reduction="none")
    event = F.binary_cross_entropy_with_logits(
        output["event_logits"] + model.event_pos_weight.log(), batch["events"],
        pos_weight=model.event_pos_weight, reduction="none",
    )
    losses = {
        "xy": _masked_mean(xy, mask, sample_weight),
        "yaw": _masked_mean(yaw, mask, sample_weight),
        "work": _masked_mean(work, batch.get("work_mask", mask), sample_weight),
        "events": _masked_mean(event, batch["event_mask"] * model.supported_events.to(event.dtype), sample_weight),
    }
    total = sum(weights.get(name, 0.0) * value for name, value in losses.items())
    return total, {**losses, "total": total}


def load_fdm_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[FiniteHorizonFDM, dict[str, Any]]:
    """Load a trusted local training checkpoint for evaluation or MPPI scoring."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    config = FDMConfig(**checkpoint["model_config"])
    model = FiniteHorizonFDM(config, checkpoint["normalization"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def model_config_dict(model: FiniteHorizonFDM) -> dict[str, Any]:
    return asdict(model.config)
