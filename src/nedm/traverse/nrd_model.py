"""WP2 NRD dynamics model with the ego-indexed scene-map token (plan v1.3 §5/§8.1).

Shared by the WP2 trainer (``scripts/traverse_wp2_train_map.py``) and the WP3
tracker imagination env (``nedm.traverse.tracker_env``), so the checkpoint
format and the rollout conventions live in exactly one place:

* ``z1`` is the 15-D ``tire_normal_force_omega`` preset, kept NORMALIZED inside
  the model and its rollouts (the head predicts the delta of normalized z1);
  physical units are needed only for dead reckoning and rewards.
* the sensor token is ``MapCropper(scene_map, pose)``: an 8x8 ego-aligned crop
  of the frozen encoder's stage-2 map at the vehicle's (dead-reckoned) pose.
* pose ``(x, y, yaw)`` is integrated outside the network from vx, vy, yaw rate
  with the repo's semi-implicit convention (yaw first, then the world velocity
  uses the new yaw) -- the same function the offline evaluator uses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from nedm.traverse import nrd_data as D
from nedm.traverse.map_crop import MapCropper
from nedm.training.model_transformer import ContinuousTransformer, TransformerConfig

DT_S = 0.05
VX, VY, YAW_RATE = 0, 1, 6  # indices into the tire_normal_force_omega preset
ROLL, PITCH = 2, 3
POSE_CHANNELS = [0, 1, 6]  # the only channels dead reckoning reads
TERRAIN_CHANNELS = [2, 3, 4, 5, 7, 8, 9, 10]  # attitude + tire normal loads


def integrate_pose(pose: torch.Tensor, z1_phys: torch.Tensor) -> torch.Tensor:
    """(B, 3) pose, (B, 15) PHYSICAL-unit state -> (B, 3) next pose."""
    yaw = pose[:, 2] + DT_S * z1_phys[:, YAW_RATE]
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    vx_world = cos_yaw * z1_phys[:, VX] - sin_yaw * z1_phys[:, VY]
    vy_world = sin_yaw * z1_phys[:, VX] + cos_yaw * z1_phys[:, VY]
    return torch.stack([pose[:, 0] + DT_S * vx_world, pose[:, 1] + DT_S * vy_world, yaw], dim=1)


class WP2MapModel(nn.Module):
    """Backbone over [z1, crop-token, a]; heads: delta z1, aux power, optional next-token."""

    def __init__(self, z1_dim: int, act_dim: int, cfg: dict, arena_dir: Path,
                 token_dim: int = 256, predict_token: bool = False):
        super().__init__()
        self.cropper = MapCropper(arena_dir, out_dim=token_dim)
        self.token_dim = token_dim
        self.backbone = ContinuousTransformer(TransformerConfig(
            input_dim=z1_dim + token_dim + act_dim, block_size=int(cfg["block_size"]),
            n_layer=int(cfg["n_layer"]), n_head=int(cfg["n_head"]), n_embd=int(cfg["n_embd"]),
            dropout=float(cfg["dropout"]), bias=bool(cfg["bias"])))
        hidden, n_embd = int(cfg["head_hidden_dim"]), int(cfg["n_embd"])
        mlp = lambda out: nn.Sequential(nn.Linear(n_embd, hidden), nn.GELU(), nn.Linear(hidden, out))
        self.state_head, self.power_head = mlp(z1_dim), mlp(1)
        self.token_head = mlp(token_dim) if predict_token else None
        # Prediction-head target statistics (plan section 5's normalization footgun,
        # applied to the crop): the head predicts the NEXT crop in normalized space
        # so the token loss carries a fixed weight instead of the crop's raw scale.
        # Meaningful only when the cropper is frozen (two-stage predict).
        self.register_buffer("tok_mean", torch.zeros(token_dim))
        self.register_buffer("tok_std", torch.ones(token_dim))

    def forward(self, z1, token, act):
        feat = self.backbone(torch.cat([z1, token, act], dim=-1))
        return (self.state_head(feat), self.power_head(feat),
                self.token_head(feat) if self.token_head is not None else None)


def load_map_model(ckpt_path: Path | str, arena_dir: Path | str, device: str | torch.device,
                   z1_dim: int = 15, act_dim: int = 3) -> tuple[WP2MapModel, D.Normalizer, dict]:
    """Load a ``traverse_wp2_train_map.py`` checkpoint: (frozen model, normalizer, payload)."""
    payload = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    predict = payload.get("map_mode") == "predict"
    token_dim = int(payload["model"]["cropper.fc.weight"].shape[0])
    model = WP2MapModel(z1_dim, act_dim, cfg, Path(arena_dir), token_dim, predict_token=predict)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    allowed = {"tok_mean", "tok_std"}  # buffers added 2026-09-04; older index ckpts lack them
    if unexpected or (set(missing) - allowed):
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    norm = D.Normalizer.from_dict(payload["normalization"])
    return model, norm, payload
