"""Joint NRD dynamics model: [z1, z2, a] tokens -> (delta z1, next z2).

Extends the NeDM temporal backbone (``ContinuousTransformer``) at the input and
output boundaries only, per study plan section 7.2: each token is the
concatenation of the normalized physical state, the camera latent, and the
normalized action; the shared temporal feature feeds a physical head (residual
z1 in normalized target space, the NeDM convention) and a visual head (next z2,
predicted directly in the encoder's LayerNorm'd latent space).

The encoder/decoder live inside the module so one checkpoint carries the whole
surrogate; the decoder is only used when frames are explicitly requested.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from nedm.nrd.vision import ConvDecoder, ConvEncoder, frames_to_float
from nedm.training.model_transformer import ContinuousTransformer, TransformerConfig


class NRDDynamicsModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        transformer_cfg: dict[str, Any],
        vision_cfg: dict[str, Any],
        normalization: dict[str, list[float]],
        state_fields: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.z2_dim = int(transformer_cfg.get("z2_dim", 64))
        self.state_fields = list(state_fields) if state_fields is not None else None

        image_size = int(vision_cfg.get("image_size", 128))
        base_channels = int(vision_cfg.get("base_channels", 32))
        self.encoder = ConvEncoder(self.z2_dim, image_size=image_size, base_channels=base_channels)
        self.decoder = ConvDecoder(self.z2_dim, image_size=image_size, base_channels=base_channels)

        self.backbone = ContinuousTransformer(
            TransformerConfig(
                input_dim=state_dim + self.z2_dim + action_dim,
                block_size=int(transformer_cfg["block_size"]),
                n_layer=int(transformer_cfg["n_layer"]),
                n_head=int(transformer_cfg["n_head"]),
                n_embd=int(transformer_cfg["n_embd"]),
                dropout=float(transformer_cfg["dropout"]),
                bias=bool(transformer_cfg["bias"]),
            )
        )
        hidden_dim = int(transformer_cfg.get("head_hidden_dim", self.backbone.config.n_embd))
        self.state_head = nn.Sequential(
            nn.Linear(self.backbone.config.n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.latent_head = nn.Sequential(
            nn.Linear(self.backbone.config.n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.z2_dim),
        )

        self.register_buffer("state_mean", torch.tensor(normalization["state_mean"], dtype=torch.float32))
        self.register_buffer("state_std", torch.tensor(normalization["state_std"], dtype=torch.float32))
        self.register_buffer("action_mean", torch.tensor(normalization["action_mean"], dtype=torch.float32))
        self.register_buffer("action_std", torch.tensor(normalization["action_std"], dtype=torch.float32))
        self.register_buffer("target_mean", torch.tensor(normalization["target_mean"], dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(normalization["target_std"], dtype=torch.float32))
        # z2 normalization over the training frame distribution. The encoder's
        # LayerNorm'd latents share a LARGE constant component (measured raw
        # pairwise cosine 0.9998 between arbitrary frames, per-dim std 0.013):
        # without this, latent-space losses and metrics are dominated by a
        # trivially predictable constant. Set via set_z2_normalization() after
        # the encoder is warm-started; identity until then.
        self.register_buffer("z2_mean", torch.zeros(self.z2_dim, dtype=torch.float32))
        self.register_buffer("z2_std", torch.ones(self.z2_dim, dtype=torch.float32))

    # -- normalization (NeDM conventions) -----------------------------------
    def normalize_state(self, states: torch.Tensor) -> torch.Tensor:
        return (states - self.state_mean) / self.state_std

    def normalize_action(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.action_mean) / self.action_std

    def normalize_target(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.target_mean) / self.target_std

    def denormalize_target(self, normalized_targets: torch.Tensor) -> torch.Tensor:
        return normalized_targets * self.target_std + self.target_mean

    def normalize_z2(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.z2_mean) / self.z2_std

    def denormalize_z2(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        return normalized_latents * self.z2_std + self.z2_mean

    @torch.no_grad()
    def set_z2_normalization(self, sample_latents: torch.Tensor) -> None:
        """Fit z2_mean/z2_std from encoder outputs over a training-frame sample."""
        self.z2_mean.copy_(sample_latents.mean(dim=0))
        self.z2_std.copy_(sample_latents.std(dim=0).clamp_min(1e-4))

    # -- vision --------------------------------------------------------------
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Float images (N, 3, H, W) in [0, 1] -> latents (N, z2_dim)."""
        return self.encoder(images)

    def encode_frame_sequence(self, frames_uint8: torch.Tensor) -> torch.Tensor:
        """uint8 frames (B, T, H, W, 3) -> latents (B, T, z2_dim)."""
        batch, steps = frames_uint8.shape[:2]
        images = frames_to_float(frames_uint8).flatten(0, 1)
        return self.encoder(images).view(batch, steps, self.z2_dim)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Latents (..., z2_dim) -> float images (..., 3, H, W)."""
        flat = latents.reshape(-1, self.z2_dim)
        images = self.decoder(flat)
        return images.view(*latents.shape[:-1], *images.shape[1:])

    # -- dynamics ------------------------------------------------------------
    def forward(
        self, states: torch.Tensor, latents: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-position predictions over a (B, T, ·) token window.

        ``latents`` are RAW encoder outputs; both token inputs and the visual
        head's output live in the z2-NORMALIZED space. Returns
        (delta_z1 in normalized target space, next z2 in normalized z2 space),
        both (B, T, ·): position t predicts the transition t -> t+1.
        """
        tokens = torch.cat(
            [self.normalize_state(states), self.normalize_z2(latents), self.normalize_action(actions)],
            dim=-1,
        )
        features = self.backbone(tokens)
        return self.state_head(features), self.latent_head(features)

    @torch.no_grad()
    def predict_next(
        self, states: torch.Tensor, latents: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One transition from the last window position: (next z1, next z2).

        ``states``/``latents`` (raw)/``actions`` are (B, T, ·) context windows;
        returns denormalized next state (B, state_dim) and RAW next latent
        (B, z2_dim) -- directly decodable and re-feedable.
        """
        delta_norm, z2_next_norm = self.forward(states, latents, actions)
        next_state = states[:, -1, :] + self.denormalize_target(delta_norm[:, -1, :])
        return next_state, self.denormalize_z2(z2_next_norm[:, -1, :])
