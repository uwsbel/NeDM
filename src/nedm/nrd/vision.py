"""Camera encoder/decoder for NRD (study plan sections 7.1 and 7.3).

Deterministic convolutional autoencoder trained from scratch on the Chrono
camera distribution. GroupNorm + SiLU after each convolution; the latent is
LayerNorm'd so its scale is fixed for the downstream transition model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
        nn.SiLU(),
    )


def _deconv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
        nn.SiLU(),
    )


class ConvEncoder(nn.Module):
    """RGB (B, 3, H, W) in [0, 1] -> z2 (B, z2_dim). H = W = image_size."""

    def __init__(self, z2_dim: int = 64, image_size: int = 128, base_channels: int = 32) -> None:
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError(f"image_size must be divisible by 16, got {image_size}")
        self.z2_dim = z2_dim
        self.image_size = image_size
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.conv = nn.Sequential(
            _conv_block(3, channels[0]),
            _conv_block(channels[0], channels[1]),
            _conv_block(channels[1], channels[2]),
            _conv_block(channels[2], channels[3]),
        )
        self.spatial = image_size // 16
        flat_dim = channels[3] * self.spatial * self.spatial
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.SiLU(),
            nn.Linear(256, z2_dim),
            nn.LayerNorm(z2_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(images))


class ConvDecoder(nn.Module):
    """z2 (B, z2_dim) -> RGB (B, 3, H, W) in [0, 1]."""

    def __init__(self, z2_dim: int = 64, image_size: int = 128, base_channels: int = 32) -> None:
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError(f"image_size must be divisible by 16, got {image_size}")
        self.z2_dim = z2_dim
        self.image_size = image_size
        channels = [base_channels * 8, base_channels * 4, base_channels * 2, base_channels]
        self.spatial = image_size // 16
        self.head = nn.Sequential(
            nn.Linear(z2_dim, channels[0] * self.spatial * self.spatial),
            nn.SiLU(),
        )
        self.deconv = nn.Sequential(
            _deconv_block(channels[0], channels[1]),
            _deconv_block(channels[1], channels[2]),
            _deconv_block(channels[2], channels[3]),
            nn.ConvTranspose2d(channels[3], 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self._channels0 = channels[0]

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        features = self.head(latents)
        features = features.view(-1, self._channels0, self.spatial, self.spatial)
        return self.deconv(features)


def frames_to_float(frames_uint8: torch.Tensor) -> torch.Tensor:
    """(..., H, W, 3) uint8 -> (..., 3, H, W) float in [0, 1]."""
    frames = frames_uint8.to(torch.float32) / 255.0
    return frames.movedim(-1, -3)


def frames_to_uint8(frames_float: torch.Tensor) -> torch.Tensor:
    """(..., 3, H, W) float in [0, 1] -> (..., H, W, 3) uint8."""
    frames = (frames_float.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return frames.movedim(-3, -1)
