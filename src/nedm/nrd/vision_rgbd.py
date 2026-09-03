"""RGB-D camera encoder + mandatory auxiliary warm-up heads (study plan §5).

DRAFT for WP1 de-risking — not wired into `model.py` yet.

Extends Study 1's `ConvEncoder` per plan §5: 4 input channels (RGB + normalized
elevation) at 256², one extra stride-2 stage, `z2_dim=128`.

The pre-pooling spatial feature map is a first-class output rather than a local
inside `forward()`, because `open-questions.md` ("Does a global pooled z2 survive
at 256²?") is settled by running the occupancy and localization probes from both
that map and the global latent. `forward()` still returns the latent alone, so
this stays drop-in for `model.py`'s `self.encoder(frames)` call.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class EncoderOutput(NamedTuple):
    """`z2` is the global latent; `features` is the pre-pooling map (B, C, S, S)."""

    z2: torch.Tensor
    features: torch.Tensor


class ConvEncoderRGBD(nn.Module):
    """RGB-D (B, 4, 256, 256) -> z2 (B, 128), keeping the spatial map reachable."""

    def __init__(
        self,
        z2_dim: int = 128,
        image_size: int = 256,
        base_channels: int = 32,
        in_channels: int = 4,
        n_stages: int = 5,
    ) -> None:
        super().__init__()
        stride_factor = 2**n_stages
        if image_size % stride_factor != 0:
            raise ValueError(f"image_size must be divisible by {stride_factor}, got {image_size}")
        self.z2_dim = z2_dim
        self.image_size = image_size
        self.in_channels = in_channels
        self.n_stages = n_stages

        channels = [base_channels * (2**i) for i in range(n_stages)]
        stages = [_conv_block(in_channels, channels[0])]
        stages += [_conv_block(channels[i - 1], channels[i]) for i in range(1, n_stages)]
        self.conv = nn.Sequential(*stages)

        self.spatial = image_size // stride_factor
        self.feature_channels = channels[-1]
        flat_dim = self.feature_channels * self.spatial * self.spatial
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.SiLU(),
            nn.Linear(256, z2_dim),
            nn.LayerNorm(z2_dim),
        )

    def encode_spatial(self, images: torch.Tensor) -> torch.Tensor:
        """Pre-pooling spatial feature map, (B, feature_channels, S, S)."""
        return self.conv(images)

    def encode_pyramid(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Every stage's output, coarsest last.

        At 256² with five stages the final map is 8x8, i.e. 32 px per cell, and
        the vehicle is ~15x7 px — smaller than one cell. A probe reading only the
        last map therefore inherits much of the loss it is meant to measure, so
        `open-questions.md`'s fallback ("keep a low-res spatial feature map")
        needs a choice of stage, not just "the" spatial map. Stage 3 (32x32, 8 px
        per cell) is the first at which the vehicle spans more than one cell.
        """
        maps = []
        h = images
        for stage in self.conv:
            h = stage(h)
            maps.append(h)
        return maps

    def forward_with_features(self, images: torch.Tensor) -> EncoderOutput:
        features = self.encode_spatial(images)
        return EncoderOutput(z2=self.head(features), features=features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode_spatial(images))


class ConvDecoderRGBD(nn.Module):
    """z2 -> (B, out_channels, 256, 256). Mirrored decoder, diagnostic only (§5)."""

    def __init__(
        self,
        z2_dim: int = 128,
        image_size: int = 256,
        base_channels: int = 32,
        out_channels: int = 4,
        n_stages: int = 5,
        final_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        stride_factor = 2**n_stages
        if image_size % stride_factor != 0:
            raise ValueError(f"image_size must be divisible by {stride_factor}, got {image_size}")
        self.spatial = image_size // stride_factor
        channels = [base_channels * (2**i) for i in range(n_stages)][::-1]
        self._channels0 = channels[0]
        self.head = nn.Sequential(
            nn.Linear(z2_dim, channels[0] * self.spatial * self.spatial),
            nn.SiLU(),
        )
        blocks = [_deconv_block(channels[i], channels[i + 1]) for i in range(n_stages - 1)]
        blocks.append(nn.ConvTranspose2d(channels[-1], out_channels, kernel_size=4, stride=2, padding=1))
        if final_activation == "sigmoid":
            blocks.append(nn.Sigmoid())
        elif final_activation != "none":
            raise ValueError(f"unknown final_activation: {final_activation}")
        self.deconv = nn.Sequential(*blocks)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        features = self.head(latents)
        features = features.view(-1, self._channels0, self.spatial, self.spatial)
        return self.deconv(features)


class _LatentToMap(nn.Module):
    """z2 -> (B, out_channels, out_size, out_size) via a short deconv stack."""

    def __init__(self, z2_dim: int, out_channels: int, out_size: int, width: int = 128) -> None:
        super().__init__()
        if out_size % 8 != 0:
            raise ValueError(f"out_size must be divisible by 8, got {out_size}")
        self.start = out_size // 8
        self.width = width
        self.fc = nn.Sequential(nn.Linear(z2_dim, width * self.start * self.start), nn.SiLU())
        self.deconv = nn.Sequential(
            _deconv_block(width, width // 2),
            _deconv_block(width // 2, width // 4),
            nn.ConvTranspose2d(width // 4, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.fc(latents).view(-1, self.width, self.start, self.start)
        return self.deconv(x)


class OccupancyHead(nn.Module):
    """(a) occupancy / class-mask logits, (B, n_classes, out_size, out_size)."""

    def __init__(self, z2_dim: int = 128, n_classes: int = 4, out_size: int = 64) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.out_size = out_size
        self.net = _LatentToMap(z2_dim, n_classes, out_size)

    def forward(self, z2: torch.Tensor) -> torch.Tensor:
        return self.net(z2)


class VehicleHeatmapHead(nn.Module):
    """(b) vehicle-centre heatmap logits + yaw as an unnormalized (cos, sin)."""

    def __init__(self, z2_dim: int = 128, out_size: int = 64) -> None:
        super().__init__()
        self.out_size = out_size
        self.heatmap = _LatentToMap(z2_dim, 1, out_size)
        self.yaw = nn.Sequential(nn.Linear(z2_dim, 128), nn.SiLU(), nn.Linear(128, 2))

    def forward(self, z2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.heatmap(z2), self.yaw(z2)


class ElevationHead(nn.Module):
    """(d) elevation reconstruction carrying its OWN normalization (§5).

    Kept separate from the RGB path so elevation is not forced through the [0, 1]
    sigmoid the RGB reconstruction uses. `set_normalization()` mirrors
    `NRDModel.set_z2_normalization`: fit from data, then persist in the state dict.
    """

    def __init__(self, z2_dim: int = 128, image_size: int = 256, base_channels: int = 32) -> None:
        super().__init__()
        self.decoder = ConvDecoderRGBD(
            z2_dim=z2_dim,
            image_size=image_size,
            base_channels=base_channels,
            out_channels=1,
            final_activation="none",
        )
        self.register_buffer("elev_mean", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("elev_std", torch.ones(1, dtype=torch.float32))

    @torch.no_grad()
    def set_normalization(self, elevation: torch.Tensor) -> None:
        self.elev_mean.copy_(elevation.mean().reshape(1))
        self.elev_std.copy_(elevation.std().clamp_min(1e-4).reshape(1))

    def normalize(self, elevation: torch.Tensor) -> torch.Tensor:
        return (elevation - self.elev_mean) / self.elev_std

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * self.elev_std + self.elev_mean

    def forward(self, z2: torch.Tensor) -> torch.Tensor:
        """Returns elevation in NORMALIZED space; compare against `normalize(target)`."""
        return self.decoder(z2)


class AuxiliaryHeads(nn.Module):
    """The four mandatory AE warm-up heads of plan §5, as one module.

    All four read the global latent, so their gradients shape `z2` itself. Reading
    the spatial map instead would leave `z2` untouched by them; see the ambiguity
    note in the WP1 report.
    """

    def __init__(
        self,
        z2_dim: int = 128,
        image_size: int = 256,
        base_channels: int = 32,
        n_classes: int = 4,
        mask_size: int = 64,
    ) -> None:
        super().__init__()
        self.occupancy = OccupancyHead(z2_dim, n_classes=n_classes, out_size=mask_size)
        self.vehicle = VehicleHeatmapHead(z2_dim, out_size=mask_size)
        self.rgb = ConvDecoderRGBD(
            z2_dim=z2_dim, image_size=image_size, base_channels=base_channels, out_channels=3
        )
        self.elevation = ElevationHead(z2_dim, image_size=image_size, base_channels=base_channels)

    def forward(self, z2: torch.Tensor) -> dict[str, torch.Tensor]:
        heatmap, yaw = self.vehicle(z2)
        return {
            "occupancy_logits": self.occupancy(z2),
            "vehicle_heatmap_logits": heatmap,
            "vehicle_yaw": yaw,
            "rgb": self.rgb(z2),
            "elevation_normalized": self.elevation(z2),
        }


def foreground_weighted_rgb_loss(
    pred: torch.Tensor, target: torch.Tensor, foreground: torch.Tensor, weight: float = 30.0
) -> torch.Tensor:
    """(c) class-weighted RGB reconstruction. `foreground` is a (B, 1, H, W) mask."""
    per_pixel = F.mse_loss(pred, target, reduction="none").mean(dim=1, keepdim=True)
    weights = 1.0 + (weight - 1.0) * foreground
    return (per_pixel * weights).sum() / weights.sum().clamp_min(1e-6)


def yaw_loss(pred_cos_sin: torch.Tensor, yaw_rad: torch.Tensor) -> torch.Tensor:
    """Angle loss on the unit circle; avoids the wraparound a direct MSE on yaw has."""
    target = torch.stack([torch.cos(yaw_rad), torch.sin(yaw_rad)], dim=-1)
    pred = pred_cos_sin / pred_cos_sin.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return F.mse_loss(pred, target)


def pack_rgbd(rgb_uint8: torch.Tensor, elevation: torch.Tensor, head: ElevationHead) -> torch.Tensor:
    """(B, H, W, 3) uint8 + (B, 1, H, W) elevation -> (B, 4, H, W) float.

    RGB is scaled to [0, 1]; elevation uses the head's own normalization, so the
    two do not share a scale (plan §5's "own normalization").
    """
    rgb = rgb_uint8.to(torch.float32).div_(255.0).movedim(-1, -3)
    return torch.cat([rgb, head.normalize(elevation)], dim=-3)
