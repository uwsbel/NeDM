"""Ego-aligned crop of the static scene feature map (WP2 spatial-token fix).

The map is the encoder's STAGE-2 tap (64 ch @ 64x64, 1.36 m/cell), not the
final 256x16x16 spatial map: at 5.44 m/cell an 8x8 crop would span 43 m, half
the arena, with about one cell across the whole vehicle. Stage 2 puts 8 samples
over +/-5 m at 1.25 m spacing, matched to the cell size and to the privileged
terrain patch (+/-6 m) that this is the learned analogue of.

Replaces the globally pooled z2 with an indexed one: at each step, sample a
K x K ego-aligned window of the encoder's spatial map centred on the vehicle.
Same token budget as the 256-D z2, but the numbers describe *here* instead of
*everywhere* -- the learned analogue of the privileged terrain patch that took
z1mae@1s from 0.328 to 0.221.

World -> image follows the CameraModel contract exactly (plan 3.3):

    u = cx + f * x / (H - z),   v = cy - f * y / (H - z)

with z the terrain height at the sample point, bilinearly read from the same
quantized heightmap Chrono loads. The feature map is a 4x downsample of the
image and covers the same field of view, so normalized [-1, 1] grid coordinates
transfer unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from nedm.traverse.camera import CameraModel
from nedm.traverse.terrain import TerrainMap


class MapCropper(nn.Module):
    def __init__(self, arena_dir: Path, k: int = 8, half_m: float = 5.0,
                 in_ch: int = 64, mid_ch: int = 16, out_dim: int = 256) -> None:
        super().__init__()
        tmap = TerrainMap.from_dir(Path(arena_dir))
        cam = CameraModel()
        self.k = k
        self.size_m = float(tmap.size_m)
        self.f_px, self.cx, self.cy = cam.f_px, cam.cx, cam.cy
        self.cam_h = cam.cam_height_m
        self.width = cam.width
        self.register_buffer(
            "heightmap", torch.tensor(tmap.height_grid, dtype=torch.float32)[None, None]
        )
        offsets = torch.linspace(-half_m, half_m, k)
        du, dv = torch.meshgrid(offsets, offsets, indexing="ij")  # du forward, dv left
        self.register_buffer("du", du.reshape(-1))
        self.register_buffer("dv", dv.reshape(-1))
        self.proj = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 1), nn.GELU())
        self.fc = nn.Linear(k * k * mid_ch, out_dim)
        self.out_dim = out_dim

    def _terrain_height(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Bilinear height at world (x, y); grid rows increase with +y."""
        gx = (x + self.size_m / 2.0) / self.size_m * 2.0 - 1.0
        gy = (y + self.size_m / 2.0) / self.size_m * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).reshape(1, -1, 1, 2)
        out = F.grid_sample(self.heightmap, grid, mode="bilinear",
                            padding_mode="border", align_corners=False)
        return out.reshape(x.shape)

    def sample_points(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, L, 3) world pose -> (B, L, K*K) world x, y of the ego window."""
        x, y, yaw = pose[..., 0:1], pose[..., 1:2], pose[..., 2:3]
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        du, dv = self.du.view(1, 1, -1), self.dv.view(1, 1, -1)
        return (x + du * cos_yaw - dv * sin_yaw, y + du * sin_yaw + dv * cos_yaw)

    def forward(self, maps: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """maps (B, C, H, W) static per episode; pose (B, L, 3) -> (B, L, out_dim)."""
        batch, _, _, _ = maps.shape
        length = pose.shape[1]
        px, py = self.sample_points(pose)
        pz = self._terrain_height(px, py)
        scale = self.f_px / (self.cam_h - pz)
        u = self.cx + scale * px
        v = self.cy - scale * py
        gx = (u + 0.5) * 2.0 / self.width - 1.0
        gy = (v + 0.5) * 2.0 / self.width - 1.0
        grid = torch.stack([gx, gy], dim=-1).reshape(batch, length * self.k, self.k, 2)
        crop = F.grid_sample(maps, grid, mode="bilinear", padding_mode="border",
                             align_corners=False)  # (B, C, L*K, K)
        crop = crop.reshape(batch, -1, length, self.k, self.k).permute(0, 2, 1, 3, 4)
        crop = crop.reshape(batch * length, -1, self.k, self.k)
        return self.fc(self.proj(crop).flatten(1)).reshape(batch, length, self.out_dim)
