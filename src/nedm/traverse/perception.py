"""WP1 perception pilot: dataset, encoder + warm-up auxiliary heads, probes.

Plan §5: the AE encoder produces one global latent z2 (the v1 spine). During
warm-up the encoder is shaped by mandatory auxiliary losses from analytic
ground truth (labels are free, never stored):
  (a) class-mask prediction        -- label_image() from the layout manifest,
  (b) vehicle-center heatmap + yaw -- pose from states.npz,
  (c) foreground-weighted RGB recon,
  (d) elevation reconstruction with its own normalization.

After warm-up, frozen-encoder probes decode BEV occupancy and vehicle pose
from z2 AND from the pre-pooling spatial feature map; the gap quantifies what
global pooling destroys (plan §5 information-bottleneck staging, G1).

Frames are read straight from the schema-v1 stores (storage.EpisodeReader);
static targets (asset label image, BEV occupancy) are rasterized once per
episode and cached, only the vehicle box changes per frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from nedm.traverse.camera import CameraModel
from nedm.traverse.masks import (
    VEHICLE_HGT_M,
    VEHICLE_LEN_M,
    VEHICLE_WID_M,
    _box_corners,
    _rasterize_solids,
    bev_occupancy,
    label_image,
)
from nedm.traverse.storage import DEPTH_NO_HIT, DEPTH_OFFSET_M, EpisodeReader, list_episodes
from nedm.traverse.terrain import TerrainMap

N_CLASSES = 5  # 0=background, 1=rock, 2=tree, 3=house, 4=vehicle
BEV_GRID = 128
AUX_RES = 64  # rgb/elevation recon + center heatmap resolution
HEAT_SIGMA_PX = 1.5
DEPTH_RAY_SCALE = 1.2  # WP0b-calibrated ChDepthCamera ray widening

# Loss weights for the warm-up phase (recorded in the run config). Heat is a
# ~7 px^2 gaussian in a 64^2 field: raw MSE of an all-zero prediction is ~5e-4,
# so it needs a large weight to matter at all (v1 ran at 50 and the head
# stayed at zero; center error was ~random).
LOSS_WEIGHTS = {
    "seg": 1.0, "rgb": 1.0, "elev": 1.0, "heat": 2000.0, "pose": 1.0, "bev": 1.0,
    "seg_sp": 1.0, "bev_sp": 1.0,  # deep supervision on the pre-pooling map
}
SEG_CLASS_WEIGHTS = (0.2, 8.0, 4.0, 1.0, 2.0)  # bg, rock, tree, house, vehicle


# ---------------------------------------------------------------------------
# Dataset


@dataclass
class EpisodeEntry:
    ep_dir: Path
    n_frames: int
    layout: dict[str, Any]


def _episode_entries(roots: list[Path]) -> list[EpisodeEntry]:
    import json

    entries = []
    for root in roots:
        for ep_dir in list_episodes(root):
            with (ep_dir / "meta.json").open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            if meta.get("status") != "complete":
                continue
            entries.append(EpisodeEntry(ep_dir, int(meta["frames"]), meta["layout"]))
    return entries


def split_episodes(
    roots: list[Path], seed: int = 20260902, fractions: tuple[float, float] = (0.7, 0.15)
) -> tuple[list[EpisodeEntry], list[EpisodeEntry], list[EpisodeEntry]]:
    """Deterministic 70/15/15 layout-level split (each episode = unique layout)."""
    entries = _episode_entries(roots)
    order = np.random.default_rng(seed).permutation(len(entries))
    n_train = int(fractions[0] * len(entries))
    n_val = int(fractions[1] * len(entries))
    train = [entries[i] for i in order[:n_train]]
    val = [entries[i] for i in order[n_train : n_train + n_val]]
    test = [entries[i] for i in order[n_train + n_val :]]
    return train, val, test


class WP1FrameDataset(Dataset):
    """One sample = one frame: RGB-D input + analytic targets.

    Static targets (asset label image, BEV occupancy) are cached per episode;
    the vehicle box is overlaid per frame from the recorded pose.
    """

    def __init__(self, entries: list[EpisodeEntry], arena_dir: Path, max_open_readers: int = 32):
        self.entries = entries
        self.tmap = TerrainMap.from_dir(arena_dir)
        self.cam = CameraModel()
        self.max_open = max_open_readers
        self._readers: dict[int, EpisodeReader] = {}
        self._static: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._field_idx: dict[str, int] | None = None
        _, _, sec = self.cam.pixel_rays(DEPTH_RAY_SCALE)
        self._sec = sec.astype(np.float32)
        self.h_min = float(self.tmap.meta["height_min_m"])
        self.h_max = float(self.tmap.meta["height_max_m"])

    def __len__(self) -> int:
        return sum(e.n_frames for e in self.entries)

    def _locate(self, idx: int) -> tuple[int, int]:
        for ep_i, entry in enumerate(self.entries):
            if idx < entry.n_frames:
                return ep_i, idx
            idx -= entry.n_frames
        raise IndexError(idx)

    def _reader(self, ep_i: int) -> EpisodeReader:
        if ep_i not in self._readers:
            if len(self._readers) >= self.max_open:
                old_key, old = next(iter(self._readers.items()))
                old.close()
                del self._readers[old_key]
            self._readers[ep_i] = EpisodeReader(self.entries[ep_i].ep_dir)
        return self._readers[ep_i]

    def _static_targets(self, ep_i: int) -> tuple[np.ndarray, np.ndarray]:
        if ep_i not in self._static:
            layout = self.entries[ep_i].layout
            label = label_image(layout, self.tmap.height, self.cam)
            bev = bev_occupancy(layout, self.tmap.size_m, grid=BEV_GRID)
            # ~80 KB/episode: keep all of them (random sampling revisits every
            # episode constantly; evicting would re-rasterize ~30 ms per miss).
            self._static[ep_i] = (label, bev)
        return self._static[ep_i]

    def _z_map(self, depth_mm: np.ndarray) -> np.ndarray:
        """uint16 depth -> normalized height-above-datum image in ~[0, 1]."""
        depth_m = DEPTH_OFFSET_M + depth_mm.astype(np.float32) / 1000.0
        z = self.cam.cam_height_m - depth_m / self._sec
        z[depth_mm == DEPTH_NO_HIT] = self.h_min
        return (z - self.h_min) / (self.h_max - self.h_min)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_i, t = self._locate(int(idx))
        reader = self._reader(ep_i)
        win = reader.read_window(t, 1)
        if self._field_idx is None:
            self._field_idx = {name: i for i, name in enumerate(win["state_fields"])}
        fi = self._field_idx
        state = win["states"][0]
        x, y, yaw = float(state[fi["pos_x_m"]]), float(state[fi["pos_y_m"]]), float(state[fi["yaw_rad"]])
        gz = float(self.tmap.height(x, y))

        rgb = win["rgb"][0].astype(np.float32) / 255.0  # (H, W, 3)
        z_norm = self._z_map(win["depth_mm"][0])  # (H, W)

        label_static, bev = self._static_targets(ep_i)
        label = label_static.copy()
        veh_box = _box_corners(x, y, yaw, VEHICLE_LEN_M, VEHICLE_WID_M, gz, gz + VEHICLE_HGT_M)
        label[_rasterize_solids([veh_box], self.cam)] = 4

        # Vehicle center in image + heatmap target at AUX_RES.
        u, v = self.cam.world_to_pixel(x, y, gz + 0.5 * VEHICLE_HGT_M)
        scale = AUX_RES / self.cam.width
        uu, vv = np.meshgrid(np.arange(AUX_RES, dtype=np.float32), np.arange(AUX_RES, dtype=np.float32))
        heat = np.exp(-(((uu - u * scale) ** 2 + (vv - v * scale) ** 2) / (2.0 * HEAT_SIGMA_PX**2)))

        inp = np.concatenate([rgb.transpose(2, 0, 1), z_norm[None]], axis=0)
        return {
            "input": torch.from_numpy(inp),
            "label": torch.from_numpy(label.astype(np.int64)),
            "heat": torch.from_numpy(heat.astype(np.float32)),
            "yaw": torch.tensor([math.sin(yaw), math.cos(yaw)], dtype=torch.float32),
            "bev": torch.from_numpy(bev.astype(np.float32)),
            "center_xy": torch.tensor([x, y], dtype=torch.float32),
            "center_uv": torch.tensor([float(u), float(v)], dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Model


def _conv_block(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.SiLU(inplace=True),
    )


class AttnPool(nn.Module):
    """Learned-query attention pooling: n_q views of the spatial map -> z2.

    Mean+max pooling (v1) collapsed the 16x16 map to one statistic per channel
    and the BEV probe gap came out 0.40 IoU; multiple queries let the vector
    route information from different spatial regions before the bottleneck.
    """

    def __init__(self, c_map: int, z_dim: int, n_q: int = 8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_q, c_map) * 0.02)
        self.proj = nn.Sequential(nn.Linear(n_q * c_map, z_dim), nn.SiLU(), nn.Linear(z_dim, z_dim))
        self.scale = c_map**-0.5

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        f = s.flatten(2).transpose(1, 2)  # (B, HW, C)
        att = torch.softmax(self.q @ f.transpose(1, 2) * self.scale, dim=-1)  # (B, n_q, HW)
        pooled = att @ f  # (B, n_q, C)
        return self.proj(pooled.flatten(1))


class Encoder(nn.Module):
    """256^2 x 4ch -> spatial map (c_map x 16 x 16) -> global latent z2."""

    def __init__(self, z_dim: int = 256, c_map: int = 256, n_q: int = 8):
        super().__init__()
        chans = [32, 64, 128, c_map]
        layers: list[nn.Module] = []
        c_prev = 4
        for c in chans:
            layers += [_conv_block(c_prev, c, stride=2), _conv_block(c, c)]
            c_prev = c
        self.backbone = nn.Sequential(*layers)
        self.pool = AttnPool(c_map, z_dim, n_q)
        self.z_dim = z_dim
        self.c_map = c_map

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.backbone(x)  # (B, c_map, 16, 16)
        return self.pool(s), s


class _UpTrunk(nn.Module):
    """z -> (c x 16 x 16) seed -> upsampled feature pyramid."""

    def __init__(self, z_dim: int, c_seed: int = 128):
        super().__init__()
        self.c_seed = c_seed
        self.seed = nn.Linear(z_dim, c_seed * 4 * 4)
        self.up = nn.Sequential(  # 4 -> 8 -> 16 -> 32 -> 64
            _conv_block(c_seed, 128), nn.Upsample(scale_factor=2),
            _conv_block(128, 128), nn.Upsample(scale_factor=2),
            _conv_block(128, 64), nn.Upsample(scale_factor=2),
            _conv_block(64, 64), nn.Upsample(scale_factor=2),
            _conv_block(64, 64),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        s = self.seed(z).view(-1, self.c_seed, 4, 4)
        return self.up(s)  # (B, 64, 64, 64)


class WarmupHeads(nn.Module):
    """Auxiliary heads decoding from z2 (they shape the global latent)."""

    def __init__(self, z_dim: int = 256):
        super().__init__()
        self.trunk = _UpTrunk(z_dim)
        self.seg = nn.Sequential(  # 64 -> 256 full-res logits (small objects ~15 px)
            nn.Upsample(scale_factor=2), _conv_block(64, 32),
            nn.Upsample(scale_factor=2), _conv_block(32, 32),
            nn.Conv2d(32, N_CLASSES, 1),
        )
        self.rgb = nn.Conv2d(64, 3, 1)
        self.elev = nn.Conv2d(64, 1, 1)
        self.heat = nn.Conv2d(64, 1, 1)
        self.bev = nn.Sequential(nn.Upsample(scale_factor=2), _conv_block(64, 32), nn.Conv2d(32, 1, 1))
        self.pose = nn.Sequential(nn.Linear(z_dim, 128), nn.SiLU(), nn.Linear(128, 4))  # x_n, y_n, sin, cos

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.trunk(z)
        return {
            "seg": self.seg(f),
            "rgb": self.rgb(f),
            "elev": self.elev(f)[:, 0],
            "heat": self.heat(f)[:, 0],
            "bev": self.bev(f)[:, 0],
            "pose": self.pose(z),
        }


class SpatialAuxHeads(nn.Module):
    """Warm-up deep supervision decoding straight from the pre-pooling map.

    Without this branch the encoder only receives gradients through z2, so any
    information the pooling bottleneck rejects (v2: rocks, trees) is never
    perceived at all; with it, the encoder must detect every class and pooling
    alone decides what reaches z2 — which is what the probes then measure.
    """

    def __init__(self, c_map: int = 256):
        super().__init__()
        self.seg = nn.Sequential(  # 16 -> 256 full-res logits
            _conv_block(c_map, 128), nn.Upsample(scale_factor=2),
            _conv_block(128, 64), nn.Upsample(scale_factor=2),
            _conv_block(64, 64), nn.Upsample(scale_factor=2),
            _conv_block(64, 32), nn.Upsample(scale_factor=2),
            _conv_block(32, 32), nn.Conv2d(32, N_CLASSES, 1),
        )
        self.bev = nn.Sequential(  # 16 -> 128 logits
            _conv_block(c_map, 128), nn.Upsample(scale_factor=2),
            _conv_block(128, 64), nn.Upsample(scale_factor=2),
            _conv_block(64, 32), nn.Upsample(scale_factor=2),
            _conv_block(32, 32), nn.Conv2d(32, 1, 1),
        )

    def forward(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"seg": self.seg(s), "bev": self.bev(s)[:, 0]}


def warmup_losses(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    half_size_m: float,
    sp_out: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    inp = batch["input"]
    rgb_t = F.avg_pool2d(inp[:, :3], inp.shape[-1] // AUX_RES)
    elev_t = F.avg_pool2d(inp[:, 3:], inp.shape[-1] // AUX_RES)[:, 0]
    label_small = (
        F.max_pool2d(batch["label"].float().unsqueeze(1), inp.shape[-1] // AUX_RES)[:, 0] > 0
    ).float()
    fg_w = 1.0 + 4.0 * label_small  # foreground-weighted recon (plan §5c)

    cw = torch.tensor(SEG_CLASS_WEIGHTS, device=inp.device)
    pose_t = torch.cat([batch["center_xy"] / half_size_m, batch["yaw"]], dim=1)
    losses = {
        "seg": F.cross_entropy(out["seg"], batch["label"], weight=cw),
        "rgb": ((out["rgb"] - rgb_t).abs().mean(dim=1) * fg_w).mean(),
        "elev": ((out["elev"] - elev_t).abs() * fg_w).mean(),
        "heat": F.mse_loss(out["heat"], batch["heat"]),
        "pose": F.mse_loss(out["pose"], pose_t),
        "bev": F.binary_cross_entropy_with_logits(
            out["bev"], batch["bev"], pos_weight=torch.tensor(5.0, device=inp.device)
        ),
    }
    if sp_out is not None:
        losses["seg_sp"] = F.cross_entropy(sp_out["seg"], batch["label"], weight=cw)
        losses["bev_sp"] = F.binary_cross_entropy_with_logits(
            sp_out["bev"], batch["bev"], pos_weight=torch.tensor(5.0, device=inp.device)
        )
    return losses


# ---------------------------------------------------------------------------
# Frozen-encoder probes (G1: z2 vs pre-pooling spatial map)


class LatentProbe(nn.Module):
    """Decodes BEV occupancy + vehicle pose from the global z2 vector."""

    def __init__(self, z_dim: int = 256):
        super().__init__()
        self.trunk = _UpTrunk(z_dim)
        self.bev = nn.Sequential(nn.Upsample(scale_factor=2), _conv_block(64, 32), nn.Conv2d(32, 1, 1))
        self.pose = nn.Sequential(nn.Linear(z_dim, 128), nn.SiLU(), nn.Linear(128, 4))  # x_n, y_n, sin, cos

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.bev(self.trunk(z))[:, 0], self.pose(z)


class SpatialProbe(nn.Module):
    """Same targets decoded from the pre-pooling spatial map (capacity-matched)."""

    def __init__(self, c_map: int = 256):
        super().__init__()
        self.bev = nn.Sequential(  # 16 -> 128 logits
            _conv_block(c_map, 128), nn.Upsample(scale_factor=2),
            _conv_block(128, 64), nn.Upsample(scale_factor=2),
            _conv_block(64, 32), nn.Upsample(scale_factor=2),
            _conv_block(32, 32), nn.Conv2d(32, 1, 1),
        )
        self.pose_conv = nn.Sequential(_conv_block(c_map, 64, stride=2), _conv_block(64, 64, stride=2))
        self.pose = nn.Sequential(nn.Linear(64 * 4 * 4, 128), nn.SiLU(), nn.Linear(128, 4))

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pose = self.pose(self.pose_conv(s).flatten(1))
        return self.bev(s)[:, 0], pose


def probe_losses(
    bev_logits: torch.Tensor, pose: torch.Tensor, batch: dict[str, torch.Tensor], half_size_m: float
) -> torch.Tensor:
    xy_n = batch["center_xy"] / half_size_m
    pose_t = torch.cat([xy_n, batch["yaw"]], dim=1)
    bev = F.binary_cross_entropy_with_logits(
        bev_logits, batch["bev"], pos_weight=torch.tensor(5.0, device=pose.device)
    )
    return bev + F.mse_loss(pose, pose_t)


# ---------------------------------------------------------------------------
# Metrics


@torch.no_grad()
def seg_metrics(logits: torch.Tensor, label: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-class intersection/union/target-pixel counts (accumulate then divide)."""
    pred = logits.argmax(dim=1)
    inter = np.zeros(N_CLASSES)
    union = np.zeros(N_CLASSES)
    target = np.zeros(N_CLASSES)
    correct = np.zeros(N_CLASSES)
    for c in range(N_CLASSES):
        p, t = pred == c, label == c
        inter[c] = (p & t).sum().item()
        union[c] = (p | t).sum().item()
        target[c] = t.sum().item()
        correct[c] = inter[c]
    return {"inter": inter, "union": union, "target": target, "correct": correct}


@torch.no_grad()
def bev_counts(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pred = torch.sigmoid(logits) > 0.5
    t = target > 0.5
    return float((pred & t).sum().item()), float((pred | t).sum().item())


@torch.no_grad()
def pose_errors(pose: torch.Tensor, batch: dict[str, torch.Tensor], half_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    xy_err = ((pose[:, :2] * half_size_m) - batch["center_xy"]).norm(dim=1).cpu().numpy()
    yaw_pred = torch.atan2(pose[:, 2], pose[:, 3])
    yaw_true = torch.atan2(batch["yaw"][:, 0], batch["yaw"][:, 1])
    d = (yaw_pred - yaw_true + math.pi) % (2 * math.pi) - math.pi
    return xy_err, np.degrees(np.abs(d.cpu().numpy()))
