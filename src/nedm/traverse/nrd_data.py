"""WP2 NRD dynamics data: cached [z1, z2, a] windows over traverse episodes.

The cache is produced by ``scripts/traverse_wp2_encode.py`` (frozen WP1
encoder). Splits are reproduced from the cache manifest with the same
permutation ``perception.split_episodes`` uses, so WP2's held-out layouts are
exactly the encoder's held-out layouts -- otherwise encoder-training layouts
leak into WP2 val/test and "held out" stops meaning held out for the stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SPLIT_SEED = 20260902
SPLIT_FRACTIONS = (0.7, 0.15)


def split_keys(
    keys: list[str], seed: int = SPLIT_SEED, fractions: tuple[float, float] = SPLIT_FRACTIONS
) -> tuple[list[str], list[str], list[str]]:
    """Mirror of ``perception.split_episodes`` over cache keys (same order/seed)."""
    order = np.random.default_rng(seed).permutation(len(keys))
    n_train = int(fractions[0] * len(keys))
    n_val = int(fractions[1] * len(keys))
    return (
        [keys[i] for i in order[:n_train]],
        [keys[i] for i in order[n_train : n_train + n_val]],
        [keys[i] for i in order[n_train + n_val :]],
    )


@dataclass
class CacheSplit:
    """Stacked arrays for one split. Episodes are uniform length in schema v1."""

    keys: list[str]
    z1: np.ndarray  # (N, T, 15)
    z2: np.ndarray  # (N, T, Z) -- empty when the variant does not need it
    act: np.ndarray  # (N, T, 3)
    pose: np.ndarray  # (N, T, 3)
    power: np.ndarray  # (N, T, 1) motorshaft kW -- auxiliary head target only
    terrain: np.ndarray  # (N, T, 64) privileged ego terrain patch (priv rows only)

    @property
    def n_episodes(self) -> int:
        return self.z1.shape[0]

    @property
    def n_frames(self) -> int:
        return self.z1.shape[1]


def load_split(cache_dir: Path, keys: list[str], with_z2: bool = True,
               with_terrain: bool = False) -> CacheSplit:
    z1, z2, act, pose, power, terrain = [], [], [], [], [], []
    for key in keys:
        with np.load(Path(cache_dir) / f"{key}.npz") as data:
            z1.append(data["z1"])
            act.append(data["act"])
            pose.append(data["pose"])
            power.append(data["power"])
            terrain.append(data["terrain"] if with_terrain else np.zeros((0,), np.float32))
            if with_z2:
                z2.append(data["z2"])
    return CacheSplit(
        keys=list(keys),
        z1=np.stack(z1),
        z2=np.stack(z2) if with_z2 else np.zeros((len(keys), z1[0].shape[0], 0), dtype=np.float32),
        act=np.stack(act),
        pose=np.stack(pose),
        power=np.stack(power),
        terrain=np.stack(terrain) if with_terrain
        else np.zeros((len(keys), z1[0].shape[0], 0), dtype=np.float32),
    )


def load_z1_extra(sidecar: Path, keys: list[str]) -> np.ndarray:
    """(N, T, k) extra state channels from a sidecar dir (traverse_wp5_build_z1_sidecar.py)."""
    out = []
    for key in keys:
        with np.load(Path(sidecar) / f"{key}.npz") as data:
            out.append(data["z1_extra"])
    return np.stack(out)


def with_z1_extra(split: CacheSplit, sidecar: Path) -> CacheSplit:
    """Append the sidecar's channels to ``split.z1`` (frame-aligned; same key order)."""
    extra = load_z1_extra(sidecar, split.keys)
    return CacheSplit(keys=split.keys, z1=np.concatenate([split.z1, extra.astype(split.z1.dtype)], -1),
                      z2=split.z2, act=split.act, pose=split.pose, power=split.power, terrain=split.terrain)


def pose_features(pose: np.ndarray) -> np.ndarray:
    """(x, y, yaw) -> (x/40, y/40, sin yaw, cos yaw); yaw split so it is continuous.

    The arena is 80 x 80 m centred on the origin, so the scale puts x, y in
    roughly [-1, 1] without fitting statistics to a privileged channel.
    """
    x, y, yaw = pose[..., 0], pose[..., 1], pose[..., 2]
    return np.stack([x / 40.0, y / 40.0, np.sin(yaw), np.cos(yaw)], axis=-1).astype(np.float32)


@dataclass
class Normalizer:
    z1_mean: np.ndarray
    z1_std: np.ndarray
    z2_mean: np.ndarray
    z2_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray
    power_mean: np.ndarray
    power_std: np.ndarray

    @staticmethod
    def fit(split: CacheSplit, eps: float = 1e-6) -> "Normalizer":
        def stats(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            flat = a.reshape(-1, a.shape[-1])
            return flat.mean(0), np.maximum(flat.std(0), eps)

        z1_mean, z1_std = stats(split.z1)
        act_mean, act_std = stats(split.act)
        power_mean, power_std = stats(split.power)
        if split.z2.shape[-1]:
            z2_mean, z2_std = stats(split.z2)
        else:
            z2_mean = np.zeros(0, dtype=np.float32)
            z2_std = np.ones(0, dtype=np.float32)
        return Normalizer(z1_mean, z1_std, z2_mean, z2_std, act_mean, act_std,
                          power_mean, power_std)

    def to_dict(self) -> dict[str, list[float]]:
        return {k: np.asarray(v).astype(float).tolist() for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(payload: dict[str, list[float]]) -> "Normalizer":
        return Normalizer(**{k: np.asarray(v, dtype=np.float32) for k, v in payload.items()})


class WindowDataset(Dataset):
    """One sample = ``context + 1`` consecutive frames of one episode."""

    def __init__(self, split: CacheSplit, norm: Normalizer, context: int) -> None:
        self.context = context
        self.z1 = ((split.z1 - norm.z1_mean) / norm.z1_std).astype(np.float32)
        self.act = ((split.act - norm.act_mean) / norm.act_std).astype(np.float32)
        self.priv = pose_features(split.pose)
        if split.z2.shape[-1]:
            self.z2 = ((split.z2 - norm.z2_mean) / norm.z2_std).astype(np.float32)
        else:
            self.z2 = np.zeros(split.z1.shape[:2] + (0,), dtype=np.float32)
        self.per_episode = split.n_frames - context
        if self.per_episode <= 0:
            raise ValueError(f"context {context} exceeds episode length {split.n_frames}")
        self.n_episodes = split.n_episodes

    def __len__(self) -> int:
        return self.n_episodes * self.per_episode

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ep, t0 = divmod(int(index), self.per_episode)
        sl = slice(t0, t0 + self.context + 1)
        return {
            "z1": torch.from_numpy(self.z1[ep, sl]),
            "z2": torch.from_numpy(self.z2[ep, sl]),
            "act": torch.from_numpy(self.act[ep, sl]),
            "priv": torch.from_numpy(self.priv[ep, sl]),
        }


def load_cache_keys(cache_dir: Path) -> list[str]:
    payload = json.loads((Path(cache_dir) / "cache_manifest.json").read_text())
    return list(payload["episodes"])
