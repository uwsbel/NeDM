"""Collage videos and contact sheets from batches of rendered images."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def grid(tiles: np.ndarray, cols: int) -> np.ndarray:
    """Tiles (K, H, W, 3) laid out row by row, ``cols`` per row."""
    k, h, w = tiles.shape[:3]
    rows = int(math.ceil(k / cols))
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i in range(k):
        sheet[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = tiles[i]
    return sheet


def depth_to_gray(depth: np.ndarray, near: float = 0.5, far: float = 6.0) -> np.ndarray:
    """Depth (..., H, W) -> gray (..., H, W, 3): near is bright, misses are black."""
    gray = np.where(depth > 0, 255.0 * (1.0 - np.clip(depth, near, far) / far), 0.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def save_sheet(tiles: np.ndarray, path: str | Path, cols: int = 32) -> None:
    """Every world in one PNG, the evidence that all of them rendered."""
    from PIL import Image
    Image.fromarray(grid(tiles, cols)).save(path)


class CollageRecorder:
    """Video of a random sample of worlds, the same sample in every frame."""

    def __init__(self, path: str | Path, num_worlds: int, sample: int = 50, cols: int = 10,
                 fps: float = 50.0, seed: int = 0) -> None:
        import imageio.v2 as imageio
        rng = np.random.default_rng(seed)
        self.pick = np.sort(rng.choice(num_worlds, size=min(sample, num_worlds), replace=False))
        self.cols = cols
        self._writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                                          macro_block_size=None)

    def add(self, tiles: np.ndarray) -> None:
        """``tiles`` (N, H, W, 3) uint8 for all worlds. The sample is taken here."""
        self._writer.append_data(grid(tiles[self.pick], self.cols))

    def close(self) -> None:
        self._writer.close()
