"""Append the privileged local terrain patch to a WP2 latent cache (CPU only).

Plan section 8.3's privileged upper bound is "state + (x,y,yaw) (and optionally
+ a privileged local terrain patch)". The G3b triad showed the pose-only row is
NOT an upper bound -- joint beat it outright at 1 s -- because position only
buys terrain if the model memorizes the fixed heightmap. This adds the terrain
directly, giving the real ceiling on what perfect localization plus perfect
terrain knowledge can buy.

    terrain (T, K*K) float32   ego-aligned heights relative to the vehicle,
                               metres / TERRAIN_SCALE_M
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nedm.traverse.terrain import TerrainMap

PATCH_K = 8  # 8 x 8 samples
PATCH_HALF_M = 6.0  # +/- 6 m window (vehicle is ~4.8 m long)
TERRAIN_SCALE_M = 2.0  # arena height range is ~ +/- 2 m

_TMAP: TerrainMap | None = None
_ARENA = ""


def _tmap() -> TerrainMap:
    global _TMAP
    if _TMAP is None:
        _TMAP = TerrainMap.from_dir(Path(_ARENA))
    return _TMAP


def _init(arena: str) -> None:
    global _ARENA
    _ARENA = arena


def terrain_patch(tmap: TerrainMap, pose: np.ndarray) -> np.ndarray:
    """(T, 3) world pose -> (T, K*K) ego-aligned relative heights."""
    offsets = np.linspace(-PATCH_HALF_M, PATCH_HALF_M, PATCH_K, dtype=np.float32)
    du, dv = np.meshgrid(offsets, offsets, indexing="ij")  # du = forward, dv = left
    du, dv = du.ravel()[None, :], dv.ravel()[None, :]
    x, y, yaw = pose[:, 0:1], pose[:, 1:2], pose[:, 2:3]
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    px = x + du * cos_yaw - dv * sin_yaw
    py = y + du * sin_yaw + dv * cos_yaw
    heights = tmap.height(px, py)
    centre = tmap.height(pose[:, 0], pose[:, 1])[:, None]
    return ((heights - centre) / TERRAIN_SCALE_M).astype(np.float32)


def add_terrain(job: tuple[str, str]) -> str | None:
    cache_dir, key = job
    target = Path(cache_dir) / f"{key}.npz"
    with np.load(target) as data:
        payload = {k: data[k] for k in data.files}
    if "terrain" in payload:
        return None
    payload["terrain"] = terrain_patch(_tmap(), payload["pose"].astype(np.float64))
    if payload["terrain"].shape[0] != payload["z1"].shape[0]:
        return f"{key}: terrain rows != z1 rows"
    np.savez(target, **payload)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--arena", default="assets/traverse/arena_v1")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    cache = Path(args.cache)
    keys = [p.stem for p in sorted(cache.glob("*.npz"))]
    print(f"adding {PATCH_K}x{PATCH_K} terrain patches to {len(keys)} episodes", flush=True)
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.arena,)) as pool:
        for i, err in enumerate(pool.map(add_terrain, [(str(cache), k) for k in keys],
                                         chunksize=32), start=1):
            if err:
                errors.append(err)
            if i % 2000 == 0:
                print(f"  [{i}/{len(keys)}]", flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} episodes failed, first: {errors[0]}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
