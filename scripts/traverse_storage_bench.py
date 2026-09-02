#!/usr/bin/env python
"""Storage-schema benchmarks over a collected episode store (plan §6.1, WP0c).

Reports the three numbers the plan requires before pilot collection:
  1. compression ratio (raw frame bytes / bytes on disk) + tier extrapolation,
  2. random-window loader throughput at training batch size,
  3. peak disk (the writer streams compressed chunks, so peak = final size;
     measured here as the actual store size).

Usage (from repo root, nedm env):
  PYTHONPATH=src python scripts/traverse_storage_bench.py --root artifacts/traverse/smoke_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nedm.traverse.storage import EpisodeReader, decode_depth_m, list_episodes  # noqa: E402

GIB = 2**30


def bench_loader(
    readers: list[EpisodeReader],
    window: int,
    batch: int,
    n_batches: int,
    rng: np.random.Generator,
) -> dict:
    """Random (episode, t0) windows, decoded to training dtypes."""
    draws = []
    for _ in range(n_batches * batch):
        r = readers[int(rng.integers(len(readers)))]
        t0 = int(rng.integers(0, r.frames - window + 1))
        draws.append((r, t0))

    wall0 = time.perf_counter()
    raw_bytes = 0
    for i in range(n_batches):
        for r, t0 in draws[i * batch : (i + 1) * batch]:
            got = r.read_window(t0, window)
            _ = decode_depth_m(got["depth_mm"])  # include the float conversion cost
            raw_bytes += got["rgb"].nbytes + got["depth_mm"].nbytes
    wall = time.perf_counter() - wall0
    n_windows = n_batches * batch
    return {
        "window_frames": window,
        "batch_windows": batch,
        "batches": n_batches,
        "windows_per_s": round(n_windows / wall, 1),
        "frames_per_s": round(n_windows * window / wall, 1),
        "raw_equivalent_mib_per_s": round(raw_bytes / wall / 2**20, 1),
        "wall_s": round(wall, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/traverse/smoke_v1")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = (REPO_ROOT / args.root).resolve()
    ep_dirs = list_episodes(root)
    if not ep_dirs:
        print(f"no episodes under {root}", file=sys.stderr)
        return 1

    readers = [EpisodeReader(d) for d in ep_dirs]
    per_ep = []
    for r in readers:
        b = r.meta["bytes"]
        disk = b["rgb_bin"] + b["depth_bin"] + b["states_npz"]
        per_ep.append(
            {
                "episode_id": r.meta["episode_id"],
                "family": r.meta.get("family_actual", r.meta.get("family")),
                "frames": r.frames,
                "raw_mib": round(b["raw_frames"] / 2**20, 1),
                "disk_mib": round(disk / 2**20, 1),
                "ratio": round(b["raw_frames"] / max(disk, 1), 1),
                "rgb_ratio": round(r.frames * r.meta["width"] * r.meta["height"] * 3 / max(b["rgb_bin"], 1), 1),
                "depth_ratio": round(r.frames * r.meta["width"] * r.meta["height"] * 2 / max(b["depth_bin"], 1), 1),
            }
        )

    total_raw = sum(r.meta["bytes"]["raw_frames"] for r in readers)
    total_disk = sum(
        r.meta["bytes"]["rgb_bin"] + r.meta["bytes"]["depth_bin"] + r.meta["bytes"]["states_npz"]
        for r in readers
    )
    total_frames = sum(r.frames for r in readers)
    bytes_per_frame_disk = total_disk / max(total_frames, 1)

    rng = np.random.default_rng(args.seed)
    # first pass touches fresh offsets (cold-ish), second reuses the page cache
    loader_pass1 = bench_loader(readers, args.window, args.batch, args.batches, rng)
    loader_pass2 = bench_loader(readers, args.window, args.batch, args.batches, rng)

    tiers = {"pilot_160k_frames": 160_000, "full_1p6M_frames": 1_600_000}
    report = {
        "root": str(root),
        "episodes": len(readers),
        "frames": total_frames,
        "codec": readers[0].meta["codec"],
        "chunk_frames": readers[0].meta["chunk_frames"],
        "compression": {
            "total_raw_gib": round(total_raw / GIB, 3),
            "total_disk_gib": round(total_disk / GIB, 3),
            "ratio": round(total_raw / max(total_disk, 1), 1),
            "disk_bytes_per_frame": int(bytes_per_frame_disk),
            "per_episode": per_ep,
        },
        "peak_disk_note": "writer streams compressed chunks; peak disk == final store size",
        "tier_extrapolation_gib": {
            name: round(n * bytes_per_frame_disk / GIB, 1) for name, n in tiers.items()
        },
        "loader": {"pass1": loader_pass1, "pass2_warm": loader_pass2},
    }
    for r in readers:
        r.close()

    out_path = root / "storage_bench.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
