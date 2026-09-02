#!/usr/bin/env python
"""Validate analytic class masks (nedm.traverse.masks) against recorded depth.

Geometry check, runs on any machine with an episode store: back-project each
episode's first depth frame to world heights, threshold "elevated above
terrain", and require the analytic mask union to agree up to a 1-pixel
anti-aliasing band (raw boundary pixels differ because PIL fills polygon
edges while edge depth rays graze past silhouettes).

Class-identity is validated separately (one-shot ChSegmentationCamera on
newton, scripts/traverse_segcam_check.py).

Usage:
  PYTHONPATH=src python scripts/traverse_mask_check.py --root artifacts/traverse/smoke_v1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nedm.traverse.camera import CameraModel  # noqa: E402
from nedm.traverse.masks import class_masks  # noqa: E402
from nedm.traverse.storage import DEPTH_NO_HIT, DEPTH_OFFSET_M, EpisodeReader, list_episodes  # noqa: E402
from nedm.traverse.terrain import TerrainMap  # noqa: E402


def dilate1(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    out[1:, :] |= m[:-1, :]
    out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    out[1:, 1:] |= m[:-1, :-1]
    out[:-1, :-1] |= m[1:, 1:]
    out[1:, :-1] |= m[:-1, 1:]
    out[:-1, 1:] |= m[1:, :-1]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/traverse/smoke_v1")
    parser.add_argument("--arena", default="assets/traverse/arena_v1")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--elev-thresh-m", type=float, default=0.25)
    parser.add_argument("--ray-scale", type=float, default=1.2)
    parser.add_argument("--max-extra-px", type=int, default=40, help="beyond the 1-px band")
    parser.add_argument("--max-missed-px", type=int, default=5, help="beyond the 1-px band")
    args = parser.parse_args()

    tmap = TerrainMap.from_dir((REPO_ROOT / args.arena).resolve())
    half = 0.494 * tmap.size_m
    cam_rgb = CameraModel()
    cam_depth = CameraModel(hfov_rad=2 * math.atan(args.ray_scale * math.tan(cam_rgb.hfov_rad / 2)))

    failures = 0
    for ep in list_episodes((REPO_ROOT / args.root).resolve()):
        meta = json.load(open(ep / "meta.json"))
        reader = EpisodeReader(ep)
        win = reader.read_window(args.frame, 1)
        reader.close()
        depth_mm = win["depth_mm"][0].astype(np.float64)
        valid = depth_mm != int(DEPTH_NO_HIT)
        depth_m = np.where(valid, DEPTH_OFFSET_M + depth_mm / 1000.0, 100.0)
        fields = list(win["state_fields"])
        s = win["states"][0]
        x, y, yaw = (float(s[fields.index(k)]) for k in ("pos_x_m", "pos_y_m", "yaw_rad"))

        wx, wy, wz = cam_rgb.depth_to_world(depth_m, convention="ray", ray_scale=args.ray_scale)
        inb = valid & (np.abs(wx) < half) & (np.abs(wy) < half)
        ground = np.full_like(wz, 1e9)
        ground[inb] = tmap.height(wx[inb], wy[inb])
        elevated = inb & ((wz - ground) > args.elev_thresh_m)

        masks = class_masks(
            meta["layout"], tmap.height, cam_depth, vehicle_pose=(x, y, float(tmap.height(x, y)), yaw)
        )
        union = np.zeros_like(elevated)
        for m in masks.values():
            union |= m

        extra1 = int((union & ~dilate1(elevated)).sum())
        missed1 = int((elevated & ~dilate1(union)).sum())
        iou = (elevated & union).sum() / max(1, (elevated | union).sum())
        ok = extra1 <= args.max_extra_px and missed1 <= args.max_missed_px
        failures += 0 if ok else 1
        print(
            f"{ep.name}: {'ok  ' if ok else 'FAIL'} raw-IoU={iou:.3f} "
            f"beyond-1px extra={extra1} missed={missed1} "
            f"(elevated={int(elevated.sum())} mask={int(union.sum())})"
        )
    print("mask check:", "PASS" if failures == 0 else f"{failures} episode(s) FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
