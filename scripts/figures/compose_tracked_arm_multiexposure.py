"""Multi-exposure composites of the tracked+arm demo.

Writes TWO separate stills (not one stitched image):
  - drive: 10 frames evenly in [0, 479], alpha INCREASING (origin faint -> goal solid)
  - arm:   10 frames evenly in [487, 504], alpha DECREASING (goal solid -> arm reach fades)

Because the camera / ground / sky are static, each frame's vehicle is matted
against a clean background plate (the per-pixel median of a sample of frames) so
the ghosts stay crisp and visible instead of muddying the shared background.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
ANIM = REPO / "artifacts/combined_tracked_arm_demo/blender/renders/anim"
OUT = REPO / "artifacts/combined_tracked_arm_demo/blender/renders/tracked_arm_multiexposure.png"


def load(idx: int) -> np.ndarray:
    return np.asarray(Image.open(ANIM / f"frame{idx:04d}.png").convert("RGB"), dtype=np.float32)


def build_background(sample_idxs: list[int]) -> np.ndarray:
    """Per-pixel median over a spread of frames -> clean plate (vehicle is a minority per pixel)."""
    stack = np.stack([load(i) for i in sample_idxs], axis=0)  # (N, H, W, 3)
    return np.median(stack, axis=0)


def composite(layers: dict[int, float], bg: np.ndarray, diff_lo: float, diff_hi: float) -> np.ndarray:
    """Matte each frame's vehicle against bg and stack faintest->boldest over the plate."""
    acc = bg.copy()
    for f, a in sorted(layers.items(), key=lambda kv: kv[1]):
        frame = load(f)
        diff = np.abs(frame - bg).max(axis=2)  # per-pixel max-channel difference
        soft = np.clip((diff - diff_lo) / (diff_hi - diff_lo), 0.0, 1.0)
        w = (soft * a)[..., None]  # per-pixel paint weight
        acc = frame * w + acc * (1.0 - w)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha-min", type=float, default=0.12)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--diff-lo", type=float, default=8.0, help="soft-matte low threshold (0-255)")
    ap.add_argument("--diff-hi", type=float, default=40.0, help="soft-matte high threshold (0-255)")
    ap.add_argument("--drive-start", type=int, default=0)
    ap.add_argument("--drive-end", type=int, default=479, help="last frame of the drive trail")
    ap.add_argument("--arm-start", type=int, default=487)
    ap.add_argument("--arm-end", type=int, default=504)
    ap.add_argument("--n-drive", type=int, default=10)
    ap.add_argument("--n-arm", type=int, default=10)
    ap.add_argument("--output", type=Path, default=None,
                    help="Base path; '_drive'/'_arm' suffixes are inserted. Default: next to the anim renders.")
    args = ap.parse_args()

    drive = np.unique(np.round(np.linspace(args.drive_start, args.drive_end, args.n_drive)).astype(int))
    arm = np.unique(np.round(np.linspace(args.arm_start, args.arm_end, args.n_arm)).astype(int))
    drive_layers = {int(f): float(a)
                    for f, a in zip(drive, np.linspace(args.alpha_min, args.alpha_max, len(drive)))}
    arm_layers = {int(f): float(a)
                  for f, a in zip(arm, np.linspace(args.alpha_max, args.alpha_min, len(arm)))}
    print(f"drive frames {list(drive)}")
    print(f"arm frames   {list(arm)}")

    # Background plate from a spread across the whole run (independent of the chosen layers).
    bg = build_background(list(range(0, 691, 15)))

    # Two SEPARATE composites -> two files.
    drive_img = composite(drive_layers, bg, args.diff_lo, args.diff_hi)  # drive-in trail
    arm_img = composite(arm_layers, bg, args.diff_lo, args.diff_hi)      # arm-reach trail

    if args.output is not None:
        base = Path(args.output)
        suffix = base.suffix or ".png"
        drive_out = base.with_name(f"{base.stem}_drive{suffix}")
        arm_out = base.with_name(f"{base.stem}_arm{suffix}")
    else:
        drive_out = OUT.with_name("tracked_arm_multiexposure_drive.png")
        arm_out = OUT.with_name("tracked_arm_multiexposure_arm.png")

    for img, out in ((drive_img, drive_out), (arm_img, arm_out)):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
