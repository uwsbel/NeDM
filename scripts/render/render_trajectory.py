#!/usr/bin/env python
"""Render a saved joint trajectory of a URDF robot in every world, with no model and no torch GPU.

The model-agnostic half of the pipeline: any rollout that saved ``joint_q`` can be drawn
later, on another machine, by Warp alone. It is also how a new machine is checked. Render
one trajectory on two machines and compare ``sample_frames.npz`` (see docs/batch_rendering.md).

    PYTHONPATH=src python scripts/render/render_trajectory.py --traj out/go2/traj.npz \\
        --urdf go2_description.urdf --out out/go2_again

``--traj`` is an .npz with ``joint_q`` (frames, worlds, coordinates): base position, base
quaternion xyzw, then the joints in the URDF's order, and ``cam_dt`` (seconds per frame).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traj", required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--envs", type=int, default=1 << 30, help="render only the first N worlds")
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--sample", type=int, default=50, help="worlds in the collage")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="warp device, default: the GPU if there is one")
    parser.add_argument("--no-shadows", action="store_true")
    parser.add_argument("--backend", choices=["newton", "madrona"], default="newton")
    parser.add_argument("--palette", choices=["none", "go2"], default="none",
                        help="link colors: none draws every link gray, go2 uses the Go2 example's palette")
    a = parser.parse_args()

    if a.backend == "newton":
        import warp as wp
        wp.config.quiet = True
    from nedm.render import BatchRenderer, CollageRecorder, Scene, cameras, depth_to_gray, save_sheet

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(a.traj)
    trajectory = data["joint_q"][:, : a.envs]
    frames_n, n = trajectory.shape[:2]

    colors = None
    if a.palette == "go2":
        from go2_nrd_rollout import link_color
        colors = link_color
    t0 = time.perf_counter()
    scene = Scene.from_urdf(a.urdf, floating=True, colors=colors).add_ground()
    renderer = BatchRenderer(scene, n, width=a.res, height=a.res, shadows=not a.no_shadows, device=a.device,
                             backend=a.backend)
    build_s = time.perf_counter() - t0

    fps = int(round(1.0 / float(data["cam_dt"])))
    rgb_video = CollageRecorder(out / "collage_rgb.mp4", n, a.sample, fps=fps, seed=a.seed)
    depth_video = CollageRecorder(out / "collage_depth.mp4", n, a.sample, fps=fps, seed=a.seed)
    kept, render_s, first_s = [], 0.0, 0.0
    for c in range(frames_n):
        t = time.perf_counter()
        frames = renderer.render(cameras.follow(trajectory[c, :, :2]).numpy(), joint_q=trajectory[c])
        elapsed = time.perf_counter() - t
        if c == 0:
            first_s = elapsed
        else:
            render_s += elapsed
        rgb, depth = frames.rgb[:, 0], frames.depth[:, 0]
        rgb_video.add(rgb)
        depth_video.add(depth_to_gray(depth))
        if c % 25 == 0:
            kept.append((rgb[rgb_video.pick].copy(), depth[rgb_video.pick].copy()))
        if c == (frames_n - 1) // 2:
            save_sheet(rgb, out / "all_worlds_midframe.png")
        if c % 50 == 0:
            print(f"frame {c:4d}/{frames_n - 1}  render {1e3 * render_s / max(c, 1):6.1f} ms/frame", flush=True)
    rgb_video.close()
    depth_video.close()
    np.savez_compressed(out / "sample_frames.npz", pick=rgb_video.pick, rgb=np.stack([k[0] for k in kept]),
                        depth=np.stack([k[1] for k in kept]))
    info = {"envs": n, "res": a.res, "frames": frames_n, "backend": a.backend, "render_device": str(renderer.device),
            "scene_build_s": round(build_s, 2),
            "first_frame_s": round(first_s, 2), "render_ms_per_frame": round(1e3 * render_s / (frames_n - 1), 2),
            "views_per_s": round(n * (frames_n - 1) / render_s), "trajectory": str(a.traj)}
    (out / "run.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
