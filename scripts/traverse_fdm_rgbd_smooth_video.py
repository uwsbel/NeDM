#!/usr/bin/env python3
"""Record actual Chrono video and passive wheel/grade diagnostics together.

The existing physics runner, camera and measured RGB-D input remain unchanged.
This wrapper combines their optional observers for the smooth-hill diagnostic.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nedm.traverse.fdm_slope_probe import SmoothHillDiagnostics
from traverse_fdm_rgbd_video import RolloutCamera
import traverse_fdm_rgbd_chrono as runner


class CameraAndSlopeDiagnostics:
    def __init__(self, out, case, *, width=640, height=480, fps=10., name="rollout"):
        self.camera = RolloutCamera(out, width=width, height=height, fps=fps, name=name)
        self.diagnostics = SmoothHillDiagnostics(out, case)

    def on_frame(self, scene, frame, state, pose, action):
        self.diagnostics.on_frame(scene, frame, state, pose, action)
        self.camera.on_frame(scene, frame, state, pose, action)

    def finish(self, scene, frame, state, pose, action):
        self.diagnostics.finish(scene, frame, state, pose, action)
        self.camera.finish(scene, frame, state, pose, action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("execute",))
    parser.add_argument("--case", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--anchor-observation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chrono-data", required=True)
    parser.add_argument("--horizon-s", type=float)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=10.)
    parser.add_argument("--name", default="smooth_hill_rollout")
    args = parser.parse_args()
    args.backend, args.depth_ray_scale = "Vulkan_RT_lavapipe", 1.
    args.record_rgbd_stride, args.path_height_source = 0, "truth"
    args.frame_observer = CameraAndSlopeDiagnostics(args.out, args.case,
        width=args.width, height=args.height, fps=args.fps, name=args.name)
    runner.dump(Path(args.out) / "smooth_video_wrapper.json", {
        "wrapper_sha256": runner.sha256(__file__),
        "observer_order": ["passive slope diagnostics", "actual visualization camera"],
        "model_inputs": "Unchanged measured current overhead RGB-D and vehicle context",
        "physics_runner_sha256": runner.sha256(runner.__file__),
    })
    runner.run_chrono(args)


if __name__ == "__main__":
    main()
