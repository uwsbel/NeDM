#!/usr/bin/env python3
"""Execute the exact paired Chrono reference while recording a separate camera.

This camera observes actual simulated bodies during physics execution. It is
added only after the standard overhead RGB-D/state equality check has passed.
No predictions or drawn vehicle motion are used to produce its pixels. Raw
PNG frames and timestamp metadata are suitable for an external H.264 encoder.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import traverse_fdm_rgbd_chrono as runner


class RolloutCamera:
    def __init__(self, out, *, width=640, height=480, fps=10., name="rollout"):
        destination = Path(out)
        if (destination / "outcome.json").exists() or (destination / "trajectory.npz").exists() or any((destination / "frames").glob("*.png")):
            raise FileExistsError("Use a new video output directory; existing physical results are immutable")
        stride = int(round(1./(runner.DT*fps)))
        if stride < 1 or not np.isclose(stride*runner.DT*fps, 1.):
            raise ValueError("Video fps must divide the20Hz telemetry rate exactly")
        if width % 2 or height % 2:
            raise ValueError("Even image dimensions required for common H.264 encoders")
        self.out, self.width, self.height = destination, width, height
        self.fps, self.stride, self.name = float(fps), stride, name
        self.rows, self.camera, self.last_launch_count = [], None, None
        self.position, self.look_at = (-10., -8., 7.5), (3., 0., .5)
        self.hfov_rad = math.radians(65.)

    def _initialize(self, scene):
        import pychrono as chrono
        import pychrono.sensor as sens

        direction = np.asarray(self.look_at)-self.position
        yaw = math.atan2(direction[1], direction[0])
        pitch = -math.atan2(direction[2], np.linalg.norm(direction[:2]))
        rot = chrono.QuatFromAngleZ(yaw)*chrono.QuatFromAngleY(pitch)
        pose = chrono.ChFramed(chrono.ChVector3d(*self.position), rot)
        chassis = scene.hmmwv.GetChassis().GetBody()
        rate = 1./float(scene.config["simulation"]["step_size_s"])
        self.camera = sens.ChCameraSensor(chassis, rate, pose,
            self.width, self.height, self.hfov_rad)
        self.camera.SetName("actual_rollout_oblique")
        self.camera.SetLag(0.)
        self.camera.SetCollectionWindow(0.)
        self.camera.PushFilter(sens.ChFilterRGBA8Access())
        scene.manager.AddSensor(self.camera)
        (self.out / "frames").mkdir(parents=True, exist_ok=True)
        runner.dump(self.out / "video_camera.json", {
            "source": "Actual Chrono Vulkan ray tracing of the current physical state",
            "name": self.name, "width": self.width, "height": self.height,
            "nominal_fps": self.fps, "hfov_rad": self.hfov_rad,
            "mount": "chassis body frame, rear-oblique chase view",
            "offset_position_m": self.position, "look_at_in_mount_frame_m": self.look_at,
            "pixel_rows": "top down; same verified Vulkan RGBA tap convention as overhead camera",
            "model_input_boundary": "Visualization sensor added after paired initial model RGB-D/state equality; never supplied to NN",
            "raw_pixels": "No route lines, model predictions, labels or synthetic vehicle motion rendered into these PNGs",
            "video_script_sha256": runner.sha256(__file__),
            "physics_script_sha256": runner.sha256(runner.__file__),
        })

    @staticmethod
    def _drain_overhead(scene):
        # Every manager update also triggers existing overhead sensors. Consume
        # them so the original strict no-skipped-frame taps retain their contract.
        for tap in (scene.rgb_tap, scene.depth_tap):
            buffer = tap._get_buffer()
            if buffer.HasData() and buffer.LaunchedCount > tap.taken_count:
                tap.take()

    def _save(self, scene, frame, state, pose, action, *, terminal=False):
        from PIL import Image
        # Sensors added after t=0 inherit a time-based launch counter (e.g.401
        # after the0.8s settle at500Hz). Require a new buffer, not a counter
        # starting at1. The original model sensor taps remain strict unchanged.
        deadline = time.monotonic()+30.
        while time.monotonic() < deadline:
            buffer = self.camera.GetMostRecentRGBA8Buffer()
            count = int(buffer.LaunchedCount)
            if buffer.HasData() and (self.last_launch_count is None or count > self.last_launch_count):
                self.last_launch_count = count
                rgb = np.ascontiguousarray(buffer.GetRGBA8Data()[::-1, :, :3])
                break
            time.sleep(.001)
        else:
            raise RuntimeError("Visualization camera produced no new frame")
        index = len(self.rows)
        filename = f"frame_{index:06d}.png"
        Image.fromarray(rgb).save(self.out / "frames" / filename)
        self.rows.append({"video_index": index, "file": f"frames/{filename}",
            "telemetry_frame": int(frame), "recording_time_s": frame*runner.DT,
            "simulation_time_s": float(scene.system.GetChTime()),
            "sensor_launch_count": self.last_launch_count,
            "actual_pose": np.asarray(pose).tolist(), "actual_state17": np.asarray(state).tolist(),
            "applied_action": np.asarray(action).tolist(), "terminal": bool(terminal)})

    def on_frame(self, scene, frame, state, pose, action):
        if frame % self.stride:
            return
        if self.camera is None:
            self._initialize(scene)
        scene.manager.Update()
        self._save(scene, frame, state, pose, action)
        self._drain_overhead(scene)

    def finish(self, scene, frame, state, pose, action):
        # The physics runner has just performed its unchanged terminal overhead
        # render, which also produced the final visualization-camera image.
        self._save(scene, frame, state, pose, action, terminal=True)
        np.save(self.out / "frame_times_s.npy", np.array([row["recording_time_s"] for row in self.rows]))
        runner.dump(self.out / "frame_metadata.json", {"schema": 1,
            "nominal_fps": self.fps, "frame_count": len(self.rows),
            "timing": "Every2 pre-interval telemetry rows at10fps by default; final frame is the measured terminal endpoint, which may add a half-period sample",
            "encoding_note": "Use exact timestamps for timing-sensitive analysis; constant-fps display can extend the last half-interval by up to0.05s",
            "frames": self.rows})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("execute",))
    p.add_argument("--case", required=True)
    p.add_argument("--route", required=True)
    p.add_argument("--anchor-observation", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chrono-data", required=True)
    p.add_argument("--horizon-s", type=float)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=10.)
    p.add_argument("--name", default="rollout")
    args = p.parse_args()
    args.backend, args.depth_ray_scale = "Vulkan_RT_lavapipe", 1.
    args.record_rgbd_stride, args.path_height_source = 0, "truth"
    args.frame_observer = RolloutCamera(args.out, width=args.width, height=args.height,
                                      fps=args.fps, name=args.name)
    runner.run_chrono(args)


if __name__ == "__main__":
    main()
