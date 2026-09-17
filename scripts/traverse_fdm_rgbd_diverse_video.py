#!/usr/bin/env python3
"""Passive chase camera for headless-first online Chrono evaluation."""
from pathlib import Path
import math

import numpy as np

from traverse_fdm_rgbd_video import RolloutCamera


class HeadlessRolloutCamera(RolloutCamera):
    def _initialize(self, scene):
        if scene.manager is None:
            import pychrono as chrono
            import pychrono.sensor as sens
            from nedm.traverse.scene import SKY_RGB
            scene.manager = sens.ChSensorManager(scene.system)
            scene.manager.scene.SetAmbientLight(chrono.ChVector3f(.35, .35, .38))
            scene.manager.scene.AddDirectionalLight(
                chrono.ChColor(1., .95, .85), math.radians(45.), math.radians(120.))
            background = sens.Background()
            background.mode = sens.BackgroundMode_SOLID_COLOR
            background.color_zenith = chrono.ChVector3f(*SKY_RGB)
            scene.manager.scene.SetBackground(background)
        super()._initialize(scene)
        # The inherited camera renders real bodies; its original provenance
        # named the older fixed-reference runner, so replace only that metadata.
        import json
        from traverse_fdm_rgbd_diverse_online import dump, sha256
        path = self.out / "video_camera.json"
        metadata = json.loads(path.read_text())
        metadata["physics_script_sha256"] = sha256(Path(__file__).with_name("traverse_fdm_rgbd_diverse_online.py"))
        metadata["video_adapter_sha256"] = sha256(__file__)
        metadata["model_input_boundary"] = "Chase sensor created after fixed-map launch equality; never a model input"
        dump(path, metadata)

    @staticmethod
    def _drain_overhead(scene):
        for tap in (scene.rgb_tap, scene.depth_tap):
            if tap is None:
                continue
            buffer = tap._get_buffer()
            if buffer.HasData() and buffer.LaunchedCount > tap.taken_count:
                tap.take()

    def finish(self, scene, frame, state, pose, action):
        from traverse_fdm_rgbd_diverse_online import dump
        if self.camera is None:
            self._initialize(scene)
        scene.manager.Update()
        self._save(scene, frame, state, pose, action, terminal=True)
        self._drain_overhead(scene)
        np.save(self.out / "frame_times_s.npy", np.asarray([r["recording_time_s"] for r in self.rows]))
        dump(self.out / "frame_metadata.json", {
            "schema": 1, "nominal_fps": self.fps, "frame_count": len(self.rows),
            "timing": "Pre-interval physical states plus an actual terminal endpoint; use per-frame timestamps",
            "encoding_note": "Constant-fps playback can extend the final partial interval by less than one video period",
            "frames": self.rows})


class OnlineVideoObserver:
    """Forward telemetry without changing its order or simulation advances."""
    def __init__(self, telemetry, out, *, fps=5.):
        self.telemetry = telemetry
        self.camera = HeadlessRolloutCamera(out, fps=fps, name="actual_online_mppi")

    def on_frame(self, scene, frame, state, pose, action, command_context=None):
        self.telemetry.on_frame(scene, frame, state, pose, action, command_context=command_context)
        self.camera.on_frame(scene, frame, state, pose, action)

    def on_substep(self, *args, **kwargs):
        return self.telemetry.on_substep(*args, **kwargs)

    def on_post_substep(self, *args, **kwargs):
        return self.telemetry.on_post_substep(*args, **kwargs)

    def finish(self, scene, N, terminal_state, terminal_pose, last_action):
        result = self.telemetry.finish(scene, N, terminal_state, terminal_pose, last_action)
        self.camera.finish(scene, N, terminal_state, terminal_pose, last_action)
        return result
