"""Chrono step server for closed-loop control of the SO-101 push-T scene (one episode per process).

Run with the Chrono Python (scripts/so101_push/cluster/chrono_env.sh). Protocol: one JSON object per line on stdin,
one JSON reply per line on stdout.

  {"cmd": "init", "config": "<collector config>", "q_start": [5], "first_cmd": [5],
   "t_pose": [x, y, yaw] (optional, default: the config start pose), "diagnostics": false (optional)}
      Builds the scene as run_episode does (T at t_pose, arm at q_start), settles for settle_s holding
      first_cmd (not recorded) and returns record 0.
  {"cmd": "step", "q_cmds": [[5], ...]}
      Applies each command for one control step (zero-order hold, records every record_step_s, the explicit PD law
      at every physics step) exactly as run_episode's loop. Returns the new records.
  {"cmd": "close"}

Diagnostics ("diagnostics": true at init; default off, and when off every message is unchanged): record 0 and every
step record also carry the per-record values that run_episode stores in the collector shards, with the same indexing:
  tau [5]        the joint torques of the last physics step of the record interval (run_episode's tau[r] for the
                 interval r -> r+1; record 0 reports the torques of the last settle step)
  lock_err [2]   scene.lock_errors() at the record
  finger_gap, link_gap   scene.gaps() at the record (m)
  arm_t_force    the largest per-physics-step sum of arm-T normal forces in the record interval (N; 0 at record 0)
The gap check costs about 0.3 s per 4 s episode. Diagnostics do not change the dynamics.

Optional render (local visual check): the init message may hold
  "render": {"dir": <frame folder>, "width": 960, "height": 720, "camera": {"pos": [3], "target": [3]},
             "goal": [x, y, yaw] or null, "backend": "irrlicht" (default; needs pychrono.irrlicht and a display) or
             "sensor" (Chrono::Sensor GPU camera, headless; optional "hfov_rad" 0.9, "supersample" 2)}
Then the scene gets visual shapes (no effect on the dynamics) and one PNG is written after the settle (frame 0) and after
every control step (frame k = control steps since the settle), so frame j_a is the start state of a goal episode.

A record: arm [19] (q, qd, tcp, tcp velocity, angular velocity), t [13] (COM, wxyz with sign continuity, velocity,
angular velocity), contacts_link [7] (0/1 over the record interval), arm_table (0/1), t_table (0/1).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from nedm.so101_push.geometry import TShape
from nedm.so101_push.robot import ArmModel, load_config
from nedm.so101_push.sim import LINK_ORDER, PushScene


class Renderer:
    def __init__(self, scene, r):
        import os

        import pychrono as chrono
        import pychrono.irrlicht as irr
        self.dir = r["dir"]
        os.makedirs(self.dir, exist_ok=True)
        scene.add_visuals(r.get("goal"))
        vis = irr.ChVisualSystemIrrlicht()
        vis.AttachSystem(scene.sys)
        vis.SetWindowSize(int(r.get("width", 960)), int(r.get("height", 720)))
        vis.SetWindowTitle("SO-101 push-T (Chrono)")
        vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
        vis.Initialize()
        vis.AddLightDirectional(65, 45, chrono.ChColor(0.5, 0.5, 0.5), chrono.ChColor(0.1, 0.1, 0.1), chrono.ChColor(0.75, 0.75, 0.75))
        vis.AddLightDirectional(35, 225, chrono.ChColor(0.0, 0.0, 0.0), chrono.ChColor(0.0, 0.0, 0.0), chrono.ChColor(0.35, 0.35, 0.35))
        cam = r.get("camera") or {}
        p, t = cam.get("pos", [0.6, -0.3, 0.35]), cam.get("target", [0.4, 0.0, 0.0])
        self.cam_p, self.cam_t = chrono.ChVector3d(*map(float, p)), chrono.ChVector3d(*map(float, t))
        vis.AddCamera(self.cam_p, self.cam_t)
        self.vis, self.k = vis, 0

    def frame(self):
        vis = self.vis
        vis.Run()
        # the window is visible on the desktop: undo any key / mouse input (panels, camera moves) before each frame
        vis.ShowProfiler(False)
        vis.ShowInfoPanel(False)
        vis.ShowExplorer(False)
        vis.SetCameraPosition(self.cam_p)
        vis.SetCameraTarget(self.cam_t)
        vis.BeginScene()
        vis.Render()
        vis.EndScene()
        vis.WriteImageToFile(f"{self.dir}/frame_{self.k:05d}.png")
        self.k += 1


class SensorRenderer:
    """Headless renderer (render "backend": "sensor"): a Chrono::Sensor GPU camera on a fixed anchor body, one frame per
    control step written as frame_%05d.png (same numbering as Renderer). Needs no display. Visual shapes as Renderer."""

    def __init__(self, scene, r):
        import os

        import pychrono as chrono
        import pychrono.sensor as sens
        from scipy.spatial.transform import Rotation
        self.dir = r["dir"]
        os.makedirs(self.dir, exist_ok=True)
        scene.add_visuals(r.get("goal"))
        w, h = int(r.get("width", 960)), int(r.get("height", 720))
        cam = r.get("camera") or {}
        p = np.asarray(cam.get("pos", [0.6, -0.3, 0.35]), float)
        t = np.asarray(cam.get("target", [0.4, 0.0, 0.0]), float)
        fwd = (t - p) / np.linalg.norm(t - p)                 # sensor camera looks along its +x, z up
        left = np.cross([0.0, 0.0, 1.0], fwd)
        left /= np.linalg.norm(left)
        up = np.cross(fwd, left)
        qx, qy, qz, qw = Rotation.from_matrix(np.stack([fwd, left, up], axis=1)).as_quat()
        self.anchor = chrono.ChBody()
        self.anchor.SetFixed(True)
        scene.sys.Add(self.anchor)
        self.manager = sens.ChSensorManager(scene.sys)
        self.manager.scene.AddPointLight(chrono.ChVector3f(0.3, -0.6, 1.2), chrono.ChColor(1, 1, 1), 5.0)
        self.manager.scene.AddPointLight(chrono.ChVector3f(-0.4, 0.5, 0.9), chrono.ChColor(0.5, 0.5, 0.5), 5.0)
        self.manager.scene.SetAmbientLight(chrono.ChVector3f(0.35, 0.35, 0.35))
        if hasattr(self.manager.scene, "SetSceneEpsilon"):  # ray offset: stops shadow speckles on the overlapping link hulls
            self.manager.scene.SetSceneEpsilon(float(r.get("scene_epsilon", 2e-3)))
        bg = sens.Background()
        bg.mode = sens.BackgroundMode_SOLID_COLOR
        bg.color_zenith = chrono.ChVector3f(0.12, 0.16, 0.22)
        self.manager.scene.SetBackground(bg)
        self.dt = float(r.get("frame_dt", 0.02))
        self.cam = sens.ChCameraSensor(self.anchor, 1.0 / self.dt, chrono.ChFramed(chrono.ChVector3d(*p), chrono.ChQuaterniond(qw, qx, qy, qz)),
                                       w, h, float(r.get("hfov_rad", 0.9)), int(r.get("supersample", 2)))
        self.cam.SetLag(0)
        self.cam.SetCollectionWindow(0)
        self.cam.PushFilter(sens.ChFilterRGBA8Access())
        self.manager.AddSensor(self.cam)
        self.k, self.last, self.img = 0, 0, None

    def frame(self):
        import time

        from PIL import Image
        self.manager.Update()
        t0 = time.time()
        while True:                                            # wait for the launch of this control step (async GPU render)
            buf = self.cam.GetMostRecentRGBA8Buffer()
            if buf.HasData() and buf.LaunchedCount != self.last:
                self.last = buf.LaunchedCount
                self.img = Image.fromarray(np.asarray(buf.GetRGBA8Data())[::-1, :, :3].copy())
                break
            if time.time() - t0 > 2.0:                         # no new launch: keep the frame numbering, repeat the last image
                break
            time.sleep(0.002)
        if self.img is not None:
            self.img.save(f"{self.dir}/frame_{self.k:05d}.png")
        self.k += 1


def _diag(scene, arm_t_force):
    """Per-record diagnostics (module docstring)."""
    fg, lg = scene.gaps()
    e_m, e_rad = scene.lock_errors()
    return {"tau": scene.tau.tolist(), "lock_err": [float(e_m), float(e_rad)], "finger_gap": float(fg),
            "link_gap": float(lg), "arm_t_force": float(arm_t_force)}


def main():
    scene = None
    renderer = None
    diagnostics = False
    out = sys.stdout
    sys.stdout = sys.stderr          # Chrono or library prints must not corrupt the protocol
    prev_q = None
    for line in sys.stdin:
        msg = json.loads(line)
        if msg["cmd"] == "close":
            break
        if msg["cmd"] == "init":
            cfg = load_config(msg["config"])
            mc = json.loads(json.dumps(cfg, sort_keys=True))          # as episode._models (without its scripted Planner)
            model, ts = ArmModel(mc["robot"], mc["robot"]["description"]), TShape(mc["tshape"])
            sim = cfg["simulation"]
            dt, rec, ctrl = float(sim["step_s"]), float(sim["record_step_s"]), float(sim["control_step_s"])
            spr, rpc = int(round(rec / dt)), int(round(ctrl / rec))
            t_pose = np.asarray(msg.get("t_pose") or cfg["tshape"]["start_pose"], float)
            diagnostics = bool(msg.get("diagnostics", False))
            scene = PushScene(cfg, model, ts, np.asarray(msg["q_start"], float), t_pose)
            lo, hi = scene.cmd_limits[:, 0], scene.cmd_limits[:, 1]
            q_cmd = np.clip(np.asarray(msg["first_cmd"], float), lo, hi)
            for _ in range(int(round(float(sim["settle_s"]) / dt))):
                scene.step(q_cmd)
            t = scene.t_state()
            if t[3] < 0:
                t[3:7] *= -1
            prev_q = t[3:7].copy()
            if msg.get("render"):
                if msg["render"].get("backend") == "sensor":
                    # Chrono::Sensor / OptiX print from C++ to file descriptor 1 (the protocol pipe): keep the pipe on a
                    # duplicate descriptor and point descriptor 1 at stderr before the sensor starts
                    import os
                    out.flush()
                    proto = os.dup(1)
                    os.dup2(2, 1)
                    out = os.fdopen(proto, "w", buffering=1)
                    renderer = SensorRenderer(scene, msg["render"])
                else:
                    renderer = Renderer(scene, msg["render"])
                renderer.frame()
            reply = {"arm": scene.arm_state().tolist(), "t": t.tolist()}
            if diagnostics:
                reply.update(_diag(scene, 0.0))
        elif msg["cmd"] == "step":
            recs = []
            for qc in msg["q_cmds"]:
                q_cmd = np.clip(np.asarray(qc, float), lo, hi)
                for _ in range(rpc):
                    fl = np.zeros(len(LINK_ORDER), dtype=bool)
                    tt = at = False
                    fmax = 0.0
                    for _ in range(spr):
                        f, a_tab, a_arm, fn = scene.step(q_cmd)
                        fl |= f
                        tt |= a_tab
                        at |= a_arm
                        if fn > fmax:
                            fmax = fn
                    t = scene.t_state()
                    if t[3:7] @ prev_q < 0:                # quaternion sign continuity (as run_episode)
                        t[3:7] *= -1
                    prev_q = t[3:7].copy()
                    rec_ = {"arm": scene.arm_state().tolist(), "t": t.tolist(), "contacts_link": fl.astype(int).tolist(),
                            "arm_table": int(at), "t_table": int(tt)}
                    if diagnostics:
                        rec_.update(_diag(scene, fmax))
                    recs.append(rec_)
                if renderer is not None:
                    renderer.frame()
            reply = {"records": recs}
        else:
            reply = {"error": "unknown cmd " + str(msg.get("cmd"))}
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
