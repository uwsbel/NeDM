"""Videos of recorded CPU MuJoCo evaluation episodes (``eval_policy.py run --world mujoco`` output with per-step arrays).

    MUJOCO_GL=egl PYTHONPATH=src python -P scripts/so101_policy/render_eval_videos.py --eval RUN.npz --tasks TASKS.npz \
        --rows 0 40 80 120 160 --title "PPO trained only in the NRD" --out DIR

The frames are the recorded MuJoCo states of the evaluation itself (one per 20 ms control step, played at 50 frames per
second = real time), drawn with the collector's scene: a side view and a high front view side by side. The goal is drawn on the
table as a translucent green copy of the cube at the goal pose and a ring of the 15 mm success radius. Text: policy,
task, time, the decoder phase, and at the end the outcome with the final cube error. The last frame is held for 1 s.
numpy, mujoco, imageio (ffmpeg), pillow; no torch.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nedm.contact_nrd.collect import so101_grasp as G

MODES = ("free", "g_brake", "g_descend", "g_settle", "g_close", "g_grip", "g_lift", "hold", "p_brake", "p_settle",
         "p_lower", "p_open", "p_retreat")   # nedm.so101_policy.decoder.MODES (no torch import here)
PHASE = {"free": "move", "g_brake": "grasp: align", "g_descend": "grasp: descend", "g_settle": "grasp: descend",
         "g_close": "grasp: close jaw", "g_grip": "grasp: grip", "g_lift": "grasp: lift", "hold": "carry",
         "p_brake": "place: stop", "p_settle": "place: stop", "p_lower": "place: lower", "p_open": "place: open jaw",
         "p_retreat": "place: retreat"}
W, H, DT = 640, 480, 0.02
GREEN = np.array([0.15, 0.75, 0.25])


def quat_from_yaw(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def add_geom(scn, kind, size, pos, mat, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], kind, np.asarray(size, float), np.asarray(pos, float),
                        np.asarray(mat, float).reshape(9), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def goal_geoms(scn, gxy, gyaw, half):
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, quat_from_yaw(gyaw))
    add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [half, half, half], [gxy[0], gxy[1], half], r, [*GREEN, 0.28])
    eye = np.eye(3).reshape(9)
    for k in range(48):   # the 15 mm success ring: short capsules on a circle
        a0, a1 = 2 * math.pi * k / 48, 2 * math.pi * (k + 1) / 48
        p0 = np.array([gxy[0] + 0.015 * math.cos(a0), gxy[1] + 0.015 * math.sin(a0), 0.0006])
        p1 = np.array([gxy[0] + 0.015 * math.cos(a1), gxy[1] + 0.015 * math.sin(a1), 0.0006])
        if scn.ngeom < scn.maxgeom:
            mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0007, p0, p1)
            scn.geoms[scn.ngeom].rgba[:] = [*GREEN, 0.9]
            scn.ngeom += 1
    del eye


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True), ap.add_argument("--tasks", required=True)
    ap.add_argument("--rows", type=int, nargs="+", required=True), ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ev, tk = np.load(a.eval, allow_pickle=True), np.load(a.tasks)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    sc = G.Scene(G.load_config())
    m = sc.m
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, W), max(m.vis.global_.offheight, H)
    d = mujoco.MjData(m)
    ren = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption()
    front, top = mujoco.MjvCamera(), mujoco.MjvCamera()
    for c in (front, top):
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
    px, py = sc.cfg["workspace"]["pan_axis_xy_m"]
    front.lookat[:], front.distance, front.azimuth, front.elevation = [px + 0.15, py, 0.07], 0.68, 90.0, -28.0   # side
    top.lookat[:], top.distance, top.azimuth, top.elevation = [px + 0.17, py, 0.02], 0.72, 180.0, -48.0       # high front
    big, small = ImageFont.load_default(size=20), ImageFont.load_default(size=16)
    half = float(sc.half)
    stills = []
    for row in a.rows:
        n_steps = int(ev["t_end"][row]) + 1
        gxy, gyaw = tk["goal_xy"][row], float(tk["goal_yaw"][row])
        ok = bool(ev["success"][row])
        err_mm, err_deg = 1e3 * float(ev["end_xy"][row]), math.degrees(float(ev["end_yaw"][row]))
        frames = []
        for k in range(n_steps):
            q = m.qpos0.copy()
            q[sc.qadr] = ev["q"][row, k]
            q[sc.cq: sc.cq + 7] = ev["cube"][row, k]
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            views = []
            for cam in (front, top):
                ren.update_scene(d, cam, opt)
                goal_geoms(ren.scene, gxy, gyaw, half)
                views.append(ren.render())
            img = Image.fromarray(np.concatenate(views, 1))
            dr = ImageDraw.Draw(img)
            mode = MODES[int(ev["mode"][row, k])] if int(ev["mode"][row, k]) < len(MODES) else "?"
            dr.rectangle([0, 0, 2 * W, 58], fill=(255, 255, 255))
            dr.text((10, 6), f"{a.title}   |   MuJoCo (ground truth), test task {row} (seed {int(ev['seeds'][row])})",
                    fill=(0, 0, 0), font=big)
            dr.text((10, 32), f"t = {k * DT:4.2f} s    phase: {PHASE.get(mode, mode)}    green = goal pose, ring = 15 mm",
                    fill=(40, 40, 40), font=small)
            frames.append(np.asarray(img))
        last = Image.fromarray(frames[-1])
        dr = ImageDraw.Draw(last)
        msg = (f"{'SUCCESS' if ok else 'FAILED (' + str(ev['end'][row]) + ')'}: cube {err_mm:.1f} mm and "
               f"{err_deg:.1f} deg from the goal, t = {n_steps * DT - DT:.2f} s")
        dr.rectangle([0, H - 40, 2 * W, H], fill=(225, 245, 228) if ok else (250, 225, 225))
        dr.text((10, H - 32), msg, fill=(0, 90, 0) if ok else (150, 0, 0), font=big)
        frames += [np.asarray(last)] * 50
        path = out / f"test_task{row:03d}_{'success' if ok else 'fail'}.mp4"
        imageio.mimwrite(path, frames, fps=50, quality=8, macro_block_size=8)
        stills.append(np.asarray(last.resize((W, H // 2))))
        print(f"{path}: {n_steps} steps, {msg}")
    imageio.imwrite(out / "last_frames.png", np.concatenate(stills, 0))
    ren.close()


if __name__ == "__main__":
    main()
