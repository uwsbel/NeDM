"""Videos of tray-student episodes recorded in CPU MuJoCo (``vision_tray.py eval --record-states [--full-horizon]``).

Two steps, two conda environments:

    # 1. the student's own camera images of every recorded state (Madrona, the training renderer)
    source ~/so101_policy/runs/render/madrona_luffy/env_luffy.sh
    PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -P scripts/so101_policy/render_tray_videos.py cams \
        --eval EVAL.npz --student STUDENT.pt --out CAMS.npz
    # 2. MuJoCo side view (the collector scene + the visual tray) next to those camera images, text, mp4 per episode
    MUJOCO_GL=egl PYTHONPATH=src ~/miniconda3/envs/nrdsim/bin/python -P scripts/so101_policy/render_tray_videos.py video \
        --eval EVAL.npz --cams CAMS.npz --title "..." --out DIR

Frames: one per 20 ms control step at 50 frames per second (real time), the whole recorded horizon (with --full-horizon
the episode is not stopped at success or failure; the text shows when the task rule ended it and how).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

END_NAMES = {1: "success", 2: "time-out"}


def cmd_cams(a):
    import torch
    from nedm.so101_policy.vision import dataset as V
    from nedm.so101_policy.vision import tray as TR
    ev = np.load(a.eval, allow_pickle=True)
    st, _ = TR.load_student(a.student, "cuda")
    views = V.Views(64, st.cfg.res, st.cfg.cameras, goal="tray")
    q, cube = ev["state_q"], ev["state_cube"]                          # [E, S, 6], [E, S, 7] (pos, quat w x y z)
    E, S = q.shape[:2]
    goal = np.concatenate([ev["goal_xy"], ev["goal_yaw"][:, None]], 1)
    out = np.zeros((E, S, len(st.cfg.cameras), st.cfg.res, st.cfg.res, 3), np.uint8)
    f = lambda v: torch.as_tensor(v, dtype=torch.float32, device="cuda")   # noqa: E731
    for e in range(E):
        for s0 in range(0, S, 64):
            s1 = min(S, s0 + 64)
            n = s1 - s0
            img = views.render(f(q[e, s0:s1]), f(cube[e, s0:s1, :3]), f(cube[e, s0:s1, 3:]),
                               f(np.repeat(goal[e:e + 1], n, 0)))
            out[e, s0:s1] = img.cpu().numpy()
    np.savez_compressed(a.out, cams=out, cameras=np.array(st.cfg.cameras))
    print(f"{a.out}: {out.shape}")


def cmd_video(a):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from nedm.contact_nrd.collect import so101_grasp as G
    import ast
    import nedm.contact_nrd as _cn
    src = (Path(_cn.__file__).parents[1] / "render" / "so101.py").read_text()   # the TRAY_* constants (that module needs torch)
    TR_ = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(src).body
           if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith("TRAY_")}
    TRAY_BASE_THICKNESS, TRAY_COLOR, TRAY_INNER_HALF = TR_["TRAY_BASE_THICKNESS"], TR_["TRAY_COLOR"], TR_["TRAY_INNER_HALF"]
    TRAY_RIM_COLOR, TRAY_RIM_HEIGHT, TRAY_RIM_WIDTH = TR_["TRAY_RIM_COLOR"], TR_["TRAY_RIM_HEIGHT"], TR_["TRAY_RIM_WIDTH"]
    ev, cams = np.load(a.eval, allow_pickle=True), np.load(a.cams)
    cam_img, cam_names = cams["cams"], [str(c) for c in cams["cameras"]]
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    sc = G.Scene(G.load_config())
    m = sc.m
    W, H, P = 640, 480, 240                                            # MuJoCo view, camera panel side
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, W), max(m.vis.global_.offheight, H)
    d, ren, opt = mujoco.MjData(m), mujoco.Renderer(m, H, W), mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    px, py = sc.cfg["workspace"]["pan_axis_xy_m"]
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = [px + 0.15, py, 0.06], 0.70, 90.0, -30.0

    def box(scn, half, pos, rmat, rgba):
        if scn.ngeom < scn.maxgeom:
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.asarray(half, float),
                                np.asarray(pos, float), rmat.reshape(9), np.asarray(rgba, np.float32))
            scn.ngeom += 1

    def tray(scn, gx, gy, yaw):            # the renderer's TRAY_* geometry, drawn as user geoms (no physics)
        c, s = math.cos(yaw), math.sin(yaw)
        r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        hi, w, hr, tb = TRAY_INNER_HALF, TRAY_RIM_WIDTH, TRAY_RIM_HEIGHT, TRAY_BASE_THICKNESS
        box(scn, [hi + w, hi + w, tb / 2], [gx, gy, tb / 2], r, [*TRAY_COLOR, 1])
        for sx, sy, hx, hy in ((1, 0, w / 2, hi + w), (-1, 0, w / 2, hi + w), (0, 1, hi + w, w / 2), (0, -1, hi + w, w / 2)):
            off = r @ np.array([sx * (hi + w / 2), sy * (hi + w / 2), 0.0])
            box(scn, [hx, hy, hr / 2], [gx + off[0], gy + off[1], hr / 2], r, [*TRAY_RIM_COLOR, 1])

    big, small = ImageFont.load_default(size=20), ImageFont.load_default(size=15)
    E, S = ev["state_q"].shape[:2]
    stills = []
    for e in range(E):
        gx, gy, gyaw = float(ev["goal_xy"][e, 0]), float(ev["goal_xy"][e, 1]), float(ev["goal_yaw"][e])
        end_code, t_end = int(ev["end"][e]), int(ev["steps"][e])
        ok = end_code == 1
        frames = []
        for k in range(S):
            q = m.qpos0.copy()
            q[sc.qadr] = ev["state_q"][e, k]
            q[sc.cq: sc.cq + 7] = ev["state_cube"][e, k]
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            ren.update_scene(d, cam, opt)
            tray(ren.scene, gx, gy, gyaw)
            side = Image.fromarray(ren.render())
            panel = Image.new("RGB", (P, H), (255, 255, 255))
            for j in range(min(2, cam_img.shape[2])):
                panel.paste(Image.fromarray(cam_img[e, min(k, cam_img.shape[1] - 1), j]).resize((P - 10, P - 30)),
                            (5, j * P + 25))
            img = Image.new("RGB", (W + P, H + 60), (255, 255, 255))
            img.paste(side, (0, 60))
            img.paste(panel, (W, 60))
            dr = ImageDraw.Draw(img)
            dr.text((10, 6), f"{a.title}   |   CPU MuJoCo (ground truth), task seed {int(ev['seeds'][e])}", fill=(0, 0, 0), font=big)
            state = "running" if k < t_end else (f"task rule ended it at {t_end * 0.02:.2f} s: "
                                                 f"{END_NAMES.get(end_code, 'end ' + str(end_code))} (video continues)")
            dr.text((10, 34), f"t = {k * 0.02:4.2f} s    {state}", fill=(0, 110, 0) if (k >= t_end and ok) else
                    ((170, 0, 0) if k >= t_end else (40, 40, 40)), font=small)
            for j, n in enumerate(cam_names[:2]):
                dr.text((W + 8, 60 + j * P + 5), f"student input: {n} camera", fill=(40, 40, 40), font=small)
            frames.append(np.asarray(img))
        cube = ev["state_cube"][e, -1]
        err = 1e3 * math.hypot(cube[0] - gx, cube[1] - gy)
        msg = (f"{'SUCCESS' if ok else 'FAILED (' + END_NAMES.get(end_code, str(end_code)) + ')'} by the task rule; "
               f"at 8 s the cube is {err:.1f} mm from the tray centre (rule at its end: {1e3 * float(ev['end_xy'][e]):.1f} mm, "
               f"{math.degrees(float(ev['end_yaw'][e])):.1f} deg)")
        last = Image.fromarray(frames[-1])
        dr = ImageDraw.Draw(last)
        dr.rectangle([0, H + 25, W + P, H + 60], fill=(225, 245, 228) if ok else (250, 225, 225))
        dr.text((10, H + 32), msg, fill=(0, 90, 0) if ok else (150, 0, 0), font=small)
        frames += [np.asarray(last)] * 50
        name = f"tray_seed{int(ev['seeds'][e])}_{'success' if ok else 'fail'}.mp4"
        imageio.mimwrite(out / name, frames, fps=50, quality=8, macro_block_size=8)
        stills.append(np.asarray(last.resize(((W + P) // 2, (H + 60) // 2))))
        print(f"{out / name}: {msg}")
    imageio.imwrite(out / "last_frames.png", np.concatenate(stills, 0))
    ren.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cams")
    p.add_argument("--eval", required=True), p.add_argument("--student", required=True), p.add_argument("--out", required=True)
    p = sub.add_parser("video")
    p.add_argument("--eval", required=True), p.add_argument("--cams", required=True), p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    a = ap.parse_args()
    {"cams": cmd_cams, "video": cmd_video}[a.cmd](a)


if __name__ == "__main__":
    main()
