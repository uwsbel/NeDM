"""Videos of build9 camera-student episodes recorded in CPU MuJoCo (``vision_viral.py eval --record-states
--full-horizon``); ``render_tray_videos.py`` adapted to the colour-block goal and the lenient / strict success.

Two steps, two conda environments:

    # 1. the student's own camera images of every recorded state (Newton / Warp, the training renderer; nominal
    #    visuals, or --randomise with one draw per episode, the evaluation's randomised visuals)
    PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -P scripts/so101_policy/render_viral_videos.py cams \
        --eval EVAL.npz --student STUDENT.pt --out CAMS.npz [--randomise]
    # 2. MuJoCo side view (the collector scene + the visual block) next to those camera images, text, mp4 per episode
    MUJOCO_GL=egl PYTHONPATH=src ~/miniconda3/envs/nrdsim/bin/python -P scripts/so101_policy/render_viral_videos.py video \
        --eval EVAL.npz --cams CAMS.npz --title "..." --out DIR [--rows 0 3 7]

Frames: one per 20 ms control step at 50 frames per second (real time), the whole recorded horizon (with --full-horizon
the episode is not stopped at success or failure; the text shows when the lenient success first held and how the task
rule ended the episode).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

ENDS = ("running", "success", "timeout", "world_invalid", "world_arm_table", "lowest_point", "link_overlap",
        "jaw_overlap", "base_zone", "grasp_box", "keep_out", "release_high", "unsupported", "hold_cap", "cube_slide",
        "cube_tilted", "arm_self", "moved_untouched")      # nedm.so101_policy.task.ENDS (no torch import here)
END_NAMES = {**{k: n.replace("_", " ") for k, n in enumerate(ENDS)}, 1: "strict success", 2: "time-out",
             13: "held in the air > 3.5 s", 12: "cube unsupported (dropped)", 11: "released too high"}


def cmd_cams(a):
    import torch
    from nedm.so101_policy.vision import viral as VI
    ev = np.load(a.eval, allow_pickle=True)
    st, _ = VI.load_student(a.student, "cuda")
    views = VI.DRViews(64, st.cfg.res, st.cfg.cameras, dr=VI.DRConfig.from_dict(st.cfg.dr), seed=a.seed)
    q, cube = ev["state_q"], ev["state_cube"]                          # [E, S, 6], [E, S, 7] (pos, quat w x y z)
    rows = a.rows if a.rows else list(range(q.shape[0]))
    S = q.shape[1]
    goal = np.concatenate([ev["goal_xy"], ev["goal_yaw"][:, None]], 1)
    out = np.zeros((len(rows), S, len(st.cfg.cameras), st.cfg.res, st.cfg.res, 3), np.uint8)
    f = lambda v: torch.as_tensor(v, dtype=torch.float32, device="cuda")   # noqa: E731
    for i, e in enumerate(rows):
        dr = None
        if a.randomise:
            d1 = views.draw(1)
            dr = {k: v.expand(64, *v.shape[1:]).clone() for k, v in d1.items()}
        for s0 in range(0, S, 64):
            s1 = min(S, s0 + 64)
            n = s1 - s0
            img = views.render(f(q[e, s0:s1]), f(cube[e, s0:s1, :3]), f(cube[e, s0:s1, 3:]),
                               f(np.repeat(goal[e:e + 1], n, 0)), dr=None if dr is None else {k: v[:n] for k, v in
                                                                                             dr.items()},
                               light=False)
            out[i, s0:s1] = img.cpu().numpy()
    np.savez_compressed(a.out, cams=out, cameras=np.array(st.cfg.cameras), rows=np.asarray(rows))
    print(f"{a.out}: {out.shape}")


def cmd_video(a):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from nedm.contact_nrd.collect import so101_grasp as G
    import ast
    import nedm.contact_nrd as _cn
    src = (Path(_cn.__file__).parents[1] / "render" / "so101.py").read_text()   # the BLOCK_* constants (no torch here)
    C = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(src).body
         if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith("BLOCK_")}
    BH, BT, BC = C["BLOCK_HALF"], C["BLOCK_THICKNESS"], C["BLOCK_COLOR"]
    ev, cams = np.load(a.eval, allow_pickle=True), np.load(a.cams)
    cam_img, cam_names = cams["cams"], [str(c) for c in cams["cameras"]]
    cam_rows = [int(r) for r in cams["rows"]]
    rows = a.rows if a.rows else cam_rows
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

    def block(scn, gx, gy, yaw):           # the renderer's BLOCK_* geometry, drawn as a user geom (no physics)
        c, s = math.cos(yaw), math.sin(yaw)
        r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        if scn.ngeom < scn.maxgeom:
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([BH, BH, BT / 2]),
                                np.array([gx, gy, BT / 2]), r.reshape(9), np.asarray([*BC, 1], np.float32))
            scn.ngeom += 1

    big, small = ImageFont.load_default(size=20), ImageFont.load_default(size=15)
    S = ev["state_q"].shape[1]
    stills = []
    for e in rows:
        ci = cam_rows.index(e)
        gx, gy, gyaw = float(ev["goal_xy"][e, 0]), float(ev["goal_xy"][e, 1]), float(ev["goal_yaw"][e])
        end_code, t_end = int(ev["end"][e]), int(ev["steps"][e])
        len_ok, len_t = bool(ev["lenient"][e]), int(ev["lenient_t"][e])
        strict = end_code == 1
        frames = []
        for k in range(S):
            q = m.qpos0.copy()
            q[sc.qadr] = ev["state_q"][e, k]
            q[sc.cq: sc.cq + 7] = ev["state_cube"][e, k]
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            ren.update_scene(d, cam, opt)
            block(ren.scene, gx, gy, gyaw)
            side = Image.fromarray(ren.render())
            panel = Image.new("RGB", (P, H), (255, 255, 255))
            for j in range(min(2, cam_img.shape[2])):
                panel.paste(Image.fromarray(cam_img[ci, min(k, cam_img.shape[1] - 1), j]).resize((P - 10, P - 30)),
                            (5, j * P + 25))
            img = Image.new("RGB", (W + P, H + 60), (255, 255, 255))
            img.paste(side, (0, 60))
            img.paste(panel, (W, 60))
            dr = ImageDraw.Draw(img)
            dr.text((10, 6), f"{a.title}   |   CPU MuJoCo (ground truth), task seed {int(ev['seeds'][e])}", fill=(0, 0, 0),
                    font=big)
            parts = []
            if len_ok and k + 1 >= len_t:
                parts.append(f"lenient success at {len_t * 0.02:.2f} s")
            if k + 1 >= t_end:
                parts.append(f"task rule ended it at {t_end * 0.02:.2f} s: "
                             f"{END_NAMES.get(end_code, 'end ' + str(end_code))} (video continues)")
            state = "; ".join(parts) if parts else "running"
            good = len_ok and k + 1 >= len_t
            dr.text((10, 34), f"t = {k * 0.02:4.2f} s    {state}", fill=(0, 110, 0) if good else
                    ((170, 0, 0) if k + 1 >= t_end else (40, 40, 40)), font=small)
            for j, n in enumerate(cam_names[:2]):
                dr.text((W + 8, 60 + j * P + 5), f"student input: {n} camera", fill=(40, 40, 40), font=small)
            frames.append(np.asarray(img))
        cube = ev["state_cube"][e, -1]
        c, s = math.cos(gyaw), math.sin(gyaw)
        dx, dy = cube[0] - gx, cube[1] - gy
        bx, by = c * dx + s * dy, -s * dx + c * dy
        inside = max(abs(bx), abs(by)) <= BH
        msg = (f"{'LENIENT SUCCESS' if len_ok else 'NO SUCCESS'}"
               f"{' + STRICT' if strict else ''}; at {S * 0.02:.0f} s the cube centre is "
               f"{1e3 * math.hypot(dx, dy):.1f} mm from the block centre ({'inside' if inside else 'outside'} the block)")
        last = Image.fromarray(frames[-1])
        dr = ImageDraw.Draw(last)
        dr.rectangle([0, H + 25, W + P, H + 60], fill=(225, 245, 228) if len_ok else (250, 225, 225))
        dr.text((10, H + 32), msg, fill=(0, 90, 0) if len_ok else (150, 0, 0), font=small)
        frames += [np.asarray(last)] * 50
        name = f"block_seed{int(ev['seeds'][e])}_{'success' if len_ok else 'fail'}.mp4"
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
    p.add_argument("--rows", type=int, nargs="*")
    p.add_argument("--randomise", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("video")
    p.add_argument("--eval", required=True), p.add_argument("--cams", required=True), p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--rows", type=int, nargs="*")
    a = ap.parse_args()
    {"cams": cmd_cams, "video": cmd_video}[a.cmd](a)


if __name__ == "__main__":
    main()
