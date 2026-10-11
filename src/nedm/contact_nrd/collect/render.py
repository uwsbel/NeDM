"""Videos and contact timelines of simulator-versus-NRD rollouts (the .npz of ``nedm.contact_nrd.rollouts``).

    MUJOCO_GL=egl PYTHONPATH=src python -m nedm.contact_nrd.collect.render ROLLOUTS.npz --out DIR \\
        [--data DIR] [--layout overlay|side] [--slow F] [--stills S ...] [--replay]

Per episode, in DIR (stem = <system>_<mode>_row<row>_<family>):

| File | Contents |
|---|---|
| ``<stem>.mp4`` | One frame per model step (20 ms), played at 50 / F frames per second (F = --slow, default 1) |
| ``<stem>_contacts.png`` | Contact timeline: per pair the recorded label and the NRD switch of each step, and the steps where they differ |
| ``<stem>_t<S>s.png`` | Stills at S seconds after the rollout start (--stills); with --replay a simulator-replay panel on the left |
| ``<ROLLOUTS stem>_replay_check.json`` | --replay: pose differences between the simulator replay and the recorded states |

Scene: the system of the data's system.json (--data; default: the data directory written in the .npz). G1: the
collector's model (g1_locomotion.build_model; foot geoms of the data's physics_config). SO101: the collector's Scene
(table and cube of the data's physics_config). State to qpos: G1 = [base position, base quaternion wxyz, 27 joint
angles] (the first 34 channels); SO101 = the 6 arm joint angles and the cube free joint [position, quaternion].
mj_forward on qpos only (velocities are not needed to draw a pose).

Layout ``overlay`` (default): the recorded motion is drawn solid (the SO101 arm in grey: its own yellow and orange
are too close to the NRD colour) and the NRD prediction is an orange tint with a 2 px orange outline in the same
scene. A drift of millimetres to centimetres shows as the outline leaves the solid body, which two side-by-side
images do not show; where the copies coincide the body is evenly tinted. The two copies are rendered apart and
composited by depth (a transparent copy drawn into the same scene z-fights with the solid one where they coincide).
``side``: the recorded motion and the prediction in two panels with the same camera. Camera: G1 follows the
recorded pelvis slowly (1 s centred moving average of its x, y); SO101 is fixed for each episode, at right angles to
the line from the arm base to the cube's start.
Text in each frame: time, episode, errors of the target body (data.body_errors, as the evaluation), and a strip per
contact pair (the recorded label in the evaluation's switch view against the NRD switch, over the whole rollout,
with a cursor at the current step).

--replay re-runs the collector's physics from the recorded start state (with the file's instantaneous rates) with
the recorded commands, and compares its poses with the recorded ones: a check that the recorded copy is drawn at the
simulator's poses. numpy, mujoco, imageio (ffmpeg) and pillow; no torch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SURFACE, INK, INK2, TRACK = (252, 252, 251), (11, 11, 11), (82, 81, 78), (226, 225, 221)
SKY = (232, 235, 239)
RECORD_GREY = (0.78, 0.78, 0.78)
REC, NRD = (42, 120, 214), (235, 104, 52)          # categorical slots 1 and 2 (blue, orange)
GHOST_ALPHA = 0.45
HEADER_LINES = 5
TITLES = {"g1_locomotion": "G1 walking", "so101_grasp": "SO101 cube grasping"}
UNITS = {"pos_m": ("position", 1000.0, "mm", 1), "angle_rad": ("orientation", 180 / np.pi, "deg", 2),
         "joint_rms_rad": ("joint RMS", 1.0, "rad", 4)}


def font(size):
    return ImageFont.load_default(size=size)


# ---- scene ---------------------------------------------------------------------------------------------------
class SystemScene:
    """The collector's MuJoCo model of a system and the map from an NRD state [D, S] (padded) to its qpos."""

    def __init__(self, system):
        self.name = system["name"]
        pc = system.get("physics_config") or {}
        if self.name == "g1_locomotion":
            from nedm.contact_nrd.collect import g1_locomotion as mod
            cfg = mod.load_config()
            cfg["physics"].update({k: pc[k] for k in ("foot", "sole_box_min", "sole_box_max") if k in pc})
            self.model = mod.build_model(cfg)
        elif self.name == "so101_grasp":
            from nedm.contact_nrd.collect import so101_grasp as mod
            cfg = mod.load_config()
            for k in ("table", "cube"):
                cfg[k] = {**cfg[k], **pc.get(k, {})}
            self.scene = mod.Scene(cfg)
            self.model = self.scene.m
        else:
            raise SystemExit(f"no scene for system {self.name!r} (g1_locomotion, so101_grasp)")
        self.mod, self.cfg = mod, cfg
        m = self.model
        moving = m.body_weldid[m.geom_bodyid] != 0
        # geoms that the record draws in RECORD_GREY: the SO101 arm is yellow and orange, too close to the NRD
        # colour (the cube keeps its blue); the G1 is grey already
        self.grey = moving & (m.geom_bodyid != self.scene.cube_body) if self.name == "so101_grasp" else moving & False

    def qpos(self, x):
        q = self.model.qpos0.copy()
        if self.name == "g1_locomotion":
            q[:34] = x[0, :34]
            q[3:7] /= np.linalg.norm(q[3:7])
        else:
            s = self.scene
            q[s.qadr] = x[0, :6]
            q[s.cq: s.cq + 7] = x[1, :7]
            q[s.cq + 3: s.cq + 7] /= np.linalg.norm(q[s.cq + 3: s.cq + 7])
        return q

    def parts(self, q):
        """qpos -> {free-body position, quaternion, joint angles} of the moving bodies (for the replay check)."""
        if self.name == "g1_locomotion":
            return dict(base_pos=q[:3], base_quat=q[3:7], joints=q[7:34])
        s = self.scene
        return dict(cube_pos=q[s.cq: s.cq + 3], cube_quat=q[s.cq + 3: s.cq + 7], joints=q[s.qadr])

    def replay(self, x0, actions, stride):
        """The collector's physics from the recorded state x0 [D, S] (instantaneous rates) with the recorded command
        of each model step, actions [H, D, A] -> qpos [H+1, nq] on the model grid."""
        if self.name == "g1_locomotion":
            states, _ = self.mod.replay(self.cfg, x0[0, :67], np.repeat(actions[:, 0, :27], stride, 0),
                                        len(actions) * stride + 1)
            return np.array([self.qpos(x[None]) for x in states[::stride]])
        s, m = self.scene, self.model
        d = mujoco.MjData(m)
        d.qpos[:] = self.qpos(x0)
        d.qvel[s.vadr] = x0[0, 6:12]
        d.qvel[s.cv: s.cv + 3] = x0[1, 7:10]
        d.qvel[s.cv + 3: s.cv + 6] = self.mod.quat_mat(d.qpos[s.cq + 3: s.cq + 7]).T @ x0[1, 10:13]
        mujoco.mj_forward(m, d)
        out = [d.qpos.copy()]
        for a in actions:
            d.ctrl[s.act] = a[0, :6]
            for _ in range(stride * s.nsub):
                mujoco.mj_step(m, d)
            out.append(d.qpos.copy())
        return np.array(out)


class Painter:
    """Offscreen renders of the scene at a recorded pose, optionally with the NRD pose as a second, semi-transparent
    copy. The two copies are rendered apart and composited by depth: the prediction is blended over the record where
    it is in front of the record or at the same depth, so coinciding copies give one evenly tinted body (a
    transparent geom drawn into the same scene z-fights with the opaque one there). The moving bodies of the record
    can be drawn in one colour; the background is light."""

    def __init__(self, sc, width, height):
        m = self.m = sc.model
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
        m.vis.global_.offheight = max(m.vis.global_.offheight, height)
        self.r = mujoco.Renderer(m, height, width)
        self.opt = mujoco.MjvOption()
        self.d = mujoco.MjData(m)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.moving = m.body_weldid[m.geom_bodyid] != 0
        self.pert = mujoco.MjvPerturb()
        self.far = 0.999 * m.vis.map.zfar * m.stat.extent       # depth of the background (Renderer depth, m)
        self.grey = sc.grey if sc.grey.any() else None

    def _pass(self, q, color=None, which=None, ghost=False):
        """RGB and depth (m) of the scene at qpos q, and its pixels in front of the far plane; color (0-1 RGB)
        replaces the colour of the geoms in which ([ngeom] bool). ghost: only the moving bodies, without shadows and
        reflections (a software renderer spends most of a frame on them)."""
        self.d.qpos[:] = q
        mujoco.mj_forward(self.m, self.d)
        scn = self.r.scene
        if ghost:
            mujoco.mjv_updateScene(self.m, self.d, self.opt, self.pert, self.cam, int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
                                   scn)
        else:
            self.r.update_scene(self.d, self.cam, self.opt)
        if color is not None:
            for i in range(scn.ngeom):
                g = scn.geoms[i]
                if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and which[g.objid]:
                    g.rgba[:3], g.matid = color, -1
        flags = scn.flags.copy()
        if ghost:
            scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        rgb = self.r.render().astype(np.float64)
        self.r.enable_depth_rendering()
        depth = self.r.render().copy()
        self.r.disable_depth_rendering()
        scn.flags[:] = flags
        return rgb, depth, depth < self.far

    def image(self, q, ghost=None):
        """The scene at qpos q (the scene's grey geoms in RECORD_GREY); ghost: the NRD qpos, blended in orange where
        it is in front of the record or level with it, with an orange outline (2 px) there."""
        rgb, depth, scene = self._pass(q, None if self.grey is None else RECORD_GREY, self.grey)
        rgb[~scene] = SKY
        if ghost is not None:
            g_rgb, g_depth, g_moving = self._pass(ghost, np.array(NRD) / 255, self.moving, ghost=True)
            front = g_moving & (g_depth <= depth * 1.001 + 1e-3)
            rgb[front] = (1 - GHOST_ALPHA) * rgb[front] + GHOST_ALPHA * g_rgb[front]
            inner = g_moving.copy()
            for _ in range(2):
                pad = np.pad(inner, 1)
                inner = inner & pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
            rgb[front & ~inner] = NRD
        return rgb.round().astype(np.uint8)

    def close(self):
        self.r.close()


def camera_track(sc, recorded, steps, dt):
    """Camera settings of each frame. G1: looks at the 1 s centred moving average of the recorded pelvis x, y.
    SO101: fixed for the episode, at right angles to the line from the arm base to the cube's start, 15 deg towards
    the front of the gripper."""
    if sc.name == "g1_locomotion":
        xy = recorded[: steps + 1, 0, :2]
        half = round(0.5 / dt)
        pad = np.concatenate((np.repeat(xy[:1], half, 0), xy, np.repeat(xy[-1:], half, 0)))
        kernel = np.ones(2 * half + 1) / (2 * half + 1)
        smooth = np.stack([np.convolve(pad[:, c], kernel, "valid") for c in range(2)], 1)
        return [dict(lookat=(x, y, 0.55), distance=2.8, azimuth=135.0, elevation=-20.0) for x, y in smooth]
    c = recorded[0, 1, :2]
    view = dict(lookat=(0.55 * c[0], 0.55 * c[1], 0.1), distance=0.75,
                azimuth=float(np.degrees(np.arctan2(c[1], c[0]))) + 105.0, elevation=-25.0)
    return [view] * (steps + 1)


def set_camera(cam, c):
    cam.lookat[:] = c["lookat"]
    cam.distance, cam.azimuth, cam.elevation = c["distance"], c["azimuth"], c["elevation"]


# ---- drawing -------------------------------------------------------------------------------------------------
def runs(on):
    """Start and end (exclusive) of the runs of True in a 1-D bool array."""
    x = np.concatenate(([0], np.asarray(on, np.int8), [0]))
    d = np.diff(x)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def strip_height(pairs):
    return 26 + len(pairs) * 26


def draw_strip(draw, box, labels, switches, steps, k, pairs, t0, t1):
    """Contact strip in box (x0, y0, x1, y1): per pair the recorded label (blue) and the NRD switch (orange) of each
    step of the rollout, a cursor at step k."""
    x0, y0, x1, _ = box
    f, fs = font(12), font(11)
    left = x0 + 150
    draw.rectangle((left, y0 + 6, left + 10, y0 + 14), fill=REC)
    draw.text((left + 14, y0 + 3), "recorded (step or step before)", font=fs, fill=INK2)
    draw.rectangle((left + 220, y0 + 6, left + 230, y0 + 14), fill=NRD)
    draw.text((left + 234, y0 + 3), "NRD switch", font=fs, fill=INK2)
    draw.text((x1 - 4, y0 + 3), f"{t0:.2f} - {t1:.2f} s", font=fs, fill=INK2, anchor="ra")
    n = max(1, steps)
    xs = lambda j: left + (x1 - left) * j / n
    y = y0 + 22
    for p, name in enumerate(pairs):
        draw.text((x0 + 8, y + 9), name, font=f, fill=INK, anchor="lm")
        for row, (src, color) in enumerate(((labels, REC), (switches, NRD))):
            ya = y + row * 10
            draw.rectangle((left, ya, x1, ya + 7), fill=TRACK)
            for a, b in runs(src[:steps, p]):
                draw.rectangle((xs(a), ya, max(xs(a) + 1, xs(b) - 1), ya + 7), fill=color)
        y += 26
    xc = xs(min(k, n))
    draw.line((xc, y0 + 20, xc, y - 6), fill=INK, width=2)


def error_line(err, r, k, target):
    """Errors of the target body at step k (as the evaluation computes them), then the other bodies'."""
    bodies = [target] + sorted({key.split("/")[1] for key in err} - {target})
    parts = []
    for b in bodies:
        items = [(m, err[f"error/{b}/{m}"][r, k]) for m in UNITS if f"error/{b}/{m}" in err]
        if items:
            parts.append(f"{b} " + ", ".join(f"{UNITS[m][0]} {UNITS[m][1] * v:.{UNITS[m][3]}f} {UNITS[m][2]}"
                                             for m, v in items))
    return "error: " + "; ".join(parts)


# ---- timeline ------------------------------------------------------------------------------------------------
def timeline(path, title, note, time_s, steps, labels, switches, pairs, counts, acc, dt, tol):
    """Contact timeline PNG: per pair the recorded label (switch view) and the NRD switch of each 20 ms step, a
    thin row where they differ, the switch accuracy and the event counts of the evaluation."""
    left, plot_w, right = 300, 1100, 30
    block = 54
    top = 92
    height = top + len(pairs) * block + 50
    img = Image.new("RGB", (left + plot_w + right, height), SURFACE)
    d = ImageDraw.Draw(img)
    d.text((16, 12), title, font=font(18), fill=INK)
    d.text((16, 38), note, font=font(12), fill=INK2)
    lx = left
    for color, text in ((REC, "recorded: the pair touched in the step or the step before (switch view)"),
                        (NRD, "NRD: collision switch on"), (INK2, "differ")):
        h = 12 if color != INK2 else 4
        d.rectangle((lx, 64 + (12 - h) // 2, lx + 14, 64 + (12 - h) // 2 + h), fill=color)
        d.text((lx + 20, 63), text, font=font(12), fill=INK2)
        lx += 30 + font(12).getbbox(text)[2]
    n = max(1, steps)
    xs = lambda j: left + plot_w * j / n
    t0, t1 = time_s[0], time_s[0] + steps * dt
    span = t1 - t0
    tick = 0.5 if span <= 6 else 1.0
    ticks = np.arange(np.ceil(t0 / tick) * tick, t1 + 1e-9, tick)
    for t in ticks:
        x = left + plot_w * (t - t0) / max(span, 1e-9)
        d.line((x, top - 6, x, top + len(pairs) * block - 10), fill=TRACK, width=1)
        d.text((x, top + len(pairs) * block - 4), f"{t:.1f} s", font=font(12), fill=INK2, anchor="ma")
    for p, name in enumerate(pairs):
        y = top + p * block
        c = counts[p]
        d.text((16, y - 2), name, font=font(14), fill=INK)
        d.text((16, y + 16), f"switch accuracy {acc[p]:.3f}", font=font(12), fill=INK2)
        d.text((16, y + 31), f"onsets {c[0, 0]} rec / {c[0, 1]} NRD, offsets {c[1, 0]} / {c[1, 1]}; "
                             f"missed {c[:, 2].sum()}, false {c[:, 3].sum()}", font=font(11), fill=INK2)
        for row, (src, color, tag) in enumerate(((labels, REC, "recorded"), (switches, NRD, "NRD"))):
            ya = y + row * 17
            d.text((left - 8, ya + 6), tag, font=font(11), fill=INK2, anchor="rm")
            d.rectangle((left, ya, left + plot_w, ya + 13), fill=TRACK)
            for a, b in runs(src[:steps, p]):
                d.rectangle((xs(a), ya, max(xs(a) + 1, xs(b) - 1), ya + 13), fill=color)
        ya = y + 36
        for a, b in runs(labels[:steps, p] != switches[:steps, p]):
            d.rectangle((xs(a), ya, max(xs(a) + 1, xs(b) - 1), ya + 4), fill=INK2)
    d.text((left + plot_w, height - 16), f"one cell = one {1000 * dt:.0f} ms step; event counts match within "
                                         f"{tol} steps, as in the evaluation", font=font(11), fill=INK2, anchor="rm")
    img.save(path)


# ---- simulator replay check ----------------------------------------------------------------------------------
def replay_check(sc, painter, replay, recorded, cams, steps, dt, times=(0.02, 0.2, 0.5, 1.0, 2.0, 4.0)):
    """Differences between the simulator replay (qpos [H+1, nq]) and the recorded states at the given seconds after
    the rollout start and at its end: free-body position (mm) and orientation (deg), largest joint-angle difference
    (rad), and the mean absolute pixel difference of the two renders (0-255 per channel)."""
    out = {}
    for s in sorted({*times, steps * dt}):
        k = round(s / dt)
        if k > steps:
            continue
        a, b = sc.parts(replay[k]), sc.parts(sc.qpos(recorded[k]))
        diff = {}
        for name in a:
            if name.endswith("_quat"):
                rel = np.empty(4)
                mujoco.mju_mulQuat(rel, a[name] * np.array([1, -1, -1, -1]), b[name])
                diff[name[:-5] + "_angle_deg"] = float(np.degrees(2 * np.arctan2(np.linalg.norm(rel[1:]), abs(rel[0]))))
            elif name.endswith("_pos"):
                diff[name + "_mm"] = float(1000 * np.linalg.norm(a[name] - b[name]))
            else:
                diff[name + "_max_rad"] = float(np.abs(a[name] - b[name]).max())
        set_camera(painter.cam, cams[k])
        painter.image(replay[k])   # the first render after a camera change still differs in a few thousand pixels
        pix = np.abs(painter.image(replay[k]).astype(int) - painter.image(sc.qpos(recorded[k])).astype(int))
        diff["mean_pixel_diff"] = float(pix.mean())
        out[f"{k * dt:.2f}s"] = diff
    return out


# ---- main ----------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollouts", type=Path, help=".npz of nedm.contact_nrd.rollouts")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--data", type=Path, help="data directory (its system.json); default: the one in the .npz")
    parser.add_argument("--layout", choices=("overlay", "side"), default="overlay")
    parser.add_argument("--slow", type=float, default=1.0, help="slow motion factor (video at 50 / F fps)")
    parser.add_argument("--size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"), help="render size")
    parser.add_argument("--stills", type=float, nargs="*", default=[], metavar="S",
                        help="PNG stills at these seconds after the rollout start")
    parser.add_argument("--replay", action="store_true", help="simulator replay check (stills get a replay panel)")
    parser.add_argument("--no-video", action="store_true", help="only the timelines and stills")
    args = parser.parse_args()
    z = np.load(args.rollouts)
    meta = json.loads(str(z["meta"]))
    data_dir = args.data or Path(meta["data"])
    system = json.loads((data_dir / "system.json").read_text())
    if system["name"] != meta["system"]:
        raise SystemExit(f"{data_dir} holds {system['name']}, the rollouts {meta['system']}")
    sc = SystemScene(system)
    w, h = args.size
    painter = Painter(sc, w, h)
    pairs, dt, mode = meta["pairs"], meta["dt_s"], meta["mode"]
    stride = round(dt / meta["record_step_s"])
    hh = HEADER_LINES * 19 + 10                                  # text band above the render
    hs = -(-(hh + h + strip_height(pairs)) // 16) * 16 - h - hh  # frame height a multiple of 16 (H.264)
    err = {k: z[k] for k in z.files if k.startswith("error/")}
    legend = {"so101_grasp": "grey arm and blue cube: simulator record; orange tint and outline: NRD prediction"}.get(
        meta["system"], "solid: simulator record; orange tint and outline: NRD prediction")
    how = ("from the first recorded state" if mode == "first" else
           f"from the recorded {meta['history']}-state history at {meta['start_s']:.2f} s") + ", recorded commands"
    args.out.mkdir(parents=True, exist_ok=True)
    checks = []
    for r, row in enumerate(z["episode"].tolist()):
        fam = str(z["family_name"][r]) or "no family"
        split = {0: "train", 1: "validation", 2: "test"}.get(int(z["split"][r]), "split ?")
        steps = int(z["steps"][r])
        rec, pred, t = z["recorded"][r], z["predicted"][r], z["time_s"]
        tag = "first" if mode == "first" else f"window{meta['start_s']:.2f}s"
        stem = f"{meta['system']}_{tag}_row{row:05d}_{fam}"
        title = f"{TITLES.get(meta['system'], meta['system'])}, episode {int(z['episode_id'][r])} ({fam}, {split})"
        model = Path(meta["model"])
        note = (f"NRD {how}. Row {row} of {data_dir.parent.name}/{data_dir.name}. "
                f"Checkpoint {model.parent.name}/{model.name}.")
        timeline(args.out / f"{stem}_contacts.png", title, note, t, steps, z["labels"][r], z["switches"][r], pairs,
                 z["event_counts"][r], z["gate_accuracy"][r], dt, meta["contact_tolerance_steps"])
        finite = np.isfinite(pred.reshape(len(pred), -1)).all(1)
        bad = int(np.argmin(finite)) if not finite.all() else None
        cams = camera_track(sc, rec, steps, dt)
        replay = sc.replay(z["start_raw"][r], z["actions"][r, :steps], stride) if args.replay else None

        def frame(k, with_replay=False):
            set_camera(painter.cam, cams[k])
            q_rec = sc.qpos(rec[k])
            q_pred = sc.qpos(pred[k]) if finite[k] else None
            if args.layout == "side":
                imgs = [painter.image(q_rec)] + ([painter.image(q_pred)] if q_pred is not None else
                                                 [np.full((h, w, 3), SURFACE, np.uint8)])
                names = ["simulator record", "NRD prediction"]
            else:
                imgs, names = [painter.image(q_rec, q_pred)], [None]
            if with_replay:
                imgs, names = [painter.image(replay[k])] + imgs, ["simulator replay (MuJoCo re-run)"] + names
            canvas = Image.new("RGB", (w * len(imgs), hh + h + hs), SURFACE)
            for i, im in enumerate(imgs):
                canvas.paste(Image.fromarray(im), (i * w, hh))
            lines = [title, f"NRD {how}",
                     f"t = {t[k]:.2f} s in the episode, +{t[k] - t[0]:.2f} s from the rollout start",
                     error_line(err, r, k, meta["target_body"]) if finite[k] else
                     f"NRD prediction not finite from t = {t[bad]:.2f} s",
                     legend if args.layout == "overlay" else "same camera in both panels"]
            d = ImageDraw.Draw(canvas)
            for i, text in enumerate(lines):
                d.text((10, 6 + 19 * i), text, font=font(14), fill=INK if i != 4 else INK2)
            for i, name in enumerate(names):
                if name:
                    d.text((i * w + w // 2, hh + 8), name, font=font(15), fill=INK, anchor="ma")
            draw_strip(d, (0, hh + h, w * len(imgs), hh + h + hs), z["labels"][r], z["switches"][r], steps, k,
                       pairs, t[0], t[0] + steps * dt)
            return np.asarray(canvas.convert("RGB"))

        if not args.no_video:
            import imageio
            path = args.out / f"{stem}.mp4"
            with imageio.get_writer(path, fps=50.0 / args.slow, codec="libx264", quality=8,
                                    macro_block_size=16) as writer:
                for k in range(steps + 1):
                    writer.append_data(frame(k))
            print(path, flush=True)
        for s in args.stills:
            k = int(np.clip(round(s / dt), 0, steps))
            Image.fromarray(frame(k, args.replay)).save(args.out / f"{stem}_t{k * dt:.2f}s.png")
        print(args.out / f"{stem}_contacts.png", flush=True)
        if replay is not None:
            item = dict(row=row, family=fam, start_s=float(t[0]),
                        at=replay_check(sc, painter, replay, rec, cams, steps, dt))
            checks.append(item)
            print(json.dumps(item), flush=True)
    if checks:
        (args.out / f"{args.rollouts.stem}_replay_check.json").write_text(json.dumps(checks, indent=1) + "\n")
    painter.close()


if __name__ == "__main__":
    main()
