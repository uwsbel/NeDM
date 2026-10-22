#!/usr/bin/env python
"""SO101 camera student on the real robot (LeRobot 0.6.2 ``so101_follower``) or on a recorded LeRobot dataset.

DRAFT. Only the replay mode and the fake-robot backend were run (luffy, 2026-10-21). The dry and live modes were
NEVER run on the robot. README.md (this folder) says how to run them on the Orin and what is checked before motion.

Modes
- ``replay`` (offline, safe): a LeRobot v3 recording (data/chunk-000/file-000.parquet + the two mp4 files) is played
  at the robot loop's 50 Hz: the follower's recorded joint state (linear interpolation of the 30 Hz samples) is the
  measured q, the newest recorded frame at or before each query is the camera image. The student, the qdot estimate,
  the action layer and the guard run exactly as on the robot; nothing is sent anywhere. Logs the student's chunk and
  executed targets, its auxiliary predictions (plate x, y, yaw mod 90; cube x, y, z) next to the recorded leader
  actions and, if calib_v1.json has them, the episode's cube start and plate pose.
- ``dry`` (robot connected, NO motor writes): opens the follower's serial bus (LeRobot handshake = reads only) and the
  two cameras, never calls ``robot.connect`` / ``configure`` (they toggle torque and write registers), blocks every
  write method of the bus. Torque stays as it is. Runs the student loop on the measured q (move the arm by hand with
  torque off to see the outputs) and logs.
- ``live`` (moves the robot; needs ``--i-am-supervising``, a real calib_v1.json joint map and typed confirmations):
  connect, optional LeRobot motor configuration (``--configure auto``: only if the registers differ), torque on at the
  present position, a slow joint-space move to HOME (inside the sim start distribution), then the student loop at
  50 Hz (a query every 0.1 s, 5 targets executed, each through the action layer and the guard), time limit, stop on
  q / space / Esc / Ctrl+C or on any guard reason; then HOLD (arm at the measured q, gripper at its last command,
  torque on) and a menu (hold and exit / slow return to the start pose / torque off).

Examples (luffy, from the worktree root)
    ~/miniconda3/envs/so101_vision/bin/python scripts/so101_policy/real2sim/hw/run_student_real.py replay \
        --student ~/so101_policy/runs/viral/final_student_o17_it03000.pt \
        --episodes green_cube_red_plate_1:0 --out ~/so101_r2s_claude/hw/replay_test
    ... run_student_real.py live --backend fake --auto-confirm ...      # the robot loop on a simulated servo
"""
from __future__ import annotations

import argparse
import json
import math
import os
import select
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_SRC = Path(os.environ.get("NEDM_SRC", HERE.parents[3] / "src"))
if str(_SRC) not in sys.path:
    sys.path.insert(1, str(_SRC))

import hw_lib as H  # noqa: E402

# Robot machine (proxy notes SO101_CAMERA_SETUP.md / SO101_DATA_COLLECTION.md, 2026-10-21)
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115318-if00"
ROBOT_ID = "so101_follower"
CAM_PATHS = {"wrist": "/dev/v4l/by-path/platform-3610000.usb-usb-0:4.1.1:1.0-video-index0",
             "external": "/dev/v4l/by-path/platform-3610000.usb-usb-0:4.1.4:1.0-video-index0"}
# by-path (USB port), not by-id: both cameras report the serial 20250606105, so the by-id link follows the last plug-in
CAM_BACKEND = 200                      # cv2.CAP_V4L2, as the recordings (lerobot-record ... backend: 200)
HF_CALIB = Path("~/.cache/huggingface/lerobot/calibration/robots/so_follower").expanduser()
LOCAL_CALIB_COPY = Path("~/so101_r2s_claude/proxy_docs/so101_follower.json").expanduser()
DEFAULT_CALIB_V1 = Path("~/so101_r2s_claude/calib/calib_v1.json").expanduser()


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


# ---- LeRobot robot (dry / live) ------------------------------------------------------------------------------------
BUS_WRITERS = ("write", "sync_write", "enable_torque", "disable_torque", "write_calibration", "configure_motors",
               "set_operating_mode", "setup_motors", "setup_motor", "reset_calibration", "set_half_turn_homings",
               "record_ranges_of_motion", "record_calibration", "set_baudrate")
ROBOT_WRITERS = ("connect", "configure", "calibrate", "send_action", "setup_motors")


def block_writes(bus, robot=None):
    """Dry mode: replace every method of the LeRobot bus (and robot) that writes to a motor or changes the torque by
    one that raises. -> names blocked."""
    def blocked(name):
        def f(*a, **k):
            raise RuntimeError(f"dry mode: {name} is blocked (no motor writes)")
        return f
    done = []
    for obj, names, tag in ((bus, BUS_WRITERS, "bus"), (robot, ROBOT_WRITERS, "robot")):
        if obj is None:
            continue
        for n in names:
            if hasattr(obj, n):
                setattr(obj, n, blocked(f"{tag}.{n}"))
                done.append(f"{tag}.{n}")
    return done


# LeRobot 0.6.2 tools that open the robot bus or the cameras
DEVICE_TOOLS = ("lerobot-teleoperate", "lerobot-record", "lerobot-replay", "lerobot-rollout", "lerobot-calibrate",
                "lerobot-setup-motors", "lerobot-find-port", "lerobot-find-cameras", "lerobot-find-joint-limits")


def device_users(paths, skip=()):
    """Processes that can use the robot bus or a camera at the same time as this run: own processes that have one of
    ``paths`` open (/proc/<pid>/fd), and a LeRobot robot tool or a dry / live run of this script of ANY user (all
    users can read the command lines; only root can read the open files of other users' processes).
    -> ["pid user: why: command", ...]"""
    import pwd
    skip = {os.getpid(), *skip}
    want = {os.path.realpath(os.path.expanduser(str(p))) for p in paths}
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) in skip:
            continue
        try:
            argv = [a.decode(errors="replace") for a in (d / "cmdline").read_bytes().split(b"\0") if a]
            uid = d.stat().st_uid
        except OSError:
            continue
        if not argv:
            continue
        why = []
        try:
            fds = list((d / "fd").iterdir())
        except OSError:                                   # another user's process: its open files are not readable
            fds = []
        held = set()
        for f in fds:
            try:
                t = os.readlink(f)
            except OSError:
                continue
            if t in want:
                held.add(t)
        if held:
            why.append("has " + ", ".join(sorted(held)) + " open")
        names = [a.rsplit("/", 1)[-1].replace("_", "-") for a in argv[:3]]
        if any(n == t or n.endswith("." + t) for n in names for t in DEVICE_TOOLS):
            why.append("LeRobot robot tool")
        if any(a.endswith("run_student_real.py") for a in argv[:3]) and {"dry", "live"} & set(argv):
            why.append("dry / live run of this script")
        if why:
            try:
                user = pwd.getpwuid(uid).pw_name
            except KeyError:
                user = str(uid)
            out.append(f"{d.name} {user}: {'; '.join(why)}: {' '.join(argv)[:150]}")
    return out


class LeRobotIO:
    """LeRobot 0.6.2 SO follower + its two OpenCV cameras, used below the ``Robot`` API so that nothing is written
    unless asked: ``connect`` = bus handshake (reads model numbers and firmware) + calibration check (reads) + cameras;
    ``writes=False`` replaces every writing method of the bus and of the robot by one that raises."""

    def __init__(self, port, robot_id, cam_paths, writes, calibration_dir=None, fps=30):
        from lerobot.cameras.opencv.configuration_opencv import ColorMode, Cv2Backends, OpenCVCameraConfig
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
        cams = {n: OpenCVCameraConfig(index_or_path=Path(p), fps=fps, width=640, height=480, fourcc="MJPG",
                                      backend=Cv2Backends(CAM_BACKEND), color_mode=ColorMode.RGB)
                for n, p in cam_paths.items()}
        cfg = SOFollowerRobotConfig(port=port, id=robot_id, cameras=cams, disable_torque_on_disconnect=False,
                                    use_degrees=True, max_relative_target=None,
                                    calibration_dir=Path(calibration_dir) if calibration_dir else None)
        self.robot = SOFollower(cfg)
        self.bus = self.robot.bus
        self.writes = bool(writes)
        self.bgr = False                                  # LeRobot cameras deliver RGB (color_mode rgb)
        if not self.writes:
            block_writes(self.bus, self.robot)

    def calibration_dict(self):
        return {m: dict(id=c.id, drive_mode=c.drive_mode, homing_offset=c.homing_offset, range_min=c.range_min,
                        range_max=c.range_max) for m, c in self.robot.calibration.items()}

    def connect(self):
        if not self.robot.calibration:
            raise RuntimeError(f"no LeRobot calibration file {self.robot.calibration_fpath}: refusing (do not "
                               f"calibrate from this script)")
        self.bus.connect()                                # handshake: reads only
        try:
            if not self.bus.is_calibrated:
                raise RuntimeError("the motors' calibration registers differ from the calibration file: refusing "
                                   "(LeRobot would ask to recalibrate; investigate first)")
            for cam in self.robot.cameras.values():
                cam.connect()
        except BaseException:
            self.close(False)
            raise

    def read_q(self):
        d = self.bus.sync_read("Present_Position", num_retry=2)
        return np.asarray([d[m] for m in H.JOINTS], np.float64)

    def frames(self, max_age_ms=1000):
        out = {}
        for n, cam in self.robot.cameras.items():
            img = cam.read_latest(max_age_ms=int(max_age_ms))
            ts = getattr(cam, "latest_timestamp", None)
            out[n] = (img, (time.perf_counter() - ts) * 1e3 if ts else float("nan"))
        return out

    def write_goal(self, q_real):
        if not self.writes:
            raise RuntimeError("dry mode: write_goal blocked")
        self.bus.sync_write("Goal_Position", {m: float(v) for m, v in zip(H.JOINTS, q_real)})

    def torque_on_at_present(self):
        """Goal = present position (raw ticks), then torque on: no jump to an old goal register."""
        raw = self.bus.sync_read("Present_Position", normalize=False, num_retry=2)
        self.bus.sync_write("Goal_Position", raw, normalize=False)
        self.bus.enable_torque(num_retry=2)

    def disable_torque(self):
        self.bus.disable_torque(num_retry=5)

    def config_diffs(self):
        """Registers that LeRobot's ``SOFollower.configure`` would change -> list of strings (reads only)."""
        cfg = self.robot.config
        want = {"P_Coefficient": cfg.position_p_coefficient, "I_Coefficient": cfg.position_i_coefficient,
                "D_Coefficient": cfg.position_d_coefficient, "Return_Delay_Time": 0}
        diffs = []
        for m in self.bus.motors:
            w = dict(want)
            has = getattr(self.bus, "_has_register", None)     # private; not in every LeRobot release
            for r in ("Maximum_Acceleration", "Acceleration"):
                try:
                    if has is not None and has(m, r):
                        w[r] = 254
                except Exception:                          # noqa: BLE001 (skip this register check)
                    pass
            try:
                w["Operating_Mode"] = self.bus._definition(m).operating_modes["position"]
            except Exception:                              # noqa: BLE001 (private API; skip the check)
                pass
            if m == "gripper":
                w.update(Max_Torque_Limit=500, Protection_Current=250, Overload_Torque=25)
            for r, v in w.items():
                try:
                    got = self.bus.read(r, m, normalize=False, num_retry=1)
                except Exception as e:                     # noqa: BLE001
                    diffs.append(f"{m}.{r}: read failed ({e})")
                    continue
                if int(got) != int(v):
                    diffs.append(f"{m}.{r} = {got} (LeRobot writes {v})")
        return diffs

    def configure(self):
        """LeRobot's configure (torque off, register writes, torque on) after goal = present position."""
        raw = self.bus.sync_read("Present_Position", normalize=False, num_retry=2)
        self.bus.sync_write("Goal_Position", raw, normalize=False)
        self.robot.configure()

    def close(self, disable_torque=False):
        try:
            if self.bus.is_connected:
                self.bus.disconnect(disable_torque=bool(disable_torque) and self.writes)
        finally:
            for cam in self.robot.cameras.values():
                try:
                    if cam.is_connected:
                        cam.disconnect()
                except Exception as e:                     # noqa: BLE001
                    log(f"camera disconnect: {e}")


# ---- keyboard / Ctrl+C stop ----------------------------------------------------------------------------------------
class Stopper:
    """Stop request from Ctrl+C (SIGINT) or the keys q, Q, space, Esc in this terminal (cbreak mode while active, no
    Enter needed). poll() is non-blocking."""

    def __init__(self):
        self.flag = False
        self.why = ""
        self.fd = None
        self.old_tty = None
        self.old_sig = None

    def __enter__(self):
        self.flag = False
        self.old_sig = signal.signal(signal.SIGINT, self._sig)
        if sys.stdin.isatty():
            import termios
            import tty
            self.fd = sys.stdin.fileno()
            self.old_tty = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def _sig(self, *_):
        self.flag, self.why = True, "Ctrl+C"

    def poll(self):
        if self.fd is not None:
            while select.select([self.fd], [], [], 0)[0]:
                ch = os.read(self.fd, 1)
                if ch in (b"q", b"Q", b" ", b"\x1b"):
                    self.flag, self.why = True, f"key {ch!r}"
        return self.flag

    def __exit__(self, *exc):
        if self.old_tty is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_tty)
        signal.signal(signal.SIGINT, self.old_sig or signal.default_int_handler)
        self.fd = self.old_tty = None
        return False


def confirm(word, text, auto=False):
    print(text, flush=True)
    if auto:
        print(f"(fake backend: auto-confirmed {word})")
        return True
    try:
        return input(f"Type {word} to continue, anything else aborts: ").strip() == word
    except (EOFError, KeyboardInterrupt):
        return False


# ---- shared set-up ------------------------------------------------------------------------------------------------
def lerobot_calib_path(args):
    if args.lerobot_calib:
        return Path(args.lerobot_calib).expanduser()
    p = (Path(args.calibration_dir).expanduser() if args.calibration_dir else HF_CALIB) / f"{args.robot_id}.json"
    return p if p.exists() else LOCAL_CALIB_COPY


def build_common(args):
    calib = H.load_calib(args.calib)
    jm = H.JointMap.from_calib(calib, str(args.calib))
    if args.q_offset_deg:                              # extra joint-map offset (sim deg; e.g. the table-tilt correction)
        qo = np.radians(np.asarray([float(v) for v in args.q_offset_deg.split(",")], np.float64))
        if qo.shape != (5,):
            raise SystemExit("--q-offset-deg needs 5 values (pan, lift, elbow, wrist_flex, wrist_roll; sim deg)")
        jm.offset_rad = jm.offset_rad + qo
        jm.source = f"{jm.source} + --q-offset-deg {args.q_offset_deg}"
    kp = Path(args.kin_points).expanduser() if args.kin_points else H.default_kin_points()
    layer = H.ActionLayer(kp, table_z=args.table_z)
    gcfg = H.GuardConfig.from_dict(json.loads(args.guard) if args.guard else {})
    import torch
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = H.StudentPolicy(args.student, dev, args.cam_map or None, args.jaw_lead, None if args.amp is None else
                             bool(args.amp))
    views = H.view_specs(calib, H.REAL_CAMERAS, policy.res, interp=args.interp)
    # a real2sim student pins the crops it was trained with (cfg.crops, keyed by the student's camera names): use them;
    # a --crop that differs from them is refused unless --allow-crop-change (the images would differ from training)
    want = getattr(policy.cfg, "calib_sha256", "") or ""
    if want and args.calib:                            # the joint map must be the one the student was trained with
        import hashlib
        got = hashlib.sha256(Path(args.calib).expanduser().read_bytes()).hexdigest()
        if got != want:
            msg = f"--calib {args.calib} sha256 {got[:12]} != the student's calibration {want[:12]}"
            if getattr(args, "mode", "") == "live" and not args.allow_calib_change:
                raise SystemExit(msg + " (live refused; --allow-calib-change to override)")
            log("WARNING: " + msg)
    pinned = {policy.cam_map[c]: tuple(int(v) for v in b) for c, b in (getattr(policy.cfg, "crops", None) or {}).items()
              if c in policy.cam_map}
    for cam, box in pinned.items():
        views[cam] = H.ViewSpec(box, (policy.res, policy.res), args.interp, "student checkpoint (cfg.crops)")
    if args.crop:                                      # override: "external=x,y,w,h;wrist=x,y,w,h"
        for part in args.crop.split(";"):
            cam, box = part.split("=")
            cam, box = cam.strip(), tuple(int(v) for v in box.split(","))
            if cam in pinned and box != pinned[cam] and not args.allow_crop_change:
                raise SystemExit(f"--crop {cam}={box} differs from the student's training crop {pinned[cam]} "
                                 "(add --allow-crop-change to use it anyway)")
            views[cam] = H.ViewSpec(box, (policy.res, policy.res), args.interp, "--crop")
    if not pinned:
        log("NOTE: the student has no pinned crops (cfg.crops); the views come from --calib / --crop")
    info = dict(student=policy.ckpt, student_cfg=dict(cameras=policy.cams, res=policy.res, chunk=policy.chunk,
                                                      n_exec=policy.n_exec, hist=policy.H, jaw_lead=policy.jaw_lead,
                                                      amp=policy.amp), device=dev, cam_map=policy.cam_map,
                calib=str(args.calib) if calib is not None else None, joint_map=jm.describe(), kin_points=str(kp),
                table_z=args.table_z, guard=gcfg.__dict__, qdot_window=args.qdot_window,
                views={k: dict(crop_xywh=v.crop_xywh, out_wh=v.out_wh, interp=v.interp, source=v.source)
                       for k, v in views.items()})
    return calib, jm, layer, gcfg, policy, views, info


def print_setup(info):
    jm = info["joint_map"]
    log(f"student {info['student']} cameras {info['student_cfg']['cameras']} <- {info['cam_map']} res "
        f"{info['student_cfg']['res']} device {info['device']}")
    log(f"joint map: {'PLACEHOLDER ' if jm['placeholder'] else ''}{jm['source']} sign {jm['sign']} offset_rad "
        f"{np.round(jm['offset_rad'], 4).tolist()} gripper_lin {np.round(jm['gripper_lin'], 5).tolist()}")
    for k, v in info["views"].items():
        log(f"view {k}: crop {v['crop_xywh']} -> {v['out_wh']} ({v['interp']}, {v['source']})")


# ---- replay -------------------------------------------------------------------------------------------------------
class LeRobotEpisode:
    """One episode of a LeRobot v3 recording (read only; the files are data, nothing in them is executed): joints at
    30 Hz and the two camera videos (one mp4 per camera with every episode, ``meta/episodes`` from / to timestamps)."""

    def __init__(self, root, episode, cams=H.REAL_CAMERAS):
        import cv2
        import pyarrow.parquet as pq
        self.root, self.episode = Path(root).expanduser(), int(episode)
        info = json.loads((self.root / "meta" / "info.json").read_text())
        self.fps = float(info["fps"])
        t = pq.read_table(self.root / "data" / "chunk-000" / "file-000.parquet").to_pydict()
        e = pq.read_table(self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pydict()
        ep = np.asarray(t["episode_index"])
        m = ep == self.episode
        if not m.any():
            raise ValueError(f"episode {episode} not in {root}")
        order = np.argsort(np.asarray(t["frame_index"])[m])
        self.state = np.asarray(t["observation.state"], np.float64)[m][order]
        self.action = np.asarray(t["action"], np.float64)[m][order]
        self.n = len(self.state)
        self.t = np.arange(self.n) / self.fps
        self.duration = (self.n - 1) / self.fps
        k = list(e["episode_index"]).index(self.episode)
        self.caps, self.next_idx, self.cur, self.first = {}, {}, {}, {}
        for c in cams:
            key = f"videos/observation.images.{c}"
            ci, fi = int(e[f"{key}/chunk_index"][k]), int(e[f"{key}/file_index"][k])
            path = self.root / "videos" / f"observation.images.{c}" / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"cannot open {path}")
            self.first[c] = int(round(float(e[f"{key}/from_timestamp"][k]) * self.fps))
            for _ in range(self.first[c]):                 # sequential skip (exact; h264 seeks can be off)
                if not cap.grab():
                    raise RuntimeError(f"{path}: video ended before episode {episode}")
            self.caps[c], self.next_idx[c], self.cur[c] = cap, 0, None

    def _interp(self, a, t):
        x = min(max(t * self.fps, 0.0), self.n - 1.0)
        i = min(int(x), self.n - 2) if self.n > 1 else 0
        w = x - i
        return a[i] * (1 - w) + a[min(i + 1, self.n - 1)] * w

    def q_at(self, t):
        return self._interp(self.state, t)

    def action_at(self, t):
        return self._interp(self.action, t)

    def frames_at(self, t):
        """{camera: (BGR frame of the newest recorded frame at or before t, nan)}."""
        i = min(int(math.floor(t * self.fps + 1e-6)), self.n - 1)
        out = {}
        for c, cap in self.caps.items():
            while self.next_idx[c] <= i:
                ok, f = cap.read()
                if not ok:
                    raise RuntimeError(f"video {c} ended at episode frame {self.next_idx[c]}")
                self.cur[c] = f
                self.next_idx[c] += 1
            out[c] = (self.cur[c], float("nan"))
        return out

    def close(self):
        for cap in self.caps.values():
            cap.release()


def episode_labels(calib, dataset, episode):
    for e in (calib or {}).get("episodes") or []:
        if str(e.get("dataset")) == dataset and int(e.get("episode", -1)) == int(episode):
            return e
    return None


def wrap45(d):
    return (np.asarray(d) + 45.0) % 90.0 - 45.0


def run_replay(args):
    calib, jm, layer, gcfg, policy, views, info = build_common(args)
    print_setup(info)
    limits = H.RealLimits.from_lerobot_calibration(lerobot_calib_path(args), gcfg.range_margin_deg)
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.warmup()
    summaries = []
    for spec in args.episodes.split(","):
        ds, e = spec.split(":")
        ep = LeRobotEpisode(Path(args.data_root).expanduser() / ds, int(e))
        guard = H.Guard(gcfg, layer, jm, limits)
        ctrl = H.Controller(policy, jm, guard, views, H.QdotEstimator(args.qdot_window), bgr=True,
                            max_frame_age_ms=float("inf"), thumb_res=args.thumb_res)
        ctrl.reset(ep.q_at(0.0))
        T = ep.duration if args.max_seconds <= 0 else min(ep.duration, args.max_seconds)
        n = int(math.floor(T / H.DT)) + 1
        rec_act, t0 = [], time.time()
        for s in range(n):
            t = s * H.DT
            ctrl.step(t, ep.q_at(t), lambda t=t: ep.frames_at(t))
            rec_act.append(ep.action_at(t))
        wall = time.time() - t0
        ep.close()
        A = ctrl.arrays()
        rec_act = np.asarray(rec_act)
        A["rec_action_real"] = rec_act
        A["rec_action_sim"] = jm.real_to_sim(rec_act)
        # the chunk against the recorded leader action at the step each target is meant for (target j of a query at
        # step k drives step k + j)
        qs = A["qry_step"]
        K = A["qry_chunk"].shape[1]
        err = np.full((len(qs), K, 6), np.nan)
        for i, k in enumerate(qs):
            for j in range(K):
                if k + j < n:
                    err[i, j] = A["qry_chunk"][i, j] - A["rec_action_sim"][k + j]
        lab = episode_labels(calib, ds, e)
        aux = dict(labels=lab is not None)
        if lab is not None:
            tq = A["qry_t"]
            pre = tq < float(lab.get("grasp_frame", 1e9)) / ep.fps if lab.get("grasp_frame") is not None else \
                np.ones(len(tq), bool)
            cxyz = np.asarray(lab["cube_start_xyz"], np.float64)
            pxy = np.asarray(lab["plate_xy"], np.float64)
            ce = np.linalg.norm(A["qry_cube_xyz"] - cxyz, axis=-1)
            pe = np.linalg.norm(A["qry_plate_xy"] - pxy, axis=-1)
            ye = np.abs(wrap45(A["qry_plate_yaw4_deg"] - float(lab["plate_yaw_deg"])))
            aux.update(cube_err_mm_before_grasp_p50=float(np.median(ce[pre]) * 1e3) if pre.any() else None,
                       n_before_grasp=int(pre.sum()), plate_xy_err_mm_p50=float(np.median(pe) * 1e3),
                       plate_yaw_err_deg_p50=float(np.median(ye)), n_queries=int(len(tq)))
            A["lab_cube_start_xyz"], A["lab_plate_xy"] = cxyz, pxy
            A["lab_plate_yaw_deg"] = np.float64(lab["plate_yaw_deg"])
        lat = H.latency_summary({k: A[k] for k in ("ms_qdot", "ms_frames", "ms_pre", "ms_model", "ms_guard",
                                                   "ms_ctrl")})
        evc = {k: int(A["ev_" + k].sum()) for k in ("step", "lead", "range", "floor", "real_range", "max_rel")}
        summ = dict(dataset=ds, episode=int(e), seconds=T, steps=n, queries=int(len(qs)), wall_s=wall,
                    chunk_vs_rec_action_mae_deg=np.degrees(np.nanmean(np.abs(err), axis=(0, 1))).round(3).tolist(),
                    chunk_vs_rec_action_mae_deg_by_j=np.degrees(np.nanmean(np.abs(err[..., :5]), axis=(0, 2))).round(
                        3).tolist(),
                    exec_vs_rec_action_mae_deg=np.degrees(np.nanmean(np.abs(err[:, :policy.n_exec]), axis=(0, 1))).round(
                        3).tolist(),
                    aux=aux, rule_active_steps=evc, guard_reasons=ctrl.events, latency_ms=lat,
                    pred_cube_xyz_p50=np.median(A["qry_cube_xyz"], 0).round(4).tolist(),
                    pred_plate_xy_p50=np.median(A["qry_plate_xy"], 0).round(4).tolist(),
                    pred_plate_yaw4_deg_p50=float(np.median(A["qry_plate_yaw4_deg"])))
        meta = dict(mode="replay", setup=info, summary=summ, note="student outputs on real images; the student was "
                    "trained on sim images of other cameras unless the setup says otherwise")
        H.save_log(out_dir / f"{ds}_ep{int(e):02d}.npz", A, meta)
        summaries.append(summ)
        log(f"{ds} ep {e}: {n} steps, {len(qs)} queries in {wall:.1f} s; chunk vs recorded action MAE deg "
            f"{summ['chunk_vs_rec_action_mae_deg']}; model ms p50 {lat['ms_model']['p50']:.1f} p95 "
            f"{lat['ms_model']['p95']:.1f}; guard reasons {list(ctrl.events)}")
    (out_dir / "replay_summary.json").write_text(json.dumps(dict(setup=info, episodes=summaries), indent=1,
                                                            default=str))
    log(f"wrote {out_dir}")
    return summaries


# ---- robot loop (dry / live) ------------------------------------------------------------------------------------------
def hold(io, jm, limits, q_real, grip_cmd_real):
    """Hold: arm at the measured q, gripper at its last command (keeps the grip), inside the calibrated range."""
    h = np.asarray(q_real, np.float64).copy()
    if grip_cmd_real is not None and np.isfinite(grip_cmd_real):
        h[H.ARM] = grip_cmd_real
    h, _ = limits.clip(h)
    io.write_goal(h)
    return h


def move_path(io, path_sim, jm, limits, guard, stopper, max_rel, track_tol=0.25, pause_max_s=2.0, settle_tol=0.05,
              settle_s=3.0, label="move"):
    """Execute a checked joint path at 50 Hz: each command clamped to the calibrated range and to max_rel (LeRobot units)
    around the measured q; the path does not advance while an arm joint is more than track_tol (rad) from the previous
    waypoint (stop after pause_max_s); afterwards wait until every arm joint is within settle_tol of the end.
    -> (stop reason or None, last command real)."""
    i, paused, last = 0, 0.0, None
    n = len(path_sim)
    t_next = time.perf_counter()
    t_settle = None
    while True:
        if stopper.poll():
            return f"user stop ({stopper.why})", last
        q_real = io.read_q()
        q_sim = jm.real_to_sim(q_real)
        bad = guard.check_measured(q_sim, q_real)
        if bad:
            return f"{label}: " + bad[0], last
        ref = path_sim[max(i - 1, 0)]
        err = float(np.abs(q_sim[:H.ARM] - ref[:H.ARM]).max())
        if err > track_tol and i > 0:
            paused += H.DT
            if paused > pause_max_s:
                return f"{label}: arm not following (error {err:.2f} rad for {paused:.1f} s)", last
        else:
            paused = 0.0
            i = min(i + 1, n)
        cmd = jm.sim_to_real(path_sim[i - 1] if i > 0 else path_sim[0])
        cmd, _ = limits.clip(cmd)
        cmd = np.minimum(np.maximum(cmd, q_real - max_rel), q_real + max_rel)
        io.write_goal(cmd)
        last = cmd
        if i >= n:
            e_end = float(np.abs(q_sim[:H.ARM] - path_sim[-1][:H.ARM]).max())
            t_settle = t_settle or time.perf_counter()
            if e_end < settle_tol:
                return None, last
            if time.perf_counter() - t_settle > settle_s:
                log(f"{label}: end error {e_end:.3f} rad after {settle_s} s (continuing)")
                return None, last
        t_next += H.DT
        if time.perf_counter() > t_next + H.DT:
            t_next = time.perf_counter()
        H.precise_wait(t_next)


def control_loop(io, ctrl, guard, jm, limits, stopper, live, time_limit, status_every=1.0):
    """The student at 50 Hz on the robot. -> (end reason, last command real, timing columns)."""
    W = {k: [] for k in ("ms_read", "ms_write", "ms_step", "late_ms")}
    q0 = io.read_q()
    ctrl.reset(q0)
    t_start = time.perf_counter()
    t_next = t_start
    overruns, last_cmd, prev_sim = 0, None, None
    reason = "time limit"
    t_status = t_start
    while True:
        now = time.perf_counter()
        W["late_ms"].append((now - t_next) * 1e3)
        t = now - t_start
        if t >= time_limit:
            break
        if stopper.poll():
            reason = f"user stop ({stopper.why})"
            break
        t0 = time.perf_counter()
        try:
            q_real = io.read_q()
        except Exception as e:                             # noqa: BLE001
            reason = f"joint read failed: {e}"
            break
        W["ms_read"].append((time.perf_counter() - t0) * 1e3)
        try:
            out = ctrl.step(t, q_real, lambda: io.frames(1000))
        except Exception as e:                             # noqa: BLE001
            reason = f"step failed: {type(e).__name__}: {e}"
            break
        stop = list(out.stop)
        if live:
            stop += guard.stall(jm.real_to_sim(q_real), prev_sim)
            if stop:
                reason = "guard: " + stop[0]
                break
            t1 = time.perf_counter()
            try:
                io.write_goal(out.u_real)
            except Exception as e:                         # noqa: BLE001
                reason = f"goal write failed: {e}"
                break
            W["ms_write"].append((time.perf_counter() - t1) * 1e3)
            last_cmd = out.u_real
        prev_sim = None if out.u_sim is None else guard.prev.copy()   # stall check: without the integral correction
        W["ms_step"].append((time.perf_counter() - t0) * 1e3)
        if time.perf_counter() - t_status >= status_every:
            t_status = time.perf_counter()
            r = ctrl.S
            k = len(r["t"])
            rate = (k - 1) / max(r["t"][-1] - r["t"][0], 1e-9) if k > 1 else 0.0
            log(f"t {t:5.1f} s  {rate:5.1f} Hz  step ms max {max(W['ms_step'][-50:]):.1f}  q_real "
                f"{np.round(q_real, 1).tolist()}  overruns {overruns}" + (f"  WOULD STOP: {stop[0]}" if stop else ""))
        t_next += H.DT
        if time.perf_counter() > t_next + H.DT:            # more than a period behind: no burst, resync
            overruns += 1
            t_next = time.perf_counter()
        H.precise_wait(t_next)
    W["overruns"] = overruns
    return reason, last_cmd, W


def preflight(io, jm, limits, guard, layer, reads=3):
    """Reads only: q (finite, consistent between reads, inside the calibrated range), camera frames (fresh, 480 x
    640). -> (q_real, report lines, problems)."""
    qs = [io.read_q() for _ in range(reads)]
    q = qs[-1]
    rep, bad = [], []
    if not all(np.all(np.isfinite(x)) for x in qs):
        bad.append("joint read not finite")
    spread = np.max(np.ptp(np.stack(qs), axis=0))
    rep.append(f"q real {np.round(q, 2).tolist()} (spread over {reads} reads {spread:.2f})")
    qs_ = jm.real_to_sim(q)
    rep.append(f"q sim deg {np.round(np.degrees(qs_), 1).tolist()}  grasp point {np.round(layer.tcp(qs_), 3).tolist()} "
               f"lowest gripper point {float(layer.lowest(qs_)) * 1e3:.1f} mm")
    out = limits.outside(q, guard.cfg.measured_range_tol)
    if out.any():
        bad.append("measured q outside the calibrated range: " + str([H.JOINTS[i] for i in np.flatnonzero(out)]))
    fr = io.frames(1000)
    for c in H.REAL_CAMERAS:
        if c not in fr:
            bad.append(f"camera {c} missing")
            continue
        img, age = fr[c]
        rep.append(f"camera {c}: {img.shape} {img.dtype} age {age:.0f} ms mean {img.mean():.0f}")
        if img.shape != (480, 640, 3):
            bad.append(f"camera {c} shape {img.shape}")
        if np.isfinite(age) and age > 200:
            bad.append(f"camera {c} frame {age:.0f} ms old")
    return q, rep, bad, fr


def run_robot(args, live):
    mode = "live" if live else "dry"
    fake = args.backend == "fake"
    if args.auto_confirm and not fake:
        raise SystemExit("--auto-confirm is only for --backend fake")
    calib, jm, layer, gcfg, policy, views, info = build_common(args)
    print_setup(info)
    if live and not args.i_am_supervising:
        raise SystemExit("live mode moves the robot: pass --i-am-supervising and stay at the robot")
    if live and jm.placeholder and not fake:
        raise SystemExit(f"live mode needs a real joint map in {args.calib} (calib_v1.json joint_map); the placeholder "
                         f"identity map is not verified")
    home_sim = np.radians(np.asarray([float(v) for v in args.home.split(",")], np.float64))
    if home_sim.shape != (6,):
        raise SystemExit("--home needs 6 values (sim degrees)")
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.warmup()
    if fake:
        q_rest = np.asarray([0.0, -88.0, 84.0, 65.0, 0.0, 20.0])       # folded rest pose of the recordings
        io = H.FakeRobotIO(q_rest, writes=live)
        limits = H.RealLimits.from_lerobot_calibration(lerobot_calib_path(args), gcfg.range_margin_deg)
    else:
        cam_paths = CAM_PATHS if not args.cam_paths else json.loads(args.cam_paths)
        busy = device_users([args.port, *cam_paths.values()])
        if busy:                                          # two writers on the motor bus = two programs move the arm
            raise SystemExit(f"{mode} refused: another process can use the robot bus or a camera. Stop it first:\n  "
                             + "\n  ".join(busy))
        io = LeRobotIO(args.port, args.robot_id, cam_paths, writes=live, calibration_dir=args.calibration_dir)
        limits = H.RealLimits.from_lerobot_calibration(io.calibration_dict(), gcfg.range_margin_deg,
                                                       source=str(io.robot.calibration_fpath))
    guard = H.Guard(gcfg, layer, jm, limits)
    log(f"{mode}: connecting ({'fake robot' if fake else args.port}) ...")
    io.connect()
    meta = dict(mode=mode, setup=info, limits=dict(lo=limits.lo.tolist(), hi=limits.hi.tolist(), source=limits.source),
                home_sim_deg=np.degrees(home_sim).tolist(), start=time.strftime("%Y-%m-%d %H:%M:%S"))
    disable_torque, last_cmd, torque_on = False, None, False
    st_ = dict(ctrl=None, W=None, saved=False, dir=out_dir)
    sess = live and args.episodes_n > 0                    # eval session (even with 1 episode); 0 = plain live run
    n_ep = args.episodes_n if sess else 1

    def save(reason):
        """Episode log (npz + json); also called from ``finally`` after an error (partial log)."""
        meta["end_reason"] = reason
        ctrl_, W_ = st_["ctrl"], st_["W"]
        A = ctrl_.arrays() if ctrl_ is not None else {}
        if W_ is not None:
            for k, v in W_.items():
                A["robot_" + k] = np.asarray(v)
            lat = H.latency_summary({**{k: A[k] for k in ("ms_qdot", "ms_frames", "ms_pre", "ms_model", "ms_guard",
                                                          "ms_ctrl") if k in A},
                                     **{k: W_[k] for k in ("ms_read", "ms_write", "ms_step", "late_ms")}})
            meta.update(latency_ms=lat, overruns=W_["overruns"])
        if ctrl_ is not None:
            meta.update(guard_reasons=ctrl_.events, rule_active_steps={
                k: int(A["ev_" + k].sum()) for k in ("step", "lead", "range", "floor", "real_range", "max_rel")
                if "ev_" + k in A})
        H.save_log(st_["dir"] / f"episode_{mode}.npz", A, meta)
        st_["saved"] = True
        return meta.get("latency_ms", {})

    def finish(reason):
        """Save the episode log, print the timing lines."""
        lat = save(reason)
        for k in ("ms_read", "ms_model", "ms_ctrl", "ms_write", "ms_step"):
            if k in lat:
                v = lat[k]
                log(f"latency {k}: p50 {v['p50']:.1f} p95 {v['p95']:.1f} max {v['max']:.1f} ms (n {v['n']})")
        if st_["W"] is not None:
            log(f"overruns (> one period behind): {st_['W']['overruns']}; log {st_['dir'] / f'episode_{mode}.npz'}")

    def new_ctrl():
        return H.Controller(policy, jm, guard, views, H.QdotEstimator(args.qdot_window), bgr=io.bgr,
                            max_frame_age_ms=args.max_frame_age_ms, thumb_res=args.thumb_res)

    reason = "not started"
    try:
        q_start, rep, bad, fr = preflight(io, jm, limits, guard, layer)
        for r in rep:
            log("preflight: " + r)
        meta.update(preflight=rep, preflight_problems=bad, q_start_real=q_start.tolist())
        for c, (img, _) in fr.items():
            meta.setdefault("first_frame_mean", {})[c] = float(img.mean())
        np.savez_compressed(out_dir / "first_frames.npz", **{c: img for c, (img, _) in fr.items()})
        if bad:
            raise SystemExit("preflight failed: " + "; ".join(bad))
        ctrl = new_ctrl()
        st_["ctrl"] = ctrl
        if not live:
            log("dry: NO motor writes; torque untouched. q / space / Esc / Ctrl+C stops.")
            with Stopper() as st:
                reason, _, W = control_loop(io, ctrl, guard, jm, limits, st, False, args.time_limit)
            st_["W"] = W
            log(f"dry ended: {reason}")
        else:
            # 1. LeRobot motor configuration (only if needed / asked)
            if not fake:
                diffs = io.config_diffs()
                meta["config_diffs"] = diffs
                do = args.configure == "always" or (args.configure == "auto" and diffs)
                if diffs:
                    log("registers that LeRobot's configure would change: " + "; ".join(diffs))
                if do:
                    if not confirm("CONFIGURE", "LeRobot's configure turns the torque OFF for a moment and writes motor "
                                   "registers (some in EEPROM). The arm must rest on its stops (folded rest pose)."):
                        raise SystemExit("aborted before configure")
                    io.configure()
                    torque_on = True
                    log("configured (torque on, goal = present position)")
            # 2. HOME (checked first), 3. the student; eval (--episodes-n >= 1): back to HOME between the episodes
            results, session = [], out_dir / "session.json"

            def record(ep, reason, outcome):
                meta["outcome"] = outcome
                finish(reason)
                results.append(dict(episode=ep + 1, end_reason=reason, dir=str(st_["dir"]),
                                    start=meta.get("episode_start"), **outcome))
                session.write_text(json.dumps(dict(start=meta["start"], args=vars(args), episodes=results), indent=1,
                                              default=str))
            for ep in range(n_ep):
                if ep > 0:
                    st_.update(ctrl=new_ctrl(), W=None, saved=False)
                    ctrl = st_["ctrl"]
                    for k in ("outcome", "latency_ms", "overruns", "guard_reasons", "rule_active_steps",
                              "home_path_steps", "home_path_s", "home_problems"):
                        meta.pop(k, None)
                if sess:
                    st_["dir"] = out_dir / f"ep_{ep + 1:02d}"
                    st_["dir"].mkdir(parents=True, exist_ok=True)
                    meta.update(episode=ep + 1, episode_start=time.strftime("%Y-%m-%d %H:%M:%S"))
                    log(f"==== episode {ep + 1} / {n_ep}")
                q_now = jm.real_to_sim(io.read_q())
                path, dur = H.plan_joint_path(q_now, home_sim, math.radians(args.home_speed_deg_s))
                probs = H.check_path(path, layer, jm, limits, clearance=args.home_clearance, start=q_now)
                for k in range(0, len(path), 5):
                    probs += [f"waypoint {k}: {s}" for s in guard._box(layer.tcp(path[k]))][:1]
                home_real = jm.sim_to_real(home_sim)
                log(f"HOME sim deg {np.round(np.degrees(home_sim), 1).tolist()} = real {np.round(home_real, 1).tolist()}; "
                    f"grasp point {np.round(layer.tcp(home_sim), 3).tolist()} lowest {float(layer.lowest(home_sim)) * 1e3:.0f}"
                    f" mm; {len(path)} steps = {dur:.1f} s at <= {args.home_speed_deg_s} deg/s")
                meta.update(home_path_steps=len(path), home_path_s=dur, home_problems=probs)
                if probs:
                    if not torque_on:
                        raise SystemExit("HOME path refused: " + "; ".join(probs[:5]))
                    reason = "HOME path refused: " + "; ".join(probs[:5])
                    log(reason + " (holding; menu below)")
                    break
                if ep == 0 and not confirm("MOVE", f"Torque ON at the present position, then a {dur:.1f} s joint move to "
                                           f"HOME. Keep a hand near the arm; q / space / Esc / Ctrl+C holds.",
                                           args.auto_confirm):
                    if not torque_on:
                        raise SystemExit("aborted before HOME")
                    reason = "aborted before HOME"
                    break
                if not torque_on:
                    io.torque_on_at_present()
                    torque_on = True
                with Stopper() as st:
                    why, last_cmd = move_path(io, path, jm, limits, guard, st, np.asarray(gcfg.max_rel_real),
                                              label="home")
                if why:
                    q = io.read_q()
                    last_cmd = hold(io, jm, limits, q, None if last_cmd is None else last_cmd[H.ARM])
                    reason = f"HOME move stopped: {why}"
                    log(reason + " (holding; menu below)")
                    break
                log(f"at HOME: q real {np.round(io.read_q(), 1).tolist()}")
                if not confirm("RUN", f"Student loop for up to {args.time_limit:.0f} s at 50 Hz (query every 0.1 s). "
                               f"{'Reset the scene: p' if ep else 'P'}lace the cube and the plate now. q / space / Esc / "
                               f"Ctrl+C stops and holds.", args.auto_confirm):
                    reason = "aborted before the student loop (holding at HOME)"
                    log(reason + "; menu below")
                    break
                with Stopper() as st:
                    reason, lc, W = control_loop(io, ctrl, guard, jm, limits, st, True, args.time_limit)
                st_["W"] = W
                last_cmd = lc if lc is not None else last_cmd
                try:
                    q = io.read_q()
                    last_cmd = hold(io, jm, limits, q, None if last_cmd is None else last_cmd[H.ARM])
                    log(f"student loop ended: {reason}; HOLDING at q real {np.round(q, 1).tolist()}")
                except Exception as e:                         # noqa: BLE001
                    log(f"student loop ended: {reason}; HOLD FAILED: {e}")
                    reason += f"; hold failed: {e}"
                    if sess:
                        record(ep, reason, UNKNOWN)
                    break
                if not sess:
                    break
                record(ep, reason, ask_outcome(ep + 1, n_ep, args.auto_confirm))
                if ep + 1 < n_ep and not ask_next(ep + 2, n_ep, args.auto_confirm):
                    break
            if results:
                print_session(results, session)
        if not st_["saved"]:
            finish(reason)
        if live:
            disable_torque, last_cmd = end_menu(io, jm, limits, layer, guard, q_start, args, last_cmd)
    except BaseException as e:
        reason = f"{type(e).__name__}: {e}"
        raise
    finally:
        if not st_["saved"]:
            try:
                save(reason)
                log(f"partial log {st_['dir'] / f'episode_{mode}.npz'} ({reason})")
            except Exception as e:                         # noqa: BLE001
                log(f"saving the log failed: {e}")
        if live and torque_on and not disable_torque:
            try:
                q = io.read_q()
                hold(io, jm, limits, q, None if last_cmd is None else last_cmd[H.ARM])
            except Exception as e:                         # noqa: BLE001
                log(f"final hold failed: {e}")
        io.close(disable_torque=disable_torque)
        log(f"closed ({'torque OFF' if disable_torque else 'torque unchanged / ON, holding' if live else 'no writes'})")
    return meta


UNKNOWN = dict(grasp=None, lift=None, place=None, note="no result typed")


def ask_outcome(ep, n, auto=False):
    """Eval: the result of an episode, typed by the person at the robot (None = not known)."""
    if auto:
        return dict(grasp=False, lift=False, place=False, note="auto")
    while True:
        try:
            a = input(f"Episode {ep} / {n} result: letters for each stage reached, g = grasp, l = lift, p = placed on "
                      f"the plate (for example glp, or - for none); a note after a space: ").strip()
        except (EOFError, KeyboardInterrupt):
            return dict(UNKNOWN)
        code, _, note = a.partition(" ")
        if code and set(code.lower()) <= set("glp-"):
            c = code.lower()
            return dict(grasp="g" in c, lift="l" in c, place="p" in c, note=note.strip(), typed=a)


def ask_next(ep, n, auto=False):
    """Eval: True = return to HOME for episode ep, False = stop (menu)."""
    if auto:
        return True
    while True:
        try:
            a = input(f"[r] return to HOME for episode {ep} / {n} (the gripper opens to its HOME value)  "
                      f"[s] stop the session (menu)  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if a in ("r", "s"):
            return a == "r"


def print_session(results, path):
    n = len(results)
    def c(k):
        return sum(r[k] is True for r in results)

    def f(r, k, ch):
        return "?" if r[k] is None else ch if r[k] else "-"
    log(f"session: {n} episodes; grasp {c('grasp')}, lift {c('lift')}, placed {c('place')} "
        f"(not known: {sum(r['grasp'] is None for r in results)}); {path}")
    for r in results:
        log(f"  episode {r['episode']:2d}: {f(r, 'grasp', 'g')}{f(r, 'lift', 'l')}{f(r, 'place', 'p')}  "
            f"{r['end_reason']}  {r.get('note', '')}")


def end_menu(io, jm, limits, layer, guard, q_start_real, args, last_cmd):
    """-> (disable torque at close?, the last command: the final hold keeps its gripper). Default: keep holding."""
    if args.auto_confirm:
        return False, last_cmd
    while True:
        try:
            c = input("[h] hold and exit (torque stays ON)  [r] slow return to the start pose  [t] torque OFF "
                      "(support the arm first)  > ").strip().lower() or "h"
        except (EOFError, KeyboardInterrupt):
            return False, last_cmd
        if c == "h":
            return False, last_cmd
        if c == "t":
            try:
                ok = input("Type TORQUE OFF: ").strip() == "TORQUE OFF"
            except (EOFError, KeyboardInterrupt):
                ok = False
            if ok:
                return True, last_cmd
            continue
        if c == "r":
            q_now = jm.real_to_sim(io.read_q())
            goal = jm.real_to_sim(q_start_real)
            path, dur = H.plan_joint_path(q_now, goal, math.radians(args.home_speed_deg_s))
            probs = H.check_path(path, layer, jm, limits, clearance=args.home_clearance, start=q_now, goal_measured=True)
            if probs:
                log("return path refused: " + "; ".join(probs[:3]))
                continue
            if not confirm("MOVE", f"{dur:.1f} s joint move back to the start pose."):
                continue
            with Stopper() as st:
                why, lc = move_path(io, path, jm, limits, guard, st, np.asarray(guard.cfg.max_rel_real), label="return")
            q = io.read_q()
            last_cmd = hold(io, jm, limits, q, None if lc is None else lc[H.ARM])
            log(f"return: {why or 'done'}; q real {np.round(q, 1).tolist()}")


# ---- CLI --------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter,
                                 allow_abbrev=False)          # robot_test.sh checks the full flag names
    ap.add_argument("mode", choices=("replay", "dry", "live"))
    ap.add_argument("--student", required=True, help="ViralNet checkpoint (vision.viral.save_student)")
    ap.add_argument("--calib", default=str(DEFAULT_CALIB_V1), help="calib_v1.json (joint_map, student_view, episodes)")
    ap.add_argument("--kin-points", default="", help="kin_points_v1.npz (default: hw_lib.default_kin_points)")
    ap.add_argument("--device", default="", help="cuda / cpu (default cuda if available)")
    ap.add_argument("--cam-map", default="", help="student camera=real camera, e.g. top=external,wrist=wrist")
    ap.add_argument("--crop", default="", help="override crops: external=x,y,w,h;wrist=x,y,w,h")
    ap.add_argument("--q-offset-deg", default="", help="extra joint-map offset, 5 sim deg (pan, lift, elbow, wrist_flex, "
                    "roll), added to calib offset_rad; e.g. 0,-2.086,0,0,0 = the table-tilt correction (README)")
    ap.add_argument("--allow-calib-change", action="store_true",
                    help="live: accept a --calib whose sha256 differs from the student's pinned calibration")
    ap.add_argument("--allow-crop-change", action="store_true",
                    help="accept a --crop that differs from the crops pinned in the student checkpoint")
    ap.add_argument("--interp", default="area", choices=("area", "linear", "nearest", "cubic"))
    ap.add_argument("--jaw-lead", type=int, default=None, help="default: the checkpoint's cfg.jaw_lead")
    ap.add_argument("--amp", type=int, default=None, help="bf16 autocast 1 / 0 (default: cfg.amp on cuda)")
    ap.add_argument("--qdot-window", type=int, default=3)
    ap.add_argument("--table-z", type=float, default=0.0, help="real table top z in the arm base frame (m)")
    ap.add_argument("--guard", default="", help="GuardConfig overrides as JSON")
    ap.add_argument("--thumb-res", type=int, default=112, help="logged student input images (px; 0 = full)")
    ap.add_argument("--out", default="", help="output folder (default ~/so101_r2s_claude/hw/<mode>_<time>)")
    ap.add_argument("--lerobot-calib", default="", help="LeRobot calibration json for the joint limits (replay/fake)")
    # replay
    ap.add_argument("--data-root", default="~/so101_r2s_claude/raw")
    ap.add_argument("--episodes", default="green_cube_red_plate_1:0", help="dataset:episode,...")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="replay at most this long per episode (0 = all)")
    # robot
    ap.add_argument("--backend", default="lerobot", choices=("lerobot", "fake"))
    ap.add_argument("--port", default=FOLLOWER_PORT)
    ap.add_argument("--robot-id", default=ROBOT_ID)
    ap.add_argument("--calibration-dir", default="", help="LeRobot calibration dir (default the HF cache)")
    ap.add_argument("--cam-paths", default="", help='JSON {"wrist": path, "external": path} (default: the notes)')
    ap.add_argument("--max-frame-age-ms", type=float, default=150.0)
    ap.add_argument("--time-limit", type=float, default=15.0, help="student loop seconds")
    ap.add_argument("--episodes-n", type=int, default=0, help="live: eval session of n episodes (result prompt, return "
                    "to HOME and scene reset between them; logs <out>/ep_NN, summary <out>/session.json); 0 = one run")
    ap.add_argument("--home", default=",".join(str(v) for v in H.HOME_SIM_DEG), help="HOME pose, 6 SIM degrees")
    ap.add_argument("--home-speed-deg-s", type=float, default=15.0, help="peak joint speed of HOME / return moves")
    ap.add_argument("--home-clearance", type=float, default=0.02, help="m above the table for HOME / return paths")
    ap.add_argument("--configure", default="auto", choices=("auto", "always", "never"))
    ap.add_argument("--i-am-supervising", action="store_true")
    ap.add_argument("--auto-confirm", action="store_true", help="fake backend only: skip the typed confirmations")
    args = ap.parse_args(argv)
    if not args.out:
        args.out = str(Path("~/so101_r2s_claude/hw").expanduser() / f"{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}")
    if args.mode == "replay":
        return run_replay(args)
    return run_robot(args, live=args.mode == "live")


if __name__ == "__main__":
    main()
