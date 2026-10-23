"""Real-robot side of the SO101 camera student (real2sim study): the pieces that the replay, dry and live modes of
``run_student_real.py`` share, and that ``tests/so101_policy/test_real2sim_hw.py`` tests.

Pure numpy / torch (+ cv2 for images). No mujoco, no newton / warp, no LeRobot import in this file: it must run on
the Jetson AGX Orin of the real robot (aarch64, torch 2.11, LeRobot 0.6.2, no mujoco).

Units and frames
- Real joints = LeRobot 0.6.2 ``so101_follower`` units (``use_degrees=True``): 5 arm joints in degrees
  (0 = middle of the calibrated range: deg = (raw - (range_min + range_max) / 2) * 360 / 4095), gripper 0-100
  (0 = range_min). Order ``JOINTS``.
- Sim joints = the pinned MuJoCo Menagerie so101 joint angles (rad), same order, gripper = the moving jaw angle.
- Joint map (``calib_v1.json`` "joint_map"): sim_rad[i] = sign[i] * radians(real_deg[i]) + offset_rad[i] for the 5 arm
  joints, sim jaw rad = a * real_gripper + b. ``JointMap.placeholder_identity`` (sign +1, offset 0, a = pi / 180,
  b = 0) is a placeholder only (signs and offsets NOT verified); live mode refuses it.
- World = arm base frame, z up, sim table top z = 0 (``table_z`` shifts the floor if the real table is elsewhere).

Control timing (as in the sim: ``joint_env``): one 20 ms step per target; a student query every ``n_exec`` (5) steps
= 0.1 s; the query sees the measured q, a qdot estimate and the newest camera frames; it returns a chunk of 10 absolute
targets, the first 5 are executed, each through the stateless action layer (``ActionLayer`` = ``joint_env.safety``:
lead band around the measured q of THAT step, the collector's joint range, the table floor from the kinematics).
The hardware guard (``Guard``) adds limits that the sim does not have (command change per step, LeRobot-unit range
and max relative target, workspace box, hard floor, tracking error); it never loosens the action layer.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM = 5
DT = 0.02                      # s, one control step (world.CONTROL_DT)
FPS_REC = 30                   # LeRobot recordings
STS_RES = 4096                 # STS3215 ticks per turn (LeRobot: max_res = resolution - 1 = 4095)
REAL_CAMERAS = ("external", "wrist")
DEFAULT_CAM_MAP = {"top": "external", "external": "external", "wrist": "wrist"}
# Default HOME pose in SIM degrees: the median start of the build9 development tasks (dev100.npz: 1.0, -5.2, -14.8,
# 92.1, 11.0, 36.3) rounded, pan set to 0, wrist roll 45 (teacher jt_h110 in CPU MuJoCo, 1,500 paired tasks: roll 45
# 98.1 % vs roll 0 95.1 %, McNemar p 0.002 on the 500 tasks not used to choose it; roll 30-60 all >= 97 %, roll -45
# 91 %; ~/so101_r2s_claude/teacher_check). Real LeRobot degrees with calib_v1: (0.0, -3.2, -6.5, 93.1, 120.8), gripper
# 31.2. Grasp point (0.27, 0.0, 0.14) m, lowest gripper point 121 mm (roll 0).
HOME_SIM_DEG = (0.0, -5.0, -15.0, 92.0, 45.0, 30.0)


# ---- calibration file ---------------------------------------------------------------------------------------------
def load_calib(path):
    """calib_v1.json (the study's calibration contract) -> dict, or None if the path is empty / missing."""
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists():
        return None
    return json.loads(p.read_text())


@dataclass
class JointMap:
    """real (LeRobot units) <-> sim (rad), module docstring."""
    sign: np.ndarray
    offset_rad: np.ndarray
    grip_a: float
    grip_b: float
    placeholder: bool = False
    source: str = ""

    def __post_init__(self):
        self.sign = np.asarray(self.sign, np.float64).reshape(ARM)
        self.offset_rad = np.asarray(self.offset_rad, np.float64).reshape(ARM)
        if not np.all(np.isin(self.sign, (-1.0, 1.0))):
            raise ValueError(f"joint_map sign must be +1 / -1, got {self.sign.tolist()}")
        if not np.all(np.isfinite(self.offset_rad)):
            raise ValueError("joint_map offset_rad must be finite")
        self.grip_a, self.grip_b = float(self.grip_a), float(self.grip_b)
        if not (math.isfinite(self.grip_a) and math.isfinite(self.grip_b)) or abs(self.grip_a) < 1e-9:
            raise ValueError(f"joint_map gripper_lin invalid: {[self.grip_a, self.grip_b]}")

    @staticmethod
    def placeholder_identity():
        return JointMap(np.ones(ARM), np.zeros(ARM), math.pi / 180.0, 0.0, placeholder=True,
                        source="placeholder identity (no calib_v1.json joint_map; signs / offsets NOT verified)")

    @staticmethod
    def from_dict(d, source="calib"):
        names = d.get("names")
        if names is not None and tuple(names)[:ARM] != JOINTS[:ARM]:
            raise ValueError(f"joint_map names {names} differ from {JOINTS}")
        lin = d["gripper_lin"]
        return JointMap(d["sign"][:ARM], d["offset_rad"][:ARM], lin[0], lin[1],
                        placeholder=bool(d.get("placeholder", False)), source=source)

    @staticmethod
    def from_calib(calib, source="calib"):
        if calib is None or "joint_map" not in calib:
            return JointMap.placeholder_identity()
        return JointMap.from_dict(calib["joint_map"], source)

    def real_to_sim(self, real):
        r = np.asarray(real, np.float64)
        out = np.empty_like(r)
        out[..., :ARM] = self.sign * np.radians(r[..., :ARM]) + self.offset_rad
        out[..., ARM] = self.grip_a * r[..., ARM] + self.grip_b
        return out

    def sim_to_real(self, sim):
        s = np.asarray(sim, np.float64)
        out = np.empty_like(s)
        out[..., :ARM] = np.degrees(self.sign * (s[..., :ARM] - self.offset_rad))
        out[..., ARM] = (s[..., ARM] - self.grip_b) / self.grip_a
        return out

    def describe(self):
        return dict(sign=self.sign.tolist(), offset_rad=self.offset_rad.tolist(), gripper_lin=[self.grip_a, self.grip_b],
                    placeholder=self.placeholder, source=self.source)


@dataclass
class RealLimits:
    """Allowed real targets (LeRobot units) from the LeRobot calibration file: the calibrated range of every motor
    minus ``margin`` (arm deg / gripper units)."""
    lo: np.ndarray
    hi: np.ndarray
    source: str = ""

    @staticmethod
    def from_lerobot_calibration(cal, margin_deg=3.0, margin_grip=0.0, source=""):
        if isinstance(cal, (str, Path)):
            source = source or str(cal)
            cal = json.loads(Path(cal).expanduser().read_text())
        lo, hi = np.zeros(6), np.zeros(6)
        for i, m in enumerate(JOINTS):
            c = cal[m]
            mn, mx = float(c["range_min"]), float(c["range_max"])
            if int(c.get("drive_mode", 0)) != 0:
                raise ValueError(f"{m}: drive_mode {c['drive_mode']} not handled (expected 0)")
            if i < ARM:
                mid = (mn + mx) / 2
                lo[i] = (mn - mid) * 360.0 / (STS_RES - 1) + margin_deg
                hi[i] = (mx - mid) * 360.0 / (STS_RES - 1) - margin_deg
            else:
                lo[i], hi[i] = 0.0 + margin_grip, 100.0 - margin_grip
        return RealLimits(lo, hi, source)

    def clip(self, real):
        r = np.asarray(real, np.float64)
        c = np.minimum(np.maximum(r, self.lo), self.hi)
        return c, np.abs(c - r) > 1e-9

    def outside(self, real, tol=0.0):
        r = np.asarray(real, np.float64)
        return (r < self.lo - tol) | (r > self.hi + tol)


# ---- images ---------------------------------------------------------------------------------------------------------
# ONE crop + resize for sim and real: src/nedm/render/student_view.py (numpy + cv2 only), also used by the renderer
# (nedm.render.so101.RealScene). Loaded by file path, so no nedm.render package import (no torch / newton / warp).
def _load_student_view():
    import importlib.util
    import os
    import sys
    mod = sys.modules.get("nedm.render.student_view")
    if mod is not None:
        return mod
    roots = [os.environ.get("NEDM_SRC"), str(Path(__file__).resolve().parents[4] / "src")]
    for root in roots:
        if not root:
            continue
        path = Path(root) / "nedm/render/student_view.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location("_nedm_student_view", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod                  # dataclasses need the module registered
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("src/nedm/render/student_view.py not found (NEDM_SRC or the repo / bundle layout): the robot "
                      "bundle must contain it (make_orin_bundle.sh)")


_SV = _load_student_view()
ViewSpec = _SV.ViewSpec
centre_square = _SV.centre_square
view_specs = _SV.view_specs                 # (calib, real_cams, out_res, width=640, height=480, interp="area")
preprocess = _SV.crop_resize                # (img, spec, bgr): crop box, then cv2 resize -> RGB uint8 [out_h, out_w, 3]
_INTERP = _SV._INTERP


def thumb(img, res):
    import cv2
    return cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA) if res and img.shape[0] != res else img


# ---- qdot ---------------------------------------------------------------------------------------------------------
class QdotEstimator:
    """Joint velocity (rad/s) from the measured q of the last ``window`` control steps: least-squares slope over the
    window (window 3 = 40 ms span, about 20 ms lag). Restarts after a gap > ``max_gap`` s. The sim student saw the exact
    joint velocity (training noise 0.3 x its std); the STS3215 position resolution is 1.5 mrad."""

    def __init__(self, window=3, max_gap=0.1):
        self.window, self.max_gap = int(window), float(max_gap)
        self.reset()

    def reset(self):
        self.t, self.q = [], []

    def update(self, t, q):
        q = np.asarray(q, np.float64)
        if self.t and (t - self.t[-1] > self.max_gap or t <= self.t[-1]):
            self.reset()
        self.t.append(float(t))
        self.q.append(q)
        if len(self.t) > self.window:
            self.t.pop(0)
            self.q.pop(0)
        if len(self.t) < 2:
            return np.zeros_like(q)
        tt = np.asarray(self.t)
        qq = np.stack(self.q)
        tc = tt - tt.mean()
        return (tc[:, None] * (qq - qq.mean(0))).sum(0) / (tc ** 2).sum()


# ---- the action layer (sim) and the hardware guard ---------------------------------------------------------------
def default_kin_points():
    """The SO101 gripper / moving-jaw collision points (.npz, ``kin.save_points``): first that exists of
    $SO101_HW_ASSETS/kin_points_v1.npz, <bundle>/assets/kin_points_v1.npz, ~/so101_policy/ckpt/kin_points_v1.npz."""
    import os
    here = Path(__file__).resolve()
    cands = []
    if os.environ.get("SO101_HW_ASSETS"):
        cands.append(Path(os.environ["SO101_HW_ASSETS"]) / "kin_points_v1.npz")
    if len(here.parents) > 4:
        cands.append(here.parents[4] / "assets" / "kin_points_v1.npz")
    cands.append(Path("~/so101_policy/ckpt/kin_points_v1.npz").expanduser())
    for c in cands:
        if c.exists():
            return c
    return None


class ActionLayer:
    """``joint_env.safety`` for one robot (the action layer of teacher and student): arm targets within ``joint_env.LEAD``
    of the measured q, all targets within the collector's range (kin.TLO .. THI), the table floor (``floor_guard``:
    lowest gripper / moving-jaw point >= ``joint_env.FLOOR`` + table_z unless the measured pose is lower). The kinematics
    use the saved collision points (no mujoco)."""

    def __init__(self, kin_points, table_z=0.0, lead=True, floor=True):
        import torch

        from nedm.so101_policy import joint_env as JE
        from nedm.so101_policy.kin import SO101Kin
        if kin_points is None or not Path(kin_points).expanduser().exists():
            raise FileNotFoundError(f"kin points file {kin_points} missing (needed for the table floor without mujoco)")
        self.torch, self.JE = torch, JE
        self.kin = SO101Kin("cpu", points=str(Path(kin_points).expanduser()))
        self.tlo, self.thi = self.kin.tlo.clone(), self.kin.thi.clone()
        self.lead = torch.tensor(JE.LEAD, dtype=torch.float64) if lead else None
        self.table_z = float(table_z)
        self.floor = JE.FLOOR + self.table_z if floor else None
        self.lowest(np.zeros(6))            # builds the reduced point set now (0.4 s on luffy), not in the first step

    def __call__(self, u, q):
        """u, q: sim rad [6] -> (u' [6], dict lead / range / floor: which rule changed the target)."""
        t = self.torch
        u0 = t.as_tensor(np.asarray(u, np.float64))[None]
        q0 = t.as_tensor(np.asarray(q, np.float64))[None]
        out = self.JE.safety(u0, q0, self.tlo, self.thi, self.lead, self.kin, self.floor)[0].numpy()
        # attribution of the change (diagnostics only)
        u1 = u0.clone()
        if self.lead is not None:
            u1[:, :ARM] = t.minimum(t.maximum(u1[:, :ARM], q0[:, :ARM] - self.lead), q0[:, :ARM] + self.lead)
        lead_on = bool((u1 - u0).abs().max() > 1e-12)
        u2 = t.minimum(t.maximum(u1, self.tlo), self.thi)
        range_on = bool((u2 - u1).abs().max() > 1e-12)
        floor_on = bool(np.abs(out - u2[0].numpy()).max() > 1e-12)
        return out, dict(lead=lead_on, range=range_on, floor=floor_on)

    def lowest(self, q):
        """Lowest gripper / moving-jaw point (m, world z) of sim poses q [..., 6] (fast point set, as the floor rule)."""
        t = self.torch
        qq = t.as_tensor(np.asarray(q, np.float64)).reshape(-1, 6)
        return self.kin.lowest_point(qq, exact=False).numpy().reshape(np.shape(q)[:-1])

    def tcp(self, q):
        """Grasp point (m, world) of sim poses q [..., 6]."""
        t = self.torch
        qq = t.as_tensor(np.asarray(q, np.float64)).reshape(-1, 6)
        p, _ = self.kin.fk(qq)
        return p.numpy().reshape(*np.shape(q)[:-1], 3)


@dataclass
class GuardConfig:
    """Hardware limits on top of the action layer (module docstring). Sim teacher labels for scale (build9 d1 r0,
    49k decisions): per-step target change p99 0.030-0.037 rad (arm), max 0.04-0.07 rad inside a decision and up to
    0.18 rad between decisions; jaw up to 1.4 rad between decisions."""
    max_step_rad: tuple = (0.06, 0.06, 0.06, 0.06, 0.08, 0.5)     # command change per 20 ms step (sim rad)
    max_rel_real: tuple = (10.0, 10.0, 10.0, 10.0, 10.0, 60.0)    # |command - measured| (LeRobot units)
    range_margin_deg: float = 3.0                                 # inside the LeRobot calibrated range
    measured_range_tol: float = 5.0                               # measured q may be this far outside (deg) before a stop
    hard_floor_m: float = -0.005       # stop if the MEASURED lowest gripper point is below table_z + this
    cmd_floor_m: float = -0.002        # stop if the COMMANDED lowest point is below table_z + this (after the layer)
    tcp_x: tuple = (-0.10, 0.45)       # grasp point box (m, world)
    tcp_y: tuple = (-0.42, 0.42)
    tcp_z: tuple = (-0.01, 0.40)
    tcp_r_max: float = 0.45            # grasp point distance from the pan axis (m)
    stall_frac: float = 0.9            # live only: an arm joint that stays >= stall_frac x its lead band (joint_env.LEAD)
    stall_steps: int = 25              #   from the previous command for stall_steps steps (0.5 s) is blocked: stop
    i_time_s: float = 0.0              # integral correction of the real servos' static error (0 = off; arm joints):
    i_max_deg: float = 3.0             #   each step adds (command - measured q) * DT / i_time_s, the sum <= i_max_deg,
    i_v_min: float = 0.02              #   only while the joint does not move to its command faster than this (rad/s)
    i_e_max_deg: float = 1.0           #   and only while |command - measured q| < this (not for a blocked joint)

    @staticmethod
    def from_dict(d):
        d = dict(d or {})
        known = set(GuardConfig.__dataclass_fields__)
        bad = sorted(set(d) - known)
        if bad:
            raise ValueError(f"unknown guard keys {bad}")
        return GuardConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


class Guard:
    """Hardware guard (``GuardConfig``). ``check_measured`` -> stop reasons of the measured state; ``apply`` -> the
    command (sim and real units) after: step limit vs the previous command, the action layer, LeRobot-unit range and
    max relative target vs the measured q, then the final checks (finite, workspace box, commanded floor).

    ``u_layer`` (sim rad [6], None before the first ``apply`` or after a non-finite target): the target of the last
    ``apply`` right after the step limit and the action layer, BEFORE the integral correction and the LeRobot-unit
    clamps = the sim's executed target (``joint_env`` ``u_exe``). The command input of a ``cmd_in`` student reads it,
    never the sent command (``u_sim``: + the integral correction) or ``prev`` (+ the LeRobot clamps)."""

    def __init__(self, cfg: GuardConfig, layer: ActionLayer, jmap: JointMap, limits: RealLimits):
        self.cfg, self.layer, self.jmap, self.limits = cfg, layer, jmap, limits
        self.max_step = np.asarray(cfg.max_step_rad, np.float64)
        self.max_rel = np.asarray(cfg.max_rel_real, np.float64)
        self.prev = None
        self.err_run = 0
        self.integ = np.zeros(ARM)
        self.u_layer = None

    def reset(self, q_sim):
        self.prev = np.asarray(q_sim, np.float64).copy()
        self.err_run = 0
        self.integ = np.zeros(ARM)
        self.u_layer = None

    def _box(self, p):
        c = self.cfg
        r = math.hypot(p[0] - 0.0388, p[1])
        bad = []
        if not (c.tcp_x[0] <= p[0] <= c.tcp_x[1] and c.tcp_y[0] <= p[1] <= c.tcp_y[1] and c.tcp_z[0] <= p[2] <= c.tcp_z[1]):
            bad.append(f"workspace box: grasp point {np.round(p, 3).tolist()} outside")
        if r > c.tcp_r_max:
            bad.append(f"reach: grasp point {r:.3f} m from the pan axis > {c.tcp_r_max}")
        return bad

    def check_measured(self, q_sim, q_real):
        c, stop = self.cfg, []
        if not (np.all(np.isfinite(q_sim)) and np.all(np.isfinite(q_real))):
            return ["measured q: not finite"]
        out = self.limits.outside(q_real, c.measured_range_tol)
        if out.any():
            stop.append("measured range: outside the calibrated range: " + ", ".join(
                f"{JOINTS[i]} {q_real[i]:.1f}" for i in np.flatnonzero(out)))
        low = float(self.layer.lowest(q_sim))
        if low < self.layer.table_z + c.hard_floor_m:
            stop.append(f"measured floor: lowest gripper point {low * 1e3:.1f} mm, below the hard floor")
        stop += ["measured " + s for s in self._box(self.layer.tcp(q_sim))]
        return stop

    def apply(self, u_sim, q_sim, q_real, qd_sim=None):
        """-> (command sim [6], command real [6], events dict, stop reasons). ``qd_sim``: the measured joint speed for
        the integral correction (None: the joints count as not moving)."""
        c = self.cfg
        ev = dict(step=False, lead=False, range=False, floor=False, real_range=False, max_rel=False)
        u = np.asarray(u_sim, np.float64)
        self.u_layer = None
        if not np.all(np.isfinite(u)):
            return None, None, ev, ["student target: not finite"]
        if self.prev is None:
            self.reset(q_sim)
        d = np.clip(u - self.prev, -self.max_step, self.max_step)
        ev["step"] = bool(np.any(np.abs(d - (u - self.prev)) > 1e-12))
        u = self.prev + d
        u, info = self.layer(u, q_sim)
        self.u_layer = np.array(u, np.float64)   # the sim's executed target (class docstring): before the integral
        ev.update(info)
        if c.i_time_s > 0:                 # the real servos stop short of a small command under load (sim: they follow)
            imax = math.radians(c.i_max_deg)
            e = u[:ARM] - np.asarray(q_sim)[:ARM]
            if qd_sim is not None:         # a joint that follows its command needs no correction (no wind-up in motion)
                e = np.where(np.asarray(qd_sim)[:ARM] * np.sign(e) > c.i_v_min, 0.0, e)
            e = np.where(np.abs(e) < math.radians(c.i_e_max_deg), e, 0.0)   # blocked joint / new target: no more push
            self.integ = np.clip(self.integ + e * (DT / c.i_time_s), -imax, imax)
            u = u.copy()
            u[:ARM] += self.integ
        ur = self.jmap.sim_to_real(u)
        ur, m1 = self.limits.clip(ur)
        lo, hi = np.asarray(q_real) - self.max_rel, np.asarray(q_real) + self.max_rel
        ur2 = np.minimum(np.maximum(ur, lo), hi)
        ev["real_range"], ev["max_rel"] = bool(m1.any()), bool(np.any(np.abs(ur2 - ur) > 1e-9))
        ur = ur2
        u = self.jmap.real_to_sim(ur)
        stop = []
        if not np.all(np.isfinite(ur)):
            stop.append("command: not finite")
        low = float(self.layer.lowest(u))
        if low < self.layer.table_z + c.cmd_floor_m and low < float(self.layer.lowest(q_sim)) - 1e-4:
            stop.append(f"command floor: lowest gripper point {low * 1e3:.1f} mm, below the floor")
        stop += ["command " + s for s in self._box(self.layer.tcp(u))]
        self.prev = u.copy()               # the step limit and the stall check: the command without the correction
        self.prev[:ARM] -= self.integ
        return u, ur, ev, stop

    def stall(self, q_sim, cmd_prev_sim):
        """Live only: the lead band keeps every arm command within LEAD of the measured q, so a blocked joint shows as a
        command error that stays near its lead band. -> stop reason after stall_steps such steps in a row."""
        if cmd_prev_sim is None or self.layer.lead is None:
            return []
        lead = self.layer.lead.numpy()
        e = np.abs(np.asarray(q_sim)[:ARM] - np.asarray(cmd_prev_sim)[:ARM])
        sat = e >= self.cfg.stall_frac * lead
        self.err_run = self.err_run + 1 if sat.any() else 0
        if self.err_run >= self.cfg.stall_steps:
            j = int(np.argmax(e / lead))
            return [f"stall: {JOINTS[j]} blocked, command error {e[j]:.3f} rad (lead band {lead[j]:.3f}) for {self.err_run} steps"]
        return []


# ---- joint-space moves (HOME, return) --------------------------------------------------------------------------
def plan_joint_path(q_from, q_to, max_speed, dt=DT, min_time=1.0):
    """Joint-space straight line q_from -> q_to (sim rad [6]) with a cosine time law whose peak joint speed stays
    within max_speed (rad/s, scalar or [6]). -> waypoints [N, 6] (excludes q_from, ends at q_to), duration s."""
    a, b = np.asarray(q_from, np.float64), np.asarray(q_to, np.float64)
    v = np.broadcast_to(np.asarray(max_speed, np.float64), a.shape)
    T = max(float(min_time), float(np.max(np.abs(b - a) / v)) * math.pi / 2)
    n = max(1, int(math.ceil(T / dt)))
    s = 0.5 - 0.5 * np.cos(math.pi * np.arange(1, n + 1) / n)
    return a + s[:, None] * (b - a), n * dt


def check_path(path, layer: ActionLayer, jmap: JointMap, limits: RealLimits, clearance=0.02, start=None,
               goal_measured=False):
    """Problems of a joint path (sim rad [N, 6]): a waypoint outside the collector's range or the LeRobot calibrated
    range, or with its lowest gripper point below min(table_z + clearance, the lowest point of either end) - 1 mm, or
    its grasp point outside the guard box (checked by the caller's Guard). A measured end pose (``start``; the last
    waypoint if ``goal_measured``) may lie outside the collector's range (the real gripper closed: LeRobot gripper
    < 2.9 = sim jaw < kin.TLO); a straight path stays between its ends, so waypoints between that end and the range
    are accepted. -> list of strings (empty = ok)."""
    p = np.asarray(path, np.float64)
    ends = p[[0, -1]] if start is None else np.stack((np.asarray(start, np.float64), p[-1]))
    bad = []
    tlo, thi = layer.tlo.numpy(), layer.thi.numpy()
    for m in ([] if start is None else [np.asarray(start, np.float64)]) + ([p[-1]] if goal_measured else []):
        tlo, thi = np.minimum(tlo, m), np.maximum(thi, m)
    if np.any(p < tlo - 1e-9) or np.any(p > thi + 1e-9):
        bad.append("path leaves the sim joint range (kin.TLO .. THI)")
    if limits.outside(jmap.sim_to_real(p)).any():
        bad.append("path leaves the LeRobot calibrated range (minus margin)")
    low = layer.lowest(p)
    ref = min(layer.table_z + clearance, float(layer.lowest(ends).min()))
    if float(low.min()) < ref - 1e-3:
        k = int(low.argmin())
        bad.append(f"path waypoint {k} lowest gripper point {low[k] * 1e3:.1f} mm < {ref * 1e3:.1f} mm")
    return bad


# ---- the student ----------------------------------------------------------------------------------------------------
def resolve_cam_map(student_cams, spec=None):
    """{student camera: real camera}: ``spec`` "top=external,wrist=wrist" or a dict; default DEFAULT_CAM_MAP."""
    m = dict(DEFAULT_CAM_MAP)
    if isinstance(spec, str) and spec:
        for kv in spec.split(","):
            k, v = kv.split("=")
            m[k.strip()] = v.strip()
    elif isinstance(spec, dict):
        m.update(spec)
    out = {}
    for c in student_cams:
        if c not in m:
            raise ValueError(f"student camera {c!r} has no real camera (use --cam-map {c}=external)")
        if m[c] not in REAL_CAMERAS:
            raise ValueError(f"real camera {m[c]!r} unknown (expected one of {REAL_CAMERAS})")
        out[c] = m[c]
    if len(set(out.values())) != len(out):
        raise ValueError(f"two student cameras read the same real camera: {out}")
    return out


class StudentPolicy:
    """The build9 camera student (``vision.viral.ViralNet``) as a real-robot policy: images of its cameras (the student's
    RGB uint8 [res, res, 3], ``preprocess``), q and qdot (sim units) -> absolute chunk [K, 6] (sim rad), the executed
    part [n_exec, 6] (jaw ``jaw_lead`` steps ahead, as ``viral.Actor``), the auxiliary predictions (block = plate
    x, y, yaw mod 90 deg; cube x, y, z). Same normalisation and bf16 autocast as ``viral.Actor``.

    Input options (as ``viral.OnlineDAgger`` / ``viral.Actor``; ``reset`` at every episode start):
    - ``cfg.hist``: [q, qd] of the previous ``hist`` queries, newest first (the current values repeated at the first
      query).
    - ``cfg.cmd_in``: the previous decision's 5 executed targets - q now, raw rad [5, 6] flattened after the history
      (the net divides by ``cmd_scale``); 0 at the first query. ``query(..., u_prev=)`` = ``Guard.u_layer`` of the 5
      loop steps since the last query (``Controller`` passes them); a missing ``u_prev`` after the first query is
      refused.
    - ``cfg.img_hist``: the previous query's image features (cached, no second encoder pass; the current ones at the
      first query) appended to the head input.
    A student whose head input is not exactly [features, robot vector above, (previous features)] (an input option this
    class does not know) is refused. ``query`` returns the robot vector it fed (``robot``)."""

    def __init__(self, ckpt, device="cuda", cam_map=None, jaw_lead=None, amp=None):
        import torch

        from nedm.so101_policy.vision import viral as VI
        self.torch, self.VI = torch, VI
        self.dev = torch.device(device)
        self.st, ck = VI.load_student(ckpt, device)
        self.ckpt = str(Path(ckpt).expanduser())
        self.meta = {k: v for k, v in ck.items() if k not in ("state_dict", "stats")}
        self.cfg = self.st.cfg
        self.cams = tuple(self.cfg.cameras)
        self.cam_map = resolve_cam_map(self.cams, cam_map)
        self.res = int(self.cfg.res)
        self.n_exec, self.chunk = int(self.cfg.n_exec), int(self.cfg.chunk)
        self.jaw_lead = int(self.cfg.jaw_lead if jaw_lead is None else jaw_lead)
        self.amp = bool(self.cfg.amp if amp is None else amp) and self.dev.type == "cuda"
        self.H = int(self.cfg.hist)
        self.cmd_in = bool(getattr(self.cfg, "cmd_in", False))
        self.img_hist = bool(getattr(self.cfg, "img_hist", False))
        fd = int(self.st.fd)
        want = fd + 12 * (1 + self.H) + (6 * self.n_exec if self.cmd_in else 0) + (fd if self.img_hist else 0)
        got = int(self.st.head[0].in_features)
        if got != want:
            raise ValueError(f"student {self.ckpt}: head input {got} != {want} = features {fd} + [q, qd] x (1 + hist "
                             f"{self.H}) + cmd_in {self.cmd_in} + img_hist {self.img_hist}: an input option that the "
                             f"robot code does not implement")
        self.rhist = None
        self.prev_feat = None       # img_hist: the previous query's image features
        self.n_query = 0            # queries since ``reset``

    def reset(self):
        self.rhist = None
        self.prev_feat = None
        self.n_query = 0

    def warmup(self, n=3):
        """``n`` (>= 2) queries on black images (the history / image-history paths included), then ``reset``."""
        z = {self.cam_map[c]: np.zeros((self.res, self.res, 3), np.uint8) for c in self.cams}
        for i in range(max(2, int(n))):
            self.query(z, np.zeros(6), np.zeros(6), u_prev=np.zeros((self.n_exec, 6)) if i else None)
        self.reset()

    def query(self, images, q_sim, qd_sim, u_prev=None):
        """``u_prev`` (``cmd_in`` students; others ignore it): the previous decision's executed targets [n_exec, 6]
        (sim rad, ``Guard.u_layer`` of the loop steps since the last query); ignored at the first query (input 0)."""
        torch, VI = self.torch, self.VI
        t0 = time.perf_counter()
        with torch.no_grad():
            rgb = np.stack([images[self.cam_map[c]] for c in self.cams])[None]          # [1, C, H, W, 3]
            if rgb.shape[2:4] != (self.res, self.res):
                raise ValueError(f"student images must be {self.res} px, got {rgb.shape[2:4]}")
            img = VI.to_input(torch.from_numpy(rgb).to(self.dev))
            q = torch.as_tensor(np.asarray(q_sim, np.float32), device=self.dev)[None]
            qd = torch.as_tensor(np.nan_to_num(np.asarray(qd_sim, np.float32)), device=self.dev)[None]
            robot = torch.cat((q, qd), -1)
            if self.H:
                if self.rhist is None:
                    self.rhist = robot[:, None, :].expand(-1, self.H, -1).clone()
                now = robot
                robot = torch.cat((now, self.rhist.reshape(1, -1)), -1)
                self.rhist = torch.cat((now[:, None, :], self.rhist[:, :-1]), 1)
            if self.cmd_in:             # the previous decision's executed targets - q now (0 at the first query)
                if self.n_query == 0:
                    cmd = torch.zeros(1, 6 * self.n_exec, device=self.dev)
                else:
                    if u_prev is None:
                        raise ValueError("cmd_in student: pass u_prev (Guard.u_layer of the loop steps since the last "
                                         "query) after the first query of an episode")
                    up = np.asarray(u_prev, np.float32)
                    if up.shape != (self.n_exec, 6):
                        raise ValueError(f"u_prev shape {up.shape}, expected ({self.n_exec}, 6)")
                    cmd = (torch.as_tensor(up, device=self.dev) - q[:, None, :]).nan_to_num().reshape(1, -1)
                robot = torch.cat((robot, cmd), -1)
            with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.amp):
                if self.img_hist:       # the previous query's features (the first query: the current ones)
                    f = self.st.features(img)
                    p = self.st.head_from_features(f, robot, self.prev_feat)
                    self.prev_feat = f
                else:
                    p, f = self.st(img, robot)
                aux = self.st.aux(f)
            delta = VI.denormalise(self.st, p.float())
            aux = aux.float() * self.st.aux_std + self.st.aux_mean
            u = (q[:, None, :] + delta)[0]                                              # [K, 6]
            jj = torch.arange(self.n_exec, device=self.dev)
            kk = (jj + self.jaw_lead).clamp_max(u.shape[0] - 1)
            ex = torch.cat((u[jj, :ARM], u[kk, ARM:]), -1)
            out = dict(chunk=u.double().cpu().numpy(), exec=ex.double().cpu().numpy(),
                       aux=aux[0].double().cpu().numpy(), robot=robot[0].double().cpu().numpy())
        self.n_query += 1
        a = out["aux"]
        out["plate_xy"] = a[:2]
        out["plate_yaw4_deg"] = math.degrees(math.atan2(a[2], a[3]) / 4)
        out["cube_xyz"] = a[4:7]
        out["t_model_ms"] = (time.perf_counter() - t0) * 1e3
        return out


# ---- one control step (replay, dry, live) ---------------------------------------------------------------------------
@dataclass
class StepOut:
    u_sim: np.ndarray | None
    u_real: np.ndarray | None
    query: bool
    stop: list = field(default_factory=list)          # stop reasons (enforced in live)
    events: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)


class Controller:
    """The student's control step, identical for a replayed recording and the robot: measured q (real units) at time
    t -> qdot estimate, every n_exec steps a query (``get_frames()`` -> {real camera: (image uint8 [480, 640, 3],
    age ms)}; ``bgr``: images are BGR), the executed target of this step through ``Guard.apply`` (step limit, action
    layer, LeRobot-unit limits). Keeps a per-step and per-query log (``arrays``: per-query keys start with "qry_").
    Per step ``u_layer`` (``Guard.u_layer``: the action-layer target, before the integral correction and the LeRobot
    clamps); a ``cmd_in`` student gets the ``u_layer`` of the 5 loop steps since the last query (by loop step: a query
    every ``n_exec`` steps); a student with history / new inputs logs the robot vector it was fed (``qry_robot``)."""

    def __init__(self, policy: StudentPolicy, jmap: JointMap, guard: Guard, views: dict, qdot: QdotEstimator,
                 bgr: bool, max_frame_age_ms=150.0, thumb_res=112, qd_gain=1.0):
        self.p, self.jm, self.g, self.views, self.qdot = policy, jmap, guard, views, qdot
        # real2sim (2026-10-22): the student's qdot INPUT x qd_gain (1 = off). The student's step size follows the
        # measured speed; the real arm is ~5x slower than the sim arm, so its steps shrink until the arm holds still
        # above the plate (robot-like sim, today's student: 50 % at gain 1, 88 % at 2, 94 % at 3; nominal sim 98 / 90 /
        # 75 %). The guard and the log keep the measured qdot.
        self.qd_gain = float(qd_gain)
        self.bgr, self.max_age, self.thumb_res = bool(bgr), float(max_frame_age_ms), int(thumb_res)
        self.k = 0
        self.ex = None
        self.u_layer = []           # Guard.u_layer of the last n_exec loop steps (NaN: no command that step)
        self.S = {k: [] for k in ("t", "q_real", "q_sim", "qd_sim", "u_target", "u_layer", "u_sim", "u_real", "i_corr",
                                  "query", "ev_step", "ev_lead", "ev_range", "ev_floor", "ev_real_range", "ev_max_rel",
                                  "n_stop", "ms_qdot", "ms_frames", "ms_pre", "ms_model", "ms_guard", "ms_ctrl")}
        self.Q = {k: [] for k in ("t", "step", "chunk", "exec", "aux", "plate_xy", "plate_yaw4_deg", "cube_xyz",
                                  "frame_age_ms", "thumbs", "robot")}     # robot: empty (no key) for a plain student
        self.events = {}

    def reset(self, q_real=None):
        self.k = 0
        self.ex = None
        self.u_layer = []
        self.qdot.reset()
        self.p.reset()
        if q_real is not None:
            self.g.reset(self.jm.real_to_sim(q_real))

    def step(self, t, q_real, get_frames):
        T = {}
        t0 = time.perf_counter()
        q_real = np.asarray(q_real, np.float64)
        q_sim = self.jm.real_to_sim(q_real)
        qd = self.qdot.update(t, q_sim)
        T["ms_qdot"] = (time.perf_counter() - t0) * 1e3
        stop = self.g.check_measured(q_sim, q_real)
        query = self.k % self.p.n_exec == 0
        T["ms_frames"] = T["ms_pre"] = T["ms_model"] = float("nan")          # query steps only
        if query:
            t1 = time.perf_counter()
            fr = get_frames()
            T["ms_frames"] = (time.perf_counter() - t1) * 1e3
            t2 = time.perf_counter()
            imgs, ages = {}, []
            for c in self.p.cams:
                rc = self.p.cam_map[c]
                im, age = fr[rc]
                ages.append(float(age))
                if np.isfinite(age) and age > self.max_age:
                    stop.append(f"camera age: {rc} frame {age:.0f} ms old")
                imgs[rc] = preprocess(im, self.views[rc], self.bgr)
            T["ms_pre"] = (time.perf_counter() - t2) * 1e3
            qd_in = qd if self.qd_gain == 1.0 else qd * self.qd_gain
            if getattr(self.p, "cmd_in", False):          # the action-layer targets of the last n_exec loop steps
                n = self.p.n_exec
                r = self.p.query(imgs, q_sim, qd_in, u_prev=np.stack(self.u_layer[-n:]) if len(self.u_layer) >= n
                                 else None)
            else:
                r = self.p.query(imgs, q_sim, qd_in)
            T["ms_model"] = r["t_model_ms"]
            self.ex = r["exec"]
            Q = self.Q
            if "robot" in r and (getattr(self.p, "H", 0) or getattr(self.p, "cmd_in", False)
                                 or getattr(self.p, "img_hist", False)):
                Q["robot"].append(np.asarray(r["robot"]))
            Q["t"].append(t)
            Q["step"].append(self.k)
            for kk in ("chunk", "exec", "aux", "plate_xy", "plate_yaw4_deg", "cube_xyz"):
                Q[kk].append(np.asarray(r[kk]))
            Q["frame_age_ms"].append(np.asarray(ages))
            Q["thumbs"].append(np.stack([thumb(imgs[self.p.cam_map[c]], self.thumb_res) for c in self.p.cams]))
        t3 = time.perf_counter()
        target = self.ex[self.k % self.p.n_exec]
        u_sim, u_real, ev, stop2 = self.g.apply(target, q_sim, q_real, qd)
        nan6 = np.full(6, np.nan)
        ul = nan6 if self.g.u_layer is None else self.g.u_layer.copy()
        self.u_layer = (self.u_layer + [ul])[-self.p.n_exec:]
        stop += stop2
        T["ms_guard"] = (time.perf_counter() - t3) * 1e3
        T["ms_ctrl"] = (time.perf_counter() - t0) * 1e3
        S = self.S
        for kk, v in (("t", t), ("q_real", q_real), ("q_sim", q_sim), ("qd_sim", qd), ("u_target", target),
                      ("u_layer", ul), ("u_sim", nan6 if u_sim is None else u_sim),
                      ("u_real", nan6 if u_real is None else u_real),
                      ("i_corr", self.g.integ.copy()), ("query", query), ("n_stop", len(stop))):
            S[kk].append(v)
        for kk in ("step", "lead", "range", "floor", "real_range", "max_rel"):
            S["ev_" + kk].append(bool(ev.get(kk, False)))
        for kk, v in T.items():
            S[kk].append(v)
        for s in stop:
            key = s.split(":")[0]                              # one entry per kind of reason: first time, count
            e = self.events.setdefault(key, dict(first_t=float(t), first=s, count=0))
            e["count"] += 1
        self.k += 1
        return StepOut(u_sim, u_real, query, stop, ev, T)

    def arrays(self):
        out = {}
        for pre, D in (("", self.S), ("qry_", self.Q)):
            for k, v in D.items():
                if len(v):
                    out[pre + k] = np.asarray(np.stack(v) if isinstance(v[0], np.ndarray) else v)
        return out


# ---- logs and latency ----------------------------------------------------------------------------------------------
def latency_summary(cols):
    """{name: list of ms} -> {name: {n, p50, p95, p99, max}}."""
    out = {}
    for k, v in cols.items():
        a = np.asarray([x for x in v if x is not None and np.isfinite(x)], np.float64)
        if len(a):
            out[k] = dict(n=int(len(a)), p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)),
                          p99=float(np.percentile(a, 99)), max=float(a.max()))
    return out


def save_log(path, arrays, meta):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays, meta=np.asarray(json.dumps(meta, default=str)))
    p.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))


def precise_wait(deadline):
    """Sleep until time.perf_counter() >= deadline (sleep, then spin the last 2 ms)."""
    while True:
        r = deadline - time.perf_counter()
        if r <= 0:
            return
        if r > 0.003:
            time.sleep(r - 0.002)


# ---- a fake robot (tests of the dry / live loop without hardware) -------------------------------------------------
class FakeRobotIO:
    """Stand-in for the LeRobot robot in tests: 6 servos that move toward the goal with a first-order lag (``tau`` s)
    and a speed limit (deg/s), torque flag, a write counter, constant camera images. Same methods as
    ``run_student_real.LeRobotIO``."""

    def __init__(self, q0_real, writes=True, tau=0.04, vmax=200.0, image=None):
        self.q = np.asarray(q0_real, np.float64).copy()
        self.goal = self.q.copy()
        self.writes, self.tau, self.vmax = bool(writes), float(tau), float(vmax)
        self.torque = False
        self.n_writes = 0
        self.t_last = None
        self.img = np.full((480, 640, 3), 90, np.uint8) if image is None else image
        self.bgr = False
        self.connected = False

    def connect(self):
        self.connected = True

    def _advance(self):
        now = time.perf_counter()
        if self.t_last is not None and self.torque:
            dt = now - self.t_last
            d = (self.goal - self.q) * (1 - math.exp(-dt / self.tau))
            self.q += np.clip(d, -self.vmax * dt, self.vmax * dt)
        self.t_last = now

    def read_q(self):
        self._advance()
        return self.q.copy()

    def frames(self, max_age_ms=500):
        return {c: (self.img, 5.0) for c in REAL_CAMERAS}

    def write_goal(self, q_real):
        if not self.writes:
            raise RuntimeError("fake robot: writes are blocked (dry mode)")
        self._advance()
        self.goal = np.asarray(q_real, np.float64).copy()
        self.n_writes += 1

    def torque_on_at_present(self):
        if not self.writes:
            raise RuntimeError("fake robot: writes are blocked (dry mode)")
        self.goal = self.q.copy()
        self.torque = True

    def disable_torque(self):
        if not self.writes:
            raise RuntimeError("fake robot: writes are blocked (dry mode)")
        self.torque = False

    def configure(self, mode):
        return "fake: nothing to configure"

    def close(self, disable_torque=False):
        if disable_torque:
            self.disable_torque()
        self.connected = False
