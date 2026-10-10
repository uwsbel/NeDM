"""Closed-loop evaluation of SO-101 push-T tracking controllers in Chrono (the collector's scene, settle, PD and timing).

One Chrono step server per episode (nedm.so101_push.chrono_server). The controller acts every 20 ms (one control step =
two 10 ms records). Its observation is built from the MEASURED history on the 20 ms grid: joint angles and the T pose
from the records, rates as step averages over one model step (frame 0 keeps the recorded rates), exactly as the NRD
training data (evaluate_v6.step_average_rates). The same observation, action map (joint and slew limits) and tracking
terms as the learned environment are used (so101_push_common).

Controllers (all through the same action map):
  recorded       the recorded q_cmd of the reference (open loop)
  ref_feedback   recorded q_cmd(t) + gain (q_ref(t) - q_measured(t))
  policy         actor mean on the normalised observation (BC-only or PPO)

Protocols:
  clean          frame 0 to the task end (2 s: 100 steps, 4 s: 200 steps), no reset after the start
  recovery       a command disturbance prefix (recorded q_cmd + offset ramped in over 5 steps and held for 10) from
                 frame 0 to the hand-off frame, identical for every controller; the controller acts from the hand-off
                 to the task end. Paired controllers run on the same node (Chrono is deterministic per node).

Validity in Chrono: arm-table contact, contact of a link other than the gripper / jaw with the T, T height / tilt
outside the collector QA bounds. Success: final T position error < 5 mm AND yaw error < 3 deg and no invalid event.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time

import numpy as np
import torch

from nedm.so101_push_rl import so101_push_common as C

CFG_PATH = "configs/so101_push/collector_v2.json"
FINGER_LINKS = (5, 6)                     # gripper, jaw in LINK_ORDER (base, shoulder, upper_arm, lower_arm, wrist, gripper, jaw)


def step_average_np(arm, t, stride, dt):
    """Records [N, R, 10] / [N, R, 13] -> the same with step-average rates (records >= stride), as
    evaluate_v6.step_average_rates for the arm (angles) and the T (position, quaternion)."""
    arm, t = arm.copy(), t.copy()
    s = stride
    arm[:, s:, 5:10] = (arm[:, s:, 0:5] - arm[:, :-s, 0:5]) / dt
    t_out = t.copy()
    t_out[:, s:, 7:10] = (t[:, s:, 0:3] - t[:, :-s, 0:3]) / dt
    q1, q0 = t[:, s:, 3:7], t[:, :-s, 3:7]
    w0, v0 = q0[..., :1], -q0[..., 1:]
    w1, v1 = q1[..., :1], q1[..., 1:]
    w = w1 * w0 - (v1 * v0).sum(-1, keepdims=True)
    v = w1 * v0 + w0 * v1 + np.cross(v1, v0)
    sign = np.where(w < 0, -1.0, 1.0)
    w, v = w * sign, v * sign
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2 * np.arctan2(n, w)
    rotvec = np.where(n > 1e-12, v / np.maximum(n, 1e-12) * ang, 2 * v)
    t_out[:, s:, 10:13] = rotvec / dt
    return arm, t_out


class ChronoServer:
    def __init__(self, code_root, q_start, first_cmd, cfg_path=CFG_PATH, render=None, diagnostics=False, t_pose=None):
        """The Chrono runtime comes from the bash script in $SO101_CHRONO_ENV (default: the AMD cluster script;
        scripts/so101_push/local/chrono_env_local.sh for this workstation). render: optional dict, see chrono_server.
        diagnostics: every record also carries tau, lock_err, finger_gap, link_gap, arm_t_force (chrono_server; the
        init message is unchanged when False). t_pose: T start pose [x, y, yaw] (default: the config start pose)."""
        env = dict(os.environ, SO101_CODE=str(code_root))
        env_sh = os.environ.get("SO101_CHRONO_ENV", f"{code_root}/scripts/so101_push/cluster/chrono_env.sh")
        self.p = subprocess.Popen(["bash", "-c", f"source {env_sh} >/dev/null 2>&1; "
                                   "exec $NRD_PYTHON -m nedm.so101_push.chrono_server"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env, bufsize=1)
        msg = dict(cmd="init", config=cfg_path, q_start=list(map(float, q_start)), first_cmd=list(map(float, first_cmd)))
        if render:
            msg["render"] = render
        if diagnostics:
            msg["diagnostics"] = True
        if t_pose is not None:
            msg["t_pose"] = list(map(float, t_pose))
        self.send(msg)
        self.first = self.recv()

    def send(self, msg):
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def recv(self):
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("chrono server ended")
        return json.loads(line)

    def step(self, q_cmd):
        self.send(dict(cmd="step", q_cmds=[list(map(float, q_cmd))]))
        return self.recv()["records"]

    def close(self):
        try:
            self.send(dict(cmd="close"))
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


class MeasuredHistory:
    """Records -> model-grid states (23 values, step-average rates) and contact flags per model step."""

    def __init__(self, first, stride=2, dt=0.02):
        self.arm = [np.asarray(first["arm"], float)[:10]]
        self.t = [np.asarray(first["t"], float)]
        self.stride, self.dt = stride, dt
        self.frames = [self._frame(0)]
        self.contacts = []                    # per model step: dict(arm_table, links [7], t_table)

    def _frame(self, r):
        a, t = np.asarray(self.arm), np.asarray(self.t)
        lo = max(0, r - self.stride)
        aa, tt = step_average_np(a[None, lo:r + 1], t[None, lo:r + 1], self.stride, self.dt) if r >= self.stride else (a[None, :r + 1], t[None, :r + 1])
        return np.r_[aa[0, -1], tt[0, -1]]

    def add(self, records):
        links = np.zeros(7, bool)
        at = tt = False
        for rec in records:
            self.arm.append(np.asarray(rec["arm"], float)[:10])
            self.t.append(np.asarray(rec["t"], float))
            links |= np.asarray(rec["contacts_link"], bool)
            at |= bool(rec["arm_table"])
            tt |= bool(rec["t_table"])
        self.frames.append(self._frame(len(self.arm) - 1))
        self.contacts.append(dict(arm_table=at, links=links, t_table=tt))

    def history(self, k=C.HISTORY):
        """Last k frames (oldest first) and validity (frames before 0 are zero / invalid)."""
        n = len(self.frames)
        out = np.zeros((k, C.N_STATE))
        ok = np.zeros(k, bool)
        for i in range(k):
            f = n - k + i
            if f >= 0:
                out[i], ok[i] = self.frames[f], True
        return out, ok


def disturbance_prefix(ref_cmds, t_start, amp, ramp=5, hold=10):
    """Recorded commands with an offset ramped in over `ramp` steps from t_start and held `hold` steps.
    Returns (prefix commands [t_h, 5], hand-off frame t_h)."""
    t_h = t_start + ramp + hold
    pre = np.array(ref_cmds[:t_h], float).copy()
    for t in range(t_start, t_h):
        s = min(1.0, (t - t_start + 1) / ramp)
        pre[t] += s * np.asarray(amp, float)
    return pre, t_h


def task_offset_to_joints(arm_model, q, dxy, dyaw, w_tilt=0.05, w_yaw=0.2):
    """Joint offset for a gripper shift (dx, dy in m, dz = 0) and a gripper yaw change (rad) at joint angles q:
    weighted least squares on the 6x5 TCP Jacobian (rows vx vy vz wx wy wz, weights 1 1 1 w_tilt w_tilt w_yaw),
    the trade-off of the collector's IK (position first, small tilt)."""
    _, _, J = arm_model.tcp_jacobian(np.asarray(q, float))
    w = np.array([1.0, 1.0, 1.0, w_tilt, w_tilt, w_yaw])
    target = np.array([dxy[0], dxy[1], 0.0, 0.0, 0.0, dyaw])
    dq, *_ = np.linalg.lstsq(J * w[:, None], target * w, rcond=None)
    return dq


def make_disturbances(bank, seed, amp_rad=0.03, joints=(0, 1, 2, 3), refs=None, space="joint", arm_model=None,
                      xy_m=0.005, yaw_deg=3.0):
    """One disturbance per reference: phase 'approach' (start before the first contact) or 'contact' (during it),
    alternating by reference. space 'joint': amplitude amp_rad on random signs of the listed joints scaled by U(0.5, 1).
    space 'task': a gripper shift of xy_m x U(0.5, 1) in a random direction and a yaw change of +-yaw_deg x U(0.5, 1),
    mapped to a joint offset at the reference joint angles of the start frame (task_offset_to_joints): the joints
    move together as in the collector's IK commands."""
    rng = np.random.default_rng(seed)
    con = bank["contacts"].any(-1)
    out = []
    for i in (range(bank.n) if refs is None else refs):
        c = np.flatnonzero(con[i])
        first, last = (int(c[0]), int(c[-1])) if len(c) else (150, 150)
        kind = "approach" if (i % 2 == 0 or len(c) == 0) else "contact"
        if kind == "approach":
            lo, hi = 5, max(6, first - 20)
        else:
            lo, hi = first, max(first + 1, last - 15)
        t_s = int(rng.integers(lo, hi))
        t_s = min(t_s, 200 - 40)
        a = np.zeros(5)
        if space == "joint":
            for j in joints:
                a[j] = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0) * amp_rad
            extra = {}
        else:
            ang = rng.uniform(0, 2 * np.pi)
            r = xy_m * rng.uniform(0.5, 1.0)
            dxy = (r * np.cos(ang), r * np.sin(ang))
            dyaw = np.radians(yaw_deg) * rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0)
            q = bank["states"][i, t_s, 0, :5]
            a = task_offset_to_joints(arm_model, q, dxy, dyaw)
            extra = dict(dxy_m=list(dxy), dyaw_rad=float(dyaw))
        out.append(dict(ref=int(i), kind=kind, t_start=t_s, amp=a.tolist(), space=space, **extra))
    return out


class Controller:
    name = "base"

    def __call__(self, ctx):
        raise NotImplementedError


class Recorded(Controller):
    name = "recorded"

    def __call__(self, ctx):
        return ctx["ref_cmds"][min(ctx["t"], len(ctx["ref_cmds"]) - 1)]


class RefFeedback(Controller):
    name = "ref_feedback"

    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, ctx):
        t = min(ctx["t"], len(ctx["ref_cmds"]) - 1)
        return ctx["ref_cmds"][t] + self.gain * (ctx["ref23"][t, :5] - ctx["hist"][-1, :5])


class Policy(Controller):
    def __init__(self, name, actor, obs_norm, amap):
        self.name, self.actor, self.norm, self.amap = name, actor, obs_norm, amap

    @torch.no_grad()
    def __call__(self, ctx):
        h = torch.as_tensor(ctx["hist"][None], dtype=torch.float64)
        ok = torch.as_tensor(ctx["ok"][None])
        prev = torch.as_tensor(ctx["prev_cmd"][None], dtype=torch.float64)
        T = ctx["task_end"]
        pf = [min(ctx["t"] + o, T) for o in C.PREVIEW_STEPS]
        refp = torch.as_tensor(ctx["ref23"][pf][None], dtype=torch.float64)
        phase = torch.tensor([ctx["t"] / T], dtype=torch.float64)
        raw = C.raw_observation(h, ok, prev, phase, refp)
        a = self.actor(self.norm(raw))
        cmd = self.amap.center + self.amap.scale * a[0].to(torch.float64)    # the action map clamps afterwards
        return cmd.numpy()


def run_episode(code_root, ref, task_end, controller, amap, prefix=None, collector_qa=None):
    """ref: dict(q_start, first_cmd, cmds [200, 5], ref23 [201, 23]). prefix: commands [t_h, 5] executed before the
    controller takes over (recovery) or None. Returns per-step arrays and metrics."""
    qa = collector_qa or {"t_z_tolerance_m": [0.002, 0.01], "max_t_tilt_deg": 15.0}
    srv = ChronoServer(code_root, ref["q_start"], ref["first_cmd"])
    try:
        mh = MeasuredHistory(srv.first)
        prev = np.asarray(ref["first_cmd"], float)
        cmds, sats, lat = [], [], []
        t_h = 0 if prefix is None else len(prefix)
        lo, hi, slew = (amap.lo.numpy(), amap.hi.numpy(), amap.slew.numpy())
        for t in range(task_end):
            if t < t_h:
                target = prefix[t]
            else:
                hist, ok = mh.history()
                ctx = dict(t=t, hist=hist, ok=ok, prev_cmd=prev, ref23=ref["ref23"], ref_cmds=ref["cmds"], task_end=task_end)
                t0 = time.perf_counter()
                target = np.asarray(controller(ctx), float)
                lat.append(time.perf_counter() - t0)
            bounded = np.clip(target, lo, hi)
            cmd = prev + np.clip(bounded - prev, -slew, slew)
            sats.append(bool((bounded != target).any() or (cmd != bounded).any()) if t >= t_h else False)
            mh.add(srv.step(cmd))
            cmds.append(cmd)
            prev = cmd
    finally:
        srv.close()
    S = np.asarray(mh.frames)                                     # [T+1, 23]
    U = np.asarray(cmds)
    con = mh.contacts
    arm_table = np.array([c["arm_table"] for c in con])
    other = np.array([c["links"][[i for i in range(7) if i not in FINGER_LINKS]].any() for c in con])
    h = 0.015                                                     # T half height
    tz = S[1:, 12]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (S[1:, 14] ** 2 + S[1:, 15] ** 2), -1, 1)))
    t_bad = (tz < h - qa["t_z_tolerance_m"][0]) | (tz > h + qa["t_z_tolerance_m"][1]) | (tilt > qa["max_t_tilt_deg"])
    invalid = arm_table | other | t_bad
    return dict(states=S, cmds=U, arm_table=arm_table, other_links=other, t_bad=t_bad, invalid=invalid,
                saturated=np.asarray(sats), latency_s=np.asarray(lat), handoff=t_h)


def metrics(res, ref23, prev0, slew, task_end):
    S = torch.as_tensor(res["states"][1:task_end + 1])
    R = torch.as_tensor(ref23[1:task_end + 1])
    t_h = res["handoff"]
    terms = C.tracking_terms(S, R)
    pe, ye = terms["pos_err_m"].numpy(), terms["yaw_err_rad"].numpy()
    qe = (S[:, :5] - R[:, :5]).numpy()
    U = res["cmds"][:task_end]
    du = np.diff(np.vstack([prev0[None], U]), axis=0)[t_h:]
    inv = bool(res["invalid"][:task_end].any())
    sl = slice(t_h, task_end)
    return dict(success=bool(not inv and pe[-1] < 0.005 and ye[-1] < math.radians(3.0)), invalid=inv,
                arm_table=bool(res["arm_table"][:task_end].any()), other_links=bool(res["other_links"][:task_end].any()),
                t_tip=bool(res["t_bad"][:task_end].any()),
                final_pos_mm=float(1e3 * pe[-1]), final_yaw_deg=float(np.degrees(ye[-1])),
                pos_rmse_mm=float(1e3 * np.sqrt((pe[sl] ** 2).mean())), yaw_rmse_deg=float(np.degrees(np.sqrt((ye[sl] ** 2).mean()))),
                joint_rmse_rad=float(np.sqrt((qe[sl] ** 2).mean())),
                action_change_rms_rad=float(np.sqrt((du ** 2).mean())) if len(du) else 0.0,
                action_change_p95_rad=float(np.quantile(np.abs(du), 0.95)) if len(du) else 0.0,
                action_change_rel_slew_max=float((np.abs(du) / slew).max()) if len(du) else 0.0,
                saturation_rate=float(res["saturated"][t_h:task_end].mean()) if task_end > t_h else 0.0,
                final_t_speed_mps=float(np.linalg.norm(res["states"][task_end, 17:19])),
                latency_median_ms=float(1e3 * np.median(res["latency_s"])) if len(res["latency_s"]) else 0.0,
                latency_p95_ms=float(1e3 * np.quantile(res["latency_s"], 0.95)) if len(res["latency_s"]) else 0.0)
