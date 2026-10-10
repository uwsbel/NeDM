"""Planar pusher PPO for the SO-101 push-T (large T rotations; spec SPEC_lead_v1.md A1, A3, A5, A6): action map,
observation, support (validity) rules and goal coverage. Shared by the NRD environment (so101_planar_env.py), the
Chrono controller (so101_planar_chrono.py) and the support calibration (experiment path scripts/evaluation/so101_support_calibration.py).
The decoder that turns a decision into joint commands is so101_planar_decoder.PlanarDecoder.

State order (23 values, so101_push_common): arm q 0-4, averaged q_dot 5-9, T x y z 10-12, quaternion 13-16,
averaged velocity 17-19, averaged angular velocity 20-22.

Action map (PlanarActionMap; a [N, 3] raw latent, PPO keeps a):
  u = tanh(a[:, 0:2]); d_pol = d_max_m u, then |d_pol| <= d_max_m (radial clip, the direction is kept)
  frame "t" (default): d_pol is in the T frame at decision time (x along the bar, y toward the stem),
                       d_world = R(T yaw) d_pol, d_tframe = d_pol
  frame "world" (ablation): d_world = d_pol, d_tframe = R(-T yaw) d_pol
  dyaw = dyaw_max_rad tanh(a[:, 2]) (gripper yaw change over the decision); 0 when yaw_control is false

Observation (OBS_DIM = 139, float32; nan_to_num, clamp(-50, 50)):
  [0:129)   so101_direct_common.DirectObs unchanged (stats_v1.pt normalisation; delta_max of the direct env for the
            previous command change)
  [129:131) decoder target xy - TCP xy (FK of the NRD q), T frame / 0.01 m
  [131:133) decoder target velocity (nominal profile), T frame / 0.1 m/s
  [133:135) target gripper yaw - T yaw: sin, cos
  [135]     time since the last fingertip contact / 1 s, clipped to [0, 1]
  [136]     T at rest (|v| < 2 mm/s and |w| < 0.02 rad/s; t_at_rest)
  [137]     wrapped goal yaw - T yaw / 0.5 rad (the DirectObs entry uses 0.1 rad and saturates at 90 deg)
  [138]     coverage of the goal T by the current T (coverage)

Support rules (support_rules; validity block of the env config; a rule with threshold None is off):
  nonfinite        any state value not finite
  joint_limit      q outside the raw joint limits +- 1e-3 rad
  tip_table        lowest fingertip collision point z < tip_floor_m (1 mm)
  link_table       lowest shoulder..wrist collision point z < link_floor_m (1 mm)
  link_t_contact   base..wrist collision points to the T < link_gap_m (2 mm) (ArmFK points)
  t_height         T COM z outside [t_z_rest_m - t_z_tol_m[0], t_z_rest_m + t_z_tol_m[1]] ([13, 25] mm)
  t_tilt           T tilt acos(1 - 2 (qx^2 + qy^2)) > max_tilt_deg (15 deg)
  gripper_tilt     gripper z axis vs world z (FK of the NRD q) > max_gripper_tilt_deg (3 deg)
  t_speed          T planar speed |v_xy| > max_t_speed_mps (0.40 m/s)
  t_yaw_rate       |w_z| > max_t_yaw_rate_radps (4.5 rad/s)
  kick             |v_xy| > kick_ratio (1.3) x max(TCP planar speed over the last kick_window_steps (5) steps, this
                   step included) + kick_excess_mps (0.03 m/s); TCP velocity = Jacobian x averaged joint rates
  unexplained      |v_xy| > unexplained_speed_mps (10 mm/s) and the dense fingertip-T signed distance > unexplained_sd_m
                   (spec 1.5 mm, env_v1 2 mm) at every one of the last unexplained_window_steps (spec 8 = 0.15 s, env_v1
                   13 = 0.26 s) steps, this step included
  penetration      dense fingertip-T signed distance < -penetration_m (spec 1.5 mm, env_v1 9 mm)
  keepout          T COM within keepout_radius_m (0.066 m = base keep-out 50.6 mm disc + 15 mm) of the pan axis
The two window rules read ring buffers that the caller keeps (SupportBuffers): the TCP planar speed and the dense
fingertip-T signed distance of the recent steps, the current step last.

Coverage (A6): the 175 cell centres of a 5 mm grid inside the goal T (bar 100 x 25 mm: 20 x 5, stem 25 x 75 mm: 5 x 15),
moved to the world with the goal pose, tested against the two boxes of the current T in its frame; coverage = share of
points inside (1.0 at the goal pose; gym-pusht counts 0.95 as success).
"""
from __future__ import annotations

import torch

from nedm.so101_push_rl import so101_direct_common as D
from nedm.so101_push_rl import so101_push_common as C
from nedm.so101_push_rl.so101_goal_common import t_frame, t_pose
from nedm.so101_push_rl.so101_planar_decoder import norm2, pan_axis_xy

OBS_DIM = 139
N_EXTRA = OBS_DIM - D.OBS_DIM
RULES = ("nonfinite", "joint_limit", "tip_table", "link_table", "link_t_contact", "t_height", "t_tilt", "gripper_tilt",
         "t_speed", "t_yaw_rate", "kick", "unexplained", "penetration", "keepout")
# = configs/rl/so101_planar/env_v1.json validity (spec A5 values; penetration and unexplained motion loosened by the
# calibration: at the spec values they flag 8.0 % and 1.4-2.5 % of the NRD replays of recorded RL-train episodes)
DEFAULT_VALIDITY = dict(tip_floor_m=0.001, link_floor_m=0.001, link_gap_m=0.002, t_z_rest_m=0.015, t_z_tol_m=[0.002, 0.010],
                        max_tilt_deg=15.0, max_gripper_tilt_deg=3.0, max_t_speed_mps=0.40, max_t_yaw_rate_radps=4.5,
                        kick_ratio=1.3, kick_excess_mps=0.03, kick_window_steps=5, unexplained_speed_mps=0.010,
                        unexplained_sd_m=0.002, unexplained_window_steps=13, penetration_m=0.009, keepout_radius_m=0.066)


def rot2(xy, yaw):
    """Rotate vectors xy [N, 2] by yaw [N] (body frame -> world frame); t_frame is the inverse."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]), -1)


class PlanarActionMap:
    """a [N, 3] -> (d_world [N, 2] m, dyaw [N] rad, d_tframe [N, 2] m); cfg = env_cfg["action"] (d_max_m, dyaw_max_rad,
    yaw_control, frame)."""

    def __init__(self, cfg):
        self.d_max = float(cfg.get("d_max_m", 0.030))
        self.dyaw_max = float(cfg.get("dyaw_max_rad", 0.3))
        self.yaw_control = bool(cfg.get("yaw_control", True))
        self.frame = cfg.get("frame", "t")
        if self.frame not in ("t", "world"):
            raise ValueError(f"action.frame must be 't' or 'world', got {self.frame!r}")

    def __call__(self, a, t_yaw):
        a = a.to(torch.float64)
        t_yaw = t_yaw.to(a.device, torch.float64)
        d = self.d_max * torch.tanh(a[:, 0:2])
        n = norm2(d)
        d = torch.where((n > self.d_max)[:, None], d * (self.d_max / torch.where(n > 0, n, torch.ones_like(n)))[:, None], d)
        if self.frame == "t":
            d_t, d_w = d, rot2(d, t_yaw)
        else:
            d_w, d_t = d, t_frame(d, t_yaw)
        dyaw = self.dyaw_max * torch.tanh(a[:, 2]) if self.yaw_control else torch.zeros_like(a[:, 2])
        return d_w, dyaw, d_t


class PlanarObs:
    """139-value observation: DirectObs (129) + 10 planar-pusher values (module docstring)."""

    def __init__(self, hist_norm, amap, delta_max):
        self.direct = D.DirectObs(hist_norm, amap, delta_max)

    def __call__(self, hist23, ok, prev_cmd, prev_inc, goal, t_left, horizon, geo, dec, t_since_contact_s, t_rest, coverage):
        base = self.direct(hist23, ok, prev_cmd, prev_inc, goal, t_left, horizon, geo)
        dev, dt = hist23.device, geo["tcp"].dtype
        yaw = geo["t_yaw"].to(dt)
        f = lambda x: torch.as_tensor(x, device=dev).to(dt)
        rel = t_frame(f(dec["target_xy"]) - geo["tcp"][:, :2], yaw) / 0.01
        vel = t_frame(f(dec["target_v"]), yaw) / 0.1
        dg = f(dec["target_yaw"]) - yaw
        gy = C.wrap(f(goal[:, 2]) - yaw) / 0.5
        extra = torch.cat((rel, vel, torch.sin(dg)[:, None], torch.cos(dg)[:, None], f(t_since_contact_s).clamp(0.0, 1.0)[:, None],
                           f(t_rest)[:, None], gy[:, None], f(coverage)[:, None]), -1)
        extra = torch.nan_to_num(extra, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-50, 50).to(torch.float32)
        out = torch.cat((base, extra), -1)
        assert out.shape[1] == OBS_DIM, out.shape
        return out


def t_at_rest(s23, v_tol=0.002, w_tol=0.02):
    """T at rest: planar speed < v_tol (m/s) and |yaw rate| < w_tol (rad/s), averaged rates of the state [N] bool."""
    return (norm2(s23[:, 17:19]) < v_tol) & (s23[:, 22].abs() < w_tol)


def goal_grid(centres, halves, step=0.005):
    """Cell centres of a `step` grid over the T boxes, T (COM) frame: [M, 2] (M = 175 for the collector T at 5 mm).
    centres, halves: [2, 2] (or [2, 3]) box centres and half sizes (ArmFK.t_centres / t_half)."""
    pts = []
    for b in range(centres.shape[0]):
        c, h = centres[b, :2].to(torch.float64), halves[b, :2].to(torch.float64)
        nx, ny = int(round(float(2 * h[0] / step))), int(round(float(2 * h[1] / step)))
        xs = c[0] - h[0] + step * (torch.arange(nx, dtype=torch.float64, device=c.device) + 0.5)
        ys = c[1] - h[1] + step * (torch.arange(ny, dtype=torch.float64, device=c.device) + 0.5)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        pts.append(torch.stack((gx.reshape(-1), gy.reshape(-1)), -1))
    return torch.cat(pts)


def coverage(s23, goal, centres, halves, step=0.005):
    """Share of the goal-T grid points (goal_grid) inside the current T [N] float64. s23 [N, 23], goal [N, 3]."""
    g = goal_grid(centres, halves, step).to(s23.device)                     # [M, 2]
    xy, yaw = t_pose(s23.to(torch.float64))
    goal = goal.to(s23.device, torch.float64)
    cg, sg = torch.cos(goal[:, 2])[:, None], torch.sin(goal[:, 2])[:, None]
    wx = goal[:, 0:1] + (cg * g[None, :, 0] - sg * g[None, :, 1])
    wy = goal[:, 1:2] + (sg * g[None, :, 0] + cg * g[None, :, 1])
    dx, dy = wx - xy[:, 0:1], wy - xy[:, 1:2]
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    inside = torch.zeros_like(lx, dtype=torch.bool)
    for b in range(centres.shape[0]):
        cb, hb = centres[b, :2].to(lx), halves[b, :2].to(lx)
        inside = inside | (((lx - cb[0]).abs() <= hb[0]) & ((ly - cb[1]).abs() <= hb[1]))
    return inside.to(torch.float64).mean(-1)


def keepout_distance(s23, kin):
    """Distance of the T COM from the shoulder pan axis (joint 0 origin) [N] (m)."""
    a = pan_axis_xy(kin).to(s23.device, torch.float64)
    return norm2(s23[:, 10:12].to(torch.float64) - a)


class SupportBuffers:
    """Ring buffers of the two window rules, the current step last: tcp_speed [N, W1] (TCP planar speed, m/s) and
    finger_sd [N, W2] (dense fingertip-T signed distance, m); W1 = kick_window_steps, W2 = unexplained_window_steps.
    reset(ids, tcp_speed, finger_sd) fills every slot of ids with the given [m] values (a rest start: the values of the
    start state); push(tcp_speed [N], finger_sd [N]) appends the values of a new step (after the NRD step, before
    support_rules)."""

    def __init__(self, n, vc, device, dtype=torch.float64):
        vc = {**DEFAULT_VALIDITY, **(vc or {})}
        self.tcp_speed = torch.zeros(n, int(vc["kick_window_steps"]), dtype=dtype, device=device)
        self.finger_sd = torch.zeros(n, int(vc["unexplained_window_steps"]), dtype=dtype, device=device)

    def reset(self, ids, tcp_speed, finger_sd):
        self.tcp_speed[ids] = tcp_speed.to(self.tcp_speed)[:, None].expand(-1, self.tcp_speed.shape[1])
        self.finger_sd[ids] = finger_sd.to(self.finger_sd)[:, None].expand(-1, self.finger_sd.shape[1])

    def push(self, tcp_speed, finger_sd):
        self.tcp_speed = torch.cat((self.tcp_speed[:, 1:], tcp_speed.to(self.tcp_speed)[:, None]), 1)
        self.finger_sd = torch.cat((self.finger_sd[:, 1:], finger_sd.to(self.finger_sd)[:, None]), 1)

    def bufs(self):
        return dict(tcp_speed=self.tcp_speed, finger_sd=self.finger_sd)


def support_rules(s23, bufs, geo, vc, kin):
    """A5 support rules of the current state (module docstring).

    s23 [N, 23] state after the NRD step; bufs = dict(tcp_speed [N, W1], finger_sd [N, W2]) ring buffers that already
    contain this step as the last column (SupportBuffers.bufs()); geo = so101_direct_common.DirectGeometry(s23) built with
    dense=True (finger_z, link_z, link_gap, tilt_deg, finger_sd); vc = validity block (missing keys: DEFAULT_VALIDITY);
    kin = So101Kin (joint limits, pan axis). Only the last kick_window_steps / unexplained_window_steps columns are used.
    Returns (bad [N] bool, reasons dict name -> [N] bool) with the names of RULES (a rule with a None threshold is
    absent)."""
    vc = {**DEFAULT_VALIDITY, **(vc or {})}
    s = s23.to(torch.float64)
    dev = s.device
    f = lambda x: x.to(dev, torch.float64)
    q = s[:, :5]
    lim = kin.limits.to(dev)
    v_t = norm2(s[:, 17:19])
    tilt = torch.rad2deg(torch.acos((1 - 2 * (s[:, 14] ** 2 + s[:, 15] ** 2)).clamp(-1, 1)))
    sd = f(geo["finger_sd"])
    reasons = dict(nonfinite=~torch.isfinite(s).all(-1),
                   joint_limit=((q < lim[:, 0] - 1e-3) | (q > lim[:, 1] + 1e-3)).any(-1))
    on = lambda key: vc.get(key) is not None
    if on("tip_floor_m"):
        reasons["tip_table"] = f(geo["finger_z"]) < vc["tip_floor_m"]
    if on("link_floor_m"):
        reasons["link_table"] = f(geo["link_z"]) < vc["link_floor_m"]
    if on("link_gap_m"):
        reasons["link_t_contact"] = f(geo["link_gap"]) < vc["link_gap_m"]
    if on("t_z_tol_m"):
        h, (lo, hi) = vc["t_z_rest_m"], vc["t_z_tol_m"]
        reasons["t_height"] = (s[:, 12] < h - lo) | (s[:, 12] > h + hi)
    if on("max_tilt_deg"):
        reasons["t_tilt"] = tilt > vc["max_tilt_deg"]
    if on("max_gripper_tilt_deg"):
        reasons["gripper_tilt"] = f(geo["tilt_deg"]) > vc["max_gripper_tilt_deg"]
    if on("max_t_speed_mps"):
        reasons["t_speed"] = v_t > vc["max_t_speed_mps"]
    if on("max_t_yaw_rate_radps"):
        reasons["t_yaw_rate"] = s[:, 22].abs() > vc["max_t_yaw_rate_radps"]
    if on("kick_ratio"):
        w = int(vc["kick_window_steps"])
        tcp_max = f(bufs["tcp_speed"])[:, -w:].max(1).values
        reasons["kick"] = v_t > vc["kick_ratio"] * tcp_max + vc["kick_excess_mps"]
    if on("unexplained_speed_mps"):
        w = int(vc["unexplained_window_steps"])
        far = (f(bufs["finger_sd"])[:, -w:] > vc["unexplained_sd_m"]).all(1)
        reasons["unexplained"] = (v_t > vc["unexplained_speed_mps"]) & far
    if on("penetration_m"):
        reasons["penetration"] = sd < -vc["penetration_m"]
    if on("keepout_radius_m"):
        reasons["keepout"] = keepout_distance(s, kin) < vc["keepout_radius_m"]
    bad = torch.zeros(s.shape[0], dtype=torch.bool, device=dev)
    for v in reasons.values():
        bad = bad | v
    return bad, reasons


def tcp_planar_speed(geo):
    """TCP planar speed (m/s) of DirectGeometry output: |J v| xy with the averaged joint rates [N]."""
    return norm2(geo["tcp_v"][:, :2].to(torch.float64))
