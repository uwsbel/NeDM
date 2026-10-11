"""Policy action -> 6 absolute joint targets inside the scripted command family of the grasp NRD data.

The grasp NRD is accurate only on the collector's commands (critic 3.5): top-down gripper, straight grasp-point moves
with minimum-jerk timing, arm targets <= 0.06 rad and jaw targets <= 0.09 rad per 20 ms, a firm jaw command
(<= 0.05 rad) to grip and an open command 0.55-0.9 rad. ``CommandDecoder`` maps a policy action to such commands:

    a [B, 5] in [-1, 1]: a[0:3] grasp-point velocity (world, x 0.35 m/s, norm clipped to 0.35 m/s), a[3] yaw rate of
    the closing axis (x 2 rad/s), a[4] jaw (> 0: close to the firm command 0.0 rad, else open to 0.7 rad).

Per 20 ms step (``step``):

1. Smoothing filter: the velocity and yaw-rate commands pass a critically damped second-order low pass
   (natural frequency ``wn`` = 12.5 rad/s, unit gain, no overshoot of a step; a velocity step reaches 90 % in 0.31 s with
   a peak acceleration of 0.37 wn |dv| = 1.6 m/s^2 for 0.35 m/s; the integral of the velocity is kept, so a command that
   starts and ends at rest moves the target by exactly its own integral). Solved exactly for a command held over the
   step (zero-order hold), so CPU and GPU give the same targets up to rounding.
2. The internal Cartesian target (grasp point position, yaw) integrates the filtered velocity and is clamped to the
   workspace: radius about the pan axis 0.10-0.32 m, |pan angle| <= 1.75 rad, grasp point z 0.015-0.22 m (z 15 mm =
   table clearance: top-down with the jaw at the firm-grip angle 0.167 rad the lowest collision point is then 0.76 mm
   above the table), |yaw - radial direction| <= 2.5 rad (wrist roll range). A clamped axis loses its filter state.
3. Orientation: top-down at the target yaw (``kin.topdown``). The tilt of the start pose (data start poses tilt up to
   60 deg) is removed by a minimum-jerk blend over ``tilt_s`` = 0.6 s. Where top-down is out of reach (grasp point above
   about 7 cm) the IK tilts the gripper (the collector's fallback rule).
4. IK (``SO101Kin.ik_track``: the collector's two IK weights, 3 + 3 iterations warm-started from the last command,
   joints at a limit leave the Jacobian) -> 5 arm targets, change limited to 0.06 rad per step,
   clipped to the collector's target range (50 mrad inside the joint limits). If the arm cannot follow (reach, rate
   limit), the Cartesian target is pulled back to within ``max_lead`` = 1 cm of the commanded arm pose (anti-windup).
5. Jaw: a minimum-jerk move to the commanded end (0.0 or 0.7 rad) of duration max(0.2 s, 1.875 |delta| / 4.5 rad/s)
   (the collector's ``Plan.jaw`` rule), restarted from the current jaw target when the command flips; steps clipped to
   0.09 rad.

All state is per world ([B, ...] tensors on one device); ``reset(idx, q)`` re-initialises worlds idx at measured joint
angles q (the commands then hold that pose). float64 throughout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from nedm.so101_policy.kin import PAN_AXIS_XY, SO101Kin, rotvec, rotz, wrap, yaw_of

CONTROL_DT = 0.02


@dataclass
class DecoderConfig:
    v_max: float = 0.35            # m/s, grasp-point speed (norm)
    yaw_rate_max: float = 2.0      # rad/s
    wn: float = 12.5               # rad/s, natural frequency of the critically damped velocity filter
    jaw_closed: float = 0.0        # rad, firm command (the data close to -0.1245..0.05)
    jaw_open: float = 0.7          # rad (the data open to 0.55-0.9)
    jaw_speed: float = 4.5         # rad/s, Plan.jaw
    jaw_min_s: float = 0.2         # s, shortest jaw move
    arm_step: float = 0.06         # rad per step
    jaw_step: float = 0.09         # rad per step
    r_range: tuple = (0.10, 0.32)  # m, grasp point radius about the pan axis
    angle_max: float = 1.75        # rad, |pan angle| of the grasp point
    z_range: tuple = (0.015, 0.22)  # m, grasp point height
    roll_max: float = 2.5          # rad, |yaw - radial direction|
    tilt_s: float = 0.6            # s, removal of the start tilt
    max_lead: float = 0.01         # m, Cartesian target ahead of the commanded arm pose at most
    ik_iters: int = 3              # IK iterations per weight per step (SO101Kin.ik_track)


def minjerk(s):
    s = s.clamp(0.0, 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def expmap(v):
    """Rotation vectors [..., 3] -> rotation matrices (Rodrigues)."""
    th = v.norm(dim=-1, keepdim=True)[..., None]
    k = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-300)
    z = torch.zeros_like(k[..., 0])
    kx = torch.stack((z, -k[..., 2], k[..., 1], k[..., 2], z, -k[..., 0], -k[..., 1], k[..., 0], z), -1).reshape(*v.shape, 3)
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand(*v.shape[:-1], 3, 3)
    return eye + torch.sin(th) * kx + (1 - torch.cos(th)) * (kx @ kx)


class CommandDecoder:
    """Batched decoder (see the module docstring). ``step(a) -> targets [B, 6]`` (rad, jaw last)."""

    def __init__(self, num_envs, device="cpu", kin: SO101Kin | None = None, cfg: DecoderConfig | None = None):
        self.cfg = cfg or DecoderConfig()
        self.dev = torch.device(device)
        self.kin = kin or SO101Kin(self.dev)
        self.B = num_envs
        f = dict(dtype=torch.float64, device=self.dev)
        self.pos = torch.zeros(num_envs, 3, **f)          # Cartesian target: grasp point (world, m)
        self.yaw = torch.zeros(num_envs, **f)             # target yaw of the closing axis (rad, not wrapped)
        self.vel = torch.zeros(num_envs, 3, **f)          # filtered velocity (m/s) and its derivative
        self.acc = torch.zeros(num_envs, 3, **f)
        self.yrate = torch.zeros(num_envs, **f)
        self.yacc = torch.zeros(num_envs, **f)
        self.tilt0 = torch.zeros(num_envs, 3, **f)        # start tilt (rotation vector in the top-down frame)
        self.t = torch.zeros(num_envs, **f)               # steps since reset
        self.q_cmd = torch.zeros(num_envs, 6, **f)        # last joint targets
        self.jaw_from = torch.zeros(num_envs, **f)        # jaw minimum-jerk segment: start, end, step, length
        self.jaw_to = torch.zeros(num_envs, **f)
        self.jaw_k = torch.ones(num_envs, **f)
        self.jaw_n = torch.ones(num_envs, **f)
        self.jaw_mode = torch.full((num_envs,), -1, dtype=torch.long, device=self.dev)   # -1 none, 0 open, 1 closed
        self.ik_err = torch.zeros(num_envs, 2, **f)       # position (m) and angle (rad) error of the last IK
        self.lead = torch.zeros(num_envs, **f)            # Cartesian target minus commanded arm pose (m)
        self.pan_axis = torch.tensor(PAN_AXIS_XY, **f)

    # ---- reset ---------------------------------------------------------------------------------------------
    def reset(self, idx, q):
        """Worlds idx (index tensor or bool mask) start holding the measured joint angles q [n, 6]."""
        q = torch.minimum(torch.maximum(q.to(self.q_cmd), self.kin.tlo), self.kin.thi)
        p, r = self.kin.fk(q)
        yaw = yaw_of(r)
        self.q_cmd[idx] = q
        self.pos[idx], self.yaw[idx] = p, yaw
        self.tilt0[idx] = rotvec(rotz(yaw).transpose(-1, -2) @ r)
        for x in (self.vel, self.acc, self.yrate, self.yacc, self.t, self.ik_err, self.lead):
            x[idx] = 0.0
        self.jaw_from[idx], self.jaw_to[idx] = q[:, 5], q[:, 5]
        self.jaw_k[idx], self.jaw_n[idx] = 1.0, 1.0
        self.jaw_mode[idx] = -1

    # ---- one step ------------------------------------------------------------------------------------------
    def _filter(self, x, xd, u):
        """Critically damped 2nd-order filter x'' = wn^2 (u - x) - 2 wn x', input held over one step ->
        (x, x', integral of x over the step)."""
        w, T = self.cfg.wn, CONTROL_DT
        E = math.exp(-w * T)
        e0 = x - u
        x1 = u + E * (e0 * (1 + w * T) + xd * T)
        xd1 = E * (xd * (1 - w * T) - e0 * w * w * T)
        disp = u * T + e0 * (2 - E * (2 + w * T)) / w + xd * (1 - E * (1 + w * T)) / (w * w)
        return x1, xd1, disp

    def target_rot(self):
        """Target gripper rotation [B, 3, 3]: top-down at the target yaw, with the start tilt blended out."""
        lam = 1 - minjerk(self.t * CONTROL_DT / self.cfg.tilt_s)
        return rotz(self.yaw) @ expmap(self.tilt0 * lam[:, None])

    @torch.no_grad()
    def step(self, action):
        """action [B, 5] in [-1, 1] -> joint targets [B, 6] (rad)."""
        c = self.cfg
        a = action.to(self.pos).clamp(-1.0, 1.0)
        u = a[:, :3] * c.v_max
        n = u.norm(dim=-1, keepdim=True)
        u = u * torch.where(n > c.v_max, c.v_max / n.clamp_min(1e-12), torch.ones_like(n))
        self.vel, self.acc, disp = self._filter(self.vel, self.acc, u)
        self.yrate, self.yacc, ydisp = self._filter(self.yrate, self.yacc, a[:, 3] * c.yaw_rate_max)
        pos = self.pos + disp
        yaw = self.yaw + ydisp
        # workspace clamp (anti-windup: a clamped axis loses its filter state)
        z = pos[:, 2].clamp(*c.z_range)
        zc = z != pos[:, 2]
        rel = pos[:, :2] - self.pan_axis
        r = rel.norm(dim=-1)
        ang = torch.atan2(rel[:, 1], rel[:, 0])
        r2, ang2 = r.clamp(*c.r_range), ang.clamp(-c.angle_max, c.angle_max)
        xyc = (r2 != r) | (ang2 != ang)
        xy = self.pan_axis + torch.stack((torch.cos(ang2), torch.sin(ang2)), -1) * r2[:, None]
        pos = torch.cat((xy, z[:, None]), -1)
        roll = wrap(yaw - ang2)
        yc = roll.abs() > c.roll_max
        yaw = yaw - (roll - roll.clamp(-c.roll_max, c.roll_max))
        self.vel[:, 2] = torch.where(zc, 0.0, self.vel[:, 2])
        self.acc[:, 2] = torch.where(zc, 0.0, self.acc[:, 2])
        self.vel[:, :2] = torch.where(xyc[:, None], 0.0, self.vel[:, :2])
        self.acc[:, :2] = torch.where(xyc[:, None], 0.0, self.acc[:, :2])
        self.yrate = torch.where(yc, 0.0, self.yrate)
        self.yacc = torch.where(yc, 0.0, self.yacc)
        self.pos, self.yaw = pos, yaw
        self.t = self.t + 1
        # IK, rate limit, joint range
        q, ep, ea = self.kin.ik_track(self.q_cmd[:, :5], self.pos, self.target_rot(), c.ik_iters)
        q = self.q_cmd[:, :5] + (q - self.q_cmd[:, :5]).clamp(-c.arm_step, c.arm_step)
        q = torch.minimum(torch.maximum(q, self.kin.tlo[:5]), self.kin.thi[:5])
        self.ik_err = torch.stack((ep, ea), -1)
        p_cmd, _ = self.kin.fk(q)
        lead = self.pos - p_cmd
        ln = lead.norm(dim=-1, keepdim=True)
        self.pos = p_cmd + lead * torch.where(ln > c.max_lead, c.max_lead / ln.clamp_min(1e-12), torch.ones_like(ln))
        self.lead = ln[:, 0].clamp_max(c.max_lead)
        # jaw: minimum-jerk move to the commanded end, restarted when the command flips
        mode = (a[:, 4] > 0).long()
        flip = mode != self.jaw_mode
        goal = torch.where(mode == 1, c.jaw_closed, c.jaw_open).to(self.pos)
        jaw = self.q_cmd[:, 5]
        steps = torch.maximum(torch.full_like(jaw, round(c.jaw_min_s / CONTROL_DT)),
                              torch.ceil(1.875 * (goal - jaw).abs() / (c.jaw_speed * CONTROL_DT)))
        self.jaw_from = torch.where(flip, jaw, self.jaw_from)
        self.jaw_to = torch.where(flip, goal, self.jaw_to)
        self.jaw_n = torch.where(flip, steps, self.jaw_n)
        self.jaw_k = torch.where(flip, torch.zeros_like(jaw), self.jaw_k)
        self.jaw_mode = mode
        self.jaw_k = torch.minimum(self.jaw_k + 1, self.jaw_n)
        j = self.jaw_from + (self.jaw_to - self.jaw_from) * minjerk(self.jaw_k / self.jaw_n)
        j = jaw + (j - jaw).clamp(-c.jaw_step, c.jaw_step)
        j = j.clamp(float(self.kin.tlo[5]), float(self.kin.thi[5]))
        self.q_cmd = torch.cat((q, j[:, None]), -1)
        return self.q_cmd.clone()

    # ---- views ---------------------------------------------------------------------------------------------
    @property
    def jaw_done(self):
        """[B] bool: the jaw target has reached the end of its move."""
        return self.jaw_k >= self.jaw_n

    def observation(self):
        """Decoder target state for a policy [B, 11]: target position (3), sin and cos of the target yaw, filtered
        velocity (3), filtered yaw rate, jaw target (rad), jaw command (1 closed, 0 open, -1 none)."""
        return torch.cat((self.pos, torch.sin(self.yaw)[:, None], torch.cos(self.yaw)[:, None], self.vel,
                          self.yrate[:, None], self.q_cmd[:, 5:6], self.jaw_mode[:, None].to(self.pos)), -1)
