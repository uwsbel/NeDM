"""Observation, action mapping and reward of the SO-101 push-T tracking task, shared by the learned (NRD) environment and
the Chrono evaluation.

State order (unpadded, 23 values; units m, rad, m/s, rad/s; rates are step averages over the 20 ms model step):
  arm  q0..q4, qd0..qd4                               (indices 0-9)
  T    x y z, qw qx qy qz, vx vy vz, wx wy wz         (indices 10-22)

Observation (150 values, in this order):
  [0:92)    four actual state-history entries (oldest first; entries before the episode start are zero), 4 x 23
  [92:96)   history validity mask (1 = valid), oldest first
  [96:101)  previous applied q_cmd (rad)
  [101]     reference phase = reference frame / final frame of the task
  [102:150) four desired-motion previews at +0, +0.1, +0.2, +0.4 s (clamped to the task's final frame), 4 x 12:
            q_ref - q (5), x_ref - x, y_ref - y (2), sin, cos of (yaw_ref - yaw) (2), vx_ref - vx, vy_ref - vy (2),
            wz_ref - wz (1)
Normalisation: (raw - mean) / scale with a frozen ObsNorm (state features: RL-train statistics with physical floors;
error features: not centred, scale = max(RL-train std, physical floor)).

Action: five values a -> q_target = center + scale * a (rad), clamped to the joint limits, then to
prev_cmd +- slew (rad per 20 ms step). The clamped q_cmd is the applied command; PPO keeps the raw sample a.

Reward (after the step, against the reference at the next frame):
  Ep = |xy - xy_ref|^2 / s_p^2, Eyaw = wrap(yaw - yaw_ref)^2 / s_yaw^2, Eq = mean_j (q - q_ref)^2 / s_q^2,
  Ev = mean([(vx - vx_ref) / s_v, (vy - vy_ref) / s_v, (wz - wz_ref) / s_w]^2)
  L = w_p Ep + w_yaw Eyaw + w_q Eq + w_v Ev;  A = mean_j ((u - u_prev) / slew)^2
  r = max(0, exp(-L) - w_a A) if valid, else r_invalid
"""
from __future__ import annotations

import math

import torch

N_STATE = 23
ARM_Q, ARM_QD = slice(0, 5), slice(5, 10)
T_XY, T_Z, T_QUAT, T_V, T_W = slice(10, 12), 12, slice(13, 17), slice(17, 20), slice(20, 23)
PREVIEW_STEPS = (0, 5, 10, 20)
HISTORY = 4
OBS_DIM = HISTORY * N_STATE + HISTORY + 5 + 1 + len(PREVIEW_STEPS) * 12
assert OBS_DIM == 150

DEFAULT_REWARD = dict(pos_scale_m=0.010, yaw_scale_deg=5.0, q_scale_rad=0.05, v_scale_mps=0.05, w_scale_radps=0.5,
                      w_pos=0.45, w_yaw=0.25, w_q=0.20, w_v=0.10, w_action=0.02, invalid_reward=-1.0)
DEFAULT_FAILURE = dict(max_pos_err_m=0.03, max_yaw_err_deg=20.0, max_joint_err_rad=0.35)

# physical floors of the observation scales
STATE_FLOOR = torch.tensor([0.05] * 5 + [0.1] * 5 + [0.005, 0.005, 0.001] + [0.01] * 4 + [0.01] * 3 + [0.1] * 3, dtype=torch.float64)
PREVIEW_FLOOR = torch.tensor([0.05] * 5 + [0.01, 0.01] + [0.087, 0.01] + [0.05, 0.05] + [0.5], dtype=torch.float64)
CMD_FLOOR = 0.05


def flat23(state):
    """Model-layout state [..., 2, 13] (arm padded) -> [..., 23]."""
    return torch.cat((state[..., 0, :10], state[..., 1, :13]), -1)


def unflat23(x, smax=13):
    out = torch.zeros(*x.shape[:-1], 2, smax, dtype=x.dtype, device=x.device)
    out[..., 0, :10] = x[..., :10]
    out[..., 1, :13] = x[..., 10:23]
    return out


def yaw_of_q(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def quat_mul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), -1)


def preview_errors(cur, ref):
    """cur [B, 23], ref [B, P, 23] -> [B, P, 12]."""
    c = cur[:, None]
    dyaw = wrap(yaw_of_q(ref[..., T_QUAT]) - yaw_of_q(c[..., T_QUAT]))
    return torch.cat((ref[..., ARM_Q] - c[..., ARM_Q], ref[..., T_XY] - c[..., T_XY], torch.sin(dyaw)[..., None],
                      torch.cos(dyaw)[..., None], ref[..., 17:19] - c[..., 17:19], (ref[..., 22] - c[..., 22])[..., None]), -1)


def raw_observation(hist23, ok, prev_cmd, phase, ref_prev):
    """hist23 [B, 4, 23] (oldest first), ok [B, 4] bool, prev_cmd [B, 5], phase [B], ref_prev [B, 4, 23] (reference
    states at the preview frames) -> raw observation [B, 150] (float64)."""
    B = hist23.shape[0]
    h = hist23 * ok[..., None]
    pe = preview_errors(hist23[:, -1], ref_prev)
    return torch.cat((h.reshape(B, -1), ok.to(hist23.dtype), prev_cmd, phase[:, None].to(hist23.dtype), pe.reshape(B, -1)), -1)


class ObsNorm:
    """Frozen observation normalisation: obs = (raw - mean) / scale."""

    def __init__(self, mean, scale):
        self.mean, self.scale = mean.to(torch.float64), scale.to(torch.float64)

    @classmethod
    def fit(cls, raw):
        """raw [N, 150] observations of RL-train recorded trajectories."""
        raw = raw.to(torch.float64)
        mean = raw.mean(0)
        std = raw.std(0)
        dev = raw.device
        hist_mean = mean[: HISTORY * N_STATE].reshape(HISTORY, N_STATE)
        hist_std = std[: HISTORY * N_STATE].reshape(HISTORY, N_STATE)
        # one statistic per state channel (the four history entries share it)
        m = hist_mean.mean(0)
        s = torch.maximum(hist_std.mean(0), STATE_FLOOR.to(dev))
        mean_v = torch.zeros(OBS_DIM, dtype=torch.float64, device=dev)
        scale_v = torch.ones(OBS_DIM, dtype=torch.float64, device=dev)
        mean_v[: HISTORY * N_STATE] = m.repeat(HISTORY)
        scale_v[: HISTORY * N_STATE] = s.repeat(HISTORY)
        o = HISTORY * N_STATE
        mean_v[o:o + HISTORY], scale_v[o:o + HISTORY] = 0.5, 0.5                      # mask {0, 1} -> {-1, 1}
        o += HISTORY
        mean_v[o:o + 5] = mean[o:o + 5]
        scale_v[o:o + 5] = std[o:o + 5].clamp_min(CMD_FLOOR)
        o += 5
        mean_v[o], scale_v[o] = 0.5, 0.5                                               # phase [0, 1] -> [-1, 1]
        o += 1
        pstd = std[o:].reshape(len(PREVIEW_STEPS), 12)
        pmean = torch.zeros_like(pstd)
        pmean[:, 8] = 1.0                                                              # cos of the yaw error: centred at 1
        mean_v[o:] = pmean.reshape(-1)
        scale_v[o:] = torch.maximum(pstd, PREVIEW_FLOOR.to(dev)).reshape(-1)
        return cls(mean_v, scale_v)

    def __call__(self, raw):
        return ((raw - self.mean.to(raw.device)) / self.scale.to(raw.device)).to(torch.float32)

    def state_dict(self):
        return {"mean": self.mean.cpu(), "scale": self.scale.cpu()}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["mean"], d["scale"])


class ActionMap:
    """a [B, 5] -> applied q_cmd [B, 5] (rad) and saturation flags."""

    def __init__(self, center, scale, lo, hi, slew):
        self.center, self.scale = center.to(torch.float64), scale.to(torch.float64)
        self.lo, self.hi, self.slew = lo.to(torch.float64), hi.to(torch.float64), slew.to(torch.float64)

    def to(self, device):
        return ActionMap(*(t.to(device) for t in (self.center, self.scale, self.lo, self.hi, self.slew)))

    def __call__(self, a, prev_cmd):
        target = self.center + self.scale * a.to(torch.float64)
        bounded = torch.minimum(torch.maximum(target, self.lo), self.hi)
        step = bounded - prev_cmd
        cmd = prev_cmd + torch.minimum(torch.maximum(step, -self.slew), self.slew)
        sat_bound = (bounded != target)
        sat_slew = (cmd != bounded)
        return cmd, sat_bound, sat_slew

    def inverse(self, cmd):
        """Raw action that maps to `cmd` when no clamp is active (BC labels, replay)."""
        return ((cmd.to(torch.float64) - self.center) / self.scale)

    def state_dict(self):
        return {k: getattr(self, k).cpu() for k in ("center", "scale", "lo", "hi", "slew")}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["center"], d["scale"], d["lo"], d["hi"], d["slew"])


def tracking_terms(nxt, ref, cfg=DEFAULT_REWARD):
    """nxt, ref [B, 23] -> dict of raw errors and normalised terms."""
    dxy = nxt[:, T_XY] - ref[:, T_XY]
    pos = dxy.norm(dim=-1)
    yaw = wrap(yaw_of_q(nxt[:, T_QUAT]) - yaw_of_q(ref[:, T_QUAT]))
    dq = nxt[:, ARM_Q] - ref[:, ARM_Q]
    ev = torch.stack(((nxt[:, 17] - ref[:, 17]) / cfg["v_scale_mps"], (nxt[:, 18] - ref[:, 18]) / cfg["v_scale_mps"],
                      (nxt[:, 22] - ref[:, 22]) / cfg["w_scale_radps"]), -1)
    Ep = (pos / cfg["pos_scale_m"]) ** 2
    Eyaw = (yaw / math.radians(cfg["yaw_scale_deg"])) ** 2
    Eq = (dq ** 2).mean(-1) / cfg["q_scale_rad"] ** 2
    Ev = (ev ** 2).mean(-1)
    L = cfg["w_pos"] * Ep + cfg["w_yaw"] * Eyaw + cfg["w_q"] * Eq + cfg["w_v"] * Ev
    return dict(pos_err_m=pos, yaw_err_rad=yaw.abs(), joint_err_rad=dq.abs().max(-1).values, joint_rms_rad=dq.square().mean(-1).sqrt(),
                v_err_mps=(nxt[:, 17:19] - ref[:, 17:19]).norm(dim=-1), wz_err_radps=(nxt[:, 22] - ref[:, 22]).abs(),
                Ep=Ep, Eyaw=Eyaw, Eq=Eq, Ev=Ev, L=L)


def reward(terms, cmd, prev_cmd, slew, invalid, cfg=DEFAULT_REWARD, penalty=None):
    """penalty (optional, [B]): subtracted inside the floor, so a valid step never pays less than 0."""
    A = (((cmd - prev_cmd) / slew) ** 2).mean(-1)
    track = torch.exp(-terms["L"])
    r = track - cfg["w_action"] * A
    if penalty is not None:
        r = r - penalty
    r = torch.clamp(r, min=0.0)
    r = torch.where(invalid, torch.full_like(r, cfg["invalid_reward"]), r)
    return r, dict(track=track, action_change=A)
