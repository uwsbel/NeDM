"""The batched G1 contact-NRD simulator: one or several frozen checkpoints ("members"), re-centred x, y, a dense
contact evaluation without data-dependent shapes, and a gradient path to the action.

One step (20 ms) of NrdSim.step(hist, ok, act):

1. Re-centre: subtract the newest pelvis x, y from all 4 history entries (the NRD is not translation invariant).
   The true world x, y stay in float64 outside the model.
2. Core change, collision logits of the 2 pairs (left foot - ground, right foot - ground), switch = logit >= 0.
3. Contact change: the contact network runs for ALL (environment, pair) entries; each change is multiplied by its
   switch (torch.where, so a non-finite output of a switched-off pair cannot leak). ContactNRD.contact_sum runs only
   the switched-on entries (torch.nonzero); both give the same next state (checked to <= 1e-9 on recorded steps).
4. ContactNRD.assemble integrates the poses from the new rates and normalises the quaternion; x, y are shifted back.

Gradients: the switch is a hard threshold (no gradient). Everything else is differentiable; with grad=True the
step keeps the autograd graph from the action (and the history) to the next state.

Members: each checkpoint is one member. step(..., member=[B] long) runs each environment with its own member;
step_ensemble() runs all members on all environments and returns the across-member standard deviation of the next
rates as a disagreement value.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from nedm.contact_nrd.model import load_model
from nedm.g1_footstep_rl import REPO_ROOT

STATE, ACTION, HISTORY, PAIRS = 67, 27, 4, 2
RATES = slice(34, 67)               # pelvis velocity, angular velocity, joint rates
FALL_Z, FALL_TILT = 0.35, math.pi / 4   # evaluation fall flag (configs/contact_nrd/evaluation/g1_locomotion.json)
HEADLINE = [REPO_ROOT / f"artifacts/contact_nrd/g1_locomotion/models/seed{s}.pt" for s in (61, 62)]


class NrdSim:
    """ckpt_paths: one checkpoint path or a list (each one member). members: optional list of indices into
    ckpt_paths to load (default all). All parameters are frozen (requires_grad False, eval mode)."""

    def __init__(self, ckpt_paths=None, device="cpu", members=None):
        paths = HEADLINE if ckpt_paths is None else ([ckpt_paths] if isinstance(ckpt_paths, (str, Path)) else list(ckpt_paths))
        if members is not None:
            paths = [paths[i] for i in members]
        self.device = torch.device(device)
        self.paths = [str(p) for p in paths]
        self.models = []
        for p in paths:
            m, _ = load_model(p, self.device)
            for q in m.parameters():
                q.requires_grad_(False)
            if (m.k, m.n_bodies, m.smax, m.amax, m.n_pairs) != (HISTORY, 1, STATE, ACTION, PAIRS) or abs(m.dt - 0.02) > 1e-12:
                raise ValueError(f"{p}: not the G1 contact NRD (history 4, one 67-channel robot, 27 actions, 2 pairs)")
            if not ((m.pair_pos[:, 0] == 0).all() and (~m.pair_mm).all()):
                raise ValueError(f"{p}: both pairs must be robot - fixed ground")
            self.models.append(m)
        self.n_members = len(self.models)
        self.dt = 0.02
        self.k = HISTORY

    # ---- one member ---------------------------------------------------------------------------------------
    def _member_step(self, model, hist, ok, act, dense=True):
        """hist [B, 4, 67] float64 (already re-centred), ok [B, 4], act [B, 27] -> next [B, 67], gates, logits [B, 2]."""
        b = hist.shape[0]
        h = hist[:, :, None, :]
        a = act.to(torch.float64)[:, None, :]
        cur, diff, lag_ok = model.code(h, ok)
        d_core = model.run_core(cur, diff, a, lag_ok)
        logits = model.logits(cur, diff, a, lag_ok)
        g = (logits >= 0).to(cur.dtype)
        if dense:
            rows = torch.arange(b, device=cur.device).repeat_interleave(PAIRS)
            pairs = torch.arange(PAIRS, device=cur.device).repeat(b)
            out = model.contact(*model.pair_inputs(cur, diff, a, lag_ok, rows, pairs)).reshape(b, PAIRS, 2, STATE)
            own = out[:, :, 0]                                                      # change of the robot
            change = torch.where(g[..., None] > 0, own * g[..., None], torch.zeros_like(own)).sum(1)
            change = change[:, None] * model.smask
        else:
            change = model.contact_sum(cur, diff, a, lag_ok, g)
        nxt = model.assemble(cur, d_core, change)
        return nxt[:, 0], g, logits

    @staticmethod
    def recentre(hist):
        """(history with the newest x, y subtracted from every entry, the newest x, y [B, 2])."""
        off = hist[:, -1, 0:2]
        return torch.cat((hist[..., 0:2] - off[:, None], hist[..., 2:]), -1), off

    # ---- public steps -------------------------------------------------------------------------------------------
    def step(self, hist, ok, act, member=None, grad=False, recentre=True, dense=True):
        """hist [B, 4, 67] float64 (newest last), ok [B, 4] bool, act [B, 27] (absolute joint targets), member [B]
        long (index of each environment's member) or None (member 0) -> next [B, 67] float64 (world x, y), gates
        [B, 2] (0/1 contact switches), logits [B, 2]. grad=False runs under torch.no_grad (fast path); grad=True keeps
        the graph. recentre=False and dense=False exist for checks only."""
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            hist = hist.to(torch.float64)
            if recentre:
                hist, off = self.recentre(hist)
            if member is None or self.n_members == 1:
                nxt, g, lg = self._member_step(self.models[0], hist, ok, act, dense)
            else:
                member = torch.as_tensor(member, device=hist.device).long()
                nxt = torch.zeros(hist.shape[0], STATE, dtype=torch.float64, device=hist.device)
                g = torch.zeros(hist.shape[0], PAIRS, dtype=torch.float64, device=hist.device)
                lg = torch.zeros_like(g)
                for m in range(self.n_members):
                    sel = member == m
                    if not bool(sel.any()):
                        continue
                    n_, g_, l_ = self._member_step(self.models[m], hist[sel], ok[sel], act[sel], dense)
                    nxt, g, lg = nxt.index_put((sel,), n_), g.index_put((sel,), g_), lg.index_put((sel,), l_)
            if recentre:
                nxt = torch.cat((nxt[:, 0:2] + off, nxt[:, 2:]), -1)
            return nxt, g, lg

    def step_ensemble(self, hist, ok, act, member=None, grad=False):
        """Every member on every environment. Returns (next [B, 67], gates [B, 2], logits [B, 2], info): next, gates
        and logits of each environment's own member (member [B] long, None = member 0); info = {"next_all"
        [M, B, 67], "gates_all" [M, B, 2], "rate_std" [B, 33] across-member std of the next rates (channels 34-66),
        "disagreement" [B] = RMS over the 33 rate channels of rate_std (mixed units: m/s, rad/s)}."""
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            hist = hist.to(torch.float64)
            hc, off = self.recentre(hist)
            outs = [self._member_step(m, hc, ok, act) for m in self.models]
            nxt_all = torch.stack([torch.cat((o[0][:, 0:2] + off, o[0][:, 2:]), -1) for o in outs])
            g_all, l_all = torch.stack([o[1] for o in outs]), torch.stack([o[2] for o in outs])
            idx = torch.zeros(hist.shape[0], dtype=torch.long, device=hist.device) if member is None else \
                torch.as_tensor(member, device=hist.device).long()
            ar = torch.arange(hist.shape[0], device=hist.device)
            std = nxt_all[:, :, RATES].std(0, unbiased=False) if self.n_members > 1 else torch.zeros_like(nxt_all[0, :, RATES])
            info = {"next_all": nxt_all, "gates_all": g_all, "rate_std": std,
                    "disagreement": std.square().mean(-1).sqrt()}
            return nxt_all[idx, ar], g_all[idx, ar], l_all[idx, ar], info

    @staticmethod
    def advance(hist, ok, nxt):
        """Drop the oldest entry, append the new state (as ContactNRD.advance, for [B, 4, 67])."""
        return torch.cat((hist[:, 1:], nxt[:, None]), 1), torch.cat((ok[:, 1:], torch.ones_like(ok[:, :1])), 1)


# ---- state helpers (67-channel layout) -------------------------------------------------------------------------
def tilt_cos(state):
    """cos of the angle between the pelvis z axis and the world z axis (data._condition)."""
    q = state[..., 3:7]
    return 1 - 2 * (q[..., 1].square() + q[..., 2].square()) / q.square().sum(-1)


def fallen(state):
    """Evaluation fall flag of each state: pelvis z below 0.35 m or tilt above 45 deg."""
    return (state[..., 2] < FALL_Z) | (tilt_cos(state) < math.cos(FALL_TILT))


def yaw(state):
    """Heading (rad) of the pelvis quaternion (z-y-x Euler yaw)."""
    w, x, y, z = state[..., 3:7].unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def heading_velocity(state):
    """[..., 3] (forward, sideways velocity in the heading frame, world yaw rate) of 67-channel states."""
    th = yaw(state)
    c, s = torch.cos(th), torch.sin(th)
    vx, vy = state[..., 34], state[..., 35]
    return torch.stack((c * vx + s * vy, -s * vx + c * vy, state[..., 39]), -1)


ENVELOPE_FEATURES = (["z", "roll", "pitch", "v_fwd", "v_lat", "vz", "w_fwd", "w_lat", "wz"]
                     + [f"q{j}" for j in range(27)] + [f"dq{j}" for j in range(27)])


def envelope_features(state):
    """[..., 67] -> [..., 63] heading-free features for data-envelope checks: pelvis z, roll, pitch (rad), pelvis
    velocity and angular velocity in the heading frame (forward, sideways, vertical), 27 joint angles, 27 joint
    rates (names ENVELOPE_FEATURES). Independent of the world x, y and the heading."""
    w, x, y, z = state[..., 3:7].unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
    th = yaw(state)
    c, s = torch.cos(th), torch.sin(th)

    def head(a):
        return c * state[..., a] + s * state[..., a + 1], -s * state[..., a] + c * state[..., a + 1]

    v_f, v_l = head(34)
    w_f, w_l = head(37)
    return torch.cat((torch.stack((state[..., 2], roll, pitch, v_f, v_l, state[..., 36], w_f, w_l, state[..., 39]), -1),
                      state[..., 7:34], state[..., 40:67]), -1)


def step_average(prev, cur, dt=0.02):
    """The 20 ms step-average rates of state `cur` given the state `prev` one model step earlier (both [..., 67],
    instantaneous or averaged rates; only their poses are used) -> cur with rate channels replaced, exactly as
    nedm.contact_nrd.data._step_average: rate = (pose - pose_prev) / dt; angular velocity = rotation vector of
    q * conj(q_prev) / dt (world frame, shortest rotation). Use it to feed a policy the same rates in MuJoCo as in
    the NRD."""
    out = cur.clone()
    out[..., 34:37] = (cur[..., 0:3] - prev[..., 0:3]) / dt
    out[..., 40:67] = (cur[..., 7:34] - prev[..., 7:34]) / dt
    q1, q0 = cur[..., 3:7], prev[..., 3:7]
    w0, v0 = q0[..., :1], -q0[..., 1:]
    w1, v1 = q1[..., :1], q1[..., 1:]
    w = w1 * w0 - (v1 * v0).sum(-1, keepdim=True)
    v = w1 * v0 + w0 * v1 + torch.cross(v1, v0, dim=-1)
    sign = torch.where(w < 0, -1.0, 1.0)
    w, v = w * sign, v * sign
    n = v.norm(dim=-1, keepdim=True)
    ang = 2 * torch.atan2(n, w)
    rotvec = torch.where(n > 1e-12, v / n.clamp_min(1e-12) * ang, 2 * v)
    out[..., 37:40] = rotvec / dt
    return out
