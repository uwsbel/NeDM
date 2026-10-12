"""Pick-and-place expert (nedm.so101_policy.expert) through DecoderV2 in a kinematic toy world: joint angles = commands,
the cube follows the gripper while the jaw command is closed on it and drops flat to the table when the jaw opens.
Checks the phase sequence and the place error, a recovery after a failed grasp, an abort when the cube is moved during
the descent, the mid-task reset and the helpers. CPU, about 20 s.

    PYTHONPATH=src python -m unittest tests.so101_policy.test_expert -v
"""
from __future__ import annotations

import math
import unittest

import torch

from nedm.so101_policy.decoder import M, DecoderV2
from nedm.so101_policy.expert import PH, PHASES, ExpertConfig, PickPlaceExpert, cube_yaw_tilt, wrap90
from nedm.so101_policy.kin import SO101Kin, yaw_of
from nedm.so101_policy.world import WorldState

torch.set_num_threads(2)
F = torch.float64
KIN = SO101Kin()
TOY = ExpertConfig(release_shift=0.0)        # the toy cube does not move when the jaw opens


class ToyWorld:
    """Joint angles = commands (the jaw stops at the cube width 0.167 rad while it holds the cube)."""

    def __init__(self, q, cube_xy, cube_yaw, kin, grab=True):
        B = len(q)
        self.kin, self.grab = kin, grab
        self.q = q.clone()
        self.pos = torch.cat((cube_xy, torch.full((B, 1), 0.015, dtype=F)), -1)
        self.yaw = cube_yaw.clone()
        self.att = torch.zeros(B, dtype=torch.bool)
        self.rel = torch.zeros(B, 3, dtype=F)
        self.dy = torch.zeros(B, dtype=F)
        self.v = torch.zeros(B, 3, dtype=F)

    def state(self):
        B = len(self.q)
        z = torch.zeros(B, dtype=F)
        quat = torch.stack((torch.cos(self.yaw / 2), z, z, torch.sin(self.yaw / 2)), -1)
        zb = torch.zeros(B, dtype=torch.bool)
        return WorldState(q=self.q.clone(), qd=torch.zeros(B, 6, dtype=F), cube_pos=self.pos.clone(), cube_quat=quat,
                          cube_v=self.v.clone(), cube_w=torch.zeros(B, 3, dtype=F),
                          contact=torch.stack((self.att, self.att, ~self.att), -1), arm_table=zb, invalid=zb)

    def step(self, tgt):
        q = tgt.clone()
        p, r = self.kin.fk(q)
        rel = (r.transpose(-1, -2) @ (self.pos - p)[..., None])[..., 0]
        between = (rel[:, :2].abs() < 0.012).all(-1) & (rel[:, 2].abs() < 0.02)
        grab = self.grab & ~self.att & between & (tgt[:, 5] < 0.167)
        self.att |= grab
        self.rel = torch.where(grab[:, None], rel, self.rel)
        self.dy = torch.where(grab, self.yaw - yaw_of(r), self.dy)
        self.att &= ~(tgt[:, 5] > 0.25)
        q[:, 5] = torch.where(self.att | (self.grab & between), q[:, 5].clamp_min(0.167), q[:, 5])
        self.q = q
        p, r = self.kin.fk(q)
        old = self.pos.clone()
        self.pos = torch.where(self.att[:, None], p + (r @ self.rel[..., None])[..., 0], self.pos)
        self.yaw = torch.where(self.att, yaw_of(r) + self.dy, self.yaw)
        self.pos[:, 2] = torch.where(self.att, self.pos[:, 2].clamp_min(0.015), torch.full_like(self.pos[:, 2], 0.015))
        self.v = (self.pos - old) / 0.02
        return self.state()


def tasks(n, seed):
    g = torch.Generator().manual_seed(seed)
    u = lambda *s: torch.rand(*s, generator=g, dtype=F)

    def annulus():
        r, a = 0.15 + 0.12 * u(n), (u(n) * 2 - 1) * math.radians(75)
        return torch.stack((0.0388 + r * torch.cos(a), r * torch.sin(a)), -1)
    cube, goal = annulus(), annulus()
    cyaw = u(n) * math.pi / 2
    gyaw = torch.remainder(cyaw + (u(n) * 2 - 1) * math.radians(30), math.pi / 2)
    start = torch.cat((annulus(), 0.08 + 0.06 * u(n, 1)), -1)
    return cube, goal, cyaw, gyaw, start


def setup(n, seed, grab=True):
    cube, goal, cyaw, gyaw, start = tasks(n, seed)
    q0, ok, _, _ = KIN.ik_topdown(None, start, torch.atan2(start[:, 1], start[:, 0] - 0.0388), max_tilt=math.radians(60))
    q0 = torch.cat((q0, torch.full((n, 1), 0.5, dtype=F)), -1)
    w = ToyWorld(q0, cube, cyaw, KIN, grab)
    dec = DecoderV2(n, kin=KIN, seed=seed)
    ex = PickPlaceExpert(dec, TOY)
    st = w.state()
    dec.reset(None, st.q)
    ex.reset(None, st, goal, gyaw)
    return w, dec, ex, goal, gyaw, st


def run(w, dec, ex, st, decisions, hook=None):
    n = dec.B
    seen = torch.zeros(n, len(PHASES), dtype=torch.bool)
    modes = torch.zeros(n, len(M), dtype=torch.bool)
    for k in range(decisions):
        if hook is not None:
            st = hook(k, w, dec, ex) or st
        dec.decide(ex.act(st), st)
        for _ in range(dec.cfg.decision_steps):
            cmd, _ = dec.step(st.q)
            st = w.step(cmd)
            modes[torch.arange(n), dec.mode] = True
        seen[torch.arange(n), ex.phase] = True
        if bool((ex.phase == PH["done"]).all()):
            break
    return st, seen, modes


class TestExpert(unittest.TestCase):
    def test_pick_and_place_in_toy_world(self):
        w, dec, ex, goal, gyaw, st = setup(8, 0)
        st, seen, modes = run(w, dec, ex, st, 80)
        self.assertTrue(bool((ex.phase == PH["done"]).all()), ex.phase_names)
        err = (w.pos[:, :2] - goal).norm(dim=-1)
        yerr = wrap90(w.yaw - gyaw).abs()
        self.assertLess(float(err.max()), 1e-3)
        self.assertLess(float(yerr.max()), math.radians(1))
        order = ["approach", "grasp", "carry", "place", "done"]
        self.assertTrue(bool(seen[:, [PH[p] for p in order]].all()))
        self.assertTrue(bool(modes[:, [M[m] for m in ("g_close", "g_lift", "hold", "p_lower", "p_open")]].all()))
        self.assertEqual(int(ex.recoveries.sum()), 0)
        self.assertEqual(int(dec.counts["aborted"].sum()), 0)
        self.assertEqual(int(dec.counts["floor_override"].sum()), 0)
        p, _ = KIN.fk(w.q)
        self.assertTrue(bool((p[:, 2] >= w.pos[:, 2] + 0.015 + 0.03).all()))     # TCP >= 3 cm above the cube top

    def test_failed_grasp_recovers(self):
        """The cube never sticks: after the GRASP lift the expert sees no cube in the jaws, re-opens with a PLACE (no
        lowering), re-approaches, and gives up after max_recover tries."""
        w, dec, ex, goal, gyaw, st = setup(2, 1, grab=False)
        st, seen, modes = run(w, dec, ex, st, 160)
        self.assertTrue(bool(seen[:, PH["recover"]].all()))
        self.assertTrue(bool((ex.recoveries >= 1).all()))
        self.assertFalse(bool(seen[:, PH["carry"]].any()))
        self.assertTrue(bool(modes[:, M["p_open"]].all()))

    def test_abort_when_cube_moves_during_descent(self):
        """The cube is moved 3 cm while the GRASP descends: the expert aborts before the close, re-approaches the new
        pose and completes the task."""
        w, dec, ex, goal, gyaw, st = setup(2, 2)
        moved = torch.zeros(2, dtype=torch.bool)

        def hook(k, w, dec, ex):
            go = (dec.mode == M["g_descend"]) & ~moved
            if bool(go.any()):
                w.pos[go, 0] += 0.03
                moved[:] |= go
                return w.state()
            return None
        st, seen, modes = run(w, dec, ex, st, 120, hook)
        self.assertTrue(bool(moved.all()))
        self.assertEqual(int(dec.counts["aborted"].sum()), 2)
        self.assertTrue(bool((ex.phase == PH["done"]).all()), ex.phase_names)
        self.assertLess(float((w.pos[:, :2] - goal).norm(dim=-1).max()), 1e-3)

    def test_reset_mid_task_infers_carry(self):
        """A world started with the cube in the closed jaws (decoder reset in mode hold) carries it."""
        pos = torch.tensor([[0.25, 0.05, 0.06]], dtype=F)
        q, ok, _, _ = KIN.ik_topdown(None, pos, torch.tensor([0.2], dtype=F))
        q = torch.cat((q, torch.tensor([[0.167]], dtype=F)), -1)
        p, r = KIN.fk(q)
        w = ToyWorld(q, p[:, :2], torch.tensor([0.2], dtype=F), KIN)
        w.pos[:, 2] = p[0, 2]
        w.att[:] = True
        w.rel[:] = (r.transpose(-1, -2) @ (w.pos - p)[..., None])[..., 0]
        dec = DecoderV2(1, kin=KIN)
        ex = PickPlaceExpert(dec, TOY)
        st = w.state()
        dec.reset(None, st.q, hold=torch.ones(1, dtype=torch.bool))
        ex.reset(None, st, torch.tensor([[0.20, -0.10]], dtype=F), torch.tensor([0.3], dtype=F))
        self.assertEqual(int(ex.phase[0]), PH["carry"])
        st, seen, _ = run(w, dec, ex, st, 60)
        self.assertEqual(int(ex.phase[0]), PH["done"])
        self.assertLess(float((w.pos[0, :2] - torch.tensor([0.20, -0.10], dtype=F)).norm()), 1e-3)

    def test_cube_yaw_tilt(self):
        yaw = torch.tensor([0.1, 1.0, 2.0, -0.4], dtype=F)
        z = torch.zeros_like(yaw)
        q = torch.stack((torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)), -1)
        y, t = cube_yaw_tilt(q)
        self.assertLess(float(wrap90(y - yaw).abs().max()), 1e-12)
        self.assertLess(float(t.max()), 1e-6)
        # the same cube with another face up (90 deg about x) has the same yaw mod 90
        w1, x1, y1, z1 = q.unbind(-1)
        w2, x2, y2, z2 = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0], dtype=F)
        q2 = torch.stack((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), -1)
        y2_, t2 = cube_yaw_tilt(q2)
        self.assertLess(float(wrap90(y2_ - yaw).abs().max()), 1e-9)
        self.assertLess(float(t2.max()), 1e-6)


if __name__ == "__main__":
    unittest.main()
