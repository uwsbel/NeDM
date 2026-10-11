"""Pick-and-place expert (nedm.so101_policy.expert) in a kinematic toy world: perfect servos, the cube follows the
gripper while the jaw command is closed on it and drops flat to the table when the jaw opens. Checks the phase
sequence, the place error, a recovery after a failed grasp and the helpers. CPU, about 10 s.

    PYTHONPATH=src python -m unittest tests.so101_policy.test_expert -v
"""
from __future__ import annotations

import math
import unittest

import torch

from nedm.so101_policy.decoder import CommandDecoder
from nedm.so101_policy.expert import PH, PickPlaceExpert, cube_yaw_tilt, wrap90
from nedm.so101_policy.kin import SO101Kin, yaw_of
from nedm.so101_policy.world import WorldState

F = torch.float64


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

    def state(self):
        B = len(self.q)
        z = torch.zeros(B, dtype=F)
        quat = torch.stack((torch.cos(self.yaw / 2), z, z, torch.sin(self.yaw / 2)), -1)
        zb = torch.zeros(B, dtype=torch.bool)
        contact = torch.stack((self.att, self.att, ~self.att), -1)
        return WorldState(q=self.q.clone(), qd=torch.zeros(B, 6, dtype=F), cube_pos=self.pos.clone(), cube_quat=quat,
                          cube_v=torch.zeros(B, 3, dtype=F), cube_w=torch.zeros(B, 3, dtype=F), contact=contact,
                          arm_table=zb, invalid=zb)

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
        self.pos = torch.where(self.att[:, None], p + (r @ self.rel[..., None])[..., 0], self.pos)
        self.yaw = torch.where(self.att, yaw_of(r) + self.dy, self.yaw)
        self.pos[:, 2] = torch.where(self.att, self.pos[:, 2], torch.full_like(self.pos[:, 2], 0.015))
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
    start = torch.cat((annulus(), 0.08 + 0.1 * u(n, 1)), -1)
    return cube, goal, cyaw, gyaw, start


def run(n, seed, steps=400, grab=True):
    kin = SO101Kin()
    cube, goal, cyaw, gyaw, start = tasks(n, seed)
    q0, ok, _, _ = kin.ik_topdown(None, start, torch.atan2(start[:, 1], start[:, 0] - 0.0388), max_tilt=math.radians(60))
    q0 = torch.cat((q0, torch.full((n, 1), 0.5, dtype=F)), -1)
    w = ToyWorld(q0, cube, cyaw, kin, grab)
    dec = CommandDecoder(n, kin=kin)
    ex = PickPlaceExpert(dec)
    st = w.state()
    dec.reset(torch.arange(n), st.q)
    ex.reset(torch.arange(n), st, goal, gyaw)
    seen = torch.zeros(n, len(PH), dtype=torch.bool)
    for _ in range(steps):
        st = w.step(dec.step(ex.act(st)))
        seen[torch.arange(n), ex.phase] = True
    return ex, w, goal, gyaw, seen


class TestExpert(unittest.TestCase):
    def test_pick_and_place_in_toy_world(self):
        ex, w, goal, gyaw, seen = run(12, 0)
        done = ex.phase == PH["done"]
        self.assertGreaterEqual(int(done.sum()), 11)
        err = (w.pos[done, :2] - goal[done]).norm(dim=-1)
        yerr = wrap90(w.yaw[done] - gyaw[done]).abs()
        self.assertLess(float(err.max()), 1e-3)
        self.assertLess(float(yerr.max()), math.radians(1))
        order = ["approach", "descend", "close", "grip", "lift", "carry", "lower", "release", "retreat", "done"]
        self.assertTrue(bool(seen[done][:, [PH[p] for p in order]].all()))
        self.assertEqual(int(ex.recoveries.sum()), 0)

    def test_failed_grasp_recovers(self):
        """The cube never sticks: after closing the expert sees no cube between the jaws, re-opens and re-approaches,
        then gives up after max_recover tries."""
        ex, w, goal, gyaw, seen = run(3, 1, steps=700, grab=False)
        self.assertTrue(bool(seen[:, PH["recover"]].all()))
        self.assertTrue(bool((ex.recoveries >= 1).all()))
        self.assertFalse(bool(seen[:, PH["carry"]].any()))

    def test_cube_yaw_tilt(self):
        yaw = torch.tensor([0.1, 1.0, 2.0, -0.4], dtype=F)
        z = torch.zeros_like(yaw)
        q = torch.stack((torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)), -1)
        y, t = cube_yaw_tilt(q)
        self.assertLess(float(wrap90(y - yaw).abs().max()), 1e-12)
        self.assertLess(float(t.max()), 1e-6)
        # the same cube with another face up (90 deg about x) has the same yaw mod 90
        rx = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0], dtype=F)
        from nedm.so101_policy.expert import cube_yaw_tilt as cyt
        w1, x1, y1, z1 = q.unbind(-1)
        w2, x2, y2, z2 = rx
        q2 = torch.stack((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), -1)
        y2_, t2 = cyt(q2)
        self.assertLess(float(wrap90(y2_ - yaw).abs().max()), 1e-9)
        self.assertLess(float(t2.max()), 1e-6)

    def test_act_from_mid_task_state(self):
        """reset infers 'held' from the state: a world started with the cube in the closed jaws carries it."""
        kin = SO101Kin()
        pos = torch.tensor([[0.25, 0.05, 0.08]], dtype=F)
        q, ok, _, _ = kin.ik_topdown(None, pos, torch.tensor([0.2], dtype=F))
        q = torch.cat((q, torch.tensor([[0.167]], dtype=F)), -1)
        p, r = kin.fk(q)
        w = ToyWorld(q, p[:, :2], torch.tensor([0.2], dtype=F), kin)
        w.pos[:, 2] = p[0, 2]
        w.att[:] = True
        w.rel[:] = (r.transpose(-1, -2) @ (w.pos - p)[..., None])[..., 0]
        w.dy[:] = 0.0
        dec = CommandDecoder(1, kin=kin)
        ex = PickPlaceExpert(dec)
        st = w.state()
        dec.reset(torch.arange(1), st.q)
        ex.reset(torch.arange(1), st, torch.tensor([[0.20, -0.10]], dtype=F), torch.tensor([0.3], dtype=F))
        self.assertEqual(int(ex.phase[0]), PH["lift"])
        a = ex.act(st)
        self.assertGreater(float(a[0, 4]), 0)      # keeps the jaw closed


if __name__ == "__main__":
    unittest.main()
