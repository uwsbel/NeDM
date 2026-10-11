"""CommandDecoder (nedm.so101_policy.decoder): rate limits, holding, workspace, jaw, filter integral, CPU/GPU parity.

CPU, a few seconds (the GPU test skips without CUDA).

    PYTHONPATH=src python -m unittest tests.so101_policy.test_decoder -v
"""
from __future__ import annotations

import math
import unittest

import torch

from nedm.so101_policy.decoder import CONTROL_DT, CommandDecoder
from nedm.so101_policy.kin import SO101Kin, rotz


def start_q(n, seed=0):
    """Top-down start poses 8-18 cm above the table, jaw 0.5 rad."""
    kin = SO101Kin()
    g = torch.Generator().manual_seed(seed)
    r = 0.15 + 0.12 * torch.rand(n, generator=g, dtype=torch.float64)
    a = (torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1) * 1.2
    z = 0.08 + 0.10 * torch.rand(n, generator=g, dtype=torch.float64)
    pos = torch.stack((0.0388 + r * torch.cos(a), r * torch.sin(a), z), -1)
    q, ok, _, _ = kin.ik_topdown(None, pos, a, max_tilt=math.radians(60))
    assert bool(ok.all())
    return torch.cat((q, torch.full((n, 1), 0.5, dtype=torch.float64)), -1)


class TestDecoder(unittest.TestCase):
    def test_zero_action_holds_the_pose(self):
        q0 = start_q(8)
        dec = CommandDecoder(8)
        dec.reset(torch.arange(8), q0)
        for _ in range(30):
            out = dec.step(torch.zeros(8, 5, dtype=torch.float64))
        # arm: the start tilt is blended out over 0.6 s, the grasp point stays
        p0, _ = dec.kin.fk(q0)
        p1, _ = dec.kin.fk(out)
        self.assertLess(float((p1 - p0).norm(dim=-1).max()), 1e-3)
        self.assertTrue(torch.allclose(out[:, 5], torch.full((8,), 0.7, dtype=torch.float64)))   # a4 = 0 -> open

    def test_rate_limits_and_joint_range_under_random_actions(self):
        n = 16
        dec = CommandDecoder(n)
        dec.reset(torch.arange(n), start_q(n, 1))
        g = torch.Generator().manual_seed(1)
        prev = dec.q_cmd.clone()
        for _ in range(150):
            out = dec.step(torch.rand(n, 5, generator=g, dtype=torch.float64) * 2 - 1)
            d = (out - prev).abs()
            self.assertLessEqual(float(d[:, :5].max()), 0.06 + 1e-12)
            self.assertLessEqual(float(d[:, 5].max()), 0.09 + 1e-12)
            self.assertTrue(bool((out >= dec.kin.tlo - 1e-12).all() & (out <= dec.kin.thi + 1e-12).all()))
            self.assertTrue(bool((dec.pos[:, 2] >= 0.015 - 1e-12).all()))
            prev = out

    def test_velocity_integral_and_filter(self):
        """A held velocity command then zero: the target moves by exactly the commanded integral (filter keeps the
        integral), no overshoot past it."""
        q0 = start_q(4, 2)
        dec = CommandDecoder(4)
        dec.reset(torch.arange(4), q0)
        p0 = dec.pos.clone()
        a = torch.zeros(4, 5, dtype=torch.float64)
        a[:, 0] = 0.5                                                   # 0.175 m/s along x
        for _ in range(10):
            dec.step(a)
        a[:, 0] = 0.0
        xs = []
        for _ in range(60):
            dec.step(a)
            xs.append(float(dec.pos[0, 0] - p0[0, 0]))
        expect = 0.5 * dec.cfg.v_max * 10 * CONTROL_DT
        self.assertAlmostEqual(xs[-1], expect, delta=1e-4)
        self.assertLessEqual(max(xs), expect + 1e-6)

    def test_jaw_moves_minimum_jerk_to_the_commanded_end(self):
        dec = CommandDecoder(2)
        dec.reset(torch.arange(2), start_q(2, 3))
        a = torch.zeros(2, 5, dtype=torch.float64)
        a[:, 4] = 1.0
        jaws = [float(dec.step(a)[0, 5]) for _ in range(40)]
        self.assertEqual(jaws[-1], dec.cfg.jaw_closed)
        self.assertTrue(all(b <= x + 1e-15 for x, b in zip(jaws, jaws[1:])))   # monotone closing
        a[:, 4] = -1.0
        jaws = [float(dec.step(a)[0, 5]) for _ in range(40)]
        self.assertEqual(jaws[-1], dec.cfg.jaw_open)

    def test_reset_subset(self):
        dec = CommandDecoder(4)
        q = start_q(4, 4)
        dec.reset(torch.arange(4), q)
        for _ in range(5):
            dec.step(torch.full((4, 5), 0.5, dtype=torch.float64))
        dec.reset(torch.tensor([1, 3]), q[[1, 3]])
        self.assertTrue(torch.equal(dec.q_cmd[[1, 3]], q[[1, 3]]))
        self.assertTrue(bool((dec.vel[[1, 3]] == 0).all()))
        self.assertFalse(torch.equal(dec.q_cmd[[0, 2]], q[[0, 2]]))

    def test_workspace_clamp_low(self):
        """Pushing down keeps the grasp point at z 15 mm, top-down."""
        dec = CommandDecoder(2)
        dec.reset(torch.arange(2), start_q(2, 5))
        a = torch.zeros(2, 5, dtype=torch.float64)
        a[:, 2] = -1.0
        for _ in range(80):
            out = dec.step(a)
        p, r = dec.kin.fk(out)
        self.assertLess(float((p[:, 2] - 0.015).abs().max()), 5e-4)
        self.assertGreater(float(r[:, 2, 2].min()), math.cos(math.radians(3)))

    @unittest.skipUnless(torch.cuda.is_available(), "no CUDA")
    def test_cpu_gpu_same_commands(self):
        n = 8
        q = start_q(n, 6)
        g = torch.Generator().manual_seed(6)
        acts = torch.rand(100, n, 5, generator=g, dtype=torch.float64) * 2 - 1
        outs = []
        for dev in ("cpu", "cuda"):
            dec = CommandDecoder(n, dev)
            dec.reset(torch.arange(n, device=dev), q.to(dev))
            outs.append(torch.stack([dec.step(a.to(dev)).cpu() for a in acts]))
        self.assertLess(float((outs[0] - outs[1]).abs().max()), 1e-9)


if __name__ == "__main__":
    unittest.main()
