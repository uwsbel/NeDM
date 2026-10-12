"""Expert agreement gate harness (scripts/so101_policy/gate_expert.py): counter-based noise, the brake-end prediction
against DecoderV2.macro_goal, the harness triggers, phase latches, goals, statistics, and a short closed-loop run in
the kinematic world (skips without the task geometry / the v1 expert file).

CPU, a few seconds.

    PYTHONPATH=src python -m unittest tests.so101_policy.test_gate -v
"""
from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import decoder as D
from nedm.so101_policy import task as T
from nedm.so101_policy.kin import SO101Kin

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("gate_expert", ROOT / "scripts" / "so101_policy" / "gate_expert.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
torch.set_num_threads(2)
F = torch.float64
KIN = SO101Kin()


def start_q(n, seed=0, z=(0.06, 0.10), jaw=0.7):
    """Top-down start poses r 0.16-0.25 m, |angle| <= 1 rad, TCP z in z."""
    g = torch.Generator().manual_seed(seed)
    out = []
    while sum(len(x) for x in out) < n:
        r = 0.16 + 0.09 * torch.rand(n, generator=g, dtype=F)
        a = torch.rand(n, generator=g, dtype=F) * 2 - 1
        h = z[0] + (z[1] - z[0]) * torch.rand(n, generator=g, dtype=F)
        pos = torch.stack((0.0388 + r * torch.cos(a), r * torch.sin(a), h), -1)
        q, ok, _, _ = KIN.ik_topdown(None, pos, a, max_tilt=math.radians(30))
        out.append(q[ok])
    q = torch.cat(out)[:n]
    return torch.cat((q, torch.full((n, 1), jaw, dtype=F)), -1)


class Noise(unittest.TestCase):
    def test_hash_normal_layout_free(self):
        seeds = torch.arange(1000, 1064, dtype=torch.long)
        a = G.hash_normal(seeds, 7, 4)
        b = G.hash_normal(seeds[10:20], 7, 4)
        self.assertTrue(torch.equal(a[10:20], b))                              # same task, same counter: same noise
        self.assertFalse(torch.equal(a, G.hash_normal(seeds, 8, 4)))
        big = G.hash_normal(torch.arange(4096, dtype=torch.long), 3, 4)
        self.assertLess(abs(float(big.mean())), 0.05)
        self.assertLess(abs(float(big.std()) - 1.0), 0.05)
        self.assertLess(abs(float(torch.corrcoef(big.T)[0, 1])), 0.05)

    def test_pretanh_noise(self):
        a = torch.tensor([[0.0, 0.5, -1.0, 1.0], [0.2, 0.2, 0.2, 0.2]], dtype=F)
        eps = torch.ones(2, 4, dtype=F)
        out = G.pretanh_noise(a, eps, 0.3, torch.tensor([True, False]))
        self.assertTrue(torch.equal(out[1], a[1]))                              # unmasked row untouched
        self.assertAlmostEqual(float(out[0, 0]), math.tanh(0.3), places=12)
        self.assertAlmostEqual(float(out[0, 1]), math.tanh(math.atanh(0.5) + 0.3), places=12)
        self.assertTrue(bool((out.abs() < 1).all()))
        self.assertTrue(torch.allclose(G.pretanh_noise(a.clamp(-0.99, 0.99), eps, 0.0), a.clamp(-0.99, 0.99)))


class BrakeStop(unittest.TestCase):
    def test_equals_macro_goal(self):
        """The predicted brake end equals the decoder's GRASP macro_goal (x, y, yaw) for moving targets."""
        B = 6
        dec = D.DecoderV2(B, kin=KIN)
        dec.reset(None, start_q(B, 3), seed=torch.arange(B))
        g = torch.Generator().manual_seed(1)
        for k in range(6):                                                      # jaw opens (0.2 s), target moves
            a = torch.zeros(B, 6, dtype=F)
            a[:, :4] = (torch.rand(B, 4, generator=g, dtype=F) * 2 - 1) * 0.6
            a[:, 4] = -1
            dec.decide(a, None)
            for _ in range(5):
                q, _ = dec.step(None)
        self.assertGreater(float(dec.vel.norm(dim=-1).min()), 0.01)
        sxy, syaw = G.brake_stop(dec)
        a = torch.zeros(B, 6, dtype=F)
        a[:, 4] = 1
        info = dec.decide(a, None)
        self.assertTrue(bool(info["grasp_started"].all()))
        self.assertLess(float((dec.macro_goal[:, :2] - sxy).abs().max()), 1e-12)
        self.assertLess(float((dec.macro_goal[:, 3] - syaw).abs().max()), 1e-12)


class Triggers(unittest.TestCase):
    def test_grasp_ready(self):
        stop = torch.zeros(5, 2, dtype=F)
        yaw = torch.tensor([0.0, 0.0, 0.0, 0.0, math.pi / 2], dtype=F)
        cube = torch.tensor([[0.004, 0.0], [0.005, 0.0], [0.0, 0.0095], [0.0, 0.0], [0.0, 0.005]], dtype=F)
        cyaw = torch.tensor([0.0, 0.0, 0.0, math.radians(3.0), math.radians(1.0)], dtype=F)
        ok = G.grasp_ready(stop, yaw, cube, cyaw, tol=(0.0045, 0.010), tol_yaw=math.radians(2.5))
        # in / x too far along the closing axis / 9.5 mm across it: in / yaw off / 90 deg gripper: world y = closing axis
        self.assertEqual(ok.tolist(), [True, False, True, False, False])

    def test_place_ready(self):
        stop = torch.tensor([[0.2, 0.0], [0.2, 0.0]], dtype=F)
        syaw = torch.tensor([0.0, 0.0], dtype=F)
        rel = torch.tensor([[-0.0025, 0.001, -0.01], [-0.0025, 0.001, -0.01]], dtype=F)
        goal = torch.tensor([[0.2, 0.001], [0.2, 0.006]], dtype=F)
        gy = torch.tensor([0.1, 0.1], dtype=F)
        ok = G.place_ready(stop, syaw, rel, torch.zeros(2, dtype=F), gy, goal, gy, 0.0025, tol=0.004)
        self.assertEqual(ok.tolist(), [True, False])
        ok = G.place_ready(stop, syaw, rel, torch.zeros(2, dtype=F), gy, goal, gy + math.radians(5), 0.0025, tol=0.004,
                           tol_yaw=math.radians(3))
        self.assertEqual(ok.tolist(), [False, False])


class Latch(unittest.TestCase):
    def test_sequence(self):
        lat = G.Latches(1, "cpu")
        b = lambda x: torch.tensor([x])                                         # noqa: E731
        seq = [  # approach, carry, near_cube, near_goal -> label
            ((1, 0, 0, 0), "approach"), ((1, 0, 1, 0), "align"), ((1, 0, 0, 0), "align"),   # latched
            ((0, 0, 0, 0), "none"), ((1, 0, 0, 0), "approach"),                             # reset when the phase ends
            ((0, 1, 0, 0), "carry"), ((0, 1, 0, 1), "place"), ((0, 1, 0, 0), "place"), ((0, 0, 0, 0), "none")]
        for (ap, ca, nc, ng), want in seq:
            lab = lat.label(b(bool(ap)), b(bool(ca)), b(bool(nc)), b(bool(ng)))
            self.assertEqual(G.LABELS[int(lab)], want)
        lab = lat.label(b(False), b(False), b(False), b(False), extra_align=b(True))
        self.assertEqual(G.LABELS[int(lab)], "align")


class Stats(unittest.TestCase):
    def test_table(self):
        ref = np.array([1, 1, 1, 0, 0, 1], bool)
        oth = np.array([1, 0, 1, 0, 1, 1], bool)
        t = G.table2x2(ref, oth)
        self.assertEqual((t["n11"], t["n10"], t["n01"], t["n00"]), (3, 1, 1, 1))
        self.assertAlmostEqual(t["agree"], 4 / 6)
        self.assertAlmostEqual(t["phi"], (3 - 1) / math.sqrt(4 * 2 * 4 * 2))
        self.assertEqual(G.mcnemar(1, 1), 1.0)
        self.assertAlmostEqual(G.mcnemar(0, 6), 2 / 64)
        lo, hi = G.wilson(95, 100)
        self.assertTrue(0.88 < lo < 0.95 < hi < 0.99)

    def test_first_divergence(self):
        a = np.zeros((3, 5, 3))
        b = a.copy()
        b[0, 3, 0] = 0.006
        b[1, 4, 1] = 0.006
        fd = G.first_divergence(a, b, np.array([5, 4, 5]))
        self.assertEqual(fd.tolist(), [3, -1, -1])                              # world 1 diverges after its end


class Goals(unittest.TestCase):
    def test_goals(self):
        n = 256
        xy = T.sample_annulus(n, torch.Generator().manual_seed(0))
        yaw = torch.rand(n, generator=torch.Generator().manual_seed(1), dtype=F) * (math.pi / 2)
        g1, y1, g2, y2 = G.make_goals(xy, yaw, 5)
        h1, *_ = G.make_goals(xy, yaw, 5)
        self.assertTrue(torch.equal(g1, h1))
        for g, y in ((g1, y1), (g2, y2)):
            d = (g - xy).norm(dim=-1)
            self.assertTrue(bool(((d >= 0.06 - 1e-12) & (d <= 0.30 + 1e-12)).all()))
            r = (g - torch.tensor((0.0388, 0.0), dtype=F)).norm(dim=-1)
            self.assertTrue(bool(((r >= 0.15 - 1e-9) & (r <= 0.27 + 1e-9)).all()))
            self.assertLessEqual(float(T.wrap90(y - yaw).abs().max()), math.radians(30) + 1e-12)
        d12 = (g2 - g1).norm(dim=-1)
        self.assertTrue(bool(((d12 >= G.GOAL2_SEP[0]) & (d12 <= G.GOAL2_SEP[1])).all()))


@unittest.skipUnless(G.GEOM_NPZ.exists() or importlib.util.find_spec("mujoco"), "task geometry needs the npz or mujoco")
class ClosedLoop(unittest.TestCase):
    """A few steps of the full loop (expert, decoder, kinematic world, task rules) on CPU."""

    @classmethod
    def setUpClass(cls):
        n = 3
        q = start_q(n, 5, z=(0.08, 0.10), jaw=0.3)
        g = torch.Generator().manual_seed(2)
        xy = T.sample_annulus(n, g)
        yaw = torch.rand(n, generator=g, dtype=F)
        quat = torch.stack((torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)), -1)
        g1, y1, g2, y2 = G.make_goals(xy, torch.remainder(yaw, math.pi / 2), 3)
        z3 = np.zeros((n, 3))
        cls.tasks = dict(seeds=np.arange(5000, 5000 + n), state_q=q.numpy(), state_qd=np.zeros((n, 6)),
                         state_cube_pos=torch.cat((xy, torch.full((n, 1), T.CUBE_REST_Z, dtype=F)), -1).numpy(),
                         state_cube_quat=quat.numpy(), state_cube_v=z3, state_cube_w=z3, goal_xy=g1.numpy(),
                         goal_yaw=y1.numpy(), goal2_xy=g2.numpy(), goal2_yaw=y2.numpy())

    def check(self, r, T_):
        self.assertEqual(r["q"].shape, (3, T_, 6))
        self.assertEqual(r["cube"].shape, (3, T_, 7))
        self.assertTrue(np.isfinite(r["q"]).all())
        self.assertTrue((r["t_end"] == T_).all())                               # nothing else ends in a few steps:
        self.assertTrue((r["end"] == T.EC["timeout"]).all())                   # the task's horizon = the run's

    def test_v2_align_noise(self):
        r = G.run_episodes(self.tasks, "kin", "approach", "v2", horizon=25, log=lambda *x, **k: None)
        self.check(r, 25)
        self.assertGreater(int(r["noisy_n"].sum()), 0)                          # approach decisions got noise
        self.assertTrue((r["label"][:, 0] == G.LB["approach"]).all())
        r2 = G.run_episodes(self.tasks, "kin", "approach", "v2", horizon=25, log=lambda *x, **k: None)
        self.assertTrue(np.array_equal(r["q"], r2["q"]))                       # deterministic

    @unittest.skipUnless(G.EXPERT_V1.exists(), "v1 expert copy not present")
    def test_v1(self):
        r = G.run_episodes(self.tasks, "kin", "base", "v1", horizon=10, log=lambda *x, **k: None)
        self.check(r, 10)
        self.assertTrue((r["mode"] == -1).all())


if __name__ == "__main__":
    unittest.main()
