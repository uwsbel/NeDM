"""nedm.so101_policy.cube_symmetry: the 24 cube rotations and the canonical relabelling (CPU, seconds).

    PYTHONPATH=src python -m unittest tests.so101_policy.test_cube_symmetry -v
"""
from __future__ import annotations

import math
import unittest

import torch

from nedm.contact_nrd.model import _qmat
from nedm.so101_policy import cube_symmetry as cs


def random_quats(n, seed=0):
    q = torch.randn(n, 4, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def critic_canonical(q_hist):
    """The rule of ~/so101_policy/scripts/critic/symmetry_canon.py (canonical), written out: q_hist [B, K, 4] -> (q, k)."""
    g = cs.group()
    qn = q_hist[:, -1]
    cand = cs.qmul(qn[:, None, :].expand(-1, 24, -1), g[None].expand(len(qn), -1, -1))
    ref = torch.tensor(cs.Q_REF, dtype=torch.float64)
    k = (cand * ref).sum(-1).abs().argmax(1)
    q2 = cs.qmul(q_hist, g[k][:, None, :].expand_as(q_hist))
    return q2 * torch.where(q2[:, -1:, :1] < 0, -1.0, 1.0), k


class Group(unittest.TestCase):
    def test_elements(self):
        g = cs.group()
        self.assertEqual(g.shape, (24, 4))
        self.assertTrue(torch.equal(g[0], torch.tensor([1.0, 0, 0, 0], dtype=torch.float64)))
        self.assertLess(float((g.norm(dim=-1) - 1).abs().max()), 1e-15)
        dots = (g @ g.T).abs() - torch.eye(24, dtype=torch.float64)
        self.assertLess(float(dots.max()), 1 - 1e-6)                        # 24 distinct rotations
        r = _qmat(g)                                                        # signed permutation matrices, det 1
        self.assertTrue(torch.allclose(r.abs().sum(-1), torch.ones(24, 3, dtype=torch.float64), atol=1e-12))
        self.assertTrue(torch.allclose(torch.linalg.det(r), torch.ones(24, dtype=torch.float64), atol=1e-12))

    def test_tables(self):
        g, mul, inv = cs.group(), cs.mul_table(), cs.inverse_table()
        prod = cs.qmul(g[:, None], g[None])
        self.assertTrue(torch.allclose((prod * g[mul]).sum(-1).abs(), torch.ones(24, 24, dtype=torch.float64), atol=1e-12))
        self.assertTrue(torch.equal(mul[torch.arange(24), inv], torch.zeros(24, dtype=torch.long)))

    def test_order_matches_critic(self):
        """Same closure loop as the critic's cube_group (floating point, no snapping): same elements in the same order."""
        gens = [cs.axis_quat(a, 90) for a in "xyz"]
        G = [torch.tensor([1.0, 0, 0, 0], dtype=torch.float64)]
        while True:
            new = []
            for a in G:
                for h in gens:
                    q = cs.qmul(a, h)
                    if all(abs(float((q * x).sum())) < 1 - 1e-9 for x in G + new):
                        new.append(q)
            if not new:
                break
            G += new
        self.assertLess(float((torch.stack(G) - cs.group()).abs().max()), 1e-15)


class Canonical(unittest.TestCase):
    def test_flat_yaw(self):
        for deg in (0.3, 17.0, 45.0, 89.7):
            q = cs.axis_quat("z", deg)[None]
            self.assertEqual(int(cs.canonical_index(q)), 0, deg)
        for deg in (-30.0, 91.0, 135.0, 200.0, 300.0, 1000.0):
            qc, k = cs.canonicalise(cs.axis_quat("z", deg)[None])
            self.assertNotEqual(int(k), 0)
            self.assertAlmostEqual(math.degrees(float(cs.yaw_of_flat(qc))), deg % 90.0, places=9)
            self.assertGreaterEqual(float(qc[0, 0]), 0.0)
            self.assertLess(float(_qmat(qc)[0, 2, 2] - 1).__abs__(), 1e-12)   # still flat (body z = world z)

    def test_invariance(self):
        """Every labelling of the same physical cube gets the same canonical quaternion (up to sign)."""
        q = random_quats(500, seed=1)
        base, _ = cs.canonicalise(q)
        for k in range(24):
            other, _ = cs.canonicalise(cs.relabel(q, torch.full((500,), k), newest_sign=False))
            self.assertLess(float((other - base).abs().max()), 1e-12, k)

    def test_matches_critic_rule(self):
        q = random_quats(1000, seed=2)
        hist = torch.stack([cs.qmul(q, cs.axis_quat("z", 2.0 * j)) for j in range(4)], 1)  # [B, 4, 4]
        ref, k_ref = critic_canonical(hist)
        k = cs.canonical_index(hist[:, -1])
        self.assertTrue(torch.equal(k, k_ref))
        self.assertLess(float((cs.relabel(hist, k) - ref).abs().max()), 1e-15)

    def test_margin(self):
        q = cs.axis_quat("z", -0.5)[None]                                    # just outside the band
        self.assertNotEqual(int(cs.canonical_index(q)), 0)
        self.assertEqual(int(cs.canonical_index(q, margin=0.02)), 0)
        self.assertNotEqual(int(cs.canonical_index(cs.axis_quat("z", -30.0)[None], margin=0.02)), 0)

    def test_true_round_trip(self):
        q = random_quats(200, seed=3)
        qc, k = cs.canonicalise(q)
        back = cs.true_from_canonical(qc, k)
        self.assertLess(float(((back * q).sum(-1).abs() - 1).abs().max()), 1e-12)


if __name__ == "__main__":
    unittest.main()
