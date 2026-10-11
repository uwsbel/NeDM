"""Batched SO101 kinematics (nedm.so101_policy.kin) against the collector's MuJoCo scene and the NRD's KinematicChain.

CPU, a few seconds. The MuJoCo checks skip when mujoco or the vendor Menagerie model is missing.

    PYTHONPATH=src python -m unittest tests.so101_policy.test_kin -v
"""
from __future__ import annotations

import json
import math
import unittest

import numpy as np
import torch

from nedm.so101_policy import kin as K

try:
    import mujoco  # noqa: F401
    from nedm.contact_nrd.collect import so101_grasp as g
    _SCENE = g.Scene(g.load_config())
except Exception:   # mujoco or the vendor model missing
    _SCENE = None
needs_mujoco = unittest.skipIf(_SCENE is None, "mujoco / vendor Menagerie model not available")


def random_q(n, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = np.array(K.TLO) - 0.05, np.array(K.THI) + 0.05   # the joint ranges
    return rng.uniform(lo, hi, size=(n, 6))


class TestRotations(unittest.TestCase):
    def test_quat_round_trip_and_rotvec(self):
        torch.manual_seed(0)
        q = torch.randn(500, 4, dtype=torch.float64)
        q = q / q.norm(dim=-1, keepdim=True)
        q = torch.where(q[:, :1] < 0, -q, q)
        r = K.quat_mat(q)
        self.assertLess(float((K.mat_quat(r) - q).abs().max()), 1e-12)
        v = K.rotvec(r)
        ang = v.norm(dim=-1)
        self.assertLess(float((torch.cos(ang) - (r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).abs().max()), 1e-12)
        small = ang < math.radians(170)
        self.assertLess(float((K.rotvec_fast(r)[small] - v[small]).abs().max()), 1e-9)


class TestFK(unittest.TestCase):
    def test_matches_nrd_kinematic_chain(self):
        """Grasp point and gripper rotation equal the NRD's derived-channel FK (configs/.../so101_grasp_fk.json)."""
        from nedm.contact_nrd.collect import export
        from nedm.contact_nrd.model import KinematicChain
        spec = json.loads((export.REPO_ROOT / "configs/contact_nrd/derived/so101_grasp_fk.json").read_text())
        chain = KinematicChain(spec)
        q = torch.as_tensor(random_q(200))
        p_ref, r_ref = chain.pose(q[:, :5])
        p, r = K.SO101Kin().fk(q)
        self.assertLess(float((p - p_ref).abs().max()), 1e-12)
        self.assertLess(float((r - r_ref).abs().max()), 1e-12)

    def test_jacobian_finite_difference(self):
        kin = K.SO101Kin()
        q = torch.as_tensor(random_q(20, 1))
        jp, jr, p, r = kin.jacobian(q)
        h = 1e-6
        for j in range(5):
            dq = torch.zeros(6, dtype=torch.float64)
            dq[j] = h
            p1, r1 = kin.fk(q + dq)
            p0, r0 = kin.fk(q - dq)
            self.assertLess(float((jp[:, :, j] - (p1 - p0) / (2 * h)).abs().max()), 1e-7)
            w = K.rotvec(r1 @ r0.transpose(-1, -2)) / (2 * h)
            self.assertLess(float((jr[:, :, j] - w).abs().max()), 1e-6)

    @needs_mujoco
    def test_body_poses_grasp_point_lowest_point_vs_mujoco(self):
        """Every body pose, the grasp point and lowest_point (gripper, moving jaw) equal Scene.fk / grasp_pose /
        lowest to 1e-12 m on 300 random joint configurations."""
        s, m = _SCENE, _SCENE.m
        kin = K.SO101Kin()
        Q = random_q(300, 2)
        P, R = kin.body_poses(torch.as_tensor(Q))
        gp, gr = kin.fk(torch.as_tensor(Q))
        low = kin.lowest_point(torch.as_tensor(Q), per_body=True)
        for i, q in enumerate(Q):
            dk = s.fk(q)
            for b, name in enumerate(K.BODIES):
                bid = m.body(name).id
                self.assertLess(np.abs(dk.xpos[bid] - P[i, b].numpy()).max(), 1e-12)
                self.assertLess(np.abs(dk.xmat[bid].reshape(3, 3) - R[i, b].numpy()).max(), 1e-12)
            p, r = s.grasp_pose(q)
            self.assertLess(np.abs(p - gp[i].numpy()).max(), 1e-12)
            self.assertLess(np.abs(r - gr[i].numpy()).max(), 1e-12)
            for j, name in enumerate(K.GUARD_BODIES):
                self.assertLess(abs(s.lowest(dk, which=m.body(name).id) - float(low[i, j])), 1e-12)

    @needs_mujoco
    def test_points_file_round_trip(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            K.save_points(f"{d}/pts.npz")
            q = torch.as_tensor(random_q(50, 3))
            a = K.SO101Kin(points=f"{d}/pts.npz").lowest_point(q)
            b = K.SO101Kin().lowest_point(q)
            self.assertLess(float((a - b).abs().max()), 1e-15)


class TestIK(unittest.TestCase):
    def targets(self, n, seed):
        rng = np.random.default_rng(seed)
        r, a = rng.uniform(0.15, 0.27, n), np.radians(rng.uniform(-75, 75, n))
        pos = np.stack((0.0388 + r * np.cos(a), r * np.sin(a), rng.uniform(0.015, 0.12, n)), -1)
        return torch.as_tensor(pos), torch.as_tensor(a + rng.uniform(-1.0, 1.0, n))

    def test_ik_topdown_reaches_workspace(self):
        kin = K.SO101Kin()
        pos, yaw = self.targets(32, 4)
        q, ok, ep, ea = kin.ik_topdown(None, pos, yaw, max_tilt=math.radians(45))
        self.assertTrue(bool(ok.all()))
        p, r = kin.fk(q)
        self.assertLess(float((p - pos).norm(dim=-1).max()), K.IK_TOL_M)
        self.assertLess(float(ep.max()), K.IK_TOL_M)

    def test_ik_track_follows_small_moves(self):
        """Warm-started tracking keeps the grasp point on a slowly moving target (2 mm per step)."""
        kin = K.SO101Kin()
        pos, yaw = self.targets(16, 5)
        pos[:, 2] = 0.04                                      # top-down reachable height
        q, ok, _, _ = kin.ik_topdown(None, pos, yaw)
        self.assertTrue(bool(ok.all()))
        for _ in range(20):
            pos = pos + torch.tensor([0.002, 0.0, 0.0], dtype=torch.float64)
            q, ep, ea = kin.ik_track(q, pos, K.rotz(yaw))
        self.assertLess(float(ep.max()), 1e-5)
        self.assertLess(float(ea.max()), math.radians(0.5))

    @needs_mujoco
    def test_ik_topdown_same_branch_as_collector(self):
        """Scene.solve vs SO101Kin.ik_topdown on 24 random targets (no hint): same success and the same joint
        solution to 1e-6 rad (the full parity run of 1,000 poses is check_expert.py ikparity)."""
        s = _SCENE
        kin = K.SO101Kin()
        pos, yaw = self.targets(24, 6)
        q, ok, _, _ = kin.ik_topdown(None, pos, yaw, max_tilt=math.radians(45))
        for i in range(len(pos)):
            try:
                ref = s.solve(pos[i].numpy(), g.topdown(float(yaw[i])), 0.5, max_tilt=math.radians(45))
            except g.PlanError:
                self.assertFalse(bool(ok[i]))
                continue
            self.assertTrue(bool(ok[i]))
            self.assertLess(np.abs(ref[:5] - q[i].numpy()).max(), 1e-6)


if __name__ == "__main__":
    unittest.main()
