"""The kinematic control world (nedm.so101_policy.kin_world): servo response, capture, carry, release, the task on it.

CPU, a few seconds. Skips when mujoco or the vendor Menagerie model is missing (the arm geometry comes from the
collector scene); the CUDA parity test skips without CUDA.

    PYTHONPATH=src python -m unittest tests.so101_policy.test_kin_world -v
"""
from __future__ import annotations

import unittest

import torch

from nedm.so101_policy import task as T
from nedm.so101_policy.kin import SO101Kin
from nedm.so101_policy.kin_world import DEFAULT_COEF, N_FEAT, KinWorld, ServoModel, fit_rollout
from nedm.so101_policy.world import WorldState
from tests.so101_policy.test_task import _GEOM, Episode, make_state, needs_geom


def commands(q, macro):
    """Joint targets from a kinematic episode: the jaw closes to the firm command 0.0 rad (not to the grip angle)."""
    u = q.clone()
    closing = (macro == T.MC["grasp_close"]) | (macro == T.MC["grasp_grip"]) | (macro == T.MC["grasp_lift"]) | \
        (macro == T.MC["place_lower"])
    held_free = (macro == T.MC["free"]) & (q[:, 5] < 0.3)
    u[closing | held_free, 5] = torch.where(q[closing | held_free, 5] < 0.3, 0.0, q[closing | held_free, 5])
    return u


class TestServo(unittest.TestCase):
    def test_fit_recovers_known_model(self):
        torch.manual_seed(0)
        coef = torch.zeros(6, N_FEAT, dtype=torch.float64)
        coef[:, 0], coef[:, 1], coef[:, 2], coef[:, 3] = 0.6, 0.2, 0.3, -0.05
        coef[1, 4] = 1e-3
        model = ServoModel(coef)
        N, J = 64, 60
        u = torch.cumsum(torch.randn(N, J, 6, dtype=torch.float64) * 0.01, 1)
        q = torch.zeros(N, J, 6, dtype=torch.float64)
        q[:, :3] = u[:, :1]
        for k in range(2, J - 1):
            q[:, k + 1] = model.step(q[:, k], q[:, k] - q[:, k - 1], q[:, k - 1] - q[:, k - 2], u[:, k], u[:, k - 1])
        valid = torch.ones(N, J, dtype=torch.bool)

        def pred_err(c):   # one-step prediction error on the data (the gravity columns are collinear with "1" here)
            m = ServoModel(c)
            k = slice(2, J - 1)
            d = q[:, 1:] - q[:, :-1]
            p = m.step(q[:, k].reshape(-1, 6), d[:, 1:-1].reshape(-1, 6), d[:, :-2].reshape(-1, 6),
                       u[:, k].reshape(-1, 6), u[:, 1:-2].reshape(-1, 6))
            return float((p - q[:, 3:].reshape(-1, 6)).abs().max())
        got = ServoModel.fit(q, u, valid)
        self.assertLess(pred_err(got), 1e-5)
        self.assertLess(float((got[:, :4] - coef[:, :4]).abs().max()), 1e-3)
        ref = fit_rollout(got, q, u, valid, horizon=10, iters=20)
        self.assertLess(pred_err(ref), 1e-5)

    def test_default_holds_and_follows(self):
        model = ServoModel(DEFAULT_COEF)
        q = torch.tensor([[0.1, -0.2, 0.3, 1.4, 0.2, 0.6]], dtype=torch.float64)
        d0 = torch.zeros_like(q)
        qq, d1, d2, up = q.clone(), d0, d0, q.clone()
        for _ in range(100):                                # constant target: settles near it
            qn = model.step(qq, d1, d2, q, up)
            d2, d1, qq, up = d1, qn - qq, qn, q
        self.assertLess(float((qq - q).abs().max()), 0.01)
        self.assertLess(float(d1.abs().max()), 1e-5)
        for k in range(30):                                 # ramp of 0.02 rad per step on the pan joint
            u = q.clone()
            u[0, 0] += 0.02 * (k + 1)
            qn = model.step(qq, d1, d2, u, up)
            d2, d1, qq, up = d1, qn - qq, qn, u
        self.assertLess(float((u[0, 0] - qq[0, 0]).abs()), 0.03)


@needs_geom
class TestKinWorld(unittest.TestCase):
    def world(self, device="cpu"):
        return KinWorld(1, device, kin=SO101Kin(device), geom=T.TaskGeometry.from_mujoco(device))

    def run_episode(self, release_bottom=0.001, dy=0.0, device="cpu"):
        kin = SO101Kin()
        ep = Episode(kin)
        q, cube, quat, macro = ep.build(release_bottom=release_bottom)
        u = commands(q, macro).to(device)
        states = make_state(q, cube, quat)
        w = self.world(device)
        st0 = states[0].to(device)
        st0.cube_pos = st0.cube_pos + torch.tensor([[0.0, dy, 0.0]], dtype=torch.float64, device=device)
        w.reset(None, st0)
        task = T.PickPlaceTask(1, device, kin=SO101Kin(device), geom=T.TaskGeometry.from_mujoco(device))
        gyaw, _ = T.cube_yaw_tilt(states[-1].cube_quat)
        task.reset(torch.tensor([0], device=device), w.observe(), cube[-1:, :2].to(device), gyaw.to(device))
        out = []
        for k in range(1, len(u)):
            ws = w.step(u[k - 1: k])
            r = task.step(ws, macro[k: k + 1].to(device))
            out.append((ws, bool(w.held[0]), r))
        return out, macro

    def test_pick_carry_place_success(self):
        out, macro = self.run_episode()
        held = [h for _, h, _ in out]
        self.assertTrue(any(held))
        k0 = held.index(True)
        k1 = max(i for i, h in enumerate(held) if h)
        self.assertTrue(all(held[k0: k1 + 1]))                                      # held without a gap until release
        self.assertEqual(int(macro[k1 + 2]), T.MC["place_open"])
        self.assertIn(int(macro[k0 + 1]), (T.MC["grasp_close"], T.MC["grasp_grip"]))
        ws = [o[0] for o in out]
        jaw_held = torch.stack([s.q[0, 5] for s, h in zip(ws, held) if h])
        self.assertTrue(bool(((jaw_held >= 0.167 - 1e-12) & (jaw_held <= 0.19 + 1e-12)).all()))
        self.assertLess(float((jaw_held[2:12] - 0.167).abs().max()), 1e-12)          # the jaw stops on the cube
        self.assertGreater(max(float(s.cube_pos[0, 2]) for s in ws), T.CUBE_REST_Z + 0.025)
        ends = [int(r[3]["end"][0]) for _, _, r in out]
        first = next(e for e in ends if e)
        self.assertEqual(T.ENDS[first], "success")
        last = ws[-1]
        self.assertAlmostEqual(float(last.cube_pos[0, 2]), T.CUBE_REST_Z, places=9)
        _, tilt = T.cube_yaw_tilt(last.cube_quat)
        self.assertLess(float(tilt), 1e-9)
        self.assertLess(float(last.cube_v.norm()), 1e-9)

    def test_high_release_falls_flat(self):
        out, macro = self.run_episode(release_bottom=0.03)
        ws = [o[0] for o in out]
        held = [h for _, h, _ in out]
        k_rel = max(i for i, h in enumerate(held) if h) + 1
        vz = [float(s.cube_v[0, 2]) for s in ws[k_rel: k_rel + 4]]
        self.assertTrue(all(b < a for a, b in zip(vz, vz[1:3])))                     # accelerating downwards
        self.assertAlmostEqual(float(ws[-1].cube_pos[0, 2]), T.CUBE_REST_Z, places=9)
        ends = [int(r[3]["end"][0]) for _, _, r in out]
        first = next(e for e in ends if e)
        self.assertIn(T.ENDS[first], ("release_high", "unsupported"))

    def test_no_capture_outside_box(self):
        out, _ = self.run_episode(dy=0.02)
        self.assertFalse(any(h for _, h, _ in out))
        self.assertLess(min(float(s.q[0, 5]) for s, _, _ in out), 0.05)              # the jaw closes fully
        ends = [int(r[3]["end"][0]) for _, _, r in out]
        first = next((e for e in ends if e), 0)
        self.assertNotEqual(T.ENDS[first], "success")

    def test_reset_held_and_set_cube(self):
        kin = SO101Kin()
        q, cube, quat, macro = Episode(kin).build()
        k = int(torch.nonzero(macro == T.MC["free"])[8])                            # a carry state
        st = make_state(q, cube, quat)[k]
        w = self.world()
        w.reset(None, st)
        self.assertTrue(bool(w.held[0]))
        tcp0, _ = kin.fk(w.q)
        rel0 = w.cube_pos - tcp0
        u = q[k: k + 1].clone()
        u[0, 5] = 0.0
        for _ in range(5):
            ws = w.step(u)
        tcp1, _ = kin.fk(ws.q)
        self.assertLess(float((ws.cube_pos - tcp1 - rel0).norm()), 0.003)
        ret = w.set_cube(None, torch.tensor([[0.2, -0.1, T.CUBE_REST_Z]], dtype=torch.float64),
                         torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64))
        self.assertIsInstance(ret, WorldState)                                     # review F4
        self.assertTrue(torch.equal(ret.cube_pos, w.observe().cube_pos))
        self.assertFalse(bool(w.held[0]))
        ws = w.step(u)
        self.assertLess(float(ws.cube_v.norm()), 1e-12)
        self.assertTrue(bool(ws.contact[0, 2]))

    def test_targets_applied_like_mujoco(self):
        """review F3: targets are clipped to kin.TLO..THI and rounded to float32 before the servo model."""
        kin = SO101Kin()
        q, cube, quat, macro = Episode(kin).build()
        st = make_state(q, cube, quat)[0]
        u = q[:1].clone()
        u[0, 0] += 0.3
        u[0, 5] = -0.4
        applied = torch.minimum(torch.maximum(u, kin.tlo), kin.thi).float().double()
        outs = []
        for x in (u, applied):
            w = self.world()
            w.reset(None, st)
            for _ in range(4):
                ws = w.step(x)
            outs.append(ws.q)
        self.assertTrue(torch.equal(outs[0], outs[1]))

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA")
    def test_cpu_cuda_same(self):
        a, _ = self.run_episode(device="cpu")
        b, _ = self.run_episode(device="cuda")
        ca = torch.stack([s.cube_pos[0] for s, _, _ in a])
        cb = torch.stack([s.cube_pos[0].cpu() for s, _, _ in b])
        self.assertLess(float((ca - cb).abs().max()), 1e-9)
        self.assertEqual([int(r[3]["end"][0]) for _, _, r in a], [int(r[3]["end"][0]) for _, _, r in b])


if __name__ == "__main__":
    unittest.main()
