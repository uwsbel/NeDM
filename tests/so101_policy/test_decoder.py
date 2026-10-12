"""Command decoders (nedm.so101_policy.decoder): DecoderV2 bounds, macros, abort, yaw rules, determinism, CPU/GPU parity;
DecoderV1 (ablation) basics. A perfect-servo loop (joint angles = commands) stands in for a world.

CPU, about 20 s (the GPU test skips without CUDA).

    PYTHONPATH=src python -m unittest tests.so101_policy.test_decoder -v
"""
from __future__ import annotations

import math
import unittest

import torch

from nedm.so101_policy import decoder as D
from nedm.so101_policy.kin import SO101Kin, quat_mat, yaw_of
from nedm.so101_policy.world import WorldState

torch.set_num_threads(2)
F = torch.float64
KIN = SO101Kin()


def cuda_ok():
    """CUDA present and a small allocation works (the shared GPU may be full)."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1024, device="cuda")
        return True
    except Exception:
        return False


def start_q(n, seed=0, z=(0.06, 0.10), jaw=0.7):
    """Top-down-ish start poses r 0.16-0.25 m, |angle| <= 1 rad, TCP z in z, jaw angle jaw."""
    g = torch.Generator().manual_seed(seed)
    out = []
    while sum(len(x) for x in out) < n:
        r = 0.16 + 0.09 * torch.rand(n, generator=g, dtype=F)
        a = (torch.rand(n, generator=g, dtype=F) * 2 - 1)
        h = z[0] + (z[1] - z[0]) * torch.rand(n, generator=g, dtype=F)
        pos = torch.stack((0.0388 + r * torch.cos(a), r * torch.sin(a), h), -1)
        q, ok, _, _ = KIN.ik_topdown(None, pos, a, max_tilt=math.radians(30))
        out.append(q[ok])
    q = torch.cat(out)[:n]
    return torch.cat((q, torch.full((n, 1), jaw, dtype=F)), -1)


def state(q, cube_xy=None, cube_z=D.CUBE_REST_Z, yaw=None):
    n = len(q)
    if cube_xy is None:
        cube_xy = torch.full((n, 2), 5.0, dtype=F)                       # far away
    yaw = torch.zeros(n, dtype=F) if yaw is None else yaw
    z = torch.zeros(n, dtype=F)
    quat = torch.stack((torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)), -1)
    zb = torch.zeros(n, dtype=torch.bool)
    return WorldState(q=q.clone(), qd=torch.zeros(n, 6, dtype=F),
                      cube_pos=torch.cat((cube_xy, torch.as_tensor(cube_z, dtype=F).expand(n)[:, None]), -1), cube_quat=quat,
                      cube_v=torch.zeros(n, 3, dtype=F), cube_w=torch.zeros(n, 3, dtype=F),
                      contact=torch.zeros(n, 3, dtype=torch.bool), arm_table=zb, invalid=zb)


def run(dec, q, actions, st_fn=None):
    """Perfect-servo loop: per decision one action [B, 6], then decision_steps steps. -> commands [B, T, 6], modes."""
    cmds, modes = [q.clone()], []
    cur = q.clone()
    for a in actions:
        dec.decide(a, st_fn(cur) if st_fn else None)
        for _ in range(dec.cfg.decision_steps):
            cur, info = dec.step(cur)
            cmds.append(cur)
            modes.append(dec.mode.clone())
    return torch.stack(cmds, 1), torch.stack(modes, 1)


class TestHelpers(unittest.TestCase):
    def test_hash_uniform_device_and_layout_free(self):
        seed = torch.arange(1000, dtype=torch.long) * 7919 + 20261011
        cnt = torch.arange(1000, dtype=torch.long) % 13
        u = D.hash_uniform(seed, cnt)
        self.assertTrue(bool(((u >= 0) & (u < 1)).all()))
        self.assertAlmostEqual(float(u.mean()), 0.5, delta=0.03)
        perm = torch.randperm(1000)
        self.assertTrue(torch.equal(D.hash_uniform(seed[perm], cnt[perm]), u[perm]))
        if cuda_ok():
            self.assertTrue(torch.equal(D.hash_uniform(seed.cuda(), cnt.cuda()).cpu(), u))

    def test_blend_closed_form(self):
        v0, vn = torch.tensor([0.2, -0.1], dtype=F), torch.tensor([-0.1, 0.3], dtype=F)
        td = torch.full((2,), 0.1, dtype=F)
        p, v = D.blend(v0, vn, td, td)
        self.assertTrue(torch.allclose(p, (v0 + vn) * 0.05))
        self.assertTrue(torch.allclose(v, vn))
        p0, v0b = D.blend(v0, vn, torch.zeros(2, dtype=F), td)
        self.assertTrue(torch.allclose(v0b, v0) and bool((p0 == 0).all()))


class TestDecoderV2(unittest.TestCase):
    def test_zero_action_holds_and_opens_jaw(self):
        q0 = start_q(6, 0, jaw=0.3)
        dec = D.DecoderV2(6, kin=KIN)
        dec.reset(None, q0)
        cmds, modes = run(dec, q0, [torch.zeros(6, 6, dtype=F)] * 12)
        p0, _ = KIN.fk(q0)
        p1, _ = KIN.fk(cmds[:, -1])
        self.assertLess(float((p1 - p0).norm(dim=-1).max()), 1e-3)
        self.assertTrue(torch.allclose(cmds[:, -1, 5], torch.full((6,), 0.7, dtype=F)))
        self.assertTrue(bool((modes == D.M["free"]).all()))

    def test_bounds_under_random_actions(self):
        """Uniform random actions (worst case, no triggers): command steps <= dq_max, step changes <= ddq_max except
        on steps flagged as floor overrides or joint-range clips (counted), lowest collision point >= low_min - 0.05 mm,
        TCP z <= 0.20 m."""
        n = 16
        q0 = start_q(n, 1)
        dec = D.DecoderV2(n, kin=KIN, seed=1)
        dec.reset(None, q0)
        g = torch.Generator().manual_seed(1)
        acts = []
        for _ in range(40):
            a = torch.rand(n, 6, generator=g, dtype=F) * 2 - 1
            a[:, 4] = -1
            acts.append(a)
        cmds, _ = run(dec, q0, acts)
        d1 = cmds.diff(dim=1)[:, 1:, :5].abs()
        d2 = cmds.diff(n=2, dim=1)[:, 1:, :5].abs()     # (the first step starts from the measured pose)
        self.assertTrue(bool((d1 <= dec.dq_max + 1e-12).all()))
        bad = (d2 > dec.ddq_max + 1e-9).any(-1).sum()
        exempt = dec.counts["floor_override"].sum() + dec.counts["range_clipped"].sum()
        self.assertLessEqual(int(bad), int(exempt) * 2)
        self.assertLess(int(bad), 0.01 * d2.shape[0] * d2.shape[1])
        low = KIN.lowest_point(cmds.reshape(-1, 6)).reshape(n, -1)
        self.assertGreater(float(low.min()), dec.cfg.low_min - 5e-5)
        self.assertLess(float(KIN.fk(cmds.reshape(-1, 6))[0][:, 2].max()), dec.cfg.z_max + 1e-3)

    def _grasp_setup(self, n=4, seed=2):
        q0 = start_q(n, seed, z=(0.06, 0.07))
        p, r = KIN.fk(q0)
        cube_xy = p[:, :2].clone()
        dec = D.DecoderV2(n, kin=KIN, seed=seed)
        dec.reset(None, q0)
        st_fn = lambda q: state(q, cube_xy, yaw=yaw_of(r))
        run(dec, q0, [torch.zeros(n, 6, dtype=F)] * 4, st_fn)          # jaw opened, at rest
        return dec, dec.q_cmd.clone(), st_fn, cube_xy

    def test_grasp_macro_sequence_and_arm_still_while_closing(self):
        n = 4
        dec, q, st_fn, cube_xy = self._grasp_setup(n)
        a = torch.zeros(n, 6, dtype=F)
        a[:, 4] = 1.0
        a[:, 5] = -1.0                                                   # h = 3 cm
        cmds, modes = run(dec, q, [a], st_fn)
        goal = dec.macro_goal.clone()
        for _ in range(60):                                              # until every world holds the cube
            c2, m2 = run(dec, dec.q_cmd, [a * (dec.mode != D.M["hold"]).to(F)[:, None]], st_fn)
            cmds, modes = torch.cat((cmds, c2[:, 1:]), 1), torch.cat((modes, m2), 1)
            if bool((dec.mode == D.M["hold"]).all()):
                break
        seq = [D.M[m] for m in ("g_descend", "g_settle", "g_close", "g_grip", "g_lift", "hold")]
        for i in range(n):
            seen = torch.unique_consecutive(modes[i]).tolist()
            self.assertEqual([m for m in seen if m in seq][:6], seq, f"world {i}: {[D.MODES[m] for m in seen]}")
            close = modes[i] == D.M["g_close"]
            arm = cmds[i, 1:][close, :5]
            self.assertTrue(bool((arm == arm[0]).all()))                 # arm exactly still while the jaw closes
            # vertical descent: the command's TCP stays within 1.5 mm of the line (the limiter lags where the IK
            # leaves the wrist-flex limit, about 7 cm up) and ends on it (first descent, last step before the settle)
            desc = modes[i] == D.M["g_descend"]
            tcp = KIN.fk(cmds[i, 1:][desc])[0]
            self.assertLess(float((tcp[:, :2] - cube_xy[i]).norm(dim=-1).max()), 1.5e-3)
            end = int(torch.nonzero(modes[i] == D.M["g_settle"])[0])
            self.assertLess(float((KIN.fk(cmds[i, end + 1: end + 2])[0][0, :2] - cube_xy[i]).norm()), 1e-4)
        # descent end: cube centre height + 0-2 mm, raised to the floor (lowest point >= 1.5 mm at the deepest jaw)
        self.assertTrue(bool((goal[:, 2] >= D.CUBE_REST_Z - 1e-9).all()))
        floor = dec.cfg.low_min + KIN.fk(torch.cat((dec.q_des, torch.full((n, 1), dec.theta_worst, dtype=F)), -1))[0][:, 2] \
            - KIN.lowest_point(torch.cat((dec.q_des, torch.full((n, 1), dec.theta_worst, dtype=F)), -1))
        self.assertTrue(bool((goal[:, 2] <= torch.maximum(floor, torch.full_like(floor, D.CUBE_REST_Z + 0.002)) + 1e-4).all()))
        # end of the lift: 3 cm above the grasp height
        self.assertTrue(torch.allclose(dec.pos[:, 2], goal[:, 2] + 0.03, atol=1e-6))
        self.assertTrue(bool((dec.q_cmd[:, 5] == dec.cfg.jaw_closed).all()))

    def test_trigger_held_through_grasp_does_not_place(self):
        """A trigger held from the GRASP start on stays disarmed in mode hold (no PLACE right after the lift); one
        decision with the trigger released arms it, the next press starts PLACE."""
        n = 2
        dec, q, st_fn, _ = self._grasp_setup(n, 9)
        a = torch.zeros(n, 6, dtype=F)
        a[:, 4] = 1.0
        for _ in range(80):
            run(dec, dec.q_cmd, [a], st_fn)
            if bool((dec.mode == D.M["hold"]).all()):
                break
        self.assertTrue(bool((dec.mode == D.M["hold"]).all()))
        run(dec, dec.q_cmd, [a] * 3, st_fn)
        self.assertTrue(bool((dec.mode == D.M["hold"]).all()))
        self.assertFalse(bool(dec.armed.any()))
        run(dec, dec.q_cmd, [-a], st_fn)
        self.assertTrue(bool(dec.armed.all()))
        info = dec.decide(a, st_fn(dec.q_cmd))
        self.assertTrue(bool(info["place_started"].all()))
        self.assertFalse(bool(dec.armed.any()))

    def test_abort_before_close_only(self):
        n = 2
        dec, q, st_fn, _ = self._grasp_setup(n, 3)
        a = torch.zeros(n, 6, dtype=F)
        a[:, 4] = 1.0
        run(dec, q, [a], st_fn)
        self.assertTrue(bool((dec.mode == D.M["g_descend"]).all()))
        b = a.clone()
        b[0, 4] = -1.0                                                   # world 0 aborts, world 1 holds the trigger
        info = dec.decide(b, st_fn(dec.q_cmd))
        self.assertEqual(info["aborted"].tolist(), [True, False])
        self.assertEqual(dec.mode.tolist(), [D.M["free"], D.M["g_descend"]])
        while int(dec.mode[1]) != D.M["g_close"]:
            dec.step(dec.q_cmd)
        info = dec.decide(b * torch.tensor([[1.0], [-1.0]], dtype=F), st_fn(dec.q_cmd))   # world 1 tries to abort
        self.assertFalse(bool(info["aborted"][1]))
        self.assertEqual(int(dec.mode[1]), D.M["g_close"])

    def test_place_macro_opens_still_and_retreats(self):
        n = 3
        q0 = start_q(n, 4, z=(0.05, 0.06), jaw=0.167)
        p0, r0 = KIN.fk(q0)
        dec = D.DecoderV2(n, kin=KIN, seed=4)
        dec.reset(None, q0, hold=torch.ones(n, dtype=torch.bool))
        cube = p0.clone()
        cube[:, 2] = p0[:, 2] - 0.001                                    # cube centre 1 mm under the TCP
        st_fn = lambda q: state(q, cube[:, :2], cube_z=cube[:, 2], yaw=yaw_of(r0))
        a = torch.zeros(n, 6, dtype=F)
        a[:, 4] = 1.0
        b = a.clone()
        b[:, 4] = -1.0                                                   # trigger once, then off
        cmds, modes = run(dec, q0, [a] + [b] * 30, st_fn)
        for i in range(n):
            seen = [D.MODES[m] for m in torch.unique_consecutive(modes[i]).tolist()]
            self.assertEqual(seen[-4:], ["p_lower", "p_open", "p_retreat", "free"], seen)
            opening = modes[i] == D.M["p_open"]
            arm = cmds[i, 1:][opening, :5]
            self.assertTrue(bool((arm == arm[0]).all()))
            low = KIN.fk(cmds[i, 1:][opening][:1])[0][0, 2]
            rise = KIN.fk(cmds[i, -1:])[0][0, 2] - low
            self.assertGreaterEqual(float(rise), 0.05 - 1e-6)
            # lowering: cube bottom (moves with the TCP) ends 0-3 mm above the table (or at the floor)
            bottom_end = (cube[i, 2] - 0.015) + (low - p0[i, 2])
            self.assertLess(float(bottom_end), 0.003 + 1e-6)
        self.assertTrue(torch.allclose(cmds[:, -1, 5], torch.full((n,), 0.7, dtype=F)))

    def test_yaw_lock_near_cube_and_hold_turn_cap(self):
        n = 2
        q0 = start_q(n, 5, z=(0.045, 0.05))
        p0, r0 = KIN.fk(q0)
        dec = D.DecoderV2(n, kin=KIN)
        dec.reset(None, q0)
        st_fn = lambda q: state(q, p0[:, :2])                            # cube right under the TCP
        run(dec, q0, [torch.zeros(n, 6, dtype=F)] * 3, st_fn)
        y0 = dec.yaw.clone()
        a = torch.zeros(n, 6, dtype=F)
        a[:, 3] = 1.0
        run(dec, dec.q_cmd, [a] * 5, st_fn)
        self.assertLess(float((dec.yaw - y0).abs().max()), 1e-9)
        self.assertGreater(int(dec.counts["yaw_blocked"].sum()), 0)
        # holding: the yaw turns at most hold_turn_max in total
        dec2 = D.DecoderV2(n, kin=KIN)
        q1 = start_q(n, 6, z=(0.06, 0.07), jaw=0.167)
        dec2.reset(None, q1, hold=torch.ones(n, dtype=torch.bool))
        y1 = dec2.yaw.clone()
        run(dec2, q1, [a] * 30)
        self.assertLess(float((dec2.yaw - y1).abs().max()), dec2.cfg.hold_turn_max + 1e-6)
        self.assertGreater(float((dec2.yaw - y1).abs().min()), dec2.cfg.hold_turn_max - 0.05)

    def test_batch_invariance_and_determinism(self):
        """The same world gives the same commands in any batch position (macro durations by counter hash)."""
        n = 4
        dec, q, st_fn, cube_xy = self._grasp_setup(n, 7)
        perm = torch.tensor([2, 0, 3, 1])
        dec2, q2, _, _ = self._grasp_setup(n, 7)
        dec2.reset(None, q2[perm], seed=dec.seed[perm])
        dec.reset(None, q, seed=dec.seed)
        a = torch.zeros(n, 6, dtype=F)
        a[:, 4] = 1.0
        c1, _ = run(dec, q, [a] * 25, st_fn)
        r = KIN.fk(q)[1]
        c2, _ = run(dec2, q[perm], [a] * 25, lambda qq: state(qq, cube_xy[perm], yaw=yaw_of(r[perm])))
        self.assertLess(float((c1[perm] - c2).abs().max()), 1e-9)

    def test_macro_codes_match_task(self):
        try:
            from nedm.so101_policy.task import MACROS
        except Exception:
            self.skipTest("task.py not importable")
        self.assertEqual(tuple(MACROS), D.MACROS)
        dec = D.DecoderV2(len(D.MODES), kin=KIN)
        dec.mode = torch.arange(len(D.MODES))
        self.assertEqual([D.MACROS[k] for k in dec.macro().tolist()], list(D.MACRO_OF_MODE))

    @unittest.skipUnless(cuda_ok(), "no usable CUDA")
    def test_cpu_gpu_same_commands(self):
        n = 4
        q0 = start_q(n, 8, z=(0.06, 0.07))
        p, r = KIN.fk(q0)
        outs = []
        for dev in ("cpu", "cuda"):
            kin = SO101Kin(dev)
            dec = D.DecoderV2(n, dev, kin=kin, seed=8)
            q = q0.to(dev)
            dec.reset(None, q)
            st_fn = lambda qq: state(qq.cpu(), p[:, :2], yaw=yaw_of(r)).to(dev)
            g = torch.Generator().manual_seed(8)
            acts = [((torch.rand(n, 6, generator=g, dtype=F) * 2 - 1) * 0.5).to(dev) for _ in range(10)]
            trig = torch.zeros(n, 6, dtype=F, device=dev)
            trig[:, 4] = 1.0
            cm, _ = run(dec, q, acts + [trig] * 20, st_fn)
            outs.append(cm.cpu())
        self.assertLess(float((outs[0] - outs[1]).abs().max()), 1e-9)


class TestDecoderV1(unittest.TestCase):
    def test_v1_holds_and_rate_limit(self):
        q0 = start_q(4, 9)
        dec = D.DecoderV1(4, kin=KIN)
        dec.reset(torch.arange(4), q0)
        g = torch.Generator().manual_seed(9)
        prev = dec.q_cmd.clone()
        for _ in range(40):
            out = dec.step(torch.rand(4, 5, generator=g, dtype=F) * 2 - 1)
            self.assertLessEqual(float((out - prev)[:, :5].abs().max()), 0.06 + 1e-12)
            prev = out
        self.assertIs(D.CommandDecoder, D.DecoderV1)


if __name__ == "__main__":
    unittest.main()
