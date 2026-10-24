"""Tests of the real2sim robot timing / servo emulation (``joint_env.RobotDynConfig`` / ``RobotDyn``, the
``JointVecEnv.step`` stage, ``$NEDM_SO101_DYN`` in ``joint_teacher.make_env``) and of the online-loop options of
``vision.viral`` (robot-like qdot ``robot_qd``, image lag up to 6 steps, ``OnlineConfig`` horizon / hold cap, the
per-sample qdot noise scale). CPU, a few seconds; the env tests need the kinematic world's files (kin points, task
geometry, the build9 development tasks; skipped without them), the MuJoCo test needs mujoco.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_robot_dyn -v
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from nedm.so101_policy import joint_env as JE

TASKS = Path("~/so101_policy/runs/viral/tasks/dev100.npz").expanduser()
HAVE_KIN = Path(JE.JointEnvConfig().kin_points).expanduser().exists() and \
    Path(JE.JointEnvConfig().geom_path).expanduser().exists() and TASKS.exists()
HAVE_MUJOCO = importlib.util.find_spec("mujoco") is not None
TRAIN = dict(p=0.8, d_lo=0, d_hi=3, jitter=0.4, s_lo=0.05, s_hi=1.0, tau_lo=0.02, tau_hi=0.2, sag_max_deg=2.5,
             noise_max=0.008)                                            # the first requested (wide) ranges
NEUTRAL = dict(p=1.0)                                                    # every world perturbed, every stage neutral
BIG = (-10.0 * torch.ones(6, dtype=torch.float64), 10.0 * torch.ones(6, dtype=torch.float64))   # no actuator clamp


def dyn(B=4, seed=0, **kw):
    """A RobotDyn of B worlds, reset at q = 0 (fixed settings: ``kw`` short forms)."""
    d = JE.RobotDyn(JE.RobotDynConfig.from_dict(kw), B, "cpu", seed)
    d.reset(torch.arange(B), torch.zeros(B, 6, dtype=torch.float64))
    return d


def decision(d, u5, q=None, lim=BIG):
    """One decision of the stage alone: u5 [B, 5, 6] action-layer targets, q [B, 6] measured (fixed) -> [B, 5, 6]."""
    B = u5.shape[0]
    q = torch.zeros(B, 6, dtype=torch.float64) if q is None else q
    d.begin()
    ex, out = [], []
    for j in range(JE.STEPS):
        ex.append(u5[:, j])
        out.append(d.apply(j, ex, u5[:, j], q, *lim))
    return torch.stack(out, 1)


# ---- the stage alone -------------------------------------------------------------------------------------------------
class TestConfig(unittest.TestCase):
    def test_short_forms_validation_and_off(self):
        c = JE.RobotDynConfig.from_dict(dict(d=2, s=0.1, tau=0.1, sag_deg=1.7, noise=0.0, note="x"))
        self.assertEqual((c.d_lo, c.d_hi, c.s_lo, c.s_hi, c.tau_lo, c.tau_hi), (2, 2, 0.1, 0.1, 0.1, 0.1))
        self.assertEqual((c.sag_lo_deg, c.sag_max_deg, c.p), (1.7, 1.7, 1.0))
        self.assertTrue(JE.RobotDynConfig().neutral)
        self.assertFalse(c.neutral)
        for bad in (dict(d=5), dict(d_lo=2, d_hi=1), dict(s=0.0), dict(tau_lo=0.0, tau_hi=0.1), dict(p=1.5),
                    dict(jitter=-0.1), dict(noise_lo=-1.0), dict(d=1, d_lo=1), dict(gain=0.5)):
            with self.assertRaises(ValueError, msg=str(bad)):
                JE.RobotDynConfig.from_dict(bad)
        for off in (None, {}, "", False, "{}", " ", "null", "false"):
            self.assertIsNone(JE.dyn_config(off), msg=repr(off))
        with self.assertRaises(ValueError):
            JE.dyn_config("[1, 2]")
        self.assertEqual(JE.dyn_config('{"d": 1}'), JE.RobotDynConfig(d_lo=1, d_hi=1))
        self.assertFalse(JE.RobotDynConfig(i_time_s=0.2).neutral)        # the integral correction acts
        with self.assertRaises(ValueError):
            JE.RobotDynConfig(i_time_s=-0.1)
        self.assertIsNone(JE.JointEnvConfig().dyn)
        cfg = JE.JointEnvConfig(dyn=TRAIN)                              # a dict -> the dataclass, serialised
        self.assertIsInstance(cfg.dyn, JE.RobotDynConfig)
        back = JE.JointEnvConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
        self.assertEqual(back.dyn, cfg.dyn)
        self.assertEqual(JE.JointEnvConfig.from_dict(JE.JointEnvConfig().to_dict()), JE.JointEnvConfig())

    def test_env_variable(self):
        with mock.patch.dict(os.environ, {JE.DYN_ENV: '{"d": 2, "s": 0.2, "tau": 0.08, "sag_deg": 1.5}'}):
            c = JE.dyn_from_env()
            self.assertEqual((c.d_lo, c.d_hi, c.s_lo, c.tau_hi, c.sag_max_deg, c.p), (2, 2, 0.2, 0.08, 1.5, 1.0))
        for off in ("", "{}"):
            with mock.patch.dict(os.environ, {JE.DYN_ENV: off}):
                self.assertIsNone(JE.dyn_from_env(), msg=off)
        env = dict(os.environ)
        env.pop(JE.DYN_ENV, None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsNone(JE.dyn_from_env())


class TestStage(unittest.TestCase):
    def test_delay_holds_the_last_command_then_shifts(self):
        B = 5
        d = dyn(B, d=0)
        d.d0 = torch.arange(B)                                          # world b: delay b steps (b = 4: the maximum)
        u1 = torch.arange(1, 6, dtype=torch.float64)[None, :, None].expand(B, 5, 6) * 0.01     # targets 1..5
        u2 = u1 + 0.1                                                   # the next decision: 11..15
        o1, o2 = decision(d, u1), decision(d, u2)
        for b in range(B):
            for j in range(JE.STEPS):
                want1 = 0.0 if j < b else u1[b, j - b, 0]               # the hold of the first decision: q at reset
                want2 = u1[b, JE.STEPS - 1 - b, 0] if j < b else u2[b, j - b, 0]   # the previous decision's last
                self.assertAlmostEqual(float(o1[b, j, 0]), float(want1), places=15)
                self.assertAlmostEqual(float(o2[b, j, 0]), float(want2), places=15)
        self.assertTrue(torch.equal(o1[0], u1[0]) and torch.equal(o2[0], u2[0]))   # d = 0: the targets themselves

    def test_jitter_statistics_and_clip(self):
        d = dyn(20000, d=2, jitter=0.4)
        d.begin()
        h = torch.bincount(d.d - 2 + 1, minlength=3).double() / 20000
        np.testing.assert_allclose(h.numpy(), [0.2, 0.6, 0.2], atol=0.015)
        d0 = dyn(20000, d=0, jitter=1.0)
        d0.begin()
        self.assertEqual(int(d0.d.min()), 0)                              # clipped at 0 ...
        self.assertAlmostEqual(float((d0.d == 1).double().mean()), 0.5, delta=0.02)
        d4 = dyn(1000, d=4, jitter=1.0)
        d4.begin()
        self.assertEqual(int(d4.d.max()), JE.STEPS - 1)                   # ... and at STEPS - 1

    def test_gain_arm_only(self):
        d = dyn(3, s=0.25)
        q = torch.full((3, 6), 0.1, dtype=torch.float64)
        u = q[:, None].repeat(1, 5, 1) + 0.4
        o = decision(d, u, q)
        np.testing.assert_allclose(o[..., :5].numpy(), 0.1 + 0.25 * 0.4, atol=1e-15)
        np.testing.assert_allclose(o[..., 5].numpy(), 0.5, atol=1e-15)   # the jaw keeps its command

    def test_lag_filter_converges(self):
        d = dyn(2, tau=0.1)                                              # a = DT / tau = 0.2
        self.assertAlmostEqual(float(d.a[0]), 0.2, places=12)
        u = torch.ones(2, 5, 6, dtype=torch.float64)
        outs = torch.cat([decision(d, u) for _ in range(20)], 1)       # 100 steps of a step command from f = 0
        n = torch.arange(1, 101, dtype=torch.float64)
        np.testing.assert_allclose(outs[0, :, 5].numpy(), (1 - 0.8 ** n).numpy(), atol=1e-12)   # jaw: no gain
        self.assertLess(float((outs[:, -1] - 1).abs().max()), 1e-9)
        self.assertEqual(float(dyn(1, tau=0.01).a[0]), 1.0)             # tau <= DT: no lag (a = 1)
        self.assertEqual(float(dyn(1).a[0]), 1.0)                       # tau 0: off

    def test_sag_sign_and_gain_independence(self):
        d = dyn(2, sag_deg=2.0)
        u = torch.zeros(2, 5, 6, dtype=torch.float64)
        o = decision(d, u)
        s = math.radians(2.0)
        np.testing.assert_allclose(o[:, :, 1].numpy(), s, atol=1e-15)    # + sag: measured q = command + sag
        np.testing.assert_allclose(o[:, :, 2].numpy(), s, atol=1e-15)
        self.assertEqual(float(o[:, :, [0, 3, 4, 5]].abs().max()), 0.0)
        # with a weak servo the sim arm still settles at command + sag: the target equals q there
        dw = dyn(1, s=0.05, sag_deg=2.0)
        q = torch.zeros(1, 6, dtype=torch.float64)
        q[0, 1] = q[0, 2] = s
        o = decision(dw, torch.zeros(1, 5, 6, dtype=torch.float64), q)
        self.assertLess(float((o[0, :, :5] - q[0, :5]).abs().max()), 1e-15)

    def test_sag_lowers_the_gripper(self):
        if not HAVE_KIN:
            self.skipTest("kinematics files missing")
        from nedm.so101_policy import vec_env as VE
        kin = VE.kinematics(JE.JointEnvConfig().kin_points, "cpu")
        q = torch.tensor(np.radians([[0.0, -5.0, -15.0, 92.0, 0.0, 30.0], [0, 50, -20, 70, 0, 5]]))
        for j in JE.SAG_JOINTS:
            dq = torch.zeros(6, dtype=torch.float64)
            dq[j] = math.radians(2.0)
            self.assertTrue(bool((kin.lowest_point(q + dq, exact=False) < kin.lowest_point(q, exact=False)).all()))

    def test_noise_statistics(self):
        d = dyn(4000, noise=0.01)
        u = torch.zeros(4000, 5, 6, dtype=torch.float64)
        o = torch.cat([decision(d, u) for _ in range(4)], 1)
        self.assertAlmostEqual(float(o[..., :5].std()), 0.01, delta=3e-4)
        self.assertLess(abs(float(o[..., :5].mean())), 3e-4)
        self.assertEqual(float(o[..., 5].abs().max()), 0.0)              # no jaw noise
        d2 = JE.RobotDyn(JE.RobotDynConfig(noise_lo=0.0, noise_max=0.008), 20000, "cpu", 1)
        d2.reset(torch.arange(20000), torch.zeros(20000, 6, dtype=torch.float64))
        self.assertAlmostEqual(float(d2.sigma.mean()), 0.004, delta=1e-4)   # sigma ~ U(0, noise_max) per world
        self.assertLessEqual(float(d2.sigma.max()), 0.008)

    def test_actuator_range(self):
        d = dyn(2, sag_deg=10.0)
        lim = (torch.full((6,), -0.05, dtype=torch.float64), torch.full((6,), 0.05, dtype=torch.float64))
        o = decision(d, torch.zeros(2, 5, 6, dtype=torch.float64), lim=lim)
        self.assertAlmostEqual(float(o[..., 1].max()), 0.05, places=15)    # sag 0.17 rad clamped to thi

    def test_draws_per_world_reset_and_reproducible(self):
        B = 3000
        d = JE.RobotDyn(JE.RobotDynConfig.from_dict(TRAIN), B, "cpu", 7)
        q = torch.randn(B, 6, dtype=torch.float64)
        d.reset(torch.arange(B), q)
        self.assertAlmostEqual(float(d.on.double().mean()), 0.8, delta=0.025)
        self.assertEqual(set(d.d0.tolist()), {0, 1, 2, 3})
        ls = torch.log(d.s)
        self.assertTrue(math.log(0.05) <= float(ls.min()) and float(ls.max()) <= 0.0)
        self.assertAlmostEqual(float(ls.mean()), (math.log(0.05) + 0.0) / 2, delta=0.06)       # log-uniform
        self.assertTrue(0.02 <= float(d.tau.min()) and float(d.tau.max()) <= 0.2)
        self.assertTrue(0.0 <= float(torch.rad2deg(d.sag).min()) and float(torch.rad2deg(d.sag).max()) <= 2.5)
        self.assertTrue(torch.equal(d.hold, q) and torch.equal(d.f, q))   # hold / lag state = the measured q
        self.assertGreater(len(set(d.s.tolist())), B - 5)                # independent per world
        p0 = d.params().clone()
        d.hold[:] = 9.0
        q2 = torch.randn(2, 6, dtype=torch.float64)
        d.reset(torch.tensor([5, 11]), q2)                               # only the reset worlds change
        keep = torch.ones(B, dtype=torch.bool)
        keep[[5, 11]] = False
        self.assertTrue(torch.equal(d.params()[keep], p0[keep]))
        self.assertFalse(torch.equal(d.params()[~keep], p0[~keep]))
        self.assertTrue(torch.equal(d.hold[[5, 11]], q2) and bool((d.hold[keep] == 9.0).all()))
        e = JE.RobotDyn(JE.RobotDynConfig.from_dict(TRAIN), B, "cpu", 7)
        e.reset(torch.arange(B), q)
        self.assertTrue(torch.equal(e.params(), p0))                      # same seed: same draws
        # paired settings share their draws: other ranges, the same perturbed worlds and delays
        f = JE.RobotDyn(JE.RobotDynConfig.from_dict(dict(TRAIN, s_lo=0.5, tau_hi=0.4)), B, "cpu", 7)
        f.reset(torch.arange(B), q)
        self.assertTrue(torch.equal(f.on, p0[:, 0].bool()) and torch.equal(f.d0, p0[:, 1].long()))

    def test_p0_and_neutral_return_the_input(self):
        u = torch.randn(6, 5, 6, dtype=torch.float64)
        q = torch.randn(6, 6, dtype=torch.float64)
        for kw in (dict(TRAIN, p=0.0), NEUTRAL):
            d = dyn(6, **kw)
            d.reset(torch.arange(6), q)
            self.assertTrue(torch.equal(decision(d, u, q), u), msg=str(kw))
        self.assertFalse(dyn(6, **dict(TRAIN, p=0.0)).active)
        self.assertFalse(dyn(6, **NEUTRAL).active)                       # neutral: the stage never runs
        self.assertTrue(dyn(6, i_time_s=0.2).active)

    def test_summary_has_every_column(self):
        off = dyn(8, **dict(TRAIN, p=0.0)).summary()
        self.assertEqual(tuple(off), JE.RobotDyn.SUMMARY)                # the csv header of the first log row
        self.assertEqual(off["dyn_on"], 0.0)
        self.assertTrue(all(math.isnan(v) for k, v in off.items() if k != "dyn_on"))
        on = dyn(8, **dict(TRAIN, p=1.0)).summary()
        self.assertEqual(tuple(on), JE.RobotDyn.SUMMARY)
        self.assertFalse(any(math.isnan(v) for v in on.values()))


class TestIntegral(unittest.TestCase):
    """The robot guard's integral correction (``hw_lib.Guard.apply``: i_time_s, i_max_deg, i_e_max_deg, i_v_min)."""

    def loop(self, d, n, cmd, q, qd_fn=None):
        """n steps of an ideal servo (q = the stage's output) at a constant command -> q."""
        for i in range(n):
            j = i % JE.STEPS
            if j == 0:
                d.begin()
                ex = []
            ex.append(cmd)
            qd = None if qd_fn is None else qd_fn(q)
            q_new = d.apply(j, ex, cmd, q, *BIG, qd)
            q = q_new
        return q

    def test_cancels_a_small_sag(self):
        d = dyn(2, sag_deg=0.8, i_time_s=0.2)
        cmd = torch.zeros(2, 6, dtype=torch.float64)
        q = self.loop(d, 400, cmd, cmd.clone())
        s = math.radians(0.8)
        np.testing.assert_allclose(d.integ[:, 1:3].numpy(), -s, atol=1e-9)   # integ -> - sag
        self.assertLess(float(q[:, :5].abs().max()), 1e-9)                    # the arm settles at its command
        self.assertEqual(float(d.integ[:, [0, 3, 4]].abs().max()), 0.0)       # no error, no correction
        # the same sag without the integral: measured = command + sag
        q0 = self.loop(dyn(2, sag_deg=0.8), 50, cmd, cmd.clone())
        np.testing.assert_allclose(q0[:, 1:3].numpy(), s, atol=1e-15)

    def test_gates_cap_and_reset(self):
        DT = JE.CONTROL_DT
        cmd = torch.zeros(3, 6, dtype=torch.float64)
        big = torch.zeros(3, 6, dtype=torch.float64)
        big[:, :5] = math.radians(1.5)                                   # |e| >= 1 deg: a blocked joint, no push
        d = dyn(3, i_time_s=0.2)
        decision(d, cmd[:, None].expand(3, 5, 6), q=big)
        self.assertEqual(float(d.integ.abs().max()), 0.0)
        small = torch.zeros(3, 6, dtype=torch.float64)
        small[:, :5] = math.radians(0.5)                                 # e = -0.5 deg
        d = dyn(3, i_time_s=0.2)
        d.begin()
        qd = torch.zeros(3, 6, dtype=torch.float64)
        qd[0, :5] = -0.05                                                # moves to its command faster than i_v_min
        qd[1, :5] = 0.05                                                 # moves away from it
        d.apply(0, [cmd], cmd, small, *BIG, qd)
        want = -math.radians(0.5) * DT / 0.2
        np.testing.assert_allclose(d.integ[0].numpy(), 0.0, atol=0)
        np.testing.assert_allclose(d.integ[1:].numpy(), want, atol=1e-15)
        for i in range(1, 2000):                                         # the cap
            if i % JE.STEPS == 0:
                d.begin()
            d.apply(i % JE.STEPS, [cmd] * (i % JE.STEPS + 1), cmd, small, *BIG)
        np.testing.assert_allclose(d.integ.numpy(), -math.radians(3.0), atol=1e-15)
        d.reset(torch.tensor([2]), torch.zeros(1, 6, dtype=torch.float64))   # only the reset world restarts
        self.assertEqual(float(d.integ[2].abs().max()), 0.0)
        self.assertEqual(float(d.integ[:2].max()), -math.radians(3.0))
        off = dyn(3, **dict(p=0.0, i_time_s=0.2, sag_deg=0.5))           # unperturbed worlds never integrate
        off.on[:] = False
        off.active = True
        o = decision(off, cmd[:, None].expand(3, 5, 6), q=small)
        self.assertTrue(torch.equal(o, cmd[:, None].expand(3, 5, 6)) and float(off.integ.abs().max()) == 0.0)

    def test_antiwindup_option(self):
        """``i_antiwindup`` (off by default: the old rule bit for bit; the robot guard's rule, ``hw_lib.Guard``): a
        joint held at its actuator range does not wind the integral up, the correction never passes the range."""
        self.assertFalse(JE.RobotDynConfig().i_antiwindup)
        self.assertNotIn("i_antiwindup", JE.RobotDynConfig(i_time_s=0.2).to_dict())       # off: the old record
        on = JE.RobotDynConfig.from_dict(dict(i_time_s=0.2, i_antiwindup=True))
        self.assertIs(on.to_dict()["i_antiwindup"], True)
        self.assertEqual(JE.RobotDynConfig.from_dict(json.loads(json.dumps(on.to_dict()))), on)
        self.assertFalse(on.neutral)
        hi = math.radians(101.2)                                         # the real wrist-roll stop (roll cap)
        lim = (-10.0 * torch.ones(6, dtype=torch.float64),
               torch.tensor([10.0, 10.0, 10.0, 10.0, hi, 10.0], dtype=torch.float64))
        cmd = torch.zeros(2, 6, dtype=torch.float64)
        cmd[:, 4] = math.radians(101.08)                                 # below the range end ...
        cmd[:, 1] = -math.radians(0.45)                                  # ... and a stuck lift (the correction acts)
        q = torch.zeros(2, 6, dtype=torch.float64)
        q[:, 4] = math.radians(100.69)                                   # the roll held at 100.69 deg
        runs = {}
        for tag, kw in (("default", {}), ("off", dict(i_antiwindup=False)), ("on", dict(i_antiwindup=True))):
            d = dyn(2, i_time_s=0.2, **kw)
            o = torch.cat([decision(d, cmd[:, None].expand(2, 5, 6), q=q, lim=lim) for _ in range(40)], 1)
            runs[tag] = (o, d.integ.clone())
        self.assertTrue(torch.equal(runs["default"][0], runs["off"][0]))             # default = off, bit for bit
        self.assertTrue(torch.equal(runs["default"][1], runs["off"][1]))
        (o0, i0), (o1, i1) = runs["off"], runs["on"]
        np.testing.assert_allclose(torch.rad2deg(i0[:, 4]).numpy(), 3.0, atol=1e-9)    # old: wound up at the stop
        np.testing.assert_allclose(i1[:, 4].numpy(), hi - math.radians(101.08), atol=1e-12)   # new: to the end only
        np.testing.assert_allclose(torch.rad2deg(i1[:, 1]).numpy(), -3.0, atol=1e-9)  # the lift: corrected as before
        self.assertTrue(torch.equal(o1[..., [0, 1, 2, 3, 5]], o0[..., [0, 1, 2, 3, 5]]))
        np.testing.assert_allclose(o1[..., 4].numpy(), o0[..., 4].numpy(), atol=1e-12)  # the same clamped roll
        self.assertLessEqual(float(o1[..., 4].max()), hi + 1e-12)
        # leaving the stop: the old rule keeps the +3 deg on the roll, the new one at most what it had (0.12 deg)
        cmd2 = cmd.clone()
        cmd2[:, 4] = math.radians(90.0)
        q2 = q.clone()
        q2[:, 4] = math.radians(90.0)
        for tag in ("off", "on"):
            d = dyn(2, i_time_s=0.2, i_antiwindup=tag == "on")
            for _ in range(40):
                decision(d, cmd[:, None].expand(2, 5, 6), q=q, lim=lim)
            o = decision(d, cmd2[:, None].expand(2, 5, 6), q=q2, lim=lim)
            want = 3.0 if tag == "off" else math.degrees(hi) - 101.08
            np.testing.assert_allclose(torch.rad2deg(o[:, 0, 4]).numpy(), 90.0 + want, atol=1e-6)


# ---- the env: bit-identical when off, the stage in the step loop -----------------------------------------------------
def head_step(env, targets):
    """``JointVecEnv.step`` of the unmodified code (commit 3a4f84b34) for absolute targets, no q offset, servo scale 1:
    the action layer, the world step, the task step per 20 ms step."""
    u5 = targets.to(dtype=torch.float64)
    fl = (JE.FLOOR if env.floor_z == 0.0 else JE.FLOOR + env.floor_z) if env.cfg.floor_guard else None
    ex = []
    for j in range(JE.STEPS):
        u = JE.safety(u5[:, j], env.ws.q.to(torch.float64), env.tlo, env.thi, env.lead, env.kin, fl, env.floor_plane)
        ex.append(u)
        ws = env.world.step(u)
        env.ws = ws if env.cfg.world != "mujoco" else ws.to(env.dev)
        env.task.step(env.ws, None, None)
    env.u_exe = torch.stack(ex, 1)


def tasks(n):
    from nedm.so101_policy.vision.demos import load_tasks
    return load_tasks(TASKS, np.arange(n))


def kin_env(t, dyn_=False, **kw):
    from nedm.so101_policy import joint_teacher as JT
    from nedm.so101_policy.vision.demos import start_state
    B = len(t["seeds"])
    env = JT.make_env("kin", B, JE.Codec(np.zeros(18), np.ones(18)), "cpu", dyn=dyn_, **kw)
    env.reset_tasks(torch.arange(B), start_state(t, "cpu"), torch.as_tensor(t["goal_xy"]),
                    torch.as_tensor(t["goal_yaw"]), torch.as_tensor(t["seeds"]))
    return env


def roll(env, n=8, step=None):
    """n decisions moving the arm down and forward and closing the jaw -> stacked state + executed targets."""
    out = []
    for k in range(n):
        tg = env.ws.q[:, None, :].expand(env.B, JE.STEPS, 6).clone()
        tg[..., 1] += 0.03 * (k + 1)
        tg[..., 2] -= 0.015 * (k + 1)
        tg[..., 5] -= 0.02 * (k + 1)
        if step is None:
            env.step(None, targets=tg)
        else:
            step(env, tg)
        out.append(torch.cat((env.ws.q, env.ws.cube_pos, env.ws.cube_quat, env.task.tr.end[:, None].double(),
                              env.u_exe.reshape(env.B, -1)), -1))
    return torch.stack(out, 1)


@unittest.skipUnless(HAVE_KIN, "kinematics files / development tasks missing")
class TestEnvOff(unittest.TestCase):
    def test_off_p0_and_neutral_are_the_old_step(self):
        t = tasks(6)
        ref = roll(kin_env(t), step=head_step)                           # the unmodified per-step loop
        for d_ in (False, dict(TRAIN, p=0.0), NEUTRAL):
            env = kin_env(t, d_)
            self.assertEqual(env.dyn is None, d_ is False)
            got = roll(env)
            self.assertTrue(torch.equal(got, ref), msg=str(d_))

    def test_on_changes_the_motion_and_records(self):
        t = tasks(6)
        ref = roll(kin_env(t))
        env = kin_env(t, dict(TRAIN, p=1.0))
        got = roll(env)
        self.assertFalse(torch.equal(got[..., :6], ref[..., :6]))
        self.assertTrue(torch.equal(got[:, 0, -30:-24], ref[:, 0, -30:-24]))     # u_exe = the action layer's targets
        info = env.dyn_info()
        self.assertEqual(info["cfg"]["d_hi"], 3)
        self.assertEqual(np.asarray(info["params"]).shape, (6, len(JE.RobotDyn.COLS)))
        json.dumps(info)
        # a slow servo moves less: achieved / commanded motion < 1
        env_s = kin_env(t, dict(s=0.1, tau=0.1))
        slow = roll(env_s)
        self.assertLess(float((slow[:, -1, 1] - slow[:, 0, 1]).abs().mean()),
                        float((ref[:, -1, 1] - ref[:, 0, 1]).abs().mean()) * 0.8)

    def test_reset_tasks_draws_are_reproducible_and_make_env_reads_the_variable(self):
        t = tasks(6)
        a = kin_env(t, TRAIN)
        p_a = a.dyn.params().clone()
        r_a = roll(a, 3)
        b = kin_env(t, TRAIN)
        self.assertTrue(torch.equal(b.dyn.params(), p_a))
        self.assertTrue(torch.equal(roll(b, 3), r_a))
        from nedm.so101_policy import joint_teacher as JT
        codec = JE.Codec(np.zeros(18), np.ones(18))
        with mock.patch.dict(os.environ, {JE.DYN_ENV: json.dumps(dict(d=2, s=0.2))}):
            e = JT.make_env("kin", 2, codec, "cpu")
            self.assertEqual((e.dyn.cfg.d_lo, e.dyn.cfg.s_hi), (2, 0.2))
            self.assertEqual(e.resolved()["env"]["dyn"]["d_hi"], 2)
            self.assertIsNone(JT.make_env("kin", 2, codec, "cpu", dyn=False).dyn)   # explicit off wins
        env = dict(os.environ)
        env.pop(JE.DYN_ENV, None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsNone(JT.make_env("kin", 2, codec, "cpu").dyn)

    def test_auto_reset_path_draws(self):
        from nedm.so101_policy.vision.demos import load_tasks
        t = load_tasks(TASKS, np.arange(16))
        cfg = JE.JointEnvConfig(world="kin", num_envs=8, device="cpu", auto_reset=True, stagger=True, starts="",
                                seed=3, dyn=TRAIN, horizon=20)
        env = JE.JointVecEnv(cfg, JE.Codec(np.zeros(18), np.ones(18)))
        f64 = dict(dtype=torch.float64)
        env.bank = {k: torch.as_tensor(t[f"state_{k}"], **f64) for k in ("q", "qd", "cube_pos", "cube_quat", "cube_v",
                                                                          "cube_w")}
        env.pool = dict(i=torch.arange(16), goal_xy=torch.as_tensor(t["goal_xy"], **f64),
                        goal_yaw=torch.as_tensor(t["goal_yaw"], **f64))
        env.reset()
        self.assertTrue(torch.equal(env.dyn.hold, env.ws.q.to(torch.float64)))
        p0 = env.dyn.params().clone()
        done_any = torch.zeros(8, dtype=torch.bool)
        for _ in range(6):                                             # horizon 20 steps = 4 decisions (staggered)
            prev = env.dyn.params().clone()
            _, _, done, _ = env.step(None, targets=env.ws.q[:, None].expand(8, JE.STEPS, 6).clone())
            if bool(done.any()):
                now = env.dyn.params()
                self.assertTrue(torch.equal(now[~done], prev[~done]))   # only the reset worlds draw again
                self.assertTrue(torch.equal(env.dyn.hold[done], env.ws.q[done].to(torch.float64)))
            done_any |= done
        self.assertTrue(bool(done_any.any()))
        self.assertFalse(torch.equal(env.dyn.params()[done_any], p0[done_any]))


@unittest.skipUnless(HAVE_MUJOCO and HAVE_KIN, "mujoco / development tasks missing")
class TestMujocoOff(unittest.TestCase):
    def env(self, t, d_):
        from nedm.so101_policy import joint_teacher as JT
        from nedm.so101_policy.vision.demos import start_state
        B = len(t["seeds"])
        e = JT.make_env("mujoco", B, JE.Codec(np.zeros(18), np.ones(18)), "cpu", dyn=d_)
        mj = dict(q=t["start_q"], cube_pos=t["start_cube_pos"], cube_quat=t["start_cube_quat"])
        e.reset_tasks(torch.arange(B), start_state(t, "cpu"), torch.as_tensor(t["goal_xy"]),
                      torch.as_tensor(t["goal_yaw"]), torch.as_tensor(t["seeds"]), mj, check_tol=None)
        return e

    def test_mujoco_off_p0_neutral_and_fixed(self):
        t = tasks(2)
        e0 = self.env(t, False)
        ref = roll(e0, 6, step=head_step)
        e0.close()
        for d_ in (dict(TRAIN, p=0.0), NEUTRAL):
            e = self.env(t, d_)
            self.assertTrue(torch.equal(roll(e, 6), ref), msg=str(d_))
            e.close()
        e = self.env(t, dict(d=2, s=0.2, tau=0.1, sag_deg=1.7))      # a fixed evaluation setting acts
        self.assertFalse(torch.equal(roll(e, 6)[..., :6], ref[..., :6]))
        e.close()

    def test_mujoco_sag_and_integral_correction(self):
        """A held command under gravity: the sag stays without the integral correction, the correction removes it."""
        t = tasks(2)
        err = {}
        for tag, d_ in (("sag", dict(sag_deg=0.8)), ("sag_i", dict(sag_deg=0.8, i_time_s=0.2))):
            e = self.env(t, d_)
            tg = e.ws.q.to(torch.float64)[:, None].expand(2, JE.STEPS, 6).clone()
            for _ in range(30):                                          # 3 s at a constant command
                e.step(None, targets=tg)
            err[tag] = torch.rad2deg(e.ws.q.to(torch.float64)[:, 1:3] - tg[:, 0, 1:3]).abs().max().item()
            e.close()
        self.assertGreater(err["sag"], 0.6)                              # measured = command + sag (0.8 deg)
        self.assertLess(err["sag_i"], 0.2)


# ---- the online loop's options -------------------------------------------------------------------------------------
class TestRobotQd(unittest.TestCase):
    def test_quantisation_zeros_and_ramp(self):
        from nedm.so101_policy.vision import viral as VI
        self.assertAlmostEqual(VI.QD_TICK, 2 * math.pi / 4096, places=15)
        q = torch.tensor([[0.3, -1.2, 0.0, 2.0, 1e-4, float("nan")]])
        k = VI.qd_ticks(q)
        self.assertTrue(torch.equal(k, torch.round(k)))
        self.assertLessEqual(float((k[0, :5] * VI.QD_TICK - q[0, :5]).abs().max()), VI.QD_TICK / 2 + 1e-15)
        self.assertEqual(float(k[0, 5]), 0.0)                             # non-finite -> 0
        for W in (2, 3, 4):
            w = VI.qd_weights(W)
            self.assertAlmostEqual(float(w.sum()), 0.0, places=12)
            t = -torch.arange(W, dtype=torch.float64) * 0.02
            self.assertAlmostEqual(float((w * t).sum()), 1.0, places=12)  # the slope of q = t is 1
        w3 = VI.qd_weights(3)
        np.testing.assert_allclose(w3.numpy(), [25.0, 0.0, -25.0], atol=1e-12)      # (q0 - q2) / 40 ms
        # still arm with sub-tick motion: exactly 0 (the robot reads 0 in 68-85 % of still queries)
        base = torch.tensor([0.3, -1.2, 0.5, 2.0, 0.1, 0.4], dtype=torch.float64)
        ticks = torch.stack([VI.qd_ticks(base + 1e-5 * i) for i in range(3)], 0)[None]
        self.assertEqual(VI.robot_qd(ticks, w3).abs().max().item(), 0.0)
        # a ramp of v rad/s: the slope within one tick per 40 ms
        v = 0.17
        hist = torch.stack([VI.qd_ticks(base - v * 0.02 * i) for i in range(3)], 0)[None]
        got = VI.robot_qd(hist, w3)[0]
        self.assertLess(float((got - v).abs().max()), VI.QD_TICK / 0.04 + 1e-12)
        self.assertTrue(torch.allclose(got / (VI.QD_TICK / 0.04), torch.round(got / (VI.QD_TICK / 0.04))))
        with self.assertRaises(ValueError):
            VI.qd_weights(1)

    def test_online_reset_input_and_hook(self):
        from nedm.so101_policy.vision import viral as VI
        N, W = 4000, 3
        q = torch.randn(N, 6, dtype=torch.float64)
        st = types.SimpleNamespace(o=VI.OnlineConfig(qd_robot_p=0.5), dev=torch.device("cpu"),
                                   env=types.SimpleNamespace(ws=types.SimpleNamespace(q=q)),
                                   qd_rob=torch.zeros(N, dtype=torch.bool), lag_hist=None,
                                   qd_hist=torch.zeros(N, W, 6, dtype=torch.float64), qd_w=VI.qd_weights(W),
                                   qd_gen=torch.Generator().manual_seed(0))
        st._lag_hook = lambda e, j: None
        VI.OnlineDAgger._qd_reset(st, torch.arange(N))
        self.assertAlmostEqual(float(st.qd_rob.float().mean()), 0.5, delta=0.03)
        sim_qd = torch.randn(N, 6)
        got = VI.OnlineDAgger.qd_input(st, sim_qd)
        self.assertEqual(float(got[st.qd_rob].abs().max()), 0.0)          # at the start: still -> exactly 0
        self.assertTrue(torch.equal(got[~st.qd_rob], sim_qd[~st.qd_rob]))
        e = types.SimpleNamespace(ws=types.SimpleNamespace(q=q + 0.17 * 0.02))        # one step of 0.17 rad/s
        VI.OnlineDAgger._step_hook(st, e, 0)
        e2 = types.SimpleNamespace(ws=types.SimpleNamespace(q=q + 2 * 0.17 * 0.02))
        VI.OnlineDAgger._step_hook(st, e2, 1)
        got = VI.OnlineDAgger.qd_input(st, sim_qd)[st.qd_rob]
        self.assertLess(float((got - 0.17).abs().max()), VI.QD_TICK / 0.04 + 1e-6)
        keep = st.qd_hist.clone()
        VI.OnlineDAgger._qd_reset(st, torch.tensor([0, 1]))              # only the reset worlds restart
        self.assertTrue(torch.equal(st.qd_hist[2:], keep[2:]))
        self.assertTrue(bool((st.qd_hist[:2] == st.qd_hist[:2, :1]).all()))
        off = types.SimpleNamespace(qd_hist=None)
        self.assertIs(VI.OnlineDAgger.qd_input(off, sim_qd), sim_qd)     # off: the sim qdot object itself


    def test_actor_robot_qd_for_evaluation(self):
        from nedm.so101_policy.vision import viral as VI

        class Views:
            fx = None

            def draw(self, n):
                return None

            def render(self, q, cp, cq, gp, dr=None, light=None):
                return torch.zeros(len(q), 2, 8, 8, 3, dtype=torch.uint8)

        class Env:                                                     # q += 0.004 rad per 20 ms step (0.2 rad/s)
            def __init__(self, B):
                self.ws = types.SimpleNamespace(q=torch.full((B, 6), 0.3, dtype=torch.float64),
                                                qd=torch.full((B, 6), 7.0, dtype=torch.float64),
                                                cube_pos=torch.zeros(B, 3, dtype=torch.float64),
                                                cube_quat=torch.tensor([[1.0, 0, 0, 0]] * B, dtype=torch.float64))

            def step(self, actions=None, targets=None, use_targets=None, step_hook=None):
                for j in range(JE.STEPS):
                    self.ws = types.SimpleNamespace(**{**vars(self.ws), "q": self.ws.q + 0.004})
                    if step_hook is not None:
                        step_hook(self, j)

        cfg = VI.ViralConfig(res=8, keypoints=2, pool_dim=4, hidden=(8,), amp=False, lag=0,
                             cameras=("external", "wrist"))
        st = VI.ViralNet(cfg, dict(robot_mean=torch.zeros(12), robot_std=torch.ones(12), tgt_mean=torch.zeros(10, 6),
                                   tgt_std=torch.ones(10, 6), aux_mean=torch.zeros(7), aux_std=torch.ones(7)),
                         pretrained=False).eval()
        seen = []
        st.predict = lambda img, robot: (seen.append(robot.clone()), torch.zeros(len(robot), 10, 6))[1]
        self.assertFalse(VI.Actor(st, Views(), device="cpu").needs_tap)
        actor = VI.Actor(st, Views(), device="cpu", qd_window=3)
        self.assertTrue(actor.needs_tap)
        actor.reset(2)
        env, goal = Env(2), torch.zeros(2, 3)
        with VI.step_tap(env, actor.push):
            actor(env.ws, goal, 0)
            env.step()
            actor(env.ws, goal, 1)
        self.assertEqual(float(seen[0][:, 6:].abs().max()), 0.0)          # the first decision: at rest
        self.assertLess(float((seen[1][:, 6:] - 0.2).abs().max()), VI.QD_TICK / 0.04 + 1e-6)   # not the sim's 7.0
        self.assertEqual(len(actor.qd_log), 2)                            # the input per decision (eval rows)
        self.assertTrue(torch.equal(torch.stack(actor.qd_log), torch.stack([x[:, 6:] for x in seen])))
        lost = VI.Actor(st, Views(), device="cpu", qd_window=3)          # no tap: the second decision raises
        lost.reset(2)
        env2 = Env(2)
        lost(env2.ws, goal, 0)
        env2.step()
        with self.assertRaises(RuntimeError):
            lost(env2.ws, goal, 1)
        seen.clear()
        off = VI.Actor(st, Views(), device="cpu")
        off.reset(2)
        off(env.ws, goal, 0)
        self.assertEqual(float(seen[-1][0, 6]), 7.0)                      # off: the sim qdot


class TestLagRange6(unittest.TestCase):
    def test_config_history_and_round_rows(self):
        from nedm.so101_policy.vision import viral as VI
        cfg = VI.ViralConfig.from_dict(dict(lag=2, lag_range=[1, 6]))
        self.assertEqual((cfg.lag_range, cfg.lag_max), ((1, 6), 6))
        h = VI.LagHistory(2, cfg.lag_max, "cpu")
        z = torch.zeros(2, 6)
        cp, cq = torch.zeros(2, 3), torch.tensor([[1.0, 0, 0, 0]] * 2)
        h.fill(torch.arange(2), z, cp, cq)
        for s in range(1, 9):
            h.push(z + s, cp, cq)
        self.assertEqual(h.get(torch.tensor([6, 1]))[0][:, 0].tolist(), [2.0, 7.0])
        # round files: k > 4 reaches the decision before; before the episode start: the start state
        B, D, S = 2, 3, JE.STEPS
        sub = np.zeros((B, D, S, 13), np.float32)
        sub[..., 0] = np.arange(1, D * S + 1, dtype=np.float32).reshape(1, D, S)
        sub[..., 9] = 1.0
        lengths = [3, 2]
        rb = np.repeat(np.arange(B), lengths)
        rt = np.concatenate([np.arange(n) for n in lengths])
        qq = np.zeros((len(rb), 6), np.float32)
        qq[:, 0] = rt * S
        r = dict(row_ep=rb, row_t=rt, row_q=qq, row_cube_pos=np.zeros((len(rb), 3), np.float32),
                 row_cube_quat=np.tile([1.0, 0, 0, 0], (len(rb), 1)).astype(np.float32))
        l6, l4 = VI.lag_rows(r, sub, 6), VI.lag_rows(r, sub, 4)
        self.assertEqual(l6.shape, (len(rb), 6, 13))
        self.assertTrue(np.array_equal(l6[:, :4], l4))                   # the old columns unchanged
        for i in range(len(rb)):
            for k in range(1, 7):
                self.assertEqual(float(l6[i, k - 1, 0]), float(max(0, rt[i] * S - k)))
        actor = VI.Actor(types.SimpleNamespace(cfg=types.SimpleNamespace(jaw_lead=0, lag=2)), None, device="cpu",
                         lag_range=(1, 6))
        self.assertEqual(actor.kmax, 6)


class TestOnlineOptions(unittest.TestCase):
    def test_defaults_are_off_and_the_env_config_is_the_old_one(self):
        from nedm.so101_policy.vision import viral as VI
        o = VI.OnlineConfig.from_dict({})
        self.assertEqual((o.dyn, o.qd_robot_p, o.qd_window, o.horizon, o.hold_cap_s, o.hold_obs_s, o.qd_noise_zero_p,
                          o.lag_range), ({}, 0.0, 3, 400, 3.5, 0.0, None, ()))
        self.assertEqual(VI.OnlineConfig.from_dict(dict(lag_range=[1, 6])).lag_range, (1, 6))
        for bad in ([3, 1], [1, 2, 3], [-1, 2]):
            with self.assertRaises(ValueError, msg=str(bad)):
                VI.OnlineConfig.from_dict(dict(lag_range=bad))
        self.assertEqual(JE.hold_cap_kw(o.hold_cap_s, o.hold_obs_s), {})
        old = JE.JointEnvConfig(world="nrd", num_envs=8, device="cpu", ckpts=["x.pt"], auto_reset=True, stagger=True,
                                starts="", seed=3)                       # OnlineDAgger's env config before the change
        new = JE.JointEnvConfig(world="nrd", num_envs=8, device="cpu", ckpts=["x.pt"], auto_reset=True, stagger=True,
                                starts="", seed=3, horizon=int(o.horizon), dyn=o.dyn or None,
                                **JE.hold_cap_kw(o.hold_cap_s, o.hold_obs_s))
        self.assertEqual(old, new)
        self.assertIsNone(VI.OnlineDAgger.qd_noise_scale(types.SimpleNamespace(o=o), 4))

    def test_new_config_file(self):
        from nedm.so101_policy.vision import viral as VI
        p = Path(__file__).resolve().parents[2] / "configs" / "so101_policy" / "vision" / "viral_r2s_online_dyn_v1.json"
        d = json.loads(p.read_text())
        o = VI.OnlineConfig.from_dict(d)
        c = JE.RobotDynConfig.from_dict(o.dyn)
        self.assertEqual((c.p, c.d_lo, c.d_hi, c.jitter, c.tau_lo, c.tau_hi), (0.8, 0, 3, 0.4, 0.02, 0.2))
        self.assertEqual((c.i_time_s, c.i_max_deg, c.i_e_max_deg, c.i_v_min), (0.2, 3.0, 1.0, 0.02))   # robot guard
        self.assertEqual(c.noise_max, 0.0)                               # no command noise on the robot
        self.assertGreaterEqual(c.s_lo, 0.2)                             # a gain below 0.2 cannot lift the cube
        self.assertLessEqual(c.sag_max_deg, 2.0)
        self.assertEqual((o.horizon, o.hold_cap_s, o.qd_robot_p, o.qd_noise_zero_p, o.iterations, o.lag_range),
                         (1500, 20.0, 0.5, 0.3, 6000, (1, 6)))
        base = json.loads((p.parent / "viral_r2s_online_v1.json").read_text())
        for k, v in base.items():
            if k not in ("note", "iterations"):
                self.assertEqual(d[k], v, msg=k)

    @unittest.skipUnless(HAVE_KIN, "kinematics files / development tasks missing")
    def test_hold_cap_and_observation_scale(self):
        t = tasks(2)
        self.assertEqual(JE.hold_cap_kw(20.0), dict(task=dict(hold_cap_s=20.0), hold_obs_s=3.5))
        self.assertEqual(JE.hold_cap_kw(None, 5.0), dict(hold_obs_s=5.0))
        e0 = kin_env(t)
        e1 = kin_env(t, **dict(hold_cap_s=20.0))                         # the JT's scale kept by default
        e2 = kin_env(t, **dict(hold_cap_s=20.0, hold_obs_s=20.0))
        self.assertEqual((e0.tcfg.hold_cap_s, e1.tcfg.hold_cap_s, e2.tcfg.hold_cap_s), (3.5, 20.0, 20.0))
        self.assertIsNone(e0.obs_tcfg)
        for e in (e0, e1, e2):
            e.task.tr.air_run[:] = 50                                    # 1 s held in the air
        o0, o1, o2 = (e._observe() for e in (e0, e1, e2))
        self.assertAlmostEqual(float(o0[0, -1]), 1.0 / 3.5, places=6)    # the held-time input (last column)
        self.assertTrue(torch.equal(o1, o0))                             # hold_obs_s 3.5: the teacher's old scale
        self.assertAlmostEqual(float(o2[0, -1]), 1.0 / 20.0, places=6)
        self.assertTrue(torch.equal(o0[:, :-1], o2[:, :-1]))

    def test_output_record_of_the_robot_setting(self):
        spec = importlib.util.spec_from_file_location(
            "vision_viral_script", Path(__file__).resolve().parents[2] / "scripts/so101_policy/vision_viral.py")
        S = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S)
        from nedm.so101_policy import task as T
        stub = lambda d_, cap=3.5, obs=0.0: types.SimpleNamespace(                     # noqa: E731
            dyn=None if d_ is None else JE.RobotDyn(JE.RobotDynConfig.from_dict(d_), 2, "cpu", 0),
            tcfg=T.TaskConfig(hold_cap_s=cap), cfg=types.SimpleNamespace(hold_obs_s=obs))
        self.assertIsNone(S.robot_rec(stub(None)))                       # all defaults: no record (old files)
        r = json.loads(S.robot_rec(stub(dict(d=2, i_time_s=0.2), 20.0, 3.5)))
        self.assertEqual((r["cfg"]["d_hi"], r["cfg"]["i_time_s"], r["hold_cap_s"], r["hold_obs_s"], r["kp_scale"]),
                         (2, 0.2, 20.0, 3.5, 1.0))
        self.assertIsNone(json.loads(S.robot_rec(stub(None, 20.0)))["cfg"])
        with mock.patch.object(JE, "EVAL_KP_SCALE", 0.5):                 # the servo scale alone is recorded too
            self.assertEqual(json.loads(S.robot_rec(stub(None)))["kp_scale"], 0.5)

    def test_qd_noise_scale(self):
        from nedm.so101_policy.vision import viral as VI
        st = types.SimpleNamespace(o=VI.OnlineConfig(qd_noise_zero_p=0.3), dev=torch.device("cpu"),
                                   gen=torch.Generator().manual_seed(0))
        s = VI.OnlineDAgger.qd_noise_scale(st, 20000)
        self.assertAlmostEqual(float((s == 0).float().mean()), 0.3, delta=0.015)
        nz = s[s > 0]
        self.assertTrue(float(nz.max()) < 1.0)
        self.assertAlmostEqual(float(nz.mean()), 0.5, delta=0.01)
        # the network: scale None = the old noise (same RNG), scale 0 = no noise
        net = types.SimpleNamespace(cfg=types.SimpleNamespace(hist=0, qd_noise=0.3), training=True,
                                    robot_mean=torch.zeros(12), robot_std=torch.ones(12) * 2)
        x = torch.randn(5, 12)
        torch.manual_seed(1)
        a = VI.ViralNet.robot(net, x)
        torch.manual_seed(1)
        r = (x.view(-1, 1, 12) / 2).clamp(-10, 10)
        old = torch.cat((r[..., :6], r[..., 6:] + 0.3 * torch.randn_like(r[..., 6:])), -1).reshape(-1, 12)
        self.assertTrue(torch.equal(a, old))
        torch.manual_seed(1)
        self.assertTrue(torch.equal(VI.ViralNet.robot(net, x, None), old))
        b = VI.ViralNet.robot(net, x, torch.zeros(5))
        self.assertTrue(torch.equal(b, (x / 2).clamp(-10, 10)))
        torch.manual_seed(1)
        c = VI.ViralNet.robot(net, x, torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]))
        self.assertTrue(torch.equal(c[[0, 2, 4]], old[[0, 2, 4]]) and torch.equal(c[[1, 3]], (x[[1, 3]] / 2)))

    def test_loss_with_scale_runs(self):
        from nedm.so101_policy.vision import viral as VI
        stats = dict(robot_mean=torch.zeros(12), robot_std=torch.ones(12), tgt_mean=torch.zeros(10, 6),
                     tgt_std=torch.ones(10, 6), aux_mean=torch.zeros(VI.AUX_DIM), aux_std=torch.ones(VI.AUX_DIM))
        cfg = VI.ViralConfig(res=32, keypoints=4, pool_dim=8, hidden=(16, 16), amp=False, qd_noise=0.3)
        st = VI.ViralNet(cfg, stats, pretrained=False).train()
        img, robot = torch.randn(3, 2, 3, 32, 32), torch.randn(3, 12)
        tgt, valid = torch.randn(3, 10, 6), torch.ones(3, 10)
        torch.manual_seed(2)
        l0, _ = st.loss(img, robot, tgt, valid)
        torch.manual_seed(2)
        l1, _ = st.loss(img, robot, tgt, valid, qd_scale=None)
        self.assertTrue(torch.equal(l0, l1))
        l2, _ = st.loss(img, robot, tgt, valid, qd_scale=torch.zeros(3))
        self.assertTrue(bool(torch.isfinite(l2)))


# ---- the fitted real servo (RobotDynConfig.servo, RealServo) ---------------------------------------------------------
# the evaluation setting of /tmp/claude_servo_fit/servo_fit_params.json ("proposed") and its training randomisation
EVAL_SERVO = dict(delay_ms=[25, 25, 20, 25, 20, 45], tau_ms=[50, 50, 45, 40, 40, 45],
                  deadband_deg=[0.35, 0.30, 0.74, 0.37, 0.44, 0.37], deadband_per_Nm_deg=[0.0, 1.7, 1.3, 0.8, 0.0, 0.0],
                  breakaway_extra_deg=[0.14, 0.0, 0.02, 0.09, 0.09, 0.29],
                  compliance_rad_per_Nm=[0.0, 0.060, 0.055, 0.055, 0.0, 0.0],
                  bias_deg=[-0.07, -0.22, 0.18, 0.0, -0.07, 0.03], vmax_rad_s=[3.0, 2.3, 3.0, 3.0, 1.0, 2.2],
                  cube_mass_kg=0.03)
EVAL_DYN = dict(d=2, jitter=0.4, i_time_s=0.2, servo=EVAL_SERVO)
TRAIN_SERVO = dict(delay_ms=[[10, 40]] * 5 + [[30, 55]], tau_ms=[[30, 75]] * 5 + [[35, 75]],
                   deadband_deg=[[0.25, 0.5], [0.2, 0.5], [0.55, 0.95], [0.25, 0.5], [0.3, 0.55], [0.25, 0.5]],
                   deadband_per_Nm_deg=[0, [1.4, 2.0], [0.8, 1.6], [0.0, 1.5], 0, 0],
                   breakaway_extra_deg=[[0.0, 0.3]] * 5 + [[0.1, 0.5]],
                   compliance_rad_per_Nm=[0, [0.045, 0.075], [0.045, 0.075], [0.04, 0.075], 0, 0],
                   cube_mass_kg=[0.01, 0.05], bias_deg=[[-0.35, 0.35]] * 6,
                   vmax_rad_s=[[1.0, 3.0]] * 4 + [[0.5, 1.5], [1.5, 3.5]])
REF_EMU = Path("/tmp/claude_servo_fit/servo_emu.py")
MENAGERIE = Path(__file__).resolve().parents[2] / "contact_nrd/so101_grasp/vendor/mujoco_menagerie" / \
    "robotstudio_so101/so101.xml"
_RW_BIAS = torch.tensor([[-0.08956, -0.41704, -0.35361, -0.01302, -0.12852, -0.00093],
                         [0.0, 0.0, -0.35361, -0.01302, -0.12852, -0.00093],
                         [0.0, 0.0, 0.0, 0.0, -0.12852, -0.00093]], dtype=torch.float64)
_RW_CUBE = torch.tensor([[-0.27491, -1.10161, -1.32276, -0.05156, -1.50263, -0.01843],
                         [0.0, 0.0, -1.32276, -0.05156, -1.50263, -0.01843],
                         [0.0, 0.0, 0.0, 0.0, -1.50263, -0.01843]], dtype=torch.float64)


def ref_servo(p, q0, cmds, qws, held, n_sub=10, nh=5, reanchor_deg=3.0):
    """Frozen copy of the reference emulator (/tmp/claude_servo_fit/servo_emu.py ``RealServoEmu.step``, 2026-10-22)
    plus its sim.py's load-dependent dead band db + db1 |G|. p {delay, tau, db, db1, dbs, kg, kh, b, vmax} [B, 6] (SI),
    q0 [B, 6], cmds / qws [N, B, 6], held [N, B] bool -> outputs [N, B, 6]."""
    h = 0.02 / n_sub
    y, mov = q0.clone(), torch.zeros(q0.shape, dtype=torch.bool)
    hist = q0[:, None, :].repeat(1, nh, 1)
    out = []
    for cmd, qw, hd in zip(cmds, qws, held):
        hist = torch.cat((cmd[:, None], hist[:, :-1]), 1)
        y = torch.where((y - qw).abs() > math.radians(reanchor_deg), qw, y)
        a1 = qw[:, 1]
        a2 = a1 + qw[:, 2]
        a3 = a2 + qw[:, 3]
        X = torch.stack((a1.cos(), a1.sin(), a2.cos(), a2.sin(), a3.cos(), a3.sin()), -1)
        bias, cube = torch.zeros_like(qw), torch.zeros_like(qw)
        bias[:, 1:4], cube[:, 1:4] = X @ _RW_BIAS.T, X @ _RW_CUBE.T
        o = p["b"] - p["kg"] * bias - p["kh"] * cube * hd.to(torch.float64)[:, None]
        tau, dbs, vmax = p["tau"].clamp_min(1e-4), p["dbs"], p["vmax"]
        db = p["db"] + p["db1"] * bias.abs()
        vt = vmax * tau
        dsub = torch.round(p["delay"] / h).long()
        for i in range(n_sub):
            m = torch.div(dsub - i + n_sub - 1, n_sub, rounding_mode="floor").clamp(0, nh - 1)
            c = torch.gather(hist, 1, m[:, None, :]).squeeze(1)
            r = c + o
            e = r - y
            ae, sg = e.abs(), torch.sign(e)
            mv = torch.where(mov, ae > db + 1.7e-4, ae > db + dbs)
            rp = r - db * sg
            ep = (rp - y).abs()
            t1 = (ep - vt) / vmax
            q_lin = y + sg * vmax * h
            tb = t1.clamp(0.0, h)
            qb = torch.where(ep > vt, rp - sg * vt, y)
            q_exp = rp - (rp - qb) * torch.exp(-(h - tb) / tau)
            y = torch.where(mv, torch.where(t1 >= h, q_lin, q_exp), y)
            mov = mv
        out.append(y)
    return torch.stack(out)


def servo(B=4, seed=0, spec=None, q=None):
    """A RealServo of B worlds (``spec``: default the evaluation setting), reset at q (default 0), held flag fresh."""
    s = JE.RealServo(EVAL_SERVO if spec is None else spec, B, "cpu", seed)
    s.reset(torch.arange(B), torch.zeros(B, 6, dtype=torch.float64) if q is None else q)
    s.fresh[:] = False
    return s


def free_run(s, cmds, held=None):
    """The servo alone with an ideal world (world q = the servo's output): cmds [N, B, 6] -> outputs [N, B, 6]."""
    out = []
    for c in cmds:
        out.append(s.step(c, s.y.clone(), held).clone())
    return torch.stack(out)


def random_case(B, N, seed):
    """Random command sequences (random walks with held stretches and jumps), world q = y + noise with jumps of
    0-6 deg (re-anchor events), random held flags."""
    g = torch.Generator().manual_seed(seed)
    f = dict(dtype=torch.float64, generator=g)
    q0 = 0.8 * (2 * torch.rand(B, 6, **f) - 1)
    steps = torch.randn(N, B, 6, **f) * 0.01 * (torch.rand(N, B, 6, **f) < 0.5)
    steps += (torch.rand(N, B, 6, **f) < 0.01) * 0.2 * torch.randn(N, B, 6, **f)
    cmds = q0[None] + steps.cumsum(0)
    noise = torch.randn(N, B, 6, **f) * 0.002 + (torch.rand(N, B, 6, **f) < 0.02) * math.radians(6) * \
        (2 * torch.rand(N, B, 6, **f) - 1)
    held = torch.rand(N, B, generator=g) < 0.5
    return q0, cmds, noise, held


class TestRealServo(unittest.TestCase):
    def test_equals_the_reference_on_random_commands(self):
        for spec in (TRAIN_SERVO, EVAL_SERVO, dict(TRAIN_SERVO, deadband_per_Nm_deg=0.0, n_sub=4)):
            n = spec.get("n_sub", 10)
            q0, cmds, noise, held = random_case(48, 400, 1)
            for rule, hd in ((False, held), (True, held), (True, None)):     # held_jaw False = the reference's rule
                s = servo(48, 3, dict(spec, held_jaw=rule), q0)
                got, qws = [], []
                for k in range(400):
                    qw = s.y + noise[k]
                    qws.append(qw.clone())
                    got.append(s.step(cmds[k], qw, None if hd is None else hd[k]).clone())
                want = ref_servo({k: v for k, v in s.p.items()}, q0, cmds, torch.stack(qws),
                                 torch.zeros(400, 48, dtype=torch.bool) if hd is None else hd, n_sub=n, nh=s.nh)
                d = (torch.stack(got) - want).abs()
                msg = f"{str(spec)[:60]} held_jaw {rule} held {hd is not None}"
                self.assertEqual(float(d[..., :5].max()), 0.0, msg=msg)     # the arm: always the reference
                if rule and hd is not None:                                 # the held-cube jaw rule acts
                    self.assertGreater(float(d[..., 5].max()), 1e-3, msg=msg)
                else:
                    self.assertEqual(float(d[..., 5].max()), 0.0, msg=msg)
                self.assertGreater(float((torch.stack(qws) - torch.stack(got)).abs().max()), math.radians(3))
            if spec is TRAIN_SERVO:
                self.assertTrue(bool((s.p["db1"][:, 1:3] > 0).all()))  # the load term is in the test

    @unittest.skipUnless(REF_EMU.exists(), "reference emulator /tmp/claude_servo_fit/servo_emu.py missing")
    def test_equals_the_reference_file(self):
        """The reference file itself (no load-dependent dead band: db1 0)."""
        spec = importlib.util.spec_from_file_location("servo_emu_ref", REF_EMU)
        SE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(SE)
        B, N = 48, 400
        q0, cmds, noise, held = random_case(B, N, 2)
        s = servo(B, 5, dict(TRAIN_SERVO, deadband_per_Nm_deg=0.0, held_jaw=False), q0)
        R = SE.RealServoEmu(B, "cpu", n_sub=10)
        R.set_params(list(range(B)), {k: s.p[k] for k in SE.KEYS})
        R.reset(list(range(B)), q0)
        mx = 0.0
        for k in range(N):
            qw = s.y + noise[k]
            mx = max(mx, float((s.step(cmds[k], qw, held[k]) - R.step(cmds[k], qw, held[k])).abs().max()))
        self.assertEqual(mx, 0.0)

    @unittest.skipUnless(HAVE_MUJOCO and MENAGERIE.exists(), "mujoco / the Menagerie SO101 model missing")
    def test_gravity_closed_form_vs_mujoco(self):
        import mujoco
        from nedm.so101_policy.kin import GRASP_POINT
        m = mujoco.MjModel.from_xml_path(str(MENAGERIE))
        d = mujoco.MjData(m)
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        rng = np.random.default_rng(0)
        lo = np.array([-1.9, -1.74, -1.69, -1.65, -2.7, 0.0])
        hi = np.array([1.9, 1.74, 1.69, 1.65, 2.7, 1.7])
        Q = lo + (hi - lo) * rng.random((300, 6))
        jac = np.zeros((3, m.nv))
        bias, cube = np.zeros((300, 6)), np.zeros((300, 6))
        for k, q in enumerate(Q):
            d.qpos[:6], d.qvel[:] = q, 0.0
            mujoco.mj_forward(m, d)
            bias[k] = d.qfrc_bias[:6]
            p = d.xpos[gid] + d.xmat[gid].reshape(3, 3) @ np.asarray(GRASP_POINT)
            mujoco.mj_jac(m, d, jac, None, p, gid)
            cube[k] = jac[:, :6].T @ np.array([0.0, 0.0, 9.81])
        w = (torch.tensor(JE.SERVO_W_BIAS, dtype=torch.float64), torch.tensor(JE.SERVO_W_CUBE, dtype=torch.float64))
        G, Gc = JE.servo_gravity(torch.as_tensor(Q), w)
        rb = np.sqrt(((G.numpy() - bias)[:, 1:4] ** 2).mean(0))
        rc = np.sqrt(((Gc.numpy() - cube)[:, 1:4] ** 2).mean(0))
        self.assertLess(float(rb.max()), 0.01)                          # N m (the fit: 0.004)
        self.assertLess(float(rc.max()), 0.05)                          # N m / kg (x 0.05 kg: < 2.5 mN m)
        self.assertEqual(float(G[:, [0, 4, 5]].abs().max()), 0.0)

    def test_dead_band_delay_lag_and_speed_limit(self):
        D = math.radians
        spec = dict(delay_ms=26, tau_ms=50, deadband_deg=0.4, breakaway_extra_deg=0.1)
        s = servo(1, spec=spec)
        z = torch.zeros(1, 6, dtype=torch.float64)
        small = torch.full((1, 6), D(0.45), dtype=torch.float64)        # < dead band + break-away (0.5 deg)
        out = free_run(s, small[None].expand(50, 1, 6))
        self.assertTrue(torch.equal(out, z[None].expand(50, 1, 6)))     # holds still, exactly
        big = torch.full((1, 6), D(2.0), dtype=torch.float64)
        out = free_run(s, big[None].expand(60, 1, 6))
        self.assertTrue(torch.equal(out[0], z))                         # 26 ms delay: the first step is still
        self.assertGreater(float(out[1].min()), 0.0)
        settle = big - D(0.4)                                           # stops at the dead band's edge: 0.4 deg short
        self.assertLess(float((out[-1] - settle).abs().max()), JE.SERVO_EPS + 1e-12)
        self.assertLessEqual(float((out[-1] - settle).max()), 1e-12)
        t = (torch.arange(1, 7, dtype=torch.float64) * 0.02 - 0.026).clamp(min=0.0)       # first-order lag
        np.testing.assert_allclose(out[:6, 0, 0].numpy(), (D(1.6) * (1 - torch.exp(-t / 0.05))).numpy(), atol=1e-12)
        rest = out[-1].clone()
        for dc in (0.08, -0.3, -0.75):                                  # small changes of the command: no motion
            o = free_run(s, (big + D(dc))[None].expand(20, 1, 6))
            self.assertTrue(torch.equal(o[-1], rest), msg=str(dc))
        o = free_run(s, (big + D(0.3))[None].expand(40, 1, 6))          # 0.3 + 0.4 > 0.5: it moves (to 0.4 short)
        self.assertLess(float((o[-1] - (big + D(0.3) - D(0.4))).abs().max()), JE.SERVO_EPS + 1e-12)
        v = servo(1, spec=dict(spec, vmax_rad_s=1.0))                   # speed limit: <= vmax DT per step
        o = free_run(v, torch.full((30, 1, 6), D(30.0), dtype=torch.float64))
        self.assertLessEqual(float(o.diff(dim=0).abs().max()), 1.0 * JE.CONTROL_DT + 1e-12)
        self.assertGreater(float(o.diff(dim=0).abs().max()), 0.99 * JE.CONTROL_DT)

    def test_gravity_sag_and_held_cube(self):
        q = torch.tensor([[0.0, 0.3, 0.4, 0.5, 0.0, 0.2]], dtype=torch.float64)
        w = (torch.tensor(JE.SERVO_W_BIAS, dtype=torch.float64), torch.tensor(JE.SERVO_W_CUBE, dtype=torch.float64))
        G, Gc = JE.servo_gravity(q, w)
        spec = dict(delay_ms=0, tau_ms=40, deadband_deg=0.0, compliance_rad_per_Nm=[0, 0.06, 0.055, 0.055, 0, 0],
                    cube_mass_kg=0.03, bias_deg=0.0)
        kg = torch.tensor([[0, 0.06, 0.055, 0.055, 0, 0]], dtype=torch.float64)
        for held in (False, True):
            s = servo(1, spec=spec, q=q)
            for _ in range(100):                                        # a constant command at q, the world at q
                y = s.step(q, q, torch.tensor([held]))
            want = q - kg * (G + 0.03 * Gc * held)                      # settles at command - compliance x torque
            self.assertLess(float((y - want).abs().max()), 1e-9, msg=str(held))
        self.assertGreater(float(G[0, 1:3].abs().min()), 0.1)           # a loaded pose (N m)
        # the dead band grows with the load: db + db1 |G| (shoulder_lift: 0.3 + 1.7 |G| deg)
        s = servo(1, spec=dict(spec, deadband_deg=0.3, deadband_per_Nm_deg=[0, 1.7, 0, 0, 0, 0],
                               compliance_rad_per_Nm=0.0), q=q)
        c = q.clone()
        c[0, 1] += math.radians(0.3 + 1.7 * abs(float(G[0, 1]))) * 0.98
        y = free_run(s, c[None].expand(20, 1, 6))[-1]
        self.assertTrue(torch.equal(y, q))                              # inside the load-grown dead band: still

    def test_held_cube_jaw(self):
        """A held cube blocks the jaw at 15 deg (the world: max(target, 15 deg)), the command closes to 5 deg, then
        opens to 40 deg: with ``held_jaw`` the world's jaw target is the settle point (command + bias) once the delayed
        command acts, the emulated jaw stays at the held angle, and the release opens from there; the reference's
        target is only 2.5 / 4.9 deg below the held jaw, and it opens later."""
        D = math.radians
        blk = D(15.0)

        def run(rule, close=5.0, n_close=30, n_open=12, after=40.0):
            q0 = torch.tensor([[0.0, 0.3, 0.4, 0.5, 0.0, D(20.0)]], dtype=torch.float64)
            s = servo(1, spec=dict(EVAL_SERVO, held_jaw=rule), q=q0)
            qw = s.y.clone()
            out, st = [], []
            for k in range(n_close + n_open):
                c = q0.clone()
                c[0, 5] = D(close) if k < n_close else D(after)
                held = torch.tensor([bool(qw[0, 5] <= blk + 1e-12)])
                o = s.step(c, qw, held)
                out.append(float(o[0, 5]))
                st.append(float(s.y[0, 5]))
                qw = o.clone()
                qw[0, 5] = max(float(o[0, 5]), blk)                     # the stiff world, the cube blocks the jaw
            return np.degrees(out), np.degrees(st), s

        out, st, s = run(True)
        held = np.flatnonzero(out < 15.0)
        sq = out[8:30]                                                  # squeezing: the 45 ms delayed 5 deg command
        np.testing.assert_allclose(sq, 5.0 + 0.03, atol=1e-9)           # + the jaw bias (0.03 deg)
        self.assertTrue(np.all(st[8:30] >= 15.0 - 2.53))                # the emulated jaw: one step from the held angle
        ref, ref_st, _ = run(False)
        g = ref[8:30] - 15.0
        self.assertTrue(np.all((g < -2.4) & (g > -5.0)), msg=str(g))    # the reference: 2.5 / 4.9 deg below
        self.assertGreater(np.ptp(g), 2.0)                              # alternating
        o_new, o_ref = [], []
        for n in (29, 30):                                              # both phases of the reference's alternation
            a_, b_ = run(True, n_close=n)[0], run(False, n_close=n)[0]
            o_new.append(int(np.flatnonzero(a_[n:] > 15.2)[0]))
            o_ref.append(int(np.flatnonzero(b_[n:] > 15.2)[0]))
            self.assertLessEqual(a_[n + o_new[-1]] - 15.0, 2.2 * JE.CONTROL_DT * 180 / math.pi + 1e-9)
        self.assertEqual(o_new, [2, 2])                                 # opens from the held angle after the delay
        self.assertEqual(sorted(o_ref), [2, 3])                         # the reference: one step later from 4.9 deg
        self.assertGreater(len(held), 20)
        # held, then a command inside the dead band (0.3 deg below the held jaw): no squeeze, the target = the held jaw
        out, _, _ = run(True, n_close=15, n_open=15, after=14.7)
        np.testing.assert_allclose(out[15:17], 5.03, atol=1e-9)        # the delayed squeeze (step 17: the change)
        np.testing.assert_allclose(out[18:], 15.0, atol=1e-9)
        self.assertEqual(JE.servo_spec(dict(EVAL_SERVO))[2]["held_jaw"], True)    # the default
        with self.assertRaises(ValueError):
            JE.servo_spec(dict(EVAL_SERVO, held_jaw=1))

    def test_reset_reanchor_and_stale_held_flag(self):
        B = 6
        s = servo(B, spec=TRAIN_SERVO)
        cmds = torch.full((30, B, 6), 0.1, dtype=torch.float64)
        free_run(s, cmds)
        y0, h0, m0, d0 = s.y.clone(), s.hist.clone(), s.moving.clone(), s.draw.clone()
        q2 = torch.randn(2, 6, dtype=torch.float64)
        s.reset(torch.tensor([1, 4]), q2)
        keep = torch.tensor([0, 2, 3, 5])
        self.assertTrue(torch.equal(s.y[[1, 4]], q2) and not bool(s.moving[[1, 4]].any()))
        self.assertTrue(bool((s.hist[[1, 4]] == q2[:, None]).all()))
        self.assertTrue(torch.equal(s.y[keep], y0[keep]) and torch.equal(s.hist[keep], h0[keep])
                        and torch.equal(s.moving[keep], m0[keep]) and torch.equal(s.draw[keep], d0[keep]))
        self.assertFalse(torch.equal(s.draw[[1, 4]], d0[[1, 4]]))       # new draws for the reset worlds
        self.assertEqual(s.fresh.tolist(), [False, True, False, False, True, False])
        # re-anchor: a joint whose world q is > 3 deg from y restarts there, the others keep y
        a = servo(1, spec=dict(delay_ms=0, tau_ms=40, deadband_deg=0.4))
        qw = torch.zeros(1, 6, dtype=torch.float64)
        qw[0, 5], qw[0, 2] = math.radians(3.5), math.radians(2.5)
        cmd = torch.zeros(1, 6, dtype=torch.float64)
        cmd[0, 5] = math.radians(3.5)                                   # commands = y after the re-anchor: no motion
        y = a.step(cmd, qw)
        self.assertEqual(float(y[0, 5]), math.radians(3.5))             # 3.5 deg off: re-anchored to the world q
        self.assertEqual(float(y[0, 2]), 0.0)                           # 2.5 deg off: kept
        # the held flag of a reset world is the old episode's: ignored for one step
        q = torch.tensor([[0.0, 0.3, 0.4, 0.5, 0.0, 0.2]] * 2, dtype=torch.float64)
        spec = dict(delay_ms=0, tau_ms=40, deadband_deg=0.0, compliance_rad_per_Nm=[0, 0.06, 0.055, 0.055, 0, 0],
                    cube_mass_kg=0.03)
        b = JE.RealServo(spec, 2, "cpu", 0)
        b.reset(torch.arange(2), q)
        hd = torch.tensor([True, False])
        o1 = b.step(q, q, hd)
        self.assertTrue(torch.equal(o1[0], o1[1]))                      # first step: both as not held
        o2 = b.step(q, b.y.clone(), hd)
        self.assertFalse(torch.equal(o2[0], o2[1]))

    def test_off_is_the_old_stage(self):
        self.assertEqual(JE.RobotDynConfig(servo={}), JE.RobotDynConfig())
        self.assertEqual(JE.RobotDynConfig.from_dict(dict(TRAIN, servo={})), JE.RobotDynConfig.from_dict(TRAIN))
        self.assertEqual(JE.RobotDynConfig.from_dict(dict(servo=None)), JE.RobotDynConfig())
        self.assertTrue(JE.RobotDynConfig(servo={}).neutral)
        self.assertFalse(JE.RobotDynConfig(servo=EVAL_SERVO).neutral)     # the servo alone acts
        self.assertTrue(JE.dyn_config(dict(servo={})).neutral)          # only an empty servo: the stage never runs
        g = torch.Generator().manual_seed(0)
        u = torch.randn(6, 5, 6, generator=g, dtype=torch.float64)
        q = torch.randn(6, 6, generator=g, dtype=torch.float64)
        for kw in (TRAIN, dict(d=2, jitter=0.4, s=0.5, tau=0.1, sag_deg=1.5, i_time_s=0.2)):
            a, b = dyn(6, 3, **kw), dyn(6, 3, **dict(kw, servo={}))
            self.assertIsNone(b.servo)
            a.reset(torch.arange(6), q)
            b.reset(torch.arange(6), q)
            for _ in range(3):                                          # the held flag is ignored without the servo
                a.begin(), b.begin()
                ex, oa, ob = [], [], []
                for j in range(JE.STEPS):
                    ex.append(u[:, j])
                    oa.append(a.apply(j, ex, u[:, j], q, *BIG))
                    ob.append(b.apply(j, ex, u[:, j], q, *BIG, None, torch.ones(6, dtype=torch.bool)))
                self.assertTrue(torch.equal(torch.stack(oa), torch.stack(ob)))
            self.assertEqual(tuple(b.summary()), JE.RobotDyn.SUMMARY)    # the old log columns
        cfg = JE.JointEnvConfig(dyn=EVAL_DYN)                           # serialised with the servo, round trip
        back = JE.JointEnvConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
        self.assertEqual(back.dyn, cfg.dyn)
        self.assertEqual(back.dyn.servo, EVAL_SERVO)

    @unittest.skipUnless(HAVE_KIN, "kinematics files / development tasks missing")
    def test_env_off_and_empty_servo_are_the_old_step(self):
        t = tasks(6)
        ref = roll(kin_env(t), step=head_step)
        self.assertTrue(torch.equal(roll(kin_env(t, dict(NEUTRAL, servo={}))), ref))
        a, b = roll(kin_env(t, TRAIN)), roll(kin_env(t, dict(TRAIN, servo={})))
        self.assertTrue(torch.equal(a, b))

    def test_draws_per_world_and_reproducible(self):
        B = 4000
        s = JE.RealServo(TRAIN_SERVO, B, "cpu", 11)
        s.reset(torch.arange(B), torch.zeros(B, 6, dtype=torch.float64))
        lo, hi = s.lo, s.hi
        self.assertTrue(bool(((s.draw >= lo - 1e-15) & (s.draw <= hi + 1e-15)).all()))
        K = {k: i for i, k in enumerate(JE.SERVO_KEYS)}
        dl = s.draw[:, K["delay_ms"], 0]
        self.assertAlmostEqual(float(dl.mean()), 0.025, delta=0.0006)   # uniform 10-40 ms
        lt = torch.log(s.draw[:, K["tau_ms"], 0])
        self.assertAlmostEqual(float(lt.mean()), (math.log(0.03) + math.log(0.075)) / 2, delta=0.01)   # log-uniform
        cm = s.draw[:, K["cube_mass_kg"]]
        self.assertTrue(bool((cm == cm[:, :1]).all()))                  # one cube mass per world
        self.assertGreater(len(set(cm[:, 0].tolist())), B - 5)          # independent per world
        self.assertEqual(float(s.draw[:, K["compliance_rad_per_Nm"], [0, 4, 5]].abs().max()), 0.0)   # fixed 0
        np.testing.assert_allclose(s.p["kh"].numpy(), (s.draw[:, K["compliance_rad_per_Nm"]] * cm).numpy(), atol=0)
        e = JE.RealServo(TRAIN_SERVO, B, "cpu", 11)
        e.reset(torch.arange(B), torch.zeros(B, 6, dtype=torch.float64))
        self.assertTrue(torch.equal(e.draw, s.draw))                    # same seed: same draws
        # in the stage: the servo's generator is its own, the stage's draws are those without the servo
        a = JE.RobotDyn(JE.RobotDynConfig.from_dict(dict(d_lo=0, d_hi=3, jitter=0.4, p=0.8)), 500, "cpu", 4)
        b = JE.RobotDyn(JE.RobotDynConfig.from_dict(dict(d_lo=0, d_hi=3, jitter=0.4, p=0.8, servo=TRAIN_SERVO)), 500,
                        "cpu", 4)
        for x in (a, b):
            x.reset(torch.arange(500), torch.zeros(500, 6, dtype=torch.float64))
            x.begin()
        self.assertTrue(torch.equal(a.params(), b.params()) and torch.equal(a.d, b.d))
        p0 = b.servo_params().clone()
        b.reseed(5)
        b.reset(torch.arange(500), torch.zeros(500, 6, dtype=torch.float64))
        c = JE.RobotDyn(b.cfg, 500, "cpu", 99)
        c.reseed(5)
        c.reset(torch.arange(500), torch.zeros(500, 6, dtype=torch.float64))
        self.assertTrue(torch.equal(b.servo_params(), c.servo_params()))   # reseed: reproducible
        self.assertFalse(torch.equal(b.servo_params(), p0))
        sm = b.summary()
        self.assertEqual(tuple(sm), JE.RobotDyn.SUMMARY + JE.SERVO_SUMMARY)

    def test_eval_json_parse_and_errors(self):
        js = json.dumps(EVAL_DYN)
        with mock.patch.dict(os.environ, {JE.DYN_ENV: js}):
            c = JE.dyn_from_env()
        self.assertEqual((c.d_lo, c.d_hi, c.jitter, c.i_time_s, c.s_lo, c.tau_hi), (2, 2, 0.4, 0.2, 1.0, 0.0))
        self.assertEqual(c.servo, EVAL_SERVO)
        d = JE.RobotDyn(c, 3, "cpu", 0)
        d.reset(torch.arange(3), torch.zeros(3, 6, dtype=torch.float64))
        for i, k in enumerate(JE.SERVO_KEYS):                           # fixed: every world gets the exact values
            want = np.broadcast_to(np.asarray(EVAL_SERVO[k], dtype=np.float64) * JE.SERVO_SI[k], (3, 6))
            np.testing.assert_allclose(d.servo.draw[:, i].numpy(), want, rtol=1e-15, atol=0, err_msg=k)
        np.testing.assert_allclose(d.servo.p["kh"][0].numpy(), np.asarray([0, 0.06, 0.055, 0.055, 0, 0]) * 0.03,
                                   rtol=1e-15)
        self.assertEqual(d.servo.m[0, 0].tolist(), [2, 2, 1, 2, 1, 3])   # delays 25 / 20 / 45 ms: commands 2, 1, 3 back
        bad = [dict(EVAL_DYN, s=0.5), dict(EVAL_DYN, tau=0.1), dict(EVAL_DYN, sag_deg=1.5),       # counted twice
               dict(EVAL_DYN, servo=dict(EVAL_SERVO, gain=1)), dict(EVAL_DYN, servo=dict(tau_ms=40, deadband_deg=0.3)),
               dict(EVAL_DYN, servo=dict(EVAL_SERVO, tau_ms=[40] * 5)),
               dict(EVAL_DYN, servo=dict(EVAL_SERVO, deadband_deg=[[0.5, 0.3]] * 6)),
               dict(EVAL_DYN, servo=dict(EVAL_SERVO, deadband_deg=-0.1)),
               dict(EVAL_DYN, servo=dict(EVAL_SERVO, delay_ms=300)), dict(EVAL_DYN, servo=dict(EVAL_SERVO, tau_ms=0)),
               dict(EVAL_DYN, servo=dict(EVAL_SERVO, n_sub=0)),
               dict(EVAL_DYN, servo=dict(EVAL_SERVO, cube_mass_kg=[1] * 6)),
               dict(EVAL_DYN, servo=dict(EVAL_SERVO, bias_deg=True)), dict(EVAL_DYN, servo=[1, 2])]
        for b in bad:
            with self.assertRaises(ValueError, msg=str(b)[:120]):
                JE.dyn_config(b)
        self.assertEqual(JE.dyn_config(dict(servo=dict(EVAL_SERVO, note="x", n_sub=5))).servo["n_sub"], 5)

    @unittest.skipUnless(HAVE_KIN, "kinematics files / development tasks missing")
    def test_env_make_env_records_and_dead_band_in_the_loop(self):
        from nedm.so101_policy import joint_teacher as JT
        codec = JE.Codec(np.zeros(18), np.ones(18))
        with mock.patch.dict(os.environ, {JE.DYN_ENV: json.dumps(EVAL_DYN)}):
            e = JT.make_env("kin", 2, codec, "cpu")
        self.assertIsNotNone(e.dyn.servo)
        self.assertEqual(e.resolved()["env"]["dyn"]["servo"], EVAL_SERVO)
        t = tasks(4)
        a, b = kin_env(t, EVAL_DYN), kin_env(t, EVAL_DYN)
        self.assertTrue(torch.equal(a.dyn.servo_params(), b.dyn.servo_params()))
        self.assertTrue(torch.equal(roll(a, 4), roll(b, 4)))            # reproducible from the task seeds
        info = a.dyn_info()
        self.assertEqual(np.asarray(info["servo_params"]).shape, (4, len(JE.SERVO_KEYS), 6))
        self.assertEqual(info["servo_keys"], list(JE.SERVO_KEYS))
        json.dumps(info)
        off = kin_env(t, TRAIN)                                         # off: the old records, no "servo" key
        self.assertNotIn("servo_params", off.dyn_info())
        self.assertNotIn("servo", off.dyn_info()["cfg"])
        self.assertNotIn("servo", off.resolved()["env"]["dyn"])
        old = {k: v for k, v in dataclasses.asdict(off.dyn.cfg).items() if k not in ("servo", "i_antiwindup")}
        self.assertEqual(off.dyn_info()["cfg"], old)
        self.assertEqual(JE.JointEnvConfig.from_dict(off.cfg.to_dict()), off.cfg)
        with mock.patch.object(JE, "EVAL_KP_SCALE", 0.5):              # the softness counted twice: an error
            with self.assertRaises(ValueError):
                kin_env(t, EVAL_DYN)
            kin_env(t, TRAIN)                                           # without the servo: as before
        # in the loop, without the loop delay: a 0.25 deg pan target change does not move the arm, 2 deg does
        moved = {}
        for tag, d_ in (("servo", dict(servo=EVAL_SERVO)), ("off", False)):
            for dc in (0.25, 2.0):
                env = kin_env(t, d_)
                q0 = env.ws.q.to(torch.float64).clone()
                for _ in range(4):
                    tg = q0[:, None].expand(env.B, JE.STEPS, 6).clone()
                    tg[..., 0] += math.radians(dc)
                    env.step(None, targets=tg)
                moved[tag, dc] = float(torch.rad2deg(env.ws.q[:, 0].to(torch.float64) - q0[:, 0]).abs().max())
        self.assertLess(moved["servo", 0.25], 0.01)
        self.assertGreater(moved["off", 0.25], 0.2)
        self.assertGreater(moved["servo", 2.0], 1.3)

    @unittest.skipUnless(HAVE_MUJOCO and HAVE_KIN, "mujoco / development tasks missing")
    def test_mujoco_world_follows_the_servo(self):
        from nedm.so101_policy import joint_teacher as JT
        from nedm.so101_policy.vision.demos import start_state
        t = tasks(2)
        e = JT.make_env("mujoco", 2, JE.Codec(np.zeros(18), np.ones(18)), "cpu", dyn=dict(servo=EVAL_SERVO))
        mj = dict(q=t["start_q"], cube_pos=t["start_cube_pos"], cube_quat=t["start_cube_quat"])
        e.reset_tasks(torch.arange(2), start_state(t, "cpu"), torch.as_tensor(t["goal_xy"]),
                      torch.as_tensor(t["goal_yaw"]), torch.as_tensor(t["seeds"]), mj, check_tol=None)
        q0 = e.ws.q.to(torch.float64).clone()
        gap = []
        for k in range(8):
            tg = q0[:, None].expand(2, JE.STEPS, 6).clone()
            tg[..., 0] += math.radians(1.5 * (k + 1))
            e.step(None, targets=tg)
            gap.append(float(torch.rad2deg(e.ws.q.to(torch.float64)[:, :5] - e.dyn.servo.y[:, :5]).abs().max()))
        e.close()
        self.assertLess(max(gap), 0.3)                                  # the stiff sim servo follows the output y
        self.assertGreater(float(torch.rad2deg(e.dyn.servo.y[:, 0] - q0[:, 0]).min()), 5.0)   # and the arm moved


if __name__ == "__main__":
    unittest.main()


class TestEligible(unittest.TestCase):
    def test_only_eligible_worlds_perturbed(self):
        from nedm.so101_policy import joint_env as JE
        cfg = JE.dyn_config(dict(p=1.0, d_lo=2, d_hi=2, s_lo=0.5, s_hi=0.5))
        dyn = JE.RobotDyn(cfg, 8, "cpu", 0)
        q = torch.zeros(8, 6, dtype=torch.float64)
        dyn.reset(torch.arange(8), q)
        self.assertTrue(bool(dyn.on.all()))
        m = torch.tensor([False] * 4 + [True] * 4)
        dyn.set_eligible(m)
        self.assertFalse(bool(dyn.on[:4].any()))
        self.assertTrue(bool(dyn.on[4:].all()))
        dyn.reset(torch.arange(8), q)                     # new episodes: still only the eligible worlds
        self.assertFalse(bool(dyn.on[:4].any()))
        self.assertTrue(bool(dyn.on[4:].all()))


class TestPause(unittest.TestCase):
    def _task(self, B, held=True, dxy=0.01, zb=0.03):
        from nedm.so101_policy import task as T
        c = torch.zeros(B, 3, dtype=torch.float64)
        c[:, 0], c[:, 2] = dxy, zb + T.CUBE_HALF
        F = dict(c=c, held=torch.full((B,), held))
        return types.SimpleNamespace(F=F, goal_xy=torch.zeros(B, 2, dtype=torch.float64),
                                     tr=types.SimpleNamespace(lifted=torch.ones(B, dtype=torch.bool)))

    def test_freeze_arm_only_then_release(self):
        cfg = JE.pause_config(dict(p=1.0, s_lo=0.1, s_hi=0.1, rate=1.0, max_n=1))
        pz = JE.Pause(cfg, 4, "cpu", 0)
        pz.reset(torch.arange(4))
        q = torch.full((4, 6), 0.3, dtype=torch.float64)
        pz.begin(self._task(4), q)
        self.assertTrue(bool((pz.left == 5).all()))                    # 0.1 s = 5 steps
        u = torch.full((4, 6), 0.9, dtype=torch.float64)
        for _ in range(5):
            v = pz.apply(u)
            self.assertTrue(torch.allclose(v[:, :5], q[:, :5]))         # arm frozen at the pause start q
            self.assertTrue(torch.allclose(v[:, 5], u[:, 5]))           # jaw follows its command
        self.assertTrue(torch.equal(pz.apply(u), u))                    # over: the command again
        pz.begin(self._task(4), q)                                      # max_n 1: no second pause
        self.assertFalse(bool((pz.left > 0).any()))

    def test_condition(self):
        cfg = JE.pause_config(dict(p=1.0, rate=1.0))
        for kw in (dict(held=False), dict(dxy=0.2), dict(zb=0.002), dict(zb=0.2)):
            pz = JE.Pause(cfg, 3, "cpu", 0)
            pz.reset(torch.arange(3))
            pz.begin(self._task(3, **kw), torch.zeros(3, 6, dtype=torch.float64))
            self.assertFalse(bool((pz.left > 0).any()), kw)
        self.assertIsNone(JE.pause_config({}))
        with self.assertRaises(ValueError):
            JE.pause_config(dict(bogus=1))


class TestQdInput(unittest.TestCase):
    def test_qd_input_zero_and_online_keys(self):
        from nedm.so101_policy.vision import viral as VI
        cfg = VI.ViralConfig.from_dict(dict(qd_input_zero=True, qd_noise=0.0))
        self.assertTrue(cfg.qd_input_zero)
        net = types.SimpleNamespace(cfg=cfg, robot_mean=torch.zeros(12), robot_std=torch.ones(12), training=False)
        r = (torch.arange(12, dtype=torch.float32)[None] + 1.0) / 4.0           # inside the [-10, 10] clamp
        out = VI.ViralNet.robot(net, r)
        self.assertTrue(torch.equal(out[0, :6], r[0, :6]))
        self.assertTrue(torch.equal(out[0, 6:], torch.zeros(6)))
        cfg.qd_input_zero = False
        self.assertTrue(torch.equal(VI.ViralNet.robot(net, r), r))
        oc = VI.OnlineConfig.from_dict(dict(qd_in_lo=0.0, qd_in_hi=1.0, qd_in_zero_p=0.3, relabel_cmd_speed=True))
        self.assertTrue(oc.relabel_cmd_speed)


class TestQueueDelay(unittest.TestCase):
    def test_queue_runs_every_target_late(self):
        cfg = JE.dyn_config(dict(p=1.0, d_lo=2, d_hi=2, delay_mode="queue"))
        dyn = JE.RobotDyn(cfg, 1, "cpu", 0)
        q0 = torch.zeros(1, 6, dtype=torch.float64)
        dyn.reset(torch.arange(1), q0)
        tlo, thi = torch.full((6,), -10.0, dtype=torch.float64), torch.full((6,), 10.0, dtype=torch.float64)
        sent = []
        for dec in range(3):
            dyn.begin()
            ex = []
            for j in range(JE.STEPS):
                u = torch.full((1, 6), float(10 * dec + j + 1), dtype=torch.float64) / 100.0
                ex.append(u)
                sent.append(float(dyn.apply(j, ex, u, q0, tlo, thi)[0, 0]) * 100.0)
        want = [0.0, 0.0] + [float(10 * dec + j + 1) for dec in range(3) for j in range(JE.STEPS)][:13]
        np.testing.assert_allclose(sent, want, atol=1e-9)     # the command stream 2 steps late, nothing dropped
        with self.assertRaises(ValueError):
            JE.dyn_config(dict(p=1.0, d_lo=1, d_hi=1, delay_mode="bogus"))
