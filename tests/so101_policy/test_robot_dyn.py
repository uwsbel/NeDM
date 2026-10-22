"""Tests of the real2sim robot timing / servo emulation (``joint_env.RobotDynConfig`` / ``RobotDyn``, the
``JointVecEnv.step`` stage, ``$NEDM_SO101_DYN`` in ``joint_teacher.make_env``) and of the online-loop options of
``vision.viral`` (robot-like qdot ``robot_qd``, image lag up to 6 steps, ``OnlineConfig`` horizon / hold cap, the
per-sample qdot noise scale). CPU, a few seconds; the env tests need the kinematic world's files (kin points, task
geometry, the build9 development tasks; skipped without them), the MuJoCo test needs mujoco.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_robot_dyn -v
"""
from __future__ import annotations

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
