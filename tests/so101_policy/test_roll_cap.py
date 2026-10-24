"""Tests of the wrist-roll target cap (``joint_env.JointEnvConfig.roll_hi_deg``, real2sim 2026-10-23: the real wrist
roll ends at +101.2 sim deg): off = the old env bit for bit (the old step loop written out; a cap above the target
range changes no output either); on = no target sent to the world, executed target (``u_exe``) or JT label above the
cap (action layer, robot dynamics stage with noise and the integral correction, a policy-frame joint offset), start-bank
rows above the cap dropped, fixed-task starts above it warned, per-episode roll records and summaries; the plumbing
(``OnlineConfig``, ``joint_teacher.make_env`` / ``run_tasks``, the online loop's labels and pool, the PPO statistic,
the eval CLI option, the new config files). CPU, a few seconds; the env tests need the kinematic world's files and the
build9 development tasks, the online test also the JT (skipped without them).

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_roll_cap -v
"""
from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import types
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import joint_env as JE

ROOT = Path(__file__).resolve().parents[2]
TASKS = Path("~/so101_policy/runs/viral/tasks/dev100.npz").expanduser()
JT_CK = Path("~/so101_policy/runs/viral/jt/jt_h110.pt").expanduser()
HAVE_KIN = Path(JE.JointEnvConfig().kin_points).expanduser().exists() and \
    Path(JE.JointEnvConfig().geom_path).expanduser().exists() and TASKS.exists()
HAVE_JT = HAVE_KIN and JT_CK.exists()
CAP = 60.0                     # deg: binds on the development starts (wrist roll up to 79 deg) within a few decisions
OVER = math.radians(1.5)       # the kinematic world's servo overshoots a step target by up to ~1 deg (measured q)
torch.set_num_threads(4)


def tasks(n, first=0):
    from nedm.so101_policy.vision.demos import load_tasks
    return load_tasks(TASKS, np.arange(first, first + n))


def kin_env(t, roll=None, dyn=False, **kw):
    from nedm.so101_policy import joint_teacher as JT
    from nedm.so101_policy.vision.demos import start_state
    B = len(t["seeds"])
    env = JT.make_env("kin", B, JE.Codec(np.zeros(18), np.ones(18)), "cpu", dyn=dyn, roll_hi_deg=roll, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env.reset_tasks(torch.arange(B), start_state(t, "cpu"), torch.as_tensor(t["goal_xy"]),
                        torch.as_tensor(t["goal_yaw"]), torch.as_tensor(t["seeds"]))
    return env


def old_step(env, targets):
    """``JointVecEnv.step`` of the code before the cap for absolute targets (no q offset, servo scale 1, no robot
    dynamics): the action layer, the world step and the task step per 20 ms step."""
    u5 = targets.to(dtype=torch.float64)
    fl = (JE.FLOOR if env.floor_z == 0.0 else JE.FLOOR + env.floor_z) if env.cfg.floor_guard else None
    ex = []
    for j in range(JE.STEPS):
        u = JE.safety(u5[:, j], env.ws.q.to(torch.float64), env.kin.tlo, env.kin.thi, env.lead, env.kin, fl,
                      env.floor_plane)
        ex.append(u)
        env.ws = env.world.step(u)
        env.task.step(env.ws, None, None)
    env.u_exe = torch.stack(ex, 1)


def push_targets(env, k, roll_step=0.05):
    """Targets that turn the wrist roll up by ``roll_step`` per 20 ms step and move the arm a little (decision k)."""
    tg = env.ws.q.to(torch.float64)[:, None, :].expand(env.B, JE.STEPS, 6).clone()
    tg[..., ROLL] += roll_step * torch.arange(1, JE.STEPS + 1, dtype=torch.float64)
    tg[..., 1] += 0.01 * (k + 1)
    tg[..., 5] -= 0.02
    return tg


ROLL = JE.ROLL


def same(a, b):
    """Bit-identical tensors, NaN == NaN (records of worlds that did not end)."""
    return a.shape == b.shape and a.dtype == b.dtype and bool(((a == b) | (a.isnan() & b.isnan())).all())


def rollout(env, n=10, step=None, sent=None):
    """n decisions of ``push_targets`` -> [B, n, ...] states, executed targets, rewards and done flags."""
    out = []
    if sent is not None:          # record every target sent to the world
        w0 = env.world.step

        def wstep(u):
            sent.append(u.clone())
            return w0(u)
        env.world.step = wstep
    for k in range(n):
        tg = push_targets(env, k)
        if step is None:
            obs, rew, done, ex = env.step(None, targets=tg)
            extra = torch.cat((obs.double(), rew.double()[:, None], done.double()[:, None],
                               ex["time_outs"].double()[:, None]), -1)
        else:
            step(env, tg)
            extra = torch.zeros(env.B, 0, dtype=torch.float64)
        out.append(torch.cat((env.ws.q, env.ws.cube_pos, env.ws.cube_quat, env.task.tr.end[:, None].double(),
                              env.u_exe.reshape(env.B, -1)), -1))
        if step is None:
            out[-1] = torch.cat((out[-1], extra), -1)
    return torch.stack(out, 1)


class TestConfig(unittest.TestCase):
    def test_default_off_records_and_validation(self):
        c = JE.JointEnvConfig()
        self.assertIsNone(c.roll_hi_deg)
        self.assertNotIn("roll_hi_deg", c.to_dict())                          # off: the old record
        on = JE.JointEnvConfig.from_dict(dict(roll_hi_deg=100))
        self.assertEqual(on.roll_hi_deg, 100.0)
        self.assertIsInstance(on.roll_hi_deg, float)
        self.assertEqual(JE.JointEnvConfig.from_dict(json.loads(json.dumps(on.to_dict()))), on)
        self.assertEqual(JE.JointEnvConfig.from_dict(c.to_dict()), c)
        for bad in ("100", float("nan"), float("inf"), True, -170.0, [100.0]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                JE.JointEnvConfig(roll_hi_deg=bad)
        self.assertIsNone(JE.roll_hi_config(None))

    def test_online_config(self):
        from nedm.so101_policy.vision import viral as VI
        self.assertIsNone(VI.OnlineConfig.from_dict({}).roll_hi_deg)
        self.assertEqual(VI.OnlineConfig.from_dict(dict(roll_hi_deg=100)).roll_hi_deg, 100.0)
        with self.assertRaises(ValueError):
            VI.OnlineConfig.from_dict(dict(roll_hi_deg="high"))

    def test_drop_high_roll(self):
        q = torch.zeros(5, 6, dtype=torch.float64)
        q[:, ROLL] = torch.tensor([0.0, 1.9, 1.0, float("nan"), 2.5])
        bank = dict(q=q, qd=torch.arange(5.0)[:, None].expand(5, 6))
        out, n = JE.drop_high_roll(bank, math.radians(100.0))
        self.assertEqual(n, 2)
        self.assertEqual(out["qd"][:, 0].tolist(), [0.0, 2.0, 3.0])          # the NaN row is kept (as before)
        same, n0 = JE.drop_high_roll(bank, 3.0)
        self.assertIs(same, bank)                                             # nothing above: the same dict
        self.assertEqual(n0, 0)

    def test_new_config_files(self):
        from nedm.so101_policy.vision import viral as VI
        t = json.loads((ROOT / "configs/so101_policy/ppo/r2s/teacher_roll_v8.json").read_text())
        v7 = json.loads((ROOT / "configs/so101_policy/ppo/r2s/teacher_dyn_v7.json").read_text())
        c = JE.JointEnvConfig.from_dict(t)
        self.assertEqual(c.roll_hi_deg, 100.0)
        self.assertIsNone(c.dyn)                                              # nominal NRD worlds
        self.assertEqual((c.starts, c.horizon), (v7["starts"], v7["horizon"]))
        self.assertEqual((c.task, c.hold_obs_s), (v7["task"], v7["hold_obs_s"]))   # the online env's hold cap / scale
        p = ROOT / "configs/so101_policy/vision"
        new = json.loads((p / "viral_r2s_online_relabel_roll_v8.json").read_text())
        old = json.loads((p / "viral_r2s_online_relabel_v4.json").read_text())
        self.assertEqual(VI.OnlineConfig.from_dict(new).roll_hi_deg, 100.0)
        self.assertEqual({k: v for k, v in new.items() if k not in ("note", "roll_hi_deg")},
                         {k: v for k, v in old.items() if k != "note"})


@unittest.skipUnless(HAVE_KIN, "kinematics files / development tasks missing")
class TestEnv(unittest.TestCase):
    def test_off_is_the_old_step(self):
        t = tasks(6)
        ref = rollout(kin_env(t), step=old_step)
        env = kin_env(t)
        self.assertIsNone(env.roll_hi)
        self.assertIsNone(env.roll_info())
        got = rollout(env, step=None)
        self.assertTrue(torch.equal(got[..., :ref.shape[-1]], ref))
        _, _, _, ex = env.step(None, targets=push_targets(env, 0))
        self.assertNotIn("roll_cmd_max", ex["episode"])

    def test_cap_above_the_range_changes_no_output(self):
        """A cap above THI (159.9 deg) runs every new code path but changes nothing (also with the robot dynamics
        stage): obs, reward, done, states, executed targets and every old episode record are bit-identical."""
        t = tasks(6)
        for dyn in (False, dict(p=1.0, d=1, sag_deg=0.8, i_time_s=0.2, noise=0.004)):
            a, b = kin_env(t, dyn=dyn), kin_env(t, 170.0, dyn=dyn)
            self.assertTrue(torch.equal(a.thi, b.thi))
            ra, rb = rollout(a), rollout(b)
            self.assertTrue(torch.equal(ra, rb), msg=str(dyn))
            ea = a.step(None, targets=push_targets(a, 0))[3]["episode"]
            eb = b.step(None, targets=push_targets(b, 0))[3]["episode"]
            self.assertEqual(set(eb) - set(ea), {"roll_cmd_max", "roll_q_max"})
            for k, v in ea.items():
                self.assertTrue(same(v, eb[k]), msg=k)

    def test_cap_on_every_sent_and_executed_target(self):
        t = tasks(8)
        cap = math.radians(CAP)
        for dyn in (False, dict(p=1.0, d=2, sag_deg=1.0, i_time_s=0.2, noise=0.01),
                    dict(p=1.0, d=1, delay_mode="queue", i_time_s=0.2, noise=0.01,
                         servo=dict(delay_ms=20, tau_ms=40, deadband_deg=0.3))):
            env = kin_env(t, CAP, dyn=dyn)
            start = env.ws.q[:, ROLL].to(torch.float64).clone()                    # one start is above the cap
            self.assertAlmostEqual(float(env.thi[ROLL]), cap, places=15)
            self.assertEqual(env.roll_cap(), float(env.thi[ROLL]))
            sent = []
            r = rollout(env, 12, sent=sent)
            S = torch.stack(sent)
            self.assertLessEqual(float(S[..., ROLL].max()), cap, msg=str(dyn))   # every target sent to the world
            self.assertGreater(float(S[..., ROLL].max()), cap - 1e-3)            # ... and the cap was reached
            u_exe = r[..., 14:44].reshape(8, 12, JE.STEPS, 6)
            self.assertLessEqual(float(u_exe[..., ROLL].max()), cap)             # executed targets (u_exe)
            self.assertLessEqual(float(env.ws.q[:, ROLL].max()), cap + OVER)     # the world follows the target
            ep = env.step(None, targets=push_targets(env, 12))[3]["episode"]
            at = ep["roll_q_max"] >= cap - 1e-3                                  # worlds that reached the cap
            self.assertTrue(bool(at.any()))
            self.assertTrue(bool((ep["roll_cmd_max"][at] > cap).all()))          # ... their command pushed above it
            self.assertTrue(bool((ep["roll_q_max"] <= torch.clamp(start, min=cap) + OVER).all()))
        # the original kinematics range is untouched (shared object)
        self.assertAlmostEqual(float(env.kin.thi[ROLL]), 2.7912063000000003, places=12)

    def test_servo_scale_and_policy_frame_offset(self):
        t = tasks(4)
        cap = math.radians(CAP)
        env = kin_env(t, CAP)
        start = env.ws.q[:, ROLL].to(torch.float64).clone()
        env.q_offset = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.1, 0.0], dtype=torch.float64)   # policy frame = q + 0.1
        self.assertAlmostEqual(env.roll_cap(), cap - 0.1, places=12)
        sent = []
        from unittest import mock
        with mock.patch.object(JE, "EVAL_KP_SCALE", 1.5):                         # an overshooting servo scale
            rollout(env, 10, sent=sent)
        self.assertLessEqual(float(torch.stack(sent)[..., ROLL].max()), cap - 0.1 + 1e-12)
        ep = env.step(None, targets=push_targets(env, 0))[3]["episode"]
        ok = ep["roll_q_max"] <= torch.clamp(start + 0.1, min=cap) + OVER           # records in the policy frame
        self.assertTrue(bool(ok.all()))

    def test_records_reset_with_the_episode(self):
        t = tasks(4)
        env = kin_env(t, CAP)
        r0 = env.ws.q[:, ROLL].to(torch.float64)
        self.assertTrue(torch.equal(env.roll_q_max, r0))                          # the start roll
        self.assertTrue(bool(torch.isinf(env.roll_cmd_max).all()))
        rollout(env, 2)
        self.assertTrue(bool(torch.isfinite(env.roll_cmd_max).all()))
        from nedm.so101_policy.vision.demos import start_state
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                                       # one start is above the cap
            env.reset_tasks(torch.arange(4), start_state(t, "cpu"), torch.as_tensor(t["goal_xy"]),
                            torch.as_tensor(t["goal_yaw"]), torch.as_tensor(t["seeds"]))
        self.assertTrue(bool(torch.isinf(env.roll_cmd_max).all()))

    def test_fixed_task_start_above_the_cap_warns(self):
        t = tasks(100)
        hi = np.degrees(t["state_q"][:, ROLL]) > CAP
        self.assertTrue(hi.any())
        with self.assertWarns(UserWarning):
            from nedm.so101_policy import joint_teacher as JT
            from nedm.so101_policy.vision.demos import start_state
            env = JT.make_env("kin", 100, JE.Codec(np.zeros(18), np.ones(18)), "cpu", dyn=False, roll_hi_deg=CAP)
            env.reset_tasks(torch.arange(100), start_state(t, "cpu"), torch.as_tensor(t["goal_xy"]),
                            torch.as_tensor(t["goal_yaw"]), torch.as_tensor(t["seeds"]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")                                        # off: no warning
            kin_env(t)

    def test_start_bank_rows_above_the_cap_are_dropped(self):
        t = tasks(100)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "bank.npz"
            np.savez(f, **{k: t[f"state_{k}"] for k in ("q", "qd", "cube_pos", "cube_quat", "cube_v", "cube_w")})
            kw = dict(world="kin", num_envs=16, device="cpu", starts=str(f), horizon=30, codec="")
            codec = JE.Codec(np.zeros(18), np.ones(18))
            off = JE.JointVecEnv(JE.JointEnvConfig(**kw), codec)
            on = JE.JointVecEnv(JE.JointEnvConfig(roll_hi_deg=CAP, **kw), codec)
            n_hi = int((np.degrees(t["state_q"][:, ROLL]) > CAP).sum())
            self.assertGreater(n_hi, 0)
            self.assertEqual((len(off.bank["q"]), len(on.bank["q"])), (100, 100 - n_hi))
            self.assertEqual(on.roll_info()["bank_dropped"], n_hi)
            self.assertLessEqual(float(on.bank["q"][:, ROLL].max()), math.radians(CAP))
            on.reset()
            for _ in range(8):                                                    # automatic resets
                on.step(None, targets=on.ws.q[:, None].expand(16, JE.STEPS, 6).clone())
                self.assertLessEqual(float(on.ws.q[:, ROLL].max()), math.radians(CAP) + OVER)
            far = JE.JointVecEnv(JE.JointEnvConfig(roll_hi_deg=170.0, **kw), codec)
            self.assertEqual(far.roll_dropped, 0)
            self.assertTrue(all(torch.equal(far.bank[k], off.bank[k]) for k in off.bank))

    def test_run_tasks_labels_and_summary(self):
        """``run_tasks`` with a teacher that turns the roll up: recorded labels capped, summary roll metrics; off: no
        roll keys and the raw labels."""
        from nedm.so101_policy import joint_teacher as JT
        t = tasks(6)
        a = torch.zeros(6, JE.ACT_DIM, dtype=torch.float64)
        a[:, ROLL] = 0.15                                      # constant term of the wrist roll: q + 0.15 rad
        teacher = types.SimpleNamespace(act=lambda obs: a[:obs.shape[0]].float(), std=torch.ones(JE.ACT_DIM))
        cap = math.radians(CAP)
        out = {}
        for roll in (None, CAP):
            env = JT.make_env("kin", 6, JE.Codec(np.zeros(18), np.ones(18)), "cpu", dyn=False, horizon=60,
                              roll_hi_deg=roll)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[roll] = JT.run_tasks(env, t, teacher, record=True)
        r = out[CAP]
        self.assertLessEqual(float(r["row_lab"][..., ROLL].max()), cap + 1e-6)   # float32 record of a capped label
        self.assertGreater(float(out[None]["row_lab"][..., ROLL].max()), cap + 0.1)
        self.assertNotIn("roll_cmd_max", out[None])
        s = JT.summarize(r)
        self.assertEqual(s["roll_cmd_over"], 1.0)
        self.assertEqual(s["roll_q_at_cap"], 1.0)
        self.assertGreater(s["roll_cmd_max_deg"], CAP)
        self.assertNotIn("roll_cmd_over", JT.summarize(out[None]))
        self.assertEqual(float(r["roll_hi_deg"]), CAP)

    def test_ppo_statistic(self):
        """``JointPPO.collect``: the share of finished episodes whose command went above the cap (``roll_cmd_over``)."""
        from nedm.so101_policy import joint_teacher as JT
        from nedm.so101_policy import ppo as P
        t = tasks(32)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "bank.npz"
            np.savez(f, **{k: t[f"state_{k}"] for k in ("q", "qd", "cube_pos", "cube_quat", "cube_v", "cube_w")})
            rows = {}
            for roll in (None, CAP):
                env = JE.JointVecEnv(JE.JointEnvConfig(world="kin", num_envs=16, device="cpu", starts=str(f),
                                                       horizon=20, roll_hi_deg=roll), JE.Codec(np.zeros(18),
                                                                                                np.ones(18)))
                tc = JT.ppo_train_cfg(dict(num_steps_per_env=8, init_std=[0.02, 0.02], std_floor=[0.02, 0.02]))
                bias = [0.0] * JE.ACT_DIM
                bias[ROLL] = 0.3                              # mean action: the roll's constant term up (+0.3 rad)
                pol = P.Policy(JE.OBS_DIM, JE.ACT_DIM, groups=env.action_groups, actor_hidden_dims=(16,),
                               critic_hidden_dims=(16,), init_std=tc["init_std"], actor_out_gain=0.0,
                               action_bias=bias)
                algo = JT.JointPPO(env, pol, tc)
                rows[roll] = algo.iterate()
            self.assertNotIn("roll_cmd_over", rows[None])
            self.assertGreater(rows[CAP]["episodes"], 0)
            self.assertGreater(rows[CAP]["roll_cmd_over"], 0.0)
            self.assertLessEqual(rows[CAP]["roll_cmd_over"], 1.0)


@unittest.skipUnless(HAVE_JT, "JT / development tasks missing")
class TestOnline(unittest.TestCase):
    def test_labels_pool_and_targets(self):
        """The online loop in the kinematic world (``studentlab_ref.make_online``): the pool keeps only starts at or
        below the cap, every buffer label and every target sent to the world has the roll at or below it; off: the
        pool is the whole task table."""
        import sys
        sys.path.insert(0, str(ROOT / "tests" / "so101_policy"))
        import studentlab_ref as R
        cap = math.radians(CAP)
        st = R.tiny_student(seed=0)
        od, _ = R.make_online(st, dict(roll_hi_deg=CAP, teacher_share=0.5, horizon=40), envs=16)
        self.assertAlmostEqual(od.roll_cap, cap, places=15)
        bank_roll = od.env.bank["q"][:, ROLL]
        self.assertLessEqual(float(bank_roll[od.env.pool["i"]].max()), cap)
        self.assertLess(len(od.env.pool["i"]), len(bank_roll))
        self.assertEqual(len(od.env.pool["goal_xy"]), len(od.env.pool["i"]))
        sent = []
        w0 = od.env.world.step
        od.env.world.step = lambda u: (sent.append(u.clone()), w0(u))[1]
        for it in range(12):
            od.it = it
            od.act()
        self.assertLessEqual(float(torch.stack(sent)[..., ROLL].max()), cap)
        n = od.buf.n
        self.assertLessEqual(float(od.buf.lab1[:n, :, ROLL].max()), cap + 1e-6)
        v2 = od.buf.valid2[:n]
        if bool(v2.any()):
            self.assertLessEqual(float(od.buf.lab2[:n][v2][:, :, ROLL].max()), cap + 1e-6)
        self.assertIn("roll_over_student", od.stats)
        off, _ = R.make_online(R.tiny_student(seed=0), dict(teacher_share=0.5, horizon=40), envs=16)
        self.assertIsNone(off.roll_cap)
        self.assertEqual(len(off.env.pool["i"]), len(off.env.bank["q"]))
        self.assertNotIn("roll_over_student", off.stats)


class TestCli(unittest.TestCase):
    def test_eval_option_and_record(self):
        spec = importlib.util.spec_from_file_location("vision_viral_script",
                                                      ROOT / "scripts/so101_policy/vision_viral.py")
        S = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S)
        seen = {}
        S.cmd_eval = lambda a: seen.update(vars(a))
        S.main(["eval", "--jt", "x.pt", "--student", "s.pt", "--tasks", "t.npz", "--roll-hi-deg", "100"])
        self.assertEqual(seen["roll_hi_deg"], 100.0)
        seen.clear()
        S.main(["eval", "--jt", "x.pt", "--student", "s.pt", "--tasks", "t.npz"])
        self.assertIsNone(seen["roll_hi_deg"])
        cache = S.EnvCache(None, "cpu", None, 0, None, 400, roll_hi_deg=100.0)
        self.assertEqual(cache.roll_hi_deg, 100.0)
        from nedm.so101_policy import task as T
        stub = types.SimpleNamespace(dyn=None, tcfg=T.TaskConfig(), cfg=types.SimpleNamespace(hold_obs_s=0.0,
                                                                                             roll_hi_deg=100.0),
                                     roll_hi=math.radians(100.0))
        self.assertEqual(json.loads(S.robot_rec(stub))["roll_hi_deg"], 100.0)
        stub.roll_hi = None
        self.assertIsNone(S.robot_rec(stub))                                      # off: no record (old files)

    @unittest.skipUnless(HAVE_JT, "JT / development tasks missing")
    def test_script_evaluate_with_the_cap(self):
        """``vision_viral.evaluate`` (the gates' path) in the kinematic world: with --roll-hi-deg every target sent to
        the world is capped and the summary has the roll keys; off: no roll keys."""
        import sys
        sys.path.insert(0, str(ROOT / "tests" / "so101_policy"))
        import studentlab_ref as R
        spec = importlib.util.spec_from_file_location("vision_viral_script",
                                                      ROOT / "scripts/so101_policy/vision_viral.py")
        S = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S)
        jt = JE.JointTeacher(JT_CK, "cpu")
        t = tasks(6)
        out = {}
        for roll in (None, CAP):
            cache = S.EnvCache(jt, "cpu", None, 0, None, 40, roll_hi_deg=roll)
            st = R.tiny_student(seed=0)
            env = cache.get("kin", 6)
            sent = []
            w0 = env.world.step
            env.world.step = lambda u, w0=w0, sent=sent: (sent.append(u.clone()), w0(u))[1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = S.evaluate(st, jt, t, "kin", cache, R.FakeViews(st.cfg.res, st.cfg.cameras), device="cpu")
            out[roll] = (json.loads(str(r["summary"])), torch.stack(sent))
        self.assertNotIn("roll_cmd_over", out[None][0])
        self.assertIn("roll_cmd_over", out[CAP][0])
        self.assertLessEqual(float(out[CAP][1][..., ROLL].max()), math.radians(CAP))

    def test_teacher_script_option(self):
        spec = importlib.util.spec_from_file_location("vision_viral_teacher_script",
                                                      ROOT / "scripts/so101_policy/vision_viral_teacher.py")
        S = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S)
        seen = {}
        S.cmd_eval = lambda a: seen.update(vars(a))
        S.main(["eval", "--jt", "x.pt", "--tasks", "t.npz", "--roll-hi-deg", "100"])
        self.assertEqual(seen["roll_hi_deg"], 100.0)


if __name__ == "__main__":
    unittest.main()
