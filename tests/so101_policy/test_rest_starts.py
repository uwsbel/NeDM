"""Tests of the held-cube rest starts (``joint_env.RestConfig``, real2sim round v9, 2026-10-25) and of the widened arm
pause (``PauseConfig`` with ``near_m`` 0.30 / ``z_lo`` 0.002, a config change only): the defaults are unchanged (no
``rest`` key in the env record, the old ``PauseConfig`` values, the automatic reset of the code before the option
written out = bit for bit, share 0 = off bit for bit); on = rest rows with the bank's goal, lifted flag and held time
(observation included), the robot dynamics stage's jaw history, the share, the roll-cap drop; ``run_tasks`` presets the
flags only for task files with rest_* keys; the round v9 teacher configs. CPU, a few seconds; the env tests need the
kinematic world's files and the build9 development tasks (skipped without them).

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_rest_starts -v
"""
from __future__ import annotations

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
HAVE_KIN = Path(JE.JointEnvConfig().kin_points).expanduser().exists() and \
    Path(JE.JointEnvConfig().geom_path).expanduser().exists() and TASKS.exists()
STATE = ("q", "qd", "cube_pos", "cube_quat", "cube_v", "cube_w")
# JointEnvConfig().to_dict() keys of the code before the option (claude/r2s-v8 ddb08332d)
OLD_KEYS = {"world", "num_envs", "device", "seed", "ckpts", "members", "nrd_cuda_graph", "starts", "start_split",
            "pool_size", "gamma", "horizon", "task", "log_only", "lead_clamp", "floor_guard", "codec", "auto_reset",
            "stagger", "block_half", "geom_path", "kin_points", "mj_workers", "mj_python", "mjwarp", "dyn", "pause",
            "hold_obs_s"}
torch.set_num_threads(4)


def tasks(n, first=0):
    from nedm.so101_policy.vision.demos import load_tasks
    return load_tasks(TASKS, np.arange(first, first + n))


def codec():
    return JE.Codec(np.zeros(18), np.ones(18))


def old_pool_reset(env, idx):
    """``JointVecEnv._pool_reset`` of the code before the rest option, written out."""
    from nedm.so101_policy.vec_env import start_rows
    if env.pool is None:
        env.refresh_pool()
    k = torch.randint(len(env.pool["i"]), (len(idx),), generator=env.gen, device=env.dev)
    i = env.pool["i"][k]
    s = (int(env.cfg.seed) * 0x9E3779B1 + env.ep_count[idx] * env.B + idx) & 0xFFFFFFFF
    env.ep_count[idx] += 1
    env._reset_rows(idx, start_rows(env.bank, i), env.pool["goal_xy"][k], env.pool["goal_yaw"][k], s)


def write_banks(d, t, n_rest=None, roll_deg=None):
    """A start bank (the development states) and a rest bank (the same states + rest_* keys) in folder d."""
    st = {k: np.asarray(t[f"state_{k}"], np.float64) for k in STATE}
    np.savez(Path(d) / "bank.npz", **st)
    n = len(st["q"]) if n_rest is None else n_rest
    rs = {k: v[:n].copy() for k, v in st.items()}
    if roll_deg is not None:
        rs["q"][: n // 2, JE.ROLL] = math.radians(roll_deg)
    rng = np.random.default_rng(5)
    ctrl = rs["q"].copy()
    ctrl[:, 5] = -0.05
    np.savez(Path(d) / "rest.npz", **rs, rest_goal_xy=rs["cube_pos"][:, :2] + rng.uniform(0.04, 0.1, (n, 2)),
             rest_goal_yaw=rng.uniform(-0.4, 0.4, n), rest_lifted=np.ones(n, bool), rest_air_s=rng.uniform(0.3, 8, n),
             rest_ctrl=ctrl, rest_row_seed=np.arange(760000, 760000 + n))
    return Path(d) / "bank.npz", Path(d) / "rest.npz"


def bank_env(bank, B=16, **kw):
    cfg = dict(world="kin", num_envs=B, device="cpu", starts=str(bank), horizon=30, codec="", seed=3)
    cfg.update(kw)
    return JE.JointVecEnv(JE.JointEnvConfig(**cfg), codec())


def rollout(env, n=14):
    """reset + n decisions of small random targets (horizon 30 steps: automatic resets every 6 decisions) -> one
    tensor of states, observations, rewards, done flags, goals and task seeds per decision."""
    g = torch.Generator().manual_seed(11)
    out = [torch.cat((env.reset().double(), env.task.goal_xy, env.task_seed[:, None].double()), -1)]
    for _ in range(n):
        tg = env.ws.q.to(torch.float64)[:, None, :] + 0.02 * torch.randn(env.B, JE.STEPS, 6, generator=g,
                                                                        dtype=torch.float64)
        obs, rew, done, ex = env.step(None, targets=tg)
        out.append(torch.cat((env.ws.q, env.ws.cube_pos, env.ws.cube_quat, obs.double(), rew.double()[:, None],
                              done.double()[:, None], env.task.goal_xy, env.task_seed[:, None].double(),
                              env.task.tr.lifted.double()[:, None], env.task.tr.air_run.double()[:, None]), -1))
    return out


def equal(a, b):
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


class TestConfig(unittest.TestCase):
    def test_defaults_unchanged(self):
        c = JE.JointEnvConfig()
        self.assertIsNone(c.rest)
        self.assertEqual(set(c.to_dict()), OLD_KEYS)                          # the old record
        self.assertEqual(JE.JointEnvConfig.from_dict(c.to_dict()), c)
        p = JE.PauseConfig()
        self.assertEqual((p.p, p.s_lo, p.s_hi, p.near_m, p.z_lo, p.z_hi, p.rate, p.max_n),
                         (0.5, 0.5, 3.0, 0.04, 0.008, 0.06, 0.25, 2))
        for off in (None, False, {}, "", "{}", "null"):
            self.assertIsNone(JE.rest_config(off), msg=repr(off))
            self.assertIsNone(JE.JointEnvConfig(rest=off).rest)

    def test_rest_config(self):
        on = JE.JointEnvConfig.from_dict(dict(rest=dict(bank="x.npz", share=0.25, note="n")))
        self.assertEqual(on.rest, JE.RestConfig("x.npz", 0.25))
        self.assertEqual(on.to_dict()["rest"], dict(bank="x.npz", share=0.25))
        self.assertEqual(JE.JointEnvConfig.from_dict(json.loads(json.dumps(on.to_dict()))), on)
        self.assertEqual(JE.rest_config('{"bank": "y.npz"}').share, 0.25)
        for bad in (dict(share=0.3), dict(bank="x.npz", share=1.5), dict(bank="x.npz", share=-0.1),
                    dict(bank="x.npz", share=True), dict(bank="x.npz", p=0.2)):
            with self.assertRaises(ValueError, msg=repr(bad)):
                JE.rest_config(bad)

    def test_widened_pause_config(self):
        p = JE.pause_config(dict(near_m=0.30, z_lo=0.002, z_hi=0.08, rate=0.15))
        self.assertEqual((p.near_m, p.z_lo, p.z_hi, p.rate, p.p, p.max_n), (0.30, 0.002, 0.08, 0.15, 0.5, 2))

    def test_v9_teacher_configs(self):
        d = ROOT / "configs/so101_policy/ppo/r2s"
        dyn = json.loads((ROOT / "configs/so101_policy/vision/viral_r2s_online_relabel_roll_v8.json").read_text())["dyn"]
        roll = json.loads((d / "teacher_roll_v8.json").read_text())
        for n, want_dyn in (("teacher_restart_v9a", None), ("teacher_restart_v9b", dyn)):
            raw = json.loads((d / f"{n}.json").read_text())
            c = JE.JointEnvConfig.from_dict(raw)
            for k in ("horizon", "task", "hold_obs_s", "roll_hi_deg"):           # as D's teacher fine-tune
                self.assertEqual(raw[k], roll[k], msg=(n, k))
            self.assertEqual((c.rest.share, Path(c.rest.bank).name), (0.25, "rest_bank_v9_8k.npz"))
            self.assertEqual((c.pause.near_m, c.pause.z_lo), (0.30, 0.002))
            self.assertTrue(raw["starts"].endswith("bank_pp_h45h0mix.npz"))
            if want_dyn is None:
                self.assertIsNone(c.dyn)
            else:                 # the student recipe's slow-arm stage: no command delay, no fitted servo
                self.assertEqual(raw["dyn"], want_dyn)
                self.assertEqual((c.dyn.d_hi, c.dyn.servo, c.dyn.s_lo, c.dyn.tau_hi, c.dyn.sag_max_deg),
                                 (0, {}, 0.3, 0.15, 1.5))


@unittest.skipUnless(HAVE_KIN, "kinematics files / development tasks missing")
class TestEnv(unittest.TestCase):
    def test_off_is_the_old_reset(self):
        """Default env and share 0 (also with the robot dynamics stage and the pause): the automatic resets, states,
        observations, rewards and goals of the code before the option, bit for bit."""
        t = tasks(40)
        with tempfile.TemporaryDirectory() as d:
            bank, rest = write_banks(d, t)
            for kw in (dict(), dict(dyn=dict(p=0.8, s_lo=0.3, s_hi=1.0, tau_lo=0.02, tau_hi=0.15, sag_max_deg=1.5,
                                             i_time_s=0.2), pause=dict(p=1.0, near_m=0.30, z_lo=0.002))):
                ref_env = bank_env(bank, **kw)
                ref_env._pool_reset = types.MethodType(old_pool_reset, ref_env)
                ref = rollout(ref_env)
                env = bank_env(bank, **kw)
                self.assertIsNone(env.rest)
                self.assertIsNone(env.rest_info())
                self.assertTrue(equal(rollout(env), ref), msg=str(kw))
                z = bank_env(bank, rest=dict(bank=str(rest), share=0.0), **kw)
                self.assertTrue(equal(rollout(z), ref), msg=f"share 0 {kw}")
                self.assertEqual(z.rest_info()["rest_starts"], 0)
                self.assertGreater(z.rest_info()["resets"], 16)

    def test_rest_rows(self):
        t = tasks(40)
        with tempfile.TemporaryDirectory() as d:
            bank, rest = write_banks(d, t)
            R = np.load(rest)
            env = bank_env(bank, rest=dict(bank=str(rest), share=1.0), hold_obs_s=3.5, task=dict(hold_cap_s=20.0))
            obs = env.reset()
            # every row is a rest row: find its bank row by the start q
            q0 = env.ws.q.numpy()
            j = np.array([int(np.argmin(np.abs(R["q"] - q).sum(1))) for q in q0])
            self.assertTrue(np.allclose(R["q"][j], q0, atol=1e-6))
            self.assertTrue(np.allclose(env.task.goal_xy.numpy(), R["rest_goal_xy"][j]))
            self.assertTrue(np.allclose(env.task.goal_yaw.numpy(), R["rest_goal_yaw"][j]))
            self.assertTrue(bool(env.task.tr.lifted.all()))
            air = np.round(R["rest_air_s"][j] / 0.02).astype(np.int64)
            self.assertTrue(np.array_equal(env.task.tr.air_run.numpy(), air))
            self.assertTrue(torch.equal(obs[:, -2], torch.ones(16, dtype=obs.dtype)))         # lifted input
            self.assertTrue(np.allclose(obs[:, -1].numpy(), np.clip(air * 0.02 / 3.5, 0, 1), atol=1e-6))
            self.assertEqual(env.rest_info()["rest_starts"], 16)
            # share 0.5: about half of the automatic resets; the other rows are the old bank rows (lifted 0)
            h = bank_env(bank, B=64, rest=dict(bank=str(rest), share=0.5))
            h.reset()
            for _ in range(30):
                h.step(None, targets=h.ws.q[:, None].expand(64, JE.STEPS, 6).clone())
            info = h.rest_info()
            self.assertGreater(info["resets"], 64 * 4)
            self.assertLess(abs(info["rest_starts"] / info["resets"] - 0.5), 0.08)

    def test_dyn_jaw_history_and_roll_cap(self):
        t = tasks(40)
        with tempfile.TemporaryDirectory() as d:
            bank, rest = write_banks(d, t, roll_deg=110.0)
            dyn = dict(p=1.0, tau_lo=0.05, tau_hi=0.05)
            env = bank_env(bank, rest=dict(bank=str(rest), share=1.0), dyn=dyn, roll_hi_deg=98.0)
            self.assertEqual(env.rest_info()["dropped"], 20)
            self.assertEqual(env.rest_info()["rows"], 20)
            self.assertLessEqual(float(env.rest["q"][:, JE.ROLL].max()), math.radians(98.0))
            env.reset()
            self.assertTrue(torch.equal(env.dyn.hold[:, 5], torch.full((16,), -0.05, dtype=torch.float64)))
            self.assertTrue(torch.equal(env.dyn.f[:, 5], torch.full((16,), -0.05, dtype=torch.float64)))
            self.assertTrue(bool((env.dyn.prev5[:, :, 5] == -0.05).all()))
            with self.assertRaises(ValueError):                 # fixed-task envs (no start bank): no rest option
                JE.JointVecEnv(JE.JointEnvConfig(world="kin", num_envs=4, device="cpu", starts="",
                                                 rest=dict(bank=str(rest))), codec())
            with self.assertRaises(ValueError):                 # a task file is not a rest bank
                JE.load_rest_bank(TASKS, "cpu")

    def test_far_open_cost(self):
        """FarOpenConfig: cost 0 = off bit for bit; cost > 0 lowers the reward by exactly cost x max(0, executed jaw
        target - measured jaw) summed over the steps of live worlds whose cube is held (forced here), lifted and far."""
        from nedm.so101_policy import joint_teacher as JT
        from nedm.so101_policy.vision.demos import start_state
        t = tasks(12)

        def run(fo):
            env = JT.make_env("kin", 12, codec(), "cpu", horizon=60, dyn=False)
            env.cfg.far_open = JE.far_open_config(fo)
            env.reset_tasks(torch.arange(12), start_state(t, "cpu"), torch.as_tensor(t["goal_xy"]),
                            torch.as_tensor(t["goal_yaw"]), torch.as_tensor(t["seeds"]))
            env.task.tr.lifted[:] = True
            orig, wstep = env.task.step, env.world.step
            rec = []

            def step(*a, **k):                 # every cube counts as held (the kin world grasps only by closing)
                out = orig(*a, **k)
                env.task.F["held"] = torch.ones_like(env.task.F["held"])
                return out

            def world_step(u):                 # q, cube, liveness before each 20 ms step
                F = env.task.F
                rec.append((env.ws.q[:, 5].clone(), None if F is None else F["c"][:, :2].clone(),
                            (env.task.tr.end == 0).clone()))
                return wstep(u)
            env.task.step, env.world.step = step, world_step
            out, exp = [], []
            for k in range(4):
                rec.clear()
                tg = env.ws.q.to(torch.float64)[:, None, :].expand(12, JE.STEPS, 6).clone()
                tg[:6, :, 5] += 0.3                                              # open
                tg[6:, :, 5] -= 0.1                                              # close
                obs, rew, done, ex = env.step(None, targets=tg)
                out.append(rew.double())
                e = torch.zeros(12, dtype=torch.float64)
                for j, (qj, c, live) in enumerate(rec):
                    if k > 0:                  # decision 1: no task.F yet (the cost starts with the 2nd decision)
                        far = live & ((c - env.task.goal_xy).norm(dim=-1) > 0.04)
                        e += torch.where(far, (env.u_exe[:, j, 5] - qj).clamp(min=0.0), torch.zeros_like(e))
                exp.append(e)
            return torch.stack(out, 1), torch.stack(exp, 1), env
        off, _, e0 = run(None)
        zero, _, _ = run(dict(cost=0.0))
        on, exp, e1 = run(dict(cost=2.0, far_m=0.04))
        self.assertTrue(torch.equal(off, zero))
        self.assertEqual(float(e0.far_open_sum), 0.0)
        self.assertTrue(torch.allclose(off - on, 2.0 * exp, atol=1e-6))
        self.assertTrue(bool((exp[:6, 1] > 0.1).all()))       # the opening worlds pay (later: the jaw reaches its range end)
        self.assertLess(float(exp[6:].max()), 0.1)            # closing jaw: only the servo overshoot (5 steps)
        self.assertAlmostEqual(float(e1.far_open_sum), float(2.0 * exp.sum()), places=5)
        c = JE.JointEnvConfig.from_dict(dict(far_open=dict(cost=0.5, note="x")))
        self.assertEqual(c.far_open, JE.FarOpenConfig(0.5, 0.04))
        self.assertEqual(c.to_dict()["far_open"], dict(cost=0.5, far_m=0.04))
        self.assertNotIn("far_open", JE.JointEnvConfig().to_dict())
        for bad in (dict(cost=-1.0), dict(far_m=0.0), dict(cost=True), dict(k=1)):
            with self.assertRaises(ValueError, msg=repr(bad)):
                JE.far_open_config(bad)

    def test_run_tasks_rest_keys(self):
        from nedm.so101_policy import joint_teacher as JT
        t = tasks(8)
        still = types.SimpleNamespace(act=lambda o: torch.zeros(len(o), 18), std=torch.ones(18))
        out = {}
        for name, extra in (("plain", {}), ("rest", dict(rest_lifted=np.ones(8, bool),
                                                          rest_air_s=np.full(8, 2.0),
                                                          rest_ctrl=np.asarray(t["state_q"])))):
            env = JT.make_env("kin", 8, codec(), "cpu", horizon=20, dyn=False)
            seen = {}
            orig = env.set_rest

            def rec(idx, lifted, air, jaw=None, orig=orig, seen=seen):
                seen.update(lifted=lifted.clone(), air=air.clone())
                return orig(idx, lifted, air, jaw)
            env.set_rest = rec
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[name] = JT.run_tasks(env, {**t, **extra}, still)
            out[name + "_seen"] = seen
        self.assertEqual(out["plain_seen"], {})
        self.assertTrue(bool(out["rest_seen"]["lifted"].all()))
        self.assertTrue(bool((out["rest_seen"]["air"] == 100).all()))
        self.assertTrue(bool(np.asarray(out["rest"]["lifted"]).all()))


if __name__ == "__main__":
    unittest.main()
