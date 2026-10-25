"""Tests of the kinematic arm-table end (``joint_env.ArmTableConfig``, real2sim round v9c, 2026-10-25): off by default
(no ``arm_table`` key in the env record; a rule that never fires = off bit for bit: states, observations, rewards, done
flags, resets); on = the world's arm_table flag is exactly ``arm_table_rule`` at the stepped state (lowest gripper /
moving-jaw point at q + tau x qd below the margin), the episode ends with ``world_arm_table`` (unless world_invalid
fired first) and pays the failure cost, the world's own state record is not changed; the config parser; the round v9c
teacher configs. CPU, a few seconds; the env tests need the kinematic world's files and the build9 development tasks
(skipped without them).

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_arm_table_rule -v
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import joint_env as JE
from nedm.so101_policy import task as T

ROOT = Path(__file__).resolve().parents[2]
TASKS = Path("~/so101_policy/runs/viral/tasks/dev100.npz").expanduser()
HAVE_KIN = Path(JE.JointEnvConfig().kin_points).expanduser().exists() and \
    Path(JE.JointEnvConfig().geom_path).expanduser().exists() and TASKS.exists()
torch.set_num_threads(4)


def tasks(n, first=0):
    from nedm.so101_policy.vision.demos import load_tasks
    return load_tasks(TASKS, np.arange(first, first + n))


def codec():
    return JE.Codec(np.zeros(18), np.ones(18))


def env_for(t, arm_table=None, B=24, horizon=60):
    from nedm.so101_policy.vision.demos import start_state
    cfg = JE.JointEnvConfig(world="kin", num_envs=B, device="cpu", starts="", horizon=horizon, auto_reset=False,
                            stagger=False, seed=3, arm_table=arm_table)
    env = JE.JointVecEnv(cfg, codec())
    env.reset_tasks(torch.arange(B), start_state(t, "cpu"), torch.as_tensor(t["goal_xy"]),
                    torch.as_tensor(t["goal_yaw"]), torch.as_tensor(t["seeds"]))
    return env


def drive(env, n=10, down=0.012, hook=None):
    """n decisions of targets that move every arm joint by a fixed random step and push the wrist / elbow down (the
    TCP sinks toward the table): one tensor of states, observations, rewards, done flags and end codes per decision."""
    g = torch.Generator().manual_seed(7)
    step = 0.01 * torch.randn(env.B, 6, generator=g, dtype=torch.float64)
    step[:, 1] += down                  # shoulder lift forward
    step[:, 3] += down                  # wrist flex down
    out = []
    for _ in range(n):
        q = env.ws.q.to(torch.float64)
        tg = q[:, None, :] + step[:, None, :] * torch.arange(1, JE.STEPS + 1, dtype=torch.float64)[None, :, None]
        obs, rew, done, ex = env.step(None, targets=tg, step_hook=hook)
        out.append(torch.cat((env.ws.q, env.ws.qd, env.ws.cube_pos, obs.double(), rew.double()[:, None],
                              done.double()[:, None], env.task.tr.end.double()[:, None]), -1))
    return out


class TestConfig(unittest.TestCase):
    def test_config(self):
        c = JE.JointEnvConfig()
        self.assertIsNone(c.arm_table)
        self.assertNotIn("arm_table", c.to_dict())                            # off: the old record
        for off in (None, False, {}, "", "{}", "null"):
            self.assertIsNone(JE.arm_table_config(off), msg=repr(off))
            self.assertIsNone(JE.JointEnvConfig(arm_table=off).arm_table)
        on = JE.JointEnvConfig.from_dict(dict(arm_table=dict(margin_m=0.001, tau_s=0.03, grasp_only=False, note="n")))
        self.assertEqual(on.arm_table, JE.ArmTableConfig(0.001, 0.03, False))
        self.assertEqual(on.to_dict()["arm_table"], dict(margin_m=0.001, tau_s=0.03, grasp_only=False))
        self.assertEqual(JE.JointEnvConfig.from_dict(json.loads(json.dumps(on.to_dict()))), on)
        self.assertEqual(JE.arm_table_config('{"tau_s": 0.0}'), JE.ArmTableConfig(0.0, 0.0, True))
        for bad in (dict(margin_m=0.05), dict(tau_s=-0.01), dict(tau_s=0.5), dict(margin_m=True),
                    dict(tau_s="0.01"), dict(margin_m=float("nan")), dict(grasp_only=1), dict(k=1)):
            with self.assertRaises(ValueError, msg=repr(bad)):
                JE.arm_table_config(bad)

    def test_v9c_teacher_configs(self):
        d = ROOT / "configs/so101_policy/ppo/r2s"
        base = json.loads((d / "teacher_restart_v9a_fo.json").read_text())
        raw = json.loads((d / "teacher_restart_v9c_at.json").read_text())
        self.assertEqual({k: v for k, v in raw.items() if k not in ("note", "arm_table")},
                         {k: v for k, v in base.items() if k != "note"})       # everything of v9a_fo
        c = JE.JointEnvConfig.from_dict(raw)
        self.assertEqual(c.arm_table, JE.ArmTableConfig(0.0, 0.02, True))
        s10 = json.loads((d / "joint_ppo_v3_kl02_std10.json").read_text())
        s25 = json.loads((d / "joint_ppo_v3_kl02_std25.json").read_text())
        self.assertEqual({k: v for k, v in s10.items() if k not in ("note", "init_std", "std_floor", "std_cap",
                                                                   "std_reset")},
                         {k: v for k, v in s25.items() if k not in ("note", "init_std", "std_floor", "std_cap",
                                                                   "std_reset")})
        self.assertEqual((s10["init_std"], s10["std_reset"]), ([0.1, 0.1], [0.1, 0.1]))


@unittest.skipUnless(HAVE_KIN, "kinematics files / development tasks missing")
class TestEnv(unittest.TestCase):
    def test_never_firing_rule_is_off(self):
        """margin -5 mm, tau 0 (the kin world's floor guard keeps the jaws above 0): the rollout of the env without
        the option, bit for bit, and no rule end."""
        t = tasks(24)
        ref = drive(env_for(t))
        e = env_for(t, dict(margin_m=-0.005, tau_s=0.0))
        got = drive(e)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(ref, got)))
        self.assertEqual(int(e.arm_table_n), 0)

    def test_rule_ends(self):
        """margin 15 mm, tau 20 ms: the world flag after the env step = the world's own flag (the kin world's lowest-point
        rule) OR arm_table_rule at the stepped state; a live world ends with world_arm_table (or world_invalid) at the
        first step where either fires; the rule count; worlds the rule never touched = the rollout without the option;
        a decision that ends a world by the rule alone pays the failure cost."""
        t = tasks(24)
        at = JE.ArmTableConfig(0.015, 0.02, grasp_only=False)
        e = env_for(t, at)
        rec, own = [], []
        wstep = e.world.step

        def world_step(u):
            ws = wstep(u)
            own.append(ws.arm_table.clone())
            return ws
        e.world.step = world_step

        def hook(env, j):
            want = JE.arm_table_rule(env.ws.q, env.ws.qd, env.kin, at)
            rec.append((want.clone(), env.ws.arm_table.clone(), env.task.tr.end.clone()))
        ref = drive(env_for(t), hook=None)
        got = drive(e, hook=hook)
        want = torch.stack([w for w, _, _ in rec], 1)                           # [B, steps]
        flag = torch.stack([f for _, f, _ in rec], 1)
        ownf = torch.stack(own, 1)
        ends = torch.stack([c for _, _, c in rec], 1)
        self.assertTrue(torch.equal(flag, want | ownf))
        live_before = torch.cat((torch.ones(e.B, 1, dtype=torch.bool), ends[:, :-1] == 0), 1)
        self.assertGreater(int((want & live_before & ~ownf).any(1).sum()), 3)  # the rule alone ends some worlds
        self.assertGreater(int((~want.any(1)).sum()), 3)                         # some worlds never fire
        for b in range(e.B):
            ev = (want[b] | ownf[b]) & live_before[b]
            if ev.any():
                k = int(ev.nonzero()[0])
                self.assertIn(int(ends[b, k]), (T.EC["world_arm_table"], T.EC["world_invalid"]), msg=b)
        self.assertEqual(int(e.arm_table_n), int((want & live_before).sum()))
        never = ~want.any(1)
        for a, b in zip(ref, got):
            self.assertTrue(torch.equal(a[never], b[never]))
        rew_col, done_col, end_col = -3, -2, -1
        n_checked = 0
        for k, (a, b) in enumerate(zip(ref, got)):
            new = (b[:, done_col] == 1) & (b[:, end_col] == T.EC["world_arm_table"]) & (a[:, end_col] == 0)
            if k > 0:
                new &= got[k - 1][:, end_col] == 0
            if new.any():
                n_checked += int(new.sum())
                self.assertTrue(bool((b[new, rew_col] < a[new, rew_col] - 1.0).all()), msg=k)
        self.assertGreater(n_checked, 0)

    def test_grasp_only(self):
        """grasp_only: worlds already lifted (forced here) are never touched by the rule (= the rollout without the
        option); the other worlds = the rollout with grasp_only False."""
        t = tasks(24)
        runs = {}
        for name, at in (("off", None), ("all", dict(margin_m=0.015, tau_s=0.02, grasp_only=False)),
                         ("grasp", dict(margin_m=0.015, tau_s=0.02, grasp_only=True))):
            e = env_for(t, at)
            e.task.tr.lifted[:12] = True
            runs[name] = (drive(e), e)
        lifted_differs = any(not torch.equal(a[:12], b[:12]) for a, b in zip(runs["off"][0], runs["all"][0]))
        self.assertTrue(lifted_differs)                                          # the rule acts there without it
        for a, b in zip(runs["off"][0], runs["grasp"][0]):
            self.assertTrue(torch.equal(a[:12], b[:12]))
        for a, b in zip(runs["all"][0], runs["grasp"][0]):
            self.assertTrue(torch.equal(a[12:], b[12:]))
        self.assertLess(int(runs["grasp"][1].arm_table_n), int(runs["all"][1].arm_table_n))

    def test_world_record_unchanged(self):
        """The rule makes a new WorldState: the world's returned states keep their own flags (not changed in place)."""
        t = tasks(24)
        e = env_for(t, dict(margin_m=0.015, tau_s=0.02, grasp_only=False))
        seen = []
        orig = e.world.step

        def wstep(u):
            ws = orig(u)
            seen.append((ws, ws.arm_table.clone()))
            return ws
        e.world.step = wstep
        changed = []

        def hook(env, j):
            changed.append(not torch.equal(env.ws.arm_table, seen[-1][1]))
        drive(e, n=4, hook=hook)
        self.assertTrue(all(torch.equal(ws.arm_table, f) for ws, f in seen))
        self.assertTrue(any(changed))


if __name__ == "__main__":
    unittest.main()
