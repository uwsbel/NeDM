"""Tests of the round v12 table-drop track (2026-10-26): the action-layer floor raise of the joint env
(``joint_env.JointEnvConfig.floor_raise``), the fixed table plane of the student's table frame
(``table_bias.TableBiasConfig.plane``), the flat observed-frame floor of the online loop
(``viral.OnlineConfig.student_floor`` "observed", ``viral.online_floor_raise``), the labelling teacher's floor check
(``vision_viral.check_jt_floor``), the robot deploy of such students (``hw_lib.student_floor_mode`` flat,
``student_table_frame``, ``floor_frame_problems``, ``deploy.json`` table_frame, ``robot_test.sh live-tilt``) and the
v12 teacher configs.

- Config (CPU): parsers, records (off = no key), the configs = teacher_roll_v8.json + floor_raise.
- FloorGuard (CPU, kin points): a zero lift = the old guard bit for bit; a flat lift = the floor shifted (the robot's
  --table-z rule); the tilted lift IS the robot's flat arm-frame floor seen through the table frame map: the lowest
  point of a pose minus raise(grasp xy) = the lowest point of its reading Phi(pose) (to 1e-9 m), the same rows are
  clamped, and the guarded targets keep the observed-frame floor.
- Env (CPU, kin world + dev tasks): off = no record key; a flat raise = the env with floor_z set (bit for bit); a
  tilted raise executes ``safety(..., floor_lift=raise)`` at every 20 ms step; make_env / run_tasks records; the
  floor creep measure (``JointVecEnv.floor_gap``, ``run_tasks(floor_gap=True)``: no change of the rollout).
- Config also: ``joint_teacher.resolve_floor_raise`` (the eval / collect ``--floor-raise auto`` default = the JT's
  own) and an online plane beyond the floor raise's limit refused by ``OnlineConfig.from_dict``.
- BitIdentity (CPU, git): the base commit (``NEDM_V12_FLOOR_BASE``, default cf0dd6687 = claude/r2s-v11) and this code
  give the same arrays with the options off: the online loop + evaluation actor (``studentlab_ref.py``: defaults, the
  C family, the v8 random table planes, the new keys present but off) and joint-env rollouts (actions and targets,
  the floor active).
- Online (CPU, jt_h110 + dev tasks, kin world): student_floor observed -> the online env's floor raise = -plane,
  planes fixed in every world, the checkpoint record; every executed target (student and teacher slots) keeps the
  robot's flat floor in the observed frame; the default keeps the v8 rule; the JT floor check.
- HwDeploy (CPU, kin points): a v12 student deploys flat with table_frame true (no --q-offset-deg / live-tilt), old v8
  students keep the plane floor, an unknown student_floor is refused; deploy.json; the bundle's live-tilt refusal.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_floor_raise_v12 -v
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import joint_env as JE
from nedm.so101_policy import table_bias as TB

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import studentlab_ref as R  # noqa: E402

BASE = os.environ.get("NEDM_V12_FLOOR_BASE", "cf0dd6687")       # claude/r2s-v11 = r2s-v12 before the floor raise
PLANE_CAL = (-0.004681260916979778, -0.03640954226125585, 0.01836320115536056)   # calib_v1 scene.table_plane
KIN_OK = Path(JE.JointEnvConfig().kin_points).expanduser().exists() and \
    Path(JE.JointEnvConfig().geom_path).expanduser().exists()
f64 = torch.float64
torch.set_num_threads(4)


def kin():
    from nedm.so101_policy.vec_env import kinematics
    return kinematics(JE.JointEnvConfig().kin_points, "cpu")


def poses(k, n=1200, seed=0, z=(0.0, 0.06)):
    """Top-down IK poses of the grasp point over the task area (0.12-0.30 m from the pan axis, pan +-60 deg), the
    grasp point ``z`` above the sim table (the fingertips reach below it), jaw 0-0.6 rad."""
    g = torch.Generator().manual_seed(seed)
    r = 0.12 + 0.18 * torch.rand(n, generator=g, dtype=f64)
    ph = (torch.rand(n, generator=g, dtype=f64) * 2 - 1) * math.radians(60)
    pos = torch.stack((0.0388 + r * torch.cos(ph), r * torch.sin(ph),
                       z[0] + (z[1] - z[0]) * torch.rand(n, generator=g, dtype=f64)), -1)
    yaw = ph + (torch.rand(n, generator=g, dtype=f64) - 0.5) * math.pi / 2
    q5, ok, _, _ = k.ik_topdown(None, pos, yaw)
    jaw = 0.6 * torch.rand(n, 1, generator=g, dtype=f64)
    return torch.cat((q5, jaw), -1)[ok]


def down_targets(q, seed=1, scale=0.03):
    """Targets around q pushed toward the table (shoulder lift forward, wrist down) plus noise, jaw changed."""
    g = torch.Generator().manual_seed(seed)
    d = scale * (torch.rand(q.shape, generator=g, dtype=f64) - 0.5)
    d[:, 1] += scale
    d[:, 3] += scale
    return q + d


# ---- configs ---------------------------------------------------------------------------------------------------------
class Config(unittest.TestCase):
    def test_floor_raise_config(self):
        c = JE.JointEnvConfig()
        self.assertIsNone(c.floor_raise)
        self.assertNotIn("floor_raise", c.to_dict())                              # off: the old record
        for off in (None, False, 0, 0.0, "", " ", [], (), [0, 0, 0], "0,0,0", [-0.0, 0.0, 0.0]):
            self.assertIsNone(JE.floor_raise_config(off), msg=repr(off))
            self.assertIsNone(JE.JointEnvConfig(floor_raise=off).floor_raise, msg=repr(off))
        self.assertEqual(JE.floor_raise_config(0.01), (0.01, 0.0, 0.0))
        self.assertEqual(JE.floor_raise_config("0.01"), (0.01, 0.0, 0.0))
        self.assertEqual(JE.floor_raise_config("0.004,0.036,-0.018"), (0.004, 0.036, -0.018))
        on = JE.JointEnvConfig.from_dict(dict(floor_raise=[0.005, 0, 0]))
        self.assertEqual(on.floor_raise, (0.005, 0.0, 0.0))
        self.assertEqual(on.to_dict()["floor_raise"], [0.005, 0.0, 0.0])
        self.assertEqual(JE.JointEnvConfig.from_dict(json.loads(json.dumps(on.to_dict()))), on)
        for bad in (True, [1, 2], [0.0, 0.0, float("nan")], 0.05, [0.0, 0.5, 0.0], [0.0, 0.0, -0.3], "x", [True, 0, 0],
                    {"a": 1}):
            with self.assertRaises(ValueError, msg=repr(bad)):
                JE.floor_raise_config(bad)
        # the raise of a lower table = -plane (a 10 mm lower table: the floor 10 mm up in the true frame)
        self.assertEqual(JE.raise_of_table((-0.01, 0.0, 0.0)), (0.01, 0.0, 0.0))
        self.assertEqual(JE.raise_of_table(PLANE_CAL), tuple(-c for c in PLANE_CAL))
        self.assertIsNone(JE.raise_of_table(None))
        self.assertIsNone(JE.raise_of_table((0.0, 0.0, 0.0)))
        self.assertTrue(all(math.copysign(1.0, v) > 0 for v in JE.raise_of_table((-0.01, 0.0, 0.0))[1:]))  # no -0.0

    def test_table_bias_plane(self):
        c = TB.TableBiasConfig.from_dict({"plane": list(PLANE_CAL), "note": "n"})
        self.assertEqual((c.plane, c.p), (PLANE_CAL, 1.0))
        self.assertEqual(c.to_dict(), {"p": 1.0, "x_ref_m": 0.25, "plane": list(PLANE_CAL)})  # no unused ranges
        self.assertEqual(TB.TableBiasConfig.from_dict(c.to_dict()), c)
        self.assertEqual(TB.TableBiasConfig.from_dict({"plane": [0, 0, 0], "p": 1}).plane, (0.0, 0.0, 0.0))
        old = {"p": 0.7, "drop_mm": [0.0, 22.0], "slope_x": [-55.0, 25.0], "slope_y": [-10.0, 35.0], "x_ref_m": 0.25}
        self.assertEqual(TB.TableBiasConfig.from_dict(old).to_dict(), old)        # the v8 record unchanged
        self.assertEqual(TB.TableBiasConfig.from_dict({**old, "plane": []}).to_dict(), old)
        for bad in ({"plane": [0, 0]}, {"plane": 0.01}, {"plane": "0,0,0"}, {"plane": [True, 0, 0]},
                    {"plane": [-0.01, 0, 0], "p": 0.5}, {"plane": [-0.01, 0, 0], "drop_mm": [5, 5]},
                    {"plane": [-0.01, 0, 0], "slope_y": [0, 0]}, {"plane": [-0.1, 0, 0]}, {"plane": [0, 0.5, 0]},
                    {"plane": [float("inf"), 0, 0]}):
            with self.assertRaises(ValueError, msg=repr(bad)):
                TB.TableBiasConfig.from_dict(bad)
        g = torch.Generator().manual_seed(5)
        s0 = g.get_state()
        pl, on = TB.sample(7, c, g)                                               # no random numbers
        self.assertTrue(torch.equal(g.get_state(), s0))
        self.assertTrue(bool(on.all()))
        self.assertTrue(torch.equal(pl, torch.tensor(PLANE_CAL, dtype=f64).expand(7, 3)))
        pl[0, 0] = 1.0                                                            # own memory per call
        self.assertEqual(float(TB.sample(2, c, g)[0][0, 0]), PLANE_CAL[0])
        o = TB.TableBiasConfig.from_dict(old)                                     # random planes: draws as before
        g1, g2 = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
        a, b = TB.sample(50, o, g1), TB.sample(50, o, g2)
        self.assertTrue(torch.equal(a[0], b[0]) and not torch.equal(g1.get_state(), s0))

    def test_online_config(self):
        from nedm.so101_policy.vision import viral as VI
        self.assertEqual(VI.OnlineConfig().student_floor, "true")
        self.assertIsNone(VI.online_floor_raise(VI.OnlineConfig()))
        o = VI.OnlineConfig.from_dict({"table_bias": {"plane": list(PLANE_CAL)}, "student_floor": "observed"})
        self.assertEqual(VI.online_floor_raise(o), tuple(-c for c in PLANE_CAL))
        o = VI.OnlineConfig.from_dict({"table_bias": {"plane": [-0.01, 0, 0]}, "student_floor": "observed"})
        self.assertEqual(VI.online_floor_raise(o), (0.01, 0.0, 0.0))
        o = VI.OnlineConfig.from_dict({"table_bias": {"plane": [0, 0, 0]}, "student_floor": "observed"})
        self.assertIsNone(VI.online_floor_raise(o))                               # the 0 mm arm: no raise
        o = VI.OnlineConfig.from_dict({"table_bias": {"plane": [-0.01, 0, 0]}})   # a plane with the v8 floor rule
        self.assertIsNone(VI.online_floor_raise(o))
        self.assertEqual(VI.OnlineConfig.from_dict({"student_floor": "true"}).student_floor, "true")
        for bad in ({"student_floor": "observed"}, {"student_floor": "observed", "table_bias": {"note": "x"}},
                    {"student_floor": "observed", "table_bias": {"p": 1.0, "drop_mm": [10, 10]}},
                    {"student_floor": "flat"}, {"student_floor": "observed", "table_bias": {"plane": [0.0, 0.0]}},
                    {"student_floor": "observed", "table_bias": {"plane": [-0.045, 0.0, 0.0]}}):   # (review F6)
            with self.assertRaises(ValueError, msg=repr(bad)):
                VI.OnlineConfig.from_dict(bad)
        # a plane beyond the floor raise's limit is still a valid v8-rule plane (the table_bias limit is 0.06 m)
        self.assertIsNone(VI.online_floor_raise(VI.OnlineConfig.from_dict({"table_bias": {"plane": [-0.045, 0, 0]}})))

    def test_resolve_floor_raise(self):
        """v12 review finding 4: the JT commands' --floor-raise "auto" = the JT's own (none: off), "0" = off, another
        value than the JT's own: a WARNING."""
        from nedm.so101_policy import joint_teacher as JT

        class J:
            def __init__(self, env):
                self.env_rec = {"env": env} if env is not None else {}
        old, new = J({"floor_guard": True}), J({"floor_raise": [0.015, 0.0, 0.0]})
        msgs = []
        for spec in ("", "auto", "AUTO", None):
            self.assertIsNone(JT.resolve_floor_raise(spec, old, msgs.append))
            self.assertIsNone(JT.resolve_floor_raise(spec, J(None), msgs.append))
            self.assertEqual(JT.resolve_floor_raise(spec, new, msgs.append), (0.015, 0.0, 0.0))
        self.assertFalse([m for m in msgs if "WARNING" in m])
        msgs.clear()
        for off in ("0", "off", "none", "0.0"):
            self.assertIsNone(JT.resolve_floor_raise(off, old, msgs.append))
        self.assertFalse(msgs)
        self.assertIsNone(JT.resolve_floor_raise("0", new, msgs.append))                 # explicit off: warned
        self.assertTrue(msgs and "WARNING" in msgs[-1])
        msgs.clear()
        self.assertEqual(JT.resolve_floor_raise("0.01", new, msgs.append), (0.01, 0.0, 0.0))
        self.assertTrue(msgs and "WARNING" in msgs[-1])
        msgs.clear()
        self.assertEqual(JT.resolve_floor_raise("0.015,0,0", new, msgs.append), (0.015, 0.0, 0.0))
        self.assertFalse(msgs)
        self.assertEqual(JT.jt_floor_raise(new), (0.015, 0.0, 0.0))
        self.assertIsNone(JT.jt_floor_raise(old))
        from importlib import util
        spec = util.spec_from_file_location("vvt", REPO / "scripts/so101_policy/vision_viral_teacher.py")
        m = util.module_from_spec(spec)
        spec.loader.exec_module(m)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            m.main(["eval", "--help"])

    def test_v12_teacher_configs(self):
        d = REPO / "configs/so101_policy/ppo/r2s"
        base = json.loads((d / "teacher_roll_v8.json").read_text())
        want = {"fr05": (0.005, 0.0, 0.0), "fr10": (0.010, 0.0, 0.0), "fr15": (0.015, 0.0, 0.0),
                "frpl": JE.raise_of_table(PLANE_CAL)}
        cal = Path("~/so101_r2s_claude/calib/calib_v1.json").expanduser()
        if cal.exists():                                   # the measured plane of the calibration file itself
            cp = json.loads(cal.read_text())["scene"]["table_plane"]["coef"]
            self.assertEqual(tuple(float(v) for v in cp), PLANE_CAL)
        for tag, fr in want.items():
            raw = json.loads((d / f"teacher_roll_v12_{tag}.json").read_text())
            self.assertEqual({k: v for k, v in raw.items() if k not in ("note", "floor_raise")},
                             {k: v for k, v in base.items() if k != "note"}, tag)          # everything of D's env
            c = JE.JointEnvConfig.from_dict(raw)
            self.assertEqual(c.floor_raise, fr, tag)
            self.assertIsNone(c.grasp_shift)
            self.assertIsNone(c.arm_table)


# ---- the floor guard ---------------------------------------------------------------------------------------------------
@unittest.skipUnless(KIN_OK, "kin points / task geometry missing")
class FloorGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.k = kin()
        cls.q = poses(cls.k)
        cls.u = down_targets(cls.q)

    def test_zero_lift_is_the_old_guard(self):
        a = JE.floor_guard(self.u, self.q, self.k, JE.FLOOR)
        b = JE.floor_guard(self.u, self.q, self.k, JE.FLOOR, lift=(0.0, 0.0, 0.0))
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, self.u))                                  # the floor acts on these targets
        tlo, thi = self.k.tlo, self.k.thi
        lead = torch.tensor(JE.LEAD, dtype=f64)
        s0 = JE.safety(self.u, self.q, tlo, thi, lead, self.k, JE.FLOOR)
        self.assertTrue(torch.equal(s0, JE.safety(self.u, self.q, tlo, thi, lead, self.k, JE.FLOOR, None)))

    def test_flat_lift_is_a_shifted_floor(self):
        """A flat raise h at the grasp point = the floor FLOOR + h (the robot's --table-z h)."""
        for h in (0.005, 0.010, 0.015):
            a = JE.floor_guard(self.u, self.q, self.k, JE.FLOOR + h)
            b = JE.floor_guard(self.u, self.q, self.k, JE.FLOOR, lift=(h, 0.0, 0.0))
            self.assertLess(float((a - b).abs().max()), 1e-6, h)        # bisection decisions at 1e-18 m rounding
            self.assertGreater(int((a != self.u).any(-1).sum()), int((JE.floor_guard(
                self.u, self.q, self.k, JE.FLOOR) != self.u).any(-1).sum()))          # a raise clamps more rows

    def test_lift_is_the_observed_flat_floor(self):
        """The robot keeps its floor flat in the frame it reads (arm frame z 0); the student's reading is Phi(q)
        (``table_bias.observe``: the gripper moved by plane(grasp xy), same rotation). For every pose: lowest point
        of u minus raise(grasp xy of u) (raise = -plane) = the lowest point of Phi(u), exactly; so the same targets
        are clamped and the guarded targets keep the observed-frame floor."""
        k, q, u = self.k, self.q, self.u
        for plane in (PLANE_CAL, (-0.010, 0.0, 0.0), (-0.015, -0.04, 0.0), (-0.002, 0.03, -0.02)):
            n = len(q)
            pl = torch.tensor(plane, dtype=f64).expand(n, 3)
            uo, ru = TB.observe(k, u, pl)
            qo, rq = TB.observe(k, q, pl)
            ok = (ru < 1e-9) & (rq < 1e-9)
            self.assertGreater(float(ok.double().mean()), 0.99)
            lift = JE.raise_of_table(plane)
            lt = JE._low(k, u, q, lift=lift)
            lo = JE._low(k, uo, qo)
            self.assertLess(float((lt - lo)[ok].abs().max()), 1e-9, plane)
            lqt, lqo = JE._low(k, q, q, lift=lift), JE._low(k, qo, qo)
            self.assertLess(float((lqt - lqo)[ok].abs().max()), 1e-9, plane)
            gt = JE.floor_guard(u, q, k, JE.FLOOR, lift=lift)                   # the env (true frame)
            go = JE.floor_guard(uo, qo, k, JE.FLOOR)                            # the robot (its frame)
            ct, co = (gt != u).any(-1), (go != uo).any(-1)
            self.assertGreater(int(ct.sum()), 50)
            self.assertTrue(torch.equal(ct[ok], co[ok]), plane)                # the same rows clamped
            gto, rg = TB.observe(k, gt, pl)                                       # the env's result, read
            m = ok & (rg < 1e-9)
            ref = torch.minimum(lqo, torch.full_like(lqo, JE.FLOOR))
            at_q = (gt == torch.cat((q[:, :5], u[:, 5:]), -1)).all(-1)        # (lam 0: the guard's own exception)
            self.assertTrue(bool((JE._low(k, gto, qo)[m & ~at_q] >= ref[m & ~at_q] - 1e-8).all()), plane)
            # the pull-back line differs (straight in true vs read joints): grasp points of the clamped rows agree
            # to well below a millimetre
            dp = (k.fk(gto)[0] - k.fk(go)[0]).norm(dim=-1)[m & ct]
            self.assertLess(float(dp.max()), 5e-4, plane)

    def test_tilted_lift_guard(self):
        """With a tilted raise every guarded target is at or above min(FLOOR, the measured pose's height) above the
        raise at its own grasp point, except the guard's own exception (as without a raise): lam 0 = the measured
        arm with the COMMANDED jaw (the guard does not limit the jaw) is already lower -> that pose; unclamped rows
        are returned unchanged."""
        lift = JE.raise_of_table(PLANE_CAL)
        g = JE.floor_guard(self.u, self.q, self.k, JE.FLOOR, lift=lift)
        lq = JE._low(self.k, self.q, self.q, lift=lift)
        lu = JE._low(self.k, self.u, self.q, lift=lift)
        ref = torch.minimum(lq, torch.full_like(lq, JE.FLOOR))
        at_q = (g == torch.cat((self.q[:, :5], self.u[:, 5:]), -1)).all(-1)
        low_g = JE._low(self.k, g, self.q, lift=lift)
        self.assertTrue(bool((low_g[~at_q] >= ref[~at_q] - 1e-8).all()))
        self.assertLess(float((ref - low_g)[at_q].max()), 5e-4)               # (sub-millimetre: the jaw command)
        keep = (lu >= JE.FLOOR - 1e-9) | (lu >= ref - 1e-9)
        self.assertTrue(torch.equal(g[keep], self.u[keep]))
        self.assertTrue(torch.equal(JE.lift_at(self.k, self.u, lift), JE.lift_at(self.k, self.u[:, :5], lift)))


# ---- the joint env -------------------------------------------------------------------------------------------------
@unittest.skipUnless(KIN_OK and R.TASKS.exists(), "kin files / development tasks missing")
class Env(unittest.TestCase):
    @staticmethod
    def env_for(t, floor_raise=None, B=24, horizon=60):
        from nedm.so101_policy.vision.demos import start_state
        cfg = JE.JointEnvConfig(world="kin", num_envs=B, device="cpu", starts="", horizon=horizon, auto_reset=False,
                                stagger=False, seed=3, floor_raise=floor_raise)
        env = JE.JointVecEnv(cfg, JE.Codec(np.zeros(18), np.ones(18)))
        env.reset_tasks(torch.arange(B), start_state(t, "cpu"), torch.as_tensor(t["goal_xy"]),
                        torch.as_tensor(t["goal_yaw"]), torch.as_tensor(t["seeds"]))
        return env

    @staticmethod
    def drive(env, n=10, down=0.012, record=None):
        """n decisions of targets that step every arm joint by a fixed random amount and push the gripper toward the
        table (the floor acts); per decision one tensor of states, observations, rewards, done flags, end codes."""
        g = torch.Generator().manual_seed(7)
        step = 0.01 * torch.randn(env.B, 6, generator=g, dtype=f64)
        step[:, 1] += down
        step[:, 3] += down
        out = []
        for _ in range(n):
            q = env.ws.q.to(f64)
            tg = q[:, None, :] + step[:, None, :] * torch.arange(1, JE.STEPS + 1, dtype=f64)[None, :, None]
            if record is not None:
                record.append(tg.clone())
            obs, rew, done, ex = env.step(None, targets=tg)
            out.append(torch.cat((env.ws.q, env.ws.qd, env.ws.cube_pos, obs.double(), rew.double()[:, None],
                                  done.double()[:, None], env.task.tr.end.double()[:, None]), -1))
        return out

    @classmethod
    def setUpClass(cls):
        cls.t = R.load(24)

    def test_off(self):
        e = self.env_for(self.t)
        self.assertIsNone(e.raise_z)
        self.assertIsNone(e.raise_lift)
        self.assertNotIn("floor_raise", e.resolved()["env"])
        self.assertNotIn("floor_raise", e.table_info())

    def test_flat_raise_is_floor_z(self):
        """floor_raise (h, 0, 0) = the env with its flat floor height floor_z = h (the robot's --table-z h), bit for
        bit; and it changes the rollout (the floor acts)."""
        h = 0.010
        a = self.env_for(self.t, (h, 0.0, 0.0))
        self.assertEqual((a.raise_z, a.raise_lift), (h, None))
        b = self.env_for(self.t)
        b.floor_z = h
        ra, rb = self.drive(a), self.drive(b)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(ra, rb)))
        r0 = self.drive(self.env_for(self.t))
        self.assertFalse(all(torch.equal(x, y) for x, y in zip(ra, r0)))
        self.assertEqual(a.resolved()["env"]["floor_raise"], [h, 0.0, 0.0])
        self.assertEqual(a.table_info()["floor_raise"], [h, 0.0, 0.0])

    def test_tilted_raise_executes_lifted_safety(self):
        """A tilted raise: the target of every 20 ms world step = ``safety(target, measured q, ..., FLOOR,
        floor_lift=raise)``; the floor acts in some steps."""
        lift = JE.raise_of_table(PLANE_CAL)
        e = self.env_for(self.t, lift)
        self.assertEqual((e.raise_z, e.raise_lift), (None, lift))
        sent = []
        wstep = e.world.step

        def world_step(u):
            sent.append((e.ws.q.to(f64).clone(), u.clone()))
            return wstep(u)
        e.world.step = world_step
        tg = []
        self.drive(e, n=8, record=tg)
        n_act = 0
        for i, (q, u) in enumerate(sent):
            raw = tg[i // JE.STEPS][:, i % JE.STEPS]
            want = JE.safety(raw, q, e.tlo, e.thi, e.lead, e.kin, JE.FLOOR, None, floor_lift=lift)
            self.assertTrue(torch.equal(u, want), i)
            n_act += int((want != JE.safety(raw, q, e.tlo, e.thi, e.lead, e.kin, JE.FLOOR)).any(-1).sum())
        self.assertGreater(n_act, 10)

    def test_make_env_and_run_tasks_record(self):
        from nedm.so101_policy import joint_teacher as JT
        e = JT.make_env("kin", 4, JE.Codec(np.zeros(18), np.ones(18)), "cpu", floor_raise=0.01)
        self.assertEqual(e.cfg.floor_raise, (0.01, 0.0, 0.0))
        self.assertIsNone(JT.make_env("kin", 4, JE.Codec(np.zeros(18), np.ones(18)), "cpu").cfg.floor_raise)
        if not R.HAVE:
            self.skipTest("jt_h110 missing")
        jt = JE.JointTeacher(str(R.JT_PATH), "cpu")
        t = R.load(4)
        r = JT.run_tasks(JT.make_env("kin", 4, jt.codec, "cpu", horizon=40, floor_raise=list(PLANE_CAL)), t, jt)
        self.assertEqual(json.loads(str(r["floor_raise"])), list(PLANE_CAL))
        r0 = JT.run_tasks(JT.make_env("kin", 4, jt.codec, "cpu", horizon=40), t, jt)
        self.assertNotIn("floor_raise", r0)
        self.assertNotIn("floor_gap_prelift", r0)                                 # off: no key
        self.assertNotIn("floor_gap_mm_p10", JT.summarize(r0))
        # v12 review finding 2: the floor creep before the lift (floor_gap); the rollout itself does not change
        r1 = JT.run_tasks(JT.make_env("kin", 4, jt.codec, "cpu", horizon=40), t, jt, floor_gap=True)
        for k in r0:
            if k not in ("wall_s", "student_s"):
                self.assertTrue(np.array_equal(np.asarray(r0[k]), np.asarray(r1[k])), k)
        g = r1["floor_gap_prelift"]
        self.assertEqual(g.shape, (4,))
        s1 = JT.summarize(r1)
        self.assertEqual(s1["floor_gap_n"], int(np.isfinite(g).sum()))
        self.assertAlmostEqual(s1["floor_gap_mm_p50"], float(np.median(g[np.isfinite(g)]) * 1e3))

    def test_floor_gap(self):
        """JointVecEnv.floor_gap = the floor guard's own measure minus the floor: 0 for a pose whose lowest point is
        on the floor; a flat raise h moves it by -h; a tilted raise is measured at the pose's grasp point."""
        e0 = self.env_for(self.t)
        q = e0.ws.q.to(f64)
        g0 = e0.floor_gap()
        k = e0.kin
        self.assertTrue(torch.allclose(g0, k.lowest_point(q, exact=False) - JE.FLOOR, atol=1e-12))
        h = 0.010
        e1 = self.env_for(self.t, (h, 0.0, 0.0))
        self.assertTrue(torch.allclose(e1.floor_gap(q), g0 - h, atol=1e-12))
        lift = JE.raise_of_table(PLANE_CAL)
        e2 = self.env_for(self.t, lift)
        self.assertTrue(torch.allclose(e2.floor_gap(q), g0 - JE.lift_at(k, q, lift), atol=1e-12))
        # the guard keeps a target's own gap >= min(0, the measured pose's gap)
        tg = down_targets(q, scale=0.2)
        u = JE.safety(tg, q, e2.tlo, e2.thi, None, k, JE.FLOOR, None, floor_lift=lift)
        gu = JE._low(k, u, q, None, lift) - JE.FLOOR
        self.assertTrue(bool((gu >= torch.minimum(e2.floor_gap(q), torch.zeros_like(gu)) - 1e-6).all()))


# ---- bit identity with the base commit -----------------------------------------------------------------------------
ENV_REF = r'''
import json, sys
import numpy as np, torch
torch.set_num_threads(1)
from nedm.so101_policy import joint_env as JE
from nedm.so101_policy.vision.demos import load_tasks, start_state
t = load_tasks(sys.argv[2], np.arange(24))
over = json.loads(sys.argv[3])
out = {}
for mode in ("targets", "actions"):
    cfg = JE.JointEnvConfig(world="kin", num_envs=24, device="cpu", starts="", horizon=80, auto_reset=False,
                            stagger=False, seed=3, **over)
    env = JE.JointVecEnv(cfg, JE.Codec(np.zeros(18), np.full(18, 0.02)))
    env.reset_tasks(torch.arange(24), start_state(t, "cpu"), torch.as_tensor(t["goal_xy"]),
                    torch.as_tensor(t["goal_yaw"]), torch.as_tensor(t["seeds"]))
    g = torch.Generator().manual_seed(7)
    step = 0.01 * torch.randn(24, 6, generator=g, dtype=torch.float64)
    step[:, 1] += 0.012
    step[:, 3] += 0.012
    for k in range(12):
        q = env.ws.q.to(torch.float64)
        if mode == "targets":
            tg = q[:, None, :] + step[:, None, :] * torch.arange(1, 6, dtype=torch.float64)[None, :, None]
            obs, rew, done, ex = env.step(None, targets=tg)
        else:
            a = torch.randn(24, 18, generator=g, dtype=torch.float64)
            a[:, 1] += 1.0
            a[:, 3] += 1.0
            obs, rew, done, ex = env.step(a)
        for n_, v in (("q", env.ws.q), ("qd", env.ws.qd), ("cube", env.ws.cube_pos), ("obs", obs), ("rew", rew),
                      ("done", done), ("end", env.task.tr.end), ("u_exe", env.u_exe)):
            out[f"{mode}_{n_}_{k}"] = v.detach().cpu().numpy()
np.savez(sys.argv[1], **out)
'''


@unittest.skipUnless(R.HAVE and KIN_OK and shutil.which("git"), "jt_h110 / dev tasks / kin files / git missing")
class BitIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="v12floor_base_"))
        r = subprocess.run(f"git -C {REPO} archive {BASE} src | tar -x -C {cls.tmp}", shell=True, capture_output=True)
        cls.ok = r.returncode == 0 and (cls.tmp / "src").exists()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_sub(self, src, args):
        env = dict(os.environ, PYTHONPATH=str(src), CUDA_VISIBLE_DEVICES="")
        r = subprocess.run([sys.executable] + args, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])

    def compare(self, tag, a, b):
        a, b = np.load(a), np.load(b)
        self.assertEqual(set(a.files), set(b.files))
        bad = [k for k in a.files if not np.array_equal(a[k], b[k], equal_nan=True)]
        self.assertEqual(bad, [], f"{tag}: outputs differ from {BASE}")

    def online(self, tag, cfg, ocfg, ocfg_new=None):
        if not self.ok:
            self.skipTest(f"git archive {BASE} failed")
        out = []
        for name, src, oc in (("base", self.tmp / "src", ocfg), ("new", REPO / "src", ocfg_new or ocfg)):
            f = self.tmp / f"{name}_{tag}.npz"
            self.run_sub(src, [str(HERE / "studentlab_ref.py"), "--out", str(f), "--cfg", json.dumps(cfg), "--ocfg",
                               json.dumps(oc)])
            out.append(f)
        self.compare(tag, *out)

    def test_online_defaults(self):
        self.online("default", {}, {})

    def test_online_c_family_and_v8_table(self):
        """The C configs' options + the v8 random table planes; also with the new keys present but off."""
        oc = {"dyn": {"p": 0.8, "d_lo": 0, "d_hi": 1, "s_lo": 0.3, "s_hi": 1.0, "tau_lo": 0.02, "tau_hi": 0.15,
                      "sag_max_deg": 1.5, "i_time_s": 0.2, "i_max_deg": 3.0}, "dyn_student_only": True,
              "relabel_cmd_speed": True, "qd_robot_p": 0.5, "qd_window": 3, "roll_hi_deg": 98.0,
              "table_bias": {"p": 0.7, "drop_mm": [0.0, 22.0], "slope_x": [-55.0, 25.0], "slope_y": [-10.0, 35.0]}}
        cfg = {"hist": 2, "cmd_in": True, "lastcmd": True, "lag_range": [1, 4]}
        self.online("cfam_v8", cfg, oc, {**oc, "student_floor": "true",
                                         "table_bias": {**oc["table_bias"], "plane": []}})

    def test_joint_env_rollouts(self):
        """Joint-env rollouts (kin world; targets and PPO actions; the floor active): the base commit's arrays with
        floor_raise absent, None and zero."""
        if not self.ok:
            self.skipTest(f"git archive {BASE} failed")
        ref = self.tmp / "env_base.npz"
        script = self.tmp / "env_ref.py"
        script.write_text(ENV_REF)
        self.run_sub(self.tmp / "src", [str(script), str(ref), str(R.TASKS), "{}"])
        for tag, over in (("absent", {}), ("none", {"floor_raise": None}), ("zero", {"floor_raise": [0, 0, 0]})):
            f = self.tmp / f"env_new_{tag}.npz"
            self.run_sub(REPO / "src", [str(script), str(f), str(R.TASKS), json.dumps(over)])
            self.compare(tag, ref, f)


# ---- the online loop -----------------------------------------------------------------------------------------------
@unittest.skipUnless(R.HAVE and KIN_OK, "jt_h110 / dev tasks / kin files missing")
class Online(unittest.TestCase):
    OBS = {"table_bias": {"plane": list(PLANE_CAL)}, "student_floor": "observed"}

    def run_loop(self, over, n_dec=6, envs=16):
        od, _ = R.make_online(R.tiny_student(0), over, 0, envs)
        sent = []
        wstep = od.env.world.step

        def world_step(u):
            sent.append((od.env.ws.q.to(f64).clone(), u.to(f64).clone()))
            return wstep(u)
        od.env.world.step = world_step
        for it in range(n_dec):
            od.it = it
            od.act()
        return od, sent

    def test_observed_floor(self):
        """student_floor observed: the env's floor raise = -plane; every world has the plane; every executed target
        (student and teacher slots) keeps the robot's flat floor in the observed frame; the checkpoint record."""
        from nedm.so101_policy.vision import viral as VI
        od, sent = self.run_loop(self.OBS)
        lift = JE.raise_of_table(PLANE_CAL)
        self.assertEqual(od.env.cfg.floor_raise, lift)
        self.assertEqual(od.env.raise_lift, lift)
        self.assertTrue(bool(od.tb_on.all()))
        self.assertTrue(torch.equal(od.tb_plane, torch.tensor(PLANE_CAL, dtype=f64).expand_as(od.tb_plane)))
        rec = od.ckpt_record()["v8_online"]
        self.assertEqual(rec["student_floor"], "observed")
        self.assertEqual(rec["table_bias"]["plane"], list(PLANE_CAL))
        self.assertEqual(rec["table_bias"]["p"], 1.0)
        k = od.env.kin
        n_clamped, n_rows = 0, 0
        for q, u in sent:
            n = len(q)
            pl = torch.tensor(PLANE_CAL, dtype=f64).expand(n, 3)
            uo, ru = TB.observe(k, u, pl)
            qo, rq = TB.observe(k, q, pl)
            m = (ru < 1e-9) & (rq < 1e-9) & torch.isfinite(u).all(-1) & torch.isfinite(q).all(-1)
            lq = JE._low(k, qo, qo)
            ref = torch.minimum(lq, torch.full_like(lq, JE.FLOOR))
            m = m & ~(u[:, :5] == q[:, :5]).all(-1)          # (the guard's lam-0 exception: the measured arm)
            self.assertTrue(bool((JE._low(k, uo, qo)[m] >= ref[m] - 1e-8).all()))
            n_rows += int(m.sum())
            n_clamped += int((JE._low(k, uo, qo)[m] < JE.FLOOR + 1e-6).sum())
        self.assertGreater(n_rows, 0.9 * len(sent) * od.env.B)
        self.assertEqual(VI.online_floor_raise(od.o), lift)

    def test_v8_rule_default(self):
        """A fixed plane with the default student_floor: the v8 rule (the floor above the TRUE table, no raise), no
        student_floor key in the record."""
        od, _ = self.run_loop({"table_bias": {"plane": list(PLANE_CAL)}}, n_dec=2)
        self.assertIsNone(od.env.cfg.floor_raise)
        self.assertNotIn("student_floor", od.ckpt_record()["v8_online"])
        od0, _ = self.run_loop({}, n_dec=1, envs=4)
        self.assertEqual(od0.ckpt_record(), {})

    def test_check_jt_floor(self):
        sys.path.insert(0, str(REPO / "scripts" / "so101_policy"))
        import vision_viral as VV

        class J:
            def __init__(self, fr):
                self.path = "jt.pt"
                self.env_rec = {"env": {} if fr is None else {"floor_raise": fr}}
        lift = JE.raise_of_table(PLANE_CAL)
        self.assertIsNone(VV.check_jt_floor(J(None), None))                       # D's teacher, no raise: ok
        self.assertEqual(VV.check_jt_floor(J(list(lift)), lift), lift)
        self.assertEqual(VV.check_jt_floor(J([0.01, 0, 0]), (0.01, 0.0, 0.0)), (0.01, 0.0, 0.0))
        for jf, fr in ((None, lift), (list(lift), None), ([0.01, 0, 0], (0.015, 0.0, 0.0)),
                       ([0.01, 0, 0], lift)):
            with self.assertRaises(SystemExit, msg=(jf, fr)):
                VV.check_jt_floor(J(jf), fr)
            self.assertEqual(VV.check_jt_floor(J(jf), fr, allow=True), JE.floor_raise_config(jf))
        jt = JE.JointTeacher(str(R.JT_PATH), "cpu")                               # a real record without the key
        self.assertIsNone(VV.jt_floor_raise(jt))
        src = (REPO / "scripts/so101_policy/vision_viral.py").read_text()
        i = src.index("def cmd_online")
        body = src[i:src.index("\ndef ", i + 10)]
        self.assertIn("check_jt_floor(jt, o_raise, a.allow_floor_raise_mismatch)", body)
        self.assertLess(body.index("check_jt_floor("), body.index("VI.OnlineDAgger("))


# ---- robot deploy ----------------------------------------------------------------------------------------------------
HW = REPO / "scripts" / "so101_policy" / "real2sim" / "hw"
V12_REC = {"v8_online": {"table_bias": {"p": 1.0, "x_ref_m": 0.25, "plane": list(PLANE_CAL)}, "table_render": True,
                         "student_floor": "observed"}}


def hw_lib():
    sys.path.insert(0, str(HW))
    import hw_lib as H
    return H


class HwDeploy(unittest.TestCase):
    def test_floor_mode(self):
        H = hw_lib()
        from test_real2sim_hw_deploy import V8_REC
        m, why = H.student_floor_mode(V12_REC)
        self.assertEqual(m, "flat")
        self.assertIn("student_floor observed", why)
        self.assertIn("fixed plane", why)
        self.assertEqual(H.student_floor_mode(V8_REC)[0], "plane")                 # old v8 students: unchanged
        self.assertEqual(H.student_floor_mode({"v8_online": {**V8_REC["v8_online"], "student_floor": "true"}})[0],
                         "plane")
        self.assertEqual(H.student_floor_mode({})[0], "flat")
        with self.assertRaises(ValueError):
            H.student_floor_mode({"v8_online": {**V12_REC["v8_online"], "student_floor": "flat"}})
        self.assertTrue(H.student_table_frame(V12_REC))
        self.assertTrue(H.student_table_frame(V8_REC))
        self.assertFalse(H.student_table_frame({}))
        p0 = {"v8_online": {"table_bias": {**V8_REC["v8_online"]["table_bias"], "p": 0.0}}}
        self.assertFalse(H.student_table_frame(p0))
        self.assertEqual(H.floor_frame_problems("flat", [0, -2.086, 0, 0, 0]), [])              # C1 / D: as before
        self.assertTrue(H.floor_frame_problems("flat", [0, -2.086, 0, 0, 0], table_frame=True))
        self.assertEqual(H.floor_frame_problems("flat", [0, 0, 0, 0, 0], table_frame=True), [])
        # review finding 3: a v12 flat-floor student refuses a --table-z other than 0; older students: as before
        self.assertTrue(H.student_floor_observed(V12_REC))
        self.assertFalse(H.student_floor_observed(V8_REC))
        self.assertFalse(H.student_floor_observed({}))
        self.assertEqual(H.floor_frame_problems("flat", None, table_frame=True, observed=True, table_z=0.0), [])
        bad = H.floor_frame_problems("flat", None, table_frame=True, observed=True, table_z=-0.0126)
        self.assertEqual(len(bad), 1)
        self.assertIn("12.6 mm lower", bad[0])
        self.assertEqual(len(H.floor_frame_problems("flat", [0, -2.086, 0, 0, 0], table_frame=True, observed=True,
                                                    table_z=-0.0126)), 2)
        self.assertEqual(H.floor_frame_problems("flat", None, observed=False, table_z=-0.0126), [])   # C1 / D
        self.assertEqual(H.floor_problems("flat", "flat", H.GuardConfig.from_dict(H.eval_guard("flat"))), [])

    def test_floor_setup(self):
        H = hw_lib()
        from test_real2sim_hw import _load_runner
        R_ = _load_runner()
        calib = {"scene": {"table_plane": {"coef": list(PLANE_CAL)}}}

        def a(mode="live", **kw):
            d = dict(mode=mode, floor_mode="auto", table_plane="", table_z=0.0, allow_floor_mismatch=False, calib="c")
            d.update(kw)
            return argparse.Namespace(**d)
        G = H.GuardConfig.from_dict
        f = R_.floor_setup(a(), calib, V12_REC, G(H.eval_guard("flat")))
        self.assertEqual((f["mode"], f["student_mode"], f["plane"], f["problems"]), ("flat", "flat", None, []))
        with self.assertRaises(SystemExit):                                         # the plane floor: refused
            R_.floor_setup(a(table_plane="calib"), calib, V12_REC, G(H.eval_guard("plane")))
        for allow in (False, True):                                                 # live-tilt: no override
            with self.assertRaises(SystemExit) as cm:
                R_.floor_setup(a(q_offset_deg="0,-2.086,0,0,0", allow_floor_mismatch=allow), calib, V12_REC,
                               G(H.eval_guard("flat")))
            self.assertIn("no override", str(cm.exception))
        f = R_.floor_setup(a("dry", q_offset_deg="0,-2.086,0,0,0"), calib, V12_REC, G(H.eval_guard("flat")))
        self.assertTrue(any("table frame" in p for p in f["problems"]))
        f = R_.floor_setup(a(q_offset_deg="0,-2.086,0,0,0"), calib, {}, G(H.eval_guard("flat")))   # D: as before
        self.assertEqual(f["problems"], [])
        # review finding 3: --table-z other than 0 for a v12 flat-floor student: live refused (no override), dry warns
        for allow in (False, True):
            with self.assertRaises(SystemExit) as cm:
                R_.floor_setup(a(table_z=-0.0126, allow_floor_mismatch=allow), calib, V12_REC, G(H.eval_guard("flat")))
            self.assertIn("no override", str(cm.exception))
            self.assertIn("--table-z", str(cm.exception))
        f = R_.floor_setup(a("dry", table_z=-0.0126), calib, V12_REC, G(H.eval_guard("flat")))
        self.assertTrue(any("--table-z" in p for p in f["problems"]))
        f = R_.floor_setup(a(table_z=-0.0126), calib, {}, G(H.eval_guard("flat")))           # C1 / D: as before
        self.assertEqual(f["problems"], [])

    @unittest.skipUnless(KIN_OK, "kin points missing")
    def test_deploy_json_and_bundle(self):
        H = hw_lib()
        from nedm.so101_policy.vision import viral as VI
        from test_real2sim_hw import _load_runner, tiny_student
        from test_real2sim_hw_deploy import calib_file, tiny_v8
        R_ = _load_runner()
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            plain = tiny_student(t / "plain.pt")
            v8 = tiny_v8(t / "v8.pt", plain)
            st, _ = VI.load_student(plain, "cpu")
            v12 = t / "v12.pt"
            VI.save_student(st, v12, **V12_REC)
            cal = calib_file(t / "calib_v1.json", PLANE_CAL)
            for ck, mode, tf in ((plain, "flat", False), (v8, "plane", True), (v12, "flat", True)):
                out = t / f"deploy_{ck.stem}.json"
                R_.main(["deploy-json", "--student", str(ck), "--calib", str(cal), "--out", str(out)])
                d = json.loads(out.read_text())
                self.assertEqual((d["floor_mode"], d["table_frame"], d["eval_guard"]), (mode, tf, H.eval_guard(mode)),
                                 ck.stem)
                self.assertEqual(d["table_plane"], "calib" if mode == "plane" else "")
            if shutil.which("script") is None:
                return
            env = dict(os.environ, PYTHONPATH="", PYTHON=sys.executable, CUDA_VISIBLE_DEVICES="",
                       SO101_KIN_POINTS=str(H.default_kin_points()))
            out = t / "bundle_v12"
            r = subprocess.run(["bash", str(HW / "make_orin_bundle.sh"), str(v12), str(out), str(cal)], env=env,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(json.loads((out / "assets" / "deploy.json").read_text())["floor_mode"], "flat")
            b = t / "shimbin"
            b.mkdir()
            (b / "python").write_text(f"#!/usr/bin/env bash\ncase \"${{1:-}}\" in *run_student_real.py) echo SHIM_RUN; "
                                      f"exit 99 ;; esac\nexec {sys.executable} \"$@\"\n")
            (b / "python").chmod(0o755)
            env2 = dict(os.environ, PATH=f"{b}:{os.environ['PATH']}", CONDA_DEFAULT_ENV="lerobot", RUNS=str(b / "runs"))
            r = subprocess.run(["script", "-qec", "./robot_test.sh live-tilt 4", "/dev/null"], cwd=out, env=env2,
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn("trained in the table frame", r.stdout)
            self.assertNotIn("SHIM_RUN", r.stdout)


if __name__ == "__main__":
    unittest.main()
