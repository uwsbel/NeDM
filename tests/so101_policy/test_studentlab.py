"""Tests of the studentlab student inputs and goal jumps (2026-10-22; ``viral.ViralConfig.cmd_in`` / ``img_hist`` /
``hist``, ``viral.expand_student``, ``viral.ReplayBuffer`` previous-sample links, ``joint_env.GoalJumpConfig`` in the
env, the online loop and ``joint_teacher.run_tasks``). CPU; the online-loop tests run ``viral.OnlineDAgger`` in the
kinematic world with a tiny student and fake views (``studentlab_ref.py``), and need jt_h110 and the dev task file.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_studentlab -v

``TestBitIdentity`` runs ``studentlab_ref.py`` against the base commit's code (``git archive``; skipped without git
or the commit) and against this tree, and compares every output exactly (default options and existing options).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from nedm.so101_policy import joint_env as JE
from nedm.so101_policy.vision import viral as VI

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import studentlab_ref as R  # noqa: E402

BASE = os.environ.get("NEDM_STUDENTLAB_BASE", "96ddde26d")       # the commit before the studentlab change
S28 = Path(os.environ.get("NEDM_STUDENTLAB_S28", "~/so101_studentlab_claude/ckpt/robot_v2_s28_it03000.pt")).expanduser()


def rand_inputs(st, B, gen, res=None):
    res = res or st.cfg.res
    C = len(st.cfg.cameras)
    img = torch.randn(B, C, 3, res, res, generator=gen)
    rob = torch.randn(B, 12, generator=gen)
    return img, rob


class TestDefaults(unittest.TestCase):
    def test_options_default_off(self):
        c = VI.ViralConfig()
        self.assertEqual((c.cmd_in, c.img_hist, c.hist), (False, False, 0))
        self.assertEqual(tuple(c.cmd_scale), VI.CMD_SCALE)
        self.assertEqual(VI.CMD_SCALE[:5], tuple(JE.LEAD))
        d = VI.cfg_dict(c)
        for k in VI.NEW_INPUT_KEYS:
            self.assertNotIn(k, d)                                     # old readers keep working
        self.assertEqual(VI.OnlineConfig().goal_jump, {})
        self.assertIsNone(JE.JointEnvConfig().goal_jump)
        self.assertNotIn("goal_jump", JE.JointEnvConfig().to_dict())
        self.assertIsNone(JE.goal_jump_config({}))
        self.assertIsNone(JE.goal_jump_config(""))

    def test_default_net_layout(self):
        st = R.tiny_student(0)
        self.assertEqual(st.head[0].in_features, st.fd + 12)
        self.assertFalse(any("cmd_s" in k for k in st.state_dict()))
        img, rob = rand_inputs(st, 3, torch.Generator().manual_seed(0))
        st.eval()
        f = st.features(img)
        ref = st.head(torch.cat((f, ((rob - st.robot_mean) / st.robot_std).clamp(-10, 10)), -1)).view(-1, 10, 6)
        self.assertTrue(torch.equal(st.predict(img, rob), ref))


@unittest.skipUnless(R.HAVE and shutil.which("git"), "jt_h110 / dev tasks / git missing")
class TestBitIdentity(unittest.TestCase):
    """The online loop (kinematic world, 14 decisions with training steps) and the evaluation actor give the same
    numbers as the base commit's code, bit for bit, with the new options off."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="studentlab_base_"))
        r = subprocess.run(f"git -C {REPO} archive {BASE} src | tar -x -C {cls.tmp}", shell=True, capture_output=True)
        cls.ok = r.returncode == 0 and (cls.tmp / "src").exists()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_ref(self, src, out, cfg, ocfg):
        env = dict(os.environ, PYTHONPATH=str(src), CUDA_VISIBLE_DEVICES="")
        r = subprocess.run([sys.executable, str(HERE / "studentlab_ref.py"), "--out", str(out), "--cfg",
                            json.dumps(cfg), "--ocfg", json.dumps(ocfg)], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return np.load(out)

    def check(self, tag, cfg, ocfg):
        if not self.ok:
            self.skipTest(f"git archive {BASE} failed")
        a = self.run_ref(self.tmp / "src", self.tmp / f"base_{tag}.npz", cfg, ocfg)
        b = self.run_ref(REPO / "src", self.tmp / f"new_{tag}.npz", cfg, ocfg)
        self.assertEqual(set(a.files), set(b.files))
        bad = [k for k in a.files if not np.array_equal(a[k], b[k], equal_nan=True)]
        self.assertEqual(bad, [], f"{tag}: outputs differ from {BASE}")
        self.assertGreater(int(a["samples"]), 0)

    def test_defaults(self):
        self.check("default", {}, {})

    def test_existing_options(self):
        self.check("opts", {"hist": 3, "lag": 2, "lag_range": [1, 3]},
                   {"qd_robot_p": 0.5, "q_bias_deg": [1, 2, 2, 2, 2, 0], "q_bias_p": 0.5, "qd_noise_zero_p": 0.3,
                    "qd_in_lo": 0.5, "qd_in_zero_p": 0.2, "nominal_share": 0.3})


class TestExpansion(unittest.TestCase):
    def check_same(self, st0, st1, B=4, seed=0):
        """st1 (expanded) = st0 on the same images / q, qd whatever the new inputs are (fp32)."""
        g = torch.Generator().manual_seed(seed)
        img, rob = rand_inputs(st0, B, g)
        st0.eval()
        # the same old inputs; any new history / cmd / previous images
        H0, H1 = int(st0.cfg.hist), int(st1.cfg.hist)
        old_hist = torch.randn(B, 12 * H0, generator=g)
        p0 = st0.predict(img, torch.cat((rob, old_hist), -1)) if H0 else st0.predict(img, rob)
        rob1 = torch.cat((rob, old_hist, torch.randn(B, 12 * (H1 - H0), generator=g)), -1)
        if st1.cfg.cmd_in:
            rob1 = torch.cat((rob1, 0.1 * torch.randn(B, 30, generator=g)), -1)
        prev = torch.randn_like(img) if st1.cfg.img_hist else None
        p1 = st1.eval().predict(img, rob1, prev)
        return float((p0 - p1).abs().max())

    def test_zero_init_keeps_the_chunk(self):
        st = R.tiny_student(0)
        a1 = VI.expand_student(st, hist=4, cmd_in=True, device="cpu")
        a4 = VI.expand_student(st, hist=4, cmd_in=True, img_hist=True, device="cpu")
        self.assertEqual((a1.cfg.hist, a1.cfg.cmd_in, a1.cfg.img_hist), (4, True, False))
        self.assertEqual(a4.head[0].in_features, 2 * st.fd + 12 * 5 + 30)
        self.assertLess(self.check_same(st, a1), 1e-5)
        self.assertLess(self.check_same(st, a4), 1e-5)
        chained = VI.expand_student(a1, img_hist=True, device="cpu")          # A1 -> A4: the same student as A4
        for k, v in a4.state_dict().items():
            self.assertTrue(torch.equal(v, chained.state_dict()[k]), k)
        self.assertIs(VI.expand_student(a4, hist=4, device="cpu"), a4)
        with self.assertRaises(ValueError):
            VI.expand_student(a4, hist=2, device="cpu")
        with self.assertRaises(ValueError):
            VI.expand_student(a4, cmd_in=False, device="cpu")
        h3 = VI.expand_hist(st, 3, "cpu")                                     # the old helper: same columns
        self.assertLess(self.check_same(st, h3), 1e-5)

    def test_new_inputs_move_the_output_after_training(self):
        """The new columns are trainable: one gradient step on them changes the prediction with new inputs."""
        st = VI.expand_student(R.tiny_student(0), hist=1, cmd_in=True, img_hist=True, device="cpu")
        g = torch.Generator().manual_seed(1)
        img, rob = rand_inputs(st, 4, g)
        rob = torch.cat((rob, torch.randn(4, 12 + 30, generator=g)), -1)
        st.train()
        loss, _ = st.loss(img, rob, torch.randn(4, 10, 6, generator=g), torch.ones(4, 10, dtype=torch.bool),
                          prev=torch.randn_like(img))
        loss.backward()
        lay = VI.head_layout(st.cfg, st.fd)
        gw = st.head[0].weight.grad
        for seg in ("robot", "cmd", "feat_prev"):
            a, b = lay[seg]
            self.assertGreater(float(gw[:, a:b].abs().sum()), 0.0, seg)

    def test_checkpoint_round_trip(self):
        st = VI.expand_student(R.tiny_student(0), hist=4, cmd_in=True, img_hist=True, device="cpu",
                               cmd_scale=(0.05, 0.1, 0.05, 0.05, 0.05, 0.2))
        with tempfile.TemporaryDirectory() as d:
            VI.save_student(st, Path(d) / "s.pt", it=1)
            ck = torch.load(Path(d) / "s.pt", weights_only=False)
            self.assertEqual((ck["cfg"]["cmd_in"], ck["cfg"]["img_hist"], ck["cfg"]["hist"]), (True, True, 4))
            st2, _ = VI.load_student(Path(d) / "s.pt", "cpu")
            self.assertEqual(st2.cfg, st.cfg)
            self.assertEqual(tuple(st2.cfg.cmd_scale), (0.05, 0.1, 0.05, 0.05, 0.05, 0.2))
            g = torch.Generator().manual_seed(2)
            img, rob = rand_inputs(st, 3, g)
            rob = torch.cat((rob, torch.randn(3, 48 + 30, generator=g)), -1)
            prev = torch.randn_like(img)
            self.assertTrue(torch.equal(st.eval().predict(img, rob, prev), st2.predict(img, rob, prev)))
            VI.save_student(R.tiny_student(0), Path(d) / "old.pt")             # options off: the old cfg keys
            ck = torch.load(Path(d) / "old.pt", weights_only=False)
            self.assertFalse(set(VI.NEW_INPUT_KEYS) & set(ck["cfg"]))

    def test_cmd_input_normalisation(self):
        st = VI.expand_student(R.tiny_student(0), cmd_in=True, device="cpu").eval()
        rob = torch.zeros(2, 12 + 30)
        rob[0, 12:18] = torch.tensor(VI.CMD_SCALE)
        rob[1, 12] = 1.0                                                     # 22 lead bands: clipped at 10
        r = st.robot(rob)
        self.assertTrue(torch.allclose(r[0, 12:18], torch.ones(6)))
        self.assertAlmostEqual(float(r[1, 12]), 10.0)

    @unittest.skipUnless(S28.exists(), "robot student s28 checkpoint missing")
    def test_s28_expansions(self):
        """The real robot student s28: A1 / A3 (hist 4 + cmd) and A4 (+ image history) give s28's chunk (fp32)."""
        st, _ = VI.load_student(S28, "cpu")
        self.assertEqual((st.cfg.hist, st.cfg.res), (0, 224))
        for kw in (dict(hist=4, cmd_in=True), dict(hist=4, cmd_in=True, img_hist=True)):
            e = VI.expand_student(st, device="cpu", **kw)
            self.assertLess(self.check_same(st, e, B=2), 1e-5, kw)


class TestReplayPrev(unittest.TestCase):
    def add(self, b, k, n, prev=None):
        return b.add(torch.full((n, 1, 2, 2, 3), k, dtype=torch.uint8), torch.zeros(n, 6), torch.zeros(n, 12),
                     torch.zeros(n, 5, 6), torch.zeros(n, VI.AUX_DIM), torch.ones(n, dtype=torch.bool), prev=prev)

    def test_prev_links_and_overwrite(self):
        b = VI.ReplayBuffer(5, 1, 2, "cpu", prev_link=True)
        s0 = self.add(b, 10, 2)                                          # episode starts: no previous sample
        self.assertFalse(bool(b.prev_valid(s0).any()))
        s1 = self.add(b, 11, 2, prev=s0)                                 # slots 2, 3 -> 0, 1
        self.assertTrue(bool(b.prev_valid(s1).all()))
        img, ok = b.prev_rgb(s1)
        self.assertEqual(img[:, 0, 0, 0, 0].tolist(), [10, 10])
        s2 = self.add(b, 12, 2, prev=torch.tensor([s1[0], -1]))          # slots 4, 0: overwrites s0[0]
        self.assertEqual(s2.tolist(), [4, 0])
        self.assertEqual(b.prev_valid(s1).tolist(), [False, True])       # s1[0]'s previous (slot 0) is gone
        self.assertEqual(b.prev_valid(s2).tolist(), [True, False])
        img, _ = b.prev_rgb(s1)
        self.assertEqual(img[:, 0, 0, 0, 0].tolist(), [11, 10])          # invalid: its own image
        s3 = self.add(b, 13, 4, prev=torch.cat((s2, s1)))                # slots 1-4: overwrites s1 and s2[0] itself
        self.assertEqual(b.prev_valid(s3).tolist(), [False, True, False, False])

    def test_off_by_default(self):
        b = VI.ReplayBuffer(5, 1, 2, "cpu")
        self.assertIsNone(b.sid)
        self.add(b, 1, 2, prev=torch.tensor([-1, -1]))                    # ignored without links


class FakeTask:
    def __init__(self, B, cube_xy, goal_xy, held=True, lifted=True, bottom=0.02, t=100):
        f = dict(dtype=torch.float64)
        self.F = dict(c=torch.cat((torch.as_tensor(cube_xy, **f), torch.full((B, 1), 0.04, **f)), -1),
                      held=torch.full((B,), held), bottom=torch.full((B,), bottom, **f))
        self.tr = SimpleNamespace(t=torch.full((B,), t, dtype=torch.long), lifted=torch.full((B,), lifted))
        self.goal_xy = torch.as_tensor(goal_xy, **f).clone()
        self.goal_yaw = torch.zeros(B, **f)

    def set_goal(self, idx, goal_xy, goal_yaw):
        self.goal_xy[idx] = goal_xy
        self.goal_yaw[idx] = goal_yaw


class TestGoalJumpUnit(unittest.TestCase):
    def setup(self, B=2000, **kw):
        cfg = JE.GoalJumpConfig(**{**dict(p=1.0, rate=1.0), **kw})
        gj = JE.GoalJump(cfg, B, "cpu", 0)
        start = torch.tensor([[0.20, -0.08]]).double().expand(B, 2)
        goal = torch.tensor([[0.20, 0.08]]).double().expand(B, 2).clone()      # carry +y
        gj.reset(torch.arange(B), start, goal)
        cube = goal + torch.tensor([0.0, -0.01]).double()                        # 10 mm before the goal
        return gj, FakeTask(B, cube, goal), cube, goal

    def test_geometry(self):
        gj, task, cube, goal = self.setup()
        idx = gj.apply(task, torch.ones(2000, dtype=torch.bool))
        self.assertEqual(len(idx), 2000)
        rel = cube - task.goal_xy                                               # cube - new goal
        along, across = rel[:, 1], rel[:, 0] * -1.0                             # carry dir (0, 1), perp (-1, 0)
        past = along > 0
        self.assertAlmostEqual(float(past.double().mean()), 0.8, delta=0.03)
        self.assertTrue(bool(((along.abs() >= 0.008 - 1e-12) & (along.abs() <= 0.028 + 1e-12)).all()))
        self.assertTrue(bool((across.abs() <= 0.008 + 1e-12).all()))
        self.assertTrue(torch.allclose(gj.d_last, along) and torch.allclose(gj.lat_last, across))
        self.assertTrue(bool(gj.just.all()) and bool((gj.n == 1).all()))
        self.assertTrue(torch.equal(task.goal_yaw, torch.zeros(2000, dtype=torch.float64)))

    def test_trigger_conditions(self):
        B = 7
        gj, task, cube, goal = self.setup(B)
        task.F["held"][1] = False
        task.tr.lifted[2] = False
        task.F["bottom"][3] = 0.004                                             # below 5 mm
        task.F["bottom"][4] = 0.046                                             # above 45 mm
        task.F["c"][5, 1] -= 0.025                                              # 35 mm from the goal
        live = torch.ones(B, dtype=torch.bool)
        live[6] = False
        idx = gj.apply(task, live)
        self.assertEqual(idx.tolist(), [0])
        g0 = task.goal_xy.clone()
        self.assertEqual(len(gj.apply(task, live).tolist()), 0)                # gap 1 s: none
        self.assertTrue(torch.equal(task.goal_xy, g0))
        self.assertFalse(bool(gj.just.any()))
        task.tr.t += 50                                                         # 1.0 s later
        task.F["c"][0, :2] = task.goal_xy[0]                                    # near the new goal again
        self.assertEqual(gj.apply(task, live).tolist(), [0])
        task.tr.t += 50
        task.F["c"][0, :2] = task.goal_xy[0]
        self.assertEqual(gj.apply(task, live).tolist(), [])                     # max_n 2
        self.assertEqual(int(gj.n[0]), 2)

    def test_episode_share_rate_and_workspace(self):
        gj, task, cube, goal = self.setup(4000, p=0.4)
        self.assertAlmostEqual(float(gj.on.double().mean()), 0.4, delta=0.03)
        idx = gj.apply(task, torch.ones(4000, dtype=torch.bool))
        self.assertTrue(bool(gj.on[idx].all()))
        gj, task, cube, goal = self.setup(4000, rate=0.35)
        n = len(gj.apply(task, torch.ones(4000, dtype=torch.bool)))
        self.assertAlmostEqual(n / 4000, 0.35, delta=0.03)
        from nedm.so101_policy.kin import PAN_AXIS_XY
        pan = torch.tensor(PAN_AXIS_XY).double()
        B = 200
        # carry outward to a goal at the outer edge: a goal moved AHEAD (cube before it) leaves the annulus: no jump;
        # moved back (cube past it): inside, every world jumps
        goal = (pan + torch.tensor([0.265, 0.0]).double()).expand(B, 2).clone()
        for before, n_want in ((1.0, 0), (0.0, B)):
            cfg = JE.GoalJumpConfig(p=1.0, rate=1.0, before_p=before, margin_m=0.0, d_lo=0.02, d_hi=0.028)
            gj = JE.GoalJump(cfg, B, "cpu", 0)
            gj.reset(torch.arange(B), goal - torch.tensor([0.1, 0.0]).double(), goal)
            task = FakeTask(B, goal.clone(), goal)
            self.assertEqual(len(gj.apply(task, torch.ones(B, dtype=torch.bool))), n_want, before)
            self.assertTrue(bool(JE.in_workspace(task.goal_xy).all()))
        # at the angular edge (74.5 deg), radial carry: a lateral offset toward > 75 deg is mirrored
        a = math.radians(74.5)
        goal = (pan + 0.2 * torch.tensor([math.cos(a), math.sin(a)]).double()).expand(B, 2).clone()
        cfg = JE.GoalJumpConfig(p=1.0, rate=1.0, before_p=0.0, margin_m=0.0, d_lo=0.008, d_hi=0.01, lat_m=0.008)
        gj = JE.GoalJump(cfg, B, "cpu", 0)
        gj.reset(torch.arange(B), pan.expand(B, 2) + 0.1 * (goal - pan) / 0.2, goal)
        task = FakeTask(B, goal.clone(), goal)
        # without the mirror ~40 % of the draws (lateral < -1.7 mm) would leave the annulus and be skipped
        self.assertEqual(len(gj.apply(task, torch.ones(B, dtype=torch.bool))), B)
        self.assertTrue(bool(JE.in_workspace(task.goal_xy).all()))
        self.assertTrue(bool((gj.lat_last >= -0.0017).all()))                 # no new goal beyond 75 deg
        rel = goal - task.goal_xy                                              # cube - new goal
        u = (goal - pan) / 0.2
        perp = torch.stack((-u[:, 1], u[:, 0]), -1)
        self.assertTrue(torch.allclose((rel * u).sum(-1), gj.d_last) and torch.allclose((rel * perp).sum(-1),
                                                                                        gj.lat_last))

    def test_spec(self):
        c = JE.GoalJumpConfig.from_spec("p=1,d=10:25,lat=8", max_n=1, before_p=0.0)
        self.assertEqual((c.p, c.max_n, c.before_p), (1.0, 1, 0.0))
        self.assertAlmostEqual(c.d_lo, 0.010)
        self.assertAlmostEqual(c.d_hi, 0.025)
        self.assertAlmostEqual(c.lat_m, 0.008)
        c = JE.GoalJumpConfig.from_spec("n=2,before=0.2,rate=0.5,near=25,z=4:40,gap=0.5")
        self.assertEqual((c.max_n, c.before_p, c.rate, c.gap_s), (2, 0.2, 0.5, 0.5))
        self.assertAlmostEqual(c.z_hi, 0.040)
        with self.assertRaises(ValueError):
            JE.GoalJumpConfig.from_spec("dd=3")
        with self.assertRaises(ValueError):
            JE.goal_jump_config({"p": 0.5, "bogus": 1})


@unittest.skipUnless(R.HAVE, "jt_h110 / dev task file missing")
class TestGoalJumpEnv(unittest.TestCase):
    def test_run_tasks_reads_the_moved_goal(self):
        """Fixed tasks (kinematic world): the JT observation's goal, the task's goal and the goal the student gets
        are the moved goal at every decision; jumps happen; the old outputs stay as they were without jumps."""
        from nedm.so101_policy import joint_teacher as JT
        jt = JE.JointTeacher(str(R.JT_PATH), "cpu")
        t = R.load(12)
        seen = []

        class Spy:                      # a "student" that drives with the JT and records the goal it is given
            def reset(self, B):
                pass

            def __call__(self, ws, goal_pose, k):
                seen.append((goal_pose.clone(), env.task.goal_xy.clone(), env.obs[:, 38:40].double().clone()))
                return jt.targets(env.obs, ws.q)

        env = JT.make_env("kin", 12, jt.codec, "cpu", goal_jump=dict(p=1.0, rate=1.0))
        r = JT.run_tasks(env, t, jt, Spy(), np.ones(12, np.int64), record=True)
        self.assertGreater(int((r["goal_jumps"] > 0).sum()), 6)
        for gp, g, og in seen:
            self.assertTrue(torch.equal(gp[:, :2], g))
            self.assertLess(float((og - g).abs().max()), 1e-6)                 # float32 observation
        self.assertLess(float(np.abs(r["row_obs"][:, 38:40] - r["row_goal_xy"]).max()), 1e-6)
        moved = np.abs(r["goal_xy_final"] - r["goal_xy"]).max(-1) > 1e-6
        self.assertTrue(np.array_equal(moved, r["goal_jumps"] > 0))
        env0 = JT.make_env("kin", 12, jt.codec, "cpu")
        r0 = JT.run_tasks(env0, t, jt, record=True)
        self.assertNotIn("goal_jumps", r0)
        self.assertNotIn("row_goal_xy", r0)


@unittest.skipUnless(R.HAVE, "jt_h110 / dev task file missing")
class TestOnlineLoop(unittest.TestCase):
    """``OnlineDAgger`` in the kinematic world (staggered clocks, automatic resets, a small ring buffer)."""

    def run_loop(self, cfg, ocfg, n_dec, envs=16):
        st = R.tiny_student(0)
        st = VI.expand_student(st, device="cpu", **cfg)
        od, views = R.make_online(st, ocfg, 0, envs)
        A = torch.arange(envs)
        od.env.reset()                  # what the first act() does (so the episode ids are readable before it)
        od._draw(A), od._lag_reset(A), od._bias_reset(A), od._qd_reset(A)
        rec = []
        for it in range(n_dec):
            od.it = it
            pre = dict(ep=od.env.ep_count.clone(), q=od.env.ws.q.float().clone(),
                       u=None if od.env.u_exe is None else od.env.u_exe.float().clone(), goal=od.env.task.goal_xy.clone(),
                       obs_goal=od.env.obs[:, 38:40].double().clone(), ptr=od.buf.ptr, count=getattr(od.buf, "count", 0),
                       just=None if od.env.goal_jump is None else od.env.goal_jump.just.clone(),
                       prev_slot=od.prev_slot.clone())
            od.act()
            last = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in od.last.items()}
            rec.append(dict(pre=pre, last=last, render_goal=views.goals[-1].clone(),
                            slots=(pre["ptr"] + torch.arange(envs)) % od.buf.cap))
            if od.buf.n >= 16 and it % 3 == 0:
                od.train_steps()
        return od, rec

    def test_history_cmd_and_images_reset_at_episode_starts(self):
        H = 2
        od, rec = self.run_loop(dict(hist=H, cmd_in=True, img_hist=True), dict(horizon=40, buffer=40), 30)
        N, P = 16, 12 * (1 + H)
        starts = 0
        for t, d in enumerate(rec):
            rob, fresh = d["last"]["robot"], d["last"]["fresh"]
            ep = d["pre"]["ep"]
            exp_fresh = torch.ones(N, dtype=torch.bool) if t == 0 else ep != rec[t - 1]["pre"]["ep"]
            self.assertTrue(torch.equal(fresh, exp_fresh), t)
            starts += int(fresh.sum())
            now = rob[:, :12]
            # robot history: the same episode's earlier decisions, the episode's first decision before that
            for w in range(N):
                s = t
                while s > 0 and torch.equal(rec[s - 1]["pre"]["ep"][w], ep[w]) and t > 0:
                    s -= 1
                for h in range(1, H + 1):
                    src = max(t - h, s)
                    self.assertTrue(torch.equal(rob[w, 12 * h:12 * (h + 1)], rec[src]["last"]["robot"][w, :12]),
                                    (t, w, h))
            # cmd: 0 at an episode start, else the last decision's executed targets - q now
            cmd = rob[:, P:P + 30]
            self.assertEqual(float(cmd[fresh].abs().max()) if bool(fresh.any()) else 0.0, 0.0)
            if t > 0:
                want = (d["pre"]["u"] - d["pre"]["q"][:, None, :]).reshape(N, 30)
                self.assertLess(float((cmd[~fresh] - want[~fresh]).abs().max()), 1e-6, t)
                # previous images: the last decision's own images inside an episode, the current ones at a start
                prev, cur, last_rgb = d["last"]["prev_rgb"], d["last"]["rgb"], rec[t - 1]["last"]["rgb"]
                self.assertTrue(torch.equal(prev[~fresh], last_rgb[~fresh]))
                self.assertTrue(torch.equal(prev[fresh], cur[fresh]))
            self.assertTrue(torch.equal(now, torch.cat((d["pre"]["q"], now[:, 6:]), -1)))
        self.assertGreater(starts, N + 10)                                     # resets happened at many decisions
        # buffer links: (t, w) -> (t - 1, w) in the same episode, valid until that slot is overwritten
        cap, total = od.buf.cap, len(rec) * N
        checked = 0
        for t in range(max(1, len(rec) - 2), len(rec)):
            sl = rec[t]["slots"]
            fresh = rec[t]["last"]["fresh"]
            exp = ~fresh & torch.tensor([(t - 1) * N + w >= total - cap for w in range(N)])
            self.assertTrue(torch.equal(od.buf.prev_valid(sl), exp), t)
            img, ok = od.buf.prev_rgb(sl)
            self.assertTrue(torch.equal(img[ok], rec[t - 1]["last"]["rgb"][ok]))
            checked += int(ok.sum())
        self.assertGreater(checked, 0)
        self.assertEqual(od.buf.robot.shape[1], P + 30)

    def test_goal_jump_moves_render_teacher_and_aux_goal_together(self):
        od, rec = self.run_loop({}, dict(horizon=400, buffer=4000, teacher_share=1.0, noise_share=0.0,
                                         goal_jump=dict(p=1.0, rate=1.0), hold_cap_s=6.0), 70)
        self.assertEqual(od.env.tcfg.hold_cap_s, 6.0)
        self.assertGreater(od.env.goal_jump.jumps, 4)
        moved = 0
        for t, d in enumerate(rec):
            g = d["pre"]["goal"]
            self.assertLess(float((d["render_goal"][:, :2] - g).abs().max()), 1e-6, t)    # rendered plate (fp32)
            self.assertLess(float((d["pre"]["obs_goal"] - g).abs().max()), 1e-6, t)          # JT observation
            aux = od.buf.aux[d["slots"]][:, :2].double()                                   # (never overwritten)
            self.assertLess(float((aux - g).abs().max()), 1e-6, t)                           # aux target
            if d["pre"]["just"] is not None and t > 0:
                j = d["pre"]["just"]
                moved += int(j.sum())
                self.assertTrue(bool((g[j] != rec[t - 1]["pre"]["goal"][j]).any(-1).all()))
                pv = d["pre"]["prev_slot"][j]
                self.assertFalse(bool(od.buf.valid2[pv[pv >= 0]].any()))                 # no 2nd-half label
        self.assertGreater(moved, 0)


if __name__ == "__main__":
    unittest.main()
