"""Tests of build9 (Markov joint teacher ``nedm.so101_policy.joint_env`` / ``joint_teacher`` and the camera student
``nedm.so101_policy.vision.viral``). CPU, kinematic world, no renderer process:

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_vision_viral

The episode tests need the current teacher checkpoint and the build9 development task file (skipped without them).
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import joint_env as JE

TEACHER = Path("~/so101_policy/runs/c1ppo/ckpt/nC1_s62_b_model_00171.pt").expanduser()
TASKS = Path("~/so101_policy/runs/viral/tasks/dev100.npz").expanduser()
HAVE = TEACHER.exists() and TASKS.exists()


class TestCodec(unittest.TestCase):
    def test_quadratic_round_trip(self):
        g = torch.Generator().manual_seed(0)
        c = torch.randn(64, 3, 6, generator=g, dtype=torch.float64)
        D = torch.einsum("jk,nkd->njd", JE.PHI, c)                       # exact quadratics over the 5 steps
        codec = JE.Codec.fit(D)
        a = codec.encode(D)
        self.assertEqual(tuple(a.shape), (64, JE.ACT_DIM))
        self.assertLess(float((codec.decode(a) - D).abs().max()), 1e-10)
        q = torch.randn(64, 6, generator=g, dtype=torch.float64)
        self.assertLess(float((codec.targets(a, q) - (q[:, None] + D)).abs().max()), 1e-10)

    def test_safety_layer(self):
        q = torch.zeros(2, 6, dtype=torch.float64)
        u = torch.tensor([[1.0, -1.0, 0.01, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, -5.0]], dtype=torch.float64)
        from nedm.so101_policy.kin import THI, TLO
        lo, hi = torch.tensor(TLO, dtype=torch.float64), torch.tensor(THI, dtype=torch.float64)
        lead = torch.tensor(JE.LEAD, dtype=torch.float64)
        v = JE.safety(u, q, lo, hi, lead)
        self.assertAlmostEqual(float(v[0, 0]), JE.LEAD[0])
        self.assertAlmostEqual(float(v[0, 1]), -JE.LEAD[1])
        self.assertAlmostEqual(float(v[0, 2]), 0.01)
        self.assertAlmostEqual(float(v[0, 5]), 1.0)                       # the jaw has no lead band
        self.assertAlmostEqual(float(v[1, 5]), TLO[5])                    # the target range


class TestLenient(unittest.TestCase):
    def F(self, xy, yaw_g=0.0, held=False, jaw=0.7, tcp_z=0.08):
        n = len(xy)
        c = torch.tensor([[x, y, 0.015] for x, y in xy], dtype=torch.float64)
        return dict(c=c, held=torch.full((n,), held), on_table=torch.ones(n, dtype=torch.bool),
                    tilt=torch.zeros(n, dtype=torch.float64), speed=torch.zeros(n, dtype=torch.float64),
                    jaw=torch.full((n,), jaw, dtype=torch.float64),
                    tcp=torch.tensor([[0.0, 0.0, tcp_z]] * n, dtype=torch.float64))

    def test_square_any_yaw(self):
        from nedm.so101_policy.task import TaskConfig
        cfg = TaskConfig()
        g = torch.tensor([[0.2, 0.0]] * 4, dtype=torch.float64)
        lifted = torch.ones(4, dtype=torch.bool)
        # 20 mm along x (inside the 25 mm half side, outside the strict 15 mm), 30 mm (outside), corner 24/24 mm
        F = self.F([(0.22, 0.0), (0.23, 0.0), (0.224, 0.024), (0.2, 0.0)])
        ok = JE.lenient_now(F, cfg, g, torch.zeros(4, dtype=torch.float64), lifted)
        self.assertEqual(ok.tolist(), [True, False, True, True])
        # the block turned by 45 deg: the corner point (24, 24) mm is outside (33.9 mm along the block x), the point
        # 30 mm along the world x is inside (21.2 / -21.2 mm in the block frame)
        ok = JE.lenient_now(F, cfg, g, torch.full((4,), math.pi / 4, dtype=torch.float64), lifted)
        self.assertEqual(ok.tolist(), [True, True, False, True])
        # held / jaw closed / gripper too low / no lift: never lenient
        for kw in (dict(held=True), dict(jaw=0.1), dict(tcp_z=0.05)):
            self.assertFalse(bool(JE.lenient_now(self.F([(0.2, 0.0)], **kw), cfg, g[:1], torch.zeros(1),
                                                 lifted[:1]).any()))
        self.assertFalse(bool(JE.lenient_now(self.F([(0.2, 0.0)]), cfg, g[:1], torch.zeros(1),
                                             torch.zeros(1, dtype=torch.bool)).any()))


class TestBlockScene(unittest.TestCase):
    def test_block_and_table_body(self):
        from nedm.render import so101 as R
        sc = R.SO101Scene(goal="block", cameras=("top", "wrist"), table_body=True)
        self.assertEqual(sc.goal_meshes, 1)
        self.assertEqual(sc.body_names[-1], "table")
        self.assertNotIn("table", [m.name for m, _ in sc.scene.statics])
        q = torch.zeros(2, 6)
        cp = torch.tensor([[0.2, 0.05, 0.015]] * 2)
        cq = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
        gp = torch.tensor([[0.25, -0.1, 0.3], [0.2, 0.1, -0.2]])
        body_q, cam_q = sc.transforms(q, cp, cq, gp)
        gi = sc.body_names.index("goal")
        self.assertTrue(torch.allclose(body_q[:, gi, :2], gp[:, :2]))
        self.assertAlmostEqual(float(body_q[0, gi, 2]), 0.0)
        yaw = 2 * torch.atan2(body_q[:, gi, 5], body_q[:, gi, 6])                    # x y z w about z
        self.assertTrue(torch.allclose(yaw, gp[:, 2], atol=1e-6))
        self.assertTrue(torch.allclose(body_q[:, -1], sc.table_pose.float().expand(2, 7)))
        st = R.SO101Scene(goal="tray")                                  # other styles still build
        self.assertEqual(st.goal_meshes, 2)


class TestAugment(unittest.TestCase):
    def test_shapes_and_range(self):
        from nedm.so101_policy.vision import viral as VI
        x = torch.rand(6, 3, 32, 32)
        d = VI.DRConfig(p_bright=1, p_contrast=1, p_sat=1, p_hue=1, p_noise=1, p_blur=1)
        y = VI.augment(x, d, torch.Generator().manual_seed(0))
        self.assertEqual(y.shape, x.shape)
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 1.0)
        z = VI.augment(x, VI.DRConfig(p_bright=0, p_contrast=0, p_sat=0, p_hue=0, p_noise=0, p_blur=0))
        self.assertLess(float((z - x).abs().max()), 1e-5)                 # all off: the identity (hue 0 rotation)


class TestStudentParts(unittest.TestCase):
    def stats(self):
        from nedm.so101_policy.vision import viral as VI
        return dict(robot_mean=torch.zeros(12), robot_std=torch.ones(12), tgt_mean=torch.zeros(10, 6),
                    tgt_std=torch.ones(10, 6), aux_mean=torch.zeros(VI.AUX_DIM), aux_std=torch.ones(VI.AUX_DIM))

    def test_expand_hist_keeps_the_policy(self):
        from nedm.so101_policy.vision import viral as VI
        cfg = VI.ViralConfig(res=32, keypoints=4, pool_dim=8, hidden=(16, 16), amp=False)
        st = VI.ViralNet(cfg, self.stats(), pretrained=False).eval()
        img = torch.randn(3, 2, 3, 32, 32)
        robot = torch.randn(3, 12)
        p0 = st.predict(img, robot)
        st2 = VI.expand_hist(st, 3, "cpu").eval()
        self.assertEqual(st2.cfg.hist, 3)
        p1 = st2.predict(img, torch.cat((robot, torch.randn(3, 36)), -1))     # history inputs have zero weights
        self.assertLess(float((p0 - p1).abs().max()), 1e-5)

    def test_replay_buffer_ring(self):
        from nedm.so101_policy.vision import viral as VI
        b = VI.ReplayBuffer(5, 2, 4, "cpu")
        for k in range(3):
            n = 2
            idx = b.add(torch.full((n, 2, 4, 4, 3), k, dtype=torch.uint8), torch.zeros(n, 6), torch.zeros(n, 12),
                        torch.zeros(n, 5, 6), torch.zeros(n, VI.AUX_DIM), torch.ones(n, dtype=torch.bool))
        self.assertEqual(b.n, 5)
        self.assertEqual(idx.tolist(), [4, 0])                            # wrapped around
        self.assertEqual(int(b.rgb[0, 0, 0, 0, 0]), 2)
        self.assertFalse(bool(b.valid2.any()))


@unittest.skipUnless(HAVE, "teacher checkpoint / task file missing")
class TestEpisodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nedm.so101_policy import joint_teacher as JT
        from nedm.so101_policy.vision import teacher as TT
        from nedm.so101_policy.vision.demos import load_tasks
        cls.JT = JT
        cls.te = TT.Teacher(str(TEACHER), "cpu")
        cls.tasks = load_tasks(TASKS, np.arange(4))
        cls.d = JT.expert_collect(cls.tasks, cls.te, "kin", np.array([0.0, 0.0, 1.0, 1.0]), 0, "cpu", batch=4,
                                  log=lambda s: None)

    def test_expert_rows(self):
        d = self.d
        self.assertEqual(d["row_mobs"].shape, (len(d["row_ep"]), JE.OBS_DIM))
        self.assertTrue(np.isfinite(d["row_mobs"]).all())
        # the observation's q is the recorded row q
        self.assertLess(float(np.abs(d["row_mobs"][:, :6] - d["row_q"]).max()), 1e-5)
        self.assertTrue(d["lenient"][d["success"].astype(bool)].all())    # strict implies lenient

    def test_bc_jt_run_and_student_set(self):
        import tempfile
        from nedm.so101_policy.vision import viral as VI
        JT = self.JT
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "e.npz"
            np.savez(p, **self.d)
            data = JT.BCSet([p], "cpu", val_frac=0.25)
            pol, codec = JT.train_bc(data, Path(tmp) / "bc", hidden=(32, 32), steps=20, batch=256, log=lambda s: None,
                                     val_every=10)
            jt = JE.JointTeacher(Path(tmp) / "bc" / "bc.pt", "cpu")
            env = JT.make_env("kin", 4, jt.codec, "cpu")
            r = JT.run_tasks(env, self.tasks, jt, record=True, noise=np.array([0.0, 1.0, 0.0, 1.0]))
            n = int(r["length"].sum())
            self.assertEqual(len(r["row_ep"]), n)
            self.assertEqual(r["row_lab"].shape, (n, JE.STEPS, 6))
            # the JT label of the first decision equals the codec decode of its action at the start state
            obs = env.reset_tasks(torch.arange(4), __import__("nedm.so101_policy.vision.demos",
                                                              fromlist=["x"]).start_state(self.tasks, "cpu"),
                                  torch.as_tensor(self.tasks["goal_xy"]), torch.as_tensor(self.tasks["goal_yaw"]),
                                  torch.as_tensor(self.tasks["seeds"]))
            lab0 = jt.targets(obs, env.ws.q)
            self.assertLess(float((lab0[0].float() - torch.as_tensor(r["row_lab"][0])).abs().max()), 1e-5)
            # student set: chunk = this decision's 5 labels ++ the next decision's 5, relative to q
            np.savez(Path(tmp) / "s.npz", **r)
            ss = VI.StudentSet([Path(tmp) / "s.npz"], "cpu", chunk=10, val_frac=0.0)
            rows = torch.arange(ss.n_rows)
            tg, ok = ss.targets(rows)
            e0 = int(r["length"][0])
            self.assertTrue(torch.allclose(tg[0, :5] + ss.q[0], ss.lab[0]))
            self.assertTrue(torch.allclose(tg[0, 5:] + ss.q[0], ss.lab[1]))
            self.assertFalse(bool(ok[e0 - 1, 5:].any()))                  # past the episode end: masked
            self.assertTrue(bool(ok[e0 - 1, :5].all()))
            b = ss.batch(rows[:3])
            self.assertEqual(tuple(b["aux"].shape), (3, VI.AUX_DIM))


if __name__ == "__main__":
    unittest.main()
