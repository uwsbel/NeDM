"""Tests of the tray student (``nedm.so101_policy.vision.tray``; build8_tray_student.md). CPU, kinematic world (no
MuJoCo processes, no renderer):

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_vision_tray

Needs the teacher checkpoint (``TEACHER``) and the stage-A development task file; skipped without them.
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import torch

TEACHER = Path("~/so101_policy/runs/c1ppo/ckpt/nC1_s62_b_model_00171.pt").expanduser()
TASKS = Path("~/so101_policy/runs/vision_tray/tasks/tasks_A_dev100.npz").expanduser()
HAVE = TEACHER.exists() and TASKS.exists()


class StubViews:
    """A renderer stand-in: images that encode nothing (the student then acts from q, qdot only)."""

    def __init__(self, res=32, cameras=("top", "wrist")):
        self.res, self.cameras = res, cameras
        self.views, self.seconds = 0, 0.0

    def render(self, q, cube_pos, cube_quat, goal_pose):
        n = q.shape[0]
        self.views += n * len(self.cameras)
        g = (goal_pose[:, :1, None, None] * 0 + 128).to(torch.uint8)
        return g.expand(n, len(self.cameras), self.res, self.res).unsqueeze(-1).expand(-1, -1, -1, -1, 3).clone()


class Replay:
    """A student that sends the commands of a recorded teacher episode (a perfect follower)."""

    def __init__(self, r, B, chunk=10):
        S = int(r["steps"].max())
        self.U = np.zeros((B, S + 400, 6))
        for b in range(B):
            s0, n = int(r["ep_step_start"][b]), int(r["steps"][b])
            self.U[b, :n] = r["u_exe"][s0:s0 + n]
            self.U[b, n:] = r["u_exe"][s0 + n - 1]
        self.k, self.chunk = 0, chunk
        self.t_render = self.t_model = 0.0

    def reset(self, B):
        self.k = 0

    def __call__(self, ws, goal):
        c = torch.as_tensor(self.U[:, 5 * self.k: 5 * self.k + self.chunk], dtype=torch.float64)
        self.k += 1
        return c.to(ws.q.device)


@unittest.skipUnless(HAVE, "teacher checkpoint / task file missing")
class TestTray(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nedm.so101_policy.vision import teacher as TT
        from nedm.so101_policy.vision import tray as TR
        cls.TR = TR
        cls.te = TT.Teacher(str(TEACHER), "cpu")
        cls.tasks = TR.load_tasks(TASKS, np.arange(4))
        cls.tw = TR.TrayWorld("kin", 4, cls.te, "cpu")
        cls.r = cls.tw.run(cls.tasks, None, 1.0, None, 0, True)

    def test_teacher_labels_are_what_it_sends(self):
        r = self.r
        self.assertEqual(float(np.abs(r["u_lab"] - r["u_exe"]).max()), 0.0)
        self.assertTrue(r["success"].all())
        self.assertTrue((r["row_src"] == 0).all())
        self.assertEqual(len(r["u_lab"]), int(r["steps"].sum()))
        self.assertEqual(len(r["row_ep"]), int(((r["steps"] + 4) // 5).sum()))

    def test_rules_log_macro_phase_rules(self):
        self.assertEqual(set(self.TR.LOG_ONLY), {"grasp_box", "keep_out", "cube_slide"})
        self.assertEqual(self.tw.tcfg.log_only, self.TR.LOG_ONLY)
        self.assertAlmostEqual(self.tw.tcfg.tol_xy, 0.015)
        self.assertAlmostEqual(self.tw.tcfg.tol_yaw, math.radians(15))

    def test_noisy_teacher_labels_are_clean(self):
        r = self.tw.run(self.tasks, None, 1.0, 1.0, 0, True)
        self.assertTrue((r["row_src"] == 1).all())
        self.assertFalse(r["row_cont"].any())
        d = np.abs(r["u_lab"] - r["u_exe"]).max()
        self.assertGreater(d, 1e-4)          # the noise moved the executed targets ...
        self.assertLess(d, 0.2)              # ... but the labels are the clean continuation from the same state

    def test_perfect_follower_resync_near_identity(self):
        rep = Replay(self.r, 4)
        r2 = self.tw.run(self.tasks, rep, 0.0, None, 0, True)
        self.assertTrue((r2["steps"] == self.r["steps"]).all())
        self.assertTrue((r2["row_src"] == 2).all())
        d = np.abs(r2["u_lab"] - self.r["u_lab"])
        self.assertLess(float(np.percentile(d[:, :5], 99)), 0.03)
        self.assertEqual(int(r2["row_to_free"].sum() + r2["row_to_grip"].sum()), 0)

    def test_dec_copy_covers_state(self):
        from nedm.so101_policy.decoder import DecoderV2
        TR = self.TR
        B = 7
        d = DecoderV2(B, "cpu", kin=self.tw.kin, cfg=self.tw.dcfg)
        per_world = {n for n, v in vars(d).items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == B}
        missing = per_world - set(TR.DEC_STATE)
        self.assertEqual(missing, set(), f"per-world decoder state not copied: {missing}")
        d2 = DecoderV2(B, "cpu", kin=self.tw.kin, cfg=self.tw.dcfg)
        q = torch.zeros(B, 6, dtype=torch.float64)
        q[:, 1], q[:, 2] = -0.5, 0.8
        d.reset(torch.arange(B), q)
        d2.reset(torch.arange(B), q + 0.1)
        TR.dec_copy(d2, d, torch.ones(B, dtype=torch.bool))
        a = torch.zeros(B, 7, dtype=torch.float64)
        a[:, 0] = 0.5
        ws = self.tw.ws.index(torch.zeros(B, dtype=torch.long))
        d.decide(a, ws)
        d2.decide(a, ws)
        for _ in range(5):
            u1, _ = d.step(q)
            u2, _ = d2.step(q)
            self.assertTrue(torch.equal(u1, u2))

    def test_dataset_targets_and_validity(self):
        import tempfile
        TR = self.TR
        mix = self.tw.run(self.tasks, Replay(self.r, 4), 0.5, None, 3, True)
        with tempfile.TemporaryDirectory() as tmp:
            out = []
            for k, (rr, beta) in enumerate(((self.r, 1.0), (mix, 0.5))):
                rr = dict(rr)
                rr["meta"] = np.array('{"beta": %s}' % beta)
                p = Path(tmp) / f"r{k}.npz"
                np.savez(p, **rr)
                out.append(str(p))
            ds = TR.TraySet(out, "cpu", chunk=10, val_frac=0.0)
            src = np.concatenate([self.r["row_src"], mix["row_src"]])
            rows = torch.arange(ds.n_rows)
            tgt, valid = ds.targets(rows)
            # teacher file: chunk = the recorded future targets minus q at the query
            r0 = 3
            e = int(ds.s_ep[r0])
            g0 = int(ds.ep_step_start[e]) + int(ds.s_t[r0])
            self.assertTrue(torch.allclose(tgt[r0], ds.u_lab[g0:g0 + 10] - ds.q[r0]))
            self.assertTrue(valid[r0].all())
            # the last row of an episode: targets beyond the end masked
            last = int(ds.ep_row_start[e] + ds.ep_rows[e] - 1)
            left = int(ds.ep_steps[e] - ds.s_t[last])
            self.assertEqual(int(valid[last].sum()), min(10, left))
            # mixed file: steps 5-9 valid only where the next boundary continues the clean teacher's plan
            n0 = len(self.r["row_ep"])
            for rr in range(n0, ds.n_rows - 1):
                e = int(ds.s_ep[rr])
                if int(ds.s_ep[rr + 1]) != e or int(ds.s_t[rr]) + 10 > int(ds.ep_steps[e]):
                    continue
                self.assertTrue(bool(valid[rr, :5].all()))
                self.assertEqual(bool(valid[rr, 5:].all()), bool(ds.cont[rr + 1]))
                self.assertEqual(bool(ds.cont[rr + 1]), int(src[rr]) == 0)
            # per-step samples: the same targets at the boundaries, the states of the boundary rows
            dss = TR.TraySet(out, "cpu", chunk=10, val_frac=0.0, step_rows=True)
            self.assertEqual(dss.n_rows, int(np.sum([len(TR.load(f)["u_lab"]) for f in out])))
            bnd = torch.nonzero(dss.s_t % 5 == 0).flatten()
            self.assertEqual(len(bnd), ds.n_rows)
            tg2, va2 = dss.targets(bnd)
            self.assertTrue(torch.allclose(dss.q[bnd], ds.q))
            self.assertTrue(torch.allclose(tg2, tgt))
            self.assertTrue(torch.equal(va2, valid))
            # a mid-decision step of a student-driven episode: valid up to its decision's end only
            k = int(torch.nonzero((dss.s_t % 5 == 2) & (dss.s_ep == n0 // 1000 + len(self.r["end"]))).flatten()[3])
            e = int(dss.s_ep[k]); r = int(dss.ep_row_start[e] + dss.s_t[k] // 5 + 1)
            _, vk = dss.targets(torch.tensor([k]))
            if int(dss.s_t[k]) + 10 <= int(dss.ep_steps[e]):
                self.assertTrue(bool(vk[0, :3].all()))
                self.assertEqual(bool(vk[0, 3:8].all()), bool(dss.cont[r]))

    def test_student_train_and_act(self):
        import tempfile
        TR = self.TR
        with tempfile.TemporaryDirectory() as tmp:
            rr = dict(self.r)
            rr["meta"] = np.array("{}")
            p = Path(tmp) / "r0.npz"
            np.savez(p, **rr)
            ds = TR.TraySet([str(p)], "cpu", chunk=10, val_frac=0.25)
            cfg = TR.TrayConfig(encoder="small", keypoints=4, pool_dim=8, hidden=(32,), res=32, batch=8,
                                render_batch=16, warmup=2, amp=False)
            views = StubViews(32)
            st = TR.train(ds, views, cfg, Path(tmp) / "st", max_steps=4, device="cpu", val_n=16, val_every=2,
                          log=lambda s: None)
            st2, _ = TR.load_student(Path(tmp) / "st" / "last.pt", "cpu")
            actor = TR.TrayActor(st2, views, "cpu")
            ws = self.tw.ws
            u = actor(ws, torch.zeros(4, 3, dtype=torch.float64))
            self.assertEqual(tuple(u.shape), (4, 10, 6))
            r2 = self.tw.run(self.tasks, actor, 0.0, None, 0, True)
            self.assertTrue((r2["row_src"] == 2).all())
            self.assertEqual(len(r2["u_exe"]), int(r2["steps"].sum()))

    def test_stage_constants(self):
        TR = self.TR
        d = math.hypot(TR.CUBE_A[0] - TR.TRAY_AB[0], TR.CUBE_A[1] - TR.TRAY_AB[1])
        self.assertTrue(0.06 <= d <= 0.30)
        self.assertLessEqual(abs(TR.wrap90(TR.TRAY_AB[2] - TR.CUBE_A[2])), math.radians(30))
        from nedm.so101_policy.vec_env import reserved_seed_hits
        for st in TR.STAGES.values():
            self.assertEqual(reserved_seed_hits(np.arange(st["train0"], st["train0"] + 5000)), 0)
            self.assertEqual(reserved_seed_hits(np.arange(st["dev0"], st["dev0"] + 100)), 0)
        self.assertTrue((np.asarray(self.tasks["goal_xy"]) == np.float64(TR.TRAY_AB[:2])).all())


class TestTrayRender(unittest.TestCase):
    def test_tray_scene(self):
        from nedm.render import so101 as R
        try:
            sc = R.SO101Scene(goal="tray", cameras=("top", "wrist"))
        except FileNotFoundError as e:       # no vendor model on this machine
            self.skipTest(str(e))
        self.assertEqual(sc.goal_meshes, 2)
        names = sc.body_names
        self.assertIn("goal", names)
        self.assertIn("goal_rim", names)
        q = torch.zeros(2, 6)
        body, cams = sc.transforms(q, torch.tensor([[0.2, 0.1, 0.015]] * 2), torch.tensor([[1.0, 0, 0, 0]] * 2),
                                   torch.tensor([[0.22, -0.1, 0.3], [0.2, 0.0, 0.0]]))
        self.assertEqual(body.shape[1], len(names))
        gi, ri = names.index("goal"), names.index("goal_rim")
        self.assertTrue(torch.equal(body[:, gi], body[:, ri]))
        self.assertAlmostEqual(float(body[0, gi, 0]), 0.22, places=6)
        self.assertAlmostEqual(float(body[0, gi, 2]), 0.0, places=6)
        # yaw 0.3 rad about z: quaternion x y z w = (0, 0, sin 0.15, cos 0.15)
        self.assertAlmostEqual(float(body[0, gi, 5]), math.sin(0.15), places=5)
        sc2 = R.SO101Scene(goal="plate", cameras=("top",))
        self.assertEqual(sc2.goal_meshes, 1)
        sc3 = R.SO101Scene(goal=None, cameras=("top",))
        self.assertEqual(sc3.goal_meshes, 0)


if __name__ == "__main__":
    unittest.main()
