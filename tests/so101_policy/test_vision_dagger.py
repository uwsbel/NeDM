"""Unit tests of the PPO-teacher DAgger pipeline (``vision.teacher``, ``vision.perstep``, ``vision.act7``), CPU.

    PYTHONPATH=src python -m unittest tests.so101_policy.test_vision_dagger            # env nedm: LeRobot tests skip
    PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest tests.so101_policy.test_vision_dagger
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy.vision import teacher as TT

HAS_LEROBOT = importlib.util.find_spec("lerobot") is not None
GATE_TASKS = Path.home() / "so101_policy" / "runs" / "gate" / "tasks_1024.npz"
DEMO_TASKS = Path.home() / "so101_policy" / "runs" / "vision" / "demos" / "tasks_demo_2560.npz"
GEOM = Path.home() / "so101_policy" / "ckpt" / "task_geom_v2.npz"
TEACHER = Path(TT.TEACHERS["mj_w5dc"]).expanduser()
HAVE_FILES = DEMO_TASKS.exists() and GEOM.exists() and TEACHER.exists()


class StubViews:
    """Stands in for ``dataset.Views`` (no GPU): a picture whose brightness encodes the cube x."""

    def __init__(self, res=32, cameras=("top", "wrist")):
        self.res, self.cameras, self.views, self.seconds, self.num_worlds = res, tuple(cameras), 0, 1e-6, 10 ** 6

    def render(self, q, cube_pos, cube_quat, goal_pose):
        n = q.shape[0]
        base = ((cube_pos[:, 0] * 400).clamp(0, 255)).to(torch.uint8)
        self.views += n * len(self.cameras)
        return base.view(n, 1, 1, 1, 1).expand(n, len(self.cameras), self.res, self.res, 3).clone()


def synthetic_file(path, lengths=(5, 7, 4), modes=None, seed=0):
    """A ``teacher.collect``-like file: label[k] = episode * 100 + decision index in every column (traceable)."""
    from nedm.so101_policy import task as T
    from nedm.so101_policy.vec_env import obs_layout
    rng = np.random.default_rng(seed)
    D_ = T.obs_dim(T.TaskConfig(), 24)
    N = sum(lengths)
    ep = np.repeat(np.arange(len(lengths)), lengths)
    t = np.concatenate([np.arange(n) for n in lengths])
    lab = (ep * 100 + t)[:, None].repeat(7, 1).astype(np.float32) / 100.0
    d = dict(obs=rng.normal(size=(N, D_)).astype(np.float32), q=rng.normal(size=(N, 6)).astype(np.float32),
             cube_pos=np.abs(rng.normal(size=(N, 3))).astype(np.float32) * 0.1,
             cube_quat=np.tile([1.0, 0, 0, 0], (N, 1)).astype(np.float32), label=lab, executed=lab.copy(),
             src=np.zeros(N, np.int8), mode=(np.zeros(N, np.int8) if modes is None else np.asarray(modes, np.int8)),
             ep_index=ep.astype(np.int32), t=t.astype(np.int16),
             ep_start=np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64))
    E = len(lengths)
    for k in ("end", "length", "steps"):
        d[f"ep_{k}"] = np.asarray(lengths if k == "length" else np.zeros(E), np.int64)
    for k in ("success", "success10", "lifted"):
        d[f"ep_{k}"] = np.ones(E, bool)
    for k in ("end_xy", "end_yaw"):
        d[f"ep_{k}"] = np.zeros(E)
    d["ep_seed"] = 510000 + np.arange(E)
    d["ep_goal_xy"] = np.tile([0.25, 0.0], (E, 1)).astype(np.float32)
    d["ep_goal_yaw"] = np.zeros(E, np.float32)
    meta = dict(teacher=dict(decoder_cfg={"grasp_gate": True}, obs_rates=True), world="synthetic")
    d["meta"] = np.array(json.dumps(meta))
    np.savez(path, **d)
    _ = obs_layout
    return d


class TestBasics(unittest.TestCase):
    def test_read_mask(self):
        from nedm.so101_policy.decoder import M
        m = TT.read_mask(torch.tensor([M["free"], M["g_brake"], M["g_descend"], M["g_settle"], M["g_close"], M["hold"],
                                       M["p_lower"]]))
        np.testing.assert_array_equal(m.numpy(), [[1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1],
                                                  [0, 0, 0, 0, 0, 0, 1], [0] * 7, [1, 1, 1, 1, 1, 0, 0], [0] * 7])

    def test_robot_index_excludes_cube(self):
        from nedm.so101_policy import task as T
        from nedm.so101_policy.vec_env import obs_layout
        lay = obs_layout(T.TaskConfig(obs_rates=True))
        idx = set(TT.robot_index(lay).tolist())
        self.assertEqual(len(idx), 57)
        for k in ("cube_pos", "cube_yaw4", "cube_tilt", "cube_v", "cube_w", "cube_in_grasp", "rel_yaw4",
                  "goal_minus_cube", "yaw_err", "lifted", "hold_time"):
            self.assertFalse(idx & set(range(lay[k].start, lay[k].stop)), k)

    def test_seed_guard(self):
        TT.check_seeds(np.arange(510000, 510010))
        for bad in (np.array([400001]), np.array([1500]), np.array([900000])):
            with self.assertRaises(ValueError):
                TT.check_seeds(bad)

    def test_dataset_chunks_and_modes(self):
        from nedm.so101_policy.decoder import M
        from nedm.so101_policy.vision.perstep import DaggerSet
        modes = [M["free"]] * 3 + [M["g_brake"], M["g_close"]] + [M["free"]] * 7 + [M["hold"]] * 4
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_file(Path(tmp) / "a.npz", modes=modes)
            ds = DaggerSet([str(Path(tmp) / "a.npz")], "cpu", chunk=4, val_frac=0.0)
            # g_close rows read nothing: not sampled
            self.assertNotIn(4, ds.train_rows.tolist())
            b = ds.batch(torch.tensor([0, 2, 10]))
            # row 0 (episode 0, t 0): targets t 0..3, mode change after t 2 -> t 3 masked
            np.testing.assert_allclose(b["label"][0, :, 0].numpy(), [0.0, 0.01, 0.02, 0.03], atol=1e-6)
            np.testing.assert_array_equal(b["mask"][0, :, 0].numpy(), [1, 1, 1, 0])
            # row 2: t 2 free, t 3 g_brake (other segment) -> only the first target
            np.testing.assert_array_equal(b["mask"][1].sum(-1).numpy() > 0, [True, False, False, False])
            # row 10 (episode 1, t 5 of 7): t 5, 6 then the episode end (padded)
            self.assertTrue(bool(b["is_pad"][2, 2]) and bool(b["is_pad"][2, 3]))
            np.testing.assert_array_equal(b["mask"][2, :, 0].numpy(), [1, 1, 0, 0])
            ds1 = DaggerSet([str(Path(tmp) / "a.npz")], "cpu", chunk=1, val_frac=0.0)
            b1 = ds1.batch(torch.tensor([3, 12]))
            np.testing.assert_array_equal(b1["mask"].numpy(), [[0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 0, 0]])
            self.assertEqual(ds1.robot.shape[1], 57)

    def test_validation_from_first_file(self):
        from nedm.so101_policy.vision.perstep import DaggerSet
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_file(Path(tmp) / "a.npz", lengths=(3,) * 20)
            synthetic_file(Path(tmp) / "b.npz", lengths=(4,) * 10)
            ds1 = DaggerSet([str(Path(tmp) / "a.npz")], "cpu", val_frac=0.2)
            ds2 = DaggerSet([str(Path(tmp) / "a.npz"), str(Path(tmp) / "b.npz")], "cpu", val_frac=0.2)
            self.assertEqual(ds1.val_rows.tolist(), ds2.val_rows.tolist())
            self.assertEqual(ds2.n_val_eps, 4)
            self.assertEqual(len(ds2.train_rows), 16 * 3 + 40)


class TestPerStep(unittest.TestCase):
    def test_train_save_load(self):
        from nedm.so101_policy.vision import perstep as P
        cfg = P.PerStepConfig(encoder="small", keypoints=4, pool_dim=8, hidden=(32,), res=32, batch=8, render_batch=16,
                              amp=False, warmup=2)
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_file(Path(tmp) / "a.npz", lengths=(6,) * 10)
            ds = P.DaggerSet([str(Path(tmp) / "a.npz")], "cpu", val_frac=0.2)
            st = P.train(ds, StubViews(32), cfg, Path(tmp) / "run", max_steps=6, log=lambda s: None, val_every=3,
                         device="cpu", val_n=16)
            st2, ck = P.load_perstep(Path(tmp) / "run" / "last.pt", "cpu")
            img = torch.randn(3, 2, 3, 32, 32)
            rob = torch.randn(3, 57)
            torch.testing.assert_close(st(img, rob), st2(img, rob))
            self.assertEqual(ck["kind"], "perstep")
            rows = list(__import__("csv").DictReader(open(Path(tmp) / "run" / "learning_curve.csv")))
            self.assertTrue(any(r["val_mse"] for r in rows))

    def test_spatial_softmax_peak(self):
        from nedm.so101_policy.vision.perstep import SpatialSoftmax
        s = SpatialSoftmax(1, 1)
        with torch.no_grad():
            s.proj.weight.fill_(1.0)
            s.proj.bias.zero_()
            s.log_temp.fill_(-5.0)
        x = torch.zeros(1, 1, 9, 9)
        x[0, 0, 2, 6] = 10.0                 # row 2 (y), column 6 (x)
        out = s(x)[0]
        self.assertAlmostEqual(float(out[0]), -1 + 2 * 6 / 8, places=3)
        self.assertAlmostEqual(float(out[1]), -1 + 2 * 2 / 8, places=3)


class Const:
    """A student that always returns the same pre-tanh action."""

    def __init__(self, a):
        self.a = torch.as_tensor(a, dtype=torch.float64)

    def reset(self, env):
        pass

    def __call__(self, obs, env):
        return self.a.to(env.dev).expand(env.B, -1).clone()


@unittest.skipUnless(HAVE_FILES, "needs the demo tasks, task geometry and the w5_dc teacher checkpoint")
class TestTeacherWorld(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nedm.so101_policy.vision import demos as D
        cls.te = TT.Teacher(TEACHER, "cpu")
        cls.tasks = D.load_tasks(DEMO_TASKS, np.arange(6))

    def test_teacher_collect_equals_eval_policy(self):
        """beta 1, noise 0: executed = label, and the episodes equal the PPO evaluation path (policy_act_fn)."""
        import importlib.util as iu
        torch.manual_seed(0)
        d = TT.collect(self.tasks, "kin", self.te, device="cpu", log=lambda s: None)
        np.testing.assert_array_equal(d["label"], d["executed"])
        self.assertTrue((d["src"] == TT.SRC["teacher"]).all())
        spec = iu.spec_from_file_location("eval_policy", Path(__file__).resolve().parents[2] / "scripts" /
                                          "so101_policy" / "eval_policy.py")
        E = iu.module_from_spec(spec)
        spec.loader.exec_module(E)
        r = E.run_policy(self.tasks, "kin", str(TEACHER), device="cpu", record=False, log=None)
        np.testing.assert_array_equal(d["ep_end"], r["end"])
        np.testing.assert_array_equal(d["ep_length"], r["length"])

    def test_beta0_and_noise(self):
        a = np.zeros(7)
        a[4], a[6] = -2.0, -3.0
        d = TT.collect(self.tasks[:0] if False else self.tasks, "kin", self.te, Const(a), beta=0.0, device="cpu",
                       log=lambda s: None)
        np.testing.assert_allclose(d["executed"], np.broadcast_to(a, d["executed"].shape), atol=1e-6)
        self.assertTrue((d["src"] == TT.SRC["student"]).all())
        n = TT.collect(self.tasks, "kin", self.te, None, beta=1.0, noise=1.0, device="cpu", log=lambda s: None)
        diff = n["executed"] - n["label"]
        self.assertGreater(np.abs(diff[:, :4]).max(), 0.01)
        np.testing.assert_array_equal(diff[:, 4:], 0.0)

    def test_beta_mix(self):
        a = np.zeros(7)
        a[4], a[6] = -2.0, -3.0
        d = TT.collect(self.tasks, "kin", self.te, Const(a), beta=0.5, device="cpu", log=lambda s: None)
        frac = float((d["src"] == TT.SRC["teacher"]).mean())
        self.assertTrue(0.3 < frac < 0.7, frac)
        st = d["src"] == TT.SRC["student"]
        np.testing.assert_allclose(d["executed"][st], np.broadcast_to(a, d["executed"][st].shape), atol=1e-6)
        np.testing.assert_array_equal(d["executed"][~st], d["label"][~st])


@unittest.skipUnless(HAS_LEROBOT, "LeRobot not installed (env so101_vision has it)")
class TestAct7(unittest.TestCase):
    def test_train_save_load_predict(self):
        from nedm.so101_policy.vision import act7 as A
        from nedm.so101_policy.vision.perstep import DaggerSet
        cfg = A.Act7Config(chunk=4, res=32, dim_model=32, n_heads=2, dim_feedforward=64, n_encoder_layers=1,
                           n_vae_encoder_layers=1, latent_dim=4, pretrained=False, batch=8, render_batch=16, amp=False,
                           warmup=2)
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_file(Path(tmp) / "a.npz", lengths=(6,) * 10)
            ds = DaggerSet([str(Path(tmp) / "a.npz")], "cpu", chunk=4, val_frac=0.2)
            st = A.train(ds, StubViews(32), cfg, Path(tmp) / "run", max_steps=4, log=lambda s: None, val_every=2,
                         device="cpu", val_n=16)
            st2, _ = A.load_act7(Path(tmp) / "run" / "last.pt", "cpu")
            img, rob, goal = torch.randn(3, 2, 3, 32, 32), torch.randn(3, 57), torch.randn(3, 4)
            p1, p2 = st.predict_chunk(img, rob, goal), st2.predict_chunk(img, rob, goal)
            self.assertEqual(tuple(p1.shape), (3, 4, 7))
            torch.testing.assert_close(p1, p2)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_FILES and HAS_LEROBOT, "needs the task files, the teacher and LeRobot")
class TestStudentsInLoop(unittest.TestCase):
    """Both students drive a DAgger collection in the kinematic world (CPU, stub renderer)."""

    @classmethod
    def setUpClass(cls):
        from nedm.so101_policy.vision import demos as D
        cls.te = TT.Teacher(TEACHER, "cpu")
        cls.tasks = D.load_tasks(DEMO_TASKS, np.arange(4))

    def _robot_idx(self):
        from nedm.so101_policy import task as T
        from nedm.so101_policy.vec_env import obs_layout
        return TT.robot_index(obs_layout(T.TaskConfig(obs_rates=self.te.obs_rates)))

    def test_act7_and_perstep_drive(self):
        from nedm.so101_policy.vision import act7 as A
        from nedm.so101_policy.vision import perstep as P
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_file(Path(tmp) / "a.npz", lengths=(6,) * 10)
            ds = P.DaggerSet([str(Path(tmp) / "a.npz")], "cpu", chunk=4, val_frac=0.2)
            cfg = A.Act7Config(chunk=4, res=32, dim_model=32, n_heads=2, dim_feedforward=64, n_encoder_layers=1,
                               n_vae_encoder_layers=1, latent_dim=4, pretrained=False, batch=8, render_batch=16,
                               amp=False, warmup=2)
            st = A.train(ds, StubViews(32), cfg, Path(tmp) / "act", max_steps=2, log=lambda s: None, device="cpu",
                         val_n=8)
            ps = P.PerStepStudent(P.PerStepConfig(encoder="small", keypoints=4, pool_dim=8, hidden=(16,), res=32,
                                                  amp=False), P.DaggerSet([str(Path(tmp) / "a.npz")], "cpu").stats(),
                                  pretrained=False).eval()
        for stu in (A.Act7Act(st, StubViews(32), self._robot_idx(), device="cpu"),
                    P.PerStepAct(ps, StubViews(32), self._robot_idx(), device="cpu")):
            d = TT.collect(self.tasks, "kin", self.te, stu, beta=0.0, device="cpu", log=lambda s: None)
            self.assertTrue((d["src"] == TT.SRC["student"]).all())
            self.assertEqual(d["label"].shape[1], 7)
            self.assertGreater(stu.calls, 0)
