"""Unit tests of the camera-student pipeline (``nedm.so101_policy.vision``), CPU, seconds.

    PYTHONPATH=src python -m unittest tests.so101_policy.test_vision            # env nedm: LeRobot tests skip
    PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest tests.so101_policy.test_vision   # all
"""
from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy.vision import dataset as V
from nedm.so101_policy.vision import demos as D

HAS_LEROBOT = importlib.util.find_spec("lerobot") is not None
GATE_TASKS = Path.home() / "so101_policy" / "runs" / "gate" / "tasks_1024.npz"
GEOM = Path.home() / "so101_policy" / "ckpt" / "task_geom_v2.npz"


class StubViews:
    """Stands in for ``dataset.Views`` (no GPU): a deterministic picture from the state (cube x, y as brightness)."""

    def __init__(self, res=32, cameras=V.CAMERAS):
        self.res, self.cameras, self.views, self.seconds, self.num_worlds = res, tuple(cameras), 0, 1e-6, 10 ** 6

    def render(self, q, cube_pos, cube_quat, goal_pose):
        n = q.shape[0]
        base = ((cube_pos[:, 0] * 400).clamp(0, 255)).to(torch.uint8)
        img = base.view(n, 1, 1, 1, 1).expand(n, len(self.cameras), self.res, self.res, 3).clone()
        self.views += n * len(self.cameras)
        return img


def tiny_cfg(**kw):
    from nedm.so101_policy.vision.student import StudentConfig
    d = dict(chunk=5, res=32, dim_model=32, n_heads=2, dim_feedforward=64, n_encoder_layers=1, n_vae_encoder_layers=1,
             latent_dim=4, pretrained=False, batch=8, render_batch=16, amp=False, warmup=2, shift_px=2)
    d.update(kw)
    return StudentConfig(**d)


def synthetic_demo(E=3, lengths=(4, 6, 3), seed=0):
    """A packed demo dict like ``demos.pack`` makes, with action[k] = episode * 100 + decision index (traceable)."""
    rng = np.random.default_rng(seed)
    B, K = E, max(lengths) + 2
    per = {k: np.zeros((B, K) + s, np.float32) for k, s in (("q", (6,)), ("qd", (6,)), ("cube_pos", (3,)),
                                                              ("cube_quat", (4,)), ("dec_obs", (24,)), ("action", (6,)))}
    per["mode"] = np.zeros((B, K), np.int8)
    per["live"] = np.zeros((B, K), bool)
    for b, L in enumerate(lengths):
        per["live"][b, :L] = True
        per["action"][b, :, :] = (b * 100 + np.arange(K))[:, None]
        per["q"][b] = rng.normal(size=(K, 6))
        per["cube_quat"][b, :, 0] = 1.0
    ep = dict(end=np.array([1, 3, 1][:E]), success=np.array([True, False, True][:E]), success10=np.ones(E, bool),
              lifted=np.ones(E, bool), end_xy=np.zeros(E), end_yaw=np.zeros(E), length=np.array(lengths),
              steps=np.array(lengths) * 5)
    tasks = dict(seeds=np.arange(E) + 500000, goal_xy=rng.uniform(0.1, 0.2, (E, 2)), goal_yaw=rng.uniform(0, 1, E))
    out = D.pack(per, ep, tasks)
    out["meta"] = {"decoder_cfg": {"carry_macro": True}}
    return out


class TestPackAndDataset(unittest.TestCase):
    def test_pack_rows(self):
        d = synthetic_demo()
        self.assertEqual(len(d["t"]), 13)
        np.testing.assert_array_equal(d["ep_start"], [0, 4, 10])
        np.testing.assert_array_equal(d["ep_index"], [0] * 4 + [1] * 6 + [2] * 3)
        np.testing.assert_array_equal(d["action"][:, 0], [0, 1, 2, 3, 100, 101, 102, 103, 104, 105, 200, 201, 202])

    def test_pack_rejects_gap(self):
        per = {k: np.zeros((1, 3, 1), np.float32) for k in D.REC_KEYS}
        per["live"] = np.array([[True, False, True]])
        ep = {k: np.zeros(1) for k in D.EP_KEYS}
        ep["length"] = np.array([3])
        with self.assertRaises(RuntimeError):
            D.pack(per, ep, dict(seeds=np.zeros(1), goal_xy=np.zeros((1, 2)), goal_yaw=np.zeros(1)))

    def test_concat_shifts(self):
        a, b = synthetic_demo(), synthetic_demo(seed=1)
        a["tasks_file"], b["tasks_file"] = np.array("x.npz"), np.array("y.npz")     # scalar records: the first file's
        c = D.concat([{k: v for k, v in x.items() if k != "meta"} for x in (a, b)])
        np.testing.assert_array_equal(c["ep_start"], [0, 4, 10, 13, 17, 23])
        self.assertEqual(int(c["ep_index"][13]), 3)
        self.assertEqual(str(c["tasks_file"]), "x.npz")

    def _demoset(self, chunk=3, **kw):
        d = synthetic_demo()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.npz"
            np.savez(p, **{k: v for k, v in d.items() if k != "meta"}, meta=np.array('{"decoder_cfg": {"carry_macro": true}}'))
            return V.DemoSet([p], "cpu", chunk=chunk, val_frac=0.0, **kw)

    def test_success_only_and_chunks(self):
        ds = self._demoset(chunk=3)
        self.assertEqual(sorted(ds.train_eps.tolist()), [0, 2])          # episode 1 failed
        self.assertEqual(len(ds.train_rows), 7)
        b = ds.batch(torch.tensor([2, 3, 10]))                             # rows: ep0 t2, ep0 t3, ep2 t0
        np.testing.assert_array_equal(b["action"][:, :, 0].numpy(), [[2, 3, 3], [3, 3, 3], [200, 201, 202]])
        np.testing.assert_array_equal(b["action_is_pad"].numpy(), [[False, False, True], [False, True, True],
                                                                    [False, False, False]])
        self.assertEqual(tuple(b["state"].shape), (3, 30))
        self.assertEqual(ds.decoder_cfg, {"carry_macro": True})

    def test_max_episodes(self):
        ds = self._demoset(max_episodes=1)
        self.assertEqual(ds.n_train_eps, 1)

    def test_goal_vec_symmetry(self):
        g1 = V.goal_vec(torch.tensor([[0.2, -0.1]]), torch.tensor([0.3]))
        g2 = V.goal_vec(torch.tensor([[0.2, -0.1]]), torch.tensor([0.3 + math.pi / 2]))
        torch.testing.assert_close(g1, g2, atol=1e-6, rtol=0)
        self.assertEqual(tuple(g1.shape), (1, 4))

    def test_random_shift(self):
        x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).view(2, 3, 8, 8)
        torch.testing.assert_close(V.random_shift(x, 0), x)
        g = torch.Generator().manual_seed(0)
        y = V.random_shift(x, 2, g)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(set(y.unique().tolist()) <= set(x.unique().tolist()))
        rgb = torch.zeros(2, 2, 8, 8, 3, dtype=torch.uint8)
        inp = V.to_input(rgb)
        self.assertEqual(tuple(inp.shape), (2, 2, 3, 8, 8))
        torch.testing.assert_close(inp[0, 0, :, 0, 0], -torch.tensor(V.IMAGENET_MEAN) / torch.tensor(V.IMAGENET_STD))


@unittest.skipUnless(GATE_TASKS.exists() and GEOM.exists(), "needs the gate tasks and task geometry files")
class TestRecorderKinWorld(unittest.TestCase):
    """The recorder in the kinematic world (CPU): replaying the recorded actions reproduces the episodes."""

    def test_noisy_expert(self):
        from nedm.so101_policy.vision import rollout as R
        t = D.load_tasks(GATE_TASKS, np.arange(4))
        clean = R.collect_noisy_expert(t, "kin", device="cpu", sigma=0.0, log=lambda s: None)
        noisy = R.collect_noisy_expert(t, "kin", device="cpu", sigma=0.3, log=lambda s: None)
        np.testing.assert_array_equal(clean["action"], clean["executed"])
        self.assertTrue(clean["ep_success"].all())
        free = noisy["mode"] == 0
        self.assertGreater(float(np.abs(noisy["action"][free, :4] - noisy["executed"][free, :4]).max()), 0.01)
        np.testing.assert_array_equal(noisy["action"][:, 4:], noisy["executed"][:, 4:])     # triggers: no noise
        self.assertTrue(json.loads(str(noisy["meta"]))["dart"])

    def test_record_and_replay(self):
        from nedm.so101_policy.vec_env import run_episodes
        t = D.load_tasks(GATE_TASKS, np.arange(3))
        out = D.collect(t, "kin", device="cpu", log=lambda s: None)
        self.assertEqual(json_dec(out), {"carry_macro": True})
        L = out["ep_length"]
        self.assertEqual(int(L.sum()), len(out["t"]))
        self.assertTrue(np.all(np.abs(out["action"]) <= 1.0))
        # replay the recorded actions decision by decision in a fresh env
        env = D.make_env("kin", 3, "cpu")
        acts = np.zeros((3, int(L.max()) + 2, 6), np.float32)
        for e in range(3):
            s = out["ep_start"][e]
            acts[e, : L[e]] = out["action"][s: s + L[e]]
        k = {"i": 0}

        def act(obs, env_):
            a = torch.as_tensor(acts[:, min(k["i"], acts.shape[1] - 1)], dtype=torch.float64)
            k["i"] += 1
            return a, True
        g = lambda key: torch.as_tensor(t[key], dtype=torch.float64)        # noqa: E731
        r = run_episodes(env, act, D.start_state(t, "cpu"), g("goal_xy"), g("goal_yaw"), torch.as_tensor(t["seeds"]))
        np.testing.assert_array_equal(r["end"], out["ep_end"])
        np.testing.assert_array_equal(r["length"], out["ep_length"])
        np.testing.assert_allclose(r["end_xy"], out["ep_end_xy"], atol=1e-6)    # actions stored as float32


def json_dec(out):
    return json.loads(str(out["meta"]))["decoder_cfg"]


@unittest.skipUnless(HAS_LEROBOT, "LeRobot not installed (env so101_vision has it)")
class TestStudent(unittest.TestCase):
    def test_ensembler_matches_offline_average(self):
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        c, T, coeff = 4, 7, 0.3
        g = torch.Generator().manual_seed(0)
        chunks = torch.randn(T, 2, c, 3, generator=g)
        ens = ACTTemporalEnsembler(coeff, c)
        w = torch.exp(-coeff * torch.arange(c))
        for t in range(T):
            got = ens.update(chunks[t])
            preds = [chunks[s][:, t - s] for s in range(max(0, t - c + 1), t + 1)]     # oldest first
            ww = w[: len(preds)]
            ref = sum(p * wi for p, wi in zip(preds, ww)) / ww.sum()
            torch.testing.assert_close(got, ref, atol=1e-5, rtol=0)

    def test_mode_ensembler(self):
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        from nedm.so101_policy.vision.rollout import ModeEnsembler
        c, T, coeff = 4, 9, 0.3
        g = torch.Generator().manual_seed(1)
        chunks = torch.randn(T, 3, c, 2, generator=g)
        ref, me = ACTTemporalEnsembler(coeff, c), ModeEnsembler(coeff, c)
        mode = torch.zeros(3, dtype=torch.long)
        for t in range(T):                       # no mode change: equal to ACT's ensembler
            torch.testing.assert_close(me.update(chunks[t], mode), ref.update(chunks[t]), atol=1e-5, rtol=0)
        me.reset()
        for t in range(T):                       # world 1 changes mode at t = 5: only the newest prediction then
            mode = torch.tensor([0, 1 if t >= 5 else 0, 0])
            a = me.update(chunks[t], mode)
            if t == 5:
                torch.testing.assert_close(a[1], chunks[5][1, 0])
            if t == 6:
                w = torch.exp(-coeff * torch.tensor([0.0, 1.0]))
                torch.testing.assert_close(a[1], (chunks[5][1, 1] * w[0] + chunks[6][1, 0] * w[1]) / w.sum())

    def test_train_save_load_predict(self):
        from nedm.so101_policy.vision import student as S
        d = synthetic_demo(lengths=(12, 6, 9))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.npz"
            np.savez(p, **{k: v for k, v in d.items() if k != "meta"}, meta=np.array('{"decoder_cfg": {"carry_macro": true}}'))
            ds = V.DemoSet([p], "cpu", chunk=5, val_frac=0.0)
            cfg = tiny_cfg()
            st = S.train(ds, StubViews(32), cfg, Path(tmp) / "run", max_minutes=0, max_steps=4, val_n=16, val_every=2,
                         log_every=2, device="cpu", log=lambda s: None)
            self.assertTrue((Path(tmp) / "run" / "learning_curve.csv").exists())
            st2, ck = S.load_student(Path(tmp) / "run" / "last.pt", "cpu")
            self.assertEqual(ck["step"], 4)
            self.assertEqual(st2.decoder_cfg, {"carry_macro": True})
            b = ds.batch(torch.arange(4))
            img = V.to_input(StubViews(32).render(b["q"], b["cube_pos"], b["cube_quat"], b["goal_pose"]))
            a1 = st.predict_chunk(img, b["state"], b["goal"])
            a2 = st2.predict_chunk(img, b["state"], b["goal"])
            self.assertEqual(tuple(a1.shape), (4, 5, 6))
            torch.testing.assert_close(a1, a2)
            loss, info = st2.loss(img, b["state"], b["goal"], b["action"], b["action_is_pad"])
            self.assertTrue(torch.isfinite(loss))

    def test_trigger_bce(self):
        from nedm.so101_policy.vision import student as S
        d = synthetic_demo(lengths=(12, 6, 9))
        d["action"][:, 4] = np.where(np.arange(len(d["t"])) % 5 == 0, 1.0, -1.0)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.npz"
            np.savez(p, **{k: v for k, v in d.items() if k != "meta"}, meta=np.array('{"decoder_cfg": {"carry_macro": true}}'))
            ds = V.DemoSet([p], "cpu", chunk=5, val_frac=0.0)
            st = S.train(ds, StubViews(32), tiny_cfg(trigger_bce=True), Path(tmp) / "run", max_minutes=0, max_steps=3,
                         val_n=8, val_every=1, log_every=1, device="cpu", log=lambda s: None)
            self.assertEqual((float(st.action_mean[4]), float(st.action_std[4])), (0.0, 1.0))
            b = ds.batch(torch.arange(4))
            img = V.to_input(StubViews(32).render(b["q"], b["cube_pos"], b["cube_quat"], b["goal_pose"]))
            a = st.predict_chunk(img, b["state"], b["goal"])
            self.assertTrue(bool(((a[..., 4] > -1) & (a[..., 4] < 1)).all()))
            loss, info = st.loss(img, b["state"], b["goal"], b["action"], b["action_is_pad"])
            self.assertIn("bce_loss", info)
            self.assertTrue(torch.isfinite(loss))
            st2, _ = S.load_student(Path(tmp) / "run" / "last.pt", "cpu")
            torch.testing.assert_close(st2.predict_chunk(img, b["state"], b["goal"]), a)

    def test_privileged_ablation_shapes(self):
        from nedm.so101_policy.vision import student as S
        d = synthetic_demo(lengths=(12, 6, 9))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "d.npz"
            np.savez(p, **{k: v for k, v in d.items() if k != "meta"}, meta=np.array('{"decoder_cfg": {"carry_macro": true}}'))
            ds = V.DemoSet([p], "cpu", chunk=5, val_frac=0.0, privileged=True)
            b = ds.batch(torch.arange(3))
            self.assertEqual(tuple(b["goal"].shape), (3, 9))
            self.assertEqual(tuple(ds.stats()["goal_mean"].shape), (9,))
            st = S.train(ds, StubViews(32), tiny_cfg(privileged_cube=True), Path(tmp) / "run", max_minutes=0, max_steps=2,
                         val_n=8, val_every=1, log_every=1, device="cpu", log=lambda s: None)
            img = V.to_input(StubViews(32).render(b["q"], b["cube_pos"], b["cube_quat"], b["goal_pose"]))
            self.assertEqual(tuple(st.predict_chunk(img, b["state"], b["goal"]).shape), (3, 5, 6))

    @unittest.skipUnless(GATE_TASKS.exists() and GEOM.exists(), "needs the gate tasks and task geometry files")
    def test_rollout_kin_world(self):
        from nedm.so101_policy.vision import rollout as R
        from nedm.so101_policy.vision import student as S
        stats = {f"{k}_{s}": (torch.zeros(n) if s == "mean" else torch.ones(n))
                 for k, n in (("state", 30), ("goal", 4), ("action", 6)) for s in ("mean", "std")}
        st = S.Student(tiny_cfg(), stats, "cpu", {"carry_macro": True}).eval()
        t = D.load_tasks(GATE_TASKS, np.arange(2))
        r = R.evaluate(st, t, "kin", StubViews(32), device="cpu", horizon=20, record=False, log=lambda s: None)
        self.assertEqual(len(r["end"]), 2)
        self.assertEqual(int(r["model_calls"]), 4)          # 20 steps = 4 decisions
        self.assertIn('"carry_macro": true', str(r["env_resolved"]))

    @unittest.skipUnless(GATE_TASKS.exists() and GEOM.exists(), "needs the gate tasks and task geometry files")
    def test_dagger_beta1_equals_expert(self):
        """beta = 1: the shadow expert drives; labels = executed actions and the episodes equal the plain expert's."""
        from nedm.so101_policy.vision import rollout as R
        from nedm.so101_policy.vision import student as S
        stats = {f"{k}_{s}": (torch.zeros(n) if s == "mean" else torch.ones(n))
                 for k, n in (("state", 30), ("goal", 4), ("action", 6)) for s in ("mean", "std")}
        st = S.Student(tiny_cfg(), stats, "cpu", {"carry_macro": True}).eval()
        t = D.load_tasks(GATE_TASKS, np.arange(2))
        dag = R.collect_dagger(st, t, "kin", StubViews(32), device="cpu", beta=1.0, log=lambda s: None, harness=False)
        ref = D.collect(t, "kin", device="cpu", log=lambda s: None)
        np.testing.assert_array_equal(dag["action"], dag["executed"])
        np.testing.assert_array_equal(dag["ep_end"], ref["ep_end"])
        np.testing.assert_allclose(dag["action"], ref["action"], atol=1e-6)
        self.assertTrue(json.loads(str(dag["meta"]))["dagger"])
        self.assertEqual(json.loads(str(dag["meta"]))["resyncs"], 0)

    @unittest.skipUnless(GATE_TASKS.exists() and GEOM.exists(), "needs the gate tasks and task geometry files")
    def test_dagger_student_drives(self):
        from nedm.so101_policy.vision import rollout as R
        from nedm.so101_policy.vision import student as S
        stats = {f"{k}_{s}": (torch.zeros(n) if s == "mean" else torch.ones(n))
                 for k, n in (("state", 30), ("goal", 4), ("action", 6)) for s in ("mean", "std")}
        st = S.Student(tiny_cfg(), stats, "cpu", {"carry_macro": True}).eval()
        t = D.load_tasks(GATE_TASKS, np.arange(2))
        dag = R.collect_dagger(st, t, "kin", StubViews(32), device="cpu", beta=0.0, horizon=50, log=lambda s: None)
        self.assertEqual(len(dag["t"]), int(dag["ep_length"].sum()))
        self.assertFalse(np.allclose(dag["action"], dag["executed"]))     # random student != expert labels
        self.assertTrue(json.loads(str(dag["meta"]))["harness"])
        self.assertEqual(json.loads(str(dag["meta"]))["expert_cfg_resolved"]["abort_off"], 0.013)

    @unittest.skipUnless(GATE_TASKS.exists() and GEOM.exists(), "needs the gate tasks and task geometry files")
    def test_harness_trigger_fires_in_box(self):
        """The expert drives (beta 1) with the harness trigger: it triggers in the box before its own 1.5 mm rule, and
        every GRASP it labels starts inside the box (|x| 4.5 mm, |y| 10 mm at the close)."""
        from nedm.so101_policy.vision import rollout as R
        from nedm.so101_policy.vision import student as S
        stats = {f"{k}_{s}": (torch.zeros(n) if s == "mean" else torch.ones(n))
                 for k, n in (("state", 30), ("goal", 4), ("action", 6)) for s in ("mean", "std")}
        st = S.Student(tiny_cfg(), stats, "cpu", {"carry_macro": True}).eval()
        t = D.load_tasks(GATE_TASKS, np.arange(4))
        dag = R.collect_dagger(st, t, "kin", StubViews(32), device="cpu", beta=1.0, log=lambda s: None, harness=True)
        meta = json.loads(str(dag["meta"]))
        self.assertGreater(meta["harness_triggers"], 0)
        self.assertTrue(dag["ep_success"].all())


if __name__ == "__main__":
    unittest.main()
