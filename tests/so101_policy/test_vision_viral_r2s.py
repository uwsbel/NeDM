"""Tests of the real2sim pieces of the camera student (``nedm.so101_policy.vision.viral``, ``vision.tray`` stage H;
``~/so101_r2s_claude/docs/integrate.md``): image-space camera effects, ``to_input`` unchanged without them, the image
lag (round-file lag states, interpolation, the per-world history, the env step tap, the actor's missing-tap check),
the real-scene setup (scene file -> crops, sha256 pinning), HOME task starts and the task mix, old configs unchanged.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_vision_viral_r2s -v

GPU tests (newton + CUDA, skipped otherwise) build a small real-scene view set from a placeholder scene file.
"""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from nedm.so101_policy import joint_env as JE
from nedm.so101_policy.vision import viral as VI

REPO = Path(__file__).resolve().parents[2]
CALIB = Path(os.environ.get("NEDM_R2S_CALIB", Path.home() / "so101_r2s_claude/calib/calib_v1.json"))
HAVE_MUJOCO = importlib.util.find_spec("mujoco") is not None
GPU = (importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None
       and torch.cuda.is_available())


def old_to_input(rgb, d=None, gen=None):
    """``viral.to_input`` before the real2sim change (reference for bit identity)."""
    x = rgb.permute(0, 1, 4, 2, 3).float().div_(255.0)
    n, C = x.shape[:2]
    if d is not None and d.on:
        x = VI.augment(x.reshape(n * C, *x.shape[2:]), d, gen).reshape(x.shape)
    from nedm.so101_policy.vision import dataset as V
    m = torch.tensor(V.IMAGENET_MEAN, device=x.device).view(1, 1, 3, 1, 1)
    s = torch.tensor(V.IMAGENET_STD, device=x.device).view(1, 1, 3, 1, 1)
    return (x - m) / s


def placeholder_scene(tmp: Path, **extra) -> Path:
    d = {"version": "real_scene_test", "calib": str(CALIB),
         "student_view": {"external": {"crop": [80, 0, 480, 480], "out": [64, 64], "interp": "area"},
                          "wrist": {"crop": [0, 0, 640, 480], "out": [64, 64], "interp": "area"}},
         "image_fx": {"p_noise": 0.5}, **extra}
    p = tmp / "scene.json"
    p.write_text(json.dumps(d))
    return p


class TestCameraEffects(unittest.TestCase):
    def test_off_is_identity_and_fixed_is_deterministic(self):
        x = torch.rand(5, 3, 32, 32, generator=torch.Generator().manual_seed(0))
        off = VI.CamFX()
        g = torch.Generator().manual_seed(1)
        self.assertTrue(torch.equal(VI.fx_sensor(VI.fx_optics(x, off, g, True), off, g, True), x.clamp(0, 1)))
        fx = VI.CamFX(fixed_blur=0.7, fixed_vignette=0.2, fixed_jpeg=80)
        a = VI.fx_sensor(VI.fx_optics(x, fx), fx)
        b = VI.fx_sensor(VI.fx_optics(x, fx), fx)
        self.assertTrue(torch.equal(a, b))                              # no random draw without ``random``
        self.assertGreater(float((a - x).abs().mean()), 1e-3)

    def test_random_effects_shapes_range_and_one_draw_per_image(self):
        x = torch.rand(6, 3, 32, 32, generator=torch.Generator().manual_seed(0))
        fx = VI.CamFX(p_vignette=1, p_motion=1, p_blur=1, p_wb=1, p_gamma=1, p_noise=1, p_jpeg=1)
        g = torch.Generator().manual_seed(2)
        y = VI.fx_sensor(VI.fx_optics(x, fx, g, True), fx, g, True)
        self.assertEqual(y.shape, x.shape)
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 1.0)
        same = torch.stack([x[0]] * 6)                                  # one image 6 times: 6 different draws
        z = VI.fx_sensor(VI.fx_optics(same, fx, g, True), fx, g, True)
        self.assertGreater(float((z[0] - z[1]).abs().max()), 1e-3)

    def test_jpeg_quality(self):
        g = torch.Generator().manual_seed(3)
        x = torch.nn.functional.interpolate(torch.rand(2, 3, 8, 8, generator=g), size=(64, 64), mode="bilinear",
                                            align_corners=False)          # smooth: chroma subsampling loses little
        e = [float((VI.jpeg_sim(x, torch.full((2,), q)) - x).abs().mean()) for q in (100.0, 90.0, 30.0)]
        self.assertLess(e[0], 0.01)
        self.assertLess(e[0], e[1])
        self.assertLess(e[1], e[2])
        y = VI.jpeg_sim(torch.rand(1, 3, 30, 22, generator=g), torch.tensor([75.0]))   # not multiples of 8 / 16
        self.assertEqual(tuple(y.shape), (1, 3, 30, 22))

    def test_per_camera_and_unknown_keys(self):
        f = VI.CamFX.for_cameras({"p_jpeg": 0.5, "per_camera": {"wrist": {"p_motion": 0.4}}}, ("external", "wrist"))
        self.assertEqual((f[0].p_jpeg, f[0].p_motion, f[1].p_jpeg, f[1].p_motion), (0.5, 0.0, 0.5, 0.4))
        with self.assertRaises(ValueError):
            VI.CamFX.for_cameras({"p_jepg": 0.5}, ("external",))
        with self.assertRaises(ValueError):
            VI.CamFX.for_cameras({"per_camera": {"top": {}}}, ("external", "wrist"))

    def test_to_input_without_fx_is_the_old_function(self):
        rgb = torch.randint(0, 256, (4, 2, 16, 16, 3), dtype=torch.uint8, generator=torch.Generator().manual_seed(4))
        d = VI.DRConfig(p_bright=1, p_contrast=1, p_sat=1, p_hue=1, p_noise=1, p_blur=1)
        a = VI.to_input(rgb, d, torch.Generator().manual_seed(5))
        b = old_to_input(rgb, d, torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(VI.to_input(rgb), old_to_input(rgb)))

    def test_view_fx_nominal_vs_training(self):
        rgb = torch.randint(0, 256, (3, 2, 16, 16, 3), dtype=torch.uint8, generator=torch.Generator().manual_seed(6))
        fx = VI.ViewFX(VI.CamFX.for_cameras({"fixed_blur": 0.5, "p_wb": 1.0}, ("external", "wrist")), None)
        n1 = VI.to_input(rgb, fx=fx)
        n2 = VI.to_input(rgb, VI.DRConfig(on=False), torch.Generator().manual_seed(0), fx=fx)
        self.assertTrue(torch.equal(n1, n2))                            # nominal: the fixed effects only
        t = VI.to_input(rgb, VI.DRConfig(p_bright=0, p_contrast=0, p_sat=0, p_hue=0, p_noise=0, p_blur=0),
                        torch.Generator().manual_seed(0), fx=fx)
        self.assertFalse(torch.equal(n1, t))                            # training: + the random white balance


class FakeEnv:
    """``JointVecEnv.step`` stand-in: 5 steps, q += 1 per step (q[:, 0] counts the steps), hook after each."""

    def __init__(self, B):
        self.ws = SimpleNamespace(q=torch.zeros(B, 6, dtype=torch.float64), qd=torch.zeros(B, 6, dtype=torch.float64),
                                  cube_pos=torch.zeros(B, 3, dtype=torch.float64),
                                  cube_quat=torch.tensor([[1.0, 0, 0, 0]] * B, dtype=torch.float64))

    def step(self, actions=None, targets=None, use_targets=None, step_hook=None):
        for j in range(JE.STEPS):
            self.ws = SimpleNamespace(**{**vars(self.ws), "q": self.ws.q + 1.0})
            if step_hook is not None:
                step_hook(self, j)


class TestImageLag(unittest.TestCase):
    def test_step_tap_composes_and_restores(self):
        env = FakeEnv(2)
        log, seen = VI.SubstepLog(), []
        with VI.step_tap(env, log):
            env.step(step_hook=lambda e, j: seen.append(j))
            env.step()
        self.assertEqual(seen, list(range(JE.STEPS)))
        self.assertNotIn("step", vars(env))                             # restored
        a = log.array()                                                 # [B, D, S, 13]
        self.assertEqual(a.shape, (2, 2, JE.STEPS, 13))
        self.assertEqual(a[0, :, :, 0].ravel().tolist(), [float(k) for k in range(1, 11)])

    def test_lag_rows_from_the_substep_log(self):
        B, D, S = 2, 3, JE.STEPS
        sub = np.zeros((B, D, S, 13), np.float32)
        sub[..., 0] = np.arange(1, D * S + 1, dtype=np.float32).reshape(1, D, S)      # q0 = global step count
        sub[..., 9] = 1.0
        lengths = [3, 2]
        rb = np.repeat(np.arange(B), lengths)
        rt = np.concatenate([np.arange(n) for n in lengths])
        q = np.zeros((len(rb), 6), np.float32)
        q[:, 0] = rt * S                                                # boundary t = step 5 t
        r = dict(row_ep=rb, row_t=rt, row_q=q, row_cube_pos=np.zeros((len(rb), 3), np.float32),
                 row_cube_quat=np.tile([1.0, 0, 0, 0], (len(rb), 1)).astype(np.float32))
        lag = VI.lag_rows(r, sub, 4)
        self.assertEqual(lag.shape, (len(rb), 4, 13))
        for i in range(len(rb)):
            for k in range(1, 5):
                want = max(0, rt[i] * S - k) if rt[i] > 0 else 0
                self.assertEqual(float(lag[i, k - 1, 0]), float(want))
        bad = dict(r, row_q=q + 1)
        with self.assertRaises(RuntimeError):
            VI.lag_rows(bad, sub, 4)

    def test_interpolation_without_recorded_states(self):
        rows_q = np.zeros((4, 6))
        rows_q[:, 0] = [0.0, 1.0, 2.0, 10.0]
        z = _NpzLike(dict(row_q=rows_q, row_cube_pos=np.zeros((4, 3)), row_cube_quat=np.tile([1.0, 0, 0, 0], (4, 1)),
                          row_t=np.array([0, 1, 2, 0])))
        lag = VI.lag_interp(z, 2)
        self.assertAlmostEqual(float(lag[1, 0, 0]), 1.0 - 1.0 / JE.STEPS, places=6)   # 20 ms back toward row 0
        self.assertAlmostEqual(float(lag[2, 1, 0]), 2.0 - 2.0 / JE.STEPS, places=6)
        self.assertEqual(float(lag[0, 1, 0]), 0.0)                       # first row of an episode: itself
        self.assertEqual(float(lag[3, 0, 0]), 10.0)

    def test_history_fill_push_get(self):
        h = VI.LagHistory(3, 3, "cpu")
        q = torch.zeros(3, 6)
        cp, cq = torch.zeros(3, 3), torch.tensor([[1.0, 0, 0, 0]] * 3)
        h.fill(torch.arange(3), q, cp, cq)
        for s in range(1, 5):
            h.push(q + s, cp, cq)
        got = h.get(torch.tensor([0, 1, 3]))[0][:, 0]
        self.assertEqual(got.tolist(), [4.0, 3.0, 1.0])
        h.fill(torch.tensor([1]), q[:1] + 9, cp[:1], cq[:1])           # new episode: the start state everywhere
        self.assertEqual(h.get(torch.tensor([2, 2, 2]))[0][:, 0].tolist(), [2.0, 9.0, 2.0])

    def test_actor_renders_the_lagged_state_and_needs_the_tap(self):
        class Views:
            fx = None

            def __init__(self):
                self.seen = []

            def draw(self, n):
                return None

            def render(self, q, cp, cq, gp, dr=None, light=None):
                self.seen.append(q[:, 0].clone())
                return torch.zeros(len(q), 2, 8, 8, 3, dtype=torch.uint8)

        cfg = VI.ViralConfig(res=8, keypoints=2, pool_dim=4, hidden=(8,), amp=False, lag=2, cameras=("external", "wrist"))
        st = VI.ViralNet(cfg, dict(robot_mean=torch.zeros(12), robot_std=torch.ones(12), tgt_mean=torch.zeros(10, 6),
                                   tgt_std=torch.ones(10, 6), aux_mean=torch.zeros(7), aux_std=torch.ones(7)),
                         pretrained=False).eval()
        views, env = Views(), FakeEnv(2)
        actor = VI.Actor(st, views, device="cpu")
        self.assertTrue(actor.needs_tap)
        actor.reset(2)
        goal = torch.zeros(2, 3)
        with VI.step_tap(env, actor.push):
            actor(env.ws, goal, 0)
            env.step()
            actor(env.ws, goal, 1)
        self.assertEqual(views.seen[0].tolist(), [0.0, 0.0])
        self.assertEqual(views.seen[1].tolist(), [3.0, 3.0])            # q 5 now, the image from 2 steps earlier
        env.step()                                                      # without the tap: refused
        with self.assertRaises(RuntimeError):
            actor(env.ws, goal, 2)
        a0 = VI.Actor(st, views, device="cpu", lag=0)
        self.assertFalse(a0.needs_tap)

    def test_student_set_lag_batch(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r0.npz"
            N = 4
            q = np.zeros((N, 6), np.float32)
            q[:, 0] = [0, 5, 10, 0]
            lag = np.zeros((N, 4, 13), np.float32)
            lag[:, :, 0] = q[:, :1] - np.arange(1, 5)[None]
            lag[:, :, 9] = 1
            np.savez(p, row_q=q, row_qd=np.zeros((N, 6), np.float32), row_cube_pos=np.zeros((N, 3), np.float32),
                     row_cube_quat=np.tile([1.0, 0, 0, 0], (N, 1)).astype(np.float32),
                     row_lab=np.zeros((N, 5, 6), np.float32), row_ep=np.array([0, 0, 0, 1]), row_t=np.array([0, 1, 2, 0]),
                     length=np.array([3, 1]), ep_row_start=np.array([0, 3]), goal_xy=np.zeros((2, 2)),
                     goal_yaw=np.zeros(2), src=np.zeros(2), lenient=np.zeros(2, bool), success=np.zeros(2, bool),
                     row_lag=lag)
            ds = VI.StudentSet([p], "cpu", 10, val_frac=0, lag=3)
            self.assertEqual(ds.lag_exact, [True])
            b = ds.batch(torch.tensor([1, 2]), lag=torch.tensor([0, 3]))
            self.assertEqual(b["img_q"][:, 0].tolist(), [5.0, 7.0])
            self.assertEqual(b["q"][:, 0].tolist(), [5.0, 10.0])        # robot input stays current
            b0 = ds.batch(torch.tensor([1, 2]))
            self.assertTrue(torch.equal(b0["img_q"], b0["q"]))
            no = VI.StudentSet([p], "cpu", 10, val_frac=0)
            with self.assertRaises(ValueError):
                no.batch(torch.tensor([1]), lag=torch.tensor([1]))


class _NpzLike:
    """A dict with ``files`` (as ``np.load``'s NpzFile)."""

    def __init__(self, d):
        self._d, self.files = dict(d), list(d)

    def __getitem__(self, k):
        return self._d[k]


class TestConfig(unittest.TestCase):
    def test_old_configs_keep_the_defaults(self):
        for name in ("viral_v1.json", "viral_v2.json", "viral_v3.json", "viral_v4.json"):
            c = VI.ViralConfig.from_dict(json.loads((REPO / "configs/so101_policy/vision" / name).read_text()))
            self.assertEqual((c.scene, c.lag, c.lag_range, c.lag_max, c.cam_fx, c.crops), ("", 0, (0, 0), 0, {}, {}))

    def test_r2s_config(self):
        c = VI.ViralConfig.from_dict(json.loads((REPO / "configs/so101_policy/vision/viral_r2s_v1.json").read_text()))
        self.assertEqual(c.cameras, ("external", "wrist"))
        self.assertEqual((c.lag, c.lag_range, c.lag_max, c.jaw_lead), (2, (1, 3), 3, 0))
        VI.CamFX.for_cameras(c.cam_fx, c.cameras)
        o = VI.OnlineConfig.from_dict(json.loads((REPO / "configs/so101_policy/vision/viral_r2s_online_v1.json")
                                                 .read_text()))
        self.assertEqual(o.iterations, 3000)


@unittest.skipUnless(HAVE_MUJOCO and CALIB.exists(), "needs mujoco (arm meshes) and the calibration")
class TestRealSetup(unittest.TestCase):
    def test_scene_file_crops_pin_and_change(self):
        with tempfile.TemporaryDirectory() as d:
            sp = placeholder_scene(Path(d))
            cfg = VI.ViralConfig(res=64, cameras=("external", "wrist"), scene=str(sp))
            rs = VI.load_real_setup(cfg)
            self.assertEqual(rs.crops, {"external": [80, 0, 480, 480], "wrist": [0, 0, 640, 480]})
            self.assertEqual(rs.calib_path, str(CALIB))
            self.assertEqual(set(rs.image_fx), {"external", "wrist"})
            VI.pin_real(cfg)
            self.assertEqual(len(cfg.scene_sha256), 64)
            self.assertEqual(cfg.crops, rs.crops)
            data = json.loads(sp.read_text())
            data["student_view"]["external"]["crop"] = [0, 0, 480, 480]
            sp.write_text(json.dumps(data))
            with self.assertRaises(ValueError):                         # the scene changed under the student
                VI.make_views(cfg, 2, device="cpu")
            with self.assertRaises(ValueError):
                VI.pin_real(cfg)
            cfg2 = VI.ViralConfig(res=32, cameras=("external", "wrist"), scene=str(sp))
            with self.assertRaises(ValueError):                         # student view output 64 != res 32
                VI.load_real_setup(cfg2)

    def test_relative_calib_and_override(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "scene"
            sub.mkdir()
            (Path(d) / "calib.json").write_text(CALIB.read_text())
            sp = placeholder_scene(sub)
            data = json.loads(sp.read_text())
            data["calib"] = "../calib.json"
            sp.write_text(json.dumps(data))
            rs = VI.load_real_setup(VI.ViralConfig(res=64, cameras=("external", "wrist"), scene=str(sp)))
            self.assertEqual(Path(rs.calib_path), (Path(d) / "calib.json").resolve())
            rs2 = VI.load_real_setup(VI.ViralConfig(res=64, cameras=("external", "wrist"), scene=str(sp),
                                                    calib=str(CALIB)))
            self.assertEqual(rs2.calib_path, str(CALIB))
            self.assertEqual(rs.calib_sha256, rs2.calib_sha256)


@unittest.skipUnless(HAVE_MUJOCO, "the HOME task warm-up needs mujoco")
class TestHomeTasks(unittest.TestCase):
    def test_home_tasks_and_mix(self):
        from nedm.so101_policy.vision import tray as TR
        home = {"noise_deg": 3.0}
        t = TR.make_tasks("H", 4, first=900, dev=True, log=lambda s: None, home=home)
        self.assertEqual(t["seeds"].tolist(), list(range(597_900, 597_904)))
        dq = np.degrees(t["start_q"]) - np.asarray(TR.HOME_SIM_DEG)
        self.assertLessEqual(float(np.abs(dq).max()), 3.0 + 1e-9)
        self.assertGreater(float(np.abs(dq).max()), 0.0)
        self.assertLess(float(np.abs(t["state_qd"]).max()), 0.5)        # at rest after the warm-up
        r = np.hypot(t["goal_xy"][:, 0] - t["state_cube_pos"][:, 0], t["goal_xy"][:, 1] - t["state_cube_pos"][:, 1])
        self.assertTrue(((r > 0.05) & (r < 0.31)).all())
        t2 = TR.make_tasks("H", 4, first=900, dev=True, log=lambda s: None, home=home)
        self.assertTrue(np.array_equal(t["start_q"], t2["start_q"]))    # seeded
        c = {k: v for k, v in t.items()}
        c["seeds"] = c["seeds"] + 100_000
        m = TR.mix_tasks([t, c], [3, 2], seed=1)
        self.assertEqual(len(m["seeds"]), 5)
        self.assertEqual(sorted(m["kind"].tolist()), [0, 0, 0, 1, 1])
        with self.assertRaises(ValueError):
            TR.mix_tasks([t, t], [2, 2])                                # duplicate seeds
        with self.assertRaises(ValueError):
            TR._home_cfg({"home": 1})

    def test_home_pose_is_the_robot_scripts(self):
        from nedm.so101_policy.vision import tray as TR
        import sys
        name = "_r2s_test_hw_lib"
        spec = importlib.util.spec_from_file_location(name, REPO / "scripts/so101_policy/real2sim/hw/hw_lib.py")
        H = importlib.util.module_from_spec(spec)
        sys.modules[name] = H                                           # dataclasses need the module registered
        try:
            spec.loader.exec_module(H)
        finally:
            sys.modules.pop(name, None)
        self.assertEqual(tuple(H.HOME_SIM_DEG), tuple(TR.HOME_SIM_DEG))
        cfg = json.loads((REPO / "configs/so101_policy/vision/viral_r2s_home.json").read_text())
        self.assertEqual(tuple(cfg["home_deg"]), tuple(H.HOME_SIM_DEG))


@unittest.skipUnless(GPU and HAVE_MUJOCO and CALIB.exists(), "needs newton on CUDA, mujoco and the calibration")
class TestRealViewsOnGpu(unittest.TestCase):
    def test_render_draws_nominal_merge(self):
        with tempfile.TemporaryDirectory() as d:
            sp = placeholder_scene(Path(d))
            cfg = VI.pin_real(VI.ViralConfig(res=64, cameras=("external", "wrist"), scene=str(sp)))
            v = VI.make_views(cfg, 4, seed=0)
            q = torch.tensor([[0.0, -0.09, -0.26, 1.6, 0.0, 0.5]] * 3, device="cuda")
            cp = torch.tensor([[0.25, 0.0, 0.015]] * 3, device="cuda")
            cq = torch.tensor([[1.0, 0, 0, 0]] * 3, device="cuda")
            gp = torch.tensor([[0.2, 0.1, 0.0]] * 3, device="cuda")
            a = v.render(q, cp, cq, gp)
            self.assertEqual(tuple(a.shape), (3, 2, 64, 64, 3))
            d_ = v.draw(3)
            b = v.render(q, cp, cq, gp, dr=d_, light=True)
            self.assertFalse(torch.equal(a, b))
            self.assertTrue(torch.equal(a, v.render(q, cp, cq, gp)))     # back to nominal: the same pictures
            m = v.merge_nominal(d_, torch.tensor([True, False, True], device="cuda"))
            c = v.render(q, cp, cq, gp, dr=m, light=False)
            self.assertTrue(torch.equal(c[0], a[0]) and torch.equal(c[2], a[2]))
            self.assertTrue(torch.equal(c[1], v.render(q, cp, cq, gp, dr=d_, light=False)[1]))
            self.assertIsNotNone(v.fx)
            x = VI.to_input(a, fx=v.fx)
            self.assertEqual(tuple(x.shape), (3, 2, 3, 64, 64))

class TestQBias(unittest.TestCase):
    """OnlineConfig.q_bias_*: per-world offset of the student's q input, drawn at the episode start."""

    def test_bias_reset(self):
        import types
        import torch
        from nedm.so101_policy.vision import viral as VI
        o = VI.OnlineConfig.from_dict(dict(q_bias_deg=[1, 3, 2, 2, 3, 0], q_bias_p=0.7))
        st = types.SimpleNamespace(o=o, dev=torch.device("cpu"), qb=torch.zeros(4000, 6),
                                   qb_h=torch.deg2rad(torch.tensor([1.0, 3, 2, 2, 3, 0])),
                                   qb_gen=torch.Generator().manual_seed(0))
        VI.OnlineDAgger._bias_reset(st, torch.arange(4000))
        d = torch.rad2deg(st.qb)
        self.assertTrue(bool((d.abs() <= torch.tensor([1.0, 3, 2, 2, 3, 0]) + 1e-5).all()))
        self.assertEqual(float(d[:, 5].abs().max()), 0.0)
        on = (d.abs().sum(1) > 0).float().mean()
        self.assertAlmostEqual(float(on), 0.7, delta=0.03)
        self.assertGreater(float(d[:, 1].abs().max()), 2.9)
        keep = st.qb.clone()
        VI.OnlineDAgger._bias_reset(st, torch.tensor([0, 1]))            # only the reset worlds change
        self.assertTrue(torch.equal(st.qb[2:], keep[2:]))

    def test_off_by_default(self):
        from nedm.so101_policy.vision import viral as VI
        o = VI.OnlineConfig.from_dict({})
        self.assertEqual(o.q_bias_p, 0.0)
        self.assertEqual(tuple(o.q_bias_deg), ())


if __name__ == "__main__":
    unittest.main()
