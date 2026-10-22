"""Tests of the real-robot side of the SO101 camera student (``scripts/so101_policy/real2sim/hw``): the joint conversion
(real LeRobot units <-> sim rad), the LeRobot-unit limits, the safety limits (action layer and hardware guard), the
table floor without mujoco (against the MuJoCo path when mujoco and the vendor model are present), image
preprocessing, joint paths, the dry-mode write block, and the dry / live robot loop and the replay path on a fake robot
/ a synthetic recording with a tiny random student. CPU, about 30 s:

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_real2sim_hw -v
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "scripts" / "so101_policy" / "real2sim" / "hw"
sys.path.insert(0, str(HW))
import hw_lib as H  # noqa: E402

torch.set_num_threads(2)
KIN_POINTS = H.default_kin_points()
needs_points = unittest.skipIf(KIN_POINTS is None, "kin_points_v1.npz not found")
# LeRobot calibration of the real follower (copy of the robot machine's so101_follower.json, 2026-10-20)
LR_CAL = {"shoulder_pan": dict(id=1, drive_mode=0, homing_offset=1252, range_min=999, range_max=2937),
          "shoulder_lift": dict(id=2, drive_mode=0, homing_offset=-1476, range_min=811, range_max=3194),
          "elbow_flex": dict(id=3, drive_mode=0, homing_offset=1483, range_min=840, range_max=3030),
          "wrist_flex": dict(id=4, drive_mode=0, homing_offset=-673, range_min=914, range_max=3263),
          "wrist_roll": dict(id=5, drive_mode=0, homing_offset=-182, range_min=0, range_max=4095),
          "gripper": dict(id=6, drive_mode=0, homing_offset=758, range_min=2033, range_max=3551)}


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_student_real", HW / "run_student_real.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def lerobot_unnormalize_deg(val, c):
    """LeRobot 0.6.2 SerialMotorsBus._unnormalize, DEGREES mode (the raw tick a degree command becomes)."""
    mid = (c["range_min"] + c["range_max"]) / 2
    return int((val * 4095 / 360) + mid)


class TestJointMap(unittest.TestCase):
    def test_formula_and_round_trip(self):
        jm = H.JointMap([1, -1, 1, -1, 1], [0.1, -0.2, 0.0, 0.3, -1.0], 0.015, -0.1)
        real = np.array([30.0, -45.0, 10.0, 80.0, -120.0, 50.0])
        sim = jm.real_to_sim(real)
        want = [math.radians(30) + 0.1, -math.radians(-45) - 0.2, math.radians(10), -math.radians(80) + 0.3,
                math.radians(-120) - 1.0, 0.015 * 50 - 0.1]
        np.testing.assert_allclose(sim, want, atol=1e-12)
        np.testing.assert_allclose(jm.sim_to_real(sim), real, atol=1e-10)
        rng = np.random.default_rng(0)
        for _ in range(20):
            jm = H.JointMap(rng.choice([-1.0, 1.0], 5), rng.uniform(-1, 1, 5), rng.uniform(0.005, 0.03) *
                            rng.choice([-1, 1]), rng.uniform(-0.5, 0.5))
            r = rng.uniform(-150, 150, (50, 6))
            np.testing.assert_allclose(jm.sim_to_real(jm.real_to_sim(r)), r, atol=1e-9)
            s = rng.uniform(-2, 2, (50, 6))
            np.testing.assert_allclose(jm.real_to_sim(jm.sim_to_real(s)), s, atol=1e-12)

    def test_placeholder_and_calib(self):
        jm = H.JointMap.from_calib(None)
        self.assertTrue(jm.placeholder)
        np.testing.assert_allclose(jm.real_to_sim([90, 0, 0, 0, 0, 100]), [math.pi / 2, 0, 0, 0, 0, math.radians(100)])
        cal = dict(joint_map=dict(names=list(H.JOINTS), sign=[1, -1, 1, 1, -1], offset_rad=[0, 0.1, 0, 0, 0],
                                  gripper_lin=[0.02, -0.17], evidence="test"))
        jm = H.JointMap.from_calib(cal)
        self.assertFalse(jm.placeholder)
        self.assertEqual(jm.sign.tolist(), [1, -1, 1, 1, -1])

    def test_validation(self):
        with self.assertRaises(ValueError):
            H.JointMap([1, 1, 0.5, 1, 1], np.zeros(5), 0.01, 0)
        with self.assertRaises(ValueError):
            H.JointMap(np.ones(5), np.zeros(5), 0.0, 0)
        with self.assertRaises(ValueError):
            H.JointMap.from_dict(dict(names=["a"] * 6, sign=[1] * 5, offset_rad=[0] * 5, gripper_lin=[0.01, 0]))


class TestRealLimits(unittest.TestCase):
    def test_from_lerobot_calibration(self):
        lim = H.RealLimits.from_lerobot_calibration(LR_CAL, margin_deg=3.0)
        self.assertAlmostEqual(lim.lo[0], (999 - 1968) * 360 / 4095 + 3, places=9)       # pan -85.2 + 3
        self.assertAlmostEqual(lim.hi[0], (2937 - 1968) * 360 / 4095 - 3, places=9)
        self.assertAlmostEqual(lim.hi[4], 2047.5 * 360 / 4095 - 3, places=9)            # full-turn wrist roll
        self.assertEqual((lim.lo[5], lim.hi[5]), (0.0, 100.0))
        c, m = lim.clip([200, 0, 0, 0, 0, 120])
        self.assertAlmostEqual(c[0], lim.hi[0])
        self.assertEqual(c[5], 100.0)
        self.assertEqual(m.tolist(), [True, False, False, False, False, True])
        # every clipped degree command lands inside the motor's position-limit registers (LeRobot's tick formula)
        for i, mname in enumerate(H.JOINTS[:5]):
            for v in (lim.lo[i], lim.hi[i]):
                raw = lerobot_unnormalize_deg(v, LR_CAL[mname])
                self.assertTrue(LR_CAL[mname]["range_min"] <= raw <= LR_CAL[mname]["range_max"], (mname, v, raw))
        bad = json.loads(json.dumps(LR_CAL))
        bad["elbow_flex"]["drive_mode"] = 1
        with self.assertRaises(ValueError):
            H.RealLimits.from_lerobot_calibration(bad)


class TestQdot(unittest.TestCase):
    def test_ramp_window_gap(self):
        est = H.QdotEstimator(window=3, max_gap=0.1)
        v = np.array([0.5, -1.0, 0.0, 2.0, 0.1, -0.3])
        np.testing.assert_allclose(est.update(0.0, np.zeros(6)), 0.0)
        for k in range(1, 6):
            qd = est.update(k * 0.02, v * k * 0.02)
            np.testing.assert_allclose(qd, v, atol=1e-9)
        self.assertEqual(len(est.t), 3)
        np.testing.assert_allclose(est.update(1.0, np.ones(6)), 0.0)                    # gap: restart


@needs_points
class TestActionLayerAndGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nedm.so101_policy import joint_env as JE
        cls.JE = JE
        cls.layer = H.ActionLayer(KIN_POINTS)
        cls.lim = H.RealLimits.from_lerobot_calibration(LR_CAL, 3.0)
        cls.jm = H.JointMap.placeholder_identity()
        cls.home = np.radians(H.HOME_SIM_DEG)
        cls.mid = np.radians([0.0, -5.0, -15.0, 80.0, 0.0, 30.0])        # away from every joint limit

    def guard(self, **kw):
        return H.Guard(H.GuardConfig(**kw), self.layer, self.jm, self.lim)

    def test_equals_joint_env_safety(self):
        g = torch.Generator().manual_seed(0)
        q = self.home + 0.3 * (torch.rand(64, 6, generator=g, dtype=torch.float64).numpy() - 0.5)
        u = q + 0.4 * (torch.rand(64, 6, generator=g, dtype=torch.float64).numpy() - 0.5)
        k = self.layer.kin
        want = self.JE.safety(torch.as_tensor(u), torch.as_tensor(q), k.tlo, k.thi,
                              torch.tensor(self.JE.LEAD, dtype=torch.float64), k, self.JE.FLOOR).numpy()
        got = np.stack([self.layer(u[i], q[i])[0] for i in range(len(q))])
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_lead_band_and_floor(self):
        q = self.mid.copy()
        u, info = self.layer(q + 1.0, q)
        np.testing.assert_allclose(u[:5], q[:5] + np.asarray(self.JE.LEAD), atol=1e-12)
        self.assertTrue(info["lead"])
        # a measured pose just above the floor and a target that dips the gripper into the table
        q2 = np.radians([0.0, 30.0, 40.0, 92.0, 0.0, 20.0])
        low_q = float(self.layer.lowest(q2))
        u2 = q2.copy()
        u2[1] += 0.12
        if float(self.layer.lowest(u2)) < self.JE.FLOOR and low_q > self.JE.FLOOR:
            v, info = self.layer(u2, q2)
            self.assertTrue(info["floor"])
            self.assertGreaterEqual(float(self.layer.lowest(v)), self.JE.FLOOR - 1e-6)
        # the floor follows table_z
        lay2 = H.ActionLayer(KIN_POINTS, table_z=0.05)
        self.assertAlmostEqual(lay2.floor, self.JE.FLOOR + 0.05)

    def test_floor_pulls_back_a_low_target(self):
        """Search a target below the floor from a measured pose above it: the layer output is at the floor."""
        rng = np.random.default_rng(1)
        hit = 0
        for _ in range(400):
            q = np.radians(rng.uniform([-60, 0, -20, 40, -90, 0], [60, 60, 60, 92, 90, 60]))
            if float(self.layer.lowest(q)) < 0.01:
                continue
            u = q.copy()
            u[:5] += rng.uniform(-1, 1, 5) * np.asarray(self.JE.LEAD)
            if float(self.layer.lowest(u)) >= self.JE.FLOOR:
                continue
            v, info = self.layer(u, q)
            hit += 1
            self.assertTrue(info["floor"])
            self.assertGreaterEqual(float(self.layer.lowest(v)), self.JE.FLOOR - 2e-4)
        self.assertGreater(hit, 0, "no low target found")

    def test_step_limit_max_rel_range_nan(self):
        q = self.mid.copy()
        qr = self.jm.sim_to_real(q)
        g = self.guard()
        g.reset(q)
        u = q.copy()
        u[5] += 1.0                                                                 # jaw: no lead band
        us, ur, ev, stop = g.apply(u, q, qr)
        self.assertAlmostEqual(us[5], q[5] + 0.5, places=9)
        self.assertTrue(ev["step"])
        self.assertEqual(stop, [])
        g = self.guard(max_step_rad=(0.01,) * 6)
        g.reset(q)
        us, ur, ev, stop = g.apply(q + 0.04, q, qr)
        np.testing.assert_allclose(us, q + 0.01, atol=1e-9)
        g = self.guard(max_rel_real=(0.5,) * 6)
        g.reset(q)
        us, ur, ev, stop = g.apply(q + 0.04, q, qr)
        self.assertLessEqual(float(np.abs(ur - qr).max()), 0.5 + 1e-9)
        self.assertTrue(ev["max_rel"])
        # real range: measured near the pan limit, command beyond
        qe = q.copy()
        qe[0] = math.radians(self.lim.hi[0] - 0.01)
        g = self.guard()
        g.reset(qe)
        us, ur, ev, stop = g.apply(qe + np.array([0.04, 0, 0, 0, 0, 0]), qe, self.jm.sim_to_real(qe))
        self.assertLessEqual(ur[0], self.lim.hi[0] + 1e-9)
        self.assertTrue(ev["real_range"])
        us, ur, ev, stop = g.apply(np.full(6, np.nan), q, qr)
        self.assertIsNone(us)
        self.assertTrue(stop and stop[0].startswith("student target"))

    def test_measured_checks_and_stall(self):
        g = self.guard()
        q = self.home.copy()
        self.assertEqual(g.check_measured(q, self.jm.sim_to_real(q)), [])
        self.assertTrue(g.check_measured(q * np.nan, q)[0].startswith("measured q"))
        low = np.radians([0.0, 80.0, 40.0, 60.0, 0.0, 20.0])                        # gripper 55 mm into the table
        self.assertLess(float(self.layer.lowest(low)), -0.005)
        r = g.check_measured(low, self.jm.sim_to_real(low))
        self.assertTrue(any(s.startswith("measured floor") for s in r), r)
        far = np.radians([0.0, 90.0, -90.0, 0.0, 0.0, 20.0])                       # arm stretched out: x 0.47 m
        r = g.check_measured(far, self.jm.sim_to_real(far))
        self.assertTrue(any("workspace box" in s or "reach" in s for s in r), r)
        out = self.jm.sim_to_real(q)
        out[0] = self.lim.hi[0] + 10
        self.assertTrue(any(s.startswith("measured range") for s in g.check_measured(q, out)))
        # stall: a joint held at its lead band
        cmd = q.copy()
        cmd[1] += self.JE.LEAD[1]
        g = self.guard(stall_steps=5)
        for i in range(4):
            self.assertEqual(g.stall(q, cmd), [])
        self.assertTrue(g.stall(q, cmd)[0].startswith("stall"))
        self.assertEqual(g.stall(q, q), [])                                          # moving again: reset

    def test_paths(self):
        rest = np.radians([0.0, -88.0, 84.0, 65.0, 0.0, 20.0])
        path, dur = H.plan_joint_path(rest, self.home, math.radians(15.0))
        np.testing.assert_allclose(path[-1], self.home, atol=1e-12)
        sp = np.abs(np.diff(np.vstack((rest, path)), axis=0)).max(0) / H.DT
        self.assertTrue(np.all(sp <= math.radians(15.0) * (1 + 1e-3)), sp)
        self.assertEqual(H.check_path(path, self.layer, self.jm, self.lim, start=rest), [])
        # a path through the table is refused
        low = np.radians([0.0, 80.0, 40.0, 60.0, 0.0, 20.0])
        p1, _ = H.plan_joint_path(self.home, low, 1.0)
        p2, _ = H.plan_joint_path(low, self.home, 1.0)
        bad = H.check_path(np.vstack((p1, p2)), self.layer, self.jm, self.lim, start=self.home)
        self.assertTrue(any("lowest gripper point" in s for s in bad), bad)


class TestImages(unittest.TestCase):
    def test_crop_resize_colour_order(self):
        img = np.zeros((480, 640, 3), np.uint8)
        img[:, :, 0] = 200                                    # BGR: blue channel
        v = H.centre_square(640, 480, 224)
        self.assertEqual(v.crop_xywh, (80, 0, 480, 480))
        a = H.preprocess(img, v, bgr=True)
        self.assertEqual(a.shape, (224, 224, 3))
        self.assertEqual(a[..., 2].min(), 200)                 # now RGB: blue last
        self.assertEqual(a[..., 0].max(), 0)
        b = H.preprocess(img, v, bgr=False)
        self.assertEqual(b[..., 0].min(), 200)
        with self.assertRaises(ValueError):
            H.preprocess(img, H.ViewSpec((500, 0, 200, 200), (224, 224)), True)
        cal = dict(student_view=dict(external=dict(crop_xywh=[10, 20, 300, 300], out_wh=[224, 224]),
                                     wrist=dict(crop_xyxy=[0, 0, 480, 480])))
        vs = H.view_specs(cal, H.REAL_CAMERAS, 224)
        self.assertEqual(vs["external"].crop_xywh, (10, 20, 300, 300))
        self.assertEqual(vs["wrist"].crop_xywh, (0, 0, 480, 480))
        self.assertEqual(H.view_specs(None, ("wrist",), 128)["wrist"].out_wh, (128, 128))

    def test_cam_map(self):
        self.assertEqual(H.resolve_cam_map(("top", "wrist")), {"top": "external", "wrist": "wrist"})
        self.assertEqual(H.resolve_cam_map(("external", "wrist")), {"external": "external", "wrist": "wrist"})
        with self.assertRaises(ValueError):
            H.resolve_cam_map(("top", "wrist"), "top=wrist")


class TestWriteBlock(unittest.TestCase):
    def test_dry_blocks_every_writer(self):
        R = _load_runner()

        class Bus:
            def __getattr__(self, n):
                raise AttributeError(n)

        bus, robot = Bus(), Bus()
        for n in R.BUS_WRITERS + ("sync_read", "read", "disconnect"):
            setattr(bus, n, lambda *a, **k: "ok")
        for n in R.ROBOT_WRITERS:
            setattr(robot, n, lambda *a, **k: "ok")
        done = R.block_writes(bus, robot)
        self.assertEqual(len(done), len(R.BUS_WRITERS) + len(R.ROBOT_WRITERS))
        for n in R.BUS_WRITERS:
            with self.assertRaises(RuntimeError):
                getattr(bus, n)("Goal_Position", {})
        for n in R.ROBOT_WRITERS:
            with self.assertRaises(RuntimeError):
                getattr(robot, n)()
        self.assertEqual(bus.sync_read("Present_Position"), "ok")                    # reads stay


@needs_points
class TestNoMujocoOnTheRobot(unittest.TestCase):
    def test_import_and_floor_without_mujoco_newton_warp_lerobot(self):
        """The Orin has no mujoco: hw_lib + the action layer + the student code must work with mujoco, newton, warp and
        lerobot imports failing."""
        code = f"""
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in ('mujoco', 'newton', 'warp', 'lerobot', 'mujoco_warp'):
            raise ImportError('blocked ' + name)
sys.meta_path.insert(0, Block())
sys.path[:0] = [{str(HW)!r}, {str(ROOT / 'src')!r}]
import numpy as np, hw_lib as H
from nedm.so101_policy.vision import viral
layer = H.ActionLayer({str(KIN_POINTS)!r})
q = np.radians(H.HOME_SIM_DEG)
u, info = layer(q + 0.5, q)
assert np.all(np.isfinite(u))
print('ok', float(layer.lowest(q)), [m for m in sys.modules if m.split('.')[0] in ('mujoco', 'newton', 'warp', 'lerobot')])
"""
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                           env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"})
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertTrue(r.stdout.strip().startswith("ok"), r.stdout)
        self.assertTrue(r.stdout.strip().endswith("[]"), r.stdout)

    def test_saved_points_equal_mujoco(self):
        try:
            import mujoco  # noqa: F401

            from nedm.contact_nrd.collect import so101_grasp as g
            scene = g.Scene(g.load_config())
        except BaseException as e:                                                    # noqa: BLE001
            self.skipTest(f"mujoco / vendor model not available: {e}")
        from nedm.so101_policy import kin as K
        rng = np.random.default_rng(3)
        q = torch.as_tensor(np.column_stack([rng.uniform(lo, hi, 300) for lo, hi in zip(K.TLO, K.THI)]))
        saved = K.SO101Kin(points=str(KIN_POINTS))
        mj = K.SO101Kin(points=None)                                                  # points from MuJoCo now
        for exact in (True, False):
            np.testing.assert_allclose(saved.lowest_point(q, exact=exact).numpy(), mj.lowest_point(q, exact=exact).numpy(),
                                       atol=1e-12)
        # and the saved-point floor against MuJoCo's own Scene.lowest (gripper + moving jaw), 50 poses
        m = scene.m
        ids = [m.body(n).id for n in K.GUARD_BODIES]
        low = saved.lowest_point(q[:50], per_body=True).numpy()
        for i in range(50):
            d = scene.fk(q[i].numpy())
            for j, b in enumerate(ids):
                self.assertLess(abs(scene.lowest(d, which=b) - low[i, j]), 1e-9)


# ---- the robot loop on a fake robot and the replay path, with a tiny random student ----------------------------------
def tiny_student(path, res=64):
    from nedm.so101_policy.vision import viral as VI
    cfg = VI.ViralConfig(res=res, encoder="small", keypoints=8, pool_dim=16, hidden=(32,), aux_hidden=16, amp=False)
    stats = dict(robot_mean=torch.zeros(12), robot_std=torch.ones(12), tgt_mean=torch.zeros(10, 6),
                 tgt_std=torch.full((10, 6), 0.01), aux_mean=torch.zeros(7), aux_std=torch.ones(7))
    torch.manual_seed(0)
    st = VI.ViralNet(cfg, stats, pretrained=False)
    VI.save_student(st, path)
    return path


class Clock:
    """poll() True after n calls (a scripted stop key)."""

    def __init__(self, n):
        self.n, self.k, self.why = n, 0, "test"

    def poll(self):
        self.k += 1
        return self.k > self.n


@needs_points
class TestRobotLoopFake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.R = _load_runner()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ck = tiny_student(Path(cls.tmp.name) / "tiny.pt")
        cls.layer = H.ActionLayer(KIN_POINTS)
        cls.lim = H.RealLimits.from_lerobot_calibration(LR_CAL, 3.0)
        cls.jm = H.JointMap.placeholder_identity()
        cls.pol = H.StudentPolicy(cls.ck, "cpu")
        cls.views = H.view_specs(None, H.REAL_CAMERAS, cls.pol.res)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def ctrl(self, gcfg=None):
        g = H.Guard(gcfg or H.GuardConfig(), self.layer, self.jm, self.lim)
        return H.Controller(self.pol, self.jm, g, self.views, H.QdotEstimator(), bgr=False), g

    def test_dry_never_writes(self):
        io = H.FakeRobotIO(self.jm.sim_to_real(np.radians(H.HOME_SIM_DEG)), writes=False)
        c, g = self.ctrl()
        reason, last, W = self.R.control_loop(io, c, g, self.jm, self.lim, Clock(30), False, 10.0, status_every=99)
        self.assertTrue(reason.startswith("user stop"))
        self.assertEqual(io.n_writes, 0)
        A = c.arrays()
        self.assertEqual(len(A["t"]), 30)
        self.assertEqual(A["query"].sum(), 6)                                       # every 5th step
        self.assertEqual(A["qry_step"].tolist(), [0, 5, 10, 15, 20, 25])
        with self.assertRaises(RuntimeError):
            io.write_goal(np.zeros(6))

    def test_live_writes_bounded_commands_then_hold(self):
        q0 = self.jm.sim_to_real(np.radians(H.HOME_SIM_DEG))
        io = H.FakeRobotIO(q0, writes=True)
        io.torque_on_at_present()
        c, g = self.ctrl()
        reason, last, W = self.R.control_loop(io, c, g, self.jm, self.lim, Clock(40), True, 10.0, status_every=99)
        self.assertTrue(reason.startswith("user stop"), reason)
        A = c.arrays()
        n = len(A["t"])
        self.assertEqual(io.n_writes, n)
        self.assertTrue(np.all(np.abs(A["u_real"] - A["q_real"]) <= np.asarray(g.cfg.max_rel_real) + 1e-9))
        self.assertTrue(np.all(A["u_real"] >= self.lim.lo - 1e-9) and np.all(A["u_real"] <= self.lim.hi + 1e-9))
        lead = np.asarray(self.layer.JE.LEAD)
        self.assertTrue(np.all(np.abs(A["u_sim"][:, :5] - A["q_sim"][:, :5]) <= lead + 1e-9))
        q = io.read_q()
        h = self.R.hold(io, self.jm, self.lim, q, last[5])
        np.testing.assert_allclose(h[:5], np.clip(q, self.lim.lo, self.lim.hi)[:5])
        self.assertAlmostEqual(h[5], float(np.clip(last[5], 0, 100)))
        np.testing.assert_allclose(io.goal, h)

    def test_live_stops_on_guard(self):
        io = H.FakeRobotIO(self.jm.sim_to_real(np.radians(H.HOME_SIM_DEG)), writes=True)
        io.torque_on_at_present()
        c, g = self.ctrl(H.GuardConfig(tcp_z=(0.5, 0.6)))                            # every pose is "outside"
        reason, last, W = self.R.control_loop(io, c, g, self.jm, self.lim, Clock(40), True, 10.0, status_every=99)
        self.assertTrue(reason.startswith("guard: measured workspace box"), reason)
        self.assertEqual(io.n_writes, 0)

    def test_home_move(self):
        rest = np.array([0.0, -88.0, 84.0, 65.0, 0.0, 20.0])
        io = H.FakeRobotIO(rest, writes=True)
        io.torque_on_at_present()
        home = np.radians(H.HOME_SIM_DEG)
        path, dur = H.plan_joint_path(self.jm.real_to_sim(rest), home, math.radians(90.0))
        g = H.Guard(H.GuardConfig(), self.layer, self.jm, self.lim)
        why, last = self.R.move_path(io, path, self.jm, self.lim, g, Clock(10 ** 6), np.asarray(g.cfg.max_rel_real))
        self.assertIsNone(why)
        self.assertLess(float(np.abs(self.jm.real_to_sim(io.read_q())[:5] - home[:5]).max()), 0.05)

    def test_replay_synthetic_recording(self):
        """A 2 s LeRobot-v3-shaped recording (parquet + mp4) through ``run_student_real.py replay``."""
        import cv2
        import pyarrow as pa
        import pyarrow.parquet as pq
        root = Path(self.tmp.name) / "rec"
        n, fps = 60, 30
        st = np.zeros((n, 6), np.float32)
        st[:, 0] = np.linspace(0, 10, n)
        st[:, 1:5] = H.HOME_SIM_DEG[1:5]
        st[:, 5] = 30
        (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (root / "data" / "chunk-000").mkdir(parents=True)
        (root / "meta" / "info.json").write_text(json.dumps(dict(fps=fps)))
        pq.write_table(pa.table({"observation.state": list(st), "action": list(st + 1), "episode_index": [0] * n,
                                 "frame_index": list(range(n)), "timestamp": list(np.arange(n) / fps)}),
                       root / "data" / "chunk-000" / "file-000.parquet")
        ep = {"episode_index": [0]}
        for c in H.REAL_CAMERAS:
            k = f"videos/observation.images.{c}"
            ep.update({f"{k}/chunk_index": [0], f"{k}/file_index": [0], f"{k}/from_timestamp": [0.0],
                       f"{k}/to_timestamp": [n / fps]})
            d = root / "videos" / f"observation.images.{c}" / "chunk-000"
            d.mkdir(parents=True)
            w = cv2.VideoWriter(str(d / "file-000.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
            if not w.isOpened():
                self.skipTest("cv2 VideoWriter (mp4v) unavailable")
            for i in range(n):
                f = np.full((480, 640, 3), 40, np.uint8)
                f[:, : 10 * i % 640] = (0, 200, 0)
                w.write(f)
            w.release()
        pq.write_table(pa.table(ep), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        out = Path(self.tmp.name) / "replay_out"
        lr = Path(self.tmp.name) / "lr.json"
        lr.write_text(json.dumps(LR_CAL))
        summ = self.R.main(["replay", "--student", str(self.ck), "--device", "cpu", "--data-root", str(root.parent),
                            "--episodes", "rec:0", "--out", str(out), "--calib", "", "--lerobot-calib", str(lr)])
        self.assertEqual(summ[0]["steps"], int(math.floor((n - 1) / fps / H.DT)) + 1)
        z = np.load(out / "rec_ep00.npz")
        self.assertEqual(z["qry_chunk"].shape[1:], (10, 6))
        np.testing.assert_allclose(z["q_real"][0], st[0], atol=1e-5)
        np.testing.assert_allclose(z["q_real"][5], st[3], atol=1e-4)                  # t 0.1 s = recorded frame 3


class TestDeviceUsers(unittest.TestCase):
    """The dry / live refusal while another program can use the robot bus or a camera."""

    def test_holders_and_tools(self):
        R = _load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp) / "ttyACM9"
            dev.write_text("")
            fake = Path(tmp) / "run_student_real.py"                                          # sleeps, like a run
            fake.write_text("import time\ntime.sleep(30)\n")
            procs = [subprocess.Popen(["sleep", "30"], stdin=dev.open()),                       # has the "port" open
                     subprocess.Popen(["bash", "-c", "exec -a lerobot-teleoperate sleep 30"]),  # a LeRobot robot tool
                     subprocess.Popen([sys.executable, str(fake), "live", "--i-am-supervising"]),  # a live run
                     subprocess.Popen([sys.executable, str(fake), "replay"])]                     # replay: no devices
            try:
                for _ in range(50):
                    busy = R.device_users([str(dev), str(Path(tmp) / "missing")])
                    if len(busy) >= 3:
                        break
                    time.sleep(0.05)
            finally:
                for pr in procs:
                    pr.kill()
                    pr.wait()
            got = {int(b.split()[0]): b for b in busy}
            self.assertIn(f"has {dev.resolve()} open", got.get(procs[0].pid, ""))
            self.assertIn("LeRobot robot tool", got.get(procs[1].pid, ""))
            self.assertIn("dry / live run of this script", got.get(procs[2].pid, ""))
            self.assertNotIn(procs[3].pid, got)
            self.assertEqual(R.device_users([str(dev)], skip=[pr.pid for pr in procs]), [])

    def test_no_abbreviated_flags(self):
        R = _load_runner()
        with self.assertRaises(SystemExit):              # "--time" must not mean --time-limit (robot_test.sh checks)
            R.main(["replay", "--student", "x.pt", "--time", "60"])


if __name__ == "__main__":
    unittest.main()
