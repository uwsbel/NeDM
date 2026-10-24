"""Tests of the robot deploy settings of the SO101 camera student (real2sim 2026-10-24,
``scripts/so101_policy/real2sim/hw``): the guard's anti-windup of the servo-error integral at the real joint range
(the wrist-roll stall of 2026-10-23), the floor mode (flat for students trained on the sim table, plane above the
calibrated mat for v8 table-trained students; no double lowering), the bundle's assets/deploy.json and MANIFEST, HOME
with wrist roll 0, and the removal of the two-student mode (user rule: no policy stitching). CPU, about 30 s:

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_real2sim_hw_deploy -v
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "scripts" / "so101_policy" / "real2sim" / "hw"
sys.path.insert(0, str(HW))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hw_lib as H  # noqa: E402
from test_real2sim_hw import LR_CAL, _load_runner, tiny_student  # noqa: E402

torch.set_num_threads(2)
KIN_POINTS = H.default_kin_points()
needs_points = unittest.skipIf(KIN_POINTS is None, "kin_points_v1.npz not found")
PLANE = (-0.004681260916979778, -0.03640954226125585, 0.01836320115536056)      # calib_v1.json scene.table_plane
ROLL_OFF = math.radians(-75.789)            # calib_v1 wrist-roll offset: the real range end +177 deg = 101.21 sim deg
V8_REC = {"v8_online": {"table_bias": {"p": 0.7, "drop_mm": [0.0, 22.0], "slope_x": [-55.0, 25.0],
                                       "slope_y": [-10.0, 35.0], "x_ref_m": 0.25},
                        "table_render": True, "plate_mm": [1.0, 5.0]}}


def tiny_v8(path, base):
    """The tiny student re-saved with a v8 table-randomisation record (``OnlineDAgger.ckpt_record``)."""
    from nedm.so101_policy.vision import viral as VI
    st, _ = VI.load_student(base, "cpu")
    VI.save_student(st, path, **V8_REC)
    return path


def calib_file(path, plane=PLANE):
    """A calib_v1-shaped file with a joint map and the mat plane (no crops: the tiny student has none)."""
    d = {"joint_map": {"sign": [1.0] * 5, "offset_rad": [0.0, 0.0, 0.0, 0.0, ROLL_OFF],
                       "gripper_lin": [math.pi / 180, 0.0], "names": list(H.JOINTS)}}
    if plane is not None:
        d["scene"] = {"table_plane": {"coef": list(plane)}}
    Path(path).write_text(json.dumps(d))
    return path


# ---- anti-windup ---------------------------------------------------------------------------------------------------
@needs_points
class TestAntiWindup(unittest.TestCase):
    """The robot day's wrist-roll stall: measured roll 100.69 sim deg at the real stop, target ~101.0 deg, the LeRobot
    range end 101.21 deg (calibrated +-180 real deg - 3 deg margin, real = sim + 75.79 deg)."""

    @classmethod
    def setUpClass(cls):
        cls.layer = H.ActionLayer(KIN_POINTS)
        cls.lim = H.RealLimits.from_lerobot_calibration(LR_CAL, 3.0)
        cls.jm = H.JointMap(np.ones(5), [0.0, 0.0, 0.0, 0.0, ROLL_OFF], math.pi / 180, 0.0)
        cls.q = np.radians([0.0, -5.0, -15.0, 92.0, 100.69, 20.0])     # HOME, the roll at its real stop
        cls.lift = np.radians(0.45)            # a stuck shoulder lift as well (the correction is needed there)

    def guard(self, aw, **kw):
        return H.Guard(H.GuardConfig(i_time_s=0.2, i_max_deg=3.0, i_antiwindup=aw, **kw), self.layer, self.jm,
                       self.lim)

    def run_loop(self, g, target, n=300):
        """n live-loop steps at a fixed measured q (the blocked roll): -> sent commands [n, 6], prevs, stall step."""
        g.reset(self.q)
        qr = self.jm.sim_to_real(self.q)
        out, prevs, stall, prev = [], [], None, None
        for k in range(n):
            u, ur, ev, stop = g.apply(target, self.q, qr, np.zeros(6))
            self.assertEqual(stop, [])
            st = g.stall(self.q, prev)
            if st and stall is None:
                stall = k
            prev = g.prev.copy()
            out.append(u)
            prevs.append(prev)
        return np.asarray(out), np.asarray(prevs), stall

    def test_range_in_sim_units(self):
        g = self.guard(True)
        lo, hi = g.real_range_sim()
        self.assertAlmostEqual(math.degrees(hi[4]), 177.0 - 75.789, places=9)
        np.testing.assert_allclose(np.degrees(hi[:4]), self.lim.hi[:4], atol=1e-9)
        jm2 = H.JointMap([1.0, -1.0, 1.0, 1.0, 1.0], np.zeros(5), math.pi / 180, 0.0)     # a mirrored joint
        lo2, hi2 = H.Guard(H.GuardConfig(), self.layer, jm2, self.lim).real_range_sim()
        self.assertAlmostEqual(math.degrees(lo2[1]), -self.lim.hi[1], places=9)
        self.assertAlmostEqual(math.degrees(hi2[1]), -self.lim.lo[1], places=9)
        self.assertTrue(H.GuardConfig().i_antiwindup)                          # the robot default

    def test_stall_without_and_none_with(self):
        tgt = self.q.copy()
        tgt[4] = math.radians(101.08)                                           # below the range end, above the q
        tgt[1] -= self.lift
        u0, p0, s0 = self.run_loop(self.guard(False), tgt)
        u1, p1, s1 = self.run_loop(self.guard(True), tgt)
        hi = math.radians(177.0 - 75.789)
        self.assertIsNotNone(s0)                                                # the old rule: the robot's stall
        self.assertAlmostEqual(math.degrees(u0[-1, 4]), math.degrees(hi), places=6)
        self.assertAlmostEqual(math.degrees(u0[-1, 4] - p0[-1, 4]), 3.0, places=6)    # wound up to i_max
        self.assertIsNone(s1)
        np.testing.assert_allclose(u1[:, 4], u0[:, 4], atol=1e-9)              # the same (clipped) roll command
        self.assertLessEqual(float(u1[:, 4].max()), hi + 1e-12)
        np.testing.assert_allclose(p1[-1, 4], tgt[4], atol=1e-12)               # prev = the target
        np.testing.assert_array_equal(u1[:, [0, 1, 2, 3, 5]], u0[:, [0, 1, 2, 3, 5]])   # other joints: unchanged
        self.assertAlmostEqual(math.degrees(u1[-1, 1] - tgt[1]), -3.0, places=6)      # the lift still corrected

    def test_target_beyond_the_range_bleeds_off(self):
        g = self.guard(False)
        tgt = self.q.copy()
        tgt[4] = math.radians(101.08)
        self.run_loop(g, tgt, 200)
        self.assertAlmostEqual(math.degrees(g.integ[4]), 3.0, places=6)         # wound up (old rule)
        g.cfg.i_antiwindup = True                                               # the new rule from the next step on
        far = self.q.copy()
        far[4] = math.radians(104.0)                                            # the student pushes past the stop
        qr = self.jm.sim_to_real(self.q)
        u, _, _, _ = g.apply(far, self.q, qr, np.zeros(6))
        self.assertEqual(float(g.integ[4]), 0.0)                                # bled off at once
        self.assertTrue(g.i_aw[4] and not g.i_aw[:4].any())
        self.assertAlmostEqual(math.degrees(u[4]), 177.0 - 75.789, places=6)
        self.assertAlmostEqual(math.degrees(g.prev[4]), 177.0 - 75.789, places=6)
        self.assertEqual(g.stall(self.q, g.prev), [])
        lo = math.radians(self.lim.lo[0])                                       # the low side: the pan at -82.2 deg
        g2, g3 = self.guard(True), self.guard(False)
        q2 = self.q.copy()
        q2[0] = lo + math.radians(0.5)
        t2 = q2.copy()
        t2[0] -= math.radians(0.4)                                              # inside the range, below the q
        g2.reset(q2)
        g3.reset(q2)
        for _ in range(100):
            u, _, _, _ = g2.apply(t2, q2, self.jm.sim_to_real(q2), np.zeros(6))
            u3, _, _, _ = g3.apply(t2, q2, self.jm.sim_to_real(q2), np.zeros(6))
        self.assertAlmostEqual(float(u[0]), lo, places=9)                       # the same clipped command
        self.assertAlmostEqual(float(u3[0]), lo, places=9)
        self.assertAlmostEqual(math.degrees(g3.integ[0]), -3.0, places=6)       # old: wound up
        self.assertAlmostEqual(float(g2.integ[0]), float(lo - t2[0]), places=12)    # new: just to the range end
        np.testing.assert_allclose(g2.prev[0], t2[0], atol=1e-12)

    def test_inside_the_range_bit_for_bit(self):
        """No joint near its range: the two rules give the same commands and correction."""
        q = np.radians([0.0, -5.0, -15.0, 80.0, 0.0, 30.0])
        rng = np.random.default_rng(0)
        g0, g1 = self.guard(False), self.guard(True)
        g0.reset(q)
        g1.reset(q)
        for k in range(300):
            qk = q + np.radians(rng.uniform(-0.6, 0.6, 6))
            t = q.copy()
            t[:5] += np.radians(0.4 + rng.uniform(-0.3, 0.3, 5))              # a steady error: the correction grows
            qd = rng.uniform(-0.03, 0.03, 6)
            a = g0.apply(t, qk, self.jm.sim_to_real(qk), qd)
            b = g1.apply(t, qk, self.jm.sim_to_real(qk), qd)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(g0.integ, g1.integ)
            self.assertFalse(g1.i_aw.any())
        self.assertGreater(float(np.abs(g1.integ).max()), math.radians(0.5))    # the correction was active


# ---- floor mode ----------------------------------------------------------------------------------------------------
class TestFloorMode(unittest.TestCase):
    def test_student_mode_and_eval_guards(self):
        self.assertEqual(H.student_floor_mode({})[0], "flat")
        self.assertEqual(H.student_floor_mode({"v8_online": {"img_lag": {"lo_ms": 10.0}}})[0], "flat")
        self.assertEqual(H.student_floor_mode({"v8_online": {"table_bias": {"note": "off"}}})[0], "flat")
        m, why = H.student_floor_mode(V8_REC)
        self.assertEqual(m, "plane")
        self.assertIn("v8_online.table_bias", why)
        # flat = the eval values of the robot days (unchanged), plane = the guard defaults above the mat plane
        self.assertEqual(H.eval_guard("flat"), {"cmd_floor_m": -0.015, "hard_floor_m": -0.020, "i_time_s": 0.2,
                                                "i_max_deg": 3.0, "i_antiwindup": True})
        gp = H.eval_guard("plane")
        d = H.GuardConfig()
        self.assertEqual((gp["cmd_floor_m"], gp["hard_floor_m"]), (d.cmd_floor_m, d.hard_floor_m))
        self.assertEqual({k: gp[k] for k in H.EVAL_I_GUARD}, H.EVAL_I_GUARD)
        with self.assertRaises(ValueError):
            H.eval_guard("auto")

    def test_problems(self):
        G = H.GuardConfig.from_dict
        self.assertEqual(H.floor_problems("flat", "flat", G(H.eval_guard("flat"))), [])
        self.assertEqual(H.floor_problems("plane", "plane", G(H.eval_guard("plane"))), [])
        self.assertEqual(H.floor_problems("plane", "plane", H.GuardConfig()), [])
        dbl = H.floor_problems("plane", "plane", G(H.eval_guard("flat")))     # the flat eval stops on the plane
        self.assertEqual(len(dbl), 2)
        self.assertTrue(all("double lowering" in s for s in dbl))
        self.assertTrue(H.floor_problems("flat", "plane", H.GuardConfig()))   # a v8 student on the flat floor
        self.assertTrue(H.floor_problems("plane", "flat", H.GuardConfig()))   # C1 / s28 on the plane floor

    def test_banner_numbers(self):
        fl = H.floor_lines("flat", "x", 0.0, None, H.GuardConfig.from_dict(H.eval_guard("flat")))
        self.assertIn("floor +9.8 .. +19.0 mm, command stop -6.7 .. +2.5 mm, measured stop -11.7 .. -2.5 mm", fl[1])
        pl = H.floor_lines("plane", "x", 0.0, PLANE, H.GuardConfig.from_dict(H.eval_guard("plane")))
        self.assertIn("floor +1.5 mm, command stop -2.0 mm, measured stop -5.0 mm", pl[1])
        self.assertTrue(fl[0].startswith("FLOOR MODE flat") and pl[0].startswith("FLOOR MODE plane"))

    def test_print_setup(self):
        """The run's banner: floor lines, the integral with its anti-windup, HOME, then the views and the qd gain."""
        import contextlib
        import io
        R = _load_runner()
        g = H.GuardConfig.from_dict(H.eval_guard("plane"))
        info = dict(student="s.pt", student_cfg=dict(cameras=("external",), res=8), cam_map={"external": "external"},
                    device="cpu", joint_map=H.JointMap.placeholder_identity().describe(),
                    floor=dict(lines=H.floor_lines("plane", "x", 0.0, PLANE, g)), guard=g.__dict__,
                    home_sim_deg=list(H.ROBOT_HOME_SIM_DEG), qd_gain=2.0,
                    views={"external": dict(crop_xywh=(0, 0, 8, 8), out_wh=(8, 8), interp="area", source="t")})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            R.print_setup(info)
        out = buf.getvalue()
        for want in ("FLOOR MODE plane", "command stop -2.0 mm", "anti-windup at the real joint range ON",
                     "HOME sim deg [0.0, -5.0, -15.0, 92.0, 0.0, 30.0]", "view external: crop", "qdot input gain 2.0"):
            self.assertIn(want, out)

    def test_floor_setup(self):
        R = _load_runner()
        calib = {"scene": {"table_plane": {"coef": list(PLANE)}}}

        def a(mode="live", **kw):
            d = dict(mode=mode, floor_mode="auto", table_plane="", table_z=0.0, allow_floor_mismatch=False, calib="c")
            d.update(kw)
            return argparse.Namespace(**d)
        G = H.GuardConfig.from_dict
        f = R.floor_setup(a(), calib, V8_REC, G(H.eval_guard("plane")))
        self.assertEqual((f["mode"], f["plane"], f["table_plane_spec"], f["problems"]), ("plane", PLANE, "calib", []))
        f = R.floor_setup(a(), calib, {}, G(H.eval_guard("flat")))
        self.assertEqual((f["mode"], f["plane"], f["problems"]), ("flat", None, []))
        for kw, meta, g in ((dict(floor_mode="flat"), V8_REC, {}),                       # v8 student, flat floor
                            (dict(table_plane="calib"), {}, {}),                       # C1 on the plane
                            ({}, V8_REC, H.eval_guard("flat"))):                       # double lowering
            with self.assertRaises(SystemExit, msg=str(kw)):
                R.floor_setup(a(**kw), calib, meta, G(g))
            f = R.floor_setup(a("dry", **kw), calib, meta, G(g))                      # dry / replay: a warning
            self.assertTrue(f["problems"])
            f = R.floor_setup(a(allow_floor_mismatch=True, **kw), calib, meta, G(g))
            self.assertTrue(f["problems"])
        with self.assertRaises(SystemExit):
            R.floor_setup(a(floor_mode="plane", table_z=-0.01), calib, V8_REC, H.GuardConfig())
        with self.assertRaises(SystemExit):
            R.floor_setup(a(floor_mode="flat", table_plane="calib"), calib, {}, H.GuardConfig())
        with self.assertRaises(SystemExit):                                               # no plane in the calib
            R.floor_setup(a(), {}, V8_REC, H.GuardConfig())


# ---- HOME, no stitching, deploy.json, bundle -------------------------------------------------------------------------
class TestHomeAndNoStitching(unittest.TestCase):
    def test_home_roll_0(self):
        self.assertEqual(H.ROBOT_HOME_SIM_DEG, (0.0, -5.0, -15.0, 92.0, 0.0, 30.0))
        src = (HW / "run_student_real.py").read_text()
        self.assertIn('ap.add_argument("--home", default=",".join(f"{v:g}" for v in H.ROBOT_HOME_SIM_DEG)', src)
        self.assertEqual(",".join(f"{v:g}" for v in H.ROBOT_HOME_SIM_DEG), "0,-5,-15,92,0,30")

    def test_no_two_student_mode(self):
        self.assertFalse(hasattr(H, "SwitchPolicy"))
        for f in ("hw_lib.py", "run_student_real.py", "robot_test.sh", "make_orin_bundle.sh", "README.md"):
            t = (HW / f).read_text()
            for w in ("student2", "SwitchPolicy", "switch-hold", "switch-plate", "STUDENT2"):
                self.assertNotIn(w, t, (f, w))
        R = _load_runner()
        with self.assertRaises(SystemExit):
            R.main(["replay", "--student", "x.pt", "--student2", "y.pt"])


@needs_points
class TestDeployJsonAndBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        cls.plain = tiny_student(t / "plain.pt")
        cls.v8 = tiny_v8(t / "v8.pt", cls.plain)
        cls.calib = calib_file(t / "calib_v1.json")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_deploy_json(self):
        R = _load_runner()
        t = Path(self.tmp.name)
        for ck, mode in ((self.plain, "flat"), (self.v8, "plane")):
            out = t / f"deploy_{mode}.json"
            R.main(["deploy-json", "--student", str(ck), "--calib", str(self.calib), "--out", str(out)])
            d = json.loads(out.read_text())
            self.assertEqual(d["floor_mode"], mode)
            self.assertEqual(d["home_sim_deg"], "0,-5,-15,92,0,30")
            self.assertEqual(d["eval_guard"], H.eval_guard(mode))
            self.assertEqual(d["table_plane"], "calib" if mode == "plane" else "")
            self.assertEqual(d["table_plane_coef"], list(PLANE) if mode == "plane" else None)
        out = t / "deploy_home.json"
        R.main(["deploy-json", "--student", str(self.plain), "--calib", str(self.calib), "--out", str(out), "--home",
                "0,-5,-15,92,45,30"])
        self.assertEqual(json.loads(out.read_text())["home_sim_deg"], "0,-5,-15,92,45,30")
        with self.assertRaises(SystemExit):                                     # a v8 student without the plane
            R.main(["deploy-json", "--student", str(self.v8), "--calib", str(calib_file(t / "noplane.json", None)),
                    "--out", str(t / "x.json")])

    def test_bundle_manifest_and_deploy(self):
        t = Path(self.tmp.name)
        env = dict(os.environ, PYTHON=sys.executable, SO101_KIN_POINTS=str(KIN_POINTS), CUDA_VISIBLE_DEVICES="")
        for ck, mode in ((self.plain, "flat"), (self.v8, "plane")):
            out = t / f"bundle_{mode}"
            r = subprocess.run(["bash", str(HW / "make_orin_bundle.sh"), str(ck), str(out), str(self.calib)], env=env,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            man = (out / "MANIFEST.txt").read_text().splitlines()
            self.assertTrue(man[0].startswith("built ") and man[1].startswith("student ")
                            and man[2].startswith("calib "), man[:3])
            self.assertIn(f"floor mode {mode}", man[1])
            listed = [ln.split("  ", 1)[1] for ln in man[3:]]
            self.assertIn("./assets/deploy.json", listed)
            self.assertIn("./robot_test.sh", listed)
            self.assertFalse(any("__pycache__" in f for f in listed))
            c = subprocess.run("sed -n '4,$p' MANIFEST.txt | sha256sum -c --quiet", shell=True, cwd=out,
                               capture_output=True, text=True)
            self.assertEqual(c.returncode, 0, c.stdout + c.stderr)
            d = json.loads((out / "assets" / "deploy.json").read_text())
            self.assertEqual(d["floor_mode"], mode)
            self.assertFalse((out / "assets" / "student2.pt").exists())


if __name__ == "__main__":
    unittest.main()
