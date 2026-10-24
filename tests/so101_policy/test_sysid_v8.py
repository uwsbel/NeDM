"""Tests of the v8 real-data system identification helpers (``scripts/so101_policy/real2sim/sysid``): the image-lag
recommendation rule (mixture of whole-step lag shares), the time blocks and student groups of the lag fit, the plane
over the task annulus, and (when the robot logs, calibration and kinematic points are present) that the free-cube lag
fit recovers a known pipeline lag on synthetic blobs on the logged robot time base. CPU, about 20 s:

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_sysid_v8 -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SYSID = ROOT / "scripts" / "so101_policy" / "real2sim" / "sysid"
sys.path.insert(0, str(SYSID))
import sysid_lib as L  # noqa: E402
import lag_fit as LF  # noqa: E402
import make_json as MJ  # noqa: E402

HAVE_DATA = L.CALIB.exists() and L.KIN_POINTS.exists() and L.LOGS.exists()


class TestRules(unittest.TestCase):
    def test_lag_mixture(self):
        fr = [{"0": 0.05, "1": 0.59, "2": 0.36}, {"1": 0.5, "2": 0.5},
              {"1": 0.24, "2": 0.59, "3": 0.17}, {"1": 0.09, "2": 0.6, "3": 0.31}]
        m = MJ.lag_mixture(fr)
        self.assertEqual(sorted(m), ["0", "1", "2", "3"])
        self.assertAlmostEqual(sum(m.values()), 1.0, places=9)
        self.assertTrue(all(v >= 0.05 - 1e-9 for v in m.values()))
        self.assertGreater(m["2"], m["1"])                   # the two estimates together lean to 2 steps
        # 4+ steps pool into 3; a single estimate at 2 steps keeps the 0.05 minimum elsewhere
        m2 = MJ.lag_mixture([{"2": 0.5, "5": 0.5}])
        self.assertAlmostEqual(m2["0"], 0.05)
        self.assertAlmostEqual(sum(m2.values()), 1.0, places=9)
        self.assertGreater(m2["3"], 0.4)

    def test_blocks_and_groups(self):
        self.assertEqual(LF.block_of("eval_20261022_151223"), "10-22 day")
        self.assertEqual(LF.block_of("live_20261023_214309"), "10-23 night")
        self.assertEqual(LF.block_of("eval_20261023_075959"), "10-23 night")
        self.assertEqual(LF.block_of("eval_20261023_080000"), "10-23 day")
        self.assertEqual(LF.student_group("s28"), "s28")
        self.assertEqual(LF.student_group("C1"), "C1")
        self.assertEqual(LF.student_group("s28+C1"), "other")

    def test_annulus_plane(self):
        lo, hi = MJ.annulus_plane_mm((-0.01, 0.0, 0.0))
        self.assertAlmostEqual(lo, -10.0, places=6)
        self.assertAlmostEqual(hi, -10.0, places=6)
        lo, hi = MJ.annulus_plane_mm((0.0, 0.0, 0.1))     # rises with y: +-75 deg edges at r 0.27
        self.assertAlmostEqual(hi, 0.27 * np.sin(np.radians(75.0)) * 100.0, places=3)
        self.assertAlmostEqual(lo, -hi, places=6)


@unittest.skipUnless(HAVE_DATA, "robot logs / calibration / kinematic points not present")
class TestLagIdentifiability(unittest.TestCase):
    def test_synthetic_recovery(self):
        rec = [r for r in LF.collect_robot()[:12] if len(r["W"]) >= 3][:6]
        self.assertGreater(len(rec), 2)
        cubes = LF.fitted_cubes(rec, "robot", 10.0)
        taus = np.arange(0.0, 60.1, 2.0)
        for tt in (15.0, 35.0):
            f = LF.fit_free(LF.synth(rec, cubes, tt, noise_px=0.05, seed=1), "robot", nb=20, taus=taus)[0]
            self.assertLess(abs(f["tau_ms"] - tt), 3.0, f"tau_true {tt}: fitted {f['tau_ms']:.1f} ms")


if __name__ == "__main__":
    unittest.main()
