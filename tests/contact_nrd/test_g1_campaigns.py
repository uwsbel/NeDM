"""G1 collector campaigns: core-only branches in pilot and production, none in the test campaign (stdlib unittest;
collector environment nrdsim: mujoco, onnxruntime, numpy; skipped without mujoco).

The test campaign has core_scale 0 (test.py never scores core-only episodes): its counts and job plan are those of the
collector before the branches. With the sealed test data present (CONTACT_NRD_G1_TEST, default
artifacts/contact_nrd/test_sealed/g1_locomotion) every stored episode (episode_id, group_id, seed, family) is a planned
one and the job count is that of its manifest.

    PYTHONPATH=src ~/miniconda3/envs/nrdsim/bin/python -m unittest tests.contact_nrd.test_g1_campaigns -v
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import numpy as np

try:
    from nedm.contact_nrd.collect import g1_locomotion as g
except ImportError as exc:   # the NRD environment has no mujoco
    g, MISSING = None, str(exc)

REPO = Path(__file__).resolve().parents[2]
SEALED = Path(os.environ.get("CONTACT_NRD_G1_TEST", REPO / "artifacts/contact_nrd/test_sealed/g1_locomotion"))


@unittest.skipIf(g is None, "collector environment needed (mujoco)")
class Campaigns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = g.load_config()

    def test_core_counts(self):
        base = self.cfg["pilot_counts"][g.CORE]
        self.assertEqual(g.campaign_counts(self.cfg, "pilot")[g.CORE], base)
        self.assertEqual(g.campaign_counts(self.cfg, "production")[g.CORE], 10 * base)
        self.assertEqual(g.campaign_counts(self.cfg, "test")[g.CORE], 0)
        self.assertEqual(g.campaign_counts(self.cfg, "test", probe_scale=2)[g.CORE], 0)   # core_scale stays
        self.assertEqual(g.campaign_counts(self.cfg, "test", probe_scale=2)["probe_drop"],
                         2 * self.cfg["pilot_counts"]["probe_drop"])
        g.check_campaigns(self.cfg)   # no repeated or shared episode seeds

    def test_test_plan(self):
        camp = self.cfg["campaigns"]["test"]
        jobs = g.make_jobs(self.cfg, g.campaign_counts(self.cfg, "test"), int(camp["master_seed"]))
        self.assertFalse([j for j in jobs if j["family"] == g.CORE])
        if not (SEALED / "unified_data.npz").exists():
            self.skipTest(f"no sealed test data in {SEALED}")
        plan = set()
        for j in jobs:
            fam = g.FAMILY[j["family"]]
            if j["family"].startswith("probe"):
                plan.update((j["episode_id"] + b, j["group"], j["seed"] * 100 + b, fam)
                            for b in range(self.cfg[j["family"]]["branches"]))
            else:
                plan.add((j["episode_id"], j["group"], j["seed"], fam))
        with np.load(SEALED / "unified_data.npz") as f:   # lazy: reads only these arrays
            stored = set(zip(*(f[k].astype(np.int64).tolist() for k in ("episode_id", "group_id", "seed", "family"))))
        self.assertEqual(stored - plan, set())
        self.assertEqual(len(jobs), json.loads((SEALED / "manifest.json").read_text())["jobs"])


if __name__ == "__main__":
    unittest.main()
