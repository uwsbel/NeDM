"""train.py --smoke and test.py end to end on synthetic data of both new layouts (stdlib unittest; GPU if present).

G1 layout: one floating articulated body with 67 channels and 27 actions, two pairs to the same fixed ground, an
explicit channel_mask that keeps the base height (the automatic rule drops it: its spread is under 1 % of the
travel), families with one probe family that checkpoint selection skips, the evaluation block whose score uses the
windows only (first-state rollouts and long windows are reports).
SO101 layout: arm 12 (no channel_mask: automatic mask and a mask_warning log row) + cube 13 (explicit mask), two
pairs of the same two moving bodies and one to the fixed table, the free + windows evaluation block with the timing
rule of the controller phase. The data pass the collection checker first. The final report has the event windows and
the outcome flags; the checks during training skip the report-only modes. An explicit mask that gives a pose
channel and its rate channel different values is refused.

    PYTHONPATH=src python -m unittest tests.contact_nrd.test_smoke -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from nedm.contact_nrd.collect.validate import validate
from tests.contact_nrd import synthetic

REPO = Path(__file__).resolve().parents[2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run(*args):
    env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
    env["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run([sys.executable, "-m", *args], cwd=tempfile.gettempdir(), env=env, capture_output=True,
                          text=True, check=True).stdout


class Smoke(unittest.TestCase):
    def pipeline(self, name):
        with tempfile.TemporaryDirectory() as tmp:
            data, out = Path(tmp) / "data", Path(tmp) / "run"
            synthetic.write(data, name)
            report = validate(data)
            self.assertTrue(report["ok"], report["fails"])
            system = json.loads((data / "system.json").read_text())
            spec = json.loads((REPO / "configs/contact_nrd/evaluation" / f"{name}.json").read_text())
            self.assertEqual(system["evaluation"], spec)            # embedded by the collection writer
            run("nedm.contact_nrd.train", "--data", str(data), "--output", str(out), "--smoke", "--device", DEVICE)
            done = json.loads((out / "complete.json").read_text())
            self.assertTrue(done["complete"] and done["smoke"])
            self.assertEqual(done["finite"], 1.0)
            log = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
            config = json.loads((out / "run_config.json").read_text())
            validation = json.loads((out / "validation.json").read_text())
            names = [f["name"] for f in system["families"]]
            self.assertEqual(set(validation["families"]), set(names))
            self.assertEqual(config["selection_families"], names[:2])
            self.assertEqual(validation["selected"]["selection_score"], done["validation_score"])
            self.assertEqual({t["stat"] for t in validation["score_terms"]}, {t["stat"] for t in system["evaluation"]["score"]})
            for h in ("0.02", "0.2", "0.5", "1.0"):
                self.assertGreater(validation["windows"][h]["windows"], 0)
            self.assertGreater(validation["event_windows"]["0.2"]["windows"], 0)
            self.assertEqual(set(validation["windows"]["flags"]), set(system["evaluation"]["flags"]))
            printed = run("nedm.contact_nrd.test", "--data", str(data), "--model", f"smoke={out}", "--output", str(Path(tmp) / "test.json"),
                          "--device", DEVICE)
            result = json.loads((Path(tmp) / "test.json").read_text())["models"]["smoke"]
            self.assertEqual(result["episodes"], 60)
            self.assertEqual(set(result["families"]), set(names))
            for group in ["all"] + [f"family {n}" for n in names] + ["selected families"]:
                self.assertIn(f"smoke, {group} (", printed)
            self.assertIn("windows 1.0 s", printed)
            return log, config, validation

    def test_g1_layout(self):
        log, config, validation = self.pipeline("g1_locomotion")
        masks = next(r for r in log if r["stage"] == "masks")
        self.assertFalse([r for r in log if r["stage"] == "mask_warning"])
        self.assertEqual(masks["explicit"], ["robot"])
        self.assertEqual(masks["automatic"][0][2], 0.0)          # base height: the automatic rule drops it
        self.assertEqual(config["masks"][0], [1.0] * 67)          # the explicit mask keeps every channel
        self.assertIn("free", validation)                          # a report: the score uses the windows only
        self.assertIn("long_windows", validation)
        self.assertEqual({t["stat"].split("/")[0] for t in validation["score_terms"]}, {"windows", "nonfinite_fraction"})
        checks = [r for r in log if r["stage"] == "cotrain"]
        self.assertTrue(checks and all(set(r["terms"]) == {t["stat"] for t in validation["score_terms"]} for r in checks))
        self.assertEqual(set(validation["windows"]["contacts"]), {"left_foot_ground", "right_foot_ground", "all"})

    def test_unpaired_mask_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            synthetic.write(data, "g1_locomotion", episodes=12, records=121)
            path = data / "system.json"
            system = json.loads(path.read_text())
            robot = system["bodies"][0]
            robot["channel_mask"][robot["channel_types"].index("angular_velocity") + 2] = 0   # yaw rate off, quaternion on
            path.write_text(json.dumps(system))
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                run("nedm.contact_nrd.train", "--data", str(data), "--output", str(Path(tmp) / "run"), "--smoke", "--device", DEVICE)
            self.assertIn("channel_mask must give a pose channel and its rate channel", caught.exception.stderr)

    def test_so101_layout(self):
        log, config, validation = self.pipeline("so101_grasp")
        masks = next(r for r in log if r["stage"] == "masks")
        self.assertEqual(masks["explicit"], ["cube"])
        warning = [r for r in log if r["stage"] == "mask_warning"]
        self.assertEqual([r["bodies"] for r in warning], [["arm"]])   # data with families but no channel_mask for the arm
        self.assertEqual(config["masks"][1], [1.0] * 13)
        self.assertEqual(config["masks"][0][12], 0.0)             # padding of the 12-channel arm
        self.assertIn("free", validation)
        self.assertEqual(set(validation["free"]["contacts"]), {"fixed_jaw_cube", "moving_jaw_cube", "cube_table", "all"})


if __name__ == "__main__":
    unittest.main()
