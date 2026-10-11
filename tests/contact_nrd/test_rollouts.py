"""rollouts.py (input of the videos and contact timelines) against the evaluation code (CPU, stdlib unittest).

On synthetic G1 and SO101 data with a small random model: the memory-mapped npz reader gives the arrays of np.load;
load_rows gives the rows of load_data; the first-state and window rollouts give the errors, valid steps and contact
event counts of data._free_raw and data._window_raw on the same episodes; --pick takes validation episodes of every
family that are long enough, the same ones twice.

    PYTHONPATH=src python -m unittest tests.contact_nrd.test_rollouts -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.contact_nrd import data as D
from nedm.contact_nrd.rollouts import load_rows, npz_members, pick_rows, rollouts
from tests.contact_nrd import synthetic
from tests.contact_nrd.test_conventions import KINDS, random_model

ROWS = [7, 2, 10]


class Rollouts(unittest.TestCase):
    def check(self, name):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic.write(tmp, name, episodes=12, records=161, seed=3)
            members = npz_members(Path(tmp) / "unified_data.npz")
            with np.load(Path(tmp) / "unified_data.npz") as f:
                for key in f.files:
                    np.testing.assert_array_equal(members[key], f[key], err_msg=key)
            full = D.load_data(tmp, "cpu", model_step_s=0.02)
            subs = [load_rows(tmp, ROWS, "cpu", 0.02) for _ in range(2)]
            picks = [pick_rows(tmp, 2, None, 5, 40, 2) for _ in range(2)]
        for key in ("states", "actions", "contacts", "lengths", "splits", "family"):
            self.assertTrue(torch.equal(full[key][ROWS], subs[0][key]), key)
        self.assertEqual(picks[0], picks[1])
        fam, n_steps = full["family"].numpy(), (full["lengths"].numpy() - 1) // 2
        long_val = (full["splits"].numpy() == 1) & (n_steps >= 45)       # validation, start 5 + 40 steps fit
        self.assertTrue(all(long_val[r] for r in picks[0]))
        for f in set(fam.tolist()):
            self.assertEqual(sum(fam[r] == f for r in picks[0]), min(2, int((long_val & (fam == f)).sum())))
        model = random_model(full["system"])
        D.step_average_rates(full, set(KINDS))
        system, stride, tol = full["system"], full["stride"], 2
        moving = [b for b in system["bodies"] if b["moving"]]
        cols = [{m: c for m, c in ((m, [k for k, t in enumerate(b["channel_types"]) if t == typ])
                                   for m, typ in D.TASK_ERRORS.items()) if c} for b in moving]
        idx = torch.tensor(ROWS)
        labels = D.model_labels(full["contacts"][idx], stride).numpy()
        phys = D.physical_labels(full["contacts"][idx], stride).numpy()
        flags = D.compile_flags(system["evaluation"].get("flags"), system, [m["name"] for m in system["pair_metadata"]])
        for (mode, start, h), sub in zip((("first", 0, None), ("window", 5, 40)), subs):
            with torch.no_grad():
                out = rollouts(model, sub, mode, start, h, tol)
                if mode == "first":
                    raw = D._free_raw(model, full, idx, cols, labels, phys, flags, 64, tol)
                    ref, at = raw["err"]["final"], out["steps"]
                    np.testing.assert_array_equal(out["steps"], (full["lengths"][idx].numpy() - 1) // stride)
                else:
                    raw = D._window_raw(model, full, idx, cols, labels, phys, flags, [np.array([start])] * len(ROWS),
                                        [h * full["dt"]], 64, tol)
                    ref, at = raw["err"][str(h * full["dt"])], np.full(len(ROWS), h)
                    np.testing.assert_array_equal(out["steps"], raw["fit"])
            np.testing.assert_array_equal(out["event_counts"], raw["contacts"]["counts"])
            for d, b in enumerate(moving):
                for m, v in ref[d].items():
                    mine = out[f"error/{b['name']}/{m}"][np.arange(len(ROWS)), at]
                    ok = np.isfinite(v)   # a window error is NaN where the horizon passes the episode end
                    self.assertTrue(ok.any())
                    np.testing.assert_allclose(mine[ok], v[ok], rtol=1e-10, atol=1e-12, err_msg=f"{mode} {m}")

    def test_g1(self):
        self.check("g1_locomotion")

    def test_so101(self):
        self.check("so101_grasp")


if __name__ == "__main__":
    unittest.main()
