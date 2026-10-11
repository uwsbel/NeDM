"""Core-only families ("use": "core_only") train the core and nothing else (stdlib unittest; GPU if present).

Synthetic G1-layout data (tests/contact_nrd/synthetic.py) with a fourth family "floor_free" of core-only branches as
the G1 collector writes them (probe_core): 6 parent records with walking contact labels, then a contact-free branch
(labels 0, branch_start = 6). train.py --smoke --trace-draws (train.json with a rollout loss at every update, so the
5 smoke updates draw rollout starts) records the episodes every component drew windows from:
- core-only episodes never appear in the channel-mask statistics, the collision inputs, batches and validation, the
  contact inputs, linear fit, batches and one-step check, the rollout starts, the checks or the final report;
- they do appear in the core draws (fit and loss) and in the core validation report, but not in the history noise;
- run_config.json counts them apart; the final report and test.py have no core-only family, test.py names how many
  episodes it left out, and the per-episode indices of both are rows of unified_data.npz (also with the core-only
  episodes stored first).
load_data keeps core-only episodes ("keep"), leaves them out ("drop") or keeps them apart, cut after their longest
episode ("separate"; step_average_rates treats that store too). The collection checker counts core-only windows apart
(the other families' counts and the frame check are those of the same data without the core-only episodes) and
checks branch_start and the labels after it. load_data refuses a core-only family with select: true and an unknown
"use".

    PYTHONPATH=src python -m unittest tests.contact_nrd.test_core_only -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.contact_nrd.collect import export
from nedm.contact_nrd.collect.validate import validate
from nedm.contact_nrd.data import core_only_episodes, load_data, step_average_rates, task_episodes
from tests.contact_nrd import synthetic

REPO = Path(__file__).resolve().parents[2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TASK_ONLY = ("channel_masks", "collision_inputs", "collision", "collision_validation", "contact_inputs", "contact_fit",
             "contact", "onestep", "rollout", "checks", "final_validation")
CORE_PARTS = ("core", "core_validation")
PARENT = 6   # parent records before the branch (as the G1 collector's probe_core.parent_records)


def run(*args):
    env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
    env["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run([sys.executable, "-m", *args], cwd=tempfile.gettempdir(), env=env, capture_output=True,
                          text=True, check=True).stdout


def write(out, with_core=True, records=301, base=(30, 18, 12), core=30, branch_records=115, seed=0, core_first=False):
    """Synthetic G1 data: walk / perturbed / airborne (synthetic.g1_episode) and, with with_core, `core` core-only
    branches of PARENT + 1 + up to branch_records records (the last branch is the longest). The families list always
    names floor_free (use core_only), so the data without it differ only by its episodes. core_first: the core-only
    episodes are stored before the others. -> (system, arrays)."""
    rng = np.random.default_rng(seed)
    system = synthetic.g1_system(records)
    system["families"].append(dict(name="floor_free", group="probe", select=False, use="core_only",
                                   note="synthetic core-only branches"))
    family = np.repeat(np.arange(3), base)
    eps = []
    for e, f in enumerate(family):
        length = records if f == 0 else int(rng.integers(records // 2, records))
        ep = synthetic.g1_episode(rng, records, int(f))
        eps.append(dict(states={"robot": ep["states"]["robot"][:length]}, action=ep["action"][:length],
                        contacts=ep["contacts"][:length - 1], scalars=dict(episode_id=e, group_id=e, branch_start=-1)))
    if with_core:
        crng = np.random.default_rng(seed + 1)
        for k in range(core):
            length = PARENT + 1 + branch_records - 2 * (core - 1 - k)
            ep = synthetic.g1_episode(crng, records, 0)               # walking motion with contact labels
            contacts = ep["contacts"][:length - 1].copy()
            contacts[PARENT:] = 0                                       # the branch: no floor
            contacts[PARENT - 1] = 1                                    # the parent touched just before the branch
            eps.append(dict(states={"robot": ep["states"]["robot"][:length]}, action=ep["action"][:length],
                            contacts=contacts, scalars=dict(episode_id=len(family) + k, group_id=len(family) + k,
                                                            branch_start=PARENT)))
        family = np.concatenate((family, np.full(core, 3)))
        if core_first:
            order = np.concatenate((np.arange(len(family) - core, len(family)), np.arange(len(family) - core)))
            eps, family = [eps[k] for k in order], family[order]
    arrays = export.pack_episodes(eps, system)
    arrays.update(family=family.astype(np.int16), terminal_reason=np.zeros(len(family), np.int8),
                  branch_start=arrays["branch_start"].astype(np.int16),
                  splits=export.assign_splits(family, arrays["group_id"], seed=seed))
    export.write_dataset(out, system, arrays, manifest=dict(synthetic=True))
    return system, arrays


class CoreOnly(unittest.TestCase):
    CORE_FIRST = False   # the G1 collector stores the core-only branches last

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = Path(cls.tmp.name) / "data"
        cls.system, cls.arrays = write(cls.data, core_first=cls.CORE_FIRST)
        cls.core = np.flatnonzero(cls.arrays["family"] == 3)
        cls.other = np.flatnonzero(cls.arrays["family"] != 3)
        cls.run_dir = Path(cls.tmp.name) / "run"
        cls.trace = Path(cls.tmp.name) / "draws.json"
        cfg = json.loads((REPO / "configs/contact_nrd/train.json").read_text())
        cfg["cotrain"]["traj_every"] = 1
        config = Path(cls.tmp.name) / "train.json"
        config.write_text(json.dumps(cfg))
        run("nedm.contact_nrd.train", "--data", str(cls.data), "--output", str(cls.run_dir), "--smoke", "--device",
            DEVICE, "--config", str(config), "--trace-draws", str(cls.trace))
        cls.drawn = {k: set(v) for k, v in json.loads(cls.trace.read_text()).items()}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_core_only_never_in_other_draws(self):
        for part in TASK_ONLY:
            with self.subTest(part=part):
                self.assertIn(part, self.drawn)                       # the component drew windows
                self.assertFalse(self.drawn[part] & set(self.core.tolist()),
                                 f"{part} drew windows of core-only episodes")

    def test_core_draws_core_only(self):
        core_val = set(self.core[self.arrays["splits"][self.core] == 1].tolist())
        self.assertTrue(core_val)
        self.assertTrue(self.drawn["core"] & set(self.core.tolist()))     # the core fit and loss use them
        self.assertTrue(self.drawn["core_validation"] & core_val)          # and their validation windows
        others = set(np.flatnonzero(self.arrays["family"] != 3).tolist())
        self.assertTrue(self.drawn["core"] & others)                       # with the other families' free windows

    def test_reports_leave_core_only_out(self):
        config = json.loads((self.run_dir / "run_config.json").read_text())
        self.assertEqual(config["core_only_families"], ["floor_free"])
        self.assertNotIn("floor_free", json.loads((self.run_dir / "validation.json").read_text())["families"])
        by_use = json.loads((self.run_dir / "core_validation.json").read_text())["by_use"]
        self.assertGreater(by_use["core_only"]["robot"]["windows"], 0)
        self.assertGreater(by_use["task"]["robot"]["windows"], 0)
        out = Path(self.tmp.name) / "test.json"
        printed = run("nedm.contact_nrd.test", "--data", str(self.data), "--model", f"smoke={self.run_dir}",
                      "--output", str(out), "--device", DEVICE)
        result = json.loads(out.read_text())
        self.assertEqual(result["core_only_left_out"], len(self.core))
        self.assertIn(f"{len(self.core)} episodes of core-only families (floor_free) are not scored", printed)
        self.assertEqual(result["models"]["smoke"]["episodes"], len(self.arrays["lengths"]) - len(self.core))
        self.assertNotIn("floor_free", result["models"]["smoke"]["families"])
        self.assertNotIn("family floor_free", printed)
        # per-episode indices are rows of unified_data.npz
        self.assertEqual([e["index"] for e in result["models"]["smoke"]["per_episode"]], self.other.tolist())
        validated = [e["index"] for e in json.loads((self.run_dir / "validation.json").read_text())["per_episode"]]
        self.assertTrue(validated)
        self.assertTrue(set(validated) <= set(self.other[self.arrays["splits"][self.other] == 1].tolist()))

    def test_run_config_counts(self):
        config = json.loads((self.run_dir / "run_config.json").read_text())
        splits = self.arrays["splits"]
        self.assertEqual(config["training_episodes"], int((splits[self.other] == 0).sum()))   # the other families
        self.assertEqual(config["core_only_episodes"], {"train": int((splits[self.core] == 0).sum()),
                                                        "validation": int((splits[self.core] == 1).sum())})

    def test_history_noise_without_core_only(self):
        """The history noise comes from the core validation windows of the other families only."""
        log = [json.loads(x) for x in (self.run_dir / "train_log.jsonl").read_text().splitlines()]
        rows = [r for r in log if r["stage"] == "history_noise_windows"]
        self.assertEqual(len(rows), 1)
        by_use = json.loads((self.run_dir / "core_validation.json").read_text())["by_use"]
        self.assertEqual(rows[0]["bodies"]["robot"], {"used": by_use["task"]["robot"]["windows"],
                                                      "core_only_left_out": by_use["core_only"]["robot"]["windows"]})
        # the rate channels of the noise (no history_noise rule in train.json) = the RMS error of those windows
        sigma = next(r["sigma"][0] for r in log if r["stage"] == "core_sigma")
        masks = json.loads((self.run_dir / "run_config.json").read_text())["masks"][0]
        types = self.system["bodies"][0]["channel_types"]
        rates = [c for c, t in enumerate(types) if t in ("velocity", "angular_velocity", "angular_rate")]
        self.assertEqual([sigma[c] for c in rates], [by_use["task"]["robot"]["rms"][c] * masks[c] for c in rates])
        self.assertNotEqual([sigma[c] for c in rates], [by_use["core_only"]["robot"]["rms"][c] * masks[c] for c in rates])

    def test_task_episodes(self):
        data = load_data(self.data, "cpu")
        flag = core_only_episodes(data).numpy()
        self.assertTrue(np.array_equal(np.flatnonzero(flag), self.core))
        ids = torch.arange(len(flag))
        self.assertTrue(np.array_equal(task_episodes(data, ids).numpy(), np.flatnonzero(~flag)))


class CoreOnlyStoredFirst(CoreOnly):
    """The same with the core-only episodes stored before the others (the loaded rows are not the file rows)."""
    CORE_FIRST = True


class Checker(unittest.TestCase):
    def test_counts_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(Path(tmp) / "with", with_core=True)
            write(Path(tmp) / "without", with_core=False)
            a, b = validate(Path(tmp) / "with"), validate(Path(tmp) / "without")
            self.assertTrue(a["ok"], a["fails"])
            for block in ("split0", "split1", "all", "family_walk", "family_airborne"):
                with self.subTest(block=block):
                    self.assertEqual({k: a["coverage"][block][k] for k in b["coverage"][block]}, b["coverage"][block])
            self.assertEqual(a["rate_consistency"], b["rate_consistency"])   # the same 400 other episodes
            self.assertIn("robot_angular", a["rate_consistency_core_only"])
            self.assertEqual(a["rate_consistency_episodes"], {"other": 60, "core_only": 30})
            train = a["coverage"]["split0"]
            self.assertGreater(train["core_only_core_robot"], 0)
            self.assertEqual(train["core_fit_robot"], train["core_robot"] + train["core_only_core_robot"])
            fam = a["coverage"]["family_floor_free"]
            self.assertEqual(fam["use"], "core_only")
            self.assertFalse([k for k in fam if k.startswith(("single_", "on_", "onsets_", "multi"))])

    def test_branch_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            _, arrays = write(root, core=6, base=(10, 6, 4))
            core = np.flatnonzero(arrays["family"] == 3)
            with np.load(root / "unified_data.npz") as f:
                packet = {k: f[k] for k in f.files}
            packet["contacts"][core[0], PARENT + 3, 0] = 1      # a label inside the branch
            np.savez(root / "unified_data.npz", **packet)
            self.assertIn("core-only episodes have contact labels from their branch_start on", validate(root)["fails"])
            del packet["branch_start"]
            np.savez(root / "unified_data.npz", **packet)
            self.assertIn("core-only episodes need a branch_start [N] array", validate(root)["fails"])


class Loader(unittest.TestCase):
    def test_modes(self):
        """keep: all episodes; drop: the others only; separate: the others and a store of the core-only episodes cut
        after the longest one. The same values in every mode, also after step_average_rates."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            _, arrays = write(root, core=5, base=(6, 4, 2), core_first=True)
            core, other = np.flatnonzero(arrays["family"] == 3), np.flatnonzero(arrays["family"] != 3)
            cut = int(arrays["lengths"][core].max())
            self.assertLess(cut, arrays["contacts"].shape[1])
            keep, drop, sep = (load_data(root, "cpu", core_only=m) for m in ("keep", "drop", "separate"))
            self.assertNotIn("file_index", keep)
            self.assertNotIn("core_store", drop)
            for d in (drop, sep):
                self.assertEqual(d["file_index"].tolist(), other.tolist())
                self.assertEqual(d["core_only_left_out"], len(core))
                for key in ("states", "actions", "contacts", "lengths", "splits", "family"):
                    self.assertTrue(torch.equal(d[key], keep[key][other]), key)
            store = sep["core_store"]
            self.assertEqual(store["file_index"].tolist(), core.tolist())
            for key, n in (("states", cut), ("actions", cut), ("contacts", cut - 1)):
                self.assertTrue(torch.equal(store[key], keep[key][core, :n]), key)
            for key in ("lengths", "splits", "family"):
                self.assertTrue(torch.equal(store[key], keep[key][core]), key)
            kinds = {"articulated"}
            full = step_average_rates(keep, kinds)["states"]
            step_average_rates(sep, kinds)
            self.assertTrue(torch.equal(sep["states"], full[other]))
            self.assertTrue(torch.equal(sep["core_store"]["states"], full[core, :cut]))
            with self.assertRaisesRegex(ValueError, "core_only must be"):
                load_data(root, "cpu", core_only="all")
            plain = Path(tmp) / "plain"   # no core-only episodes: every mode gives the arrays of keep, no extra keys
            write(plain, with_core=False, base=(6, 4, 2))
            ref = load_data(plain, "cpu")
            for mode in ("drop", "separate"):
                d = load_data(plain, "cpu", core_only=mode)
                self.assertEqual(set(d), set(ref))
                self.assertTrue(all(torch.equal(d[k], ref[k]) for k in ("states", "actions", "contacts", "lengths")))

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            system, _ = write(root, core=4, base=(6, 4, 2))
            for change, message in ((dict(select=True), "must have select: false"), (dict(use="contact_only"), "use must be")):
                bad = json.loads(json.dumps(system))
                bad["families"][3].update(change)
                (root / "system.json").write_text(json.dumps(bad))
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                    load_data(root, "cpu")


if __name__ == "__main__":
    unittest.main()
