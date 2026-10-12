"""NRD data round of the SO101 policy study: the noisy-command campaign of the grasp collector and the fine-tune
(init_from) option of contact NRD training.

    PYTHONPATH=src python -m unittest tests.so101_policy.test_nrd_round -v

CPU only, seconds. Tests that need MuJoCo (or torch) skip without it; the bit-identity test of the production campaign also needs
scipy and the production validation data (~/so101_nrd_fix/data/val_only, or NEDM_GRASP_VAL). The fine-tune smoke run
(train.py --smoke twice on synthetic SO101 data, 4-5 min on newton's CPU) runs only with NEDM_SLOW_TESTS=1.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
VAL = Path(os.environ.get("NEDM_GRASP_VAL", Path.home() / "so101_nrd_fix/data/val_only"))
NOISY = ("noisy_nominal", "noisy_off_centre", "noisy_carry", "noisy_offset")
# master seeds that the noisy campaign must not reuse: the config's campaigns and convergence check, the sealed test,
# the critic's command-noise generator (scripts/critic/perturb.py) and its planning probe (plan_time.py)
USED_SEEDS = {20261008, 20261021, 20261034, 20261085, 20261123, 20261011, 99990001}


def config():
    return json.loads((REPO / "configs/contact_nrd/collection/so101_grasp.json").read_text())


@unittest.skipUnless(HAS_MUJOCO, "needs mujoco")
class Campaign(unittest.TestCase):
    def test_existing_campaigns_unchanged(self):
        """The noisy families come after every existing family and run only in noisy_v1: each existing campaign gives
        the same (family index, count) list as the config without the round."""
        from nedm.contact_nrd.collect import so101_grasp as g
        cfg = config()
        old = json.loads(json.dumps(cfg))
        old["families"] = [f for f in old["families"] if f["name"] not in NOISY]
        del old["campaigns"]["noisy_v1"]
        names = [f["name"] for f in cfg["families"]]
        self.assertEqual(names[: len(old["families"])], [f["name"] for f in old["families"]])
        for c in old["campaigns"]:
            self.assertEqual(g.campaign_counts(cfg, c), g.campaign_counts(old, c), c)
        self.assertEqual(g.campaign_counts(cfg, "production", families=["nominal"]), [(0, 11250)])

    def test_noisy_campaign(self):
        from nedm.contact_nrd.collect import so101_grasp as g
        cfg = config()
        counts = g.campaign_counts(cfg, "noisy_v1")
        names = [f["name"] for f in cfg["families"]]
        self.assertEqual([names[f] for f, _ in counts], list(NOISY))
        self.assertEqual(sum(n for _, n in counts), 10000)
        master = cfg["campaigns"]["noisy_v1"]["master_seed"]
        self.assertNotIn(master, USED_SEEDS)
        self.assertEqual(g.check_seeds(cfg, "noisy_v1", counts), 10000)   # no episode seed shared with any campaign
        self.assertTrue(all(name in g.FAMILIES for name in NOISY))


@unittest.skipUnless(HAS_MUJOCO, "needs mujoco")
class NoisyCommands(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nedm.contact_nrd.collect import so101_grasp as g
        cls.g, cls.s = g, g.Scene(g.load_config())

    def test_ou_noise(self):
        rng = np.random.default_rng(0)
        x = self.g.ou_noise(rng, 200000, 2, 0.01, 0.2, 0.02)
        self.assertTrue((x[0] == 0).all())
        self.assertAlmostEqual(float(x[1000:].std()), 0.01, delta=0.0006)
        lag1 = float(np.mean(x[1001:, 0] * x[1000:-1, 0]) / x[1000:, 0].var())
        self.assertAlmostEqual(lag1, math.exp(-0.02 / 0.2), delta=0.01)

    def plan(self, seed, family="noisy_nominal", fn=None):
        """(episode, control steps) of a family (or of fn(s, rng, n)) with the first seed seed + 1000 k that plans."""
        g, s = self.g, self.s
        n = (601 - 1) // s.per_ctrl + 1
        for _ in range(50):
            rng = np.random.default_rng(seed)
            try:
                return (fn or g.FAMILIES[family])(s, rng, n), n
            except g.PlanError:
                seed += 1000
        raise AssertionError("no plan")

    def test_rate_range_phases_and_seed(self):
        g, s = self.g, self.s
        ep, n = self.plan(3)
        rows, phases = map(np.array, zip(*ep["script"]))
        again, _ = self.plan(3)
        rows2, _ = map(np.array, zip(*again["script"]))
        np.testing.assert_array_equal(rows, rows2)
        self.assertEqual(rows.shape, (n, 6))
        step = np.abs(np.diff(np.vstack((np.clip(ep["q0"], s.tlo, s.thi)[None], rows)), axis=0))
        self.assertLessEqual(step[:, :5].max(), 0.06 + 1e-12)
        self.assertLessEqual(step[:, 5].max(), 0.09 + 1e-12)
        self.assertTrue(((rows >= s.tlo - 1e-12) & (rows <= s.thi + 1e-12)).all())
        # the same draws without the noise give the plan: the phases are the plan's and the commands differ from it
        clean, _ = self.plan(3, fn=g.fam_nominal)
        plan_rows, plan_phases = map(np.array, zip(*clean["script"]))
        np.testing.assert_array_equal(phases, plan_phases)
        self.assertGreater(np.abs(rows[:, :5] - plan_rows[:, :5]).max(), 1e-3)
        m = ep["meta"]
        self.assertTrue(0.005 <= m["ou_std_rad"] <= 0.02 and 0.1 <= m["ou_tau_s"] <= 0.4 and abs(m["jaw_offset_rad"]) <= 0.03)
        self.assertLessEqual(len(m["kind"]), 16)   # stored as the U16 variant

    def test_table_guard(self):
        """Where the plan's grasp point is below 5 cm, the commanded gripper's lowest point stays at least
        min(the plan's lowest point, 3 mm) high, except at a few steps where the rate limit lags (0.1 mm slack)."""
        g, s = self.g, self.s
        ep, _ = self.plan(11, "noisy_carry")
        clean, _ = self.plan(11, fn=lambda s_, rng, n: g.fam_policy_grasp(s_, rng, n, move_prob=1.0))
        rows, _ = map(np.array, zip(*ep["script"]))
        plan_rows, _ = map(np.array, zip(*clean["script"]))
        near = np.array([s.grasp_pose(q)[0][2] < 0.05 for q in plan_rows])
        self.assertGreater(near.sum(), 10)
        floor = np.array([min(s.lowest(s.fk(q)), 0.003) for q in plan_rows[near]])
        low = np.array([s.lowest(s.fk(q)) for q in rows[near]])
        self.assertLessEqual(int((low < floor - 1e-4).sum()), max(2, int(0.02 * near.sum())))
        self.assertGreaterEqual(ep["meta"]["guarded_steps"], 0)

    def test_offset_family(self):
        ep, _ = self.plan(5, "noisy_offset")
        off = np.asarray(ep["meta"]["offset_m"])
        self.assertTrue(0.001 - 1e-12 <= math.hypot(off[0], off[1]) <= 0.008 + 1e-12)
        self.assertTrue(0.0 <= off[2] <= 0.002)


@unittest.skipUnless(HAS_MUJOCO and HAS_SCIPY and (VAL / "unified_data.npz").exists(),
                     "needs mujoco, scipy and the production validation data")
class ProductionBitIdentical(unittest.TestCase):
    def test_regenerate_validation_episodes(self):
        """make_episode with the production master seed reproduces stored validation episodes exactly (nominal,
        off_centre, weak_closure: the closed-loop script)."""
        from nedm.contact_nrd.collect import so101_grasp as g
        cfg = g.load_config()
        s = g.Scene(cfg)
        with np.load(VAL / "unified_data.npz") as f:
            arr = {k: f[k] for k in ("family", "family_index", "lengths", "arm", "cube", "action", "contacts")}
        fam, idx = arr["family"], arr["family_index"]
        for f in (0, 1, 4):
            row = int(np.flatnonzero(fam == f)[0])
            ep = g.make_episode(s, f, int(idx[row]), cfg["campaigns"]["production"]["master_seed"])
            n = int(arr["lengths"][row])
            self.assertEqual(len(ep["action"]), n)
            np.testing.assert_array_equal(ep["states"]["arm"].astype(np.float32), arr["arm"][row, :n])
            np.testing.assert_array_equal(ep["action"].astype(np.float32), arr["action"][row, :n])
            np.testing.assert_array_equal(ep["contacts"], arr["contacts"][row, : n - 1])
            np.testing.assert_allclose(ep["states"]["cube"].astype(np.float32), arr["cube"][row, :n], rtol=0, atol=2e-7)


@unittest.skipUnless(HAS_TORCH, "needs torch")
class InitFrom(unittest.TestCase):
    def test_init_spec(self):
        from nedm.contact_nrd.train import INIT_DEFAULTS, init_spec
        self.assertIsNone(init_spec({}))
        spec = init_spec({"init_from": "~/run"})
        self.assertEqual(spec["run"], str(Path("~/run").expanduser().resolve()))
        self.assertEqual({k: spec[k] for k in INIT_DEFAULTS}, INIT_DEFAULTS)
        self.assertEqual(init_spec({"init_from": {"run": "/r", "collision_updates": 7}})["collision_updates"], 7)
        with self.assertRaises(ValueError):
            init_spec({"init_from": {"run": "/r", "refit": True}})
        with self.assertRaises(ValueError):
            init_spec({"init_from": {"checkpoint": "best.pt"}})

    def test_same_structure(self):
        from nedm.contact_nrd.train import same_structure
        sys.path.insert(0, str(REPO))
        from tests.contact_nrd import synthetic
        a = synthetic.so101_system(301)
        b = json.loads(json.dumps(a))
        b["families"].append(dict(name="noisy", group="noisy", select=True, note=""))   # families may differ
        self.assertEqual(same_structure(a, b), [])
        b["bodies"][1]["channels"] = b["bodies"][1]["channels"][::-1]
        b["pairs"] = [[0, 1], [1, 2]]
        self.assertEqual(same_structure(a, b), ["pairs", "bodies[1].channels"])

    @unittest.skipUnless(os.environ.get("NEDM_SLOW_TESTS") == "1", "slow: NEDM_SLOW_TESTS=1")
    def test_finetune_smoke(self):
        """train.py --smoke from scratch, then --smoke with init_from that run: the fine-tune starts from its weights,
        maps and loss scales (no refit), checks the starting model first and completes."""
        import torch
        sys.path.insert(0, str(REPO))
        from tests.contact_nrd import synthetic
        env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
        env["PYTHONPATH"] = str(REPO / "src")
        with tempfile.TemporaryDirectory() as tmp:
            data, base, tuned = Path(tmp) / "data", Path(tmp) / "base", Path(tmp) / "tuned"
            synthetic.write(data, "so101_grasp")
            cfg = json.loads((REPO / "configs/contact_nrd/train.json").read_text())
            cfg["init_from"] = {"run": str(base), "collision_updates": 2}
            (Path(tmp) / "ft.json").write_text(json.dumps(cfg))
            for out, extra in ((base, []), (tuned, ["--config", str(Path(tmp) / "ft.json")])):
                subprocess.run([sys.executable, "-m", "nedm.contact_nrd.train", "--data", str(data), "--output", str(out),
                                "--smoke", "--device", "cpu", *extra], cwd=tmp, env=env, check=True, capture_output=True)
            done = json.loads((tuned / "complete.json").read_text())
            self.assertTrue(done["complete"])
            self.assertEqual(done["init_from"], str(base.resolve()))
            log = [json.loads(x) for x in (tuned / "train_log.jsonl").read_text().splitlines()]
            stages = [r["stage"] for r in log]
            self.assertIn("init_from", stages)
            self.assertIn("init_contact_scale", stages)
            self.assertNotIn("core_linear_fit", stages)
            self.assertNotIn("contact_linear_fit", stages)
            self.assertEqual(next(r for r in log if r["stage"] == "init")["update"], 0)
            self.assertEqual(next(r for r in log if r["stage"] == "collision_fit")["updates"], 2)
            self.assertEqual((tuned / "contact_scale.json").read_text(), (base / "contact_scale.json").read_text())
            src = torch.load(base / "best.pt", weights_only=False)
            start = torch.load(tuned / "core.pt", weights_only=False)   # saved before any fine-tune update
            for k in ("core.in_mean", "core.lin", "contact.out_std", "collision.in_std"):
                self.assertTrue(torch.equal(src["model_state_dict"][k], start["model_state_dict"][k]), k)
            self.assertEqual(src["masks"], start["masks"])
            cfg_run = json.loads((tuned / "run_config.json").read_text())
            self.assertEqual(cfg_run["init_from"]["source_stage"], src["stage"])


@unittest.skipUnless(HAS_TORCH, "needs torch")
class EvalCompare(unittest.TestCase):
    """scripts/so101_policy/eval_nrd_noisy.py --compare: before / after rows, paired lift flags, forgetting gate."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(REPO / "src"))
        spec = importlib.util.spec_from_file_location("eval_nrd_noisy", REPO / "scripts/so101_policy/eval_nrd_noisy.py")
        cls.ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ev)

    @staticmethod
    def fake(wrong_rows, p95_1s, p95_4s):
        """An output JSON of 6 episodes (rows 10..15, all recorded lifts) where wrong_rows have no predicted lift."""
        rows = list(range(10, 16))
        per = [dict(row=r, family="noisy_nominal", lifted_recorded=True, lifted_predicted=r not in wrong_rows,
                    free_4s_cube_mm=float(r)) for r in rows]
        summ = dict(episodes=6, win1s_cube_mm=[1.0, p95_1s], free_4s_cube_mm=[2.0, p95_4s])
        wrong = len(wrong_rows)
        ens = {g: {"episodes": 6, "recorded_lifts": 6,
                   "s61": dict(wrong=wrong, false_lift=0, missed_lift=wrong, wrong_pct=round(100 * wrong / 6, 2))}
               for g in ("all", "noisy_nominal", "std_05_10mrad")}
        return dict(data_sha256="x", split=1, families=["noisy_nominal"], groups={"all": 6, "noisy_nominal": 6, "std_05_10mrad": 6},
                    group_rows={"noisy_nominal": rows, "std_05_10mrad": rows}, ensemble=ens,
                    models={"s61": dict(summary={"all": summ, "noisy_nominal": summ}, per_episode=per)})

    def test_compare(self):
        self.assertEqual(self.ev.sign_test(0, 0), 1.0)
        self.assertAlmostEqual(self.ev.sign_test(6, 0), 2 / 64)
        with tempfile.TemporaryDirectory() as tmp:
            b, a = Path(tmp) / "b.json", Path(tmp) / "a.json"
            b.write_text(json.dumps(self.fake({10, 11, 12}, 4.0, 10.0)))
            a.write_text(json.dumps(self.fake({12, 13}, 4.3, 12.0)))
            rows, pairs = self.ev.compare([f"before={b}", f"after={a}"], Path(tmp) / "c.json")
            self.assertTrue((Path(tmp) / "c.json").exists())
        self.assertEqual(len(rows), 6)   # 3 groups x 2 files, one model
        binr = next(r for r in rows if r["group"] == "std_05_10mrad" and r["label"] == "after")
        self.assertIsNone(binr["win1s_cube_p95_mm"])
        self.assertAlmostEqual(binr["free_4s_cube_p95_mm"], float(np.percentile(range(10, 16), 95)), places=3)
        p = next(x for x in pairs if x["group"] == "all")
        self.assertEqual((p["lift_fixed"], p["lift_broken"]), (2, 1))   # 10, 11 fixed; 13 broken
        self.assertTrue(p["gate_checks"]["win1s_cube_p95_mm"])          # 4.3 <= 1.1 x 4.0
        self.assertFalse(p["gate_checks"]["free_4s_cube_p95_mm"])       # 12 > 1.1 x 10
        self.assertTrue(p["gate_checks"]["lift_wrong"])
        self.assertFalse(p["gate_pass"])


if __name__ == "__main__":
    unittest.main()
