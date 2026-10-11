"""Task metrics, families and the selection score of data.py (CPU, stdlib unittest).

A stub model replays the recorded trajectory and the recorded labels (a perfect model), optionally with a known
position offset, a non-finite episode or non-finite long rollouts, so every statistic has a known value: errors 0
(or the offset), no missed or false contact events, switch accuracy 1, the expected window counts (fixed starts, the
end window, the long windows, the event windows), outcome flags that agree (or, with an offset, a known
disagreement), per-family and selected-family summaries, report-only modes that neither enter the score nor run in
the checks, statistics without rollouts that are None (and a score that needs one: None for a family with select:
false, an error otherwise). Event matching is checked against brute force; the physical contact view and the tilt
condition on hand-made cases.

    PYTHONPATH=src python -m unittest tests.contact_nrd.test_task_metrics -v
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.contact_nrd.data import (FAIL, NONE, ZERO, compile_flags, contact_tally, evaluate, event_starts,
                                   exclude_timing, flag_trace, free_metrics, load_data, match_events, model_labels,
                                   physical_labels, physical_switches, step_average_rates, task_metrics)
from nedm.contact_nrd.test import describe
from tests.contact_nrd import synthetic

KINDS = {"sphere", "rigid", "articulated"}


class Replay:
    """The recorded trajectory from the start state's episode and record; recorded labels as switches. offset: added
    to every predicted position; broken: an episode whose rollouts are NaN from half their steps on; nan_after: every
    rollout longer than that many steps is NaN from that step on; late_pair: the switch of that pair is one step
    late."""

    def __init__(self, data, offset=0.0, broken=None, nan_after=None, late_pair=None):
        self.data, self.offset, self.broken, self.nan_after, self.late_pair = data, offset, broken, nan_after, late_pair
        self.dt, self.k = data["dt"], 4
        self.moving = [k for k, b in enumerate(data["system"]["bodies"]) if b["moving"]]
        self.labels = model_labels(data["contacts"], data["stride"])
        self.pos = [[c for c, t in enumerate(b["channel_types"]) if t == "position"] for b in data["system"]["bodies"] if b["moving"]]
        self.where = {}   # state bytes -> first (episode, record) with that state
        for n, episode in enumerate(data["states"].numpy()):
            for r, x in enumerate(episode):
                self.where.setdefault(x.tobytes(), (n, r))

    def rollout_from(self, hist, ok, actions, return_gates=False):
        st, s = self.data["states"], self.data["stride"]
        steps = actions.shape[1]
        out, gates = [], []
        for row in range(len(hist)):
            n, r = self.where[hist[row, -1].numpy().tobytes()]
            traj = st[n, (r + s * torch.arange(steps + 1)).clamp_max(st.shape[1] - 1)].clone()
            for d, pc in enumerate(self.pos):
                traj[1:, d, pc] += self.offset
            if n == self.broken:
                traj[steps // 2:] = torch.nan
            if self.nan_after is not None and steps > self.nan_after:
                traj[self.nan_after:] = torch.nan
            out.append(traj)
            g = self.labels[n, (r // s + torch.arange(steps)).clamp_max(self.labels.shape[1] - 1)].double()
            if self.late_pair is not None:
                g[:, self.late_pair] = self.labels[n, (r // s + torch.arange(steps) - 1).clamp(0, self.labels.shape[1] - 1),
                                                   self.late_pair].double()
            gates.append(g)
        out, gates = torch.stack(out), torch.stack(gates)
        return (out, gates) if return_gates else out

    def rollout(self, initial, actions, return_gates=False):
        hist = torch.zeros(len(initial), self.k, *initial.shape[1:], dtype=initial.dtype)
        hist[:, -1] = initial
        return self.rollout_from(hist, None, actions, return_gates)


def dataset(tmp, name, evaluation=True, records=181):
    synthetic.write(tmp, name, episodes=20, records=records, seed=5)
    if not evaluation:
        path = Path(tmp) / "system.json"
        system = json.loads(path.read_text())
        system.pop("evaluation")
        path.write_text(json.dumps(system))
    return step_average_rates(load_data(tmp, "cpu", model_step_s=0.02), KINDS)


def brute(true, pred, tol):
    """(most matches, smallest sum of |pred - true| among them) over all one-to-one matchings within tol."""
    best = (0, 0)

    def go(i, used, count, cost):
        nonlocal best
        if i == len(true):
            best = max(best, (count, cost), key=lambda x: (x[0], -x[1]))
            return
        go(i + 1, used, count, cost)
        for j, p in enumerate(pred):
            d = abs(int(p) - int(true[i]))
            if j not in used and d <= tol:
                go(i + 1, used | {j}, count + 1, cost + d)
    go(0, frozenset(), 0, 0)
    return best


class Events(unittest.TestCase):
    def test_match(self):
        dts, missed, false = match_events([10, 20, 30], [11, 25, 30, 40], 2)
        self.assertEqual((dts, missed, false), ([1, 0], 1, 2))
        self.assertEqual(match_events([], [3], 1), ([], 0, 1))
        self.assertEqual(match_events([3], [], 1), ([], 1, 0))

    def test_match_prefers_exact(self):   # earliest-first matching kept the 2-step pair and called the exact one false
        self.assertEqual(match_events([10], [8, 10], 2), ([0], 0, 1))
        self.assertEqual(match_events([10, 12], [12], 2), ([0], 1, 0))

    def test_match_optimal(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            true = np.sort(rng.choice(25, rng.integers(0, 6), replace=False))
            pred = np.sort(rng.choice(25, rng.integers(0, 6), replace=False))
            dts, missed, false = match_events(true, pred, 2)
            self.assertEqual((len(dts), sum(abs(d) for d in dts)), brute(true, pred, 2))
            self.assertEqual((missed, false), (len(true) - len(dts), len(pred) - len(dts)))

    def test_physical_switches(self):
        phys = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0], bool)    # contact during each step; a one-step gap at 6
        gate = phys.copy()
        gate[1:] |= phys[:-1]                                         # the model's switch: this step or the step before
        expect = phys[:-1].copy()
        expect[6] = True                                              # a switch cannot show a one-step gap
        self.assertEqual(physical_switches(gate[None, :, None])[0, :, 0].tolist(), expect.tolist())

    def test_exclude_timing(self):
        """Matched events move from timing to left_out by the value at the record of their recorded step: free rollouts
        from step 0, windows from their start step (record = step x stride)."""
        values = np.zeros((2, 41), bool)
        values[1, 20:30] = True                                        # episode 1, records 20-29 (steps 10-14)
        tim = np.array([[0, 0, 0, 1, 12], [1, 0, 0, 1, 12], [1, 1, 1, 0, 9], [1, 0, 1, 0, 15]])
        raw = {k: dict(timing=tim.copy()) for k in ("contacts", "contacts_physical")}
        exclude_timing(raw, "free", values, [0], 2)                     # pair 0 only
        self.assertEqual(raw["contacts"]["left_out"].tolist(), [[1, 0, 0, 1, 12]])
        self.assertEqual(raw["contacts_physical"]["timing"].tolist(), [tim[0].tolist(), tim[2].tolist(), tim[3].tolist()])
        tim = np.array([[0, 0, 0, 1, 12], [1, 1, 1, 0, 9], [1, 1, 0, 0, 12], [2, 0, 1, 0, 5]])
        raw = {"row": np.array([0, 1, 1]), "start": np.array([0, 0, 5]),   # windows 1 and 2: episode 1 from steps 0 and 5
               **{k: dict(timing=tim.copy()) for k in ("contacts", "contacts_physical")}}
        exclude_timing(raw, "windows", values, [0, 1], 2)
        self.assertEqual(raw["contacts"]["left_out"].tolist(), [[1, 1, 0, 0, 12], [2, 0, 1, 0, 5]])   # steps 12, 5 + 5
        self.assertEqual(raw["contacts"]["timing"].tolist(), [[0, 0, 0, 1, 12], [1, 1, 1, 0, 9]])

    def test_tally(self):
        lab = np.zeros((1, 12, 1), bool)
        lab[0, 3:7] = True                       # onset at 3, offset at 7
        gate = np.zeros_like(lab)
        gate[0, 4:10] = True                     # onset at 4 (late by 1), offset at 10 (late by 3)
        t = contact_tally(gate, lab, np.array([12]), 2)
        self.assertEqual(t["counts"][0, 0].tolist(), [[1, 1, 0, 0], [1, 1, 1, 1]])
        self.assertEqual(t["timing"].tolist(), [[0, 0, 0, 1, 3]])   # row, pair, onset, 1 step late, recorded at 3
        self.assertEqual(int(t["acc"][0, 0]), 12 - 4)
        t = contact_tally(gate, lab, np.array([6]), 2)   # only the first 6 steps are valid
        self.assertEqual(t["counts"][0, 0].tolist(), [[1, 1, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(int(t["den"][0]), 6)


class Flags(unittest.TestCase):
    def test_tilt(self):
        system = synthetic.g1_system(11)
        flags = compile_flags({"fall": {"any": [{"body": "robot", "tilt_above_rad": math.radians(45)}]}}, system,
                              ["left_foot_ground", "right_foot_ground"])
        x = torch.zeros(1, 3, 1, 67, dtype=torch.float64)
        half = [math.radians(a) / 2 for a in (50, 90, 40)]
        x[0, 0, 0, 3:7] = torch.tensor([math.cos(half[0]), math.sin(half[0]), 0, 0])    # 50 deg about x
        x[0, 1, 0, 3:7] = torch.tensor([math.cos(half[1]), 0, 0, math.sin(half[1])])    # 90 deg about z (yaw)
        x[0, 2, 0, 3:7] = 2 * torch.tensor([math.cos(half[2]), 0, math.sin(half[2]), 0])  # 40 deg about y, not unit
        hit = flag_trace(flags["fall"], x, torch.zeros(1, 3, 2, dtype=torch.bool))
        self.assertEqual(hit[0].tolist(), [True, False, False])

    def test_bad_conditions(self):
        system = synthetic.so101_system(11)
        pairs = ["fixed_jaw_cube", "moving_jaw_cube", "cube_table"]
        for bad in ({"all": [{"body": "cube", "channel": "height", "above": 0.02}]},
                    {"all": [{"body": "box", "channel": "z", "above": 0.02}]},
                    {"all": [{"pair": "jaw_cube"}]},
                    {"all": [{"body": "arm", "tilt_above_rad": 0.5}]},
                    {"any": []}):
            with self.assertRaises(ValueError):
                compile_flags({"f": bad}, system, pairs)


class TaskMetrics(unittest.TestCase):
    def test_perfect_g1(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = dataset(tmp, "g1_locomotion")
        ids = torch.arange(len(data["splits"]))
        r = evaluate(Replay(data), data, ids, by_family=True)
        self.assertLess(r["selection_score"], 1e-12)
        for at in ("target", "final"):   # the first-state rollout: a report for G1
            self.assertEqual({m: q["max"] for m, q in r["free"][at]["robot"].items()},
                             dict(pos_m=0.0, angle_rad=0.0, joint_rms_rad=0.0, vel_mps=0.0))
        self.assertEqual(r["free"]["contacts"]["all"]["event_error_rate"], 0.0)
        self.assertEqual(r["long_windows"]["windows"], 0)   # 1.8 s episodes: no 2 s window
        self.assertEqual(r["long_windows"]["5.0"]["robot"]["pos_m"], NONE)
        self.assertEqual(r["finite_fraction"], 1.0)
        n = ((data["lengths"] - 1) // 2 + 1).numpy()
        horizons = (("0.02", 1), ("0.2", 10), ("0.5", 25), ("1.0", 50))
        starts = [set(range(5, m - 1, 25)) | {max(0, m - 1 - 50)} for m in n]   # 5, 30, ... and the end window
        for h, hs in horizons:   # a horizon counts if start + horizon <= last step
            self.assertEqual(r["windows"][h]["windows"], sum(s + hs <= m - 1 for st, m in zip(starts, n) for s in st))
        for h, _ in horizons:
            for m in ("pos_m", "angle_rad", "joint_rms_rad", "vel_mps"):
                self.assertLess(r["windows"][h]["robot"][m]["max"], 1e-12)
        c = r["windows"]["contacts"]
        self.assertGreater(c["all"]["onset"]["true"], 10)
        self.assertEqual(c["all"]["event_error_rate"], 0.0)
        self.assertEqual(c["all"]["gate_accuracy"], 1.0)
        self.assertEqual(set(c), {"left_foot_ground", "right_foot_ground", "all"})
        p = r["windows"]["contacts_physical"]["all"]   # the synthetic gait has no one-step gaps
        self.assertGreater(p["onset"]["true"], 10)
        self.assertEqual(p["event_error_rate"], 0.0)
        f = r["windows"]["flags"]["fall"]
        self.assertEqual((f["rollouts"], f["recorded"], f["predicted"], f["disagreement"]), (r["windows"]["windows"], 0, 0, 0.0))
        by = r["windows"]["by_start_contact"]
        self.assertLessEqual(set(by), {"none", "left_foot_ground", "right_foot_ground", "left_foot_ground+right_foot_ground"})
        for h, _ in horizons:
            self.assertEqual(sum(v[h]["windows"] for v in by.values()), r["windows"][h]["windows"])
        contacts = data["contacts"].numpy()   # event windows: 5 steps before each recorded onset, while 0.2 s fits
        expect = 0
        for e, m in enumerate(n):
            ph = np.stack([contacts[e, 2 * k:2 * k + 2].any(0) for k in range(m - 1)])
            onsets = np.flatnonzero((ph[1:] & ~ph[:-1]).any(-1)) + 1
            expect += len({max(0, s) for s in onsets - 5 if max(0, s) < m - 10})   # an early onset: from step 0
        ev = r["event_windows"]
        self.assertGreater(expect, 20)
        self.assertEqual(ev["0.2"]["windows"], expect)
        self.assertEqual(ev["contacts"]["all"]["event_error_rate"], 0.0)
        quick = evaluate(Replay(data), data, ids, score_only=True)   # checks during training: no report-only modes
        self.assertEqual([k for k in ("free", "windows", "long_windows", "event_windows") if k in quick], ["windows"])
        self.assertEqual(quick["windows"]["1.0"], r["windows"]["1.0"])
        fallen = evaluate(Replay(data, offset=-0.5), data, ids)       # every predicted base below 0.35 m
        f = fallen["windows"]["flags"]["fall"]
        self.assertEqual((f["false"], f["disagreement"]), (f["rollouts"], 1.0))
        term = next(t for t in fallen["score_terms"] if t["stat"] == "windows/flags/fall/disagreement")
        self.assertAlmostEqual(term["term"], 100.0)
        self.assertEqual(set(r["families"]), {"walk", "perturbed", "airborne"})
        self.assertEqual(r["families"]["airborne"]["windows"]["contacts"]["all"]["onset"]["true"], 0)
        fam = data["family"].numpy()
        self.assertEqual(r["selected"]["episodes"], int((fam != 2).sum()))
        self.assertEqual(len(r["per_episode"]), len(ids))
        json.dumps(r, allow_nan=False)

    def test_offset_and_nonfinite_so101(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = dataset(tmp, "so101_grasp")
        ids = torch.arange(len(data["splits"]))
        r = task_metrics(Replay(data, offset=0.003, broken=4), data, ids, data["system"]["evaluation"])
        self.assertAlmostEqual(r["nonfinite_fraction"], 1 / len(ids))
        expect = np.sqrt(3) * 0.003
        for key in ("median", "p95", "max"):   # episode 4 is finite for 0.2 s from every start
            self.assertAlmostEqual(r["windows"]["0.2"]["cube"]["pos_m"][key], expect, places=12)
        target = r["free"]["target"]["cube"]["pos_m"]   # episode 4 is NaN at its target: one error of FAIL
        self.assertEqual(r["free"]["target"]["nonfinite"], 1)
        self.assertAlmostEqual(target["median"], expect, places=12)
        self.assertAlmostEqual(target["p95"], np.quantile([expect] * (len(ids) - 1) + [FAIL], 0.95), places=9)
        self.assertEqual(target["max"], FAIL)
        self.assertLess(r["windows"]["0.2"]["arm"]["joint_rms_rad"]["max"], 1e-12)
        self.assertLess(r["windows"]["0.2"]["cube"]["angle_rad"]["max"], 1e-12)
        self.assertEqual(r["free"]["contacts"]["all"]["event_error_rate"], 0.0)
        terms = {t["stat"]: t for t in r["score_terms"]}
        self.assertAlmostEqual(terms["free/target/cube/pos_m/p95"]["term"], target["p95"] / 0.005, places=9)
        self.assertAlmostEqual(terms["nonfinite_fraction"]["term"], 1 / len(ids) / 0.01, places=9)
        self.assertAlmostEqual(r["selection_score"] / sum(t["term"] for t in r["score_terms"]), 1.0, places=12)   # sum() compensates
        self.assertFalse(r["per_episode"][4]["scored_finite"])
        self.assertLessEqual({"free", "windows"}, set(r["per_episode"][4]["nonfinite_modes"]))
        self.assertTrue(all(e["scored_finite"] and not e["nonfinite_modes"] for e in r["per_episode"] if e["index"] != 4))
        json.dumps(r, allow_nan=False)

    def test_errors_capped_so101(self):
        """Every error counts as at most FAIL. A finite rollout far off (offset 1e5 m) gets the same position statistics
        as a NaN one; an offset of 1e200 m overflows the error (inf) and counts as FAIL too, so the score stays finite
        and the report is valid JSON. The rollouts stay finite. Per episode: the raw error, None if it is not finite."""
        with tempfile.TemporaryDirectory() as tmp:
            data = dataset(tmp, "so101_grasp")
        ids = torch.arange(len(data["splits"]))
        spec = data["system"]["evaluation"]
        nan = task_metrics(Replay(data, nan_after=1), data, ids, spec)
        failed = dict(mean=FAIL, median=FAIL, p95=FAIL, max=FAIL)
        self.assertEqual(nan["free"]["target"]["cube"]["pos_m"], failed)
        for offset, raw in ((1e5, math.sqrt(3) * 1e5), (1e200, None)):
            r = task_metrics(Replay(data, offset=offset), data, ids, spec)
            self.assertEqual(r["finite_fraction"], 1.0)
            for q in (r["free"]["target"]["cube"]["pos_m"], r["free"]["path_rmse_m"], r["windows"]["1.0"]["cube"]["pos_m"],
                      r["windows"]["0.02"]["cube"]["pos_m"], r["event_windows"]["0.2"]["cube"]["pos_m"]):
                self.assertEqual(q, failed)
            self.assertEqual(r["free"]["target"]["nonfinite"], 0)
            self.assertLess(r["windows"]["1.0"]["cube"]["angle_rad"]["max"], 1e-12)
            self.assertTrue(math.isfinite(r["selection_score"]))
            self.assertLess(r["selection_score"], nan["selection_score"])
            got = r["per_episode"][0]["free"]["target"]["cube"]["pos_m"]
            if raw is None:
                self.assertIsNone(got)
            else:
                self.assertAlmostEqual(got / raw, 1.0, places=9)
            json.dumps(r, allow_nan=False)

    def test_free_metrics_overflow(self):
        """free_metrics (data without an evaluation block): an error that overflows makes the rollout count as not
        finite (the score stays finite, valid JSON); a finite error, however large, keeps the released rule."""
        with tempfile.TemporaryDirectory() as tmp:
            data = dataset(tmp, "so101_grasp", evaluation=False)
        ids = torch.arange(len(data["splits"]))
        r = free_metrics(Replay(data, offset=1e200), data, ids)
        self.assertEqual((r["finite_fraction"], r["target_pos_m"]["p95"]), (0.0, 1e3))
        self.assertTrue(math.isfinite(r["selection_score"]))
        json.dumps(r, allow_nan=False)
        r = free_metrics(Replay(data, offset=1e5), data, ids)
        self.assertEqual(r["finite_fraction"], 1.0)
        self.assertAlmostEqual(r["target_pos_m"]["max"] / (math.sqrt(3) * 1e5), 1.0, places=9)

    def test_late_nonfinite_so101(self):
        """Episode 4 turns NaN halfway through each rollout (window: at step 25 of 50). Its windows keep their errors
        at the horizons before that; at later horizons they count as FAIL; the flags count them as disagreements."""
        with tempfile.TemporaryDirectory() as tmp:
            data = dataset(tmp, "so101_grasp")
        ids = torch.arange(len(data["splits"]))
        spec = data["system"]["evaluation"]
        r = task_metrics(Replay(data, broken=4), data, ids, spec)
        n = ((data["lengths"] - 1) // 2 + 1).numpy()
        fits = [min(50, m - 1 - s) for e, m in enumerate(n) for s in range(5, m - 1, 25)]
        mine = [min(50, n[4] - 1 - s) for s in range(5, n[4] - 1, 25)]
        self.assertTrue(any(f < 25 for f in mine) and any(f >= 50 for f in mine))
        w = r["windows"]
        for h, hs in (("0.02", 1), ("0.2", 10), ("0.5", 25), ("1.0", 50)):
            bad = sum(hs <= f and hs >= 25 for f in mine)
            self.assertEqual((w[h]["windows"], w[h]["nonfinite"]), (sum(f >= hs for f in fits), bad))
            q = w[h]["cube"]["pos_m"]
            self.assertEqual(q["max"], FAIL if bad else 0.0)
            self.assertAlmostEqual(q["p95"], np.quantile([0.0] * (w[h]["windows"] - bad) + [FAIL] * bad, 0.95), places=9)
        broken = sum(f >= 25 for f in mine)   # NaN before the end of the window
        self.assertAlmostEqual(w["nonfinite_fraction"], broken / len(fits))
        for name, f in w["flags"].items():
            self.assertEqual((f["rollouts"], f["nonfinite"], f["missed"], f["false"]), (len(fits), broken, 0, 0))
            self.assertAlmostEqual(f["disagreement"], broken / len(fits))
        for name, f in r["free"]["flags"].items():
            self.assertEqual((f["rollouts"], f["nonfinite"], f["missed"], f["false"]), (len(ids), 1, 0, 0))
            self.assertAlmostEqual(f["disagreement"], 1 / len(ids))
        self.assertEqual((r["free"]["target"]["nonfinite"], r["free"]["final"]["nonfinite"]), (1, 1))
        terms = {t["stat"]: t["value"] for t in r["score_terms"]}
        self.assertAlmostEqual(terms["free/flags/lifted/disagreement"], 1 / len(ids))
        self.assertEqual(terms["windows/1.0/cube/pos_m/p95"], w["1.0"]["cube"]["pos_m"]["p95"])
        worst = r["per_episode"][4]["windows_worst"]
        self.assertEqual(worst["0.2"]["cube"]["pos_m"], 0.0)
        self.assertIsNone(worst["1.0"]["cube"]["pos_m"])
        self.assertIsNone(r["per_episode"][4]["free"]["target"]["cube"]["pos_m"])
        self.assertEqual(r["per_episode"][4]["windows_flag_errors"], broken * len(w["flags"]))
        json.dumps(r, allow_nan=False)
        data["target_step"] = 20   # before the NaN at step 45 of the free rollout: the target error stays
        r = task_metrics(Replay(data, broken=4), data, ids, spec)
        self.assertEqual((r["free"]["target"]["nonfinite"], r["free"]["final"]["nonfinite"]), (0, 1))
        self.assertEqual((r["free"]["target"]["cube"]["pos_m"]["max"], r["free"]["path_rmse_m"]["max"]), (0.0, 0.0))
        self.assertEqual(r["free"]["final"]["cube"]["pos_m"]["max"], FAIL)
        self.assertEqual(r["per_episode"][4]["free"]["target"]["cube"]["pos_m"], 0.0)
        self.assertIsNone(r["per_episode"][4]["free"]["final"]["cube"]["pos_m"])

    def test_event_starts_early_onset(self):
        phys = np.zeros((1, 99, 1), bool)
        for onset in (3, 20, 96):
            phys[0, onset:onset + 2] = True
        spec = {"horizons_s": [0.2, 1.0], "lead_s": 0.1, "events": ["onset"]}
        self.assertEqual(event_starts(phys, [100], spec, 0.02)[0].tolist(), [0, 15])   # 96 - 5 leaves no 0.2 s

    def test_family_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic.write(tmp, "g1_locomotion", episodes=10, records=61, seed=5)
            path, packet = Path(tmp) / "system.json", Path(tmp) / "unified_data.npz"
            system = json.loads(path.read_text())
            with np.load(packet) as z:
                arrays = {k: z[k] for k in z.files}
            for fams, with_family in ((None, True), (system["families"], False),
                                      ([{k: v for k, v in f.items() if k != "select"} for f in system["families"]], True),
                                      ([dict(f, select=1) for f in system["families"]], True)):
                bad = {k: v for k, v in system.items() if k != "families"}
                if fams is not None:
                    bad["families"] = fams
                path.write_text(json.dumps(bad))
                np.savez(packet, **{k: v for k, v in arrays.items() if with_family or k != "family"})
                with self.assertRaises(ValueError):
                    load_data(tmp, "cpu")

    def test_report_only_modes_g1(self):
        """6 s episodes: long windows at 2 and 5 s. Rollouts longer than 120 steps turn NaN at step 120: the windows
        (50 steps), the long windows at 2 s and the first-state rollouts at the 2 s target stay finite, the long
        windows at 5 s and the first-state rollouts at the episode end do not. For G1 the first-state rollouts and
        the long windows are reports (the score does not use them): the score and nonfinite_fraction stay those of a
        perfect model."""
        with tempfile.TemporaryDirectory() as tmp:
            data = dataset(tmp, "g1_locomotion", records=601)
        ids = torch.arange(len(data["splits"]))
        r = evaluate(Replay(data, nan_after=120), data, ids, by_family=True)
        n = ((data["lengths"] - 1) // 2 + 1).numpy()
        starts = [range(5, m - 100, 50) for m in n]
        lw = r["long_windows"]
        self.assertEqual(lw["windows"], sum(len(s) for s in starts))
        for h, hs in (("2.0", 100), ("5.0", 250)):
            count = sum(s + hs <= m - 1 for st, m in zip(starts, n) for s in st)
            self.assertGreater(count, 0)
            self.assertEqual((lw[h]["windows"], lw[h]["nonfinite"]), (count, count if hs > 120 else 0))
            self.assertEqual(lw[h]["robot"]["pos_m"]["max"], FAIL if hs > 120 else 0.0)
        broken = sum(min(250, m - 1 - s) >= 120 for st, m in zip(starts, n) for s in st)
        self.assertAlmostEqual(lw["nonfinite_fraction"], broken / lw["windows"])
        self.assertEqual((r["free"]["target"]["nonfinite"], r["free"]["final"]["nonfinite"]), (0, len(ids)))
        self.assertEqual(r["windows"]["nonfinite_fraction"], 0.0)
        self.assertEqual((r["nonfinite_fraction"], r["finite_fraction"]), (0.0, 1.0))
        self.assertLess(r["selection_score"], 1e-12)
        self.assertTrue(all(e["scored_finite"] for e in r["per_episode"]))
        self.assertTrue(all("windows" not in e["nonfinite_modes"] and "free" in e["nonfinite_modes"] for e in r["per_episode"]))
        self.assertEqual(sum("long_windows" in e["nonfinite_modes"] for e in r["per_episode"]),
                         len({e for e, (st, m) in enumerate(zip(starts, n)) for s in st if min(250, m - 1 - s) >= 120}))
        quick = evaluate(Replay(data, nan_after=120), data, ids, score_only=True)
        self.assertEqual(quick["selection_score"], r["selection_score"])
        self.assertNotIn("long_windows", quick)
        lines = describe("replay", "all", r)
        self.assertIn("finite in scored modes 1.0000", lines[0])
        self.assertTrue(any(x.startswith("  long windows 5.0 s (") for x in lines))
        json.dumps(r, allow_nan=False)

    def test_timing_exclude_array_required(self):
        """An evaluation block whose timing rule names a per-record array: the data must have it ([N, T] integers)."""
        with tempfile.TemporaryDirectory() as tmp:
            synthetic.write(tmp, "so101_grasp", episodes=10, records=61, seed=5)
            packet = Path(tmp) / "unified_data.npz"
            self.assertEqual(load_data(tmp, "cpu")["record_arrays"]["phase"].shape, (10, 61))
            with np.load(packet) as z:
                arrays = {k: z[k] for k in z.files}
            for phase in (None, arrays["phase"][:, :-1], arrays["phase"].astype(np.float32)):
                np.savez(packet, **{k: v for k, v in arrays.items() if k != "phase"}, **({} if phase is None else {"phase": phase}))
                with self.assertRaises(ValueError):
                    load_data(tmp, "cpu")
            np.savez(packet, **arrays)
            path = Path(tmp) / "system.json"
            system = json.loads(path.read_text())
            for phases in (None, [p for p in system["phases"] if p != "grip"]):   # the list that names the values
                path.write_text(json.dumps({**{k: v for k, v in system.items() if k != "phases"},
                                            **({} if phases is None else {"phases": phases})}))
                with self.assertRaisesRegex(ValueError, "phases"):
                    load_data(tmp, "cpu")

    def test_no_rollouts(self):
        """Statistics without any rollout are None, not 1000. The score that needs one: None for a family with select:
        false, an error for a family with select: true, the selected families and all episodes."""
        with tempfile.TemporaryDirectory() as tmp:
            data = dataset(tmp, "g1_locomotion")
        ids = torch.arange(len(data["splits"]))
        spec = data["system"]["evaluation"]
        fam = data["family"]
        data["lengths"][fam == 2] = 81   # airborne (select: false): 40 steps, no 1 s window
        r = task_metrics(Replay(data), data, ids, spec, by_family=True)
        air = r["families"]["airborne"]
        self.assertIsNone(air["selection_score"])
        self.assertEqual((air["windows"]["1.0"]["windows"], air["windows"]["1.0"]["robot"]["pos_m"]), (0, NONE))
        self.assertGreater(air["windows"]["0.5"]["windows"], 0)
        terms = {t["stat"]: t for t in air["score_terms"]}
        self.assertEqual((terms["windows/1.0/robot/pos_m/p95"]["value"], terms["windows/1.0/robot/pos_m/p95"]["term"]), (None, None))
        self.assertEqual(terms["nonfinite_fraction"]["value"], 0.0)
        for v in air["windows"]["by_start_contact"].values():
            self.assertEqual(v["1.0"]["robot"]["joint_rms_rad"], NONE)
        self.assertLess(max(r["selection_score"], r["selected"]["selection_score"]), 1e-12)
        lines = describe("replay", "family airborne", air, detail=True)
        self.assertIn("score not available (3 scored statistics without rollouts)", lines[0])
        self.assertFalse(any("1.0 s" in x for x in lines))
        self.assertTrue(any(x.startswith("  windows 0.5 s (") for x in lines))
        self.assertTrue(any(x.startswith("  windows 0.5 s from contacts none (") for x in lines))
        json.dumps(r, allow_nan=False)
        data["lengths"][fam == 0] = 81   # walk (select: true) too
        with self.assertRaisesRegex(ValueError, "evaluation score of family walk: no rollouts for windows/1.0/robot/pos_m/p95"):
            task_metrics(Replay(data), data, ids, spec, by_family=True)
        self.assertLess(task_metrics(Replay(data), data, ids, spec)["selection_score"], 1e-12)   # all: perturbed reach 1 s
        late = dict(spec, windows=dict(spec["windows"], first_start_s=100.0, end_window=False))   # no window at all
        with self.assertRaisesRegex(ValueError, "evaluation score of all: no rollouts"):
            task_metrics(Replay(data), data, ids, late)
        r = task_metrics(Replay(data), data, ids, dict(late, score=[{"stat": "free/target/robot/pos_m/p95", "scale": 0.05}]),
                         by_family=True)
        w = r["windows"]
        self.assertEqual((w["windows"], w["nonfinite_fraction"], w["1.0"]["robot"]["pos_m"]), (0, None, NONE))
        self.assertEqual((w["contacts"]["all"]["onset"]["true"], w["contacts"]["all"]["event_error_rate"],
                          w["contacts"]["all"]["gate_accuracy"], w["contacts"]["all"]["timing_s"]), (0, None, None, NONE))
        self.assertEqual((w["flags"]["fall"]["rollouts"], w["flags"]["fall"]["disagreement"]), (0, None))
        self.assertEqual(w["by_start_contact"], {})
        self.assertEqual(r["selection_score"], 0.0)
        self.assertFalse(any(x.startswith("  windows") for x in describe("replay", "all", r, detail=True)))
        json.dumps(r, allow_nan=False)

    def test_flags_so101(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = dataset(tmp, "so101_grasp")
        ids = torch.arange(len(data["splits"]))
        spec = dict(data["system"]["evaluation"], score=[{"stat": "free/flags/up/disagreement", "scale": 0.02}],
                    flags={"up": {"all": [{"body": "cube", "channel": "z", "above": 0.02}, {"pair": "cube_table", "on": False}]}})
        f = task_metrics(Replay(data), data, ids, spec)["free"]["flags"]["up"]
        self.assertGreater(f["recorded"], 0)
        self.assertEqual((f["predicted"], f["matched"], f["disagreement"], f["first_s"]["max"]), (f["recorded"], f["recorded"], 0.0, 0.0))
        r = task_metrics(Replay(data, offset=-0.01), data, ids, spec)   # the predicted cube stays below 0.02 m
        f = r["free"]["flags"]["up"]
        self.assertEqual((f["predicted"], f["missed"], f["false"]), (0, f["recorded"], 0))
        self.assertAlmostEqual(r["selection_score"], f["recorded"] / f["rollouts"] / 0.02)
        lifted = task_metrics(Replay(data), data, ids, data["system"]["evaluation"])["free"]["flags"]
        self.assertEqual(set(lifted), {"lifted", "dropped"})
        self.assertEqual([lifted[k]["disagreement"] for k in lifted], [0.0, 0.0])

    def test_timing_exclude_so101(self):
        """The SO101 block leaves the matched cube-table events at records in the phases close and grip out of the
        timing (synthetic phase: 0.25 s per phase, close = records 75-99, grip = 100-124). Event counts do not
        change. A model whose cube-table switch is one step late has a 20 ms timing error at every matched cube-table
        event; with all phases left out its timing has no event left."""
        with tempfile.TemporaryDirectory() as tmp:
            data = dataset(tmp, "so101_grasp")
        ids = torch.arange(len(data["splits"]))
        spec = data["system"]["evaluation"]
        self.assertEqual(spec["timing_exclude"]["values"], ["close", "grip"])
        plain = {k: v for k, v in spec.items() if k != "timing_exclude"}
        r, r0 = task_metrics(Replay(data), data, ids, spec), task_metrics(Replay(data), data, ids, plain)
        n = ((data["lengths"] - 1) // 2 + 1).numpy()
        phase = data["record_arrays"]["phase"]
        for key, lab in (("contacts", model_labels(data["contacts"], 2)), ("contacts_physical", physical_labels(data["contacts"], 2))):
            lab = lab[..., 2].numpy()
            for k, kind in enumerate(("onset", "offset")):
                change = (lab[:, 1:] & ~lab[:, :-1]) if k == 0 else (~lab[:, 1:] & lab[:, :-1])   # at step j + 1
                last = n - (1 if key == "contacts" else 2)   # valid steps of the view
                expect = sum(int(np.isin(phase[e, 2 * (j + 1)], [3, 4])) for e in range(len(n))
                             for j in np.flatnonzero(change[e]) if j + 1 < last[e])
                self.assertGreater(expect, 0)
                got = r["free"][key]
                self.assertEqual(got["cube_table"][kind]["timing_left_out"], expect, (key, kind))
                self.assertEqual(got["all"][kind]["timing_left_out"], expect)
                self.assertEqual(got["fixed_jaw_cube"][kind]["timing_left_out"], 0)
                self.assertEqual({x: got["all"][kind][x] for x in ("true", "matched", "missed", "false")},
                                 {x: r0["free"][key]["all"][kind][x] for x in ("true", "matched", "missed", "false")})
        self.assertIn("left out of the timing", "\n".join(describe("replay", "all", r)))
        json.dumps(r, allow_nan=False)
        late = Replay(data, late_pair=2)
        every = dict(spec, timing_exclude=dict(spec["timing_exclude"], values=data["system"]["phases"]))
        r0, r = task_metrics(late, data, ids, plain), task_metrics(Replay(data, late_pair=2), data, ids, every)
        c0, c = r0["free"]["contacts"]["cube_table"]["onset"], r["free"]["contacts"]["cube_table"]["onset"]
        self.assertEqual((c0["timing_s"]["p95"], c0["bias_s"]), (0.02, 0.02))
        self.assertEqual((c["timing_s"], c["bias_s"], c["timing_left_out"]), (ZERO, 0.0, c0["matched"]))
        self.assertEqual(r["free"]["contacts"]["all"]["timing_s"], ZERO)   # the jaw pairs are on time
        for bad in (dict(values=["lift_off"]), dict(names="stages"), dict(pairs=["jaw_table"]), dict(array="stage"),
                    {"names": None}):
            rule = {k: v for k, v in dict(spec["timing_exclude"], **bad).items() if v is not None}   # None: key removed
            with self.assertRaises(ValueError, msg=bad):
                task_metrics(Replay(data), data, ids, dict(spec, timing_exclude=rule))

    def test_bad_statistic_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = dataset(tmp, "g1_locomotion")
        spec = dict(data["system"]["evaluation"], score=[{"stat": "windows/1.0/robot/pos_mm/p95", "scale": 1.0}])
        with self.assertRaises(ValueError):
            task_metrics(Replay(data), data, torch.arange(4), spec)

    def test_without_evaluation_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = dataset(tmp, "so101_grasp", evaluation=False)
        ids = torch.arange(len(data["splits"]))
        model = Replay(data)
        plain = free_metrics(model, data, ids)
        r = evaluate(model, data, ids)
        self.assertEqual(json.dumps(r), json.dumps(plain))
        r = evaluate(model, data, ids, by_family=True)
        self.assertEqual(set(r["families"]), {"lift", "slip", "free_motion"})
        self.assertIn("selected", r)
        self.assertEqual(r["selected"]["episodes"], int((data["family"] != 2).sum()))


if __name__ == "__main__":
    unittest.main()
