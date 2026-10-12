"""NRD data round 2 (nedm.so101_policy.data_round, scripts/so101_policy/collect_round.py): task options, chunking
coverage, chunk slicing, collector phases of the decoder modes, the collector labels against run_episode, a cube move
cut into segments against MujocoWorlds, and the written data directory against the contract (export.check_arrays,
validate, merge_data's system.json rules).

CPU, about 20 s (the MuJoCo tests build one collector Scene; they skip without the visual-overlap cache or scipy).

    PYTHONPATH=src python -m unittest tests.so101_policy.test_data_round -v
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nedm.so101_policy import data_round as R

NOISY_SYSTEM = Path.home() / "so101_policy" / "data" / "noisy_v1" / "system.json"


def _scene_or_skip(test):
    try:
        return R.scene()
    except (RuntimeError, FileNotFoundError) as e:   # no overlap cache and no scipy
        test.skipTest(f"no collector scene with overlap grids: {e}")


def _short_episode(s, seed=12345, steps=30):
    """A collector start (task_start, warm-up) and a fixed command sequence: hold, then a slow move of the pan joint
    and the jaw (no contact needed)."""
    from nedm.contact_nrd.collect import so101_grasp as g
    rng = np.random.default_rng(seed)
    while True:
        try:
            cube, cyaw, q0, _ = g.task_start(s, rng, 40)
            break
        except g.PlanError:
            continue
    c = g.resting_cube(cube, cyaw)
    q = np.asarray(q0, float)
    cmd = np.repeat(q[None], steps + 1, 0)
    cmd[:, 0] += np.linspace(0, 0.15, steps + 1)
    cmd[:, 5] = np.clip(q[5] + np.linspace(0, 0.3, steps + 1), s.tlo[5], s.thi[5])
    cmd = np.clip(cmd, s.tlo, s.thi).astype(np.float32)
    ph = np.full(steps + 1, g.P["approach"], np.int8)
    return dict(q0=q, cube_pos=np.asarray(c["pos"], float), cube_quat=np.asarray(c["quat"], float), commands=cmd,
                phases=ph, n_steps=steps, events=[])


class Options(unittest.TestCase):
    def test_family_counts(self):
        for n in (13000, 1024, 7):
            c = R.family_counts(n)
            self.assertEqual(int(c.sum()), n)
            self.assertTrue((np.abs(c - n * np.array([f["share"] for f in R.FAMILIES])) <= 1).all())

    def test_options_deterministic_and_per_family(self):
        for f, fam in enumerate(R.FAMILIES):
            a = R.draw_options(R.task_rng(R.MASTER_SEED, 0, 17), f)
            b = R.draw_options(R.task_rng(R.MASTER_SEED, 0, 17), f)
            self.assertEqual(json.dumps({k: np.asarray(v).tolist() for k, v in a.items()}),
                             json.dumps({k: np.asarray(v).tolist() for k, v in b.items()}))
            self.assertEqual(a["horizon"], fam["horizon"])
            self.assertTrue(0.03 <= a["lift_h"] <= 0.12 and 10 <= a["tail"] <= 30)
            code = fam["code"]
            self.assertEqual(a["noise"] > 0, code == "N")
            self.assertEqual(a["dist_r"] > 0, code == "D")
            self.assertEqual(a["hold_s"] > 0, code == "L")
            if code == "R":
                self.assertTrue(2 <= a["rt1"] <= 10)
            if code == "N":
                self.assertTrue(0.15 <= a["sigma"] <= 0.3 and 1 <= a["noise"] <= 4)
            if code == "D":
                self.assertTrue(((a["dist_cand"][:, 0] >= 0.02) & (a["dist_cand"][:, 0] <= 0.06)).all())

    def test_refused_start_seeds(self):
        with self.assertRaises(ValueError):
            R.make_tasks(10, start0=1000)
        with self.assertRaises(ValueError):
            R.make_tasks(10, start0=900000)

    def test_start_seed_ranges_not_reserved(self):
        from nedm.so101_policy.mujoco_env import reserved_seeds
        res = reserved_seeds()
        for s0, n in ((R.START0, 13000), (R.GATE2_START0, 1024)):
            self.assertFalse(set(range(s0, s0 + n)) & res)
            self.assertTrue(s0 + n <= 900000 and s0 > 2023)


class Chunks(unittest.TestCase):
    def test_chunk_starts(self):
        self.assertEqual(R.chunk_starts(601), [0])
        self.assertEqual(R.chunk_starts(602), [0, 492])
        self.assertEqual(R.chunk_starts(1201), [0, 492, 984])
        self.assertEqual(R.CHUNK_STEP % 2, 0)

    def test_every_window_has_history_and_horizon_in_a_chunk(self):
        """A training window at record w (command grid) with its 4-state history (w - 6) and a 1 s rollout (w + 100)
        lies inside one chunk, for every segment length up to 12 s."""
        hist, horizon = 6, 100
        for L in range(2, 1202, 7):
            starts = R.chunk_starts(L)
            ends = [min(a + R.RECORDS, L) for a in starts]          # exclusive
            for w in range(hist, L - horizon, 2):
                self.assertTrue(any(a <= w - hist and w + horizon < e for a, e in zip(starts, ends)), (L, w))

    def test_slice_segment(self):
        L = 700
        q = np.tile(np.array([0.6, 0.0, 0.0, 0.8]), (L, 1))
        q[300:] *= -1.0     # continuous from record 300 on, w < 0: a chunk starting there must start with w >= 0
        cube = np.zeros((L, 13))
        cube[:, 3:7] = q
        r = dict(states=dict(arm=np.arange(L * 12, dtype=float).reshape(L, 12), cube=cube),
                 action=np.zeros((L, 6), np.float32), contacts=np.ones((L - 1, 3), np.uint8),
                 per_record=dict(phase=np.arange(L) % 7), per_interval=dict(contact_force=np.arange(L - 1, dtype=np.float32)))
        ep = R.slice_segment(r, 300, 400)
        self.assertEqual(ep["states"]["arm"].shape, (400, 12))
        self.assertEqual(ep["contacts"].shape, (399, 3))
        self.assertEqual(ep["per_interval"]["contact_force"][0], 300)
        self.assertGreaterEqual(ep["states"]["cube"][0, 3], 0)
        self.assertTrue(np.array_equal(np.abs(ep["states"]["cube"][:, 3:7]), np.abs(cube[300:700, 3:7])))


class Phases(unittest.TestCase):
    def test_collector_phase(self):
        import torch

        from nedm.contact_nrd.collect.so101_grasp import P
        from nedm.so101_policy.decoder import M, MODES
        n = len(MODES)
        mode = torch.arange(n).repeat(2)
        done = torch.tensor([False] * n + [True] * n)
        moving = torch.tensor([True] * n + [False] * n)
        ph = R.collector_phase(mode, done, moving).tolist()
        self.assertEqual(ph[M["free"]], P["approach"])
        self.assertEqual(ph[n + M["free"]], P["idle"])
        self.assertEqual(ph[M["hold"]], P["move"])
        self.assertEqual(ph[n + M["hold"]], P["hold"])
        for m, name in (("g_descend", "descend"), ("g_close", "close"), ("g_grip", "grip"), ("g_lift", "lift"),
                        ("p_lower", "lower"), ("p_open", "release"), ("p_retreat", "retreat")):
            self.assertEqual(ph[M[m]], P[name])


class CollectorLabels(unittest.TestCase):
    def test_label_equals_run_episode(self):
        """label_episode without events = the collector's run_episode on the same plan (every array bit-equal)."""
        from nedm.contact_nrd.collect import so101_grasp as g
        s = _scene_or_skip(self)
        ep = _short_episode(s)
        segs = R.label_episode(s, ep)
        self.assertEqual(len(segs), 1)
        plan = dict(q0=ep["q0"], cube=dict(pos=ep["cube_pos"], quat=ep["cube_quat"], spawn=False),
                    script=g.play(ep["commands"].astype(np.float64), ep["phases"]))
        ref = g.run_episode(s, plan, 2 * ep["n_steps"] + 1)
        a = segs[0]
        for k in ("arm", "cube"):
            self.assertTrue(np.array_equal(a["states"][k], ref["states"][k]), k)
        for k in ("action", "contacts"):
            self.assertTrue(np.array_equal(a[k], ref[k]), k)
        for grp in ("per_record", "per_interval"):
            for k in ref[grp]:
                self.assertTrue(np.array_equal(a[grp][k], ref[grp][k]), k)
        self.assertEqual(a["terminal"], ref["terminal"])
        self.assertEqual(len(a["action"]), 2 * ep["n_steps"] + 1)

    def test_cube_move_segments_match_mujoco_worlds(self):
        """A cube move: pass 1 in MujocoWorlds (records + set_cube) and pass 2 (two segments) agree bit for bit."""
        import torch

        from nedm.so101_policy.mujoco_env import REST_Z, MujocoWorlds
        s = _scene_or_skip(self)
        ep = _short_episode(s, seed=777, steps=24)
        t_move = 10
        pose = np.zeros(13)
        with MujocoWorlds(1, workers=0) as w:
            w.reset(None, dict(q=ep["q0"][None], cube_pos=ep["cube_pos"][None], cube_quat=ep["cube_quat"][None]),
                    warmup=True)
            c0 = w.state().cube_pos[0].numpy()
            pose[0:3] = [c0[0] + 0.03, c0[1] - 0.01, REST_Z]
            pose[3:7] = R.qz_mul(np.array([0.4]), w.state().cube_quat.numpy())[0]
            rec = np.zeros((2 * ep["n_steps"] + 1, 25), np.float32)
            rcon = np.zeros((2 * ep["n_steps"], 3), bool)
            rec[0] = w.raw_state()[0].numpy()
            jump = None
            for t in range(ep["n_steps"]):
                if t == t_move:
                    w.set_cube([0], torch.as_tensor(pose[None, 0:3]), torch.as_tensor(pose[None, 3:7]))
                    jump = w.raw_state()[0].numpy().astype(np.float32)
                w.step(torch.as_tensor(ep["commands"][t: t + 1], dtype=torch.float64))
                r = w.records.numpy().astype(np.float32)[0]
                rec[2 * t + 1], rec[2 * t + 2] = r[0], r[1]
                rcon[2 * t: 2 * t + 2] = w.record_contact.numpy()[0]
            env_end = int(w.end_reason[0])
        ep["events"] = [(t_move, pose)]
        segs = R.label_episode(s, ep, dict(records=rec, record_contact=rcon, jump_raw={t_move: jump}, env_end=env_end))
        self.assertEqual(len(segs), 2)
        self.assertEqual([len(x["action"]) for x in segs], [2 * t_move + 1, 2 * (ep["n_steps"] - t_move) + 1])
        for x in segs:
            self.assertEqual(x["check"], dict(states=True, contacts=True, end=True))
        self.assertTrue(np.allclose(segs[1]["states"]["cube"][0, :3], pose[:3]))


class Contract(unittest.TestCase):
    def test_write_round(self):
        from nedm.contact_nrd.collect import validate
        s = _scene_or_skip(self)
        ep = _short_episode(s, seed=4321, steps=40)
        segs = R.label_episode(s, ep)
        n = 3
        ro = dict(family=np.array([0, 2, 4]), index=np.arange(n), seeds=R.START0 + np.arange(n), tries=np.ones(n, int),
                  task_end=np.zeros(n, int), success=np.zeros(n, bool), success10=np.zeros(n, bool),
                  lifted=np.zeros(n, bool), end_xy=np.zeros(n), end_yaw=np.zeros(n), n_steps=np.full(n, 40),
                  env_end=np.zeros(n, int), noise=np.array([0, 2, 0]), sigma=np.array([0, 0.2, 0]),
                  rt1=np.full(n, -1), rt2=np.full(n, -1), rt_step1=np.full(n, -1), rt_step2=np.full(n, -1),
                  dist_step=np.full(n, -1), speed=np.ones(n, int), recoveries=np.zeros(n, int),
                  noisy_n=np.zeros(n, int), pause_s=np.zeros(n), hold_s=np.array([0, 0, 1.5]))
        system = R.round_system(s)
        off = R.family_offset(system)
        rows = []
        for k in range(n):
            rows += R.rows_of(ro, k, segs, off, k, chunked=True, records=31, step=20)   # 81 records -> 4 chunks
        self.assertEqual(len(rows), 3 * len(R.chunk_starts(81, 31, 20)))
        names = [f["name"] for f in system["families"]]
        self.assertEqual(names[-6:], list(R.FAMILY_NAMES) + [R.TAIL["name"]])
        self.assertFalse(system["families"][-1]["select"])
        splits = np.array([1 if r["meta"]["parent"] == 1 else 0 for r in rows], np.int8)
        with tempfile.TemporaryDirectory() as d:
            # the chunk length of this test is 31 records; the real one is RECORDS
            R.write_round(d, rows, splits, system, 31, dict(test=True))
            rep = validate.validate(d)
            self.assertTrue(rep["ok"], rep["fails"])
            z = np.load(Path(d) / "unified_data.npz")
            self.assertEqual(z["arm"].shape[1:], (31, 12))
            fam = z["family"]
            self.assertEqual(set(fam[np.array([r["meta"]["chunk"] for r in rows]) > 0].tolist()),
                             {names.index(R.TAIL["name"])})
            meta = np.load(Path(d) / "episodes.npz")
            self.assertEqual(len(meta["parent"]), len(rows))
        if NOISY_SYSTEM.exists():   # merge_data.py: same bodies, pairs, phases; families extend noisy_v1's in order
            other = json.loads(NOISY_SYSTEM.read_text())
            for key in ("bodies", "pairs", "phases"):
                self.assertEqual(json.dumps(system[key], sort_keys=True), json.dumps(other[key], sort_keys=True), key)
            theirs = [f["name"] for f in other["families"]]
            self.assertEqual(names[: len(theirs)], theirs)


# ---- round 3 (campaign policy_v3): PPO policy rollouts ----------------------------------------------------------------
EXPERT_SYSTEM = Path.home() / "so101_policy" / "data" / "expert_v2" / "system.json"


class PolicyOptions(unittest.TestCase):
    def test_policy_cells(self):
        c = R.policy_cells(13000).reshape(len(R.POLICIES), len(R.NOISE_LEVELS))
        self.assertEqual(int(c.sum()), 13000)
        self.assertEqual(c.sum(1).tolist(), [1820] * 5 + [1950] * 2)          # 70 % competent, 30 % NRD-trained
        self.assertTrue((c.max(1) - c.min(1) <= 1).all())                      # deterministic : noisy = 1 : 2
        comp = sum(p["share"] for p in R.POLICIES if p["family"] < 2)
        self.assertAlmostEqual(comp, 0.70)

    def test_policy_seeds(self):
        from nedm.so101_policy.mujoco_env import reserved_seeds
        from nedm.so101_policy.vec_env import reserved_seed_hits
        camp = np.arange(R.P_START0, R.P_START0 + 13000)
        pg = np.arange(R.PGATE_START0, R.PGATE_START0 + 512)
        R.check_policy_seeds(camp, "campaign")
        R.check_policy_seeds(pg, "pgate")
        self.assertEqual(reserved_seed_hits(camp), 0)              # trainable starts
        self.assertEqual(reserved_seed_hits(pg), len(pg))          # pgate: refused by load_starts (review R9)
        res = reserved_seeds()
        self.assertFalse(set(camp.tolist()) & res)
        self.assertFalse(set(pg.tolist()) & res)
        # disjoint from the earlier ranges: expert_v2 campaign / gate2, vision DAgger tasks 510,000-518,191
        for lo, n in ((R.START0, 13000), (R.GATE2_START0, 1024), (510000, 8192)):
            self.assertFalse(set(range(lo, lo + n)) & (set(camp.tolist()) | set(pg.tolist())))
        for bad, name in (((1000,), "campaign"), ((900001,), "pgate"), ((400005,), "campaign"), ((600000,), "pgate")):
            with self.assertRaises(ValueError):
                R.check_policy_seeds(np.array(bad), name)

    def test_expand_pgate_and_variant(self):
        n = 5
        t = dict(seeds=R.PGATE_START0 + np.arange(n), task=np.arange(n), policy=np.full(n, -1), noise_scale=np.zeros(n),
                 noise_level=np.zeros(n, int), family=np.full(n, -1), index=np.arange(n), master_seed=np.int64(1))
        e = R.expand_pgate(t)
        m = len(R.POLICIES) * len(R.PGATE_NOISE)
        self.assertEqual(len(e["seeds"]), m * n)
        self.assertEqual(sorted(e["index"].tolist()), list(range(m * n)))
        self.assertTrue(np.array_equal(e["seeds"], np.tile(t["seeds"], m)))
        for k in range(m * n):
            p, j = divmod(int(e["index"][k]) // n, len(R.PGATE_NOISE))
            self.assertEqual((int(e["policy"][k]), float(e["noise_scale"][k])), (p, R.PGATE_NOISE[j]))
            self.assertEqual(int(e["family"][k]), R.POLICIES[p]["family"])
        ro = dict(policy=np.array([0, 6]), noise_scale=np.array([0.5, 1.0]))
        self.assertEqual(R.policy_variant_of(0, ro, 0, 0, 0), "W:w5a.h")
        self.assertEqual(R.policy_variant_of(2, ro, 1, 0, 2), "Q:nRa.f/c2")
        self.assertTrue(all(len(R.policy_variant_of(f, ro, k, 0, 9)) <= 16 for f in range(3) for k in range(2)))


class PolicyRollouts(unittest.TestCase):
    """Pass 1 of a PPO policy in CPU MuJoCo (short horizon): deterministic rows = eval_policy.run_policy bit for bit;
    pass 2 = pass 1 bit for bit; the written folder passes validate and extends expert_v2's family list."""

    @classmethod
    def setUpClass(cls):
        cls.ckpt = R.policy_path("w5a")
        if not cls.ckpt.exists():
            raise unittest.SkipTest(f"no checkpoint {cls.ckpt}")
        try:
            R.scene()
        except (RuntimeError, FileNotFoundError) as e:
            raise unittest.SkipTest(f"no collector scene with overlap grids: {e}")
        tasks = R.make_policy_tasks(512, "pgate", idx=np.array([3, 7]), log=lambda *a, **k: None)
        cls.tasks = tasks
        e = R.expand_pgate(tasks)
        cls.sel = np.flatnonzero(e["policy"] == 0)
        cls.ep = R.subset(e, cls.sel)                         # w5a: 2 tasks x (deterministic, 1.0 x std)
        cls.ro = R.run_policy_rollouts(cls.ep, cls.ckpt, workers=0, horizon=60, log=lambda *a, **k: None)

    def test_deterministic_rows_equal_eval_policy(self):
        """Same batch layout (the deterministic rows alone): bit for bit eval_policy.run_policy. (Mixed with noisy rows
        the CPU float rounding of the policy MLP / decoder may differ at the 1e-7 level: batch-size dependent kernels.)"""
        E = R.eval_policy_module()
        ev = E.run_policy(self.tasks, "mujoco", str(self.ckpt), horizon=60, log=None)
        det_rows = np.flatnonzero(self.ep["noise_scale"] == 0)
        ro = R.run_policy_rollouts(R.subset(self.ep, det_rows), self.ckpt, workers=0, horizon=60, log=lambda *a, **k: None)
        det = ro["noise_scale"] == 0
        self.assertEqual(int(det.sum()), 2)
        for k_ev, k_ro in (("end", "task_end"), ("t_end", "task_steps"), ("success", "success"), ("lifted", "lifted")):
            self.assertTrue(np.array_equal(ev[k_ev], ro[k_ro][det]), k_ev)
        T = ev["q"].shape[1]
        rec = ro["records"][det]
        self.assertTrue(np.array_equal(ev["q"], rec[:, 2: 2 * T + 1: 2, :6]))
        self.assertTrue(np.array_equal(ev["cube"], rec[:, 2: 2 * T + 1: 2, 12:19]))
        self.assertTrue(np.array_equal(ev["u"].astype(np.float32), ro["commands"][det][:, :T]))
        self.assertTrue(np.array_equal(ro["act_exec"][det], ro["act_mean"][det]))
        mix, nz = self.ro, self.ro["noise_scale"] > 0
        self.assertTrue((np.abs(mix["act_exec"][nz] - mix["act_mean"][nz]).max(axis=(1, 2)) > 0).all())
        self.assertTrue(np.array_equal(mix["act_exec"][~nz], mix["act_mean"][~nz]))
        self.assertTrue((mix["n_steps"] == np.minimum(mix["task_steps"] + mix["tail"],
                                                      np.where(mix["env_end_step"] >= 1, mix["env_end_step"], 10 ** 9))).all())

    def test_pass2_and_write(self):
        from nedm.contact_nrd.collect import validate
        s = R.scene()
        ro = self.ro
        segs = []
        for k in range(len(ro["n_steps"])):
            ep, chk = R.episodes_of(ro, k)
            sg = R.label_episode(s, ep, chk)
            self.assertEqual(len(sg), 1)
            self.assertEqual(sg[0]["check"], dict(states=True, contacts=True, end=True))
            self.assertEqual(len(sg[0]["action"]), 2 * int(ro["n_steps"][k]) + 1)
            segs.append(sg)
        system = R.round_system(s, campaign=R.P_CAMPAIGN)
        off = R.family_offset(system, R.P_CAMPAIGN)
        names = [f["name"] for f in system["families"]]
        self.assertEqual(names[-4:], list(R.P_FAMILY_NAMES) + [R.POLICY_TAIL["name"]])
        self.assertEqual([f["select"] for f in system["families"][-4:]], [True, True, True, False])
        if EXPERT_SYSTEM.exists():       # merge order: production_plus + noisy_v1 + expert_v2 + policy_v3
            theirs = [f["name"] for f in json.loads(EXPERT_SYSTEM.read_text())["families"]]
            self.assertEqual(names[: len(theirs)], theirs)
        rows = []
        for k in range(len(segs)):
            rows += R.policy_rows_of(ro, k, segs[k], off, int(ro["index"][k]), chunked=True, records=31, step=20)
        tail = names.index(R.POLICY_TAIL["name"])
        self.assertTrue(all(int(r["scalars"]["family"]) in (off, tail) for r in rows))
        self.assertTrue(all(str(r["scalars"]["variant"]).startswith("W:w5a.") for r in rows))
        with tempfile.TemporaryDirectory() as d:
            R.write_round(d, rows, np.zeros(len(rows), np.int8), system, 31, dict(test=True), R.P_META_KEYS)
            rep = validate.validate(d)
            self.assertTrue(rep["ok"], rep["fails"])
            meta = np.load(Path(d) / "episodes.npz")
            self.assertEqual(sorted(set(meta["noise_scale"].tolist())), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
