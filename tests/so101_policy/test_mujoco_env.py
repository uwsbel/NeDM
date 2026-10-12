"""Tests of nedm.so101_policy.mujoco_env (CPU, a few seconds; MuJoCo / data / checkpoint tests skip without them).

    PYTHONPATH=src python -m unittest tests.so101_policy.test_mujoco_env -v
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import unittest

import numpy as np
import torch

from nedm.so101_policy import mujoco_env as me
from nedm.so101_policy.world import WorldState

HAS_MJ = importlib.util.find_spec("mujoco") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
# extract of the validation split (bench script extract) and an NRD checkpoint; env vars for other machines
VAL = Path(os.environ.get("NEDM_MJENV_VAL", Path.home() / "so101_policy/runs/mujoco_env/val_replay"))
CKPT = Path(os.environ.get("NEDM_MJENV_CKPT", Path.home() / "so101_policy/ckpt/nrd_final_s61.pt"))
mj = unittest.skipUnless(HAS_MJ, "needs mujoco")


def _overlap_ready():
    """The overlap grids can be installed: scipy here, or the cache exists."""
    if not HAS_MJ:
        return False
    if HAS_SCIPY:
        return True
    from nedm.contact_nrd.collect import so101_grasp as g
    s = g.Scene(g.load_config())
    try:
        me.install_overlap(s, build=False)
        return True
    except FileNotFoundError:
        return False


OVERLAP = unittest.skipUnless(_overlap_ready(), "needs scipy or the overlap cache (mujoco_env overlap-cache)")


def _seq(n, seed=0):
    """A random smooth raw-record sequence [n, 25] (float64): arm angles and rates, cube pose (unit quaternion,
    sign continuous) and rates."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)[:, None] * 0.02
    q = 0.3 * np.sin(t * rng.uniform(1, 5, 6) + rng.uniform(0, 6, 6))
    pos = np.array([0.2, 0.0, 0.015]) + 0.05 * np.sin(t * rng.uniform(1, 4, 3))
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    ang = 2.0 * t[:, 0] + 0.3
    quat = np.concatenate((np.cos(ang / 2)[:, None], np.sin(ang / 2)[:, None] * ax), 1)
    return np.concatenate((q, rng.normal(size=(n, 6)), pos, quat, rng.normal(size=(n, 6))), 1)


class MujocoEnvTest(unittest.TestCase):

    def test_step_rates_equal_data_pipeline(self):
        """step_rates on consecutive pairs == data.step_average_rates of the whole sequence (stride 2 records = 20 ms)."""
        from nedm.contact_nrd.data import step_average_rates
        raw = torch.as_tensor(_seq(40).astype(np.float32).astype(np.float64))
        rec = torch.zeros(1, 2 * len(raw) - 1, 2, 13, dtype=torch.float64)   # every second record = our grid state
        rec[0, ::2, 0, :12], rec[0, ::2, 1] = raw[:, :12], raw[:, 12:]
        rec[0, 1::2] = rec[0, :-1:2]                                            # odd records: anything (not on the grid)
        data = dict(states=rec, stride=2, dt=2 * 0.01, system=me._RATE_SYSTEM)
        st = step_average_rates(data, {"articulated", "rigid"})["states"][0, ::2]
        qd, v, w = me.step_rates(raw[:-1], raw[1:])
        assert torch.equal(qd, st[1:, 0, 6:12]) and torch.equal(v, st[1:, 1, 7:10])
        # spin: torch's float64 atan2 / norm can differ by an ulp with the batch layout (vector body vs scalar tail)
        assert (w - st[1:, 1, 10:13]).abs().max() <= 1e-12
        rot = (w * 0.02).norm(dim=-1)
        assert torch.allclose(rot, torch.full_like(rot, 0.04), atol=1e-6)   # 2 rad/s about a fixed axis


    def test_map_linear(self):
        """Trilinear fallback: exact on grid points, linear in between, clamped outside; equals scipy if present."""
        rng = np.random.default_rng(1)
        grid = rng.normal(size=(5, 6, 7)).astype(np.float32)
        idx = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0]]).T
        assert np.array_equal(me._map_linear(grid, idx.astype(float)), grid[tuple(idx)])
        mid = me._map_linear(grid, np.array([[1.5], [2.0], [3.0]]))
        assert np.isclose(mid[0], 0.5 * (grid[1, 2, 3] + grid[2, 2, 3]), atol=1e-6)
        assert np.array_equal(me._map_linear(grid, np.array([[-3.0], [9.0], [2.0]])), grid[[0], [5], [2]])
        if HAS_SCIPY:
            from scipy.ndimage import map_coordinates
            c = rng.uniform(-1, 8, size=(3, 5000))
            a, b = me._map_linear(grid, c), map_coordinates(grid, c, order=1, mode="nearest")
            assert np.abs(a.astype(float) - b).max() <= 1e-6 and (a == b).mean() > 0.95


    def test_pack_start_defaults(self):
        st = dict(q=np.ones((2, 6)), cube_pos=np.zeros((2, 3)), cube_quat=np.tile([1.0, 0, 0, 0], (2, 1)))
        rows = me.pack_start(st)
        assert rows.shape == (2, me.RAW) and (rows[:, 6:12] == 0).all() and (rows[:, 19:] == 0).all()


    @mj
    def test_forces_equal_collector(self):
        """Both lean force sums == Scene.contact_forces()[0][:7] bit for bit at every physics step of a scripted
        nominal grasp (collector plan, seed 5): cube-table, both jaw pairs."""
        import mujoco
        from nedm.contact_nrd.collect import so101_grasp as g
        blk = me._Block(1, overlap=False)
        s = blk.s
        s.d = blk.data[0]
        rng = np.random.default_rng(5)
        while True:
            try:
                plan = g.fam_nominal(s, rng, 301)
                break
            except g.PlanError:
                continue
        c = plan["cube"]
        s.reset(plan["q0"], c["pos"], c["quat"])
        seen = np.zeros(7, bool)
        for k, (target, _) in enumerate(plan["script"]):
            if k >= 150:   # 3 s: approach, close, lift
                break
            s.d.ctrl[s.act] = np.clip(target, s.tlo, s.thi).astype(np.float32)
            for _ in range(40):
                mujoco.mj_step(s.m, s.d)
                ref = g.Scene.contact_forces(s)[0][:7]
                assert np.array_equal(blk._forces(s.d), ref)
                assert np.array_equal(np.array(blk._forces_list(s.d)), ref)
                seen |= ref > s.thr
        assert seen[:3].all(), seen

    @mj
    def test_workers_deterministic_and_equal(self):
        """Same starts and commands: bit-identical states in-process and over 2 worker processes; run twice."""
        st = me.sample_starts([1011, 1012, 1013])
        rng = np.random.default_rng(0)
        cmds = st["q"][None] + np.cumsum(rng.normal(scale=0.01, size=(15, 3, 6)), 0)
        out = []
        for workers in (0, 2, 0):
            with me.MujocoWorlds(3, workers=workers, overlap=False) as env:
                env.reset(None, st, warmup=True)
                seq = [env.step(torch.as_tensor(c)) for c in cmds]
            out.append(torch.cat([torch.cat((s.q, s.qd, s.cube_pos, s.cube_quat, s.cube_v, s.cube_w, s.contact), -1) for s in seq]))
        assert torch.equal(out[0], out[1]) and torch.equal(out[0], out[2])


    @mj
    def test_record_labels_same_states(self):
        """labels="record" steps the same physics: bit-identical states; its contacts are a subset of the exact ones."""
        st = me.sample_starts([1004, 1005])
        cmds = st["q"][None] + np.linspace(0, 1, 30)[:, None, None] * np.array([0, 0.5, 0.4, 0, 0, 0])
        seqs = []
        for labels in ("exact", "record"):
            with me.MujocoWorlds(2, workers=0, overlap=False, labels=labels) as env:
                env.reset(None, st, warmup=True)
                seqs.append([env.step(torch.as_tensor(c)) for c in cmds])
        for a, b in zip(*seqs):
            for f in ("q", "qd", "cube_pos", "cube_quat", "cube_v", "cube_w"):
                assert torch.equal(getattr(a, f), getattr(b, f)), f
            assert not (b.contact & ~a.contact).any()

    @mj
    def test_reset_exact_and_set_cube(self):
        """Exact reset reproduces the given state (float32-rounded); set_cube moves the cube, keeps the arm fields
        (rates included), and the next step's cube rate starts from the new pose; the resting cube touches the table."""
        st = me.sample_starts([1006, 1007])
        with me.MujocoWorlds(2, workers=0, overlap=False) as env:
            ws = env.reset(None, st, warmup=True)
            assert ws.contact[:, 2].all() and not ws.contact[:, :2].any()
            assert not (ws.arm_table | ws.invalid).any()
            assert torch.allclose(ws.cube_pos[:, 2], torch.full((2,), me.REST_Z, dtype=torch.float64), atol=1e-9)
            ws2 = env.reset(None, ws, warmup=False)
            for f in ("q", "qd", "cube_pos", "cube_quat", "cube_v", "cube_w"):
                assert torch.equal(getattr(ws2, f), getattr(ws, f)), f
            u = torch.as_tensor(st["q"]) + torch.tensor([0.02, -0.03, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
            ws = env.step(u)                                                   # the arm moves: qd != 0
            new = ws.cube_pos.clone()
            new[:, 0] += 0.02
            ws3 = env.set_cube([0, 1], new, ws.cube_quat)
            assert torch.allclose(ws3.cube_pos, new.float().double(), atol=0) and (ws3.cube_v == 0).all()
            assert torch.equal(ws3.q, ws.q) and torch.equal(ws3.qd, ws.qd) and ws.qd.abs().max() > 0.1
            ws4 = env.step(u)
            assert (ws4.cube_pos - ws3.cube_pos).abs().max() < 1e-3        # resting where it was put, no jump in the rate
            assert ws4.cube_v.abs().max() < 0.05

    @mj
    def test_sample_starts(self):
        """Fresh starts: per-seed deterministic and order-free, on the collector annulus, flat at REST_Z; the env's
        warm-up reset reproduces ``state`` bit for bit (in-process and in a worker); reserved seeds raise."""
        seeds = [1000, 1001, 1002, 1003]
        a, b = me.sample_starts(seeds), me.sample_starts(seeds[::-1])
        for k in ("q", "cube_pos", "cube_quat", "raw"):
            assert np.array_equal(a[k], b[k][::-1]), k
        r = np.hypot(a["cube_pos"][:, 0] - 0.0388, a["cube_pos"][:, 1])
        assert ((r >= 0.15) & (r <= 0.27)).all() and (a["cube_quat"][:, 1:3] == 0).all()
        assert torch.allclose(a["state"].cube_pos[:, 2], torch.full((4,), me.REST_Z, dtype=torch.float64), atol=1e-7)
        for workers in (0, 2):
            with me.MujocoWorlds(4, workers=workers) as env:
                ws = env.reset(None, a, warmup=True)
            for f in ("q", "qd", "cube_pos", "cube_quat", "cube_v", "cube_w", "contact", "arm_table", "invalid"):
                assert torch.equal(getattr(ws, f), getattr(a["state"], f)), f
        with self.assertRaises(ValueError):
            me.sample_starts([1000, 20261034])                               # the sealed test's master seed

    @mj
    def test_contact_rules(self):
        """gate = physical label of this step OR the step before (reset: none before); records hold the two 10 ms
        states of the step (the end one = the state) and record_contact ORs to the physical label."""
        st = me.sample_starts([1008, 1009])
        cmds = st["q"][None] + np.linspace(0, 1, 40)[:, None, None] * np.array([0, 0.4, 0.35, 0, 0, 0])
        envs = [me.MujocoWorlds(2, workers=0, overlap=False, contact_rule=r) for r in ("gate", "physical")]
        try:
            for env in envs:
                env.reset(None, st, warmup=True)
            prev = torch.zeros(2, 3, dtype=torch.bool)
            for c in cmds:
                g, p = (env.step(torch.as_tensor(c)) for env in envs)
                assert torch.equal(p.contact, envs[1].physical_contact) and torch.equal(g.contact, p.contact | prev)
                assert torch.equal(envs[0].switch_contact, g.contact)
                assert torch.equal(envs[0].record_contact.any(1), p.contact)
                assert torch.equal(envs[0].records[:, 1], envs[0].raw_state()) and torch.equal(envs[0].records[:, 1, :6], g.q)
                prev = p.contact
        finally:
            for env in envs:
                env.close()

    @mj
    def test_end_conditions(self):
        """Gripper driven into the table -> arm_table; cube off the table -> invalid (cube_off_table), end_step set."""
        st = me.sample_starts([1002, 1003])
        with me.MujocoWorlds(2, workers=0, overlap=False) as env:
            ws = env.reset(None, st, warmup=True)
            pos = ws.cube_pos.clone()
            pos[1] = torch.tensor([0.9, 0.0, 0.015], dtype=torch.float64)    # beyond the table minus 5 cm
            env.set_cube([1], pos[1:], ws.cube_quat[1:])
            assert env.state().invalid[1] and env.end_reason[1] == me.end_names().index("cube_off_table")
            down = torch.as_tensor(st["q"]).clone()
            hit = False
            for k in range(150):                                              # shoulder lift forward: the arm sweeps down
                down[0, 1] = min(1.6, float(st["q"][0, 1]) + 0.03 * (k + 1))
                down[0, 2] = min(1.6, float(st["q"][0, 2]) + 0.02 * (k + 1))
                ws = env.step(down)
                hit |= bool(ws.arm_table[0])
            assert hit and env.end_reason[0] in (1, 2, 3, 4, 6)
            assert env.end_step[0] > 0


    @mj
    @OVERLAP
    @unittest.skipUnless((VAL / "unified_data.npz").exists(), "needs the validation extract (bench extract)")
    def test_replay_two_stored_episodes_bit_exact(self):
        """Two nominal validation episodes: regenerated collector start, stored commands -> the stored records: every
        field bit-identical except the cube spin (torch's float64 atan2 / norm kernels differ by an ulp with the tensor
        layout, also inside the data pipeline itself: <= 1e-12 rad/s), contacts, no end."""
        import importlib.util as iu
        spec = iu.spec_from_file_location("bench", Path(__file__).parents[2] / "scripts/so101_policy/bench_mujoco_env.py")
        bench = iu.module_from_spec(spec)
        spec.loader.exec_module(bench)
        from nedm.contact_nrd.rollouts import npz_members
        a = npz_members(VAL / "unified_data.npz")
        fam, term = np.asarray(a["family"]), np.asarray(a["terminal_reason"])
        rows = np.flatnonzero((fam == 0) & (term == 0))[:2]
        bench._plan_init()
        plans = [bench.regenerate((int(fam[e]), int(a["family_index"][e]), int(a["seed"][e]), np.array(a["arm"][e, 0]),
                                   np.array(a["cube"][e, 0]))) for e in rows]
        truth, phys, actions, lengths = bench._truth(VAL, rows)
        with me.MujocoWorlds(2, workers=0) as env:
            start = dict(q=np.stack([p["q"] for p in plans]), cube_pos=np.stack([p["pos"] for p in plans]),
                         cube_quat=np.stack([p["quat"] for p in plans]))
            seq, er, es, _ = bench._run(env, actions, 300, lambda: env.reset(None, start, warmup=True))
        res = bench.compare(seq, er, es, truth, phys, lengths, term[rows])
        assert all(r["exact_1e12"] for r in res), res
        for r in res:
            assert all(v == 0.0 for k, v in r["max_abs"].items() if k != "cube_w"), r


    @unittest.skipUnless(CKPT.exists(), "needs the NRD checkpoint")
    def test_nrd_state_round_trip(self):
        """WorldState -> NRD layout (finish fills FK + relative channels) -> WorldState."""
        from nedm.contact_nrd.model import load_model
        model, _ = load_model(CKPT)
        raw = torch.as_tensor(_seq(5))
        f = torch.zeros(5, dtype=torch.bool)
        ws = WorldState(q=raw[:, 0:6], qd=raw[:, 6:12], cube_pos=raw[:, 12:15], cube_quat=raw[:, 15:19],
                        cube_v=raw[:, 19:22], cube_w=raw[:, 22:25], contact=torch.zeros(5, 3, dtype=torch.bool),
                        arm_table=f, invalid=f)
        x = me.nrd_state(ws, model)
        assert x.shape == (5, model.n_bodies, model.smax)
        assert torch.equal(x, model.finish(x))                                  # already finished
        back = me.world_from_nrd(x, model)
        for k in ("q", "qd", "cube_pos", "cube_v", "cube_w"):
            assert torch.equal(getattr(back, k), getattr(ws, k)), k
        assert torch.allclose(back.cube_quat, ws.cube_quat, atol=1e-15)
        names = [b for b in model.system["bodies"] if b["moving"]][0]["channels"]
        fz = x[:, 0, names.index("fk_z")]
        assert (fz.abs() < 1).all() and (x[:, 0, names.index("fk_x")] != 0).all()
        if not (VAL / "unified_data.npz").exists():
            return
        # the NRD evaluation's states of stored episodes (data pipeline) from their base channels alone
        from nedm.contact_nrd.data import add_derived, add_relative, step_average_rates
        from nedm.contact_nrd.rollouts import load_rows
        d = load_rows(VAL, [0, 50, 100], "cpu", 0.02)
        step_average_rates(d, set(model.config["integrate"]["kinds"]))
        add_derived(d, model.config.get("derived"))
        add_relative(d, model.config.get("relative"))
        st = d["states"][:, ::2].reshape(-1, 2, model.smax)
        n = len(st)
        f = torch.zeros(n, dtype=torch.bool)
        ws = WorldState(q=st[:, 0, 0:6], qd=st[:, 0, 6:12], cube_pos=st[:, 1, 0:3], cube_quat=st[:, 1, 3:7],
                        cube_v=st[:, 1, 7:10], cube_w=st[:, 1, 10:13], contact=torch.zeros(n, 3, dtype=torch.bool),
                        arm_table=f, invalid=f)
        y = me.nrd_state(ws, model)
        assert torch.equal(y[:, 0], st[:, 0])                                # arm incl. FK: bit-identical
        # cube: finish re-normalises the float32-stored quaternion (|q| - 1 up to 4e-8), the data pipeline does not
        assert (y[:, 1] - st[:, 1]).abs().max() < 3e-7


if __name__ == "__main__":
    unittest.main()
