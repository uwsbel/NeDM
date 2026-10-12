"""nedm.so101_policy.nrd_sim.GraspNrdSim on CPU (seconds): small random models in the two state layouts (with the
derived + relative channels of the final checkpoints, and without them as an "ns" model would have), and the real
checkpoint on recorded validation windows when it is on this machine. The CUDA-graph test needs a GPU.

    PYTHONPATH=src python -m unittest tests.so101_policy.test_nrd_sim -v
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch

from nedm.contact_nrd.model import (DERIVED_NAMES, DERIVED_TYPES, RELATIVE_NAMES, RELATIVE_TYPES, ContactNRD,
                                    load_model, model_config)
from nedm.so101_policy import cube_symmetry as cs
from nedm.so101_policy.nrd_sim import STUDY_CKPTS, GraspNrdSim, cube_rest_envelope, lowest_point_guard
from nedm.so101_policy.world import CONTACT_PAIRS, WorldState
from tests.contact_nrd import synthetic
from tests.contact_nrd.test_conventions import KINDS, SMALL, full_masks

VAL = Path.home() / "so101_nrd_fix" / "data" / "val_only"
# a "ns" model of the fix study (no derived / relative channels; read only)
NS_CKPT = Path.home() / "so101_nrd_fix" / "design" / "signals" / "ckpt" / "short_ns_rk_cc_s62" / "best.pt"
LINKS = [dict(pos=[0.0388353, 0.0, 0.0624], quat=[0.0, 0.0, -1.0, 0.0]),
         dict(pos=[-0.0303992, -0.0182778, -0.0542], quat=[0.5, -0.5, -0.5, -0.5]),
         dict(pos=[-0.11257, -0.028, 0.0], quat=[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]),
         dict(pos=[-0.1349, 0.0052, 0.0], quat=[math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)]),
         dict(pos=[0.0, -0.0611, 0.0181], quat=[0.017209108230571014, -0.017209108230571014, 0.7068973380865913, 0.7068973380865913])]


_THREADS = []


def setUpModule():   # small batches: one thread is fastest and avoids oversubscription on a loaded machine
    _THREADS.append(torch.get_num_threads())
    torch.set_num_threads(1)


def tearDownModule():
    torch.set_num_threads(_THREADS.pop())


def so101_model(seed=0, derived=True):
    """A small random model with the SO101 grasp layout (synthetic system; derived + relative channels optional)."""
    system = synthetic.so101_system(101)
    arm, cube, table = system["bodies"]
    table = {**table, "size_xy_m": [1.2, 1.2], "center_xy_m": [0.25, 0.0]}
    kw = {}
    if derived:
        arm = {**arm, "state_dim": 27, "channels": arm["channels"] + DERIVED_NAMES, "channel_types": arm["channel_types"] + DERIVED_TYPES}
        cube = {**cube, "state_dim": 25, "channels": cube["channels"] + RELATIVE_NAMES,
                "channel_types": cube["channel_types"] + RELATIVE_TYPES, "channel_mask": [1] * 25}
        kw = dict(derived=dict(body="arm", links=LINKS, point_m=[0.0105, 0.0, -0.092]), relative=dict(body="cube", frame="arm"))
    system = {**system, "bodies": [arm, cube, table]}
    torch.manual_seed(seed)
    model = ContactNRD(model_config(0.02, 4, SMALL, KINDS, **kw), system, full_masks(system))
    for net in (model.core, model.collision, model.contact):
        for head in net.heads:
            torch.nn.init.normal_(head.weight, std=0.02)
    with torch.no_grad():
        model.collision.out_mean.fill_(0.2)                                 # some switches on
    return model.eval()


def starts(n, seed=0, yaw_deg=None):
    g = torch.Generator().manual_seed(seed)
    q = torch.zeros(n, 6, dtype=torch.float64)
    q[:, 1:4] = torch.tensor([0.1, 0.2, 1.3], dtype=torch.float64) + 0.1 * torch.rand(n, 3, generator=g, dtype=torch.float64)
    q[:, 5] = 0.6
    yaw = (torch.rand(n, generator=g, dtype=torch.float64) * math.pi / 2) if yaw_deg is None else torch.full((n,), math.radians(yaw_deg), dtype=torch.float64)
    quat = torch.stack((torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)), -1)
    pos = torch.stack((0.2 + 0.05 * torch.rand(n, generator=g, dtype=torch.float64), 0.05 * torch.randn(n, generator=g, dtype=torch.float64),
                       torch.full((n,), 0.015, dtype=torch.float64)), -1)
    return dict(q=q, cube_pos=pos, cube_quat=quat)


def same_rotation(a, b):
    """Max over rows of 1 - |<a, b>| (0 for the same rotation, either sign)."""
    return float((1 - (a * b).sum(-1).abs()).abs().max())


def targets(n, steps, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = starts(n, seed)["q"]
    t = torch.arange(steps, dtype=torch.float64)[None, :, None] * 0.02
    return base[:, None] + 0.05 * torch.sin(3 * t + torch.rand(n, 1, 6, generator=g, dtype=torch.float64) * 6)


class Layout(unittest.TestCase):
    def test_finish_and_assemble_equal_model(self):
        for derived in (True, False):
            sim = GraspNrdSim({"a": so101_model(0, derived)}, 4, "cpu")   # the constructor's self-check raises otherwise
            self.assertEqual(sim.S, 27 if derived else 13)


    def test_fallback_to_model_finish(self):
        """A model whose finish differs from the sim's copy: the sim warns and uses the model's own finish."""
        model = so101_model(14)

        def finish(x, m=model):
            out = ContactNRD.finish(m, x)
            out[:, 1, 2] = out[:, 1, 2].clamp_min(0.02)                   # a rule the sim's copy does not know
            return out
        model.finish = finish
        with self.assertWarns(UserWarning):
            sim = GraspNrdSim({"a": model}, 4, "cpu", cuda_graph=True)
        self.assertFalse(sim.fast_layout or sim.use_graph)
        sim.reset(None, starts(4, 14))
        sim.step(starts(4, 14)["q"])
        self.assertTrue(bool((sim.state[:, 1, 2] >= 0.02).all()))


class DenseStep(unittest.TestCase):
    def test_parity_with_rollout(self):
        """canonical=False: the dense step and ContactNRD.rollout (sparse contact step) give the same states."""
        for derived in (True, False):
            model = so101_model(1, derived)
            sim = GraspNrdSim({"a": model}, 16, "cpu", canonical=False)
            sim.reset(None, starts(16, 1))
            x0 = sim.state.clone()
            tg = targets(16, 25, 1)
            act = torch.zeros(16, 25, model.n_bodies, model.amax, dtype=torch.float64)
            act[:, :, 0, :6] = tg
            with torch.no_grad():
                ref, gates = model.rollout(x0, act, return_gates=True)
            got, sw = [x0], []
            for t in range(25):
                sim.step(tg[:, t])
                got.append(sim.state.clone())
                sw.append(sim.gates.clone())
            self.assertLess(float((torch.stack(got, 1) - ref).abs().max()), 1e-6)
            self.assertTrue(torch.equal(torch.stack(sw, 1), gates))
            self.assertGreater(float(gates.mean()), 0.05)                     # the contact network was exercised

    def test_both_bodies_get_jaw_contact(self):
        """A jaw pair (arm, cube) writes its change to both bodies: switching it off changes both."""
        model = so101_model(2)
        sim = GraspNrdSim({"a": model}, 8, "cpu", canonical=False)
        sim.reset(None, starts(8, 2))
        tg = targets(8, 1, 2)[:, 0]
        with torch.no_grad():
            model.collision.out_mean.fill_(5.0)
            sim.step(tg)
            on = sim.state.clone()
            sim.reset(None, starts(8, 2))
            model.collision.out_mean[:, 0:2] = -5.0                         # jaw pairs off, table pair on
            sim.step(tg)
            off = sim.state.clone()
        self.assertGreater(float((on[:, 0, 6:12] - off[:, 0, 6:12]).abs().max()), 1e-6)   # arm rates
        self.assertGreater(float((on[:, 1, 7:13] - off[:, 1, 7:13]).abs().max()), 1e-6)   # cube rates

    def test_members_split(self):
        a, b = so101_model(3), so101_model(4)
        sim = GraspNrdSim({"a": a, "b": b}, 6, "cpu", canonical=False)
        self.assertEqual(sim.member_of_world.tolist(), [0, 0, 0, 1, 1, 1])
        st = starts(6, 3)
        sim.reset(None, st)
        tg = targets(6, 5, 3)
        for t in range(5):
            sim.step(tg[:, t])
        for m, sl in ((a, slice(0, 3)), (b, slice(3, 6))):
            one = GraspNrdSim({"x": m}, 3, "cpu", canonical=False)
            one.reset(None, {k: v[sl] for k, v in st.items()})
            for t in range(5):
                one.step(tg[sl, t])
            self.assertLess(float((one.state - sim.state[sl]).abs().max()), 1e-6)
        only = GraspNrdSim({"s61x": a, "s62x": b}, 4, "cpu", members="62x")
        self.assertEqual(only.member_names, ["s62x"])


class Canonicalisation(unittest.TestCase):
    def test_relabelled_starts_give_the_same_physics(self):
        """The same physical start with another cube labelling: same positions, true quaternions differ by the label."""
        model = so101_model(5)
        st = starts(24, 5)
        turned = dict(st, cube_quat=cs.relabel(st["cube_quat"], torch.arange(24), newest_sign=False))
        tg = targets(24, 30, 5)
        runs = []
        for s in (st, turned):
            sim = GraspNrdSim({"a": model}, 24, "cpu", canonical=True)
            sim.reset(None, s)
            ws = [sim.step(tg[:, t]) for t in range(30)]
            runs.append(ws)
        for wa, wb in zip(*runs):
            self.assertLess(float((wa.cube_pos - wb.cube_pos).abs().max()), 1e-9)
            rel = cs.qmul(cs.qconj(wa.cube_quat), wb.cube_quat)              # = G[k] up to sign
            self.assertLess(float(((rel * cs.group()[torch.arange(24)]).sum(-1).abs() - 1).abs().max()), 1e-9)

    def test_true_orientation_and_bookkeeping(self):
        model = so101_model(6)
        sim = GraspNrdSim({"a": model}, 3, "cpu", canonical=True)
        st = starts(3, 6)
        st["cube_quat"] = torch.stack([cs.axis_quat("z", d) for d in (30.0, 100.0, 200.0)])
        sim.reset(None, st)
        ws = sim.observe()
        self.assertLess(same_rotation(ws.cube_quat, st["cube_quat"]), 1e-12)               # true orientation out
        self.assertTrue(bool((ws.cube_quat[:, 0] >= 0).all()))                              # w >= 0 at a reset
        inner = sim.state[:, 1, 3:7]
        yaw = torch.rad2deg(cs.yaw_of_flat(inner))
        self.assertTrue(bool(((yaw > 0) & (yaw < 90)).all()))                               # canonical inside
        self.assertEqual(sim.canon.tolist()[0], 0)
        q_true = cs.true_from_canonical(inner, sim.canon)
        self.assertLess(same_rotation(q_true, ws.cube_quat), 1e-12)
        prev = ws.cube_quat
        for t in range(10):                                                 # sign-continuous true quaternion
            ws = sim.step(st["q"])
            self.assertTrue(bool(((ws.cube_quat * prev).sum(-1) > 0).all()))
            prev = ws.cube_quat

    def test_off_is_plain_rollout(self):
        model = so101_model(7)
        sim = GraspNrdSim({"a": model}, 2, "cpu", canonical=False)
        st = starts(2, 7)
        st["cube_quat"] = torch.stack([cs.axis_quat("z", d) for d in (100.0, 200.0)])
        sim.reset(None, st)
        self.assertLess(same_rotation(sim.state[:, 1, 3:7], st["cube_quat"]), 1e-12)
        self.assertLess(float((sim.state[0, 1, 3:7] - st["cube_quat"][0]).abs().max()), 1e-12)   # w > 0: as given
        self.assertEqual(sim.canon.tolist(), [0, 0])


class ResetsAndDisturbances(unittest.TestCase):
    def test_reset_from_worldstate(self):
        model = so101_model(8)
        sim = GraspNrdSim({"a": model}, 4, "cpu")
        st = starts(4, 8)
        st["cube_quat"] = -st["cube_quat"]                                  # w < 0 in: normalised to w >= 0
        sim.reset(torch.tensor([1, 3]), {k: v[[1, 3]] for k, v in st.items()})
        self.assertEqual(sim.ok[1].tolist(), [False, False, False, True])
        self.assertTrue(torch.equal(sim.hist[1, :3], torch.zeros_like(sim.hist[1, :3])))
        self.assertLess(float((sim.state[[1, 3], 0, 0:6] - st["q"][[1, 3]]).abs().max()), 1e-15)
        self.assertTrue(torch.equal(sim.act[[1, 3], 0, :6], st["q"][[1, 3]]))
        self.assertTrue(bool((sim.true_q[[1, 3], 0] >= 0).all()))
        self.assertTrue(torch.equal(sim.state[[1, 3]], model.finish(sim.state[[1, 3]])))      # derived channels filled
        ws = WorldState(q=st["q"], qd=torch.zeros(4, 6, dtype=torch.float64), cube_pos=st["cube_pos"],
                        cube_quat=st["cube_quat"], cube_v=torch.zeros(4, 3, dtype=torch.float64),
                        cube_w=torch.zeros(4, 3, dtype=torch.float64), contact=None, arm_table=None, invalid=None)
        sim.reset(None, ws)
        self.assertFalse(bool(sim.observe().invalid.any()))

    def test_set_cube_rebuilds_history(self):
        """Review sec. 5: the cube base channels in ALL K slots (v = w = 0), arm channels and ok kept, finish on each slot."""
        model = so101_model(9)
        sim = GraspNrdSim({"a": model}, 4, "cpu", canonical=True)
        sim.reset(None, starts(4, 9))
        for t in range(2):                                                   # 3 valid entries, 1 masked
            sim.step(starts(4, 9)["q"])
        arm_before, ok_before = sim.hist[:, :, 0].clone(), sim.ok.clone()
        pos = torch.tensor([[0.25, 0.05, 0.014993]] * 4, dtype=torch.float64)
        quat = torch.stack([cs.axis_quat("z", d) for d in (10.0, 80.0, 120.0, 250.0)])
        ws = sim.set_cube(None, pos, quat)
        h = sim.hist
        self.assertIsInstance(ws, WorldState)
        self.assertTrue(torch.equal(sim.ok, ok_before))                      # masks kept
        self.assertTrue(torch.equal(h[:, :, 0, :12], arm_before[:, :, :12]))  # arm base channels kept, every slot
        self.assertLess(float((h[:, 1:, 0] - arm_before[:, 1:]).abs().max()), 1e-15)   # valid slots: derived too
        for j in range(4):                                                    # every slot: the same resting cube
            self.assertLess(float((h[:, j, 1, 0:3] - pos).abs().max()), 1e-15)
            self.assertEqual(float(h[:, j, 1, 7:13].abs().max()), 0.0)
            self.assertTrue(torch.equal(h[:, j, 1, 3:7], h[:, -1, 1, 3:7]))
            self.assertLess(float((model.finish(h[:, j]) - h[:, j]).abs().max()), 1e-15)   # finished slot by slot
        self.assertLess(same_rotation(ws.cube_quat, quat), 1e-12)
        self.assertLess(float((ws.cube_pos - pos).abs().max()), 1e-15)
        yaw = torch.rad2deg(cs.yaw_of_flat(h[:, -1, 1, 3:7]))
        self.assertTrue(bool(((yaw >= 0) & (yaw <= 90)).all()))              # canonical inside
        self.assertTrue(bool(cube_rest_envelope(pos, quat).all()))
        sim.step(starts(4, 9)["q"])                                          # and it steps on
        self.assertTrue(bool(torch.isfinite(sim.hist).all()))

    def test_set_cube_with_rates(self):
        """Nonzero v, w (outside the data): constant-rate history that obeys the model's integration rule."""
        model = so101_model(9)
        sim = GraspNrdSim({"a": model}, 4, "cpu", canonical=False)
        sim.reset(None, starts(4, 9))
        sim.step(starts(4, 9)["q"])
        pos = torch.tensor([[0.25, 0.05, 0.015]] * 4, dtype=torch.float64)
        quat = torch.stack([cs.axis_quat("z", d) for d in (10.0, 80.0, 120.0, 250.0)])
        v = torch.tensor([[0.1, -0.05, 0.0]] * 4, dtype=torch.float64)
        w = torch.tensor([[0.0, 0.0, 2.0]] * 4, dtype=torch.float64)
        sim.set_cube(None, pos, quat, v, w)
        h = sim.hist
        self.assertLess(same_rotation(sim.observe().cube_quat, quat), 1e-12)
        for j in range(3):
            self.assertLess(float((h[:, j + 1, 1, 0:3] - h[:, j, 1, 0:3] - 0.02 * v).abs().max()), 1e-15)
            nxt = model.assemble(h[:, j], h[:, j + 1] - h[:, j], torch.zeros_like(h[:, j]))
            self.assertLess(float((nxt[:, 1, 3:7] - h[:, j + 1, 1, 3:7]).abs().max()), 1e-12)

    def test_relabel_refinish_equals_finish(self):
        """After steps with relabels every slot is a finished state, and the cube-only refinish was used."""
        for derived in (True, False):
            model = so101_model(15, derived)
            sim = GraspNrdSim({"a": model}, 24, "cpu", canonical=True)
            self.assertTrue(sim.fast_relabel)
            st = starts(24, 15)
            st["cube_quat"] = cs.relabel(st["cube_quat"], torch.arange(24), newest_sign=False)
            sim.reset(None, st)
            tg = targets(24, 6, 15)
            sim.step(tg[:, 0])
            ws = sim.observe()                                               # spin about z: the label band is crossed
            sim.set_cube(None, ws.cube_pos, cs.qmul(cs.axis_quat("z", 1.0), ws.cube_quat), w=torch.tensor([[0.0, 0.0, -3.0]] * 24))
            n = 0
            for t in range(1, 6):
                sim.step(tg[:, t])
                n += int(sim.relabelled.sum())
            self.assertGreater(n, 0)
            fin = model.finish(sim.hist.flatten(0, 1)).view_as(sim.hist)
            valid = sim.ok[..., None, None].expand_as(fin)
            self.assertLess(float((fin - sim.hist)[valid].abs().max()), 1e-15)


class Helpers(unittest.TestCase):
    def test_lowest_point_guard(self):
        class Kin:
            def lowest_point(self, q):
                return q[:, 1] * 0.01
        fn = lowest_point_guard(Kin())
        q = torch.zeros(3, 6, dtype=torch.float64)
        q[:, 1] = torch.tensor([-1.0, 0.04, 0.06], dtype=torch.float64)          # -10 mm, 0.4 mm, 0.6 mm
        self.assertEqual(fn(q).tolist(), [True, True, False])
        model = so101_model(16)
        sim = GraspNrdSim({"a": model}, 3, "cpu", arm_table_fn=fn)
        st = starts(3, 16)
        st["q"][:, 1] = q[:, 1]
        self.assertEqual(sim.reset(None, st).arm_table.tolist(), [True, True, False])

    def test_cube_rest_envelope(self):
        flat = cs.axis_quat("z", 30.0)
        pos = torch.tensor([[0.25, 0.0, 0.014993], [0.25, 0.0, 0.0165], [0.0388 + 0.14, 0.0, 0.014993],
                            [0.0388 + 0.2 * math.cos(math.radians(80)), 0.2 * math.sin(math.radians(80)), 0.014993],
                            [0.25, 0.0, 0.014993], [0.25, 0.0, 0.014993]], dtype=torch.float64)
        quat = torch.stack([flat] * 4 + [cs.qmul(cs.axis_quat("x", 2.0), flat), flat])
        v = torch.zeros(6, 3, dtype=torch.float64)
        v[5, 0] = 0.01
        self.assertEqual(cube_rest_envelope(pos, quat, v).tolist(), [True, False, False, False, False, False])


class Flags(unittest.TestCase):
    def test_contact_rules_and_arm_table(self):
        model = so101_model(10)
        order = [model.system["pair_metadata"][k]["name"] for k in range(3)]
        perm = [order.index(n) for n in CONTACT_PAIRS]
        tg = targets(8, 12, 10)
        sims = {r: GraspNrdSim({"a": model}, 8, "cpu", contact_rule=r) for r in ("gate", "physical")}
        calls = []
        sims["gate"].arm_table_fn = lambda q: (calls.append(q.shape), q[:, 0] > 0)[1]
        for s in sims.values():
            s.reset(None, starts(8, 10))
        prev = torch.zeros(8, 3, dtype=torch.bool)
        for t in range(12):
            wg, wp = (sims[r].step(tg[:, t]) for r in ("gate", "physical"))
            g = sims["gate"].gates.bool()[:, perm]
            self.assertTrue(torch.equal(wg.contact, g))
            self.assertTrue(torch.equal(wp.contact, g & prev))
            self.assertTrue(torch.equal(wg.arm_table, wg.q[:, 0] > 0))
            self.assertFalse(bool(wp.arm_table.any()))                      # default: never
            prev = g
        self.assertEqual(calls[0], (8, 6))

    def test_targets_applied_like_mujoco(self):
        """review F3: the step applies float32(clip(target, kin.TLO, kin.THI)), as MuJoCo and the data."""
        from nedm.so101_policy.kin import THI, TLO
        model = so101_model(12)
        tg = targets(4, 6, 12)
        wild = tg.clone()
        wild[:, :, 5] = -0.3                                                  # below the jaw's applied range
        wild[:, :, 0] = 2.5                                                   # above the pan's
        lo, hi = torch.tensor(TLO, dtype=torch.float64), torch.tensor(THI, dtype=torch.float64)
        applied = torch.minimum(torch.maximum(wild, lo), hi).float().double()
        out = []
        for u in (wild, applied):
            sim = GraspNrdSim({"a": model}, 4, "cpu")
            sim.reset(None, starts(4, 12))
            for t in range(6):
                sim.step(u[:, t])
            out.append(sim.state.clone())
            self.assertTrue(torch.equal(sim.act[:, sim.ia, :6], applied[:, -1]))
        self.assertTrue(torch.equal(out[0], out[1]))                          # (applied: in-range values float32 too)

    def test_invalid(self):
        model = so101_model(11)
        sim = GraspNrdSim({"a": model}, 6, "cpu")
        st = starts(6, 11)
        st["cube_pos"][1, 0] = 0.81                                         # off the table (edge margin)
        st["cube_pos"][2, 2] = 0.005                                        # sunk 10 mm
        st["cube_pos"][3, 2] = 0.5                                          # absurd height
        st["cube_pos"][4, 2] = float("nan")
        st["cube_v"] = torch.zeros(6, 3, dtype=torch.float64)
        st["cube_v"][5, 0] = 5.0                                            # absurd speed
        sim.reset(None, st)
        self.assertEqual(sim.observe().invalid.tolist(), [False, True, True, True, True, True])


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class Graph(unittest.TestCase):
    def test_graph_equals_eager(self):
        model = so101_model(12)
        tg = targets(32, 20, 12).cuda()
        out = []
        for graph in (False, True):
            sim = GraspNrdSim({"a": model, "b": so101_model(13)}, 32, "cuda", cuda_graph=graph)
            sim.reset(None, {k: v.cuda() for k, v in starts(32, 12).items()})
            ws = [sim.step(tg[:, t]) for t in range(10)]
            sim.reset(torch.arange(0, 32, 2, device="cuda"), {k: v[::2].cuda() for k, v in starts(32, 14).items()})
            sim.set_cube(torch.tensor([1, 5], device="cuda"), torch.tensor([[0.2, 0.0, 0.015]] * 2), cs.axis_quat("z", 120.0).expand(2, 4))
            ws += [sim.step(tg[:, t]) for t in range(10, 20)]
            out.append(torch.stack([torch.cat((w.q, w.cube_pos, w.cube_quat), -1) for w in ws]))
        self.assertLess(float((out[0] - out[1]).abs().max()), 1e-9)


def val_windows(model, rows, steps=(60, 120)):
    """Recorded 4-state validation histories (model layout) at the given 20 ms steps and the recorded commands."""
    from nedm.contact_nrd import rollouts as R
    data = R.load_rows(VAL, rows, "cpu", 0.02)
    R.step_average_rates(data, set(model.config["integrate"]["kinds"]))
    R.add_derived(data, model.config.get("derived"))
    R.add_relative(data, model.config.get("relative"))
    h = torch.cat([data["states"][:, 2 * s - 6: 2 * s + 1: 2] for s in steps])
    a = torch.cat([data["actions"][:, 2 * s] for s in steps])
    return h, torch.ones(len(h), model.k, dtype=torch.bool), a


def window_parity(test, model, name):
    h, ok, a = val_windows(model, list(range(0, 1200, 50)))
    with torch.no_grad():
        ref, info = model.details(h, ok, a)
    sim = GraspNrdSim({name: model}, len(h), "cpu", canonical=False)
    test.assertTrue(sim.fast_layout and sim.fast_relabel)
    sim.reset_history(None, h, ok)
    sim.step(a[:, 0, :6])
    test.assertLess(float((sim.state - ref).abs().max()), 1e-6)
    test.assertTrue(torch.equal(sim.gates, info["gates"]))
    sim = GraspNrdSim({name: model}, len(h), "cpu", canonical=True)        # canonical, from a WorldState: runs
    s0 = h[:, -1]
    ws = sim.reset(None, dict(q=s0[:, 0, 0:6], qd=s0[:, 0, 6:12], cube_pos=s0[:, 1, 0:3], cube_quat=s0[:, 1, 3:7],
                              cube_v=s0[:, 1, 7:10], cube_w=s0[:, 1, 10:13]))
    test.assertLess(same_rotation(ws.cube_quat, s0[:, 1, 3:7] / s0[:, 1, 3:7].norm(dim=-1, keepdim=True)), 1e-12)
    for t in range(2):
        ws = sim.step(a[:, 0, :6])
    test.assertTrue(bool(torch.isfinite(sim.hist).all()))
    return sim


@unittest.skipUnless(all(Path(p).exists() for p in STUDY_CKPTS.values()) and (VAL / "unified_data.npz").exists(),
                     "needs the study checkpoints and the validation data")
class RealCheckpoint(unittest.TestCase):
    def test_windows_parity(self):
        """One dense step from 48 recorded validation windows of the final s61 model = ContactNRD.details."""
        model, _ = load_model(STUDY_CKPTS["s61"], "cpu")
        sim = window_parity(self, model, "s61")
        self.assertEqual(sim.S, 27)


@unittest.skipUnless(NS_CKPT.exists() and (VAL / "unified_data.npz").exists(), "needs the ns checkpoint and the validation data")
class OtherLayoutCheckpoint(unittest.TestCase):
    def test_no_derived_channels(self):
        """A real checkpoint without derived and relative channels (12 + 13 state channels): layout read from the model."""
        model, _ = load_model(NS_CKPT, "cpu")
        self.assertIsNone(model.kin)
        sim = window_parity(self, model, "ns")
        self.assertEqual(sim.S, 13)


if __name__ == "__main__":
    unittest.main()
