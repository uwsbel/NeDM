"""Task env (src/nedm/so101_policy/vec_env.py) in the kinematic world on the CPU: observation layout, the decision =
5 task steps (reward sum), action masks, time-out / final observation / auto-reset, per-episode decoder seeds, the task
pool, the task charge discount, start banks and their seed guard (review R9), and (if the gate files exist) the FSM
expert reproducing the gate's kinematic-world episodes exactly. Decoder PPO options (build4_ppo2.md): the brake rest
point = macro_goal for every blend (R4), the GRASP gate, the GRASP latch and the abort action (R3), arming only where
a[4] is read (R5), the grasp-pose shaping (potential telescopes), the evaluation env rebuilt from a checkpoint's
decoder record (R1) and the resolved configs in it (R6).

CPU, about a minute (skips without the task geometry: ~/so101_policy/ckpt/task_geom_v2.npz or the vendor model).

    PYTHONPATH=src python -m unittest tests.so101_policy.test_vec_env -v
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import task as T
from nedm.so101_policy import vec_env as V
from nedm.so101_policy.decoder import M

torch.set_num_threads(2)
GATE = Path.home() / "so101_policy" / "runs" / "gate"
FRESH = Path.home() / "so101_policy" / "runs" / "ppo" / "starts_fresh_200000_256.npz"


def bank_from_tasks():
    """A small TRAINING start bank: fresh collector starts (seeds 200000+, ``train_ppo.py starts``) if present, else
    None (the gate tasks are validation starts: refused as a bank since review R9)."""
    return FRESH if FRESH.exists() and (GATE / "tasks_1024.npz").exists() else None


def have_geom():
    if V.GEOM_NPZ.exists():
        return True
    try:
        import mujoco  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(have_geom() and bank_from_tasks() is not None, "needs the task geometry and runs/gate/tasks_1024.npz")
class KinEnv(unittest.TestCase):
    def make(self, B=6, **kw):
        cfg = dict(world="kin", num_envs=B, device="cpu", seed=3, starts=str(bank_from_tasks()), pool_size=64)
        cfg.update(kw)
        return V.PickPlaceVecEnv(cfg)

    def test_layout_and_dims(self):
        env = self.make()
        obs = env.reset()
        self.assertEqual(tuple(obs.shape), (6, T.obs_dim(env.tcfg, 24)))
        self.assertEqual(env.num_obs, 79)
        F = T.features(env.ws, env.kin, None, env.tcfg)
        F2 = dict(F)
        F2["v"] = F["v"] + 1.0
        F2["w"] = F["w"] + 2.0
        F2["xg"] = F["xg"] + 3.0
        args = (env.task.goal_xy, env.task.goal_yaw, env.task.tr.lifted, env.task.tr.air_run, env.tcfg,
                env.dec.observation(), env.dec.macro())
        d = T.observation(F2, *args) - T.observation(F, *args)
        L = env.layout
        self.assertTrue(torch.allclose(d[:, L["cube_v"]], torch.ones(6, 3)))
        self.assertTrue(torch.allclose(d[:, L["cube_w"]], torch.full((6, 3), 2.0)))
        self.assertTrue(torch.allclose(d[:, L["cube_in_grasp"]], torch.full((6, 3), 3.0)))
        rest = torch.ones(env.num_obs, dtype=torch.bool)
        for k in ("cube_v", "cube_w", "cube_in_grasp"):
            rest[L[k]] = False
        self.assertEqual(float(d[:, rest].abs().max()), 0.0)
        self.assertEqual(L["decoder"].stop, env.num_obs)
        self.assertAlmostEqual(env.tcfg.gamma, 0.99 ** 0.2, places=12)

    def test_reward_is_sum_of_task_steps_and_masks(self):
        a = self.make()
        b = self.make()
        a.reset()
        b.reset()
        g = torch.Generator().manual_seed(0)
        for k in range(4):
            act = torch.tanh(0.3 * torch.randn(6, 6, generator=g))
            act[:, 4] = -1.0
            if k == 2:
                act[:, 4] = 1.0                       # trigger GRASP everywhere (training rule: box check)
            mode0 = b.dec.mode.clone()
            info = b.dec.decide(act, b.ws)
            r = torch.zeros(6, dtype=torch.float64)
            for _ in range(5):
                u, _ = b.dec.step(b.ws.q)
                b.ws = b.world.step(u)
                rr, _, _, _ = b.task.step(b.ws, b.dec.macro(), b.dec.macro_goal)
                r += rr
            _, rew, done, ex = a.step(act, squashed=True)
            self.assertLess(float((rew.double() - r).abs().max()), 1e-6)
            m = ex["action_mask"]
            take = (mode0 == M["free"]) | (mode0 == M["hold"])
            self.assertTrue(torch.equal(m[:, 5], info["grasp_started"]))
            self.assertTrue(torch.equal(m[:, 4], take | ((mode0 >= M["g_brake"]) & (mode0 <= M["g_settle"]))))
            self.assertTrue(torch.equal(m[:, 0], (take | info["aborted"]) & ~info["grasp_started"] & ~info["place_started"]))
            if k == 2:
                self.assertTrue(bool(info["grasp_started"].any()))       # the trigger decision is exercised
                self.assertFalse(bool(m[info["grasp_started"], :4].any()))
            if bool(done.any()):
                break
            # keep b in step with a's resets (none expected in 4 decisions unless a rule ended a world)

    def test_carry_macro_mask(self):
        from nedm.so101_policy.decoder import DecoderV2Config
        if not hasattr(DecoderV2Config(), "carry_macro"):
            self.skipTest("decoder without the CARRY macro option")
        env = self.make(B=4, decoder=dict(carry_macro=True), eval_rules=True)
        env.reset()
        env.dec.reset(torch.arange(4), env.ws.q, hold=torch.ones(4, dtype=torch.bool))     # mode hold, at rest
        act = torch.zeros(4, 6)
        act[:, 0], act[:, 4], act[:, 5] = 0.5, -1.0, -1.0
        _, _, _, ex = env.step(act, squashed=True)
        m = ex["action_mask"]
        self.assertTrue(bool(env.dec.carry_on.all()))                     # the decision started a CARRY line
        self.assertEqual(m[0].tolist(), [True, True, False, True, True, False])
        _, _, _, ex = env.step(act, squashed=True)                         # during the line: only the abort (a[5])
        self.assertEqual(ex["action_mask"][0].tolist(), [False, False, False, False, True, True])

    def test_timeout_final_obs_and_auto_reset(self):
        env = self.make(task=dict(horizon=10), stagger=False)
        obs0 = env.reset()
        seeds0 = env.dec.seed.clone()
        zero = torch.zeros(6, 6)
        zero[:, 4] = -1.0
        _, _, d1, ex1 = env.step(zero, squashed=True)
        self.assertFalse(bool(d1.any()))
        self.assertTrue(torch.equal(ex1["final_obs"], env.obs))
        obs, rew, d2, ex2 = env.step(zero, squashed=True)
        self.assertTrue(bool(d2.all()) and bool(ex2["time_outs"].all()))
        self.assertTrue(bool((ex2["episode"]["end"] == T.EC["timeout"]).all()))
        self.assertEqual(ex2["episode"]["length"].tolist(), [2] * 6)
        self.assertEqual(ex2["episode"]["steps"].tolist(), [10] * 6)
        self.assertFalse(torch.equal(ex2["final_obs"], obs))              # obs = the new episode's
        self.assertEqual(env.task.tr.t.tolist(), [0] * 6)
        self.assertEqual(env.ep_count.tolist(), [2] * 6)
        self.assertFalse(bool((env.dec.seed == seeds0).any()))            # new macro seeds per episode (review F5)
        expect = (3 * 0x9E3779B1 + 1 * 6 + torch.arange(6)) & 0xFFFFFFFF
        self.assertTrue(torch.equal(env.dec.seed, expect))
        self.assertEqual(obs0.shape, obs.shape)

    def test_stagger(self):
        env = self.make(B=64, task=dict(horizon=400))
        env.reset()
        t = env.task.tr.t
        self.assertTrue(bool((t % 5 == 0).all()) and bool((t < 400).all()) and len(torch.unique(t)) > 20)
        env2 = self.make(B=8, stagger=False)
        env2.reset()
        self.assertEqual(env2.task.tr.t.tolist(), [0] * 8)

    def test_pool_goals(self):
        env = self.make()
        env.refresh_pool(256)
        p = env.pool
        cube = env.bank["cube_pos"][p["i"], :2]
        d = (p["goal_xy"] - cube).norm(dim=-1)
        self.assertTrue(bool((d >= 0.06 - 1e-9).all()) and bool((d <= 0.30 + 1e-9).all()))
        r = (p["goal_xy"] - torch.tensor([0.0388, 0.0], dtype=torch.float64)).norm(dim=-1)
        self.assertTrue(bool((r >= 0.15 - 1e-9).all()) and bool((r <= 0.27 + 1e-9).all()))
        self.assertTrue(bool((p["goal_yaw"] >= 0).all()) and bool((p["goal_yaw"] < math.pi / 2).all()))

    def test_starts_bank_formats(self):
        b = V.load_starts(bank_from_tasks())
        self.assertEqual(b["q"].shape[1], 6)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bank.npz"
            np.savez(p, **{k: v.numpy() for k, v in b.items()})
            b2 = V.load_starts(p)
            self.assertTrue(all(torch.equal(b[k], b2[k]) for k in b))
            # review R9: validation / test seeds or a task file are no training starts (evaluation may allow them)
            for seeds in ([1000, 5], [400003], [900000]):
                q = Path(d) / f"res_{seeds[0]}.npz"
                np.savez(q, seeds=np.array(seeds), **{k: v[:len(seeds)].numpy() for k, v in b.items()})
                with self.assertRaises(ValueError):
                    V.load_starts(q)
                self.assertEqual(V.load_starts(q, allow_eval=True)["q"].shape[0], len(seeds))
            ok = Path(d) / "fresh.npz"
            np.savez(ok, seed=np.array([200000, 300001]), **{k: v[:2].numpy() for k, v in b.items()})
            self.assertEqual(V.load_starts(ok)["q"].shape[0], 2)
        with self.assertRaises(ValueError):
            V.load_starts(GATE / "tasks_1024.npz")
        self.assertEqual(V.load_starts(GATE / "tasks_1024.npz", allow_eval=True)["q"].shape[0], 1024)
        with self.assertRaises(ValueError):
            V.load_starts("/x/test_sealed/unified_data.npz")
        with self.assertRaises(ValueError):
            V.load_starts(Path.home() / "so101_policy" / "data" / "expert_v2" / "gate2")
        with self.assertRaises(ValueError):
            V.load_starts(V.PP_DATA, split=1)
        with self.assertRaises(ValueError):
            V.EnvConfig.from_dict(dict(wrold="kin"))
        self.assertEqual(V.GEOM_NPZ.name, "task_geom_v2.npz")                 # review R10: no MuJoCo needed

    def test_brake_stop_and_grasp_assist(self):
        off = self.make(B=8)
        on = self.make(B=8, grasp_assist=True)
        ev = self.make(B=8, eval_rules=True)          # grasp box logs only: started GRASPs keep running
        off.reset()
        on.reset()
        ev.reset()
        g = torch.Generator().manual_seed(5)
        for k in range(4):
            act = torch.tanh(0.5 * torch.randn(8, 6, generator=g))
            act[:, 4] = 1.0 if k == 3 else -1.0
            if k == 3:
                sxy, syaw = V.brake_stop(off.dec)
                pred_block = on._assist(act.double())
            _, r_off, d_off, ex_off = off.step(act, squashed=True)
            _, r_on, d_on, ex_on = on.step(act, squashed=True)
            _, _, d_ev, ex_ev = ev.step(act, squashed=True)
        st = ex_off["counts"]["grasp_started"]
        self.assertTrue(bool(st.any()))
        mg = ev.dec.macro_goal                       # started and not ended (ended worlds were reset)
        ok = ex_ev["counts"]["grasp_started"] & ~d_ev
        self.assertTrue(bool(ok.any()))
        self.assertLess(float((mg[ok, :2] - sxy[ok]).abs().max()), 1e-12)        # brake_stop = the decoder's rule
        self.assertLess(float((mg[ok, 3] - syaw[ok]).abs().max()), 1e-12)
        blocked = ex_on["counts"]["assist_blocked"]
        self.assertTrue(torch.equal(blocked, pred_block))
        self.assertTrue(bool(blocked.any()))
        # without the assist the blocked triggers end by the grasp-box rule; with it they do not
        box = ex_off["episode"]["end"] == T.EC["grasp_box"]
        self.assertTrue(bool(box[blocked].all()))
        self.assertFalse(bool((ex_on["episode"]["end"] == T.EC["grasp_box"]).any()))
        self.assertTrue(torch.equal(ex_on["counts"]["grasp_started"], st & ~blocked))
        same = ~blocked & ~d_off
        if bool(same.any()):
            self.assertLess(float((r_on - r_off)[same].abs().max()), 1e-6)
        self.assertGreater(float((r_on - r_off)[blocked].min()), 1.0)     # blocked: assist cost, not the failure

    def test_grasp_assist_holds_the_trigger(self):
        from nedm.so101_policy.expert import PickPlaceExpert
        import importlib.util
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("gate_expert", root / "scripts" / "so101_policy" / "gate_expert.py")
        G = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(G)
        t = G.load_tasks(GATE / "tasks_1024.npz", np.arange(3))
        env = self.make(B=3, grasp_assist=True, auto_reset=False)
        env.reset_tasks(torch.arange(3), G.start_state(t, "cpu"), torch.as_tensor(t["goal_xy"]), torch.as_tensor(t["goal_yaw"]),
                        torch.as_tensor(t["seeds"]))
        ex = PickPlaceExpert(env.dec)
        ex.reset(torch.arange(3), env.ws, env.task.goal_xy.clone(), env.task.goal_yaw.clone())
        for _ in range(40):
            _, _, _, e = env.step(ex.act(env.ws), squashed=True)
            self.assertFalse(bool(e["counts"]["assist_blocked"].any()))       # the expert triggers inside the box
            pre = (env.dec.mode >= M["g_brake"]) & (env.dec.mode <= M["g_settle"])
            if bool(pre.any()):
                break
        self.assertTrue(bool(pre.any()))
        a = ex.act(env.ws)
        a[:, 4] = -1.0                                                          # would abort without the assist
        _, _, _, e = env.step(a, squashed=True)
        self.assertFalse(bool(e["counts"]["aborted"][pre].any()))
        self.assertTrue(bool(e["counts"]["assist_held"][pre].all()))
        self.assertFalse(bool(e["action_mask"][pre, 4].any()))                 # the trigger was not read
        self.assertFalse(bool((env.dec.mode[pre] == M["free"]).any()))

    @unittest.skipUnless((GATE / "all" / "v2_base_kin.npz").exists(), "needs the gate's kinematic-world run")
    def test_expert_reproduces_gate(self):
        import importlib.util
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("eval_policy", root / "scripts" / "so101_policy" / "eval_policy.py")
        E = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(E)
        tasks = E.tasks_subset(GATE / "tasks_1024.npz", 3, 0)
        r = E.run_policy(tasks, "kin", "expert", device="cpu", log=None)
        ref = np.load(GATE / "all" / "v2_base_kin.npz")
        self.assertTrue(np.array_equal(r["end"], ref["end"][:3]))
        self.assertTrue(np.array_equal(r["t_end"], ref["t_end"][:3]))
        n = min(r["q"].shape[1], ref["q"].shape[1])
        self.assertLess(float(np.abs(r["q"][:, :n] - ref["q"][:3, :n]).max()), 1e-5)


def _gate_tasks(n):
    import importlib.util
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("gate_expert", root / "scripts" / "so101_policy" / "gate_expert.py")
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    return G, G.load_tasks(GATE / "tasks_1024.npz", np.arange(n))


@unittest.skipUnless(have_geom() and bank_from_tasks() is not None, "needs the task geometry and the gate / fresh starts")
class DecoderPpoOptions(unittest.TestCase):
    """Decoder options for PPO (decoder module docstring, 'PPO options') through the env, kinematic world, CPU."""

    def env_at_tasks(self, n=4, **kw):
        G, t = _gate_tasks(n)
        cfg = dict(world="kin", num_envs=n, device="cpu", seed=3, starts="", auto_reset=False)
        cfg.update(kw)
        env = V.PickPlaceVecEnv(cfg)
        env.reset_tasks(torch.arange(n), G.start_state(t, "cpu"), torch.as_tensor(t["goal_xy"]),
                        torch.as_tensor(t["goal_yaw"]), torch.as_tensor(t["seeds"]))
        return env

    def expert_to_trigger(self, env, max_dec=40):
        """Run the FSM expert until its GRASP trigger decision (not taken); -> the expert's action of that decision."""
        from nedm.so101_policy.expert import PH, PickPlaceExpert
        ex = PickPlaceExpert(env.dec)
        ex.reset(torch.arange(env.B), env.ws, env.task.goal_xy.clone(), env.task.goal_yaw.clone())
        for _ in range(max_dec):
            a = ex.act(env.ws)
            if bool((ex.phase == PH["grasp"]).all()):
                return a
            a[:, 4] = torch.where(ex.phase == PH["grasp"], -1.0, a[:, 4])     # hold back early ones
            ex.phase = torch.where(ex.phase == PH["grasp"], PH["approach"], ex.phase)
            env.step(a, squashed=True)
        self.fail("the expert did not reach its GRASP trigger")

    def test_brake_stop_equals_macro_goal_every_blend(self):
        """R4: DecoderV2.brake_stop (no state change) = macro_goal of a GRASP started at that decision."""
        for blend in ("cosine", "hann", "filter"):
            env = self.make_moving(blend)
            dec = env.dec
            sp, syaw = dec.brake_stop()
            xy, yaw = V.brake_stop(dec)                                      # the env's wrapper (the assist uses it)
            a = torch.zeros(env.B, 6, dtype=torch.float64)
            a[:, 4] = 1.0
            info = dec.decide(a, env.ws)
            st = info["grasp_started"]
            self.assertTrue(bool(st.all()), blend)
            self.assertLess(float((dec.macro_goal[:, :2] - sp[:, :2]).abs().max()), 1e-12, blend)
            self.assertLess(float((dec.macro_goal[:, 3] - syaw).abs().max()), 1e-12, blend)
            self.assertTrue(torch.equal(xy, sp[:, :2]) and torch.equal(yaw, syaw))

    def make_moving(self, blend, n=4):
        env = self.env_at_tasks(n, decoder=dict(blend=blend))
        g = torch.Generator().manual_seed(4)
        for _ in range(6):                                                  # jaw opens; the target moves
            a = torch.tanh(0.6 * torch.randn(n, 6, generator=g, dtype=torch.float64))
            a[:, 2], a[:, 4] = a[:, 2].abs(), -1.0
            env.step(a, squashed=True)
        self.assertGreater(float(env.dec.vel.norm(dim=-1).min()), 1e-3)
        return env

    def test_grasp_gate_ignores_out_of_box_triggers_at_no_cost(self):
        gated = self.env_at_tasks(4, decoder=dict(grasp_gate=True))
        quiet = self.env_at_tasks(4, decoder=dict(grasp_gate=True))
        plain = self.env_at_tasks(4)
        a = torch.zeros(4, 6, dtype=torch.float64)
        a[:, 4] = -1.0
        for _ in range(6):                                                  # jaw open, arm at its start pose
            for e in (gated, quiet, plain):
                e.step(a, squashed=True)
        _, r_q, _, _ = quiet.step(a, squashed=True)                         # no trigger
        a[:, 4] = 1.0                                                       # trigger far from the cube
        self.assertFalse(bool(gated.dec.gate_ok(gated.ws).any()))
        _, r_g, d_g, ex_g = gated.step(a, squashed=True)
        _, r_p, d_p, ex_p = plain.step(a, squashed=True)
        self.assertTrue(bool(ex_g["counts"]["gate_blocked"].all()))
        self.assertFalse(bool(ex_g["counts"]["grasp_started"].any()))
        self.assertTrue(bool((gated.dec.mode == M["free"]).all()) and bool(gated.dec.armed.all()))
        self.assertFalse(bool(d_g.any()))
        self.assertTrue(bool((ex_p["episode"]["end"] == T.EC["grasp_box"]).all()))     # the plain decoder: rule end
        self.assertTrue(bool(ex_g["action_mask"][:, :4].all()))           # the decision moved the target
        self.assertTrue(torch.equal(r_g, r_q))                             # no cost: the same as no trigger
        self.assertTrue(torch.equal(gated.ws.q, quiet.ws.q))
        self.assertGreater(float((r_g - r_p).min()), 2.0)                 # the plain decoder: failure charge
        # inside the gate box (the FSM expert's trigger) the GRASP starts and passes the task's box rule
        env = self.env_at_tasks(4, decoder=dict(grasp_gate=True))
        a = self.expert_to_trigger(env)
        self.assertTrue(bool(env.dec.gate_ok(env.ws).all()))
        _, _, d, ex = env.step(a, squashed=True)
        self.assertTrue(bool(ex["counts"]["grasp_started"].all()) and not bool(d.any()))
        self.assertFalse(bool(env.task.tr.box_log.any()))

    def test_latch_and_abort_action(self):
        """grasp_latch: a[4] <= 0 does not abort a started GRASP; abort_action: a[6] > 0 does (R3); masks follow."""
        env = self.env_at_tasks(4, decoder=dict(grasp_gate=True, grasp_latch=True, abort_action=True))
        self.assertEqual(env.num_actions, 7)
        self.assertEqual(env.action_group_names, ["move", "trigger", "lift", "abort"])
        a6 = self.expert_to_trigger(env)
        a = torch.cat((a6, -torch.ones(4, 1, dtype=torch.float64)), -1)
        env.step(a, squashed=True)
        pre = (env.dec.mode >= M["g_brake"]) & (env.dec.mode <= M["g_settle"])
        self.assertTrue(bool(pre.all()))
        b = a.clone()
        b[:, 4] = -1.0                                                      # trigger released: no abort (latched)
        b[0, 6] = 1.0                                                       # world 0: explicit abort
        _, _, _, ex = env.step(b, squashed=True)
        self.assertEqual(ex["counts"]["aborted"].tolist(), [True, False, False, False])
        self.assertEqual(ex["action_mask"][:, 4].tolist(), [False] * 4)        # a[4] not read in the latched descent
        self.assertEqual(ex["action_mask"][:, 6].tolist(), [True] * 4)
        self.assertEqual(int(env.dec.mode[0]), M["free"])
        self.assertTrue(bool((env.dec.mode[1:] != M["free"]).all()))

    def test_carry_abort_on_its_own_output(self):
        """R3: with abort_action the CARRY line ignores a[5] and stops on a[6]; mask a[5] off, a[6] on."""
        env = self.env_at_tasks(2, decoder=dict(carry_macro=True, abort_action=True), eval_rules=True)
        env.dec.reset(torch.arange(2), env.ws.q, hold=torch.ones(2, dtype=torch.bool))
        a = torch.zeros(2, 7, dtype=torch.float64)
        a[:, 0], a[:, 4], a[:, 6] = 0.5, -1.0, -1.0
        _, _, _, ex = env.step(a, squashed=True)
        self.assertTrue(bool(env.dec.carry_on.all()))
        b = a.clone()
        b[:, 5] = 1.0                                                       # the old abort channel: ignored now
        b[1, 6] = 1.0
        _, _, _, ex = env.step(b, squashed=True)
        self.assertEqual(env.dec.counts["carry_aborts"].tolist(), [0, 1])
        self.assertEqual(ex["action_mask"][0].tolist(), [False, False, False, False, True, False, True])

    def test_arming_only_where_read(self):
        """R5: a[4] in g_close .. g_lift neither arms the trigger nor enters the likelihood."""
        env = self.env_at_tasks(2)
        a = self.expert_to_trigger(env)
        env.step(a, squashed=True)
        for _ in range(40):
            mid = (env.dec.mode >= M["g_close"]) & (env.dec.mode <= M["g_lift"])
            if bool(mid.all()):
                break
            env.step(a, squashed=True)                                       # trigger held through the descent
        self.assertTrue(bool(mid.all()))
        self.assertFalse(bool(env.dec.armed.any()))
        b = a.clone()
        b[:, 4] = -1.0
        _, _, _, ex = env.step(b, squashed=True)
        if bool(((env.dec.mode >= M["g_close"]) & (env.dec.mode <= M["g_lift"])).all()):
            self.assertFalse(bool(env.dec.armed.any()))
        self.assertFalse(bool(ex["action_mask"][:, 4].any()))

    def test_grasp_shaping_telescopes(self):
        """The grasp-pose term is gamma Phi(s') - Phi(s): with gamma = 1 its sum is Phi(end) - Phi(start)."""
        kw = dict(decoder=dict(grasp_gate=True), gamma=1.0)
        on = self.env_at_tasks(3, grasp_shaping=dict(w_x=20.0, w_y=10.0, w_yaw=2.0), **kw)
        off = self.env_at_tasks(3, **kw)
        phi0 = on.phi.clone()
        self.assertTrue(bool((phi0 < 0).all()))
        g = torch.Generator().manual_seed(7)
        tot = torch.zeros(3, dtype=torch.float64)
        for _ in range(8):
            a = torch.tanh(0.5 * torch.randn(3, 6, generator=g, dtype=torch.float64))
            a[:, 4] = -1.0
            _, r1, _, _ = on.step(a, squashed=True)
            _, r0, _, _ = off.step(a, squashed=True)
            tot += (r1 - r0).double()
        self.assertLess(float((tot - (on.phi - phi0)).abs().max()), 1e-5)

    def test_eval_env_from_checkpoint(self):
        """R1: a checkpoint holds the resolved env; eval_policy rebuilds the decoder from it and refuses a mismatch."""
        import importlib.util

        from nedm.so101_policy import ppo as P
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("eval_policy", root / "scripts" / "so101_policy" / "eval_policy.py")
        E = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(E)
        env = V.PickPlaceVecEnv(dict(world="kin", num_envs=2, device="cpu", seed=1, starts=str(bank_from_tasks()),
                                     pool_size=8, decoder=dict(carry_macro=True, abort_action=True, grasp_gate=True)))
        pol = P.Policy(env.num_obs, env.num_actions, env.action_groups, (16,), (16,), init_std=(0.3, 1.0, 0.5, 0.5))
        ppo = P.PPO(env, pol, P.merge(P.train_cfg_default(1), dict(num_steps_per_env=2, std_floor=[0.15, 0.3, 0.2],
                                                                    std_cap=[0.3, 2.0, 1.0])))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.pt"
            ppo.save(path)
            ck = torch.load(path, weights_only=False)
            self.assertTrue(ck["env"]["decoder"]["carry_macro"] and ck["env"]["task"]["self_rule"] is not None)
            tasks = E.tasks_subset(GATE / "tasks_1024.npz", 2, 0)
            r = E.run_policy(tasks, "kin", str(path), device="cpu", log=None, horizon=10, record=False)
            import json
            dec = json.loads(str(r["env_resolved"]))["decoder"]
            self.assertTrue(dec["carry_macro"] and dec["abort_action"] and dec["grasp_gate"])
            with self.assertRaises(ValueError):
                E.run_policy(tasks, "kin", str(path), device="cpu", log=None, horizon=10, record=False,
                             decoder_cfg=dict(carry_macro=False))


if __name__ == "__main__":
    unittest.main()
