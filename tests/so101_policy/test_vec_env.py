"""Task env (src/nedm/so101_policy/vec_env.py) in the kinematic world on the CPU: observation layout, the decision =
5 task steps (reward sum), action masks, time-out / final observation / auto-reset, per-episode decoder seeds, the task
pool, the task charge discount, start banks, and (if the gate files exist) the FSM expert reproducing the gate's
kinematic-world episodes exactly.

CPU, a few seconds (skips without the task geometry: ~/so101_policy/ckpt/task_geom_v1.npz or the vendor model).

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


def bank_from_tasks():
    """A small start bank from the gate tasks file (settled collector starts), else None."""
    p = GATE / "tasks_1024.npz"
    return p if p.exists() else None


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
        with self.assertRaises(ValueError):
            V.load_starts("/x/test_sealed/unified_data.npz")
        with self.assertRaises(ValueError):
            V.EnvConfig.from_dict(dict(wrold="kin"))

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


if __name__ == "__main__":
    unittest.main()
