"""Tests of the grasp target toward the fixed jaw (``joint_env.GraspShiftConfig``, real2sim round v11, 2026-10-25):
off by default (no ``grasp_shift`` key in the env record); a zero-cost option = off bit for bit (states, observations,
rewards, done flags, end codes); on = the same states and observations, and the reward is lower by exactly
cost x min(max(0, |hx - d| - tol), cap) at the descent event (the TCP goes below h_m above the cube centre within r_m
xy, cube free, before the first firm lift) and at the close event (the measured jaw goes below jaw_rad with the TCP
below h_m); one descent and one close event per scripted grasp, at the crossing steps; no events after a firm lift or
from a start below h_m; the sign of the jaw axis (the fixed jaw is on the -x side of the grasp point, so hx > 0 = cube
toward the moving jaw); the config parser; the round v11 teacher configs. CPU (the kinematic world), a few seconds;
the env tests need the kinematic world's files (skipped without them).

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_grasp_shift -v
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import joint_env as JE
from nedm.so101_policy import kin as K
from nedm.so101_policy import task as T
from nedm.so101_policy.world import WorldState

ROOT = Path(__file__).resolve().parents[2]
HAVE_KIN = Path(JE.JointEnvConfig().kin_points).expanduser().exists() and \
    Path(JE.JointEnvConfig().geom_path).expanduser().exists()
torch.set_num_threads(4)
OFFS = (-0.002, 0.0, 0.004, 0.008, 0.012, 0.016)     # cube offset along the closing axis (+ = toward the moving jaw)
H_START, H_LOW = 0.045, 0.004                        # TCP above the cube centre: start, end of the scripted descent
JAW_OPEN = 0.72


def codec():
    return JE.Codec(np.zeros(18), np.ones(18))


def scene(kin):
    """B = 2 places x 6 offsets x 2 (repeat; the second half gets the lifted flag in some tests): the cube at rest,
    yaw = the radial gripper yaw (closing axis radially outward, the HOME roll 0 orientation), the TCP H_START above
    the cube centre and OFFS[k] short of it along the closing axis; per-step joint targets of a straight vertical
    descent to H_LOW (6 decisions), then the jaw closing to 0 (3 decisions) -> (WorldState, targets [B, 45, 6], hx0)."""
    rows = [(r, a, s) for r in (0.18, 0.22) for a in (-30.0, 20.0) for s in OFFS][:12] * 2
    a = np.array(rows)
    B = len(a)
    th = np.radians(a[:, 1])
    cxy = np.stack([K.PAN_AXIS_XY[0] + a[:, 0] * np.cos(th), a[:, 0] * np.sin(th)], 1)
    yaw = np.arctan2(cxy[:, 1], cxy[:, 0] - K.PAN_AXIS_XY[0])
    ex = np.stack([np.cos(yaw), np.sin(yaw)], 1)
    t64 = lambda x: torch.as_tensor(np.asarray(x, np.float64))          # noqa: E731
    hs = np.concatenate([np.full(1, H_START), np.linspace(H_START, H_LOW, 31)[1:]])     # start + 30 descent steps
    qs, q0 = [], None
    for h in hs:
        tcp = np.concatenate([cxy - a[:, 2:3] * ex, np.full((B, 1), T.CUBE_REST_Z + h)], 1)
        q5, ok, ep, _ = kin.ik_topdown(q0, t64(tcp), t64(yaw))
        assert bool(ok.all()) and float(ep.max()) < 1e-4
        q0 = q5
        qs.append(torch.cat((q5, torch.full((B, 1), JAW_OPEN, dtype=torch.float64)), -1))
    tg = torch.stack(qs[1:], 1)                                          # [B, 30, 6]
    close = tg[:, -1:].repeat(1, 15, 1)
    close[:, :, 5] = 0.0
    tg = torch.cat((tg, close), 1)                                       # [B, 45, 6] = 9 decisions
    cq = K.mat_quat(K.rotz(t64(yaw)))
    z3 = torch.zeros(B, 3, dtype=torch.float64)
    ws = WorldState(q=qs[0], qd=torch.zeros(B, 6, dtype=torch.float64),
                    cube_pos=t64(np.concatenate([cxy, np.full((B, 1), T.CUBE_REST_Z)], 1)), cube_quat=cq, cube_v=z3,
                    cube_w=z3.clone(), contact=torch.zeros(B, 3, dtype=torch.bool),
                    arm_table=torch.zeros(B, dtype=torch.bool), invalid=torch.zeros(B, dtype=torch.bool))
    return ws, tg, torch.as_tensor(a[:, 2])


def env_for(grasp_shift=None, B=24, start=None):
    cfg = JE.JointEnvConfig(world="kin", num_envs=B, device="cpu", starts="", horizon=200, auto_reset=False,
                            stagger=False, seed=3, grasp_shift=grasp_shift)
    env = JE.JointVecEnv(cfg, codec())
    ws, tg, hx0 = scene(env.kin)
    if start is not None:
        ws = start(env, ws)
    goal = ws.cube_pos[:, :2] + torch.tensor([0.0, 0.08], dtype=torch.float64)
    env.reset_tasks(torch.arange(B), ws, goal, torch.zeros(B, dtype=torch.float64), torch.arange(B))
    return env, tg, hx0


def drive(env, tg, hook=None):
    out = []
    for k in range(tg.shape[1] // JE.STEPS):
        obs, rew, done, ex = env.step(None, targets=tg[:, k * JE.STEPS:(k + 1) * JE.STEPS], step_hook=hook)
        out.append(torch.cat((env.ws.q, env.ws.qd, env.ws.cube_pos, obs.double(), rew.double()[:, None],
                              done.double()[:, None], env.task.tr.end.double()[:, None]), -1))
    return out


def recorder(gs):
    """step hook: the per-step features and the env's event statistics; expected events re-derived here."""
    rec = dict(h=[], hx=[], jaw=[], free=[], dxy=[], stats=[], live=[])

    def hook(env, j):
        F = T.features(env.ws, env.kin, None, env.tcfg)
        rec["h"].append(F["tcp"][:, 2] - F["c"][:, 2])
        rec["hx"].append(F["hx"])
        rec["jaw"].append(F["jaw"].clone())
        rec["free"].append(~F["held"] & F["on_table"] & ~env.task.tr.lifted)
        rec["dxy"].append(F["dxy"])
        rec["stats"].append(env.gs_stats.clone())
        rec["live"].append(env.task.tr.end == 0)
    return rec, hook


class TestConfig(unittest.TestCase):
    def test_config(self):
        c = JE.JointEnvConfig()
        self.assertIsNone(c.grasp_shift)
        self.assertNotIn("grasp_shift", c.to_dict())                          # off: the old record
        for off in (None, False, {}, "", "{}", "null"):
            self.assertIsNone(JE.grasp_shift_config(off), msg=repr(off))
            self.assertIsNone(JE.JointEnvConfig(grasp_shift=off).grasp_shift)
        on = JE.JointEnvConfig.from_dict(dict(grasp_shift=dict(d_m=0.008, cost=20, note="n")))
        self.assertEqual(on.grasp_shift, JE.GraspShiftConfig(d_m=0.008, cost=20.0))
        self.assertEqual(on.to_dict()["grasp_shift"], dict(d_m=0.008, cost=20.0, tol_m=0.001, cap_m=0.015,
                                                           h_m=0.028, r_m=0.03, jaw_rad=0.45))
        self.assertEqual(JE.JointEnvConfig.from_dict(json.loads(json.dumps(on.to_dict()))), on)
        self.assertEqual(JE.grasp_shift_config('{"d_m": 0.004}'), JE.GraspShiftConfig())
        for bad in (dict(d_m=0.02), dict(d_m=-0.001), dict(cost=-1.0), dict(tol_m=0.01), dict(cap_m=0.0),
                    dict(h_m=0.1), dict(r_m=0.001), dict(jaw_rad=1.0), dict(d_m=True), dict(cost="1"),
                    dict(h_m=float("nan")), dict(k=1)):
            with self.assertRaises(ValueError, msg=repr(bad)):
                JE.grasp_shift_config(bad)

    def test_cost(self):
        gs = JE.GraspShiftConfig(d_m=0.004, cost=50.0, tol_m=0.001, cap_m=0.015)
        err = torch.tensor([0.0, 0.0009, -0.001, 0.003, -0.006, 0.05, -0.05], dtype=torch.float64)
        want = torch.tensor([0.0, 0.0, 0.0, 0.1, 0.25, 0.75, 0.75], dtype=torch.float64)
        self.assertTrue(torch.allclose(JE.grasp_shift_cost(err, gs), want, atol=1e-12))

    def test_v11_teacher_configs(self):
        d = ROOT / "configs/so101_policy/ppo/r2s"
        base = json.loads((d / "teacher_roll_v8.json").read_text())
        for mm in (4, 8):
            raw = json.loads((d / f"teacher_roll_v11_gs{mm}.json").read_text())
            self.assertEqual({k: v for k, v in raw.items() if k not in ("note", "arm_table", "grasp_shift")},
                             {k: v for k, v in base.items() if k != "note"})      # everything of D's teacher env
            c = JE.JointEnvConfig.from_dict(raw)
            self.assertEqual(c.arm_table, JE.ArmTableConfig(0.0, 0.02, True))  # the v9c rule, on
            self.assertEqual(c.grasp_shift, JE.GraspShiftConfig(d_m=mm / 1000))


@unittest.skipUnless(HAVE_KIN, "kinematics files missing")
class TestGeometry(unittest.TestCase):
    def test_fixed_jaw_on_minus_x(self):
        """The fixed jaw (gripper body collision points) at the cube height band is on the -x side of the grasp
        point, 17-19 mm from it (2.8 mm from a centred 30 mm cube); the open moving jaw is on +x."""
        pts = K.load_points(Path(JE.JointEnvConfig().kin_points).expanduser())
        pg, rg = pts["gripper"]
        loc = pg - np.asarray(K.GRASP_POINT)
        band = (loc[:, 2] - rg < 0.011) & (loc[:, 2] + rg > -0.019) & (np.abs(loc[:, 1]) < 0.015)
        inner = (loc[band, 0] + rg[band])[loc[band, 0] < 0].max()
        self.assertTrue(-0.019 < inner < -0.017, msg=inner)
        self.assertFalse(bool((np.abs(loc[band, 0]) < 0.015).any()))           # nothing of the fixed jaw near x 0
        kin = K.SO101Kin("cpu", points=JE.JointEnvConfig().kin_points)
        pos, rot = kin.body_poses(torch.tensor([[0.0, 0.0, 0.0, 1.5, 0.0, JAW_OPEN]], dtype=torch.float64))
        pj, _ = pts["moving_jaw_so101_v1"]
        w = pos[0, K.JAW] + torch.as_tensor(pj) @ rot[0, K.JAW].T
        locj = ((w - pos[0, K.GRIPPER]) @ rot[0, K.GRIPPER]).numpy() - np.asarray(K.GRASP_POINT)
        self.assertGreater(float(np.median(locj[:, 0])), 0.015)


@unittest.skipUnless(HAVE_KIN, "kinematics files missing")
class TestEnv(unittest.TestCase):
    GS = JE.GraspShiftConfig(d_m=0.004, cost=50.0)

    def test_zero_cost_is_off(self):
        """cost 0: the rollout without the option, bit for bit (and the events are still counted)."""
        e0, tg, _ = env_for()
        ref = drive(e0, tg)
        e1, tg1, _ = env_for(dict(d_m=0.004, cost=0.0))
        got = drive(e1, tg1)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(ref, got)))
        self.assertGreater(float(e1.gs_stats[0]), 0)
        self.assertEqual(float(e1.gs_stats[-1]), 0.0)

    def test_events_and_cost(self):
        """cost 50: states, observations, done flags = off; reward(off) - reward(on) per decision = the summed costs
        of the events re-derived from the recorded steps; one descent and one close event per world at the crossing
        steps; hx at the events = the constructed offset (sign: + = toward the moving jaw)."""
        e0, tg, hx0 = env_for()
        ref = drive(e0, tg)
        e1, tg1, _ = env_for(self.GS)
        rec, hook = recorder(self.GS)
        got = drive(e1, tg1, hook)
        for a, b in zip(ref, got):
            self.assertTrue(torch.equal(torch.cat((a[:, :-3], a[:, -2:]), -1), torch.cat((b[:, :-3], b[:, -2:]), -1)))
        S = {k: torch.stack(v, 1) for k, v in rec.items() if k != "stats"}      # [B, steps]
        B, n = S["h"].shape
        h_prev = torch.cat((torch.full((B, 1), H_START), S["h"][:, :-1]), 1)
        jaw_prev = torch.cat((torch.full((B, 1), JAW_OPEN), S["jaw"][:, :-1]), 1)
        zone = S["free"] & (S["dxy"] < self.GS.r_m)
        e1w = zone & (S["h"] < self.GS.h_m) & (h_prev >= self.GS.h_m)
        e2w = zone & (S["h"] < self.GS.h_m) & (jaw_prev >= self.GS.jaw_rad) & (S["jaw"] < self.GS.jaw_rad)
        cost = (e1w.double() + e2w.double()) * JE.grasp_shift_cost(S["hx"] - self.GS.d_m, self.GS)
        for k, (a, b) in enumerate(zip(ref, got)):
            dr = (a[:, -3] - b[:, -3])
            want = cost[:, k * JE.STEPS:(k + 1) * JE.STEPS].sum(1)
            self.assertTrue(torch.allclose(dr, want.float().double(), atol=1e-5), msg=(k, dr, want))
        self.assertTrue(torch.equal(e1w.sum(1), torch.ones(B, dtype=torch.long)))
        self.assertTrue(torch.equal(e2w.sum(1), torch.ones(B, dtype=torch.long)))
        st = rec["stats"][-1]
        self.assertEqual(float(st[0]), B)
        self.assertEqual(float(st[3]), B)
        hx1 = (S["hx"] * e1w).sum(1)
        hx2 = (S["hx"] * e2w).sum(1)
        self.assertLess(float((hx1 - hx0).abs().max()), 5e-4)                   # the constructed offset, + = moving jaw
        self.assertLess(float((hx2 - hx0).abs().max()), 5e-4)
        self.assertAlmostEqual(float(st[1]), float(hx1.sum()), places=9)
        self.assertAlmostEqual(float(st[6]), float(cost.sum()), places=9)
        # the shifted target is cheaper: the world with the cube 4 mm toward the moving jaw pays 0
        on_target = (hx0 - self.GS.d_m).abs() < 1e-9
        self.assertTrue(bool((cost.sum(1)[on_target] == 0).all()))
        self.assertTrue(bool((cost.sum(1)[~on_target] > 0).all()))

    def test_lifted_and_low_start(self):
        """Worlds with the lifted flag (forced) pay nothing; a world reset with the TCP already below h_m has no
        descent event at its first step."""
        e0, tg, _ = env_for()
        e1, tg1, _ = env_for(self.GS)
        e0.task.tr.lifted[12:] = True
        e1.task.tr.lifted[12:] = True
        ref, got = drive(e0, tg), drive(e1, tg1)
        for a, b in zip(ref, got):
            self.assertTrue(torch.equal(a[12:], b[12:]))
        self.assertTrue(any(not torch.equal(a[:12], b[:12]) for a, b in zip(ref, got)))
        # a start below h_m: put the arm at the end of the descent (TCP H_LOW above the cube centre)
        e2, tg2, _ = env_for(self.GS)
        low = tg2[:, 29].clone()
        ws = e2.ws
        ws2 = WorldState(**{**{f: getattr(ws, f) for f in ws.__dataclass_fields__}, "q": low})
        e2.reset_tasks(torch.arange(e2.B), ws2, e2.task.goal_xy.clone(), e2.task.goal_yaw.clone(),
                       torch.arange(e2.B))
        hold = low[:, None].repeat(1, JE.STEPS, 1)
        e2.step(None, targets=hold)
        self.assertEqual(float(e2.gs_stats[0]), 0.0)
        self.assertLess(float(e2.gs_h.max()), self.GS.h_m)


if __name__ == "__main__":
    unittest.main()
