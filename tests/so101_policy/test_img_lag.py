"""Tests of the real2sim v8 image latency of the online loop (``viral.OnlineConfig.img_lag`` -> ``ImgLagConfig``,
``LagHistory(goal=True).at``, ``OnlineDAgger`` with it) and of the v8 latency configs. CPU; the online-loop tests run
``viral.OnlineDAgger`` in the kinematic world with a tiny student and fake views (``studentlab_ref.py``; need jt_h110
and the dev task file).

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_img_lag -v

``TestBitIdentity`` runs ``studentlab_ref.py`` against the base commit's code (``git archive``; skipped without git or
the commit) and against this tree, and compares every output exactly (default options, the existing integer lag with
goal jumps and the robot dynamics' queued delay).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import joint_env as JE
from nedm.so101_policy.vision import viral as VI

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import studentlab_ref as R  # noqa: E402

BASE = os.environ.get("NEDM_IMGLAG_BASE", "39d2e5b72")           # the commit before the img_lag change
CONFIGS = REPO / "configs" / "so101_policy" / "vision"
REAL_LAG_MS = (27.8, 62.2)          # robot logs 2026-10-22/23, p1 / p99 of the per-query image lag (both cameras)


class RecViews(R.FakeViews):
    """``FakeViews`` that also records the rendered q, cube position, cube quaternion of every call."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rec = []

    def render(self, q, cube_pos, cube_quat, goal_pose, dr=None, light=None, keep=False):
        out = super().render(q, cube_pos, cube_quat, goal_pose, dr, light, keep)
        self.rec.append(dict(q=q.detach().clone(), cp=cube_pos.detach().clone(), cq=cube_quat.detach().clone(),
                             goal=goal_pose.detach().clone(), keep=bool(keep), out=out.clone()))
        return out


def make(st, ocfg_over, seed=0, envs=12):
    jt = JE.JointTeacher(str(R.JT_PATH), "cpu")
    o = dict(envs=envs, buffer=400, steps_per_iter=1, horizon=60, lr=1e-3, lr_backbone=1e-3)
    o.update(ocfg_over or {})
    views = RecViews(st.cfg.res, st.cfg.cameras)
    with R.kin_world():
        od = VI.OnlineDAgger(st, jt, R.load(), "unused-nrd.pt", views, VI.OnlineConfig.from_dict(o), "cpu", seed,
                             lambda s: None)
    return od, views


def run(st, ocfg_over, n_dec, seed=0, envs=12, train=True):
    """-> (od, views, per decision dict(sub = the 5 world states after each 20 ms step of the decision [5, N, 13],
    pre = the state at the decision [N, 13], fresh, lag, calls = the render calls of the decision))."""
    od, views = make(st, ocfg_over, seed, envs)
    rec = []
    for it in range(n_dec):
        od.it = it
        n0 = len(views.rec)
        sub = []
        with VI.step_tap(od.env, lambda ws: sub.append(VI._state13(*VI._sane(ws.q, ws.cube_pos, ws.cube_quat,
                                                                                "cpu")).clone())):
            fresh = od.fresh.clone()
            pre = None if od.env.obs is None else VI._state13(*VI._sane(od.env.ws.q, od.env.ws.cube_pos,
                                                                         od.env.ws.cube_quat, "cpu")).clone()
            od.act()
        rec.append(dict(sub=torch.stack(sub), pre=pre, fresh=fresh, calls=views.rec[n0:],
                        gp=od.last["goal_pose"].clone(), rgb=od.last["rgb"].clone(),
                        lag=None if od.lag_x is None else od.last["img_lag"].clone(), cur=od.last.get("cur_goal")))
        if train and od.buf.n >= 16:
            od.train_steps()
    return od, views, rec


class TestConfig(unittest.TestCase):
    def test_default_off(self):
        self.assertEqual(VI.OnlineConfig().img_lag, {})
        self.assertEqual(VI.OnlineConfig.from_dict({}).img_lag, {})

    def test_validation_and_kmax(self):
        c = VI.ImgLagConfig.from_dict(dict(lo_ms=20, hi_ms=70))
        self.assertEqual((c.lo_ms, c.hi_ms, c.redraw_p, c.per_camera, c.quantize, c.kmax), (20.0, 70.0, 1.0, False,
                                                                                            False, 4))
        self.assertEqual(VI.ImgLagConfig.from_dict(dict(lo_ms=0, hi_ms=60)).kmax, 4)
        self.assertEqual(VI.ImgLagConfig.from_dict(dict(lo_ms=0, hi_ms=59.9)).kmax, 3)
        self.assertEqual(VI.ImgLagConfig.from_dict(dict(lo_ms=0, hi_ms=0)).kmax, 1)
        for bad in (dict(lo_ms=30, hi_ms=20), dict(lo_ms=-1, hi_ms=20), dict(lo_ms=0, hi_ms=500),
                    dict(lo_ms=0, hi_ms=40, redraw_p=1.5), dict(lo_ms=0, hi_ms=40, bogus=1)):
            with self.assertRaises(ValueError, msg=str(bad)):
                VI.OnlineConfig.from_dict(dict(img_lag=bad))
        with self.assertRaises(ValueError):                                 # one lag source for the online loop
            VI.OnlineConfig.from_dict(dict(img_lag=dict(lo_ms=20, hi_ms=60), lag_range=[1, 3]))

    def test_draws(self):
        g = torch.Generator().manual_seed(0)
        c = VI.ImgLagConfig.from_dict(dict(lo_ms=20, hi_ms=70))
        x = c.draw(20000, 2, g, "cpu")
        self.assertEqual(tuple(x.shape), (20000, 2))
        self.assertGreaterEqual(float(x.min()), 1.0)
        self.assertLessEqual(float(x.max()), 3.5)
        self.assertAlmostEqual(float(x.mean()), 2.25, delta=0.02)
        xq = VI.ImgLagConfig.from_dict(dict(lo_ms=20, hi_ms=70, quantize=True)).draw(2000, 1, g, "cpu")
        self.assertTrue(torch.equal(xq, xq.round()))
        self.assertEqual(set(xq.flatten().tolist()) - {1.0, 2.0, 3.0, 4.0}, set())


class TestHistoryAt(unittest.TestCase):
    def hist(self, B=3, kmax=4, steps=9, seed=0):
        g = torch.Generator().manual_seed(seed)
        h = VI.LagHistory(B, kmax, "cpu", goal=True)
        q0, cp0 = torch.randn(B, 6, generator=g), torch.randn(B, 3, generator=g)
        cq0 = torch.nn.functional.normalize(torch.randn(B, 4, generator=g), dim=-1)
        h.fill(torch.arange(B), q0, cp0, cq0, torch.zeros(B, 3))
        states = [(q0, cp0, cq0, torch.zeros(B, 3))]
        for s in range(steps):
            q, cp = torch.randn(B, 6, generator=g), torch.randn(B, 3, generator=g)
            cq = torch.nn.functional.normalize(torch.randn(B, 4, generator=g), dim=-1)
            gl = torch.full((B, 3), float(s + 1))
            h.push(q, cp, cq, gl)
            states.append((q, cp, cq, gl))
        return h, states

    def test_whole_steps_are_the_entries_exactly(self):
        h, states = self.hist()
        now = torch.full((3, 3), -7.0)
        for k in range(5):
            q, cp, cq, g = h.at(torch.full((3,), float(k)), now)
            q2, cp2, cq2 = h.get(torch.full((3,), k))
            self.assertTrue(torch.equal(q, q2) and torch.equal(cp, cp2) and torch.equal(cq, cq2), k)
            self.assertTrue(torch.equal(q, states[-1 - k][0].float()), k)
            self.assertTrue(torch.equal(g, now if k == 0 else states[-1 - k][3].float()), k)

    def test_fraction_interpolates(self):
        h, states = self.hist()
        lag = torch.tensor([0.25, 1.5, 2.75])
        q, cp, cq, g = h.at(lag, torch.zeros(3, 3))
        for i, x in enumerate(lag.tolist()):
            k0, a = int(math.floor(x)), x - math.floor(x)
            s0, s1 = states[-1 - k0], states[-2 - k0]
            torch.testing.assert_close(q[i], (1 - a) * s0[0][i] + a * s1[0][i])
            torch.testing.assert_close(cp[i], (1 - a) * s0[1][i] + a * s1[1][i])
            q1 = s1[2][i] if float((s0[2][i] * s1[2][i]).sum()) >= 0 else -s1[2][i]
            want = torch.nn.functional.normalize((1 - a) * s0[2][i] + a * q1, dim=-1)
            torch.testing.assert_close(cq[i], want)
            self.assertAlmostEqual(float(cq[i].norm()), 1.0, places=6)
            self.assertTrue(torch.equal(g[i], s0[3][i]))                  # the goal of the newer neighbour

    def test_goal_required_and_old_api_unchanged(self):
        h = VI.LagHistory(2, 3, "cpu")
        self.assertIsNone(h.g)
        with self.assertRaises(ValueError):
            h.at(torch.zeros(2), torch.zeros(2, 3))
        hg = VI.LagHistory(2, 3, "cpu", goal=True)
        with self.assertRaises(ValueError):
            hg.push(torch.zeros(2, 6), torch.zeros(2, 3), torch.zeros(2, 4))


@unittest.skipUnless(R.HAVE, "jt_h110 / dev tasks missing")
class TestOnlineLoop(unittest.TestCase):
    """``OnlineDAgger`` with ``img_lag``: the rendered state is the world state the lag back."""

    def test_zero_lag_is_the_default_loop_bit_for_bit(self):
        od0, v0, _ = run(R.tiny_student(0), {}, 14)
        od1, v1, rec = run(R.tiny_student(0), dict(img_lag=dict(lo_ms=0, hi_ms=0)), 14)
        self.assertEqual(int(od0.samples), int(od1.samples))
        for k in ("rgb", "q", "robot", "lab1", "lab2", "valid2", "aux", "aug"):
            self.assertTrue(torch.equal(getattr(od0.buf, k), getattr(od1.buf, k)), k)
        for (n0, p0), (n1, p1) in zip(od0.st.state_dict().items(), od1.st.state_dict().items()):
            self.assertTrue(torch.equal(p0, p1), n0)
        self.assertTrue(all(float(r["lag"].abs().max()) == 0.0 for r in rec))

    def test_whole_step_lag_matches_the_integer_lag_and_the_state_back(self):
        """img_lag 40 ms = 2 steps: the rendered q / cube are the world 2 steps before (also across a decision
        boundary: k = 2 < 5), and the run is bit for bit the old integer lag 2."""
        od_old, v_old, _ = run(R.tiny_student(0, lag=2, lag_range=[0, 0]), {}, 14)
        od, views, rec = run(R.tiny_student(0, lag=2, lag_range=[0, 0]), dict(img_lag=dict(lo_ms=40, hi_ms=40)), 14)
        for k in ("rgb", "q", "robot", "lab1", "lab2", "valid2", "aux", "aug"):
            self.assertTrue(torch.equal(getattr(od_old.buf, k), getattr(od.buf, k)), k)
        checked = 0
        for t in range(1, len(rec)):
            (c,) = rec[t]["calls"]
            want = rec[t - 1]["sub"][JE.STEPS - 1 - 2]                       # 2 steps before the decision
            ok = ~rec[t]["fresh"]                                            # an episode start: its start state
            got = torch.cat((c["q"], c["cp"], c["cq"]), -1)
            self.assertTrue(torch.equal(got[ok], want[ok]), t)
            self.assertTrue(torch.equal(got[~ok], rec[t]["pre"][~ok]), t)
            self.assertFalse(torch.equal(got[ok], rec[t]["pre"][ok]), t)    # the state really is an older one
            checked += int(ok.sum())
        self.assertGreater(checked, 100)

    def test_fractional_lag_interpolates_world_states(self):
        od, views, rec = run(R.tiny_student(0), dict(img_lag=dict(lo_ms=30, hi_ms=30)), 8, train=False)
        for t in range(1, len(rec)):
            (c,) = rec[t]["calls"]
            ok = ~rec[t]["fresh"]
            s1, s2 = rec[t - 1]["sub"][JE.STEPS - 2], rec[t - 1]["sub"][JE.STEPS - 3]    # 1 and 2 steps back
            torch.testing.assert_close(c["q"][ok], (0.5 * (s1[:, :6] + s2[:, :6]))[ok])
            torch.testing.assert_close(c["cp"][ok], (0.5 * (s1[:, 6:9] + s2[:, 6:9]))[ok])

    def test_redraw_per_decision_and_per_episode(self):
        _, _, rec1 = run(R.tiny_student(0), dict(img_lag=dict(lo_ms=20, hi_ms=70, redraw_p=1.0)), 6, train=False)
        _, _, rec0 = run(R.tiny_student(0), dict(img_lag=dict(lo_ms=20, hi_ms=70, redraw_p=0.0)), 6, train=False)
        lags1 = torch.stack([r["lag"][:, 0] for r in rec1])
        lags0 = torch.stack([r["lag"][:, 0] for r in rec0])
        self.assertTrue(bool((lags1[1:] != lags1[:-1]).all()))                   # a new lag every decision
        same = torch.stack([~r["fresh"] for r in rec0[1:]])
        self.assertTrue(bool((lags0[1:] == lags0[:-1])[same].all()))             # per episode: only at a reset
        for L in (lags0, lags1):
            self.assertGreaterEqual(float(L.min()), 1.0)
            self.assertLessEqual(float(L.max()), 3.5)

    def test_per_camera_lags_two_calls_shared_light(self):
        """per_camera (whole steps: quantize): camera c's image = the camera-c call's image of the world lag[:, c]
        steps back; the second call keeps the first call's light / intrinsics."""
        od, views, rec = run(R.tiny_student(0), dict(img_lag=dict(lo_ms=20, hi_ms=70, per_camera=True, quantize=True),
                                                     light_redraw_p=0.5), 8, train=False)
        differ = 0
        for t, r in enumerate(rec):
            self.assertEqual(len(r["calls"]), 2)
            self.assertTrue(r["calls"][1]["keep"])
            self.assertEqual(views.states[2 * t], views.states[2 * t + 1])      # same per-call render state
            lag = r["lag"]
            self.assertEqual(tuple(lag.shape), (12, 2))
            differ += int((lag[:, 0] != lag[:, 1]).sum())
            for c in range(2):
                call = r["calls"][c]
                self.assertTrue(torch.equal(r["rgb"][:, c], call["out"][:, c]), (t, c))
                if t == 0:
                    continue
                ok = ~r["fresh"]
                k = lag[:, c].long()
                want = torch.stack([rec[t - 1]["sub"][JE.STEPS - 1 - int(k[i])][i] for i in range(len(k))])
                got = torch.cat((call["q"], call["cp"], call["cq"]), -1)
                self.assertTrue(torch.equal(got[ok], want[ok]), (t, c))
        self.assertGreater(differ, 10)

    def test_queue_delay_with_lag_renders_the_delayed_world(self):
        """robot dynamics queue delay 0-2 (every world): the history follows the world after the dynamics stage."""
        dyn = dict(p=1.0, d_lo=0, d_hi=2, delay_mode="queue")
        od, views, rec = run(R.tiny_student(0), dict(img_lag=dict(lo_ms=40, hi_ms=40), dyn=dyn), 10, train=False)
        self.assertTrue(od.env.dyn is not None and bool(od.env.dyn.on.all()))
        self.assertEqual(set(od.env.dyn.d0.tolist()) - {0, 1, 2}, set())
        for t in range(1, len(rec)):
            (c,) = rec[t]["calls"]
            ok = ~rec[t]["fresh"]
            got = torch.cat((c["q"], c["cp"], c["cq"]), -1)
            self.assertTrue(torch.equal(got[ok], rec[t - 1]["sub"][JE.STEPS - 3][ok]), t)

    def test_goal_jump_lagged_goal_and_no_stale_sample(self):
        """Goal jumps (end of a decision) with lags 1-3.5 steps: the images of the next decision still show the old
        goal (the goal during the previous decision's steps); those decisions give no buffer sample."""
        od, views, rec = run(R.tiny_student(0), dict(img_lag=dict(lo_ms=20, hi_ms=70), horizon=400, buffer=4000,
                                                     teacher_share=1.0, noise_share=0.0, hold_cap_s=6.0,
                                                     goal_jump=dict(p=1.0, rate=1.0)), 70, train=False)
        self.assertGreater(od.env.goal_jump.jumps, 4)
        stale = 0
        for t in range(1, len(rec)):
            (c,) = rec[t]["calls"]
            ok = ~rec[t]["fresh"]
            self.assertTrue(torch.equal(c["goal"][ok], rec[t - 1]["gp"][ok]), t)       # the goal of then
            self.assertTrue(torch.equal(c["goal"][~ok], rec[t]["gp"][~ok]), t)         # episode start: the new one
            jumped = ok & (rec[t]["gp"] != rec[t - 1]["gp"]).any(-1)
            self.assertTrue(torch.equal(~rec[t]["cur"], jumped), t)
            stale += int(jumped.sum())
        self.assertGreater(stale, 0)
        self.assertEqual(int(od.samples), sum(int(r["cur"].sum()) for r in rec))      # stale decisions: no sample
        self.assertEqual(int(od.buf.n), int(od.samples))
        # aux goal of the kept samples: the image goal, which is the CURRENT goal for every kept sample (the stale
        # decisions, whose images show the old goal, add none; v8 integration review: the check below says so)
        aux_goal = torch.cat([r["calls"][0]["goal"][r["cur"]] for r in rec])[:, :2]
        self.assertTrue(torch.equal(od.buf.aux[:od.buf.n, :2], aux_goal))
        cur_goal = torch.cat([r["gp"][r["cur"]] for r in rec])[:, :2].to(aux_goal)
        self.assertTrue(torch.equal(od.buf.aux[:od.buf.n, :2], cur_goal))


@unittest.skipUnless(R.HAVE and shutil.which("git"), "jt_h110 / dev tasks / git missing")
class TestBitIdentity(unittest.TestCase):
    """The online loop (kinematic world, 14 decisions with training steps) and the evaluation actor give the same
    numbers as the base commit's code, bit for bit, with img_lag off."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="imglag_base_"))
        r = subprocess.run(f"git -C {REPO} archive {BASE} src | tar -x -C {cls.tmp}", shell=True, capture_output=True)
        cls.ok = r.returncode == 0 and (cls.tmp / "src").exists()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_ref(self, src, out, cfg, ocfg):
        env = dict(os.environ, PYTHONPATH=str(src), CUDA_VISIBLE_DEVICES="")
        r = subprocess.run([sys.executable, str(HERE / "studentlab_ref.py"), "--out", str(out), "--cfg",
                            json.dumps(cfg), "--ocfg", json.dumps(ocfg)], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return np.load(out)

    def check(self, tag, cfg, ocfg):
        if not self.ok:
            self.skipTest(f"git archive {BASE} failed")
        a = self.run_ref(self.tmp / "src", self.tmp / f"base_{tag}.npz", cfg, ocfg)
        b = self.run_ref(REPO / "src", self.tmp / f"new_{tag}.npz", cfg, ocfg)
        self.assertEqual(set(a.files), set(b.files))
        bad = [k for k in a.files if not np.array_equal(a[k], b[k], equal_nan=True)]
        self.assertEqual(bad, [], f"{tag}: outputs differ from {BASE}")
        self.assertGreater(int(a["samples"]), 0)

    def test_defaults(self):
        self.check("default", {}, {})

    def test_integer_lag_goal_jump_queue_delay(self):
        self.check("lag", {"lag": 2, "lag_range": [1, 3]},
                   {"lag_range": [1, 4], "goal_jump": {"p": 1.0, "rate": 1.0}, "qd_robot_p": 0.5,
                    "dyn": {"p": 0.8, "d_lo": 0, "d_hi": 2, "delay_mode": "queue"}, "dyn_student_only": True})


class TestV8Configs(unittest.TestCase):
    """viral_r2s_online_v8_C_lat.json = relabel_v4 + img_lag + queued delay 0-2; v8_S_lat = online_v1 + the same."""

    def load(self, name):
        return json.loads((CONFIGS / name).read_text())

    def check_lat(self, d):
        o = VI.OnlineConfig.from_dict(d)
        il = VI.ImgLagConfig.from_dict(o.img_lag)
        self.assertLessEqual(il.lo_ms, REAL_LAG_MS[0])                      # covers the robot's p1 - p99
        self.assertGreaterEqual(il.hi_ms, REAL_LAG_MS[1])
        self.assertEqual(il.redraw_p, 1.0)
        self.assertEqual(o.lag_range, ())
        dc = JE.dyn_config(o.dyn)
        self.assertEqual((dc.d_lo, dc.d_hi, dc.delay_mode, dc.jitter), (0, 2, "queue", 0.0))
        self.assertTrue(o.dyn_student_only)
        return o

    def test_c(self):
        base, new = self.load("viral_r2s_online_relabel_v4.json"), self.load("viral_r2s_online_v8_C_lat.json")
        self.check_lat(new)
        diff = {k for k in set(base) | set(new) if base.get(k) != new.get(k)} - {"note"}
        self.assertEqual(diff, {"img_lag", "lag_range", "dyn"})
        self.assertNotIn("lag_range", new)
        bd, nd = dict(base["dyn"]), dict(new["dyn"])
        self.assertEqual({k for k in set(bd) | set(nd) if bd.get(k) != nd.get(k)}, {"d_hi", "delay_mode"})

    def test_s(self):
        base, new = self.load("viral_r2s_online_v1.json"), self.load("viral_r2s_online_v8_S_lat.json")
        o = self.check_lat(new)
        diff = {k for k in set(base) | set(new) if base.get(k) != new.get(k)} - {"note"}
        self.assertEqual(diff, {"img_lag", "dyn", "dyn_student_only"})
        dc = JE.dyn_config(o.dyn)
        neutral = JE.RobotDynConfig(p=dc.p, d_lo=dc.d_lo, d_hi=dc.d_hi, delay_mode="queue")
        self.assertEqual(dc, neutral)                                       # only the delay: no servo change
        self.assertFalse(o.relabel_cmd_speed)


if __name__ == "__main__":
    unittest.main()
