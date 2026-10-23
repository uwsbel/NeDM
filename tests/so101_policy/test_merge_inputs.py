"""Tests of the merge of the studentlab student inputs (``ViralConfig.cmd_in`` / ``img_hist`` / ``hist``,
``expand_student``, robot ``Guard.u_layer``) with the real2sim last-command input (``ViralConfig.lastcmd``,
``expand_lastcmd``, robot ``Guard.prev``), 2026-10-23: the two "last command" inputs are independent options and can be
on together (robot vector [q, qd, history, cmd, lastcmd]); each expansion keeps the given student's chunk; the
robot ``Controller`` feeds both; the online loop builds and stores the combined vector. CPU, about 20 s; the robot and
online-loop tests need the kinematic world's files and the JT (skipped without them).

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_merge_inputs -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy.vision import viral as VI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import studentlab_ref as R  # noqa: E402
from test_viral_lastcmd import HAVE_KIN, HW, fake_robot_run, hw, tiny  # noqa: E402

torch.set_num_threads(4)
HIST = 2
P = 12 * (1 + HIST)             # [q, qd] now + history
NC = 30                         # cmd input: n_exec 5 x 6


def full(seed=0):
    """A tiny student with every input option on (hist 2, cmd_in, img_hist, lastcmd); all weights perturbed."""
    return tiny(lastcmd=True, hist=HIST, seed=seed, cmd_in=True, img_hist=True)


def rand(B=4, seed=1, res=32):
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(B, 2, 3, res, res, generator=g)
    prev = torch.randn(B, 2, 3, res, res, generator=g)
    robot = torch.cat((torch.randn(B, P, generator=g), 0.05 * torch.randn(B, NC, generator=g),
                       0.05 * torch.randn(B, 6, generator=g), torch.randn(B, 6, generator=g)), -1)
    return img, prev, robot


class TestLayout(unittest.TestCase):
    def test_dims_and_forward_by_hand(self):
        st = full()
        fd = st.fd
        self.assertEqual(st.cfg.robot_dim, P + NC + 12)
        self.assertEqual(st.head[0].in_features, fd + P + NC + fd)            # lastcmd: own layer, not in the head
        self.assertEqual(tuple(st.lastcmd_in.weight.shape), (st.head[0].out_features, 12))
        img, prev, robot = rand()
        with torch.no_grad():
            p, f = st(img, robot, prev=prev)
            # by hand: head input [f, robot (with cmd) normalised, previous features], + the last-command term
            fp = st.features(prev)
            fc = st.features(img)
            rn = torch.cat((((robot[:, :P].view(-1, 1 + HIST, 12) - st.robot_mean) / st.robot_std).clamp(-10, 10)
                            .reshape(-1, P), (robot[:, P:P + NC] / st.cmd_s).clamp(-10, 10)), -1)
            h = st.head[0](torch.cat((fc, rn, fp), -1)) + st.lastcmd_in(
                (robot[:, P + NC:] / st.lastcmd_scale).clamp(-10, 10))
            want = st.head[1:](h).view(-1, 10, 6)
            self.assertLess(float((p - want).abs().max()), 1e-5)
            self.assertLess(float((f - fc).abs().max()), 1e-5)
            # the robot's cached-feature path = forward
            p2 = st.head_from_features(fc, robot, fp)
            self.assertLess(float((p2 - p).abs().max()), 1e-5)
            # each input moves the output (none is dropped)
            for a, b in ((0, P), (P, P + NC), (P + NC, P + NC + 12)):
                r2 = robot.clone()
                r2[:, a:b] += 0.5
                self.assertGreater(float((st(img, r2, prev=prev)[0] - p).abs().max()), 1e-4, (a, b))
            self.assertGreater(float((st(img, robot, prev=img)[0] - p).abs().max()), 1e-4)
        with self.assertRaises(ValueError):
            st(img, robot[:, :-1], prev=prev)
        with self.assertRaises(ValueError):
            st.head_from_features(fc, robot[:, :P + NC], fp)

    def test_cfg_dict_keys(self):
        new = ("cmd_in", "cmd_scale", "img_hist", "lastcmd")
        cases = {(): set(), ("lastcmd",): {"lastcmd"}, ("cmd_in",): {"cmd_in", "cmd_scale", "img_hist"},
                 ("cmd_in", "lastcmd"): set(new)}
        for on, keys in cases.items():
            d = VI.cfg_dict(VI.ViralConfig(**{k: True for k in on}))
            self.assertEqual({k for k in new if k in d}, keys, on)

    def test_checkpoint_round_trip(self):
        st = full()
        img, prev, robot = rand()
        with tempfile.TemporaryDirectory() as d:
            VI.save_student(st, Path(d) / "s.pt")
            st2, ck = VI.load_student(Path(d) / "s.pt", "cpu")
        self.assertTrue(ck["cfg"]["lastcmd"] and ck["cfg"]["cmd_in"] and ck["cfg"]["img_hist"])
        with torch.no_grad():
            self.assertTrue(torch.equal(st(img, robot, prev=prev)[0], st2(img, robot, prev=prev)[0]))


class TestExpansion(unittest.TestCase):
    def test_both_orders_keep_the_chunk(self):
        """plain -> + hist / cmd_in / img_hist -> + lastcmd, and the other order: the same weights, and the given
        student's chunk on its own inputs whatever the new inputs are."""
        base = tiny(seed=3)
        a = VI.expand_lastcmd(VI.expand_student(base, hist=HIST, cmd_in=True, img_hist=True, device="cpu"), "cpu")
        b = VI.expand_student(VI.expand_lastcmd(base, "cpu"), hist=HIST, cmd_in=True, img_hist=True, device="cpu")
        for st in (a, b):
            self.assertTrue(st.cfg.lastcmd and st.cfg.cmd_in and st.cfg.img_hist and st.cfg.hist == HIST)
        sa, sb = a.state_dict(), b.state_dict()
        self.assertEqual(sorted(sa), sorted(sb))
        for k in sa:
            self.assertTrue(torch.equal(sa[k], sb[k]), k)
        img, prev, robot = rand(seed=5)
        a.eval(), b.eval(), base.eval()
        with torch.no_grad():
            p0 = base(img, robot[:, :12])[0]
            for st in (a, b):
                self.assertLess(float((st(img, robot, prev=prev)[0] - p0).abs().max()), 1e-5)
        # a lastcmd student expanded by expand_student keeps its last-command weights
        lc = tiny(lastcmd=True, seed=4)
        e = VI.expand_student(lc, cmd_in=True, device="cpu").eval()
        self.assertTrue(torch.equal(e.lastcmd_in.weight, lc.lastcmd_in.weight))
        r = torch.cat((robot[:, :12], torch.zeros(len(robot), NC), robot[:, P + NC:]), -1)
        with torch.no_grad():
            self.assertLess(float((e(img, r)[0] - lc(img, torch.cat((robot[:, :12], robot[:, P + NC:]), -1))[0])
                                  .abs().max()), 1e-5)


@unittest.skipUnless((HW / "hw_lib.py").exists() and HAVE_KIN, "hw_lib or the kinematic world's files missing")
class TestRobot(unittest.TestCase):
    def policy(self, st):
        H = hw()
        if H.default_kin_points() is None:
            self.skipTest("kin_points_v1.npz not found")
        with tempfile.TemporaryDirectory() as d:
            VI.save_student(st, Path(d) / "s.pt")
            return H, H.StudentPolicy(Path(d) / "s.pt", "cpu")

    def test_warmup_with_the_last_command(self):
        for st in (tiny(lastcmd=True), full()):
            _, pol = self.policy(st)
            pol.warmup()
            self.assertEqual(pol.n_query, 0)

    def test_controller_feeds_both(self):
        """Every input block of the vector the robot fed: q / qd now, cmd = Guard.u_layer of the 5 steps before - q
        (0 at the first query), last command = lastcmd_raw of Guard.prev (u_sim - i_corr) of the 2 steps before."""
        H, pol = self.policy(full())
        self.assertTrue(pol.cmd_in and pol.lastcmd and pol.img_hist and pol.H == HIST)
        A = fake_robot_run(H, pol, 32)
        self.assertTrue({"qry_robot", "qry_lastcmd", "u_layer"} <= set(A))
        rob, steps = A["qry_robot"], A["qry_step"].astype(int)
        self.assertEqual(rob.shape, (len(steps), P + NC + 12))
        prev = A["u_sim"].copy()
        prev[:, :5] -= A["i_corr"]
        self.assertGreater(float(np.abs(A["i_corr"]).max()), 0.0)
        for i, k in enumerate(steps):
            np.testing.assert_allclose(rob[i, :6], A["q_sim"][k].astype(np.float32), atol=1e-6)
            cmd, lc = rob[i, P:P + NC], rob[i, P + NC:]
            np.testing.assert_allclose(lc, A["qry_lastcmd"][i].astype(np.float32), atol=1e-6)
            if i == 0:
                self.assertEqual(float(np.abs(cmd).max()), 0.0)
                self.assertEqual(float(np.abs(lc).max()), 0.0)
                continue
            want = A["u_layer"][k - 5:k].astype(np.float32) - A["q_sim"][k].astype(np.float32)
            np.testing.assert_allclose(cmd.reshape(5, 6), want, atol=1e-6)
            q = torch.as_tensor(A["q_sim"][k])[None]
            sim = VI.lastcmd_raw(torch.as_tensor(prev[k - 1])[None], torch.as_tensor(prev[k - 2])[None], q)
            np.testing.assert_allclose(lc, sim[0].float().numpy(), atol=1e-6)
        self.assertGreater(float(np.abs(rob[1:, P + NC + 6:]).max()), 0.0)


@unittest.skipUnless(R.HAVE and HAVE_KIN, "jt_h110 / dev tasks / kinematic world missing")
class TestOnlineLoop(unittest.TestCase):
    def test_vector_buffer_and_training(self):
        st = VI.expand_lastcmd(VI.expand_student(R.tiny_student(0), hist=HIST, cmd_in=True, img_hist=True,
                                                 device="cpu"), "cpu")
        od, views = R.make_online(st, dict(horizon=40, buffer=40, light_redraw_p=0.3), 0, 16)
        self.assertEqual(od.buf.robot.shape[1], P + NC + 12)
        A = torch.arange(16)
        od.env.reset()                  # what the first act() does (as test_studentlab.TestOnlineLoop)
        od._draw(A), od._lag_reset(A), od._bias_reset(A), od._qd_reset(A)
        info = []
        for it in range(12):
            od.it = it
            u = None if od.env.u_exe is None else od.env.u_exe.clone()
            q = od.env.ws.q.clone()
            lcf = od.lc_fresh.clone()
            od.act()
            rob, fresh = od.last["robot"], od.last["fresh"]
            self.assertEqual(rob.shape[1], P + NC + 12)
            self.assertTrue(torch.equal(lcf, fresh), it)                   # the two episode-start flags agree
            lc = rob[:, P + NC:]
            if u is None:
                self.assertEqual(float(lc.abs().max()), 0.0)
            else:
                want = VI.lastcmd_raw(u[:, -1], u[:, -2], q, fresh).float()
                self.assertTrue(torch.equal(lc, want), it)
                cmd = rob[:, P:P + NC]
                wc = (u.float() - q.float()[:, None, :]).reshape(16, -1)
                self.assertLess(float((cmd[~fresh] - wc[~fresh]).abs().max()), 1e-6, it)
            if od.buf.n >= 16:
                info.append(od.train_steps())
        self.assertTrue(info and all(np.isfinite(v["loss"]) for v in info if "loss" in v))
        self.assertGreater(float(od.st.lastcmd_in.weight.detach().abs().max()), 0.0)  # the new layer trains
        r = R.run_eval(VI.expand_lastcmd(VI.expand_student(R.tiny_student(1), hist=HIST, cmd_in=True, img_hist=True,
                                                           device="cpu"), "cpu"))
        self.assertEqual(len(r["end"]), 8)


if __name__ == "__main__":
    unittest.main()
