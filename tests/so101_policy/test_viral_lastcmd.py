"""Tests of the camera student's last-command input (``vision.viral.ViralConfig.lastcmd``; VIRAL's last-action input):
off = the old network, checkpoint layout and loop vectors; a student expanded from one without it acts exactly as that
student (zero weights on the new inputs: bit for bit, also with bf16 autocast and in the online loop); the input values
on toy envs (the kinematic world through ``run_tasks`` + ``Actor`` and the script's ``evaluate``, the online loop on
the NRD world, teacher slots and automatic resets included); the robot ``Controller`` (``hw_lib``) builds the same
vector as the sim's ``Actor`` and ``OnlineDAgger`` for the same command sequence; the new config files. CPU, about
20 s; the env tests need the kinematic world's files, the JT and the NRD checkpoint (skipped without them).

    CUDA_VISIBLE_DEVICES= PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_viral_lastcmd -v
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import joint_env as JE
from nedm.so101_policy.vision import viral as VI

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs" / "so101_policy" / "vision"
HW = ROOT / "scripts" / "so101_policy" / "real2sim" / "hw"
TASKS = Path("~/so101_policy/runs/viral/tasks/dev100.npz").expanduser()
JT_CK = Path("~/so101_policy/runs/viral/jt/jt_h110.pt").expanduser()
NRD_CK = Path("~/so101_policy/ckpt/nrd_c1_s62.pt").expanduser()
HAVE_KIN = Path(JE.JointEnvConfig().kin_points).expanduser().exists() and \
    Path(JE.JointEnvConfig().geom_path).expanduser().exists() and TASKS.exists()
HAVE_JT = JT_CK.exists()
HAVE_NRD = HAVE_JT and NRD_CK.exists() and TASKS.exists()
torch.set_num_threads(4)


def stats(seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(robot_mean=torch.randn(12, generator=g), robot_std=torch.rand(12, generator=g) + 0.5,
                tgt_mean=0.01 * torch.randn(10, 6, generator=g), tgt_std=torch.full((10, 6), 0.01),
                aux_mean=torch.zeros(VI.AUX_DIM), aux_std=torch.ones(VI.AUX_DIM))


def tiny(lastcmd=False, hist=0, seed=0, perturb=True, **kw):
    """A small ViralNet (eval mode); ``perturb``: every weight moved off its initial value (biases, the new inputs)."""
    cfg = VI.ViralConfig(res=32, encoder="small", keypoints=8, pool_dim=16, hidden=(32, 32), aux_hidden=16, amp=False,
                         qd_noise=0.3, hist=hist, lastcmd=lastcmd, **kw)
    torch.manual_seed(seed)
    st = VI.ViralNet(cfg, stats(seed), pretrained=False)
    if perturb:
        with torch.no_grad():
            for p in st.parameters():
                p.add_(0.05 * torch.randn_like(p))
    return st.eval()


def old_forward(st, images, robot, qd_scale=None):
    """``ViralNet.forward`` + ``robot`` before the last-command input (commit 9f9f2fa5e)."""
    H = 1 + int(st.cfg.hist)
    if getattr(st.cfg, "qd_input_zero", False):
        robot = robot.view(-1, H, 12)
        robot = torch.cat((robot[..., :6], torch.zeros_like(robot[..., 6:])), -1)
    r = ((robot.view(-1, H, 12) - st.robot_mean) / st.robot_std).clamp(-10, 10)
    if st.training and st.cfg.qd_noise > 0:
        n = st.cfg.qd_noise * torch.randn_like(r[..., 6:])
        if qd_scale is not None:
            n = n * qd_scale.to(n).view(-1, 1, 1)
        r = torch.cat((r[..., :6], r[..., 6:] + n), -1)
    r = r.reshape(-1, 12 * H)
    f = st.features(images)
    return st.head(torch.cat((f, r.to(f.dtype)), -1).float()).view(-1, st.cfg.chunk, 6), f


def inputs(B=5, hist=0, seed=1):
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(B, 2, 3, 32, 32, generator=g)
    robot = torch.randn(B, 12 * (1 + hist), generator=g)
    lc = torch.cat((0.05 * torch.randn(B, 6, generator=g), torch.randn(B, 6, generator=g)), -1)
    lc[0] = 0.0
    lc[1] *= 400.0                                                       # beyond the +-10 clamp
    return img, robot, lc


class Views:
    """Stand-in for ``DRViews`` (real-scene layout): black images, no camera effects."""
    real, fx, dr = True, None, None

    def __init__(self, res=32, cams=2):
        self.res, self.cams = res, cams

    def draw(self, n):
        return {"colors": torch.zeros(n, 1, 3)}

    def merge_nominal(self, d, nom):
        return d

    def render(self, q, cp, cq, gp, dr=None, light=None):
        return torch.zeros(len(q), self.cams, self.res, self.res, 3, dtype=torch.uint8)


def chunk_stub(B):
    """Normalised chunk of a stand-in student (with tgt_std 1, mean 0 = the delta): arm +2 mrad per step, jaw -4."""
    p = torch.zeros(B, 10, 6)
    p[..., :5] = 0.002 * torch.arange(1, 11, dtype=torch.float32)[None, :, None]
    p[..., 5] = -0.004 * torch.arange(1, 11, dtype=torch.float32)[None]
    return p


def by_hand(u5, q):
    """The last-command input from the executed targets of the decision before (u5 [B, 5, 6]) and q, written out."""
    u5, q = u5.to(torch.float64), q.to(torch.float64)
    return torch.cat((u5[:, 4] - q, (u5[:, 4] - u5[:, 3]) / 0.02), -1)


# ---- off: the old network, checkpoints and loop ---------------------------------------------------------------------
class TestOff(unittest.TestCase):
    def test_defaults_and_dims(self):
        self.assertFalse(VI.ViralConfig().lastcmd)
        self.assertFalse(VI.ViralConfig.from_dict({}).lastcmd)
        self.assertFalse(VI.OnlineConfig().require_lastcmd)
        self.assertEqual((VI.ViralConfig().robot_dim, VI.ViralConfig(hist=2).robot_dim), (12, 36))
        self.assertEqual((VI.ViralConfig(lastcmd=True).robot_dim, VI.ViralConfig(hist=1, lastcmd=True).robot_dim),
                         (24, 36))
        self.assertEqual(len(VI.LASTCMD_DQ_SCALE + VI.LASTCMD_V_SCALE), 12)

    def test_off_network_is_the_old_one(self):
        for hist in (0, 1):
            st = tiny(hist=hist)
            self.assertFalse(any("lastcmd" in k for k in st.state_dict()))
            self.assertFalse(any("lastcmd" in k for k, _ in st.named_buffers()))
            img, robot, _ = inputs(hist=hist)
            with torch.no_grad():
                self.assertTrue(torch.equal(st(img, robot)[0], old_forward(st, img, robot)[0]))
                st.train()
                for qs in (None, torch.rand(5)):
                    torch.manual_seed(3)
                    a = st(img, robot, qs)[0]
                    torch.manual_seed(3)
                    self.assertTrue(torch.equal(a, old_forward(st, img, robot, qs)[0]))
                st.eval()

    def test_checkpoint_layout(self):
        with tempfile.TemporaryDirectory() as d:
            st = tiny()
            VI.save_student(st, Path(d) / "off.pt", it=3)
            ck = torch.load(Path(d) / "off.pt", weights_only=False)
            self.assertEqual(set(ck["cfg"]), set(dataclasses.asdict(VI.ViralConfig())) - {"lastcmd"})   # old layout
            back, _ = VI.load_student(Path(d) / "off.pt", "cpu")
            self.assertFalse(back.cfg.lastcmd)
            img, robot, lc = inputs()
            self.assertTrue(torch.equal(back.predict(img, robot), st.predict(img, robot)))
            on = tiny(lastcmd=True)
            VI.save_student(on, Path(d) / "on.pt")
            self.assertTrue(torch.load(Path(d) / "on.pt", weights_only=False)["cfg"]["lastcmd"])
            back, _ = VI.load_student(Path(d) / "on.pt", "cpu")
            x = torch.cat((robot, lc), -1)
            self.assertTrue(back.cfg.lastcmd and torch.equal(back.predict(img, x), on.predict(img, x)))

    def test_round_training_refuses_it(self):
        with self.assertRaises(NotImplementedError):
            VI.train(None, None, VI.ViralConfig(lastcmd=True), "/nonexistent")


# ---- zero-initialised new inputs: the expanded student is the old one ---------------------------------------------
class TestZeroInit(unittest.TestCase):
    def test_expanded_student_acts_as_the_old_one(self):
        for hist in (0, 1):
            old = tiny(hist=hist)
            new = VI.expand_lastcmd(old, "cpu")
            self.assertTrue(new.cfg.lastcmd and not new.training)
            self.assertEqual(float(new.lastcmd_in.weight.detach().abs().max()), 0.0)
            sd = new.state_dict()
            self.assertEqual(set(sd) - set(old.state_dict()), {"lastcmd_in.weight", "lastcmd_scale"})
            self.assertTrue(all(torch.equal(v, sd[k]) for k, v in old.state_dict().items()))
            img, robot, lc = inputs(hist=hist)
            x = torch.cat((robot, lc), -1)
            for amp in (False, True):
                with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
                    a, b = old(img, robot)[0], new(img, x)[0]
                self.assertTrue(torch.equal(a, b), msg=f"hist {hist} amp {amp}")
                self.assertTrue(torch.equal(VI.denormalise(old, a.float()), VI.denormalise(new, b.float())))
            old.train()
            new.train()
            tgt, valid, aux = torch.randn(5, 10, 6), torch.ones(5, 10), torch.randn(5, VI.AUX_DIM)
            torch.manual_seed(5)
            la, _ = old.loss(img, robot, tgt, valid, aux, qd_scale=torch.rand(5))
            torch.manual_seed(5)
            lb, _ = new.loss(img, x, tgt, valid, aux, qd_scale=torch.rand(5))
            self.assertTrue(torch.equal(la, lb))
            with self.assertRaises(ValueError):                         # the old 12-value vector: refused
                new(img, robot)
        self.assertIs(VI.expand_lastcmd(new, "cpu"), new)

    def test_new_inputs_learn(self):
        new = VI.expand_lastcmd(tiny(), "cpu").train()
        img, robot, lc = inputs()
        loss, _ = new.loss(img, torch.cat((robot, lc), -1), torch.randn(5, 10, 6), torch.ones(5, 10))
        loss.backward()
        self.assertGreater(float(new.lastcmd_in.weight.grad.abs().sum()), 0.0)
        opt = torch.optim.SGD([new.lastcmd_in.weight], lr=1.0)
        opt.step()
        new.eval()
        a = new.predict(img, torch.cat((robot, lc), -1))
        b = new.predict(img, torch.cat((robot, torch.zeros_like(lc)), -1))
        self.assertFalse(torch.equal(a, b))                             # trained: the output depends on the input

    def test_expand_hist_keeps_the_last_command(self):
        st = tiny(lastcmd=True)
        st2 = VI.expand_hist(st, 1, "cpu").eval()
        self.assertTrue(st2.cfg.lastcmd and st2.cfg.hist == 1)
        self.assertTrue(torch.equal(st2.lastcmd_in.weight, st.lastcmd_in.weight))
        img, robot, lc = inputs()
        a = st.predict(img, torch.cat((robot, lc), -1))
        b = st2.predict(img, torch.cat((robot, torch.randn(5, 12), lc), -1))      # zero weights on the history
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


# ---- the input values ------------------------------------------------------------------------------------------------
class TestValues(unittest.TestCase):
    def test_raw_formula_fresh_and_non_finite(self):
        u1 = torch.tensor([[0.10, 0.20, -0.30, 0.0, 1.0, 0.5], [0.0] * 6])
        u0 = torch.tensor([[0.09, 0.22, -0.30, 0.0, 1.0, 0.4], [0.0] * 6])
        q = torch.tensor([[0.08, 0.25, -0.31, 0.01, 1.0, 0.3], [float("nan")] + [0.0] * 5], dtype=torch.float64)
        out = VI.lastcmd_raw(u1, u0, q)
        self.assertEqual((out.shape, out.dtype), ((2, 12), torch.float64))
        want = torch.cat((u1.double()[0] - q[0], (u1.double()[0] - u0.double()[0]) / 0.02))
        self.assertTrue(torch.equal(out[0], want))
        np.testing.assert_allclose(out[0, 6:].numpy(), [0.5, -1.0, 0.0, 0.0, 0.0, 5.0], atol=2e-6)
        self.assertEqual(float(out[1, 0]), 0.0)                         # non-finite -> 0
        f = VI.lastcmd_raw(u1, u0, q, torch.tensor([True, False]))
        self.assertTrue(torch.equal(f[0], torch.zeros(12, dtype=torch.float64)) and torch.equal(f[1], out[1]))
        self.assertTrue(torch.equal(VI.lastcmd_zeros(3, "cpu"), torch.zeros(3, 12, dtype=torch.float64)))

    def test_fixed_normalisation(self):
        st = tiny(lastcmd=True)
        lc = torch.tensor([[0.05, -0.1, 0.0, 0.025, 1.0, 0.25, 1.0, -2.0, 0.5, 0.0, 30.0, 5.0]])
        np.testing.assert_allclose(st.lastcmd_norm(lc).numpy(), [[1, -2, 0, 0.5, 10, 1, 1, -2, 0.5, 0, 10, 1]],
                                   atol=1e-6)
        self.assertTrue(torch.equal(st.lastcmd_scale, torch.tensor(VI.LASTCMD_DQ_SCALE + VI.LASTCMD_V_SCALE)))
        self.assertIn("lastcmd_scale", st.state_dict())                  # saved with the checkpoint (review 10-23)

    @unittest.skipUnless(HAVE_KIN, "kinematics files / development tasks missing")
    def test_actor_through_run_tasks_on_the_kin_env(self):
        from nedm.so101_policy import joint_teacher as JT
        from nedm.so101_policy.vision.demos import load_tasks
        B = 3
        t = load_tasks(TASKS, np.arange(B))
        env = JT.make_env("kin", B, JE.Codec(np.zeros(18), np.ones(18)), "cpu", horizon=40, dyn=False)
        st = tiny(lastcmd=True)
        st.tgt_mean.zero_()
        st.tgt_std.fill_(1.0)
        seen = []

        def predict(img, robot):
            seen.append((robot.clone(), None if env.u_exe is None else env.u_exe.clone(), env.ws.q.clone()))
            return chunk_stub(len(robot))
        st.predict = predict
        actor = VI.Actor(st, Views(), device="cpu")
        actor.env = env
        JT.run_tasks(env, t, None, actor, full_horizon=True)
        self.assertEqual(len(seen), env.max_episode_length + 1)
        self.assertEqual(seen[0][0].shape, (B, 24))
        self.assertEqual(float(seen[0][0][:, 12:].abs().max()), 0.0)     # the episode start
        n_hand = 0
        for k in range(1, len(seen)):
            robot, u5, q = seen[k]
            self.assertTrue(torch.equal(robot[:, :6], q.float()))           # [q, qd] first, as before
            self.assertTrue(torch.equal(robot[:, 12:], by_hand(u5, q).float()), msg=f"decision {k}")
            # where the action layer passed the student's targets: +2 mrad per step = 0.1 rad/s, jaw -0.2 rad/s
            q_dec = seen[k - 1][2].float()                               # the Actor adds the chunk to float32 q
            asked = (q_dec[:, None, :] + chunk_stub(B)[:, :5]).double()
            passed = (u5 - asked).abs().amax(1) < 1e-9                   # [B, 6] joints the layer left alone
            v = robot[:, 18:].double()
            want = torch.tensor([0.1] * 5 + [-0.2], dtype=torch.float64).expand(B, 6)
            self.assertTrue(bool(((v - want).abs() < 1e-5)[passed].all()))
            n_hand += int(passed.sum())
        self.assertGreater(n_hand, 20)
        actor.reset(B)                                                   # a decision without a step since: refused
        actor(env.ws, torch.zeros(B, 3), 0)
        with self.assertRaises(RuntimeError):
            actor(env.ws, torch.zeros(B, 3), 1)
        unbound = VI.Actor(st, Views(), device="cpu")                    # no env: refused
        unbound.reset(B)
        with self.assertRaises(RuntimeError):
            unbound(env.ws, torch.zeros(B, 3), 0)

    @unittest.skipUnless(HAVE_KIN and HAVE_JT, "kinematics files / JT missing")
    def test_script_evaluate_binds_the_env(self):
        spec = importlib.util.spec_from_file_location("vision_viral_script",
                                                      ROOT / "scripts/so101_policy/vision_viral.py")
        S = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(S)
        from nedm.so101_policy.vision.demos import load_tasks
        jt = JE.JointTeacher(JT_CK, "cpu")
        cache = S.EnvCache(jt, "cpu", None, 0, None, 30)
        st = tiny(lastcmd=True)
        r = S.evaluate(st, jt, load_tasks(TASKS, np.arange(2)), "kin", cache, Views(), device="cpu")
        self.assertIn("summary", r)
        out = S.collect(load_tasks(TASKS, np.arange(2)), "kin", jt, cache, np.ones(2, np.int64), np.zeros(2), 0, 2,
                        VI.Actor(st, Views(), device="cpu"), log=lambda s: None)
        self.assertIn("row_q", out)

    @unittest.skipUnless(HAVE_NRD, "NRD checkpoint / JT / development tasks missing")
    def test_online_loop_buffer_and_zero_init_trajectory(self):
        """The online loop on 8 NRD worlds (4 teacher slots, 20-step episodes: automatic resets): the buffer's robot
        vector = [q, qd, the last command] with the last command written out from the env's executed targets (0 after
        a reset), teacher slots too; a student expanded with zero weights drives exactly the old trajectory."""
        from nedm.so101_policy.vision.demos import load_tasks
        tasks = load_tasks(TASKS)
        jt = JE.JointTeacher(JT_CK, "cpu")
        oc = VI.OnlineConfig(envs=8, buffer=512, horizon=20, teacher_share=0.5)
        D = 10

        def run(st):
            od = VI.OnlineDAgger(st, jt, tasks, NRD_CK, Views(), oc, "cpu", 7, lambda s: None)
            rec = []
            orig = od.env.step

            def step(*a, **k):
                out = orig(*a, **k)
                rec.append((od.env.u_exe.clone(), out[2].clone()))
                return out
            od.env.step = step
            qs = []
            for _ in range(D):
                qs.append(None if od.env.ws is None else od.env.ws.q.clone())
                od.act()
            return od, rec, qs

        old = tiny(seed=4)
        new = VI.expand_lastcmd(old, "cpu")
        a, ra, _ = run(old)
        b, rb, qs = run(new)
        N, n = 8, 8 * D
        self.assertEqual((a.buf.robot.shape[1], b.buf.robot.shape[1]), (12, 24))
        self.assertEqual(b.buf.n, n)
        for k in ("q", "lab1", "lab2", "valid2", "aux"):
            self.assertTrue(torch.equal(getattr(a.buf, k)[:n], getattr(b.buf, k)[:n]), msg=k)
        self.assertTrue(torch.equal(a.buf.robot[:n], b.buf.robot[:n, :12]))     # the old vector, unchanged
        self.assertTrue(torch.equal(a.env.ws.q, b.env.ws.q))                     # the same trajectory
        lc = b.buf.robot[:n, 12:].view(D, N, 12)
        self.assertEqual(float(lc[0].abs().max()), 0.0)                  # the first decision: no command yet
        resets = 0
        for i in range(1, D):
            u5, done = rb[i - 1]
            want = by_hand(u5, qs[i]).float()
            want[done] = 0.0                                             # an automatic reset: a new episode
            resets += int(done.sum())
            self.assertTrue(torch.equal(lc[i], want), msg=f"decision {i}")
        self.assertGreater(resets, 0)
        live = lc[1:].abs().amax(-1) > 0                                 # [D - 1, N]
        self.assertTrue(bool(live[:, b.teacher].any()) and bool(live[:, ~b.teacher].any()))   # both slot kinds


# ---- the robot builds the sim's vector ----------------------------------------------------------------------------
OLD_LOG_KEYS = {"t", "q_real", "q_sim", "qd_sim", "u_target", "u_sim", "u_real", "i_corr", "query", "ev_step",
                "ev_lead", "ev_range", "ev_floor", "ev_real_range", "ev_max_rel", "n_stop", "ms_qdot", "ms_frames",
                "ms_pre", "ms_model", "ms_guard", "ms_ctrl"} | \
    {"qry_" + k for k in ("t", "step", "chunk", "exec", "aux", "plate_xy", "plate_yaw4_deg", "cube_xyz", "frame_age_ms",
                          "thumbs")}                                     # the episode log before the last command


def fake_robot_run(H, pol, n=32, i_time_s=0.2):
    """``n`` control steps of ``pol`` on a slow fake servo from HOME (Controller + Guard, integral correction on)."""
    pts = H.default_kin_points()
    jm = H.JointMap.placeholder_identity()
    lim = H.RealLimits(np.array([-150.0] * 5 + [0.0]), np.array([150.0] * 5 + [100.0]))
    g = H.Guard(H.GuardConfig(i_time_s=i_time_s, i_e_max_deg=5.0), H.ActionLayer(pts), jm, lim)
    ctrl = H.Controller(pol, jm, g, H.view_specs(None, H.REAL_CAMERAS, pol.res), H.QdotEstimator(), bgr=False)
    q = jm.sim_to_real(np.radians(H.HOME_SIM_DEG))
    ctrl.reset(q)
    img = np.full((480, 640, 3), 90, np.uint8)
    frames = lambda: {c: (img, 5.0) for c in H.REAL_CAMERAS}             # noqa: E731
    for s in range(n):
        out = ctrl.step(s * H.DT, q, frames)
        q = q + 0.3 * (out.u_real - q)                                  # a slow fake servo (stops short)
    return ctrl.arrays()


def hw():
    sys.path.insert(0, str(HW))
    import hw_lib as H
    return H


@unittest.skipUnless((HW / "hw_lib.py").exists(), "hw_lib missing")
class TestRobotMatchesSim(unittest.TestCase):
    def test_off_keeps_the_old_call_and_log(self):
        H = hw()
        if H.default_kin_points() is None:
            self.skipTest("kin_points_v1.npz not found")
        with tempfile.TemporaryDirectory() as d:
            VI.save_student(tiny(), Path(d) / "off.pt")
            pol = H.StudentPolicy(Path(d) / "off.pt", "cpu")
        self.assertFalse(pol.lastcmd)
        calls = []
        q0 = pol.query

        def query(images, q_sim, qd_sim):                                # the old signature: no last_cmd passed
            calls.append(1)
            return q0(images, q_sim, qd_sim)
        pol.query = query
        A = fake_robot_run(H, pol, 12)
        self.assertEqual(set(A), OLD_LOG_KEYS)                           # no qry_lastcmd (old episode files)
        self.assertEqual(len(calls), 3)

    def test_controller_vector_equals_the_sim(self):
        """A lastcmd student on a fake robot (Controller + Guard with the integral correction, so Guard.prev != the sent
        command): its logged input and the vector fed to the network equal the sim Actor's and the online loop's for
        the same executed-command sequence (the decision before = Guard.prev of the last 5 steps), 0 at the start."""
        H = hw()
        if H.default_kin_points() is None:
            self.skipTest("kin_points_v1.npz not found")
        with tempfile.TemporaryDirectory() as d:
            ck = Path(d) / "lc.pt"
            VI.save_student(tiny(lastcmd=True), ck)
            pol = H.StudentPolicy(ck, "cpu")
        self.assertTrue(pol.lastcmd)
        fed = []
        fwd = pol.st.forward
        pol.st.forward = lambda img, robot, qd_scale=None: (fed.append(robot.clone()), fwd(img, robot, qd_scale))[1]
        A = fake_robot_run(H, pol)
        self.assertEqual(set(A), OLD_LOG_KEYS | {"qry_lastcmd"})
        prev = A["u_sim"].copy()
        prev[:, :5] -= A["i_corr"]                                       # Guard.prev from the log (class docstring)
        self.assertGreater(float(np.abs(A["i_corr"]).max()), 0.0)        # the integral acts: prev != the sent command
        steps = A["qry_step"]
        self.assertEqual(A["qry_lastcmd"].shape, (len(steps), 12))
        # the sim: Actor (fake env, u_exe = Guard.prev of the 5 steps before the decision) and the online loop
        st = pol.st
        seen = []
        st_pred = st.predict
        st.predict = lambda im, robot: (seen.append(robot.clone()), st_pred(im, robot))[1]
        actor = VI.Actor(st, Views(pol.res), device="cpu")
        env = types.SimpleNamespace(u_exe=None, decisions=0, B=1)
        actor.env = env
        actor.reset(1)
        for j, k in enumerate(steps):
            if j:
                env.u_exe = torch.as_tensor(prev[k - 5:k])[None]
                env.decisions += 1
            qk = torch.as_tensor(A["q_sim"][k])[None]
            ws = types.SimpleNamespace(q=qk, qd=torch.as_tensor(A["qd_sim"][k])[None],
                                       cube_pos=torch.zeros(1, 3, dtype=torch.float64),
                                       cube_quat=torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64))
            actor(ws, torch.zeros(1, 3), j)
            od = types.SimpleNamespace(env=env if j else types.SimpleNamespace(u_exe=None), dev=torch.device("cpu"),
                                       lc_fresh=torch.tensor([j == 0]))
            sim = VI.OnlineDAgger.lastcmd_input(od, qk)
            self.assertTrue(torch.equal(torch.as_tensor(A["qry_lastcmd"][j])[None], sim), msg=f"query {j}")
            self.assertTrue(torch.equal(fed[j][:, 12:], seen[j][:, 12:]), msg=f"query {j}")
            self.assertTrue(torch.equal(fed[j][:, 12:], sim.float()), msg=f"query {j}")
            if j:
                self.assertTrue(torch.equal(sim, by_hand(torch.as_tensor(prev[k - 5:k])[None], qk)))
        self.assertEqual(float(np.abs(A["qry_lastcmd"][0]).max()), 0.0)     # the episode start
        self.assertGreater(float(np.abs(A["qry_lastcmd"][1:, 6:]).max()), 0.0)


# ---- the config files ----------------------------------------------------------------------------------------------
class TestConfigs(unittest.TestCase):
    def test_new_config_files(self):
        base, new = (json.loads((CFG / f).read_text()) for f in ("viral_r2s_v1.json", "viral_r2s_v1_lastcmd.json"))
        self.assertEqual(set(new) - set(base), {"lastcmd"})
        self.assertTrue(new["lastcmd"] and VI.ViralConfig.from_dict(new).lastcmd)
        self.assertTrue(all(new[k] == v for k, v in base.items() if k != "note"))
        base, new = (json.loads((CFG / f).read_text()) for f in ("viral_r2s_online_relabel_place_v5.json",
                                                                 "viral_r2s_online_lastcmd_v6.json"))
        self.assertEqual(set(new) - set(base), {"require_lastcmd"})
        self.assertTrue(VI.OnlineConfig.from_dict(new).require_lastcmd)
        self.assertTrue(all(new[k] == v for k, v in base.items() if k != "note"))

    def test_require_lastcmd_refuses_an_old_student(self):
        with self.assertRaises(ValueError):
            VI.OnlineDAgger(tiny(), None, None, None, None, VI.OnlineConfig(require_lastcmd=True), "cpu")


if __name__ == "__main__":
    unittest.main()
