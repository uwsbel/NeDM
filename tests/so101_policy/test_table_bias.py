"""Tests of the real2sim v8 table randomisation (2026-10-23): the table height / tilt of the student's JOINT INPUT
(``nedm.so101_policy.table_bias``, ``viral.OnlineConfig.table_bias``) and the goal plate thickness of the training
renders (``OnlineConfig.plate_mm`` -> ``RealDR.plate_thickness``, ``SO101Scene(plate_depth=)``,
``transforms(..., plate_dz=)``, ``viral.DRViews``; ported from the studentlab round C WIP e8efd92ea).

- TableMap (CPU): config defaults / validation; the per-world planes (share, ranges, drop at x_ref); the joint map is
  exact (grasp point moved by exactly the plane height under it, same gripper rotation, only the 3 pitch joints) and
  its SIGN (a lower table -> a lower joint-implied gripper, shoulder_lift up); the inverse; at a grasp the reading =
  an independent top-down IK of the same cube on the lowered / tilted table; the applied target (observed target -
  b) reaches the true pose the student's observed-frame step aims at.
- TableOnline (CPU, jt_h110 + dev tasks; ``studentlab_ref.py``): the online loop in the kinematic world with the
  option on (hist 2 + cmd_in + lastcmd): every joint quantity of the student is in the observed frame, exactly: its q
  input Phi(q) (plane fixed in an episode, redrawn at resets; b follows the pose), its qdot J_Phi qdot, hist = earlier
  inputs, cmd / lastcmd = Phi(executed) - Phi(q), the buffer's q and labels Phi(q), Phi(teacher label); the executed
  targets of the student slots = Phi^-1(Phi(q) + predicted delta); worlds without a plane exactly as before; aux
  targets true; with q_bias the two add.
- BitIdentity (CPU, git): ``studentlab_ref.py`` (online loop + evaluation actor) with the defaults and with older
  options gives the base commit's arrays bit for bit.
- OnlineViews (CPU): ``online_views_cfg`` (training views only, the student's config unchanged), the loop refuses
  plate_mm without plate-randomised views.
- PlateGeometry (CPU, mujoco), PlateRender / PlateViews / RenderBitIdentity (GPU, the v2 scene): the plate mesh depth,
  shift and the cube's visual lift; renders; the base commit's renders bit for bit with the option off.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_table_bias -v
"""
from __future__ import annotations

import importlib.util
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

from nedm.so101_policy import table_bias as TB
from nedm.so101_policy.kin import SO101Kin
from nedm.so101_policy.vision import viral as VI

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import studentlab_ref as R  # noqa: E402

BASE = os.environ.get("NEDM_V8_TABLE_BASE", "39d2e5b72")      # the commit before the v8 table change
SCENE_V2 = Path(os.environ.get("NEDM_R2S_SCENE", "~/so101_r2s_claude/scene/real_scene_v2.json")).expanduser()
CALIB = Path(os.environ.get("NEDM_R2S_CALIB", "~/so101_r2s_claude/calib/calib_v1.json")).expanduser()
HAVE_MUJOCO = importlib.util.find_spec("mujoco") is not None
GPU = (importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None
       and torch.cuda.is_available())
HAVE_SCENE = SCENE_V2.exists() and CALIB.exists()
PLANE_REAL = (-0.00468, -0.03641, 0.01836)     # calib_v1 scene.table_plane (the measured real table)
f64 = torch.float64


def workspace_poses(n=1500, seed=0, z=(0.01, 0.09)):
    """Top-down IK poses of the grasp point over the workspace (0.12-0.32 m from the pan axis, pan +-60 deg)."""
    kin = SO101Kin()
    g = torch.Generator().manual_seed(seed)
    r = 0.12 + 0.20 * torch.rand(n, generator=g, dtype=f64)
    ph = (torch.rand(n, generator=g, dtype=f64) * 2 - 1) * math.radians(60)
    pos = torch.stack((0.0388 + r * torch.cos(ph), r * torch.sin(ph),
                       z[0] + (z[1] - z[0]) * torch.rand(n, generator=g, dtype=f64)), -1)
    yaw = ph + (torch.rand(n, generator=g, dtype=f64) - 0.5) * math.pi / 2
    q5, ok, _, _ = kin.ik_topdown(None, pos, yaw)
    jaw = 0.6 * torch.rand(n, 1, generator=g, dtype=f64)
    return kin, torch.cat((q5, jaw), -1)[ok], pos[ok], yaw[ok]


def rot_angle(r1, r0):
    rel = r0.transpose(-1, -2) @ r1
    return torch.acos(((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1.0, 1.0))


class TableMap(unittest.TestCase):
    def test_defaults_and_validation(self):
        o = VI.OnlineConfig()
        self.assertEqual((o.table_bias, o.plate_mm), ({}, ()))
        c = TB.TableBiasConfig.from_dict({})
        self.assertEqual((c.p, c.drop_mm, c.slope_x, c.slope_y, c.x_ref_m), (0.7, (0.0, 20.0), (-40.0, 40.0),
                                                                           (0.0, 0.0), 0.25))
        self.assertEqual(TB.TableBiasConfig.from_dict(c.to_dict()), c)
        for bad in ({"p": 1.5}, {"drop_mm": [20, 0]}, {"slope_x": [1.0]}, {"foo": 1}, {"drop_mm": [0, 80]},
                    {"slope_y": [-500, 0]}):
            with self.assertRaises(ValueError, msg=bad):
                TB.TableBiasConfig.from_dict(bad)
        with self.assertRaises(ValueError):
            VI.OnlineConfig.from_dict({"table_bias": {"p": -0.1}})
        for bad in ([5, 1], [0, 3], [1], [1, 30]):
            with self.assertRaises(ValueError, msg=bad):
                VI.OnlineConfig.from_dict({"plate_mm": bad})
        o = VI.OnlineConfig.from_dict({"table_bias": {"p": 0.7}, "plate_mm": [1, 5]})
        self.assertEqual((o.table_bias, o.plate_mm), ({"p": 0.7}, (1.0, 5.0)))

    def test_planes(self):
        cfg = TB.TableBiasConfig(p=0.7, slope_y=(-20.0, 20.0))
        pl, on = TB.sample(20000, cfg, torch.Generator().manual_seed(0))
        self.assertEqual(pl.dtype, f64)
        self.assertAlmostEqual(float(on.float().mean()), 0.7, delta=0.015)
        self.assertEqual(float(pl[~on].abs().max()), 0.0)
        drop = -(pl[on, 0] + pl[on, 1] * 0.25)                         # the drop at x_ref, y = 0
        self.assertGreaterEqual(float(drop.min()), -1e-12)
        self.assertLessEqual(float(drop.max()), 0.020 + 1e-12)
        self.assertGreater(float(drop.max()), 0.0195)
        self.assertLessEqual(float(pl[on, 1].abs().max()), 0.040 + 1e-12)
        self.assertGreater(float(pl[on, 1].abs().max()), 0.039)
        self.assertLessEqual(float(pl[on, 2].abs().max()), 0.020 + 1e-12)
        # the measured real table is inside the default ranges: drop 13.8 mm at x 0.25, slope -36.4 mm/m
        real_drop = -(PLANE_REAL[0] + PLANE_REAL[1] * 0.25)
        self.assertTrue(0.0 <= real_drop <= 0.020 and abs(PLANE_REAL[1]) <= 0.040, real_drop)
        pl2, _ = TB.sample(20000, cfg, torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(pl, pl2))                          # deterministic per generator seed
        self.assertEqual(float(TB.sample(100, TB.TableBiasConfig(p=0.0), torch.Generator())[0].abs().max()), 0.0)
        c = TB.plane_coef(torch.tensor([0.01], dtype=f64), -0.03, 0.0, 0.25)     # 10 mm lower at x 0.25
        self.assertAlmostEqual(float(TB.height(c[0], torch.tensor([0.25, 0.1], dtype=f64))), -0.01, places=12)
        self.assertAlmostEqual(float(TB.height(c[0], torch.tensor([0.35, 0.0], dtype=f64))), -0.013, places=12)

    def test_shift_exact_and_sign(self):
        kin, q, _, _ = workspace_poses()
        self.assertGreater(len(q), 800)
        p0, r0 = kin.fk(q)
        for dz in (-0.030, -0.020, -0.0125, -0.005, 0.005):
            qo, res = TB.shift(kin, q, torch.full((len(q),), dz, dtype=f64))
            p1, r1 = kin.fk(qo)
            d = p1 - p0
            self.assertLess(float((d[:, 2] - dz).abs().max()), 1e-8, dz)       # exactly dz along world z ...
            self.assertLess(float(d[:, :2].norm(dim=-1).max()), 1e-8, dz)      # ... no horizontal motion
            self.assertLess(float(rot_angle(r1, r0).max()), 1e-7, dz)           # the same gripper rotation
            self.assertLess(float(res.max()), 1e-8, dz)
            self.assertEqual(float((qo - q)[:, [0, 4, 5]].abs().max()), 0.0)   # pan, roll, jaw untouched
            if dz < 0:      # SIGN: a lower table -> the reading is lower -> shoulder_lift up (the table test's
                self.assertTrue(bool((d[:, 2] < 0).all()))                      # compensation was -2.09 deg)
                self.assertGreater(float((qo - q)[:, 1].median()), 0.0)
        qo, _ = TB.shift(kin, q, torch.zeros(len(q), dtype=f64))
        self.assertTrue(torch.equal(qo, q))                                     # dz 0: exactly q
        # far outside the task poses (uniform over the joint range) the map may not converge: then it returns its
        # best iterate (never worse than q itself, residual <= |dz|), never a diverged pose; non-finite rows pass
        from nedm.so101_policy.kin import THI, TLO
        lo, hi = torch.tensor(TLO, dtype=f64), torch.tensor(THI, dtype=f64)
        qu = lo + (hi - lo) * torch.rand(4000, 6, generator=torch.Generator().manual_seed(9), dtype=f64)
        qu[0] = float("nan")
        qo, res = TB.shift(kin, qu, torch.full((4000,), -0.02, dtype=f64))
        self.assertTrue(bool(torch.isnan(qo[0]).all()) and float(res[0]) == 0.0)
        self.assertLessEqual(float(res[1:].max()), 0.02 + 1e-12)
        self.assertGreater(float((res[1:] < 1e-8).float().mean()), 0.65)          # (measured 0.73)
        # dz -20 mm at the grasp heights: shoulder_lift / elbow / wrist_flex offsets about +2.5 / +5.8 / -8.5 deg
        b = torch.rad2deg(TB.shift(kin, q, torch.full((len(q),), -0.02, dtype=f64))[0] - q)[:, 1:4].median(0).values
        self.assertTrue(np.allclose(b.numpy(), [2.5, 5.8, -8.45], atol=0.6), b)

    def test_observe_tilted_and_inverse(self):
        kin, q, _, _ = workspace_poses(seed=1)
        n = len(q)
        g = torch.Generator().manual_seed(2)
        pl, _ = TB.sample(n, TB.TableBiasConfig(p=1.0, slope_y=(-20.0, 20.0)), g)
        pl[0] = torch.tensor(PLANE_REAL, dtype=f64)
        qo, res = TB.observe(kin, q, pl)
        p0, r0 = kin.fk(q)
        p1, r1 = kin.fk(qo)
        h = TB.height(pl, p0[:, :2])                                           # the plane under the TRUE gripper
        self.assertLess(float(((p1 - p0)[:, 2] - h).abs().max()), 1e-8)
        self.assertLess(float((p1 - p0)[:, :2].norm(dim=-1).max()), 1e-8)
        self.assertLess(float(rot_angle(r1, r0).max()), 1e-7)
        self.assertLess(float(res.max()), 1e-8)
        # the measured real plane under a grasp at (0.25, 0): 13.8 mm lower
        self.assertAlmostEqual(float(TB.height(torch.tensor(PLANE_REAL, dtype=f64),
                                               torch.tensor([0.25, 0.0], dtype=f64))), -0.0138, places=4)
        back, _ = TB.to_true(kin, qo, pl)
        self.assertLess(float((back - q).abs().max()), 1e-9)                   # exact inverse

    def test_grasp_matches_real_ik(self):
        """At a top-down grasp of a cube on the sim table, the student's reading = the joints a real arm needs for the
        SAME cube on a table lying on the plane (independent IK, hinted by the sim pose)."""
        kin = SO101Kin()
        g = torch.Generator().manual_seed(3)
        n = 400
        r = 0.15 + 0.15 * torch.rand(n, generator=g, dtype=f64)
        ph = (torch.rand(n, generator=g, dtype=f64) * 2 - 1) * math.radians(45)
        xy = torch.stack((0.0388 + r * torch.cos(ph), r * torch.sin(ph)), -1)
        yaw = ph + (torch.rand(n, generator=g, dtype=f64) - 0.5) * math.pi / 2
        for plane in ((-0.0125, 0.0, 0.0), PLANE_REAL, (-0.02 + 0.04 * 0.25, -0.04, 0.0)):
            pl = torch.tensor(plane, dtype=f64).expand(n, 3)
            sim = torch.cat((xy, torch.full((n, 1), 0.015, dtype=f64)), -1)          # grasp point at the cube centre
            q5, ok, _, _ = kin.ik_topdown(None, sim, yaw)
            real = sim.clone()
            real[:, 2] += TB.height(pl, xy)                                            # the cube on the lower table
            q5r, okr, epr, _ = kin.ik_topdown(q5, real, yaw, seeds=False)
            m = ok & okr
            self.assertGreater(int(m.sum()), 300, plane)
            qs = torch.cat((q5, torch.zeros(n, 1, dtype=f64)), -1)[m]
            qo, _ = TB.observe(kin, qs, pl[m])
            dq = (qo[:, :5] - q5r[m]).abs()
            self.assertLess(float(dq.max()), 2e-4, (plane, float(dq.max())))           # IK tolerance (1e-6 m)
            self.assertLess(float((kin.fk(qo)[0] - real[m]).norm(dim=-1).max()), 1e-6)
            d = qo - qs             # a lower table: elbow up, wrist_flex down at every grasp, shoulder_lift up at
            self.assertTrue(bool((d[:, 2] > 0).all() and (d[:, 3] < 0).all()), plane)  # most (not at full reach)
            self.assertGreater(float((d[:, 1] > 0).float().mean()), 0.95, plane)

    def test_applied_target(self):
        """The student at its reading q_obs = Phi(q) commands an observed-frame target u_obs = q_obs + delta (here: the
        reading of the true pose 10 mm lower, the descent it aims at). The loop applies Phi^-1(u_obs): exactly that
        true pose, and the observed step equals the world's step. A constant offset per decision (u = u_obs - b(q) =
        q + delta) would be ~10 % off on this vertical move (the reason the loop maps every pose)."""
        kin, q, pos, yaw = workspace_poses(seed=4, z=(0.025, 0.08))
        n = len(q)
        pl = torch.tensor(PLANE_REAL, dtype=f64).expand(n, 3)
        qo, _ = TB.observe(kin, q, pl)
        aim, _ = TB.shift(kin, q, torch.full((n,), -0.010, dtype=f64))              # true pose 10 mm lower
        u_obs, _ = TB.observe(kin, aim, pl)                                          # the student's target, its frame
        u, _ = TB.to_true(kin, u_obs, pl)                                            # applied (the loop's rule)
        self.assertLess(float((u - aim).abs().max()), 1e-9)
        d_obs = kin.fk(u_obs)[0] - kin.fk(qo)[0]
        d_true = kin.fk(u)[0] - kin.fk(q)[0]
        self.assertLess(float((d_obs - d_true).norm(dim=-1).max()), 1e-8)
        approx = (kin.fk(u_obs - (qo - q))[0] - kin.fk(aim)[0]).norm(dim=-1)         # one offset per decision
        self.assertTrue(0.5e-3 < float(approx.median()) < 1.6e-3, float(approx.median()))
        # qdot reading: J_Phi qd (forward difference) = a central difference of Phi; qd 0 -> exactly 0
        qd = (torch.rand(n, 6, generator=torch.Generator().manual_seed(5), dtype=f64) - 0.5) * 2.0
        qd[:5] = 0.0
        v = TB.observe_qd(kin, q, qd, pl)
        h = 1e-3
        c = (TB.observe(kin, q + qd * h, pl)[0] - TB.observe(kin, q - qd * h, pl)[0]) / (2 * h)
        self.assertLess(float((v - c).abs().max()), 1e-3 * float(qd.abs().max()))
        self.assertEqual(float(v[:5].abs().max()), 0.0)
        self.assertGreater(float((v - qd)[:, 1:4].abs().max()), 0.01)                 # J_Phi is not the identity
        big = qd[5:6] * 1e6                       # a huge qdot: the difference step shrinks, J_Phi qd stays linear
        vb = TB.observe_qd(kin, q[5:6], big, pl[5:6])
        self.assertTrue(bool(torch.isfinite(vb).all()))
        self.assertLess(float((vb / 1e6 - v[5:6]).abs().max()), 1e-2 * float(qd[5].abs().max()))

    def test_reach_limit_fallback(self):
        """A student target near the reach limit (online smoke, s28 in the NRD: arm almost straight, wrist above the
        shoulder) whose sim preimage would be 14.7 mm higher at the same gripper rotation: out of reach, the
        inverse does not converge; ``OnlineDAgger.table_applied`` then applies u_obs - b of the decision."""
        import types
        kin = SO101Kin()
        u_obs = torch.tensor([[0.402166, 0.611313, -0.835384, 1.423152, -1.033823, 0.681594]], dtype=f64)
        pl = torch.tensor([[-0.0147, 0.0, 0.0]], dtype=f64)
        u, res = TB.to_true(kin, u_obs, pl)
        self.assertGreater(float(res[0]), TB.TOL)
        self.assertLessEqual(float(res[0]), 0.0147 + 1e-12)                       # never worse than no shift
        b = torch.tensor([[0.0, 0.03, 0.08, -0.11, 0.0, 0.0]], dtype=f64)      # the decision's offset b(q)
        ns = types.SimpleNamespace(tb_kin=kin, tb_plane=pl, tb_note=lambda r: None)
        out = VI.OnlineDAgger.table_applied(ns, u_obs[:, None, :].expand(1, 2, 6), torch.tensor([0]), b)
        self.assertTrue(torch.equal(out[0], (u_obs - b).expand(2, 6)))
        ok = torch.tensor([[0.3, 0.6, 0.8, 0.9, 0.0, 0.2]], dtype=f64)          # a task pose: the exact inverse
        o2 = VI.OnlineDAgger.table_applied(types.SimpleNamespace(tb_kin=kin, tb_plane=pl, tb_note=lambda r: None),
                                           ok[:, None, :], torch.tensor([0]), b)
        self.assertLess(float((o2[0, 0] - TB.to_true(kin, ok, pl)[0][0]).abs().max()), 1e-12)


@unittest.skipUnless(R.HAVE, "jt_h110 / dev tasks missing")
class TableOnline(unittest.TestCase):
    """The online loop with ``table_bias`` on (kinematic world, 16 worlds, staggered clocks, resets)."""

    def test_input_targets_labels_aux_hist_cmd(self):
        H, N = 2, 16
        st = VI.expand_lastcmd(VI.expand_student(R.tiny_student(0), hist=H, cmd_in=True, device="cpu"), "cpu")
        od, views = R.make_online(st, dict(horizon=40, buffer=4000, table_bias=dict(p=0.7, slope_y=[-20, 20])), 0, N)
        self.assertIsNotNone(od.tb)
        kin = SO101Kin()
        A = torch.arange(N)
        od.env.reset()
        od._draw(A), od._lag_reset(A), od._bias_reset(A), od._qd_reset(A)
        sent = []
        orig = od.env.step

        def spy(actions=None, targets=None, use_targets=None, step_hook=None):
            sent.append(targets.clone())
            return orig(actions, targets=targets, use_targets=use_targets, step_hook=step_hook)

        od.env.step = spy
        rec = []
        P = 12 * (1 + H)
        for t in range(36):
            od.it = t
            pre = dict(ep=od.env.ep_count.clone(), q=od.env.ws.q.float().clone(), qd=od.env.ws.qd.clone(),
                       obs=od.env.obs.clone(), plane=od.tb_plane.clone(), on=od.tb_on.clone(), ptr=od.buf.ptr,
                       cube=od.env.ws.cube_pos.float().clone(), u=None if od.env.u_exe is None else
                       od.env.u_exe.clone(), goal=torch.cat((od.env.task.goal_xy, od.env.task.goal_yaw[:, None]),
                                                            -1).float())
            od.act()
            last = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in od.last.items()}
            si = (~od.teacher).nonzero().flatten()
            st.eval()
            with torch.no_grad():
                delta = VI.denormalise(st, st.predict(VI.to_input(last["rgb"][si]), last["robot"][si]))
            st.train()
            rec.append(dict(pre=pre, last=last, delta=delta, si=si, sent=sent[-1],
                            slots=(pre["ptr"] + torch.arange(N)) % od.buf.cap))
            if od.buf.n >= 16 and t % 3 == 0:
                od.train_steps()
        on_seen, resets_changed, b_moved, n_off_checked = 0, 0, 0, 0
        n_exec, K = st.cfg.n_exec, st.cfg.chunk
        jj = torch.arange(n_exec)
        kk = (jj + st.cfg.jaw_lead).clamp_max(K - 1)
        for t, d in enumerate(rec):
            pre, last = d["pre"], d["last"]
            rob, fresh, TF = last["robot"], last["fresh"], last["tb"]
            b, q, on, pl = TF["b"], pre["q"], pre["on"], pre["plane"]
            off = ~on
            on_seen += int(on.sum())
            if bool(off.any()):
                self.assertEqual(float(pl[off].abs().max()), 0.0)
                self.assertEqual(float(b[off].abs().max()), 0.0)
            # b = Phi(q) - q of this decision's true q under the world's plane (independent recomputation)
            qo, _ = TB.observe(kin, q.double(), pl)
            self.assertLess(float((b.double() - (qo - q.double())).abs().max()), 1e-6, t)
            # the q input: Phi(q); its FK is the true FK moved by the plane height under the gripper
            self.assertTrue(torch.equal(rob[:, :6], q + b), t)
            p0, r0 = kin.fk(q.double())
            p1, r1 = kin.fk(rob[:, :6].double())
            self.assertLess(float(((p1 - p0)[:, 2] - TB.height(pl, p0[:, :2])).abs().max()), 2e-6, t)
            self.assertLess(float(rot_angle(r1, r0).max()), 2e-5, t)
            # the qdot input: J_Phi qd (worlds without a plane: the true qd exactly)
            qd = pre["qd"].float().nan_to_num()
            self.assertTrue(torch.equal(rob[:, 6:12], TF["qd"]), t)
            if bool(off.any()):
                self.assertTrue(torch.equal(TF["qd"][off], qd[off]), t)
            v = TB.observe_qd(kin, q.double(), qd.double(), pl)
            self.assertLess(float((TF["qd"].double() - v).abs().max()), 1e-4 + 1e-3 * float(qd.abs().max()), t)
            if t > 0:
                same_ep = pre["ep"] == rec[t - 1]["pre"]["ep"]
                self.assertTrue(torch.equal(pl[same_ep], rec[t - 1]["pre"]["plane"][same_ep]), t)
                resets_changed += int((~same_ep & (pl != rec[t - 1]["pre"]["plane"]).any(-1)).sum())
                moved = same_ep & on & ((q - rec[t - 1]["pre"]["q"]).abs().amax(-1) > 1e-3)
                b_moved += int((moved & ((b - rec[t - 1]["last"]["tb"]["b"]).abs().amax(-1) > 1e-6)).sum())
            # hist blocks: the same episode's earlier (observed-frame) inputs
            for w in range(N):
                s_ = t
                while s_ > 0 and torch.equal(rec[s_ - 1]["pre"]["ep"][w], pre["ep"][w]):
                    s_ -= 1
                for k in range(1, H + 1):
                    src = max(t - k, s_)
                    self.assertTrue(torch.equal(rob[w, 12 * k:12 * (k + 1)], rec[src]["last"]["robot"][w, :12]),
                                    (t, w, k))
            # cmd: Phi(executed) - Phi(q), 0 at a start; lastcmd: Phi(u_last) - Phi(q), (Phi(u_last) - Phi(u_prev)) / dt
            cmd = rob[:, P:P + 30]
            if t > 0:
                ue, _ = TB.observe(kin, pre["u"].reshape(-1, 6).double(), pl.repeat_interleave(5, 0))
                ue = ue.reshape(N, 5, 6)
                want = (ue - (q + b).double()[:, None, :]).reshape(N, 30)
                self.assertLess(float((cmd[~fresh].double() - want[~fresh]).abs().max()), 1e-5, t)
                lc = rob[:, P + 30:P + 42].double()
                self.assertLess(float((lc[~fresh, :6] - (ue[:, -1] - qo)[~fresh]).abs().max()), 1e-5, t)
                self.assertLess(float((lc[~fresh, 6:] - ((ue[:, -1] - ue[:, -2]) / 0.02)[~fresh]).abs().max()), 1e-3,
                                t)
                if bool((off & ~fresh).any()):          # no plane: the old inputs bit for bit
                    m = off & ~fresh
                    old = (pre["u"].float() - q[:, None, :]).reshape(N, 30)
                    self.assertTrue(torch.equal(cmd[m], old[m]), t)
            self.assertEqual(float(cmd[fresh].abs().max()) if bool(fresh.any()) else 0.0, 0.0)
            # executed targets of the student slots: Phi^-1(reading + delta); no plane: q + delta exactly
            si = d["si"]
            u_obs = (q + b)[si][:, None, :] + d["delta"]
            u_obs = torch.cat((u_obs[:, jj, :5], u_obs[:, kk, 5:]), -1).double()
            ex, _ = TB.to_true(kin, u_obs.reshape(-1, 6), pl[si].repeat_interleave(n_exec, 0))
            self.assertLess(float((d["sent"][si] - ex.reshape(u_obs.shape)).abs().max()), 1e-6, t)
            back, _ = TB.observe(kin, d["sent"][si].reshape(-1, 6), pl[si].repeat_interleave(n_exec, 0))
            self.assertLess(float((back.reshape(u_obs.shape) - u_obs).abs().max()), 1e-6, t)
            so = off[si]
            if bool(so.any()):
                self.assertTrue(torch.equal(d["sent"][si][so], u_obs[so]), t)
                n_off_checked += int(so.sum())
            moved_on = on[si] & ((u_obs - q[si][:, None, :].double()).abs().amax((1, 2)) > 1e-3)
            if bool(moved_on.any()):            # not the old rule (q + delta) where a plane is on
                naive = q[si][moved_on][:, None, :].double() + (u_obs[moved_on] - (q + b)[si][moved_on][:, None, :])
                self.assertGreater(float((d["sent"][si][moved_on] - naive).abs().amax((1, 2)).min()), 1e-6, t)
            # buffer: q = Phi(q), labels = Phi(teacher label on the TRUE state), aux targets of the true cube
            sl = d["slots"]
            self.assertTrue(torch.equal(od.buf.q[sl], q + b), t)
            self.assertTrue(torch.equal(od.buf.robot[sl], rob), t)
            a = od.jt.act(pre["obs"]).to(torch.float64)
            lab = od.env.codec.targets(a, q.double())
            lo, _ = TB.observe(kin, lab.reshape(-1, 6), pl.repeat_interleave(5, 0))
            self.assertLess(float((od.buf.lab1[sl].double() - lo.reshape(lab.shape)).abs().max()), 1e-5, t)
            if bool(off.any()):
                self.assertLess(float((od.buf.lab1[sl][off] - lab[off].float()).abs().max()), 1e-5, t)
            self.assertTrue(torch.equal(od.buf.aux[sl], VI.aux_targets(pre["goal"], pre["cube"]).float()), t)
        self.assertGreater(on_seen, 0.5 * N * len(rec))
        self.assertLess(on_seen, 0.9 * N * len(rec))
        self.assertGreater(resets_changed, 5)
        self.assertGreater(b_moved, 20)                    # b follows the pose inside an episode
        self.assertGreater(n_off_checked, 10)
        ex = od.log_extra()
        for k in ("tb_on", "tb_drop_mm", "tb_res_um", "tb_bad"):
            self.assertIn(k, ex)
        self.assertTrue(0.0 < ex["tb_on"] <= 1.0)
        self.assertLess(ex["tb_res_um"], 1.0)
        self.assertEqual(ex["tb_bad"], 0.0)

    def test_q_bias_and_table_bias_add(self):
        st = R.tiny_student(0)
        od, _ = R.make_online(st, dict(q_bias_deg=[0, 1, 1, 1, 0, 0], q_bias_p=1.0, table_bias=dict(p=1.0)), 0, 8)
        od2, _ = R.make_online(st, dict(q_bias_deg=[0, 1, 1, 1, 0, 0], q_bias_p=1.0), 0, 8)
        for o in (od, od2):
            o.it = 0
            o.act()
        self.assertTrue(torch.equal(od.qb, od2.qb))                                   # q_bias draws unchanged
        self.assertGreater(float(od.qb.abs().max()), 1e-3)
        self.assertTrue(torch.equal(od.last["robot"][:, :6], od2.last["robot"][:, :6] + od.last["tb"]["b"]))
        self.assertGreater(float(od.last["tb"]["b"].abs().max()), 1e-3)
        self.assertTrue(torch.equal(od.buf.q[:8], od.last["tb"]["q"]))     # the buffer's q: Phi(q), no q_bias


@unittest.skipUnless(R.HAVE and shutil.which("git"), "jt_h110 / dev tasks / git missing")
class BitIdentity(unittest.TestCase):
    """studentlab_ref.py (online loop with training, evaluation actor) = the base commit's arrays with the v8 options
    off."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="v8table_base_"))
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

    def check(self, tag, cfg, ocfg, ocfg_new=None):
        if not self.ok:
            self.skipTest(f"git archive {BASE} failed")
        a = self.run_ref(self.tmp / "src", self.tmp / f"base_{tag}.npz", cfg, ocfg)
        b = self.run_ref(REPO / "src", self.tmp / f"new_{tag}.npz", cfg, ocfg if ocfg_new is None else ocfg_new)
        self.assertEqual(set(a.files), set(b.files))
        bad = [k for k in a.files if not np.array_equal(a[k], b[k], equal_nan=True)]
        self.assertEqual(bad, [], f"{tag}: outputs differ from {BASE}")

    def test_defaults(self):
        self.check("default", {}, {})

    def test_existing_options(self):
        self.check("opts", {"hist": 3, "cmd_in": True, "lag": 2, "lag_range": [1, 3]},
                   {"qd_robot_p": 0.5, "q_bias_deg": [1, 2, 2, 2, 2, 0], "q_bias_p": 0.5, "qd_noise_zero_p": 0.3,
                    "nominal_share": 0.3, "goal_jump": {"p": 1.0}, "light_redraw_p": 0.5})

    def test_explicit_off(self):        # the new keys present but off: the same arrays as the base defaults
        self.check("off", {"hist": 2, "cmd_in": True}, {"qd_robot_p": 0.5},
                   {"qd_robot_p": 0.5, "table_bias": {}, "plate_mm": []})

    def test_plane_p0(self):            # on with p 0: no world gets a plane -> the base arrays bit for bit
        self.check("p0", {"hist": 2, "cmd_in": True}, {"qd_robot_p": 0.5},
                   {"qd_robot_p": 0.5, "table_bias": {"p": 0.0}})


class OnlineViews(unittest.TestCase):
    def test_views_cfg(self):
        from dataclasses import asdict, replace
        cfg = replace(VI.ViralConfig(), scene="/x/scene.json", real_dr={"wall_xy": 0.02})
        o0 = VI.OnlineConfig()
        self.assertIs(VI.online_views_cfg(cfg, o0), cfg)
        o1 = VI.OnlineConfig.from_dict({"plate_mm": [1, 5]})
        v = VI.online_views_cfg(cfg, o1)
        self.assertIsNot(v, cfg)
        self.assertEqual(v.real_dr, {"wall_xy": 0.02, "plate_thickness": [0.001, 0.005]})
        self.assertEqual(cfg.real_dr, {"wall_xy": 0.02})                               # the student keeps its own
        d0, d1 = asdict(cfg), asdict(v)
        self.assertEqual({k for k in d0 if d0[k] != d1[k]}, {"real_dr"})
        with self.assertRaises(ValueError):                                            # sim scene: no plate DR
            VI.online_views_cfg(VI.ViralConfig(), o1)

    @unittest.skipUnless(R.HAVE, "jt_h110 / dev tasks missing")
    def test_loop_refuses_views_without_plate(self):
        with self.assertRaises(ValueError):
            R.make_online(R.tiny_student(0), dict(plate_mm=[1, 5]), 0, 4)            # FakeViews: no plate_on

    def test_online_script_uses_training_views_cfg(self):
        src = (REPO / "scripts/so101_policy/vision_viral.py").read_text()
        i = src.index("def cmd_online")
        body = src[i:src.index("\ndef ", i + 10)]
        self.assertIn("vcfg = VI.online_views_cfg(st.cfg, ocfg)", body)
        self.assertIn("views = VI.make_views(vcfg, ocfg.envs", body)
        self.assertIn("eviews = VI.make_views(st.cfg, a.render_worlds", body)


@unittest.skipUnless(HAVE_MUJOCO, "the arm meshes need mujoco")
class PlateGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nedm.render.so101 import RealScene, SO101Scene, load_assets
        cls.assets = load_assets()
        cls.real = RealScene()
        cls.SO101Scene = SO101Scene
        cls.s0 = SO101Scene(cls.assets, real=cls.real, cameras=("external", "wrist"), goal="block")
        cls.s1 = SO101Scene(cls.assets, real=cls.real, cameras=("external", "wrist"), goal="block", plate_depth=0.0045)

    def test_defaults_off(self):
        from nedm.render.so101 import RealDR
        self.assertEqual(RealDR().plate_thickness, ())
        self.assertFalse(RealDR().plate_on)
        self.assertIsNone(RealDR().plate_depth())
        self.assertIsNone(self.s0.plate_depth)
        with self.assertRaises(ValueError):
            RealDR(plate_thickness=(0.004, 0.001)).plate_depth()
        with self.assertRaises(ValueError):                         # the sim scene has no plate depth
            self.SO101Scene(self.assets, plate_depth=0.004)

    def test_mesh_and_shift(self):
        i = self.s0.body_names.index("goal")
        v0, v1 = self.s0.scene.bodies[i].vertices, self.s1.scene.bodies[i].vertices
        pt = self.real.plate_thickness
        self.assertAlmostEqual(float(v0[:, 2].min()), 0.0, places=7)
        self.assertAlmostEqual(float(v1[:, 2].max()), pt, places=7)          # the same top
        self.assertAlmostEqual(float(v1[:, 2].min()), pt - 0.0045, places=7)
        n = 6
        q = torch.zeros(n, 6, dtype=f64)
        goal = torch.tensor([[0.25, 0.0, 0.0]] * n, dtype=f64)
        goal[5] = float("nan")                                                  # hidden plate
        cp = torch.tensor([[0.25, 0.0, 0.015], [0.25, 0.0, 0.017], [0.25, 0.0, 0.0185], [0.285, 0.0, 0.015],
                           [0.25, 0.0, 0.05], [0.25, 0.0, 0.015]], dtype=f64)
        cq = torch.tensor([[1.0, 0, 0, 0]] * n, dtype=f64)
        b0, c0 = self.s1.transforms(q, cp, cq, goal)
        bo, co = self.s0.transforms(q, cp, cq, goal)
        self.assertTrue(torch.equal(b0, bo) and torch.equal(c0, co))           # no plate_dz: as the old scene
        dz = torch.full((n,), 0.0035, dtype=f64)
        b1, c1 = self.s1.transforms(q, cp, cq, goal, plate_dz=dz)
        ci = self.s1.body_names.index("cube")
        self.assertTrue(torch.allclose(b1[:5, i, 2] - b0[:5, i, 2], torch.full((5,), 0.0035), atol=1e-7))
        self.assertEqual(float(b1[5, i, 2]), float(b0[5, i, 2]))               # hidden stays hidden
        lift = (b1[:, ci, 2] - b0[:, ci, 2]).double()
        # resting (bottom 0) -> +3.5 mm; bottom 2 mm -> +1.5; bottom 3.5 -> 0; 35 mm off centre (1/3 over) -> 1.17;
        # in the air -> 0; hidden plate -> 0
        want = torch.tensor([0.0035, 0.0015, 0.0, 0.0035 / 3, 0.0, 0.0], dtype=f64)
        self.assertTrue(torch.allclose(lift, want, atol=2e-6), lift)
        self.assertTrue(torch.equal(c1, c0))                                    # cameras unchanged
        with self.assertRaises(ValueError):                                     # plate_dz needs plate_depth
            self.s0.transforms(q, cp, cq, goal, plate_dz=dz)

    def test_cube_never_deeper_than_nominal(self):
        """Over the plate, any thickness 1-4.5 mm and any cube height / small tilt: the drawn cube sinks into the
        drawn plate no deeper than into the nominal plate (the plate thickness, plus what the physics cube is below
        the table)."""
        g = torch.Generator().manual_seed(0)
        n = 4000
        th = 0.001 + 0.0035 * torch.rand(n, generator=g, dtype=f64)
        pt = self.real.plate_thickness
        goal = torch.stack((0.2 + 0.1 * torch.rand(n, generator=g, dtype=f64),
                            0.2 * (torch.rand(n, generator=g, dtype=f64) - 0.5),
                            torch.rand(n, generator=g, dtype=f64) * 3), -1)
        off = (torch.rand(n, 2, generator=g, dtype=f64) - 0.5) * 0.04             # centre within 20 mm
        ax = torch.nn.functional.normalize(torch.randn(n, 3, generator=g, dtype=f64), dim=-1)
        ang = math.radians(8) * torch.rand(n, generator=g, dtype=f64)
        cq = torch.cat((torch.cos(ang / 2)[:, None], torch.sin(ang / 2)[:, None] * ax), -1)
        bottom0 = 0.010 * torch.rand(n, generator=g, dtype=f64) - 0.001
        w, x, y, z = cq.unbind(-1)
        rz = torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)
        cz = bottom0 + 0.015 * rz.abs().sum(-1)
        cp = torch.cat((goal[:, :2] + off, cz[:, None]), -1)
        b1, _ = self.s1.transforms(torch.zeros(n, 6), cp, cq, goal, plate_dz=th - pt)
        ci, gi = self.s1.body_names.index("cube"), self.s1.body_names.index("goal")
        drawn_bottom = b1[:, ci, 2].double() - 0.015 * rz.abs().sum(-1)
        top = b1[:, gi, 2].double() + pt
        inside = off.abs().max(-1).values <= 0.025 - 0.015                    # footprint fully over the plate
        sink = (top - drawn_bottom)[inside]
        nominal_sink = (pt - bottom0)[inside]                                   # the nominal scene: plate top at pt
        self.assertLessEqual(float((sink - nominal_sink.clamp_min(pt)).max()), 2e-6)
        self.assertGreater(float((th[inside] - bottom0[inside] > pt + 1e-4).float().mean()), 0.2)   # lifts happened
        hi = inside & (bottom0 > th)        # a cube whose bottom is above the drawn plate top is drawn where it is
        self.assertTrue(torch.allclose(b1[hi, ci, 2].double(), cp[hi, 2], atol=1e-6))

    def test_draws_after_the_old_ones(self):
        from nedm.render.so101 import RealDR
        a = RealDR(wall_xy=0.02).draw(self.s1, 64, torch.Generator().manual_seed(0))
        b = RealDR(wall_xy=0.02, plate_thickness=(0.001, 0.0045)).draw(self.s1, 64, torch.Generator().manual_seed(0))
        self.assertEqual(set(b) - set(a), {"plate_dz"})
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)
        pt = self.real.plate_thickness
        self.assertGreaterEqual(float(b["plate_dz"].min()), 0.001 - pt - 1e-12)
        self.assertLessEqual(float(b["plate_dz"].max()), 0.0045 - pt + 1e-12)
        self.assertGreater(float(b["plate_dz"].std()), 0.0007)
        with self.assertRaises(ValueError):                                    # renderer without plate_depth
            RealDR(plate_thickness=(0.001, 0.0045)).draw(self.s0, 2, torch.Generator().manual_seed(0))
        with self.assertRaises(ValueError):                                    # deeper than the renderer's mesh
            RealDR(plate_thickness=(0.001, 0.006)).draw(self.s1, 2, torch.Generator().manual_seed(0))


@unittest.skipUnless(GPU and HAVE_SCENE, "needs newton on CUDA and the v2 scene")
class PlateRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nedm.render.so101 import RealScene, SO101Renderer
        d = json.loads(SCENE_V2.read_text())
        cls.real = RealScene.from_scene_dict(d, calib=json.loads(CALIB.read_text()), tone=None, image_fx={})
        kw = dict(backend="newton", cameras=("external", "wrist", "top"), goal="block", real=cls.real, segment=True)
        cls.r1 = SO101Renderer(4, 224, 224, plate_depth=0.0045, **kw)
        cls.r0 = SO101Renderer(4, 224, 224, **kw)

    def states(self, cube_xy_off=0.0, cube_z=0.015):
        q = torch.zeros(4, 6, dtype=f64)
        q[:, 1], q[:, 2] = -1.2, 1.2                                           # arm folded up, away from the plate
        goal = torch.tensor([[0.25, 0.0, 0.2], [0.22, 0.05, 0.6], [0.28, -0.04, 1.0], [0.24, 0.02, 0.0]], dtype=f64)
        cp = torch.cat((goal[:, :2] + torch.tensor([cube_xy_off, 0.0], dtype=f64),
                        torch.full((4, 1), cube_z, dtype=f64)), -1)
        cq = torch.tensor([[1.0, 0, 0, 0]] * 4, dtype=f64)
        return q, cp, cq, goal

    def test_top_camera_depth_moves_by_plate_dz(self):
        q, cp, cq, goal = self.states(cube_xy_off=0.08)                        # cube beside the plate
        dz = torch.full((4,), 0.0035, dtype=f64)
        rgb0, dep0, seg0 = self.r1.render(q, cp, cq, goal, depth=True, segment=True)
        rgb1, dep1, seg1 = self.r1.render(q, cp, cq, goal, depth=True, segment=True, plate_dz=dz)
        gi = self.r1.scene.body_names.index("goal")
        ci = self.r1.scene.body_names.index("cube")
        top = 2
        both = (seg0[:, top] == gi) & (seg1[:, top] == gi)
        self.assertGreater(int(both.sum()), 400)
        d = (dep0[:, top] - dep1[:, top])[both]
        self.assertLess(float((d - 0.0035).abs().median()), 2e-4)
        cube = (seg0[:, top] == ci) & (seg1[:, top] == ci)                       # off the plate: no lift
        self.assertGreater(int(cube.sum()), 100)
        self.assertLess(float((dep0[:, top] - dep1[:, top])[cube].abs().max()), 1e-5)
        rgbo = self.r0.render(q, cp, cq, goal)          # the old renderer draws the nominal picture the same
        frac = (rgbo.int() - rgb0.int()).abs().amax(-1).gt(0).float().mean()
        self.assertLess(float(frac), 1e-3)
        for c in (0, 1):                                # both student cameras see the change only near the plate
            ch = (rgb0[:, c].int() - rgb1[:, c].int()).abs().amax(-1) > 0
            near = (seg0[:, c] == gi) | (seg1[:, c] == gi)
            self.assertGreater(int(ch.sum()), 0)
            self.assertLess(float((ch & ~near).float().sum() / ch.float().sum()), 0.25)

    def test_resting_cube_rides_on_the_plate(self):
        q, cp, cq, goal = self.states(cube_xy_off=0.0, cube_z=0.015)          # resting on the table, on the plate
        dz = torch.full((4,), 0.0035, dtype=f64)
        _, dep0, seg0 = self.r1.render(q, cp, cq, goal, depth=True, segment=True)
        _, dep1, seg1 = self.r1.render(q, cp, cq, goal, depth=True, segment=True, plate_dz=dz)
        ci = self.r1.scene.body_names.index("cube")
        cube = (seg0[:, 2] == ci) & (seg1[:, 2] == ci)
        self.assertGreater(int(cube.sum()), 150)                               # the cube's top in the top camera
        d = (dep0[:, 2] - dep1[:, 2])[cube]
        top = (d - 0.0035).abs() < 2e-4             # the top face: 3.5 mm closer; the side faces (vertical planes
        side = d.abs() < 2e-4                       # moved along themselves) keep their depth
        self.assertGreater(float(top.float().mean()), 0.6)
        self.assertTrue(bool((top | side).all()), d[~(top | side)])


@unittest.skipUnless(GPU and HAVE_SCENE, "needs newton on CUDA and the v2 scene")
class PlateViews(unittest.TestCase):
    def test_views_draw_nominal_merge_render(self):
        from dataclasses import replace
        pt0, pt1 = 0.001, 0.0045
        cfg = replace(VI.ViralConfig(), cameras=("external", "wrist"), res=224, scene=str(SCENE_V2), calib=str(CALIB))
        vcfg = VI.online_views_cfg(cfg, VI.OnlineConfig.from_dict({"plate_mm": [pt0 * 1e3, pt1 * 1e3]}))
        v = VI.make_views(vcfg, 4, seed=1, device="cuda")
        self.assertTrue(v.plate_on)
        self.assertAlmostEqual(v.r.scene.plate_depth, pt1, places=12)
        d = v.draw(4)
        pt = v.r.scene.real.plate_thickness
        self.assertTrue(bool(((d["plate_dz"] >= pt0 - pt - 1e-12) & (d["plate_dz"] <= pt1 - pt + 1e-12)).all()))
        nd = v.nominal_draw(4)
        self.assertEqual(set(nd), set(d))
        self.assertEqual(float(nd["plate_dz"].abs().max()), 0.0)
        m = v.merge_nominal(d, torch.tensor([True, False, True, False], device="cuda"))
        self.assertEqual(float(m["plate_dz"][0]), 0.0)
        self.assertEqual(float(m["plate_dz"][1]), float(d["plate_dz"][1]))
        import bgdr_ref
        q, cp, cq, goal = (t.cuda() for t in bgdr_ref.states(4))
        a = v.render(q, cp, cq, goal, dr=nd, light=False)
        d["plate_dz"][:] = pt1 - pt
        b = v.render(q, cp, cq, goal, dr={**nd, "plate_dz": d["plate_dz"]}, light=False)
        self.assertFalse(torch.equal(a, b))
        v0 = VI.make_views(cfg, 4, seed=1, device="cuda")          # the student's own config: no plate DR
        self.assertFalse(getattr(v0, "plate_on", False))
        c = v0.render(q, cp, cq, goal, dr=v0.nominal_draw(4), light=False)
        self.assertLess(float((a.int() - c.int()).abs().amax(-1).gt(0).float().mean()), 2e-3)


@unittest.skipUnless(GPU and HAVE_SCENE and shutil.which("git"), "needs newton on CUDA, the v2 scene and git")
class RenderBitIdentity(unittest.TestCase):
    """bgdr_ref.py (DRViews nominal / randomised renders, draws, merge, RealDR + render_dr with segment and hdr) on
    the v2 scene: the base commit's arrays bit for bit with the v8 options off."""

    def test_v2_scene_bit_identical(self):
        tmp = Path(tempfile.mkdtemp(prefix="v8table_render_"))
        try:
            r = subprocess.run(f"git -C {REPO} archive {BASE} src configs contact_nrd | tar -x -C {tmp}", shell=True,
                               capture_output=True)
            if r.returncode != 0:
                self.skipTest(f"git archive {BASE} failed")
            vendor = REPO / "contact_nrd/so101_grasp/vendor"
            if vendor.exists() and not (tmp / "contact_nrd/so101_grasp/vendor").exists():
                (tmp / "contact_nrd/so101_grasp/vendor").symlink_to(vendor)
            out = {}
            for tag, src in (("base", tmp / "src"), ("new", REPO / "src")):
                env = dict(os.environ, PYTHONPATH=str(src))
                p = subprocess.run([sys.executable, str(HERE / "bgdr_ref.py"), "--scene", str(SCENE_V2), "--calib",
                                    str(CALIB), "--out", str(tmp / f"{tag}.npz")], env=env, capture_output=True,
                                   text=True)
                self.assertEqual(p.returncode, 0, p.stderr[-3000:])
                out[tag] = np.load(tmp / f"{tag}.npz")
            a, b = out["base"], out["new"]
            self.assertEqual(set(a.files), set(b.files))
            bad = [k for k in a.files if not np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f")]
            self.assertEqual(bad, [], f"outputs differ from {BASE}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
