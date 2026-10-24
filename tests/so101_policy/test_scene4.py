"""Tests of the scene v4 round (2026-10-23, ~/so101_r2s_claude/v8/scene4): parts on the FIXED arm base
(``RealScene.parts`` with ``"body": "base"``, scene v4's table clamp) and the base add-on randomisation
(``RealDR.base_addon``, ``nedm.render.base_addon``, ``RealScene.addon_slots``, ``viral.load_real_setup``).

- Structure (CPU, mujoco for the meshes): fixed parts after the base meshes and before the floor, their pose, group,
  colour override, bad bodies / groups; the add-on slots (hidden env bodies, group ``addon``); the base outline; the
  add-on config (defaults off, bad keys and ranges); the draw (every other draw of a seed unchanged, counts, the
  world probability, placement at the outline, colours, determinism, slot mismatch); load_real_setup adds the slots
  only when the add-ons are on.
- OffIdentity (GPU, git): ``scene4_ref.py`` (views nominal / randomised, draws, merge_nominal, RealDR + render_dr with
  segment, hdr and the background layers) gives the base commit's arrays bit for bit on the v2 and v3 scenes.
- Pixels (GPU): a v4-like scene (v3-like + a clamp on the base) differs from the same scene without it only on the
  clamp's pixels; hidden add-on slots change no pixel, depth or segment; drawn add-ons (also flat cables and their
  sloped pieces) change only their own pixels; the views of a config with ``real_dr.base_addon`` (slots, draws,
  nominal draw, merge_nominal); the add-ons' own generator keeps every other draw of DRViews paired over calls.
- Review round 1 (CPU): every (lo, hi) range and non-negative scalar is checked; ``slot_spread="grid"`` covers both
  ends of every range; flat cables (on the table, start at the outline, outward, heading range, sloped piece joined
  to the start, counts, colours) and ``cable_phi`` / ``cable_heading``; ``block_reach``; ``addon_generator``.
- RealV4Config (GPU, skipped without the files): the shipped config viral_r2s_v8_scene4_student.json with the real
  scene v4 file: pin_real (sha256), make_views (background layers on, the add-on slots last), the clamp in the
  external camera, add-on pixels untouched by the background layers, nominal draw = nominal render.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_scene4 -v
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = os.environ.get("NEDM_SCENE4_BASE", "39d2e5b72")          # the commit before the scene v4 change
SCENE_V2 = Path(os.environ.get("NEDM_R2S_SCENE", "~/so101_r2s_claude/scene/real_scene_v2.json")).expanduser()
SCENE_V3 = Path(os.environ.get("NEDM_R2S_SCENE_V3", "~/so101_studentlab_claude/scene/real_scene_v3.json")).expanduser()
CALIB = Path(os.environ.get("NEDM_R2S_CALIB", "~/so101_r2s_claude/calib/calib_v1.json")).expanduser()
HAVE_MUJOCO = importlib.util.find_spec("mujoco") is not None
GPU = (importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None
       and torch.cuda.is_available())
HAVE_SCENE = SCENE_V2.exists() and CALIB.exists()
SCENE_V4 = Path(os.environ.get("NEDM_R2S_SCENE_V4", "~/so101_r2s_claude/v8/scene4/real_scene_v4.json")).expanduser()
CFG_V8 = REPO / "configs/so101_policy/vision/viral_r2s_v8_scene4_student.json"

# a clamp-like part on the fixed base (scene v4 layout: a jaw beyond the base's -x side, a frame below it)
JAW = {"name": "clamp_jaw", "body": "base", "group": "printed",
       "polygon": [[-0.015, -0.036, 0.0], [-0.015, -0.002, 0.0], [-0.077, -0.002, 0.0], [-0.077, -0.036, 0.0]],
       "offset": [0.0, 0.0, 0.010]}
FRAME = {"name": "clamp_frame", "body": "base", "group": "printed", "color": [0.36, 0.34, 0.31],
         "polygon": [[-0.079, -0.037, -0.044], [-0.072, -0.037, -0.044], [-0.072, -0.016, -0.044],
                     [-0.079, -0.016, -0.044]], "offset": [0.0, 0.0, 0.046]}
ADDON = {"p": 1.0, "n_blocks": [2, 2], "block_slots": 3, "n_cables": [1, 2], "cable_slots": 2}
# with flat cables (review round 1): 3 blocks, 2 cables, 3 flats + 3 sloped pieces
ADDON_FLAT = {**ADDON, "n_flat": [1, 2], "flat_slots": 3, "flat_drop": 0.6}


def v3_like(d2: dict) -> dict:
    """A v3-like scene dict from the v2 dict (as test_bgdr.v3_like): floor turn, wall end, background layers."""
    d = copy.deepcopy(d2)
    d["version"] = "real_scene_v3_test"
    g = d["geometry"]
    g["floor_xy"] = [-0.045, 0.80, -0.60, 0.70]
    g["floor_turn"] = {"deg": -4.0, "pivot": [-0.046, -0.300]}
    g["walls"][0][0] = [0.015, 0.457]
    g["wall_end"] = {"wall": 0, "slide": 0.04}
    d["dr"].update(floor_turn_deg=8.0, floor_xy=0.03, wall_end=0.04,
                   bg={"cameras": ["external"], "p": 1.0, "tex_amp": [0.02, 0.16]})
    return d


@unittest.skipUnless(HAVE_MUJOCO, "the arm meshes need mujoco")
class Structure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nedm.render.so101 import RealScene, SO101Scene, load_assets
        cls.assets = load_assets()
        cls.RealScene, cls.SO101Scene = RealScene, SO101Scene

    def make(self, **kw):
        real = self.RealScene(**kw)
        return self.SO101Scene(self.assets, real=real, cameras=("external", "wrist")), real

    def test_fixed_part_order_pose_colour(self):
        from nedm.render.so101 import REAL_GROUPS
        sc0, _ = self.make()
        sc1, real = self.make(parts=(JAW, FRAME))
        n0 = sc0.body_names
        k = n0.index("floor")
        self.assertEqual(sc1.body_names, n0[:k] + ["clamp_jaw", "clamp_frame"] + n0[k:])
        self.assertTrue(n0[k - 1].startswith("base#"))
        self.assertTrue(torch.equal(sc0.env_pose, sc1.env_pose))
        self.assertEqual(sc0.env_names, sc1.env_names)
        j, f = sc1.body_names.index("clamp_jaw"), sc1.body_names.index("clamp_frame")
        self.assertIn(j, sc1.groups["printed"])
        self.assertIn(f, sc1.groups["printed"])
        self.assertEqual(sc1.scene.bodies[j].color, real.color("printed"))
        self.assertEqual(sc1.scene.bodies[f].color, tuple(FRAME["color"]))
        nfix = len(sc1.fixed_pose) - sc1.n_env
        rows = sc1.fixed_pose[nfix - 2:nfix]
        self.assertTrue(torch.allclose(rows, torch.tensor([[0, 0, 0, 0, 0, 0, 1.0]] * 2, dtype=rows.dtype)))
        idx = sorted(i for g in REAL_GROUPS for i in sc1.groups[g])
        self.assertEqual(idx, list(range(len(sc1.body_names))))
        self.assertEqual(sc1.groups["addon"], [])
        # body transforms: the jaw sits at the base pose for any arm state
        q = torch.tensor([[0.5, -0.3, 0.4, 0.2, 0.1, 0.0]], dtype=torch.float64)
        bq, _ = sc1.transforms(q, torch.zeros(1, 3), torch.tensor([[1.0, 0, 0, 0]]))
        self.assertTrue(torch.allclose(bq[0, j], torch.tensor([0, 0, 0, 0, 0, 0, 1.0])))

    def test_bad_parts(self):
        with self.assertRaises(ValueError):
            self.make(parts=({**JAW, "body": "no_such_body"},))
        with self.assertRaises(ValueError):
            self.make(parts=({**JAW, "group": "no_such_group"},))

    def test_slots_hidden_env_bodies(self):
        from nedm.render.so101 import HIDDEN_Z, RealDR
        slots = RealDR(base_addon=ADDON).addon_slots()
        self.assertEqual(len(slots["blocks"]), 3)
        self.assertEqual(len(slots["cables"]), 2)
        sc0, _ = self.make()
        sc, _ = self.make(addon_slots=slots)
        names = sc.env_names
        self.assertEqual(names[:len(sc0.env_names)], sc0.env_names)
        self.assertEqual(names[len(sc0.env_names):], ["addon_block0", "addon_block1", "addon_block2", "addon_cable0",
                                                      "addon_cable1"])
        nb = len(sc.body_names)
        self.assertEqual(sc.groups["addon"], list(range(nb - 5, nb)))
        self.assertTrue((sc.env_pose[-5:, 2] == HIDDEN_Z).all())
        self.assertTrue(torch.equal(sc.env_pose[:-5], sc0.env_pose))
        # slot sizes: inside the ranges, distinct
        b = np.array(slots["blocks"])
        for k, (lo, hi) in enumerate(((0.015, 0.07), (0.010, 0.035), (0.006, 0.025))):
            self.assertTrue(((b[:, k] >= lo) & (b[:, k] <= hi)).all())
        self.assertEqual(len({tuple(x) for x in b.tolist()}), 3)

    def test_base_outline(self):
        sc, _ = self.make()
        hull, ztop = sc.base_outline()
        self.assertGreater(len(hull), 4)
        self.assertTrue(0.05 < ztop < 0.12)
        self.assertTrue(-0.04 < hull[:, 0].min() < -0.02 and 0.05 < hull[:, 0].max() < 0.08)
        self.assertTrue(-0.065 < hull[:, 1].min() < -0.045 and 0.045 < hull[:, 1].max() < 0.065)
        # ccw and convex
        e = np.roll(hull, -1, 0) - hull
        cross = e[:, 0] * np.roll(e, -1, 0)[:, 1] - e[:, 1] * np.roll(e, -1, 0)[:, 0]
        self.assertTrue((cross > -1e-12).all())

    def test_config_defaults_and_errors(self):
        from nedm.render.base_addon import BaseAddonDR
        from nedm.render.so101 import RealDR
        d = RealDR()
        self.assertEqual(d.base_addon, {})
        self.assertFalse(d.addon_on)
        self.assertIsNone(d.addon_slots())
        self.assertFalse(RealDR(base_addon={"p": 0.0}).addon_on)
        self.assertFalse(RealDR(base_addon={"n_blocks": [0, 0], "n_cables": [0, 0]}).addon_on)
        self.assertTrue(RealDR(base_addon={"p": 0.5}).addon_on)
        for bad in ({"nope": 1}, {"p": 1.5}, {"n_blocks": [0, 5]}, {"n_blocks": [2, 1]},
                    {"block_size": [[0.01, 0.02], [0.01, 0.02]]}, {"cable_len": [0.1, 0.05]},
                    # review round 1: every (lo, hi) range, the non-negative scalars, the new options
                    {"block_gap": [0.002, -0.008]}, {"block_z": [0.03, -0.01]}, {"block_bright": [0.8, 0.3]},
                    {"block_bright": [-0.1, 0.5]}, {"cable_in": [1.0, 0.2]}, {"cable_in": [-0.1, 0.5]},
                    {"cable_z": [0.06, 0.01]}, {"cable_pitch": [1.3, 0.4]}, {"cable_pitch": [0.4, 2.0]},
                    {"cable_bright": [0.12, 0.02]}, {"block_tint": -0.1}, {"cable_tint": -0.1},
                    {"block_tilt": -0.2}, {"cable_yaw": -1.0}, {"block_reach": -0.1}, {"reach_xy": [0.0]},
                    {"slot_spread": "sobol"}, {"flat_drop": 1.5}, {"flat_yaw": 1.6}, {"flat_yaw": -0.1},
                    {"n_flat": [0, 1]}, {"n_flat": [2, 1], "flat_slots": 3}, {"flat_len": [0.0, 0.1]},
                    {"drop_len": [0.09, 0.04]}, {"drop_z": [0.0, 0.05]}, {"flat_gap": [0.004, -0.004]},
                    {"flat_phi": [2.0, 1.0]}, {"flat_heading": [4.0, 2.0]}, {"cable_heading": [0.0, 7.0]},
                    {"cable_phi": [3.0, 1.0]}, {"block_slots": -1}, {"cable_width": [0.01, 0.004]}):
            with self.assertRaises(ValueError, msg=str(bad)):
                BaseAddonDR.from_dict(bad)
        ok = BaseAddonDR.from_dict({"n_flat": [0, 1], "flat_slots": 2, "flat_heading": [1.6, 4.7],
                                    "cable_heading": None, "slot_spread": "grid"})
        self.assertTrue(ok.flat_on and ok.on)
        self.assertFalse(BaseAddonDR.from_dict({"n_flat": [0, 1], "flat_slots": 2, "n_blocks": [0, 0],
                                                "n_cables": [0, 0], "p": 0.0}).on)
        self.assertTrue(BaseAddonDR.from_dict({"n_flat": [0, 1], "flat_slots": 2, "n_blocks": [0, 0],
                                               "n_cables": [0, 0]}).on)
        # the defaults of the new options = the first build: no flats, Halton slots, no reach limit
        d0 = BaseAddonDR()
        self.assertEqual((d0.slot_spread, d0.block_reach, d0.n_flat, d0.flat_slots, d0.flat_heading,
                          d0.cable_heading), ("halton", 0.0, (0, 0), 0, None, None))
        self.assertEqual(set(d0.slots()), {"blocks", "cables"})
        with self.assertRaises(ValueError):
            RealDR.from_dict({"base_addon": {"nope": 1}}).addon_on

    def test_draw(self):
        from nedm.render.so101 import HIDDEN_Z, RealDR
        old = RealDR(wall_xy=0.02, mat_xy=0.01, floor_turn_deg=8.0, floor_xy=0.03)
        new = RealDR(wall_xy=0.02, mat_xy=0.01, floor_turn_deg=8.0, floor_xy=0.03, base_addon=ADDON)
        sc0, _ = self.make()
        sc, _ = self.make(addon_slots=new.addon_slots())
        n = 256
        a = old.draw(sc0, n, torch.Generator().manual_seed(0))
        b = new.draw(sc, n, torch.Generator().manual_seed(0))
        b2 = new.draw(sc, n, torch.Generator().manual_seed(0))
        for k in b:
            self.assertTrue(torch.equal(b[k], b2[k]), k)                  # determinism
        ne = len(sc0.env_names)
        nb0 = len(sc0.body_names)
        self.assertTrue(torch.equal(a["colors"], b["colors"][:, :nb0]))   # every other draw of the seed unchanged
        self.assertTrue(torch.equal(a["cam_offset"], b["cam_offset"]))
        self.assertTrue(torch.equal(a["env_pose"], b["env_pose"][:, :ne]))
        slots = b["env_pose"][:, ne:]
        shown = slots[..., 2] > HIDDEN_Z / 2
        nblk = shown[:, :3].sum(1)
        ncab = shown[:, 3:].sum(1)
        self.assertTrue(bool((nblk == 2).all()))                          # p = 1, n_blocks (2, 2)
        self.assertTrue(bool(((ncab >= 1) & (ncab <= 2)).all()))
        self.assertTrue(bool((ncab == 1).any() and (ncab == 2).any()))
        # placement: shown blocks touch the base outline (centre within half diagonal + 1 cm of the hull)
        hull, _ = sc.base_outline()
        dims = np.array(new.addon_slots()["blocks"])
        pts = slots[:, :3, :2].numpy()
        for s in range(3):
            m = shown[:, s].numpy()
            dist = np.array([_dist_to_polygon(p, hull) for p in pts[m, s]])
            self.assertLess(dist.max(), np.linalg.norm(dims[s, :2]) / 2 + 0.01)
            z = slots[m, s, 2].numpy()
            self.assertTrue(((z >= -0.015) & (z <= 0.035 + dims[s, 2])).all())
        # colours: blocks grey 0.3..0.8 x tint, cables dark
        cols = b["colors"][:, nb0:]
        blk = cols[:, :3][shown[:, :3]]
        self.assertTrue(bool((blk >= 0.3 * 0.92 - 1e-6).all() and (blk <= 0.8 * 1.08 + 1e-6).all()))
        cab = cols[:, 3:][shown[:, 3:]]
        self.assertTrue(bool((cab <= 0.12 * 1.1 + 1e-6).all()))
        # world probability
        half = RealDR(base_addon={**ADDON, "p": 0.5})
        c = half.draw(sc, 2000, torch.Generator().manual_seed(1))["env_pose"][:, ne:, 2] > HIDDEN_Z / 2
        frac = float(c.any(1).float().mean())
        self.assertTrue(0.44 < frac < 0.56, frac)
        # a scene with other slots is refused
        sc_other, _ = self.make(addon_slots=RealDR(base_addon={**ADDON, "block_slots": 4}).addon_slots())
        with self.assertRaises(ValueError):
            new.draw(sc_other, 4, torch.Generator().manual_seed(0))
        # slots in a scene whose RealDR has the add-ons off: they stay hidden
        off = old.draw(sc, 4, torch.Generator().manual_seed(0))
        self.assertTrue(bool((off["env_pose"][:, ne:, 2] == HIDDEN_Z).all()))

    def test_slot_spread(self):
        from nedm.render.base_addon import BaseAddonDR
        # Halton (default): the first build's slots (k = 1..4): only the middle of each range
        h = np.array(BaseAddonDR().slots()["blocks"])
        self.assertTrue(np.allclose(h, [[0.0425, 0.018333, 0.0098], [0.02875, 0.026667, 0.0136],
                                        [0.05625, 0.012778, 0.0174], [0.021875, 0.021111, 0.0212]], atol=1e-6))
        # grid: per dimension the S values lo + j / (S - 1) (hi - lo), both ends included, distinct boxes
        for S in (2, 4, 12):
            d = BaseAddonDR(slot_spread="grid", block_slots=S, n_blocks=(1, min(2, S)), cable_slots=S,
                            n_cables=(0, 1), flat_slots=S, n_flat=(0, 1))
            sl = d.slots()
            b = np.array(sl["blocks"])
            for k, (lo, hi) in enumerate(d.block_size):
                self.assertTrue(np.allclose(np.sort(b[:, k]), np.linspace(lo, hi, S), atol=1e-6), (S, k))
            self.assertEqual(len({tuple(x) for x in b.tolist()}), S)
            c = np.array(sl["cables"])
            self.assertTrue(np.allclose(np.sort(c[:, 0]), np.linspace(*d.cable_len, S), atol=1e-6))
            f, dr = np.array(sl["flats"]), np.array(sl["drops"])
            self.assertTrue(np.allclose(np.sort(f[:, 0]), np.linspace(*d.flat_len, S), atol=1e-6))
            self.assertTrue(np.allclose(np.sort(dr[:, 0]), np.linspace(*d.drop_len, S), atol=1e-6))
            self.assertTrue(np.array_equal(f[:, 1], dr[:, 1]))                   # a sloped piece = its cable's width
            if S >= 4:                                                           # dimensions not in lock step
                self.assertFalse(np.array_equal(np.argsort(b[:, 0]), np.argsort(b[:, 1])))
        one = BaseAddonDR(slot_spread="grid", block_slots=1, n_blocks=(1, 1)).slots()["blocks"][0]
        self.assertTrue(np.allclose(one, [np.mean(r) for r in BaseAddonDR().block_size]))

    def test_flat_cables(self):
        from nedm.render.base_addon import BaseAddonDR
        from nedm.render.so101 import HIDDEN_Z, RealDR, wxyz
        heading = (1.8, 4.5)
        cfg = {**ADDON_FLAT, "flat_heading": list(heading), "flat_phi": [1.4, 4.9]}
        dr = RealDR(wall_xy=0.02, mat_xy=0.01, base_addon=cfg)
        sc, real = self.make(addon_slots=dr.addon_slots())
        names = sc.env_names
        self.assertEqual(names[-11:], [f"addon_block{k}" for k in range(3)] + [f"addon_cable{k}" for k in range(2)]
                         + [f"addon_flat{k}" for k in range(3)] + [f"addon_drop{k}" for k in range(3)])
        n = 4000
        out = dr.draw(sc, n, torch.Generator().manual_seed(5))
        # adding flat cables keeps the blocks' and cables' draws (they are drawn after them)
        ref = RealDR(wall_xy=0.02, mat_xy=0.01, base_addon=ADDON)
        sc0, _ = self.make(addon_slots=ref.addon_slots())
        o0 = ref.draw(sc0, n, torch.Generator().manual_seed(5))
        ne0 = len(sc0.env_names)
        self.assertTrue(torch.equal(o0["env_pose"], out["env_pose"][:, :ne0]))
        self.assertTrue(torch.equal(o0["colors"], out["colors"][:, :len(sc0.body_names)]))
        sl = dr.addon_slots()
        add = BaseAddonDR.from_dict(cfg)
        fl = [names.index(f"addon_flat{k}") for k in range(3)]
        dp = [names.index(f"addon_drop{k}") for k in range(3)]
        env = out["env_pose"].double()
        fs = env[:, fl, 2] > HIDDEN_Z / 2
        ds = env[:, dp, 2] > HIDDEN_Z / 2
        nf = fs.sum(1)
        self.assertTrue(bool(((nf >= 1) & (nf <= 2)).all()))                    # p = 1, n_flat (1, 2)
        self.assertTrue(bool((ds <= fs).all()))                                  # a sloped piece only with its cable
        share = float(ds.sum()) / float(fs.sum())
        self.assertTrue(0.55 < share < 0.65, share)                              # flat_drop 0.6
        hull, _ = sc.base_outline()
        c, _, _ = BaseAddonDR.outline_table(hull)
        top = -float(real.mat_thickness)
        for k in range(3):
            m = fs[:, k]
            L, W = sl["flats"][k]
            pose = env[m, fl[k]]
            # lies on the table: centre at the table top + W / 2 + 0.5 mm, its axis horizontal
            self.assertTrue(torch.allclose(pose[:, 2], torch.full_like(pose[:, 2], top + W / 2 + 0.0005)))
            qw = wxyz(pose[:, 3:])
            head = 2 * torch.atan2(qw[:, 3], qw[:, 0])
            self.assertTrue(torch.allclose(qw[:, 1:3], torch.zeros_like(qw[:, 1:3]), atol=1e-12))
            hv = torch.stack([torch.cos(head), torch.sin(head)], -1)
            start = pose[:, :2] - L / 2 * hv
            # the start is at the outline (+ flat_gap): its distance to the hull boundary <= 4 mm + lookup error
            dist = np.array([_dist_to_polygon(p, hull) for p in start.numpy()])
            self.assertLess(dist.max(), 0.004 + 0.0015)
            # outward: the far end is farther from the footprint centre than the start, and outside the hull
            cc = torch.tensor(c)
            self.assertTrue(bool(((pose[:, :2] + L / 2 * hv - cc).norm(dim=-1) > (start - cc).norm(dim=-1)).all()))
            # heading within flat_heading (every outline point of flat_phi has an outward heading in it)
            mid, half = (heading[0] + heading[1]) / 2, (heading[1] - heading[0]) / 2
            off = torch.remainder(head - mid + math.pi, 2 * math.pi) - math.pi
            self.assertTrue(bool((off.abs() <= half + 1e-9).all()), float(off.abs().max()))
            # the sloped piece joins the cable's start and rises to drop_z (pitch clamped)
            md = ds[:, k]
            Ld, Wd = sl["drops"][k]
            self.assertEqual(Wd, W)
            dpose = env[md, dp[k]]
            dq = wxyz(dpose[:, 3:])
            ax = _rot(dq, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64).expand(len(dq), 3))
            low = dpose[:, :3] + Ld / 2 * ax                                     # +x end: down at the cable start
            fpose = env[md, fl[k]]
            fq = wxyz(fpose[:, 3:])
            fh = 2 * torch.atan2(fq[:, 3], fq[:, 0])
            fstart = fpose[:, :2] - L / 2 * torch.stack([torch.cos(fh), torch.sin(fh)], -1)
            self.assertTrue(torch.allclose(low[:, :2], fstart, atol=1e-9))
            self.assertTrue(torch.allclose(low[:, 2], fpose[:, 2], atol=1e-9))
            rise = (dpose[:, 2] - fpose[:, 2]) * 2
            self.assertTrue(bool((rise >= Ld * 0.05 - 1e-9).all() and (rise <= Ld * 0.97 + 1e-9).all()))
            # the same grey for the cable and its sloped piece, dark
            nb = len(sc.body_names)
            first = nb - len(names)
            cf, cd = out["colors"][md, first + fl[k]], out["colors"][md, first + dp[k]]
            self.assertTrue(torch.equal(cf, cd))
            self.assertTrue(bool((cf <= 0.12 * 1.1 + 1e-6).all()))
        # cable_phi / cable_heading: the sloped cables start in the range and head within it
        cfg2 = {**ADDON, "cable_phi": [2.0, 4.3], "cable_heading": [2.2, 4.1]}
        dr2 = RealDR(base_addon=cfg2)
        sc2, _ = self.make(addon_slots=dr2.addon_slots())
        o2 = dr2.draw(sc2, 3000, torch.Generator().manual_seed(6))
        kc = [sc2.env_names.index(f"addon_cable{k}") for k in range(2)]
        e2 = o2["env_pose"].double()[:, kc]
        sh = e2[..., 2] > HIDDEN_Z / 2
        q2 = wxyz(e2[..., 3:])
        ax = _rot(q2.reshape(-1, 4), torch.tensor([1.0, 0, 0], dtype=torch.float64).expand(q2.numel() // 4, 3))
        ax = ax.reshape(*q2.shape[:-1], 3)
        h2 = torch.atan2(ax[..., 1], ax[..., 0])[sh]
        off = torch.remainder(h2 - 3.15 + math.pi, 2 * math.pi) - math.pi
        self.assertTrue(bool((off.abs() <= 0.95 + 1e-9).all()), float(off.abs().max()))
        self.assertTrue(bool((ax[..., 2][sh] < 0).all()))                       # downwards
        # a scene without the flat slots is refused
        with self.assertRaises(ValueError):
            dr.draw(sc0, 4, torch.Generator().manual_seed(0))

    def test_block_reach(self):
        from nedm.render.so101 import HIDDEN_Z, RealDR, wxyz
        reach = 0.09
        base = {**ADDON, "block_slots": 6, "slot_spread": "grid", "n_blocks": [2, 3]}
        on = RealDR(base_addon={**base, "block_reach": reach})
        off = RealDR(base_addon=base)
        sc, _ = self.make(addon_slots=on.addon_slots())
        n = 3000
        a = on.draw(sc, n, torch.Generator().manual_seed(8))
        b = off.draw(sc, n, torch.Generator().manual_seed(8))
        kb = [sc.env_names.index(f"addon_block{k}") for k in range(6)]
        dims = torch.tensor(on.addon_slots()["blocks"], dtype=torch.float64)
        ea, eb = a["env_pose"].double()[:, kb], b["env_pose"].double()[:, kb]
        sh = ea[..., 2] > HIDDEN_Z / 2
        self.assertTrue(torch.equal(sh, eb[..., 2] > HIDDEN_Z / 2))
        corners = torch.tensor([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                               dtype=torch.float64)
        ctr = torch.tensor([0.0388, 0.0], dtype=torch.float64)
        far = []
        for k in range(6):
            m = sh[:, k]
            pts = corners[None] * dims[k] / 2
            q = wxyz(ea[m, k, 3:])[:, None].expand(-1, 8, 4)
            P = ea[m, k, None, :3] + _rot(q, pts.expand(len(q), 8, 3))
            far.append((P[..., :2] - ctr).norm(dim=-1).max(1).values)
        far = torch.cat(far)
        self.assertLessEqual(float(far.max()), reach + 1e-9)
        moved = ~torch.isclose(ea[..., :2], eb[..., :2]).all(-1) & sh
        self.assertTrue(bool(moved.any()) and bool((~moved & sh).any()))        # some pulled in, the rest kept
        self.assertTrue(torch.equal(ea[..., 2:][~moved], eb[..., 2:][~moved]))
        self.assertTrue(torch.equal(a["colors"], b["colors"]))

    def test_addon_generator_pairs_draws(self):
        from nedm.render.so101 import RealDR
        kw = dict(wall_xy=0.02, mat_xy=0.01, floor_turn_deg=8.0, floor_xy=0.03)
        off, on = RealDR(**kw), RealDR(**kw, base_addon=ADDON_FLAT)
        sc0, _ = self.make()
        sc, _ = self.make(addon_slots=on.addon_slots())
        ne, nb = len(sc0.env_names), len(sc0.body_names)
        g0, g1 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
        ga = torch.Generator().manual_seed(9)
        for call in range(3):                                   # several calls (the online loop's per-world resets)
            a = off.draw(sc0, 64, g0)
            b = on.draw(sc, 64, g1, addon_generator=ga)
            self.assertTrue(torch.equal(a["env_pose"], b["env_pose"][:, :ne]), call)
            self.assertTrue(torch.equal(a["colors"], b["colors"][:, :nb]), call)
            self.assertTrue(torch.equal(a["cam_offset"], b["cam_offset"]), call)
        # the shared generator (no addon_generator) pairs only the first call
        g0, g1 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
        off.draw(sc0, 64, g0), on.draw(sc, 64, g1)
        a, b = off.draw(sc0, 64, g0), on.draw(sc, 64, g1)
        self.assertFalse(torch.equal(a["cam_offset"], b["cam_offset"]))

    @unittest.skipUnless(HAVE_SCENE, "needs the v2 scene and the calibration")
    def test_load_real_setup_slots(self):
        from nedm.so101_policy.vision import viral as VI
        cfg = replace(VI.ViralConfig(), cameras=("external", "wrist"), res=224, scene=str(SCENE_V2), calib=str(CALIB))
        rs = VI.load_real_setup(cfg)
        self.assertIsNone(rs.real.addon_slots)
        self.assertFalse(rs.dr.addon_on)
        rs2 = VI.load_real_setup(replace(cfg, real_dr={"base_addon": ADDON}))
        self.assertTrue(rs2.dr.addon_on)
        self.assertEqual(rs2.real.addon_slots, rs2.dr.addon_slots())
        self.assertEqual(rs2.scene_sha256, rs.scene_sha256)


def _rot(q_wxyz, v):
    """Rotate v [..., 3] by the unit quaternions q [..., 4] (w x y z)."""
    w, xyz = q_wxyz[..., :1], q_wxyz[..., 1:]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _dist_to_polygon(p, poly):
    """Distance (m) of point p (2,) to the boundary of the polygon (k, 2)."""
    a, b = poly, np.roll(poly, -1, 0)
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / (ab * ab).sum(1).clip(1e-18), 0, 1)
    return float(np.linalg.norm(a + t[:, None] * ab - p, axis=1).min())


@unittest.skipUnless(GPU and HAVE_SCENE and shutil.which("git"), "needs newton on CUDA, the scenes and git")
class OffIdentity(unittest.TestCase):
    """The default options give the base commit's arrays bit for bit (scene4_ref.py on the v2 and v3 scenes)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="scene4_base_"))
        r = subprocess.run(f"git -C {REPO} archive {BASE} src configs contact_nrd | tar -x -C {cls.tmp}", shell=True,
                           capture_output=True)
        vendor = REPO / "contact_nrd/so101_grasp/vendor"
        if vendor.exists() and not (cls.tmp / "contact_nrd/so101_grasp/vendor").exists():
            (cls.tmp / "contact_nrd/so101_grasp/vendor").symlink_to(vendor)
        cls.ok = r.returncode == 0 and (cls.tmp / "src").exists()
        cls.v3 = SCENE_V3
        if not cls.v3.exists():                                   # no v3 file: a v3-like scene from v2
            cls.v3 = cls.tmp / "scene_v3_like.json"
            cls.v3.write_text(json.dumps(v3_like(json.loads(SCENE_V2.read_text()))))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_ref(self, src, scene, out):
        env = dict(os.environ, PYTHONPATH=str(src))
        r = subprocess.run([sys.executable, str(HERE / "scene4_ref.py"), "--scene", str(scene), "--calib", str(CALIB),
                            "--out", str(out)], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        return np.load(out)

    def check(self, scene, tag):
        if not self.ok:
            self.skipTest(f"git archive {BASE} failed")
        a = self.run_ref(self.tmp / "src", scene, self.tmp / f"base_{tag}.npz")
        b = self.run_ref(REPO / "src", scene, self.tmp / f"new_{tag}.npz")
        self.assertEqual(set(a.files), set(b.files))
        bad = [k for k in a.files if not np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f")]
        self.assertEqual(bad, [], f"outputs differ from {BASE}")
        return b

    def test_v2_scene_bit_identical(self):
        self.check(SCENE_V2, "v2")

    def test_v3_scene_bit_identical(self):
        b = self.check(self.v3, "v3")
        self.assertTrue(any(k.startswith("rdraw_bg") for k in b.files))   # the background layers were exercised


@unittest.skipUnless(GPU and HAVE_SCENE, "needs newton on CUDA and the v2 scene")
class Pixels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nedm.so101_policy.vision import viral as VI
        cls.VI = VI
        cls.tmp = Path(tempfile.mkdtemp(prefix="scene4_px_"))
        d3 = v3_like(json.loads(SCENE_V2.read_text()))
        d3["image_fx"] = {}                                       # no nominal blur / chroma: exact pixel checks
        d3["dr"].pop("bg")                                        # the layers are tested elsewhere
        d4 = copy.deepcopy(d3)
        d4["version"] = "real_scene_v4_test"
        d4["geometry"]["parts"] = list(d4["geometry"].get("parts") or []) + [JAW, FRAME]
        cls.p3, cls.p4 = cls.tmp / "v3.json", cls.tmp / "v4.json"
        cls.p3.write_text(json.dumps(d3))
        cls.p4.write_text(json.dumps(d4))
        sys.path.insert(0, str(HERE))
        import scene4_ref
        n = 6
        q, cp, cq, goal = scene4_ref.states(n)
        q[:, 0] = 0.9                                             # the arm turned left: the base's -x side is free
        cls.states = (q, cp, cq, goal)
        cls.n = n

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def renderer(self, path, slots=None, n=None):
        from nedm.render.so101 import SO101Renderer
        cfg = replace(self.VI.ViralConfig(), cameras=("external", "wrist"), res=224, scene=str(path), calib=str(CALIB))
        rs = self.VI.load_real_setup(cfg)
        real = rs.real if slots is None else replace(rs.real, addon_slots=slots)
        return SO101Renderer(n or self.n, 224, 224, backend="newton", cameras=("external", "wrist"), goal="block",
                             real=real, segment=True), rs

    def named(self, r, seg):
        names = np.array(r.scene.body_names + ["<none>", "<mixed>"])
        s = seg.cpu().numpy()
        return names[np.where(s >= 0, s, np.where(s == -1, len(names) - 2, len(names) - 1))]

    def test_clamp_only_changes_its_pixels(self):
        r3, _ = self.renderer(self.p3)
        r4, _ = self.renderer(self.p4)
        a3, d3, s3 = r3.render(*self.states, depth=True, segment=True)
        a4, d4, s4 = r4.render(*self.states, depth=True, segment=True)
        n3, n4 = self.named(r3, s3), self.named(r4, s4)
        clamp = np.isin(n4, ["clamp_jaw", "clamp_frame"])
        self.assertGreater(int(clamp[:, 0].sum()), 40 * self.n)              # visible in the external camera
        self.assertTrue((n3[~clamp] == n4[~clamp]).all())
        keep = torch.as_tensor(~clamp, device=a3.device)
        self.assertTrue(torch.equal(a3[keep], a4[keep]))
        self.assertTrue(torch.equal(d3[keep], d4[keep]))
        self.assertFalse(torch.equal(a3, a4))

    def test_hidden_slots_change_nothing(self):
        from nedm.render.so101 import RealDR
        slots = RealDR(base_addon=ADDON).addon_slots()
        r0, rs = self.renderer(self.p4)
        r1, _ = self.renderer(self.p4, slots)
        out0 = r0.render(*self.states, depth=True, segment=True, hdr=True)
        out1 = r1.render(*self.states, depth=True, segment=True, hdr=True)
        for x, y in zip(out0, out1):
            self.assertTrue(torch.equal(x, y))
        # a draw without the add-ons in the slot scene = the same draw in the plain scene
        dr = rs.dr
        d0 = dr.draw(r0.scene, self.n, torch.Generator().manual_seed(2))
        d1 = dr.draw(r1.scene, self.n, torch.Generator().manual_seed(2))
        x0 = r0.render_dr(*self.states, draw=d0)
        x1 = r1.render_dr(*self.states, draw=d1)
        self.assertTrue(torch.equal(x0, x1))

    def test_addons_change_only_their_pixels(self):
        from nedm.render.so101 import HIDDEN_Z, RealDR
        dr = RealDR.from_dict({"base_addon": ADDON})
        r, _ = self.renderer(self.p4, dr.addon_slots(), n=16)
        q, cp, cq, goal = (t[:1].expand(16, *t.shape[1:]).clone() for t in self.states)
        d = dr.draw(r.scene, 16, torch.Generator().manual_seed(4))
        hid = {k: v.clone() for k, v in d.items()}
        hid["env_pose"][:, -5:] = r.scene.env_pose[-5:].to(hid["env_pose"])
        on, seg = r.render_dr(q, cp, cq, goal, draw=d, segment=True)
        off = r.render_dr(q, cp, cq, goal, draw=hid)
        names = self.named(r, seg)
        add = np.char.startswith(names.astype(str), "addon_")
        self.assertGreater(int(add[:, 0].sum()), 16 * 20)                    # add-ons seen by the external camera
        self.assertTrue(add[:, 0].reshape(16, -1).any(1).mean() > 0.5)
        keep = torch.as_tensor(~add, device=on.device)
        self.assertTrue(torch.equal(on[keep], off[keep]))
        self.assertTrue((hid["env_pose"][:, -5:, 2] == HIDDEN_Z).all())

    def test_flat_addons_change_only_their_pixels(self):
        from nedm.render.so101 import RealDR
        dr = RealDR.from_dict({"base_addon": {**ADDON_FLAT, "flat_phi": [1.4, 4.9], "flat_heading": [1.8, 4.5]}})
        r, _ = self.renderer(self.p4, dr.addon_slots(), n=16)
        ns = 11
        self.assertEqual(r.scene.env_names[-ns:][-6:], [f"addon_flat{k}" for k in range(3)]
                         + [f"addon_drop{k}" for k in range(3)])
        q, cp, cq, goal = (t[:1].expand(16, *t.shape[1:]).clone() for t in self.states)
        d = dr.draw(r.scene, 16, torch.Generator().manual_seed(4))
        hid = {k: v.clone() for k, v in d.items()}
        hid["env_pose"][:, -ns:] = r.scene.env_pose[-ns:].to(hid["env_pose"])
        on, seg = r.render_dr(q, cp, cq, goal, draw=d, segment=True)
        off = r.render_dr(q, cp, cq, goal, draw=hid)
        names = self.named(r, seg).astype(str)
        add = np.char.startswith(names, "addon_")
        flat = np.char.startswith(names, "addon_flat") | np.char.startswith(names, "addon_drop")
        self.assertGreater(int(flat[:, 0].sum()), 16 * 10)                   # flat cables seen by the external camera
        keep = torch.as_tensor(~add, device=on.device)
        self.assertTrue(torch.equal(on[keep], off[keep]))

    def test_views_addon_generator_pairs(self):
        """DRViews: the add-ons' own generator keeps every other draw the same as without them, call after call."""
        base = replace(self.VI.ViralConfig(), cameras=("external", "wrist"), res=224, scene=str(self.p4),
                       calib=str(CALIB))
        v0 = self.VI.make_views(base, 4, seed=5, device="cuda")
        v1 = self.VI.make_views(replace(base, real_dr={"base_addon": ADDON_FLAT}), 4, seed=5, device="cuda")
        self.assertIsNone(v0.addon_gen)
        self.assertIsNotNone(v1.addon_gen)
        ne, nb = len(v0.r.scene.env_names), len(v0.r.scene.body_names)
        for call in range(3):
            a, b = v0.draw(8), v1.draw(8)
            self.assertEqual(set(a), set(b))
            for k in a:
                x = b[k][:, :ne] if k == "env_pose" else b[k][:, :nb] if k == "colors" else b[k]
                self.assertTrue(torch.equal(a[k], x), (call, k))
            self.assertTrue(bool((b["env_pose"][:, ne:, 2] > -1).any()))

    def test_views_with_addons(self):
        cfg = replace(self.VI.ViralConfig(), cameras=("external", "wrist"), res=224, scene=str(self.p4),
                      calib=str(CALIB), real_dr={"base_addon": ADDON})
        v = self.VI.make_views(cfg, 4, seed=1, device="cuda")
        names = v.r.scene.env_names
        self.assertEqual(names[-5:], ["addon_block0", "addon_block1", "addon_block2", "addon_cable0", "addon_cable1"])
        q, cp, cq, goal = (t[:4].cuda() for t in self.states)
        nominal = v.render(q, cp, cq, goal)
        zero = v.render(q, cp, cq, goal, dr=v.nominal_draw(4), light=False)
        self.assertTrue(torch.equal(nominal, zero))
        d = v.draw(4)
        self.assertTrue(bool((d["env_pose"][:, -5:, 2] > -1).any()))
        m = v.merge_nominal(d, torch.tensor([True, False, True, False], device="cuda"))
        self.assertTrue(torch.equal(m["env_pose"][0], v.env0))
        self.assertTrue(torch.equal(m["env_pose"][1], d["env_pose"][1]))
        img = v.render(q, cp, cq, goal, dr=d)
        self.assertEqual(tuple(img.shape), (4, 2, 224, 224, 3))


@unittest.skipUnless(GPU and SCENE_V4.exists() and CALIB.exists(), "needs newton on CUDA, scene v4 and the calibration")
class RealV4Config(unittest.TestCase):
    """The shipped training combination: viral_r2s_v8_scene4_student.json + the real scene v4 file (sha256 pinned),
    background layers and base add-ons together."""

    @classmethod
    def setUpClass(cls):
        from nedm.so101_policy.vision import viral as VI
        cls.VI = VI
        cls.d = json.loads(CFG_V8.read_text())
        cls.cfg = VI.ViralConfig.from_dict({**{k: v for k, v in cls.d.items() if k != "note"},
                                            "scene": str(SCENE_V4), "calib": str(CALIB)})
        sys.path.insert(0, str(HERE))
        import scene4_ref
        q, cp, cq, goal = scene4_ref.states(8)
        q[:, 0] = 0.9                                             # the arm turned left: the clamp side is free
        cls.states = tuple(t.cuda() for t in (q, cp, cq, goal))

    def test_pin_views_render(self):
        from nedm.render.base_addon import BaseAddonDR
        VI = self.VI
        cfg = VI.pin_real(replace(self.cfg))                      # refuses a scene file other than the pinned one
        self.assertEqual(cfg.scene_sha256, self.d["scene_sha256"])
        self.assertTrue(cfg.crops)
        v = VI.make_views(cfg, 8, seed=3, device="cuda")
        self.assertEqual(v.bg_cams, ("external",))               # scene v3's background layers stay on
        add = BaseAddonDR.from_dict(self.d["real_dr"]["base_addon"])
        sl = add.slots()
        want = [f"addon_{kind[:-1]}{k}" for kind in ("blocks", "cables", "flats", "drops")
                for k in range(len(sl[kind]))]
        self.assertEqual(v.r.scene.env_names[-len(want):], want)
        self.assertEqual(len(v.r.scene.env_names) - len(want), len(VI.make_views(
            replace(cfg, real_dr={}), 1, device="cuda").r.scene.env_names))
        names = np.array(v.r.scene.body_names + ["<none>", "<mixed>"])
        q, cp, cq, goal = self.states
        # nominal: the clamp is in the external camera; the nominal draw renders the nominal picture
        nominal = v.render(q, cp, cq, goal)
        self.assertTrue(torch.equal(nominal, v.render(q, cp, cq, goal, dr=v.nominal_draw(8), light=False)))
        _, seg = v.r.render_dr(q, cp, cq, goal, draw=v.nominal_draw(8), segment=True)
        sg = seg.cpu().numpy()
        nm = names[np.where(sg >= 0, sg, np.where(sg == -1, len(names) - 2, len(names) - 1))]
        self.assertGreater(int(np.isin(nm[:, 0], ["base_clamp_jaw", "base_clamp_frame"]).sum()), 8 * 40)
        # randomised draws with the add-ons in every world: the background layers do not touch add-on pixels
        cfg1 = replace(cfg, real_dr={"base_addon": {**self.d["real_dr"]["base_addon"], "p": 1.0}})
        v1 = VI.make_views(cfg1, 8, seed=4, device="cuda")
        d = v1.draw(8)
        self.assertTrue(any(k.startswith("bg_") for k in d))
        nobg = dict(d)
        nobg.update(v1.rdr.bg_nominal(8, v1.cameras, v1.res, v1.dev))
        ids = torch.tensor([i for i, nm_ in enumerate(v1.r.scene.body_names) if nm_.startswith("addon_")],
                           device="cuda")
        # the renderer applies the scene's nominal image effects (external: chroma blur k 9, blur k 7) AFTER the
        # layers, so changed background pixels bleed up to 7 px into an add-on's edge: compare with them off
        fx = v1.r._fx
        self.assertIsNotNone(fx)
        try:
            v1.r._fx = None
            rgb, seg = v1.r.render_dr(q, cp, cq, goal, draw=d, segment=True)
            rgb0, seg0 = v1.r.render_dr(q, cp, cq, goal, draw=nobg, segment=True)
        finally:
            v1.r._fx = fx
        self.assertTrue(torch.equal(seg, seg0))
        addm = torch.isin(seg, ids)
        self.assertGreater(int(addm[:, 0].sum()), 8 * 20)                    # add-ons seen by the external camera
        self.assertTrue(torch.equal(rgb[addm], rgb0[addm]))
        self.assertFalse(torch.equal(rgb[:, 0], rgb0[:, 0]))                # the layers did change the background
        img = v1.render(q, cp, cq, goal, dr=d)
        self.assertEqual(tuple(img.shape), (8, 2, 224, 224, 3))


if __name__ == "__main__":
    unittest.main()
