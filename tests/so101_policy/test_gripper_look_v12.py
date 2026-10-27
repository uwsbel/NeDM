"""Tests of the v12 visual fix: the gripper-look randomisation (``nedm.render.gripper_look.GripperLookDR``,
``RealDR.gripper_look``, ``RealScene.look_slots``, ``SO101Scene`` look parts / ``part_pose``), its online-loop wiring
(``OnlineConfig.gripper_look`` / ``require_scene_sha256``, ``viral.online_views_cfg``, ``check_scene_requirement``,
``OnlineDAgger.ckpt_record``) and the shipped configs (``viral_r2s_online_v12_visR.json``,
``viral_r2s_online_v12_visH.json``, ``viral_r2s_v12_visR_student.json``).

- Config (no mujoco): defaults off-safe, bad keys and ranges, slot specs, held-out zones, the shipped configs (visR:
  plate range includes the fitted black and the old grey, all draws; visH: no screws, no edge, plate range above the
  fitted black, zones = held_out_zones), online-config validation, online_views_cfg, the scene guard, ckpt_record.
- Structure (CPU, mujoco for the meshes): no look parts when off (body list, groups and transforms as without the
  field); slot parts after the moving parts and before the cube, group ``look``, hidden by default; movable parts at
  identity; part_pose maps local poses; hidden poses go far below the scene.
- Draws (CPU): every other draw of a seed unchanged (own generator), only the look targets' colours change; marks lie
  inside their surface outline (inset) and outside every exclusion zone; mark counts within n_marks; screws move in
  their plane by <= jitter, hidden ones are hidden; p = 0 / p = 1; determinism; slot mismatch refused; load_real_setup
  adds the parts only when on.
- Real files (skipped without them): the visR student config pins the real-look scene sha256 and resolves D's crops;
  D's pinned config + the visR config pass pin_real without --allow-scene-change; the sim-check scene lookup
  (lowtable_gate.student_scene) finds the real-look scene by its sha256.
- OffIdentity (GPU, git): scene4_ref.py gives the base commit's arrays bit for bit on scene v2 and on the real-look
  scene (gripper look off).
- Pixels (GPU): with the look on, the nominal draw renders exactly the scene without the look (hidden slots and
  identity screws change no pixel); a drawn look changes only pixels of its own bodies (segment output); DRViews
  pairing over two calls.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_gripper_look_v12 -v
"""
from __future__ import annotations

import hashlib
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
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VCFG = REPO / "configs/so101_policy/vision"
BASE = os.environ.get("NEDM_LOOK_BASE", "362b48801")          # the commit before the gripper-look change
SCENE_V2 = Path(os.environ.get("NEDM_R2S_SCENE", "~/so101_r2s_claude/scene/real_scene_v2.json")).expanduser()
SCENE_WB = Path(os.environ.get("NEDM_R2S_SCENE_WB",
                               "~/so101_r2s_claude/v12/scene_wb/real_scene_v2_wb.json")).expanduser()
CALIB = Path(os.environ.get("NEDM_R2S_CALIB", "~/so101_r2s_claude/calib/calib_v1.json")).expanduser()
HAVE_MUJOCO = importlib.util.find_spec("mujoco") is not None
GPU = (importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None
       and torch.cuda.is_available())
HAVE_V2 = SCENE_V2.exists() and CALIB.exists()
HAVE_WB = SCENE_WB.exists() and CALIB.exists()
SHA_WB = "ab3e30edcf896cfcec8613ce897ecd89c68c1b0cc181a3815fd7c8bdaa8f41bf"
SHA_V2 = "4bce0b8dd6f22b5499b556275bca0bf248d1e96b1cd3babf169b34a8cb539d82"


def load_online(tag):
    return json.loads((VCFG / f"viral_r2s_online_v12_{tag}.json").read_text())


def look_r():
    return load_online("visR")["gripper_look"]


def look_h():
    return load_online("visH")["gripper_look"]


# a few fitted-like screw parts on the camera mount for scenes built in the tests (the real-look scene's layout)
def disc(c, r, k=10):
    return [[c[0] + r * math.cos(2 * math.pi * i / k), c[1], c[2] + r * math.sin(2 * math.pi * i / k)] for i in range(k)]


SCREW_PARTS = [
    {"name": "plate_screw0_head_lo", "body": "camera_mount", "group": "board", "color": [0.55, 0.41, 0.38],
     "polygon": disc((-0.0035, 0.0285, -0.0284), 0.0024), "offset": [0.0, -0.0005, 0.0]},
    {"name": "plate_screw0_recess", "body": "camera_mount", "group": "board", "color": [0.09, 0.09, 0.09],
     "polygon": disc((-0.0035, 0.0287, -0.0284), 0.0008), "offset": [0.0, -0.0003, 0.0]},
    {"name": "jaw_screw0_rim", "body": "moving_jaw_so101_v1", "group": "servo", "color": [0.32, 0.26, 0.24],
     "polygon": [[-0.0038 + 0.0025 * math.cos(a), -0.0083 + 0.0025 * math.sin(a), -0.0063]
                 for a in np.linspace(0, 2 * math.pi, 10, endpoint=False)], "offset": [0.0, 0.0, 0.0006]},
    {"name": "jaw_screw0_head", "body": "moving_jaw_so101_v1", "group": "servo", "color": [0.12, 0.10, 0.09],
     "polygon": [[-0.0038 + 0.0012 * math.cos(a), -0.0083 + 0.0012 * math.sin(a), -0.0064]
                 for a in np.linspace(0, 2 * math.pi, 10, endpoint=False)], "offset": [0.0, 0.0, 0.0002]},
]
PLATE = {"name": "camera_bracket_plate", "body": "camera_mount", "group": "board",
         "polygon": [[0.013159, 0.029292, -0.007253], [-0.006522, 0.033162, -0.006874],
                     [-0.010424, 0.026461, -0.032272], [0.012545, 0.021276, -0.034992]],
         "offset": [-0.000541, -0.002832, 0.00083]}
LOOK_T = {"p": 1.0, "plate_parts": ["camera_bracket_plate"], "plate_albedo": [0.02, 0.2],
          "bracket_parts": ["moving_jaw_so101_v1#8"], "screws": ["plate_screw0", "jaw_screw0"], "p_screw": 0.6,
          "n_marks": [1, 4], "mark_slots": 2, "p_edge": 0.5}


class Config(unittest.TestCase):
    def test_defaults_and_validation(self):
        from nedm.render.gripper_look import GripperLookDR
        from nedm.render.so101 import RealDR
        self.assertEqual(RealDR().gripper_look, {})
        self.assertFalse(RealDR().look_on)
        self.assertFalse(RealDR(gripper_look={"p": 0.0}).look_on)
        self.assertFalse(RealDR(gripper_look={"plate_albedo": None, "n_marks": [0, 0]}).look_on)
        self.assertTrue(RealDR(gripper_look={"n_marks": [0, 1]}).look_on)
        self.assertIsNone(RealDR().look_slots(None))
        for bad in ({"nope": 1}, {"p": 1.5}, {"plate_albedo": [0.3, 0.1]}, {"mark_size": [0.0, 0.002]},
                    {"n_marks": [0, 99]}, {"mark_surfaces": ["no_such_surface"]}, {"mark_tries": 0},
                    {"zones": [{"body": "camera_mount", "lo": [0, 0, 0], "hi": [-1, 0, 0]}]},
                    {"mark_standoff": 0.0001, "mark_thickness": 0.0002}, {"rect_aspect": [0.5, 2.0]},
                    {"screws": ["a", "a"]}, {"plate_parts": "camera_bracket_plate"}, {"edge_width": 0.0}):
            with self.assertRaises(ValueError, msg=str(bad)):
                GripperLookDR.from_dict(bad)
        g = GripperLookDR.from_dict({"note": "x", "mark_slots": 3})
        self.assertEqual(g.n_slots, 15)

    def test_slot_specs(self):
        from nedm.render.gripper_look import GripperLookDR
        g = GripperLookDR.from_dict({"mark_slots": 3})
        specs = g.mark_specs()
        self.assertEqual(len(specs), 15)
        sizes = sorted({s["size"] for s in specs})
        self.assertAlmostEqual(sizes[0], 0.0015)                   # both ends of mark_size
        self.assertAlmostEqual(sizes[-1], 0.004)
        self.assertEqual({s["shape"] for s in specs}, {"disc", "rect"})
        for s in specs:
            self.assertGreaterEqual(s["aspect"], 0.4)
            self.assertLessEqual(s["aspect"], 1.0)
        for surf in g.mark_surfaces:                              # every surface gets small and large slots
            ss = [s["size"] for s in specs if s["surface"] == surf]
            self.assertEqual(len(set(ss)), 3)

    def test_held_out_zones(self):
        from nedm.render.gripper_look import V12_EDGE, V12_PLATE_SCREWS, held_out_zones
        z = held_out_zones(0.003, 0.0015)
        self.assertEqual(len(z), 3)
        for s, zz in zip(V12_PLATE_SCREWS, z):
            c = np.asarray(s["centre"])
            self.assertTrue(np.allclose(np.asarray(zz["hi"]) - c, s["radius"] + 0.003, atol=2e-6))
            self.assertTrue(np.allclose(c - np.asarray(zz["lo"]), s["radius"] + 0.003, atol=2e-6))
        for k in ("a", "b"):
            p = np.asarray(V12_EDGE[k])
            self.assertTrue(((p >= np.asarray(z[2]["lo"]) + 0.003 - 1e-6) & (p <= np.asarray(z[2]["hi"]) - 0.003 + 1e-6)).all())

    def test_shipped_configs(self):
        from nedm.render.gripper_look import GripperLookDR, held_out_zones
        from nedm.so101_policy.vision import viral as VI
        d_on = json.loads((VCFG / "viral_r2s_online_relabel_roll_v8.json").read_text())
        for tag in ("visR", "visH"):
            d = load_online(tag)
            self.assertEqual({k: v for k, v in d.items() if k not in ("note", "gripper_look", "require_scene_sha256")},
                             {k: v for k, v in d_on.items() if k != "note"}, f"{tag}: D's online config + 2 keys")
            oc = VI.OnlineConfig.from_dict(d)
            self.assertTrue(GripperLookDR.from_dict(oc.gripper_look).on)
        r, h = GripperLookDR.from_dict(look_r()), GripperLookDR.from_dict(look_h())
        self.assertEqual(load_online("visR")["require_scene_sha256"], SHA_WB)
        self.assertEqual(load_online("visH")["require_scene_sha256"], SHA_V2)
        # visR: the fitted real look and the old sim look both inside the ranges, every draw on
        self.assertTrue(r.plate_albedo[0] <= 0.0318 and 0.0351 <= r.plate_albedo[1])
        self.assertTrue(r.plate_albedo[0] <= 0.104 * 0.7 and 0.104 * 1.5 <= r.plate_albedo[1])
        self.assertTrue(r.bracket_albedo[0] <= 0.36 and 0.42 <= r.bracket_albedo[1] and 0.713 <= r.bracket_albedo[1])
        self.assertEqual(len(r.screws), 6)
        self.assertGreater(r.p_screw, 0)
        self.assertAlmostEqual(r.screw_jitter, 0.001)
        self.assertGreater(r.p_edge, 0)
        self.assertEqual(tuple(r.n_marks), (0, 4))
        self.assertEqual(r.zones, ())
        # visH: no fitted screws, no edge, plate albedo above the fitted black (and the check's 0.065), zones
        self.assertEqual(h.screws, ())
        self.assertEqual(h.p_screw, 0.0)
        self.assertEqual(h.p_edge, 0.0)
        self.assertGreater(h.plate_albedo[0], 0.065)
        self.assertEqual([{k: v for k, v in z.items() if k != "what"} for z in h.zones], held_out_zones(0.003, 0.0015))
        same = {k for k in r.to_dict() if r.to_dict()[k] == h.to_dict()[k]}
        self.assertTrue({"n_marks", "mark_surfaces", "mark_slots", "mark_size", "mark_light", "mark_dark",
                         "p_light", "bracket_albedo"} <= same)

    def test_online_config_and_views_cfg(self):
        from nedm.so101_policy.vision import viral as VI
        with self.assertRaises(ValueError):
            VI.OnlineConfig.from_dict({"gripper_look": {"nope": 1}})
        with self.assertRaises(ValueError):
            VI.OnlineConfig.from_dict({"require_scene_sha256": "abc"})
        oc0 = VI.OnlineConfig.from_dict({})
        self.assertEqual(oc0.gripper_look, {})
        self.assertEqual(oc0.require_scene_sha256, "")
        cfg = replace(VI.ViralConfig(), scene="x.json", real_dr={"bright": {"mat": [0.5, 2.0]}})
        self.assertIs(VI.online_views_cfg(cfg, oc0), cfg)
        oc = VI.OnlineConfig.from_dict({"gripper_look": look_r(), "plate_mm": [1, 5]})
        v = VI.online_views_cfg(cfg, oc)
        self.assertIsNot(v, cfg)
        self.assertEqual(v.real_dr["gripper_look"], look_r())
        self.assertEqual(v.real_dr["bright"], {"mat": [0.5, 2.0]})
        self.assertEqual(v.real_dr["plate_thickness"], [0.001, 0.005])
        self.assertEqual(cfg.real_dr, {"bright": {"mat": [0.5, 2.0]}})          # the student's config untouched
        v1 = VI.online_views_cfg(cfg, VI.OnlineConfig.from_dict({"gripper_look": look_h()}))
        self.assertNotIn("plate_thickness", v1.real_dr)
        with self.assertRaises(ValueError):
            VI.online_views_cfg(replace(cfg, scene=""), oc)

    def test_scene_requirement(self):
        from nedm.so101_policy.vision import viral as VI
        cfg = replace(VI.ViralConfig(), scene="s.json", scene_sha256=SHA_V2)
        VI.check_scene_requirement(cfg, VI.OnlineConfig.from_dict({}))
        VI.check_scene_requirement(cfg, VI.OnlineConfig.from_dict({"require_scene_sha256": SHA_V2}))
        with self.assertRaises(ValueError):
            VI.check_scene_requirement(cfg, VI.OnlineConfig.from_dict({"require_scene_sha256": SHA_WB}))
        with self.assertRaises(ValueError):
            VI.check_scene_requirement(replace(cfg, scene_sha256=""),
                                       VI.OnlineConfig.from_dict({"require_scene_sha256": SHA_V2}))

    def test_ckpt_record(self):
        from nedm.so101_policy.vision import viral as VI
        st = SimpleNamespace(cfg=replace(VI.ViralConfig(), scene_sha256=SHA_WB))
        off = SimpleNamespace(o=VI.OnlineConfig.from_dict({}), st=st, il=None, tb=None)
        self.assertEqual(VI.OnlineDAgger.ckpt_record(off), {})
        on = SimpleNamespace(o=VI.OnlineConfig.from_dict({"gripper_look": look_r()}), st=st, il=None, tb=None)
        rec = VI.OnlineDAgger.ckpt_record(on)
        self.assertEqual(set(rec), {"v12_vis"})
        self.assertEqual(rec["v12_vis"]["gripper_look"], look_r())
        self.assertEqual(rec["v12_vis"]["scene_sha256"], SHA_WB)
        # the robot's floor mode (hw_lib, read from the checkpoint keys) ignores the record
        sys.path.insert(0, str(REPO / "scripts/so101_policy/real2sim/hw"))
        try:
            import hw_lib as H
        finally:
            sys.path.pop(0)
        self.assertEqual(H.student_floor_mode({"it": 1500, **rec}), H.student_floor_mode({"it": 1500}))
        tb = {"v8_online": {"table_bias": {"plane": [-0.01, 0.0, 0.0], "p": 1.0}, "student_floor": "observed"}}
        self.assertEqual(H.student_floor_mode({**tb, **rec}), H.student_floor_mode(tb))


@unittest.skipUnless(HAVE_MUJOCO, "the arm meshes need mujoco")
class Structure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nedm.render.so101 import RealDR, RealScene, SO101Scene, load_assets
        cls.assets = load_assets()
        cls.RealScene, cls.SO101Scene, cls.RealDR = RealScene, SO101Scene, RealDR

    def make(self, look=None, parts=(PLATE, *SCREW_PARTS)):
        real = self.RealScene(parts=tuple(parts))
        dr = self.RealDR(gripper_look=look or {})
        if dr.look_on:
            real = replace(real, look_slots=dr.look_slots(real))
        return self.SO101Scene(self.assets, real=real, cameras=("external", "wrist")), real, dr

    def state(self, n=3, seed=0):
        g = torch.Generator().manual_seed(seed)
        q = (torch.rand(n, 6, generator=g, dtype=torch.float64) - 0.5) * 1.2
        return q, torch.tensor([[0.2, 0.0, 0.015]] * n, dtype=torch.float64), \
            torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n, dtype=torch.float64)

    def test_off_no_parts(self):
        sc0, real0, _ = self.make()
        self.assertIsNone(real0.look_slots)
        self.assertEqual(sc0.look_names, [])
        self.assertIsNone(sc0.look_index)
        self.assertEqual(sc0.groups["look"], [])
        with self.assertRaises(ValueError):
            sc0.transforms(*self.state(), part_pose=torch.zeros(3, 0, 7))
        # a RealScene without the field = with look_slots None (the transforms of the old layout)
        sc1 = self.SO101Scene(self.assets, real=replace(real0, look_slots={}), cameras=("external", "wrist"))
        self.assertEqual(sc0.body_names, sc1.body_names)
        a, b = sc0.transforms(*self.state()), sc1.transforms(*self.state())
        self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))

    def test_look_parts_layout(self):
        from nedm.render.so101 import HIDDEN_Z
        sc0, _, _ = self.make()
        sc, real, dr = self.make(LOOK_T)
        n0, n1 = sc0.body_names, sc.body_names
        k = n0.index("cube")
        slots = [p["name"] for p in real.look_slots["slots"]]
        self.assertEqual(n1, n0[:k] + slots + n0[k:])           # the slots after the moving parts, before the cube
        self.assertEqual(len(slots), 5 * 2 + 1)                  # 2 per surface + the edge strip
        self.assertEqual(real.look_slots["movable"], [p["name"] for p in SCREW_PARTS])
        self.assertEqual(sc.look_names, real.look_slots["movable"] + slots)
        self.assertEqual(sorted(sc.groups["look"]), [n1.index(s) for s in slots])
        self.assertTrue(all(i < len(sc.mesh_body) for i in sc.look_index.tolist()))
        nm = len(real.look_slots["movable"])
        self.assertTrue(torch.equal(sc.look_pose0[:nm], torch.tensor([[0, 0, 0, 0, 0, 0, 1.0]] * nm,
                                                                     dtype=torch.float64)))
        self.assertTrue((sc.look_pose0[nm:, 2] == HIDDEN_Z).all())
        q, cp, cq = self.state()
        b0, c0 = sc0.transforms(q, cp, cq)
        b1, c1 = sc.transforms(q, cp, cq)
        keep = [n1.index(x) for x in n0]
        self.assertTrue(torch.equal(b0, b1[:, keep]))           # movable parts at identity: the same poses
        self.assertTrue(torch.equal(c0, c1))
        hid = b1[:, [n1.index(s) for s in slots]]
        self.assertTrue(torch.allclose(hid[..., :3], torch.tensor([0.0, 0.0, HIDDEN_Z])))
        # part_pose = look_pose0 gives the same as None; a local shift moves the part in its body frame
        pp = sc.look_pose0.expand(3, *sc.look_pose0.shape).clone()
        self.assertTrue(torch.equal(sc.transforms(q, cp, cq, part_pose=pp)[0], b1))
        pp[:, 0, :3] = torch.tensor([0.001, -0.002, 0.003])
        b2, _ = sc.transforms(q, cp, cq, part_pose=pp)
        i0 = n1.index(sc.look_names[0])
        from nedm.render.so101 import qrot, wxyz
        exp = b1[:, i0, :3].double() + qrot(wxyz(b1[:, i0, 3:].double()), pp[:, 0, :3])
        self.assertTrue(torch.allclose(b2[:, i0, :3].double(), exp, atol=1e-6))
        self.assertTrue(torch.equal(b2[:, i0, 3:], b1[:, i0, 3:]))
        others = [i for i in range(len(n1)) if i != i0]
        self.assertTrue(torch.equal(b2[:, others], b1[:, others]))

    def test_bad_slots(self):
        sc, real, dr = self.make(LOOK_T)
        bad = dict(real.look_slots)
        bad["slots"] = [dict(bad["slots"][0], body="base")]
        with self.assertRaises(ValueError):
            self.SO101Scene(self.assets, real=replace(real, look_slots=bad), cameras=("external", "wrist"))
        bad["slots"] = [dict(real.look_slots["slots"][0], group="board")]
        with self.assertRaises(ValueError):
            self.SO101Scene(self.assets, real=replace(real, look_slots=bad), cameras=("external", "wrist"))
        with self.assertRaises(ValueError):                    # screws named in the config must exist in the scene
            self.make({**LOOK_T, "screws": ["plate_screw7"]})
        # a scene built for another look config: the draw refuses it
        sc2, _, _ = self.make({**LOOK_T, "mark_slots": 1})
        with self.assertRaises(ValueError):
            dr.draw(sc2, 2, torch.Generator().manual_seed(0))
        sc3, _, _ = self.make()
        with self.assertRaises(ValueError):
            dr.draw(sc3, 2, torch.Generator().manual_seed(0))

    def draw_pair(self, look, n=64, seed=3):
        sc, real, dr = self.make(look)
        d_on = dr.draw(sc, n, torch.Generator().manual_seed(seed), look_generator=torch.Generator().manual_seed(9))
        d_off = replace(dr, gripper_look={}).draw(sc, n, torch.Generator().manual_seed(seed))
        return sc, real, dr, d_on, d_off

    def test_other_draws_unchanged(self):
        sc, real, dr, d_on, d_off = self.draw_pair(LOOK_T)
        self.assertEqual(set(d_on), set(d_off) | {"part_pose"})
        for k in d_off:
            if k != "colors":
                self.assertTrue(torch.equal(d_on[k], d_off[k]), k)
        names = sc.body_names
        targets = {"camera_bracket_plate", "moving_jaw_so101_v1#8"} | set(sc.look_names)
        for i, nm in enumerate(names):
            if nm not in targets:
                self.assertTrue(torch.equal(d_on["colors"][:, i], d_off["colors"][:, i]), nm)
        self.assertFalse(torch.equal(d_on["colors"][:, names.index("camera_bracket_plate")],
                                     d_off["colors"][:, names.index("camera_bracket_plate")]))
        # without the own generator the look draws come after every other draw of the shared generator
        d_sh = dr.draw(sc, 64, torch.Generator().manual_seed(3))
        for k in d_off:
            if k != "colors":
                self.assertTrue(torch.equal(d_sh[k], d_off[k]), k)
        # determinism
        d2 = dr.draw(sc, 64, torch.Generator().manual_seed(3), look_generator=torch.Generator().manual_seed(9))
        self.assertTrue(all(torch.equal(d_on[k], d2[k]) for k in d_on))

    def test_marks_inside_and_counts(self):
        from nedm.render.gripper_look import GripperLookDR, _inside
        for look in (LOOK_T, {**LOOK_T, "zones": look_h()["zones"], "n_marks": [4, 4], "mark_slots": 3}):
            sc, real, dr, d, _ = self.draw_pair(look, n=200)
            g = GripperLookDR.from_dict(look)
            specs = g.mark_specs()
            pp = d["part_pose"]
            shown = torch.stack([pp[:, sc.look_names.index(f"look_mark{m}"), 2] > -50 for m in range(len(specs))], 1)
            self.assertTrue((shown.sum(1) <= int(look["n_marks"][1])).all())
            self.assertGreater(int(shown.sum()), 50)
            for m, spec in enumerate(specs):
                surf = g.SURFACES[spec["surface"]]
                o, nrm, e1 = (torch.tensor(surf[k], dtype=torch.float64) for k in ("origin", "normal", "e1"))
                e2 = torch.linalg.cross(nrm, e1)
                s = shown[:, m]
                if not s.any():
                    continue
                p = pp[s, sc.look_names.index(f"look_mark{m}"), :3] - g.mark_standoff * nrm
                rel = p - o
                self.assertTrue(torch.allclose(rel @ nrm, torch.zeros(len(p), dtype=torch.float64), atol=1e-7))
                xy = torch.stack([rel @ e1, rel @ e2], -1)              # (the 6-digit axes: 1e-6 from orthonormal)
                self.assertTrue(_inside(xy, surf["poly2"], spec["radius"] + g.MARK_INSET - 1e-7).all())
                for z in g.zones:
                    if z["body"] != surf["body"]:
                        continue
                    lo, hi = torch.tensor(z["lo"]), torch.tensor(z["hi"])
                    gap = torch.maximum(torch.maximum(lo - p, p - hi), torch.zeros_like(p)).norm(dim=-1)
                    self.assertTrue((gap >= spec["radius"] - 1e-7).all(), (m, spec["surface"]))
                # the mark's turn: its z axis is the surface normal
                from nedm.render.so101 import qrot, wxyz
                qm = wxyz(pp[s, sc.look_names.index(f"look_mark{m}"), 3:])
                qz = qrot(qm, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(len(qm), 3))
                self.assertTrue(torch.allclose(qz, nrm.expand_as(qz), atol=1e-5))

    def test_screws(self):
        sc, real, dr, d, _ = self.draw_pair(LOOK_T, n=400)
        pp = d["part_pose"]
        ring, centre = sc.look_names.index("plate_screw0_head_lo"), sc.look_names.index("plate_screw0_recess")
        vis = pp[:, ring, 2] > -50
        self.assertTrue(torch.equal(vis, pp[:, centre, 2] > -50))        # a screw's parts move together
        self.assertTrue(torch.equal(pp[vis, ring, :3], pp[vis, centre, :3]))
        frac = float(vis.double().mean())
        self.assertLess(abs(frac - 0.6), 0.1)
        shift = pp[vis, ring, :3]
        self.assertTrue((shift.norm(dim=-1) <= 0.001 * math.sqrt(2) + 1e-12).all())
        self.assertGreater(float(shift.norm(dim=-1).max()), 0.0005)
        from nedm.render.gripper_look import polygon_plane
        _, nrm, _, _ = polygon_plane(SCREW_PARTS[0]["polygon"])
        self.assertTrue(torch.allclose(shift @ torch.tensor(nrm), torch.zeros(len(shift), dtype=torch.float64),
                                       atol=1e-12))
        # colour roles: the recess keeps its ratio to the ring / centre reference of its screw
        names = sc.body_names
        c = d["colors"][vis][:, names.index("plate_screw0_recess")]
        self.assertTrue((c > 0).all())

    def test_world_probability(self):
        sc, real, dr, d, d_off = self.draw_pair({**LOOK_T, "p": 0.3}, n=600)
        names = sc.body_names
        i = names.index("camera_bracket_plate")
        changed = (d["colors"][:, i] != d_off["colors"][:, i]).any(-1)
        self.assertLess(abs(float(changed.double().mean()) - 0.3), 0.07)
        # worlds without the look: every look part at its nominal pose
        self.assertTrue(torch.equal(d["part_pose"][~changed], sc.look_pose0.expand(int((~changed).sum()),
                                                                                    *sc.look_pose0.shape)))

    @unittest.skipUnless(HAVE_V2, "needs the v2 scene file")
    def test_load_real_setup(self):
        from nedm.so101_policy.vision import viral as VI
        cfg = replace(VI.ViralConfig(), cameras=("external", "wrist"), res=224, scene=str(SCENE_V2), calib=str(CALIB))
        rs0 = VI.load_real_setup(cfg)
        self.assertIsNone(rs0.real.look_slots)
        self.assertFalse(rs0.dr.look_on)
        rs1 = VI.load_real_setup(replace(cfg, real_dr={"gripper_look": look_h()}))
        self.assertTrue(rs1.dr.look_on)
        self.assertEqual(rs1.real.look_slots["movable"], [])
        self.assertEqual(len(rs1.real.look_slots["slots"]), 15)
        self.assertEqual(rs1.scene_sha256, rs0.scene_sha256)
        with self.assertRaises(ValueError):                    # visR's screws are not in scene v2
            VI.load_real_setup(replace(cfg, real_dr={"gripper_look": look_r()}))


@unittest.skipUnless(HAVE_WB and HAVE_V2, "needs the real-look scene and scene v2")
class RealFiles(unittest.TestCase):
    def test_visR_student_config_pin(self):
        from nedm.so101_policy.vision import viral as VI
        st = json.loads((VCFG / "viral_r2s_v12_visR_student.json").read_text())
        d1 = json.loads((VCFG / "viral_r2s_v1.json").read_text())
        self.assertEqual({k: v for k, v in st.items() if k not in ("note", "scene", "scene_sha256")},
                         {k: v for k, v in d1.items() if k not in ("note", "scene")})
        self.assertEqual(st["scene_sha256"], SHA_WB)
        self.assertEqual(hashlib.sha256(SCENE_WB.read_bytes()).hexdigest(), SHA_WB)
        # D's pinned config (scene v2) + the visR config keys (the cmd_online update) pass pin_real knowingly
        d = replace(VI.ViralConfig.from_dict({k: v for k, v in d1.items() if k != "note"}), scene=str(SCENE_V2),
                    calib=str(CALIB))
        VI.pin_real(d)
        self.assertEqual(d.scene_sha256, SHA_V2)
        crops_d = dict(d.crops)
        cd = {**VI.asdict(d), **{k: v for k, v in st.items() if k != "note"}}
        cd["scene"], cd["calib"] = str(SCENE_WB), str(CALIB)
        r = VI.ViralConfig.from_dict(cd)
        VI.pin_real(r)                                          # no allow_change needed: the config pins the new file
        self.assertEqual(r.scene_sha256, SHA_WB)
        self.assertEqual(r.crops, crops_d)                     # the robot bundle reads cfg.crops: unchanged
        with self.assertRaises(ValueError):                    # the pin refuses scene v2 under the visR config
            VI.pin_real(replace(r, scene=str(SCENE_V2)))
        with self.assertRaises(ValueError):                    # and D's pin refuses the new scene without the config
            VI.pin_real(replace(d, scene=str(SCENE_WB)))
        VI.check_scene_requirement(r, VI.OnlineConfig.from_dict(load_online("visR")))
        VI.check_scene_requirement(d, VI.OnlineConfig.from_dict(load_online("visH")))
        with self.assertRaises(ValueError):
            VI.check_scene_requirement(r, VI.OnlineConfig.from_dict(load_online("visH")))
        with self.assertRaises(ValueError):
            VI.check_scene_requirement(d, VI.OnlineConfig.from_dict(load_online("visR")))
        # the saved student config (no gripper_look: training views only) loads on the robot side
        r2 = VI.ViralConfig.from_dict(VI.cfg_dict(r))
        self.assertNotIn("gripper_look", r2.real_dr or {})
        self.assertEqual(r2.crops, crops_d)

    def test_sim_check_finds_scene(self):
        sys.path.insert(0, str(REPO / "scripts/so101_policy/real2sim"))
        try:
            import lowtable_gate as LG
        except Exception as e:                                  # pragma: no cover
            self.skipTest(f"lowtable_gate import: {e}")
        finally:
            sys.path.pop(0)
        tmp = Path(tempfile.mkdtemp(prefix="look_scene_"))
        try:
            ck = tmp / "student.pt"
            torch.save({"cfg": {"scene_sha256": SHA_WB}, "it": 7}, ck)
            p, it = LG.student_scene(ck, tmp / "data", [SCENE_V2.parent, SCENE_WB.parent])
            self.assertEqual(Path(p).resolve(), SCENE_WB.resolve())
            self.assertEqual(it, 7)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(GPU and HAVE_V2 and HAVE_WB, "needs newton on CUDA and the scene files")
class OffIdentity(unittest.TestCase):
    """The default options give the base commit's arrays bit for bit (scene4_ref.py on scene v2 and the real-look
    scene)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="look_base_"))
        r = subprocess.run(f"git -C {REPO} archive {BASE} src configs contact_nrd | tar -x -C {cls.tmp}", shell=True,
                           capture_output=True)
        vendor = REPO / "contact_nrd/so101_grasp/vendor"
        (cls.tmp / "contact_nrd/so101_grasp").mkdir(parents=True, exist_ok=True)
        if vendor.exists() and not (cls.tmp / "contact_nrd/so101_grasp/vendor").exists():
            (cls.tmp / "contact_nrd/so101_grasp/vendor").symlink_to(vendor.resolve())
        cls.ok = r.returncode == 0 and (cls.tmp / "src").exists()

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

    def test_v2_scene_bit_identical(self):
        self.check(SCENE_V2, "v2")

    def test_wb_scene_bit_identical(self):
        self.check(SCENE_WB, "wb")


@unittest.skipUnless(GPU and HAVE_V2 and HAVE_WB, "needs newton on CUDA and the scene files")
class Pixels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nedm.so101_policy.vision import viral as VI
        cls.VI = VI
        cls.tmp = Path(tempfile.mkdtemp(prefix="look_px_"))
        cls.plain = {}
        for sc in (SCENE_V2, SCENE_WB):                       # copies without the nominal blur / chroma (exact pixels)
            d = json.loads(sc.read_text())
            d["image_fx"] = {}
            d["calib"] = str(CALIB)
            cls.plain[sc] = cls.tmp / sc.name
            cls.plain[sc].write_text(json.dumps(d))
        g = torch.Generator().manual_seed(4)
        n = 8
        q = (torch.rand(n, 6, generator=g) - 0.5) * torch.tensor([0.6, 0.4, 0.4, 0.4, 0.6, 0.3])
        q += torch.tensor([0.0, -0.2, 0.5, 1.2, 0.0, 0.15])
        cls.n = n
        cls.q = q.cuda()
        cls.cp = torch.tensor([[0.22, 0.0, 0.015]] * n).cuda()
        cls.cq = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n).cuda()
        cls.goal = torch.tensor([[0.25, 0.05, 0.3]] * n).cuda()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cfg(self, scene, look=None):
        c = replace(self.VI.ViralConfig(), cameras=("external", "wrist"), res=224, scene=str(scene), calib=str(CALIB))
        return replace(c, real_dr={"gripper_look": look}) if look else c

    def test_nominal_unchanged_and_own_pixels(self):
        from nedm.render.so101 import SO101Renderer
        for scene, look in ((SCENE_WB, look_r()), (SCENE_V2, look_h())):
            v_off = self.VI.make_views(self.cfg(scene), self.n, seed=5, device="cuda")
            v_on = self.VI.make_views(self.cfg(scene, look), self.n, seed=5, device="cuda")
            self.assertTrue(v_on.look_on)
            self.assertFalse(v_off.look_on)
            a = v_off.render(self.q, self.cp, self.cq, self.goal)
            b = v_on.render(self.q, self.cp, self.cq, self.goal)
            self.assertTrue(torch.equal(a, b), f"{scene.name}: nominal renders differ")
            nd_on, nd_off = v_on.nominal_draw(self.n), v_off.nominal_draw(self.n)
            self.assertEqual(set(nd_on), set(nd_off) | {"part_pose"})
            a = v_off.render(self.q, self.cp, self.cq, self.goal, dr=nd_off, light=False)
            b = v_on.render(self.q, self.cp, self.cq, self.goal, dr=nd_on, light=False)
            self.assertTrue(torch.equal(a, b), f"{scene.name}: nominal-draw renders differ")
            # a drawn look (other draws nominal) changes only pixels of the look bodies (segment, either render;
            # the scene copy without image effects, no shadows: exact per-pixel ownership)
            rs = self.VI.load_real_setup(self.cfg(self.plain[scene], look))
            rr = SO101Renderer(self.n, 224, 224, backend="newton", cameras=v_on.cameras, goal="block", real=rs.real,
                               segment=True, shadows=False)
            g = torch.Generator(device="cuda").manual_seed(1)
            full = rs.dr.draw(rr.scene, self.n, g, device="cuda", look_generator=torch.Generator(device="cuda")
                              .manual_seed(2))
            nd = {k: v.clone() for k, v in v_on.nominal_draw(self.n).items()}
            names = rr.scene.body_names
            gl = rs.dr.look_dr()
            targets = [names.index(x) for x in list(gl.plate_parts) + list(gl.bracket_parts) + rr.scene.look_names]
            look_only = dict(nd)
            look_only["part_pose"] = full["part_pose"]
            cols = nd["colors"].clone()
            cols[:, targets] = full["colors"][:, targets].to(cols)
            look_only["colors"] = cols
            r0, s0 = rr.render_dr(self.q, self.cp, self.cq, self.goal, draw=nd, segment=True)
            r1, s1 = rr.render_dr(self.q, self.cp, self.cq, self.goal, draw=look_only, segment=True)
            ch = (r0 != r1).any(-1)
            own = torch.zeros_like(ch)
            for t in targets:
                own |= (s0 == t) | (s1 == t)
            own |= (s0 == -2) | (s1 == -2)
            self.assertGreater(int(ch.sum()), 0)
            self.assertEqual(int((ch & ~own).sum()), 0, f"{scene.name}: pixels outside the look bodies changed")
            wi = v_on.cameras.index("wrist")
            self.assertGreater(int(ch[:, wi].sum()), int(ch[:, 1 - wi].sum()))   # mostly the wrist view
            # hidden slots have no pixel
            for k, nm in enumerate(rr.scene.look_names):
                hid = full["part_pose"][:, k, 2] < -50
                self.assertEqual(int((s1[hid] == names.index(nm)).sum()), 0, nm)
            del rr, v_on, v_off
            torch.cuda.empty_cache()

    def test_views_pairing(self):
        v_off = self.VI.make_views(self.cfg(SCENE_V2), self.n, seed=5, device="cuda")
        v_on = self.VI.make_views(self.cfg(SCENE_V2, look_h()), self.n, seed=5, device="cuda")
        for _ in range(2):                                     # the own generator keeps every call paired
            a, b = v_off.draw(self.n), v_on.draw(self.n)
            self.assertEqual(set(b), set(a) | {"part_pose"})
            for k in a:
                if k != "colors":
                    self.assertTrue(torch.equal(a[k], b[k]), k)
            m = v_on.merge_nominal(b, torch.tensor([True, False] * (self.n // 2), device="cuda"))
            self.assertTrue(torch.equal(m["part_pose"][0], v_on.r.scene.look_pose0.to(m["part_pose"])))


if __name__ == "__main__":
    unittest.main()
