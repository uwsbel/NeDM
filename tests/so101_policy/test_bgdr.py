"""Tests of the background randomisation of the real-setup renderer (studentlab external-camera fix, 2026-10-23;
``nedm.render.bg_dr``, ``RealScene.floor_turn`` / ``wall_end``, ``RealDR.floor_turn_deg`` / ``floor_xy`` /
``wall_end`` / ``bg``, ``SO101Renderer(bg_cameras=)``, ``viral.DRViews`` with the layers).

- BgLayers (CPU): draw shapes / determinism / off rows, ``apply_bg`` touches only the floor, walls, mat and dark area
  (bit for bit elsewhere and in rows that are off), the prior placement, bad configs.
- SceneGeometry (CPU, mujoco): the nominal floor turn, the wall end piece, RealDR's new draws (and that every other
  draw of a seed stays the same with them on).
- BitIdentity (GPU, git): ``bgdr_ref.py`` (DRViews nominal / randomised renders, draws, merge_nominal, RealDR +
  render_dr with segment and hdr outputs of the v2 scene) gives the same arrays with the base commit's code and with
  this tree.
- WithLayers (GPU): a v3-like scene: nominal renders unchanged by the zero layers, the plate / cube / arm pixels
  unchanged by the layers (no image_fx), merge_nominal with the layers, the placement prior.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest \
        tests.so101_policy.test_bgdr -v
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
from pathlib import Path

import numpy as np
import torch

from nedm.render import bg_dr as BG

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = os.environ.get("NEDM_BGDR_BASE", "1ca7ec9cf")          # the commit before the background change
SCENE_V2 = Path(os.environ.get("NEDM_R2S_SCENE", "~/so101_r2s_claude/scene/real_scene_v2.json")).expanduser()
CALIB = Path(os.environ.get("NEDM_R2S_CALIB", "~/so101_r2s_claude/calib/calib_v1.json")).expanduser()
HAVE_MUJOCO = importlib.util.find_spec("mujoco") is not None
GPU = (importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None
       and torch.cuda.is_available())
HAVE_SCENE = SCENE_V2.exists() and CALIB.exists()
CODES = {g: i for i, g in enumerate(("printed", "servo", "cube", "plate", "floor", "mat", "walls", "board", "room",
                                     "room_floor", "strip"))}
DARK = (CODES["room"], CODES["room_floor"])


def v3_like(d2: dict) -> dict:
    """A v3-like scene dict from the v2 dict (trials/bgdr/make_scene_v3.py: the geometry and dr changes)."""
    d = copy.deepcopy(d2)
    d["version"] = "real_scene_v3_test"
    g = d["geometry"]
    g["floor_xy"] = [-0.045, 0.80, -0.60, 0.70]
    g["floor_turn"] = {"deg": -4.0, "pivot": [-0.046, -0.300]}
    g["walls"][0][0] = [0.015, 0.457]
    g["wall_end"] = {"wall": 0, "slide": 0.04}
    d["dr"].update(floor_turn_deg=8.0, floor_xy=0.03, wall_end=0.04,
                   bg={"floor_radial": [0.0, 3.0], "mat_grad": 0.5, "fold_px": [0.5, 2.0], "fold_gain": [-0.6, 1.5]})
    d["calib"] = str(CALIB)
    return d


class BgLayers(unittest.TestCase):
    def test_draw_shapes_determinism_and_off(self):
        b = BG.BgDR()
        d1 = b.draw(5, 1, 32, torch.Generator().manual_seed(0))
        d2 = b.draw(5, 1, 32, torch.Generator().manual_seed(0))
        self.assertEqual(tuple(d1["bg_tex"].shape), (5, 1, 32, 32, 3))
        self.assertEqual(tuple(d1["bg_low"].shape), (5, 1, 32, 32))
        self.assertEqual(tuple(d1["bg_par"].shape), (5, 1, BG.NPAR))
        self.assertEqual(d1["bg_tex"].dtype, torch.float16)
        for k in BG.KEYS:
            self.assertTrue(torch.equal(d1[k], d2[k]), k)
        off = BG.BgDR(p=0.0).draw(4, 2, 16, torch.Generator().manual_seed(0))
        self.assertTrue(all(float(off[k].abs().max()) == 0 for k in BG.KEYS))
        nom = BG.BgDR.nominal(3, 1, 16, "cpu")
        self.assertTrue(all(float(nom[k].abs().max()) == 0 for k in BG.KEYS))

    def test_noise_family_unit_std(self):
        n = BG.BgDR().noise(8, 64, 64, torch.Generator().manual_seed(1), torch.device("cpu"))
        s = n.std(dim=(1, 2))
        self.assertTrue(torch.allclose(s, torch.ones_like(s), atol=1e-4))

    def test_bad_configs(self):
        with self.assertRaises(ValueError):
            BG.BgDR.from_dict({"family": "gauss_v1"})                # training implements one family only
        with self.assertRaises(ValueError):
            BG.BgDR.from_dict({"texture_amp": 0.1})
        self.assertEqual(BG.BgDR.from_dict({"tex_amp": [0.01, 0.02]}).tex_amp, (0.01, 0.02))

    def scene(self, m=6, H=32):
        """Linear colour and a group map: left third dark area, middle floor, right mat, a block of arm / plate /
        cube pixels in the middle of everything, top rows walls."""
        g = torch.Generator().manual_seed(2)
        x = 0.05 + 0.5 * torch.rand(m, H, H, 3, generator=g)
        gid = torch.full((m, H, H), CODES["floor"])
        gid[:, :, : H // 3] = CODES["room_floor"]
        gid[:, :, 2 * H // 3:] = CODES["mat"]
        gid[:, :4] = CODES["walls"]
        gid[:, 12:20, 6:26] = torch.tensor([CODES["printed"], CODES["plate"], CODES["cube"], CODES["servo"],
                                            CODES["board"]]).repeat_interleave(4)[:20]
        gid[:, 0, 0] = -1
        return x, gid

    def test_apply_touches_only_background_groups(self):
        x, gid = self.scene()
        b = BG.BgDR(floor_radial=(0.5, 2.0), mat_grad=0.5, edge_blur_px=(1.0, 1.5), tex_p=1.0, clutter_p=1.0,
                    spot_p=1.0, low_p=1.0, fold_p=1.0)
        d = b.draw(6, 1, 32, torch.Generator().manual_seed(3))
        lay = {k: v[:, 0] for k, v in d.items()}
        ct = torch.linspace(0.6, 1.0, 32)[None].expand(32, 32)
        y = BG.apply_bg(x, gid, lay, CODES, DARK, 1, ct)
        prot = torch.isin(gid, torch.tensor([CODES[k] for k in ("printed", "plate", "cube", "servo", "board")])) | \
            (gid < 0)
        self.assertTrue(torch.equal(y[prot], x[prot]))
        for grp in ("floor", "mat", "walls"):
            self.assertFalse(torch.equal(y[gid == CODES[grp]], x[gid == CODES[grp]]), grp)
        self.assertFalse(torch.equal(y[gid == CODES["room_floor"]], x[gid == CODES["room_floor"]]))
        self.assertGreaterEqual(float(y.min()), 0.0)
        # rows that are off: unchanged bit for bit; x is not modified in place
        lay["bg_par"] = lay["bg_par"].clone()
        lay["bg_par"][::2, 0] = 0
        x0 = x.clone()
        y2 = BG.apply_bg(x, gid, lay, CODES, DARK, 1, ct)
        self.assertTrue(torch.equal(y2[::2], x[::2]))
        self.assertTrue(torch.equal(x, x0))
        # supersampling: layers repeated per sub-pixel (an all-dark map: no boundary band)
        g2 = torch.full((2, 64, 64), CODES["room"])
        x2 = 0.1 * torch.ones(2, 64, 64, 3)
        d2 = BG.BgDR(flat_p=0.0, ramp_amp=(0.0, 0.0)).draw(2, 1, 32, torch.Generator().manual_seed(4))
        y3 = BG.apply_bg(x2, g2, {k: v[:, 0] for k, v in d2.items()}, CODES, DARK, 2)
        exp = (0.1 + d2["bg_tex"][:, 0].float()).clamp_min(0).repeat_interleave(2, 1).repeat_interleave(2, 2)
        exp = torch.where(d2["bg_par"][:, 0, 0, None, None, None] > 0.5, exp, x2)
        self.assertTrue(torch.allclose(y3, exp, atol=1e-6))

    def test_flat_profile_and_blur_band(self):
        x, gid = self.scene(m=2)
        lay = BG.BgDR.nominal(2, 1, 32, "cpu")
        lay = {k: v[:, 0] for k, v in lay.items()}
        lay["bg_par"][:, 0] = 1.0                                     # on, everything else neutral
        y = BG.apply_bg(x, gid, lay, CODES, DARK, 1)
        self.assertTrue(torch.equal(y, x))                            # neutral layers: no change
        lay["bg_par"][:, 1] = 1.0                                     # flat dark area = its mean colour
        y = BG.apply_bg(x, gid, lay, CODES, DARK, 1)
        dk = gid == CODES["room_floor"]
        for i in range(2):
            mean = x[i][dk[i]].mean(0)
            self.assertTrue(torch.allclose(y[i][dk[i]], mean.expand_as(y[i][dk[i]]), atol=1e-6))
        lay["bg_par"][:, 1] = 0.0
        lay["bg_par"][:, 4] = 1.5                                     # boundary blur: only the band changes
        y = BG.apply_bg(x, gid, lay, CODES, DARK, 1)
        changed = (y != x).any(-1)
        cols = changed.any(0).any(0).nonzero().flatten()
        self.assertTrue(bool(changed.any()))
        self.assertTrue(int(cols.min()) >= 32 // 3 - BG.BAND_PX and int(cols.max()) < 32 // 3 + BG.BAND_PX)

    def test_prior_placement(self):
        prior = torch.zeros(1, 64, 64)
        prior[0, 40:, :20] = 1.0
        b = BG.BgDR(p=1.0, tex_p=0.0, spot_p=0.0, clutter_p=1.0, clutter_n=(6, 6), low_p=0.0, fold_p=0.0,
                    prior_floor=0.0)
        d = b.draw(64, 1, 64, torch.Generator().manual_seed(5), prior=prior)
        t = d["bg_tex"][:, 0].float().abs().sum(-1).sum(0)
        inside = float(t[40:, :20].sum() / t.sum())
        self.assertGreater(inside, 0.4)                               # uniform placement: ~0.2 incl. long bars
        u = b.draw(64, 1, 64, torch.Generator().manual_seed(5))
        tu = u["bg_tex"][:, 0].float().abs().sum(-1).sum(0)
        self.assertLess(float(tu[40:, :20].sum() / tu.sum()), inside)


@unittest.skipUnless(HAVE_MUJOCO, "the arm meshes need mujoco")
class SceneGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nedm.render.so101 import RealScene, SO101Scene, load_assets
        cls.assets = load_assets()
        cls.RealScene, cls.SO101Scene = RealScene, SO101Scene

    def make(self, **kw):
        base = dict(floor_xy=(-0.045, 0.80, -0.60, 0.70), walls=(((0.015, 0.457), (0.59, 0.457), 0.53),
                                                                   ((0.59, 0.457), (0.59, -0.456), 0.38)))
        base.update(kw)
        real = self.RealScene(**base)
        return self.SO101Scene(self.assets, real=real, cameras=("external", "wrist")), real

    def test_defaults_off(self):
        from nedm.render.so101 import RealDR
        r = self.RealScene()
        self.assertIsNone(r.floor_turn)
        self.assertIsNone(r.wall_end)
        d = RealDR()
        self.assertEqual((d.floor_turn_deg, d.floor_xy, d.wall_end, d.bg), (0.0, 0.0, 0.0, {}))
        self.assertFalse(d.bg_on)
        self.assertEqual(d.bg_cameras(("external", "wrist")), ())
        self.assertFalse(RealDR(bg={"p": 0.0}).bg_on)

    def test_floor_turn_and_wall_end_geometry(self):
        sc, real = self.make(floor_turn={"deg": -4.0, "pivot": (-0.046, -0.300)}, wall_end={"wall": 0, "slide": 0.04})
        names = sc.env_names
        self.assertEqual(names[:5], ["floor", "mat", "wall0", "wall0_end", "wall1"])
        env = sc.env_pose.numpy()
        c = np.array([(-0.045 + 0.80) / 2, (-0.60 + 0.70) / 2])
        piv = np.array([-0.046, -0.300])
        th = math.radians(-4.0)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        self.assertTrue(np.allclose(env[0, :2], piv + R @ (c - piv), atol=1e-9))
        self.assertTrue(np.allclose(env[0, 3:], [0, 0, math.sin(th / 2), math.cos(th / 2)], atol=1e-9))
        # main wall from x 0.055, end piece from 0.015 to 0.105 (inset 0.5 mm away from the mat)
        meshes = {m.name: m for m in sc.scene.bodies}
        vw = meshes["wall0"].vertices + env[2, :3].astype(np.float32)
        ve = meshes["wall0_end"].vertices + env[3, :3].astype(np.float32)
        self.assertAlmostEqual(float(vw[:, 0].min()), 0.055, places=5)
        self.assertAlmostEqual(float(ve[:, 0].min()), 0.015, places=5)
        self.assertAlmostEqual(float(ve[:, 0].max()), 0.015 + 2 * 0.04 + 0.01, places=5)
        self.assertAlmostEqual(float(ve[:, 1].min()), float(vw[:, 1].min()) + 0.0005, places=5)
        self.assertLess(float(ve[:, 2].max()), float(vw[:, 2].max()))
        self.assertIn(sc.body_names.index("wall0_end"), sc.groups["walls"])
        with self.assertRaises(ValueError):
            self.make(wall_end={"wall": 0, "slide": 0.4})            # longer than the wall allows
        with self.assertRaises(ValueError):
            self.make(wall_end={"wall": 5, "slide": 0.04})

    def test_new_draws_after_the_old_ones(self):
        from nedm.render.so101 import RealDR
        sc, real = self.make(floor_turn={"deg": -4.0, "pivot": (-0.046, -0.300)}, wall_end={"wall": 0, "slide": 0.04})
        old = RealDR(wall_xy=0.02, mat_xy=0.01)
        new = RealDR(wall_xy=0.02, mat_xy=0.01, floor_turn_deg=8.0, floor_xy=0.03, wall_end=0.04)
        a = old.draw(sc, 64, torch.Generator().manual_seed(0))
        b = new.draw(sc, 64, torch.Generator().manual_seed(0))
        for k in ("colors", "cam_offset"):
            self.assertTrue(torch.equal(a[k], b[k]), k)
        names = sc.env_names
        same = [i for i, nm in enumerate(names) if nm not in ("floor", "wall0_end")]
        self.assertTrue(torch.equal(a["env_pose"][:, same], b["env_pose"][:, same]))
        env0 = sc.env_pose
        # wall end piece: follows its wall's xy shift (old draw), then slides along the wall (x) by <= 0.04
        dw = a["env_pose"][:, 2, :2] - env0[2, :2]
        de = b["env_pose"][:, 3, :2] - env0[3, :2]
        self.assertTrue(torch.allclose(a["env_pose"][:, 3, :2] - env0[3, :2], dw))
        slide = de - dw
        self.assertTrue(torch.allclose(slide[:, 1], torch.zeros(64, dtype=slide.dtype), atol=1e-12))
        self.assertLessEqual(float(slide[:, 0].abs().max()), 0.04 + 1e-12)
        self.assertGreater(float(slide[:, 0].abs().max()), 0.02)
        # floor: turned about the pivot by <= 8 deg (distance kept up to the shift), shifted <= 30 mm
        f0, f1 = env0[0], b["env_pose"][:, 0]
        q0, q1 = f0[3:], f1[:, 3:]
        dyaw = torch.atan2(q1[:, 2], q1[:, 3]) * 2 - math.atan2(float(q0[2]), float(q0[3])) * 2
        self.assertLessEqual(float(dyaw.abs().max()), math.radians(8.0) + 1e-9)
        self.assertGreater(float(dyaw.abs().max()), math.radians(4.0))
        piv = torch.tensor([-0.046, -0.300], dtype=f1.dtype)
        c, s = torch.cos(dyaw), torch.sin(dyaw)
        d0 = f0[:2] - piv
        rot = torch.stack([piv[0] + c * d0[0] - s * d0[1], piv[1] + s * d0[0] + c * d0[1]], -1)
        sh = f1[:, :2] - rot
        self.assertLessEqual(float(sh.abs().max()), 0.03 + 1e-9)
        self.assertTrue(torch.equal(f1[:, 2], f0[2].expand(64)))
        with self.assertRaises(ValueError):                          # a wall end slide needs the scene's end piece
            RealDR(wall_end=0.04).draw(self.make()[0], 2, torch.Generator().manual_seed(0))
        with self.assertRaises(ValueError):                          # beyond the scene's slide range
            RealDR(wall_end=0.05).draw(sc, 2, torch.Generator().manual_seed(0))


@unittest.skipUnless(GPU and HAVE_SCENE and shutil.which("git"), "needs newton on CUDA, the v2 scene and git")
class BitIdentity(unittest.TestCase):
    """The default options give the base commit's arrays bit for bit (bgdr_ref.py on the v2 scene)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="bgdr_base_"))
        r = subprocess.run(f"git -C {REPO} archive {BASE} src configs contact_nrd | tar -x -C {cls.tmp}", shell=True,
                           capture_output=True)
        vendor = REPO / "contact_nrd/so101_grasp/vendor"
        if vendor.exists() and not (cls.tmp / "contact_nrd/so101_grasp/vendor").exists():
            (cls.tmp / "contact_nrd/so101_grasp/vendor").symlink_to(vendor)
        cls.ok = r.returncode == 0 and (cls.tmp / "src").exists()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_ref(self, src, out):
        env = dict(os.environ, PYTHONPATH=str(src))
        r = subprocess.run([sys.executable, str(HERE / "bgdr_ref.py"), "--scene", str(SCENE_V2), "--calib", str(CALIB),
                            "--out", str(out)], env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        return np.load(out)

    def test_v2_scene_bit_identical(self):
        if not self.ok:
            self.skipTest(f"git archive {BASE} failed")
        a = self.run_ref(self.tmp / "src", self.tmp / "base.npz")
        b = self.run_ref(REPO / "src", self.tmp / "new.npz")
        self.assertEqual(set(a.files), set(b.files))
        bad = [k for k in a.files if not np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f")]
        self.assertEqual(bad, [], f"outputs differ from {BASE}")
        self.assertFalse(any(k.startswith("draw0_bg") for k in b.files))


@unittest.skipUnless(GPU and HAVE_SCENE, "needs newton on CUDA and the v2 scene")
class WithLayers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataclasses import replace
        from nedm.so101_policy.vision import viral as VI
        d = v3_like(json.loads(SCENE_V2.read_text()))
        d["image_fx"] = {}                                            # no nominal blur / chroma: exact pixel checks
        cls.tmp = Path(tempfile.mkdtemp(prefix="bgdr_v3_"))
        p = cls.tmp / "scene.json"
        p.write_text(json.dumps(d))
        cfg = replace(VI.ViralConfig(), cameras=("external", "wrist"), res=224, scene=str(p), calib=str(CALIB))
        cls.views = VI.make_views(cfg, 4, seed=1, device="cuda")
        cls.VI = VI
        sys.path.insert(0, str(HERE))
        import bgdr_ref
        cls.states = [t.cuda() for t in bgdr_ref.states(4)]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_views_have_the_layers(self):
        v = self.views
        self.assertEqual(v.bg_cams, ("external",))
        self.assertEqual(v.r.bg_cameras, ("external",))
        self.assertEqual(tuple(v.bg_prior.shape), (1, 224, 224))
        self.assertGreater(float(v.bg_prior.mean()), 0.05)
        self.assertLessEqual(float(v.bg_prior.max()), 1.0)
        d = v.draw(4)
        self.assertEqual(tuple(d["bg_tex"].shape), (4, 1, 224, 224, 3))
        nd = v.nominal_draw(4)
        self.assertEqual(set(nd), set(d))
        m = v.merge_nominal(d, torch.tensor([True, False, True, False], device="cuda"))
        self.assertEqual(float(m["bg_par"][0].abs().max()), 0.0)
        self.assertTrue(torch.equal(m["bg_tex"][1], d["bg_tex"][1]))

    def test_nominal_and_protected_pixels(self):
        v = self.views
        q, cp, cq, goal = self.states
        nominal = v.render(q, cp, cq, goal)
        zero = v.render(q, cp, cq, goal, dr=v.nominal_draw(4), light=False)
        self.assertTrue(torch.equal(nominal, zero))                   # zero layers = the nominal picture
        d = v.draw(4)
        d["bg_par"][:, :, 0] = 1.0                                     # all worlds randomised
        on = v.render(q, cp, cq, goal, dr=d, light=False)
        off = v.render(q, cp, cq, goal, dr={k: x for k, x in d.items() if not k.startswith("bg_")}, light=False)
        r = v.r
        rgb, seg = r.render_dr(q, cp, cq, goal, draw=d, segment=True)
        lut = torch.full((len(r.scene.body_names) + 64,), -1, dtype=torch.long, device=seg.device)
        for g in ("printed", "servo", "cube", "plate", "board"):
            for b in r.scene.groups.get(g, ()):
                lut[b] = 1
        prot = torch.where(seg >= 0, lut[seg.clamp_min(0)], torch.full_like(seg, -1)) == 1
        ext = prot[:, 0]
        self.assertGreater(int(ext.sum()), 1000)
        self.assertTrue(torch.equal(on[:, 0][ext], off[:, 0][ext]))   # plate, cube, arm, board untouched
        self.assertFalse(torch.equal(on[:, 0], off[:, 0]))
        self.assertTrue(torch.equal(on[:, 1], off[:, 1]))             # the wrist camera has no layers
        self.assertTrue(torch.equal(rgb.to(on.device), on))


if __name__ == "__main__":
    unittest.main()
