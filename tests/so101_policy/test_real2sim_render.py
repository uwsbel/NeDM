"""Real-setup rendering of the SO101 scene (``nedm.render.so101.RealScene``) and the unchanged default renderer.

- HeadReference (GPU, Newton): the DEFAULT renderer (front + wrist 128 px plate; top + wrist + front 224 px block with
  the table body; build9's ``viral.DRViews`` nominal and with one visual-randomisation draw) gives the pictures and
  depths stored from the HEAD code before the camera-model change, bit for bit
  (``~/so101_r2s_claude/render_ref/ref_head.npz``, written by ``~/so101_r2s_claude/scripts/render_ref.py``; skipped if
  the file is missing; override with NEDM_R2S_RENDER_REF).
- RealSceneStructure (CPU, mujoco for the meshes): body groups, colours, the placeholder calibration's wrist camera =
  the MJCF wrist camera, the external pose, mat corners from the calibration, outward prism faces, bad parents.
- RealSceneOnGpu (Newton): image shapes, the mat plane depth against the calibrated rays, mat dark / floor bright,
  the cube where cv2.fisheye.projectPoints puts it, per-world randomisation and its reset, light randomisation.
- RealSceneFeatures (Newton): supersampling = the pooled 2x render, tone map = sRGB of the linear output, the body
  index per pixel, the near-occluder check, the table plane, render_dr without colours = nominal colours, RealDR on a
  CPU generator with a CUDA scene, intrinsics draws hit the ray cache, the bracket plate part.

    PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest tests/so101_policy/test_real2sim_render.py -v
"""
import copy
import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HAVE_MUJOCO = importlib.util.find_spec("mujoco") is not None
HAVE_CV2 = importlib.util.find_spec("cv2") is not None
GPU = (importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None
       and torch.cuda.is_available())
REF = Path(os.environ.get("NEDM_R2S_RENDER_REF", Path.home() / "so101_r2s_claude/render_ref/ref_head.npz"))
_ASSETS = None


def assets():
    global _ASSETS
    if _ASSETS is None:
        from nedm.render.so101 import SO101Assets
        _ASSETS = SO101Assets.from_mjcf()
    return _ASSETS


def inside(poly, pts):
    """Points (N, 2) strictly inside the convex polygon (k, 2) (any vertex order)."""
    poly = np.asarray(poly, np.float64)
    mid = poly.mean(0)
    poly = poly[np.argsort(np.arctan2(poly[:, 1] - mid[1], poly[:, 0] - mid[0]))]
    e = np.roll(poly, -1, 0) - poly
    rel = pts[:, None, :] - poly[None]
    return ((e[None, :, 0] * rel[..., 1] - e[None, :, 1] * rel[..., 0]) > 0).all(1)


def far_arm(n):
    """Arm turned away from the mat (pan 1.5 rad) and folded: the mat and cube in front are unoccluded."""
    return torch.tensor([[1.5, -1.6, 1.5, 1.2, 0.0, 0.0]], dtype=torch.float64).expand(n, 6)


@unittest.skipUnless(GPU and REF.exists(), f"needs newton on CUDA and the HEAD reference {REF}")
class HeadReference(unittest.TestCase):
    def test_default_renderer_is_bit_identical_to_head(self):
        from nedm.render.so101 import SO101Renderer, poses_from_nrd
        from nedm.so101_policy.vision.viral import DRViews
        ref = np.load(REF)
        x, goal = ref["x"], ref["goal"]
        K = len(x)
        q, cp, cq = poses_from_nrd(torch.tensor(x))
        g = torch.tensor(goal)
        r = SO101Renderer(K, 128, 128, backend="newton")
        rgb, d = r.render(q, cp, cq, g, depth=True)
        self.assertTrue(np.array_equal(rgb.cpu().numpy(), ref["a_rgb"]) and np.array_equal(d.cpu().numpy(),
                                                                                           ref["a_depth"]))
        del r
        r = SO101Renderer(K, 224, 224, backend="newton", cameras=("top", "wrist", "front"), goal="block",
                          table_body=True)
        rgb, d = r.render(q, cp, cq, g, depth=True)
        self.assertTrue(np.array_equal(rgb.cpu().numpy(), ref["b_rgb"]) and np.array_equal(d.cpu().numpy(),
                                                                                           ref["b_depth"]))
        del r
        v = DRViews(K, res=224, seed=0, device="cuda")
        dev = "cuda"
        nominal = v.render(q.to(dev), cp.to(dev), cq.to(dev), g.to(dev)).cpu().numpy()
        self.assertTrue(np.array_equal(nominal, ref["c_nominal"]))
        dr = v.draw(K)
        self.assertTrue(np.array_equal(dr["colors"].cpu().numpy(), ref["c_dr_colors"]))
        out = v.render(q.to(dev), cp.to(dev), cq.to(dev), g.to(dev), dr=dr).cpu().numpy()
        self.assertTrue(np.array_equal(out, ref["c_dr"]))


@unittest.skipUnless(HAVE_MUJOCO, "the arm meshes need mujoco")
class RealSceneStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nedm.render.so101 import RealScene, SO101Scene
        cls.real = RealScene()
        cls.scene = SO101Scene(assets(), real=cls.real)
        cls.default = SO101Scene(assets(), cameras=("front", "wrist"))

    def test_groups_partition_the_bodies(self):
        from nedm.render.so101 import REAL_GROUPS, arm_part
        sc = self.scene
        idx = sorted(i for g in REAL_GROUPS for i in sc.groups[g])
        self.assertEqual(idx, list(range(len(sc.body_names))))
        self.assertEqual([len(sc.groups[g]) for g in ("cube", "plate", "floor", "mat", "walls")], [1, 1, 1, 1, 2])
        mjcf = {m.name: m.color for _, m in assets().meshes}
        for g in ("printed", "servo"):
            self.assertTrue(sc.groups[g])
            for i in sc.groups[g]:
                body = sc.scene.bodies[i]
                self.assertEqual(arm_part(mjcf[body.name]), g)
        for g in REAL_GROUPS:
            for i in sc.groups[g]:
                self.assertEqual(sc.scene.bodies[i].color, self.real.color(g))
        self.assertEqual(sc.scene.statics, [])
        self.assertEqual(sc.cameras, ("external", "wrist"))

    def test_placeholder_wrist_is_the_mjcf_wrist_camera(self):
        """The placeholder calibration's wrist pose is the MJCF wrist_cam turned into the OpenCV convention, so the
        real scene's wrist camera transform equals the default scene's (the OpenCV -> renderer conversion end to end);
        the external transform is pose_from_opencv of the calibration's T_parent_cam."""
        from nedm.render.cameras import pose_from_opencv
        rng = np.random.default_rng(0)
        q = torch.tensor(rng.uniform(-1, 1, (32, 6)))
        cube = torch.tensor([[0.2, 0.0, 0.015]]).expand(32, 3)
        quat = torch.tensor([[1.0, 0, 0, 0]]).expand(32, 4)
        _, real = self.scene.transforms(q, cube, quat)
        _, dflt = self.default.transforms(q, cube, quat)
        self.assertLess((real[:, 1, :3] - dflt[:, 1, :3]).abs().max().item(), 1e-6)
        dq = torch.minimum((real[:, 1, 3:] - dflt[:, 1, 3:]).abs(), (real[:, 1, 3:] + dflt[:, 1, 3:]).abs())
        self.assertLess(dq.max().item(), 1e-6)                                  # same rotation (q or -q)
        ext = pose_from_opencv(self.real.calib["cameras"]["external"]["T_parent_cam"]).float()
        self.assertTrue(torch.allclose(real[:, 0], ext.expand(32, 7), atol=1e-7))

    def test_env_pose_moves_only_the_environment(self):
        sc = self.scene
        n = 3
        q = torch.zeros(n, 6, dtype=torch.float64)
        args = (q, torch.tensor([[0.2, 0.0, 0.015]]).expand(n, 3), torch.tensor([[1.0, 0, 0, 0]]).expand(n, 4),
                torch.tensor([[0.25, 0.05, 0.3]]).expand(n, 3))
        a, _ = sc.transforms(*args)
        env = sc.env_pose.expand(n, sc.n_env, 7).clone()
        env[:, sc.env_groups.index("mat"), 0] += 0.02
        b, _ = sc.transforms(*args, env_pose=env)
        mat = sc.groups["mat"][0]
        self.assertAlmostEqual((b[:, mat, 0] - a[:, mat, 0]).mean().item(), 0.02, places=6)
        others = [i for i in range(len(sc.body_names)) if i != mat]
        self.assertTrue(torch.equal(a[:, others], b[:, others]))
        with self.assertRaises(ValueError):
            self.default.transforms(*args, env_pose=env)

    def test_mat_corners_from_calibration_and_parameter(self):
        from nedm.render.so101 import RealScene
        calib = RealScene().calib
        calib["scene"]["mat"]["corners_xy"] = [[0.1, -0.2], [0.4, -0.2], [0.4, 0.25], [0.1, 0.25]]
        self.assertTrue(np.allclose(RealScene(calib=calib).mat_corners(), calib["scene"]["mat"]["corners_xy"]))
        own = [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]]
        self.assertTrue(np.allclose(RealScene(calib=calib, mat_corners_xy=own).mat_corners(), own))
        self.assertTrue(np.allclose(RealScene().mat_corners(), RealScene.DEFAULT_MAT))

    def test_prism_faces_point_outwards(self):
        from nedm.render.so101 import prism_mesh
        for corners in ([[0, 0], [1, 0], [1, 1], [0, 1]], [[0.3, 0.1], [0.0, 0.0], [0.2, 0.4], [0.5, 0.35]],
                        [[0.0, 0.0], [0.6, 0.0], [0.6, 0.005], [0.0, 0.005]]):
            v, f = prism_mesh(corners, -0.01, 0.2)
            tri = v[f].astype(np.float64)
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            out = tri.mean(1) - v.mean(0)
            self.assertTrue((np.einsum("ij,ij->i", n, out) > 0).all(), corners)

    def test_calibration_from_a_file_and_bad_parents(self):
        from nedm.render.so101 import RealScene, SO101Scene, placeholder_calibration
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "calib.json"
            p.write_text(json.dumps(placeholder_calibration()))
            real = RealScene.from_path(p)
        self.assertEqual(real.camera_model("external").model, "opencv_fisheye")
        bad = copy.deepcopy(real.calib)
        bad["cameras"]["wrist"]["parent"] = "no_such_body"
        with self.assertRaises(ValueError):
            SO101Scene(assets(), real=RealScene(calib=bad))
        bad = copy.deepcopy(real.calib)
        bad["cameras"]["external"]["parent"] = "gripper"
        with self.assertRaises(ValueError):
            SO101Scene(assets(), real=RealScene(calib=bad))
        with self.assertRaises(ValueError):
            SO101Scene(assets(), real=real, cameras=("external", "side"))


@unittest.skipUnless(GPU and HAVE_MUJOCO and HAVE_CV2, "needs newton on CUDA, mujoco and cv2")
class RealSceneOnGpu(unittest.TestCase):
    K, RES = 6, 224

    @classmethod
    def setUpClass(cls):
        from nedm.render.so101 import RealScene, SO101Renderer
        cls.real = RealScene()
        cls.r = SO101Renderer(cls.K, cls.RES, cls.RES, backend="newton", assets=assets(), real=cls.real)

    def _args(self, n=None):
        n = n or self.K
        cube = torch.tensor([[0.25, -0.10, 0.015]]).expand(n, 3)
        quat, goal = torch.tensor([[1.0, 0, 0, 0]]).expand(n, 4), torch.tensor([[0.3, 0.1, 0.4]]).expand(n, 3)
        return far_arm(n), cube, quat, goal

    def test_shapes_mat_depth_and_colours(self):
        """224 x 224 from both cameras; on the external camera the mat's pixels have the z-depth of the plane z = 0
        along the calibrated rays (median < 0.05 mm on > 90 % of the pixels whose plane point is 1 cm inside the
        mat, away from the cube), the mat is dark and the floor bright."""
        from nedm.render.cameras import pose_from_opencv
        from nedm.render.poses import quat_rotate
        rgb, depth = self.r.render(*self._args(), depth=True)
        self.assertEqual(tuple(rgb.shape), (self.K, 2, self.RES, self.RES, 3))
        self.assertEqual(rgb.dtype, torch.uint8)
        rgb, depth = rgb.cpu().numpy().astype(int), depth.cpu().numpy()
        cam = self.real.camera_model("external")
        dirs, ok = cam.rays(self.RES, self.RES)                                  # renderer camera frame
        pose = pose_from_opencv(self.real.calib["cameras"]["external"]["T_parent_cam"])
        world = quat_rotate(pose[3:].expand(self.RES * self.RES, 4), torch.tensor(dirs.reshape(-1, 3))).numpy()
        t = -pose[2].item() / world[:, 2]
        hit = pose[:3].numpy() + t[:, None] * world
        z = (t * -dirs.reshape(-1, 3)[:, 2]).reshape(self.RES, self.RES)
        corners = self.real.mat_corners()
        mid = corners.mean(0)
        inner = inside(mid + (corners - mid) * (1 - 0.02 / np.abs(corners - mid).max(0)), hit[:, :2])
        far_cube = np.hypot(hit[:, 0] - 0.25, hit[:, 1] + 0.10) > 0.05
        sel = (inner & far_cube & (t > 0)).reshape(self.RES, self.RES)
        self.assertGreater(sel.sum(), 1000)
        err = np.abs(depth[0, 0] - z)[sel]
        self.assertGreater((err < 0.002).mean(), 0.9)
        self.assertLess(np.median(err) * 1e3, 0.05)
        self.assertLess(rgb[0, 0][sel].mean(), 40)
        floor = ~inside(corners, hit[:, :2]).reshape(self.RES, self.RES)
        x0, x1, y0, y1 = self.real.floor_xy
        on_floor = floor & ((hit[:, 0] > x0 + 0.03) & (hit[:, 0] < x1 - 0.05) & (hit[:, 1] > y0 + 0.03)
                            & (hit[:, 1] < y1 - 0.05) & (t > 0)).reshape(self.RES, self.RES)
        good = on_floor & (np.abs(depth[0, 0] - z) < 0.003)
        self.assertGreater(good.sum(), 500)
        self.assertGreater(rgb[0, 0][good].mean(), 150)

    def test_cube_where_cv2_projects_it(self):
        """The centre of the cube's top face, projected with cv2.fisheye.projectPoints into the external image (crop
        mapping to output pixels), lands on green pixels."""
        import cv2
        from scipy.spatial.transform import Rotation
        rgb = self.r.render(*self._args()).cpu().numpy().astype(int)
        cam = self.real.camera_model("external")
        T = np.asarray(self.real.calib["cameras"]["external"]["T_parent_cam"], np.float64)
        R = Rotation.from_quat(np.r_[T[4:7], T[3]]).as_matrix()
        rvec, _ = cv2.Rodrigues(R.T)
        uv, _ = cv2.fisheye.projectPoints(np.array([[[0.25, -0.10, 0.03]]]), rvec, -R.T @ T[:3], cam.matrix,
                                          np.array(cam.D))
        j, i = cam.output_pixels(uv[0, 0, 0], uv[0, 0, 1], self.RES, self.RES)
        px = rgb[0, 0, int(round(i)), int(round(j))]
        self.assertTrue(px[1] > px[0] + 30 and px[1] > px[2] + 30, px)

    def test_randomisation_per_world_and_reset(self):
        """RealDR draws: every world's picture changes, the mat moves with env_pose (depth changes on the external
        camera), render_dr(None) restores the nominal pictures bit for bit; jittered lights change the pictures and the
        scene's own lights restore them."""
        from nedm.render.so101 import RealDR
        args = self._args()
        a = self.r.render_dr(*args).clone()
        g = torch.Generator(device="cuda").manual_seed(0)
        draw = RealDR().draw(self.r.scene, self.K, g, "cuda")
        self.assertEqual(tuple(draw["colors"].shape), (self.K, len(self.r.scene.body_names), 3))
        self.assertEqual(tuple(draw["cam_offset"].shape), (self.K, 2, 6))
        b, db = self.r.render_dr(*args, draw=draw, depth=True)
        self.assertTrue(bool((a != b).flatten(1).any(1).all()))
        only_env = {"env_pose": draw["env_pose"]}
        c, dc = (t.clone() for t in self.r.render_dr(*args, draw=only_env, depth=True))
        _, d0 = self.r.render(*args, depth=True)
        self.assertTrue(bool((dc[:, 0] != d0[:, 0]).flatten(1).any(1).all()))
        self.assertTrue(torch.equal(self.r.render_dr(*args), a))
        lights = RealDR(light_deg=20).lights(self.real.lights, torch.Generator().manual_seed(1))
        self.r.set_lights(lights)
        e = self.r.render(*args).clone()
        self.assertFalse(torch.equal(e, a))
        self.r.set_lights(self.real.lights)
        self.assertTrue(torch.equal(self.r.render(*args), a))

    def test_new_camera_models_update_rays_and_depth_factors(self):
        """set_camera_models: a crop hanging off the sensor blacks out those columns (valid mask, depth 0); the
        calibrated models restore the original pictures."""
        from nedm.render.cameras import CalibratedCamera
        args = self._args()
        a = self.r.render(*args).clone()
        base = self.real.camera_model("external")
        off = CalibratedCamera(base.model, base.K, base.D, base.width, base.height, crop=(-160, 0, 640, 480))
        self.r.set_camera_models([off, self.real.camera_model("wrist")])
        try:
            b, d = self.r.render(*args, depth=True)
            cols = int(self.RES * 160 / 640) - 1
            self.assertFalse(bool(self.r.valid_mask[0, :, :cols].any()))
            self.assertTrue(bool((b[:, 0, :, :cols] == 0).all()) and bool((d[:, 0, :, :cols] == 0).all()))
            self.assertTrue(torch.equal(b[:, 1], a[:, 1]))
        finally:
            self.r.set_camera_models(self.r.scene.camera_models)
        self.assertTrue(torch.equal(self.r.render(*args), a))


@unittest.skipUnless(GPU and HAVE_MUJOCO and HAVE_CV2, "needs newton on CUDA, mujoco and cv2")
class RealSceneFeatures(unittest.TestCase):
    K = 3

    def _args(self, n=None):
        n = n or self.K
        cube = torch.tensor([[0.25, -0.10, 0.015]]).expand(n, 3)
        quat, goal = torch.tensor([[1.0, 0, 0, 0]]).expand(n, 4), torch.tensor([[0.3, 0.1, 0.4]]).expand(n, 3)
        q = torch.tensor([[0.2, -0.3, 0.5, 1.2, 0.3, 0.4]], dtype=torch.float64).expand(n, 6)
        return q, cube, quat, goal

    def _renderer(self, res, **kw):
        from nedm.render.so101 import RealScene, SO101Renderer
        real = kw.pop("real", None) or RealScene(room={"center": (0.3, 0, 0.6), "half": (2.0, 2.0, 1.5),
                                                       "floor_z": -0.75})
        return SO101Renderer(self.K, res, res, backend="newton", assets=assets(), real=real, **kw)

    def test_supersample_is_the_pooled_double_resolution(self):
        from dataclasses import replace
        from nedm.render.so101 import RealScene
        real = RealScene()
        r2 = self._renderer(32, real=replace(real, supersample=2))
        a, da = r2.render(*self._args(), depth=True)
        del r2
        r1 = self._renderer(64, real=real)
        b, db = r1.render(*self._args(), depth=True)
        pooled = (b.float().reshape(self.K, 2, 32, 2, 32, 2, 3).mean((3, 5)) + 0.5).clamp(0, 255).to(torch.uint8)
        self.assertTrue(torch.equal(a, pooled))
        hit = (db > 0).float().reshape(self.K, 2, 32, 2, 32, 2)
        dmean = (db.reshape(self.K, 2, 32, 2, 32, 2) * hit).sum((3, 5)) / hit.sum((3, 5)).clamp_min(1)
        self.assertLess((da - dmean).abs().max().item(), 1e-5)

    def test_tone_map_and_hdr_and_segments(self):
        """With an identity tone map the colour is sRGB(clamp(linear)); hdr returns that linear colour (sky filled);
        the body index per pixel puts the mat where the mat colour is."""
        from dataclasses import replace
        from nedm.render.camera_fx import srgb_oetf, to_uint8
        from nedm.render.so101 import RealScene
        ident = {"exposure": 1.0, "wb": [1, 1, 1], "black": [0, 0, 0], "gamma": 1.0, "vignette": 0.0}
        real = replace(RealScene(), tone={"external": ident, "wrist": ident}, sky=(0, 0, 0))
        r = self._renderer(48, real=real, segment=True)
        rgb, seg, hdr = r.render(*self._args(), segment=True, hdr=True)
        self.assertTrue(torch.equal(rgb, to_uint8(srgb_oetf(hdr.clamp(0, 1)))))
        mat = r.scene.groups["mat"][0]
        self.assertGreater(int((seg == mat).sum()), 100)
        floor = r.scene.groups["floor"][0]
        lum = rgb.float().mean(-1)
        self.assertLess(lum[seg == mat].mean().item() + 50, lum[seg == floor].mean().item())
        self.assertTrue(bool(((seg >= -1) & (seg < len(r.scene.body_names))).all()))

    def test_nominal_camera_effects_applied_once(self):
        """A tone-mapped real scene with ``image_fx`` nominal blur / chroma returns the renderer's picture with
        ``ImageFX.nominal`` applied (before the 8-bit rounding), per camera."""
        from dataclasses import replace
        from nedm.render.camera_fx import ImageFX, srgb_oetf, to_uint8
        from nedm.render.so101 import RealScene
        ident = {"exposure": 1.0, "wb": [1, 1, 1], "black": [0, 0, 0], "gamma": 1.0, "vignette": 0.0}
        fxd = {"nominal_blur_px": 0.6, "nominal_chroma_px": {"external": 2.0}}
        base = replace(RealScene(), tone={"external": ident, "wrist": ident}, sky=(0, 0, 0))
        r = self._renderer(40, real=replace(base, image_fx=fxd), hdr=True)
        rgb, hdr = r.render(*self._args(), hdr=True)
        y = srgb_oetf(hdr.clamp(0, 1))
        fx = ImageFX.from_dict(fxd)
        want = torch.stack([fx.nominal(y[:, c], cam) for c, cam in enumerate(r.cameras)], 1)
        self.assertLessEqual((rgb.int() - to_uint8(want).int()).abs().max().item(), 1)
        self.assertFalse(torch.equal(rgb, to_uint8(y)))

    def test_near_occluder_check(self):
        """The MJCF camera board 9.5 mm behind the MJCF wrist camera: with the mount NOT hidden and the wrist camera
        moved 12 mm back, most wrist pixels are closer than 12 mm and the renderer refuses to build."""
        import copy
        from nedm.render.cameras import pose_from_opencv, pose_to_opencv
        from nedm.render.so101 import RealScene, qrot, wxyz
        real = RealScene(hide_bodies=(), board=None)
        cal = copy.deepcopy(real.calib)
        gl = pose_from_opencv(cal["cameras"]["wrist"]["T_parent_cam"])
        back = qrot(wxyz(gl[3:]), torch.tensor([0.0, 0.0, 0.012], dtype=torch.float64))   # renderer +z = backwards
        gl[:3] += back
        cal["cameras"]["wrist"]["T_parent_cam"] = pose_to_opencv(gl).tolist()
        with self.assertRaises(ValueError):
            self._renderer(32, real=RealScene(calib=cal, hide_bodies=(), board=None))
        r = self._renderer(32, real=RealScene(calib=cal))                  # mount hidden: fine
        self.assertLess(max(r.check_near().values()), 0.2)

    def test_table_plane_and_plate(self):
        """The floor, mat and goal plate follow the table plane (z = c0 + c1 x + c2 y): a lower plane moves the
        external camera's mat depth by about the plane offset along the view."""
        from dataclasses import replace
        from nedm.render.so101 import RealScene
        real = RealScene()
        r0 = self._renderer(48, real=real, segment=True)
        _, d0, s0 = r0.render(*self._args(), depth=True, segment=True)
        mat = r0.scene.groups["mat"][0]
        del r0
        r1 = self._renderer(48, real=replace(real, table_plane=(-0.02, 0.0, 0.0)), segment=True)
        _, d1, s1 = r1.render(*self._args(), depth=True, segment=True)
        both = (s0[0, 0] == mat) & (s1[0, 0] == mat)
        self.assertGreater(int(both.sum()), 50)
        self.assertGreater((d1[0, 0] - d0[0, 0])[both].median().item(), 0.005)
        g = r1.scene.goal_transform(torch.tensor([[0.3, 0.1, 0.4]]), 1)
        body_q, _ = r1.scene.transforms(*self._args(1))
        self.assertAlmostEqual(body_q[0, r1.scene.groups["plate"][0], 2].item(), -0.02, places=5)
        self.assertAlmostEqual(g[0, 2].item(), 0.0, places=9)

    def test_render_dr_reset_cpu_generator_and_ray_cache(self):
        from nedm.render.so101 import RealDR
        r = self._renderer(32)
        args = self._args()
        a = r.render_dr(*args).clone()
        dr = RealDR(focal=0.03, pp_px=2.0, focal_levels=3)
        draw = dr.draw(r.scene, self.K, torch.Generator().manual_seed(0))       # CPU generator, CUDA scene
        self.assertEqual(draw["colors"].device.type, "cuda")
        b = r.render_dr(*args, draw=draw).clone()
        self.assertFalse(torch.equal(a, b))
        c = r.render_dr(*args, draw={"cam_offset": torch.zeros(self.K, 2, 6)})   # no colours: nominal colours
        self.assertTrue(torch.equal(c, a))
        g = torch.Generator().manual_seed(3)
        cache = r.renderer.backend._ray_cache
        for _ in range(20):
            r.set_camera_models(dr.camera_models(r.scene, g, out_res=32))
        self.assertLessEqual(len(cache), r.renderer.backend.RAY_CACHE)
        models = dr.camera_models(r.scene, g, out_res=32)
        r.set_camera_models(models)
        n, rays = len(cache), r.renderer.backend._rays
        r.set_camera_models(models)                                      # the same draw again: a cache hit
        self.assertEqual(len(cache), n)
        self.assertIs(r.renderer.backend._rays, rays)
        r.set_camera_models(r.scene.camera_models)
        self.assertTrue(torch.equal(r.render_dr(*args), a))

    def test_parts_prism(self):
        """A prism part on the camera mount is drawn (group board) and its faces point outwards."""
        from nedm.render.so101 import RealScene, SO101Scene, polygon_prism
        poly = [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.02, 0.02, 0.001], [0.0, 0.02, 0.001]]
        v, f = polygon_prism(poly, [0.0, 0.0, 0.003])
        tri = v[f].astype(np.float64)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.assertTrue((np.einsum("ij,ij->i", n, tri.mean(1) - v.mean(0)) > 0).all())
        sc = SO101Scene(assets(), real=RealScene(parts=({"name": "p", "body": "camera_mount", "group": "board",
                                                        "polygon": poly, "offset": [0, 0, 0.003]},)))
        self.assertEqual(len(sc.groups["board"]), 2)                    # the camera board + the part
        mount = {m.name for body, m in assets().meshes if body == "camera_mount"}
        self.assertTrue(mount and not mount & set(sc.body_names))       # the MJCF camera mount is left out

if __name__ == "__main__":
    unittest.main()
