"""Tone map, image-space camera effects and the shared student-view crop (CPU, no renderer).

    PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest tests/render/test_camera_fx.py -v
"""
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

HAVE_CV2 = importlib.util.find_spec("cv2") is not None
ROOT = Path(__file__).resolve().parents[2]


class ToneMap(unittest.TestCase):
    def test_identity_is_srgb_and_inverse(self):
        from nedm.render.camera_fx import srgb_eotf, srgb_oetf, tone_map, tone_vector
        x = torch.linspace(0, 1, 1001)
        self.assertLess((srgb_eotf(srgb_oetf(x)) - x).abs().max().item(), 1e-5)
        hdr = torch.rand(2, 2, 4, 5, 3) * 1.5
        p = torch.tensor([tone_vector(None)] * 2)
        y = tone_map(hdr, p)
        self.assertTrue(torch.allclose(y, srgb_oetf(hdr.clamp(0, 1)), atol=1e-6))

    def test_parameters_act_as_documented(self):
        from nedm.render.camera_fx import srgb_oetf, tone_map, tone_vector
        hdr = torch.full((1, 1, 2, 2, 3), 0.2)
        p = torch.tensor([tone_vector({"exposure": 2.0, "wb": (1.0, 0.5, 1.0), "black": (0.01, 0.0, 0.0),
                                       "gamma": 1.2})])
        y = tone_map(hdr, p)[0, 0, 0, 0]
        want = srgb_oetf(torch.tensor([0.41, 0.2, 0.4])) ** 1.2
        self.assertTrue(torch.allclose(y, want, atol=1e-6))
        cos = torch.full((1, 2, 2), 0.5)
        pv = torch.tensor([tone_vector({"vignette": 2.0})])
        yv = tone_map(hdr, pv, cos)[0, 0, 0, 0]
        self.assertTrue(torch.allclose(yv, srgb_oetf(torch.full((3,), 0.05)), atol=1e-6))
        # per-world parameters
        pw = torch.stack([torch.tensor([tone_vector(None)]), torch.tensor([tone_vector({"exposure": 0.5})])])
        yw = tone_map(hdr.expand(2, -1, -1, -1, -1), pw)
        self.assertFalse(torch.allclose(yw[0], yw[1]))


class ImageEffects(unittest.TestCase):
    def test_blur_noise_motion_keep_mean_and_shape(self):
        from nedm.render.camera_fx import gaussian_blur, gaussian_noise, motion_blur
        g = torch.Generator().manual_seed(0)
        img = torch.rand(3, 32, 40, 3, generator=g)
        for out in (gaussian_blur(img, torch.tensor([0.5, 1.0, 2.0])),
                    motion_blur(img, torch.tensor([1.0, 3.0, 5.0]), torch.tensor([0.0, 0.7, 2.0]))):
            self.assertEqual(out.shape, img.shape)
            self.assertLess((out.mean() - img.mean()).abs().item(), 0.01)
            self.assertLess(out.std().item(), img.std().item())
        flat = torch.full((2, 64, 64, 3), 0.5)
        n = gaussian_noise(flat, torch.tensor([0.0, 0.02]), g)
        self.assertEqual(n[0].std().item(), 0.0)
        self.assertAlmostEqual(n[1].std().item(), 0.02, delta=0.002)

    @unittest.skipUnless(HAVE_CV2, "needs cv2")
    def test_jpeg_close_to_libjpeg(self):
        """The own DCT JPEG round trip is as lossy as libjpeg (cv2) at the same quality: PSNR within 2 dB, and the
        two decoded images are closer to each other than to the input."""
        import cv2
        from nedm.render.camera_fx import jpeg
        rng = np.random.default_rng(0)
        base = cv2.resize(rng.uniform(0, 255, (14, 14, 3)).astype(np.uint8), (112, 112),
                          interpolation=cv2.INTER_CUBIC)
        img = np.clip(base.astype(np.float32) + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
        psnr = lambda a, b: 10 * math.log10(255 ** 2 / np.mean((a.astype(float) - b.astype(float)) ** 2))  # noqa
        for q in (50, 75, 95):
            ok, enc = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
            ref = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
            own = (jpeg(torch.from_numpy(img)[None].float() / 255, torch.tensor([float(q)]))[0].numpy() * 255
                   ).round().clip(0, 255).astype(np.uint8)
            self.assertLess(abs(psnr(img, own) - psnr(img, ref)), 2.0, q)
            self.assertGreater(psnr(own, ref), psnr(img, ref) - 1.0, q)

    def test_image_fx_draw_apply(self):
        from nedm.render.camera_fx import ImageFX
        fx = ImageFX.from_dict({"p_noise": 1.0, "noise": [0.01, 0.02], "p_jpeg": 1.0, "jpeg_quality": [60, 90],
                                "p_blur": 0.5, "p_motion": 0.5, "nominal_blur_px": 0.3})
        g = torch.Generator().manual_seed(1)
        d = fx.draw(8, g)
        self.assertTrue(bool((d["noise"] >= 0.01).all() and (d["jpeg"] >= 60).all()))
        img = (torch.rand(8, 24, 24, 3, generator=g) * 255).to(torch.uint8)
        out = fx.apply(img, d, g)
        self.assertEqual((out.dtype, out.shape), (torch.uint8, img.shape))
        self.assertFalse(torch.equal(out, img))
        self.assertEqual(fx.apply(img.float() / 255).dtype, torch.float32)


class StudentView(unittest.TestCase):
    def test_parse_and_default(self):
        from nedm.render import student_view as sv
        v = sv.view_specs({"student_view": {"external": {"crop": [52, 103, 288, 288], "out": [224, 224]},
                                            "wrist": {"crop_xyxy": [0, 0, 640, 480]}}},
                          ("external", "wrist", "other"), 128)
        self.assertEqual(v["external"].crop_xywh, (52, 103, 288, 288))
        self.assertEqual(v["external"].out_wh, (224, 224))
        self.assertEqual(v["wrist"].crop_xywh, (0, 0, 640, 480))
        self.assertEqual(v["wrist"].out_wh, (128, 128))
        self.assertEqual(v["other"].crop_xywh, (80, 0, 480, 480))
        with self.assertRaises(ValueError):
            sv.parse_view({"out": [1, 1]})

    def test_subpixel_centres_average_to_the_pixel_centre(self):
        from nedm.render import student_view as sv
        spec = sv.ViewSpec((52, 103, 288, 288), (224, 224))
        u1, v1 = sv.output_pixel_centres(spec, 1)
        u2, v2 = sv.output_pixel_centres(spec, 2)
        pool = lambda a: a.reshape(224, 2, 224, 2).mean((1, 3))      # noqa: E731
        self.assertLess(np.abs(pool(u2) - u1).max(), 1e-9)
        self.assertLess(np.abs(pool(v2) - v1).max(), 1e-9)
        from nedm.render.cameras import CalibratedCamera
        cam = CalibratedCamera("opencv_fisheye", (270, 262, 327, 242), (0.06, -0.07, 0.02, 0), 640, 480,
                               crop=spec.crop_xywh)
        cu, cv_ = cam.real_pixels(224, 224)
        self.assertLess(np.abs(cu - u1).max(), 1e-9)
        self.assertLess(np.abs(cv_ - v1).max(), 1e-9)

    @unittest.skipUnless(HAVE_CV2, "needs cv2")
    def test_hw_lib_uses_the_same_function(self):
        """The robot code (hw_lib, loaded without the nedm.render package) and the renderer share ONE crop: same
        ViewSpec class source file, same boxes from the same dict, same crop + resize."""
        sys.path.insert(0, str(ROOT / "scripts/so101_policy/real2sim/hw"))
        try:
            import hw_lib as H
        finally:
            sys.path.pop(0)
        from nedm.render import student_view as sv
        self.assertEqual(Path(H._SV.__file__).resolve(), Path(sv.__file__).resolve())
        d = {"student_view": {"external": {"crop": [52, 103, 288, 288], "out": [224, 224]},
                              "wrist": {"crop": [0, 0, 640, 480], "out": [224, 224]}}}
        a, b = H.view_specs(d, ("external", "wrist"), 224), sv.view_specs(d, ("external", "wrist"), 224)
        for cam in a:
            self.assertEqual((a[cam].crop_xywh, a[cam].out_wh), (b[cam].crop_xywh, b[cam].out_wh))
        img = (np.random.default_rng(0).uniform(0, 255, (480, 640, 3))).astype(np.uint8)
        self.assertTrue(np.array_equal(H.preprocess(img, a["external"], True), sv.crop_resize(img, b["external"],
                                                                                             True)))

    def test_real_scene_reads_the_same_view(self):
        """RealScene with crop=None uses the student view (scene file > calibration > centred square)."""
        from nedm.render.so101 import RealScene, placeholder_calibration
        cal = placeholder_calibration()
        self.assertEqual(RealScene(calib=cal).camera_model("external").crop, (80.0, 0.0, 480.0, 480.0))
        cal["student_view"] = {"external": {"crop": [10, 20, 300, 300], "out": [224, 224]}}
        self.assertEqual(RealScene(calib=cal).camera_model("external").crop, (10.0, 20.0, 300.0, 300.0))
        rs = RealScene(calib=cal, student_view={"external": {"crop": [0, 0, 640, 480], "out": [224, 224]}})
        self.assertEqual(rs.camera_model("external").crop, (0.0, 0.0, 640.0, 480.0))
        self.assertEqual(RealScene(calib=cal, crop={"external": None}).camera_model("external").crop,
                         (0.0, 0.0, 640.0, 480.0))
        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "calib.json"
            cp.write_text(json.dumps(cal))
            sp = Path(td) / "scene.json"
            sp.write_text(json.dumps({"version": "real_scene_v1", "calib": str(cp),
                                      "student_view": {"external": {"crop": [1, 2, 400, 400], "out": [224, 224]}},
                                      "geometry": {"table_plane": [0.0, 0.0, 0.0], "plate_thickness": 0.002},
                                      "colors": {"mat": [0.02, 0.02, 0.02]}, "render": {"supersample": 2},
                                      "dr": {"focal": 0.03}}))
            rs = RealScene.from_path(sp)
        self.assertEqual(rs.camera_model("external").crop, (1.0, 2.0, 400.0, 400.0))
        self.assertEqual((rs.supersample, rs.plate_thickness, rs.color("mat")), (2, 0.002, (0.02, 0.02, 0.02)))
        self.assertEqual(rs.dr, {"focal": 0.03})


class LookAt(unittest.TestCase):
    def test_degenerate_view_gives_a_unit_quaternion(self):
        from nedm.render.cameras import opencv_look_at
        q = opencv_look_at([[0.2, 0, 0.6], [0.35, -0.35, 0.45], [0, 0, 0]], [[0.2, 0, 0], [0.2, 0, 0], [0, 0, 1]])
        self.assertLess((q[:, 3:].norm(dim=-1) - 1).abs().max().item(), 1e-12)
        with self.assertRaises(ValueError):
            opencv_look_at([0, 0, 1], [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
