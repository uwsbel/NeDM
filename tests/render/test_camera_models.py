"""Camera models of the batch renderer (nedm.render.cameras): calibrated fisheye / radial-tangential lenses, the crop
box, and the OpenCV <-> renderer pose convention.

CPU (numpy, torch, cv2; < 10 s): the pose helper round trip and axes; the lens inverses against cv2
(``cv2.fisheye.undistortPoints`` / ``projectPoints``, ``cv2.projectPoints``); ``opencv`` with D = 0 equals the pinhole
of the same K; ``opencv_fisheye`` with D = 0 is the equidistant lens (NOT a pinhole); the crop box samples like
``cv2.resize``; pixels outside the sensor are invalid. GPU (Newton/Warp on CUDA): a small box at a known 3D point is
drawn where ``cv2.fisheye.projectPoints`` / ``cv2.projectPoints`` put it (within 1 px, full image and a cropped 224 px
image); a PinholeCamera model draws bit-identically to the default renderer; opencv D = 0 draws like the pinhole;
invalid pixels are black with depth 0; camera poses stay per world.

    PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -m unittest tests/render/test_camera_models.py -v
"""
import importlib.util
import math
import unittest

import numpy as np
import torch

HAVE_CV2 = importlib.util.find_spec("cv2") is not None
HAVE_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None
GPU = HAVE_NEWTON and torch.cuda.is_available()

FISHEYE_D = ([0.05, 0.0, 0.0, 0.0], [0.05, 0.01, -0.004, 0.0005], [-0.03, 0.02, -0.01, 0.002], [0.0, 0.0, 0.0, 0.0])
K = [300.0, 300.0, 320.0, 240.0]


def random_poses(n, seed=0):
    """OpenCV-convention poses (n, 7) x y z qw qx qy qz, random rotations."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    return torch.cat([torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5, q], -1)


def matrix_wxyz(q):
    from scipy.spatial.transform import Rotation
    q = np.asarray(q, np.float64)
    return Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], -1)).as_matrix()


class PoseConvention(unittest.TestCase):
    def test_round_trip_and_axes(self):
        """pose_to_opencv(pose_from_opencv(T)) = T (quaternion sign fixed qw >= 0); renderer x = OpenCV x, renderer
        y = -OpenCV y, renderer -z = OpenCV z (the optical axis); positions untouched."""
        from nedm.render.cameras import pose_from_opencv, pose_to_opencv
        T = random_poses(200)
        T[:, 3:] = torch.where(T[:, 3:4] < 0, -T[:, 3:], T[:, 3:])
        gl = pose_from_opencv(T)
        back = pose_to_opencv(gl)
        self.assertLess((back - T).abs().max().item(), 1e-12)
        R_cv = matrix_wxyz(T[:, 3:].numpy())
        R_gl = matrix_wxyz(np.c_[gl[:, 6:].numpy(), gl[:, 3:6].numpy()])
        self.assertLess(np.abs(R_gl - R_cv @ np.diag([1.0, -1.0, -1.0])).max(), 1e-12)
        self.assertTrue(torch.equal(gl[:, :3], T[:, :3]))

    def test_opencv_look_at(self):
        """The placeholder external camera at (0.35, -0.35, 0.45) looking at (0.2, 0, 0): OpenCV z points at the
        target, image down (OpenCV y) has a negative world z component, image right is horizontal."""
        from nedm.render.cameras import opencv_look_at
        T = opencv_look_at([0.35, -0.35, 0.45], [0.2, 0.0, 0.0])[0].numpy()
        R = matrix_wxyz(T[3:])
        fwd = np.array([0.2, 0.0, 0.0]) - T[:3]
        self.assertLess(np.abs(R[:, 2] - fwd / np.linalg.norm(fwd)).max(), 1e-12)
        self.assertLess(R[2, 1], 0)
        self.assertLess(abs(R[2, 0]), 1e-12)


@unittest.skipUnless(HAVE_CV2, "needs cv2")
class LensModels(unittest.TestCase):
    def test_fisheye_inverse_matches_cv2(self):
        """Own Kannala-Brandt inverse = cv2.fisheye.undistortPoints (theta < 88 deg, < 1e-9 in normalised
        coordinates), and cv2.fisheye.projectPoints maps the rays back onto the pixels (< 1e-6 px)."""
        import cv2
        from nedm.render.cameras import CalibratedCamera
        for D in FISHEYE_D:
            c = CalibratedCamera("opencv_fisheye", K, D, 640, 480)
            u, v = c.real_pixels(160, 120)
            d, ok = c.directions(u, v)
            self.assertTrue(ok.all())
            front = (d[..., 2] > math.cos(math.radians(88))).ravel()
            pts = np.stack([u.ravel(), v.ravel()], -1)[front, None]
            xy = cv2.fisheye.undistortPoints(pts, c.matrix, np.array(D), criteria=(3, 200, 1e-15)).reshape(-1, 2)
            mine = (d[..., :2] / d[..., 2:]).reshape(-1, 2)[front]
            self.assertLess(np.abs(mine - xy).max(), 1e-9, D)
            back = c.project(d.reshape(-1, 3)[front])
            self.assertLess(np.abs(back - pts[:, 0]).max(), 1e-6, D)

    def test_opencv_inverse_matches_cv2(self):
        """Radial-tangential inverse: cv2.projectPoints of the rays returns the pixels (< 1e-3 px) wherever a pixel is
        valid; a mild lens is valid everywhere, a strong barrel lens loses its far corners (folded model)."""
        from nedm.render.cameras import CalibratedCamera
        for D, all_valid in (([-0.05, 0.01, 0.001, -0.001, 0.0], True), ([-0.3, 0.1, 0.001, -0.001, -0.02], False),
                             ([0.1, -0.05, 0.0, 0.0, 0.01, 0.05, -0.02, 0.005], None)):
            c = CalibratedCamera("opencv", K, D, 640, 480)
            u, v = c.real_pixels(160, 120)
            d, ok = c.directions(u, v)
            if all_valid is not None:
                self.assertEqual(bool(ok.all()), all_valid, D)
            self.assertGreater(ok.mean(), 0.5, D)
            back = c.project(d.reshape(-1, 3))
            err = np.hypot(back[:, 0] - u.ravel(), back[:, 1] - v.ravel())[ok.ravel()]
            self.assertLess(err.max(), 1e-3, D)

    def test_opencv_with_zero_distortion_equals_the_pinhole(self):
        """(b) ``opencv`` with D = 0 and K = (f, f, (W-1)/2, (H-1)/2), f = (H/2)/tan(fovy/2), has exactly the rays of
        PinholeCamera(fovy) (and of Newton's pinhole kernel, whose float64 copy PinholeCamera.rays is)."""
        from nedm.render.cameras import CalibratedCamera, PinholeCamera
        for W, H, fovy in ((224, 224, 61.0), (640, 480, 75.0), (320, 200, 40.0)):
            f = (H / 2) / math.tan(math.radians(fovy) / 2)
            c = CalibratedCamera("opencv", [f, f, (W - 1) / 2, (H - 1) / 2], [0.0] * 5, W, H)
            dc, okc = c.rays(W, H)
            dp, okp = PinholeCamera(fovy).rays(W, H)
            self.assertTrue(okc.all() and okp.all())
            self.assertLess(np.abs(dc - dp).max(), 1e-12)

    def test_fisheye_with_zero_distortion_is_equidistant_not_pinhole(self):
        """Kannala-Brandt with D = 0 is r = f theta (equidistant), so it is NOT the pinhole of the same K (r = f tan
        theta): equal at the centre, 0.2 % off at 5 deg, many pixels off at 30 deg."""
        from nedm.render.cameras import CalibratedCamera
        f = 300.0
        c = CalibratedCamera("opencv_fisheye", [f, f, 319.5, 239.5], [0.0] * 4, 640, 480)
        u = 319.5 + f * np.radians([0.0, 5.0, 30.0])
        d, ok = c.directions(u, np.full(3, 239.5))
        theta = np.arccos(d[:, 2])
        self.assertLess(np.abs(theta - np.radians([0.0, 5.0, 30.0])).max(), 1e-12)
        pin = 319.5 + f * np.tan(np.radians([0.0, 5.0, 30.0]))
        self.assertGreater(pin[2] - u[2], 15.0)          # px

    def test_crop_box_samples_like_cv2_resize(self):
        """Output pixel (i, j) of crop (x0, y0, w, h) and size (W, H) samples the real image at
        CalibratedCamera.real_pixels: cv2.remap there equals cv2.resize(real[y0:y0+h, x0:x0+w], (W, H)) (bilinear,
        to 1 grey level), and output_pixels inverts real_pixels."""
        import cv2
        from nedm.render.cameras import CalibratedCamera
        rng = np.random.default_rng(0)
        real = cv2.GaussianBlur(rng.uniform(0, 255, (480, 640)).astype(np.float32), (0, 0), 3)
        for crop, (W, H) in (((80, 0, 480, 480), (224, 224)), ((0, 0, 640, 480), (320, 240)),
                             ((100, 40, 400, 300), (200, 150))):
            c = CalibratedCamera("opencv_fisheye", K, [0.0] * 4, 640, 480, crop=crop)
            u, v = c.real_pixels(W, H)
            x0, y0, w, h = crop
            ref = cv2.resize(real[y0:y0 + h, x0:x0 + w], (W, H), interpolation=cv2.INTER_LINEAR)
            got = cv2.remap(real, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)
            self.assertLess(np.abs(got - ref)[2:-2, 2:-2].max(), 1.0, crop)
            j, i = c.output_pixels(u, v, W, H)
            self.assertLess(np.abs(j - np.arange(W)[None]).max(), 1e-9)
            self.assertLess(np.abs(i - np.arange(H)[:, None]).max(), 1e-9)

    def test_pixels_outside_the_sensor_are_invalid(self):
        from nedm.render.cameras import CalibratedCamera
        c = CalibratedCamera("opencv_fisheye", K, FISHEYE_D[1], 640, 480, crop=(-80, 0, 640, 480))
        _, ok = c.rays(64, 48)
        self.assertFalse(ok[:, :8].any())           # real u < -0.5
        self.assertTrue(ok[:, 8:].all())

    def test_contract_entry(self):
        from nedm.render.cameras import CalibratedCamera
        cam = {"model": "opencv_fisheye", "width": 640, "height": 480, "K": K, "D": FISHEYE_D[1], "parent": "world",
               "T_parent_cam": [0, 0, 0, 1, 0, 0, 0], "rms_px": 0.3, "n_points": 10, "holdout_rms_px": 0.4}
        c = CalibratedCamera.from_contract(cam, crop=[80, 0, 480, 480])
        self.assertEqual((c.K, c.D, c.crop), (tuple(K), tuple(FISHEYE_D[1]), (80.0, 0.0, 480.0, 480.0)))
        with self.assertRaises(ValueError):
            CalibratedCamera("kb8", K, FISHEYE_D[1], 640, 480)


def box_scene(half=0.002):
    from nedm.render import MeshBody, Scene
    from nedm.render.so101 import box_mesh
    v, f = box_mesh([half] * 3)
    return Scene.from_meshes([MeshBody("b", vertices=v, faces=f, color=(1.0, 1.0, 1.0))])


@unittest.skipUnless(GPU and HAVE_CV2, "needs newton + warp on CUDA and cv2")
class RenderWithCameraModels(unittest.TestCase):
    N = 16

    def _project_world(self, cam, T, pts):
        """cv2 projection (n, 2) of world points through the camera at OpenCV pose T (camera -> world)."""
        import cv2
        R = matrix_wxyz(T[3:].numpy())
        rvec, _ = cv2.Rodrigues(R.T)
        tvec = -R.T @ T[:3].numpy()
        D = np.array(cam.D)
        if cam.model == "opencv_fisheye":
            uv, _ = cv2.fisheye.projectPoints(pts.numpy().reshape(-1, 1, 3), rvec, tvec, cam.matrix, D)
        else:
            uv, _ = cv2.projectPoints(pts.numpy().reshape(-1, 1, 3), rvec, tvec, cam.matrix, D)
        return uv.reshape(-1, 2)

    def _projection_errors(self, cam, W, H, half, u_range, v_range, dist=(0.3, 0.6), seed=1):
        """Pixel distance between the centroid of a box (half side ``half``) and cv2's projection of its centre, for
        N boxes at real-image points drawn in u_range x v_range at distances ``dist`` (m), in output pixels."""
        from nedm.render import BatchRenderer
        from nedm.render.cameras import opencv_look_at, pose_from_opencv
        from nedm.render.poses import quat_rotate
        T = opencv_look_at([0.35, -0.35, 0.45], [0.2, 0.0, 0.0])[0]
        rng = np.random.default_rng(seed)
        d, ok = cam.directions(rng.uniform(*u_range, self.N), rng.uniform(*v_range, self.N))
        self.assertTrue(ok.all())
        w, x, y, z = T[3:].tolist()
        rq = torch.tensor([x, y, z, w], dtype=torch.float64).expand(self.N, 4)
        pts = T[:3] + quat_rotate(rq, torch.tensor(d * rng.uniform(*dist, self.N)[:, None]))
        r = BatchRenderer(box_scene(half), self.N, W, H, cameras=1, camera_models=[cam], shadows=False)
        body = torch.zeros(self.N, 1, 7)
        body[:, 0, :3], body[:, 0, 6] = pts.float(), 1.0
        depth = r.render(pose_from_opencv(T).float().expand(self.N, 7), body_q=body).depth[:, 0]
        uv = self._project_world(cam, T, pts)
        j, i = cam.output_pixels(uv[:, 0], uv[:, 1], W, H)
        errs = []
        for k in range(self.N):
            ys, xs = np.nonzero(depth[k] > 0)
            self.assertGreater(len(xs), 2)
            errs.append(math.hypot(xs.mean() - j[k], ys.mean() - i[k]))
        return np.array(errs)

    def test_box_lands_on_the_cv2_projection_fisheye_full_image(self):
        """(c) A 6 mm box at 16 known points (0.3-0.6 m, all over the 640 x 480 image), strong fisheye: the centroid
        of its pixels is within 1 px of cv2.fisheye.projectPoints of its centre."""
        from nedm.render.cameras import CalibratedCamera
        cam = CalibratedCamera("opencv_fisheye", K, FISHEYE_D[2], 640, 480)
        errs = self._projection_errors(cam, 640, 480, 0.003, (25, 615), (25, 455))
        self.assertLess(errs.max(), 1.0, errs)

    def test_box_lands_on_the_cv2_projection_cropped_224_and_opencv(self):
        """(c) The same through the crop box [80, 0, 480, 480] resized to 224 x 224 (output pixels, 8 mm box),
        fisheye; and a radial-tangential lens on the full image."""
        from nedm.render.cameras import CalibratedCamera
        cam = CalibratedCamera("opencv_fisheye", K, FISHEYE_D[1], 640, 480, crop=(80, 0, 480, 480))
        errs = self._projection_errors(cam, 224, 224, 0.004, (110, 530), (30, 450), dist=(0.3, 0.5), seed=2)
        self.assertLess(errs.max(), 1.0, errs)
        cam = CalibratedCamera("opencv", K, [-0.05, 0.01, 0.001, -0.001, 0.0], 640, 480)
        errs = self._projection_errors(cam, 640, 480, 0.003, (25, 615), (25, 455), seed=3)
        self.assertLess(errs.max(), 1.0, errs)

    def test_pinhole_model_draws_like_the_default_renderer(self):
        """camera_models=[PinholeCamera(fov)] * C (Newton's own kernel) = the renderer without camera models, bit for
        bit (and hdr=True keeps the same colour image, hdr = colour before the 8-bit clamp); with two different fovs
        each camera matches a one-fov renderer of its own fov."""
        from nedm.render import BatchRenderer, MeshBody, Scene, cameras
        from nedm.render.cameras import PinholeCamera
        from nedm.render.so101 import box_mesh
        v, f = box_mesh([0.05, 0.03, 0.02])
        scene = Scene.from_meshes([MeshBody("b", vertices=v, faces=f, color=(0.8, 0.3, 0.2))]).add_ground()
        n = 4
        body = torch.zeros(n, 1, 7)
        body[:, 0, 0], body[:, 0, 2], body[:, 0, 6] = torch.linspace(-0.1, 0.1, n), 0.05, 1.0
        eye = torch.tensor([[0.6, -0.4, 0.4]]).expand(n, 3)
        cam = cameras.look_at(eye, torch.zeros(n, 3)).float()
        cam2 = torch.stack([cam, cam], 1)
        a = BatchRenderer(scene, n, 96, 64, cameras=2, fov_deg=50.0)
        b = BatchRenderer(scene, n, 96, 64, cameras=2, camera_models=[PinholeCamera(50.0)] * 2)
        fa, fb = a.render(cam2, body_q=body), b.render(cam2, body_q=body)
        self.assertTrue(np.array_equal(fa.rgb, fb.rgb) and np.array_equal(fa.depth, fb.depth))
        h = BatchRenderer(scene, n, 96, 64, cameras=2, fov_deg=50.0, hdr=True).render(cam2, body_q=body)
        lin = (h.hdr_torch().clamp(0, 1) * 255).floor().to(torch.uint8).cpu().numpy()
        self.assertTrue(np.array_equal(h.rgb, fa.rgb) and np.array_equal(lin, fa.rgb))     # hdr = colour before clamp
        mixed = BatchRenderer(scene, n, 96, 64, cameras=2, camera_models=[PinholeCamera(50.0), PinholeCamera(30.0)])
        one30 = BatchRenderer(scene, n, 96, 64, cameras=2, fov_deg=30.0)
        fm = mixed.render(cam2, body_q=body)
        rgb_m, depth_m = fm.rgb.copy(), fm.depth.copy()
        f30 = one30.render(cam2, body_q=body)
        self.assertTrue(np.array_equal(rgb_m[:, 0], fa.rgb[:, 0]) and np.array_equal(rgb_m[:, 1], f30.rgb[:, 1]))
        self.assertTrue(np.array_equal(depth_m[:, 1], f30.depth[:, 1]))

    def test_opencv_zero_distortion_draws_like_the_pinhole(self):
        """(b) on pictures: opencv D = 0 with the pinhole's K vs PinholeCamera: depth equal to 0.1 mm on common hits,
        hit/miss and colour differ on < 0.1 % of the pixels (float32 rays of two formulas)."""
        from nedm.render import BatchRenderer, MeshBody, Scene, cameras
        from nedm.render.cameras import CalibratedCamera, PinholeCamera
        from nedm.render.so101 import box_mesh
        v, f = box_mesh([0.05, 0.03, 0.02])
        scene = Scene.from_meshes([MeshBody("b", vertices=v, faces=f, color=(0.8, 0.3, 0.2))]).add_ground()
        n, W, H, fovy = 4, 160, 120, 55.0
        fpx = (H / 2) / math.tan(math.radians(fovy) / 2)
        cv = CalibratedCamera("opencv", [fpx, fpx, (W - 1) / 2, (H - 1) / 2], [0.0] * 5, W, H)
        body = torch.zeros(n, 1, 7)
        body[:, 0, 1], body[:, 0, 2], body[:, 0, 6] = torch.linspace(-0.1, 0.1, n), 0.05, 1.0
        cam = cameras.look_at(torch.tensor([[0.5, -0.3, 0.35]]).expand(n, 3), torch.zeros(n, 3)).float()[:, None]
        fa = BatchRenderer(scene, n, W, H, camera_models=[PinholeCamera(fovy)]).render(cam, body_q=body)
        rgb_a, d_a = fa.rgb.copy().astype(int), fa.depth.copy()
        fb = BatchRenderer(scene, n, W, H, camera_models=[cv]).render(cam, body_q=body)
        rgb_b, d_b = fb.rgb.astype(int), fb.depth
        both = (d_a > 0) & (d_b > 0)
        self.assertLess(np.abs(d_a - d_b)[both].max(), 1e-4)
        self.assertLess(((d_a > 0) != (d_b > 0)).mean(), 1e-3)
        self.assertLess((np.abs(rgb_a - rgb_b).max(-1) > 2).mean(), 1e-3)

    def test_invalid_pixels_are_black_and_poses_stay_per_world(self):
        """A crop that hangs 80 px off the left of the sensor: those output columns are black, depth 0, in every
        world; each world draws its own camera pose (a world rendered alone gives the same picture)."""
        from nedm.render import BatchRenderer, cameras
        from nedm.render.cameras import CalibratedCamera
        cam = CalibratedCamera("opencv_fisheye", K, FISHEYE_D[1], 640, 480, crop=(-80, 0, 640, 480))
        scene = box_scene(0.05).add_ground()
        n = 3
        eyes = torch.tensor([[0.5, -0.3, 0.4], [0.3, 0.4, 0.5], [-0.4, 0.2, 0.6]])
        poses = cameras.look_at(eyes, torch.zeros(n, 3)).float()[:, None]
        body = torch.tensor([[[0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0]]]).expand(n, 1, 7)
        r = BatchRenderer(scene, n, 64, 48, camera_models=[cam])
        f = r.render(poses, body_q=body)
        rgb, depth = f.rgb.copy(), f.depth.copy()
        self.assertTrue((rgb[:, 0, :, :8] == 0).all() and (depth[:, 0, :, :8] == 0).all())
        self.assertTrue((depth[:, 0, :, 8:] > 0).mean() > 0.5)
        self.assertFalse(bool(r.valid_mask[0, :, :8].any()))
        self.assertFalse(np.array_equal(rgb[0], rgb[1]))
        one = BatchRenderer(scene, 1, 64, 48, camera_models=[cam])
        for k in range(n):
            self.assertTrue(np.array_equal(one.render(poses[k:k + 1], body_q=body[k:k + 1]).rgb[0], rgb[k]))


@unittest.skipUnless(GPU, "needs newton + warp on CUDA")
class Lights(unittest.TestCase):
    def test_dimmed_directional_light(self):
        """A directional light with intensity w (drawn as a far point light) lights a 10 cm plate w times as bright as
        the full light (linear hdr colour, ambient off; within 3 %), for w = 0.25, 0.5, 0.8."""
        from nedm.render import BatchRenderer, MeshBody, Scene, cameras
        from nedm.render.so101 import box_mesh
        v, f = box_mesh([0.05, 0.05, 0.005])
        scene = Scene.from_meshes([MeshBody("plate", vertices=v, faces=f, color=(0.5, 0.5, 0.5))])
        r = BatchRenderer(scene, 1, 64, 64, fov_deg=30.0, hdr=True)
        cam = cameras.look_at(torch.tensor([[0.0, 0.0, 0.5]]), torch.zeros(1, 3), up=(0.0, 1.0, 0.0)).float()[:, None]
        body = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]])
        way = (0.3, -0.2, -1.0)

        def lit(**extra):
            r.backend.set_lights([dict(type="directional", direction=way, shadow=False, **extra)], ambient=False)
            frames = r.render(cam, body_q=body)
            return frames.hdr_torch()[0, 0, ..., 0].cpu().numpy(), frames.depth[0, 0] > 0

        full, hit = lit()
        self.assertGreater(hit.mean(), 0.1)
        expect = 0.5 * 1.0 / np.linalg.norm(way)                     # albedo * n.L
        self.assertLess(abs(full[hit].mean() / expect - 1), 1e-3)
        for w in (0.25, 0.5, 0.8):
            dim, _ = lit(intensity=w, center=(0.0, 0.0, 0.0))
            ratio = dim[hit] / full[hit]
            self.assertLess(np.abs(ratio / w - 1).max(), 0.03, w)


if __name__ == "__main__":
    unittest.main()
