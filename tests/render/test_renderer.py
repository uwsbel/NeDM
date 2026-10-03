"""Tests of nedm.render.BatchRenderer. Skipped when newton and warp are not installed.

Every expected value is geometry worked out by hand (a box at a known height under a camera
at a known height), so a pass means the poses reached the right world and the camera
convention is the one documented, not merely that an image came out.

    PYTHONPATH=src python -m unittest discover -s tests/render -p "test_renderer.py" -v

The first run compiles the ray-tracing kernels, about a minute on CPU.
"""
import importlib.util
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import torch

HAVE_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

BOX_V = np.array([[x, y, z] for x in (-0.25, 0.25) for y in (-0.25, 0.25) for z in (-0.25, 0.25)], dtype=np.float32)
BOX_F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32)
RES = 16
CAMERA_Z = 4.0
FOV_DEG = 45.0
# Depth is distance ALONG THE RAY. With an even resolution no pixel sits on the optical axis:
# pixel (RES/2, RES/2) is half a pixel off it in both directions, so its ray is this much longer
# than the perpendicular distance.
OFF_AXIS = float(np.sqrt(1.0 + 2.0 * (np.tan(np.radians(FOV_DEG) / 2) / RES) ** 2))

URDF = textwrap.dedent("""\
    <robot name="pendulum">
      <link name="base"><visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual></link>
      <link name="arm"><visual><origin xyz="0.5 0 0"/><geometry><box size="1.0 0.1 0.1"/></geometry></visual></link>
      <link name="tip"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
      <joint name="shoulder" type="revolute">
        <parent link="base"/><child link="arm"/><origin xyz="0 0 1"/><axis xyz="0 1 0"/>
        <limit lower="-3.2" upper="3.2" effort="1" velocity="1"/>
      </joint>
      <joint name="wrist" type="fixed"><parent link="arm"/><child link="tip"/><origin xyz="1 0 0"/></joint>
    </robot>
    """)


def identity_bodies(positions: np.ndarray) -> np.ndarray:
    """(N, 3) positions -> (N, 1, 7) transforms with no rotation."""
    q = np.zeros((positions.shape[0], 1, 7), dtype=np.float32)
    q[:, 0, :3] = positions
    q[:, 0, 6] = 1.0
    return q


@unittest.skipUnless(HAVE_NEWTON, "newton and warp are not installed (see requirements-render.txt)")
class BatchRendererTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import warp as wp
        from nedm.render import BatchRenderer, MeshBody, Scene, cameras
        wp.config.quiet = True
        cls.cameras = cameras
        scene = Scene.from_meshes([MeshBody("box", vertices=BOX_V, faces=BOX_F, color=(0.9, 0.1, 0.1))])
        cls.renderer = BatchRenderer(scene, num_worlds=3, width=RES, height=RES, fov_deg=FOV_DEG, shadows=False)
        # straight down, so the up hint must not be the world Z the view direction lies along
        cls.down = cameras.look_at(torch.tensor([[0.0, 0.0, CAMERA_Z]]).repeat(3, 1), torch.zeros(3, 3),
                                   up=(1.0, 0.0, 0.0))

    def center_depth(self, frames) -> np.ndarray:
        return frames.depth[:, 0, RES // 2, RES // 2]

    def test_each_world_gets_its_own_pose(self):
        heights = np.array([0.5, 1.0, 1.5], dtype=np.float32)
        positions = np.stack([np.zeros(3), np.zeros(3), heights], axis=1)
        frames = self.renderer.render(self.down, body_q=identity_bodies(positions))
        # the box top is 0.25 above its centre, and depth is the distance along the ray
        np.testing.assert_allclose(self.center_depth(frames), OFF_AXIS * (CAMERA_Z - (heights + 0.25)), atol=1e-4)

    def test_a_body_is_seen_only_in_its_own_world(self):
        positions = np.array([[100.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        depth = self.center_depth(self.renderer.render(self.down, body_q=identity_bodies(positions)))
        self.assertEqual(depth[0], 0.0)                      # world 0 moved its box away: a miss
        np.testing.assert_allclose(depth[1:], OFF_AXIS * (CAMERA_Z - 1.25), atol=1e-4)

    def test_output_shapes_and_torch_views(self):
        positions = np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32)
        frames = self.renderer.render(self.down, body_q=identity_bodies(positions))
        self.assertEqual(frames.rgb.shape, (3, 1, RES, RES, 3))
        self.assertEqual(frames.rgb.dtype, np.uint8)
        self.assertEqual(frames.depth.shape, (3, 1, RES, RES))
        np.testing.assert_array_equal(frames.rgb_torch().cpu().numpy(), frames.rgb)
        np.testing.assert_array_equal(frames.depth_torch().cpu().numpy(), frames.depth)
        center, corner = frames.rgb[0, 0, RES // 2, RES // 2], frames.rgb[0, 0, 0, 0]
        self.assertGreater(int(center[0]), int(center[2]))   # the box is red
        self.assertEqual(corner.tolist(), [200, 215, 230])   # the corner ray misses: sky

    def test_torch_and_numpy_inputs_agree(self):
        positions = np.array([[0.1, 0.0, 0.7], [0.0, 0.1, 0.9], [-0.1, 0.0, 1.1]], dtype=np.float32)
        q = identity_bodies(positions)
        from_numpy = self.renderer.render(self.down.numpy(), body_q=q).depth.copy()
        device = str(self.renderer.device)
        from_torch = self.renderer.render(self.down.to(device), body_q=torch.tensor(q, device=device)).depth
        np.testing.assert_array_equal(from_numpy, from_torch)
        self.assertEqual(self.renderer.interop, "zero-copy")

    def test_body_q_reads_back(self):
        q = identity_bodies(np.array([[1.0, 2.0, 3.0]] * 3, dtype=np.float32))
        self.renderer.set_body_q(q)
        np.testing.assert_allclose(self.renderer.body_q().cpu().numpy(), q)

    def test_joints_are_refused_without_a_urdf(self):
        with self.assertRaises(RuntimeError):
            self.renderer.set_joint_q(np.zeros((3, 1), dtype=np.float32))


@unittest.skipUnless(HAVE_NEWTON, "newton and warp are not installed (see requirements-render.txt)")
class UrdfSceneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import warp as wp
        from nedm.render import BatchRenderer, Scene, cameras
        wp.config.quiet = True
        cls.cameras = cameras
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "pendulum.urdf"
        path.write_text(URDF)
        scene = Scene.from_urdf(path, floating=False).add_ground(tile_size=None)
        cls.renderer = BatchRenderer(scene, num_worlds=2, width=RES, height=RES, cameras=2, fov_deg=FOV_DEG, shadows=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_names_and_coordinates(self):
        r = self.renderer
        self.assertEqual(r.body_names, ["base", "arm", "tip"])
        self.assertEqual(r.coord_count, 1)
        self.assertEqual(r.joint_index["shoulder"], 0)

    def test_forward_kinematics_per_world(self):
        # +90 deg about Y turns the arm's +X down to -Z: the tip goes from (1, 0, 1) to (0, 0, 0).
        r = self.renderer
        joint_map = r.joint_map(["angle"], {"shoulder": "angle"})
        r.set_joint_q(joint_map.joint_q(torch.tensor([[0.0], [np.pi / 2]])))
        tip = r.body_q()[:, r.body_index["tip"], :3].cpu().numpy()
        np.testing.assert_allclose(tip, [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], atol=1e-5)

    def test_two_cameras_per_world_and_the_ground(self):
        r = self.renderer
        eye = torch.tensor([[[5.0, 0.0, 3.0], [5.0, 0.0, 6.0]]]).repeat(2, 1, 1)       # (N, C, 3)
        cams = self.cameras.look_at(eye.reshape(-1, 3), eye.reshape(-1, 3) * torch.tensor([1.0, 1.0, 0.0]),
                                    up=(1.0, 0.0, 0.0)).reshape(2, 2, 7)
        frames = r.render(cams, joint_q=np.zeros((2, 1), dtype=np.float32))
        self.assertEqual(frames.depth.shape, (2, 2, RES, RES))
        # both cameras look straight down at empty ground 5 m from the robot
        np.testing.assert_allclose(frames.depth[:, :, RES // 2, RES // 2], OFF_AXIS * np.array([[3.0, 6.0]] * 2),
                                   atol=1e-4)


if __name__ == "__main__":
    unittest.main()
