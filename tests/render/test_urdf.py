"""Tests of nedm.render.urdf and nedm.render.scene: parsing, coordinate order and forward kinematics.

Expected poses are worked out by hand. Where Newton is installed its importer and forward
kinematics are used as a second, independent implementation on a branching model.

    PYTHONPATH=src python -m unittest discover -s tests/render -p "test_urdf.py" -v
"""
import importlib.util
import math
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.render.scene import Ground, Scene
from nedm.render.urdf import load_urdf

HAVE_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

# A base with two legs. Leg A: a revolute joint whose frame is itself rotated, then a prismatic
# joint, then a fixed tip. Leg B: a revolute joint about an axis that is not a coordinate axis.
# File order puts B's joint between A's, so depth-first order is not file order.
URDF = textwrap.dedent("""\
    <robot name="tree">
      <link name="base"><visual><geometry><box size="0.4 0.2 0.1"/></geometry></visual></link>
      <link name="a1"><visual><origin xyz="0.25 0 0"/><geometry><box size="0.5 0.05 0.05"/></geometry></visual></link>
      <link name="a2"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
      <link name="a_tip"/>
      <link name="b1"><visual><geometry><sphere radius="0.05"/></geometry></visual></link>
      <joint name="a_swing" type="revolute">
        <parent link="base"/><child link="a1"/><origin xyz="0.2 0.1 0" rpy="0 0 1.5707963267948966"/><axis xyz="0 1 0"/>
        <limit lower="-3" upper="3" effort="1" velocity="1"/>
      </joint>
      <joint name="b_swing" type="continuous">
        <parent link="base"/><child link="b1"/><origin xyz="-0.2 0 0.3" rpy="0.3 -0.2 0.1"/><axis xyz="1 1 0"/>
      </joint>
      <joint name="a_slide" type="prismatic">
        <parent link="a1"/><child link="a2"/><origin xyz="0.5 0 0"/><axis xyz="1 0 0"/>
        <limit lower="0" upper="1" effort="1" velocity="1"/>
      </joint>
      <joint name="a_end" type="fixed"><parent link="a2"/><child link="a_tip"/><origin xyz="0 0 0.2"/></joint>
    </robot>
    """)


def identity_base(n: int) -> torch.Tensor:
    base = torch.zeros(n, 7)
    base[:, 6] = 1.0
    return base


class UrdfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "tree.urdf"
        cls.path.write_text(URDF)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_coordinates_are_depth_first_not_file_order(self):
        model = load_urdf(self.path, floating=True)
        self.assertEqual(model.coord_count, 10)
        self.assertEqual(model.joint_index, {"floating_base": 0, "a_swing": 7, "a_slide": 8, "b_swing": 9})
        fixed = load_urdf(self.path, floating=False)
        self.assertEqual(fixed.joint_index, {"a_swing": 0, "a_slide": 1, "b_swing": 2})

    def test_forward_kinematics_by_hand(self):
        # a_swing's frame is yawed 90 deg, so its +X is world +Y and its Y axis is world -X.
        # Turning +90 deg about that axis swings the leg's +X from world +Y down to world -Z.
        model = load_urdf(self.path, floating=False)
        name = {link.name: i for i, link in enumerate(model.links)}
        q = torch.tensor([[0.0, 0.0, 0.0], [math.pi / 2, 0.25, 0.0]])
        _rotation, position = model.forward_kinematics(q)
        at_rest, swung = position[0], position[1]
        self.assertTrue(torch.allclose(at_rest[name["a2"]], torch.tensor([0.2, 0.6, 0.0]), atol=1e-6))
        self.assertTrue(torch.allclose(at_rest[name["a_tip"]], torch.tensor([0.2, 0.6, 0.2]), atol=1e-6))
        self.assertTrue(torch.allclose(swung[name["a2"]], torch.tensor([0.2, 0.1, -0.75]), atol=1e-6))
        # the tip sits 0.2 along the leg's +Z, which the swing has turned to world +Y
        self.assertTrue(torch.allclose(swung[name["a_tip"]], torch.tensor([0.2, 0.3, -0.75]), atol=1e-6))

    def test_floating_base_moves_everything(self):
        model = load_urdf(self.path, floating=True)
        name = {link.name: i for i, link in enumerate(model.links)}
        q = torch.zeros(1, 10)
        q[0, :3] = torch.tensor([1.0, 2.0, 3.0])
        q[0, 3:7] = torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])       # yaw 90 deg
        _rotation, position = model.forward_kinematics(q)
        # a2 at rest is (0.2, 0.6, 0) in the base frame, which the yaw turns to (-0.6, 0.2, 0)
        self.assertTrue(torch.allclose(position[0, name["a2"]], torch.tensor([0.4, 2.2, 3.0]), atol=1e-6))

    def test_visual_offsets_are_baked_into_the_link_mesh(self):
        model = load_urdf(self.path)
        a1 = next(link for link in model.links if link.name == "a1")
        np.testing.assert_allclose(a1.vertices.mean(axis=0), [0.25, 0.0, 0.0], atol=1e-6)
        self.assertIsNone(next(link for link in model.links if link.name == "a_tip").vertices)

    def test_scene_draws_only_links_with_geometry(self):
        scene = Scene.from_urdf(self.path, colors={"a2": (1.0, 0.0, 0.0)})
        self.assertEqual(scene.body_names, ["base", "a1", "a2", "b1"])
        self.assertEqual(scene.bodies[2].color, (1.0, 0.0, 0.0))
        q = torch.zeros(5, 10)
        q[:, 6] = 1.0
        self.assertEqual(tuple(scene.body_transforms(q).shape), (5, 4, 7))

    @unittest.skipUnless(HAVE_NEWTON, "newton is not installed")
    def test_matches_newton_importer_and_kinematics(self):
        import newton
        import warp as wp
        wp.config.quiet = True
        builder = newton.ModelBuilder()
        builder.add_urdf(str(self.path), floating=True)
        reference = builder.finalize()          # the default device: AMD's Warp build has no CPU kernels
        state = reference.state()
        ours = load_urdf(self.path, floating=True)
        their_index = {label.split("/")[-1]: int(builder.joint_q_start[j]) for j, label in enumerate(builder.joint_label)}
        for joint, coord in ours.joint_index.items():
            self.assertEqual(their_index[joint], coord, joint)

        g = torch.Generator().manual_seed(0)
        q = torch.rand(1, ours.coord_count, generator=g) * 2 - 1
        q[0, 3:7] = torch.nn.functional.normalize(torch.rand(4, generator=g) - 0.5, dim=0)
        q[0, ours.joint_index["a_slide"]] = 0.3
        state.joint_q.assign(q[0].numpy())
        newton.eval_fk(reference, state.joint_q, state.joint_qd, state)
        theirs = dict(zip([label.split("/")[-1] for label in builder.body_label], state.body_q.numpy(), strict=True))
        rotation, position = ours.forward_kinematics(q)
        for i, link in enumerate(ours.links):
            np.testing.assert_allclose(position[0, i].numpy(), theirs[link.name][:3], atol=2e-6, err_msg=link.name)
            x, y, z, w = theirs[link.name][3:]
            matrix = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                               [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                               [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            np.testing.assert_allclose(rotation[0, i].numpy(), matrix, atol=2e-6, err_msg=link.name)


class GroundTest(unittest.TestCase):
    def test_checker_is_one_mesh_with_every_tile(self):
        ground = Ground(tile_size=0.5, extent=(-1.0, 1.0, -1.0, 1.0))
        centers = ground.tile_centers()
        self.assertEqual(len(centers), 8)                   # half of a 4 by 4 grid
        mesh = ground.tile_mesh()
        vertices, faces = mesh.arrays()
        self.assertEqual(vertices.shape, (64, 3))
        self.assertEqual(faces.shape, (96, 3))
        self.assertEqual(int(faces.max()), 63)
        self.assertAlmostEqual(float(vertices[:, 2].max()), ground.tile_thickness, places=6)
        self.assertIsNone(Ground(tile_size=None).tile_mesh())


if __name__ == "__main__":
    unittest.main()
