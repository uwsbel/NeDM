"""Tests of nedm.render.poses and cameras (torch only, no renderer needed).

    PYTHONPATH=src python -m unittest discover -s tests/render -p "test_poses.py" -v
"""
import math
import unittest

import torch

from nedm.render import cameras
from nedm.render.poses import (
    JointMap,
    PlanarPose,
    base_transform,
    compose,
    quat_from_rpy,
    quat_rotate,
    tilt_from_gravity,
    transform_from_matrix,
)


def rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([quat_rotate(q, torch.eye(3)[i].expand(q.shape[0], 3)) for i in range(3)], dim=-1)


class QuaternionTest(unittest.TestCase):
    def test_yaw_quarter_turn_maps_x_to_y(self):
        q = quat_from_rpy(torch.zeros(1), torch.zeros(1), torch.full((1,), math.pi / 2))
        self.assertTrue(torch.allclose(q, torch.tensor([[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]]), atol=1e-6))
        self.assertTrue(torch.allclose(quat_rotate(q, torch.tensor([[1.0, 0.0, 0.0]])),
                                       torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6))

    def test_tilt_is_recovered_from_gravity_at_any_yaw_and_length(self):
        g = torch.Generator().manual_seed(0)
        roll = (torch.rand(64, generator=g) - 0.5) * 1.2
        pitch = (torch.rand(64, generator=g) - 0.5) * 1.2
        yaw = (torch.rand(64, generator=g) - 0.5) * 6.0
        rot = rotation_matrix(quat_from_rpy(roll, pitch, yaw))
        gravity_body = torch.einsum("nji,j->ni", rot, torch.tensor([0.0, 0.0, -1.0]))
        for scale in (1.0, 0.7):         # a predicted gravity vector is rarely unit length
            got_roll, got_pitch = tilt_from_gravity(scale * gravity_body)
            self.assertTrue(torch.allclose(got_roll, roll, atol=1e-5))
            self.assertTrue(torch.allclose(got_pitch, pitch, atol=1e-5))

    def test_matrix_round_trip(self):
        g = torch.Generator().manual_seed(1)
        rpy = (torch.rand(32, 3, generator=g) - 0.5) * 2.0
        q = quat_from_rpy(rpy[:, 0], rpy[:, 1], rpy[:, 2])
        matrix = torch.eye(4).repeat(32, 1, 1)
        matrix[:, :3, :3] = rotation_matrix(q)
        matrix[:, :3, 3] = torch.rand(32, 3, generator=g)
        xf = transform_from_matrix(matrix)
        self.assertTrue(torch.allclose(xf[:, :3], matrix[:, :3, 3]))
        same = torch.minimum((xf[:, 3:] - q).abs().amax(1), (xf[:, 3:] + q).abs().amax(1))
        self.assertLess(float(same.max()), 1e-5)

    def test_compose_places_child_in_rotated_parent(self):
        parent = base_transform(torch.tensor([[1.0, 2.0, math.pi / 2]]), z=0.5)
        child = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        self.assertTrue(torch.allclose(compose(parent, child)[:, :3], torch.tensor([[1.0, 3.0, 0.5]]), atol=1e-6))


class PlanarPoseTest(unittest.TestCase):
    def test_straight_line(self):
        pose = PlanarPose(2, dt_s=0.01)
        for _ in range(100):
            pose.step(torch.ones(2), torch.zeros(2), torch.zeros(2))
        self.assertTrue(torch.allclose(pose.xy_yaw, torch.tensor([[1.0, 0.0, 0.0]] * 2, dtype=torch.float64)))

    def test_matches_the_environment_update(self):
        # HMMWVNeuralTrackingEnv._integrate_pose, written out: yaw first, then the rotated velocity.
        g = torch.Generator().manual_seed(2)
        pose = PlanarPose(8, dt_s=0.05)
        ref = torch.zeros(8, 3, dtype=torch.float64)
        for _ in range(50):
            vx, vy, wz = (torch.rand(8, generator=g) for _ in range(3))
            pose.step(vx, vy, wz)
            yaw = ref[:, 2] + 0.05 * wz.double()
            ref = torch.stack([ref[:, 0] + 0.05 * (torch.cos(yaw) * vx - torch.sin(yaw) * vy),
                               ref[:, 1] + 0.05 * (torch.sin(yaw) * vx + torch.cos(yaw) * vy), yaw], dim=-1)
        self.assertTrue(torch.allclose(pose.xy_yaw, ref, atol=1e-12))

    def test_reset_some(self):
        pose = PlanarPose(3, dt_s=0.1)
        pose.step(torch.ones(3), torch.zeros(3), torch.zeros(3))
        pose.reset(torch.tensor([1]))
        self.assertEqual(pose.xy_yaw[:, 0].tolist(), [0.1, 0.0, 0.1])


class JointMapTest(unittest.TestCase):
    FIELDS = ["vx", "joint_a_pos", "joint_b_pos"]

    def test_by_name_with_sign_and_offset(self):
        m = JointMap({"A": 7, "B": 8}, 9, self.FIELDS, {"B": ("joint_b_pos", -1.0), "A": ("joint_a_pos", 1.0, 0.5)})
        state = torch.tensor([[9.0, 0.2, 0.3]])
        base = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]])
        q = m.joint_q(state, base)
        self.assertTrue(torch.allclose(q, torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.7, -0.3]])))

    def test_fixed_base_and_identity_default(self):
        fixed = JointMap({"A": 0}, 1, self.FIELDS, {"A": "joint_a_pos"}, base_coords=0)
        self.assertEqual(fixed.joint_q(torch.tensor([[0.0, 0.4, 0.0]])).tolist(), [[0.4000000059604645]])
        floating = JointMap({"A": 7}, 8, self.FIELDS, {"A": "joint_a_pos"})
        self.assertEqual(floating.joint_q(torch.zeros(1, 3))[0, 6].item(), 1.0)

    def test_unknown_names_are_refused(self):
        with self.assertRaises(KeyError):
            JointMap({"A": 7}, 8, self.FIELDS, {"C": "joint_a_pos"})
        with self.assertRaises(KeyError):
            JointMap({"A": 7}, 8, self.FIELDS, {"A": "missing"})


class CameraTest(unittest.TestCase):
    def test_look_at_points_minus_z_at_the_target(self):
        eye = torch.tensor([[2.0, -3.0, 1.5], [0.0, 0.0, 4.0]])
        target = torch.tensor([[0.0, 0.0, 0.2], [1.0, 0.0, 0.0]])
        xf = cameras.look_at(eye, target)
        forward = quat_rotate(xf[:, 3:], torch.tensor([[0.0, 0.0, -1.0]]).expand(2, 3))
        want = torch.nn.functional.normalize(target - eye, dim=-1)
        self.assertTrue(torch.allclose(forward, want, atol=1e-5))
        up = quat_rotate(xf[:, 3:], torch.tensor([[0.0, 1.0, 0.0]]).expand(2, 3))
        self.assertTrue(bool((up[:, 2] > 0).all()))
        self.assertTrue(torch.allclose(xf[:, :3], eye))

    def test_follow_keeps_the_offset(self):
        xy = torch.tensor([[0.0, 0.0], [3.0, -1.0]])
        xf = cameras.follow(xy, eye_offset=(-1.0, -1.0, 0.7))
        self.assertTrue(torch.allclose(xf[:, :3], torch.tensor([[-1.0, -1.0, 0.7], [2.0, -2.0, 0.7]])))
        self.assertTrue(torch.allclose(xf[0, 3:], xf[1, 3:]))

    def test_mounted_rides_the_body(self):
        body = base_transform(torch.tensor([[1.0, 0.0, math.pi / 2]]), z=0.3)
        local = cameras.look_at(torch.tensor([[0.2, 0.0, 0.1]]), torch.tensor([[1.2, 0.0, 0.1]]))[0]
        cam = cameras.mounted(body, local)
        self.assertTrue(torch.allclose(cam[:, :3], torch.tensor([[1.0, 0.2, 0.4]]), atol=1e-6))
        forward = quat_rotate(cam[:, 3:], torch.tensor([[0.0, 0.0, -1.0]]))
        self.assertTrue(torch.allclose(forward, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
