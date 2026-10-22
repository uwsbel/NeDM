"""Tests of the lowered / tilted table option of the CPU MuJoCo evaluation (``nedm.so101_policy.table``,
``JointVecEnv.set_table``, ``task.features(plane=)``, ``joint_env.floor_guard(plane=)``, ``viral.load_real_setup(
geometry_over=)``). CPU, a few seconds; MuJoCo tests skip without mujoco.

    PYTHONPATH=src python -m unittest tests.so101_policy.test_table_offset -v
"""
from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import joint_env as JE
from nedm.so101_policy import table as TB

HAVE_MUJOCO = importlib.util.find_spec("mujoco") is not None
PLANE = (-0.004681260916979778, -0.03640954226125585, 0.01836320115536056)     # calib_v1.json scene.table_plane
HOME = np.radians([0.0, -5.0, -15.0, 92.0, 0.0, 30.0])


def tasks():
    """Two task starts: the HOME arm pose, a flat cube (yaw 0.3 / 1.1 rad) before the warm-up, goals."""
    yaw = np.array([0.3, 1.1])
    quat = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], 1)
    return dict(q=np.stack([HOME, HOME + 0.02]), cube_pos=np.array([[0.22, 0.05, 0.015], [0.27, -0.12, 0.015]]),
                cube_quat=quat, goal_xy=np.array([[0.2, -0.1], [0.18, 0.1]]), goal_yaw=np.array([0.2, 0.5]))


def env(plane="unset", floor="zero"):
    e = JE.JointVecEnv(JE.JointEnvConfig(world="mujoco", num_envs=2, device="cpu", auto_reset=False, stagger=False,
                                         starts=""), JE.Codec(np.zeros(18), np.ones(18)))
    if plane != "unset":
        e.set_table(plane, floor)
    return e


def reset(e, t):
    from nedm.so101_policy.world import WorldState
    f = lambda x: torch.as_tensor(x, dtype=torch.float64)                                     # noqa: E731
    st = WorldState(q=f(t["q"]), qd=torch.zeros(2, 6, dtype=torch.float64), cube_pos=f(t["cube_pos"]),
                    cube_quat=f(t["cube_quat"]), cube_v=torch.zeros(2, 3, dtype=torch.float64),
                    cube_w=torch.zeros(2, 3, dtype=torch.float64), contact=torch.zeros(2, 3, dtype=torch.bool),
                    arm_table=torch.zeros(2, dtype=torch.bool), invalid=torch.zeros(2, dtype=torch.bool))
    mj = dict(q=t["q"], cube_pos=t["cube_pos"], cube_quat=t["cube_quat"])
    # the stored settled state is not known here: no warm-up check
    e.reset_tasks(torch.arange(2), st, f(t["goal_xy"]), f(t["goal_yaw"]), torch.arange(2), mj, check_tol=None)


def run(e, n=6):
    """n decisions moving the arm down toward the table (the floor guard acts) -> stacked states."""
    out = []
    for k in range(n):
        tg = e.ws.q[:, None, :].expand(2, JE.STEPS, 6).clone()
        tg[..., 1] += 0.04 * (k + 1)      # shoulder lift forward / down
        tg[..., 2] -= 0.02 * (k + 1)
        e.step(None, targets=tg)
        out.append(torch.cat((e.ws.q, e.ws.cube_pos, e.ws.cube_quat, e.u_exe.reshape(2, -1)), -1))
    return torch.stack(out, 1)


class TestPlaneMath(unittest.TestCase):
    def test_parse(self):
        self.assertIsNone(TB.parse(0.0, None))
        self.assertIsNone(TB.parse(0.0, "0,0,0"))
        self.assertEqual(TB.parse(-0.01, None), (-0.01, 0.0, 0.0))
        np.testing.assert_allclose(TB.parse(-0.01, "0.001,0.1,0.2"), (-0.009, 0.1, 0.2), atol=1e-15)
        self.assertEqual(TB.parse_floor("table"), "table")
        self.assertEqual(TB.parse_floor("-0.01"), -0.01)

    def test_map_cube_lies_on_the_plane(self):
        p, q = TB.map_cube(PLANE, np.array([[0.25, -0.1, 0.015]]), np.array([[1.0, 0, 0, 0]]))
        self.assertAlmostEqual(float(TB.above(PLANE, p)[0]), 0.015, places=12)
        n, _ = TB.normal(PLANE)
        w, x, y, z = q[0]                                     # cube z axis = plane normal
        rz = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        np.testing.assert_allclose(rz, n, atol=1e-12)
        self.assertEqual(p[0, 0], 0.25)

    def test_flat_plane_heights_are_shifts(self):
        from nedm.so101_policy import task as T
        from nedm.so101_policy.vec_env import kinematics, task_geometry
        kin, geom = kinematics(JE.JointEnvConfig().kin_points, "cpu"), task_geometry(JE.JointEnvConfig().geom_path, "cpu")
        g = torch.Generator().manual_seed(0)
        q = torch.as_tensor(HOME)[None] + 0.3 * torch.randn(16, 6, generator=g, dtype=torch.float64)
        dz = -0.012
        np.testing.assert_allclose(kin.lowest_point(q, exact=False, plane=(dz, 0, 0)),
                                   kin.lowest_point(q, exact=False) - dz, atol=1e-12)
        bpos, brot = kin.body_poses(q)
        np.testing.assert_allclose(geom.lowest(bpos, brot, (dz, 0, 0)), geom.lowest(bpos, brot) - dz, atol=1e-12)
        from nedm.so101_policy.world import WorldState
        quat = torch.nn.functional.normalize(torch.randn(16, 4, generator=g, dtype=torch.float64), dim=-1)
        ws = WorldState(q=q, qd=torch.zeros_like(q), cube_pos=torch.rand(16, 3, generator=g, dtype=torch.float64),
                        cube_quat=quat, cube_v=torch.zeros(16, 3, dtype=torch.float64),
                        cube_w=torch.zeros(16, 3, dtype=torch.float64), contact=torch.zeros(16, 3, dtype=torch.bool),
                        arm_table=torch.zeros(16, dtype=torch.bool), invalid=torch.zeros(16, dtype=torch.bool))
        cfg = T.TaskConfig()
        F0 = T.features(ws, kin, geom, cfg, overlap=False)
        F1 = T.features(ws, kin, geom, cfg, overlap=False, plane=(dz, 0, 0))
        np.testing.assert_allclose(F1["bottom"], F0["bottom"] - dz, atol=1e-12)
        np.testing.assert_allclose(F1["lowest"], F0["lowest"] - dz, atol=1e-12)
        self.assertTrue(torch.equal(F1["base_zone"], F0["base_zone"]))       # the base does not move
        for k in ("c", "tcp", "xg", "tilt", "cyaw"):                         # the observation is the world's
            self.assertTrue(torch.equal(F1[k], F0[k]), k)
        # the floor guard: a flat plane = the robot's --table-z (floor + dz)
        u = q + 0.2
        a = JE.floor_guard(u, q, kin, JE.FLOOR, plane=(dz, 0, 0))
        b = JE.floor_guard(u, q, kin, JE.FLOOR + dz)
        np.testing.assert_allclose(a, b, atol=1e-9)


@unittest.skipUnless(HAVE_MUJOCO, "needs mujoco")
class TestMujocoTable(unittest.TestCase):
    def test_off_is_unchanged(self):
        """No set_table, set_table(None) and a plane set then removed give bit-identical episodes."""
        t = tasks()
        outs = []
        for mode in ("none", "off", "restored"):
            e = env() if mode == "none" else env(None)
            if mode == "restored":
                e.set_table(PLANE, "table")
                e.set_table(None)
            reset(e, t)
            outs.append(run(e))
            e.close()
        self.assertTrue(torch.equal(outs[0], outs[1]))
        self.assertTrue(torch.equal(outs[0], outs[2]))

    def test_cube_rests_on_the_new_table(self):
        t = tasks()
        e0 = env()
        reset(e0, t)
        c0, q0 = e0.ws.cube_pos.clone(), e0.ws.cube_quat.clone()
        e0.close()
        self.assertTrue(np.allclose(c0[:, 2], 0.014993169, atol=1e-8))
        for plane, tol in (((-0.01, 0.0, 0.0), 1e-8), (PLANE, 2e-6)):
            e = env(plane)
            reset(e, t)
            c = e.ws.cube_pos
            # resting at the sim rest height above the plane (along its normal), xy where the task put it
            np.testing.assert_allclose(TB.above(plane, c.numpy()), 0.014993169, atol=tol)
            np.testing.assert_allclose(c[:, :2], c0[:, :2], atol=1e-5)
            if TB.is_flat(plane):
                np.testing.assert_allclose(c[:, 2], c0[:, 2] + plane[0], atol=1e-8)
                np.testing.assert_allclose(e.ws.cube_quat, q0, atol=1e-8)
            hold = e.ws.q[:, None, :].expand(2, JE.STEPS, 6).clone()
            for _ in range(5):                 # 0.5 s held: at rest on the table, no end
                e.step(None, targets=hold)
            F = e.task.F
            self.assertTrue(bool(F["on_table"].all()))
            np.testing.assert_allclose(F["bottom"], -7e-6, atol=5e-6)
            self.assertTrue(bool((e.task.tr.end == 0).all()))
            self.assertTrue(bool((e.world.end_reason == 0).all()))
            np.testing.assert_allclose(TB.above(plane, e.ws.cube_pos.numpy()), 0.014993169, atol=tol)
            e.close()

    def test_floor_follows_the_table(self):
        """The action layer's floor: 'zero' = 1.5 mm above z = 0 (default); 'table' = 1.5 mm above the table (flat:
        FLOOR + dz, the robot's --table-z; tilted: heights above the plane); a float = a flat floor there."""
        def lowest(e, u):
            return float(e.kin.lowest_point(u, exact=False).min())

        e = env((-0.015, 0.0, 0.0), "zero")
        self.assertEqual((e.floor_z, e.floor_plane), (0.0, None))
        q = torch.as_tensor(HOME, dtype=torch.float64)[None]
        lo, hi = 0.0, 1.5                                  # shoulder-lift step that puts the gripper 10 mm below z = 0
        for _ in range(60):
            mid = (lo + hi) / 2
            u = q.clone()
            u[:, 1] += mid
            lo, hi = (mid, hi) if lowest(e, u) > -0.010 else (lo, mid)
        u = q.clone()
        u[:, 1] += hi
        self.assertGreater(lowest(e, q), 0.01)
        self.assertAlmostEqual(lowest(e, u), -0.010, places=5)

        def layer(env_):
            fl = JE.FLOOR + env_.floor_z
            return JE.safety(u, q, env_.tlo, env_.thi, None, env_.kin, fl, env_.floor_plane)

        # clipped to 1.5 mm above z = 0 (the guard's bisection: 8 halvings of the step)
        self.assertTrue(JE.FLOOR - 1e-6 <= lowest(e, layer(e)) < JE.FLOOR + 1e-3)
        e.set_table((-0.015, 0.0, 0.0), "table")
        self.assertEqual((e.floor_z, e.floor_plane), (-0.015, None))
        self.assertTrue(torch.equal(layer(e), u))                               # -10 mm is 5 mm above the floor
        e.set_table((-0.015, 0.0, 0.0), -0.005)
        self.assertTrue(JE.FLOOR - 0.005 - 1e-6 <= lowest(e, layer(e)) < JE.FLOOR - 0.004)
        e.set_table(PLANE, "table")
        self.assertEqual(e.floor_plane, PLANE)
        for dz in (0.0, 0.004):            # the gripper's height above the plane: above / below 1.5 mm
            e.set_table((PLANE[0] + dz, PLANE[1], PLANE[2]), "table")
            plane = e.floor_plane
            hu = float(e.kin.lowest_point(u, exact=False, plane=plane).min())
            out = layer(e)
            h = float(e.kin.lowest_point(out, exact=False, plane=plane).min())
            if hu >= JE.FLOOR:
                self.assertTrue(torch.equal(out, u))
            else:
                self.assertTrue(JE.FLOOR - 1e-6 <= h < JE.FLOOR + 1e-3, (hu, h))
        e.close()


@unittest.skipUnless(HAVE_MUJOCO, "needs mujoco (arm meshes)")
class TestRenderOverride(unittest.TestCase):
    def test_geometry_override_keeps_the_pin(self):
        from tests.so101_policy.test_vision_viral_r2s import CALIB, placeholder_scene
        from nedm.so101_policy.vision import viral as VI
        if not CALIB.exists():
            self.skipTest("needs the calibration")
        with tempfile.TemporaryDirectory() as d:
            sp = placeholder_scene(Path(d))
            cfg = VI.ViralConfig(res=64, cameras=("external", "wrist"), scene=str(sp))
            VI.pin_real(cfg)
            rs0 = VI.load_real_setup(cfg)
            rs1 = VI.load_real_setup(cfg, {"table_plane": list(PLANE)})
            self.assertIsNone(rs0.real.table_plane)
            self.assertEqual(tuple(rs1.real.table_plane), PLANE)
            self.assertEqual(rs1.scene_sha256, cfg.scene_sha256)           # the file is unchanged
            self.assertEqual(rs1.crops, rs0.crops)
            self.assertEqual(json.loads(sp.read_text()).get("geometry", {}).get("table_plane"), None)
            with self.assertRaises(ValueError):
                VI.load_real_setup(cfg, {"not_a_key": 1})


if __name__ == "__main__":
    unittest.main()
