"""Task rules, goals, observation and reward of the SO101 pick-and-place task (nedm.so101_policy.task).

CPU, a few seconds. Tests that need the arm geometry (built from the collector's MuJoCo scene) skip when mujoco or the
vendor Menagerie model is missing; the CUDA parity test skips without CUDA.

    PYTHONPATH=src python -m unittest tests.so101_policy.test_task -v
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from nedm.so101_policy import task as T
from nedm.so101_policy.kin import SO101Kin, mat_quat, rotz
from nedm.so101_policy.world import WorldState

try:
    import mujoco  # noqa: F401
    from nedm.contact_nrd.collect import so101_grasp as g
    _SCENE = g.Scene(g.load_config())
    _GEOM = T.TaskGeometry.from_mujoco()
except Exception:   # mujoco or the vendor model missing
    _SCENE, _GEOM = None, None
needs_geom = unittest.skipIf(_GEOM is None, "mujoco / vendor Menagerie model not available")
GRIP_X, GRIP_Z, JAW_FIRM = -0.0043, -0.0008, 0.167     # held cube in the grasp frame (data: x -4.3 mm), firm jaw


def make_state(q, cube_pos, cube_quat, dt=0.02):
    """WorldState list [T] (one world each) from poses [T, ...], with step-average rates (the data definition)."""
    q, pos, quat = (torch.as_tensor(x, dtype=torch.float64) for x in (q, cube_pos, cube_quat))
    gr = T.grid_from_records(torch.cat((q, torch.zeros_like(q)), -1)[None],
                             torch.cat((pos, quat, torch.zeros(len(q), 6, dtype=torch.float64)), -1)[None],
                             stride=1, dt=dt)
    return [T.grid_state(gr, j) for j in range(len(q))]


class Episode:
    """A kinematic pick-and-place episode (no physics): TCP waypoints -> IK (top-down, 45 deg tilt fallback), the cube
    at rest or rigidly held at the firm-grip offset. Phases list of (macro, steps, tcp end [3], jaw end, held, drop)."""

    def __init__(self, kin, cube0=(0.22, 0.05), yaw=0.3, goal=(0.25, -0.08)):
        self.kin = kin
        self.cube0 = torch.tensor([cube0[0], cube0[1], T.CUBE_REST_Z], dtype=torch.float64)
        self.yaw = yaw
        self.goal = torch.tensor(goal, dtype=torch.float64)

    _CACHE: dict = {}

    def build(self, release_bottom=0.001, retreat=0.06, lift_bottom=0.03):
        key = (release_bottom, retreat, lift_bottom, self.yaw, tuple(self.cube0.tolist()), tuple(self.goal.tolist()))
        if key not in Episode._CACHE:
            Episode._CACHE[key] = self._build(release_bottom, retreat, lift_bottom)
        return tuple(x.clone() for x in Episode._CACHE[key])

    def _build(self, release_bottom, retreat, lift_bottom):
        c0, yaw = self.cube0, self.yaw
        top = lambda xy, z: torch.tensor([xy[0], xy[1], z], dtype=torch.float64)   # noqa: E731
        grasp_z = float(c0[2]) - GRIP_Z
        carry_z = grasp_z + lift_bottom
        place_z = grasp_z + release_bottom
        plan = [("free", 5, top(c0, grasp_z + 0.05), 0.7, False),
                ("grasp_descend", 6, top(c0, grasp_z), 0.7, False),
                ("grasp_close", 5, top(c0, grasp_z), JAW_FIRM, False),
                ("grasp_grip", 3, top(c0, grasp_z), JAW_FIRM, True),
                ("grasp_lift", 10, top(c0, carry_z), JAW_FIRM, True),
                ("free", 15, top(self.goal, carry_z), JAW_FIRM, True),
                ("place_lower", 8, top(self.goal, place_z), JAW_FIRM, True),
                ("place_open", 5, top(self.goal, place_z), 0.7, False),
                ("place_retreat", 10, top(self.goal, place_z + retreat), 0.7, False),
                ("free", 5, top(self.goal, place_z + retreat), 0.7, False)]
        tcp, jaw, held, macro = [], [], [], []
        p, j = plan[0][2].clone(), plan[0][3]
        for m, n, p1, j1, h in plan:
            for k in range(1, n + 1):
                s = k / n
                tcp.append(p + (p1 - p) * s)
                jaw.append(j + (j1 - j) * s)
                held.append(h)
                macro.append(T.MC[m])
            p, j = p1, j1
        tcp = torch.stack(tcp)
        q5, ok, _, _ = self.kin.ik_topdown(None, tcp, torch.full((len(tcp),), yaw, dtype=torch.float64),
                                           max_tilt=math.radians(45))
        assert bool(ok.all())
        q = torch.cat((q5, torch.tensor(jaw, dtype=torch.float64)[:, None]), -1)
        tp, rg = self.kin.fk(q)
        cube, quat = [], []
        rest_pos, rest_quat = self.cube0.clone(), mat_quat(rotz(torch.tensor(yaw, dtype=torch.float64)))
        for i in range(len(q)):
            if held[i]:
                pos = tp[i] + rg[i] @ torch.tensor([GRIP_X, 0.0, GRIP_Z], dtype=torch.float64)
                qq = mat_quat(rg[i])
                rest_pos, rest_quat = pos.clone(), qq.clone()
                rest_pos[2] = T.CUBE_REST_Z
            else:
                pos, qq = rest_pos.clone(), rest_quat.clone()
            cube.append(pos)
            quat.append(qq)
        return q, torch.stack(cube), torch.stack(quat), torch.tensor(macro)


def run(task, states, macro):
    out = []
    for st, m in zip(states, macro):
        out.append(task.step(st, m[None]))
    return out


class TestHelpers(unittest.TestCase):
    def test_yaw_tilt_invariant_under_cube_symmetry(self):
        torch.manual_seed(0)
        from nedm.so101_policy import cube_symmetry as cs
        G = cs.group(torch.float64, "cpu")                                         # 24 rotations, w x y z
        yaw = torch.rand(10, dtype=torch.float64) * 6
        tilt_axis = torch.randn(10, 3, dtype=torch.float64) * torch.tensor([1.0, 1.0, 0.0])
        tilt_axis = tilt_axis / tilt_axis.norm(dim=-1, keepdim=True)
        ang = torch.rand(10, dtype=torch.float64) * 0.2
        qt = torch.cat((torch.cos(ang / 2)[:, None], tilt_axis * torch.sin(ang / 2)[:, None]), -1)
        q = cs.qmul(qt, mat_quat(rotz(yaw)))
        y0, t0 = T.cube_yaw_tilt(q)
        for k in range(24):
            y, t = T.cube_yaw_tilt(cs.qmul(q, G[k].expand_as(q)))
            self.assertLess(float(T.wrap90(y - y0).abs().max()), 1e-9)
            self.assertLess(float((t - t0).abs().max()), 1e-9)
        self.assertLess(float((t0 - ang).abs().max()), 1e-9)

    def test_wrap90(self):
        a = torch.tensor([0.0, math.pi / 4, -math.pi / 4 + 1e-9, math.pi / 2, 1.0], dtype=torch.float64)
        w = T.wrap90(a)
        self.assertTrue(bool(((w > -math.pi / 4) & (w <= math.pi / 4 + 1e-12)).all()))
        self.assertAlmostEqual(float(w[3]), 0.0, places=12)
        self.assertAlmostEqual(float(w[4]), 1.0 - math.pi / 2, places=12)

    def test_sat_depth_axis_aligned_and_rotated(self):
        eye = torch.eye(3, dtype=torch.float64)
        h1 = torch.tensor([0.015] * 3, dtype=torch.float64)
        h2 = torch.tensor([0.01, 0.02, 0.005], dtype=torch.float64)
        for d, want in ((0.02, 0.005), (0.03, -0.005), (0.0, 0.025)):
            c2 = torch.tensor([d, 0.0, 0.0], dtype=torch.float64)
            got = float(T.sat_depth(torch.zeros(3, dtype=torch.float64), eye, h1, c2, eye, h2))
            self.assertAlmostEqual(got, min(want, 0.015 + 0.005), places=12)
        # a cube turned 45 deg about z, corner pointing at an axis-aligned box face along x
        r = rotz(torch.tensor(math.pi / 4, dtype=torch.float64))
        c2 = torch.tensor([0.015 * math.sqrt(2) + 0.01 - 0.002, 0.0, 0.0], dtype=torch.float64)
        got = float(T.sat_depth(torch.zeros(3, dtype=torch.float64), r, h1, c2, eye, h2))
        self.assertAlmostEqual(got, 0.002, places=9)

    def test_grid_rates(self):
        torch.manual_seed(1)
        q = torch.cumsum(torch.randn(1, 9, 12, dtype=torch.float64) * 0.01, 1)
        cube = torch.zeros(1, 9, 13, dtype=torch.float64)
        cube[..., :3] = torch.cumsum(torch.randn(1, 9, 3, dtype=torch.float64) * 0.001, 1)
        ang = torch.linspace(0, 0.4, 9, dtype=torch.float64)
        cube[..., 3:7] = mat_quat(rotz(ang))[None]
        gr = T.grid_from_records(q, cube, stride=1, dt=0.02)
        self.assertLess(float((gr["qd"][0, 1:] - (q[0, 1:, :6] - q[0, :-1, :6]) / 0.02).abs().max()), 1e-12)
        self.assertLess(float((gr["cube_w"][0, 1:, 2] - 0.05 / 0.02).abs().max()), 1e-9)


class TestGoals(unittest.TestCase):
    def test_goal_sampling(self):
        gen = torch.Generator().manual_seed(3)
        xy = T.sample_annulus(2000, gen)
        yaw = torch.rand(2000, generator=gen, dtype=torch.float64) * math.pi / 2
        gxy, gyaw = T.sample_goals(xy, yaw, gen)
        d = (gxy - xy).norm(dim=-1)
        self.assertTrue(bool(((d >= 0.06 - 1e-12) & (d <= 0.30 + 1e-12)).all()))
        rel = gxy - torch.tensor(T.PAN_AXIS_XY, dtype=torch.float64)
        r, a = rel.norm(dim=-1), torch.rad2deg(torch.atan2(rel[:, 1], rel[:, 0]))
        self.assertTrue(bool(((r >= 0.15 - 1e-12) & (r <= 0.27 + 1e-12) & (a.abs() <= 75 + 1e-9)).all()))
        turn = T.wrap90(gyaw - yaw)
        self.assertLessEqual(float(turn.abs().max()), math.radians(30) + 1e-12)
        self.assertGreater(float(turn.abs().max()), math.radians(29))
        # a goal change with 0.5 rad turned already: the total turn stays <= 0.6 rad
        turned = torch.full((2000,), 0.5, dtype=torch.float64)
        _, gy2 = T.sample_goals(xy, yaw, gen, turned=turned)
        total = turned + T.wrap90(gy2 - yaw)
        self.assertLessEqual(float(total.abs().max()), 0.6 + 1e-12)

    def test_deterministic(self):
        xy = T.sample_annulus(64, torch.Generator().manual_seed(5))
        yaw = torch.zeros(64, dtype=torch.float64)
        a = T.sample_goals(xy, yaw, torch.Generator().manual_seed(7))
        b = T.sample_goals(xy, yaw, torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))


@needs_geom
class TestGeometry(unittest.TestCase):
    def test_lowest_matches_collector(self):
        import mujoco as mj
        rng = np.random.default_rng(0)
        kin = SO101Kin()
        q = torch.as_tensor(rng.uniform(-1.2, 1.2, (40, 6)))
        bpos, brot = kin.body_poses(q)
        low = _GEOM.lowest(bpos, brot).numpy()
        s = _SCENE
        for i in range(len(q)):
            s.dk.qpos[:] = 0
            s.dk.qpos[s.qadr] = q[i].numpy()
            mj.mj_kinematics(s.m, s.dk)
            self.assertLess(abs(low[i] - s.lowest(s.dk)), 1e-9)

    def test_overlap_cube_far_and_inside(self):
        kin = SO101Kin()
        q = torch.zeros(2, 6, dtype=torch.float64)
        q[:, 5] = 0.7
        tcp, rg = kin.fk(q)
        bpos, brot = kin.body_poses(q)
        far = torch.tensor([[0.5, 0.4, 0.015]], dtype=torch.float64)
        # second: the cube 7 mm from the TCP towards the fixed jaw (inner face at x = -18.5 mm): ~3.5 mm inside it
        c = torch.cat((far, tcp[1:] + rg[1] @ torch.tensor([-0.007, 0.0, 0.0], dtype=torch.float64)))
        rc = rg.clone()
        rc[0] = torch.eye(3, dtype=torch.float64)
        link = _GEOM.link_depth(bpos, brot, c, rc)
        vis, col = _GEOM.jaw_depths(bpos, brot, c, rc)
        self.assertLess(float(link[0]), 0)
        self.assertLess(float(vis[0].max()), 0)
        self.assertLess(float(col[0].max()), 0)
        self.assertGreater(float(col[1, 0]), 0.002)                                  # fixed jaw (gripper body)
        self.assertLess(float(col[1, 1]), 0.0)                                       # moving jaw open at 0.7 rad
        # centred at the TCP with the jaws open: no contact (2.85 mm clearance)
        vis, col = _GEOM.jaw_depths(bpos, brot, tcp, rg)
        self.assertLess(float(col[1].max()), 0)


@needs_geom
class TestTaskEpisodes(unittest.TestCase):
    def setUp(self):
        self.kin = SO101Kin()

    def task(self, cfg=None, device="cpu"):
        return T.PickPlaceTask(1, device, kin=SO101Kin(device), geom=T.TaskGeometry.from_mujoco(device), cfg=cfg)

    def episode(self, **kw):
        ep = Episode(self.kin)
        q, cube, quat, macro = ep.build(**kw)
        return ep, make_state(q, cube, quat), macro, cube

    def test_clean_pick_and_place_succeeds(self):
        ep, states, macro, cube = self.episode()
        task = self.task()
        gyaw, _ = T.cube_yaw_tilt(states[-1].cube_quat)
        task.reset(torch.tensor([0]), states[0], cube[-1:, :2], gyaw)
        res = run(task, states[1:], macro[1:])
        ends = [int(r[3]["end"][0]) for r in res]
        fired = torch.stack([r[3]["flags"][0] for r in res]).any(0)
        self.assertFalse(bool(fired.any()), [T.RULES[k] for k in torch.nonzero(fired).flatten().tolist()])
        self.assertIn(T.EC["success"], ends)
        k = ends.index(T.EC["success"])
        self.assertTrue(bool(res[-1][3]["success10"][0]))                            # latched after the end
        self.assertLess(float(res[-1][3]["end_xy_err"][0]), 1e-3)
        self.assertTrue(math.isnan(float(res[0][3]["end_xy_err"][0])))
        self.assertEqual(int(macro[k + 1]) in (T.MC["place_retreat"], T.MC["free"]), True)
        self.assertTrue(bool(res[k][3]["lifted"][0]))
        # reward: lift bonus once and success bonus at the end
        rew = torch.stack([r[0][0] for r in res[:k + 1]])
        self.assertGreater(float(rew[-1]), task.cfg.success_bonus - 1.0)
        self.assertTrue(bool(res[k][1][0]) and not bool(res[k][2][0]))
        # after the end, rewards are 0 and the code stays
        st = states[-1]
        r2 = task.step(st, macro[-1:])
        self.assertEqual(float(r2[0][0]), 0.0)
        self.assertEqual(int(r2[3]["end"][0]), T.EC["success"])

    def test_short_retreat_is_not_success(self):
        ep, states, macro, cube = self.episode(retreat=0.04)
        task = self.task()
        gyaw, _ = T.cube_yaw_tilt(states[-1].cube_quat)
        task.reset(torch.tensor([0]), states[0], cube[-1:, :2], gyaw)
        ends = [int(r[3]["end"][0]) for r in run(task, states[1:], macro[1:])]
        self.assertNotIn(T.EC["success"], ends)

    def test_high_release_fails(self):
        kin = self.kin
        ep = Episode(kin)
        q, cube, quat, macro = ep.build(release_bottom=0.03)
        # after the jaw opens the cube falls freely from 3 cm
        k0 = int(torch.nonzero(macro == T.MC["place_open"])[0])
        z0 = float(cube[k0 - 1, 2])
        for i in range(k0, len(cube)):
            t = (i - k0 + 1) * 0.02
            cube[i, 2] = max(T.CUBE_REST_Z, z0 - 0.5 * 9.81 * t * t)
        states = make_state(q, cube, quat)
        task = self.task()
        task.reset(torch.tensor([0]), states[0], cube[-1:, :2], torch.zeros(1, dtype=torch.float64))
        ends = [int(r[3]["end"][0]) for r in run(task, states[1:], macro[1:])]
        first = next(e for e in ends if e)
        self.assertIn(T.ENDS[first], ("release_high", "unsupported"))

    def test_grasp_box_at_trigger_and_log_only(self):
        ep, states, macro, cube = self.episode()
        for dy, fires in ((0.02, True), (0.005, False)):
            st = [WorldState(**{**s.__dict__}) for s in states]
            k_trig = int(torch.nonzero(macro == T.MC["grasp_descend"])[0])
            # move the resting cube sideways (world y) before the trigger
            for s in st[:k_trig + 1]:
                s.cube_pos = s.cube_pos + torch.tensor([[0.0, dy, 0.0]], dtype=torch.float64)
            for log in (False, True):
                cfg = T.TaskConfig.evaluation() if log else T.TaskConfig()
                task = self.task(cfg)
                task.reset(torch.tensor([0]), st[0], cube[-1:, :2], torch.zeros(1, dtype=torch.float64))
                res = run(task, st[1:k_trig + 1], macro[1:k_trig + 1])
                flag = bool(res[-1][3]["flags"][0, T.RULES.index("grasp_box")])
                self.assertEqual(flag, fires)
                self.assertEqual(int(res[-1][3]["end"][0]) == T.EC["grasp_box"], fires and not log)
                self.assertEqual(bool(res[-1][3]["grasp_box_log"][0]), fires)

    def test_grasp_box_with_macro_goal(self):
        ep, states, macro, cube = self.episode()
        k = int(torch.nonzero(macro == T.MC["grasp_descend"])[0])
        gyaw, _ = T.cube_yaw_tilt(states[0].cube_quat)
        c0 = states[0].cube_pos[0]
        for dz, dyaw, fires in ((0.0008, 0.0, False), (0.0008, 0.07, True), (0.02, 0.0, True), (-0.0008, 0.0, False)):
            goal = torch.tensor([[float(c0[0]), float(c0[1]), float(c0[2]) + dz, float(gyaw) + dyaw]],
                                dtype=torch.float64)
            task = self.task()
            task.reset(torch.tensor([0]), states[0], cube[-1:, :2], torch.zeros(1, dtype=torch.float64))
            res = [task.step(s, m[None], goal) for s, m in zip(states[1:k + 1], macro[1:k + 1])]
            self.assertEqual(bool(res[-1][3]["flags"][0, T.RULES.index("grasp_box")]), fires, (dz, dyaw))

    def test_keep_out_and_lowest_point(self):
        kin = self.kin
        c = torch.tensor([[0.22, 0.05, T.CUBE_REST_Z]], dtype=torch.float64)
        quat = torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64)
        for tcp_z, rule in ((0.03, "keep_out"), (0.004, "lowest_point")):
            pos = torch.tensor([[0.22, 0.05 + 0.035, tcp_z]], dtype=torch.float64)
            q5, ok, _, _ = kin.ik_topdown(None, pos, torch.zeros(1, dtype=torch.float64), max_tilt=math.radians(45))
            self.assertTrue(bool(ok.all()))
            q = torch.cat((q5, torch.full((1, 1), 0.7, dtype=torch.float64)), -1)
            st = make_state(q.repeat(2, 1), c.repeat(2, 1), quat.repeat(2, 1))
            task = self.task()
            task.reset(torch.tensor([0]), st[0], c[:, :2] + 0.1, torch.zeros(1, dtype=torch.float64))
            r = task.step(st[1], torch.zeros(1, dtype=torch.long))
            self.assertEqual(T.ENDS[int(r[3]["end"][0])], rule)
            self.assertTrue(bool(r[1][0]))

    def test_hold_cap_and_charge_remaining(self):
        ep, states, macro, cube = self.episode()
        k = int(torch.nonzero(macro == T.MC["free"])[5])                           # a held carry state
        st = states[k]
        cfg = T.TaskConfig(time_cost=0.01, gamma=0.9, fail_cost=2.0)
        task = self.task(cfg)
        task.reset(torch.tensor([0]), st, cube[-1:, :2], torch.zeros(1, dtype=torch.float64))
        st0 = WorldState(**{**st.__dict__, "qd": st.qd * 0, "cube_v": st.cube_v * 0, "cube_w": st.cube_w * 0})
        n_cap = int(round(cfg.hold_cap_s / 0.02))
        for i in range(n_cap + 2):
            rew, term, trunc, info = task.step(st0, torch.zeros(1, dtype=torch.long))
            if bool(term[0]):
                break
        self.assertEqual(T.ENDS[int(info["end"][0])], "hold_cap")
        self.assertEqual(i, n_cap)                                                  # > 3.5 s: the 176th held step
        n_left = cfg.horizon - (i + 1)
        charge = cfg.time_cost * (1 - cfg.gamma ** n_left) / (1 - cfg.gamma)
        self.assertAlmostEqual(float(rew[0]), -cfg.time_cost - cfg.fail_cost - charge, places=9)

    def test_observation(self):
        ep, states, macro, cube = self.episode()
        task = self.task()
        task.reset(torch.tensor([0]), states[0], cube[-1:, :2], torch.zeros(1, dtype=torch.float64))
        dec = torch.zeros(1, 11)
        o = task.observation(states[3], dec, macro[3:4])
        self.assertEqual(o.shape, (1, T.obs_dim(task.cfg, 11)))
        self.assertEqual(o.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(o).all()))
        cfg = T.TaskConfig(obs_rates=False)
        self.assertEqual(T.obs_dim(cfg), T.obs_dim(T.TaskConfig()) - 6)

    @unittest.skipIf(not torch.cuda.is_available(), "no CUDA")
    def test_cpu_cuda_same(self):
        ep, states, macro, cube = self.episode()
        out = {}
        for dev in ("cpu", "cuda"):
            task = self.task(device=dev)
            gyaw, _ = T.cube_yaw_tilt(states[-1].cube_quat)
            task.reset(torch.tensor([0], device=dev), states[0].to(dev), cube[-1:, :2].to(dev), gyaw.to(dev))
            res = [task.step(s.to(dev), m[None].to(dev)) for s, m in zip(states[1:], macro[1:])]
            out[dev] = (torch.stack([r[0][0].cpu() for r in res]), [int(r[3]["end"][0]) for r in res])
        self.assertEqual(out["cpu"][1], out["cuda"][1])
        self.assertLess(float((out["cpu"][0] - out["cuda"][0]).abs().max()), 1e-9)


if __name__ == "__main__":
    unittest.main()
