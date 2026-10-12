"""Tests of the SO101 scene for the batch renderer (nedm.render.so101).

CPU tests (torch + mujoco, a few seconds): forward kinematics and the wrist camera against MuJoCo, NRD-layout and
MuJoCo-qpos states give the same transforms (and the same as MuJoCo's own body poses), goal marker conventions, front
and top camera framing, camera selection, asset cache round trip. GPU tests (one renderer with the three cameras,
Madrona if NEDM_MADRONA_BUILD is set, else Newton/Warp on CUDA; override with NEDM_SO101_RENDER_BACKEND=
madrona|newton|none): the same state from both layouts and from a WorldState renders identically; depth against MuJoCo's
own renderer (EGL) for 8 states and against the analytic table plane; Newton only: 224 px with a camera subset.

    PYTHONPATH=src python -m unittest tests/render/test_so101.py -v              # CPU part: env nedm
    # GPU, Newton (luffy): ~/miniconda3/envs/so101_render/bin/python (warp 1.13 + newton 1.2, torch 2.12 cu130)
    # GPU, Madrona (luffy): source ~/so101_policy/runs/render/madrona_luffy/env_luffy.sh, same python; if the run
    # aborts in Madrona's engine start (CUDA_ERROR_OUT_OF_MEMORY / LAUNCH_OUT_OF_RESOURCES), the GPU is full: rerun
"""
import importlib.util
import math
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")     # before mujoco is imported anywhere: the depth test renders headless

import numpy as np
import torch

HAVE_MUJOCO = importlib.util.find_spec("mujoco") is not None
VAL_DATA = Path(os.environ.get("NEDM_SO101_VAL", Path.home() / "so101_nrd_fix/data/val_only/unified_data.npz"))


def _gpu_backend() -> str | None:
    pick = os.environ.get("NEDM_SO101_RENDER_BACKEND")
    if pick == "none" or not torch.cuda.is_available():
        return None
    if pick:
        return pick
    if os.environ.get("NEDM_MADRONA_BUILD"):
        return "madrona"
    if importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None:
        return "newton"
    return None


GPU_BACKEND = _gpu_backend()
_ASSETS = None
_VAL = None


def assets():
    global _ASSETS
    if _ASSETS is None:
        from nedm.render.so101 import SO101Assets
        _ASSETS = SO101Assets.from_mjcf()
    return _ASSETS


def collector_scene():
    from nedm.contact_nrd.collect import so101_grasp
    return so101_grasp.Scene(so101_grasp.load_config())


def recorded_states(k: int, seed: int = 0) -> np.ndarray:
    """k states [k, 2, 13] (arm q, qd; cube pos, quat, v, w) from validation episodes (split 1), else synthetic."""
    rng = np.random.default_rng(seed)
    if VAL_DATA.exists():
        global _VAL
        if _VAL is None:
            z = np.load(VAL_DATA)
            _VAL = z["lengths"], z["arm"], z["cube"]
        lengths, arm, cube = _VAL
        ep = rng.choice(len(lengths), k, replace=False)
        t = (rng.random(k) * lengths[ep]).astype(int)
        x = np.zeros((k, 2, 13), np.float32)
        x[:, 0, :12], x[:, 1] = arm[ep, t], cube[ep, t]
        return x
    x = np.zeros((k, 2, 13), np.float32)
    x[:, 0, :6] = np.array([0.0, -0.3, 0.6, 1.0, 0.0, 0.6]) + rng.uniform(-0.4, 0.4, (k, 6))
    yaw = rng.uniform(-math.pi, math.pi, k)
    x[:, 1, 0] = rng.uniform(0.15, 0.3, k)
    x[:, 1, 1] = rng.uniform(-0.2, 0.2, k)
    x[:, 1, 2] = 0.015
    x[:, 1, 3], x[:, 1, 6] = np.cos(yaw / 2), np.sin(yaw / 2)
    return x


def project(cam: torch.Tensor, p: torch.Tensor, fovy_deg: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Image coordinates (u right, v up; +-1 = the image edges) of world points p [N, 3] in camera cam (7,) x y z,
    qx qy qz qw (looks along -z, +y up), square image."""
    from nedm.render.poses import quat_rotate
    cam = cam.double().cpu()
    local = quat_rotate(torch.cat([-cam[3:6], cam[6:]]).expand(len(p), 4), p.double() - cam[:3])
    tan = math.tan(math.radians(fovy_deg) / 2)
    return local[:, 0] / -local[:, 2] / tan, local[:, 1] / -local[:, 2] / tan


def workspace_points(z_top: float) -> torch.Tensor:
    """The annulus (r 0.15-0.27 m around the pan axis (0.0388, 0), +-75 deg) widened by the cube's half diagonal
    (21 mm), at z = 0 and z = z_top (m)."""
    r, a = np.meshgrid(np.linspace(0.15, 0.27, 7), np.radians(np.linspace(-75, 75, 31)))
    pts = [np.stack([0.0388 + r.ravel() * np.cos(a.ravel()) + dx, r.ravel() * np.sin(a.ravel()) + dy,
                     np.full(r.size, z)], 1) for z in (0.0, z_top) for dx in (-0.021, 0.021) for dy in (-0.021, 0.021)]
    return torch.tensor(np.concatenate(pts), dtype=torch.float64)


def table_plane_depth(cam: torch.Tensor, h: int, fovy_deg: float) -> np.ndarray:
    """Analytic z-depth [h, h] of the table plane z = 0 at pixel centres for camera cam (7,) (inf: misses it)."""
    import mujoco
    c = cam.double().cpu().numpy()
    f = (h / 2) / math.tan(math.radians(fovy_deg) / 2)
    uu = (np.arange(h) + 0.5 - h / 2) / f
    X, Y = np.meshgrid(uu, -uu)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.r_[c[6], c[3:6]])
    dirs = np.stack([X, Y, -np.ones_like(X)], -1) @ rot.reshape(3, 3).T
    t = -c[2] / dirs[..., 2]              # camera -z component of dirs is 1, so t is the z-depth
    return np.where(t > 0, t, np.inf)


def quat_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation angle (rad) between unit quaternions (..., 4) w x y z (atan2 form: exact for small angles)."""
    aw, av = a[..., :1], -a[..., 1:]                    # conjugate of a
    bw, bv = b[..., :1], b[..., 1:]
    w = aw * bw - (av * bv).sum(-1, keepdims=True)
    v = aw * bv + bw * av + np.cross(av, bv)
    return 2 * np.arctan2(np.linalg.norm(v, axis=-1), np.abs(w[..., 0]))


@unittest.skipUnless(HAVE_MUJOCO, "needs mujoco")
class KinematicsAgainstMuJoCo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mujoco
        from nedm.render.so101 import SO101Kinematics, SO101Scene
        cls.mj = mujoco
        cls.s = collector_scene()
        cls.kin = SO101Kinematics(assets().links)
        cls.scene = SO101Scene(assets())

    def test_body_poses_match_mujoco(self):
        mj, s = self.mj, self.s
        rng = np.random.default_rng(0)
        q = rng.uniform(s.jlo, s.jhi, (256, 6))
        pos, quat = self.kin(torch.tensor(q))
        ids = [s.m.body(name).id for name in self.kin.bodies]
        perr = aerr = 0.0
        for k in range(len(q)):
            s.dk.qpos[s.qadr] = q[k]
            mj.mj_kinematics(s.m, s.dk)
            perr = max(perr, float(np.abs(pos[k].numpy() - s.dk.xpos[ids]).max()))
            aerr = max(aerr, float(quat_angle(quat[k].numpy(), s.dk.xquat[ids]).max()))
        self.assertLess(perr, 1e-12)
        self.assertLess(aerr, 1e-12)

    def test_wrist_camera_matches_mujoco(self):
        mj, s = self.mj, self.s
        rng = np.random.default_rng(1)
        q = rng.uniform(s.jlo, s.jhi, (64, 6))
        cube = torch.tensor([[0.2, 0.0, 0.015]]).expand(64, 3)
        _, cam = self.scene.transforms(torch.tensor(q), cube, torch.tensor([[1.0, 0, 0, 0]]).expand(64, 4))
        c = s.m.camera("wrist_cam").id
        for k in range(len(q)):
            s.dk.qpos[s.qadr] = q[k]
            mj.mj_kinematics(s.m, s.dk)
            mj.mj_camlight(s.m, s.dk)
            r = np.zeros(4)
            mj.mju_mat2Quat(r, s.dk.cam_xmat[c])
            w = cam[k, 1].double().numpy()
            self.assertLess(np.abs(w[:3] - s.dk.cam_xpos[c]).max(), 1e-7)     # float32 output
            self.assertLess(quat_angle(np.r_[w[6], w[3:6]], r), 1e-6)     # float32 output

    def test_nrd_and_qpos_layouts_give_the_same_poses(self):
        """The data layout [B, 2, S] and the collector's MuJoCo qpos of the same states give bit-identical transforms,
        and the arm meshes and the cube sit at MuJoCo's own body poses."""
        from nedm.render.so101 import poses_from_nrd, poses_from_qpos
        mj, s = self.mj, self.s
        x = recorded_states(64)
        qpos = np.zeros((len(x), s.m.nq))
        xpos, xquat = [], []
        names = [n.split("#")[0] for n in self.scene.body_names[: len(self.scene.mesh_body)]]
        ids = [s.m.body(n).id for n in names] + [s.cube_body]
        for k in range(len(x)):
            s.d.qpos[:] = s.m.qpos0
            s.d.qpos[s.qadr] = x[k, 0, :6]
            s.d.qpos[s.cq: s.cq + 7] = x[k, 1, :7]
            mj.mj_kinematics(s.m, s.d)
            qpos[k] = s.d.qpos
            xpos.append(s.d.xpos[ids].copy())
            xquat.append(s.d.xquat[ids].copy())
        a = self.scene.transforms(*poses_from_nrd(torch.tensor(x)))
        b = self.scene.transforms(*poses_from_qpos(torch.tensor(qpos), s.qadr, s.cq))
        self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))
        body = a[0][:, : len(ids)].double().numpy()
        self.assertLess(np.abs(body[..., :3] - np.array(xpos)).max(), 1e-7)
        self.assertLess(quat_angle(np.concatenate([body[..., 6:], body[..., 3:6]], -1), np.array(xquat)).max(), 2e-6)


class SceneConventions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HAVE_MUJOCO:
            raise unittest.SkipTest("asset export needs mujoco")
        from nedm.render.so101 import SO101Scene
        cls.scene = SO101Scene(assets())

    def test_goal_xy_yaw_equals_goal_cube_pose(self):
        """A [B, 7] goal (flat cube pose, any face up) and the [B, 3] goal (x, y, yaw) give the same marker (yaw mod
        90 deg)."""
        from nedm.render.so101 import qmul, yaw_quat
        n = 24
        g = torch.rand(n, 3, dtype=torch.float64)
        g[:, 2] = (g[:, 2] - 0.5) * 2 * math.pi
        # face-up variants: turn the cube by 90 deg about x or y first (the cube symmetry), then by the yaw
        h = 0.5 ** 0.5
        tips = torch.tensor([[1, 0, 0, 0], [h, h, 0, 0], [h, 0, h, 0], [0, 1, 0, 0]], dtype=torch.float64)
        quat = qmul(yaw_quat(g[:, 2]), tips[torch.arange(n) % 4])
        pose7 = torch.cat([g[:, :2], torch.full((n, 1), 0.015, dtype=torch.float64), quat], -1)
        a, b = self.scene.goal_transform(g, n), self.scene.goal_transform(pose7, n)
        self.assertLess((a[:, :3] - b[:, :3]).abs().max().item(), 1e-12)
        yaw = lambda t: 2 * torch.atan2(t[:, 5], t[:, 6])                                         # noqa: E731
        d = torch.remainder(yaw(a) - yaw(b) + math.pi / 4, math.pi / 2) - math.pi / 4
        self.assertLess(d.abs().max().item(), 1e-9)

    def test_hidden_goal(self):
        g = torch.tensor([[0.2, 0.1, 0.3], [float("nan")] * 3])
        t = self.scene.goal_transform(g, 2)
        self.assertEqual(t[0, 2].item(), 0.0)
        self.assertLess(t[1, 2].item(), -10)
        self.assertTrue((self.scene.goal_transform(None, 3)[:, 2] < -10).all())

    def test_front_camera_frames_the_workspace(self):
        """Every point of the annulus widened by the cube's half diagonal, from the table to 0.18 m up, projects
        inside the front image (0.95 of the half width)."""
        u, v = project(self.scene.front, workspace_points(0.18), self.scene.fovy_deg)
        self.assertLess(max(u.abs().max().item(), v.abs().max().item()), 0.95)

    def test_top_camera_looks_down_and_frames_the_workspace(self):
        """The top camera looks straight down, image up = world +x, image right = world -y; the widened annulus from
        the table to 0.15 m up projects inside 0.96 of the half width; the arm base (x = y = 0) is in the image."""
        from nedm.render.poses import quat_rotate
        cam = self.scene.top.double()
        axes = quat_rotate(cam[3:].expand(3, 4), torch.eye(3, dtype=torch.float64))       # camera x, y, z in world
        want = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64)
        self.assertLess((axes - want).abs().max().item(), 1e-6)                          # float32 transform
        u, v = project(cam, workspace_points(0.15), self.scene.fovy_deg)
        self.assertLess(max(u.abs().max().item(), v.abs().max().item()), 0.96)
        u, v = project(cam, torch.zeros(1, 3, dtype=torch.float64), self.scene.fovy_deg)
        self.assertLess(max(abs(u.item()), abs(v.item())), 0.9)

    def test_camera_selection(self):
        """Any distinct subset of VIEWS, in the given order; the wrist transform does not depend on the set."""
        from nedm.render.so101 import SO101Scene
        x = torch.tensor(recorded_states(4, seed=2))
        args = (x[:, 0, :6], x[:, 1, :3], x[:, 1, 3:7])
        _, ref = self.scene.transforms(*args)                                         # default: front, wrist
        three = SO101Scene(assets(), cameras=("top", "wrist", "front"))
        _, cam = three.transforms(*args)
        self.assertEqual(tuple(cam.shape), (4, 3, 7))
        self.assertTrue(torch.equal(cam[:, 1], ref[:, 1]) and torch.equal(cam[:, 2], ref[:, 0]))
        self.assertTrue(torch.equal(cam[:, 0], three.top.float().expand(4, 7)))
        for bad in ((), ("front", "front"), ("side",)):
            with self.assertRaises(ValueError):
                SO101Scene(assets(), cameras=bad)

    def test_face_colours_make_six_meshes(self):
        from nedm.render.so101 import FACE_COLORS, SO101Scene
        sc = SO101Scene(assets(), cube_colors=FACE_COLORS, goal="frame")
        self.assertEqual(sc.cube_meshes, 6)
        self.assertEqual(sum(len(m.faces) for m in sc.scene.bodies if m.name.startswith("cube")), 12)
        b, _ = sc.transforms(torch.zeros(2, 6), torch.zeros(2, 3), torch.tensor([[1.0, 0, 0, 0]] * 2))
        self.assertEqual(b.shape[1], len(sc.body_names))

    def test_asset_cache_round_trip(self):
        from nedm.render.so101 import SO101Assets
        with tempfile.TemporaryDirectory() as d:
            assets().save(Path(d) / "a.npz")
            b = SO101Assets.load(Path(d) / "a.npz")
        self.assertEqual(b.links, assets().links)
        self.assertEqual(b.wrist_cam, assets().wrist_cam)
        for (ba, ma), (bb, mb) in zip(assets().meshes, b.meshes):
            self.assertEqual((ba, ma.name, ma.color), (bb, mb.name, mb.color))
            self.assertTrue(np.array_equal(ma.vertices, mb.vertices) and np.array_equal(ma.faces, mb.faces))


@unittest.skipUnless(GPU_BACKEND and HAVE_MUJOCO, "needs a GPU render backend and mujoco")
class RenderOnGpu(unittest.TestCase):
    """One renderer for the class (Madrona allows one per process), with all three cameras."""

    K, RES = 8, 128
    CAMS = ("front", "wrist", "top")

    @classmethod
    def setUpClass(cls):
        from nedm.render.so101 import SO101Renderer
        cls.r = SO101Renderer(2 * cls.K, cls.RES, cls.RES, backend=GPU_BACKEND, assets=assets(), cameras=cls.CAMS)

    def test_same_state_from_nrd_qpos_and_worldstate_renders_identically(self):
        """The data layout, the collector's MuJoCo qpos and a WorldState (what the NRD and MuJoCo worlds return) of
        the same states give bit-identical pictures."""
        from nedm.render.so101 import poses_from_nrd, poses_from_qpos
        from nedm.so101_policy.world import WorldState
        s = collector_scene()
        x = recorded_states(self.K, seed=3)
        qpos = np.tile(s.m.qpos0, (self.K, 1))
        qpos[:, s.qadr] = x[:, 0, :6]
        qpos[:, s.cq: s.cq + 7] = x[:, 1, :7]
        goal = torch.tensor([[0.2, 0.05, 0.3]]).expand(self.K, 3)
        a, da = (t.clone() for t in self.r.render(*poses_from_nrd(torch.tensor(x)), goal, depth=True))
        b, db = self.r.render(*poses_from_qpos(torch.tensor(qpos), s.qadr, s.cq), goal, depth=True)
        self.assertEqual(tuple(a.shape), (self.K, len(self.CAMS), self.RES, self.RES, 3))
        self.assertEqual(a.dtype, torch.uint8)
        self.assertTrue(torch.equal(a, b) and torch.equal(da, db))
        xt = torch.tensor(x, dtype=torch.float64)
        k = self.K
        ws = WorldState(q=xt[:, 0, :6], qd=xt[:, 0, 6:12], cube_pos=xt[:, 1, :3], cube_quat=xt[:, 1, 3:7],
                        cube_v=xt[:, 1, 7:10], cube_w=xt[:, 1, 10:13], contact=torch.zeros(k, 3, dtype=torch.bool),
                        arm_table=torch.zeros(k, dtype=torch.bool), invalid=torch.zeros(k, dtype=torch.bool))
        self.assertTrue(torch.equal(self.r.render_state(ws.to("cuda:0"), goal), a))
        for c in range(len(self.CAMS)):
            self.assertGreater(a[:, c].float().std().item(), 10.0)       # a picture, not a blank

    def test_depth_matches_mujoco(self):
        """z-depth of 8 states against MuJoCo's own renderer (same cameras, near/far tightened), and of the table
        plane against the analytic depth at pixel centres (front and top)."""
        import mujoco
        from nedm.contact_nrd.collect import so101_grasp
        from nedm.render.so101 import GOAL_COLOR, GOAL_HALF, GOAL_THICKNESS, poses_from_nrd
        K, H = self.K, self.RES
        cfg = so101_grasp.load_config()
        spec = so101_grasp.build_spec(cfg)
        for name in ("front", "top"):
            c = getattr(self.r.scene, name).double().cpu().numpy()            # x y z qx qy qz qw
            spec.worldbody.add_camera(name=name, pos=c[:3].tolist(), quat=[c[6], *c[3:6]], fovy=self.r.scene.fovy_deg)
        marker = spec.worldbody.add_body(name="goal_marker", mocap=True)     # world geoms do not move after compile
        marker.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, contype=0, conaffinity=0,
                        size=[GOAL_HALF, GOAL_HALF, GOAL_THICKNESS / 2], rgba=[*GOAL_COLOR, 1])
        m = spec.compile()
        d = mujoco.MjData(m)
        m.vis.map.znear, m.vis.map.zfar = 0.02 / m.stat.extent, 4.0 / m.stat.extent     # 24-bit depth buffer
        m.geom_rgba[m.geom("wrist_camera_lens").id, 3] = 0.0
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[0] = opt.geomgroup[2] = 1          # table, cube, goal; arm visuals (lens hidden above)
        x = recorded_states(K, seed=5)
        rng = np.random.default_rng(5)
        goal = np.stack([rng.uniform(0.12, 0.3, K), rng.uniform(-0.2, 0.2, K), rng.uniform(-1, 1, K)], 1)
        mid = m.body("goal_marker").mocapid[0]
        C = len(self.CAMS)
        ref = np.zeros((K, C, H, H), np.float32)
        ref_rgb = np.zeros((K, C, H, H, 3), np.uint8)
        rend = mujoco.Renderer(m, H, H)
        for k in range(K):
            d.qpos[:] = m.qpos0
            d.qpos[:6] = x[k, 0, :6]
            d.qpos[6:13] = x[k, 1, :7]
            d.mocap_pos[mid] = [goal[k, 0], goal[k, 1], GOAL_THICKNESS / 2]
            d.mocap_quat[mid] = [math.cos(goal[k, 2] / 2), 0, 0, math.sin(goal[k, 2] / 2)]
            mujoco.mj_forward(m, d)
            for c, cam in enumerate("wrist_cam" if n == "wrist" else n for n in self.CAMS):
                rend.update_scene(d, camera=cam, scene_option=opt)
                ref_rgb[k, c] = rend.render()
                rend.enable_depth_rendering()
                rend.update_scene(d, camera=cam, scene_option=opt)
                ref[k, c] = rend.render()
                rend.disable_depth_rendering()
        rend.close()
        rgb, depth = self.r.render(*poses_from_nrd(torch.tensor(x)), torch.tensor(goal), depth=True)
        rgb, depth = rgb.cpu().numpy().astype(int), depth.cpu().numpy()
        # the goal marker covers the same pixels (green-dominant in both shadings)
        green = lambda im: (im[..., 1] > im[..., 0] + 30) & (im[..., 1] > im[..., 2] + 30)          # noqa: E731
        ours, theirs = green(rgb), green(ref_rgb.astype(int))
        for c in range(C):
            if theirs[:, c].sum() >= 20:
                iou = (ours[:, c] & theirs[:, c]).sum() / (ours[:, c] | theirs[:, c]).sum()
                self.assertGreater(iou, 0.6, f"goal marker, camera {self.CAMS[c]}")
        # depth tolerance (p50, p99 mm): far cameras get MuJoCo's coarse 24-bit depth buffer (table plane: ours 1e-4 mm)
        tol = {"front": (3.0, 12.0), "top": (3.0, 12.0), "wrist": (0.5, 4.0)}
        for c, name in enumerate(self.CAMS):
            hit = (depth[:, c] > 0) & (ref[:, c] < 3.9)
            err = np.abs(depth[:, c] - ref[:, c])[hit] * 1e3
            self.assertGreater(hit.mean(), 0.5)
            self.assertLess(np.percentile(err, 50), tol[name][0], name)
            self.assertLess(np.percentile(err, 99), tol[name][1], name)
            self.assertLess(((depth[:, c] > 0) != (ref[:, c] < 3.9)).mean(), 1e-3, name)
        # analytic: the table plane z = 0 at pixel centres, front and top
        for name in ("front", "top"):
            c = self.CAMS.index(name)
            t = table_plane_depth(getattr(self.r.scene, name), H, self.r.scene.fovy_deg)[None]
            table = (depth[:, c] > 0) & (np.abs(depth[:, c] - t) < 0.002)
            self.assertGreater(table.mean(), 0.2, name)
            self.assertLess(np.percentile(np.abs(depth[:, c] - t)[table], 50) * 1e3, 0.01, name)

    def test_set_appearance_changes_colours_and_resets(self):
        """EXPERIMENTAL per-world colours: every world changes, depth does not, and reset restores the pictures."""
        from nedm.render.so101 import poses_from_nrd
        x = torch.tensor(recorded_states(self.K, seed=7))
        goal = torch.tensor([[0.2, -0.05, 0.1]]).expand(self.K, 3)
        a, da = (t.clone() for t in self.r.render(*poses_from_nrd(x), goal, depth=True))
        g = torch.Generator().manual_seed(0)
        self.r.set_appearance(0.2 + 0.7 * torch.rand(2 * self.K, len(self.r.geom_names), 3, generator=g))
        b, db = self.r.render(*poses_from_nrd(x), goal, depth=True)
        self.assertTrue(torch.equal(da, db))
        self.assertTrue(bool((a != b).flatten(1).any(1).all()))
        self.r.set_appearance(reset=True)
        self.assertTrue(torch.equal(self.r.render(*poses_from_nrd(x), goal), a))

    @unittest.skipUnless(GPU_BACKEND == "newton", "a second renderer in one process: Newton only")
    def test_224_with_top_and_wrist(self):
        """ACT-size images (224 x 224) from a top + wrist renderer: shape, the table plane of the top camera, and the
        wrist view agrees with the 128 px renderer (z-depth medians within 1 mm)."""
        from nedm.render.so101 import SO101Renderer, poses_from_nrd
        r = SO101Renderer(self.K, 224, 224, backend="newton", assets=assets(), cameras=("top", "wrist"))
        x = torch.tensor(recorded_states(self.K, seed=11))
        goal = torch.tensor([[0.22, -0.08, 0.2]]).expand(self.K, 3)
        rgb, depth = r.render(*poses_from_nrd(x), goal, depth=True)
        self.assertEqual(tuple(rgb.shape), (self.K, 2, 224, 224, 3))
        depth = depth.cpu().numpy()
        t = table_plane_depth(r.scene.top, 224, r.scene.fovy_deg)[None]
        table = (depth[:, 0] > 0) & (np.abs(depth[:, 0] - t) < 0.002)
        self.assertGreater(table.mean(), 0.5)
        self.assertLess(np.percentile(np.abs(depth[:, 0] - t)[table], 50) * 1e3, 0.01)
        _, d128 = self.r.render(*poses_from_nrd(x), goal, depth=True)
        w128 = d128[:, self.CAMS.index("wrist")].cpu().numpy()
        for k in range(self.K):
            a, b = depth[k, 1][depth[k, 1] > 0], w128[k][w128[k] > 0]
            self.assertLess(abs(np.median(a) - np.median(b)) * 1e3, 1.0)


if __name__ == "__main__":
    unittest.main()
