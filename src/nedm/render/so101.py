"""SO101 + cube + goal scene for the batch renderer: the arm of the grasp NRD study, drawn from the visual meshes of the
pinned Menagerie MJCF, the collector's table and 30 mm cube, a goal marker, a fixed front camera and the MJCF wrist
camera.

    r = SO101Renderer(num_worlds=1024, backend="madrona")     # newton: env kyle_madrona, source madrona/env.sh
    rgb = r.render(q, cube_pos, cube_quat_wxyz, goal)          # [B, 2, 128, 128, 3] uint8, cameras in CAMERAS order
    rgb, depth = r.render(..., depth=True)                     # depth [B, 2, H, W] float32: z-depth (m), 0 = no hit

Frames and units: world z up, table top at z = 0, arm base frame = world frame (as in the collector's scene,
``nedm.contact_nrd.collect.so101_grasp.build_spec``). Joint angles in rad, in ``so101.xml`` order (jaw = joint 5),
as in the NRD data. Quaternions given to and returned by this module are w x y z (NRD, MuJoCo); the renderer's own
transforms are x y z w and are converted here. Depth is MuJoCo's convention (distance along the camera axis), not the
renderer's ray length.

The same function draws NRD states (:func:`poses_from_nrd`) and MuJoCo states (:func:`poses_from_qpos`); identical
states give identical pictures. With distinct cube face colours, give the TRUE cube orientation (``WorldState``), not
the NRD's canonical relabelling; a one-colour cube looks the same under all 24 relabellings.

Mesh export (:meth:`SO101Assets.from_mjcf`) needs ``mujoco``; everything else needs torch and numpy only, and the
assets can be cached to one ``.npz`` (:meth:`SO101Assets.save` / :meth:`SO101Assets.load`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from nedm.render.scene import MeshBody, Scene

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/contact_nrd/collection/so101_grasp.json"
CAMERAS = ("front", "wrist")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# colours of the collector's MuJoCo scene (build_spec); the arm keeps its MJCF material colours
TABLE_COLOR = (0.55, 0.5, 0.45)
CUBE_COLOR = (0.15, 0.35, 0.85)
GOAL_COLOR = (0.2, 0.72, 0.3)
# optional distinct face colours, order +x, -x, +y, -y, +z, -z of the cube frame
FACE_COLORS = ((0.85, 0.15, 0.15), (0.95, 0.5, 0.1), (0.15, 0.35, 0.85), (0.92, 0.92, 0.92), (0.75, 0.2, 0.75),
               (0.1, 0.75, 0.8))
SKY = (200, 215, 230)
# Front camera: across the table, facing the arm, 45 deg down. With the shared 61 deg field of view it frames the
# whole workspace annulus (r 0.15-0.27 m around the pan axis, +-75 deg) from the table up to z = 0.18 m, with the
# cube's half diagonal as margin (worst point at 0.9 of the half width), and the arm base.
FRONT_EYE = (0.616, 0.0, 0.456)
FRONT_TARGET = (0.22, 0.0, 0.06)
GOAL_HALF = 0.023          # goal plate half side (m): a 30 mm cube on the goal leaves 8 mm of plate on every side
GOAL_THICKNESS = 0.001     # m; the plate lies on the table top, so a cube resting on it sinks 1 mm into it (hidden)
GOAL_FRAME_WIDTH = 0.006   # m, the "frame" style: an outline of this width, outer half side GOAL_HALF
HIDDEN_Z = -100.0          # z of a hidden goal marker (far below the table)
BOX_FACES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32)
# BOX_FACES rows of each cube face, order +x, -x, +y, -y, +z, -z
FACE_ROWS = ((2, 3), (0, 1), (6, 7), (4, 5), (10, 11), (8, 9))


def box_mesh(half, center=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Box with half sizes ``half`` (3,) around ``center``: vertices (8, 3) float32, faces (12, 3) int32."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    return (corners * np.asarray(half, float) + np.asarray(center, float)).astype(np.float32), BOX_FACES.copy()


def load_config(path: str | Path = CONFIG) -> dict:
    return json.loads(Path(path).read_text())


def vendor_xml(cfg: dict | None = None) -> Path:
    """Path of the pinned ``so101.xml`` the collector loads (same rule as ``so101_grasp.vendor_model``)."""
    cfg = cfg or load_config()
    mpath = REPO_ROOT / cfg["source_manifest"]
    manifest = json.loads(mpath.read_text())
    path = mpath.parent / manifest["source"]["local_path"] / manifest["model"]["path"]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run python -I contact_nrd/so101_grasp/fetch_sources.py")
    return path


# ---- quaternions (w x y z), batched, any float dtype ----------------------------------------------------------------
def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def qrot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` (..., 3) by unit quaternions ``q`` (..., 4) w x y z."""
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 1:], q[..., :1]], -1)


def wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 3:], q[..., :3]], -1)


def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Quaternion w x y z of a turn by ``yaw`` (rad) about world z."""
    z = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)], -1)


def flat_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Heading (rad) of a box orientation (..., 4) w x y z: the angle of the box axis that is closest to horizontal
    among the two that are not closest to vertical. For a box lying flat on any face this is its yaw mod 90 deg."""
    w, x, y, z = quat.unbind(-1)
    # columns of the rotation matrix = box axes in the world
    cols = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1),
        torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1),
        torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)], -2)   # (..., 3, 3)
    up = cols[..., 2].abs().argmax(-1)                                                  # axis closest to vertical
    pick = torch.where(up == 0, 1, 0)                                                   # first remaining axis
    axis = torch.gather(cols, -2, pick[..., None, None].expand(*pick.shape, 1, 3)).squeeze(-2)
    return torch.atan2(axis[..., 1], axis[..., 0])


# ---- assets -----------------------------------------------------------------------------------------------------
@dataclass
class Link:
    """One body of the MJCF tree. ``pos``/``quat`` (w x y z) in the parent frame; a hinge about ``axis`` at the body
    origin turns by joint ``qadr`` (None: welded to the parent)."""

    name: str
    parent: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    qadr: int | None = None
    axis: tuple[float, float, float] | None = None


@dataclass
class SO101Assets:
    """What the renderer needs from the MJCF: the body tree, one mesh per (body, colour) in its body frame, and the
    wrist camera (``body``, ``pos``, ``quat`` w x y z in the body frame, ``fovy_deg``)."""

    links: list[Link]
    meshes: list[tuple[str, MeshBody]]
    wrist_cam: dict
    source: str = ""

    @classmethod
    def from_mjcf(cls, xml: str | Path | None = None) -> "SO101Assets":
        """Visual geoms (groups 1-2) of ``so101.xml`` (default: the collector's pinned copy). The wrist camera lens is
        left out: the camera sits inside it and a ray tracer has no near plane. The camera board (pcb) stays: it is
        behind the camera (+9.5 mm along the camera's z), so only the front camera sees it."""
        import mujoco
        xml = Path(xml) if xml is not None else vendor_xml()
        m = mujoco.MjModel.from_xml_path(str(xml))
        name = lambda kind, i: mujoco.mj_id2name(m, kind, i)                                    # noqa: E731
        bodies = [name(mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
        groups: dict[tuple[str, tuple], list] = {}
        for g in range(m.ngeom):
            if m.geom_group[g] not in (1, 2) or name(mujoco.mjtObj.mjOBJ_GEOM, g) == "wrist_camera_lens":
                continue
            rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
            r = np.zeros(9)
            mujoco.mju_quat2Mat(r, m.geom_quat[g])
            t = m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
                f = m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + m.mesh_facenum[mid]].astype(np.int64)
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                v, f = box_mesh(m.geom_size[g])
                v = v.astype(np.float64)
            else:
                raise ValueError(f"visual geom {g} of type {t} is not handled")
            key = (bodies[m.geom_bodyid[g]], tuple(float(c) for c in np.round(rgba[:3], 3)))
            groups.setdefault(key, []).append((v @ r.reshape(3, 3).T + m.geom_pos[g], f))
        meshes = []
        for (body, rgb), parts in sorted(groups.items()):
            offsets = np.cumsum([0] + [len(v) for v, _ in parts])[:-1]
            vertices = np.concatenate([v for v, _ in parts]).astype(np.float32)
            faces = np.concatenate([f + o for (_, f), o in zip(parts, offsets)]).astype(np.int32)
            meshes.append((body, MeshBody(f"{body}#{len(meshes)}", vertices=vertices, faces=faces, color=rgb)))
        links = []
        for b in range(1, m.nbody):
            jn = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b]
            if jn:
                if m.jnt_type[jn[0]] != mujoco.mjtJoint.mjJNT_HINGE or np.abs(m.jnt_pos[jn[0]]).max() > 0:
                    raise ValueError(f"body {bodies[b]}: only hinges at the body origin are handled")
            links.append(Link(bodies[b], bodies[m.body_parentid[b]], tuple(m.body_pos[b].tolist()),
                              tuple(m.body_quat[b].tolist()), int(m.jnt_qposadr[jn[0]]) if jn else None,
                              tuple(m.jnt_axis[jn[0]].tolist()) if jn else None))
        c = m.camera("wrist_cam").id
        cam = {"body": bodies[m.cam_bodyid[c]], "pos": m.cam_pos[c].tolist(), "quat": m.cam_quat[c].tolist(),
               "fovy_deg": float(m.cam_fovy[c])}
        return cls(links, meshes, cam, str(xml))

    def save(self, path: str | Path) -> None:
        arrays = {}
        for i, (_, mesh) in enumerate(self.meshes):
            arrays[f"v{i}"], arrays[f"f{i}"] = mesh.arrays()
        meta = {"links": [vars(link) for link in self.links], "wrist_cam": self.wrist_cam, "source": self.source,
                "meshes": [{"body": body, "name": mesh.name, "color": list(mesh.color)} for body, mesh in self.meshes]}
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "SO101Assets":
        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        links = [Link(**{**x, "pos": tuple(x["pos"]), "quat": tuple(x["quat"]),
                         "axis": tuple(x["axis"]) if x["axis"] is not None else None}) for x in meta["links"]]
        meshes = [(x["body"], MeshBody(x["name"], vertices=z[f"v{i}"], faces=z[f"f{i}"], color=tuple(x["color"])))
                  for i, x in enumerate(meta["meshes"])]
        return cls(links, meshes, meta["wrist_cam"], meta["source"])

    @property
    def triangles(self) -> int:
        return int(sum(len(mesh.faces) for _, mesh in self.meshes))


def load_assets(cache: str | Path | None = None, xml: str | Path | None = None) -> SO101Assets:
    """Assets from ``cache`` if it exists, else exported from the MJCF (and written to ``cache`` if given)."""
    if cache is not None and Path(cache).exists():
        return SO101Assets.load(cache)
    assets = SO101Assets.from_mjcf(xml)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        assets.save(cache)
    return assets


# ---- kinematics ---------------------------------------------------------------------------------------------------
class SO101Kinematics:
    """Batched torch forward kinematics of the MJCF tree (MuJoCo's rule: x_child = x_parent + R_parent pos,
    R_child = R_parent R(quat) R(axis, q)). ``bodies`` = the links in tree order (parents first)."""

    def __init__(self, links: Sequence[Link], device=None, dtype=torch.float64):
        self.links, self.device, self.dtype = list(links), torch.device(device or "cpu"), dtype
        self.bodies = [link.name for link in self.links]
        index = {name: i for i, name in enumerate(self.bodies)}
        self.parent = [index.get(link.parent, -1) for link in self.links]
        if any(p >= i for i, p in enumerate(self.parent)):
            raise ValueError("links must list parents before children")
        kw = dict(device=self.device, dtype=dtype)
        self.pos = torch.tensor([link.pos for link in self.links], **kw)
        self.quat = torch.tensor([link.quat for link in self.links], **kw)
        self.axis = torch.tensor([link.axis or (0.0, 0.0, 1.0) for link in self.links], **kw)

    def __call__(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q [B, 6] (rad) -> world positions [B, bodies, 3] (m) and orientations [B, bodies, 4] w x y z."""
        q = q.to(self.device, self.dtype)
        n = q.shape[0]
        pos, quat = [], []
        for i, link in enumerate(self.links):
            p, r = self.pos[i].expand(n, 3), self.quat[i].expand(n, 4)
            if self.parent[i] >= 0:
                pp, pq = pos[self.parent[i]], quat[self.parent[i]]
                p, r = pp + qrot(pq, p), qmul(pq, r)
            if link.qadr is not None:
                half = 0.5 * q[:, link.qadr: link.qadr + 1]
                r = qmul(r, torch.cat([torch.cos(half), torch.sin(half) * self.axis[i]], -1))
            pos.append(p)
            quat.append(r)
        return torch.stack(pos, 1), torch.stack(quat, 1)


# ---- state layouts ------------------------------------------------------------------------------------------------
def poses_from_nrd(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NRD / data state [..., 2, S] (body 0 = arm: q[0:6]; body 1 = cube: pos[0:3], quat w x y z[3:7]) ->
    (q [..., 6], cube_pos [..., 3], cube_quat [..., 4])."""
    return x[..., 0, 0:6], x[..., 1, 0:3], x[..., 1, 3:7]


def poses_from_qpos(qpos: torch.Tensor, qadr: Sequence[int] = range(6), cube_adr: int = 6):
    """MuJoCo qpos [..., nq] of the collector's scene (arm joints at ``qadr``, cube free joint at ``cube_adr``) ->
    (q [..., 6], cube_pos [..., 3], cube_quat [..., 4] w x y z)."""
    qpos = torch.as_tensor(qpos)
    return qpos[..., list(qadr)], qpos[..., cube_adr: cube_adr + 3], qpos[..., cube_adr + 3: cube_adr + 7]


# ---- scene ----------------------------------------------------------------------------------------------------------
class SO101Scene:
    """The renderer-independent part: the :class:`~nedm.render.scene.Scene` and the map from a state to the body and
    camera transforms (x y z w, float32) of every world.

    Bodies, in order: the moving arm meshes (one per body and colour), the cube (1 mesh, or 6 with ``cube_colors``
    of 6 colours), the goal marker (none with ``goal=None``). Static: the arm base meshes and the table.
    ``goal``: ``"plate"`` (a 1 mm square plate on the table, side 2 * GOAL_HALF) or ``"frame"`` (its outline).
    """

    def __init__(self, assets: SO101Assets | None = None, cube_colors=None, goal: str | None = "plate",
                 front_eye=FRONT_EYE, front_target=FRONT_TARGET, device=None, cfg: dict | None = None):
        from nedm.render import cameras
        self.assets = assets = assets if assets is not None else load_assets()
        cfg = cfg or load_config()
        self.device = torch.device(device or "cpu")
        self.kin = SO101Kinematics(assets.links, self.device)
        moving = self._moving_bodies()
        bodies, mesh_body = [], []
        for body, mesh in assets.meshes:
            if body in moving:
                bodies.append(mesh)
                mesh_body.append(self.kin.bodies.index(body))
        self.mesh_body = torch.tensor(mesh_body, device=self.device)
        # cube: the collector's 30 mm box, one colour or one colour per face
        self.cube_half = 0.5 * float(cfg["cube"]["size_m"])
        v, f = box_mesh([self.cube_half] * 3)
        colors = CUBE_COLOR if cube_colors is None else cube_colors
        if len(colors) == 6 and not isinstance(colors[0], (int, float)):
            for k, rows in enumerate(FACE_ROWS):
                ff = f[list(rows)]
                used, inv = np.unique(ff, return_inverse=True)
                bodies.append(MeshBody(f"cube_face{k}", vertices=v[used], faces=inv.reshape(-1, 3).astype(np.int32),
                                       color=tuple(colors[k])))
        else:
            bodies.append(MeshBody("cube", vertices=v, faces=f, color=tuple(colors)))
        self.cube_meshes = len(bodies) - len(mesh_body)
        self.goal_style = goal
        if goal == "plate":
            v, f = box_mesh([GOAL_HALF, GOAL_HALF, GOAL_THICKNESS / 2], (0.0, 0.0, GOAL_THICKNESS / 2))
            bodies.append(MeshBody("goal", vertices=v, faces=f, color=GOAL_COLOR))
        elif goal == "frame":
            h, w, t = GOAL_HALF, GOAL_FRAME_WIDTH, GOAL_THICKNESS
            parts = [box_mesh([h, w / 2, t / 2], (0.0, s * (h - w / 2), t / 2)) for s in (-1, 1)]
            parts += [box_mesh([w / 2, h - w, t / 2], (s * (h - w / 2), 0.0, t / 2)) for s in (-1, 1)]
            bodies.append(MeshBody("goal", vertices=np.concatenate([p[0] for p in parts]),
                                   faces=np.concatenate([p[1] + 8 * k for k, p in enumerate(parts)]), color=GOAL_COLOR))
        elif goal is not None:
            raise ValueError(f"goal style {goal!r}: 'plate', 'frame' or None")
        self.scene = Scene.from_meshes(bodies)
        # static: the base (welded to the world) and the table, as the collector builds them
        zero = torch.zeros(1, 6, dtype=torch.float64, device=self.device)
        bpos, bquat = self.kin(zero)
        for body, mesh in assets.meshes:
            if body not in moving:
                i = self.kin.bodies.index(body)
                self.scene.add_static_mesh(mesh, torch.cat([bpos[0, i], xyzw(bquat[0, i])]).tolist())
        t = cfg["table"]
        tv, tf = box_mesh([0.5 * t["size_xy_m"][0], 0.5 * t["size_xy_m"][1], 0.5 * t["thickness_m"]])
        self.scene.add_static_mesh(MeshBody("table", vertices=tv, faces=tf, color=TABLE_COLOR),
                                   (*t["center_xy_m"], -0.5 * t["thickness_m"], 0.0, 0.0, 0.0, 1.0))
        # cameras: front fixed; wrist = the MJCF camera on its body
        self.front = cameras.look_at(torch.tensor([front_eye], dtype=torch.float64),
                                     torch.tensor([front_target], dtype=torch.float64))[0].to(self.device)
        cam = assets.wrist_cam
        self.cam_body = self.kin.bodies.index(cam["body"])
        self.cam_pos = torch.tensor(cam["pos"], dtype=torch.float64, device=self.device)
        self.cam_quat = torch.tensor(cam["quat"], dtype=torch.float64, device=self.device)
        self.fovy_deg = float(cam["fovy_deg"])

    def _moving_bodies(self) -> set[str]:
        """Bodies with a hinge on the path to the world."""
        moving: set[str] = set()
        for link in self.assets.links:      # parents first
            if link.qadr is not None or link.parent in moving:
                moving.add(link.name)
        return moving

    @property
    def body_names(self) -> list[str]:
        return self.scene.body_names

    def transforms(self, q, cube_pos, cube_quat, goal_pose=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Body transforms [B, bodies, 7] and camera transforms [B, 2, 7] (x y z, qx qy qz qw; float32).

        q [B, 6] rad; cube_pos [B, 3] m; cube_quat [B, 4] w x y z (normalised here); goal_pose None (hidden),
        [B, 3] = goal cube centre x, y (m) and yaw (rad), or [B, 7] = goal cube pose (position, quaternion w x y z),
        of which x, y and the yaw of the face closest to the table (:func:`flat_yaw`) are used. The marker lies on the
        table top under that point. Rows with a non-finite goal are hidden."""
        f64 = dict(device=self.device, dtype=torch.float64)
        q = torch.as_tensor(q).to(**f64)
        n = q.shape[0]
        pos, quat = self.kin(q)
        parts = [torch.cat([pos[:, self.mesh_body], xyzw(quat[:, self.mesh_body])], -1)]
        cq = torch.as_tensor(cube_quat).to(**f64)
        cube = torch.cat([torch.as_tensor(cube_pos).to(**f64), xyzw(cq / cq.norm(dim=-1, keepdim=True))], -1)
        parts.append(cube[:, None].expand(n, self.cube_meshes, 7))
        if self.goal_style is not None:
            parts.append(self.goal_transform(goal_pose, n)[:, None])
        body_q = torch.cat(parts, 1)
        # wrist camera: body pose composed with the MJCF camera pose
        bp, bq = pos[:, self.cam_body], quat[:, self.cam_body]
        wrist = torch.cat([bp + qrot(bq, self.cam_pos.expand(n, 3)), xyzw(qmul(bq, self.cam_quat.expand(n, 4)))], -1)
        cam_q = torch.stack([self.front.to(torch.float64).expand(n, 7), wrist], 1)
        return body_q.float(), cam_q.float()

    def goal_transform(self, goal_pose, n: int) -> torch.Tensor:
        """Goal marker transform [n, 7] (float64, x y z w) on the table top; see :meth:`transforms`."""
        out = torch.zeros(n, 7, dtype=torch.float64, device=self.device)
        out[:, 2], out[:, 6] = HIDDEN_Z, 1.0
        if goal_pose is None:
            return out
        g = torch.as_tensor(goal_pose).to(device=self.device, dtype=torch.float64)
        if g.shape[-1] == 3:
            yaw = g[:, 2]
        elif g.shape[-1] == 7:
            yaw = flat_yaw(g[:, 3:7] / g[:, 3:7].norm(dim=-1, keepdim=True))
        else:
            raise ValueError(f"goal_pose must be [B, 3] (x, y, yaw) or [B, 7] (pos, quat wxyz), got {tuple(g.shape)}")
        ok = torch.isfinite(g).all(-1)
        out[ok, :2] = g[ok, :2]
        out[ok, 2] = 0.0
        out[ok, 3:] = xyzw(yaw_quat(yaw[ok]))
        return out


# ---- renderer -----------------------------------------------------------------------------------------------------
class SO101Renderer:
    """Front + wrist images of ``num_worlds`` SO101 worlds per call, with Kyle's batch renderer.

    ``backend``: ``"madrona"`` (NVIDIA, always ``cuda:0``; one renderer per process) or ``"newton"`` (Warp: NVIDIA,
    AMD, CPU; ``device`` = a Warp device, default the GPU if there is one). Both cameras share one resolution and the
    wrist camera's field of view (fovy 61 deg, ``so101.xml``). Calls with fewer than ``num_worlds`` rows are padded
    (the extra worlds repeat row 0) and the outputs cut back.
    """

    def __init__(self, num_worlds: int, width: int = 128, height: int = 128, backend: str = "madrona",
                 shadows: bool = True, assets: SO101Assets | None = None, cube_colors=None, goal: str | None = "plate",
                 front_eye=FRONT_EYE, front_target=FRONT_TARGET, sky=SKY, device=None):
        from nedm.render.renderer import BatchRenderer
        if backend == "madrona":
            torch_device = "cuda:0"
        else:
            torch_device = "cuda:0" if ("cuda" in str(device) or (device is None and torch.cuda.is_available())) \
                else "cpu"
        self.scene = SO101Scene(assets, cube_colors, goal, front_eye, front_target, device=torch_device)
        self.renderer = BatchRenderer(self.scene.scene, num_worlds, width, height, cameras=len(CAMERAS),
                                      fov_deg=self.scene.fovy_deg, device=device, shadows=shadows, sky=sky,
                                      backend=backend)
        self.device = torch.device(torch_device)
        self.num_worlds, self.width, self.height, self.backend = int(num_worlds), int(width), int(height), backend
        self.shadows = bool(shadows)
        # ray length -> z-depth, at pixel centres (fovy vertical, square pixels)
        focal = 0.5 * height / math.tan(math.radians(self.scene.fovy_deg) / 2)
        u = (torch.arange(width, dtype=torch.float64) + 0.5 - width / 2) / focal
        v = (torch.arange(height, dtype=torch.float64) + 0.5 - height / 2) / focal
        self.z_per_ray = (1.0 / torch.sqrt(1 + u[None, :] ** 2 + v[:, None] ** 2)).float().to(self.device)

    @property
    def geom_names(self) -> list[str]:
        """Names of the backend's drawn objects in its order: the bodies, then the statics (base meshes, table)."""
        return self.scene.body_names + [mesh.name for mesh, _ in self.scene.scene.statics]

    def render(self, q, cube_pos, cube_quat, goal_pose=None, depth: bool = False, copy: bool = True):
        """rgb [B, 2, H, W, 3] uint8 (cameras in CAMERAS order), and with ``depth`` also z-depth [B, 2, H, W] float32
        (m along the camera axis, 0 = nothing hit). Inputs as :meth:`SO101Scene.transforms`, any device and float
        dtype, B <= num_worlds. ``copy=False`` returns a view of the renderer's buffer (overwritten by the next call)."""
        body_q, cam_q = self.scene.transforms(q, cube_pos, cube_quat, goal_pose)
        n = body_q.shape[0]
        if n > self.num_worlds:
            raise ValueError(f"{n} states for {self.num_worlds} worlds")
        if n < self.num_worlds:
            pad = self.num_worlds - n
            body_q = torch.cat([body_q, body_q[:1].expand(pad, *body_q.shape[1:])])
            cam_q = torch.cat([cam_q, cam_q[:1].expand(pad, *cam_q.shape[1:])])
        frames = self.renderer.render(cam_q, body_q=body_q)
        rgb = frames.rgb_torch()[:n]
        if copy:
            rgb = rgb.clone(memory_format=torch.contiguous_format)
        if not depth:
            return rgb
        return rgb, frames.depth_torch()[:n] * self.z_per_ray.to(rgb.device)

    def render_state(self, state, goal_pose=None, **kw):
        """Render a :class:`nedm.so101_policy.world.WorldState` (true cube orientation)."""
        return self.render(state.q, state.cube_pos, state.cube_quat, goal_pose, **kw)

    # ---- EXPERIMENTAL: per-world appearance (domain randomisation) ----------------------------------------------
    def set_appearance(self, colors=None, light_dir=None, shadows=None, reset: bool = False) -> None:
        """Per-world colours and light, for robustness training. Reaches into private attributes of the backends.

        ``colors`` [N, geoms, 3] in 0-1, order :attr:`geom_names` (bodies, then statics), or [geoms, 3] for all worlds.
        Madrona: every geom of every world; flat colour quantised to 8 bits, shading = colour * max(0.2, n . l)
        (``bvh_raycast.cpp``), so brightness is randomised through the colour. Newton: the bodies only (arm, cube,
        goal); the statics (base, table) are one shape shared by all worlds.
        ``light_dir`` [N, 3] (the direction the light travels, world) and ``shadows`` [N] bool: Madrona only (Newton's
        lights are shared by all worlds). ``reset`` restores the scene's own materials, light and shadows first.

        Madrona copies materials, colours and lights into its scene only in ``init``; the next :meth:`render` re-runs
        ``init`` with the new values (it re-sorts the entities by world and rebuilds the render state; measured on
        the RTX 4090 at 16 worlds: 18.6 ms instead of 3.5 ms, no memory growth over 31 re-inits)."""
        b = self.renderer.backend
        if self.backend == "newton":
            if light_dir is not None or shadows is not None:
                raise NotImplementedError("the Newton backend has one light for all worlds")
            nb = len(self.scene.body_names)
            base = torch.tensor([m.color for m in self.scene.scene.bodies], dtype=torch.float32)
            if reset:
                self._newton_colors(base.expand(self.num_worlds, nb, 3))
            if colors is not None:
                c = torch.as_tensor(colors, dtype=torch.float32).cpu()
                self._newton_colors(c[..., :nb, :].expand(self.num_worlds, nb, 3))
            return
        if reset:
            from nedm.render.backends.madrona import LIGHT_DIRECTION
            b._materials.fill_(-1)
            b._light_dir.copy_(torch.tensor(LIGHT_DIRECTION, device=b.device).expand_as(b._light_dir))
            b._light_shadow.fill_(self.shadows)
        if colors is not None:
            c = (torch.as_tensor(colors, dtype=torch.float32, device=b.device).clamp(0, 1) * 255).round().long()
            packed = (c[..., 0] << 16) + (c[..., 1] << 8) + c[..., 2]
            b._colors.copy_(packed.to(torch.uint32).expand_as(b._colors))
            b._materials.fill_(-2)          # MaterialOverride::UseOverrideColor; -1 = the geom's own colour
        if light_dir is not None:
            d = torch.as_tensor(light_dir, dtype=torch.float32, device=b.device)
            b._light_dir.copy_(torch.nn.functional.normalize(d, dim=-1).expand_as(b._light_dir))
        if shadows is not None:
            b._light_shadow.copy_(torch.as_tensor(shadows, dtype=torch.bool, device=b.device).expand_as(b._light_shadow))
        b._started = False      # the next render calls init with these values

    def _newton_colors(self, colors: torch.Tensor) -> None:
        """Body shape colours [N, bodies, 3] of the Newton backend (world-major shapes, then the shared statics)."""
        model = self.renderer.backend.model
        n = colors.shape[0] * colors.shape[1]
        current = model.shape_color.numpy()
        current[:n] = colors.reshape(n, 3).numpy()
        model.shape_color.assign(current)

    def base_colors(self) -> torch.Tensor:
        """The scene's colours [geoms, 3] in 0-1, order :attr:`geom_names`."""
        return torch.tensor([m.color for m in self.scene.scene.bodies] + [m.color for m, _ in self.scene.scene.statics])
