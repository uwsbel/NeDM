"""SO101 + cube + goal scene for the batch renderer: the arm of the grasp NRD study, drawn from the visual meshes of the
pinned Menagerie MJCF, the collector's table and 30 mm cube, a goal marker, and up to three cameras: ``front`` (fixed,
across the table, facing the arm), ``top`` (fixed, straight down over the workspace) and ``wrist`` (the MJCF
``wrist_cam``).

    r = SO101Renderer(num_worlds=1024, backend="newton")      # Warp: any CUDA GPU (luffy 5090), AMD, CPU
    rgb = r.render(q, cube_pos, cube_quat_wxyz, goal)          # [B, 2, 128, 128, 3] uint8, cameras in r.cameras order
    rgb, depth = r.render(..., depth=True)                     # depth [B, 2, H, W] float32: z-depth (m), 0 = no hit
    r = SO101Renderer(1024, 224, 224, cameras=("front", "wrist"))   # ACT input (spec v2 B7); or ("top", "wrist")
    rgb = r.render_state(world_state, goal)                    # a WorldState of the NRD or of MuJoCo, same function

One renderer has one resolution and one field of view for all its cameras (the wrist camera's fovy 61 deg, as
``BatchRenderer`` takes one ``fov_deg``); the fixed cameras are placed so that 61 deg frames the workspace. The REAL
setup (off by default; Newton only) instead draws the calibrated real cameras (fisheye / radial-tangential lens models
from the calibration contract, a crop box of the real 640 x 480 image per camera) in a light-box scene:

    real = RealScene.from_path("~/so101_r2s_claude/scene/real_scene_v1.json")   # fitted scene + crops + DR ranges
    r = SO101Renderer(1024, 224, 224, real=real)              # cameras ("external", "wrist"), student crops
    dr = RealDR.from_dict(real.dr)
    rgb = r.render_dr(q, cube_pos, cube_quat, goal, draw=dr.draw(r.scene, 1024, gen))   # per-world DR
    r.set_lights(dr.lights(real.lights, gen)); r.set_camera_models(dr.camera_models(r.scene, gen))   # per call
    rgb = ImageFX.from_dict(real.image_fx).apply(rgb[:, c], fx_draw, camera=name)   # nominal + random camera effects

The student crop (``nedm.render.student_view``) is the same code the real-robot script uses (``hw_lib``).

Madrona:
one renderer per process, NVIDIA only (luffy: ``source ~/so101_policy/runs/render/madrona_luffy/env_luffy.sh``),
7-20x faster than Newton on recorded states (luffy 5090, 2026-10-11); its engine start aborts the whole process (C++
FATAL, CUDA out of memory / out of resources at the first launch) when other jobs fill the GPU: retry the process.

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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from nedm.render.scene import MeshBody, Scene

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/contact_nrd/collection/so101_grasp.json"
VIEWS = ("front", "top", "wrist")           # the cameras a scene can hold
CAMERAS = ("front", "wrist")                # the default set
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
# Top camera: straight down over the middle of the annulus (x 0.057-0.330 m, |y| <= 0.282 m with the cube's half
# diagonal), image up = world +x (arm base in the lower half, the robot's left = image left). At fovy 61 deg and
# 0.66 m it frames that box from the table up to z = 0.15 m (worst point at 0.95 of the half width); 6.1 mm per pixel
# on the table at 128 px, 3.5 mm at 224 px.
TOP_EYE = (0.19, 0.0, 0.66)
TOP_TARGET = (0.19, 0.0, 0.0)
TOP_UP = (1.0, 0.0, 0.0)
GOAL_HALF = 0.023          # goal plate half side (m): a 30 mm cube on the goal leaves 8 mm of plate on every side
GOAL_THICKNESS = 0.001     # m; the plate lies on the table top, so a cube resting on it sinks 1 mm into it (hidden)
GOAL_FRAME_WIDTH = 0.006   # m, the "frame" style: an outline of this width, outer half side GOAL_HALF
# "tray" style (build8_tray_student.md): a VISUAL-ONLY tray at the goal pose (no collision anywhere: the MuJoCo scene and
# the NRD do not know it). A flat base slightly larger than the cube's 30 mm footprint (inner side 40 mm) with a low
# rim (5 mm high, 4 mm wide); base light cream, rim orange, against the blue cube and the grey-brown table. The base is
# 1 mm thick on the table top, so a cube resting on the goal hides 1 mm of it (as the plate); the rim intersects the
# cube only when the cube is more than 5 mm off the tray centre (visual overlap, no contact).
TRAY_INNER_HALF = 0.020    # m, half side of the inner square
TRAY_RIM_WIDTH = 0.004     # m
TRAY_RIM_HEIGHT = 0.005    # m above the table top
TRAY_BASE_THICKNESS = 0.001
TRAY_COLOR = (0.93, 0.91, 0.84)
TRAY_RIM_COLOR = (0.98, 0.62, 0.15)
# "block" style (build9_viral_student.md): a VISUAL-ONLY flat colour block at the goal pose (no collision anywhere):
# a square of 50 mm side, 1 mm thick on the table top (a cube resting on it hides 1 mm of it), saturated red against the
# blue cube and the grey-brown table. The lenient success of build9 = cube centre inside this square (any yaw).
BLOCK_HALF = 0.025         # m, half side
BLOCK_THICKNESS = 0.001    # m
BLOCK_COLOR = (0.85, 0.12, 0.12)
HIDDEN_Z = -100.0          # z of a hidden goal marker (far below the table)
BOX_FACES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32)
# BOX_FACES rows of each cube face, order +x, -x, +y, -y, +z, -z
FACE_ROWS = ((2, 3), (0, 1), (6, 7), (4, 5), (10, 11), (8, 9))


def box_mesh(half, center=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Box with half sizes ``half`` (3,) around ``center``: vertices (8, 3) float32, faces (12, 3) int32."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    return (corners * np.asarray(half, float) + np.asarray(center, float)).astype(np.float32), BOX_FACES.copy()


def prism_mesh(corners_xy, z0: float, z1: float, center=None) -> tuple[np.ndarray, np.ndarray]:
    """Closed prism over a convex polygon ``corners_xy`` (k, 2) from z0 to z1, outward-facing triangles (Newton culls
    back faces), vertices relative to ``center`` (3,) (default: the polygon centroid at z = 0): (2k, 3) float32,
    (4k - 4, 3) int32."""
    c = np.asarray(corners_xy, np.float64).reshape(-1, 2)
    mid = c.mean(0)
    c = c[np.argsort(np.arctan2(c[:, 1] - mid[1], c[:, 0] - mid[0]))]           # counter-clockwise from above
    k = len(c)
    v = np.concatenate([np.c_[c, np.full(k, z0)], np.c_[c, np.full(k, z1)]])
    faces = [[k, k + i, k + i + 1] for i in range(1, k - 1)]                        # top (+z)
    faces += [[0, i + 1, i] for i in range(1, k - 1)]                               # bottom (-z)
    for i in range(k):
        j = (i + 1) % k
        faces += [[i, j, k + j], [i, k + j, k + i]]                                 # sides, outward
    center = np.r_[mid, 0.0] if center is None else np.asarray(center, np.float64)
    return (v - center).astype(np.float32), np.array(faces, np.int32)


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
    of 6 colours), the goal marker (none with ``goal=None``; 2 meshes for the tray). Static: the arm base meshes and
    the table. ``goal``: ``"plate"`` (a 1 mm square plate on the table, side 2 * GOAL_HALF), ``"frame"`` (its outline)
    or ``"tray"`` (a visual-only tray: 40 mm inner base, 5 mm rim; ``TRAY_*`` constants) or ``"block"`` (a visual-only
    flat colour square of 50 mm side; ``BLOCK_*`` constants).
    ``cameras``: names from VIEWS, in the order of the output's camera axis.
    ``table_body``: draw the table as the LAST body (fixed pose) instead of a static mesh, so that a backend with
    per-world body colours (Newton: :meth:`SO101Renderer.set_appearance`) can colour it per world (build9's visual
    randomisation); the pictures are the same as with the static table.

    ``plate_depth`` (real scene; None = off, the old plate): the goal plate's mesh is this deep (m, >= the scene's
    ``plate_thickness``) below its nominal top, so a per-world plate thickness up to ``plate_depth`` is a vertical
    shift of the plate (``transforms(..., plate_dz=)``; :class:`RealDR` ``plate_thickness``); the part below the mat
    top is hidden in the mat.
    ``real``: a :class:`RealScene` (off by default) builds the REAL setup instead (:meth:`_init_real`): no sim table;
    cameras from REAL_VIEWS (default ``external`` + ``wrist``, calibrated, :attr:`camera_models`); everything is a body
    (per-world colours and poses for all of it): the moving arm meshes, the cube, the goal plate (the 50 mm ``block``
    style in the plate colour; none with ``goal=None``), the arm base meshes, then the floor, the mat and the walls
    (:attr:`env_pose`, overridable per world). :attr:`groups` = body indices of REAL_GROUPS.
    """

    def __init__(self, assets: SO101Assets | None = None, cube_colors=None, goal: str | None = "plate",
                 front_eye=FRONT_EYE, front_target=FRONT_TARGET, device=None, cfg: dict | None = None,
                 cameras: Sequence[str] | None = None, top_eye=TOP_EYE, top_target=TOP_TARGET,
                 table_body: bool = False, real: "RealScene | None" = None, plate_depth: float | None = None):
        from nedm.render import cameras as cams
        self.real, self.camera_models = real, None
        self.plate_depth = None
        if plate_depth is not None and real is None:
            raise ValueError("plate_depth (plate thickness randomisation) needs the real scene")
        if real is not None:
            self._init_real(real, assets, cube_colors, goal, front_eye, front_target, device, cfg, cameras, top_eye,
                            top_target, plate_depth)
            return
        cameras = CAMERAS if cameras is None else cameras
        self.cameras = tuple(cameras)
        self.table_body = bool(table_body)
        if not self.cameras or len(set(self.cameras)) != len(self.cameras) or not set(self.cameras) <= set(VIEWS):
            raise ValueError(f"cameras {cameras!r}: distinct names from {VIEWS}")
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
        elif goal == "tray":
            hi, w, hr, tb = TRAY_INNER_HALF, TRAY_RIM_WIDTH, TRAY_RIM_HEIGHT, TRAY_BASE_THICKNESS
            ho = hi + w
            v, f = box_mesh([ho, ho, tb / 2], (0.0, 0.0, tb / 2))
            bodies.append(MeshBody("goal", vertices=v, faces=f, color=TRAY_COLOR))
            parts = [box_mesh([ho, w / 2, hr / 2], (0.0, s * (ho - w / 2), hr / 2)) for s in (-1, 1)]
            parts += [box_mesh([w / 2, hi, hr / 2], (s * (ho - w / 2), 0.0, hr / 2)) for s in (-1, 1)]
            bodies.append(MeshBody("goal_rim", vertices=np.concatenate([p[0] for p in parts]),
                                   faces=np.concatenate([p[1] + 8 * k for k, p in enumerate(parts)]),
                                   color=TRAY_RIM_COLOR))
        elif goal == "block":
            v, f = box_mesh([BLOCK_HALF, BLOCK_HALF, BLOCK_THICKNESS / 2], (0.0, 0.0, BLOCK_THICKNESS / 2))
            bodies.append(MeshBody("goal", vertices=v, faces=f, color=BLOCK_COLOR))
        elif goal is not None:
            raise ValueError(f"goal style {goal!r}: 'plate', 'frame', 'tray', 'block' or None")
        self.goal_meshes = 0 if goal is None else (2 if goal == "tray" else 1)
        t = cfg["table"]
        tv, tf = box_mesh([0.5 * t["size_xy_m"][0], 0.5 * t["size_xy_m"][1], 0.5 * t["thickness_m"]])
        table_mesh = MeshBody("table", vertices=tv, faces=tf, color=TABLE_COLOR)
        self.table_pose = torch.tensor([*t["center_xy_m"], -0.5 * t["thickness_m"], 0.0, 0.0, 0.0, 1.0],
                                       dtype=torch.float64, device=self.device)
        if self.table_body:
            bodies.append(table_mesh)
        self.scene = Scene.from_meshes(bodies)
        # static: the base (welded to the world) and the table, as the collector builds them
        zero = torch.zeros(1, 6, dtype=torch.float64, device=self.device)
        bpos, bquat = self.kin(zero)
        for body, mesh in assets.meshes:
            if body not in moving:
                i = self.kin.bodies.index(body)
                self.scene.add_static_mesh(mesh, torch.cat([bpos[0, i], xyzw(bquat[0, i])]).tolist())
        if not self.table_body:
            self.scene.add_static_mesh(table_mesh, self.table_pose.tolist())
        # cameras: front and top fixed (x y z, qx qy qz qw; look along -z, +y up); wrist = the MJCF camera on its body
        f64 = lambda v: torch.tensor([v], dtype=torch.float64)                                     # noqa: E731
        self.front = cams.look_at(f64(front_eye), f64(front_target))[0].to(self.device)
        self.top = cams.look_at(f64(top_eye), f64(top_target), up=TOP_UP)[0].to(self.device)
        cam = assets.wrist_cam
        self.cam_body = self.kin.bodies.index(cam["body"])
        self.cam_pos = torch.tensor(cam["pos"], dtype=torch.float64, device=self.device)
        self.cam_quat = torch.tensor(cam["quat"], dtype=torch.float64, device=self.device)
        self.fovy_deg = float(cam["fovy_deg"])

    def _init_real(self, real: "RealScene", assets, cube_colors, goal, front_eye, front_target, device, cfg, cameras,
                   top_eye, top_target, plate_depth=None) -> None:
        """The real-setup scene (class docstring, :class:`RealScene`)."""
        from nedm.render import cameras as cams
        self.cameras = tuple(REAL_CAMERAS if cameras is None else cameras)
        if not self.cameras or len(set(self.cameras)) != len(self.cameras) or not set(self.cameras) <= set(REAL_VIEWS):
            raise ValueError(f"cameras {cameras!r}: distinct names from {REAL_VIEWS} in the real scene")
        self.table_body = False
        self.assets = assets = assets if assets is not None else load_assets()
        cfg = cfg or load_config()
        self.device = torch.device(device or "cpu")
        self.kin = SO101Kinematics(assets.links, self.device)
        moving = self._moving_bodies()
        hidden = set(real.hide_bodies or ())
        if hidden - set(self.kin.bodies):
            raise ValueError(f"hide_bodies {sorted(hidden - set(self.kin.bodies))} are not bodies of the MJCF")
        groups: dict[str, list[int]] = {g: [] for g in REAL_GROUPS}
        bodies, mesh_body = [], []
        for body, mesh in assets.meshes:
            if body in moving and body not in hidden:
                part = arm_part(mesh.color)
                groups[part].append(len(bodies))
                bodies.append(replace(mesh, color=real.color(part)))
                mesh_body.append(self.kin.bodies.index(body))
        wrist_cal = real.calib["cameras"].get("wrist")
        board = real.board_mesh(wrist_cal["T_parent_cam"]) if wrist_cal is not None else None
        if board is not None:                       # the real camera board, on the wrist camera's parent body
            if wrist_cal["parent"] not in self.kin.bodies:
                raise ValueError(f"wrist camera parent {wrist_cal['parent']!r} is not a body of the MJCF")
            groups["board"].append(len(bodies))
            bodies.append(MeshBody("camera_board", vertices=board[0], faces=board[1], color=real.color("board")))
            mesh_body.append(self.kin.bodies.index(wrist_cal["parent"]))
        for part in real.parts:
            if part["body"] not in self.kin.bodies or part["body"] not in moving:
                raise ValueError(f"part {part.get('name')!r}: body {part['body']!r} is not a moving MJCF body")
            v, f = polygon_prism(part["polygon"], part["offset"])
            groups[part.get("group", "board")].append(len(bodies))
            bodies.append(MeshBody(part.get("name", "part"), vertices=v, faces=f,
                                   color=real.color(part.get("group", "board"))))
            mesh_body.append(self.kin.bodies.index(part["body"]))
        self.mesh_body = torch.tensor(mesh_body, device=self.device)
        self.cube_half = 0.5 * float(cfg["cube"]["size_m"])
        v, f = box_mesh([self.cube_half] * 3)
        groups["cube"].append(len(bodies))
        bodies.append(MeshBody("cube", vertices=v, faces=f,
                               color=tuple(cube_colors) if cube_colors is not None else real.color("cube")))
        self.cube_meshes = 1
        self.goal_style = None if goal is None else "block"
        self.goal_meshes = 0 if goal is None else 1
        if goal is not None:
            pt = float(real.plate_thickness)
            if plate_depth is not None:     # thickness randomisation (v8): the mesh reaches this far below its top
                self.plate_depth = max(pt, float(plate_depth))
            if self.plate_depth is None or self.plate_depth <= pt:
                v, f = box_mesh([BLOCK_HALF, BLOCK_HALF, pt / 2], (0.0, 0.0, pt / 2))
            else:                           # top at pt as before
                v, f = box_mesh([BLOCK_HALF, BLOCK_HALF, self.plate_depth / 2], (0.0, 0.0, pt - self.plate_depth / 2))
            groups["plate"].append(len(bodies))
            bodies.append(MeshBody("goal", vertices=v, faces=f, color=real.color("plate")))
        # fixed bodies: the arm base meshes (at their welded pose), then the floor, the mat and the walls
        zero = torch.zeros(1, 6, dtype=torch.float64, device=self.device)
        bpos, bquat = self.kin(zero)
        fixed = []
        for body, mesh in assets.meshes:
            if body not in moving and body not in hidden:
                i = self.kin.bodies.index(body)
                part = arm_part(mesh.color)
                groups[part].append(len(bodies))
                bodies.append(replace(mesh, color=real.color(part)))
                fixed.append(torch.cat([bpos[0, i], xyzw(bquat[0, i])]))
        self.env_groups, self.env_names = [], []
        for g, name, mesh, pose in real.env_meshes():
            groups[g].append(len(bodies))
            bodies.append(mesh)
            fixed.append(torch.tensor(pose, dtype=torch.float64, device=self.device))
            self.env_groups.append(g)
            self.env_names.append(name)
        self.fixed_pose = torch.stack(fixed)                         # [n_fixed, 7] x y z qx qy qz qw, float64
        self.n_env = len(self.env_names)
        self.groups = groups
        self.scene = Scene.from_meshes(bodies)
        # cameras: calibrated external (world-fixed) and wrist (on its calibrated body); top and front as the sim's
        f64 = lambda v: torch.tensor([v], dtype=torch.float64)                                     # noqa: E731
        self.front = cams.look_at(f64(front_eye), f64(front_target))[0].to(self.device)
        self.top = cams.look_at(f64(top_eye), f64(top_target), up=TOP_UP)[0].to(self.device)
        self.fovy_deg = float(assets.wrist_cam["fovy_deg"])
        calib = real.calib["cameras"]
        cam = assets.wrist_cam
        self.cam_body = self.kin.bodies.index(cam["body"])
        self.cam_pos = torch.tensor(cam["pos"], dtype=torch.float64, device=self.device)
        self.cam_quat = torch.tensor(cam["quat"], dtype=torch.float64, device=self.device)
        if "wrist" in self.cameras:
            c = calib["wrist"]
            if c["parent"] not in self.kin.bodies:
                raise ValueError(f"wrist camera parent {c['parent']!r} is not a body of the MJCF: {self.kin.bodies}")
            local = cams.pose_from_opencv(c["T_parent_cam"]).to(self.device)
            self.cam_body = self.kin.bodies.index(c["parent"])
            self.cam_pos, self.cam_quat = local[:3], wxyz(local[3:])
        if "external" in self.cameras:
            c = calib["external"]
            if c["parent"] != "world":
                raise ValueError(f"external camera parent {c['parent']!r}: only 'world' is handled")
            self.external = cams.pose_from_opencv(c["T_parent_cam"]).to(self.device)
        self.camera_models = [real.camera_model(n) if n in REAL_CAMERAS else cams.PinholeCamera(self.fovy_deg)
                              for n in self.cameras]
        # per camera tone map [C, 9] (camera_fx.TONE_KEYS), None: Newton's own 8-bit output
        self.tone_params = None
        if real.tone is not None:
            from nedm.render.camera_fx import tone_vector
            missing = [n for n in self.cameras if n not in real.tone]
            if missing:
                raise ValueError(f"tone map missing for cameras {missing}")
            self.tone_params = torch.tensor([tone_vector(real.tone[n]) for n in self.cameras], dtype=torch.float32)
        o, qp = real.plane_pose()
        self.plane_o = torch.tensor(o, dtype=torch.float64, device=self.device)
        self.plane_q = torch.tensor(qp, dtype=torch.float64, device=self.device)        # w x y z

    @property
    def env_pose(self) -> torch.Tensor:
        """Nominal poses [n_env, 7] (x y z qx qy qz qw, float64) of the floor, mat and walls (real scene)."""
        return self.fixed_pose[len(self.fixed_pose) - self.n_env:]

    def _plate_shift(self, g, cube_pos, cq, plate_dz):
        """Plate thickness randomisation (v8; table frame of the flat model): the plate rows ``g`` [n, 7]
        (x y z qx qy qz qw, before the table-plane map) raised by ``plate_dz`` [n] (m: the plate top above the nominal
        one; hidden plates stay), and the cube's visual lift [n] (m, along the table normal): a cube over the plate
        sinks into the drawn plate no deeper than into the nominal plate (lift = plate_dz - its lowest corner's height
        above the table clipped at 0, if positive: a cube resting on the table sits plate_thickness deep in the plate,
        as in the nominal scene; full lift with the cube centre over the plate, none once its footprint is off the
        plate, linear between), so a thicker plate does not swallow a resting or placed cube; the cube's physics
        state is unchanged. -> (g, lift)."""
        f64 = dict(device=self.device, dtype=torch.float64)
        n = g.shape[0]
        shown = g[:, 2] > HIDDEN_Z / 2
        dz = torch.where(shown, torch.as_tensor(plate_dz).to(**f64).reshape(n), 0.0)
        g = g.clone()
        g[:, 2] = g[:, 2] + dz
        qi = (self.plane_q * torch.tensor([1.0, -1.0, -1.0, -1.0], **f64)).expand(n, 4)
        c = qrot(qi, cube_pos - self.plane_o)                                     # cube centre, table frame
        w_, x_, y_, z_ = qmul(qi, cq).unbind(-1)                                  # cube rotation, table frame
        rz = torch.stack((2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)), -1)
        bottom = c[:, 2] - self.cube_half * rz.abs().sum(-1)                      # lowest corner above the table
        yaw = 2 * torch.atan2(g[:, 5], g[:, 6])                                   # the plate's yaw (z turn, xyzw)
        d = c[:, :2] - g[:, :2]
        lx = torch.cos(yaw) * d[:, 0] + torch.sin(yaw) * d[:, 1]
        ly = -torch.sin(yaw) * d[:, 0] + torch.cos(yaw) * d[:, 1]
        over = ((BLOCK_HALF + self.cube_half - torch.maximum(lx.abs(), ly.abs())) / self.cube_half).clamp(0.0, 1.0)
        lift = over * (dz - bottom.clamp_min(0.0)).clamp_min(0.0)
        return g, torch.where(shown & torch.isfinite(lift), lift, 0.0)

    @staticmethod
    def plane_frames(plane) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-world table planes [n, 3] (c0, c1, c2; z = c0 + c1 x + c2 y, float64) -> (origin [n, 3], quaternion
        [n, 4] w x y z) of the table frame, as :meth:`RealScene.plane_pose` (a flat plane: exactly the identity
        turn)."""
        c = torch.as_tensor(plane, dtype=torch.float64)
        nrm = torch.sqrt(1.0 + c[:, 1] ** 2 + c[:, 2] ** 2)
        z = torch.zeros_like(nrm)
        qv = torch.stack([1.0 + 1.0 / nrm, c[:, 2] / nrm, -c[:, 1] / nrm, z], -1)     # z x n, n ~ (-c1, -c2, 1)
        return torch.stack([z, z, c[:, 0]], -1), qv / qv.norm(dim=-1, keepdim=True)

    def _real_transforms(self, q, cube_pos, cube_quat, goal_pose=None, env_pose=None, plate_dz=None,
                         world_plane=None, cube_held=None):
        f64 = dict(device=self.device, dtype=torch.float64)
        q = torch.as_tensor(q).to(**f64)
        n = q.shape[0]
        pos, quat = self.kin(q)
        parts = [torch.cat([pos[:, self.mesh_body], xyzw(quat[:, self.mesh_body])], -1)]
        cq = torch.as_tensor(cube_quat).to(**f64)
        cq = cq / cq.norm(dim=-1, keepdim=True)
        cp = torch.as_tensor(cube_pos).to(**f64)
        g = None
        if plate_dz is not None:
            if self.goal_style is None or self.plate_depth is None:
                raise ValueError("plate_dz needs the goal plate and a renderer built with plate_depth")
            g, lift = self._plate_shift(self.goal_transform(goal_pose, n), cp, cq, plate_dz)
            if cube_held is not None:       # a cube the jaws hold is drawn at its physics pose (no lift)
                lift = torch.where(torch.as_tensor(cube_held, device=self.device).reshape(n).bool(), 0.0, lift)
            cp = cp + lift[:, None] * qrot(self.plane_q.expand(n, 4), torch.tensor([0.0, 0.0, 1.0], **f64).expand(n, 3))
        wp = None
        if world_plane is not None:         # per-world table plane (v8 table_render): the world drawn on it
            if self.real.table_plane is not None:
                raise ValueError("world_plane: the scene has its own table_plane (the physics poses are not flat)")
            wp = torch.as_tensor(world_plane).to(**f64).reshape(n, 3)
            wo, wq = self.plane_frames(wp)
            wo, wq = wo.to(self.device), wq.to(self.device)
            hz = wp[:, 0] + wp[:, 1] * cp[:, 0] + wp[:, 2] * cp[:, 1]           # the cube: sheared (xy kept, as the
            cp = cp + torch.stack([torch.zeros_like(hz), torch.zeros_like(hz), hz], -1)  # joint map), tilted
            cq = qmul(wq, cq)
        cube = torch.cat([cp, xyzw(cq)], -1)
        parts.append(cube[:, None])
        if self.goal_style is not None:
            g = self.goal_transform(goal_pose, n) if g is None else g
            if self.real.table_plane is not None:          # the plate lies on the measured table plane
                qp = self.plane_q.expand(n, 4)
                shown = g[:, 2:3] > HIDDEN_Z / 2
                on = torch.cat([self.plane_o + qrot(qp, g[:, :3]), xyzw(qmul(qp, wxyz(g[:, 3:])))], -1)
                g = torch.where(shown, on, g)
            elif wp is not None:                            # ... on the world's plane (as RealScene.on_plane)
                shown = g[:, 2:3] > HIDDEN_Z / 2
                on = torch.cat([wo + qrot(wq, g[:, :3]), xyzw(qmul(wq, wxyz(g[:, 3:])))], -1)
                g = torch.where(shown, on, g)
            parts.append(g[:, None])
        fixed = self.fixed_pose.expand(n, *self.fixed_pose.shape)
        if env_pose is not None:
            e = torch.as_tensor(env_pose).to(**f64).reshape(n, self.n_env, 7)
            fixed = torch.cat([fixed[:, : len(self.fixed_pose) - self.n_env], e], 1)
        if wp is not None:              # floor, mat, walls, strip, ceiling on the world's plane (not the lab room)
            k0, m = len(self.fixed_pose) - self.n_env, self.n_env
            e = fixed[:, k0:]
            wqe = wq[:, None].expand(n, m, 4)
            on = torch.cat([wo[:, None] + qrot(wqe, e[..., :3]), xyzw(qmul(wqe, wxyz(e[..., 3:])))], -1)
            mv = torch.tensor([gr not in ("room", "room_floor") for gr in self.env_groups], device=self.device)
            fixed = torch.cat([fixed[:, :k0], torch.where(mv[None, :, None], on, e)], 1)
        parts.append(fixed)
        body_q = torch.cat(parts, 1)
        views = []
        for name in self.cameras:
            if name == "wrist":
                bp, bq = pos[:, self.cam_body], quat[:, self.cam_body]
                views.append(torch.cat([bp + qrot(bq, self.cam_pos.expand(n, 3)),
                                        xyzw(qmul(bq, self.cam_quat.expand(n, 4)))], -1))
            else:
                views.append(getattr(self, name).to(torch.float64).expand(n, 7))
        return body_q.float(), torch.stack(views, 1).float()

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

    def transforms(self, q, cube_pos, cube_quat, goal_pose=None, env_pose=None, plate_dz=None, world_plane=None,
                   cube_held=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Body transforms [B, bodies, 7] and camera transforms [B, cameras, 7] (x y z, qx qy qz qw; float32).

        q [B, 6] rad; cube_pos [B, 3] m; cube_quat [B, 4] w x y z (normalised here); goal_pose None (hidden),
        [B, 3] = goal cube centre x, y (m) and yaw (rad), or [B, 7] = goal cube pose (position, quaternion w x y z),
        of which x, y and the yaw of the face closest to the table (:func:`flat_yaw`) are used. The marker lies on the
        table top under that point. Rows with a non-finite goal are hidden. ``env_pose`` (real scene only)
        [B, n_env, 7] x y z qx qy qz qw replaces the floor, mat and wall poses (:attr:`env_pose`). ``plate_dz`` (real
        scene built with ``plate_depth`` only) [B]: per-world plate thickness (:meth:`_plate_shift`); ``cube_held`` [B]
        bool: cubes the jaws hold (not lifted by the plate). ``world_plane`` (real scene without its own
        ``table_plane``) [B, 3]: per-world table plane z = c0 + c1 x + c2 y (v8 consistent table images): the cube
        (given in the flat model) is sheared by the plane height under it (xy kept) and tilted with the plane; the
        goal plate, floor, mat, walls, strip and ceiling are put on the plane as :meth:`RealScene.on_plane` does; the
        arm, its base, the lab room and the cameras stay (the arm frame); the caller draws the arm at the joints it
        wants (the student's reading)."""
        if self.real is not None:
            if plate_dz is None and world_plane is None and cube_held is None:
                return self._real_transforms(q, cube_pos, cube_quat, goal_pose, env_pose)
            return self._real_transforms(q, cube_pos, cube_quat, goal_pose, env_pose, plate_dz, world_plane,
                                         cube_held)
        if env_pose is not None or plate_dz is not None or world_plane is not None or cube_held is not None:
            raise ValueError("env_pose / plate_dz / world_plane / cube_held need the real scene")
        f64 = dict(device=self.device, dtype=torch.float64)
        q = torch.as_tensor(q).to(**f64)
        n = q.shape[0]
        pos, quat = self.kin(q)
        parts = [torch.cat([pos[:, self.mesh_body], xyzw(quat[:, self.mesh_body])], -1)]
        cq = torch.as_tensor(cube_quat).to(**f64)
        cube = torch.cat([torch.as_tensor(cube_pos).to(**f64), xyzw(cq / cq.norm(dim=-1, keepdim=True))], -1)
        parts.append(cube[:, None].expand(n, self.cube_meshes, 7))
        if self.goal_style is not None:
            parts.append(self.goal_transform(goal_pose, n)[:, None].expand(n, self.goal_meshes, 7))
        if self.table_body:
            parts.append(self.table_pose.expand(n, 1, 7))
        body_q = torch.cat(parts, 1)
        views = []
        for name in self.cameras:
            if name == "wrist":         # body pose composed with the MJCF camera pose
                bp, bq = pos[:, self.cam_body], quat[:, self.cam_body]
                views.append(torch.cat([bp + qrot(bq, self.cam_pos.expand(n, 3)),
                                        xyzw(qmul(bq, self.cam_quat.expand(n, 4)))], -1))
            else:
                views.append(getattr(self, name).to(torch.float64).expand(n, 7))
        return body_q.float(), torch.stack(views, 1).float()

    def goal_transform(self, goal_pose, n: int) -> torch.Tensor:
        """Goal marker transform [n, 7] (float64, x y z w) on the table top; see :meth:`transforms`."""
        hidden = torch.zeros(n, 7, dtype=torch.float64, device=self.device)
        hidden[:, 2], hidden[:, 6] = HIDDEN_Z, 1.0
        if goal_pose is None:
            return hidden
        g = torch.as_tensor(goal_pose).to(device=self.device, dtype=torch.float64)
        if g.shape[-1] not in (3, 7):
            raise ValueError(f"goal_pose must be [B, 3] (x, y, yaw) or [B, 7] (pos, quat wxyz), got {tuple(g.shape)}")
        ok = torch.isfinite(g).all(-1, keepdim=True)
        g = torch.where(ok, g, 1.0)             # no host sync (mask indexing would cost one per call)
        yaw = g[:, 2] if g.shape[-1] == 3 else flat_yaw(g[:, 3:7] / g[:, 3:7].norm(dim=-1, keepdim=True))
        shown = torch.cat([g[:, :2], torch.zeros_like(g[:, :1]), xyzw(yaw_quat(yaw))], -1)
        return torch.where(ok, shown, hidden)


# ---- the real setup (real2sim) ------------------------------------------------------------------------------------
# Off by default: ``SO101Scene(real=RealScene(...))`` / ``SO101Renderer(real=...)``. The calibration contract
# (``~/so101_r2s_claude/calib/calib_v1.json``): cameras with K, D, image size, parent and T_parent_cam in the OpenCV
# convention; scene.mat.corners_xy; colours measured from real frames. The scene file (``real_scene_v1.json``,
# :meth:`RealScene.from_path`) holds what was fitted to the real frames: the student crops, the light-box geometry,
# albedos, lights, the per-camera tone maps and the randomisation ranges. The class defaults are PLACEHOLDERS (a
# white light box with a black cloth mat, a white printed arm with black servos, a green 30 mm cube, a red 50 mm
# square plate, an LED strip light on the right).
REAL_VIEWS = ("external", "wrist", "top", "front")      # external, wrist: calibrated; top, front: the 61 deg pinholes
REAL_CAMERAS = ("external", "wrist")
REAL_GROUPS = ("printed", "servo", "cube", "plate", "floor", "mat", "walls", "board", "room", "room_floor", "strip")
SERVO_MAX_MJCF = 0.3       # so101.xml: the sts3215 servo materials (and the wrist camera board) are rgba 0.1 0.1 0.1
# The real wrist camera's board (black PCB on a black bracket), drawn behind the calibrated wrist camera in its OpenCV
# frame (x right, y down = towards the jaws, z forward): a w x h x t board whose front face is ``behind`` m behind the
# camera centre, and a bracket block below and behind it. Everything has camera z < 0, so the wrist camera never sees
# it; the external camera does (calib REVIEW.md issue 2). Sizes estimated from the external frames, not measured.
DEFAULT_BOARD = {"size": (0.032, 0.032, 0.002), "behind": 0.004, "bracket": (0.014, 0.022, 0.018)}


def arm_part(color) -> str:
    """``"servo"`` for the MJCF's black servo material (rgb 0.1), else ``"printed"`` (the yellow printed parts)."""
    return "servo" if max(color) < SERVO_MAX_MJCF else "printed"


def load_calibration(path: str | Path) -> dict:
    """The calibration contract (JSON) at ``path``."""
    return json.loads(Path(path).expanduser().read_text())


def placeholder_calibration() -> dict:
    """A TEMPORARY calibration in the contract's layout for tests (not measured): both cameras fisheye fx = fy = 300,
    cx = 320, cy = 240, D = (0.05, 0, 0, 0), 640 x 480; external at (0.35, -0.35, 0.45) looking at (0.2, 0, 0); wrist =
    the MJCF ``wrist_cam`` pose on ``camera_mount`` (turned to the OpenCV convention)."""
    from nedm.render import cameras as cams
    ext = cams.opencv_look_at([0.35, -0.35, 0.45], [0.2, 0.0, 0.0])[0].tolist()
    q = (0.958280, -0.285792, -0.001556, 0.004547)                                    # so101.xml wrist_cam, w x y z
    gl = torch.tensor([-0.00296, 0.05436, -0.05310, q[1], q[2], q[3], q[0]], dtype=torch.float64)
    wrist = cams.pose_to_opencv(gl).tolist()
    lens = dict(model="opencv_fisheye", width=640, height=480, K=[300.0, 300.0, 320.0, 240.0], D=[0.05, 0.0, 0.0, 0.0],
                rms_px=float("nan"), n_points=0, holdout_rms_px=float("nan"))
    return {"placeholder": True,
            "cameras": {"external": dict(lens, parent="world", T_parent_cam=ext),
                        "wrist": dict(lens, parent="camera_mount", T_parent_cam=wrist)},
            "scene": {"mat": {"corners_xy": None}, "notes": "placeholder, not measured"}}


def _quat_from_two(a, b) -> np.ndarray:
    """Unit quaternion w x y z of the shortest turn that maps unit vector a to unit vector b."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    if 1.0 + a @ b < 1e-12:                                   # opposite: half turn about any axis normal to a
        axis = np.cross(a, (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0))
        return np.r_[0.0, axis / np.linalg.norm(axis)]
    q = np.r_[1.0 + a @ b, np.cross(a, b)]
    return q / np.linalg.norm(q)


def _flip(faces: np.ndarray) -> np.ndarray:
    """Reverse the winding (outward -> inward-facing triangles: Newton culls back faces)."""
    return faces[:, ::-1].copy()


def polygon_prism(polygon, offset) -> tuple[np.ndarray, np.ndarray]:
    """Closed prism over a planar convex polygon (k, 3) (ordered) swept by ``offset`` (3,): vertices (2k, 3) float32,
    faces (4k - 4, 3) int32, every face turned outward (away from the centroid)."""
    p = np.asarray(polygon, np.float64)
    k = len(p)
    v = np.concatenate([p, p + np.asarray(offset, np.float64)])
    faces = [[0, i, i + 1] for i in range(1, k - 1)] + [[k, k + i, k + i + 1] for i in range(1, k - 1)]
    for i in range(k):
        j = (i + 1) % k
        faces += [[i, j, k + j], [i, k + j, k + i]]
    f = np.array(faces, np.int64)
    mid = v.mean(0)
    nrm = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    out = ((v[f].mean(1) - mid) * nrm).sum(-1) < 0
    f[out] = f[out][:, ::-1]
    return v.astype(np.float32), f.astype(np.int32)


@dataclass
class RealScene:
    """The real-setup scene configuration (geometry in the world = arm base frame, m).

    ``calib``: the contract dict (or :meth:`from_path`). Camera crop: ``crop`` None (default) = the student view of
    :mod:`nedm.render.student_view` (``student_view`` here, else ``calib["student_view"]``, else a centred square; the
    SAME rules as the real-robot code ``hw_lib``), whose output size must equal the renderer's size; ``crop`` a dict =
    per camera the real-pixel box [x0, y0, w, h] (None: the whole 640 x 480 image), the old explicit form.
    Geometry: the black mat = a prism over ``mat_corners_xy`` (None: the calibration's ``scene.mat.corners_xy``, else
    ``DEFAULT_MAT``) from z = -``mat_thickness`` to 0; the white floor = a box over ``floor_xy`` (x0, x1, y0, y1) with
    its top at z = -``mat_thickness``; ``walls``: vertical white panels, each a floor segment ((x0, y0), (x1, y1)) of
    the panel's inner face, ``wall_height`` tall (or its own height as a third element) and ``wall_thickness`` thick
    away from the mat (empty: none);
    ``ceiling``: z of a white panel over the floor (None: none); ``strip``: the LED strip, a box from ``"from"`` to
    ``"to"`` (m) with cross-section ``"size"`` (w, t) (group ``strip``; None: none); ``room``: the dark room outside
    the light box, an inward-facing box ``{"center": (3,), "half": (3,)}`` (group ``room``; None: rays that leave the
    box get ``sky``; ``"floor_z"``: also a lab floor at that height, group ``room_floor``). ``table_plane``
    (c0, c1, c2): put the floor, mat, walls, ceiling, strip and the goal plate on the
    plane z = c0 + c1 x + c2 y (the measured real table, calib ``scene.table_plane``); None: z = 0 (the sim physics).
    ``hide_bodies``: MJCF bodies whose meshes are left out (default the MJCF wrist camera mount and board, which sit
    in front of the calibrated wrist camera); ``board``: the real camera board drawn behind the calibrated wrist camera
    (:data:`DEFAULT_BOARD`; None: none). ``plate_thickness``: the goal plate (50 mm square) thickness. ``parts``:
    extra rigid parts on arm bodies, each ``{"name", "body", "group", "polygon": (k, 3) in the body frame, "offset":
    (3,)}`` (a prism, :func:`polygon_prism`), e.g. the real camera bracket's plate under the wrist camera.
    ``floor_turn`` (None: none): ``{"deg": a, "pivot": (x, y)}``, the floor box turned by a about the table normal
    through the pivot (flat-model frame; default pivot: the middle of the floor's -x edge), the measured turn of the
    real table edge (scene v3); it is also the pivot of ``RealDR.floor_turn_deg``. ``wall_end`` (None: none):
    ``{"wall": k, "slide": s}``: wall k's FIRST end becomes a separate piece (``wall{k}_end``, group walls) that
    ``RealDR.wall_end`` slides along the wall by up to +-s; the main wall then starts s later, the piece is 2 s + 10 mm
    long, 0.5 mm behind the main wall's inner face and 0.5 mm lower (no coincident faces); nominal picture = the wall
    ending at its listed first point.
    Colours (linear albedo, Newton multiplies by the light): ``colors[group]`` for REAL_GROUPS. ``lights``: Newton
    lights (``NewtonBackend.set_lights``), shared by all worlds; ``ambient``: Newton's fixed ambient term; ``sky``:
    8-bit colour of rays that hit nothing. ``tone``: per camera a tone map (``camera_fx``: exposure, wb, black, gamma,
    vignette) applied to Newton's linear output (None: Newton's own clamped 8-bit colour). ``supersample``: s x s rays
    per output pixel, averaged after the tone map (as the real frame is area-averaged to the student size).
    ``dr`` / ``image_fx``: the randomisation ranges (:meth:`RealDR.from_dict`, ``camera_fx.ImageFX.from_dict``)."""

    DEFAULT_MAT = ((0.08, -0.22), (0.42, -0.22), (0.42, 0.22), (0.08, 0.22))
    DEFAULT_COLORS = {"printed": (0.92, 0.92, 0.90), "servo": (0.05, 0.05, 0.05), "cube": (0.05, 0.62, 0.22),
                      "plate": (0.85, 0.08, 0.08), "floor": (0.93, 0.93, 0.93), "mat": (0.03, 0.03, 0.03),
                      "walls": (0.93, 0.93, 0.93), "board": (0.03, 0.03, 0.03), "room": (0.04, 0.035, 0.025),
                      "room_floor": (0.2, 0.18, 0.15), "strip": (1.0, 1.0, 1.0)}
    # A key light from above (shadows) and dimmed shadowless fills from the sides and from below (the light box's
    # diffuse glow and the floor's bounce: Newton has no inter-reflection and a fixed weak ambient, so without them
    # the white walls come out grey and every face turned from the key light near black). Placeholders, not measured.
    DEFAULT_LIGHTS = ({"type": "directional", "direction": (-0.25, 0.35, -1.0), "shadow": True},
                      {"type": "directional", "direction": (0.1, 1.0, -0.35), "shadow": False, "intensity": 0.6,
                       "center": (0.25, 0.0, 0.0)},
                      {"type": "directional", "direction": (1.0, -0.1, -0.35), "shadow": False, "intensity": 0.6,
                       "center": (0.25, 0.0, 0.0)},
                      {"type": "directional", "direction": (0.0, -1.0, -0.3), "shadow": False, "intensity": 0.4,
                       "center": (0.25, 0.0, 0.0)},
                      {"type": "directional", "direction": (-1.0, 0.0, -0.3), "shadow": False, "intensity": 0.4,
                       "center": (0.25, 0.0, 0.0)},
                      {"type": "directional", "direction": (0.0, 0.0, 1.0), "shadow": False, "intensity": 0.35,
                       "center": (0.25, 0.0, 0.0)})

    calib: dict = field(default_factory=placeholder_calibration)
    crop: dict | None = None
    mat_corners_xy: Sequence | None = None
    mat_thickness: float = 0.001
    floor_xy: tuple = (-0.25, 0.60, -0.45, 0.45)
    floor_thickness: float = 0.01
    walls: Sequence = (((-0.25, 0.45), (0.60, 0.45)), ((0.60, 0.45), (0.60, -0.45)))
    wall_height: float = 0.60
    wall_thickness: float = 0.005
    colors: dict = field(default_factory=dict)
    lights: Sequence = DEFAULT_LIGHTS
    shadows: bool = True
    ambient: bool = True
    sky: tuple = (25, 25, 25)
    student_view: dict | None = None
    tone: dict | None = None
    supersample: int = 1
    hide_bodies: tuple = ("camera_mount",)
    board: dict | None = field(default_factory=lambda: dict(DEFAULT_BOARD))
    room: dict | None = None
    ceiling: float | None = None
    strip: dict | None = None
    table_plane: Sequence | None = None
    plate_thickness: float = BLOCK_THICKNESS
    parts: Sequence = ()
    floor_turn: dict | None = None
    wall_end: dict | None = None
    dr: dict = field(default_factory=dict)
    image_fx: dict = field(default_factory=dict)
    source: str = ""

    # keys of the scene file's "geometry" section that map to fields
    GEOMETRY_KEYS = ("mat_corners_xy", "mat_thickness", "floor_xy", "floor_thickness", "walls", "wall_height",
                     "wall_thickness", "room", "ceiling", "strip", "table_plane", "plate_thickness", "hide_bodies",
                     "board", "parts", "floor_turn", "wall_end")
    WALL_END_OVERLAP = 0.01     # m: the end piece reaches this far past the main wall's start at its shortest
    WALL_END_INSET = 0.0005     # m: the end piece lies this far behind the main wall's inner face and below its top

    @classmethod
    def from_path(cls, path: str | Path, **kw) -> "RealScene":
        """From a calibration contract (``calib_v1.json``) or from a scene file (``real_scene_v1.json``: its
        ``"calib"`` key names the contract; its sections set the fields). ``kw`` overrides any field."""
        d = load_calibration(path)
        if not str(d.get("version", "")).startswith("real_scene"):
            return cls(calib=d, **kw)
        return cls.from_scene_dict(d, source=str(Path(path).expanduser()), **kw)

    @classmethod
    def from_scene_dict(cls, d: dict, calib: dict | None = None, source: str = "", **kw) -> "RealScene":
        """From a parsed scene file (see :meth:`from_path`); ``calib`` overrides the file's ``"calib"`` path."""
        if calib is None:
            calib = load_calibration(d["calib"])
        f: dict = {}
        geo = d.get("geometry") or {}
        for k in cls.GEOMETRY_KEYS:
            if k in geo:
                f[k] = geo[k]
        if f.get("table_plane") == "calib":
            f["table_plane"] = tuple(calib["scene"]["table_plane"]["coef"])
        if f.get("walls") is not None:
            f["walls"] = tuple(tuple(tuple(p) if isinstance(p, (list, tuple)) else float(p) for p in w)
                               for w in f["walls"])
        for k in ("floor_xy", "hide_bodies"):
            if f.get(k) is not None:
                f[k] = tuple(f[k])
        if "colors" in d:
            f["colors"] = {g: tuple(c) for g, c in d["colors"].items()}
        light = d.get("light") or {}
        if "lights" in light:
            f["lights"] = tuple(light["lights"])
        for k in ("shadows", "ambient"):
            if k in light:
                f[k] = bool(light[k])
        if "sky" in light:
            f["sky"] = tuple(int(c) for c in light["sky"])
        if "student_view" in d:
            f["student_view"] = d["student_view"]
        if "tone" in d:
            f["tone"] = d["tone"]
        if "supersample" in (d.get("render") or {}):
            f["supersample"] = int(d["render"]["supersample"])
        f["dr"] = d.get("dr") or {}
        f["image_fx"] = d.get("image_fx") or {}
        f.update(kw)
        return cls(calib=calib, source=source, **f)

    def color(self, group: str) -> tuple:
        return tuple(float(c) for c in self.colors.get(group, self.DEFAULT_COLORS[group]))

    def mat_corners(self) -> np.ndarray:
        if self.mat_corners_xy is not None:
            return np.asarray(self.mat_corners_xy, np.float64)
        mat = (self.calib.get("scene") or {}).get("mat") or {}
        corners = mat.get("corners_xy")
        return np.asarray(corners if corners is not None else self.DEFAULT_MAT, np.float64)

    # -- cameras --
    def view(self, name: str, out_res: int | None = None):
        """The camera's :class:`~nedm.render.student_view.ViewSpec` (crop box + output size) when ``crop`` is None
        (student-view rules), else None (explicit ``crop`` dict: box only, any output size)."""
        from nedm.render import student_view as sv
        if self.crop is not None:
            return None
        src = {"student_view": self.student_view} if self.student_view is not None else self.calib
        cam = self.calib["cameras"][name]
        return sv.view_specs(src, (name,), out_res, cam["width"], cam["height"])[name]

    def crop_box(self, name: str, out_res: int | None = None):
        """The real-pixel crop box [x0, y0, w, h] of a calibrated camera (None: the whole image)."""
        if self.crop is not None:
            return self.crop.get(name)
        return self.view(name, out_res).crop_xywh

    def camera_model(self, name: str, k_scale: float = 1.0, pp_shift=(0.0, 0.0), out_res: int | None = 224):
        """The calibrated camera with its crop; ``k_scale`` scales fx, fy and ``pp_shift`` (px) moves cx, cy (the
        intrinsics randomisation; the crop box stays fixed in the real image)."""
        from nedm.render.cameras import CalibratedCamera
        cam = dict(self.calib["cameras"][name])
        if k_scale != 1.0 or tuple(pp_shift) != (0.0, 0.0):
            fx, fy, cx, cy = cam["K"]
            cam["K"] = [fx * k_scale, fy * k_scale, cx + pp_shift[0], cy + pp_shift[1]]
        return CalibratedCamera.from_contract(cam, self.crop_box(name, out_res))

    # -- geometry --
    def plane_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(origin (3,), quaternion w x y z) of the table frame: z = 0 of the flat model -> the measured plane."""
        if self.table_plane is None:
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        c0, c1, c2 = (float(c) for c in self.table_plane)
        return np.array([0.0, 0.0, c0]), _quat_from_two((0.0, 0.0, 1.0), (-c1, -c2, 1.0))

    def on_plane(self, pose) -> np.ndarray:
        """A flat-model pose (7,) x y z qx qy qz qw -> the same pose on the table plane (identity without one)."""
        pose = np.asarray(pose, np.float64)
        if self.table_plane is None:
            return pose
        o, q = self.plane_pose()
        qt = torch.tensor(q)
        p = o + qrot(qt, torch.tensor(pose[:3])).numpy()
        r = qmul(qt, wxyz(torch.tensor(pose[3:7]))).numpy()
        return np.r_[p, r[1:], r[0]]

    def env_meshes(self) -> list[tuple[str, str, MeshBody, np.ndarray]]:
        """(group, name, mesh in its own frame, pose (7,) x y z qx qy qz qw) of the floor, mat, walls, then the
        ceiling, LED strip and room when set."""
        out = []
        x0, x1, y0, y1 = self.floor_xy
        top = -self.mat_thickness
        v, f = box_mesh([(x1 - x0) / 2, (y1 - y0) / 2, self.floor_thickness / 2])
        pose = [(x0 + x1) / 2, (y0 + y1) / 2, top - self.floor_thickness / 2, 0, 0, 0, 1.0]
        if self.floor_turn:                              # turned about the table normal through the pivot
            th = math.radians(float(self.floor_turn["deg"]))
            piv = self.floor_pivot()
            d = np.asarray(pose[:2]) - piv
            pose[:2] = (piv + [math.cos(th) * d[0] - math.sin(th) * d[1],
                               math.sin(th) * d[0] + math.cos(th) * d[1]]).tolist()
            pose[3:] = [0.0, 0.0, math.sin(th / 2), math.cos(th / 2)]
        out.append(("floor", "floor", MeshBody("floor", vertices=v, faces=f, color=self.color("floor")),
                    self.on_plane(pose)))
        if self.wall_end and not 0 <= int(self.wall_end["wall"]) < len(self.walls):
            raise ValueError(f"wall_end wall {self.wall_end['wall']}: no such wall ({len(self.walls)} walls)")
        corners = self.mat_corners()
        v, f = prism_mesh(corners, -self.mat_thickness, 0.0)
        out.append(("mat", "mat", MeshBody("mat", vertices=v, faces=f, color=self.color("mat")),
                    self.on_plane(np.r_[corners.mean(0), 0.0, 0, 0, 0, 1.0])))
        mid = corners.mean(0)
        for k, wall in enumerate(self.walls):
            a, b = np.asarray(wall[0], np.float64), np.asarray(wall[1], np.float64)
            height = float(wall[2]) if len(wall) > 2 else self.wall_height
            t = (b - a) / np.linalg.norm(b - a)
            nrm = np.array([t[1], -t[0]])
            if np.dot(nrm, (a + b) / 2 - mid) < 0:                  # thickness away from the mat
                nrm = -nrm
            end = self.wall_end_slide(k)
            a0 = a + t * end                                       # the main wall (shortened with an end piece)
            quad = np.stack([a0, b, b + nrm * self.wall_thickness, a0 + nrm * self.wall_thickness])
            v, f = prism_mesh(quad, top, top + height)
            out.append(("walls", f"wall{k}", MeshBody(f"wall{k}", vertices=v, faces=f, color=self.color("walls")),
                        self.on_plane(np.r_[quad.mean(0), 0.0, 0, 0, 0, 1.0])))
            if end > 0:                                            # the end piece (RealDR.wall_end slides it)
                e1, ins = a + t * (2 * end + self.WALL_END_OVERLAP), self.WALL_END_INSET
                quad = np.stack([a + nrm * ins, e1 + nrm * ins, e1 + nrm * (self.wall_thickness + ins),
                                 a + nrm * (self.wall_thickness + ins)])
                v, f = prism_mesh(quad, top, top + height - ins)
                out.append(("walls", f"wall{k}_end", MeshBody(f"wall{k}_end", vertices=v, faces=f,
                                                              color=self.color("walls")),
                            self.on_plane(np.r_[quad.mean(0), 0.0, 0, 0, 0, 1.0])))
        if self.ceiling is not None:
            v, f = box_mesh([(x1 - x0) / 2, (y1 - y0) / 2, 0.0025])
            out.append(("walls", "ceiling", MeshBody("ceiling", vertices=v, faces=f, color=self.color("walls")),
                        self.on_plane([(x0 + x1) / 2, (y0 + y1) / 2, float(self.ceiling) + 0.0025, 0, 0, 0, 1.0])))
        if self.strip is not None:
            a, b = np.asarray(self.strip["from"], np.float64), np.asarray(self.strip["to"], np.float64)
            w, t = self.strip.get("size", (0.02, 0.01))
            length = float(np.linalg.norm(b - a))
            v, f = box_mesh([length / 2, w / 2, t / 2])
            q = _quat_from_two((1.0, 0.0, 0.0), b - a)
            out.append(("strip", "strip", MeshBody("strip", vertices=v, faces=f, color=self.color("strip")),
                        self.on_plane(np.r_[(a + b) / 2, q[1:], q[0]])))
        if self.room is not None:
            v, f = box_mesh(self.room["half"])
            c = np.asarray(self.room["center"], np.float64)
            out.append(("room", "room", MeshBody("room", vertices=v, faces=_flip(f), color=self.color("room")),
                        np.r_[c, 0, 0, 0, 1.0]))
            if self.room.get("floor_z") is not None:            # the lab floor under the table (own group)
                hx, hy, _ = self.room["half"]
                v, f = box_mesh([hx * 0.98, hy * 0.98, 0.005])
                out.append(("room_floor", "room_floor", MeshBody("room_floor", vertices=v, faces=f,
                                                                 color=self.color("room_floor")),
                            np.r_[c[0], c[1], float(self.room["floor_z"]) - 0.005, 0, 0, 0, 1.0]))
        return out

    def floor_pivot(self) -> np.ndarray:
        """(x, y) of the floor turn's pivot (flat-model frame): ``floor_turn["pivot"]``, else the middle of the floor
        box's -x edge."""
        ft = self.floor_turn or {}
        if ft.get("pivot") is not None:
            return np.asarray(ft["pivot"], np.float64)
        x0, x1, y0, y1 = self.floor_xy
        return np.array([x0, (y0 + y1) / 2])

    def wall_end_slide(self, k: int) -> float:
        """The slide range s (m) of wall k's end piece (0: no end piece)."""
        we = self.wall_end
        if not we or int(we["wall"]) != int(k):
            return 0.0
        s = float(we["slide"])
        w = self.walls[k]
        length = float(np.linalg.norm(np.asarray(w[1], np.float64) - np.asarray(w[0], np.float64)))
        if not 0 < 2 * s + self.WALL_END_OVERLAP < length:
            raise ValueError(f"wall_end slide {s}: needs 0 < 2 slide + {self.WALL_END_OVERLAP} < the wall length "
                             f"{length:.3f} m")
        return s

    def board_mesh(self, T_parent_cam) -> tuple[np.ndarray, np.ndarray] | None:
        """The camera board + bracket (vertices in the camera's PARENT body frame, faces), or None."""
        if not self.board:
            return None
        w, h, t = self.board["size"]
        behind = float(self.board["behind"])
        bw, bh, bd = self.board.get("bracket", (0.0, 0.0, 0.0))
        parts = [box_mesh([w / 2, h / 2, t / 2], (0.0, 0.0, -(behind + t / 2)))]
        if bw > 0:
            parts.append(box_mesh([bw / 2, bh / 2, bd / 2], (0.0, h / 2 - bh / 2, -(behind + t + bd / 2))))
        v = np.concatenate([p[0] for p in parts]).astype(np.float64)
        f = np.concatenate([p[1] + 8 * k for k, p in enumerate(parts)])
        if (v[:, 2] >= 0).any():
            raise ValueError("the camera board must lie behind the wrist camera (camera z < 0)")
        T = np.asarray(T_parent_cam, np.float64)                  # x y z, qw qx qy qz, OpenCV camera in the parent
        r = np.zeros((3, 3))
        qw, qx, qy, qz = T[3:7] / np.linalg.norm(T[3:7])
        r[:] = [[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]]
        return (v @ r.T + T[:3]).astype(np.float32), f.astype(np.int32)


@dataclass
class RealDR:
    """Per-world visual randomisation of the real scene (as build9's ``viral.DRConfig``), ranges centred on the real
    scene (``real_scene_v1.json`` ``"dr"``: :meth:`from_dict`).

    Per world (:meth:`draw`): per group a brightness factor ``bright`` (lo, hi) and a per-channel tint ``tint`` (+-) of
    the albedo; camera pose offsets ``cam_pos`` (+- m per world axis) and ``cam_rot`` (+- rad per camera axis); the
    mat pose (``mat_xy`` +- m, ``mat_yaw`` +- rad); the walls' xy (``wall_xy`` +- m); the tone map (``exposure``
    (lo, hi) factor, ``wb`` +- per channel, ``black`` (lo, hi) linear offset added, ``gamma`` (lo, hi) factor,
    ``vignette`` (lo, hi) exponent added; each a value for all cameras or a dict per camera name). Per call: the light
    directions and point-light positions (:meth:`lights`, ``light_deg``, ``light_pos`` +- m per axis) and the
    intrinsics (:meth:`camera_models`: ``focal`` +- fraction, ``pp_px`` +- px, quantised to ``focal_levels`` levels so
    the renderer's ray cache is hit).

    Background randomisation (studentlab external-camera fix, 2026-10-23; all OFF by default, drawn AFTER every other
    per-world draw, so the other draws of a seed do not change when they are on): ``floor_turn_deg`` (+- deg about the
    table normal through the scene's floor pivot, :meth:`RealScene.floor_pivot`) and ``floor_xy`` (+- m in x and y of
    the table frame) move the floor box per world; ``wall_end`` (+- m) slides the scene's wall end piece
    (``RealScene.wall_end``) along its wall; ``bg`` (dict of :class:`nedm.render.bg_dr.BgDR` fields; {} = off): the
    image-space layers of the background (texture, clutter, spots, dark-area profile, floor gradient, low texture,
    mat folds, boundary blur) in linear colour before the tone map; needs a renderer with ``bg_cameras``
    (:meth:`bg_cameras`) and the output resolution (``draw(..., res=)``).

    Plate thickness (real2sim v8, from studentlab round C, 2026-10-23; OFF by default, drawn after everything else):
    ``plate_thickness`` (lo, hi) m: per world the goal plate's thickness ~ U(lo, hi) (real plate 3.9 mm, sim 1 mm),
    VISUAL ONLY (the physics cube still lands on z = 0): draw key ``plate_dz`` [n] = thickness - the scene's
    ``plate_thickness`` (the plate's top above its nominal top; the renderer needs ``plate_depth=``
    :meth:`plate_depth`); a cube over the plate is lifted visually so it never sinks deeper into the plate than in the
    nominal scene (:meth:`SO101Scene._plate_shift`): a resting or placed cube is drawn on the plate's top. Pass the
    held cubes (``render(..., cube_held=)``, the online loop does): a cube the jaws hold is drawn at its physics pose
    (it sinks into a thicker plate in the last ``plate_dz`` above the table, as in the MuJoCo gate renders); without
    ``cube_held`` every cube is lifted, and a held cube would slide up inside the closed jaws there, a cue the robot
    never shows (its cube stops on the plate and the arm stalls)."""

    bright: dict = field(default_factory=lambda: {"printed": (0.85, 1.1), "servo": (0.6, 2.0), "cube": (0.75, 1.25),
                                                  "plate": (0.75, 1.25), "floor": (0.8, 1.1), "mat": (0.5, 2.5),
                                                  "walls": (0.75, 1.1)})
    tint: dict = field(default_factory=lambda: {"printed": 0.05, "servo": 0.1, "cube": 0.1, "plate": 0.1,
                                                "floor": 0.05, "mat": 0.15, "walls": 0.06})
    cam_pos: dict = field(default_factory=lambda: {"external": 0.01, "wrist": 0.002, "top": 0.01, "front": 0.01})
    cam_rot: dict = field(default_factory=lambda: {"external": 0.03, "wrist": 0.02, "top": 0.03, "front": 0.03})
    mat_xy: float = 0.01
    mat_yaw: float = 0.03
    wall_xy: float = 0.03
    light_deg: float = 15.0
    light_pos: float = 0.0
    exposure: tuple | dict = (1.0, 1.0)
    wb: float | dict = 0.0
    black: tuple | dict = (0.0, 0.0)
    gamma: tuple | dict = (1.0, 1.0)
    vignette: tuple | dict = (0.0, 0.0)
    focal: float = 0.0
    pp_px: float = 0.0
    focal_levels: int = 7
    floor_turn_deg: float = 0.0
    floor_xy: float = 0.0
    wall_end: float = 0.0
    bg: dict = field(default_factory=dict)
    plate_thickness: tuple = ()

    @property
    def plate_on(self) -> bool:
        """Plate thickness randomisation on (``plate_thickness`` = (lo, hi) m)."""
        return bool(self.plate_thickness)

    def plate_depth(self) -> float | None:
        """The plate mesh depth a renderer needs for ``plate_thickness`` (its upper end; None: off)."""
        if not self.plate_on:
            return None
        lo, hi = (float(v) for v in self.plate_thickness)
        if not 0.0 < lo <= hi:
            raise ValueError(f"RealDR.plate_thickness {self.plate_thickness}: (lo, hi) m with 0 < lo <= hi")
        return hi

    @property
    def bg_on(self) -> bool:
        """Background layers on (``bg`` set with a world probability above 0)."""
        return bool(self.bg) and float(self.bg.get("p", 1.0)) > 0

    def bg_dr(self):
        from nedm.render.bg_dr import BgDR
        return BgDR.from_dict(self.bg)

    def bg_cameras(self, cameras: Sequence[str]) -> tuple:
        """The cameras (in ``cameras`` order) that get the background layers (none when off)."""
        return self.bg_dr().cams(cameras) if self.bg_on else ()

    def bg_nominal(self, n: int, cameras: Sequence[str], res: int, device=None) -> dict:
        """Background layers of n worlds without randomisation ({} when off), for nominal draws."""
        if not self.bg_on:
            return {}
        from nedm.render.bg_dr import BgDR
        return BgDR.nominal(n, len(self.bg_cameras(cameras)), res, device or "cpu")

    @classmethod
    def from_dict(cls, d: dict | None) -> "RealDR":
        """From the scene file's ``"dr"`` section (keys = the field names; lists become tuples; unknown keys are an
        error)."""
        d = dict(d or {})
        bad = set(d) - set(cls.__dataclass_fields__)
        if bad:
            raise ValueError(f"unknown RealDR keys {sorted(bad)}")
        fix = lambda v: {k: tuple(x) if isinstance(x, list) else x for k, x in v.items()} if isinstance(v, dict) \
            else (tuple(v) if isinstance(v, list) else v)                                    # noqa: E731
        return cls(**{k: fix(v) for k, v in d.items()})

    @classmethod
    def from_scene(cls, real: "RealScene") -> "RealDR":
        """The ranges of a scene file (``real.dr``, real_scene_v1.json); the class defaults are placeholders."""
        return cls.from_dict(real.dr)

    @staticmethod
    def _gen_device(generator) -> torch.device:
        return torch.device(generator.device) if generator is not None else torch.device("cpu")

    def draw(self, scene: "SO101Scene", n: int, generator: torch.Generator | None = None, device=None,
             res: int | None = None, bg_prior: torch.Tensor | None = None) -> dict:
        """Per-world draws for n worlds: ``colors`` [n, bodies, 3], ``cam_offset`` [n, C, 6] (dpos, drot),
        ``env_pose`` [n, n_env, 7] (x y z qx qy qz qw) and, when the scene has a tone map, ``tone`` [n, C, 9]
        (``camera_fx.TONE_KEYS``), for :meth:`SO101Renderer.render_dr`. Random numbers are drawn on the generator's
        device (CPU without a generator); the results are on ``device`` (default the scene's). With ``bg`` on, also
        the background layers ``bg_*`` (:mod:`nedm.render.bg_dr`) at the output resolution ``res`` (default: the
        student view's output size); ``bg_prior`` [Cb, res, res]: placement weights of their clutter and spots
        (:meth:`SO101Renderer.bg_prior`; None: uniform)."""
        gdev = self._gen_device(generator)
        dev = torch.device(device or scene.device)

        def u(lo, hi, *shape):
            return (lo + (hi - lo) * torch.rand(n, *shape, device=gdev, generator=generator,
                                                dtype=torch.float64)).to(dev)

        base = torch.tensor([m.color for m in scene.scene.bodies], dtype=torch.float64, device=dev)
        cols = base.expand(n, *base.shape).clone()
        for g, idx in scene.groups.items():
            if not idx:
                continue
            lo, hi = self.bright.get(g, (1.0, 1.0))
            ti = self.tint.get(g, 0.0)
            cols[:, idx] = (cols[:, idx] * u(lo, hi, 1, 1) * u(1 - ti, 1 + ti, 1, 3)).clamp(0, 1)
        C = len(scene.cameras)
        pos = torch.tensor([self.cam_pos.get(c, 0.0) for c in scene.cameras], dtype=torch.float64, device=dev)
        rot = torch.tensor([self.cam_rot.get(c, 0.0) for c in scene.cameras], dtype=torch.float64, device=dev)
        off = torch.cat([u(-1, 1, C, 3) * pos[None, :, None], u(-1, 1, C, 3) * rot[None, :, None]], -1)
        env0 = scene.env_pose.to(dev)
        env = env0.expand(n, *scene.env_pose.shape).clone()
        names = list(getattr(scene, "env_names", ()))
        for k, g in enumerate(scene.env_groups):
            if g == "mat":
                env[:, k, :2] += u(-1, 1, 2) * self.mat_xy
                turn = yaw_quat(u(-1, 1) * self.mat_yaw)
                env[:, k, 3:] = xyzw(qmul(wxyz(env[:, k, 3:]), turn))   # about the mat's own normal
            elif g == "walls":
                if names and names[k].endswith("_end"):                  # a wall end piece moves with its wall
                    kp = names.index(names[k][:-4])
                    env[:, k, :2] += env[:, kp, :2] - env0[kp, :2]
                else:
                    env[:, k, :2] += u(-1, 1, 2) * self.wall_xy
        out = dict(colors=cols.float(), cam_offset=off.float(), env_pose=env)
        tone = getattr(scene, "tone_params", None)
        if tone is not None:
            t = tone.to(dev, torch.float64).expand(n, *tone.shape).clone()            # [n, C, 9]

            def rng(v, default):
                """(lo, hi) [C, 2] of a per-camera dict or one value / pair for all cameras."""
                vals = [v.get(c, default) if isinstance(v, dict) else v for c in scene.cameras]
                vals = [(-x, x) if not isinstance(x, (tuple, list)) else tuple(x) for x in vals]
                return torch.tensor(vals, dtype=torch.float64, device=dev)

            def uc(v, default, *shape):
                r = rng(v, default)
                return r[:, 0].view(C, *([1] * len(shape))) + (r[:, 1] - r[:, 0]).view(C, *([1] * len(shape))) \
                    * u(0, 1, C, *shape)

            t[..., 0] *= uc(self.exposure, (1.0, 1.0))
            t[..., 1:4] *= 1 + uc(self.wb, 0.0, 3)
            t[..., 4:7] += uc(self.black, (0.0, 0.0), 1)
            t[..., 7] *= uc(self.gamma, (1.0, 1.0))
            t[..., 8] += uc(self.vignette, (0.0, 0.0))
            out["tone"] = t.float()
        # background randomisation (class docstring): after every other draw
        if self.floor_turn_deg or self.floor_xy:
            self._floor_draw(scene, env, names, u, dev)
        if self.wall_end:
            self._wall_end_draw(scene, env, names, u, dev)
        if self.bg_on:
            cams = self.bg_cameras(scene.cameras)
            if res is None:
                views = [scene.real.view(c) for c in cams]
                if not views or any(v is None for v in views):
                    raise ValueError("RealDR.bg: pass the output resolution (draw(..., res=))")
                res = views[0].out_wh[0]
            out.update(self.bg_dr().draw(n, len(cams), int(res), generator, dev, prior=bg_prior))
        if self.plate_on:            # v8: after every other draw (class docstring)
            if getattr(scene, "plate_depth", None) is None:
                raise ValueError("RealDR.plate_thickness: build the renderer with plate_depth=RealDR.plate_depth()")
            hi = self.plate_depth()
            lo = float(self.plate_thickness[0])
            if hi > scene.plate_depth + 1e-12:
                raise ValueError(f"RealDR.plate_thickness {self.plate_thickness} > the renderer's plate_depth "
                                 f"{scene.plate_depth}")
            out["plate_dz"] = u(lo, hi) - float(scene.real.plate_thickness)
        return out

    def _floor_draw(self, scene, env, names, u, dev) -> None:
        """Floor box per world: turned by +- floor_turn_deg about the table normal through the scene's floor pivot,
        then shifted by +- floor_xy in x and y of the table frame (in place on ``env``)."""
        if "floor" not in names:
            raise ValueError("RealDR.floor_turn_deg / floor_xy: the scene has no floor")
        k = names.index("floor")
        n = env.shape[0]
        th = u(-1, 1) * math.radians(float(self.floor_turn_deg))
        sh = u(-1, 1, 2) * float(self.floor_xy)
        o, qp = scene.plane_o.to(dev), scene.plane_q.to(dev)
        qi = qp * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=qp.dtype, device=dev)
        piv = torch.as_tensor(scene.real.floor_pivot(), dtype=env.dtype, device=dev)
        pf = qrot(qi.expand(n, 4), env[:, k, :3] - o)                                   # table frame
        d = pf[:, :2] - piv
        c, s = torch.cos(th), torch.sin(th)
        pf = torch.stack([piv[0] + c * d[:, 0] - s * d[:, 1] + sh[:, 0], piv[1] + s * d[:, 0] + c * d[:, 1] + sh[:, 1],
                          pf[:, 2]], -1)
        env[:, k, :3] = o + qrot(qp.expand(n, 4), pf)
        env[:, k, 3:] = xyzw(qmul(wxyz(env[:, k, 3:]), yaw_quat(th)))                  # about its own normal

    def _wall_end_draw(self, scene, env, names, u, dev) -> None:
        """Wall end pieces per world: slid by +- wall_end along their wall (in place on ``env``)."""
        ends = [k for k, nm in enumerate(names) if nm.endswith("_end")]
        if not ends:
            raise ValueError("RealDR.wall_end needs a wall end piece in the scene (geometry.wall_end)")
        n = env.shape[0]
        qp = scene.plane_q.to(dev)
        for k in ends:
            w = int(names[k][len("wall"):-4])
            if float(self.wall_end) > scene.real.wall_end_slide(w) + 1e-12:
                raise ValueError(f"RealDR.wall_end {self.wall_end} > the scene's wall_end slide "
                                 f"{scene.real.wall_end_slide(w)}")
            a, b = (np.asarray(p, np.float64) for p in scene.real.walls[w][:2])
            t = torch.tensor(np.r_[(b - a) / np.linalg.norm(b - a), 0.0], dtype=env.dtype, device=dev)
            env[:, k, :3] += (u(-1, 1) * float(self.wall_end))[:, None] * qrot(qp.expand(n, 4), t.expand(n, 3))

    def lights(self, lights: Sequence[dict], generator: torch.Generator | None = None) -> list[dict]:
        """The lights with each direction turned by up to ``light_deg`` about a random axis (one draw per call);
        lights without a direction (point lights) are kept. Random numbers on the generator's device."""
        gdev = self._gen_device(generator)
        out = []
        for lt in lights:
            lt = dict(lt)
            if "direction" in lt:
                d = np.asarray(lt["direction"], np.float64)
                axis = torch.randn(3, generator=generator, dtype=torch.float64, device=gdev).cpu().numpy()
                ang = math.radians(self.light_deg) * float(torch.rand(1, generator=generator, dtype=torch.float64,
                                                                      device=gdev))
                q = np.r_[math.cos(ang / 2), math.sin(ang / 2) * axis / np.linalg.norm(axis)]
                lt["direction"] = qrot(torch.tensor(q), torch.tensor(d / np.linalg.norm(d))).tolist()
            elif "position" in lt and self.light_pos > 0:
                dp = (2 * torch.rand(3, generator=generator, dtype=torch.float64, device=gdev) - 1).cpu().numpy()
                lt["position"] = (np.asarray(lt["position"], np.float64) + self.light_pos * dp).tolist()
            out.append(lt)
        return out

    def camera_models(self, scene: "SO101Scene", generator: torch.Generator | None = None, out_res: int = 224) -> list:
        """One intrinsics draw per call (rays are shared by all worlds): per calibrated camera fx, fy scaled by one
        of ``focal_levels`` levels in 1 +- ``focal`` and cx, cy moved by up to +- ``pp_px``, rounded to 0.5 px;
        other cameras unchanged. For :meth:`SO101Renderer.set_camera_models`."""
        gdev = self._gen_device(generator)
        models = []
        for name, m in zip(scene.cameras, scene.camera_models):
            if name not in REAL_CAMERAS:
                models.append(m)
                continue
            r = torch.rand(3, generator=generator, dtype=torch.float64, device=gdev).cpu().numpy()
            L = max(1, int(self.focal_levels))
            k = 1.0 + (self.focal * (2 * round(r[0] * (L - 1)) / (L - 1) - 1) if L > 1 else 0.0)
            pp = tuple(round(2 * self.pp_px * (x - 0.5) * 2) / 2 for x in r[1:])
            models.append(scene.real.camera_model(name, k_scale=k, pp_shift=pp, out_res=out_res))
        return models


def axis_angle_quat(v: torch.Tensor) -> torch.Tensor:
    """[..., 3] rotation vectors -> quaternions w x y z (as ``viral._quat_axis_angle``)."""
    a = v.norm(dim=-1, keepdim=True)
    s = torch.where(a > 1e-12, torch.sin(a / 2) / a.clamp_min(1e-12), torch.full_like(a, 0.5))
    return torch.cat((torch.cos(a / 2), v * s), -1)


def offset_cameras(cam_q: torch.Tensor, cam_offset: torch.Tensor) -> torch.Tensor:
    """Camera transforms [B, C, 7] (x y z qx qy qz qw) moved by ``cam_offset`` [B, C, 6]: + dpos in the world, then
    turned by the rotation vector drot in the camera's own frame (build9's camera randomisation)."""
    cq = cam_q.clone()
    off = cam_offset.to(cq)
    cq[..., :3] = cq[..., :3] + off[..., :3]
    cq[..., 3:] = xyzw(qmul(wxyz(cq[..., 3:]), axis_angle_quat(off[..., 3:])))
    return cq


# ---- renderer -----------------------------------------------------------------------------------------------------
class SO101Renderer:
    """Images of ``num_worlds`` SO101 worlds per call from the ``cameras`` (names from VIEWS, default front + wrist),
    with Kyle's batch renderer.

    ``backend``: ``"madrona"`` (NVIDIA, always ``cuda:0``; one renderer per process) or ``"newton"`` (Warp: NVIDIA,
    AMD, CPU; ``device`` = a Warp device, default the GPU if there is one). All cameras share one resolution
    (``width`` x ``height``, e.g. 224 x 224 for ACT) and the wrist camera's field of view (fovy 61 deg, ``so101.xml``).
    Calls with fewer than ``num_worlds`` rows are padded (the extra worlds repeat row 0) and the outputs cut back.

    ``real``: a :class:`RealScene` (Newton only, off by default): the real setup with its calibrated cameras (default
    ``external`` + ``wrist``; every output pixel samples the camera's crop box, pixels without a ray are black), the
    scene's lights and sky (``sky`` is ignored). Per-world randomisation: :class:`RealDR` + :meth:`render_dr`.
    With the student-view crop rules (``real.crop`` None) the view's output size must equal ``width`` x ``height``
    (ValueError otherwise). ``real.tone``: the colour is Newton's linear output through the per-camera tone map
    (:mod:`nedm.render.camera_fx`) and, with ``real.image_fx``, the nominal camera effects of each camera (blur,
    chroma bleeding: ``ImageFX.nominal``); ``real.supersample`` s: s x s rays per output pixel, tone-mapped, then
    averaged.
    ``segment``: :meth:`render` can also return the body index per pixel (real scene). ``hdr``: keep Newton's linear
    colour even without a tone map (``render(..., hdr=True)``). At construction the real scene renders one frame and
    raises if more than ``NEAR_FRAC`` of a calibrated camera's pixels are closer than ``NEAR_M`` (a body in front of
    the lens, e.g. a camera mount: review finding 2).
    ``bg_cameras`` (real scene with a tone map; default none): the cameras that take background layers
    (:class:`RealDR` ``bg``, :mod:`nedm.render.bg_dr`) in :meth:`render_dr` / :meth:`render` (``bg=``); builds the
    segment output internally (the colours are the same as without it). ``bg_dark_groups``: the groups of the dark
    area outside the light box (``BgDR.dark_groups``).
    ``plate_depth`` (real scene; None = off): the plate mesh depth for the per-world plate thickness (:class:`RealDR`
    ``plate_thickness``, ``render(..., plate_dz=)``; :class:`SO101Scene`).
    """

    NEAR_M = 0.012
    NEAR_FRAC = 0.2
    TONE_CHUNK = 1 << 25            # sub-pixels per tone-map chunk (float temporaries about 0.4 GB each)

    def __init__(self, num_worlds: int, width: int = 128, height: int = 128, backend: str = "newton",
                 shadows: bool = True, assets: SO101Assets | None = None, cube_colors=None, goal: str | None = "plate",
                 front_eye=FRONT_EYE, front_target=FRONT_TARGET, sky=SKY, device=None,
                 cameras: Sequence[str] | None = None, top_eye=TOP_EYE, top_target=TOP_TARGET,
                 table_body: bool = False, real: "RealScene | None" = None, segment: bool = False,
                 hdr: bool = False, bg_cameras: Sequence[str] = (),
                 bg_dark_groups: Sequence[str] = ("room", "room_floor"), plate_depth: float | None = None):
        from nedm.render.renderer import BatchRenderer
        if backend == "madrona":
            torch_device = "cuda:0"
        else:
            torch_device = "cuda:0" if ("cuda" in str(device) or (device is None and torch.cuda.is_available())) \
                else "cpu"
        self.scene = SO101Scene(assets, cube_colors, goal, front_eye, front_target, device=torch_device,
                                cameras=cameras, top_eye=top_eye, top_target=top_target, table_body=table_body,
                                real=real, **({} if plate_depth is None else dict(plate_depth=plate_depth)))
        self.cameras = self.scene.cameras
        self.real = real
        self.bg_cameras = tuple(bg_cameras or ())
        if self.bg_cameras:
            if real is None or real.tone is None:
                raise ValueError("bg_cameras: background layers need the real scene with a tone map")
            if not set(self.bg_cameras) <= set(self.cameras):
                raise ValueError(f"bg_cameras {self.bg_cameras} are not all in the cameras {self.cameras}")
            self.bg_cameras = tuple(c for c in self.cameras if c in self.bg_cameras)     # the cameras' order
            bad = set(bg_dark_groups) - set(REAL_GROUPS)
            if bad:
                raise ValueError(f"bg_dark_groups {sorted(bad)} are not groups of the real scene {REAL_GROUPS}")
        self._bg_dark_codes = tuple(REAL_GROUPS.index(g) for g in bg_dark_groups)
        if real is None:
            if segment or hdr:
                raise ValueError("segment / hdr outputs need the real scene")
            self.renderer = BatchRenderer(self.scene.scene, num_worlds, width, height, cameras=len(self.cameras),
                                          fov_deg=self.scene.fovy_deg, device=device, shadows=shadows, sky=sky,
                                          backend=backend)
        else:
            if backend != "newton":
                raise NotImplementedError("the real scene's calibrated cameras need the newton backend")
            for name in self.cameras:
                view = real.view(name, width) if name in REAL_CAMERAS else None
                if view is not None and view.out_wh != (int(width), int(height)):
                    raise ValueError(f"{name}: the student view's output {view.out_wh} differs from the renderer size "
                                     f"{(width, height)} (real_scene student_view / RealScene.student_view)")
            self.ss = int(real.supersample)
            if self.ss < 1:
                raise ValueError("supersample >= 1")
            self.renderer = BatchRenderer(self.scene.scene, num_worlds, width * self.ss, height * self.ss,
                                          cameras=len(self.cameras), fov_deg=self.scene.fovy_deg, device=device,
                                          shadows=shadows, sky=tuple(real.sky), backend=backend,
                                          camera_models=self.scene.camera_models,
                                          hdr=bool(hdr or real.tone is not None),
                                          segment=bool(segment or self.bg_cameras))
            self.set_lights(real.lights, shadows=bool(shadows and real.shadows), ambient=real.ambient)
        self.device = torch.device(torch_device)
        self.num_worlds, self.width, self.height, self.backend = int(num_worlds), int(width), int(height), backend
        self.shadows = bool(shadows)
        self._nominal = True
        if real is not None:        # per camera: ray length -> z-depth (0 at pixels without a ray)
            self._fx = None
            if real.tone is not None and real.image_fx:
                from nedm.render.camera_fx import ImageFX
                f = ImageFX.from_dict(real.image_fx)
                if any(f._per_camera(v, c) > 0 for v in (f.nominal_blur_px, f.nominal_chroma_px)
                       for c in self.cameras):
                    self._fx = f
            self._update_rays()
            if self.bg_cameras:     # Newton shape index -> group code (REAL_GROUPS order; -1: none)
                lut = torch.as_tensor(self.renderer.backend.shape_body, dtype=torch.long)
                body_group = torch.full((int(lut.max()) + 2,), -1, dtype=torch.long)
                for gi, g in enumerate(REAL_GROUPS):
                    for b in self.scene.groups.get(g, ()):
                        body_group[b] = gi
                self._shape_group = body_group[lut].to(self.device)
            self.check_near()
            return
        # ray length -> z-depth, at pixel centres (fovy vertical, square pixels)
        focal = 0.5 * height / math.tan(math.radians(self.scene.fovy_deg) / 2)
        u = (torch.arange(width, dtype=torch.float64) + 0.5 - width / 2) / focal
        v = (torch.arange(height, dtype=torch.float64) + 0.5 - height / 2) / focal
        self.z_per_ray = (1.0 / torch.sqrt(1 + u[None, :] ** 2 + v[:, None] ** 2)).float().to(self.device)

    @property
    def valid_mask(self) -> torch.Tensor | None:
        """[C, H, W] bool: pixels with a ray (real scene; with supersampling: all sub-pixels have one). None: the
        sim scene (Newton's own pinholes, every pixel has a ray)."""
        if self.real is None:
            return self.renderer.valid_mask
        return self._valid_out.cpu()

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Mean over the s x s sub-pixels: [..., H*s, W*s] -> [..., H, W]."""
        s = self.ss
        if s == 1:
            return x
        return x.unflatten(-1, (x.shape[-1] // s, s)).unflatten(-3, (x.shape[-2] // s, s)).mean((-1, -3))

    def _update_rays(self) -> None:
        b = self.renderer.backend
        self.z_per_ray = torch.from_numpy(b.ray_z).to(self.device)                    # [C, H*s, W*s]
        valid = torch.from_numpy(b.valid).to(self.device)
        self._valid_sub = valid
        self._valid_out = self._pool(valid.float()) > 0.999
        # cos of each sub-pixel ray's angle to the optical axis (vignetting)
        self._cos_theta = self.z_per_ray.clamp_min(0.0)

    def set_camera_models(self, models: Sequence) -> None:
        """New camera models, one per camera in :attr:`cameras` order (Newton; e.g. intrinsics randomised per call:
        rays are shared by all worlds of a call; :meth:`RealDR.camera_models`). Updates the z-depth factors. The
        renderer keeps the last ray sets (``NewtonBackend.RAY_CACHE``), so a few alternating draws cost nothing."""
        self.renderer.set_camera_models(models)
        if self.real is None:
            self.z_per_ray = torch.from_numpy(self.renderer.backend.ray_z).to(self.device)
        else:
            self._update_rays()

    def check_near(self, q=None) -> dict:
        """Render one frame of the nominal real scene (arm at ``q``, default zeros, cube and goal hidden far away) and
        raise if more than ``NEAR_FRAC`` of the valid pixels of a calibrated camera are closer than ``NEAR_M``.
        Returns {camera: fraction}."""
        q = torch.zeros(1, 6, dtype=torch.float64) if q is None else torch.as_tensor(q, dtype=torch.float64)
        cube = torch.tensor([[0.0, 0.0, HIDDEN_Z]], dtype=torch.float64)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        _, d = self.render(q, cube, quat, None, depth=True)
        out = {}
        for c, name in enumerate(self.cameras):
            if name not in REAL_CAMERAS:
                continue
            v = self._valid_out[c]
            dc = d[0, c][v]
            frac = float(((dc > 0) & (dc < self.NEAR_M)).float().mean()) if dc.numel() else 0.0
            out[name] = frac
            if frac > self.NEAR_FRAC:
                raise ValueError(f"{name}: {100 * frac:.0f} % of the pixels are closer than {self.NEAR_M} m (a body in "
                                 f"front of the lens: hide it with RealScene.hide_bodies)")
        return out

    def bg_prior(self, q=None) -> torch.Tensor:
        """[Cb, H, W] float: the share of dark-area sub-pixels (``bg_dark_groups``) per output pixel of the
        ``bg_cameras`` in one nominal frame (arm at ``q``, default zeros; cube and plate hidden): placement weights
        of the background clutter and spots (:meth:`RealDR.draw` ``bg_prior``)."""
        if not self.bg_cameras:
            raise ValueError("bg_prior: build the renderer with bg_cameras")
        q = torch.zeros(1, 6, dtype=torch.float64) if q is None else torch.as_tensor(q, dtype=torch.float64)
        cube = torch.tensor([[0.0, 0.0, HIDDEN_Z]], dtype=torch.float64)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        body_q, cam_q = self.scene.transforms(q, cube, quat, None)
        body_q = torch.cat([body_q, body_q[:1].expand(self.num_worlds - 1, *body_q.shape[1:])])
        cam_q = torch.cat([cam_q, cam_q[:1].expand(self.num_worlds - 1, *cam_q.shape[1:])])
        frames = self.renderer.render(cam_q, body_q=body_q)
        sh = frames.shape_index_torch()[:1]
        gid = torch.where(sh >= 0, self._shape_group[sh.clamp_min(0)], torch.full_like(sh, -1))[0]
        dark = torch.zeros_like(gid, dtype=torch.bool)
        for c in self._bg_dark_codes:
            dark |= gid == c
        idx = [self.cameras.index(c) for c in self.bg_cameras]
        return self._pool(dark[idx].float())

    def set_lights(self, lights: Sequence[dict], shadows: bool | None = None, ambient: bool | None = None) -> None:
        """Replace the lights of every world (Newton: :meth:`~nedm.render.backends.newton.NewtonBackend.set_lights`)."""
        if not hasattr(self.renderer.backend, "set_lights"):
            raise NotImplementedError("set_lights: newton backend only")
        self.renderer.backend.set_lights(list(lights), shadows=shadows, ambient=ambient)

    @property
    def geom_names(self) -> list[str]:
        """Names of the backend's drawn objects in its order: the bodies, then the statics (base meshes, table)."""
        return self.scene.body_names + [mesh.name for mesh, _ in self.scene.scene.statics]

    def render(self, q, cube_pos, cube_quat, goal_pose=None, depth: bool = False, copy: bool = True,
               env_pose=None, cam_offset=None, tone=None, segment: bool = False, hdr: bool = False, bg=None,
               plate_dz=None, world_plane=None, cube_held=None):
        """rgb [B, C, H, W, 3] uint8 (cameras in :attr:`cameras` order), and with ``depth`` also z-depth [B, C, H, W]
        float32 (m along the camera axis, 0 = nothing hit). Inputs as :meth:`SO101Scene.transforms`, any device and
        float dtype, B <= num_worlds. ``copy=False`` returns a view of the renderer's buffer (overwritten by the next
        call). ``env_pose`` [B, n_env, 7] (real scene): floor, mat and wall poses; ``cam_offset`` [B, C, 6]: per-world
        camera pose offsets (:func:`offset_cameras`).

        Real scene only: ``tone`` [B, C, 9] per-world tone maps (default the scene's); ``segment`` (renderer built with
        ``segment=True``): also the scene body index per pixel [B, C, H, W] int64 (-1 nothing hit, -2 sub-pixels
        disagree); ``hdr``: also the linear colour [B, C, H, W, 3] float32 (sub-pixel mean; rays that hit nothing get
        the sky colour turned linear). Extra outputs follow rgb (and depth) in the order segment, hdr. ``bg``: the
        background layers (``bg_*`` of :meth:`RealDR.draw`, rows = B) of the ``bg_cameras``, applied to the linear
        colour before the tone map (the ``hdr`` output stays the renderer's own). ``plate_dz`` [B] (renderer built with
        ``plate_depth``): the per-world plate thickness (:meth:`SO101Scene._plate_shift`); ``cube_held`` [B]: held
        cubes (no plate lift); ``world_plane`` [B, 3]: per-world table planes (:meth:`SO101Scene.transforms`)."""
        if plate_dz is not None or world_plane is not None or cube_held is not None:
            tkw = {k: v for k, v in dict(plate_dz=plate_dz, world_plane=world_plane, cube_held=cube_held).items()
                   if v is not None}
            body_q, cam_q = self.scene.transforms(q, cube_pos, cube_quat, goal_pose, env_pose=env_pose, **tkw)
        elif env_pose is None:
            body_q, cam_q = self.scene.transforms(q, cube_pos, cube_quat, goal_pose)
        else:
            body_q, cam_q = self.scene.transforms(q, cube_pos, cube_quat, goal_pose, env_pose=env_pose)
        if cam_offset is not None:
            cam_q = offset_cameras(cam_q, torch.as_tensor(cam_offset, device=cam_q.device))
        n = body_q.shape[0]
        if n > self.num_worlds:
            raise ValueError(f"{n} states for {self.num_worlds} worlds")
        if n < self.num_worlds:
            pad = self.num_worlds - n
            body_q = torch.cat([body_q, body_q[:1].expand(pad, *body_q.shape[1:])])
            cam_q = torch.cat([cam_q, cam_q[:1].expand(pad, *cam_q.shape[1:])])
        frames = self.renderer.render(cam_q, body_q=body_q)
        if self.real is not None:
            return self._real_outputs(frames, n, depth, tone, segment, hdr, bg)
        if bg is not None:
            raise ValueError("bg layers need the real scene")
        rgb = frames.rgb_torch()[:n]
        if copy:
            rgb = rgb.clone(memory_format=torch.contiguous_format)
        if not depth:
            return rgb
        return rgb, frames.depth_torch()[:n] * self.z_per_ray.to(rgb.device)

    def _real_outputs(self, frames, n, depth, tone, segment, hdr, bg=None):
        """The real scene's outputs: tone map per sub-pixel, then the sub-pixel mean (as the real frame: tone-mapped
        sensor pixels, then area-averaged to the student size)."""
        from nedm.render import camera_fx as fx
        dz = frames.depth_torch()[:n] * self.z_per_ray                     # [n, C, Hs, Ws] z-depth, 0: no hit
        sky = (dz == 0) & self._valid_sub[None]
        sky8 = torch.tensor(self.real.sky, dtype=torch.float32, device=dz.device) / 255.0
        params = self.scene.tone_params if tone is None else torch.as_tensor(tone, dtype=torch.float32)
        gid = None
        if bg is not None:
            if not self.bg_cameras:
                raise ValueError("bg layers: build the renderer with bg_cameras")
            if params is None:
                raise ValueError("bg layers need the tone map (linear colour)")
            sh = frames.shape_index_torch()[:n]
            gid = torch.where(sh >= 0, self._shape_group[sh.clamp_min(0)], torch.full_like(sh, -1))
            cb = len(self.bg_cameras)
            for k in ("bg_tex", "bg_low", "bg_mat", "bg_par"):
                if bg[k].shape[:2] != (n, cb):
                    raise ValueError(f"{k} {tuple(bg[k].shape)}: expected [{n}, {cb}, ...] (bg_cameras "
                                     f"{self.bg_cameras})")
        if params is not None:
            x = frames.hdr_torch()[:n]
            rgb = torch.empty(n, x.shape[1], self.height, self.width, 3, dtype=torch.uint8, device=x.device)
            step = max(1, self.TONE_CHUNK // (x.shape[1] * x.shape[2] * x.shape[3]))     # bounded temporaries
            for i in range(0, n, step):
                pi = params if params.dim() == 2 else params[i:i + step]
                xi = x[i:i + step]
                if gid is not None:
                    xi = self._apply_bg(xi, gid[i:i + step], {k: v[i:i + step] for k, v in bg.items()})
                y = fx.tone_map(xi, pi, self._cos_theta)
                y = torch.where(sky[i:i + step, ..., None], sky8, y)
                y = torch.where(self._valid_sub[None, ..., None], y, 0.0)
                y = self._pool(y.movedim(-1, 2)).movedim(2, -1)
                if self._fx is not None:                   # the real cameras' sharpness and colour bleeding
                    y = torch.stack([self._fx.nominal(y[:, c], cam) for c, cam in enumerate(self.cameras)], 1)
                rgb[i:i + step] = fx.to_uint8(y)
        elif self.ss > 1:
            y = frames.rgb_torch()[:n].float()
            rgb = (self._pool(y.movedim(-1, 2)).movedim(2, -1) + 0.5).clamp(0, 255).to(torch.uint8)
        else:
            rgb = frames.rgb_torch()[:n].clone(memory_format=torch.contiguous_format)
        out = [rgb]
        if depth:
            hit = (dz > 0).float()
            out.append(self._pool(dz) / self._pool(hit).clamp_min(1e-6) * (self._pool(hit) > 0))
        if segment:
            sh = frames.shape_index_torch()[:n]
            lut = torch.as_tensor(self.renderer.backend.shape_body, device=sh.device)
            seg = torch.where(sh >= 0, lut[sh.clamp_min(0)], torch.full_like(sh, -1))
            s = self.ss
            if s > 1:
                blk = seg.unflatten(-1, (seg.shape[-1] // s, s)).unflatten(-3, (seg.shape[-2] // s, s))
                first = blk[..., :1, :, :1]
                same = (blk == first).all(-1).all(-2)
                seg = torch.where(same, first[..., 0, :, 0], torch.full_like(same, -2, dtype=seg.dtype))
            out.append(seg)
        if hdr:
            x = frames.hdr_torch()[:n].clone()
            x = torch.where(sky[..., None], fx.srgb_eotf(sky8), x)
            out.append(self._pool(x.movedim(-1, 2)).movedim(2, -1))
        return out[0] if len(out) == 1 else tuple(out)

    def _apply_bg(self, x, gid, bg):
        """Background layers on the linear colour x [m, C, Hs, Ws, 3] of the bg cameras (:func:`bg_dr.apply_bg`)."""
        from nedm.render.bg_dr import apply_bg
        codes = {g: i for i, g in enumerate(REAL_GROUPS)}
        out = list(x.unbind(1))
        for j, cam in enumerate(self.bg_cameras):
            c = self.cameras.index(cam)
            out[c] = apply_bg(x[:, c], gid[:, c], {k: v[:, j] for k, v in bg.items()}, codes, self._bg_dark_codes,
                              self.ss, self._cos_theta[c])
        return torch.stack(out, 1)

    def render_dr(self, q, cube_pos, cube_quat, goal_pose=None, draw: dict | None = None, **kw):
        """:meth:`render` with per-world randomisation ``draw`` (:meth:`RealDR.draw`: colors [B, bodies, 3],
        cam_offset [B, C, 6], env_pose [B, n_env, 7], tone [B, C, 9], background layers ``bg_*``, plate_dz [B]; any
        key may be missing: a missing key means its nominal value, also for the colours), or the nominal scene with
        None. Newton colours are per world (rows beyond B repeat row 0)."""
        if draw is None or draw.get("colors") is None:
            if not self._nominal:
                self.set_appearance(reset=True)
                self._nominal = True
            if draw is None:
                return self.render(q, cube_pos, cube_quat, goal_pose, **kw)
        else:
            cols = torch.as_tensor(draw["colors"], dtype=torch.float32)
            if cols.shape[0] < self.num_worlds:
                cols = torch.cat([cols, cols[:1].expand(self.num_worlds - cols.shape[0], *cols.shape[1:])])
            self.set_appearance(cols)
            self._nominal = False
        if draw.get("tone") is not None:
            kw["tone"] = draw["tone"]
        if draw.get("bg_par") is not None:
            kw["bg"] = {k: draw[k] for k in ("bg_tex", "bg_low", "bg_mat", "bg_par")}
        if draw.get("plate_dz") is not None:
            kw["plate_dz"] = draw["plate_dz"]
        return self.render(q, cube_pos, cube_quat, goal_pose, env_pose=draw.get("env_pose"),
                           cam_offset=draw.get("cam_offset"), **kw)

    def render_state(self, state, goal_pose=None, **kw):
        """Render a :class:`nedm.so101_policy.world.WorldState` (``q``, ``cube_pos``, true ``cube_quat`` w x y z), the
        same call for the NRD world and the MuJoCo worlds; ``goal_pose`` and ``kw`` as :meth:`render`."""
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
            sh = torch.as_tensor(shadows, dtype=torch.bool, device=b.device)
            b._light_shadow.copy_(sh.expand_as(b._light_shadow))
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
