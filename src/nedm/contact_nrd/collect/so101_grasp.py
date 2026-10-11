"""SO101 cube grasping in CPU MuJoCo: scene, scripted motion, probes and the collector of the contact NRD data.

    PYTHONPATH=src python -P -m nedm.contact_nrd.collect.so101_grasp collect --out DIR [--campaign pilot]
        [--workers 6] [--chunk 16] [--scale S] [--probe-scale S] [--families ...]
    PYTHONPATH=src python -P -m nedm.contact_nrd.collect.so101_grasp convergence --out DIR/convergence.json \
        [--workers 6]
    PYTHONPATH=src python -P -m nedm.contact_nrd.collect.so101_grasp report DIR
    MUJOCO_GL=egl PYTHONPATH=src python -P -m nedm.contact_nrd.collect.so101_grasp render DIR --episodes 2 425 1462

Contract (contact_nrd/collection_contract.md): body 0 ``arm`` (12 channels: the 6 joint angles, jaw included, then
the 6 joint rates), body 1 ``cube`` (13: position, quaternion wxyz, world velocity, world angular velocity), body 2
``table`` (fixed, top at z = 0). Pairs [0,1] fixed jaw-cube, [0,1] moving jaw-cube, [1,2] cube-table. A record
every 10 ms; the 6 joint position targets (float32, the applied values, kept ``joint_target_margin_rad`` inside the
joint limits) change every 20 ms and hold in between; the physics step divides 10 ms. A pair is on in a record
interval if its summed normal force (mj_contactForce; elliptic cones: the first constraint force of each contact) in
any physics step of the interval is above ``contact_threshold_n`` (MuJoCo computes the contacts and forces of a step
from the state at its start). Any other contact above that threshold (arm-table, other arm links or the arm base
with the cube, arm self-contact) ends the episode at the last record before that interval; so does the cube leaving
the table top, and a record at which the cube overlaps the visual meshes of the gripper or the moving jaw by more
than ``visual_overlap.threshold_m`` beyond the penetration of that body's collision geoms (parts of the real gripper
without a collision shape; the episode ends at the record before). The arm is the MuJoCo Menagerie SO101 (pinned in
contact_nrd/so101_grasp/source_manifest.json), loaded unchanged with MjSpec; the table, the cube, the options, names
for the unnamed collision geoms and collision hulls of the base's visual meshes (the Menagerie base has no collision
geoms) are added here. All physical parameters are fixed; only initial states and the scripted commands are random.
Campaigns (pilot, production, test) have their own master seeds in the config. numpy, scipy (overlap grids) and
mujoco only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from nedm.contact_nrd.collect import export, validate

CONFIG = export.REPO_ROOT / "configs/contact_nrd/collection/so101_grasp.json"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
PAIRS = ("fixed_jaw_cube", "moving_jaw_cube", "cube_table")
# contact classes after the three pairs (unsupported: they end the episode)
TERMINAL = ("end", "arm_table", "cube_other_link", "arm_self", "cube_base", "cube_off_table", "cube_visual_overlap")
PHASES = ("idle", "approach", "descend", "close", "grip", "lift", "hold", "move", "lower", "release", "retreat",
          "free", "push", "open_drop", "shake", "rest", "throw", "tap", "swat", "loosen")
OUTCOMES = ("probe", "missed", "lifted_held", "lifted_dropped")
P = {name: k for k, name in enumerate(PHASES)}
NONE, FIXED, MOVING, CUBE, TABLE, LINK, BASE = range(7)


class PlanError(RuntimeError):
    """A scripted pose is out of reach (the family samples again)."""


def load_config(path=CONFIG):
    return json.loads(Path(path).read_text())


def vendor_model(cfg):
    """Path of the pinned so101.xml and the source manifest."""
    mpath = export.REPO_ROOT / cfg["source_manifest"]
    manifest = json.loads(mpath.read_text())
    path = mpath.parent / manifest["source"]["local_path"] / manifest["model"]["path"]
    if not path.exists():
        raise SystemExit(f"{path} is missing: run python -I contact_nrd/so101_grasp/fetch_sources.py")
    return path, manifest


def verify_assets(cfg):
    """sha256 of every pinned vendor file against the manifest -> {path: sha256}."""
    path, manifest = vendor_model(cfg)
    root = (export.REPO_ROOT / cfg["source_manifest"]).parent / manifest["source"]["local_path"]
    out = {}
    for item in manifest["files"]:
        digest = export.sha256_file(root / item["path"])
        if digest != item["sha256"]:
            raise SystemExit(f"vendor file {item['path']} differs from the manifest")
        out[item["path"]] = digest
    return out


# ---- scene --------------------------------------------------------------------------------------------------
def build_spec(cfg, physics=None):
    """MjSpec of the scene: the unchanged Menagerie arm (base at the origin on the table top), the table, the cube."""
    path, _ = vendor_model(cfg)
    spec = mujoco.MjSpec.from_file(str(path))
    ph = {**cfg["physics"], **(physics or {})}
    o = spec.option
    o.timestep = ph["timestep_s"]
    o.integrator = getattr(mujoco.mjtIntegrator, "mjINT_" + ph["integrator"].upper())
    o.cone = getattr(mujoco.mjtCone, "mjCONE_" + ph["cone"].upper())
    o.solver = getattr(mujoco.mjtSolver, "mjSOL_" + ph["solver"].upper())
    o.impratio, o.iterations, o.ls_iterations = ph["impratio"], ph["iterations"], ph["ls_iterations"]
    o.tolerance, o.noslip_iterations = ph["tolerance"], ph["noslip_iterations"]
    count = {}
    for g in spec.geoms:   # names for the unnamed collision geoms (pair metadata and reports); physics unchanged
        if not g.name and (g.contype or g.conaffinity):
            body = g.parent.name
            g.name = f"{body}_collision{count.get(body, 0)}"
            count[body] = count.get(body, 0) + 1
    base = spec.body("base")   # the Menagerie base is visual only: its meshes collide as convex hulls (mass 0)
    for k, g in enumerate([g for g in spec.geoms if g.parent.name == "base" and g.type == mujoco.mjtGeom.mjGEOM_MESH]):
        base.add_geom(name=f"base_hull{k}", type=g.type, meshname=g.meshname, pos=g.pos, quat=g.quat, group=3,
                      contype=1, conaffinity=1, condim=3, mass=0, rgba=[1, 0, 0, 1])
    t, c = cfg["table"], cfg["cube"]
    box = mujoco.mjtGeom.mjGEOM_BOX
    hx, hy = 0.5 * np.asarray(t["size_xy_m"])
    tab = spec.worldbody.add_geom(name="table", type=box, size=[hx, hy, 0.5 * t["thickness_m"]],
                                  pos=[*t["center_xy_m"], -0.5 * t["thickness_m"]], rgba=[0.55, 0.5, 0.45, 1])
    half = 0.5 * c["size_m"]
    body = spec.worldbody.add_body(name="cube", pos=[*c["park_xy_m"], half])
    body.add_freejoint(name="cube")
    cube = body.add_geom(name="cube", type=box, size=[half] * 3, mass=c["mass_kg"], rgba=[0.15, 0.35, 0.85, 1])
    for geom, p in ((tab, t), (cube, c)):
        geom.friction, geom.condim, geom.solref, geom.solimp = p["friction"], p["condim"], p["solref"], p["solimp"]
    spec.add_exclude(bodyname1="world", bodyname2="base")
    # the base is welded to the world, so MuJoCo's parent-child filter skips it: the shoulder turns inside its hulls
    spec.add_exclude(bodyname1="base", bodyname2="shoulder")
    spec.worldbody.add_light(pos=[0.3, -0.3, 1.2], dir=[-0.2, 0.2, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    return spec


def topdown(yaw):
    """Gripper orientation for a top-down grasp: gripper z up (fingertips down), closing axis x at the yaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def mat_quat(r):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(r, dtype=np.float64).ravel())
    return q


def quat_mat(q):
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(q, np.float64))
    return r.reshape(3, 3)


def rotvec(r):
    """Rotation vector of a rotation matrix."""
    q = mat_quat(r)
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q if q[0] >= 0 else -q, 1.0)
    return v


def slerp(r0, r1, a):
    q0, q1 = mat_quat(r0), mat_quat(r1)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    dot = min(1.0, float(np.dot(q0, q1)))
    if dot > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        th = math.acos(dot)
        q = (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)
    return quat_mat(q / np.linalg.norm(q))


def minjerk(n):
    s = np.linspace(0.0, 1.0, n + 1)[1:]
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def random_quat(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def geom_points(m, g):
    """Local points whose convex hull (plus the radius) bounds a collision geom: (points [k, 3], radius)."""
    t, size = int(m.geom_type[g]), m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * size, 0.0
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.zeros((1, 3)), float(size[0])
    if t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        return np.array([[0, 0, -size[1]], [0, 0, size[1]]]), float(size[0])
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        k = m.geom_dataid[g]
        return m.mesh_vert[m.mesh_vertadr[k]: m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(float), 0.0
    raise ValueError(f"geom type {t}")


# ---- overlap of the cube with the visual meshes -----------------------------------------------------------------
def mesh_triangles(m, g):
    """Triangles [k, 3, 3] of a mesh geom in the frame of its body."""
    k = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[k]: m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(float)
    f = m.mesh_face[m.mesh_faceadr[k]: m.mesh_faceadr[k] + m.mesh_facenum[k]]
    return (v @ quat_mat(m.geom_quat[g]).T + m.geom_pos[g])[f]


def parity_inside(tris, lo, h, shape, axis):
    """Cells of a grid (centres lo + h (i + 0.5)) inside a closed triangle mesh: odd number of mesh crossings of the
    grid line along ``axis`` before the cell centre."""
    a, b = [x for x in range(3) if x != axis]
    p, x = tris[:, :, [a, b]], tris[:, :, axis]
    first = np.maximum(np.ceil((p.min(1) - lo[[a, b]]) / h - 0.5).astype(int), 0)
    last = np.minimum(np.floor((p.max(1) - lo[[a, b]]) / h - 0.5).astype(int), np.array([shape[a], shape[b]]) - 1)
    n = np.maximum(last - first + 1, 0)
    count = n[:, 0] * n[:, 1]
    tri = np.repeat(np.arange(len(tris)), count)   # every (triangle, grid line) pair of the triangle's bounding box
    off = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    j, k = first[tri, 0] + off // n[tri, 1], first[tri, 1] + off % n[tri, 1]
    ya, yb = lo[a] + h * (j + 0.5), lo[b] + h * (k + 0.5)
    p0, p1, p2 = p[tri, 0], p[tri, 1], p[tri, 2]
    det = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    ok = np.abs(det) > 1e-18
    det = np.where(ok, det, 1.0)
    w1 = ((ya - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (yb - p0[:, 1])) / det
    w2 = ((p1[:, 0] - p0[:, 0]) * (yb - p0[:, 1]) - (ya - p0[:, 0]) * (p1[:, 1] - p0[:, 1])) / det
    ok &= (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)
    cross = (1 - w1 - w2) * x[tri, 0] + w1 * x[tri, 1] + w2 * x[tri, 2]
    i = np.maximum(np.ceil((cross - lo[axis]) / h - 0.5).astype(int), 0)   # first cell centre after the crossing
    ok &= i < shape[axis]
    toggles = np.zeros(shape, np.int32)
    index = [None] * 3
    index[axis], index[a], index[b] = i[ok], j[ok], k[ok]
    np.add.at(toggles, tuple(index), 1)
    return np.cumsum(toggles, axis=axis) % 2 == 1


def surface_points(tris, spacing):
    """Vertices of the triangles plus points spread over the triangles larger than ``spacing``."""
    pts = [tris.reshape(-1, 3)]
    n = np.ceil(np.linalg.norm(tris[:, [1, 2, 0]] - tris, axis=-1).max(1) / spacing).astype(int)
    for k in np.unique(n[n > 1]):
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = i + j <= k
        a, b = i[keep] / k, j[keep] / k
        pts.append(np.einsum("sk,tkd->tsd", np.stack((1 - a - b, a, b), 1), tris[n == k]).reshape(-1, 3))
    return np.concatenate(pts)


def box_surface(half, n):
    """Points on the surface of a cube (an n x n grid on each face, corners and edges included), cube frame."""
    s = np.linspace(-half, half, n)
    u, v = (x.ravel() for x in np.meshgrid(s, s, indexing="ij"))
    out = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            p = np.zeros((len(u), 3))
            p[:, axis] = sign * half
            p[:, [x for x in range(3) if x != axis]] = np.stack((u, v), 1)
            out.append(p)
    return np.unique(np.round(np.concatenate(out), 9), axis=0)


class VisualOverlap:
    """Overlap of the cube with the visual meshes of one arm body (the gripper or the moving jaw; the parts the real
    cube cannot enter), measured two ways in the body frame: (a) mesh surface points (the vertices and points 2 mm
    apart on large triangles, one per 0.5 mm cell) below the cube surface, (b) points on the cube surface below the
    mesh surface (a signed distance grid of the solid meshes, 0.5 mm cells, inside by majority of the crossing
    parities along x, y and z; ``parity_disagreements`` counts the cells where the three differ: 0 for closed
    meshes). The depth of an overlap is the larger of the two."""

    def __init__(self, m, body, h=0.0005, spacing=0.002, cell=0.008):
        from scipy import ndimage
        self.map = ndimage.map_coordinates
        geoms = [g for g in np.flatnonzero(m.geom_bodyid == body) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                 and not (m.geom_contype[g] or m.geom_conaffinity[g])]
        tris = np.concatenate([mesh_triangles(m, g) for g in geoms])
        v = tris.reshape(-1, 3)
        self.lo, self.hi, self.h = v.min(0) - 2 * h, v.max(0) + 2 * h, h
        shape = tuple(np.ceil((self.hi - self.lo) / h).astype(int))
        votes = sum(parity_inside(tris, self.lo, h, shape, a).astype(np.int8) for a in range(3))
        inside = votes >= 2
        self.parity_disagreements = int(((votes == 1) | (votes == 2)).sum())
        self.sdf = np.where(inside, ndimage.distance_transform_edt(inside) * h - h / 2,
                            h / 2 - ndimage.distance_transform_edt(~inside) * h).astype(np.float32)
        pts = surface_points(tris, spacing)
        _, keep = np.unique(np.floor((pts - self.lo) / h).astype(np.int64), axis=0, return_index=True)
        pts = pts[np.sort(keep)]
        key = np.floor((pts - self.lo) / cell).astype(np.int64)
        self.cell, self.ncell = cell, key.max(0) + 1
        flat = np.ravel_multi_index(key.T, self.ncell)
        order = np.argsort(flat, kind="stable")
        self.points = pts[order]
        self.start = np.searchsorted(flat[order], np.arange(np.prod(self.ncell) + 1))
        self.meshes = [m.mesh(m.geom_dataid[g]).name for g in geoms]

    def depth(self, rb, pb, rc, c, half, samples):
        """Overlap depth (m) of a cube (centre c, rotation rc, half size) with the meshes of the body at (pb, rb)."""
        r_bc, cb = rb.T @ rc, (c - pb) @ rb          # cube rotation and centre in the body frame
        ext = half * np.abs(r_bc).sum(1)             # half extents of the cube's bounding box in the body frame
        if ((cb + ext) < self.lo).any() or ((cb - ext) > self.hi).any():
            return 0.0
        first = np.maximum(np.floor((cb - ext - self.lo) / self.cell).astype(int), 0)
        last = np.minimum(np.floor((cb + ext - self.lo) / self.cell).astype(int), self.ncell - 1)
        out = 0.0
        if (first <= last).all():
            parts = []
            for i in range(first[0], last[0] + 1):
                for j in range(first[1], last[1] + 1):
                    a = np.ravel_multi_index((i, j, first[2]), self.ncell)
                    parts.append(self.points[self.start[a]: self.start[a + last[2] - first[2] + 1]])
            p = np.concatenate(parts)
            if len(p):
                out = float((half - np.abs((p - cb) @ r_bc)).min(1).max())
        u = ((samples @ r_bc.T + cb) - self.lo) / self.h - 0.5
        return max(out, float(self.map(self.sdf, u.T, order=1, mode="nearest").max()), 0.0)


class Scene:
    """One simulation (MjData ``d``) and one kinematics workspace (``dk``, for IK and planning)."""

    def __init__(self, cfg, physics=None):
        self.cfg = cfg
        self.spec = build_spec(cfg, physics)
        self.m = m = self.spec.compile()
        self.d, self.dk = mujoco.MjData(m), mujoco.MjData(m)
        self.qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in JOINTS])
        self.vadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in JOINTS])
        self.act = np.array([m.actuator(j).id for j in JOINTS])
        self.lo, self.hi = m.actuator_ctrlrange[self.act].T
        self.jlo, self.jhi = m.jnt_range[[m.joint(j).id for j in JOINTS]].T
        # scripted targets stay this far inside the joint limits (a servo that overshoots a target at the limit
        # runs into the soft joint-limit constraint)
        margin = float(cfg["joint_target_margin_rad"])
        self.tlo, self.thi = np.maximum(self.jlo + margin, self.lo), np.minimum(self.jhi - margin, self.hi)
        self.cq = m.jnt_qposadr[m.joint("cube").id]
        self.cv = m.jnt_dofadr[m.joint("cube").id]
        self.cube_body, self.gripper = m.body("cube").id, m.body("gripper").id
        self.jaw_body = m.body("moving_jaw_so101_v1").id
        self.half = 0.5 * cfg["cube"]["size_m"]
        self.grasp_point = np.asarray(cfg["grasp_point_m"], float)
        if m.opt.cone != mujoco.mjtCone.mjCONE_ELLIPTIC:
            raise ValueError("the contact labels read the normal force of elliptic cones")
        dt = m.opt.timestep
        self.nsub = round(cfg["record_step_s"] / dt)
        if abs(self.nsub * dt - cfg["record_step_s"]) > 1e-9:
            raise ValueError("the physics step must divide the record step")
        self.per_ctrl = round(cfg["control_step_s"] / cfg["record_step_s"])
        self.thr = float(cfg["contact_threshold_n"])
        cls = np.zeros(m.ngeom, np.int64)
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            b = m.geom_bodyid[g]
            cls[g] = {self.gripper: FIXED, self.jaw_body: MOVING, self.cube_body: CUBE, 0: TABLE,
                      m.body("base").id: BASE}.get(b, LINK)
        self.cls = cls
        self.palm = np.zeros(m.ngeom, bool)   # palm, servo and hinge blocks of the jaws (inside pairs 0 and 1)
        self.palm[[m.geom(g).id for g in cfg["palm_geoms"]]] = True
        # contact class of two geom classes: 0-2 the pairs, 3 arm-table, 4 cube-other link, 5 arm self, 6 cube-base,
        # -1 none (base-table and base-shoulder are excluded)
        kind = np.full((7, 7), -1, np.int64)
        for a in (FIXED, MOVING, LINK, BASE):
            kind[a, TABLE] = kind[TABLE, a] = 3
            for b in (FIXED, MOVING, LINK, BASE):
                kind[a, b] = 5
        kind[LINK, CUBE] = kind[CUBE, LINK] = 4
        kind[BASE, CUBE] = kind[CUBE, BASE] = 6
        kind[FIXED, CUBE] = kind[CUBE, FIXED] = 0
        kind[MOVING, CUBE] = kind[CUBE, MOVING] = 1
        kind[CUBE, TABLE] = kind[TABLE, CUBE] = 2
        self.kind = kind
        t = cfg["table"]
        self.table_lo = np.asarray(t["center_xy_m"]) - 0.5 * np.asarray(t["size_xy_m"]) + t["edge_margin_m"]
        self.table_hi = np.asarray(t["center_xy_m"]) + 0.5 * np.asarray(t["size_xy_m"]) - t["edge_margin_m"]
        self.park = np.array([*cfg["cube"]["park_xy_m"], self.half])
        # the shoulder turns about the vertical pan axis only: its boxes stay 24 mm above the table
        self.arm_geoms = np.flatnonzero(np.isin(cls, (FIXED, MOVING, LINK)) & (m.geom_bodyid != m.body("shoulder").id))
        self.points = [geom_points(m, g) for g in self.arm_geoms]
        th = cfg["throw_region"]
        self.throw_lo, self.throw_hi = np.array([th["x_m"][0], th["y_m"][0]]), np.array([th["x_m"][1], th["y_m"][1]])
        ov = cfg["visual_overlap"]
        self.overlap_thr = float(ov["threshold_m"])
        self.overlap_bodies = (self.gripper, self.jaw_body)
        self.cube_points = box_surface(self.half, int(ov["cube_points_per_edge"]))
        self._overlap = None

    @property
    def overlap(self):
        """{body: VisualOverlap} of the gripper and the moving jaw (built on first use, about 1.5 s)."""
        if self._overlap is None:
            ov = self.cfg["visual_overlap"]
            self._overlap = {b: VisualOverlap(self.m, b, ov["grid_m"], ov["point_spacing_m"])
                             for b in self.overlap_bodies}
        return self._overlap

    def visual_overlap(self, d):
        """Overlap (m) of the cube with the visual meshes of the gripper and of the moving jaw beyond the deepest
        penetration of that body's collision geoms into the cube (MuJoCo contacts), the larger of the two. ``d`` holds
        the kinematics and the contacts of one state (after mj_step: of the state at the start of that step)."""
        c, rc = d.xpos[self.cube_body], d.xmat[self.cube_body].reshape(3, 3)
        n = d.ncon
        g1, g2, dist = d.contact.geom1[:n], d.contact.geom2[:n], d.contact.dist[:n]
        out = 0.0
        for b, cls in zip(self.overlap_bodies, (FIXED, MOVING)):
            depth = self.overlap[b].depth(d.xmat[b].reshape(3, 3), d.xpos[b], rc, c, self.half, self.cube_points)
            if depth <= out:
                continue
            k =((self.cls[g1] == cls) & (self.cls[g2] == CUBE)) | ((self.cls[g2] == cls) & (self.cls[g1] == CUBE))
            out = max(out, depth - max(0.0, float(-dist[k].min())) if k.any() else depth)
        return out

    def overlap_at(self, qpos):
        """visual_overlap of a full qpos (kinematics and collision in the planning workspace)."""
        dk = self.dk
        dk.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, dk)
        mujoco.mj_collision(self.m, dk)
        return self.visual_overlap(dk)

    # ---- contacts and state ------------------------------------------------------------------------------
    def contact_forces(self, d=None):
        """Contacts of the last physics step (MuJoCo computes them from the state at the start of the step) ->
        summed normal force [9] by class (the 3 pairs, arm-table, cube-link, self, cube-base, then the palm blocks
        of pairs 0 and 1) and the deepest penetration [3] of the 3 pairs (m)."""
        d = self.d if d is None else d
        n = d.ncon
        out, depth = np.zeros(9), np.zeros(3)
        if not n:
            return out, depth
        c = d.contact
        a = c.efc_address[:n]
        f = np.where(a >= 0, d.efc_force[np.maximum(a, 0)], 0.0)
        g1, g2 = c.geom1[:n], c.geom2[:n]
        k = self.kind[self.cls[g1], self.cls[g2]]
        ok = k >= 0
        out[:7] = np.bincount(k[ok], weights=f[ok], minlength=7)
        palm = ok & (k <= 1) & (self.palm[g1] | self.palm[g2])
        out[7:] = np.bincount(k[palm], weights=f[palm], minlength=2)
        pair = ok & (k <= 2) & (f > 0)
        np.maximum.at(depth, k[pair], -c.dist[:n][pair])
        return out, depth

    def state(self):
        """(arm [12], cube [13]) of the simulation; the cube angular velocity rotated to the world frame."""
        d, cq, cv = self.d, self.cq, self.cv
        q = d.qpos[cq + 3: cq + 7]
        w = quat_mat(q) @ d.qvel[cv + 3: cv + 6]
        arm = np.concatenate((d.qpos[self.qadr], d.qvel[self.vadr]))
        return arm, np.concatenate((d.qpos[cq: cq + 3], q, d.qvel[cv: cv + 3], w))

    def cube_in_gripper(self):
        """Cube centre in the grasp frame (gripper frame, origin at the grasp point) of the simulation state."""
        dk = self.dk
        dk.qpos[:] = self.d.qpos
        mujoco.mj_kinematics(self.m, dk)
        r = dk.xmat[self.gripper].reshape(3, 3)
        return r.T @ (dk.xpos[self.cube_body] - dk.xpos[self.gripper]) - self.grasp_point

    def reset(self, q_arm, cube_pos, cube_quat, cube_vel=None, cube_omega=None, spawn=False):
        """Arm at q_arm (targets = q_arm) and the cube resting at its start (or parked, for a spawn) for the
        warm-up; a spawned cube is then placed with its world velocity and angular velocity. Time starts at 0."""
        m, d, cq, cv = self.m, self.d, self.cq, self.cv
        mujoco.mj_resetData(m, d)
        d.qpos[self.qadr] = q_arm
        d.ctrl[self.act] = np.clip(q_arm, self.tlo, self.thi).astype(np.float32)
        d.qpos[cq: cq + 3] = self.park if spawn else cube_pos
        d.qpos[cq + 3: cq + 7] = [1, 0, 0, 0] if spawn else cube_quat
        for _ in range(round(self.cfg["warmup_s"] / m.opt.timestep)):
            mujoco.mj_step(m, d)
        if spawn:
            d.qpos[cq: cq + 3], d.qpos[cq + 3: cq + 7] = cube_pos, cube_quat
            d.qvel[cv: cv + 3] = cube_vel
            d.qvel[cv + 3: cv + 6] = quat_mat(cube_quat).T @ np.asarray(cube_omega, float)
        d.time = 0.0
        mujoco.mj_forward(m, d)

    # ---- kinematics -----------------------------------------------------------------------------------------
    def fk(self, q, cube=None):
        dk = self.dk
        dk.qpos[self.qadr] = q
        dk.qpos[self.cq: self.cq + 3] = self.park if cube is None else cube
        dk.qpos[self.cq + 3: self.cq + 7] = [1, 0, 0, 0]
        mujoco.mj_kinematics(self.m, dk)
        mujoco.mj_comPos(self.m, dk)
        return dk

    def grasp_pose(self, q):
        dk = self.fk(q)
        r = dk.xmat[self.gripper].reshape(3, 3).copy()
        return dk.xpos[self.gripper] + r @ self.grasp_point, r

    def ik(self, q0, pos, rot=None, body=None, point=None, w_rot=0.1, iters=150):
        """Damped least squares (mj_jac) on the 5 arm joints; the jaw stays at q0[5]. Target: the world position of
        ``point`` (body frame of ``body``; default the grasp point of the gripper) and the gripper orientation
        ``rot``; the rotation weight w_rot (m/rad) trades orientation against position.
        -> (q, position error m, angle error rad)."""
        m = self.m
        body = self.gripper if body is None else body
        point = self.grasp_point if point is None else np.asarray(point, float)
        q = np.array(q0, float)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        cols = self.vadr[:5]
        lo, hi = self.tlo[:5], self.thi[:5]

        def error():
            dk = self.fk(q)
            r = dk.xmat[self.gripper].reshape(3, 3)
            rb = dk.xmat[body].reshape(3, 3)
            p = dk.xpos[body] + rb @ point
            er = rotvec(rot @ r.T) if rot is not None else np.zeros(3)
            return pos - p, er, p

        last = np.inf
        for it in range(iters):
            ep, er, p = error()
            err = np.linalg.norm(ep) + w_rot * np.linalg.norm(er)
            if np.linalg.norm(ep) < 1e-6 and np.linalg.norm(er) < 1e-5:
                break
            if it % 10 == 0:   # stop when 10 iterations gain less than 10 %
                if err > 0.9 * last:
                    break
                last = err
            mujoco.mj_jac(m, self.dk, jp, jr, p, body)
            jac = np.vstack((jp[:, cols], w_rot * jr[:, cols]))
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), np.concatenate((ep, w_rot * er)))
            q[:5] = np.clip(q[:5] + dq, lo, hi)
        ep, er, _ = error()
        return q, float(np.linalg.norm(ep)), float(np.linalg.norm(er))

    def pose_ik(self, q0, pos, rot, max_tilt=None):
        """The gripper at (pos, rot) within the IK tolerances; if that is out of reach and max_tilt (rad) is given,
        the position exactly and the orientation within max_tilt (the wrist-flex limit allows a vertical gripper
        only up to a grasp-point height of about 7 cm). -> (q, angle error) or None."""
        tol_p, tol_a = self.cfg["ik"]["position_tol_m"], math.radians(self.cfg["ik"]["angle_tol_deg"])
        q, ep, ea = self.ik(q0, pos, rot=rot)
        if ep < tol_p and ea < tol_a:
            return q, ea
        if max_tilt is not None:
            q, ep, ea = self.ik(q, pos, rot=rot, w_rot=0.001, iters=600)
            if ep < tol_p and ea < max_tilt:
                return q, ea
        return None

    def solve(self, pos, rot, jaw, hint=None, max_tilt=None):
        """IK with a few starting guesses; raises PlanError if the pose is out of reach."""
        px, py = self.cfg["workspace"]["pan_axis_xy_m"]
        pan = math.atan2(pos[1] - py, pos[0] - px)
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        guesses = [] if hint is None else [np.array(hint, float)]
        guesses += [np.array([pan, lift, 0.6, 1.1, wrap(yaw - pan), jaw]) for lift in (0.0, -0.6, 0.6)]
        for tilt in ((None,) if max_tilt is None else (None, max_tilt)):
            for g in guesses:
                g[5] = jaw
                out = self.pose_ik(g, pos, rot, tilt)
                if out is not None:
                    return out[0]
        raise PlanError("pose out of reach")

    def lowest(self, dk, which=None):
        """Lowest world z of the arm collision geoms (or of the geoms of one body) in a kinematics workspace."""
        low = np.inf
        for g, (pts, rad) in zip(self.arm_geoms, self.points):
            if which is None or self.m.geom_bodyid[g] == which:
                low = min(low, float((pts @ dk.geom_xmat[g].reshape(3, 3).T)[:, 2].min() + dk.geom_xpos[g, 2] - rad))
        return low

    def clearance(self, q, cube=None):
        """Lowest point of the arm collision geoms above the table and arm self-contact at q."""
        dk = self.fk(q, cube)
        mujoco.mj_collision(self.m, dk)
        n = dk.ncon
        k = self.kind[self.cls[dk.contact.geom1[:n]], self.cls[dk.contact.geom2[:n]]] if n else np.zeros(0, int)
        return self.lowest(dk), bool((k == 5).any())

    def cube_touches(self, q, pos, quat):
        """True if a cube placed at (pos, quat) would touch the arm or the table at q, or overlap the visual meshes of
        the gripper or the moving jaw."""
        dk = self.fk(q, pos)
        dk.qpos[self.cq + 3: self.cq + 7] = quat
        mujoco.mj_kinematics(self.m, dk)
        mujoco.mj_collision(self.m, dk)
        n = dk.ncon
        if n and ((self.cls[dk.contact.geom1[:n]] == CUBE) | (self.cls[dk.contact.geom2[:n]] == CUBE)).any():
            return True
        return self.visual_overlap(dk) > 0

    def jaw_profile(self, theta, band=0.03):
        """Grasp-frame geometry of both jaws at a jaw angle (gripper top-down): for each jaw the (x, z) of its
        lowest collision point and the x range of its collision points within ``band`` above the lowest point of
        the gripper (the height of a cube on the table under the fingertips)."""
        q = np.zeros(6)
        q[5] = theta
        dk = self.fk(q)
        r, x0 = dk.xmat[self.gripper].reshape(3, 3), dk.xpos[self.gripper]
        pts = {}
        for g, (p, rad) in zip(self.arm_geoms, self.points):
            b = self.m.geom_bodyid[g]
            if b in (self.gripper, self.jaw_body):
                local = ((p @ dk.geom_xmat[g].reshape(3, 3).T + dk.geom_xpos[g]) - x0) @ r - self.grasp_point
                local[:, 2] -= rad
                pts.setdefault("fixed" if b == self.gripper else "moving", []).append(local)
        pts = {k: np.concatenate(v) for k, v in pts.items()}
        z0 = min(v[:, 2].min() for v in pts.values())
        out = {}
        for k, v in pts.items():
            low = v[np.argmin(v[:, 2])]
            inband = v[v[:, 2] < z0 + band]
            out[k] = dict(x_low=float(low[0]), z_low=float(low[2]), x_min=float(inband[:, 0].min()),
                          x_max=float(inband[:, 0].max()))
        out["z_low"] = float(z0)
        return out

    # ---- sampling ----------------------------------------------------------------------------------------
    def sample_cube(self, rng):
        ws = self.cfg["workspace"]
        r, a = rng.uniform(*ws["r_m"]), math.radians(rng.uniform(*ws["angle_deg"]))
        px, py = ws["pan_axis_xy_m"]
        return np.array([px + r * math.cos(a), py + r * math.sin(a), self.half]), rng.uniform(0, math.pi / 2)

    def grasp_yaw(self, pos, cube_yaw, rng):
        """A closing axis along a cube face normal, near the radial direction (wrist roll near 0) or a quarter turn
        or a half turn from it (other wrist rolls)."""
        px, py = self.cfg["workspace"]["pan_axis_xy_m"]
        a = math.atan2(pos[1] - py, pos[0] - px)
        base = cube_yaw + round((a - cube_yaw) / (math.pi / 2)) * math.pi / 2
        yaws = [wrap(base + j * math.pi / 2) for j in (-1, 0, 1, 2)]
        p = np.array([0.2, 0.4, 0.2, 0.2]) * [abs(wrap(y - a)) < 2.5 for y in yaws]   # wrist roll within range
        return yaws[rng.choice(4, p=p / p.sum())]

    def sample_above(self, rng, z=(0.08, 0.2), r=(0.12, 0.27), angle=(-80, 80), jaw=(0.0, 1.2)):
        """A collision-free top-down pose above the table (start poses, free moves)."""
        px, py = self.cfg["workspace"]["pan_axis_xy_m"]
        for _ in range(50):
            rr, a = rng.uniform(*r), math.radians(rng.uniform(*angle))
            pos = np.array([px + rr * math.cos(a), py + rr * math.sin(a), rng.uniform(*z)])
            try:
                q = self.solve(pos, topdown(a + rng.uniform(-1.2, 1.2)), rng.uniform(*jaw), max_tilt=math.radians(60))
            except PlanError:
                continue
            low, self_hit = self.clearance(q)
            if low > 0.02 and not self_hit:
                return q
        raise PlanError("no start pose")

    def sample_free(self, rng, shrink=0.85, low_m=0.03):
        """A random joint configuration (ranges shrunk about their centre) clear of the table and of self-contact."""
        mid, span = 0.5 * (self.jlo + self.jhi), 0.5 * (self.jhi - self.jlo) * shrink
        for _ in range(200):
            q = mid + span * rng.uniform(-1, 1, 6)
            low, self_hit = self.clearance(q)
            if low > low_m and not self_hit:
                return q
        raise PlanError("no free pose")

    def path_clear(self, q0, q1, low_m=0.015, n=8):
        for a in np.linspace(0, 1, n + 1)[1:]:
            low, self_hit = self.clearance(q0 + (q1 - q0) * a)
            if low < low_m or self_hit:
                return False
        return True


# ---- scripted plans ---------------------------------------------------------------------------------------
class Plan:
    """Joint targets on the 20 ms control grid, with a phase per control step."""

    def __init__(self, scene, q, rng):
        self.s, self.rng = scene, rng
        self.q = np.array(q, float)
        self.rows, self.phases = [], []
        self.dt = scene.cfg["control_step_s"]
        sp = scene.cfg["speed_rad_s"]
        self.arm_max, self.jaw_speed = sp["arm_max"], sp["jaw"]

    def steps(self, seconds):
        return max(1, int(round(seconds / self.dt)))

    def __len__(self):
        return len(self.phases)

    def add(self, qs, phase):
        qs = np.asarray(qs, float)
        self.rows.append(qs)
        self.phases += [P[phase]] * len(qs)
        self.q = qs[-1].copy()

    def hold(self, seconds, phase):
        if seconds > 0.5 * self.dt:
            self.add(np.repeat(self.q[None], self.steps(seconds), 0), phase)

    def joint(self, q1, phase, speed=None, tmin=0.2):
        """Minimum-jerk joint-space move; the duration keeps the peak arm joint speed at ``speed`` (rad/s)."""
        q1 = np.array(q1, float)
        speed = self.rng.uniform(*self.s.cfg["speed_rad_s"]["arm"]) if speed is None else speed
        dq = np.abs(q1 - self.q)
        t = max(tmin, 1.875 * dq[:5].max() / speed, 1.875 * dq[5] / self.jaw_speed)
        s = minjerk(self.steps(t))
        self.add(self.q + (q1 - self.q) * s[:, None], phase)

    def jaw(self, theta, phase, seconds):
        q1 = self.q.copy()
        q1[5] = theta
        s = minjerk(self.steps(max(seconds, 1.875 * abs(theta - self.q[5]) / self.jaw_speed)))
        self.add(self.q + (q1 - self.q) * s[:, None], phase)

    def cart(self, pos, rot, phase, seconds, max_tilt=math.radians(45)):
        """Straight line of the grasp point (orientation by slerp; tilted up to max_tilt where the wrist cannot keep
        it), minimum-jerk timing, IK at every control step; slower if a joint would pass the arm speed limit."""
        p0, r0 = self.s.grasp_pose(self.q)
        pos = np.asarray(pos, float)
        for _ in range(6):
            rows, q = [], self.q.copy()
            for a in minjerk(self.steps(seconds)):
                out = self.s.pose_ik(q, p0 + (pos - p0) * a, slerp(r0, rot, a), max_tilt)
                if out is None:
                    raise PlanError("path out of reach")
                q = out[0]
                rows.append(q.copy())
            rows = np.array(rows)
            speed = np.abs(np.diff(np.vstack((self.q[None], rows)), axis=0)[:, :5]).max() / self.dt
            if speed <= self.arm_max:
                self.add(rows, phase)
                return
            seconds *= 1.3
        raise PlanError("path too fast")

    def fill(self, n, rng, free=True):
        """Free moves above the table (or holds) until n control steps."""
        tries = 0
        while free and n - len(self) > self.steps(0.8) and tries < 20:
            tries += 1
            try:
                q1 = self.s.sample_above(rng, z=(0.06, 0.22))
            except PlanError:
                break
            if not self.s.path_clear(self.q, q1):
                continue
            self.joint(q1, "free")
            self.hold(rng.uniform(0, 0.3), "idle")
        if len(self) < n:
            self.hold((n - len(self)) * self.dt, "idle")

    def targets(self, n=None):
        rows = np.concatenate(self.rows) if self.rows else np.zeros((0, 6))
        return rows[:n], np.asarray(self.phases[:n], np.int8)

    def play(self, n=None):
        rows, phases = self.targets(n)
        for q, ph in zip(rows, phases):
            yield q, ph


def grasp_approach(plan, s, rng, target, rot, theta_open, speed=None):
    """Approach above ``target`` (grasp point) with the jaw open, then descend in a straight line."""
    pre = target + [0, 0, rng.uniform(0.04, 0.08)]
    q_pre = s.solve(pre, rot, theta_open, hint=plan.q, max_tilt=math.radians(30))
    if not s.path_clear(plan.q, q_pre, low_m=0.03):
        raise PlanError("approach path")
    plan.joint(q_pre, "approach", speed=speed)
    plan.cart(target, rot, "descend", rng.uniform(0.25, 0.5))


def carry_and_place(plan, s, rng, grasp, rot, h, theta_open, n, move_prob=0.5, lift_s=(0.4, 1.0)):
    """From a grasp at grasp point ``grasp``: lift by h, hold, move while holding (move_prob, if its drawn durations
    fit in the n control steps), lower, release, retreat."""
    plan.cart(grasp + [0, 0, h], rot, "lift", rng.uniform(*lift_s))
    move = rng.random() < move_prob
    hold1, move_s, hold2 = rng.uniform(0.2, 0.5), rng.uniform(0.5, 0.9), rng.uniform(0.1, 0.3)
    rest = [rng.uniform(0.35, 0.7), rng.uniform(0.15, 0.35), rng.uniform(0.25, 0.5)]   # lower, release, retreat
    rest[1] = max(rest[1], 1.875 * abs(theta_open - plan.q[5]) / plan.jaw_speed)
    if not (move and len(plan) + plan.steps(hold1 + move_s + hold2 + sum(rest)) <= n):
        move, hold1 = False, rng.uniform(0.3, 1.0)
    plan.hold(hold1, "hold")
    place, rot2 = grasp, rot
    for _ in range(5 if move else 0):   # a reachable place for the move (else no move)
        target = s.sample_cube(rng)[0] + [0, 0, grasp[2] - s.half]
        turn = topdown(math.atan2(rot[1, 0], rot[0, 0]) + rng.uniform(-0.6, 0.6))
        try:
            s.solve(target, turn, plan.q[5], hint=plan.q)
            plan.cart(target + [0, 0, h], turn, "move", move_s)
        except PlanError:
            continue
        place, rot2 = target, turn
        plan.hold(hold2, "hold")
        break
    plan.cart(place + [0, 0, rng.uniform(0.0, 0.003)], rot2, "lower", rest[0])
    plan.jaw(theta_open, "release", rest[1])
    plan.cart(place + [0, 0, rng.uniform(0.04, 0.1)], rot2, "retreat", rest[2])


def task_start(s, rng, spread_deg=60):
    """A cube on the table and a start pose above the table within spread_deg of pan from it."""
    cube, cyaw = s.sample_cube(rng)
    px, py = s.cfg["workspace"]["pan_axis_xy_m"]
    a = math.degrees(math.atan2(cube[1] - py, cube[0] - px))
    q0 = s.sample_above(rng, angle=(a - spread_deg, a + spread_deg))
    return cube, cyaw, q0, Plan(s, q0, rng)


def resting_cube(cube, cyaw):
    return dict(pos=cube, quat=np.array([math.cos(cyaw / 2), 0, 0, math.sin(cyaw / 2)]), spawn=False)


def finish(plan, rng, n, q0, cube, cyaw, **meta):
    if len(plan) > n:
        raise PlanError("too long")
    plan.fill(n, rng)
    return dict(q0=q0, cube=resting_cube(cube, cyaw), script=plan.play(n), meta=meta)


def fam_nominal(s, rng, n):
    cube, cyaw, q0, plan = task_start(s, rng, spread_deg=40)
    plan.hold(rng.uniform(0, 0.2), "idle")
    rot = topdown(s.grasp_yaw(cube, cyaw, rng))
    theta_open = rng.uniform(0.55, 0.9)
    grasp = cube + [0, 0, rng.uniform(0, 0.002)]
    grasp_approach(plan, s, rng, grasp, rot, theta_open, speed=rng.uniform(1.2, 2.4))
    plan.hold(rng.uniform(0, 0.15), "descend")
    plan.jaw(rng.uniform(s.tlo[5], 0.05), "close", rng.uniform(0.2, 0.4))
    plan.hold(rng.uniform(0.1, 0.25), "grip")
    carry_and_place(plan, s, rng, grasp, rot, rng.uniform(0.03, 0.12), theta_open, n, lift_s=(0.35, 0.8))
    return finish(plan, rng, n, q0, cube, cyaw)


def fam_off_centre(s, rng, n):
    """The cube sits off the grasp point: along the jaw width (4-12 mm), the closing axis (+-3 mm) and lower in the
    jaws (the gripper 0-12 mm higher)."""
    cube, cyaw, q0, plan = task_start(s, rng)
    rot = topdown(s.grasp_yaw(cube, cyaw, rng))
    off = np.array([rng.uniform(-0.003, 0.003), rng.choice([-1, 1]) * rng.uniform(0.004, 0.012), 0.0])
    grasp = cube - rot @ off + [0, 0, rng.uniform(0.0, 0.012)]
    theta_open = rng.uniform(0.6, 0.9)
    grasp_approach(plan, s, rng, grasp, rot, theta_open)
    plan.jaw(rng.uniform(s.tlo[5], 0.05), "close", rng.uniform(0.2, 0.4))
    plan.hold(rng.uniform(0.1, 0.3), "grip")
    carry_and_place(plan, s, rng, grasp, rot, rng.uniform(0.05, 0.12), theta_open, n, move_prob=0.3,
                    lift_s=(0.3, 0.7))
    return finish(plan, rng, n, q0, cube, cyaw, offset_m=(grasp - cube).tolist())


def fam_one_sided(s, rng, n):
    """One jaw touches the cube on the table: a push with the fixed jaw's outer face, a swat of the moving jaw's
    outer face (the jaw opens into the cube) or a fingertip tap on the top face."""
    cube, cyaw, q0, plan = task_start(s, rng)
    rot = topdown(s.grasp_yaw(cube, cyaw, rng))
    ex = rot[:, 0]
    kind = str(rng.choice(["push", "swat", "tap"]))
    if kind == "push":   # the cube outside the fixed jaw; the gripper moves along -x into it
        jp = s.jaw_profile(rng.uniform(0.0, 0.3))
        gap = rng.uniform(0.003, 0.012)
        start = cube - ex * (jp["fixed"]["x_min"] - s.half - gap)
        start[2] = -jp["z_low"] + rng.uniform(0.002, 0.006)
        grasp_approach(plan, s, rng, start, rot, rng.uniform(0.0, 0.3))
        plan.cart(start - ex * (gap + rng.uniform(0.02, 0.06)), rot, "push", rng.uniform(0.5, 1.2))
    elif kind == "swat":   # the cube outside the nearly closed moving jaw; the jaw opens into it
        jp = s.jaw_profile(0.1)
        gap = rng.uniform(0.002, 0.015)
        start = cube - ex * (jp["moving"]["x_max"] + s.half + gap)
        start[2] = -jp["z_low"] + rng.uniform(0.002, 0.006)
        grasp_approach(plan, s, rng, start, rot, 0.1)
        plan.jaw(rng.uniform(0.8, 1.5), "swat", rng.uniform(0.15, 0.4))
    else:   # one jaw's fingertips pressed 0-3 mm onto the top face; the other jaw clear of the cube
        which = "fixed" if rng.random() < 0.5 else "moving"
        theta = rng.uniform(0.3, 0.6) if which == "fixed" else 0.4
        tip = s.jaw_profile(theta)[which]
        top = cube - ex * tip["x_low"] + ex * rng.uniform(-0.008, 0.008)
        top[2] = 2 * s.half - tip["z_low"] - rng.uniform(0.0, 0.003)
        grasp_approach(plan, s, rng, top + [0, 0, 0.004], rot, theta)
        plan.cart(top, rot, "tap", rng.uniform(0.15, 0.4))
        plan.hold(rng.uniform(0.1, 0.5), "tap")
        kind = f"tap_{which}"
    p, _ = s.grasp_pose(plan.q)
    plan.cart(p + [0, 0, rng.uniform(0.04, 0.1)], rot, "retreat", rng.uniform(0.3, 0.6))
    return finish(plan, rng, n, q0, cube, cyaw, kind=kind)


def fam_near_miss(s, rng, n):
    """The jaws close 15-30 mm beside the cube centre along the jaw width, or 22-35 mm along the closing axis."""
    cube, cyaw, q0, plan = task_start(s, rng)
    rot = topdown(s.grasp_yaw(cube, cyaw, rng))
    if rng.random() < 0.7:
        off = np.array([rng.uniform(-0.004, 0.004), rng.choice([-1, 1]) * rng.uniform(0.015, 0.03), 0.0])
    else:
        off = np.array([rng.choice([-1, 1]) * rng.uniform(0.022, 0.035), rng.uniform(-0.004, 0.004), 0.0])
    grasp = cube - rot @ off + [0, 0, rng.uniform(0, 0.003)]
    grasp_approach(plan, s, rng, grasp, rot, rng.uniform(0.6, 0.9))
    plan.jaw(rng.uniform(s.tlo[5], 0.05), "close", rng.uniform(0.2, 0.4))
    plan.hold(rng.uniform(0.1, 0.3), "grip")
    h = rng.uniform(0.04, 0.10)
    plan.cart(grasp + [0, 0, h], rot, "lift", rng.uniform(0.4, 0.8))
    plan.hold(rng.uniform(0.2, 0.6), "hold")
    plan.cart(grasp + [0, 0, rng.uniform(0.0, 0.003)], rot, "lower", rng.uniform(0.4, 0.8))
    plan.jaw(rng.uniform(0.6, 0.9), "release", rng.uniform(0.15, 0.4))
    plan.cart(grasp + [0, 0, h], rot, "retreat", rng.uniform(0.3, 0.6))
    return finish(plan, rng, n, q0, cube, cyaw, offset_m=(grasp - cube).tolist())


def fam_weak_closure(s, rng, n):
    """Slow closing until both jaws carry load, then the jaw target at the measured jaw angle plus a small opening
    (-2..+4 mrad); a fast lift, a hold, a slow opening of the jaw target (50 %, 2-20 mrad/s for 0.4-1 s: the grip
    fades and the cube slips out) and a wrist shake (60 %) when they fit in the episode, lowering and release."""
    cube, cyaw, q0, plan = task_start(s, rng, spread_deg=30)
    rot = topdown(s.grasp_yaw(cube, cyaw, rng))
    theta_open = rng.uniform(0.55, 0.8)
    grasp = cube + [0, 0, rng.uniform(0, 0.004)]
    grasp_approach(plan, s, rng, grasp, rot, theta_open)
    head, q_grasp = len(plan), plan.q.copy()
    delta, close_speed = rng.uniform(-0.002, 0.004), rng.uniform(0.3, 0.6)
    max_close = plan.steps((theta_open - 0.1) / close_speed)   # contact comes near jaw angle 0.19
    after = Plan(s, q_grasp, rng)
    after.hold(rng.uniform(0.1, 0.3), "grip")
    after.cart(grasp + [0, 0, rng.uniform(0.05, 0.12)], rot, "lift", rng.uniform(0.25, 0.5))
    after.hold(rng.uniform(0.2, 0.6), "hold")
    spare = n - head - max_close - len(after) - plan.steps(1.6)   # lowering, release and retreat
    shake, loosen = rng.random() < 0.6, rng.random() < 0.5
    dur, freq, amp = rng.uniform(0.5, 1.0), rng.uniform(1.5, 3.0), rng.uniform(0.1, 0.3)
    loosen_s, loosen = rng.uniform(0.4, 1.0), rng.uniform(0.002, 0.02) if loosen else 0.0
    if loosen and after.steps(loosen_s) > spare:
        loosen = 0.0
    if shake and after.steps(dur) + (after.steps(loosen_s) if loosen else 0) <= spare:   # tapered wrist shake
        t = np.arange(1, after.steps(dur) + 1) * after.dt
        rows = np.repeat(after.q[None], len(t), 0)
        rows[:, 3] += amp * np.sin(2 * math.pi * freq * t) * np.clip((dur - t) / 0.2, 0, 1)
        after.add(rows, "shake")
    if loosen:
        after.hold(loosen_s, "loosen")
    after.cart(grasp + [0, 0, rng.uniform(0.0, 0.003)], rot, "lower", rng.uniform(0.4, 0.8))
    after.hold(rng.uniform(0.15, 0.3), "release")
    after.cart(grasp + [0, 0, 0.06], rot, "retreat", 0.4)
    rows, phases = after.targets()
    if head + max_close + len(rows) > n:
        raise PlanError("too long")
    release = np.flatnonzero(phases == P["release"])
    carry = np.isin(phases, [P[x] for x in ("grip", "lift", "hold", "shake", "lower")])
    opening = np.cumsum(phases == P["loosen"]) * loosen * plan.dt   # jaw target opening during and after loosen
    dt = plan.dt

    def script():
        sent = 0
        for q, ph in plan.play(head):
            obs = yield q, ph
            sent += 1
        q = q_grasp.copy()
        for _ in range(max_close):   # close at close_speed until both jaws carry load
            if obs["force"][0] > s.thr and obs["force"][1] > s.thr:
                break
            q[5] = max(q[5] - close_speed * dt, s.tlo[5])
            obs = yield q.copy(), P["close"]
            sent += 1
        target = min(float(obs["q"][5]) + delta, theta_open)
        jaw = np.where(carry | (phases == P["loosen"]), target + opening, theta_open)
        start = target + opening[release[0]]
        jaw[release] = start + (theta_open - start) * minjerk(len(release))
        out = rows.copy()
        out[:, 5] = jaw
        for row, ph in zip(out, phases):
            obs = yield row, ph
            sent += 1
        tail = Plan(s, out[-1], rng)
        tail.fill(n - sent, rng)
        yield from tail.play()

    return dict(q0=q0, cube=resting_cube(cube, cyaw), script=script(),
                meta=dict(delta_rad=float(delta), close_speed=float(close_speed), loosen_rad_s=float(loosen)))


def fam_drop(s, rng, n):
    """Grasp and lift 5-15 cm, then the jaw opens in 60-200 ms: the cube falls and hits the table."""
    cube, cyaw, q0, plan = task_start(s, rng)
    rot = topdown(s.grasp_yaw(cube, cyaw, rng))
    grasp = cube + [0, 0, rng.uniform(0, 0.002)]
    grasp_approach(plan, s, rng, grasp, rot, rng.uniform(0.55, 0.9))
    plan.jaw(rng.uniform(s.tlo[5], 0.05), "close", rng.uniform(0.2, 0.45))
    plan.hold(rng.uniform(0.1, 0.3), "grip")
    h = rng.uniform(0.05, 0.15)
    plan.cart(grasp + [0, 0, h], rot, "lift", rng.uniform(0.4, 0.9))
    plan.hold(rng.uniform(0.1, 0.5), "hold")
    if rng.random() < 0.5 and len(plan) + plan.steps(2.6) <= n:
        plan.cart(s.sample_cube(rng)[0] + [0, 0, h], rot, "move", rng.uniform(0.5, 1.0))
    plan.jaw(rng.uniform(0.6, 1.5), "open_drop", rng.uniform(0.06, 0.2))
    plan.hold(rng.uniform(0.2, 0.5), "rest")
    p, _ = s.grasp_pose(plan.q)
    plan.cart(p + [0, 0, rng.uniform(0.03, 0.08)], rot, "retreat", rng.uniform(0.3, 0.6))
    return finish(plan, rng, n, q0, cube, cyaw, height_m=float(h))


def fam_push(s, rng, n):
    """Closed jaws, fingertips 2-6 mm above the table, pushed 6-12 cm through the cube position."""
    cube, cyaw, q0, plan = task_start(s, rng)
    phi = rng.uniform(-math.pi, math.pi)
    u = np.array([math.cos(phi), math.sin(phi), 0.0])
    theta = rng.uniform(s.tlo[5], 0.15)
    rot = topdown(rng.uniform(-math.pi, math.pi))
    start = cube - u * rng.uniform(0.04, 0.06)
    start[2] = -s.jaw_profile(theta)["z_low"] + rng.uniform(0.002, 0.006)
    grasp_approach(plan, s, rng, start, rot, theta)
    end = start + u * rng.uniform(0.06, 0.12)
    plan.cart(end, rot, "push", rng.uniform(0.6, 1.5))
    plan.cart(end + [0, 0, rng.uniform(0.04, 0.1)], rot, "retreat", rng.uniform(0.3, 0.6))
    return finish(plan, rng, n, q0, cube, cyaw, direction_rad=float(phi))


# ---- probes -------------------------------------------------------------------------------------------------
def free_motion(s, rng, q0, n, speed=(0.8, 3.0)):
    """Random joint-space moves between collision-free configurations (jaw included), with short holds."""
    plan = Plan(s, q0, rng)
    tries = 0
    while len(plan) < n and tries < 50:
        tries += 1
        try:
            q1 = s.sample_free(rng)
        except PlanError:
            break
        if not s.path_clear(plan.q, q1):
            continue
        plan.joint(q1, "free", speed=rng.uniform(*speed))
        plan.hold(rng.uniform(0, 0.3), "idle")
    plan.fill(n, rng, free=False)
    return plan


def fam_step_arm(s, rng, n):
    """Step commands (servo saturation, no contact): the joint target jumps and is held 0.1-0.6 s. Half of the steps
    move only the jaw, to any angle of its range (also closing from rest at any angle); the others move 1-5 arm joints
    by up to 0.6 rad between collision-free poses (and the jaw with probability 0.5). The cube rests out of reach. The
    task scripts command smooth profiles, so without this probe the core never sees a joint far from its target
    without a contact (e.g. a jaw that a false contact stopped and that must close again)."""
    q0 = s.sample_free(rng)
    plan = Plan(s, q0, rng)
    tries = 0
    while len(plan) < n and tries < 200:
        tries += 1
        q1 = plan.q.copy()
        lo, hi = s.jlo + 0.12, s.jhi - 0.12   # a saturated step overshoots its target: stay 0.12 rad inside the limits
        if rng.uniform() < 0.5:
            q1[5] = rng.uniform(lo[5], hi[5])
        else:
            q1[:5] = q1[:5] + rng.uniform(-0.6, 0.6, 5) * (rng.uniform(size=5) < 0.6)
            if rng.uniform() < 0.5:
                q1[5] = rng.uniform(lo[5], hi[5])
            q1 = np.clip(q1, lo, hi)
            low, self_hit = s.clearance(q1)
            if low < 0.03 or self_hit or not s.path_clear(plan.q, q1):
                continue
        plan.add(np.repeat(q1[None], plan.steps(rng.uniform(0.1, 0.6)), 0), "free")
    plan.fill(n, rng, free=False)
    return dict(q0=q0, cube=dict(pos=s.park.copy(), quat=np.array([1.0, 0, 0, 0]), spawn=False),
                script=plan.play(n), meta={})


def fam_free_arm(s, rng, n):
    q0 = s.sample_free(rng)
    return dict(q0=q0, cube=dict(pos=s.park.copy(), quat=np.array([1.0, 0, 0, 0]), spawn=False),
                script=free_motion(s, rng, q0, n).play(n), meta={})


def thrown_cube(s, rng, q0, clearance, vz, vxy, spin=25.0):
    """A cube in the throw region: random orientation, its lowest corner ``clearance`` above the table."""
    for _ in range(100):
        quat = random_quat(rng)
        extent = np.abs(quat_mat(quat)[2]).sum() * s.half
        pos = np.r_[rng.uniform(s.throw_lo, s.throw_hi), extent + rng.uniform(*clearance)]
        if not s.cube_touches(q0, pos, quat):
            break
    a = rng.uniform(-math.pi, math.pi)
    vel = np.array([math.cos(a), math.sin(a), 0.0]) * rng.uniform(*vxy) + [0, 0, rng.uniform(*vz)]
    w = rng.normal(size=3)
    return dict(pos=pos, quat=quat, vel=vel, omega=w / np.linalg.norm(w) * rng.uniform(0, spin), spawn=True)


def fam_cube_flight(s, rng, n):
    q0 = s.sample_free(rng)
    cube = thrown_cube(s, rng, q0, clearance=(0.005, 0.10), vz=(1.0, 2.2), vxy=(0.0, 0.5))
    return dict(q0=q0, cube=cube, script=free_motion(s, rng, q0, n).play(n), meta={})


def fam_cube_table(s, rng, n):
    q0 = s.sample_free(rng)
    cube = thrown_cube(s, rng, q0, clearance=(0.001, 0.02), vz=(-0.3, 0.3), vxy=(0.1, 1.0))
    return dict(q0=q0, cube=cube, script=free_motion(s, rng, q0, n, speed=(0.5, 1.5)).play(n), meta={})


# Inner faces of the jaws, gripper frame at jaw angle 0: a pad point near the tips, the face normal and the
# direction along the finger from the tip towards its base.
JAW_FACES = {
    "fixed": (np.array([-0.0090, 0.0, -0.0950]), np.array([0.994, 0.0, 0.110]), np.array([-0.110, 0.0, 0.994])),
    "moving": (np.array([0.0090, 0.0, -0.0950]), np.array([-0.997, 0.0, 0.070]), np.array([0.070, 0.0, 0.997]))}


def jaw_face(s, which, along):
    """A point on a jaw's inner face ``along`` metres from the tip pad towards the finger base, and the face frame
    (columns: along the finger, across, normal), in the frame of the jaw's body (the gripper for the fixed jaw)."""
    point, normal, axis = JAW_FACES[which]
    frame = np.stack((axis, np.cross(normal, axis), normal), 1)
    point = point + axis * along
    if which == "fixed":
        return s.gripper, point, frame
    dk = s.fk(np.zeros(6))
    r_g, r_j = dk.xmat[s.gripper].reshape(3, 3), dk.xmat[s.jaw_body].reshape(3, 3)
    return s.jaw_body, r_j.T @ (dk.xpos[s.gripper] + r_g @ point - dk.xpos[s.jaw_body]), r_j.T @ r_g @ frame


def fam_jaw(which):
    def family(s, rng, n):
        """The jaw's inner face held up, 10-20 cm above the table, with a slope of -25..25 degrees along the finger
        (none across it), the other jaw open out of the way. The cube, its faces parallel to the jaw face, is
        released 1-12 mm above it: it lands, rests or slides along the finger; after 0.2-0.8 s a fast turn of one
        arm joint (1.2-2 rad) tips the face over. The cube falls to the table (or onto the arm base, which ends the
        episode) or stays on the gripper against a palm block."""
        px, py = s.cfg["workspace"]["pan_axis_xy_m"]
        for _ in range(200):
            r, a = rng.uniform(0.15, 0.27), math.radians(rng.uniform(-70, 70))
            pos = np.array([px + r * math.cos(a), py + r * math.sin(a), rng.uniform(0.10, 0.20)])
            theta = rng.uniform(1.3, s.thi[5]) if which == "fixed" else rng.uniform(0.9, 1.4)
            body, point, frame = jaw_face(s, which, rng.uniform(0.0, 0.03))
            beta, psi = math.radians(rng.uniform(-25, 25)), rng.uniform(-math.pi, math.pi)
            along = np.array([math.cos(beta) * math.cos(psi), math.cos(beta) * math.sin(psi), math.sin(beta)])
            normal = np.array([-math.sin(beta) * math.cos(psi), -math.sin(beta) * math.sin(psi), math.cos(beta)])
            face_w = np.stack((along, np.cross(normal, along), normal), 1)
            q = np.array([math.atan2(pos[1] - py, pos[0] - px), rng.uniform(-0.5, 0.5), rng.uniform(0, 1.2),
                          rng.uniform(-1.2, 1.2), rng.uniform(-2.5, 2.5), theta])
            dk = s.fk(q)
            r_body_in_g = dk.xmat[s.gripper].reshape(3, 3).T @ dk.xmat[body].reshape(3, 3)
            rot = face_w @ frame.T @ r_body_in_g.T      # gripper orientation that puts the face frame at face_w
            q, ep, ea = s.ik(q, pos, rot=rot, body=body, point=point, w_rot=0.02, iters=400)
            if ep > 0.003 or ea > math.radians(25):
                continue
            low, self_hit = s.clearance(q)
            if low < 0.04 or self_hit:
                continue
            dk = s.fk(q)
            rb = dk.xmat[body].reshape(3, 3)
            face_p, face_r = dk.xpos[body] + rb @ point, rb @ frame
            if face_r[2, 2] < math.cos(math.radians(40)):
                continue
            yaw = rng.uniform(-0.35, 0.35)
            quat = mat_quat(face_r @ topdown(yaw))   # cube faces parallel to the jaw face, turned by yaw
            centre = face_p + face_r[:, 2] * (s.half + 0.0005) + [0, 0, rng.uniform(0.001, 0.012)]
            if s.cube_touches(q, centre, quat) or not drop_hits(s, q, centre, body):
                continue
            # the throw: the joint (shoulder lift, wrist flex or roll) whose axis is most perpendicular to the face
            # normal turns 1.2-2 rad towards its larger range and tips the face over
            axes = dk.xaxis[[s.m.joint(JOINTS[j]).id for j in (1, 3, 4)]]
            j = (1, 3, 4)[int(np.argmin(np.abs(axes @ face_r[:, 2])))]
            room = (s.thi[j] - q[j], s.tlo[j] - q[j])
            q1 = q.copy()
            q1[j] = np.clip(q[j] + max(room, key=abs) / abs(max(room, key=abs)) * rng.uniform(1.2, 2.0),
                            s.tlo[j], s.thi[j])
            if abs(q1[j] - q[j]) < 0.8 or not s.path_clear(q, q1, low_m=0.03):
                continue
            w = rng.normal(size=3)
            cube = dict(pos=centre, quat=quat if quat[0] >= 0 else -quat, vel=np.zeros(3),
                        omega=w / np.linalg.norm(w) * rng.uniform(0, 1), spawn=True)
            plan = Plan(s, q, rng)
            plan.hold(rng.uniform(0.2, 0.8), "rest")
            plan.joint(q1, "throw", speed=rng.uniform(2.0, 3.0))
            plan.hold(rng.uniform(0.2, 0.5), "rest")
            plan.fill(n, rng, free=False)
            return dict(q0=q, cube=cube, script=plan.play(n),
                        meta=dict(kind=which, slope_deg=round(math.degrees(beta), 1)))
        raise PlanError("no jaw pose")
    return family


def drop_hits(s, q, centre, body):
    """A vertical ray from the cube centre: the first arm or table geom below it is on ``body``, within 5 cm."""
    dk = s.fk(q)
    geom = np.zeros(1, np.int32)
    dist = mujoco.mj_ray(s.m, dk, np.asarray(centre, float), np.array([0.0, 0.0, -1.0]),
                         np.array([1, 0, 0, 1, 1, 0], np.uint8), 1, s.cube_body, geom)
    return dist >= 0 and s.m.geom_bodyid[geom[0]] == body and dist < 0.05


FAMILIES = {"nominal": fam_nominal, "off_centre": fam_off_centre, "one_sided": fam_one_sided,
            "near_miss": fam_near_miss, "weak_closure": fam_weak_closure, "drop": fam_drop, "push": fam_push,
            "free_arm": fam_free_arm, "cube_flight": fam_cube_flight, "cube_table": fam_cube_table,
            "jaw_fixed": fam_jaw("fixed"), "jaw_moving": fam_jaw("moving"), "step_arm": fam_step_arm}


# ---- one episode ----------------------------------------------------------------------------------------------
def run_episode(s, ep, n_rec):
    """Simulate a planned episode -> arrays of the contract plus diagnostics (or None after a numerical failure)."""
    m, d = s.m, s.d
    c = ep["cube"]
    s.reset(ep["q0"], c["pos"], c["quat"], c.get("vel"), c.get("omega"), c["spawn"])
    arm = np.zeros((n_rec, 12))
    cube = np.zeros((n_rec, 13))
    action = np.zeros((n_rec, 6))
    phase = np.zeros(n_rec, np.int8)
    rel = np.zeros((n_rec, 3))
    force = np.zeros((n_rec - 1, 3))
    impulse = np.zeros((n_rec - 1, 3))
    palm = np.zeros((n_rec - 1, 2))
    depth = np.zeros((n_rec - 1, 3))
    overlap = np.zeros(n_rec)
    other = np.zeros(4)   # largest unsupported force by kind in the recorded intervals (below the threshold)
    script = ep["script"]
    obs = None
    target, ph = next(script)
    terminal, length = 0, n_rec
    dt = m.opt.timestep
    ctrl_force = np.zeros(3)
    for k in range(n_rec):
        arm[k], cube[k] = s.state()
        rel[k] = s.cube_in_gripper()
        if k % s.per_ctrl == 0 and k > 0:
            obs = dict(q=arm[k, :6], cube=cube[k], force=ctrl_force)
            try:
                target, ph = script.send(obs)
            except StopIteration:
                ph = P["idle"]
            ctrl_force = np.zeros(3)
        u = np.clip(target, s.tlo, s.thi).astype(np.float32)   # stored and applied: the same float32 values
        action[k], phase[k] = u, ph
        if k == n_rec - 1:
            overlap[k] = s.overlap_at(d.qpos)
            if overlap[k] > s.overlap_thr:
                terminal, length = TERMINAL.index("cube_visual_overlap"), k
            break
        d.ctrl[s.act] = u
        fmax, bad = np.zeros(3), np.zeros(4)
        for j in range(s.nsub):
            mujoco.mj_step(m, d)
            if j == 0:   # d holds the kinematics and contacts of record k
                overlap[k] = s.visual_overlap(d)
                if overlap[k] > s.overlap_thr:
                    break
            f, pen = s.contact_forces()
            fmax = np.maximum(fmax, f[:3])
            impulse[k] += f[:3] * dt
            bad = np.maximum(bad, f[3:7])
            palm[k] = np.maximum(palm[k], f[7:])
            depth[k] = np.maximum(depth[k], pen)
        if overlap[k] > s.overlap_thr:   # the cube inside a gripper or jaw part: end at the record before
            terminal, length = TERMINAL.index("cube_visual_overlap"), k
            break
        if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number or not np.isfinite(d.qpos).all():
            return None
        if (bad > s.thr).any():
            terminal, length = 1 + int(np.argmax(bad > s.thr)), k + 1
            break
        other = np.maximum(other, bad)
        p = d.qpos[s.cq: s.cq + 3]
        if (p[:2] < s.table_lo).any() or (p[:2] > s.table_hi).any() or p[2] < -0.01:
            terminal, length = TERMINAL.index("cube_off_table"), k + 1
            break
        force[k] = fmax
        ctrl_force = np.maximum(ctrl_force, fmax)
    script.close()
    L = length
    if L < 2:   # ended at the start: the family plans again
        return "replan"
    cube[:L, 3:7] = export.quat_continuous(cube[:L, 3:7])
    out = dict(states={"arm": arm[:L], "cube": cube[:L]}, action=action[:L],
               contacts=(force[:L - 1] > s.thr).astype(np.uint8),
               per_record=dict(phase=phase[:L], cube_in_grasp=rel[:L].astype(np.float32),
                               visual_overlap=overlap[:L].astype(np.float32)),
               per_interval=dict(contact_force=force[:L - 1].astype(np.float32),
                                 contact_impulse=impulse[:L - 1].astype(np.float32),
                                 palm_force=palm[:L - 1].astype(np.float32),
                                 contact_depth=depth[:L - 1].astype(np.float32)),
               terminal=terminal, other_force=other)
    return out


def outcome(ep, half, family_group):
    """Task outcome from the records: lifted (both jaws on, cube bottom 10 mm above the table, no table contact),
    dropped (after lifting, no jaw contact while the cube is up, before the planned release), slip in the grasp."""
    if family_group == "probe":
        return dict(outcome=0, max_lift_m=0.0, hold_slip_mm=np.nan, carry_slip_mm=np.nan)
    c = ep["contacts"].astype(bool)
    z = ep["states"]["cube"][1:, 2] - half
    ph = ep["per_record"]["phase"][1:]
    gripped = c[:, 0] & c[:, 1] & ~c[:, 2]
    up = gripped & (z > 0.010)
    max_lift = float(z[gripped].max()) if gripped.any() else 0.0
    if not up.any():
        return dict(outcome=1, max_lift_m=max_lift, hold_slip_mm=np.nan, carry_slip_mm=np.nan)
    first = int(np.argmax(up))
    carrying = np.isin(ph, [P[x] for x in ("lift", "hold", "move", "shake", "loosen", "lower")])
    lost = (~c[:, 0] & ~c[:, 1]) & (z > 0.010) & carrying
    lost[:first] = False
    rel = ep["per_record"]["cube_in_grasp"][1:]
    hold = np.flatnonzero((ph == P["hold"]) & gripped)
    if len(hold) > 1:   # the first continuous hold after lifting
        hold = hold[: int(np.argmax(np.diff(hold) > 1)) + 1] if (np.diff(hold) > 1).any() else hold
    hold_slip = float(np.linalg.norm(rel[hold[-1]] - rel[hold[0]]) * 1000) if len(hold) > 1 else np.nan
    carry = np.flatnonzero(carrying & gripped & (np.arange(len(z)) >= first))
    carry_slip = float(np.linalg.norm(rel[carry] - rel[carry[0]], axis=1).max() * 1000) if len(carry) else np.nan
    return dict(outcome=3 if lost.any() else 2, max_lift_m=max_lift, hold_slip_mm=hold_slip, carry_slip_mm=carry_slip)


# ---- the collector ---------------------------------------------------------------------------------------------
_SCENE = None


def _worker_init(cfg_path):
    global _SCENE
    s = Scene(load_config(cfg_path))
    _SCENE = s, system_json(s, s.cfg)


def episode_seed(master, family, index):
    return int(np.random.SeedSequence([int(master), int(family), int(index)]).generate_state(2, np.uint64)[0] >> 1)


def campaign_counts(cfg, campaign, scale=None, probe_scale=None, families=None):
    """Episodes per family of a campaign -> [(family index, count)]: the pilot counts of the config times the
    campaign's task_scale and probe_scale. ``scale`` replaces both factors, ``probe_scale`` the probe factor."""
    c = cfg["campaigns"][campaign]
    tasks = c["task_scale"] if scale is None else scale
    probes = probe_scale if probe_scale is not None else (c["probe_scale"] if scale is None else scale)
    return [(f, max(1, int(round(fam["episodes"] * (probes if fam["group"] == "probe" else tasks)))))
            for f, fam in enumerate(cfg["families"]) if (not families or fam["name"] in families)
            and campaign in fam.get("campaigns", [campaign])]   # a family with "campaigns" runs only in those


def check_seeds(cfg, campaign, counts):
    """The master seeds of the campaigns and of the convergence check differ, and the episode seeds of this run
    overlap neither each other nor those of the other campaigns (at their configured counts) or of the convergence
    check. -> number of episode seeds of this run."""
    masters = {name: c["master_seed"] for name, c in cfg["campaigns"].items()}
    masters["convergence"] = cfg["convergence"]["master_seed"]
    if len(set(masters.values())) != len(masters):
        raise SystemExit(f"the master seeds must differ: {masters}")
    names = [f["name"] for f in cfg["families"]]

    def seeds(master, counts):
        return [episode_seed(master, f, i) for f, n in counts for i in range(n)]
    mine = seeds(masters[campaign], counts)
    if len(set(mine)) != len(mine):
        raise SystemExit(f"repeated episode seeds in campaign {campaign}")
    others = {name: seeds(masters[name], campaign_counts(cfg, name)) for name in cfg["campaigns"] if name != campaign}
    others["convergence"] = seeds(masters["convergence"], [(names.index(k), n) for k, n in
                                                            cfg["convergence"]["episodes"].items()])
    for name, other in others.items():
        if set(mine) & set(other):
            raise SystemExit(f"episode seeds of campaign {campaign} overlap with {name}")
    return len(mine)


def make_episode(s, fam_index, index, master):
    """Plan and simulate one episode (planning samples again until the poses are reachable)."""
    cfg = s.cfg
    fam = cfg["families"][fam_index]
    seed = episode_seed(master, fam_index, index)
    rng = np.random.default_rng(seed)
    n_rec = int(fam["records"])
    n_ctrl = (n_rec - 1) // s.per_ctrl + 1
    tries = 0
    while True:
        tries += 1
        try:
            plan = FAMILIES[fam["name"]](s, rng, n_ctrl)
        except PlanError:
            if tries > 200:
                raise
            continue
        out = run_episode(s, plan, n_rec)
        if out is None:
            return dict(rejected=True, family=fam_index, index=index, seed=seed)
        if isinstance(out, str):   # ended before the second record
            if tries > 200:
                raise PlanError("episodes end at the start")
            continue
        out["meta"] = plan["meta"]
        out["seed"], out["plan_tries"] = seed, tries
        return out


def _work(args):
    """Simulate a chunk of jobs [(family, index, master seed)] and pack them into one shard file -> statistics."""
    k, (jobs, shard) = args
    s, system = _SCENE
    fams = s.cfg["families"]
    episodes, rejected, cpu, wall = [], [], {}, {}
    for fam_index, index, master in jobs:
        t0, c0 = time.perf_counter(), time.process_time()
        r = make_episode(s, fam_index, index, master)
        name = fams[fam_index]["name"]
        cpu[name] = cpu.get(name, 0.0) + time.process_time() - c0
        wall[name] = wall.get(name, 0.0) + time.perf_counter() - t0
        if r.get("rejected"):
            rejected.append(dict(family=name, index=index, seed=r["seed"]))
            continue
        o = outcome(r, s.half, fams[fam_index]["group"])
        episodes.append(dict(states=r["states"], action=r["action"], contacts=r["contacts"],
                             per_record=r["per_record"], per_interval=r["per_interval"],
                             scalars=dict(family=np.int16(fam_index), episode_id=np.int64(-1), group_id=np.int64(-1),
                                          seed=np.int64(r["seed"]), terminal_reason=np.int8(r["terminal"]),
                                          outcome=np.int8(o["outcome"]), max_lift_m=np.float32(o["max_lift_m"]),
                                          hold_slip_mm=np.float32(o["hold_slip_mm"]),
                                          carry_slip_mm=np.float32(o["carry_slip_mm"]),
                                          family_index=np.int64(index), plan_tries=np.int16(r["plan_tries"]),
                                          unsupported_max_n=np.float32(r["other_force"].max()),
                                          variant=str(r["meta"].get("kind", r["meta"].get("jaw", ""))))))
    if episodes:
        export.write_shard(shard, export.pack_episodes(episodes, system))
    return k, dict(shard=str(shard) if episodes else None, episodes=len(episodes), rejected=rejected, cpu_s=cpu,
                   wall_s=wall, records=int(sum(len(e["action"]) for e in episodes)))


def _plain(x):
    return x.item() if hasattr(x, "item") else str(x)


def source_hashes():
    """sha256 of the collector's own source files (git_info covers tracked files only)."""
    here = Path(__file__).resolve().parent
    return {str(f.relative_to(export.REPO_ROOT)): export.sha256_file(f)
            for f in (here / "so101_grasp.py", here / "export.py", here / "validate.py", here / "__init__.py")}


def system_json(s, cfg):
    """system.json of the contract: bodies, pairs with their geoms, the label rule, families and the physics."""
    m = s.m
    arm_links = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "camera_mount",
                 "moving_jaw_so101_v1"]

    def names(c):
        return [m.geom(g).name for g in np.flatnonzero(s.cls == c)]
    jid = [m.joint(j).id for j in JOINTS]
    blocks = set(cfg["palm_geoms"])
    ph = cfg["physics"]
    gripper_geom = m.geom("fixed_jaw_box4").id
    return {
        "name": cfg["name"], "record_step_s": cfg["record_step_s"], "model_step_s": cfg["model_step_s"],
        "control_step_s": cfg["control_step_s"], "target_time_s": cfg["target_time_s"],
        "records_per_episode": cfg["records_per_episode"],
        "frame": "world z up, table top at z = 0, arm base frame = world frame (base at the origin), SI units, "
                 "radians, quaternions wxyz (unit, w >= 0 at the first record, sign continuous)",
        "bodies": [
            {"name": "arm", "kind": "articulated", "moving": True, "state_dim": 12,
             "channels": [f"q_{j}" for j in JOINTS] + [f"qd_{j}" for j in JOINTS],
             "channel_types": ["angle"] * 6 + ["angular_rate"] * 6, "channel_mask": [1] * 12,
             "action_dim": 6, "action_channels": [f"qcmd_{j}" for j in JOINTS], "joint_names": list(JOINTS),
             "links": arm_links,
             "actuators": {"type": "position", "kp": float(m.actuator_gainprm[s.act[0], 0]),
                           "kv": float(-m.actuator_biasprm[s.act[0], 2]),
                           "force_range_nm": m.actuator_forcerange[s.act[0]].tolist(),
                           "ctrl_range_rad": m.actuator_ctrlrange[s.act].tolist()},
             "joints": {"range_rad": m.jnt_range[jid].tolist(), "damping": m.dof_damping[s.vadr].tolist(),
                        "armature": m.dof_armature[s.vadr].tolist(),
                        "frictionloss": m.dof_frictionloss[s.vadr].tolist()},
             "grasp_point_m": cfg["grasp_point_m"],
             "note": "jaw = joint 'gripper' (the moving jaw); the fixed jaw is part of link 'gripper'. Action = the "
                     "applied position targets of the 6 Menagerie actuators (float32, the same values in d.ctrl), "
                     "kept physics_config.joint_limits.scripted_target_margin_rad inside the joint ranges"},
            {"name": "cube", "kind": "rigid", "moving": True, "state_dim": 13,
             "channels": ["x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz"],
             "channel_types": ["position"] * 3 + ["quaternion"] * 4 + ["velocity"] * 3 + ["angular_velocity"] * 3,
             "channel_mask": [1] * 13,
             "geometry": {"size_m": cfg["cube"]["size_m"], "mass_kg": float(m.body_mass[s.cube_body]),
                          "inertia_diag_kgm2": m.body_inertia[s.cube_body].tolist()},
             "note": "centre of the cube (= centre of mass); velocity and angular velocity in the world frame"},
            {"name": "table", "kind": "fixed", "moving": False, "top_z_m": 0.0,
             "size_xy_m": cfg["table"]["size_xy_m"], "center_xy_m": cfg["table"]["center_xy_m"]},
        ],
        "pairs": [[0, 1], [0, 1], [1, 2]],
        "pair_metadata": [
            {"name": PAIRS[0], "geoms": [names(FIXED), ["cube"]],
             "regions": {"fixed_finger": [g for g in names(FIXED) if g not in blocks],
                         "palm_blocks": [g for g in names(FIXED) if g in blocks and g != "gripper_collision0"],
                         "wrist_servo_box": ["gripper_collision0"]},
             "note": "all collision geoms of link 'gripper' against the cube: the fixed finger (pads, tips, finger "
                     "hull), the two palm blocks at the finger base and the wrist-servo box. One pair code for all "
                     "three regions; the per-interval palm_force array gives the palm-block and servo-box share"},
            {"name": PAIRS[1], "geoms": [names(MOVING), ["cube"]],
             "regions": {"moving_finger": [g for g in names(MOVING) if g not in blocks],
                         "hinge_block": [g for g in names(MOVING) if g in blocks]},
             "note": "the moving jaw against the cube: its finger (pads, tips, hulls) and the hinge block at its base "
                     "(palm_force, second column)"},
            {"name": PAIRS[2], "geoms": [["cube"], ["table"]], "note": "cube against the table top"},
        ],
        "target_body": 1,
        "contact_rule": {
            "threshold_n": s.thr,
            "rule": "pair on in record interval k if, in any physics step inside it, the summed mj_contactForce "
                    "normal force of the pair's contacts is above threshold_n (MuJoCo computes the contacts and "
                    "forces of a step from the state at the start of that step)",
            "force": "elliptic cones: the first constraint force of each contact (= mj_contactForce normal)",
            "physics_steps_per_record": s.nsub,
            "unsupported": "any other contact above threshold_n (arm-table, other arm links, the camera mount or the "
                           "arm base with the cube, arm self-contact) ends the episode at the last record before that "
                           "interval; an overlap of the cube with the visual meshes of the gripper or the moving jaw "
                           "(visual_overlap) ends it at the record before the overlap",
            "unsupported_geoms": {"other_arm_links": names(LINK), "arm_base": names(BASE)},
            "visual_overlap": {
                "threshold_m": s.overlap_thr,
                "bodies": [m.body(b).name for b in s.overlap_bodies],
                "meshes": {m.body(b).name: s.overlap[b].meshes for b in s.overlap_bodies},
                "rule": "at every record (the state at the start of the first physics step of the interval, with the "
                        "contacts MuJoCo computes from it): overlap = the larger of (a) the deepest visual-mesh "
                        "surface point of the body below the cube surface and (b) the deepest cube surface point below "
                        "the body's visual-mesh surface, minus the deepest penetration of the body's collision geoms "
                        "into the cube. Above threshold_m the episode ends at the record before (terminal reason "
                        "cube_visual_overlap): the cube is inside a part of the real gripper that has no collision "
                        "shape. Stored per record as visual_overlap (m)",
                "grid_m": s.cfg["visual_overlap"]["grid_m"],
                "point_spacing_m": s.cfg["visual_overlap"]["point_spacing_m"],
                "cube_points_per_edge": s.cfg["visual_overlap"]["cube_points_per_edge"],
                "parity_disagreements": {m.body(b).name: s.overlap[b].parity_disagreements
                                         for b in s.overlap_bodies}}},
        "families": [{k: f[k] for k in ("name", "group", "select", "note")} for f in cfg["families"]],
        "terminal_reasons": list(TERMINAL),
        "phases": list(PHASES),
        "outcomes": list(OUTCOMES),
        "physics_config": {
            "simulator": f"MuJoCo {mujoco.__version__} (CPU, python bindings)",
            "timestep_s": m.opt.timestep, "integrator": ph["integrator"], "cone": ph["cone"],
            "impratio": m.opt.impratio, "solver": ph["solver"], "iterations": m.opt.iterations,
            "ls_iterations": m.opt.ls_iterations, "tolerance": m.opt.tolerance,
            "noslip_iterations": m.opt.noslip_iterations, "gravity": m.opt.gravity.tolist(),
            "jaw_cube_contact": {"from": "Menagerie class collision_gripper (priority 1, used for every jaw contact)",
                                 "condim": int(m.geom_condim[gripper_geom]),
                                 "friction": m.geom_friction[gripper_geom].tolist(),
                                 "solref": m.geom_solref[gripper_geom].tolist(),
                                 "solimp": m.geom_solimp[gripper_geom].tolist()},
            "cube_table_contact": {k: cfg["table"][k] for k in ("condim", "friction", "solref", "solimp")},
            "table": cfg["table"], "cube": cfg["cube"], "warmup_s": cfg["warmup_s"],
            "excluded_contacts": [["world (table)", "base"], ["base", "shoulder"]],
            "joint_limits": {
                "joints": {j: {"range_rad": m.jnt_range[i].tolist(), "limited": bool(m.jnt_limited[i]),
                               "solref": m.jnt_solref[i].tolist(), "solimp": m.jnt_solimp[i].tolist(),
                               "margin_rad": float(m.jnt_margin[i]),
                               "ctrl_range_rad": m.actuator_ctrlrange[s.act[k]].tolist()}
                           for k, (j, i) in enumerate(zip(JOINTS, jid))},
                "scripted_target_margin_rad": s.cfg["joint_target_margin_rad"],
                "target_range_rad": np.stack((s.tlo, s.thi), 1).tolist(),
                "note": "compiled MuJoCo settings: soft joint-limit constraints (MuJoCo default solref and solimp); "
                        "every applied target lies in target_range_rad. A joint can still pass its limit by a few "
                        "mrad under load; the limit constraint then acts like a stiff spring"},
            "base_collision": "convex hulls of the 4 Menagerie base meshes (added: the Menagerie base has no "
                              "collision geoms); a contact with the cube ends the episode (terminal reason cube_base)",
            "model_sha256": hashlib.sha256(s.spec.to_xml().encode()).hexdigest(),
            "menagerie": json.loads((export.REPO_ROOT / cfg["source_manifest"]).read_text())["source"],
        },
    }


def collect(cfg, out_dir, workers, campaign="pilot", families=None, scale=None, probe_scale=None, chunk=16,
            cfg_path=CONFIG):
    """Simulate the episodes of a campaign (``campaign_counts``) on a worker pool. Each pool task simulates ``chunk``
    episodes and writes them as one shard; the shards are merged in job order (family, then episode number) and
    written as the data directory. Pilot and production: splits 0/1 by family; test: all split 2."""
    t_start = time.perf_counter()
    assets = verify_assets(cfg)
    counts = campaign_counts(cfg, campaign, scale, probe_scale, families)
    check_seeds(cfg, campaign, counts)
    master = int(cfg["campaigns"][campaign]["master_seed"])
    jobs = [(f, i, master) for f, n in counts for i in range(n)]
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    s = Scene(cfg)
    system = system_json(s, cfg)
    print(f"campaign {campaign}: master seed {master}, {len(jobs)} episodes, {workers} workers, "
          f"{chunk} episodes per shard", flush=True)
    # The shard folder stays until the data directory is written: a finished shard leaves a stats file next to it,
    # and a run started again skips those shards. A worker that dies (for example killed for lack of memory) stops
    # the run with BrokenProcessPool instead of a silent hang.
    tmp = Path(str(out_dir) + ".shards")
    tmp.mkdir(parents=True, exist_ok=True)
    tasks = [(jobs[k:k + chunk], tmp / f"shard_{k // chunk:06d}.npz") for k in range(0, len(jobs), chunk)]
    stats = [json.loads(t[1].with_suffix(".json").read_text()) if t[1].with_suffix(".json").exists() else None
             for t in tasks]
    todo = [k for k, st in enumerate(stats) if st is None]
    done, every = sum(len(tasks[k][0]) for k in range(len(tasks)) if stats[k] is not None), max(1, len(todo) // 20)
    print(f"{len(tasks) - len(todo)} of {len(tasks)} shards already done", flush=True)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_worker_init,
                             initargs=(str(cfg_path),)) as pool:
        for n, (k, st) in enumerate(pool.map(_work, [(k, tasks[k]) for k in todo])):
            side = tasks[k][1].with_suffix(".json")
            side.with_name(side.name + ".tmp").write_text(json.dumps(st, default=_plain))
            side.with_name(side.name + ".tmp").replace(side)
            stats[k] = json.loads(side.read_text())
            done += len(tasks[k][0])
            if (n + 1) % every == 0 or n + 1 == len(todo):
                el = time.perf_counter() - t_start
                print(f"{done}/{len(jobs)} episodes, {el:.0f} s, {done / el:.2f} episodes/s, "
                      f"about {el * (len(jobs) - done) / done:.0f} s left", flush=True)
    arrays = export.merge_shards([st["shard"] for st in stats if st["shard"]])
    wall = time.perf_counter() - t_start
    n_ep = len(arrays["lengths"])
    arrays["episode_id"] = np.arange(n_ep, dtype=np.int64)
    arrays["group_id"] = np.arange(n_ep, dtype=np.int64)
    arrays["variant"] = arrays["variant"].astype("U16")
    if campaign == "extra":   # supplementary training episodes: all split 0 (the validation set stays that of production)
        arrays["splits"] = np.zeros(len(arrays["family"]), np.int8)
    elif campaign == "test":
        arrays["splits"] = np.full(n_ep, 2, np.int8)
    else:
        arrays["splits"] = export.assign_splits(arrays["family"], arrays["group_id"], seed=master)
    rejected = [r for st in stats for r in st["rejected"]]
    cpu = float(sum(sum(st["cpu_s"].values()) for st in stats))
    busy = float(sum(sum(st["wall_s"].values()) for st in stats))
    transitions = int(((arrays["lengths"] - 1) // 2).sum())
    per_family = {}
    for f, fam in enumerate(cfg["families"]):
        sel = arrays["family"] == f
        if sel.any():
            per_family[fam["name"]] = dict(
                episodes=int(sel.sum()), worker_cpu_s=float(sum(st["cpu_s"].get(fam["name"], 0.0) for st in stats)),
                worker_wall_s=float(sum(st["wall_s"].get(fam["name"], 0.0) for st in stats)),
                transitions_20ms=int(((arrays["lengths"][sel] - 1) // 2).sum()))
    throughput = dict(workers=workers, chunk=chunk, wall_s=wall, worker_cpu_s=cpu, worker_wall_s=busy,
                      load_average=list(os.getloadavg()), cpu_count=os.cpu_count(),
                      episodes_per_wall_s=n_ep / wall, episodes_per_cpu_s=n_ep / cpu,
                      transitions_20ms_per_cpu_s=transitions / cpu,
                      records_10ms_per_cpu_s=float((arrays["lengths"] - 1).sum()) / cpu, per_family=per_family)
    manifest = dict(
        collector="nedm.contact_nrd.collect.so101_grasp", campaign=campaign, master_seed=master,
        episode_counts={cfg["families"][f]["name"]: n for f, n in counts},
        episode_scale=dict(tasks=scale, probes=probe_scale, families=families),
        config=cfg, config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        vendor_files_sha256=assets, collector_source_sha256=source_hashes(), rejected_numerical=rejected,
        throughput=throughput, summary=summarize(system, arrays))
    info = export.write_dataset(out_dir, system, arrays, manifest)
    shutil.rmtree(tmp)
    print(json.dumps({k: info[k] for k in ("episodes", "valid_records", "split_counts")}, indent=1))
    print(json.dumps({k: v for k, v in throughput.items() if k != "per_family"}, indent=1))
    return info


def summarize(system, arrays):
    """Terminations, end states (where the cube is in the last interval of the episodes that run to their end),
    task outcomes, slip, contact-force histogram and label flips (threshold 0 vs 2x), penetration depths, landing speed
    and energy change of cube-table impacts, cube-table label switches while the jaws close and grip, and the
    single-jaw training windows that touch a palm block; records with a joint past its limit and the stored overlap of
    the cube with the visual meshes of the gripper and the moving jaw."""
    fam_names = [f["name"] for f in system["families"]]
    lengths, contacts, family = arrays["lengths"], arrays["contacts"].astype(bool), arrays["family"]
    thr = system["contact_rule"]["threshold_n"]
    out = {"terminal_reasons": {}, "end_states": {}, "outcomes": {}, "slip": {}, "penetration_mm": {}, "impacts": {},
           "table_switches_close_grip": {}, "joint_limits": {}, "visual_overlap_mm": {}}
    valid = np.arange(arrays["arm"].shape[1])[None, :] < lengths[:, None]
    limits = np.array(system["bodies"][0]["joints"]["range_rad"])
    q = arrays["arm"][..., :6].astype(np.float64)
    past = np.maximum(np.maximum(limits[:, 0] - q, q - limits[:, 1]), 0.0) * valid[..., None]   # rad past the limit
    joints = system["bodies"][0]["joint_names"]
    cube_body = next(b for b in system["bodies"] if b["name"] == "cube")["geometry"]
    energy_args = cube_body["mass_kg"], cube_body["inertia_diag_kgm2"][0], system["physics_config"]["gravity"][2]
    phases = system["phases"]
    for f, name in enumerate(fam_names):
        sel = family == f
        if not sel.any():
            continue
        tr = arrays["terminal_reason"][sel]
        out["terminal_reasons"][name] = {r: int((tr == k).sum()) for k, r in enumerate(system["terminal_reasons"])
                                         if (tr == k).any()}
        last = np.flatnonzero(sel), np.maximum(lengths[sel] - 2, 0)
        c, palm, done = contacts[last], arrays["palm_force"][last] > thr, tr == 0
        jaw = c[:, :2].any(1)
        out["end_states"][name] = dict(ended_early=int((~done).sum()), on_table=int((done & c[:, 2] & ~jaw).sum()),
                                       on_gripper=int((done & jaw).sum()),
                                       on_a_palm_block=int((done & palm.any(1)).sum()),
                                       in_air=int((done & ~c.any(1)).sum()))
        p = past[sel]
        out["joint_limits"][name] = dict(records=int(valid[sel].sum()), past_a_limit=int((p > 0).any(-1).sum()),
                                         by_joint={j: int((p[..., k] > 0).sum()) for k, j in enumerate(joints)
                                                   if (p[..., k] > 0).any()},
                                         max_mrad=float(p.max() * 1000))
        out["visual_overlap_mm"][name] = _stats(arrays["visual_overlap"][sel][valid[sel]] * 1000)
        oc = arrays["outcome"][sel]
        out["outcomes"][name] = {r: int((oc == k).sum()) for k, r in enumerate(system["outcomes"]) if (oc == k).any()}
        for key in ("hold_slip_mm", "carry_slip_mm"):
            v = arrays[key][sel]
            v = v[np.isfinite(v)]
            if len(v):
                out["slip"].setdefault(name, {})[key] = _stats(v)
        out["penetration_mm"][name] = {p: _stats(arrays["contact_depth"][sel][..., q][contacts[sel][..., q]] * 1000)
                                       for q, p in enumerate(PAIRS)}
        ev, switches = [], []
        for i in np.flatnonzero(sel):
            n = lengths[i]
            cube = arrays["cube"][i, :n].astype(np.float64)
            ev += impacts(contacts[i, :n - 1], cube, cube_energy(cube, *energy_args))
            grip = np.flatnonzero(np.isin(arrays["phase"][i, :n - 1], [phases.index("close"), phases.index("grip")]))
            if len(grip):
                switches.append(int(np.abs(np.diff(contacts[i, grip[0]:grip[-1] + 1, 2].astype(int))).sum()))
        if ev:
            gain = np.array([g for _, g in ev if g is not None] + [-np.inf])
            out["impacts"][name] = {"landings": len(ev), "landing_speed_m_s": _stats([v for v, _ in ev]),
                                    "bounces": len(gain) - 1, "energy_gain_above_0.1_mj": int((gain > 1e-4).sum()),
                                    "largest_change_mj": float(gain.max() * 1000) if len(gain) > 1 else None}
        if switches:
            sw = np.array(switches)
            out["table_switches_close_grip"][name] = dict(episodes=len(sw), off_and_on_again=int((sw >= 2).sum()),
                                                          max=int(sw.max()))
    force = arrays["contact_force"]
    valid = np.arange(force.shape[1])[None, :] < (lengths - 1)[:, None]
    edges = np.r_[0.0, np.logspace(-6, 2, 33)]
    out["contact_force_histogram"] = {"edges_n": edges.tolist()}
    flips = {}
    for p, meta in enumerate(system["pair_metadata"]):
        f = force[..., p][valid]
        out["contact_force_histogram"][meta["name"]] = np.histogram(f, np.r_[edges, np.inf])[0].tolist()
        on = f > thr
        flips[meta["name"]] = dict(on=int(on.sum()), on_at_0=int((f > 0).sum()), on_at_2x=int((f > 2 * thr).sum()),
                                   flips_0=int(((f > 0) != on).sum()), flips_2x=int(((f > 2 * thr) != on).sum()),
                                   half_to_1x=int(((f > thr / 2) & ~on).sum()),
                                   x1_to_2x=int((on & (f <= 2 * thr)).sum()))
    out["label_flips"] = flips
    lab, _, valid_w, _ = validate.training_windows(system, arrays["contacts"], lengths, arrays["splits"])
    palm = np.concatenate((arrays["palm_force"] > thr, np.zeros_like(arrays["contacts"][..., :1], bool)), -1)
    palm_lab = validate.training_windows(system, palm, lengths, arrays["splits"])[0]
    single = valid_w & (lab.sum(-1) == 1) & (arrays["splits"] == 0)[:, None]
    out["single_jaw_windows_train"] = {}
    for q in (0, 1):
        w = single & lab[..., q]
        out["single_jaw_windows_train"][PAIRS[q]] = {
            "all": int(w.sum()), "touching_a_palm_block": int((w & palm_lab[..., q]).sum()),
            "by_family": {fam_names[f]: [int(w[family == f].sum()), int((w & palm_lab[..., q])[family == f].sum())]
                          for f in range(len(fam_names)) if w[family == f].any()}}
    return out


# ---- convergence -----------------------------------------------------------------------------------------------
def replay(s, ep, actions):
    """Open-loop replay of recorded commands (one per record) from the episode's start -> states, contact labels and
    the deepest penetration [3] of the pairs (m)."""
    m, d = s.m, s.d
    c = ep["cube"]
    s.reset(ep["q0"], c["pos"], c["quat"], c.get("vel"), c.get("omega"), c["spawn"])
    n = len(actions)
    arm, cube, contacts = np.zeros((n, 12)), np.zeros((n, 13)), np.zeros((n - 1, 3), bool)
    depth = np.zeros(3)
    for k in range(n):
        arm[k], cube[k] = s.state()
        if k == n - 1:
            break
        d.ctrl[s.act] = actions[k]
        fmax = np.zeros(3)
        for _ in range(s.nsub):
            mujoco.mj_step(m, d)
            f, pen = s.contact_forces()
            fmax = np.maximum(fmax, f[:3])
            depth = np.maximum(depth, pen)
        contacts[k] = fmax > s.thr
    return arm, cube, contacts, depth


def cube_energy(cube, mass, inertia, gravity_z):
    """Kinetic plus potential energy [T] (J) of recorded cube states (world angular velocity; a cube's inertia is
    the same about every axis)."""
    return (0.5 * mass * (cube[:, 7:10] ** 2).sum(-1) + 0.5 * inertia * (cube[:, 10:13] ** 2).sum(-1)
            - mass * gravity_z * cube[:, 2])


def impacts(contacts, cube, energy):
    """Cube-table contact events after a contact-free interval of the cube -> [(speed at the onset m/s, energy after
    - energy before J if the cube is contact-free again after the event, else None)]."""
    free = ~contacts.any(1)
    table = contacts[:, 2] & ~contacts[:, :2].any(1)
    out, k, n = [], 1, len(contacts)
    while k < n:
        if table[k] and free[k - 1]:
            k1 = k
            while k1 < n and table[k1]:
                k1 += 1
            change = float(energy[k1] - energy[k]) if k1 < n and free[k1] else None
            out.append((float(np.linalg.norm(cube[k, 7:10])), change))
            k = k1
        k += 1
    return out


def grasp_frame(s, arm, cube):
    """Cube centre in the grasp frame [T, 3] for recorded arm and cube states."""
    out = np.zeros((len(arm), 3))
    for t in range(len(arm)):
        dk = s.fk(arm[t, :6], cube[t, :3])
        r = dk.xmat[s.gripper].reshape(3, 3)
        out[t] = r.T @ (cube[t, :3] - dk.xpos[s.gripper]) - s.grasp_point
    return out


def conv_variants(cfg):
    """Physics settings of the convergence check: the chosen step, twice (if it divides the record step) and half of
    it, the tight solver, and the reference (a quarter of the step with the tight solver)."""
    dt, tight = cfg["physics"]["timestep_s"], cfg["tight_solver"]
    out = {"chosen": {}, "double_step": {"timestep_s": 2 * dt}, "half_step": {"timestep_s": dt / 2},
           "tight_solver": dict(tight), "reference": {"timestep_s": dt / 4, **tight}}
    ratio = cfg["record_step_s"] / (2 * dt)
    if abs(ratio - round(ratio)) > 1e-9:
        del out["double_step"]
    return out


_CONV = None


def _conv_init(cfg_path):
    global _CONV
    cfg = load_config(cfg_path)
    _CONV = cfg, {k: Scene(cfg, v) for k, v in conv_variants(cfg).items()}


def _conv_job(args):
    """One episode planned and run at the chosen step, its commands replayed with every variant -> row."""
    fam_index, index = args
    cfg, scenes = _CONV
    s = scenes["chosen"]
    fam = cfg["families"][fam_index]
    rng = np.random.default_rng(episode_seed(cfg["convergence"]["master_seed"], fam_index, index))
    n_rec = int(fam["records"])
    while True:
        try:
            plan = FAMILIES[fam["name"]](s, rng, (n_rec - 1) // s.per_ctrl + 1)
        except PlanError:
            continue
        rec = run_episode(s, plan, n_rec)
        if not isinstance(rec, str):
            break
    if rec is None:
        return dict(family=fam["name"], index=index, rejected=True)
    actions, phases = rec["action"], rec["per_record"]["phase"]
    runs = {k: replay(sc, plan, actions) for k, sc in scenes.items()}
    ref_arm, ref_cube, ref_con, _ = runs["reference"]
    n, rec_dt = len(actions), cfg["record_step_s"]
    marks = {}
    close = np.flatnonzero(phases == P["close"])
    for sec in ((0.2, 0.5, 1.0) if len(close) else ()):
        marks[f"close+{sec:g}s"] = int(close[0]) + round(sec / rec_dt)
    hold = np.flatnonzero(phases == P["hold"])
    if len(hold):
        marks["hold_end"] = int(hold[np.argmax(np.diff(hold) > 1)] if (np.diff(hold) > 1).any() else hold[-1])
    free = ~ref_con.any(1)
    for k in range(3, len(ref_con)):   # the first landing of a falling cube
        if ref_con[k, 2] and free[k - 3:k].all() and ref_cube[k, 9] < -0.3:
            marks["landing+50ms"] = k + round(0.05 / rec_dt)
            break
    marks = {k: v for k, v in marks.items() if v < n}
    group = fam["group"]
    energy_args = s.m.body_mass[s.cube_body], s.m.body_inertia[s.cube_body, 0], s.m.opt.gravity[2]
    row = dict(family=fam["name"], index=index, records=n, terminal=TERMINAL[rec["terminal"]], marks=marks,
               determinism_max_abs=float(max(np.abs(runs["chosen"][0] - rec["states"]["arm"]).max(),
                                             np.abs(runs["chosen"][1] - rec["states"]["cube"]).max())))
    for k, (arm, cube, con, depth) in runs.items():
        v = dict(depth_mm=(depth * 1000).tolist(), labels_differ=(con != ref_con).sum(0).tolist(),
                 impacts=impacts(con, cube, cube_energy(cube, *energy_args)))
        if k != "reference":
            v["errors"] = {name: dict(
                cube_pos_mm=float(np.linalg.norm(cube[t, :3] - ref_cube[t, :3]) * 1000),
                cube_angle_deg=float(np.degrees(2 * np.arccos(min(1.0, abs(cube[t, 3:7] @ ref_cube[t, 3:7]))))),
                cube_w_rad_s=float(np.linalg.norm(cube[t, 10:13] - ref_cube[t, 10:13])),
                joint_max_mrad=float(np.abs(arm[t, :6] - ref_arm[t, :6]).max() * 1000)) for name, t in marks.items()}
            v["timing"] = timing_errors(con, ref_con, rec_dt)
        if group != "probe":
            ep = dict(contacts=con, states={"cube": cube},
                      per_record=dict(phase=phases, cube_in_grasp=grasp_frame(s, arm, cube)))
            o = outcome(ep, s.half, group)
            v.update(outcome=OUTCOMES[o["outcome"]], carry_slip_mm=o["carry_slip_mm"])
        row[k] = v
    return row


def convergence(cfg, out=None, workers=1, scale=1.0, cfg_path=CONFIG):
    """Episodes of every family (counts in the config's ``convergence`` block, times ``scale``) simulated at the
    chosen step; their recorded commands replayed open loop with the other settings and compared with the reference.
    Per variant: cube and joint errors at marks (0.2 / 0.5 / 1 s after the jaw starts to close, the end of the first
    hold, 50 ms after the first landing of a falling cube), contact onset/offset timing and label differences of the
    three pairs, energy change in cube-table impacts, deepest penetration, grasp outcome and cube slip while
    carried."""
    names = [f["name"] for f in cfg["families"]]
    jobs = [(names.index(name), i) for name, count in cfg["convergence"]["episodes"].items()
            for i in range(max(1, int(round(count * scale))))]
    for v in conv_variants(cfg).values():   # fail here, not in the worker initializers
        Scene(cfg, v)
    with mp.get_context("spawn").Pool(workers, initializer=_conv_init, initargs=(str(cfg_path),)) as pool:
        rows = pool.map(_conv_job, jobs, chunksize=1)
    rejected = [r for r in rows if r.get("rejected")]
    rows = [r for r in rows if not r.get("rejected")]
    variants = conv_variants(cfg)
    report = {"variants": variants, "episodes": len(rows), "rejected_numerical": len(rejected),
              "summary": {"all": convergence_summary(rows, variants),
                          "families": {f: convergence_summary([r for r in rows if r["family"] == f], variants)
                                       for f in cfg["convergence"]["episodes"]}},
              "rows": rows}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=1) + "\n")
    return report


def timing_errors(a, b, dt, reach=25):
    """Onsets and offsets of two [T-1, P] label arrays: each event of ``a`` matched to the nearest unmatched event of
    the same kind in ``b`` within ``reach`` records -> per pair the matched time differences (ms), the events of
    either run left unmatched (short flickers of grazing contacts) and the events of ``b``."""
    out = {}
    for p, name in enumerate(PAIRS):
        ea, eb = (np.diff(np.r_[0, x[:, p].astype(int)]) for x in (a, b))
        dts, unmatched = [], 0
        for kind in (1, -1):
            ia, ib = list(np.flatnonzero(ea == kind)), list(np.flatnonzero(eb == kind))
            for k in ia:
                j = min(ib, key=lambda x: abs(x - k), default=None)
                if j is not None and abs(j - k) <= reach:
                    dts.append(float(abs(j - k) * dt * 1000))
                    ib.remove(j)
                else:
                    unmatched += 1
            unmatched += len(ib)
        out[name] = dict(dt_ms=dts, unmatched=unmatched, events_b=int((eb != 0).sum()))
    return out


def _stats(x):
    x = np.asarray(x, float)
    return dict(n=int(len(x)), median=float(np.median(x)), p95=float(np.quantile(x, 0.95)),
                max=float(x.max())) if len(x) else None


def convergence_summary(rows, variants):
    """Per variant: error statistics at each mark; per pair the matched timing, the share of reference events
    matched within 20 ms, unmatched events and label differences; impacts that gained energy; deepest penetration;
    outcomes that differ from the reference and the carry slip."""
    out = {}
    if not rows:
        return out
    for v in variants:
        rs = [r[v] for r in rows]
        e = {}
        if v != "reference":
            for mark in sorted({m for r in rows for m in r["marks"]}):
                for key in ("cube_pos_mm", "cube_angle_deg", "cube_w_rad_s", "joint_max_mrad"):
                    e[f"{mark} {key}"] = _stats([x["errors"][mark][key] for x in rs if mark in x["errors"]])
            for p in PAIRS:
                d = np.concatenate([x["timing"][p]["dt_ms"] for x in rs] + [np.zeros(0)])
                events = sum(x["timing"][p]["events_b"] for x in rs)
                e[f"{p} timing_ms"] = _stats(d)
                e[f"{p} reference_events"] = int(events)
                e[f"{p} within_20ms"] = float((d <= 20 + 1e-6).sum() / max(events, 1))
                e[f"{p} unmatched_events"] = int(sum(x["timing"][p]["unmatched"] for x in rs))
        intervals = sum(r["records"] - 1 for r in rows)
        for q, p in enumerate(PAIRS):
            e[f"{p} labels_differ_per_1000"] = 1000 * sum(x["labels_differ"][q] for x in rs) / intervals
            e[f"{p} depth_mm"] = _stats([x["depth_mm"][q] for x in rs if x["depth_mm"][q] > 0])
        gains = np.array([g for x in rs for _, g in x["impacts"] if g is not None] + [-np.inf])
        e["impacts"] = {"bounces": int(len(gains) - 1), "energy_gain_above_0.1_mj": int((gains > 1e-4).sum()),
                        "largest_change_mj": float(gains.max() * 1000) if len(gains) > 1 else None}
        slip = [x["carry_slip_mm"] for x in rs if "carry_slip_mm" in x and np.isfinite(x["carry_slip_mm"])]
        if slip:
            e["carry_slip_mm"] = _stats(slip)
            e["outcome_differs"] = int(sum(x["outcome"] != r["reference"]["outcome"] for x, r in zip(rs, rows)
                                           if "outcome" in x))
        e["determinism_max_abs"] = float(max(r["determinism_max_abs"] for r in rows))
        out[v] = e
    return out


# ---- rendering -------------------------------------------------------------------------------------------------
def render(cfg, data_dir, episodes, out_dir, fps=50, size=(480, 640)):
    """Videos of recorded episodes: the stored states (arm angles, cube pose) shown with the scene."""
    import imageio
    s = Scene(cfg)
    with np.load(Path(data_dir) / "unified_data.npz") as f:
        arrays = {k: f[k] for k in ("arm", "cube", "lengths", "family", "contacts")}
    system = json.loads((Path(data_dir) / "system.json").read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m, d = s.m, mujoco.MjData(s.m)
    s.m.vis.global_.offwidth, s.m.vis.global_.offheight = size[1], size[0]
    renderer = mujoco.Renderer(m, size[0], size[1])
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.2, 0.0, 0.05]
    cam.distance, cam.azimuth, cam.elevation = 0.75, 145, -25
    paths = []
    for i in episodes:
        name = system["families"][int(arrays["family"][i])]["name"]
        path = out_dir / f"episode{i:05d}_{name}.mp4"
        step = round(1 / fps / system["record_step_s"])
        with imageio.get_writer(path, fps=fps, macro_block_size=1) as w:
            for k in range(0, int(arrays["lengths"][i]), step):
                d.qpos[s.qadr] = arrays["arm"][i, k, :6]
                d.qpos[s.cq: s.cq + 7] = arrays["cube"][i, k, :7]
                mujoco.mj_forward(m, d)
                renderer.update_scene(d, cam)
                w.append_data(renderer.render())
        paths.append(str(path))
    renderer.close()
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="simulate the families and write a data directory")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--workers", type=int)
    c.add_argument("--campaign", choices=("pilot", "production", "test", "extra"), default="pilot",
                   help="master seed and episode counts from the config's campaigns block; pilot and production: "
                        "splits 0/1 by family; test: all split 2")
    c.add_argument("--chunk", type=int, default=16, help="episodes per pool task and shard file")
    c.add_argument("--families", nargs="*", help="only these families")
    c.add_argument("--scale", type=float, help="replace the campaign's task and probe factors (smoke runs)")
    c.add_argument("--probe-scale", type=float, help="replace the campaign's probe factor")
    v = sub.add_parser("convergence", help="physics-step and solver convergence on episodes of every family")
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--workers", type=int)
    v.add_argument("--scale", type=float, default=1.0, help="multiply the episode counts of the convergence block")
    r = sub.add_parser("report", help="terminations, outcomes, slip, force histogram and label flips of a directory")
    r.add_argument("data", type=Path)
    w = sub.add_parser("render", help="videos of recorded episodes")
    w.add_argument("data", type=Path)
    w.add_argument("--episodes", type=int, nargs="+", required=True)
    w.add_argument("--out", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "collect":
        collect(cfg, args.out, args.workers or cfg["workers"], args.campaign, args.families, args.scale,
                args.probe_scale, args.chunk, args.config)
    elif args.cmd == "convergence":
        report = convergence(cfg, args.out, args.workers or cfg["workers"], args.scale, args.config)
        print(json.dumps(report["summary"]["all"], indent=1))
    elif args.cmd == "report":
        system = json.loads((args.data / "system.json").read_text())
        with np.load(args.data / "unified_data.npz") as f:
            arrays = {k: f[k] for k in f.files}
        print(json.dumps(summarize(system, arrays), indent=1))
    else:
        print("\n".join(render(cfg, args.data, args.episodes, args.out or args.data / "videos")))


if __name__ == "__main__":
    main()
