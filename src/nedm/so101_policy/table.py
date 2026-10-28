"""A table that is not at z = 0 (evaluation only; ``~/so101_r2s_claude/table_test/REPORT.md``).

The calibration of the real setup found the real table top lower than the sim table relative to the arm's kinematic
frame (``calib_v1.json`` ``scene.table_plane``: z = c0 + c1 x + c2 y, m, arm base frame; 12-17 mm lower where the cubes
are). Training always has the table top at z = 0. This module holds the plane arithmetic used by the CPU MuJoCo
worlds (``mujoco_env``: the table geom on the plane), the task rules (``task.features``: heights above the plane), the
action layer (``joint_env.floor_guard``: the floor on a plane) and the joint env (``JointVecEnv.set_table``: task starts
moved onto the plane). numpy only (the MuJoCo workers import it without torch).

A plane is a tuple (c0, c1, c2); None or (0, 0, 0) = the sim table (z = 0), and then every caller runs its old code.
Heights above the plane are measured along the plane normal n = (-c1, -c2, 1) / |(-c1, -c2, 1)|. A flat offset dz is
the plane (dz, 0, 0).
"""
from __future__ import annotations

import math

import numpy as np


def parse(dz=None, plane=None):
    """CLI values -> plane tuple or None. ``dz`` float (m, negative = lower); ``plane`` "c0,c1,c2" or a sequence."""
    if plane is not None and plane != "":
        if isinstance(plane, str):
            plane = [float(v) for v in plane.split(",")]
        p = tuple(float(v) for v in plane)
        if len(p) != 3:
            raise ValueError(f"table plane needs 3 coefficients c0,c1,c2, got {plane!r}")
        if dz:
            p = (p[0] + float(dz), p[1], p[2])
    elif dz:
        p = (float(dz), 0.0, 0.0)
    else:
        return None
    return None if not any(p) else p


def is_flat(plane) -> bool:
    return plane is None or (plane[1] == 0.0 and plane[2] == 0.0)


def normal(plane):
    """(unit normal n (3,), offset d) with n . p - d = signed height of p above the plane."""
    c0, c1, c2 = (float(c) for c in plane)
    s = math.sqrt(1.0 + c1 * c1 + c2 * c2)
    return np.array([-c1 / s, -c2 / s, 1.0 / s]), c0 / s


def height(plane, xy):
    """Plane z at xy [..., 2] (vertical)."""
    c0, c1, c2 = (float(c) for c in plane)
    xy = np.asarray(xy, np.float64)
    return c0 + c1 * xy[..., 0] + c2 * xy[..., 1]


def above(plane, p):
    """Signed height (along the normal) of points p [..., 3] above the plane."""
    n, d = normal(plane)
    return np.asarray(p, np.float64) @ n - d


def tilt_quat(plane):
    """Unit quaternion w x y z of the shortest turn e_z -> plane normal (identity for a flat plane)."""
    n, _ = normal(plane)
    q = np.array([1.0 + n[2], -n[1], n[0], 0.0])
    return q / np.linalg.norm(q)


def qmul(a, b):
    """Quaternion product a b (w x y z), [..., 4]."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), -1)


def map_cube(plane, pos, quat):
    """A cube pose given on the sim table (centre z = height above z = 0; quat w x y z) -> the same pose on the plane:
    xy kept, the centre at the same height along the plane normal (vertical offset height / n_z), the cube turned by
    the plane's tilt (a cube flat on the sim table lies flat on the plane). pos [..., 3], quat [..., 4] (numpy)."""
    pos, quat = np.asarray(pos, np.float64), np.asarray(quat, np.float64)
    if plane is None:
        return pos.copy(), quat.copy()
    n, _ = normal(plane)
    out = pos.copy()
    out[..., 2] = height(plane, pos[..., :2]) + pos[..., 2] / n[2]
    q = qmul(np.broadcast_to(tilt_quat(plane), quat.shape), quat)
    return out, q


def table_geom(plane, pos0, quat0=(1.0, 0.0, 0.0, 0.0)):
    """Pose (pos (3,), quat w x y z) of the collector's table box (centre pos0, top at z = 0, axis-aligned) moved so
    that its top face lies on the plane: the top centre at (x0, y0, plane(x0, y0)), the box turned by the tilt."""
    pos0 = np.asarray(pos0, np.float64)
    if plane is None:
        return pos0.copy(), np.asarray(quat0, np.float64).copy()
    n, _ = normal(plane)
    half_t = -pos0[2]                                 # the box centre is half its thickness below the top (z = 0)
    top = np.array([pos0[0], pos0[1], float(height(plane, pos0[:2]))])
    return top - n * half_t, qmul(tilt_quat(plane), np.asarray(quat0, np.float64))


def describe(plane) -> str:
    if plane is None:
        return "sim table (z = 0)"
    if is_flat(plane):
        return f"flat table at z = {plane[0] * 1e3:+.1f} mm"
    return f"table plane z = {plane[0]:+.5f} {plane[1]:+.5f} x {plane[2]:+.5f} y"


def parse_floor(s):
    """CLI floor value -> "zero" | "table" | float (a flat floor height, m, as the robot script's --table-z)."""
    if s in (None, "", "zero"):
        return "zero"
    if s == "table":
        return "table"
    return float(s)


def parse_q_offset_deg(s):
    """ "a,b,c,d,e,f" (deg, sim joint order) -> list of 6 rad, or None."""
    if not s:
        return None
    v = [math.radians(float(x)) for x in s.split(",")]
    if len(v) != 6:
        raise ValueError(f"q offset needs 6 values (deg), got {s!r}")
    return v if any(v) else None


def add_cli(q, student=False):
    """The evaluation options of a lowered / tilted table (``vision_viral.py eval``, ``vision_viral_teacher.py eval``)."""
    q.add_argument("--table-dz", type=float, default=0.0,
                   help="evaluation only: table top moved by this (m, negative = lower), flat; added to --table-plane")
    q.add_argument("--table-plane", default="", help="evaluation only: table top on z = c0 + c1 x + c2 y ('c0,c1,c2')")
    q.add_argument("--floor", default="zero",
                   help="action-layer table floor: 'zero' (z = 0, default, the robot's --table-z 0), 'table' (follows "
                        "the table), or a height in m (a flat floor there, the robot's --table-z)")
    if student:
        q.add_argument("--q-offset-deg", default="",
                       help="sim emulation of an extra joint-map offset (6 values, deg): the student's q input and "
                            "the action layer work in q + offset")
        q.add_argument("--no-render-table", action="store_true",
                       help="with a moved table: render the images with the scene's own table (inconsistent images)")
    q.add_argument("--log-only-extra", default="",
                   help="evaluation only: task failure rules that only log (comma list, e.g. 'unsupported,release_high' "
                        "= a cube released above the table may drop onto it)")


def from_cli(a):
    """-> (plane, floor, q_offset or None, active)."""
    plane = parse(a.table_dz, a.table_plane or None)
    floor = parse_floor(a.floor)
    qo = parse_q_offset_deg(getattr(a, "q_offset_deg", ""))
    return plane, floor, qo, plane is not None or floor != "zero" or qo is not None


def log_only_extra(env, a):
    """Add ``--log-only-extra`` rules to the env's task config (evaluation only) -> the tuple added."""
    from nedm.so101_policy import task as T
    extra = tuple(r for r in (getattr(a, "log_only_extra", "") or "").split(",") if r)
    bad = [r for r in extra if r not in T.RULES]
    if bad:
        raise ValueError(f"unknown task rules {bad}")
    if extra:
        env.tcfg.log_only = tuple(env.tcfg.log_only) + extra
        if env.task.cfg is not env.tcfg:
            raise RuntimeError("task config is not the env's")
    return extra
