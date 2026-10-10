"""SO-101 kinematics and statics for the push-T collector (numpy only).

Ported from the twinfactory digital twin (no runtime dependency on it):
  * forward kinematics: chrono_vla chrono_so101/kinematics.py:link_world_poses, the code that
    twinfactory core/geometry.py ports verbatim (joint convention T_P @ F0 @ Rz(q) == T_C @ F1);
  * quaternion helpers: kinematics.py quat_to_matrix / matrix_to_quat;
  * TCP in the gripper link: configs/robots/so101.yaml tcp (scripted_policy.TCP_IN_GRIPPER rev 4);
  * the robot description comes from assets/so101/so101_robot.json (made by
    scripts/so101_push/make_assets.py from twinfactory configs/robots/so101/robot.json).

The jaw is locked at `jaw_lock_rad` (it is part of the gripper subtree), so the arm has the five
joints shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll.

World frame: z up, table top at z = 0. The robot root sits at the twinfactory `isaac_env0` pose
(x = -0.05 m, yaw +90 deg) lowered by `table_top_z_root_m`, so the base foot rests on the table.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROBOT_JSON = REPO / "assets" / "so101" / "so101_robot.json"
DEFAULT_HULLS = REPO / "assets" / "so101" / "so101_hulls.npz"
DEFAULT_FINGERS = REPO / "assets" / "so101" / "so101_finger_sections.npz"
DEFAULT_FINGER_BOXES = REPO / "assets" / "so101" / "so101_finger_boxes.json"
ARM_JOINTS = 5


# ----------------------------------------------------------------------------- math (kinematics.py)
def quat_to_matrix(q_wxyz) -> np.ndarray:
    w, x, y, z = [float(v) for v in q_wxyz]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z) with w >= 0."""
    R = np.asarray(R, dtype=float)[:3, :3]
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        q = (0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s)
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        q = ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        q = ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        q = ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
    q = np.array(q)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def make_transform(pos=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(quat_wxyz)
    T[:3, 3] = pos
    return T


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    return T


def inv_transform(T) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


@lru_cache(maxsize=4)
def load_description(path: str | None = None) -> dict:
    return json.loads(Path(path or DEFAULT_ROBOT_JSON).read_text())


@lru_cache(maxsize=4)
def load_hulls(path: str | None = None) -> dict:
    with np.load(path or DEFAULT_HULLS) as data:
        return {k: data[k].copy() for k in data.files}


def load_config(path) -> dict:
    """Collector config with the robot asset paths made absolute (repo root); moved unchanged from nedm.so101_push.episode."""
    cfg = json.loads(Path(path).read_text())
    rob = cfg["robot"]
    for key in ("description", "hulls", "finger_sections", "finger_boxes"):
        if key not in rob:
            continue
        p = Path(rob[key])
        rob[key] = str(p if p.is_absolute() else REPO / p)
    return cfg


# ----------------------------------------------------------------------------- model
class ArmModel:
    """Kinematic / static model of the 5-DOF SO-101 with the jaw locked."""

    def __init__(self, robot_cfg: dict, robot_json: str | None = None):
        self.desc = load_description(robot_json)
        d = self.desc
        self.links = list(d["link_order"])
        self.joints = list(d["joint_order"])          # Rotation .. Jaw (6)
        self.jaw_lock = float(robot_cfg["jaw_lock_rad"])
        root = d["root_in_isaac_env0"]
        pos = np.array(root["pos"], dtype=float)
        pos[2] -= float(d["table_top_z_root_m"])       # table top -> world z = 0
        self.T_root = make_transform(pos, root["quat_wxyz"])
        self.tcp_local = np.array(d["tcp"]["pos_m"], dtype=float)
        self.F0 = [make_transform(d["joints"][j]["frame_in_parent"]["pos"], d["joints"][j]["frame_in_parent"]["quat_wxyz"])
                   for j in self.joints]
        self.F1inv = [inv_transform(make_transform(d["joints"][j]["frame_in_child"]["pos"],
                                                   d["joints"][j]["frame_in_child"]["quat_wxyz"])) for j in self.joints]
        self.parent = [d["joints"][j]["parent"] for j in self.joints]
        self.child = [d["joints"][j]["child"] for j in self.joints]
        self.mass = {ln: float(d["links"][ln]["mass"]) for ln in self.links}
        self.com = {ln: np.array(d["links"][ln]["com_link_frame"], dtype=float) for ln in self.links}
        self.inertia = {ln: np.array(d["links"][ln]["inertia_about_com_link_frame"], dtype=float) for ln in self.links}
        self.T_base = self.T_root @ make_transform(d["links"]["base"]["pose_in_root"]["pos"],
                                                   d["links"]["base"]["pose_in_root"]["quat_wxyz"])
        # joint limits: intersection of the USD limits (so101.yaml joints[].limits_rad) and the variant's
        # q-space limits (so101.yaml variants.real_white_no_wrist_cam.joint_limits_deg)
        usd = np.array([d["joints"][j]["limits_rad"] for j in self.joints[:ARM_JOINTS]])
        var = np.deg2rad(np.array(d["variant_joint_limits_deg"][:ARM_JOINTS], dtype=float))
        self.limits = np.stack([np.maximum(usd[:, 0], var[:, 0]), np.minimum(usd[:, 1], var[:, 1])], axis=1)
        self.ik_margin = float(robot_cfg.get("ik_limit_margin_rad", 0.015))
        # subtree (child links) of every arm joint; the jaw is rigidly part of the gripper subtree
        self.subtree = [self.links[self.links.index(self.child[j]):] for j in range(ARM_JOINTS)]

    # ------------------------------------------------------------------ kinematics
    def q6(self, q5) -> np.ndarray:
        return np.r_[np.asarray(q5, dtype=float)[:ARM_JOINTS], self.jaw_lock]

    def link_poses(self, q5) -> dict:
        """{link: 4x4 world pose of the link frame} (kinematics.py link_world_poses for a serial chain)."""
        q = self.q6(q5)
        poses = {"base": self.T_base}
        for j in range(len(self.joints)):
            poses[self.child[j]] = poses[self.parent[j]] @ self.F0[j] @ rot_z(q[j]) @ self.F1inv[j]
        return poses

    def joint_frames(self, poses) -> list:
        """World frames of the five arm joints (parent side, z = joint axis)."""
        return [poses[self.parent[j]] @ self.F0[j] for j in range(ARM_JOINTS)]

    def tcp(self, q5) -> tuple[np.ndarray, np.ndarray]:
        T = self.link_poses(q5)["gripper"]
        return T[:3, :3] @ self.tcp_local + T[:3, 3], T[:3, :3]

    def tcp_jacobian(self, q5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """TCP position, gripper rotation and the 6x5 geometric Jacobian [v; w]."""
        poses = self.link_poses(q5)
        T = poses["gripper"]
        p = T[:3, :3] @ self.tcp_local + T[:3, 3]
        J = np.zeros((6, ARM_JOINTS))
        for j, F in enumerate(self.joint_frames(poses)):
            a, o = F[:3, 2], F[:3, 3]
            J[:3, j] = np.cross(a, p - o)
            J[3:, j] = a
        return p, T[:3, :3], J

    @staticmethod
    def gripper_yaw(R) -> float:
        """Heading of the gripper x axis (jaw closing direction) in the table plane."""
        return math.atan2(R[1, 0], R[0, 0])

    # ------------------------------------------------------------------ statics / dynamics
    def gravity_hold_torque(self, q5, g=9.81) -> np.ndarray:
        """Joint torques that hold the arm still against gravity (motor convention: + about the joint z)."""
        poses = self.link_poses(q5)
        frames = self.joint_frames(poses)
        gvec = np.array([0.0, 0.0, -g])
        tau = np.zeros(ARM_JOINTS)
        for j, F in enumerate(frames):
            a, o = F[:3, 2], F[:3, 3]
            for ln in self.subtree[j]:
                T = poses[ln]
                c = T[:3, :3] @ self.com[ln] + T[:3, 3]
                tau[j] -= a @ np.cross(c - o, self.mass[ln] * gvec)
        return tau

    def mass_matrix(self, q5) -> np.ndarray:
        """Joint-space inertia matrix (5x5) with the jaw locked to the gripper."""
        poses = self.link_poses(q5)
        frames = self.joint_frames(poses)
        M = np.zeros((ARM_JOINTS, ARM_JOINTS))
        for ln in self.links[1:]:
            T = poses[ln]
            R = T[:3, :3]
            c = R @ self.com[ln] + T[:3, 3]
            Jv = np.zeros((3, ARM_JOINTS))
            Jw = np.zeros((3, ARM_JOINTS))
            for j, F in enumerate(frames):
                if ln in self.subtree[j]:
                    a, o = F[:3, 2], F[:3, 3]
                    Jv[:, j] = np.cross(a, c - o)
                    Jw[:, j] = a
            Iw = R @ self.inertia[ln] @ R.T
            M += self.mass[ln] * Jv.T @ Jv + Jw.T @ Iw @ Jw
        return M

    # ------------------------------------------------------------------ inverse kinematics
    def ik(self, p_des, yaw_des, q_init, *, iters=60, tol=1e-6, damping=1e-3, w_pos=1.0, w_tilt=0.05,
           w_yaw=0.02) -> tuple[np.ndarray, dict]:
        """Damped least squares IK: TCP position, fingers pointing straight down (gripper z = world z),
        gripper yaw. Residual weights are in metres per radian. Joint limits with a margin are hard
        bounds (clamped every iteration). Returns (q5, info)."""
        lo, hi = self.limits[:, 0] + self.ik_margin, self.limits[:, 1] - self.ik_margin
        q = np.clip(np.asarray(q_init, dtype=float)[:ARM_JOINTS], lo, hi)
        W = np.diag([w_pos] * 3 + [w_tilt] * 2 + [w_yaw])
        err = None
        for _ in range(iters):
            p, R, J = self.tcp_jacobian(q)
            z = R[:, 2]
            yaw = self.gripper_yaw(R)
            dyaw = math.atan2(math.sin(yaw_des - yaw), math.cos(yaw_des - yaw))
            # tilt: rotate z towards world up; small-angle rotation vector = z x up
            tilt = np.cross(z, np.array([0.0, 0.0, 1.0]))
            e = np.r_[np.asarray(p_des) - p, tilt[:2], dyaw]
            # task Jacobian rows: position, tilt (x, y of the rotation vector), yaw (rotation about world z)
            Jt = np.vstack([J[:3], J[3:5], J[5:6]])
            We = W @ e
            err = float(np.linalg.norm(We))
            if err < tol:
                break
            A = W @ Jt
            dq = A.T @ np.linalg.solve(A @ A.T + damping ** 2 * np.eye(6), We)
            q = np.clip(q + dq, lo, hi)
        p, R = self.tcp(q)
        info = {"pos_err_m": float(np.linalg.norm(np.asarray(p_des) - p)),
                "tilt_rad": float(math.acos(max(-1.0, min(1.0, R[2, 2])))),
                "yaw_err_rad": float(abs(math.atan2(math.sin(yaw_des - self.gripper_yaw(R)),
                                                    math.cos(yaw_des - self.gripper_yaw(R))))),
                "at_limit": bool(np.any((q <= lo + 1e-9) | (q >= hi - 1e-9)))}
        return q, info


@lru_cache(maxsize=4)
def load_finger_sections(path: str | None = None) -> dict:
    with np.load(path or DEFAULT_FINGERS) as data:
        return {k: data[k].copy() for k in data.files}


@lru_cache(maxsize=4)
def load_finger_boxes(path: str | None = None) -> list:
    return json.loads(Path(path or DEFAULT_FINGER_BOXES).read_text())["boxes"]


def link_collision_shapes(link: str, robot_cfg: dict) -> list[dict]:
    """Collision shapes of one link in its link frame: {'kind': 'hull', 'name', 'points'} or
    {'kind': 'box', 'name', 'center', 'rot' (columns = box axes), 'half'}.

    All links use the twin's COACD hulls (backends.chrono.link_collision.hulls). For the gripper and the jaw,
    robot_cfg['finger_collision'] selects the fingertip geometry:
      'coacd'           the COACD hulls (pilot_v0: thin slivers, contacts with opposite normals);
      'convex_sections' the twin's real-SO-101 finger sections (finger_collision: convex_sections) as hulls;
      'tip_boxes'       the same sections cut into 2.5 mm slabs, each slab as one or two oriented boxes
                        (so101_finger_boxes.json). Chrono 10's Bullet hull-vs-box contacts miss up to ~1 mm of
                        penetration and let the fingertips sink into the T; box-box contacts do not.
    With 'convex_sections' and 'tip_boxes' the rest of the link is one convex hull (<link>_upper).
    """
    mode = robot_cfg.get("finger_collision", "coacd")
    if link in ("gripper", "jaw") and mode in ("convex_sections", "tip_boxes"):
        f = load_finger_sections(robot_cfg.get("finger_sections"))
        out = []
        if mode == "convex_sections":
            names = sorted((k for k in f if k.startswith(link + "_tip_")), key=lambda k: int(k.rsplit("_", 1)[1]))
            out += [{"kind": "hull", "name": k, "points": f[k]} for k in names]
        else:
            for b in load_finger_boxes(robot_cfg.get("finger_boxes")):
                if b["link"] == link:
                    out.append({"kind": "box", "name": f"{b['section']}_s{b['slab']}_{b['part']}",
                                "center": np.array(b["center_m"]), "rot": np.array(b["rotation_cols"]),
                                "half": np.array(b["half_size_m"])})
        if robot_cfg.get("finger_upper", "convex_hull") == "convex_hull":
            out.append({"kind": "hull", "name": link + "_upper", "points": f[link + "_upper"]})
        return out
    hulls = load_hulls(robot_cfg.get("hulls"))
    keys = sorted((k for k in hulls if k.rsplit("_", 1)[0] == link), key=lambda k: int(k.rsplit("_", 1)[1]))
    return [{"kind": "hull", "name": k, "points": hulls[k]} for k in keys]


def shape_points(shape: dict) -> np.ndarray:
    if shape["kind"] == "hull":
        return shape["points"]
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * shape["half"]
    return c @ shape["rot"].T + shape["center"]


def gripper_points_local(model: ArmModel, robot_cfg: dict) -> np.ndarray:
    """Collision-shape vertices of the gripper and the locked jaw, in the gripper link frame."""
    j = model.joints.index("Jaw")
    T_jaw = model.F0[j] @ rot_z(model.jaw_lock) @ model.F1inv[j]      # jaw link in the gripper frame
    g = np.concatenate([shape_points(sh) for sh in link_collision_shapes("gripper", robot_cfg)])
    jw = np.concatenate([shape_points(sh) for sh in link_collision_shapes("jaw", robot_cfg)])
    return np.concatenate([g, jw @ T_jaw[:3, :3].T + T_jaw[:3, 3]])


def gripper_footprint(model: ArmModel, slice_height_m: float, robot_cfg: dict, simplify_m: float = 2e-4
                      ) -> tuple[np.ndarray, float]:
    """Footprint of the downward-pointing gripper (gripper z = world up): the 2D convex hull of the gripper and
    jaw hull vertices within `slice_height_m` of the lowest point, in gripper (x, y) relative to the TCP.
    Returns (footprint (n, 2), tip_depth_m = TCP height above the lowest point)."""
    from nedm.so101_push.geometry import convex_hull_2d, simplify_convex

    pts = gripper_points_local(model, robot_cfg)
    zmin = pts[:, 2].min()
    sl = pts[pts[:, 2] <= zmin + slice_height_m]
    fp = convex_hull_2d(sl[:, :2] - model.tcp_local[:2])
    if simplify_m > 0:
        fp = simplify_convex(fp, simplify_m)
    return fp, float(model.tcp_local[2] - zmin)
