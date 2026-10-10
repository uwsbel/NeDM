"""Arm-to-T clearance (numpy only): exact finger-box distances and a vertex check for the other links.

Used three ways:
  * planning (scripted.Planner): the near-miss gap is set with the real 3D finger geometry, and every planned
    waypoint is checked so that the links base .. wrist stay `plan_min_m` away from the T;
  * recording (sim.PushScene): per record, the signed fingertip gap and the link gap from the simulated body frames;
  * QA (episode.simulate): an episode is rejected when a link other than the fingertips comes near the T.

Fingertips: the 104 oriented boxes (so101_finger_boxes.json). The distance between two disjoint boxes is the
minimum over corner-to-box distances (both ways) and edge-to-edge segment distances; this is exact for convex
polytopes. For overlapping boxes the value is minus the deepest corner (or 0 when only edges cross), which is
enough to flag penetration.

Other links (base, shoulder, upper_arm, lower_arm, wrist) and the upper finger parts are convex hulls; their
gap is the smallest signed distance of a hull vertex to the T boxes (a hull face can come a little closer than
its vertices, so this is used with margins of millimetres, never as a contact test).
"""
from __future__ import annotations

import math

import numpy as np

from nedm.so101_push.robot import link_collision_shapes, quat_to_matrix, rot_z

FINGER_LINKS = ("gripper", "jaw")
OTHER_LINKS = ("base", "shoulder", "upper_arm", "lower_arm", "wrist")
_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
_EDGES = np.array([(i, j) for i in range(8) for j in range(i + 1, 8)
                   if np.sum(_SIGNS[i] != _SIGNS[j]) == 1])            # 12 box edges (corner index pairs)


# ----------------------------------------------------------------------------- primitives
def sdf_box(P: np.ndarray, c: np.ndarray, R: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Signed distance of points P (..., 3) to the box (centre c, rotation columns R, half sizes h)."""
    q = np.abs((P - c) @ R) - h
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(-1), 0.0)


def box_corners(c, R, h) -> np.ndarray:
    """Corners (..., 8, 3) of boxes c (..., 3), R (..., 3, 3), h (..., 3)."""
    local = _SIGNS * np.asarray(h)[..., None, :]                       # (..., 8, 3)
    return np.einsum("...ij,...kj->...ki", R, local) + np.asarray(c)[..., None, :]


def segment_distance(p1, d1, p2, d2) -> np.ndarray:
    """Distance between segments p1 + s d1 and p2 + t d2 (s, t in [0, 1]), broadcast over leading axes
    (Ericson, Real-Time Collision Detection 5.1.9)."""
    r = p1 - p2
    a = np.sum(d1 * d1, -1)
    e = np.sum(d2 * d2, -1)
    f = np.sum(d2 * r, -1)
    c = np.sum(d1 * r, -1)
    b = np.sum(d1 * d2, -1)
    den = a * e - b * b
    s = np.where(den > 1e-20, np.clip((b * f - c * e) / np.where(den > 1e-20, den, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(p1 + s[..., None] * d1 - p2 - t[..., None] * d2, axis=-1)


def _corner_distance(cA, RA, hA, KA, cB, RB, hB, KB) -> np.ndarray:
    """Smallest corner-to-box signed distance, both ways (an upper bound of the box distance when disjoint)."""
    dA = sdf_box(KA, cB, RB, hB).min(-1)                               # A corners vs B
    loc = np.einsum("nij,nkj->nki", np.transpose(RA, (0, 2, 1)), KB[None] - cA[:, None, :])   # B corners in A frames
    q = np.abs(loc) - hA[:, None, :]
    dB = (np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(-1), 0.0)).min(-1)
    return np.minimum(dA, dB)


def _edge_distance(KA, KB) -> np.ndarray:
    """Smallest edge-to-edge distance between boxes A_i (corners KA (n, 8, 3)) and box B (corners KB (8, 3))."""
    pA, eA = KA[:, _EDGES[:, 0]], KA[:, _EDGES[:, 1]] - KA[:, _EDGES[:, 0]]          # (n, 12, 3)
    pB, eB = KB[_EDGES[:, 0]], KB[_EDGES[:, 1]] - KB[_EDGES[:, 0]]                    # (12, 3)
    return segment_distance(pA[:, :, None], eA[:, :, None], pB[None, None], eB[None, None]).min(axis=(1, 2))


def boxes_to_box_distance(cA, RA, hA, cB, RB, hB) -> np.ndarray:
    """Distance from each box A_i (cA (n, 3), RA (n, 3, 3), hA (n, 3)) to one box B. Exact when disjoint
    (corner-to-box both ways and edge-to-edge); minus the deepest corner (or 0) when they overlap."""
    KA = box_corners(cA, RA, hA)                                       # (n, 8, 3)
    KB = box_corners(cB, RB, hB)                                       # (8, 3)
    corner = _corner_distance(cA, RA, hA, KA, cB, RB, hB, KB)
    return np.where(corner < 0.0, corner, np.minimum(corner, _edge_distance(KA, KB)))


def t_boxes(tshape, com, R) -> list:
    """World boxes (centre, rotation, half sizes) of the bar and the stem for the T COM and rotation."""
    com, R = np.asarray(com, float), np.asarray(R, float)
    return [(com + R @ np.r_[tshape.bar_center, 0.0], R, np.array([tshape.bar_l, tshape.bar_w, tshape.height]) / 2),
            (com + R @ np.r_[tshape.stem_center, 0.0], R, np.array([tshape.stem_w, tshape.stem_l, tshape.height]) / 2)]


def t_boxes_planar(tshape, pose) -> list:
    """T boxes for a planar pose (x, y, yaw) resting on the table."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return t_boxes(tshape, [pose[0], pose[1], tshape.height / 2], R)


def t_boxes_state(tshape, state13) -> list:
    return t_boxes(tshape, state13[:3], quat_to_matrix(state13[3:7]))


# ----------------------------------------------------------------------------- arm geometry
class ArmTClearance:
    """Collision geometry of the arm in link frames, with distance queries against the T boxes."""

    def __init__(self, model, robot_cfg: dict, tshape):
        self.model = model
        self.ts = tshape
        # fingertip boxes (link frame) and convex hulls (other links and the upper finger parts)
        self.fbox = {}
        self.hulls = {}                       # link -> (centres (n, 3), radii (n,), [points (k, 3)])
        for ln in FINGER_LINKS + OTHER_LINKS:
            boxes, pts = [], []
            for sh in link_collision_shapes(ln, robot_cfg):
                if sh["kind"] == "box":
                    boxes.append((sh["center"], sh["rot"], sh["half"]))
                else:
                    pts.append(np.asarray(sh["points"], float))
            if boxes:
                self.fbox[ln] = (np.array([b[0] for b in boxes]), np.array([b[1] for b in boxes]),
                                 np.array([b[2] for b in boxes]))
            cen = np.array([0.5 * (P.min(0) + P.max(0)) for P in pts]).reshape(-1, 3)
            rad = np.array([np.linalg.norm(P - c, axis=1).max() for P, c in zip(pts, cen)])
            self.hulls[ln] = (cen, rad, pts)
        self.fbox_radius = {ln: np.linalg.norm(v[2], axis=1) for ln, v in self.fbox.items()}
        # the jaw link in the gripper frame (locked), for planned gripper poses without IK
        j = model.joints.index("Jaw")
        self.T_jaw_in_gripper = model.F0[j] @ rot_z(model.jaw_lock) @ model.F1inv[j]

    # ------------------------------------------------------------------ fingertips
    def finger_gap(self, frames: dict, boxes: list, cap: float = 0.02) -> float:
        """Smallest signed distance between the fingertip boxes and the T boxes. frames: link -> 4x4 world pose
        (gripper and jaw). Exact below `cap`; above it the bounding-sphere lower bound is returned."""
        lower = math.inf                  # lower bound over the boxes not checked exactly
        cand = []
        for ln, (c, R, h) in self.fbox.items():
            T = frames[ln]
            cw = c @ T[:3, :3].T + T[:3, 3]
            for cb, Rb, hb in boxes:
                lb = sdf_box(cw, cb, Rb, hb) - self.fbox_radius[ln]
                near = lb < cap
                if not near.all():
                    lower = min(lower, float(lb[~near].min()))
                if near.any():
                    Rw = np.einsum("ij,njk->nik", T[:3, :3], R[near])
                    cand.append((cw[near], Rw, h[near], lb[near], cb, Rb, hb))
        if not cand:
            return lower
        # corner distances bound the answer from above; edge pairs only for boxes that can beat that bound
        upper = math.inf
        parts = []
        for cA, RA, hA, lb, cb, Rb, hb in cand:
            KA = box_corners(cA, RA, hA)
            KB = box_corners(cb, Rb, hb)
            corner = _corner_distance(cA, RA, hA, KA, cb, Rb, hb, KB)
            upper = min(upper, float(corner.min()))
            parts.append((KA, KB, lb, corner))
        if upper < 0.0:
            return upper
        best = upper
        for KA, KB, lb, corner in parts:
            m = lb < best
            if m.any():
                best = min(best, float(_edge_distance(KA[m], KB).min()))
        return min(best, lower)

    def planned_frames(self, tcp, yaw) -> dict:
        """Gripper and jaw link poses for a planned TCP pose (fingers straight down, gripper x heading = yaw)."""
        c, s = math.cos(yaw), math.sin(yaw)
        T = np.eye(4)
        T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        T[:3, 3] = np.asarray(tcp, float) - T[:3, :3] @ self.model.tcp_local
        return {"gripper": T, "jaw": T @ self.T_jaw_in_gripper}

    def planned_finger_gap(self, tcp, yaw, t_pose, cap: float = 0.02) -> float:
        return self.finger_gap(self.planned_frames(tcp, yaw), t_boxes_planar(self.ts, t_pose), cap)

    # ------------------------------------------------------------------ hull links
    def hull_gap(self, frames: dict, boxes: list, links=OTHER_LINKS, cap: float = 0.03) -> tuple[float, str]:
        """Smallest signed distance of a hull vertex of `links` to the T boxes. Hulls whose bounding sphere is
        farther than `cap` are skipped (their sphere lower bound counts instead). Returns (gap, link)."""
        best, where = math.inf, ""
        for ln in links:
            cen, rad, pts = self.hulls[ln]
            if not len(pts):
                continue
            T = frames[ln]
            cw = cen @ T[:3, :3].T + T[:3, 3]
            lb = np.min([sdf_box(cw, cb, Rb, hb) for cb, Rb, hb in boxes], axis=0) - rad
            near = np.flatnonzero(lb < cap)
            far = lb[lb >= cap]
            if len(far) and far.min() < best:
                best, where = float(far.min()), ln
            if not len(near):
                continue
            W = np.concatenate([pts[i] for i in near]) @ T[:3, :3].T + T[:3, 3]
            d = min(float(sdf_box(W, cb, Rb, hb).min()) for cb, Rb, hb in boxes)
            if d < best:
                best, where = d, ln
        return best, where

    def planned_link_gap(self, q5, t_pose, links=OTHER_LINKS, cap: float = 0.03) -> tuple[float, str]:
        return self.hull_gap(self.model.link_poses(q5), t_boxes_planar(self.ts, t_pose), links, cap)

    # ------------------------------------------------------------------ keep-out around the base
    def base_keepout(self, z_top: float):
        """2D keep-out of the fixed base and the panning shoulder below z_top: (base polygon (k, 2), pan axis xy,
        shoulder radius). The shoulder turns with the pan joint, so its keep-out is a disc about the pan axis."""
        from nedm.so101_push.geometry import convex_hull_2d

        poses = self.model.link_poses(np.zeros(5))
        Tb = poses["base"]
        P = np.concatenate(self.hulls["base"][2]) @ Tb[:3, :3].T + Tb[:3, 3]
        poly = convex_hull_2d(P[P[:, 2] < z_top][:, :2])
        F = self.model.joint_frames(poses)[0]
        axis = F[:3, 3][:2].copy()
        Ts = poses["shoulder"]
        S = np.concatenate(self.hulls["shoulder"][2]) @ Ts[:3, :3].T + Ts[:3, 3]
        low = S[S[:, 2] < z_top]
        r = float(np.hypot(low[:, 0] - axis[0], low[:, 1] - axis[1]).max()) if len(low) else 0.0
        return poly, axis, r
