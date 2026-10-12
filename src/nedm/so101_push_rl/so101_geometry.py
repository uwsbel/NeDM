"""T-shape geometry, gripper footprint and small 2D convex-polygon helpers (numpy only).

T frame: origin at the centre of mass of the whole T (both boxes), x along the bar, y from the bar
towards the stem, z up (the bottom face lies at z = -height/2). The bar centre is at (0, -com_y)
and the stem centre at (0, bar_w/2 + stem_l/2 - com_y) in this frame.

Faces (outline edges, counter-clockwise; outward normals in the T frame):
  0 bar_back       -y   long face of the bar, opposite the stem
  1 bar_end_pos    +x   short end of the bar at +x
  2 bar_front_pos  +y   bar shoulder next to the stem, x > 0
  3 stem_side_pos  +x   long side of the stem at +x
  4 stem_end       +y   short end of the stem
  5 stem_side_neg  -x   long side of the stem at -x
  6 bar_front_neg  +y   bar shoulder next to the stem, x < 0
  7 bar_end_neg    -x   short end of the bar at -x
"""
from __future__ import annotations

import math

import numpy as np

FACE_NAMES = ["bar_back", "bar_end_pos", "bar_front_pos", "stem_side_pos", "stem_end", "stem_side_neg",
              "bar_front_neg", "bar_end_neg"]


def rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class TShape:
    def __init__(self, cfg: dict):
        self.bar_l, self.bar_w = (float(v) for v in cfg["bar_size_m"])     # along x, along y
        self.stem_w, self.stem_l = (float(v) for v in cfg["stem_size_m"])  # along x, along y
        self.height = float(cfg["height_m"])
        rho = float(cfg["density_kgpm3"])
        self.m_bar = rho * self.bar_l * self.bar_w * self.height
        self.m_stem = rho * self.stem_w * self.stem_l * self.height
        self.mass = self.m_bar + self.m_stem
        stem_cy = self.bar_w / 2 + self.stem_l / 2              # stem centre in the bar-centre frame
        self.com_y = self.m_stem * stem_cy / self.mass          # COM in the bar-centre frame (x = 0)
        self.bar_center = np.array([0.0, -self.com_y])
        self.stem_center = np.array([0.0, stem_cy - self.com_y])
        # rectangles (centre, half sizes) in the T frame
        self.rects = [(self.bar_center, np.array([self.bar_l / 2, self.bar_w / 2])),
                      (self.stem_center, np.array([self.stem_w / 2, self.stem_l / 2]))]
        yb0, yb1 = -self.bar_w / 2 - self.com_y, self.bar_w / 2 - self.com_y
        ys1 = yb1 + self.stem_l
        hb, hs = self.bar_l / 2, self.stem_w / 2
        self.outline = np.array([[-hb, yb0], [hb, yb0], [hb, yb1], [hs, yb1], [hs, ys1], [-hs, ys1], [-hs, yb1],
                                 [-hb, yb1]])
        self.faces = []
        for i, name in enumerate(FACE_NAMES):
            a, b = self.outline[i], self.outline[(i + 1) % 8]
            t = (b - a) / np.linalg.norm(b - a)
            n = np.array([t[1], -t[0]])                          # outward for a counter-clockwise outline
            self.faces.append({"id": i, "name": name, "a": a, "b": b, "tangent": t, "normal": n,
                               "length": float(np.linalg.norm(b - a))})
        # box inertias about their own centres (principal axes = T axes)
        def box_inertia(m, sx, sy, sz):
            return [m / 12 * (sy * sy + sz * sz), m / 12 * (sx * sx + sz * sz), m / 12 * (sx * sx + sy * sy)]
        self.bar_inertia = box_inertia(self.m_bar, self.bar_l, self.bar_w, self.height)
        self.stem_inertia = box_inertia(self.m_stem, self.stem_w, self.stem_l, self.height)
        # yaw inertia of the whole T about its COM (for reference)
        self.izz = (self.bar_inertia[2] + self.m_bar * self.com_y ** 2 + self.stem_inertia[2]
                    + self.m_stem * (stem_cy - self.com_y) ** 2)

    def summary(self) -> dict:
        return {"bar_size_m": [self.bar_l, self.bar_w, self.height], "stem_size_m": [self.stem_w, self.stem_l, self.height],
                "bar_mass_kg": self.m_bar, "stem_mass_kg": self.m_stem, "mass_kg": self.mass,
                "com_from_bar_centre_m": [0.0, self.com_y], "bar_centre_in_T_m": self.bar_center.tolist(),
                "stem_centre_in_T_m": self.stem_center.tolist(), "bar_inertia_diag_kgm2": self.bar_inertia,
                "stem_inertia_diag_kgm2": self.stem_inertia, "izz_about_com_kgm2": self.izz,
                "faces": [{"id": f["id"], "name": f["name"], "a": f["a"].tolist(), "b": f["b"].tolist(),
                           "normal": f["normal"].tolist(), "length_m": f["length"]} for f in self.faces]}

    def world_rects(self, pose) -> list[np.ndarray]:
        """Corner arrays (4, 2) of the bar and the stem for a planar pose (x, y, yaw)."""
        R = rot2(pose[2])
        out = []
        for c, h in self.rects:
            corners = np.array([[-h[0], -h[1]], [h[0], -h[1]], [h[0], h[1]], [-h[0], h[1]]]) + c
            out.append(corners @ R.T + np.asarray(pose[:2]))
        return out

    def to_world(self, pose, p_local) -> np.ndarray:
        return rot2(pose[2]) @ np.asarray(p_local) + np.asarray(pose[:2])

    def dir_to_world(self, pose, d_local) -> np.ndarray:
        return rot2(pose[2]) @ np.asarray(d_local)

    def to_local(self, pose, p_world) -> np.ndarray:
        return rot2(pose[2]).T @ (np.asarray(p_world) - np.asarray(pose[:2]))


# ----------------------------------------------------------------------------- convex polygons
def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull (monotone chain)."""
    pts = sorted(set(map(tuple, np.round(np.asarray(points, float), 9))))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def simplify_convex(P: np.ndarray, tol: float) -> np.ndarray:
    """Drop hull vertices that lie within `tol` of the segment joining their neighbours (inner approximation)."""
    P = [np.asarray(p, float) for p in P]
    while len(P) > 3:
        errs = []
        for i in range(len(P)):
            a, b, c = P[i - 1], P[i], P[(i + 1) % len(P)]
            ab = c - a
            errs.append(abs(ab[0] * (b[1] - a[1]) - ab[1] * (b[0] - a[0])) / max(np.linalg.norm(ab), 1e-18))
        i = int(np.argmin(errs))
        if errs[i] >= tol:
            break
        P.pop(i)
    return np.array(P)


def _points_to_edges(P: np.ndarray, Q: np.ndarray):
    """Distances from every vertex of P to every edge of polygon Q, and the closest points on Q."""
    a = Q
    ab = np.roll(Q, -1, axis=0) - Q                                    # (m, 2)
    ap = P[:, None, :] - a[None, :, :]                                  # (n, m, 2)
    t = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-18)[None], 0.0, 1.0)
    c = a[None] + t[..., None] * ab[None]                               # (n, m, 2)
    d = np.linalg.norm(P[:, None, :] - c, axis=-1)
    k = np.unravel_index(int(np.argmin(d)), d.shape)
    return float(d[k]), P[k[0]], c[k]


def polygon_distance(A: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Signed distance between convex polygons (> 0 apart, <= 0 overlapping: minus the SAT depth) and the
    closest points (on A, on B) when apart."""
    depth = math.inf
    for P in (A, B):
        e = np.roll(P, -1, axis=0) - P
        n = np.stack([e[:, 1], -e[:, 0]], axis=1) / np.maximum(np.linalg.norm(e, axis=1), 1e-18)[:, None]
        pa, pb = A @ n.T, B @ n.T                                       # (na, k), (nb, k)
        overlap = np.minimum(pa.max(0), pb.max(0)) - np.maximum(pa.min(0), pb.min(0))
        if overlap.min() <= 0:
            depth = None
            break
        depth = min(depth, float(overlap.min()))
    if depth is not None:
        return -depth, None, None
    d1, pa1, qb1 = _points_to_edges(A, B)
    d2, pb2, qa2 = _points_to_edges(B, A)
    return (d1, pa1, qb1) if d1 <= d2 else (d2, qa2, pb2)


def point_polygon_distance(p, poly: np.ndarray) -> float:
    """Signed distance of a point to a convex counter-clockwise polygon (negative inside)."""
    p = np.asarray(p, float)
    d, _, _ = _points_to_edges(p[None], np.asarray(poly, float))
    e = np.roll(poly, -1, axis=0) - poly
    cross = e[:, 0] * (p[1] - poly[:, 1]) - e[:, 1] * (p[0] - poly[:, 0])
    return -d if bool(np.all(cross >= 0)) else d


def footprint_world(footprint: np.ndarray, tcp_xy, yaw: float) -> np.ndarray:
    return footprint @ rot2(yaw).T + np.asarray(tcp_xy)


def distance_to_t(tshape: TShape, pose, poly: np.ndarray):
    """Minimum signed distance between a convex polygon and the T (two boxes) and the closest point on the T."""
    best = (math.inf, None)
    for rect in tshape.world_rects(pose):
        d, _, on_t = polygon_distance(poly, rect)
        if d < best[0]:
            best = (d, on_t)
    return best


def first_contact_along(tshape: TShape, pose, footprint, yaw, start_xy, direction, max_dist, tol=1e-4):
    """Translate the footprint from start_xy along a unit direction; return (s, contact point on the T) at the
    first touch, or (None, closest distance) if it never touches within max_dist."""
    d = np.asarray(direction, float)
    s = 0.0
    last_gap = None
    while s <= max_dist:
        gap, point = distance_to_t(tshape, pose, footprint_world(footprint, np.asarray(start_xy) + s * d, yaw))
        if gap <= tol:
            return s, point
        last_gap = gap
        s += max(gap * 0.9, tol)
    return None, last_gap


def min_gap_along(tshape: TShape, pose, footprint, yaw, path_xy: np.ndarray) -> float:
    return min(distance_to_t(tshape, pose, footprint_world(footprint, p, yaw))[0] for p in path_xy)
