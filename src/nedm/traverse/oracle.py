"""Privileged oracle planner for the HMMWV traversal study (plan §7).

Direction-aware A* on the true heightmap + asset layout, spline-free smoothing
(shortcut + Chaikin, numpy-only), a slope/curvature-modulated speed profile,
and mandatory post-smoothing feasibility validation (footprint sweep, curvature
cap, slope caps). Emits ``PlanCandidate`` — the single format shared by oracle,
Planner-B, and Planner-C scoring (plan §9.5).

Privileged by design: touches the true ``TerrainMap`` and obstacle list. Only
training/eval code may import it for labels, brackets, and reachability checks.

The v0 energy model is an uncalibrated slope proxy, replaced by the pilot-fitted
regression (plan §7.1) once driving data exists; the interface already separates
distance and energy terms so recalibration is a coefficient swap.
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nedm.traverse.terrain import TerrainMap

Obstacle = tuple[float, float, float]  # (x, y, footprint radius) in world meters


@dataclass
class PlannerParams:
    grid_res_m: float = 0.5
    # Vehicle geometry (HMMWV_Full: ~4.7 x 2.2 m, min turn radius ~8 m).
    footprint_length_m: float = 5.2  # includes margin
    footprint_width_m: float = 2.6
    kappa_max: float = 1.0 / 8.0
    # Inflation budget v0 = half width + fixed margin; the tracker's held-out
    # p95 lateral error is ADDED here once G6 measures it (plan §7.4).
    inflation_m: float = 2.0
    tracker_p95_margin_m: float = 0.0
    # Search-time slope caps (tan). Validation uses slightly looser caps so
    # smoothing across a cell corner is not brittle.
    slope_along_cap: float = math.tan(math.radians(20.0))
    slope_cross_cap: float = math.tan(math.radians(15.0))
    validate_slope_slack: float = 1.15
    # v0 energy proxy per meter: 1 + k_up*max(s,0) + k_down*max(-s,0).
    energy_k_up: float = 6.0
    energy_k_down: float = 1.5
    energy_weight: float = 1.0
    # Speed profile.
    v_cruise_mps: float = 7.0
    v_min_mps: float = 2.0
    a_lat_max: float = 2.0
    a_accel: float = 1.5
    a_decel: float = 2.0
    v_terminal_mps: float = 1.5
    terminal_taper_m: float = 4.0
    # Goal: approach ring around the house center (plan §3.2).
    approach_ring_m: float = 7.0
    approach_ring_tol_m: float = 0.75
    # Smoothing.
    sample_step_m: float = 0.5
    chaikin_iterations: int = 2
    curvature_repair_iterations: int = 12
    arena_keep_within_m: float = 38.0


@dataclass
class PlanCandidate:
    """Shared plan format: world-frame geometry + speed profile (plan §9.5)."""

    waypoints: np.ndarray  # (N, 2) world x, y
    speeds: np.ndarray  # (N,) target speed at each waypoint
    headings: np.ndarray  # (N,) path tangent yaw
    stations: np.ndarray  # (N,) cumulative arc length
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def length_m(self) -> float:
        return float(self.stations[-1])

    def to_json(self) -> dict[str, Any]:
        return {
            "waypoints": self.waypoints.tolist(),
            "speeds": self.speeds.tolist(),
            "headings": self.headings.tolist(),
            "stations": self.stations.tolist(),
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "PlanCandidate":
        return cls(
            waypoints=np.asarray(payload["waypoints"], dtype=np.float64),
            speeds=np.asarray(payload["speeds"], dtype=np.float64),
            headings=np.asarray(payload["headings"], dtype=np.float64),
            stations=np.asarray(payload["stations"], dtype=np.float64),
            meta=dict(payload.get("meta", {})),
        )

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_json(), handle)


class OracleGrid:
    """Planner-resolution view of the true map + obstacles."""

    def __init__(self, tmap: TerrainMap, obstacles: Sequence[Obstacle], params: PlannerParams):
        self.tmap = tmap
        self.params = params
        self.obstacles = list(obstacles)
        half = params.arena_keep_within_m
        self.n = int(round(2 * half / params.grid_res_m)) + 1
        self.origin = -half
        coords = self.origin + np.arange(self.n) * params.grid_res_m
        self.coords = coords
        gx_w, gy_w = np.meshgrid(coords, coords)  # [iy, ix]
        self.node_x = gx_w
        self.node_y = gy_w
        self.node_h = tmap.height(gx_w, gy_w)
        ggx, ggy = tmap.gradient(gx_w, gy_w)
        self.node_gx = ggx
        self.node_gy = ggy
        node_slope = np.sqrt(ggx**2 + ggy**2)

        inflate = params.inflation_m + params.tracker_p95_margin_m
        occupied = np.zeros_like(node_slope, dtype=bool)
        for ox, oy, orad in self.obstacles:
            occupied |= (gx_w - ox) ** 2 + (gy_w - oy) ** 2 <= (orad + inflate) ** 2
        self.occupied = occupied
        # Direction-agnostic node gate; direction-aware caps re-checked per edge.
        self.valid = (~occupied) & (node_slope <= 1.25 * params.slope_along_cap)

    def to_index(self, x: float, y: float) -> tuple[int, int]:
        ix = int(round((x - self.origin) / self.params.grid_res_m))
        iy = int(round((y - self.origin) / self.params.grid_res_m))
        return min(max(iy, 0), self.n - 1), min(max(ix, 0), self.n - 1)

    def clearance(self, x: Any, y: Any) -> np.ndarray:
        """Distance to the nearest obstacle FOOTPRINT edge (uninflated)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        best = np.full(np.broadcast(x, y).shape, np.inf)
        for ox, oy, orad in self.obstacles:
            best = np.minimum(best, np.hypot(x - ox, y - oy) - orad)
        return best


_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _energy_per_m(s_along: np.ndarray | float, params: PlannerParams) -> np.ndarray | float:
    return 1.0 + params.energy_weight * (
        params.energy_k_up * np.maximum(s_along, 0.0)
        + params.energy_k_down * np.maximum(-s_along, 0.0)
    )


def a_star(grid: OracleGrid, start_xy: tuple[float, float], goal_mask: np.ndarray) -> list[tuple[int, int]] | None:
    """Direction-aware A* to ANY goal-mask node. Returns index path or None."""
    p = grid.params
    res = p.grid_res_m
    start = grid.to_index(*start_xy)
    if not grid.valid[start]:
        return None
    if not goal_mask.any():
        return None
    goal_ys, goal_xs = np.nonzero(goal_mask)
    goal_pts = np.stack([grid.coords[goal_xs], grid.coords[goal_ys]], axis=1)

    def heuristic(iy: int, ix: int) -> float:
        dx = goal_pts[:, 0] - grid.coords[ix]
        dy = goal_pts[:, 1] - grid.coords[iy]
        return float(np.sqrt(dx**2 + dy**2).min())

    g_cost = np.full((grid.n, grid.n), np.inf)
    g_cost[start] = 0.0
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    open_heap: list[tuple[float, tuple[int, int]]] = [(heuristic(*start), start)]
    closed = np.zeros((grid.n, grid.n), dtype=bool)

    while open_heap:
        _, node = heapq.heappop(open_heap)
        if closed[node]:
            continue
        closed[node] = True
        if goal_mask[node]:
            path = [node]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            return path[::-1]
        iy, ix = node
        for diy, dix in _NEIGHBORS:
            jy, jx = iy + diy, ix + dix
            if not (0 <= jy < grid.n and 0 <= jx < grid.n):
                continue
            if closed[jy, jx] or not grid.valid[jy, jx]:
                continue
            dist = res * math.hypot(diy, dix)
            s_along = (grid.node_h[jy, jx] - grid.node_h[iy, ix]) / dist
            if abs(s_along) > p.slope_along_cap:
                continue
            # Cross slope from the midpoint gradient projected on the edge normal.
            mgx = 0.5 * (grid.node_gx[iy, ix] + grid.node_gx[jy, jx])
            mgy = 0.5 * (grid.node_gy[iy, ix] + grid.node_gy[jy, jx])
            tx, ty = dix / math.hypot(diy, dix), diy / math.hypot(diy, dix)
            s_cross = abs(-ty * mgx + tx * mgy)
            if s_cross > p.slope_cross_cap:
                continue
            cost = g_cost[iy, ix] + dist * float(_energy_per_m(s_along, p))
            if cost < g_cost[jy, jx]:
                g_cost[jy, jx] = cost
                parent[(jy, jx)] = node
                heapq.heappush(open_heap, (cost + heuristic(jy, jx), (jy, jx)))
    return None


def _segment_valid(grid: OracleGrid, a: np.ndarray, b: np.ndarray) -> bool:
    """Sampled validity of a straight segment (inflated clearance + slope caps)."""
    p = grid.params
    length = float(np.hypot(*(b - a)))
    if length < 1e-6:
        return True
    n = max(2, int(math.ceil(length / 0.25)) + 1)
    ts = np.linspace(0.0, 1.0, n)
    xs = a[0] + ts * (b[0] - a[0])
    ys = a[1] + ts * (b[1] - a[1])
    if np.any(np.abs(xs) > p.arena_keep_within_m) or np.any(np.abs(ys) > p.arena_keep_within_m):
        return False
    if np.any(grid.clearance(xs, ys) < p.inflation_m + p.tracker_p95_margin_m):
        return False
    gx, gy = grid.tmap.gradient(xs, ys)
    tx, ty = (b - a) / length
    s_along = gx * tx + gy * ty
    s_cross = -ty * gx + tx * gy
    return bool(
        np.all(np.abs(s_along) <= p.slope_along_cap) and np.all(np.abs(s_cross) <= p.slope_cross_cap)
    )


def _shortcut(grid: OracleGrid, points: np.ndarray) -> np.ndarray:
    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not _segment_valid(grid, points[i], points[j]):
            j -= 1
        out.append(points[j])
        i = j
    return np.asarray(out)


def _chaikin(points: np.ndarray, iterations: int) -> np.ndarray:
    pts = points
    for _ in range(iterations):
        if len(pts) < 3:
            return pts
        new_pts = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            new_pts.append(0.75 * a + 0.25 * b)
            new_pts.append(0.25 * a + 0.75 * b)
        new_pts.append(pts[-1])
        pts = np.asarray(new_pts)
    return pts


def _resample(points: np.ndarray, step: float) -> np.ndarray:
    seg = np.hypot(*np.diff(points, axis=0).T)
    station = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(station[-1])
    n = max(2, int(round(total / step)) + 1)
    targets = np.linspace(0.0, total, n)
    return np.stack(
        [np.interp(targets, station, points[:, 0]), np.interp(targets, station, points[:, 1])],
        axis=1,
    )


def _curvature(points: np.ndarray) -> np.ndarray:
    """Discrete curvature at interior points (three-point), zero at endpoints."""
    kappa = np.zeros(len(points))
    a = points[:-2]
    b = points[1:-1]
    c = points[2:]
    ab = b - a
    bc = c - b
    ac = c - a
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    denom = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ac, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa[1:-1] = np.where(denom > 1e-9, 2.0 * np.abs(cross) / denom, 0.0)
    return kappa


def _footprint_offsets(params: PlannerParams) -> np.ndarray:
    hl = params.footprint_length_m / 2.0
    hw = params.footprint_width_m / 2.0
    return np.array(
        [[hl, hw], [hl, -hw], [-hl, hw], [-hl, -hw], [hl, 0], [-hl, 0], [0, hw], [0, -hw]]
    )


def validate_candidate(grid: OracleGrid, points: np.ndarray, params: PlannerParams) -> dict[str, Any]:
    """Post-smoothing feasibility (plan §7.3): curvature cap, full-footprint
    swept clearance vs UNINFLATED footprints, slope caps with validation slack."""
    kappa = _curvature(points)
    tangents = np.gradient(points, axis=0)
    norms = np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    tangents = tangents / norms
    headings = np.arctan2(tangents[:, 1], tangents[:, 0])

    offsets = _footprint_offsets(params)
    cos_h, sin_h = np.cos(headings), np.sin(headings)
    fx = points[:, 0][:, None] + offsets[:, 0] * cos_h[:, None] - offsets[:, 1] * sin_h[:, None]
    fy = points[:, 1][:, None] + offsets[:, 0] * sin_h[:, None] + offsets[:, 1] * cos_h[:, None]
    clearance = grid.clearance(fx.ravel(), fy.ravel()).min()
    in_bounds = bool(
        np.all(np.abs(fx) <= grid.params.arena_keep_within_m + 1.0)
        and np.all(np.abs(fy) <= grid.params.arena_keep_within_m + 1.0)
    )

    gx, gy = grid.tmap.gradient(points[:, 0], points[:, 1])
    s_along = gx * tangents[:, 0] + gy * tangents[:, 1]
    s_cross = -tangents[:, 1] * gx + tangents[:, 0] * gy
    slack = params.validate_slope_slack
    return {
        "kappa_max": float(kappa.max()),
        "kappa_ok": bool(kappa.max() <= params.kappa_max),
        "min_footprint_clearance_m": float(clearance),
        "clearance_ok": bool(clearance > 0.0),
        "in_bounds": in_bounds,
        "slope_along_max": float(np.abs(s_along).max()),
        "slope_cross_max": float(np.abs(s_cross).max()),
        "slope_ok": bool(
            np.abs(s_along).max() <= slack * params.slope_along_cap
            and np.abs(s_cross).max() <= slack * params.slope_cross_cap
        ),
    }


def _repair_curvature(points: np.ndarray, params: PlannerParams) -> np.ndarray:
    pts = points.copy()
    for _ in range(params.curvature_repair_iterations):
        kappa = _curvature(pts)
        if kappa.max() <= params.kappa_max:
            break
        bad = np.nonzero(kappa > params.kappa_max)[0]
        smoothed = pts.copy()
        for idx in bad:
            lo = max(1, idx - 2)
            hi = min(len(pts) - 2, idx + 2)
            for j in range(lo, hi + 1):
                smoothed[j] = 0.5 * pts[j] + 0.25 * pts[j - 1] + 0.25 * pts[j + 1]
        pts = smoothed
    return pts


def speed_profile(points: np.ndarray, grid: OracleGrid, params: PlannerParams) -> np.ndarray:
    """v = min(cruise, curvature cap, slope cap), then forward/backward ramps
    and a terminal taper into the approach pose."""
    kappa = _curvature(points)
    v_curv = np.sqrt(params.a_lat_max / np.maximum(kappa, 1e-6))
    gx, gy = grid.tmap.gradient(points[:, 0], points[:, 1])
    slope = np.sqrt(gx**2 + gy**2)
    v_slope = params.v_cruise_mps * (
        1.0 - 0.65 * np.minimum(slope / params.slope_along_cap, 1.0)
    )
    v = np.minimum(params.v_cruise_mps, np.minimum(v_curv, v_slope))
    v = np.maximum(v, params.v_min_mps)

    seg = np.hypot(*np.diff(points, axis=0).T)
    station = np.concatenate([[0.0], np.cumsum(seg)])
    remaining = station[-1] - station
    taper = np.clip(remaining / params.terminal_taper_m, 0.0, 1.0)
    v = v * taper + params.v_terminal_mps * (1.0 - taper)

    for i in range(1, len(v)):  # forward accel limit
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * params.a_accel * seg[i - 1]))
    for i in range(len(v) - 2, -1, -1):  # backward decel limit
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * params.a_decel * seg[i]))
    return v


def plan_to_ring(
    tmap: TerrainMap,
    obstacles: Sequence[Obstacle],
    start_xy: tuple[float, float],
    ring_center: tuple[float, float],
    params: PlannerParams | None = None,
) -> PlanCandidate | None:
    """Full oracle pipeline: multi-goal A* to the approach ring, smooth,
    validate, attach speeds. Returns None when no feasible plan exists."""
    params = params or PlannerParams()
    grid = OracleGrid(tmap, obstacles, params)

    ring_dist = np.hypot(grid.node_x - ring_center[0], grid.node_y - ring_center[1])
    goal_mask = (
        (np.abs(ring_dist - params.approach_ring_m) <= params.approach_ring_tol_m) & grid.valid
    )
    idx_path = a_star(grid, start_xy, goal_mask)
    if idx_path is None:
        return None
    points = np.array([[grid.coords[ix], grid.coords[iy]] for iy, ix in idx_path])
    points[0] = np.asarray(start_xy)

    points = _shortcut(grid, points)
    points = _chaikin(points, params.chaikin_iterations)
    points = _resample(points, params.sample_step_m)
    points = _repair_curvature(points, params)
    checks = validate_candidate(grid, points, params)
    if not (checks["kappa_ok"] and checks["clearance_ok"] and checks["in_bounds"] and checks["slope_ok"]):
        return None

    speeds = speed_profile(points, grid, params)
    tangents = np.gradient(points, axis=0)
    headings = np.arctan2(tangents[:, 1], tangents[:, 0])
    seg = np.hypot(*np.diff(points, axis=0).T)
    stations = np.concatenate([[0.0], np.cumsum(seg)])

    gx, gy = tmap.gradient(points[:, 0], points[:, 1])
    tx = np.cos(headings)
    ty = np.sin(headings)
    s_along = gx * tx + gy * ty
    energy_proxy = float(np.sum(seg * np.asarray(_energy_per_m(s_along, params))[:-1]))

    approach_heading = math.atan2(ring_center[1] - points[-1, 1], ring_center[0] - points[-1, 0])
    return PlanCandidate(
        waypoints=points,
        speeds=speeds,
        headings=headings,
        stations=stations,
        meta={
            "checks": checks,
            "length_m": float(stations[-1]),
            "energy_proxy": energy_proxy,
            "straight_line_m": float(np.hypot(points[-1, 0] - points[0, 0], points[-1, 1] - points[0, 1])),
            "approach_pose": [float(points[-1, 0]), float(points[-1, 1]), approach_heading],
            "ring_center": [float(ring_center[0]), float(ring_center[1])],
            "est_duration_s": float(np.sum(seg / np.maximum(0.5 * (speeds[:-1] + speeds[1:]), 0.5))),
            "params": {"inflation_m": params.inflation_m, "energy_weight": params.energy_weight},
        },
    )
