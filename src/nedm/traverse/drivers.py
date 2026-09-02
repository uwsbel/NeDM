"""Collection driver mixture (plan §6.2).

Per-10-episode roster: 6 spline-following pure-pursuit over random smooth
routes (3–8 m/s, slope-modulated profile), 2 random meander, 1 near-obstacle
pass (alternating intended contact), 1 oracle-route follow (the WP0a vertical
slice as collection driver). Family is assigned by episode INDEX so any run
length keeps the 60/20/10/10 mixture; all steering is rate-limited by the
caller (repo convention 2.0 full-scale/s).

Route families return a ``PlanCandidate`` driven by ``ChPathFollowerDriver``;
meander returns a stateful 20 Hz controller instead (OU-noise steering + speed
governor + boundary/obstacle reflex).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from nedm.traverse.layout import EpisodeLayout
from nedm.traverse.oracle import (
    OracleGrid,
    PlanCandidate,
    PlannerParams,
    _chaikin,
    _repair_curvature,
    _resample,
    _segment_valid,
    speed_profile,
)
from nedm.traverse.terrain import TerrainMap

VEHICLE_HALF_WIDTH_M = 1.1  # HMMWV hull is ~2.2 m wide

FAMILY_CYCLE = (
    "spline", "meander", "spline", "spline", "oracle",
    "spline", "near_obstacle", "spline", "meander", "spline",
)


def assign_family(episode_index: int) -> str:
    return FAMILY_CYCLE[episode_index % len(FAMILY_CYCLE)]


def _finish_route(
    points: np.ndarray, grid: OracleGrid, params: PlannerParams, meta: dict
) -> PlanCandidate:
    speeds = speed_profile(points, grid, params)
    tangents = np.gradient(points, axis=0)
    headings = np.arctan2(tangents[:, 1], tangents[:, 0])
    seg = np.hypot(*np.diff(points, axis=0).T)
    stations = np.concatenate([[0.0], np.cumsum(seg)])
    meta = {
        **meta,
        "length_m": float(stations[-1]),
        "est_duration_s": float(np.sum(seg / np.maximum(0.5 * (speeds[:-1] + speeds[1:]), 0.5))),
    }
    return PlanCandidate(waypoints=points, speeds=speeds, headings=headings, stations=stations, meta=meta)


def _time_to_index(route: PlanCandidate, idx: int) -> float:
    """Speed-profile travel time from the start to waypoint ``idx``."""
    seg = np.diff(route.stations[: idx + 1])
    v_mid = np.maximum(0.5 * (route.speeds[:idx] + route.speeds[1 : idx + 1]), 0.5)
    return float(np.sum(seg / v_mid))


def random_spline_route(
    tmap: TerrainMap,
    layout: EpisodeLayout,
    rng: np.random.Generator,
    params: PlannerParams | None = None,
    min_length_m: float = 45.0,
    duration_s: float = 20.0,
) -> PlanCandidate | None:
    """Random smooth free-space route from the spawn (60% family).

    Routes must outlive the episode (est duration >= ~1.05x) so the vehicle
    never idles at a path endpoint inside the recorded window.
    """
    params = params or PlannerParams()
    grid = OracleGrid(tmap, layout.obstacles(), params)
    bound = params.arena_keep_within_m - 4.0

    for _ in range(30):
        pts = [np.asarray(layout.start_xy, dtype=np.float64)]
        heading = float(layout.start_yaw) + float(rng.uniform(-0.5, 0.5))
        for _leg in range(int(rng.integers(4, 7))):
            placed = False
            for _try in range(80):
                turn = float(rng.uniform(-1.2, 1.2))
                length = float(rng.uniform(12.0, 22.0))
                cand = pts[-1] + length * np.array([math.cos(heading + turn), math.sin(heading + turn)])
                if np.abs(cand).max() > bound:
                    # steer the next attempt back toward the arena center
                    heading = math.atan2(-pts[-1][1], -pts[-1][0]) + float(rng.uniform(-0.4, 0.4))
                    continue
                if not _segment_valid(grid, pts[-1], cand):
                    continue
                pts.append(cand)
                heading += turn
                placed = True
                break
            if not placed:
                break
        points = np.asarray(pts)
        if len(points) < 4:
            continue
        points = _chaikin(points, params.chaikin_iterations)
        points = _resample(points, params.sample_step_m)
        points = _repair_curvature(points, params)
        if float(np.hypot(*np.diff(points, axis=0).T).sum()) < min_length_m:
            continue
        route_params = replace(params, v_cruise_mps=float(rng.uniform(3.0, 8.0)))
        route = _finish_route(points, grid, route_params, {"family": "spline"})
        if route.meta["est_duration_s"] < 1.05 * duration_s:
            continue
        return route
    return None


def near_obstacle_route(
    tmap: TerrainMap,
    layout: EpisodeLayout,
    rng: np.random.Generator,
    contact_intended: bool,
    params: PlannerParams | None = None,
    duration_s: float = 20.0,
) -> PlanCandidate | None:
    """Pass close by one asset (10% family, half with intended contact).

    The pass legs deliberately violate the planner's inflation margin, so only
    the approach leg from the spawn is clearance-checked (against the OTHER
    assets); the pass segment keeps its geometry un-smoothed. The pass point
    must arrive within ~75% of the episode, or the recorded window ends before
    the interesting part (smoke-v1 lesson: an 86 m route never reached its
    target in 20 s).
    """
    params = params or PlannerParams()
    start = np.asarray(layout.start_xy, dtype=np.float64)
    targets = [a for a in layout.assets if a.kind in ("rock", "tree")]
    if not targets:
        return None
    # Closest targets first: at 2.5–4 m/s cruise, arrival time is dominated by
    # the start->target distance, and the pass must land inside the episode.
    targets.sort(key=lambda a: math.hypot(a.x_m - start[0], a.y_m - start[1]))

    for asset in targets[:6]:
        center = np.array([asset.x_m, asset.y_m])
        if not (10.0 < float(np.hypot(*(center - start))) < 35.0):
            continue
        # Aim relative to the PHYSICAL body, not the planner footprint (a
        # rock's footprint radius is its circumscribed-corner radius and a
        # tree's carries +0.4 m margin — grazing the footprint misses the box).
        # A graze means the HULL's side overlaps the asset by a fraction of the
        # vehicle half-width; aiming the CENTERLINE at the asset is a blocking
        # head-on crash that pins the vehicle (WP0c smoke run 3: 118 kN,
        # stuck 15 s at 0.4 m/s).
        physical_r = 0.5 * asset.dims["edge_m"] if asset.kind == "rock" else asset.dims["trunk_radius_m"]
        if contact_intended:  # sideswipe: 0.25–0.55 m hull overlap
            offset = float(rng.uniform(VEHICLE_HALF_WIDTH_M - 0.55, VEHICLE_HALF_WIDTH_M - 0.25))
        else:  # clear pass: hull edge misses by 0.7–1.7 m
            offset = float(rng.uniform(VEHICLE_HALF_WIDTH_M + 0.7, VEHICLE_HALF_WIDTH_M + 1.7))
        others = [o for o in layout.obstacles() if math.hypot(o[0] - asset.x_m, o[1] - asset.y_m) > 0.1]
        grid = OracleGrid(tmap, others, params)
        for _try in range(40):
            normal_ang = float(rng.uniform(0.0, 2.0 * math.pi))
            n_hat = np.array([math.cos(normal_ang), math.sin(normal_ang)])
            t_hat = np.array([-n_hat[1], n_hat[0]]) * float(rng.choice([-1.0, 1.0]))
            pass_pt = center + (physical_r + offset) * n_hat
            pre = pass_pt - 9.0 * t_hat
            post = pass_pt + 12.0 * t_hat
            far = pass_pt + 24.0 * t_hat
            legs = np.array([start, pre, pass_pt, post, far])
            if np.abs(legs).max() > params.arena_keep_within_m - 3.0:
                continue
            if not (_segment_valid(grid, start, pre) and _segment_valid(grid, post, far)):
                continue
            points = _resample(legs, params.sample_step_m)
            route_params = replace(params, v_cruise_mps=float(rng.uniform(2.5, 4.0)))
            route = _finish_route(
                points, grid, route_params,
                {
                    "family": "near_obstacle",
                    "target_asset": asset.to_json(),
                    "pass_offset_m": offset,
                    "physical_radius_m": physical_r,
                    "contact_intended": contact_intended,
                },
            )
            pass_idx = int(np.argmin(np.hypot(*(route.waypoints - pass_pt).T)))
            if _time_to_index(route, pass_idx) > 0.75 * duration_s:
                continue
            return route
    return None


@dataclass
class MeanderController:
    """20 Hz OU-noise steering + speed governor + boundary reflex (20% family)."""

    rng: np.random.Generator
    target_speed_mps: float = 0.0
    keep_within_m: float = 30.0
    ou_theta: float = 0.8
    ou_sigma: float = 0.9
    steer_clip: float = 0.75
    _ou: float = 0.0

    def __post_init__(self) -> None:
        if self.target_speed_mps <= 0.0:
            self.target_speed_mps = float(self.rng.uniform(2.0, 5.5))

    def __call__(
        self, dt_ctrl: float, x: float, y: float, yaw: float, speed: float
    ) -> tuple[float, float, float]:
        self._ou += self.ou_theta * (0.0 - self._ou) * dt_ctrl
        self._ou += self.ou_sigma * math.sqrt(dt_ctrl) * float(self.rng.standard_normal())
        steer = float(np.clip(self._ou, -self.steer_clip, self.steer_clip))

        # Boundary reflex: outside the keep-in radius, blend toward center
        # until the heading points back inside.
        r = math.hypot(x, y)
        if r > self.keep_within_m:
            to_center = math.atan2(-y, -x)
            err = math.atan2(math.sin(to_center - yaw), math.cos(to_center - yaw))
            blend = min(1.0, (r - self.keep_within_m) / 5.0)
            steer = (1.0 - blend) * steer + blend * float(np.clip(1.5 * err, -self.steer_clip, self.steer_clip))

        v_err = self.target_speed_mps - speed
        throttle = float(np.clip(0.18 + 0.28 * v_err, 0.0, 0.75))
        braking = float(np.clip(-0.4 * (v_err + 1.0), 0.0, 0.8)) if v_err < -1.0 else 0.0
        return steer, throttle, braking


def build_driver_route(
    family: str,
    tmap: TerrainMap,
    layout: EpisodeLayout,
    oracle_plan: PlanCandidate,
    rng: np.random.Generator,
    contact_intended: bool = False,
    duration_s: float = 20.0,
) -> PlanCandidate | MeanderController | None:
    """Route (or controller) for one episode; None = generation failed
    (caller falls back to the always-available oracle route)."""
    if family == "oracle":
        oracle_plan.meta["family"] = "oracle"
        return oracle_plan
    if family == "spline":
        return random_spline_route(tmap, layout, rng, duration_s=duration_s)
    if family == "near_obstacle":
        return near_obstacle_route(tmap, layout, rng, contact_intended, duration_s=duration_s)
    if family == "meander":
        return MeanderController(rng=rng)
    raise ValueError(f"unknown driver family: {family}")
