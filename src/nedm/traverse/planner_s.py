"""Planner-S: sampling + imagination instead of search (WP5).

The imagined rollout is cheap and batched, so instead of asking A* for a handful of
candidates we sample thousands of smooth routes from the start to the approach ring,
drop the ones whose footprint sweeps through camera-decoded obstacle cells, imagine
the survivors with the learned tracker inside the NRD, and keep the best score. Route
geometry comes from sampling; feasibility on the terrain (roll, pitch, speed the
vehicle actually holds) comes from the physics model, not from slope caps.

Route family: start -> a point 4 m ahead along the start heading -> three control
points scattered around the chord -> a point on the goal ring on the start-facing side;
Catmull-Rom spline, resampled at 0.5 m, curvature-checked against the HMMWV's 8 m
minimum turn radius. Cruise speed is a sampled parameter; ramps and terminal taper
reuse the oracle's speed profile so the tracker sees the same reference format.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

import numpy as np

from nedm.traverse.oracle import Obstacle, OracleGrid, PlanCandidate, PlannerParams, _repair_curvature, speed_profile
from nedm.traverse.terrain import TerrainMap

FOOTPRINT_HALF_W = 1.3
FOOTPRINT_DISCS = (-1.9, 0.0, 1.9)


def _catmull_rom(ctrl: np.ndarray, n_dense: int = 160) -> np.ndarray:
    """ctrl (N, K, 2) -> dense points (N, n_dense, 2) through all control points (vectorized)."""
    n, k, _ = ctrl.shape
    pad = np.concatenate([ctrl[:, :1] * 2 - ctrl[:, 1:2], ctrl, ctrl[:, -1:] * 2 - ctrl[:, -2:-1]], axis=1)
    t = np.linspace(0, k - 1, n_dense, endpoint=True)
    seg = np.minimum(np.floor(t).astype(int), k - 2)
    u = (t - seg)[None, :, None]
    p0, p1, p2, p3 = (pad[:, seg + i] for i in range(4))  # (N, n_dense, 2)
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)


def _resample(points: np.ndarray, step: float) -> np.ndarray:
    seg = np.hypot(*np.diff(points, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(round(s[-1] / step)) + 1)
    tgt = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(tgt, s, points[:, 0]), np.interp(tgt, s, points[:, 1])], 1)


def _kappa_dense(dense: np.ndarray) -> np.ndarray:
    """Max three-point curvature per route on the dense curve (N, M, 2) -> (N,), vectorized."""
    a, b, c = dense[:, :-2], dense[:, 1:-1], dense[:, 2:]
    ab, bc, ac = b - a, c - b, c - a
    cross = np.abs(ab[..., 0] * bc[..., 1] - ab[..., 1] * bc[..., 0])
    denom = np.linalg.norm(ab, axis=-1) * np.linalg.norm(bc, axis=-1) * np.linalg.norm(ac, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(denom > 1e-9, 2.0 * cross / denom, 0.0)
    return kappa.max(axis=1)


def _curvature_max(points: np.ndarray) -> float:
    a, b, c = points[:-2], points[1:-1], points[2:]
    ab, bc, ac = b - a, c - b, c - a
    cross = np.abs(ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0])
    denom = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ac, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(denom > 1e-9, 2.0 * cross / denom, 0.0)
    return float(kappa.max()) if len(kappa) else 0.0


def sample_routes(start_xy, start_yaw: float, goal_xy, n: int, rng: np.random.Generator,
                  tmap: TerrainMap, discs: Sequence[Obstacle], params: PlannerParams | None = None,
                  lateral_frac=(0.04, 0.14), ring_half_angle_deg: float = 75.0, v_range=(3.0, 9.0),
                  prefilter_margin_m: float = 0.2, max_routes: int | None = None) -> tuple[list[PlanCandidate], dict]:
    """Sample ``n`` routes, keep the curvature-feasible, in-arena, obstacle-clear ones (footprint
    swept against the camera's obstacle discs with ``prefilter_margin_m`` slack)."""
    params = params or PlannerParams()
    start, goal = np.asarray(start_xy, float), np.asarray(goal_xy, float)
    chord = goal - start
    length, ang = np.hypot(*chord), math.atan2(chord[1], chord[0])
    # ring end points on the side facing the start
    theta = ang + math.pi + np.radians(rng.uniform(-ring_half_angle_deg, ring_half_angle_deg, n))
    end = goal[None] + params.approach_ring_m * np.stack([np.cos(theta), np.sin(theta)], 1)
    ahead = start[None] + 4.0 * np.array([[math.cos(start_yaw), math.sin(start_yaw)]])
    # three control points along the chord from 'ahead' to 'end' with lateral / longitudinal scatter
    k = 3
    frac = (np.arange(1, k + 1) / (k + 1))[None, :, None] + rng.normal(0, 0.06, (n, k, 1))
    base = ahead[:, None] + frac * (end[:, None] - ahead[:, None])
    normal = np.stack([-(end - ahead)[:, 1], (end - ahead)[:, 0]], 1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    lo, hi = (lateral_frac, lateral_frac) if np.isscalar(lateral_frac) else lateral_frac
    sigma = rng.uniform(lo, hi, (n, 1, 1)) * length  # per-route scatter scale: near-chord and wide detours
    lat = rng.normal(0, 1.0, (n, k, 1)) * sigma * np.array([0.6, 1.0, 0.6])[None, :, None]
    ctrl = np.concatenate([np.repeat(start[None, None], n, 0), np.repeat(ahead[:, None], n, 0), base + lat * normal[:, None], end[:, None]], 1)
    dense = _catmull_rom(ctrl)
    keep_arena = (np.abs(dense) <= params.arena_keep_within_m).all(axis=(1, 2))
    # hopeless curvature is rejected on the dense curve (vectorized); borderline routes get the oracle's repair
    keep_arena &= _kappa_dense(dense) <= 2.0 * params.kappa_max
    # footprint prefilter against obstacle discs (vectorized, chunked)
    obs = np.asarray(discs, float) if len(discs) else np.zeros((0, 3))
    v_cruise = rng.uniform(*v_range, n)
    grid = OracleGrid(tmap, [], params)  # terrain only; used for the speed profile's slope term
    out, stats = [], {"sampled": n, "arena": int(keep_arena.sum()), "curvature": 0, "clear": 0}
    for i in np.nonzero(keep_arena)[0]:
        if max_routes is not None and len(out) >= max_routes:
            break
        pts = _resample(dense[i], params.sample_step_m)
        if _curvature_max(pts) > params.kappa_max:
            pts = _repair_curvature(pts, params)
            if _curvature_max(pts) > params.kappa_max:
                continue
        stats["curvature"] += 1
        if len(obs):
            tang = np.gradient(pts, axis=0); tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
            centres = np.concatenate([pts + off * tang for off in FOOTPRINT_DISCS], 0)  # (3M, 2)
            d = np.hypot(centres[:, None, 0] - obs[None, :, 0], centres[:, None, 1] - obs[None, :, 1]) - obs[None, :, 2] - FOOTPRINT_HALF_W
            if d.min() < prefilter_margin_m:
                continue
        stats["clear"] += 1
        p = replace(params, v_cruise_mps=float(v_cruise[i]))
        speeds = speed_profile(pts, grid, p)
        tang = np.gradient(pts, axis=0)
        headings = np.arctan2(tang[:, 1], tang[:, 0])
        seg = np.hypot(*np.diff(pts, axis=0).T)
        out.append(PlanCandidate(waypoints=pts, speeds=speeds, headings=headings,
                                 stations=np.concatenate([[0.0], np.cumsum(seg)]),
                                 meta={"candidate": f"sample_{i}", "v_cruise": float(v_cruise[i]), "length_m": float(seg.sum())}))
    return out, stats
