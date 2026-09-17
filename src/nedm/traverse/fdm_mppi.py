"""Constrained local MPPI over a PID reference path and spatial speed profile.

This optimizer refines ONE reference-route family. It is not a global planner
and it makes no guarantee that a numerically bounded proposal is in training
support. Geometry checks apply to samples and to the MPPI weighted mean.
The callable scorer must evaluate all proposals from the same measured context.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np

from nedm.traverse.planner_s import _curvature_max, _resample


@dataclass(frozen=True)
class MPPIConfig:
    samples: int = 128
    iterations: int = 2
    temperature: float = 1.0
    knots: int = 3
    lateral_sigma_m: float = 0.35
    speed_sigma_mps: float = 0.4
    max_lateral_m: float = 1.0
    max_speed_delta_mps: float = 1.0
    min_speed_mps: float = 0.0
    max_speed_mps: float = 10.0
    noise_correlation: float = 0.6
    max_curvature_inv_m: float = 0.125
    max_accel_mps2: float = 1.5
    max_decel_mps2: float = 2.0
    half_width_m: float = 1.3
    half_length_m: float = 2.6
    clearance_margin_m: float = 0.1
    arena_half_extent_m: float = 40.0
    path_step_m: float = 0.25

    def __post_init__(self):
        if self.samples < 2 or self.iterations < 1 or self.knots < 1:
            raise ValueError("MPPI needs >=2 samples, >=1 iteration and >=1 knot")
        if self.temperature <= 0 or not 0 <= self.noise_correlation < 1:
            raise ValueError("temperature must be positive and correlation in [0,1)")
        if self.path_step_m <= 0 or self.min_speed_mps < 0 or self.max_speed_mps <= self.min_speed_mps:
            raise ValueError("invalid reference sampling or speed limits")
        if min(self.lateral_sigma_m, self.speed_sigma_mps, self.max_lateral_m, self.max_speed_delta_mps) < 0:
            raise ValueError("noise and offset bounds must be nonnegative")
        if min(self.max_accel_mps2, self.max_decel_mps2, self.max_curvature_inv_m,
               self.half_width_m, self.half_length_m, self.arena_half_extent_m) <= 0:
            raise ValueError("vehicle and motion limits must be positive")


def _arrays(route):
    if hasattr(route, "to_json"):
        route = route.to_json()
    xy = np.asarray(route["waypoints"], np.float64)
    speed = np.asarray(route["speeds"], np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or speed.shape != (len(xy),):
        raise ValueError("reference requires >=3 XY waypoints and one speed per waypoint")
    if not np.isfinite(xy).all() or not np.isfinite(speed).all():
        raise ValueError("nonfinite reference")
    station = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    if np.any(np.diff(station) <= 1e-8):
        raise ValueError("reference contains duplicate consecutive waypoints")
    return xy, speed, station


def deform_reference(route, parameters, anchor_pose, config: MPPIConfig):
    """Smooth bounded offsets ahead of the current anchor; no future truth used.

    Parameter halves are lateral displacement [m] and desired speed change
    [m/s] at interior knots. Both changes vanish at the current station and
    the route end, preserving the path behind the vehicle and terminal speed.
    """
    xy, speed, station = _arrays(route)
    pars = np.asarray(parameters, np.float64)
    if pars.shape != (2 * config.knots,):
        raise ValueError("parameter count disagrees with MPPI knots")
    anchor = np.asarray(anchor_pose, np.float64)
    start_s = route.get("meta", {}).get("fdm_station") if isinstance(route, dict) else None
    if start_s is None:
        start_s = station[int(np.argmin(np.linalg.norm(xy - anchor[:2], axis=1)))]
    start_s = float(np.clip(start_s, station[0], station[-1]))
    span = max(station[-1] - start_s, 1e-6)
    fraction = np.clip((station - start_s) / span, 0.0, 1.0)
    # Squared-sine envelope preserves tangent continuity at both ends.
    envelope = np.sin(np.pi * fraction) ** 2
    knot_x = np.linspace(0.0, 1.0, config.knots + 2)
    lateral = np.interp(fraction, knot_x, np.r_[0., pars[:config.knots], 0.]) * envelope
    dv = np.interp(fraction, knot_x, np.r_[0., pars[config.knots:], 0.]) * envelope
    tangent = np.gradient(xy, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    points = xy + lateral[:, None] * normal
    modified_station = np.r_[0., np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
    modified_speed = np.clip(speed + dv, config.min_speed_mps, config.max_speed_mps)
    # A requested stop remains a stop; perturbations cannot turn parking into motion.
    modified_speed[speed <= 1e-6] = 0.
    if np.any(pars):
        # Project future speed requests onto the same acceleration/deceleration
        # limits used for reference construction. Keep the executed prefix.
        first_future = min(int(np.searchsorted(station, start_s, side="right")), len(speed))
        for j in range(max(1, first_future), len(speed)):
            bound = np.sqrt(modified_speed[j-1]**2 + 2.*config.max_accel_mps2*(modified_station[j]-modified_station[j-1]))
            modified_speed[j] = min(modified_speed[j], bound)
        for j in range(len(speed)-2, max(0, first_future)-1, -1):
            bound = np.sqrt(modified_speed[j+1]**2 + 2.*config.max_decel_mps2*(modified_station[j+1]-modified_station[j]))
            modified_speed[j] = min(modified_speed[j], bound)
    # Preserve the reference's waypoint sampling. Resampling before execution
    # changes which points the standard PID driver selects for its Bezier path,
    # even at zero perturbation. Dense sampling is reserved for validation.
    headings = np.arctan2(np.gradient(points[:, 1]), np.gradient(points[:, 0]))
    if not np.any(pars) and isinstance(route, dict) and "headings" in route:
        headings = np.asarray(route["headings"], np.float64).copy()
    return {"waypoints": points, "speeds": modified_speed, "stations": modified_station,
            "headings": headings, "meta": {"candidate": "fdm_mppi", "parameters": pars.tolist(),
                "fdm_station": float(np.interp(start_s, station, modified_station))}}


def validate_reference(route, obstacles: Sequence[tuple[float, float, float]], config: MPPIConfig, anchor_pose=None):
    """Conservative swept rectangular footprint against circular asset bounds."""
    xy, speed, station = _arrays(route)
    current_station = route.get("meta", {}).get("fdm_station") if isinstance(route, dict) else None
    if current_station is not None:
        index = min(max(0, int(np.searchsorted(station, current_station))-1), len(xy)-2)
        xy, speed, station = xy[index:], speed[index:], station[index:]
    elif anchor_pose is not None:
        index = min(int(np.argmin(np.linalg.norm(xy - np.asarray(anchor_pose)[:2], axis=1))), len(xy) - 2)
        xy, speed, station = xy[index:], speed[index:], station[index:]
    max_curvature = _curvature_max(xy)
    acceleration = np.diff(speed ** 2) / (2. * np.maximum(np.diff(station), 1e-8))
    failures = []
    if max_curvature > config.max_curvature_inv_m + 1e-6:
        failures.append("curvature")
    if speed.min() < config.min_speed_mps - 1e-6 or speed.max() > config.max_speed_mps + 1e-6:
        failures.append("speed")
    if acceleration.max(initial=0.) > config.max_accel_mps2 + 1e-6:
        failures.append("acceleration")
    if acceleration.min(initial=0.) < -config.max_decel_mps2 - 1e-6:
        failures.append("deceleration")
    original_heading = np.unwrap(np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0])))
    dense = _resample(xy, config.path_step_m)
    dense_station = np.r_[0., np.linalg.norm(np.diff(dense, axis=0), axis=1).cumsum()]
    heading = np.interp(dense_station, station-station[0], original_heading)
    xy = dense
    tangent = np.stack((np.cos(heading), np.sin(heading)), axis=1)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    margin = config.clearance_margin_m
    hl, hw = config.half_length_m + margin, config.half_width_m + margin
    corners = np.concatenate([xy + a * hl * tangent + b * hw * normal for a in (-1., 1.) for b in (-1., 1.)])
    if (np.abs(corners) > config.arena_half_extent_m).any():
        failures.append("arena")
    min_clearance = float("inf")
    if len(obstacles):
        obs = np.asarray(obstacles, np.float64)
        if obs.ndim != 2 or obs.shape[1] != 3 or not np.isfinite(obs).all() or (obs[:,2] < 0).any():
            raise ValueError("obstacles must be finite x,y,nonnegative radius")
        delta = obs[None, :, :2] - xy[:, None, :]
        longitudinal = np.abs((delta * tangent[:, None]).sum(-1)) - hl
        lateral = np.abs((delta * normal[:, None]).sum(-1)) - hw
        rectangle_distance = np.hypot(np.maximum(longitudinal, 0.), np.maximum(lateral, 0.))
        min_clearance = float((rectangle_distance - obs[None, :, 2]).min())
        # Include half a sample interval to cover gaps in the sampled sweep.
        if min_clearance <= np.diff(dense_station).max(initial=config.path_step_m) / 2.:
            failures.append("obstacle")
    return {"valid": not failures, "reasons": failures, "max_curvature": float(max_curvature),
            "min_clearance_m": min_clearance if np.isfinite(min_clearance) else None}


class ReferenceMPPI:
    def __init__(self, config: MPPIConfig | None = None, seed: int = 11):
        self.config = config or MPPIConfig()
        self.rng = np.random.default_rng(seed)

    def optimize(self, route, anchor_pose, obstacles, score: Callable[[list[dict]], np.ndarray],
                 initial_mean=None):
        """Return a checked candidate or explicit abstention, never an unchecked mean.

        ``score`` returns finite costs for allowed proposals, +inf to reject.
        The sampled best is retained if the final weighted mean is invalid or
        has worse measured model cost. Carry ``mean`` forward only while the
        same underlying reference/parameterization remains in use.
        """
        cfg = self.config
        dim = 2 * cfg.knots
        mean = np.zeros(dim) if initial_mean is None else np.array(initial_mean, dtype=float, copy=True)
        limits = np.r_[np.full(cfg.knots, cfg.max_lateral_m), np.full(cfg.knots, cfg.max_speed_delta_mps)]
        sigma = np.r_[np.full(cfg.knots, cfg.lateral_sigma_m), np.full(cfg.knots, cfg.speed_sigma_mps)]
        if mean.shape != (dim,) or not np.isfinite(mean).all():
            raise ValueError("invalid initial parameter mean")
        mean = np.clip(mean, -limits, limits)
        best_route, best_cost, best_parameters = None, float("inf"), None
        history, evaluations = [], 0

        def evaluate(parameters):
            nonlocal evaluations
            candidates, indices = [], []
            for i, p in enumerate(parameters):
                candidate = deform_reference(route, p, anchor_pose, cfg)
                check = validate_reference(candidate, obstacles, cfg, anchor_pose)
                if check["valid"]:
                    candidate["meta"]["geometry"] = check
                    candidates.append(candidate); indices.append(i)
            costs = np.full(len(parameters), np.inf)
            if candidates:
                values = np.asarray(score(candidates), dtype=float)
                if values.shape != (len(candidates),) or np.isnan(values).any() or np.isneginf(values).any():
                    raise ValueError("scorer must return one finite or +inf cost per proposal")
                costs[indices] = values
                evaluations += len(candidates)
            return costs, dict(zip(indices, candidates))

        for iteration in range(cfg.iterations):
            noise = self.rng.normal(size=(cfg.samples, 2, cfg.knots))
            for k in range(1, cfg.knots):
                noise[:, :, k] = cfg.noise_correlation * noise[:, :, k-1] + np.sqrt(1-cfg.noise_correlation**2) * noise[:, :, k]
            parameters = np.clip(mean + noise.reshape(cfg.samples, dim) * sigma, -limits, limits)
            parameters[0] = mean
            parameters[1] = 0.  # Always reconsider the original reference.
            # A reference already at its curvature limit can reject nearly all
            # lateral perturbations. Reserve a quarter of proposals for speed
            # changes on the original geometry, a supported local alternative.
            parameters[2:2+cfg.samples//4, :cfg.knots] = 0.
            costs, candidates = evaluate(parameters)
            valid = np.isfinite(costs)
            history.append({"iteration": iteration, "valid_geometry": len(candidates), "finite_scores": int(valid.sum())})
            if not valid.any():
                break
            j = int(np.argmin(costs))
            if costs[j] < best_cost:
                best_cost, best_route, best_parameters = float(costs[j]), candidates[j], parameters[j].copy()
            weights = np.exp(-(costs[valid] - costs[valid].min()) / cfg.temperature)
            weights /= weights.sum()
            mean = (weights[:, None] * parameters[valid]).sum(0)
            history[-1]["best_cost"] = float(costs[j])
            history[-1]["effective_sample_size"] = float(1. / (weights ** 2).sum())
        costs, candidates = evaluate(mean[None])
        selected_mean = bool(np.isfinite(costs[0]) and costs[0] <= best_cost)
        if selected_mean:
            best_cost, best_route, best_parameters = float(costs[0]), candidates[0], mean.copy()
        return {"route": best_route, "cost": best_cost if np.isfinite(best_cost) else None,
                "parameters": None if best_parameters is None else best_parameters.tolist(),
                "mean": mean.tolist(), "selected_weighted_mean": selected_mean,
                "abstained": best_route is None, "model_evaluations": evaluations,
                "iterations": history, "config": asdict(cfg)}
