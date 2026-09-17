"""RGB-D FDM costs and MPPI over several geometric route families.

Only measured RGB-D, vehicle history/localization, goal and proposed references
enter the scorer. Authored terrain heights and obstacle positions are not
accepted by this API. Kinematic checks cannot establish collision clearance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import torch

from nedm.traverse.fdm_mppi import MPPIConfig, ReferenceMPPI, validate_reference


def check_reference_contract(route):
    """Reject inconsistent station encodings and cusps hidden by triangle curvature."""
    xy = np.asarray(route["waypoints"], float)
    station = np.asarray(route["stations"], float)
    heading = np.asarray(route["headings"], float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3:
        raise ValueError("Reference needs at least three XY waypoints")
    if station.shape != (len(xy),) or heading.shape != (len(xy),):
        raise ValueError("Reference stations/headings must match waypoints")
    if any(not np.isfinite(a).all() for a in (xy, station, heading)):
        raise ValueError("Nonfinite reference geometry")
    delta = np.diff(xy, axis=0)
    ds = np.linalg.norm(delta, axis=1)
    if (ds <= 1e-8).any() or not np.allclose(station-station[0], np.r_[0., ds.cumsum()], atol=1e-4, rtol=1e-5):
        raise ValueError("Reference station encoding disagrees with waypoint geometry")
    unit = delta/ds[:, None]
    if (np.sum(unit[1:]*unit[:-1], axis=1) < np.cos(np.pi/4.)).any():
        raise ValueError("Reference contains a cusp or sharp segment-heading reversal")
    expected = np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0]))
    error = np.arctan2(np.sin(heading-expected), np.cos(heading-expected))
    if (np.abs(error) > .15).any():
        raise ValueError("Supplied reference headings disagree with waypoint tangents")


@dataclass(frozen=True)
class RGBDCostConfig:
    goal_radius_m: float = 3.0
    contact_penalty_s: float = 60.0
    low_progress_penalty_s: float = 60.0
    terminal_progress_weight: float = 0.1
    minimum_progress_speed_mps: float = 0.5
    maximum_remaining_time_s: float = 120.0
    refine_families: int = 3
    max_contact_probability: float = 0.35
    max_low_progress_probability: float = 0.5

    def __post_init__(self):
        if min(self.goal_radius_m, self.minimum_progress_speed_mps,
               self.maximum_remaining_time_s) <= 0 or self.refine_families < 1:
            raise ValueError("Invalid goal/time/refinement configuration")
        if min(self.contact_penalty_s, self.low_progress_penalty_s,
               self.terminal_progress_weight) < 0:
            raise ValueError("Risk and progress costs must be nonnegative")
        if not (0. <= self.max_contact_probability <= 1. and 0. <= self.max_low_progress_probability <= 1.):
            raise ValueError("Predicted-risk thresholds must be in [0,1]")


class RGBDReferenceScorer:
    def __init__(self, model, rgbd, history, anchor_pose, goal_xy, *,
                 elapsed_s=0., batch_size=128, cost_config=None, control="normal"):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.rgbd = np.asarray(rgbd, np.float32)
        self.history = np.asarray(history, np.float32)
        self.anchor_pose = np.asarray(anchor_pose, np.float64)
        self.goal_xy = np.asarray(goal_xy, np.float64)
        self.elapsed_s = float(elapsed_s)
        self.batch_size = int(batch_size)
        self.cost_config = cost_config or RGBDCostConfig()
        self.control = control
        self.progress_event_definition = getattr(model.config, "progress_event_definition", "net_progress")
        if self.rgbd.ndim != 3 or self.rgbd.shape[0] != 4:
            raise ValueError("Expected current four-channel calibrated RGB-D")
        if self.anchor_pose.shape != (3,) or self.goal_xy.shape != (2,) or self.batch_size < 1:
            raise ValueError("Invalid measured anchor, goal or batch size")
        if any(not np.isfinite(a).all() for a in
               (self.rgbd, self.history, self.anchor_pose, self.goal_xy)):
            raise ValueError("Nonfinite observation")
        # The observation is shared by every candidate. Encode the expensive
        # image and measured history once, retaining candidate-specific globals.
        with torch.no_grad():
            self.rgbd_tensor = torch.from_numpy(self.rgbd).to(self.device)[None]
            self.visual = model.encode_image(self.rgbd_tensor, control)
            self.history_feature = model.encode_history(torch.from_numpy(self.history).to(self.device)[None], control)

    @torch.no_grad()
    def predict(self, routes):
        from nedm.traverse.fdm_rgbd_data import build_command_features
        for route in routes:
            check_reference_contract(route)
        rows = [build_command_features(r, self.anchor_pose,
                    station=r.get("meta", {}).get("fdm_station"), elapsed_s=self.elapsed_s)
                for r in routes]
        if not rows:
            return {}
        result = {k: [] for k in ("trajectory", "work", "event_probability")}
        for offset in range(0, len(rows), self.batch_size):
            subset = rows[offset:offset+self.batch_size]
            batch = {k: torch.from_numpy(np.stack([r[k] for r in subset])).to(self.device)
                     for k in ("commands", "global_features", "nominal_pose")}
            context = self.model.context_from_features(self.visual.expand(len(subset), -1),
                        self.history_feature.expand(len(subset), -1), batch["global_features"])
            if getattr(self.model.config, "candidate_patches", False):
                patches = self.model.encode_candidate_patches(self.rgbd_tensor.expand(len(subset), -1, -1, -1),
                            batch["nominal_pose"], batch["global_features"], self.control)
                output = self.model.predict_from_context(context, batch["commands"], batch["nominal_pose"],
                                                         patch_features=patches)
            else:
                output = self.model.predict_from_context(context, batch["commands"], batch["nominal_pose"])
            for key in ("trajectory", "work"):
                result[key].append(output[key].float().cpu().numpy())
            result["event_probability"].append(torch.sigmoid(output["event_logits"]).float().cpu().numpy())
        result = {k: np.concatenate(v) for k, v in result.items()}
        if any(not np.isfinite(v).all() for v in result.values()):
            raise ValueError("Nonfinite learned forecast")
        return result

    def cost_breakdown(self, output):
        cfg = self.cost_config
        local_xy = output["trajectory"][..., :2]
        c, s = np.cos(self.anchor_pose[2]), np.sin(self.anchor_pose[2])
        world_xy = self.anchor_pose[:2] + np.stack(
            (c*local_xy[..., 0]-s*local_xy[..., 1],
             s*local_xy[..., 0]+c*local_xy[..., 1]), axis=-1)
        distance = np.linalg.norm(world_xy-self.goal_xy, axis=-1)
        initial_distance = float(np.linalg.norm(self.anchor_pose[:2]-self.goal_xy))
        dt = float(self.model.config.dt)
        horizon_s = distance.shape[1]*dt
        reached = distance <= cfg.goal_radius_m
        # Estimate first arrival within the prediction horizon; otherwise add
        # a bounded extrapolation from predicted net goal progress. This is a
        # planning heuristic, not a supervised full-route travel-time head.
        first_index = np.where(reached.any(1), np.argmax(reached, axis=1), distance.shape[1]-1)
        terminal_distance = distance[np.arange(len(distance)), first_index]
        first_arrival_s = (first_index+1)*dt
        progress_speed = np.maximum((initial_distance-terminal_distance)/horizon_s,
                                    cfg.minimum_progress_speed_mps)
        remaining_s = np.clip(np.maximum(terminal_distance-cfg.goal_radius_m, 0.)/progress_speed,
                              0., cfg.maximum_remaining_time_s)
        estimated_time = np.where(reached.any(1), first_arrival_s, horizon_s+remaining_s)
        probability = output["event_probability"]
        support = getattr(self.model, "supported_events", None)
        supported = np.ones(3, bool) if support is None else support.detach().cpu().numpy().astype(bool)
        prefix = np.arange(distance.shape[1])[None] <= first_index[:, None]
        contact = np.where(prefix, probability[..., 0], 0.).max(1) if supported[0] else np.zeros(len(distance))
        eligible = prefix & (((np.arange(distance.shape[1])+1)*dt >= 2.-1e-6)[None])
        eligible &= (~reached.any(1)[:, None]) | (np.arange(distance.shape[1])[None] < first_index[:, None])
        # Goal arrival is a predicted stopping boundary. Subsequent deliberate
        # parking has no low-progress supervision and must not penalize arrival.
        low_progress = np.where(eligible, probability[..., 2], 0.).max(1) if supported[2] else np.zeros(len(distance))
        total = (estimated_time + cfg.terminal_progress_weight*(terminal_distance-initial_distance)
                 + cfg.contact_penalty_s*contact + cfg.low_progress_penalty_s*low_progress)
        risk_allowed = (contact <= cfg.max_contact_probability) & (low_progress <= cfg.max_low_progress_probability)
        # Thresholds are declared planning choices, not calibrated guarantees.
        # When every candidate exceeds them, abstain instead of selecting a
        # known high-risk route merely because it is the least bad proposal.
        return {"cost": np.where(risk_allowed, total, np.inf), "unfiltered_cost": total,
                "allowed_by_predicted_risk": risk_allowed,
                "estimated_time_to_goal_s": estimated_time,
                "predicted_goal_progress_m": initial_distance-terminal_distance,
                "predicted_terminal_goal_distance_m": terminal_distance,
                "contact_probability": contact, "low_progress_probability": low_progress}

    def __call__(self, routes):
        allowed, indices = [], []
        for i, route in enumerate(routes):
            try:
                check_reference_contract(route)
            except (ValueError, KeyError):
                continue
            allowed.append(route); indices.append(i)
        costs = np.full(len(routes), np.inf)
        if allowed:
            costs[indices] = self.cost_breakdown(self.predict(allowed))["cost"]
        return costs


def propose_route_families(anchor_pose, goal_xy, *, speeds=(2., 4., 6.),
                           offsets=(0., -4., 4., -8., 8.), step_m=0.5):
    """Geometric Hermite references; no terrain or obstacle query is performed."""
    pose, goal = np.asarray(anchor_pose, float), np.asarray(goal_xy, float)
    delta = goal-pose[:2]
    length = float(np.linalg.norm(delta))
    if length < 1e-6 or step_m <= 0:
        raise ValueError("Distinct start and goal and positive spacing required")
    direction = delta/length
    normal = np.array([-direction[1], direction[0]])
    t = np.linspace(0., 1., max(33, int(np.ceil(length/step_m))+1))
    start_tangent = length*np.array([np.cos(pose[2]), np.sin(pose[2])])
    end_tangent = delta
    base = ((2*t**3-3*t**2+1)[:, None]*pose[:2]
            + (t**3-2*t**2+t)[:, None]*start_tangent
            + (-2*t**3+3*t**2)[:, None]*goal
            + (t**3-t**2)[:, None]*end_tangent)
    routes = []
    for offset in offsets:
        xy = base + float(offset)*np.sin(np.pi*t)[:, None]**2*normal
        station = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
        heading = np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0]))
        for speed in speeds:
            if speed <= 0:
                raise ValueError("Candidate cruising speeds must be positive")
            velocity = np.minimum(float(speed), np.sqrt(4.*np.maximum(station[-1]-station, 0.)))
            routes.append({"waypoints": xy.copy(), "speeds": velocity, "stations": station.copy(),
                           "headings": heading.copy(), "meta": {"candidate": "rgbd_geometric_family",
                           "lateral_offset_m": float(offset), "cruise_speed_mps": float(speed),
                           "fdm_station": 0.}})
    return routes


def plan_rgbd_routes(model, rgbd, history, anchor_pose, goal_xy, *, elapsed_s=0., seed=11,
                     families=None, mppi_config=None, cost_config=None, control="normal"):
    cfg = mppi_config or MPPIConfig(samples=64, iterations=2)
    costs = cost_config or RGBDCostConfig()
    scorer = RGBDReferenceScorer(model, rgbd, history, anchor_pose, goal_xy,
                                elapsed_s=elapsed_s, cost_config=costs, control=control)
    if np.linalg.norm(np.asarray(anchor_pose)[:2]-np.asarray(goal_xy)) <= costs.goal_radius_m:
        return {"route": None, "goal_reached": True, "abstained": False, "model_evaluations": 0}
    families = propose_route_families(anchor_pose, goal_xy) if families is None else families
    candidates, checks, original_indices = [], [], []
    for index, route in enumerate(families):
        try:
            check_reference_contract(route)
            check = validate_reference(route, [], cfg, anchor_pose)
        except (ValueError, KeyError) as error:
            check = {"valid": False, "reasons": ["reference_contract"], "detail": str(error)}
        checks.append(check)
        if check["valid"]:
            candidates.append(route); original_indices.append(index)
    if not candidates:
        return {"route": None, "abstained": True, "reason": "no kinematically valid reference",
                "checks": checks, "model_evaluations": 0}
    predictions = scorer.predict(candidates)
    components = scorer.cost_breakdown(predictions)
    def value_for_report(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        return float(v) if np.isfinite(v) else None
    table = [{"family_index": original_indices[i], **{k: value_for_report(v[i]) for k, v in components.items()},
              "reference_meta": candidates[i].get("meta", {})} for i in range(len(candidates))]
    order = np.array([i for i in np.argsort(components["cost"]) if np.isfinite(components["cost"][i])], dtype=int)
    if not len(order):
        return {"route": None, "abstained": True, "reason": "all families exceed declared predicted-risk limits",
                "family_scores": table, "checks": checks, "model_evaluations": len(candidates),
                "cost_config": asdict(costs), "image_intervention": control,
                "progress_event_definition": scorer.progress_event_definition}
    best_i = int(order[0])
    best_route, best_cost = candidates[best_i], float(components["cost"][best_i])
    best_family = original_indices[best_i]
    refinements, evaluations = [], len(candidates)
    # Refine distinct geometric paths, not just several speeds of one path.
    chosen, seen_geometry = [], set()
    for i in order:
        route = candidates[int(i)]
        geometry = np.asarray(route["waypoints"], np.float32).tobytes()
        if geometry in seen_geometry:
            continue
        chosen.append(int(i)); seen_geometry.add(geometry)
        if len(chosen) >= costs.refine_families:
            break
    for rank, i in enumerate(chosen):
        result = ReferenceMPPI(cfg, seed+rank).optimize(candidates[i], anchor_pose, [], scorer)
        result["family_index"] = original_indices[i]
        refinements.append(result)
        evaluations += result["model_evaluations"]
        if not result["abstained"] and result["cost"] < best_cost:
            best_route, best_cost, best_family = result["route"], result["cost"], original_indices[i]
    return {"route": best_route, "cost": best_cost, "selected_family_index": best_family,
            "abstained": False, "goal_reached": False, "family_scores": table,
            "checks": checks, "refinements": refinements, "model_evaluations": evaluations,
            "cost_config": asdict(costs), "mppi_config": asdict(cfg),
            "observation": "current measured RGB-D and vehicle history; no authored hazard inputs",
            "image_intervention": control,
            "progress_event_definition": scorer.progress_event_definition,
            "claim": "model-ranked reference; physical outcome must be measured separately"}
