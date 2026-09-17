"""Record candidate commands and actual finite-horizon FDM forecasts separately.

This wrapper preserves the existing scorer's costs. It never simulates a route,
extends the prediction horizon, or converts a later measured outcome into a
model forecast. The full commanded path is intended as a dashed visual guide;
only the measured anchor and twenty predicted poses form the forecast trace.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from nedm.traverse.fdm_rgbd_data import build_command_features
from nedm.traverse.fdm_rgbd_planner import check_reference_contract


def serializable(value):
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def world_xy(local_xy, anchor_pose):
    anchor = np.asarray(anchor_pose, np.float64)
    local = np.asarray(local_xy, np.float64)
    c, s = np.cos(anchor[2]), np.sin(anchor[2])
    return anchor[:2] + np.stack((c * local[..., 0] - s * local[..., 1],
                                  s * local[..., 0] + c * local[..., 1]), axis=-1)


def route_digest(route):
    digest = hashlib.sha256()
    for key in ("waypoints", "speeds", "stations", "headings"):
        array = np.ascontiguousarray(route[key], dtype="<f8")
        digest.update(key.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    # Explicit reference station changes the future command even when the
    # complete route arrays are identical.
    digest.update(json.dumps(serializable(route.get("meta", {}).get("fdm_station"))).encode())
    return digest.hexdigest()


def forecast_record(scorer, route, prediction, costs, row=0):
    """One candidate: no measured future, no probabilities before supervision."""
    config = scorer.model.config
    dt = float(config.dt)
    pose = np.asarray(prediction["trajectory"][row])
    horizon = len(pose)
    times = (np.arange(horizon) + 1) * dt
    nominal = build_command_features(route, scorer.anchor_pose,
        station=route.get("meta", {}).get("fdm_station"), elapsed_s=scorer.elapsed_s)
    if nominal["nominal_pose"].shape != pose.shape:
        raise ValueError("Nominal and learned forecast horizons differ")
    predicted_xy = world_xy(pose[..., :2], scorer.anchor_pose)
    nominal_xy = world_xy(nominal["nominal_pose"][..., :2], scorer.anchor_pose)
    trace = np.concatenate((np.asarray(scorer.anchor_pose[:2])[None], predicted_xy))
    initial_distance = float(np.linalg.norm(np.asarray(scorer.anchor_pose[:2]) - scorer.goal_xy))
    distance = np.linalg.norm(predicted_xy - scorer.goal_xy, axis=1)
    progress = initial_distance - distance
    nominal_progress = initial_distance - np.linalg.norm(nominal_xy - scorer.goal_xy, axis=1)
    support = getattr(scorer.model, "supported_events", None)
    supported = np.ones(3, bool) if support is None else support.detach().cpu().numpy().astype(bool)
    probability = np.asarray(prediction["event_probability"][row])
    if probability.shape != (horizon, 3):
        raise ValueError("Three per-prefix event probabilities are required")
    eligible = times >= 2. - 1e-6
    third = np.where(eligible & supported[2], probability[:, 2], np.nan)
    first = probability[:, 0] if supported[0] else np.full(horizon, np.nan)
    rollover = probability[:, 1] if supported[1] else np.full(horizon, np.nan)
    breakdown = {key: value[row] for key, value in costs.items()}
    return serializable({
        "id": route_digest(route), "route": route,
        "forecast_horizon_s": horizon * dt, "forecast_times_s": times,
        "forecast_trace_times_s": np.r_[0., times], "forecast_world_xy": trace,
        "forecast_relative_pose": pose,
        "forecast_world_yaw_rad": np.arctan2(pose[:, 2], pose[:, 3]) + scorer.anchor_pose[2],
        "nominal_world_xy": np.concatenate((np.asarray(scorer.anchor_pose[:2])[None], nominal_xy)),
        "nominal_commands": nominal["commands"],
        "predicted_goal_progress_by_prefix_m": progress,
        "predicted_interval_speed_mps": np.linalg.norm(np.diff(trace, axis=0), axis=1) / dt,
        "predicted_interval_goal_progress_m": np.diff(np.r_[0., progress]),
        "predicted_cumulative_positive_work_kj": prediction["work"][row, :, 0],
        "contact_probability_by_prefix": first,
        "progress_event_probability_by_prefix": third,
        "rollover_probability_by_prefix": rollover,
        "progress_event_definition": getattr(config, "progress_event_definition", "net_progress"),
        "progress_event_version": getattr(config, "progress_event_version", 1),
        "supported_events": supported,
        "predicted_4s_goal_progress_m": float(progress[-1]),
        "nominal_4s_goal_progress_m": float(nominal_progress[-1]),
        "predicted_to_nominal_goal_progress_fraction": float(progress[-1] / nominal_progress[-1]) if nominal_progress[-1] > 1e-6 else None,
        "scoring": breakdown,
        "visual_semantics": {
            "solid": "Measured current position followed by actual model-predicted poses, ending at the declared horizon",
            "dashed": "Full proposed PID reference path, including the portion beyond the prediction horizon",
            "risk": "Each point is a predicted event over the future prefix ending at that time, not an instantaneous hazard rate",
            "progress_event": "Head2 is structurally unsupervised before2s; those entries are null. Bounded-motion version2 means a wholly future2s low-diameter/effort interval, not generic eventual stall.",
            "time_cost": "Estimated time to goal includes analytic extrapolation after the forecast horizon; it is not a full-route learned travel-time prediction",
            "future_truth": "Chrono outcomes are not read or embedded in this forecast record",
        },
    })


class RGBDScoringRecorder:
    """Callable scorer wrapper for MPPI or a batch of reference families.

    ``recorder(routes)`` returns the unmodified scorer cost for every route.
    ``records`` deduplicates identical command geometry/speed, while ``calls``
    preserves the complete sampled evaluation order, including repeat draws.
    """
    def __init__(self, scorer, *, provenance=None):
        self.scorer = scorer
        self.provenance = dict(provenance or {})
        self.records = {}
        self.calls = []

    def __call__(self, routes):
        costs = np.full(len(routes), np.inf)
        accepted, indices = [], []
        identities = [None] * len(routes)
        rejected = []
        for index, route in enumerate(routes):
            try:
                check_reference_contract(route)
            except (ValueError, KeyError) as error:
                rejected.append({"index": index, "reason": str(error)})
                continue
            accepted.append(route)
            indices.append(index)
        if accepted:
            predictions = self.scorer.predict(accepted)
            breakdown = self.scorer.cost_breakdown(predictions)
            costs[indices] = breakdown["cost"]
            for local, original in enumerate(indices):
                record = forecast_record(self.scorer, routes[original], predictions, breakdown, local)
                identities[original] = record["id"]
                if record["id"] not in self.records:
                    record["first_call"] = len(self.calls)
                    record["occurrences"] = 0
                    self.records[record["id"]] = record
                self.records[record["id"]]["occurrences"] += 1
        self.calls.append(serializable({"candidate_ids": identities, "costs": costs, "contract_rejections": rejected}))
        return costs

    def representative_candidates(self, *, minimum_nominal_progress_m=4., maximum_progress_fraction=.6):
        """Four distinct hypotheses when forecasts support them; otherwise null.

        Low progress requires model-predicted loss relative to a moving nominal
        reference, not merely a deliberately slow command. Thresholds are
        explicit display rules and are not measured-outcome labels.
        """
        records = list(self.records.values())
        cfg = self.scorer.cost_config
        selected, reasons, used = {}, {}, set()

        def choose(name, candidates, key, missing):
            available = [record for record in candidates if record["id"] not in used]
            if available:
                candidate = min(available, key=lambda record: (key(record), record["id"]))
                selected[name] = candidate["id"]
                used.add(candidate["id"])
            else:
                selected[name] = None
                reasons[name] = missing

        choose("best", [r for r in records if r["scoring"]["allowed_by_predicted_risk"]],
               lambda r: r["scoring"]["cost"], "No candidate passes the declared predicted-risk limits")
        choose("collision_risk", [r for r in records if r["supported_events"][0]
            and r["scoring"]["contact_probability"] > cfg.max_contact_probability],
               lambda r: -r["scoring"]["contact_probability"], "No distinct candidate has above-threshold predicted contact risk")
        choose("bounded_motion_risk", [r for r in records if r["supported_events"][2]
            and r["progress_event_definition"] == "bounded_motion"
            and r["scoring"]["contact_probability"] < cfg.max_contact_probability
            and r["scoring"]["low_progress_probability"] > cfg.max_low_progress_probability],
               lambda r: -r["scoring"]["low_progress_probability"], "No distinct contact-low candidate has above-threshold within-horizon bounded-motion risk")
        choose("low_progress", [r for r in records if r["scoring"]["allowed_by_predicted_risk"]
            and r["nominal_4s_goal_progress_m"] >= minimum_nominal_progress_m
            and r["predicted_to_nominal_goal_progress_fraction"] is not None
            and r["predicted_to_nominal_goal_progress_fraction"] <= maximum_progress_fraction],
               lambda r: r["predicted_to_nominal_goal_progress_fraction"], "No distinct low-risk moving reference shows the declared predicted progress loss")
        return {"selected": selected, "unavailable": reasons,
            "display_rules": {"contact_threshold": cfg.max_contact_probability,
                "bounded_motion_threshold": cfg.max_low_progress_probability,
                "minimum_nominal_progress_m": minimum_nominal_progress_m,
                "maximum_progress_fraction": maximum_progress_fraction},
            "selection_scope": "Prediction-only representative hypotheses, not guaranteed future Chrono outcomes"}

    def export(self, path, *, representatives=None, write_npz=True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        selected = representatives if representatives is not None else self.representative_candidates()
        payload = serializable({"schema": 1, "provenance": self.provenance,
            "anchor_pose": self.scorer.anchor_pose, "goal_xy": self.scorer.goal_xy,
            "elapsed_s_at_anchor": self.scorer.elapsed_s, "image_intervention": self.scorer.control,
            "model_config": asdict(self.scorer.model.config), "cost_config": asdict(self.scorer.cost_config),
            "unique_candidates": len(self.records), "scorer_calls": len(self.calls),
            "requested_candidates": sum(len(call["candidate_ids"]) for call in self.calls),
            "candidate_evaluations": sum(sum(identity is not None for identity in call["candidate_ids"]) for call in self.calls),
            "calls": self.calls, "candidates": list(self.records.values()), "representatives": selected})
        if write_npz and self.records:
            records = list(self.records.values())
            arrays = {"candidate_ids": np.asarray([record["id"] for record in records]),
                "forecast_times_s": np.asarray(records[0]["forecast_times_s"]),
                "forecast_trace_times_s": np.asarray(records[0]["forecast_trace_times_s"])}
            for key in ("forecast_world_xy", "nominal_world_xy", "forecast_relative_pose",
                        "predicted_goal_progress_by_prefix_m", "predicted_interval_speed_mps"):
                arrays[key] = np.asarray([record[key] for record in records], np.float32)
            for key in ("contact_probability_by_prefix", "progress_event_probability_by_prefix", "rollover_probability_by_prefix"):
                values = np.asarray([record[key] for record in records], np.float32)
                arrays[key + "_valid"] = np.isfinite(values)
                arrays[key] = np.nan_to_num(values, nan=0.)
            arrays["unfiltered_cost"] = np.asarray([record["scoring"]["unfiltered_cost"] for record in records], np.float32)
            arrays["allowed_by_predicted_risk"] = np.asarray([record["scoring"]["allowed_by_predicted_risk"] for record in records], bool)
            npz_path = path.with_suffix(".npz")
            np.savez_compressed(npz_path, **arrays)
            payload["sample_cloud_npz"] = {"path": str(npz_path), "candidate_order": "Same order as candidates[] in this JSON",
                "risk_padding": "Unsupported/ineligible probabilities are zero with a separate false validity mask"}
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        return payload


__all__ = ["RGBDScoringRecorder", "forecast_record", "world_xy", "route_digest"]
