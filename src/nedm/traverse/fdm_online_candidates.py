"""Causal reference continuity for receding-horizon FDM planning.

Retain the executable reference and nearby original goal-route families. This
keeps a previously chosen long path available when fresh local Hermite curves
no longer reproduce its geometry. No terrain or measured outcome is queried.
"""
from __future__ import annotations

import hashlib
import numpy as np

from nedm.traverse.fdm_diverse_planner import check_reference_contract, propose_route_families


POLICY = {
    "version": "retained_and_nearby_global_v2",
    "nearby_distance_m": 2.,
    "nearby_heading_deg": 15.,
    "max_reference_curvature_inv_m": .025,
    "curvature_basis": "360 training references max0.016669/m; limit rounded to1.5times this geometric maximum, not a learned safety guarantee",
    "ordering": "active reference, nearby original families, fresh current-pose families; exact duplicates removed",
    "inputs": "measured pose, goal, previously proposed references; no terrain or outcomes",
    "invalid_fresh_candidates": "Reject and log individual geometry-contract failures; corrupt active/original references still raise",
}


def locate_reference(route, pose, origin):
    check_reference_contract(route)
    result = {key: np.asarray(route[key], np.float64).copy()
              for key in ("waypoints", "stations", "headings", "speeds")}
    speeds = result["speeds"]
    if (speeds.shape != result["stations"].shape or not np.isfinite(speeds).all()
            or (speeds < 0.).any() or (speeds > 6.00001).any()):
        raise ValueError("Reference speeds must be finite, station-aligned and within0–6m/s")
    distance = np.linalg.norm(result["waypoints"]-np.asarray(pose)[:2], axis=1)
    nearest = int(distance.argmin())
    angle = float(result["headings"][nearest]-pose[2])
    heading_error = float(abs(np.degrees(np.arctan2(np.sin(angle), np.cos(angle)))))
    result["meta"] = {**route.get("meta", {}), "fdm_station": float(result["stations"][nearest]),
        "candidate_origin": origin, "current_reference_distance_m": float(distance[nearest]),
        "current_reference_heading_error_deg": heading_error}
    return result


def reference_fingerprint(route):
    digest = hashlib.sha256()
    for key in ("waypoints", "speeds"):
        array = np.ascontiguousarray(route[key], dtype="<f8")
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def online_route_families(pose, goal, active_route, original_families, *,
                          speeds=(2., 4., 6.), offsets=(0., -22., 22., -44., 44.),
                          rejected=None):
    pose, goal = np.asarray(pose, np.float64), np.asarray(goal, np.float64)
    proposals = [locate_reference(active_route, pose, "active_reference")]
    for index, reference in enumerate(original_families):
        candidate = locate_reference(reference, pose, f"original_family_{index:02d}")
        meta = candidate["meta"]
        if (meta["current_reference_distance_m"] <= POLICY["nearby_distance_m"]
                and meta["current_reference_heading_error_deg"] <= POLICY["nearby_heading_deg"]):
            proposals.append(candidate)
    for index, reference in enumerate(propose_route_families(
            pose, goal, speeds=tuple(speeds), offsets=tuple(offsets), step_m=.5)):
        origin = f"fresh_family_{index:02d}"
        try:
            candidate = locate_reference(reference, pose, origin)
        except ValueError as error:
            # A Hermite proposal can fold back on itself when the current
            # heading points away from the goal. It is an invalid alternative,
            # not a simulation failure. Never pass it to the model/controller.
            if rejected is not None:
                rejected.append({"candidate_origin": origin,
                    "reference_sha256": reference_fingerprint(reference),
                    "reason": "reference_contract", "detail": str(error),
                    "reference_meta": dict(reference.get("meta", {}))})
            continue
        proposals.append(candidate)
    families, seen = [], set()
    for candidate in proposals:
        if np.linalg.norm(candidate["waypoints"][-1]-goal) > .25:
            raise ValueError("Every retained/fresh reference must target the supplied goal")
        fingerprint = reference_fingerprint(candidate)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        candidate["meta"]["reference_sha256"] = fingerprint
        families.append(candidate)
    return families
