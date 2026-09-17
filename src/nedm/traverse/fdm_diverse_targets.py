"""Measured, censored future labels for the enriched finite-horizon FDM.

All target functions use N actual intervals and N+1 actual state endpoints.
Unavailable future states are masked, never extrapolated. Known cumulative
event positives survive an unobserved tail; a negative needs complete coverage.
"""
from __future__ import annotations

import numpy as np

from nedm.traverse.fdm_bounded_targets import bounded_motion_endpoints
from nedm.traverse.fdm_data import ego_xy
from nedm.traverse.fdm_diverse_data import HORIZON, OUTPUT_DT, RECORD_DT


def prepare_episode_labels(poses, states, actions, parked, rich_intervals, *, goal_xy, goal_radius_m):
    poses, states, actions = map(np.asarray, (poses, states, actions))
    parked = np.asarray(parked, bool)
    n = len(actions)
    if poses.shape != (n+1, 3) or states.shape != (n+1, 17) or actions.shape != (n, 3) or parked.shape != (n+1,):
        raise ValueError("Expected N intervals and N+1 measured endpoints")
    if not all(np.isfinite(v).all() for v in (poses, states, actions)):
        raise ValueError("Nonfinite core physical telemetry")
    arrived = np.maximum.accumulate(np.linalg.norm(poses[:, :2]-goal_xy, axis=1) <= goal_radius_m)
    excluded = parked | arrived
    bounded = bounded_motion_endpoints(poses, actions, excluded, goal_xy=goal_xy, goal_radius_m=goal_radius_m)
    slow = (np.abs(states[:-1, 0]) < .3) & (actions[:, 1] > .3) & ~excluded[:-1] & ~excluded[1:]
    strict = np.zeros(n+1, bool)
    run, required = 0, int(round(2./RECORD_DT))
    for i, flag in enumerate(slow):
        run = run+1 if flag else 0
        strict[i+1] = run >= required
    # Retain both sources to distinguish authored-asset collision from a
    # chassis/terrain interaction. Resultants are not contact-pair identities.
    contact = np.column_stack([rich_intervals[k] for k in ("max_asset_contact_max_resultant_n", "max_chassis_contact_resultant_n")])
    attitude_peaks = np.column_stack([rich_intervals[f"max_abs_{axis}_rad"] for axis in ("roll", "pitch")])
    work = np.asarray(rich_intervals["engine_interface_positive_work_kj"])
    if contact.shape != (n, 2) or attitude_peaks.shape != (n, 2) or work.shape != (n,):
        raise ValueError("Rich intervals do not align with physical trajectory")
    if np.any(work[np.isfinite(work)] < -1e-9):
        raise ValueError("Positive mechanical work cannot be negative")
    return {"poses": poses, "states": states, "actions": actions, "parked": excluded, "n": n,
            "bounded_endpoints": bounded, "sustained_endpoints": strict, "contact": contact,
            "attitude_peaks": attitude_peaks, "work": work}


def _cumulative_event(values, complete):
    """Array of nonnegative measurements and NaNs; any positive proves event."""
    values = np.asarray(values)
    positive = bool(np.any(values[np.isfinite(values)] > 0.))
    return float(positive), float(positive or (complete and np.isfinite(values).all()))


def build_diverse_targets(ep, anchor, *, horizon=HORIZON, output_dt=OUTPUT_DT, attitude=None):
    n = ep["n"]
    if not 0 <= anchor < n:
        raise ValueError("Anchor must be an actual pre-interval state")
    stride = int(round(output_dt/RECORD_DT))
    if stride < 1 or not np.isclose(stride*RECORD_DT, output_dt) or horizon < 1:
        raise ValueError("Positive horizon and an integer output/recording ratio required")
    offsets = np.arange(1, horizon+1)*stride
    ends = anchor+offsets
    valid = ends <= n
    result = {"trajectory": np.zeros((horizon, 4), np.float32), "trajectory_mask": valid[:, None].astype(np.float32),
              "work": np.zeros((horizon, 1), np.float32), "work_mask": np.zeros((horizon, 1), np.float32),
              "events": np.zeros((horizon, 3), np.float32), "event_mask": np.zeros((horizon, 3), np.float32)}
    for key, width in (("bounded_motion", 1), ("sustained_stall", 1), ("asset_contact", 1), ("chassis_contact", 1),
                       ("attitude", 2), ("attitude_peak_abs", 2)):
        result[key] = np.zeros((horizon, width), np.float32)
        result[key+"_mask"] = np.zeros((horizon, width), np.float32)
    if attitude is None:
        attitude = ep["states"][:, 2:4]
    attitude = np.asarray(attitude)
    if attitude.shape != (n+1, 2):
        raise ValueError("Attitude requires N+1 signed roll/pitch endpoints")
    pose0 = ep["poses"][anchor]
    required = int(round(2./RECORD_DT))
    for j, end in enumerate(ends):
        seen = min(int(end), n)
        complete = bool(end <= n)
        if complete:
            pose = ep["poses"][end]
            yaw = pose[2]-pose0[2]
            result["trajectory"][j] = [*ego_xy(pose[:2], pose0), np.sin(yaw), np.cos(yaw)]
            measured = attitude[end]
            result["attitude"][j] = np.where(np.isfinite(measured), measured, 0.)
            result["attitude_mask"][j] = np.isfinite(measured)
            work = ep["work"][anchor:end]
            if np.isfinite(work).all():
                result["work"][j, 0] = work.sum(dtype=np.float64)
                result["work_mask"][j, 0] = 1.
            for axis in range(2):
                peaks = ep["attitude_peaks"][anchor:end, axis]
                if np.isfinite(peaks).all():
                    result["attitude_peak_abs"][j, axis] = peaks.max()
                    result["attitude_peak_abs_mask"][j, axis] = 1.
        contact_values = ep["contact"][anchor:seen]
        event, mask = _cumulative_event(np.where(np.isfinite(contact_values), (contact_values > 1.).astype(float), np.nan), complete)
        result["events"][j, 0], result["event_mask"][j, 0] = event, mask
        for axis, key in enumerate(("asset_contact", "chassis_contact")):
            values = contact_values[:, axis]
            event, mask = _cumulative_event(np.where(np.isfinite(values), (values > 1.).astype(float), np.nan), complete)
            result[key][j, 0], result[key+"_mask"][j, 0] = event, mask
        peaks = ep["attitude_peaks"][anchor:seen]
        event, mask = _cumulative_event(np.where(np.isfinite(peaks), (peaks > np.deg2rad(60.)).astype(float), np.nan), complete)
        result["events"][j, 1], result["event_mask"][j, 1] = event, mask
        if offsets[j] >= required:
            if complete and not ep["parked"][anchor:end+1].any():
                result["event_mask"][j, 2] = 1.
                result["events"][j, 2] = (np.linalg.norm(ep["poses"][end, :2]-pose0[:2]) < .15*offsets[j]*RECORD_DT
                    and ep["actions"][anchor:end, 1].mean() > .3)
            for key, endpoint_key in (("bounded_motion", "bounded_endpoints"), ("sustained_stall", "sustained_endpoints")):
                positive = bool(ep[endpoint_key][anchor+required:seen+1].any())
                result[key][j, 0] = positive
                result[key+"_mask"][j, 0] = positive or complete
    return result
