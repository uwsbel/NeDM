"""Explicit version-2 bounded-motion supervision; no v1 target is replaced.

This is an observed motion/effort criterion for commanded traversal. It cannot
distinguish an intentional crawl below0.125m/s from blockage solely by its
trajectory; the focused cohort commands4/6m/s cruise. Such deployment domains
would require a separately declared requested-speed gate, not relabeling here.
"""
from __future__ import annotations

import numpy as np

from nedm.traverse.fdm_data import DT, OUTPUT_OFFSETS

BOUNDED_MOTION_VERSION = 2
BOUNDED_MOTION_DEFINITION = (
    "Any entirely future contiguous40 action intervals (2s at20Hz) whose41 measured XY pose endpoints "
    "have maximum pairwise separation<=0.25m, with EVERY applied throttle>0.3, no parked pose/interval, "
    "and no pose at/after first measured goal arrival (distance<=declared goal radius). No instantaneous "
    "velocity threshold and no global startup grace. Prefixes under2s masked. Known positives remain "
    "valid after truncation; a negative needs the complete prefix. Intended very slow motion is outside "
    "this4/6m/s traversal-command cohort; throttle alone does not encode requested speed."
)


def bounded_motion_endpoints(poses, actions, parked, *, goal_xy=None, goal_radius_m=2.5):
    """Return a bool at each completed2s endpoint; all41 poses must be known."""
    poses, actions, parked = np.asarray(poses), np.asarray(actions), np.asarray(parked, bool)
    n = len(actions)
    if poses.shape != (n+1, 3) or actions.shape != (n, 3) or parked.shape != (n+1,):
        raise ValueError("Need N+1 measured poses/parking flags and N applied-control intervals")
    if not np.isfinite(poses).all() or not np.isfinite(actions).all():
        raise ValueError("Nonfinite physical telemetry")
    if goal_radius_m <= 0:
        raise ValueError("Goal radius must be positive")
    excluded = parked.copy()
    if goal_xy is not None:
        arrived = np.linalg.norm(poses[:, :2]-np.asarray(goal_xy), axis=1) <= goal_radius_m
        # Once arrival is observed, subsequent poses cannot manufacture blockage.
        excluded |= np.maximum.accumulate(arrived)
    required = int(round(2./DT))
    endpoints = np.zeros(n+1, bool)
    for end in range(required, n+1):
        begin = end-required
        if excluded[begin:end+1].any() or not np.all(actions[begin:end, 1] > .3):
            continue
        xy = poses[begin:end+1, :2].astype(np.float64)
        delta = xy[:, None]-xy[None, :]
        diameter_squared = float(np.max(np.sum(delta*delta, axis=-1)))
        endpoints[end] = diameter_squared <= .25**2+1e-12
    return endpoints


def bounded_motion_targets(endpoints, anchor):
    """Cumulative future bounded-motion event and truncation-aware horizon mask."""
    endpoints = np.asarray(endpoints, bool)
    n = len(endpoints)-1
    if anchor < 0 or anchor >= n:
        raise ValueError("Anchor must name a measured pre-interval state")
    labels = np.zeros((len(OUTPUT_OFFSETS), 1), np.float32)
    mask = np.zeros_like(labels)
    required = int(round(2./DT))
    for j, offset in enumerate(OUTPUT_OFFSETS):
        if offset < required:
            continue
        end = anchor+int(offset)
        # Completed runs starting before the anchor are deliberately excluded.
        labels[j, 0] = np.any(endpoints[anchor+required:min(end, n)+1])
        mask[j, 0] = labels[j, 0] or end <= n
    return labels, mask
