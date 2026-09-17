#!/usr/bin/env python3
"""CPU semantic checks for the explicit bounded-motion v2 target."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from nedm.traverse.fdm_bounded_targets import bounded_motion_endpoints as endpoints, bounded_motion_targets as targets


def main():
    n = 100
    actions = np.zeros((n, 3))
    actions[:, 1] = 1.
    poses, parked = np.zeros((n+1, 3)), np.zeros(n+1, bool)
    assert endpoints(poses, actions, parked)[40:].all()
    labels, mask = targets(endpoints(poses, actions, parked), 0)
    assert not labels[:9].any() and labels[9:].all()
    assert not mask[:9].any() and mask[9:].all()
    for speed in (2., .2):
        poses[:, 0] = np.arange(n+1)*.05*speed
        assert not endpoints(poses, actions, parked).any()
    # Intentional crawling is the explicitly documented command-domain limit.
    poses[:, 0] = np.arange(n+1)*.05*.1
    assert endpoints(poses, actions, parked)[40:].all()
    poses[:, 0] = np.sin(np.arange(n+1)*np.pi/40)
    assert not endpoints(poses, actions, parked).any()  # large out-and-back motion
    poses[:, 0] = .04*np.sin(np.arange(n+1)*np.pi/3)
    assert endpoints(poses, actions, parked)[40:].all()  # tiny vibrating blockage
    angle = .713
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = poses.copy()
    rotated[:, :2] = poses[:, :2]@rotation.T
    np.testing.assert_array_equal(endpoints(poses, actions, parked), endpoints(rotated, actions, parked))
    parked[40] = True
    assert not endpoints(poses, actions, parked)[40:81].any()
    parked[:] = False
    assert not endpoints(poses, actions, parked, goal_xy=[0., 0.]).any()
    actions[10, 1] = .3
    assert not endpoints(poses, actions, parked)[40:51].any()
    labels, mask = targets(endpoints(poses, actions, parked), 90)
    assert not labels.any() and not mask.any()  # insufficient future, no negative fabrication
    labels, mask = targets(endpoints(poses[:61], actions[:60], parked[:61]), 0)
    assert labels[-1, 0] == mask[-1, 0] == 1  # known positive survives truncated tail
    past_only = np.zeros(101, bool)
    past_only[50] = True
    labels, _ = targets(past_only, 20)
    assert not labels.any()  # run began before anchor and cannot become a future event
    print("PASS: launch/slow-motion controls, rollback rejection, vibration, rotation, timing, effort, parking/arrival, censoring and causal run starts")


if __name__ == "__main__":
    main()
