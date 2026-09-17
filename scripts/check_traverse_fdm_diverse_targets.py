#!/usr/bin/env python3
"""No-Chrono checks of measured endpoints, future intervals and censoring masks."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nedm.traverse.fdm_diverse_targets import prepare_episode_labels, build_diverse_targets


def fixture(n=50, *, parked=False):
    pose = np.zeros((n+1, 3))
    state = np.zeros((n+1, 17))
    actions = np.zeros((n, 3))
    actions[:, 1] = 1.
    parking = np.full(n+1, parked, bool)
    rich = {"max_asset_contact_max_resultant_n": np.zeros(n), "max_chassis_contact_resultant_n": np.zeros(n),
            "max_abs_roll_rad": np.zeros(n), "max_abs_pitch_rad": np.zeros(n),
            "engine_interface_positive_work_kj": np.ones(n)}
    return pose, state, actions, parking, rich


def episode(values):
    return prepare_episode_labels(*values, goal_xy=np.array([100., 0.]), goal_radius_m=2.5)


def main():
    values = fixture(50)
    # A contact exactly in interval4 must not leak into the first output at4.
    values[4]["max_asset_contact_max_resultant_n"][4] = 2.
    # A within-step rollover can recover before the recorded endpoint; the
    # post-step risk maximum must still label the event while endpoint is0.
    values[4]["max_abs_pitch_rad"][6] = np.deg2rad(65.)
    ep = episode(values)
    result = build_diverse_targets(ep, 0, horizon=20)
    assert result["events"][0, 0] == 0 and result["events"][1, 0] == 1
    assert result["events"][1, 1] == 1 and result["attitude"][1, 1] == 0
    np.testing.assert_allclose(result["work"][:12, 0], np.arange(1, 13)*4)
    assert result["trajectory_mask"][11, 0] == 1 and result["trajectory_mask"][12, 0] == 0
    assert result["work_mask"][12, 0] == 0 and result["work"][12, 0] == 0
    # Known positives persist after truncation, including a completed2s stall;
    # unknown late trajectories and attitude remain completely censored.
    assert result["event_mask"][-1, :2].all() and result["events"][-1, :2].all()
    assert result["bounded_motion"][8, 0] == 0 and result["bounded_motion_mask"][8, 0] == 0
    assert result["bounded_motion"][9, 0] == 1 and result["sustained_stall"][9, 0] == 1
    assert result["bounded_motion_mask"][-1, 0] == 1 and result["attitude_mask"][-1].sum() == 0
    # Events before a late anchor do not count; less than2s future motion cannot
    # inherit an old stall or manufacture a negative beyond the actual end.
    late = build_diverse_targets(ep, 20, horizon=20)
    assert late["events"][:, :2].sum() == 0
    assert late["bounded_motion"].sum() == 0 and late["bounded_motion_mask"].sum() == 0
    assert late["event_mask"][-1, :2].sum() == 0
    # The actual terminal endpoint is valid when exactly on an output tick.
    terminal = build_diverse_targets(episode(fixture(40)), 0, horizon=20)
    assert terminal["trajectory_mask"][9, 0] == 1 and terminal["attitude_mask"][9].all()
    assert terminal["trajectory_mask"][10, 0] == 0
    # Censored negatives are masked; deliberate parking does not create stall.
    parked = build_diverse_targets(episode(fixture(50, parked=True)), 0, horizon=20)
    assert parked["bounded_motion"].sum() == 0 and parked["sustained_stall"].sum() == 0
    assert parked["event_mask"][-1, :2].sum() == 0
    # Missing work and one-axis attitude censor only their actual dependencies.
    missing = fixture(50)
    missing[4]["engine_interface_positive_work_kj"][5] = np.nan
    missing[4]["max_chassis_contact_resultant_n"][2] = np.nan
    ep_missing = episode(missing)
    attitude = np.zeros((51, 2)); attitude[8, 0] = np.nan
    result = build_diverse_targets(ep_missing, 0, horizon=20, attitude=attitude)
    assert result["work_mask"][0, 0] == 1 and result["work_mask"][1, 0] == 0
    np.testing.assert_array_equal(result["attitude_mask"][1], [0., 1.])
    assert result["event_mask"][0, 0] == 0 and result["asset_contact_mask"][0, 0] == 1
    # The same measured labels are shared by H20 and H60, with no new forecast
    # used to construct a label or to fill an unmeasured tail.
    short = build_diverse_targets(ep, 0, horizon=20)
    long = build_diverse_targets(ep, 0, horizon=60)
    for key in short:
        np.testing.assert_array_equal(short[key], long[key][:20])
    print("PASS: future-only interval contact, post-step rollover, exact terminal endpoint, cumulative work, late censoring, no inherited stall, parking, missing-target masks, H20/H60 prefix equivalence")


if __name__ == "__main__":
    main()
