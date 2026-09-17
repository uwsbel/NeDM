#!/usr/bin/env python3
"""Verify frozen commands, actual video timing, and passive-observer parity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(base):
    protocol = json.loads((base / "execution_protocol.json").read_text())
    repo = Path(__file__).resolve().parents[1]
    for path, digest in protocol["source_sha256"].items():
        assert sha(repo / path) == digest, path
    assert sha(base / protocol["case"]) == protocol["case_sha256"]
    assert sha(base / protocol["observation"]) == protocol["observation_sha256"]
    for path, digest in protocol["frozen_prediction_artifacts"].items():
        assert sha(base / "prediction" / path) == digest, path
    anchor = np.load(base / protocol["observation"])
    rows = []
    for arm in protocol["runs"]:
        folder = base / "actual" / arm
        trajectory = np.load(folder / "trajectory.npz")
        obs = np.load(folder / "observation.npz")
        outcome = json.loads((folder / "outcome.json").read_text())
        metadata = json.loads((folder / "frame_metadata.json").read_text())
        assert outcome["case_sha256"] == protocol["case_sha256"]
        assert outcome["route_sha256"] == protocol["frozen_prediction_artifacts"][f"routes/{arm}.json"]
        assert all(np.array_equal(obs[k], anchor[k]) for k in
                   ("state", "pose", "rgb", "depth_m", "history", "rgbd"))
        frames = metadata["frames"]
        assert len(frames) == metadata["frame_count"]
        times = np.load(folder / "frame_times_s.npy")
        assert np.array_equal(times, [f["recording_time_s"] for f in frames])
        errors = []
        for f in frames:
            idx = f["telemetry_frame"]
            terminal = f["terminal"]
            assert (folder / f["file"]).is_file()
            pose = trajectory["terminal_pose"] if terminal else trajectory["pose"][idx]
            state = trajectory["terminal_state"] if terminal else trajectory["state"][idx]
            action = trajectory["action"][-1] if terminal else trajectory["action"][idx]
            assert np.array_equal(f["actual_pose"], pose)
            assert np.array_equal(f["actual_state17"], state)
            assert np.array_equal(f["applied_action"], action)
            errors.append(abs(f["recording_time_s"] - idx * .05))
        assert max(errors) < 1e-9
        assert frames[-1]["terminal"]
        parity = None
        previous = protocol.get("diagnostic_probe_parity", {}).get(arm)
        if previous:
            prior = np.load(base / previous / "trajectory.npz")
            keys = ("state", "action", "pose", "terminal_state", "terminal_pose",
                    "power_kw", "contact_n", "positive_work_kj_per_interval", "parked")
            parity = {k: bool(np.array_equal(trajectory[k], prior[k])) for k in keys}
            assert all(parity.values()), (arm, parity)
        rows.append({"arm": arm, "status": outcome["status"],
                     "goal_time_s": outcome["goal_time_s"],
                     "goal_progress_final_m": outcome["goal_progress_m"],
                     "asset_contact": outcome["asset_contact"],
                     "max_asset_contact_n": outcome["max_asset_contact_n"],
                     "longest_effortful_near_stop_s": outcome["longest_consecutive_effortful_near_zero_speed_s"],
                     "bounded_blockage_v1": outcome["bounded_blockage_v1"],
                     "video_frames": len(frames),
                     "maximum_time_alignment_error_s": max(errors),
                     "all_video_frames_exactly_match_recorded_telemetry": True,
                     "matched_shared_anchor": True,
                     "source_and_reference_sha_match_freeze": True,
                     "first_visual_frame_sha256": sha(folder / frames[0]["file"]),
                     "passive_video_trajectory_parity": parity})
    same_first = len({r["first_visual_frame_sha256"] for r in rows}) == 1
    assert same_first
    result = {"schema": 1, "all_contract_checks_passed": True,
              "first_visual_frame_identical_across_all4": same_first,
              "scope": "Actual fixed-reference Chrono execution on a geometry-developed scene; forecasts frozen before final recorded runs",
              "rows": rows}
    (base / "actual_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.root), indent=2))
