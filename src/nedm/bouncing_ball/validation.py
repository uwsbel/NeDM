"""Validate recorded timing, five-state fields, contact order and pre-impact ends."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from nedm.bouncing_ball.collection import CSV_FIELDS, NUMERIC_FIELDS, STATE_FIELDS, atomic_json


def validate_dataset(root: Path) -> dict:
    root = root.resolve()
    index = json.loads((root / "dataset_index.json").read_text())
    config = json.loads((root / "collector_config.resolved.json").read_text())
    schema = json.loads((root / "schema.json").read_text())
    errors, durations, frame_count = [], [], 0
    episodes = index["episodes"]
    acceptance, scene = config["acceptance"], config["scene"]
    dt = config["simulation"]["record_step_s"]
    sim_dt = config["simulation"]["step_s"]
    force_threshold = config["contact"]["force_threshold_n"]
    if not index["complete"] or len(episodes) != index["requested_episodes"]:
        errors.append("dataset is incomplete")
    if index["episode_count"] != len(episodes):
        errors.append("episode count does not match the index")
    if schema["state_fields"] != STATE_FIELDS or schema["action_fields"] != []:
        errors.append("state/action schema does not match the passive five-state contract")
    if config != index["config"]:
        errors.append("resolved config differs from index config")
    if len({episode["episode_id"] for episode in episodes}) != len(episodes):
        errors.append("duplicate episode IDs")
    split_counts = Counter(ep["split"] for ep in episodes)
    if dict(split_counts) != index["split_counts"]:
        errors.append("split counts differ from index")
    for episode in episodes:
        prefix = episode["episode_id"]
        start_errors = len(errors)

        def require(condition: bool, message: str) -> None:
            if not condition:
                errors.append(f"{prefix}: {message}")

        path = (root / episode["csv_path"]).resolve()
        if not path.is_relative_to(root):
            errors.append(f"{prefix}: CSV path escapes the dataset")
            continue
        sidecar = json.loads(path.with_suffix(".json").read_text())
        require(sidecar == episode, "sidecar differs from index record")
        require(episode["accepted"] and episode["termination"] == "next_ground_contact", "episode was not accepted")
        require(episode["split"] in ("train", "val", "test"), "unknown split")
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames == CSV_FIELDS, "CSV fields differ from schema")
            raw_rows = list(reader)
        require(len(raw_rows) == episode["rows"] and len(raw_rows) >= 2, "row count mismatch or too few samples")
        if len(errors) != start_errors:
            continue
        try:
            rows = [{key: float(row[key]) for key in NUMERIC_FIELDS} for row in raw_rows]
        except (ValueError, KeyError) as error:
            errors.append(f"{prefix}: invalid numeric field: {error}")
            continue
        require(all(math.isfinite(value) for row in rows for value in row.values()), "nonfinite data")
        if len(errors) != start_errors:
            continue
        for sample, (raw, row) in enumerate(zip(raw_rows, rows, strict=True)):
            require(raw["episode_id"] == prefix and raw["split"] == episode["split"], "mixed episode/split rows")
            require(row["sample_index"] == sample and abs(row["time_s"] - sample * dt) < 1e-9, "nonuniform sample timing")
            require(abs(row["y_m"]) <= acceptance["max_out_of_plane_position_m"], "position is not planar")
            require(abs(row["vy_mps"]) <= acceptance["max_out_of_plane_velocity_mps"], "velocity is not planar")
            require(max(abs(row["omega_x_radps"]), abs(row["omega_z_radps"])) <= acceptance["max_out_of_plane_angular_velocity_radps"], "spin is not planar")
            require(abs(row["ground_gap_m"] - (row["z_m"] - scene["ground_z_m"] - scene["radius_m"])) < 1e-10, "ground gap is inconsistent")
            require(abs(row["wall_gap_m"] - (scene["wall_front_x_m"] - row["x_m"] - scene["radius_m"])) < 1e-10, "wall gap is inconsistent")
            for surface in ("ground", "wall"):
                force = row[f"{surface}_normal_force_n"]
                require(force >= 0 and row[f"{surface}_contact"] == int(force > force_threshold), "contact flag disagrees with measured force")
        initial, final = rows[0], rows[-1]
        require(initial["x_m"] == scene["launch_x_m"] and initial["z_m"] == scene["launch_z_m"], "incorrect launch position")
        require(initial["vx_mps"] == episode["launch"]["vx_mps"] and initial["vz_mps"] == episode["launch"]["vz_mps"], "incorrect launch velocity")
        require(initial["omega_y_radps"] == config["launch"]["omega_y_radps"], "incorrect launch spin")
        require(config["launch"]["vx_range_mps"][0] <= initial["vx_mps"] <= config["launch"]["vx_range_mps"][1], "vx outside configured range")
        require(config["launch"]["vz_range_mps"][0] <= initial["vz_mps"] <= config["launch"]["vz_range_mps"][1], "vz outside configured range")
        events = episode["events"]
        kinds = [event["kind"] for event in events]
        times = {event["kind"]: event["time_s"] for event in events}
        required = ["first_ground_contact", "ground_release", "wall_contact", "wall_rebound", "wall_release", "next_ground_contact"]
        require(kinds == required, f"unexpected contact sequence {kinds}")
        if any(kind not in times for kind in required):
            continue
        require(times[required[0]] < times[required[1]] < times[required[2]] <= times[required[3]] < times[required[4]] < times[required[5]], "contact timestamps are out of order")
        for surface, event in [("ground", "first_ground_contact"), ("wall", "wall_contact")]:
            contact_rows = [row for row in rows if row[f"{surface}_contact"]]
            require(bool(contact_rows), f"{surface} contact missing from sampled force channels")
            if contact_rows:
                require(times[event] <= contact_rows[0]["time_s"] <= times[event] + dt + 1e-9, "sampled contact is not aligned with the event")
        require(not any(row["ground_contact"] for row in rows if row["time_s"] > times["wall_contact"]), "second ground impact leaked into the saved trajectory")
        require(final["ground_gap_m"] > 0 and final["vz_mps"] < 0, "episode does not end clear of the ground while descending")
        require(final["vx_mps"] <= -acceptance["min_rebound_speed_mps"], "episode does not end returning from the wall")
        require(times["next_ground_contact"] > final["time_s"] and times["next_ground_contact"] - final["time_s"] <= 3 * dt + 2 * sim_dt, "last saved sample is not immediately before the next ground impact")
        require(final["time_s"] == episode["duration_s"], "duration does not match the final sample")
        require(all(final[key] == episode["final_state"][key] for key in STATE_FIELDS), "final state differs from metadata")
        require(acceptance["duration_window_s"][0] <= final["time_s"] <= acceptance["duration_window_s"][1], "duration outside acceptance window")
        require(final["time_s"] - times["wall_contact"] >= acceptance["min_post_wall_duration_s"], "insufficient post-wall flight")
        require(scene["wall_front_x_m"] - scene["radius_m"] - final["x_m"] >= acceptance["min_return_distance_m"], "insufficient return distance")
        require(episode["max_penetration_m"] <= acceptance["max_penetration_m"], "excessive penetration in the physics trace")
        for quantity in ("position_m", "velocity_mps", "angular_velocity_radps"):
            key = f"max_out_of_plane_{quantity}"
            require(episode[key] <= acceptance[key], f"excessive out-of-plane {quantity} in the physics trace")
        durations.append(final["time_s"])
        frame_count += len(rows)
    attempts = [json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()]
    if len(attempts) != index["attempt_count"]:
        errors.append("attempt journal count mismatch")
    if [attempt for attempt in attempts if attempt["accepted"]] != episodes:
        errors.append("accepted index records do not match the attempt journal")
    rejected = Counter(attempt["termination"] for attempt in attempts if not attempt["accepted"])
    if dict(rejected) != index["rejection_counts"]:
        errors.append("rejection counts differ from the journal")
    for attempt in attempts:
        if not (root / attempt["csv_path"]).is_file():
            errors.append(f"missing preserved attempt trace: {attempt['episode_id']}")
    return {"passed": not errors, "dataset": str(root), "episodes": len(episodes), "frames": frame_count,
            "state_fields": STATE_FIELDS, "dt_s": dt, "split_counts": dict(split_counts),
            "attempts": len(attempts), "rejection_counts": dict(rejected),
            "duration_s": {"min": min(durations), "mean": sum(durations)/len(durations), "max": max(durations)} if durations else None,
            "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path, help="Optional path for the validation report")
    args = parser.parse_args(argv)
    report = validate_dataset(args.dataset)
    if args.report:
        atomic_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
