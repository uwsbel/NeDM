"""Compare AMD Chrono against the five approved local reference traces."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from nedm.bouncing_ball.collection import STATE_FIELDS, atomic_json, provenance, simulate_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.reference / "dataset_index.json").read_text())
    results = []
    for episode in index["episodes"]:
        with (args.reference / episode["csv_path"]).open(newline="") as stream:
            truth = list(csv.DictReader(stream))
        rows, metadata = simulate_episode(index["config"], episode["launch"]["vx_mps"], episode["launch"]["vz_mps"])
        times = [e["time_s"] for e in episode["events"]]
        pairs = [(a, b) for a, b in zip(truth, rows) if all(abs(float(a["time_s"])-t)>0.002 for t in times)]
        errors = {key: max(abs(float(a[key])-b[key]) for a, b in pairs) for key in STATE_FIELDS}
        original_events = {e["kind"]: e["time_s"] for e in episode["events"]}
        event_error = max(abs(e["time_s"]-original_events[e["kind"]]) for e in metadata["events"] if e["kind"] in original_events)
        passed = (metadata["accepted"] and max(errors[k] for k in ["x_m", "z_m"]) < 0.01
                  and max(errors[k] for k in ["vx_mps", "vz_mps"]) < 0.02
                  and errors["omega_y_radps"] < 0.1 and event_error < 0.001)
        results.append({"launch_label": episode["launch_label"], "passed": passed, "errors": errors,
                        "event_error_s": event_error, "metadata": metadata})
    report = {"passed": all(e["passed"] for e in results), "host": platform.node(), "runtime": provenance(),
              "reference_index_sha256": hashlib.sha256((args.reference / "dataset_index.json").read_bytes()).hexdigest(), "episodes": results}
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
