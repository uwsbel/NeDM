#!/usr/bin/env python3
"""Summarize completed FDM runs without selecting on protected test data."""
from pathlib import Path
import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="JSON output; companion Markdown uses same stem")
    args = ap.parse_args()
    rows = []
    for path in sorted(args.runs.glob("*/status.json")):
        state = json.loads(path.read_text())
        if state["state"] != "complete":
            raise RuntimeError(f"Incomplete run: {path.parent}")
        for selection in ("last", "best"):
            metrics = json.loads((path.parent/f"metrics_{selection}.json").read_text())
            val = metrics["validation"]
            row = {"run": path.parent.name, "selection": selection, "step": metrics["step"],
                   "arm": state["arm"], "seed": state["seed"], "job": state["slurm_job_id"],
                   "ade_m": val["pose"]["ade_m"], "fde_m": val["pose"]["fde_m"],
                   "nominal_fde_m": val["nominal_kinematics"]["fde_m"],
                   "final_work_mae_kj": val["work"]["final_mae_kj"],
                   "val_loss": val["loss"]["total"], "training_seconds": state["training_seconds"],
                   "updates_per_second": state["updates_per_second"], "draw_digest": metrics["draw_digest"],
                   "events": {}}
            for name in ("contact", "low_progress"):
                event = val["events"][name]
                row["events"][name] = {"four_second": event["last_horizon"], "eligible_prefixes": event["window"],
                                        "positive_unique_episodes": event["positive_unique_episodes"]}
            rows.append(row)
    if not rows:
        raise ValueError("No completed FDM runs found")
    final = [r for r in rows if r["selection"] == "last"]
    for seed in sorted({r["seed"] for r in final}):
        subset = [r for r in final if r["seed"] == seed]
        if len({r["draw_digest"] for r in subset}) != 1 or len({r["step"] for r in subset}) != 1:
            raise ValueError("Final sample sequences or update budgets differ across arms")
    if len(final) != 8 or {r["arm"] for r in final} != {"profile", "no_history", "history", "no_terrain"}:
        raise ValueError("Expected the complete four-arm, two-seed suite")
    payload = {"runs_root": str(args.runs.resolve()), "rows": rows,
               "matched_final_sampler_hashes": True,
               "interpretation": "Offline validation of one standard-PID controller on one terrain, not closed-loop MPPI success.",
               "limitations": ["Rollover has zero positive examples and is disabled.",
                   "Validation has 11 contact-positive and 10 low-progress-positive episodes; windows overlap.",
                   "Without explicit terrain features retains terrain information in speed references and measured history.",
                   "Risk retention tables use total coverage, not fixed feasible-class retention."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    def number(value, digits=3):
        return "unsupported" if value is None else f"{value:.{digits}f}"
    lines = ["# First finite-horizon PID predictor results", "",
        "All training ran on AMD. These are offline validation results on one terrain; no new closed-loop MPPI outcome was measured.", "",
        f"Nominal reference-following prediction: **{number(final[0]['nominal_fde_m'])} m** mean endpoint error at four seconds.", "",
        "| Arm | Seed | Updates | 4 s endpoint error, m | Contact AUC at 4 s | Low-progress AUC at 4 s | Work MAE at 4 s, kJ |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in final:
        lines.append(f"| {r['arm']} | {r['seed']} | {r['step']} | {number(r['fde_m'])} | "
                     f"{number(r['events']['contact']['four_second']['auroc'])} | "
                     f"{number(r['events']['low_progress']['four_second']['auroc'])} | {number(r['final_work_mae_kj'])} |")
    lines += ["", "Rows above use the fixed-budget final checkpoints. The companion JSON also contains validation-selected checkpoints and calibration/support counts.", "",
        "All arms with the same seed used identical sampled windows. The profile baseline has 35,680 parameters; the three other arms have 211,680 each.", "",
        "Validation contains 250 episodes, including only 11 contact-positive and 10 low-progress-positive episodes. Its 5,000 overlapping windows are not independent trials. Rollover prediction is unsupported and disabled.", "",
        "The geometry input is privileged BMP terrain plus authored asset footprints. The no_terrain arm removes only explicit terrain columns; speed references and state history can retain terrain information.", "",
        "The next decision gate is matched short-horizon failure judgment and physical execution under the same driver, followed by CEM versus MPPI using the same scorer."]
    args.out.with_suffix(".md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines[:9+len(final)]))


if __name__ == "__main__":
    main()
