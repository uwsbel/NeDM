#!/usr/bin/env python3
"""Condense job412094 reports into a readable, plotted evidence note.

The narrative is specific to this experiment and pinned to its report hashes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def risk(group, event):
    return group["events"][event]["planner_prefix_max_probability"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-pair", type=Path, required=True)
    parser.add_argument("--training-audit", type=Path, required=True)
    parser.add_argument("--route-support", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    expected = {"h60": "3fdecf3bf0a495278754908b5b96a8646d5f1d13a0e4b118e562623b03bd5f1b",
                "h20": "af178858e27e39cbfcf64eb86d4fd74536d15dcf368e60f35238391dc46fc4a3"}
    if any(sha(args.report_pair / horizon / "report.json") != digest for horizon, digest in expected.items()):
        raise ValueError("This numerical narrative is pinned to the frozen job412094 reports")
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / name).exists() for name in ("evidence.json", "evidence.md", "forecast_validation.png")):
        raise ValueError("Preserve existing evidence artifacts")
    reports = {key: json.loads((args.report_pair / key / "report.json").read_text()) for key in ("h60", "h20")}
    audit = json.loads(args.training_audit.read_text())
    supports = json.loads(args.route_support.read_text())
    result = {"schema": "fdm_diverse_forecast_evidence_v1", "data_scope": {"validation_scenes": 6, "validation_routes": 90,
        "validation_windows": 13625, "pre_first_observed_failure_windows": 4447, "windows_overlap": True, "test_data_opened": False},
        "source_sha256": {"h60_report": sha(args.report_pair / "h60/report.json"), "h20_report": sha(args.report_pair / "h20/report.json"),
            "training_audit": sha(args.training_audit), "route_support": sha(args.route_support), "evidence_script": sha(__file__)},
        "training_controls": {"all_eight_runs_updates": 5000, "same_seed_window_and_rotation_digests_match": audit["same_seed_window_and_rotation_digests_match_all_horizons_and_image_arms"],
            "parameter_count_h60": 3052556, "parameter_count_h20": 1649316,
            "capacity_note": "RGB-D/blank are capacity matched within each horizon. H60/H20 changes capacity, command normalization, and future supervision; it is not a capacity-matched horizon ablation."},
        "fixed_final_h60_12s": {}, "primary_h60_rgbd_seed11_best": {}, "fixed_route_baselines_validation": {}}
    lines = ["# Full-cohort forecast evidence", "", "RGB-D provides useful advance hazard information on six unseen validation arenas. Closed-loop route completion and energy efficiency still require the separate Chrono MPPI evaluation.", "",
        "All eight models used 5,000 updates, 53,830 train windows and 13,625 validation windows. Seeds and training-window/rotation draws match. The 4,447 pre-onset windows have no previously completed contact, rollover, bounded-motion, or sustained-stall event; overlapping windows are not independent trials.", "",
        "## Matched final checkpoints: 12-second forecasts", "", "| Seed | Model | Pre-onset contact AUROC | Pre-onset bounded-motion AUROC | Pre-onset FDE m | All-window FDE m |", "|---:|---|---:|---:|---:|---:|"]
    for seed in (11, 29):
        for arm in ("rgbd", "blank"):
            run = reports["h60"]["runs"][f"{arm}_s{seed}"]
            checkpoint = run["checkpoints"]["last"]
            normal = checkpoint["controls"]["normal"]["horizons"]["12s"]
            pre, all_ = normal["groups"]["pre_first_observed_failure"], normal["groups"]["all"]
            row = {"contact_pre_onset": risk(pre, "contact"), "bounded_motion_pre_onset": risk(pre, "progress_event"),
                "rollover_pre_onset": risk(pre, "rollover"), "motion_pre_onset": pre["motion"], "work_pre_onset_kj": pre["work_kj"],
                "motion_all": all_["motion"], "work_all_kj": all_["work_kj"], "per_scene_pre_onset": normal["per_scene_causal"],
                "image_controls": checkpoint["image_dependence_control_minus_normal"]}
            result["fixed_final_h60_12s"][f"{arm}_s{seed}"] = row
            lines.append(f"| {seed} | {arm} | {risk(pre,'contact')['metrics']['auroc']:.3f} | {risk(pre,'progress_event')['metrics']['auroc']:.3f} | {pre['motion']['endpoint_fde_m']:.2f} | {all_['motion']['endpoint_fde_m']:.2f} |")
    wins = {}
    for seed in (11, 29):
        a, b = [result["fixed_final_h60_12s"][f"{arm}_s{seed}"]["per_scene_pre_onset"] for arm in ("rgbd", "blank")]
        wins[str(seed)] = {event: sum(risk(a[scene]["pre_first_observed_failure"], event)["metrics"]["auroc"] >
            risk(b[scene]["pre_first_observed_failure"], event)["metrics"]["auroc"] for scene in a) for event in ("contact", "progress_event")}
    result["scenes_with_rgbd_pre_onset_auc_above_blank"] = wins
    primary = reports["h60"]["runs"]["rgbd_s11"]["checkpoints"]["best"]
    result["primary_h60_rgbd_seed11_best"] = {"selection_predeclared": True, "update": primary["step"], "controls": {}}
    for control in ("normal", "shuffle", "blank"):
        result["primary_h60_rgbd_seed11_best"]["controls"][control] = primary["controls"][control]["horizons"]["12s"]["groups"]["pre_first_observed_failure"]
    lines += ["", "RGB-D improves pre-onset contact and bounded-motion AUROC on all six arenas for both seeds. Shuffling scene images increases overall 12-second FDE by 1.76 / 2.35 m and mechanical-work MAE by 18.0 / 39.9 kJ; the learned model uses information specific to the scene.", "",
        "## Predeclared planning checkpoint and remaining limits", "",
        "The primary planning model remains H60 RGB-D seed 11, validation-best at update 4,000. Its pre-onset contact / bounded-motion AUROC is 0.871 / 0.897. With shuffled images these fall to 0.611 / 0.612.", "",
        "At the existing contact / bounded-motion probability caps (0.35 / 0.50), this primary checkpoint recalls 63.4% / 51.1% of positive pre-onset windows while retaining 92.0% / 95.2% of negative windows. Good ranking is not reliable default-threshold safety. Rollover recall is only 17.0% at the 0.35 cap, with four positive validation routes in two arenas.", "",
        "The trained stall proxy is entirely future two-second bounded motion under throttle; separately recorded sustained low-speed stall is a different target. Contact means asset or chassis resultant above 1 N. Signed roll/pitch endpoints do not guarantee detection of solver-step attitude peaks.", "",
        "Primary 12-second mechanical-work MAE is 102.2 kJ overall (17.6% weighted absolute error) and 111.5 kJ before first failure (34.5%). This is positive engine-interface mechanical work, not fuel consumption. Aggregate work error includes many already-failed/stalled windows; energy-efficient route selection must be measured among safely completed Chrono routes.", "",
        "H20 improves short-horizon average FDE but does not beat its blank controls on that metric. H60 has 3,052,556 parameters versus H20's 1,649,316, so horizon comparisons also change capacity and future supervision. Validation-best and fixed-final checkpoints are reported separately; the primary seed is unchanged.", ""]
    scenes = [row for row in supports["scenes"] if row["split"] == "val"]
    for offset in (-44., 44.):
        safe = [row["scene_id"] for row in scenes if any(r["offset_m"] == offset and r["cruise_speed_mps"] == 6. for r in row["safe_controls"])]
        result["fixed_route_baselines_validation"][str(offset)] = {"safe_scenes": safe, "safe_completions": len(safe), "total_scenes": len(scenes), "speed_mps": 6.}
    lines += ["## Coverage boundary", "", "The prescribed safe routes all use the outer ±44 m offsets. A fixed −44 m route succeeds on 4/6 validation arenas; fixed +44 m succeeds on 3/6. Both should remain visible baselines. The dataset provides diverse failure/vehicle signals, but does not yet demonstrate arbitrary safe interior-route topology.", "",
              "Next gate: measure matched candidate selection and online Chrono completion, contact/stall/rollover, time and work. Report failures and abstentions before comparing energy or time among paired safe goal completions.", ""]
    (args.out / "evidence.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (args.out / "evidence.md").write_text("\n".join(lines))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = {"rgbd": "#1965ad", "blank": "#cb6f19"}
    for arm in ("rgbd", "blank"):
        for seed in (11, 29):
            checkpoint = reports["h60"]["runs"][f"{arm}_s{seed}"]["checkpoints"]["last"]
            groups = [checkpoint["controls"]["normal"]["horizons"][f"{h}s"]["groups"]["pre_first_observed_failure"] for h in (4, 8, 12)]
            ys = [[risk(group, event)["metrics"]["auroc"] for group in groups] for event in ("contact", "progress_event")]
            ys += [[group["motion"]["endpoint_fde_m"] for group in groups], [group["work_kj"]["mae"] for group in groups]]
            for ax, values in zip(axes.flat, ys):
                ax.plot([4, 8, 12], values, color=colors[arm], linestyle="-" if seed == 11 else "--", marker="o" if seed == 11 else "s", label=f"{arm.upper()}, seed {seed}")
    titles = ("Pre-onset contact ranking", "Pre-onset bounded-motion ranking", "Pre-onset position error", "Pre-onset mechanical-work error")
    units = ("AUROC (higher is better)", "AUROC (higher is better)", "Endpoint error, m (lower is better)", "Work MAE, kJ (lower is better)")
    for i, ax in enumerate(axes.flat):
        ax.set(title=titles[i], xlabel="Forecast horizon, s", ylabel=units[i], xticks=[4, 8, 12])
        ax.grid(alpha=.22)
        if i < 2:
            ax.set_ylim(.5, 1)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("RGB-D forecast validation on six unseen arenas\nMatched H60 models at 5,000 updates; 4,447 overlapping pre-onset windows", fontsize=13)
    fig.savefig(args.out / "forecast_validation.png", dpi=180)
    fig.savefig(args.out / "forecast_validation.pdf")
    plt.close(fig)
    print(json.dumps({"out": str(args.out), "evidence_sha256": sha(args.out / "evidence.json")}))


if __name__ == "__main__":
    main()
