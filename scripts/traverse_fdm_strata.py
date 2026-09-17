#!/usr/bin/env python3
"""Read-only, causal-anchor strata for the first AMD FDM pilot predictions.

Reads saved validation predictions and their existing labels. Does not train,
select new checkpoints, collect data, or open protected test episodes. The
strata are exploratory diagnostics chosen after the first pilot feedback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EVENTS = ("contact", "rollover", "low_progress")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def auc(label, score):
    label, score = np.asarray(label, bool), np.asarray(score, float)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_score))+1]
    ends = np.r_[starts[1:], len(score)]
    ranks = np.empty(len(score))
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start+1+end)/2.
    return float((ranks[label].sum()-n_pos*(n_pos+1)/2)/(n_pos*n_neg))


def event_metrics(label, score, probability, episodes):
    label = np.asarray(label, bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    result = {"known_windows": len(label), "positive_windows": n_pos, "negative_windows": n_neg,
              "known_episodes": int(np.unique(episodes).size),
              "positive_episodes": int(np.unique(episodes[label]).size), "auroc": auc(label, score)}
    if probability is not None and len(label):
        result["brier"] = float(np.mean((probability-label)**2))
        result["positive_mean_probability"] = float(probability[label].mean()) if n_pos else None
        result["negative_mean_probability"] = float(probability[~label].mean()) if n_neg else None
    if n_pos and n_neg:
        threshold = float(np.quantile(score[~label], .95, method="higher"))
        result["roc_diagnostic_at_95pct_negative_retention"] = {
            "threshold_from_this_validation_stratum": threshold,
            "negative_retention": float((score[~label] <= threshold).mean()),
            "failure_recall": float((score[label] > threshold).mean()),
            "false_accepts": int((score[label] <= threshold).sum())}
    return result


def pose_metrics(predicted, target, valid, episodes):
    mask = valid[..., 0] > .5
    final = mask[:, -1]
    error = np.linalg.norm(predicted[..., :2]-target[..., :2], axis=-1)
    yaw = np.arctan2(predicted[..., 2], predicted[..., 3])-np.arctan2(target[..., 2], target[..., 3])
    yaw_error = np.abs(np.arctan2(np.sin(yaw), np.cos(yaw))) * 180./np.pi
    return {"observed_points": int(mask.sum()), "complete_4s_windows": int(final.sum()),
            "complete_4s_episodes": int(np.unique(episodes[final]).size),
            "ade_m": float(error[mask].mean()) if mask.any() else None,
            "fde_m": float(error[final, -1].mean()) if final.any() else None,
            "final_yaw_mae_deg": float(yaw_error[final, -1].mean()) if final.any() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/traverse/fdm_fast_data_v2"))
    parser.add_argument("--runs", type=Path, default=Path("artifacts/traverse/fdm_runs/pilot_v1"))
    parser.add_argument("--source-root", type=Path, default=Path("/home/harry/NeDM"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.runs
    records = json.loads((args.data/"val_episodes.json").read_text())
    manifest = json.loads((args.data/"manifest.json").read_text())
    with np.load(args.data/"val.npz") as handle:
        data = {k: handle[k].copy() for k in handle.files}
    val_hash = sha256(args.data/"val.npz")
    assert val_hash == manifest["splits"]["val"]["sha256"]
    history, anchors, episodes = data["history"], data["anchor"], data["episode_index"]
    count = len(anchors)
    displacement = np.linalg.norm(history[:, -1, 20:22]-history[:, 0, 20:22], axis=1)
    vx = history[:, -1, 0]
    effort = history[:, :, 18].mean(axis=1)
    full_history = anchors >= 15
    moving = full_history & (vx >= 1.) & (displacement >= .5)
    slow = full_history & (np.abs(vx) < .5)
    already_low = full_history & (displacement < .15*.75) & (effort > .3)
    past_contact, recent_contact = np.zeros(count, bool), np.zeros(count, bool)
    metadata, families = [], []
    for ep_index, record in enumerate(records):
        path = args.source_root/record["source"]/"meta.json"
        assert sha256(path) == record["meta_sha256"], f"Metadata changed: {path}"
        meta = json.loads(path.read_text())
        metadata.append(meta)
        families.append(meta["family_actual"])
        frames = np.array([event[0] for event in meta["contact"]["events"] if event[2] > 1.])
        indices = np.flatnonzero(episodes == ep_index)
        # Metadata contains the recording's complete event list; only event
        # intervals ending by the anchor define these causal strata.
        past_contact[indices] = [np.any(frames < anchor) for anchor in anchors[indices]]
        recent_contact[indices] = [np.any((frames >= anchor-4) & (frames < anchor)) for anchor in anchors[indices]]
    groups = {
        "all": np.ones(count, bool), "full_history": full_history,
        "moving": moving, "slow": slow, "already_low_progress": already_low,
        "startup_padded_history": ~full_history,
        "before_first_contact": ~past_contact,
        "recent_contact": recent_contact,
        "past_contact_not_recent": past_contact & ~recent_contact,
        "moving_before_first_contact": moving & ~past_contact,
    }
    support = {}
    for name, group in groups.items():
        support[name] = {"windows": int(group.sum()), "episodes": int(np.unique(episodes[group]).size), "events_4s": {}}
        for e, event in enumerate(EVENTS):
            known = group & (data["event_mask"][:, -1, e] > .5)
            label = data["events"][known, -1, e] > .5
            support[name]["events_4s"][event] = {"known_windows": int(known.sum()),
                "positive_windows": int(label.sum()), "negative_windows": int((~label).sum()),
                "positive_episodes": int(np.unique(episodes[known][label]).size)}
    payload = {
        "protocol": "Post-hoc exploratory strata, saved best-validation-loss checkpoints; no new selection or training",
        "horizon_seconds": 4., "validation_sha256": val_hash,
        "definitions": {
            "full_history": "anchor >=15; excludes synthetic startup history",
            "moving": "full history AND current body vx>=1 m/s AND net displacement over the past0.75 s>=0.5 m",
            "slow": "full history AND abs(current body vx)<0.5 m/s; may include deliberate parking, excluded by low-progress target mask",
            "already_low_progress": "full history AND net displacement over past0.75 s<0.1125 m AND mean previous throttle>0.3",
            "before_first_contact": "no asset-contact interval with force>1 N recorded before the anchor",
            "recent_contact": "asset-contact interval in past0.2 s before anchor",
            "moving_before_first_contact": "moving and before_first_contact",
            "target_low_progress_4s": "net displacement in next4 s<0.6 m and mean recorded commanded throttle>0.3, without deliberate route-end parking; original frozen labels",
            "future_motion_diagnostics": "For the four moving low-progress positive windows only: raw 20 Hz future path length, endpoint velocity and last2 s net displacement are evaluation diagnostics, never stratum inputs. Low net displacement can include rollback or out-and-back motion.",
            "risk_score": "probability at4 s, not maximum across prediction horizon",
            "roc_thresholds": "95% negative-retention thresholds computed within these validation strata are diagnostic only, not deployable calibrated gates",
        },
        "support": support, "runs": {}, "causal_heuristics": {},
        "limitations": [
            "One previously observed BMP arena, standard fixed PID, existing recorded references; no closed-loop MPPI or new-terrain result.",
            "Overlapping windows within episodes are dependent. Exact positive episode counts are reported; tiny strata do not support significance claims.",
            "Best checkpoints were selected using this validation split's masked loss. The separate fixed-budget last metrics are included only at their originally recorded aggregate scope.",
            "Moving/slow/past-low-progress strata use causal inputs; labels and future motion are used only for evaluation. Strata overlap and were chosen after first pilot feedback.",
            "No rollover-positive training or validation episodes. No rollover forecasting claim is supported.",
        ]}
    # These simple causal comparators expose whether a learned predictor adds
    # prospective information beyond recognizing an existing low-speed state.
    heuristics = {"negative_current_speed": -np.abs(vx), "negative_past_displacement": -displacement}
    for name, score in heuristics.items():
        payload["causal_heuristics"][name] = {}
        for group_name, group in groups.items():
            known = group & (data["event_mask"][:, -1, 2] > .5)
            payload["causal_heuristics"][name][group_name] = event_metrics(
                data["events"][known, -1, 2], score[known], None, episodes[known])
    prospective = moving & (data["event_mask"][:, -1, 2] > .5) & (data["events"][:, -1, 2] > .5)
    case_indices = np.flatnonzero(prospective)
    cases = []
    for index in case_indices:
        ep_index = int(episodes[index])
        state_path = args.source_root/records[ep_index]["source"]/"states.npz"
        assert sha256(state_path) == records[ep_index]["states_sha256"], f"States changed: {state_path}"
        with np.load(state_path) as handle:
            fields = {name: i for i, name in enumerate(handle["fields"].tolist())}
            anchor = int(anchors[index])
            future = handle["table"][anchor:anchor+81].copy()
        assert len(future) == 81, "Motion diagnostic requires the complete four-second trajectory"
        xy = future[:, [fields["pos_x_m"], fields["pos_y_m"]]]
        future_vx = future[:, fields["vel_body_x_mps"]]
        future_diagnostics = {
            "path_length_4s_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
            "max_range_from_anchor_m": float(np.linalg.norm(xy-xy[0], axis=1).max()),
            "last_2s_net_displacement_m": float(np.linalg.norm(xy[-1]-xy[40])),
            "minimum_vx_mps": float(future_vx.min()),
            "endpoint_vx_mps": float(future_vx[-1]),
            "mean_commanded_throttle": float(future[:-1, fields["driver_throttle"]].mean()),
            "raw_states_sha256": records[ep_index]["states_sha256"],
        }
        cases.append({"validation_row": int(index), "episode": records[ep_index]["id"],
                      "anchor_frame": int(anchors[index]), "family": families[ep_index],
                      "current_vx_mps": float(vx[index]), "past_075s_displacement_m": float(displacement[index]),
                      "prior_contact": bool(past_contact[index]), "recent_contact": bool(recent_contact[index]),
                      "true_4s_displacement_m": float(np.linalg.norm(data["trajectory"][index, -1, :2])),
                      "nominal_4s_displacement_m": float(np.linalg.norm(data["nominal_pose"][index, -1, :2])),
                      "future_motion_diagnostics": future_diagnostics,
                      "predictions": {}})
    for folder in sorted(args.runs.iterdir()):
        prediction_path = folder/"val_predictions_best.npz"
        if not prediction_path.exists():
            continue
        provenance = json.loads((folder/"provenance.json").read_text())
        assert provenance["data"]["val.npz"] == val_hash, folder
        best = json.loads((folder/"metrics_best.json").read_text())
        last = json.loads((folder/"metrics_last.json").read_text())
        with np.load(prediction_path) as handle:
            prediction = {k: handle[k].copy() for k in handle.files}
        assert prediction["trajectory"].shape == data["trajectory"].shape
        probability = 1./(1.+np.exp(-np.clip(prediction["event_logits"], -80., 80.)))
        run = {"best_step": best["step"], "saved_predictions_sha256": sha256(prediction_path),
               "strata": {}, "fixed_budget_last": {"step": last["step"],
                "pose": last["validation"]["pose"], "events_4s": {
                    name: last["validation"]["events"][name]["last_horizon"] for name in ("contact", "low_progress")}}}
        for name, group in groups.items():
            row = {"pose": pose_metrics(prediction["trajectory"][group], data["trajectory"][group],
                                         data["trajectory_mask"][group], episodes[group]),
                   "nominal_pose": pose_metrics(data["nominal_pose"][group], data["trajectory"][group],
                                                 data["trajectory_mask"][group], episodes[group]), "events_4s": {}}
            for e in (0, 2):
                known = group & (data["event_mask"][:, -1, e] > .5)
                score = probability[known, -1, e]
                row["events_4s"][EVENTS[e]] = event_metrics(data["events"][known, -1, e], score, score, episodes[known])
            run["strata"][name] = row
        # Verify independent four-second AUCs against original aggregate output.
        for event in ("contact", "low_progress"):
            actual = run["strata"]["all"]["events_4s"][event]["auroc"]
            expected = best["validation"]["events"][event]["last_horizon"]["auroc"]
            assert actual is not None and abs(actual-expected) < 1e-6, (folder, event, actual, expected)
        payload["runs"][folder.name] = run
        for case, index in zip(cases, case_indices):
            case["predictions"][folder.name] = {
                "displacement_m": float(np.linalg.norm(prediction["trajectory"][index, -1, :2])),
                "fde_m": float(np.linalg.norm(prediction["trajectory"][index, -1, :2]-data["trajectory"][index, -1, :2])),
                "low_progress_probability_4s": float(probability[index, -1, 2]),
                "contact_probability_4s": float(probability[index, -1, 0])}
    assert len(payload["runs"]) == 8, "Expected all eight completed pilot arms"
    payload["moving_low_progress_cases"] = cases
    out.mkdir(parents=True, exist_ok=True)
    (out/"strata.json").write_text(json.dumps(payload, indent=2)+"\n")

    lines = ["# Causal-anchor strata: first AMD FDM pilot", "",
        "The very high pooled low-progress AUC mainly reflects recognizing vehicles already moving slowly. This pilot contains very few prospective low-progress entries.", "",
        "These are post-hoc diagnostics of **best validation-loss checkpoints**, on the same development validation split used for checkpoint selection. The prediction horizon is four seconds. All inputs defining anchor strata are causal; recorded futures supply targets only.", "",
        "| Causal anchor group | Known low-progress windows | Positive windows / episodes | Known contact windows | Positive windows / episodes |",
        "|---|---:|---:|---:|---:|"]
    for name in ("all", "moving", "slow", "already_low_progress", "moving_before_first_contact", "before_first_contact", "recent_contact"):
        low, contact = [support[name]["events_4s"][key] for key in ("low_progress", "contact")]
        lines.append(f"| {name} | {low['known_windows']} | {low['positive_windows']} / {low['positive_episodes']} | {contact['known_windows']} | {contact['positive_windows']} / {contact['positive_episodes']} |")
    lines += ["", "Moving means current vx≥1 m/s and at least 0.5 m displacement in the past 0.75 s, with fully recorded history. Already low progress means past displacement<0.1125 m under mean previous throttle>0.3. Before first contact means no previous recorded asset-contact interval. Slow means |current vx|<0.5 m/s. Groups overlap. Future route-end parking is excluded only by the original low-progress target mask.", "",
        "| Best-loss checkpoint | Step | All ADE / 4 s FDE, m | Moving 4 s FDE, m | Low-progress AUC, all | Low-progress AUC, moving | Contact AUC, moving before first contact |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "unsupported" if value is None else f"{value:.4f}"
    for name, run in payload["runs"].items():
        all_, moving_, prospective_ = [run["strata"][k] for k in ("all", "moving", "moving_before_first_contact")]
        lines.append(f"| {name} | {run['best_step']} | {fmt(all_['pose']['ade_m'])} / {fmt(all_['pose']['fde_m'])} | {fmt(moving_['pose']['fde_m'])} | {fmt(all_['events_4s']['low_progress']['auroc'])} | {fmt(moving_['events_4s']['low_progress']['auroc'])} | {fmt(prospective_['events_4s']['contact']['auroc'])} |")
    first = next(iter(payload["runs"].values()))
    nominal_all, nominal_moving = first["strata"]["all"]["nominal_pose"], first["strata"]["moving"]["nominal_pose"]
    lines += ["", f"Nominal-reference kinematics: all ADE/FDE {nominal_all['ade_m']:.3f}/{nominal_all['fde_m']:.3f} m; moving FDE {nominal_moving['fde_m']:.3f} m.", "",
        "| Causal heuristic for low progress | All AUC | Moving AUC |",
        "|---|---:|---:|"]
    for name, result in payload["causal_heuristics"].items():
        lines.append(f"| {name} | {fmt(result['all']['auroc'])} | {fmt(result['moving']['auroc'])} |")
    lines += ["", "The moving low-progress positives are listed below. Two already had contact, so only two episodes combine clear current motion, no prior contact and an impending low-progress target. Low progress is a net-displacement label, not a sustained-stall label: the spline example rolls back after forward motion (3.62 m traveled, 0.308 m net, final vx −1.08 m/s). The other previously contact-free example nearly stops (0.565 m net over four seconds, 0.012 m net in the last two seconds). Individual predictions and raw motion diagnostics are preserved in strata.json. This is insufficient support for a general terrain-stall or recovery claim.", "",
        "| Episode / anchor | Prior contact | Actual 4 s displacement, m | Nominal displacement, m | History s11 / s29 displacement, m |",
        "|---|---|---:|---:|---:|"]
    for case in cases:
        pred = case["predictions"]
        lines.append(f"| {case['episode']} / {case['anchor_frame']} | {case['prior_contact']} | {case['true_4s_displacement_m']:.3f} | {case['nominal_4s_displacement_m']:.3f} | {pred['history_s11']['displacement_m']:.3f} / {pred['history_s29']['displacement_m']:.3f} |")
    lines += ["", "| Episode / anchor | Actual path traveled, m | Actual last 2 s net motion, m | Actual endpoint vx, m/s |",
        "|---|---:|---:|---:|"]
    for case in cases:
        diagnostic = case["future_motion_diagnostics"]
        lines.append(f"| {case['episode']} / {case['anchor_frame']} | {diagnostic['path_length_4s_m']:.3f} | {diagnostic['last_2s_net_displacement_m']:.3f} | {diagnostic['endpoint_vx_mps']:.3f} |")
    lines += ["", "Fixed-budget last checkpoints are separate from these selected checkpoints:", "",
        "| Last checkpoint | Step | ADE / FDE, m | Contact / low-progress AUC at 4 s |",
        "|---|---:|---:|---:|"]
    for name, run in payload["runs"].items():
        last = run["fixed_budget_last"]
        lines.append(f"| {name} | {last['step']} | {last['pose']['ade_m']:.4f} / {last['pose']['fde_m']:.4f} | {fmt(last['events_4s']['contact']['auroc'])} / {fmt(last['events_4s']['low_progress']['auroc'])} |")
    lines += ["", "No new model or optimizer updates were performed for this analysis. No protected test episodes were read. Overlapping windows are not independent trials; episode counts matter. The raw stores cover one BMP arena and one PID driver domain. This is neither unseen-terrain validation nor closed-loop MPPI performance. Neither training nor validation has rollover-positive episodes.", ""]
    (out/"strata.md").write_text("\n".join(lines))
    print(json.dumps({"out": str(out), "runs": len(payload["runs"]), "support": support,
                      "moving_low_progress_cases": [{k:v for k,v in case.items() if k != "predictions"} for case in cases]}, indent=2))


if __name__ == "__main__":
    main()
