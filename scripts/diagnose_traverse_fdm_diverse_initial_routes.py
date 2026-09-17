#!/usr/bin/env python3
"""Read-only, validation-only diagnostic of all initial fixed reference routes.

Run CPU inference on AMD using the pinned online source. Model inputs contain
only observation/history/localization/goal and prescribed reference geometry.
Measured future labels are joined after prediction for error and rank analysis.
This performs no MPPI optimization and never changes a model or cost setting.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def check_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Frozen file hash mismatch: {path}")


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def compare(left, right):
    delta = np.asarray(left, np.float64) - np.asarray(right, np.float64)
    return {"exact": bool(np.array_equal(left, right)), "max_abs": float(np.abs(delta).max()),
            "rms": float(np.sqrt(np.square(delta).mean()))}


def rank(costs):
    order = np.argsort(costs, kind="stable")
    result = np.empty(len(costs), int)
    result[order] = np.arange(1, len(costs) + 1)
    return result


def event_value(data, key, mask, row, channel=0):
    return bool(data[key][row, -1, channel]) if data[mask][row, -1, channel] else None


def world_progress(trajectory, pose, goal):
    xy = np.asarray(trajectory)[..., :2]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    world = pose[:2] + np.stack((c*xy[..., 0]-s*xy[..., 1], s*xy[..., 0]+c*xy[..., 1]), axis=-1)
    return np.linalg.norm(pose[:2]-goal) - np.linalg.norm(world-goal, axis=-1)


def summary_errors(rows):
    def avg(key):
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(values)) if values else None
    return {"routes": len(rows), "ADE_m": avg("ADE_m"), "FDE_m": avg("FDE_m"),
            "goal_progress_MAE_m": avg("goal_progress_abs_error_m"),
            "work_MAE_kj": avg("work_abs_error_kj"), "attitude_MAE_deg": avg("attitude_MAE_deg")}


def run(args):
    started = time.perf_counter()
    if args.out.exists():
        raise FileExistsError("Choose a fresh diagnostic output")
    args.out.mkdir(parents=True)
    check_hash(args.code_root / "source_manifest.json", args.source_manifest_sha256)
    sys.path.insert(0, str(args.code_root / "src"))
    sys.path.insert(0, str(args.code_root / "scripts"))
    import torch
    from nedm.traverse.fdm_diverse_data import build_command_features
    from nedm.traverse.fdm_diverse_model import load_rgbd_checkpoint
    from nedm.traverse.fdm_diverse_planner import RGBDCostConfig, RGBDReferenceScorer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    campaign = args.campaign
    pack = campaign / "packs/full_pair_v1/h60"
    manifest = read(pack / "manifest.json")
    entry = manifest["splits"]["val"]
    assert entry["episodes"] == 90 and entry["scenes"] == 6
    for name, digest in (("val.npz", entry["sha256"]), ("val_rgbd.npy", entry["rgbd_sha256"]),
                         ("val_episodes.json", entry["episodes_sha256"])):
        check_hash(pack / name, digest)
    data = npz(pack / "val.npz")
    images = np.load(pack / "val_rgbd.npy", mmap_mode="r")
    episodes = read(pack / "val_episodes.json")
    assert len(episodes) == 90 and all(ep["split"] == "val" for ep in episodes)
    manifest_path = campaign / "snapshots/campaign_v2/artifacts/traverse/fdm_diverse_v1_20260909/cases/cases.json"
    cases = read(manifest_path)
    records = [record for record in cases["records"] if record["split"] == "val"]
    assert len(records) == 6 and all(len(record["routes"]) == 15 for record in records)
    full_support = read(args.full_route_support)
    assert full_support["manifest_sha256"] == sha(manifest_path)
    full_rows = {(s["scene_id"], r["route_index"]): r for s in full_support["scenes"]
                 if s["split"] == "val" for r in s["routes"]}
    models = [("rgbd_best4000", "patch_rgbd_s11", "best", 4000),
              ("rgbd_last5000", "patch_rgbd_s11", "last", 5000),
              ("blank_best2000", "patch_blank_s11", "best", 2000)]
    report = {"schema": "fdm_diverse_initial_reference_diagnostic_v1", "observed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "All 90 original fixed references on six validation scenes, common measured initial observation. No test data, no MPPI optimization, no outcome filtering, no changes to model/cost selection.",
        "forecast_scope": "12-second learned outcomes; full-route time/work costs are explicit short-horizon extrapolation heuristics.",
        "source_snapshot": str(args.code_root), "source_manifest_sha256": args.source_manifest_sha256,
        "script_sha256": sha(__file__), "full_support_sha256": sha(args.full_route_support),
        "pack_manifest_sha256": sha(pack / "manifest.json"), "cohort_manifest_sha256": sha(manifest_path),
        "device": "AMD CPU inference", "torch_version": str(torch.__version__), "models": {}, "scenes": []}
    scene_contexts = []
    for record in records:
        scene = record["scene_id"]
        case_path = manifest_path.parent / record["case"]
        check_hash(case_path, record["case_sha256"])
        case = read(case_path)
        observation_dir = campaign / "full_cohort_v2/observations" / scene
        observation_path = observation_dir / "observation.npz"
        observation_meta = read(observation_dir / "observation.json")
        check_hash(observation_path, observation_meta["observation_sha256"])
        observation = npz(observation_path)
        pose, history, goal = observation["pose"], observation["history"], observation["goal_xy"]
        np.testing.assert_array_equal(goal, np.asarray(case["goal_xy"], np.float32))
        routes, indices, truth_rows, equivalence = [], [], [], []
        for route_index, route_relative in enumerate(record["routes"]):
            route_path = manifest_path.parent / route_relative
            check_hash(route_path, record["route_sha256"][route_index])
            route = read(route_path)
            matches = [i for i, ep in enumerate(episodes) if ep["scene_id"] == scene and ep["route_sha256"] == record["route_sha256"][route_index]]
            assert len(matches) == 1
            episode_index = matches[0]
            indices0 = np.flatnonzero((data["episode_index"] == episode_index) & (data["anchor"] == 0))
            assert len(indices0) == 1
            index = int(indices0[0])
            ep = episodes[episode_index]
            raw = campaign / "full_cohort_v2/raw" / scene / route_path.stem
            assert Path(ep["source"]).resolve() == raw.resolve()
            for name in ("anchor_state.npz", "trajectory.npz", "rich_intervals.npz", "outcome.json"):
                check_hash(raw / name, ep["source_sha256"][name])
            full = full_rows[(scene, route_index)]
            check_hash(raw / "batch_complete.json", full["completion_sha256"])
            anchor = npz(raw / "anchor_state.npz")
            for key in ("pose", "history", "goal_xy", "state"):
                np.testing.assert_allclose(anchor[key], observation[key], rtol=0., atol=2e-6 if key == "pose" else 1e-5)
            features = build_command_features(route, pose, station=route.get("meta", {}).get("fdm_station"), elapsed_s=0., horizon=60, output_dt=.2)
            checks = {key: compare(features[key], data[key][index]) for key in features}
            checks["history"] = compare(history, data["history"][index])
            image = np.asarray(images[data["image_index"][index]], np.float32)
            checks["image_raw_float32_vs_cache"] = compare(observation["rgbd"], image)
            checks["image_cache_is_fp16_roundtrip"] = bool(np.array_equal(observation["rgbd"].astype(np.float16), images[data["image_index"][index]]))
            assert checks["image_cache_is_fp16_roundtrip"]
            measured = npz(raw / "trajectory.npz")
            intervals = npz(raw / "rich_intervals.npz")
            observed = int(np.count_nonzero(data["trajectory_mask"][index, :, 0]))
            has_end = bool(data["trajectory_mask"][index, -1, 0])
            contact_series = np.maximum(intervals["max_asset_contact_max_resultant_n"], intervals["max_chassis_contact_resultant_n"]) > 1.
            rollover_series = np.maximum(intervals["max_abs_roll_rad"], intervals["max_abs_pitch_rad"]) > np.deg2rad(60.)
            def first_time(values):
                found = np.flatnonzero(values)
                return float((found[0]+1)*.05) if len(found) else None
            truth_events = {"contact": event_value(data,"events","event_mask",index,0),
                            "rollover": event_value(data,"events","event_mask",index,1),
                            "bounded_motion": event_value(data,"bounded_motion","bounded_motion_mask",index),
                            "strict_stall": event_value(data,"sustained_stall","sustained_stall_mask",index)}
            safe12 = all(value is False for key, value in truth_events.items() if key != "strict_stall") if all(value is not None for value in truth_events.values()) else None
            if has_end:
                positions = np.vstack((measured["pose"], measured["terminal_pose"]))
                endpoint = positions[240]
                progress = float(np.linalg.norm(pose[:2]-goal)-np.linalg.norm(endpoint[:2]-goal))
                np.testing.assert_allclose(progress, world_progress(data["trajectory"][index], pose, goal)[-1], atol=1e-5, rtol=0.)
                work = float(intervals["engine_interface_positive_work_kj"][:240].sum(dtype=np.float64))
                np.testing.assert_allclose(work, data["work"][index,-1,0], atol=1e-4, rtol=1e-6)
            else:
                progress, work = None, None
            truth_rows.append({"scene_id":scene,"family":record["family"],"route_index":route_index,"pack_row":index,
                "route_sha256":record["route_sha256"][route_index],"offset_m":route["meta"]["lateral_offset_m"],"speed_mps":route["meta"]["cruise_speed_mps"],
                "truth_12s":{"observed_steps":observed,"full_horizon_observed":has_end,"goal_progress_m":progress,"work_kj":work,"events":truth_events,"schema_safe":safe12},
                "truth_full_route":{**full,"first_contact_s":first_time(contact_series),"first_rollover_s":first_time(rollover_series),"safe_first12_but_not_safe_full":safe12 is True and not full["schema_safe_goal"]}})
            routes.append(route);indices.append(index);equivalence.append(checks)
        assert len(set(indices)) == 15
        scene_report = {"scene_id":scene,"family":record["family"],"observation_sha256":sha(observation_path),"input_equivalence":equivalence,"routes":truth_rows,"models":{}}
        report["scenes"].append(scene_report)
        scene_contexts.append((scene_report, observation, routes, np.asarray(indices),float(case.get("goal_radius_m",3.))))

    for name, run_name, label, expected_step in models:
        run_dir = campaign / "runs/full_learning_h60_v3" / run_name
        checkpoint_path = run_dir / f"{label}.pt"
        model, checkpoint = load_rgbd_checkpoint(checkpoint_path, "cpu")
        assert checkpoint["step"] == expected_step and model.config.horizon == 60
        assert checkpoint["provenance"]["data"]["manifest.json"] == sha(pack / "manifest.json")
        for file in ("src/nedm/traverse/fdm_diverse_model.py", "src/nedm/traverse/fdm_model.py", "src/nedm/traverse/fdm_diverse_data.py"):
            check_hash(args.code_root / file, checkpoint["provenance"]["code"][file])
        saved_path = run_dir / ("val_predictions_normal.npz" if label == "best" else "val_predictions_last_normal.npz")
        saved = npz(saved_path)
        model_report = {"checkpoint":str(checkpoint_path),"checkpoint_sha256":sha(checkpoint_path),"step":checkpoint["step"],"arm":model.config.arm,
            "saved_forecast_sha256":sha(saved_path),"source_model_matches_training":True,"all_route_errors":[],"saved_vs_cpu_pack":[],"cpu_pack_vs_deployment":[],"selections":[]}
        export = {"pack_row":[],"deployment_trajectory":[],"deployment_work":[],"deployment_event_probability":[],"deployment_attitude":[]}
        for scene_report, observation, routes, indices, goal_radius in scene_contexts:
            with torch.inference_mode():
                allowed = ("history","commands","global_features","nominal_pose")
                batch = {key:torch.from_numpy(data[key][indices].astype(np.float32)) for key in allowed}
                batch["rgbd"] = torch.from_numpy(np.asarray(images[data["image_index"][indices]],np.float32))
                pack_output = model(batch)
                pack_prediction = {key:pack_output[key].numpy() for key in ("trajectory","work","attitude")}
                pack_prediction["event_probability"] = torch.sigmoid(pack_output["event_logits"]).numpy()
            scorer = RGBDReferenceScorer(model, observation["rgbd"], observation["history"], observation["pose"], observation["goal_xy"],batch_size=15,cost_config=RGBDCostConfig(goal_radius_m=goal_radius))
            deployment = scorer.predict(routes)
            saved_prediction = {key:saved[key][indices] for key in ("trajectory","work","attitude")}
            saved_prediction["event_probability"] = 1./(1.+np.exp(-np.clip(saved["event_logits"][indices],-80.,80.)))
            numeric = {"scene_id":scene_report["scene_id"],"arrays":{key:compare(saved_prediction[key],pack_prediction[key]) for key in deployment}}
            model_report["saved_vs_cpu_pack"].append(numeric)
            model_report["cpu_pack_vs_deployment"].append({"scene_id":scene_report["scene_id"],"arrays":{key:compare(pack_prediction[key],deployment[key]) for key in deployment}})
            truth_output = {key:data[key][indices].copy() for key in ("trajectory","work","attitude")}
            truth_output["event_probability"] = data["events"][indices].copy()
            truth_output["event_probability"][...,2:3] = data["bounded_motion"][indices]
            truth_complete = data["trajectory_mask"][indices,:,-1].all(1) & data["work_mask"][indices,:,-1].all(1)
            predictions = []
            for j,index in enumerate(indices):
                valid = data["trajectory_mask"][index,:,0].astype(bool)
                delta = np.linalg.norm(deployment["trajectory"][j,:,:2]-data["trajectory"][index,:,:2],axis=-1)
                pp = float(world_progress(deployment["trajectory"][j],observation["pose"],observation["goal_xy"])[-1])
                actual = scene_report["routes"][j]["truth_12s"]
                attitude_delta = deployment["attitude"][j]-data["attitude"][index]
                attitude_error = np.abs(np.degrees(np.arctan2(np.sin(attitude_delta),np.cos(attitude_delta))))
                row={"scene_id":scene_report["scene_id"],"route_index":j,"ADE_m":float(delta[valid].mean()) if valid.any() else None,
                     "FDE_m":float(delta[-1]) if valid[-1] else None,"predicted_goal_progress_m":pp,
                     "goal_progress_abs_error_m":abs(pp-actual["goal_progress_m"]) if actual["goal_progress_m"] is not None else None,
                     "predicted_work_kj":float(deployment["work"][j,-1,0]),
                     "work_abs_error_kj":abs(float(deployment["work"][j,-1,0])-actual["work_kj"]) if actual["work_kj"] is not None else None,
                     "attitude_MAE_deg":float(attitude_error[data["attitude_mask"][index].astype(bool)].mean()),
                     "event_probability_12s":deployment["event_probability"][j,-1].tolist(),"costs":{}}
                predictions.append(row);model_report["all_route_errors"].append(row)
            selections=[]
            for energy in (0.,.02):
                scorer.cost_config=replace(scorer.cost_config,energy_weight_s_per_kj=energy)
                costs=scorer.cost_breakdown(deployment)
                oracle=scorer.cost_breakdown(truth_output)
                oracle["cost"]=np.where(truth_complete,oracle["cost"],np.inf)
                order=rank(costs["unfiltered_cost"])
                for j,prediction in enumerate(predictions):
                    prediction["costs"][str(energy)]={key:value[j] for key,value in costs.items()}
                    prediction["costs"][str(energy)]["unfiltered_rank"]=int(order[j])
                    prediction["costs"][str(energy)]["truth12_oracle_cost"]=float(oracle["cost"][j])
                chosen=int(np.argmin(costs["cost"])) if np.isfinite(costs["cost"]).any() else None
                oracle_chosen=int(np.argmin(oracle["cost"])) if np.isfinite(oracle["cost"]).any() else None
                safe=[j for j,r in enumerate(scene_report["routes"]) if r["truth_full_route"]["schema_safe_goal"]]
                selection={"scene_id":scene_report["scene_id"],"energy_weight_s_per_kj":energy,"cost_config":asdict(scorer.cost_config),
                    "selected_route":chosen,"allowed_routes":int(np.isfinite(costs["cost"]).sum()),"unfiltered_winner":int(np.argmin(costs["unfiltered_cost"])),
                    "selected_full_safe":scene_report["routes"][chosen]["truth_full_route"]["schema_safe_goal"] if chosen is not None else None,
                    "selected_truth12_safe":scene_report["routes"][chosen]["truth_12s"]["schema_safe"] if chosen is not None else None,
                    "best_full_safe_unfiltered_rank":int(min(order[j] for j in safe)),"full_safe_routes_allowed":sum(bool(costs["allowed_by_predicted_risk"][j]) for j in safe),
                    "truth12_oracle_selected_route":oracle_chosen,"truth12_oracle_full_safe":scene_report["routes"][oracle_chosen]["truth_full_route"]["schema_safe_goal"] if oracle_chosen is not None else None,
                    "truth12_oracle_complete_candidates":int(truth_complete.sum())}
                selections.append(selection);model_report["selections"].append(selection)
            scene_report["models"][name]={"predictions":predictions,"selections":selections}
            export["pack_row"].append(indices)
            for key in deployment:export["deployment_"+key].append(deployment[key])
        model_report["errors_90"]=summary_errors(model_report["all_route_errors"])
        model_report["selection_summary"]={str(energy):{"selected":sum(s["selected_route"] is not None for s in model_report["selections"] if s["energy_weight_s_per_kj"]==energy),
            "safe_full_goals":sum(s["selected_full_safe"] is True for s in model_report["selections"] if s["energy_weight_s_per_kj"]==energy),
            "truth12_oracle_safe_full_goals":sum(s["truth12_oracle_full_safe"] is True for s in model_report["selections"] if s["energy_weight_s_per_kj"]==energy)} for energy in (0.,.02)}
        del model_report["all_route_errors"]
        report["models"][name]=model_report
        np.savez_compressed(args.out/f"{name}_deployment_predictions_90.npz",**{key:np.concatenate(value) for key,value in export.items()})
        print(json.dumps(clean({"model":name,"errors":model_report["errors_90"],"selections":model_report["selection_summary"]})),flush=True)
    all_rows=[row for scene in report["scenes"] for row in scene["routes"]]
    report["counts"]={"scenes":6,"routes":len(all_rows),"full_safe_goals":sum(row["truth_full_route"]["schema_safe_goal"] for row in all_rows),
        "safe_first12":sum(row["truth_12s"]["schema_safe"] is True for row in all_rows),
        "safe_first12_but_not_safe_full":sum(row["truth_full_route"]["safe_first12_but_not_safe_full"] for row in all_rows)}
    report["wall_s"]=time.perf_counter()-started
    dump(args.out/"diagnostic.json",report)
    lines=["# Initial fixed-reference diagnostic", "", "All 90 original references on six validation scenes. CPU inference uses the exact frozen online scorer and measured float32 initial RGB-D. This is a fixed-library diagnostic, not an MPPI rollout or model-selection result.", "",
        f"Measured safe routes: {report['counts']['safe_first12']}/90 over the first 12 seconds; {report['counts']['full_safe_goals']}/90 over the full traversal. {report['counts']['safe_first12_but_not_safe_full']} references are safe for 12 seconds but fail the full-route safety/completion criterion.", "",
        "| Model | ADE / FDE, m | Progress MAE, m | Work MAE, kJ | Chosen / safe full goals (energy 0) | Chosen / safe full goals (energy .02) |", "|---|---:|---:|---:|---:|---:|"]
    for name,model in report["models"].items():
        e=model["errors_90"];a=model["selection_summary"]["0.0"];b=model["selection_summary"]["0.02"]
        lines.append(f"| {name} | {e['ADE_m']:.3f} / {e['FDE_m']:.3f} | {e['goal_progress_MAE_m']:.3f} | {e['work_MAE_kj']:.3f} | {a['selected']} / {a['safe_full_goals']} | {b['selected']} / {b['safe_full_goals']} |")
    lines += ["", "The oracle below substitutes measured first-12-second trajectories, work, event indicators and endpoint attitudes into the unchanged planning cost. It reveals the cost heuristic's full-route extrapolation limit even with perfect short-horizon forecasts. It is diagnostic truth access, never a model input or deployed planner.", "",
        "| Scene | Energy | RGB-D best winner / safe full | Best actual safe route rank | Actual safe routes passing model risk | Truth-12s oracle winner / safe full |", "|---|---:|---|---:|---:|---|"]
    for selection in report["models"]["rgbd_best4000"]["selections"]:
        s=selection
        lines.append(f"| {s['scene_id']} | {s['energy_weight_s_per_kj']} | {s['selected_route']} / {s['selected_full_safe']} | {s['best_full_safe_unfiltered_rank']} | {s['full_safe_routes_allowed']} | {s['truth12_oracle_selected_route']} / {s['truth12_oracle_full_safe']} |")
    lines += ["", "`diagnostic.json` retains every route, both cost modes and three checkpoints, measured full-route consequences, censoring, numerical input comparisons, CPU-versus-saved forecast comparisons, checkpoint/source hashes and complete cost breakdowns. Predictions are exported in three small 90-route NPZ files. Engine-interface work is mechanical work, not fuel consumption.", ""]
    (args.out/"diagnostic.md").write_text("\n".join(lines))
    dump(args.out/"complete.json",{"state":"complete","routes":90,"models":3,"diagnostic_sha256":sha(args.out/"diagnostic.json"),"wall_s":report["wall_s"]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign",type=Path,required=True)
    parser.add_argument("--code-root",type=Path,required=True)
    parser.add_argument("--source-manifest-sha256",required=True)
    parser.add_argument("--full-route-support",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--threads",type=int,default=8)
    run(parser.parse_args())


if __name__=="__main__":
    main()
