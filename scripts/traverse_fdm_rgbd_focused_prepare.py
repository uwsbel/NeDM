#!/usr/bin/env python3
"""Pack predeclared focused RGB-D siblings, preserving scene groups and causality.

This is focus-only preparation: legacy low-net-progress labels remain in events,
and a separately named sustained-stall target permits an explicit trainer choice.
No model training, terrain map, obstacle geometry, or protected-test input is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from nedm.traverse.fdm_data import (DT, OUTPUT_OFFSETS, HISTORY_FIELDS, EVENT_NAMES,
    build_history, build_targets, tracked_route_indices, parking_mask)
from nedm.traverse.fdm_rgbd_data import (COMMAND_FIELDS, GLOBAL_FIELDS,
    build_command_features, rgbd_from_arrays)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def sustained_targets(state, actions, parked, anchor):
    """Any fully future 40-interval low-speed/effort run, with no startup grace.

    Flags describe [state_i,state_{i+1}); positives survive an unobserved tail,
    while a negative needs the entire prefix. Parking interrupts a run.
    """
    n = len(actions)
    slow = (np.abs(state[:n, 0]) < .3) & (actions[:, 1] > .3) & ~np.asarray(parked[:n], bool)
    event = np.zeros((20, 1), np.float32)
    mask = np.zeros_like(event)
    for j, offset in enumerate(OUTPUT_OFFSETS):
        end = anchor+int(offset)
        run, longest = 0, 0
        for flag in slow[anchor:min(end, n)]:
            run = run+1 if flag else 0
            longest = max(longest, run)
        event[j, 0] = longest >= 40
        mask[j, 0] = offset >= 40 and (event[j, 0] or end <= n)
    return event, mask


def observed_onsets(ep):
    """Diagnostic completed-interval onsets; never passed to the network."""
    d, first_low, first_sustained, run = ep["raw"], None, None, 0
    for i in range(ep["n"]):
        slow = abs(float(d["state"][i, 0])) < .3 and d["action"][i, 1] > .3 and not ep["parked"][i]
        run = run+1 if slow else 0
        if run >= 40 and first_sustained is None:
            first_sustained = i+1
        end = i+1
        if end >= 40 and first_low is None:
            begin = end-40
            if (not ep["parked"][begin:end+1].any()
                    and np.linalg.norm(ep["poses"][end, :2]-ep["poses"][begin, :2]) < .3
                    and d["action"][begin:end, 1].mean() > .3):
                first_low = end
    return first_low, first_sustained


def load_rollout(path, route):
    with np.load(path/"trajectory.npz", allow_pickle=False) as f:
        d = {k: f[k].copy() for k in f.files}
    n = len(d["action"])
    if not np.isclose(float(d["dt_s"]), DT):
        raise ValueError("Expected 20 Hz telemetry")
    for k, shape in (("state", (n, 17)), ("action", (n, 3)), ("pose", (n, 3)),
                     ("terminal_state", (17,)), ("terminal_pose", (3,)), ("parked", (n,)),
                     ("positive_work_kj_per_interval", (n,)), ("contact_n", (n,))):
        if k not in d or d[k].shape != shape or not np.isfinite(d[k]).all():
            raise ValueError(f"Bad or missing {k} in {path}")
    if n < 1 or np.any(d["positive_work_kj_per_interval"] < -1e-7):
        raise ValueError("Invalid interval telemetry")
    state = np.concatenate((d["state"], d["terminal_state"][None]))
    poses = np.concatenate((d["pose"], d["terminal_pose"][None]))
    parked_last = bool(d["terminal_parked"]) if "terminal_parked" in d else bool(parking_mask(route, poses)[-1])
    parked = np.r_[d["parked"].astype(bool), parked_last]
    outcome = json.loads((path/"outcome.json").read_text())
    # Per-interval maxima are uncapped, unlike the legacy diagnostic event list.
    contacts = [[int(i), 0, float(v)] for i, v in enumerate(d["contact_n"]) if v > 1.]
    meta = {"status": outcome["status"], "contact": {"events": contacts}}
    with np.load(path/"rgbd_frames.npz", allow_pickle=False) as f:
        anchors = f["anchor"].copy().astype(np.int64)
        camera = json.loads(str(f["camera_json"].item()))
    cm = json.loads((path/"collection_meta.json").read_text())
    if not np.array_equal(anchors, np.asarray(cm["image_frame_indices"], np.int64)):
        raise ValueError("Image anchors differ from collection provenance")
    if cm["frames"] != n or not np.isclose(cm["frame_dt_s"], DT):
        raise ValueError("Telemetry count/timing differs from collection provenance")
    if len(anchors) == 0 or anchors[0] != 0 or np.any(np.diff(anchors) <= 0) or np.any(anchors >= n):
        raise ValueError("Sparse current images must start at frame0 and match measured states")
    if not np.isclose(camera["depth_ray_scale"], 1.):
        raise ValueError("Focused Vulkan camera must explicitly use ray scale1")
    return dict(raw=d, n=n, state=state, poses=poses, parked=parked, meta=meta,
                outcome=outcome, anchors=anchors, camera=camera)


def focused_targets(ep, anchor):
    n, d = ep["n"], ep["raw"]
    # Measured terminal state/pose provide the right endpoint of the final interval.
    actions = np.concatenate((d["action"], np.zeros((1, 3), np.float32)))
    power = np.r_[d["power_kw"], 0.]
    result = build_targets(ep["poses"], ep["state"], actions, power, ep["meta"], anchor, ep["parked"])
    prefix_work = np.r_[0., np.cumsum(d["positive_work_kj_per_interval"], dtype=np.float64)]
    for j, offset in enumerate(OUTPUT_OFFSETS):
        end = anchor+int(offset)
        result["work"][j, 0] = prefix_work[min(end, n)]-prefix_work[anchor] if end <= n else 0.
        if end > n:
            # Known event positives remain valid; no negative beyond the final interval.
            result["event_mask"][j, :2] = result["events"][j, :2] > 0
    result["sustained_stall"], result["sustained_stall_mask"] = sustained_targets(
        ep["state"], d["action"], ep["parked"], anchor)
    return result


def declared_episodes(manifests, raw_root, allow_incomplete=False):
    result, missing, scene_splits = [], [], {}
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        for record in manifest["records"]:
            scene, split = record["scene_id"], record["split"]
            if split not in ("train", "val") or scene in ("visible_rock_v1", "hill0_f111_near"):
                raise ValueError("Only declared fresh train/val scenes are permitted")
            if scene in scene_splits:
                raise ValueError("Duplicate scene declaration: "+scene)
            scene_splits[scene] = split
            case_path = manifest_path.parent/record["case"]
            if "case_sha256" in record and sha(case_path) != record["case_sha256"]:
                raise ValueError("Predeclared case changed: "+scene)
            case = json.loads(case_path.read_text())
            if case["split"] != split or case["id"] != scene:
                raise ValueError("Scene split/id mismatch")
            for j, rel in enumerate(record["routes"]):
                route_path = manifest_path.parent/rel
                if "route_sha256" in record and sha(route_path) != record["route_sha256"][j]:
                    raise ValueError("Predeclared reference changed")
                path = raw_root/scene/route_path.stem
                required = ("trajectory.npz", "rgbd_frames.npz", "outcome.json", "collection_meta.json")
                if not all((path/k).exists() for k in required):
                    missing.append(str(path))
                    continue
                cm = json.loads((path/"collection_meta.json").read_text())
                if cm["case_id"] != scene or cm["split"] != split:
                    raise ValueError("Collected source crosses scene split")
                if cm["case_sha256"] != sha(case_path) or cm["route_sha256"] != sha(route_path):
                    raise ValueError("Collected reference or scene differs from predeclaration")
                route = json.loads(route_path.read_text())
                for key in ("waypoints", "stations", "speeds", "headings"):
                    if not np.array_equal(np.asarray(cm["route"][key]), np.asarray(route[key])):
                        raise ValueError("Collected command values differ from predeclared route")
                result.append(dict(id=f"{scene}__{route_path.stem}", scene_id=scene, split=split,
                    path=path, route=route, case_path=case_path,
                    route_path=route_path, source_domain="focused_v1_standard_chrono_pid_vulkan"))
    if missing and not allow_incomplete:
        raise FileNotFoundError(f"Incomplete cohort: {len(missing)} missing rollouts; first {missing[0]}")
    return result, missing, scene_splits


def event_support(arrays, episodes):
    ei, anchors = arrays["episode_index"], arrays["anchor"]
    result = {}
    named = [(name, arrays["events"][:, :, i], arrays["event_mask"][:, :, i]) for i, name in enumerate(EVENT_NAMES)]
    named.append(("sustained_stall", arrays["sustained_stall"][:, :, 0], arrays["sustained_stall_mask"][:, :, 0]))
    for name, labels, mask in named:
        positive = ((labels > 0) & (mask > 0)).any(1)
        result[name] = {}
        for group, choose in (("all", np.ones(len(ei), bool)), ("anchor0", anchors == 0),
                              ("anchor_lt2s", anchors < 40), ("anchor_ge2s", anchors >= 40)):
            selected = positive & choose
            positive_ids = np.unique(ei[selected])
            result[name][group] = {"windows": int(choose.sum()), "positive_windows": int(selected.sum()),
                "positive_episodes": len(positive_ids), "positive_scenes": len({episodes[int(i)]["scene_id"] for i in positive_ids}),
                "positive_horizon_labels": int(((labels > 0) & (mask > 0))[choose].sum()),
                "valid_horizon_labels": int((mask[choose] > 0).sum())}
        first_contact = np.array([float("inf") if e["first_contact_frame"] is None else e["first_contact_frame"] for e in episodes])
        before = positive & (anchors < first_contact[ei])
        result[name]["before_first_contact"] = {"positive_windows": int(before.sum()),
            "positive_episodes": len(np.unique(ei[before]))}
        if name in ("low_progress", "sustained_stall"):
            onset_key = "first_observed_low_progress_frame" if name == "low_progress" else "first_observed_sustained_stall_frame"
            onset = np.array([float("inf") if e[onset_key] is None else e[onset_key] for e in episodes])
            before = positive & (anchors < onset[ei])
            result[name]["before_observed_completed2s_onset"] = {"positive_windows": int(before.sum()),
                "positive_episodes": len(np.unique(ei[before]))}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort-manifest", type=Path, action="append", required=True)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--audit-only", action="store_true", help="Count completed sources; never seal an incomplete pack")
    args = ap.parse_args()
    if not args.audit_only and args.out is None:
        ap.error("--out is required unless --audit-only")
    records, missing, scene_splits = declared_episodes(args.cohort_manifest, args.raw_root, args.audit_only)
    if not records:
        print(json.dumps({"completed_episodes": 0, "missing_episodes": len(missing)}))
        return
    manifest = {"schema": 2, "cohort_state": "complete" if not missing else "incomplete_audit_only",
        "source_domain": "focused_v1_standard_chrono_pid_vulkan", "training_scope": "focus-only; no legacy rows or silent label substitution",
        "camera": {"width": 256, "height": 256, "hfov_deg": 47., "cam_height_m": 100.,
                   "depth_ray_scale": 1., "depth_convention": "Euclidean_ray_range_m", "backend": "Vulkan"},
        "controller_domain": "Standard ChPathFollowerDriver, steering(.8,0,0), speed(.6,.05,0), lookahead5m, advance every physics substep, initialized before0.8s settle; geometric reference z uses terrainheight+.5m inside simulator, never model input",
        "cohort_manifests": {str(p.resolve()): sha(p) for p in args.cohort_manifest},
        "split_rule": "Predeclared scene-level split, all sibling routes kept together; no protected-test or demo scenes",
        "scene_splits": scene_splits, "missing_episodes": missing,
        "selection": "Every predeclared sibling, including failures, safe routes and moving routes; no outcome filtering",
        "reference_confound": "Geometric routes use declared uniform4/6m/s cruise plus distance-to-end braking; no terrain-authored speed limits or obstacle-conditioned route filtering",
        "validation_distribution": "Designed failure-focused development cohort; prevalence is not natural deployment calibration",
        "commands_fields": COMMAND_FIELDS, "global_fields": GLOBAL_FIELDS, "history_fields": HISTORY_FIELDS,
        "event_names": list(EVENT_NAMES), "head2_default": "events[...,2] remains low_progress",
        "extra_target": "sustained_stall with sustained_stall_mask; trainer must explicitly select and archive target name",
        "sustained_stall_definition": "Any entirely future contiguous40 intervals (2s at20Hz) with |body_vx|<0.3m/s, throttle>0.3 and no parking; interval flags use left-endpoint measured state and applied control. No global startup grace. Prefixes under2s masked. Positive remains valid after truncation; negative needs complete prefix.",
        "low_progress_definition": "Net displacement<0.15m/s times prefix duration and mean throttle>0.3, prefix>=2s, measured endpoint, no route-end parking; retained unchanged",
        "contact_definition": "Uncapped per-interval asset contact max>1N, interval[i,i+1); known positives survive truncation, missing negatives censored",
        "work_definition": "Cumulative positive shaft work using exact per-physics-substep integral saved for each20Hz interval; kJ",
        "startup_padding": "Unrecorded history repeats first measured state/pose and synthetic [steer0,throttle0,brake1]; not measured settle telemetry. Available history count is min(16,1+anchor).",
        "observation": "Actual sparse current RGB-D only; no future image, terrain/BMP, asset geometry, or future measured station input",
        "image_encoding": {"shape": [4, 128, 128], "dtype": "float16", "rgb": "box-resized uint8/255", "depth": "observed Euclidean ray depth -> RGB-registered elevation; fixed clip(z/10,-1,1), invalid=-2"},
        "horizon_s": 4., "output_dt_s": .2, "history_frames": 16,
        "source_domain_codes": {"1": "focused_v1_standard_chrono_pid_vulkan"},
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), ROOT/"src/nedm/traverse/fdm_data.py", ROOT/"src/nedm/traverse/fdm_rgbd_data.py")},
        "splits": {}}
    if not args.audit_only:
        if args.out.resolve().is_relative_to(Path("/home/harry/NeDM")):
            raise ValueError("Output must be outside the active source checkout")
        args.out.mkdir(parents=True, exist_ok=False)
    normalization = {}
    for split in ("train", "val"):
        selected = [r for r in records if r["split"] == split]
        if not selected:
            if args.audit_only:
                continue
            raise ValueError("Both train and validation require complete scenes")
        arrays, episodes, image_chunks = {}, [], []
        scene_ids = sorted({r["scene_id"] for r in selected})
        for epi, rec in enumerate(selected):
            ep = load_rollout(rec["path"], rec["route"])
            cam = ep["camera"]
            hfov_deg = float(cam["hfov_deg"]) if "hfov_deg" in cam else float(np.rad2deg(cam["hfov_rad"]))
            if not np.isclose(hfov_deg, 47.) or not np.isclose(cam["cam_height_m"], 100.):
                raise ValueError("Focused camera deviates from declared calibration")
            indices = tracked_route_indices(rec["route"], ep["poses"])
            frames = None if args.audit_only else np.load(rec["path"]/"rgbd_frames.npz", allow_pickle=False)
            if frames is not None:
                rgb, depth = frames["rgb"], frames["depth_m"]
                if rgb.shape != (len(ep["anchors"]), 256, 256, 3) or depth.shape != (len(ep["anchors"]), 256, 256):
                    raise ValueError("Actual RGB-D frame shapes differ from camera contract")
            for image_index, anchor in enumerate(ep["anchors"]):
                anchor = int(anchor)
                row = focused_targets(ep, anchor)
                row.update(build_command_features(rec["route"], ep["poses"][anchor],
                    station=rec["route"]["stations"][indices[anchor]], elapsed_s=anchor*DT))
                row.update(history=build_history(ep["state"], ep["raw"]["action"], ep["poses"], anchor),
                    episode_index=np.int64(epi), anchor=np.int64(anchor), source_domain=np.int64(1),
                    scene_index=np.int64(scene_ids.index(rec["scene_id"])))
                for k, value in row.items():
                    arrays.setdefault(k, []).append(value)
                if frames is not None:
                    image_chunks.append(rgbd_from_arrays(rgb[image_index], depth_m=depth[image_index], camera=ep["camera"]).astype(np.float16))
            if frames is not None:
                frames.close()
            contact_frames = np.flatnonzero(ep["raw"]["contact_n"] > 1.)
            first_low, first_sustained = observed_onsets(ep)
            episode_record = {k: rec[k] for k in ("id", "scene_id", "split", "source_domain")}
            episode_record.update(source=str(rec["path"].resolve()), frames=ep["n"], windows=len(ep["anchors"]),
                anchors=ep["anchors"].tolist(), camera=ep["camera"], status=ep["outcome"]["status"],
                first_contact_frame=int(contact_frames[0]) if len(contact_frames) else None,
                first_observed_low_progress_frame=first_low, first_observed_sustained_stall_frame=first_sustained,
                source_sha256={name: sha(rec["path"]/name) for name in ("trajectory.npz", "rgbd_frames.npz", "outcome.json", "collection_meta.json")},
                case_sha256=sha(rec["case_path"]), route_sha256=sha(rec["route_path"]))
            episodes.append(episode_record)
        arrays = {k: np.stack(v) for k, v in arrays.items()}
        if any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("Nonfinite low-dimensional pack")
        support = event_support(arrays, episodes)
        manifest["splits"][split] = {"episodes": len(episodes), "scenes": len(scene_ids), "windows": len(arrays["anchor"]),
            "scene_ids": scene_ids, "event_support": support, "shapes": {k: list(v.shape) for k, v in arrays.items()}}
        if args.audit_only:
            continue
        images = np.stack(image_chunks)
        if not np.isfinite(images).all():
            raise ValueError("Nonfinite RGB-D")
        image_path, target = args.out/(split+"_rgbd.npy"), args.out/(split+".npz")
        np.save(image_path, images)
        np.savez(target, **arrays)
        dump(args.out/(split+"_episodes.json"), episodes)
        manifest["splits"][split].update(sha256=sha(target), rgbd_file=image_path.name,
            rgbd_shape=list(images.shape), rgbd_sha256=sha(image_path))
        if split == "train":
            for k in ("history", "commands", "global_features"):
                a = arrays[k].reshape(-1, arrays[k].shape[-1])
                normalization[k] = {"mean": a.mean(0, dtype=np.float64).tolist(),
                    "std": np.maximum(a.std(0, dtype=np.float64), .01).tolist()}
        print(json.dumps({"split": split, "episodes": len(episodes), "windows": len(images)}), flush=True)
    if args.audit_only:
        print(json.dumps(manifest, indent=2))
    else:
        dump(args.out/"normalization.json", normalization)
        manifest["normalization_sha256"] = sha(args.out/"normalization.json")
        dump(args.out/"manifest.json", manifest)


if __name__ == "__main__":
    main()
