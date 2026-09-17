#!/usr/bin/env python3
"""Join frozen global RGB-D with diverse headless Chrono trajectories and labels.

Default access is train/validation only. Test packing is a separate invocation
requiring --sealed-test and previously frozen training normalization. No source
is selected by its outcome; every declared route and uniform one-second anchor
is retained, including censored late windows.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nedm.traverse.fdm_data import HISTORY_FIELDS, STATE_FIELDS, build_history, tracked_route_indices
from nedm.traverse.fdm_diverse_data import HORIZON, OUTPUT_DT, RECORD_DT, build_command_features, encode_global_rgbd
from nedm.traverse.fdm_diverse_targets import prepare_episode_labels, build_diverse_targets
from nedm.traverse.fdm_rgbd_data import COMMAND_FIELDS, GLOBAL_FIELDS
from check_traverse_fdm_rich_telemetry import verify as verify_rich

REQUIRED_RAW = ("trajectory.npz", "anchor_state.npz", "outcome.json", "collection_meta.json",
                "simulation_provenance.json", "rich_telemetry.npz", "rich_intervals.npz", "rich_telemetry.json")
REQUIRED_OBS = ("observation.npz", "observation.json", "simulation_provenance.json")
EVENT_SCHEMA = "fdm_diverse_events_v1_asset_or_chassis"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def require_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Frozen bytes changed: {path}")


def validate_provenance(provenance, record, *, source_root, protocol=None):
    for key in ("case_sha256", "arena_meta_sha256", "arena_bmp_sha256"):
        if provenance.get(key) != record[key]:
            raise ValueError(f"Collection {key} differs from scene declaration: {record['scene_id']}")
    for key in ("source_sha256", "runtime_sha256"):
        values = provenance.get(key)
        if not isinstance(values, dict) or not values or any(len(str(digest)) != 64 for digest in values.values()):
            raise ValueError(f"Missing complete {key} provenance")
        if protocol is not None and key in protocol and values != protocol[key]:
            raise ValueError(f"Collected {key} differs from frozen collection protocol")
    for filename, digest in provenance["source_sha256"].items():
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Source archive names must be repository-relative")
        require_hash(source_root / relative, digest)
    if not np.isclose(provenance["physics_dt_s"], .002, rtol=0., atol=1e-12):
        raise ValueError("Main cohort requires declared 500 Hz physics")


def verify_anchor(raw, snapshot, case):
    for key in ("state", "pose", "history", "goal_xy", "goal_radius_m"):
        if key not in raw or key not in snapshot:
            raise ValueError(f"Missing settled-anchor {key}")
        tolerance = 2e-6 if key == "pose" else 1e-5
        np.testing.assert_allclose(raw[key], snapshot[key], rtol=0., atol=tolerance, err_msg=f"Headless/rendered anchor mismatch: {key}")
    np.testing.assert_allclose(snapshot["goal_xy"], case["goal_xy"], rtol=0., atol=1e-5)
    np.testing.assert_allclose(snapshot["goal_radius_m"], case.get("goal_radius_m", 2.5), rtol=0., atol=1e-6)
    reconstructed = build_history(raw["state"][None], np.asarray([[0., 0., 1.]], np.float32), raw["pose"][None], 0)
    np.testing.assert_array_equal(raw["history"], reconstructed)


def declared_records(manifests, *, raw_root, observation_root, splits, source_root, arena_root=ROOT, protocol=None, audit_only=False):
    records, missing, all_splits, declared_hash_splits = [], [], {}, {}
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        for record in manifest["records"]:
            scene, split = record["scene_id"], record["split"]
            if scene in all_splits:
                raise ValueError(f"Duplicate scene declaration: {scene}")
            if split not in ("train", "val", "test"):
                raise ValueError("Unknown scene split")
            all_splits[scene] = split
            map_hash = record["arena_bmp_sha256"]
            if map_hash in declared_hash_splits and declared_hash_splits[map_hash] != split:
                raise ValueError("Identical physical heightmap crosses scene splits")
            declared_hash_splits[map_hash] = split
            # Do not open cases, observations or telemetry from excluded test.
            if split not in splits:
                continue
            case_path = manifest_path.parent / record["case"]
            require_hash(case_path, record["case_sha256"])
            case = json.loads(case_path.read_text())
            if case["id"] != scene or case["split"] != split:
                raise ValueError("Case declaration crosses scene split")
            arena = arena_root / record["arena"]
            require_hash(arena / "arena_meta.json", record["arena_meta_sha256"])
            arena_meta = json.loads((arena / "arena_meta.json").read_text())
            require_hash(arena / arena_meta["bmp"], record["arena_bmp_sha256"])
            observation_path = observation_root / scene
            if not all((observation_path / name).is_file() for name in REQUIRED_OBS):
                missing.append(str(observation_path))
                continue
            obs_meta = json.loads((observation_path / "observation.json").read_text())
            obs_prov = json.loads((observation_path / "simulation_provenance.json").read_text())
            validate_provenance(obs_prov, record, source_root=source_root, protocol=protocol)
            require_hash(observation_path / "observation.npz", obs_meta["observation_sha256"])
            if obs_meta["case_id"] != scene or obs_meta["camera"] != obs_prov["camera"]:
                raise ValueError("Observation identity/calibration mismatch")
            snapshot = read_npz(observation_path / "observation.npz")
            encoded = encode_global_rgbd(snapshot["rgb"], snapshot["depth_m"], obs_meta["camera"])
            np.testing.assert_array_equal(encoded, snapshot["rgbd"])
            if encoded.shape != (4, 512, 512):
                raise ValueError("Enriched main cohort requires the global 512-pixel sensor encoding")
            if not np.isclose(float(snapshot["elapsed_s"]), 0.):
                raise ValueError("Global scene snapshot must precede driving")
            for route_rel, route_hash in zip(record["routes"], record["route_sha256"], strict=True):
                route_path = manifest_path.parent / route_rel
                require_hash(route_path, route_hash)
                route = json.loads(route_path.read_text())
                raw_path = raw_root / scene / route_path.stem
                if not all((raw_path / name).is_file() for name in REQUIRED_RAW):
                    missing.append(str(raw_path))
                    continue
                meta = json.loads((raw_path / "collection_meta.json").read_text())
                prov = json.loads((raw_path / "simulation_provenance.json").read_text())
                validate_provenance(prov, record, source_root=source_root, protocol=protocol)
                for key in ("source_sha256", "runtime_sha256", "physics_dt_s", "camera", "driver", "terrain_texture_sha256"):
                    if prov.get(key) != obs_prov.get(key):
                        raise ValueError(f"Headless/snapshot {key} mismatch: {scene}")
                if (meta["case_id"], meta["split"], meta["case_sha256"], meta["route_sha256"]) != (scene, split, record["case_sha256"], route_hash):
                    raise ValueError("Collected reference or scene differs from frozen manifest")
                for key in ("waypoints", "stations", "speeds", "headings"):
                    np.testing.assert_array_equal(meta["route"][key], route[key])
                if meta["camera"] != obs_meta["camera"] or meta["driver"] != prov["driver"]:
                    raise ValueError("Collection camera/controller differs from provenance")
                anchor = read_npz(raw_path / "anchor_state.npz")
                verify_anchor(anchor, snapshot, case)
                rich_meta = json.loads((raw_path / "rich_telemetry.json").read_text())
                if rich_meta["case_sha256"] != record["case_sha256"] or rich_meta["observer_sha256"] != prov["source_sha256"]["src/nedm/traverse/fdm_rich_telemetry.py"]:
                    raise ValueError("Rich telemetry case/source hash mismatch")
                records.append({"id": scene+"__"+route_path.stem, "scene_id": scene, "split": split,
                    "family": record.get("family"), "evaluation_stratum": record.get("evaluation_stratum"),
                    "case": case, "route": route, "path": raw_path, "case_path": case_path, "route_path": route_path,
                    "observation_path": observation_path, "snapshot": snapshot, "camera": obs_meta["camera"],
                    "provenance": prov, "map_sha256": map_hash})
    if missing and not audit_only:
        raise FileNotFoundError(f"Incomplete cohort: {len(missing)} missing sources; first {missing[0]}")
    return records, missing, all_splits


def load_episode(record):
    path = record["path"]
    verify_rich(path, trajectory=path / "trajectory.npz", require_solver_steps=True)
    raw = read_npz(path / "trajectory.npz")
    samples = read_npz(path / "rich_telemetry.npz")
    intervals = read_npz(path / "rich_intervals.npz")
    n = len(raw["action"])
    if not np.isclose(raw["dt_s"], RECORD_DT) or raw["state_fields"].tolist() != list(STATE_FIELDS):
        raise ValueError("Unexpected core state or recording-period schema")
    if not np.allclose(intervals["duration_s"], RECORD_DT, rtol=0., atol=1e-7):
        raise ValueError("Recorded intervals depart from the fixed 20 Hz label contract")
    meta = json.loads((path / "collection_meta.json").read_text())
    if meta["frames"] != n or not np.isclose(meta["frame_dt_s"], RECORD_DT):
        raise ValueError("Frame count/timing differs from collection provenance")
    poses = np.vstack([raw["pose"], raw["terminal_pose"]])
    states = np.vstack([raw["state"], raw["terminal_state"]])
    parked = np.r_[raw["parked"].astype(bool), bool(raw["terminal_parked"])]
    anchor = read_npz(path / "anchor_state.npz")
    np.testing.assert_array_equal(states[0], anchor["state"])
    np.testing.assert_array_equal(poses[0], anchor["pose"])
    # Rich endpoint attitude is double precision; state17 uses float32.
    attitude = np.column_stack([samples[f"{axis}_rad"] for axis in ("roll", "pitch")])
    np.testing.assert_allclose(attitude, states[:, 2:4], rtol=1e-6, atol=1e-7)
    case = record["case"]
    ep = prepare_episode_labels(poses, states, raw["action"], parked, intervals,
        goal_xy=np.asarray(case["goal_xy"]), goal_radius_m=case.get("goal_radius_m", 2.5))
    ep.update(raw=raw, attitude=attitude, samples=samples, intervals=intervals)
    return ep


def event_support(arrays, episodes):
    support = {}
    pairs = [(name, arrays["events"][:, :, j], arrays["event_mask"][:, :, j]) for j, name in enumerate(("contact", "rollover", "low_progress"))]
    pairs += [(key, arrays[key][:, :, 0], arrays[key+"_mask"][:, :, 0]) for key in ("bounded_motion", "sustained_stall", "asset_contact", "chassis_contact")]
    for name, labels, mask in pairs:
        positive = ((labels > 0) & (mask > 0)).any(1)
        support[name] = {}
        for subset, choose in (("all", np.ones(len(positive), bool)), ("anchor0", arrays["anchor"] == 0)):
            indices = np.unique(arrays["episode_index"][positive & choose])
            support[name][subset] = {"windows": int(choose.sum()), "positive_windows": int((positive & choose).sum()),
                "positive_episodes": len(indices), "positive_scenes": len({episodes[int(i)]["scene_id"] for i in indices}),
                "positive_horizon_labels": int(((labels > 0) & (mask > 0))[choose].sum()), "valid_horizon_labels": int((mask[choose] > 0).sum())}
    return support


def prepare_one_record(payload):
    """Independent route work, returning compact chunks in declaration order."""
    epi, record, scene_number, horizon, output_dt, anchor_stride = payload
    ep = load_episode(record)
    route_indices = tracked_route_indices(record["route"], ep["poses"])
    anchors = np.arange(0, ep["n"], anchor_stride, dtype=np.int64)
    arrays = {}
    for anchor in anchors:
        anchor = int(anchor)
        row = build_diverse_targets(ep, anchor, horizon=horizon, output_dt=output_dt, attitude=ep["attitude"])
        row.update(build_command_features(record["route"], ep["poses"][anchor], station=record["route"]["stations"][route_indices[anchor]],
            elapsed_s=anchor*RECORD_DT, horizon=horizon, output_dt=output_dt))
        row.update(history=build_history(ep["states"], ep["actions"], ep["poses"], anchor),
            episode_index=np.int64(epi), anchor=np.int64(anchor), scene_index=np.int64(scene_number),
            image_index=np.int64(scene_number), source_domain=np.int64(3))
        for key, value in row.items():
            arrays.setdefault(key, []).append(value)
    episode = {key: record[key] for key in ("id", "scene_id", "split", "family", "evaluation_stratum", "map_sha256")}
    episode.update(source=str(record["path"].resolve()), frames=ep["n"], windows=len(anchors), anchors=anchors.tolist(),
        source_sha256={name: sha(record["path"] / name) for name in REQUIRED_RAW},
        observation_source=str(record["observation_path"].resolve()), observation_sha256={name: sha(record["observation_path"] / name) for name in REQUIRED_OBS},
        case_sha256=sha(record["case_path"]), route_sha256=sha(record["route_path"]),
        status=json.loads((record["path"] / "outcome.json").read_text())["status"],
        bounded_motion_ever=bool(ep["bounded_endpoints"].any()), sustained_stall_ever=bool(ep["sustained_endpoints"].any()))
    return {key: np.stack(values) for key, values in arrays.items()}, episode


def bounded_ordered_results(payloads, workers):
    if workers == 1:
        yield from map(prepare_one_record, payloads)
        return
    # At most `workers` episodes are submitted at once. This avoids an eager
    # process-map queue retaining an entire cohort's returned arrays in RAM.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        iterator, pending = iter(payloads), deque()
        for _ in range(workers):
            item = next(iterator, None)
            if item is not None:
                pending.append(pool.submit(prepare_one_record, item))
        while pending:
            yield pending.popleft().result()
            item = next(iterator, None)
            if item is not None:
                pending.append(pool.submit(prepare_one_record, item))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-manifest", type=Path, action="append", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--observation-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--output-dt", type=float, default=OUTPUT_DT)
    parser.add_argument("--anchor-stride", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1, help="Bounded independent route preparation processes; deterministic declaration order")
    parser.add_argument("--source-snapshot-root", type=Path, default=ROOT)
    parser.add_argument("--arena-root", type=Path, default=ROOT,
                        help="Repository-layout archive containing frozen arena BMP/meta bytes")
    parser.add_argument("--collection-protocol", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--sealed-test", action="store_true")
    parser.add_argument("--normalization-from", type=Path)
    args = parser.parse_args()
    if args.horizon < 1 or args.anchor_stride < 1 or args.workers < 1:
        parser.error("Horizon and anchor stride must be positive")
    if not args.audit_only and args.out is None:
        parser.error("--out is required to seal a pack")
    if args.sealed_test != (args.normalization_from is not None):
        parser.error("--sealed-test requires --normalization-from TRAIN_PACK; train/val computes training normalization only")
    if args.out and (args.out.exists() or args.out.resolve().is_relative_to(Path("/home/harry/NeDM"))):
        raise ValueError("Use a new isolated output; existing packs and active checkout are never overwritten")
    protocol = json.loads(args.collection_protocol.read_text()) if args.collection_protocol else None
    splits = ("test",) if args.sealed_test else ("train", "val")
    records, missing, scene_splits = declared_records(args.cohort_manifest, raw_root=args.raw_root,
        observation_root=args.observation_root, splits=splits, source_root=args.source_snapshot_root,
        arena_root=args.arena_root, protocol=protocol, audit_only=args.audit_only)
    if args.audit_only:
        report = {"audit_only": True, "selected_splits": list(splits), "completed_episodes": len(records),
                  "missing_sources": missing, "scene_counts": {split: len({r['scene_id'] for r in records if r['split'] == split}) for split in splits}}
        print(json.dumps(report, indent=2))
        return
    if not all(any(record["split"] == split for record in records) for split in splits):
        raise ValueError("Every selected split needs a complete scene cohort")
    observed_sources = records[0]["provenance"]["source_sha256"]
    observed_runtime = records[0]["provenance"]["runtime_sha256"]
    camera = records[0]["camera"]
    for record in records:
        if record["provenance"]["source_sha256"] != observed_sources or record["provenance"]["runtime_sha256"] != observed_runtime or record["camera"] != camera:
            raise ValueError("A pack cannot silently mix collection implementations, runtimes or cameras")
    normalization = {}
    normalized_from = None
    if args.sealed_test:
        old_manifest = json.loads((args.normalization_from / "manifest.json").read_text())
        if "train" not in old_manifest["splits"] or "test" in old_manifest["splits"]:
            raise ValueError("Normalization source must be a train/validation pack")
        require_hash(args.normalization_from / "normalization.json", old_manifest["normalization_sha256"])
        normalization = json.loads((args.normalization_from / "normalization.json").read_text())
        normalized_from = {"path": str(args.normalization_from.resolve()), "manifest_sha256": sha(args.normalization_from / "manifest.json")}
        if set(old_manifest["scene_splits"]) & {r["scene_id"] for r in records}:
            # Old pack scene_splits contains selected splits only, not held-out
            # declaration names, so any overlap here is actual leakage.
            raise ValueError("Test scenes overlap the normalization/training pack")
    args.out.mkdir(parents=True, exist_ok=False)
    # Every imported repository module, including package initializers and the
    # telemetry checker dependency, belongs to preparation provenance.
    code_files = [Path(__file__)] + sorted({Path(module.__file__).resolve()
        for module in sys.modules.values() if getattr(module, "__file__", None)
        and Path(module.__file__).resolve().is_relative_to(ROOT)
        and Path(module.__file__).suffix == ".py" and Path(module.__file__).resolve() != Path(__file__).resolve()})
    manifest = {"schema": "fdm_diverse_pack_v1", "event_schema": EVENT_SCHEMA, "cohort_state": "complete",
        "bounded_motion_version": 2, "sustained_stall_version": 1,
        "cohort_manifests": {str(path.resolve()): sha(path) for path in args.cohort_manifest},
        "scene_splits": {scene: split for scene, split in scene_splits.items() if split in splits},
        "selection": "All declared sibling routes, every uniform anchor; no outcome filtering or event-conditioned oversampling",
        "split_rule": "Scene-disjoint; byte-identical heightmaps cannot cross splits. Test not opened by default invocation.",
        "anchor_stride_frames": args.anchor_stride, "frame_dt_s": RECORD_DT, "horizon_steps": args.horizon,
        "horizon_s": args.horizon*args.output_dt, "output_dt_s": args.output_dt, "history_fields": HISTORY_FIELDS,
        "command_fields": COMMAND_FIELDS, "global_fields": GLOBAL_FIELDS, "event_names": ["contact", "rollover", "low_progress"],
        "extra_event_targets": ["bounded_motion", "sustained_stall", "asset_contact", "chassis_contact"],
        "contact_definition": "Any future interval with an asset OR chassis resultant contact >1 N; positive survives censoring. Main enriched event schema differs explicitly from legacy asset-only. Raw sources and separate labels retained; threshold not tuned on test.",
        "rollover_definition": "Any future post-physics-step abs roll OR abs pitch >60 degrees; raw sampled extrema and signed endpoint targets also retained.",
        "bounded_motion_definition": "Entirely future 2s window, measured 41-endpoint XY diameter<=0.25m, all40 throttle>0.3, no parking or measured goal arrival. Prefix<2s masked; known positive survives censoring.",
        "sustained_stall_definition": "Entirely future2s contiguous |body vx|<0.3m/s, every throttle>0.3, no parked/goal-arrived endpoints. Separate from bounded motion; no startup grace.",
        "low_progress_definition": "Measured prefix>=2s, net XY<0.15m/s times duration, mean throttle>0.3, no parking/arrival; complete endpoint needed.",
        "work_definition": "Cumulative engine_interface_positive_work_kj over exact recorded intervals; solver-step quadrature of positive engine torque*transmission motorshaft feedback speed, not fuel energy.",
        "attitude_definition": "Signed roll/pitch at actual future endpoint, per-axis mask. Cumulative abs roll/pitch interval maxima are separate reporting targets, not substituted endpoint labels.",
        "censoring": "Trajectory/work/attitude require measured end and finite needed labels. Known cumulative contact/rollover/stall positives remain valid beyond termination; unknown negatives masked.",
        "observation": "One measured pre-drive GLOBAL RGB-D map per scene, reused at all anchors. This is a fixed sensor snapshot, not current per-anchor onboard vision; assumes terrain/obstacle scene static.",
        "planner_inputs": "history, commands, global_features, nominal_pose and image_index-selected RGB-D only. Authored BMP, friction/config, asset inventory and future telemetry are labels/provenance only.",
        "startup_history": "First measured state/pose repeated to16 samples, synthetic past steering0/throttle0/brake1; no measured settling history claimed.",
        "camera": camera, "image_size": 512, "elevation_scale_m": float(camera["elevation_scale_m"]),
        "image_storage": "split_rgbd.npy has one float16[4,512,512] encoding per scene; image_index maps each window to its shared image",
        "source_sha256": observed_sources, "runtime_sha256": observed_runtime, "source_archive_root": str(args.source_snapshot_root.resolve()),
        "arena_archive_root": str(args.arena_root.resolve()),
        "preparation_workers": args.workers,
        "preparation_sha256": {str(path.relative_to(ROOT)): sha(path) for path in code_files},
        "collection_protocol_sha256": sha(args.collection_protocol) if args.collection_protocol else None,
        "normalization_from": normalized_from, "splits": {}}
    for split in splits:
        selected = [record for record in records if record["split"] == split]
        scene_ids = sorted({record["scene_id"] for record in selected})
        scene_index = {scene: i for i, scene in enumerate(scene_ids)}
        image_path = args.out / f"{split}_rgbd.npy"
        images = np.lib.format.open_memmap(image_path, mode="w+", dtype=np.float16, shape=(len(scene_ids), 4, 512, 512))
        for scene in scene_ids:
            images[scene_index[scene]] = next(record["snapshot"]["rgbd"] for record in selected if record["scene_id"] == scene)
        images.flush()
        del images
        arrays, episodes = {}, []
        payloads = ((epi, {key: value for key, value in record.items() if key != "snapshot"},
            scene_index[record["scene_id"]], args.horizon, args.output_dt, args.anchor_stride)
            for epi, record in enumerate(selected))
        for chunks, episode in bounded_ordered_results(payloads, args.workers):
            for key, value in chunks.items():
                arrays.setdefault(key, []).append(value)
            episodes.append(episode)
        arrays = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
        if any(not np.isfinite(array).all() for array in arrays.values()):
            raise ValueError("Nonfinite model tensor; missing targets must be zero-filled with masks")
        target = args.out / f"{split}.npz"
        np.savez(target, **arrays)
        dump(args.out / f"{split}_episodes.json", episodes)
        manifest["splits"][split] = {"episodes": len(episodes), "scenes": len(scene_ids), "scene_ids": scene_ids,
            "windows": len(arrays["anchor"]), "shapes": {key: list(value.shape) for key, value in arrays.items()},
            "sha256": sha(target), "rgbd_file": image_path.name, "rgbd_shape": [len(scene_ids), 4, 512, 512],
            "rgbd_sha256": sha(image_path), "event_support": event_support(arrays, episodes),
            "episodes_sha256": sha(args.out / f"{split}_episodes.json")}
        if split == "train":
            for key in ("history", "commands", "global_features"):
                values = arrays[key].reshape(-1, arrays[key].shape[-1])
                normalization[key] = {"mean": values.mean(0, dtype=np.float64).tolist(), "std": np.maximum(values.std(0, dtype=np.float64), .01).tolist()}
        print(json.dumps({"split": split, "scenes": len(scene_ids), "episodes": len(episodes), "windows": len(arrays["anchor"])}), flush=True)
    dump(args.out / "normalization.json", normalization)
    manifest["normalization_sha256"] = sha(args.out / "normalization.json")
    dump(args.out / "manifest.json", manifest)
    print(json.dumps({"out": str(args.out), "manifest_sha256": sha(args.out / "manifest.json")}))


if __name__ == "__main__":
    main()
