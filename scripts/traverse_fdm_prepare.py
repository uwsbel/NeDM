"""Prepare isolated, read-only raw-store reuse for a short-horizon PID FDM."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nedm.traverse.fdm_data import (canonical_id, frozen_split_assignments, novel_split,
    load_episode, build_history, build_candidate_features, build_targets,
    tracked_route_indices, parking_mask, HISTORY_FIELDS, CANDIDATE_FIELDS, EVENT_NAMES, DT)
from nedm.traverse.terrain import TerrainMap


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, default=Path("/home/harry/NeDM"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stores", nargs="+", default=["full_v1", "full_v2", "full_v3"])
    ap.add_argument("--train-episodes", type=int, default=1000, help="0 selects all eligible train episodes")
    ap.add_argument("--val-episodes", type=int, default=250, help="0 selects all eligible validation episodes")
    ap.add_argument("--anchor-stride", type=int, default=20, help="20 Hz frames; 20 gives one-second anchors")
    args = ap.parse_args()
    if args.anchor_stride < 1: raise ValueError("Anchor stride must be positive")
    source = args.source_root.resolve()
    if args.out.resolve().is_relative_to(source):
        raise ValueError("Output must be outside the source checkout")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()): raise ValueError("Use a new empty output directory")
    split_path = source / "artifacts/traverse/wp2_z2_cache_v6/cache_manifest.json"
    old = frozen_split_assignments(json.loads(split_path.read_text()))
    found = {}
    for store in args.stores:
        folder = source / "artifacts/traverse" / store
        if not folder.exists(): raise FileNotFoundError(folder)
        for ep in sorted(folder.glob("ep_*")):
            # Family was assigned before driving; never filter by completion.
            if ep.name.endswith("_meander"): continue
            key = canonical_id(store + "__" + ep.name)
            split = old.get(key, novel_split(key))
            if split == "test": continue  # no test metadata or state payload opened
            if key in found: raise ValueError("Duplicate canonical episode: " + key)
            found[key] = (ep, split, "legacy" if key in old else "stable_extension")
    selected = {}
    for split, limit in (("train", args.train_episodes), ("val", args.val_episodes)):
        keys = sorted((k for k,v in found.items() if v[1] == split),
                      key=lambda k: hashlib.sha256(("fdm_fast_v1:"+k).encode()).hexdigest())
        selected[split] = keys[:limit] if limit else keys
        if not selected[split]: raise ValueError("Empty " + split)
    arena = source / "assets/traverse/arena_v1"
    terrain = TerrainMap.from_dir(arena)
    manifest = {"schema": 1, "source_root": str(source), "stores": args.stores,
                "controller_domain": "standard ChPathFollowerDriver: substep PID, initialized before 0.8 s settle",
                "observation": "Privileged quantized BMP plus authored asset footprints; not camera validation",
                "history_fields": HISTORY_FIELDS, "candidate_fields": CANDIDATE_FIELDS,
                "global_fields": ["remaining_route_m", "goal_ego_x_m", "goal_ego_y_m", "elapsed_recording_s"],
                "startup_padding": "Unrecorded history repeats first measured state/pose and synthetic [steer=0,throttle=0,brake=1]; not actual settle telemetry. Available history count=min(16,1+round(global_features[3]/0.05)).",
                "event_names": EVENT_NAMES, "output_dt_s": .2, "horizon_s": 4., "history_frames": 16,
                "anchor_stride": args.anchor_stride, "split_manifest_sha256": sha(split_path),
                "split_rule": "Frozen legacy layout split; partial/full canonical aliases; stable hash for unseen identities; test payloads never opened",
                "selection": "Lowest SHA256(fdm_fast_v1:canonical_id), before reading outcomes; nominal-route families only",
                "terrain_sha256": {p.name: sha(p) for p in arena.iterdir() if p.suffix in (".bmp", ".json")},
                "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [ROOT/"src/nedm/traverse/fdm_data.py", Path(__file__)]},
                "splits": {}}
    normalization = {}
    for split in ("train", "val"):
        arrays, records = {}, []
        for epi, key in enumerate(selected[split]):
            path, _, origin = found[key]
            ep = load_episode(path)
            meta, state, actions, poses = ep["meta"], ep["state"], ep["actions"], ep["poses"]
            route = meta["route"]
            if route is None: raise ValueError("Non-route episode passed family selection: " + key)
            indices = tracked_route_indices(route, poses)
            parked = parking_mask(route, poses, indices)
            anchors = list(range(0, max(1, len(poses)-1), args.anchor_stride))
            record = {"id": key, "source": str(path.relative_to(source)), "assignment": origin,
                      "status": meta["status"], "frames": len(poses), "windows": len(anchors),
                      "meta_sha256": sha(path/"meta.json"), "states_sha256": sha(path/"states.npz"),
                      "contact_events": len(meta["contact"]["events"])}
            records.append(record)
            for anchor in anchors:
                sample = {"history": build_history(state, actions, poses, anchor)}
                sample.update(build_candidate_features(route, poses[anchor], terrain, meta["layout"],
                    station=float(route["stations"][indices[anchor]]), elapsed_s=anchor*DT))
                sample.update(build_targets(poses, state, actions, ep["power_kw"], meta, anchor, parked))
                sample.update(episode_index=np.int64(epi), anchor=np.int64(anchor))
                for k, v in sample.items(): arrays.setdefault(k, []).append(v)
            if (epi+1) % 100 == 0: print(split, epi+1, "/", len(selected[split]), flush=True)
        arrays = {k: np.stack(v) for k,v in arrays.items()}
        if any(not np.isfinite(v).all() for v in arrays.values()): raise ValueError("Nonfinite prepared array")
        dest = args.out / (split + ".npz")
        np.savez(dest, **arrays)
        (args.out/(split+"_episodes.json")).write_text(json.dumps(records, indent=1)+"\n")
        event_mask = arrays["event_mask"]
        manifest["splits"][split] = {"episodes": len(records), "windows": len(arrays["history"]),
            "frames": sum(r["frames"] for r in records), "statuses": dict(Counter(r["status"] for r in records)),
            "source_stores": dict(Counter(r["id"].split("__")[0] for r in records)),
            "event_positive_targets": dict(zip(EVENT_NAMES, (arrays["events"]*event_mask).sum((0,1)).astype(int).tolist())),
            "event_valid_targets": dict(zip(EVENT_NAMES, event_mask.sum((0,1)).astype(int).tolist())),
            "event_positive_windows": dict(zip(EVENT_NAMES, np.any(arrays["events"]*event_mask > 0, axis=1).sum(0).astype(int).tolist())),
            "shapes": {k:list(v.shape) for k,v in arrays.items()}, "sha256": sha(dest)}
        if split == "train":
            for k in ("history", "candidate", "global_features"):
                a = arrays[k].reshape(-1, arrays[k].shape[-1])
                if k == "candidate": a = a[a[:, -1] > .5]
                mean, std = a.mean(0, dtype=np.float64), a.std(0, dtype=np.float64)
                std = np.maximum(std, .01)
                if k == "candidate": mean[-1], std[-1] = 0., 1.
                normalization[k] = {"mean": mean.tolist(), "std": std.tolist()}
        print(split, json.dumps(manifest["splits"][split]), flush=True)
    (args.out/"normalization.json").write_text(json.dumps(normalization, indent=2)+"\n")
    manifest["normalization_sha256"] = sha(args.out/"normalization.json")
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__": main()
