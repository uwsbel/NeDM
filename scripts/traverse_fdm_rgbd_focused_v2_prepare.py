#!/usr/bin/env python3
"""Immutable combined obstacle/terrain pack with explicit bounded-motion v2 labels.

The v1 pack and preparation code are read-only inputs. The new terrain cohort
must be complete and scene-disjoint before preparation. Diagnostic probes are
never enumerated. Existing trajectory/contact/low-progress/strict-stall arrays
are copied unchanged, and separately named bounded_motion targets are added.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from nedm.traverse.fdm_bounded_targets import (BOUNDED_MOTION_DEFINITION,
    BOUNDED_MOTION_VERSION, bounded_motion_endpoints, bounded_motion_targets)

spec = importlib.util.spec_from_file_location("focused_v1_frozen", ROOT/"scripts/traverse_fdm_rgbd_focused_prepare.py")
V1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V1)


def bounded_support(arrays, episodes):
    labels, mask = arrays["bounded_motion"][:, :, 0], arrays["bounded_motion_mask"][:, :, 0]
    positive = ((labels > 0) & (mask > 0)).any(1)
    ei, anchors = arrays["episode_index"], arrays["anchor"]
    support = {}
    for name, group in (("all", np.ones(len(ei), bool)), ("anchor0", anchors == 0),
                        ("anchor_lt2s", anchors < 40), ("anchor_ge2s", anchors >= 40)):
        selected = positive & group
        identities = np.unique(ei[selected])
        support[name] = {"windows": int(group.sum()), "positive_windows": int(selected.sum()),
            "positive_episodes": len(identities), "positive_scenes": len({episodes[int(i)]["scene_id"] for i in identities}),
            "positive_horizon_labels": int(((labels > 0) & (mask > 0))[group].sum()),
            "valid_horizon_labels": int((mask[group] > 0).sum())}
    for name, key in (("before_first_contact", "first_contact_frame"),
                      ("before_observed_completed2s_onset", "first_observed_bounded_motion_frame")):
        onset = np.array([float("inf") if e[key] is None else e[key] for e in episodes])
        chosen = positive & (anchors < onset[ei])
        support[name] = {"positive_windows": int(chosen.sum()), "positive_episodes": len(np.unique(ei[chosen]))}
    return support


def read_verified_pack(path):
    manifest = json.loads((path/"manifest.json").read_text())
    for split in ("train", "val"):
        for name, key in ((split+".npz", "sha256"), (split+"_rgbd.npy", "rgbd_sha256")):
            if V1.sha(path/name) != manifest["splits"][split][key]:
                raise ValueError("Immutable source pack changed: "+str(path/name))
    if V1.sha(path/"normalization.json") != manifest["normalization_sha256"]:
        raise ValueError("Immutable source normalization changed")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-pack", type=Path, default=ROOT/"artifacts/traverse/fdm_rgbd_focused_pack_v1")
    ap.add_argument("--terrain-manifest", type=Path, required=True)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() or args.out.resolve().is_relative_to(Path("/home/harry/NeDM")):
        raise ValueError("Use a new isolated output path; no source/pack replacement")
    old = read_verified_pack(args.base_pack)
    # A change to the frozen v1 label/input implementation is an error, not an implicit migration.
    for name, expected in old["code_sha256"].items():
        if V1.sha(ROOT/name) != expected:
            raise ValueError("Frozen v1 preparation source changed: "+name)
    terrain_records, _, terrain_splits = V1.declared_episodes([args.terrain_manifest], args.raw_root)
    if set(terrain_splits) & set(old["scene_splits"]):
        raise ValueError("Terrain scenes overlap the base cohort")
    if any("probe" in name or "demo" in name for name in terrain_splits):
        raise ValueError("Diagnostic/probe scenes must not enter training")
    if not terrain_records:
        raise ValueError("Empty terrain cohort")
    # Reuse the tested frozen preparation contract for new physical sources.
    with tempfile.TemporaryDirectory(prefix="fdm_v2_prepare_", dir=ROOT/"artifacts/traverse") as temporary:
        terrain_pack = Path(temporary)/"terrain_pack"
        original_argv = sys.argv[:]
        try:
            sys.argv = [str(ROOT/"scripts/traverse_fdm_rgbd_focused_prepare.py"),
                "--cohort-manifest", str(args.terrain_manifest), "--raw-root", str(args.raw_root), "--out", str(terrain_pack)]
            V1.main()
        finally:
            sys.argv = original_argv
        terrain = read_verified_pack(terrain_pack)
        args.out.mkdir(parents=True, exist_ok=False)
        manifest = copy.deepcopy(old)
        manifest.update(schema=3, pack_version=2, bounded_motion_version=BOUNDED_MOTION_VERSION,
            bounded_motion_definition=BOUNDED_MOTION_DEFINITION,
            base_pack=str(args.base_pack.resolve()), base_pack_manifest_sha256=V1.sha(args.base_pack/"manifest.json"),
            terrain_component_manifest_sha256=V1.sha(terrain_pack/"manifest.json"),
            source_domain="combined focused obstacle-v1 and mesa-v2, same standard PID/Vulkan controller domain",
            source_domain_codes={"1": "focused_v1_obstacle_standard_chrono_pid_vulkan", "2": "focused_v2_mesa_standard_chrono_pid_vulkan"},
            training_scope="All predeclared obstacle and terrain siblings; focus-only, no legacy rows or diagnostic probe outcomes",
            selection="Entire two cohorts, no outcome filtering; v1 labels/images copied, bounded-motion v2 added explicitly",
            extra_target="sustained_stall and bounded_motion have separate masks; events[...,2] remains low_progress; trainer selects and archives its target",
            head2_default="events[...,2] remains low_progress; bounded_motion requires explicit progress-target choice",
            scene_splits={**old["scene_splits"], **terrain_splits},
            cohort_manifests={**old["cohort_manifests"], str(args.terrain_manifest.resolve()): V1.sha(args.terrain_manifest)},
            splits={})
        manifest["code_sha256"].update({str(p.relative_to(ROOT)): V1.sha(p) for p in (Path(__file__), ROOT/"src/nedm/traverse/fdm_bounded_targets.py")})
        V1.dump(args.out/"terrain_component_provenance.json", terrain)
        normalization = {}
        for split in ("train", "val"):
            with np.load(args.base_pack/(split+".npz")) as f:
                a = {k: f[k].copy() for k in f.files}
            with np.load(terrain_pack/(split+".npz")) as f:
                b = {k: f[k].copy() for k in f.files}
            if a.keys() != b.keys():
                raise ValueError("Different v1/new terrain tensor contract")
            old_episodes = json.loads((args.base_pack/(split+"_episodes.json")).read_text())
            new_episodes = json.loads((terrain_pack/(split+"_episodes.json")).read_text())
            for e in new_episodes:
                e["source_domain"] = "focused_v2_mesa_standard_chrono_pid_vulkan"
            b["episode_index"] += len(old_episodes)
            b["scene_index"] += len(old["splits"][split]["scene_ids"])
            b["source_domain"][:] = 2
            arrays = {k: np.concatenate((a[k], b[k])) for k in a}
            episodes = old_episodes+new_episodes
            windows = len(arrays["anchor"])
            arrays["bounded_motion"] = np.zeros((windows, 20, 1), np.float32)
            arrays["bounded_motion_mask"] = np.zeros((windows, 20, 1), np.float32)
            for ei, episode in enumerate(episodes):
                source = Path(episode["source"])
                for name, expected in episode["source_sha256"].items():
                    if V1.sha(source/name) != expected:
                        raise ValueError("Physical source changed: "+str(source/name))
                cm = json.loads((source/"collection_meta.json").read_text())
                ep = V1.load_rollout(source, cm["route"])
                endpoints = bounded_motion_endpoints(ep["poses"], ep["raw"]["action"], ep["parked"],
                    goal_xy=cm["route"]["waypoints"][-1], goal_radius_m=ep["outcome"]["goal_radius_m"])
                positive = np.flatnonzero(endpoints)
                episode["first_observed_bounded_motion_frame"] = int(positive[0]) if len(positive) else None
                episode["bounded_motion_ever"] = bool(len(positive))
                episode["bounded_motion_endpoints"] = int(len(positive))
                episode["bounded_motion_version"] = BOUNDED_MOTION_VERSION
                for row in np.flatnonzero(arrays["episode_index"] == ei):
                    labels, mask = bounded_motion_targets(endpoints, int(arrays["anchor"][row]))
                    arrays["bounded_motion"][row], arrays["bounded_motion_mask"][row] = labels, mask
            image_path = args.out/(split+"_rgbd.npy")
            images = np.lib.format.open_memmap(image_path, mode="w+", dtype=np.float16, shape=(windows, 4, 128, 128))
            begin = 0
            for source_pack in (args.base_pack, terrain_pack):
                source_images = np.load(source_pack/(split+"_rgbd.npy"), mmap_mode="r")
                for first in range(0, len(source_images), 128):
                    stop = min(first+128, len(source_images))
                    images[begin+first:begin+stop] = source_images[first:stop]
                begin += len(source_images)
            images.flush()
            del images
            for k in a:
                if not np.array_equal(arrays[k][:len(a[k])], a[k]):
                    raise ValueError("Original v1 rows were changed")
            target = args.out/(split+".npz")
            np.savez(target, **arrays)
            V1.dump(args.out/(split+"_episodes.json"), episodes)
            support = V1.event_support(arrays, episodes)
            support["bounded_motion"] = bounded_support(arrays, episodes)
            scene_ids = old["splits"][split]["scene_ids"]+terrain["splits"][split]["scene_ids"]
            manifest["splits"][split] = {"episodes": len(episodes), "scenes": len(scene_ids), "scene_ids": scene_ids,
                "windows": windows, "event_support": support, "shapes": {k: list(v.shape) for k, v in arrays.items()},
                "sha256": V1.sha(target), "rgbd_file": image_path.name, "rgbd_shape": [windows, 4, 128, 128],
                "rgbd_sha256": V1.sha(image_path), "source_window_counts": {"obstacle_v1": len(a["anchor"]), "mesa_v2": len(b["anchor"])}}
            if split == "train":
                for key in ("history", "commands", "global_features"):
                    values = arrays[key].reshape(-1, arrays[key].shape[-1])
                    normalization[key] = {"mean": values.mean(0, dtype=np.float64).tolist(),
                        "std": np.maximum(values.std(0, dtype=np.float64), .01).tolist()}
            print(json.dumps({"combined_split": split, "scenes": len(scene_ids), "episodes": len(episodes), "windows": windows,
                "anchor0_contact": support["contact"]["anchor0"], "anchor0_bounded_motion": support["bounded_motion"]["anchor0"]}), flush=True)
        V1.dump(args.out/"normalization.json", normalization)
        manifest["normalization_sha256"] = V1.sha(args.out/"normalization.json")
        V1.dump(args.out/"manifest.json", manifest)
    print(json.dumps({"out": str(args.out), "manifest_sha256": V1.sha(args.out/"manifest.json")}))


if __name__ == "__main__":
    main()
