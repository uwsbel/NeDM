#!/usr/bin/env python3
"""Reproduce two identity-selected validation frame-zero scenes across renderers.

Preparation reads only the first two identities of the already frozen validation
pack. Comparison measures image differences; it does not change a model,
calibration, checkpoint selection, or benchmark cases from those differences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    from nedm.traverse.storage import EpisodeReader
    from nedm.traverse.fdm_rgbd_data import rgbd_from_arrays

    records = json.loads((args.data / "val_episodes.json").read_text())[:2]
    manifest = []
    for record in records:
        source = args.source_root / record["source"]
        assert sha256(source / "meta.json") == record["meta_sha256"]
        meta = json.loads((source / "meta.json").read_text())
        store = json.loads((source.parent / "manifest.json").read_text())
        assert meta["route"] is not None
        reader = EpisodeReader(source)
        try:
            observation = reader.read_window(0, 1)
        finally:
            reader.close()
        rgb, depth = observation["rgb"][0], observation["depth_mm"][0]
        camera = store["camera"] | {"depth_offset_m": meta["depth_offset_m"], "depth_no_hit": meta["depth_no_hit"]}
        image = rgbd_from_arrays(rgb, depth_mm=depth, camera=camera)
        destination = args.out / record["id"]
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "original_frame0.npz", rgb=rgb, depth_mm=depth, rgbd=image)
        from PIL import Image
        Image.fromarray(rgb).save(destination / "original_rgb.png")
        case = {"id": record["id"], "arena": store["arena"]["dir"],
                "layout": meta["layout"], "goal_xy": meta["route"]["waypoints"][-1],
                "settle_reference": meta["route"], "role": "renderer consistency diagnostic only",
                "horizon_s": .1, "goal_radius_m": 2.5}
        (destination / "case.json").write_text(json.dumps(case, indent=2)+"\n")
        manifest.append({"id": record["id"], "source": record["source"], "meta_sha256": record["meta_sha256"],
                         "original_camera": camera, "arena_bmp_sha256": store["arena"]["bmp_sha256"],
                         "case_sha256": sha256(destination / "case.json")})
    (args.out / "manifest.json").write_text(json.dumps({"selection": "first two frozen validation identities, no outcome filtering", "scenes": manifest}, indent=2)+"\n")
    print(json.dumps({"out": str(args.out), "scenes": [r["id"] for r in manifest]}))


def compare(args):
    manifest = json.loads((args.out / "manifest.json").read_text())
    result = {"protocol": "Same static scene and nominal camera; separate original OptiX and new AMD Vulkan runtimes",
              "depth": "Each image uses its recorded or measured camera ray calibration; no true heightmap is used in this comparison",
              "limits": "Residual simulator settle and rendering differences may affect the vehicle pixels. Agreement is not identity of physics or proof of policy transfer.",
              "scenes": []}
    for row in manifest["scenes"]:
        folder = args.out / row["id"]
        with np.load(folder / "original_frame0.npz") as d:
            original = {k: d[k].copy() for k in d.files}
        with np.load(folder / "vulkan/observation.npz") as d:
            current = {k: d[k].copy() for k in d.files}
        old, new = original["rgbd"][3], current["rgbd"][3]
        valid = (old != -2.) & (new != -2.)
        error = 10.*np.abs(old[valid]-new[valid])
        image_error = np.abs(original["rgb"].astype(float)-current["rgb"].astype(float))
        result["scenes"].append({"id": row["id"], "rgb_mae_0_255": float(image_error.mean()),
            "rgb_p95_abs_difference_0_255": float(np.quantile(image_error, .95)),
            "old_registered_valid_fraction": float((old != -2.).mean()),
            "new_registered_valid_fraction": float((new != -2.).mean()),
            "common_registered_valid_fraction": float(valid.mean()),
            "registered_elevation_mae_m": float(error.mean()),
            "registered_elevation_median_abs_difference_m": float(np.median(error)),
            "registered_elevation_p95_abs_difference_m": float(np.quantile(error, .95)),
            "original_frame_sha256": sha256(folder / "original_frame0.npz"),
            "vulkan_frame_sha256": sha256(folder / "vulkan/observation.npz")})
    (args.out / "comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "compare"))
    parser.add_argument("--data", type=Path, default=Path("artifacts/traverse/fdm_fast_data_v2"))
    parser.add_argument("--source-root", type=Path, default=Path("/home/harry/NeDM"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/traverse/fdm_rgbd_demo_v1/renderer_replay"))
    args = parser.parse_args()
    prepare(args) if args.mode == "prepare" else compare(args)


if __name__ == "__main__":
    main()
