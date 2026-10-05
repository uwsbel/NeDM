"""Stratified pool-shot collection on AMD, per-episode checks and model-rate packing.

Launches are planned before simulation on a (speed, aim) grid. Aim is the
angle of A's launch from the A->B line; its range keeps the cut angle below
the configured limit, so every launch hits B. Each cell gets fixed train /
validation / test counts. Rejected shots are kept with their reason.

Packed data (model_data.npz):
  states   [N, T, 14] float32  A then B: x, y, vx, vy, wx, wy, wz at every record step
  contacts [N, T-1, 9] uint8   overlap of each impulsive pair during each record interval
  heights  [N, T, 2, 2] float32 z and vz of A and B (diagnostics only, not model state)
  lengths, splits (0 train / 1 val / 2 test), cells, launches (vx, vy), polar (speed, aim)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from nedm.pool_ball.physics import (PAIRS, STATE_COLUMNS, aim_limit_rad, launch_velocity, load_config,
                                    simulate_episode)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    temporary.replace(path)


def make_plan(campaign: dict, config: dict) -> list[dict]:
    plan = []
    ns, na = campaign["grid"]
    slo, shi = config["launch"]["speed_range_mps"]
    limit = aim_limit_rad(config)
    for i in range(ns):
        for j in range(na):
            for split, count in campaign["episodes_per_cell"].items():
                for repeat in range(count):
                    label = f"c{i:02d}_{j:02d}_{split}_{repeat:02d}"
                    rng = random.Random(f"{campaign['seed']}:{label}")
                    speed = slo + (i + rng.random()) / ns * (shi - slo)
                    aim = -limit + (j + rng.random()) / na * 2 * limit
                    vx, vy = launch_velocity(config, speed, aim)
                    plan.append({"episode_id": f"pool_{campaign['seed']}_{label}", "cell": [i, j], "split": split,
                                 "speed_mps": speed, "aim_rad": aim, "vx_mps": vx, "vy_mps": vy})
    return plan


def collect_shard(payload) -> dict:
    root, shard_id, config, launches = payload
    output = Path(root) / "shards" / f"shard_{shard_id:03d}"
    done = output / "shard_index.json"
    if done.exists():
        index = json.loads(done.read_text())
        if [e["episode_id"] for e in index["attempts"]] != [x["episode_id"] for x in launches]:
            raise RuntimeError(f"existing shard differs from the plan: {output}")
        return index
    output.mkdir(parents=True, exist_ok=True)
    attempts = []
    for launch in launches:
        arrays, meta = simulate_episode(config, launch["vx_mps"], launch["vy_mps"])
        meta.pop("events_full")
        meta.update({k: launch[k] for k in ("episode_id", "cell", "split", "speed_mps", "aim_rad")})
        np.savez_compressed(output / f"{launch['episode_id']}.npz", **arrays)
        attempts.append(meta)
    index = {"shard": shard_id, "attempts": attempts}
    atomic_json(done, index)
    return index


def provenance() -> dict:
    import pychrono

    bundle = {}
    for parent in Path(pychrono.__file__).resolve().parents:
        if (parent / "conda-package.json").exists():
            package = json.loads((parent / "conda-package.json").read_text())
            bundle = {k: package.get(k) for k in ("name", "version", "build")}
            break
    return {"python": sys.version, "platform": platform.platform(), "host": platform.node(), "chrono": bundle,
            "physics_sha256": hashlib.sha256((Path(__file__).with_name("physics.py")).read_bytes()).hexdigest()}


def pack(root: Path, campaign: dict, config: dict, plan: list[dict], shards: list[dict]) -> dict:
    attempts = [a for s in sorted(shards, key=lambda s: s["shard"]) for a in s["attempts"]]
    by_id = {a["episode_id"]: a for a in attempts}
    if set(by_id) != {p["episode_id"] for p in plan}:
        raise RuntimeError("attempts do not match the plan")
    accepted = [by_id[p["episode_id"]] for p in plan if by_id[p["episode_id"]]["accepted"]]
    record = config["simulation"]["record_step_s"]
    length = round(config["simulation"]["duration_s"] / record) + 1
    n = len(accepted)
    states = np.zeros((n, length, 14), np.float32)
    heights = np.zeros((n, length, 2, 2), np.float32)
    contacts = np.zeros((n, length - 1, len(PAIRS)), np.uint8)
    shard_of = {a["episode_id"]: s["shard"] for s in shards for a in s["attempts"]}
    for k, meta in enumerate(accepted):
        with np.load(root / "shards" / f"shard_{shard_of[meta['episode_id']]:03d}" / f"{meta['episode_id']}.npz") as data:
            raw = data["states"]
            if len(raw) != length:
                raise RuntimeError(f"{meta['episode_id']} has {len(raw)} records, expected {length}")
            states[k] = raw[:, :, STATE_COLUMNS].reshape(length, 14)
            heights[k] = raw[:, :, [2, 5]]
            contacts[k] = data["contact_depth"] > 0
    splits = np.array([{"train": 0, "val": 1, "test": 2}[m["split"]] for m in accepted])
    np.savez(root / "model_data.npz", states=states, contacts=contacts, heights=heights,
             lengths=np.full(n, length), splits=splits, cells=np.array([m["cell"] for m in accepted]),
             launches=np.array([[m["vx_mps"], m["vy_mps"]] for m in accepted]),
             polar=np.array([[m["speed_mps"], m["aim_rad"]] for m in accepted]))
    coverage = Counter((tuple(m["cell"]), m["split"]) for m in accepted)
    missing = [(i, j, split) for i in range(campaign["grid"][0]) for j in range(campaign["grid"][1])
               for split, count in campaign["episodes_per_cell"].items() if coverage[((i, j), split)] < count]
    report = {"complete": True, "campaign": campaign, "config": config, "provenance": provenance(),
              "episode_count": n, "attempt_count": len(attempts),
              "rejections": dict(Counter(a["termination"] for a in attempts if not a["accepted"])),
              "split_counts": dict(Counter(m["split"] for m in accepted)), "cells_short_of_quota": missing,
              "record_step_s": record, "episodes": accepted,
              "model_data_sha256": hashlib.sha256((root / "model_data.npz").read_bytes()).hexdigest()}
    atomic_json(root / "campaign_index.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--shard-range", type=int, nargs=2, metavar=("FIRST", "STOP"),
                        help="collect only shards [FIRST, STOP) (for multi-node collection); skip packing")
    parser.add_argument("--pack-only", action="store_true")
    args = parser.parse_args(argv)
    campaign = json.loads(args.campaign.read_text())
    config = load_config(Path(campaign["collection_config"]))
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = make_plan(campaign, config)
    if (root / "collection_plan.json").exists():
        if json.loads((root / "collection_plan.json").read_text()) != plan:
            raise RuntimeError("existing campaign plan differs")
    else:
        atomic_json(root / "collection_plan.json", plan)
        atomic_json(root / "campaign_config.json", campaign)
    # Interleave cells across shards so every shard spans the action grid.
    shards_total = campaign["shards"]
    payloads = [(str(root), i, copy.deepcopy(config), plan[i::shards_total]) for i in range(shards_total)]
    if args.shard_range:
        payloads = payloads[args.shard_range[0]:args.shard_range[1]]
    if not args.pack_only:
        finished = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(collect_shard, p) for p in payloads]):
                future.result()
                finished += 1
                print(f"shards {finished}/{len(payloads)}", flush=True)
        if args.shard_range:
            return 0
    shards = [json.loads((root / "shards" / f"shard_{i:03d}" / "shard_index.json").read_text()) for i in range(shards_total)]
    report = pack(root, campaign, config, plan, shards)
    print(json.dumps({k: report[k] for k in ("episode_count", "attempt_count", "rejections", "split_counts")}, indent=1))
    print(f"cells short of quota: {len(report['cells_short_of_quota'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
