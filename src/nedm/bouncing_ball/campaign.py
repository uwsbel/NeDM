"""Stratified AMD collection, independent shard validation and model-rate packing."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from nedm.bouncing_ball.collection import STATE_FIELDS, atomic_json, collect, load_config
from nedm.bouncing_ball.validation import validate_dataset


def make_plan(campaign: dict, config: dict) -> list[dict]:
    plan = []
    nx, nz = campaign["grid"]
    xlo, xhi = config["launch"]["vx_range_mps"]
    zlo, zhi = config["launch"]["vz_range_mps"]
    for ix in range(nx):
        for iz in range(nz):
            for split, count in campaign["episodes_per_cell"].items():
                for repeat in range(count):
                    label = f"c{ix:02d}_{iz:02d}_{split}_{repeat:02d}"
                    rng = random.Random(f"{campaign['seed']}:{label}")
                    plan.append({"name": label, "episode_id": f"ball_{campaign['seed']}_{label}",
                                 "cell": [ix, iz], "split": split,
                                 "vx_mps": xlo + (ix + rng.random()) / nx * (xhi-xlo),
                                 "vz_mps": zlo + (iz + rng.random()) / nz * (zhi-zlo)})
    return plan


def collect_shard(payload: tuple) -> dict:
    root, shard_id, config, launches = payload
    output = Path(root) / "shards" / f"shard_{shard_id:03d}"
    if (output / "dataset_index.json").exists():
        existing = json.loads((output / "dataset_index.json").read_text())
        if not existing["complete"] or existing.get("explicit_launches") != launches:
            raise RuntimeError(f"preserved incomplete/incompatible shard needs a new campaign directory: {output}")
    else:
        config = copy.deepcopy(config)
        config["collection"]["episodes"] = len(launches)
        config["collection"]["seed"] += shard_id
        collect(config, output, len(launches), len(launches), launches)
    validation = validate_dataset(output)
    atomic_json(output / "validation_report.json", validation)
    if not validation["passed"]:
        raise RuntimeError(f"shard validation failed: {output}")
    return {"shard": shard_id, "root": str(output), "validation": validation}


def pack(root: Path, campaign: dict, config: dict, plan: list[dict], shards: list[dict]) -> dict:
    dt = campaign["model_step_s"]
    raw_dt = config["simulation"]["record_step_s"]
    stride = round(dt/raw_dt)
    if abs(stride*raw_dt-dt) > 1e-10:
        raise ValueError("model timestep must be an integer multiple of raw recording timestep")
    episodes, states, labels, energy_gains = [], [], [], []
    for shard in sorted(shards, key=lambda item: item["shard"]):
        source = Path(shard["root"])
        index = json.loads((source / "dataset_index.json").read_text())
        for item in index["episodes"]:
            path = source / item["csv_path"]
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            trajectory = np.array([[float(row[field]) for field in STATE_FIELDS] for row in rows[::stride]], dtype=np.float32)
            contacts = np.zeros((len(trajectory)-1, 2), dtype=np.float32)
            for kind, channel in [("first_ground_contact", 0), ("wall_contact", 1)]:
                time = next(e["time_s"] for e in item["events"] if e["kind"] == kind)
                contact_index = math_ceil(time/dt - 1e-9)-1
                contacts[contact_index, channel] = 1
            energies = np.array([float(row["mechanical_energy_j"]) for row in rows])
            energy_gains.append(float(np.diff(energies).max()))
            episodes.append({**item, "csv_path": str(path.relative_to(root)),
                             "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "model_samples": len(trajectory), "runtime_provenance": index["provenance"]})
            states.append(trajectory)
            labels.append(contacts)
            if len(episodes) % 500 == 0:
                atomic_json(root / "progress.json", {"complete": False, "stage": "packing", "packed_episodes": len(episodes), "total_episodes": len(plan)})
                print(f"Packed episodes {len(episodes)}/{len(plan)}", flush=True)
    ids = [episode["episode_id"] for episode in episodes]
    if len(ids) != len(set(ids)) or set(ids) != {item["episode_id"] for item in plan}:
        raise RuntimeError("missing or duplicate planned episodes")
    coverage = Counter((tuple(item["launch_cell"]), item["split"]) for item in episodes)
    for ix in range(campaign["grid"][0]):
        for iz in range(campaign["grid"][1]):
            for split, count in campaign["episodes_per_cell"].items():
                if coverage[((ix, iz), split)] != count:
                    raise RuntimeError(f"coverage quota failed at {ix},{iz},{split}")
    if max(energy_gains) > 0.001:
        raise RuntimeError(f"energy increased in a raw trace: {max(energy_gains)} J")
    maximum_length = max(len(state) for state in states)
    padded = np.zeros((len(states), maximum_length, 5), dtype=np.float32)
    contact = np.zeros((len(states), maximum_length-1, 2), dtype=np.float32)
    for i, (state, events) in enumerate(zip(states, labels, strict=True)):
        padded[i, :len(state)] = state
        padded[i, len(state):] = state[-1]
        contact[i, :len(events)] = events
    split_ids = np.array([{"train": 0, "val": 1, "test": 2}[item["split"]] for item in episodes])
    np.savez_compressed(root / "model_data.npz", states=padded, contacts=contact,
                        lengths=np.array([len(state) for state in states]), splits=split_ids,
                        cells=np.array([item["launch_cell"] for item in episodes]),
                        launches=np.array([[item["launch"]["vx_mps"], item["launch"]["vz_mps"]] for item in episodes]))
    report = {"complete": True, "campaign": campaign, "config": config, "host": platform.node(),
              "episodes": episodes, "episode_count": len(episodes), "split_counts": dict(Counter(item["split"] for item in episodes)),
              "coverage_cells": len(coverage)//len(campaign["episodes_per_cell"]), "max_raw_energy_increase_j": max(energy_gains),
              "max_penetration_m": max(item["max_penetration_m"] for item in episodes),
              "model_dt_s": dt, "model_data_sha256": hashlib.sha256((root / "model_data.npz").read_bytes()).hexdigest(),
              "shards": shards, "duration_range_s": [min(item["duration_s"] for item in episodes), max(item["duration_s"] for item in episodes)]}
    atomic_json(root / "campaign_index.json", report)
    return report


def math_ceil(value: float) -> int:
    import math
    return math.ceil(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    campaign = json.loads(args.campaign.read_text())
    config = load_config(Path(campaign["collection_config"]))
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = make_plan(campaign, config)
    if (root / "collection_plan.json").exists() and json.loads((root / "collection_plan.json").read_text()) != plan:
        raise RuntimeError("existing campaign plan differs")
    atomic_json(root / "collection_plan.json", plan)
    atomic_json(root / "campaign_config.json", campaign)
    payloads = [(str(root), i, config, plan[i::campaign["shards"]]) for i in range(campaign["shards"])]
    shards = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(collect_shard, payload) for payload in payloads]
        for future in as_completed(futures):
            shards.append(future.result())
            atomic_json(root / "progress.json", {"complete": False, "validated_shards": len(shards),
                                                "total_shards": len(payloads), "accepted_episodes": sum(s["validation"]["episodes"] for s in shards)})
            print(f"Validated shards {len(shards)}/{len(payloads)}", flush=True)
    report = pack(root, campaign, config, plan, shards)
    atomic_json(root / "progress.json", {"complete": True, "accepted_episodes": report["episode_count"], "split_counts": report["split_counts"]})
    print(json.dumps({key: report[key] for key in ["complete", "episode_count", "split_counts", "coverage_cells", "max_penetration_m", "duration_range_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
