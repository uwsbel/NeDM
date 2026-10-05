"""System descriptions (bodies and candidate pairs) and converters to one packed format.

Packed format (unified_data.npz, every record step = 1 ms):
  states    [N, T, D, 9]  float32  moving bodies' world-frame state [p(3), v(3), w(3)]
  contacts  [N, T-1, P]   uint8    pair p in contact during record interval (t_k, t_k+1]
  lengths   [N]  valid records per episode     splits [N] (0 train, 1 val, 2 test)
  launches  [N, 2]                             (system-specific launch parameters)
system.json: bodies (name, kind sphere|plane, moving, radius, point, normal),
pairs (moving body i, partner j), record and model steps, target body and time.
Fixed bodies never move; their token comes from the description.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def pool_system(config: dict) -> dict:
    scene = config["scene"]
    R, hx, hy = scene["radius_m"], scene["half_length_m"], scene["half_width_m"]
    bodies = [
        {"name": "A", "kind": "sphere", "moving": True, "radius": R},
        {"name": "B", "kind": "sphere", "moving": True, "radius": R},
        {"name": "cushion_xp", "kind": "plane", "moving": False, "point": [hx, 0.0, R], "normal": [-1.0, 0.0, 0.0]},
        {"name": "cushion_xm", "kind": "plane", "moving": False, "point": [-hx, 0.0, R], "normal": [1.0, 0.0, 0.0]},
        {"name": "cushion_yp", "kind": "plane", "moving": False, "point": [0.0, hy, R], "normal": [0.0, -1.0, 0.0]},
        {"name": "cushion_ym", "kind": "plane", "moving": False, "point": [0.0, -hy, R], "normal": [0.0, 1.0, 0.0]},
    ]
    # Same order as pool_ball.physics.PAIRS: AB, A at xp/xm/yp/ym, B at xp/xm/yp/ym.
    pairs = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [1, 4], [1, 5]]
    return {"name": "pool", "bodies": bodies, "pairs": pairs, "record_step_s": config["simulation"]["record_step_s"],
            "model_step_s": 0.01, "target_body": 1, "target_time_s": 2.0, "gravity_axis": [0.0, 0.0, 1.0],
            "physics_config": config}


def ball_system(config: dict) -> dict:
    scene = config["scene"]
    bodies = [
        {"name": "ball", "kind": "sphere", "moving": True, "radius": scene["radius_m"]},
        {"name": "floor", "kind": "plane", "moving": False, "point": [0.0, 0.0, scene["ground_z_m"]], "normal": [0.0, 0.0, 1.0]},
        {"name": "wall", "kind": "plane", "moving": False, "point": [scene["wall_front_x_m"], 0.0, 0.0], "normal": [-1.0, 0.0, 0.0]},
    ]
    return {"name": "ball", "bodies": bodies, "pairs": [[0, 1], [0, 2]], "record_step_s": 0.001, "model_step_s": 0.01,
            "target_body": 0, "target_time_s": 1.7, "gravity_axis": [0.0, 0.0, 1.0], "physics_config": config}


def convert_pool(source: Path, output: Path) -> None:
    index = json.loads((source / "campaign_index.json").read_text())
    system = pool_system(index["config"])
    R = index["config"]["scene"]["radius_m"]
    with np.load(source / "model_data.npz") as data:
        s14, contacts = data["states"], data["contacts"]
        lengths, splits, launches = data["lengths"], data["splits"], data["launches"]
    n, t = s14.shape[:2]
    states = np.zeros((n, t, 2, 9), np.float32)
    for b in range(2):
        x = s14[..., 7 * b: 7 * b + 7]
        states[..., b, 0], states[..., b, 1], states[..., b, 2] = x[..., 0], x[..., 1], R
        states[..., b, 3], states[..., b, 4] = x[..., 2], x[..., 3]
        states[..., b, 6:9] = x[..., 4:7]
    write(output, system, states, contacts.astype(np.uint8), lengths, splits, launches, index["episodes"],
          {"source": str(source), "source_sha256": index["model_data_sha256"]})


def _ball_episode(args):
    csv_path, n_records = args
    with open(csv_path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    raw = np.array([[float(r[k]) for k in ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps", "omega_x_radps",
                                           "omega_y_radps", "omega_z_radps", "ground_contact", "wall_contact")]
                    for r in rows])
    stride = 2  # 0.5 ms rows -> 1 ms records
    states = raw[::stride][:n_records, :9]
    flags = raw[:, 9:11]
    # Row k's flag covers (t_{k-1}, t_k]; the 1 ms interval (t_m, t_{m+1}] is rows 2m+1 and 2m+2.
    labels = np.zeros((n_records - 1, 2), np.uint8)
    for m in range(n_records - 1):
        labels[m] = np.maximum(flags[2 * m + 1], flags[min(2 * m + 2, len(flags) - 1)]) > 0.5
    if len(states) < n_records:
        raise ValueError(f"{csv_path}: {len(states)} records < {n_records}")
    return states.astype(np.float32), labels


def convert_ball(primary: Path, output: Path, margin_s: float = 0.02, workers: int = 16,
                 heldout: Path | None = None) -> None:
    """Primary campaign: train/val (and its original test as diagnostic split 2 unless a held-out campaign is given)."""
    sources = [(primary, (0, 1) if heldout else (0, 1, 2))] + ([(heldout, (2,))] if heldout else [])
    episodes, jobs = [], []
    config = None
    for root, keep in sources:
        index = json.loads((root / "campaign_index.json").read_text())
        config = config or index["config"]
        for item in index["episodes"]:
            split = {"train": 0, "val": 1, "test": 2}[item["split"]]
            if split not in keep:
                continue
            end = next(e["time_s"] for e in item["events"] if e["kind"] == "next_ground_contact")
            n = int(math.floor((end - margin_s) / 0.001 + 1e-9)) + 1
            n = min(n, item["rows"] // 2 + 1 if item["rows"] % 2 else item["rows"] // 2)
            episodes.append({**item, "split_id": split, "records": n})
            jobs.append((str(root / item["csv_path"]), n))
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_ball_episode, jobs, chunksize=16))
    t = max(r[0].shape[0] for r in results)
    states = np.zeros((len(results), t, 1, 9), np.float32)
    contacts = np.zeros((len(results), t - 1, 2), np.uint8)
    for k, (s, c) in enumerate(results):
        states[k, :len(s), 0] = s
        states[k, len(s):, 0] = s[-1]
        contacts[k, :len(c)] = c
    lengths = np.array([len(r[0]) for r in results])
    splits = np.array([e["split_id"] for e in episodes])
    launches = np.array([[e["launch"]["vx_mps"], e["launch"]["vz_mps"]] for e in episodes])
    write(output, ball_system(config), states, contacts, lengths, splits, launches, episodes,
          {"sources": [str(s[0]) for s in sources], "terminal_margin_s": margin_s})


def write(output, system, states, contacts, lengths, splits, launches, episodes, provenance):
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "unified_data.npz", states=states, contacts=contacts, lengths=lengths, splits=splits, launches=launches)
    digest = hashlib.sha256((output / "unified_data.npz").read_bytes()).hexdigest()
    (output / "system.json").write_text(json.dumps(system, indent=1))
    (output / "index.json").write_text(json.dumps({"episodes": episodes, "provenance": provenance, "data_sha256": digest,
                                                    "counts": {s: int((splits == k).sum()) for k, s in enumerate(("train", "val", "test"))}}))
    print(json.dumps({"output": str(output), "states": list(states.shape), "contacts": list(contacts.shape),
                      "pair_positive_fraction": contacts.reshape(-1, contacts.shape[-1]).mean(0).round(5).tolist(),
                      "splits": np.bincount(splits).tolist(), "sha256": digest}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=["pool", "ball"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--heldout", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.system == "pool":
        convert_pool(args.source, args.output)
    else:
        convert_ball(args.source, args.output, workers=args.workers, heldout=args.heldout)


if __name__ == "__main__":
    main()
