"""Append extra training episodes (unified format) to an existing unified dataset.

The base keeps its splits (train / validation / test); every extra episode becomes
training data. Arrays are padded to the longer record count by repeating each
episode's last state (contacts padded with 0), as the converters do.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load(root):
    with np.load(root / "unified_data.npz") as d:
        arrays = {k: d[k] for k in ("states", "contacts", "lengths", "splits", "launches")}
    return arrays, json.loads((root / "system.json").read_text()), json.loads((root / "index.json").read_text())


def pad(arrays, t):
    s, c = arrays["states"], arrays["contacts"]
    if s.shape[1] < t:
        s = np.concatenate((s, np.repeat(s[:, -1:], t - s.shape[1], 1)), 1)
        c = np.concatenate((c, np.zeros((c.shape[0], t - 1 - c.shape[1], c.shape[2]), c.dtype)), 1)
    return {**arrays, "states": s, "contacts": c}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, system, ia = load(args.base)
    b, system_b, ib = load(args.extra)
    if system["pairs"] != system_b["pairs"] or system["physics_config"] != system_b["physics_config"]:
        raise ValueError("different systems")
    t = max(a["states"].shape[1], b["states"].shape[1])
    a, b = pad(a, t), pad(b, t)
    b["splits"] = np.zeros_like(b["splits"])
    out = {k: np.concatenate((a[k], b[k])) for k in a}
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output / "unified_data.npz", **out)
    digest = hashlib.sha256((args.output / "unified_data.npz").read_bytes()).hexdigest()
    (args.output / "system.json").write_text(json.dumps(system, indent=1))
    episodes = ia["episodes"] + [{**e, "split_id": 0} for e in ib["episodes"]]
    (args.output / "index.json").write_text(json.dumps({
        "episodes": episodes, "data_sha256": digest,
        "provenance": {"base": str(args.base), "base_sha256": ia["data_sha256"], "extra": str(args.extra), "extra_sha256": ib["data_sha256"]},
        "counts": {s: int((out["splits"] == k).sum()) for k, s in enumerate(("train", "val", "test"))}}))
    print(json.dumps({"states": list(out["states"].shape), "counts": {s: int((out["splits"] == k).sum()) for k, s in enumerate(("train", "val", "test"))},
                      "sha256": digest}))


if __name__ == "__main__":
    main()
