#!/usr/bin/env python
"""Sidecar cache with extra z1 channels for the recorded episodes (WP5 powertrain state).

The WP2 cache's ``z1`` is the 15-D ``tire_normal_force_omega`` preset. The stores also hold
engine speed and motorshaft torque (the collector wrote them alongside ``capture_row``), so
instead of rewriting 5 GB of cache files the extra channels go into a small sidecar directory
``<out>/<key>.npz`` with ``z1_extra`` (T, k) float32, frame-aligned with the cache rows. The
trainer / tracker env concatenate them onto ``z1`` when ``--z1-extra-cache`` is given.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nedm.traverse import nrd_data as D
from nedm.training.constants import POWERTRAIN_STATE_FIELDS


def store_dir(stores: Path, key: str) -> Path:
    store, ep = key.split("__", 1)
    d = stores / store / ep
    if not d.exists() and store == "full_v4_partial":
        d = stores / "full_v4" / ep
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
    ap.add_argument("--stores", default="artifacts/traverse")
    ap.add_argument("--out", default="artifacts/traverse/wp2_z2_cache_v6_pt")
    ap.add_argument("--fields", nargs="+", default=POWERTRAIN_STATE_FIELDS)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    keys = D.load_cache_keys(Path(args.cache))
    t0, checked = time.time(), 0
    for i, key in enumerate(keys):
        dst = out / f"{key}.npz"
        if dst.exists():
            continue
        with np.load(store_dir(Path(args.stores), key) / "states.npz") as s:
            fields, table = list(s["fields"]), s["table"]
        extra = np.stack([table[:, fields.index(f)] for f in args.fields], 1).astype(np.float32)
        if checked < 50:  # frame alignment check against the cache's own z1 (vx is column 0)
            with np.load(Path(args.cache) / f"{key}.npz") as c:
                vx = c["z1"][:, 0]
            assert len(vx) == len(extra) and np.allclose(vx, table[:, fields.index("vel_body_x_mps")]), key
            checked += 1
        np.savez(dst, z1_extra=extra)
        if (i + 1) % 1000 == 0:
            print(f"[{i + 1}/{len(keys)}] {time.time() - t0:.0f}s", flush=True)
    (out / "sidecar_manifest.json").write_text(json.dumps({"source_cache": args.cache, "fields": args.fields, "episodes": len(keys)}))
    print(f"done: {len(keys)} keys, fields {args.fields}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
