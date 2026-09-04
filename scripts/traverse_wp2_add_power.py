"""Append the powertrain channel to an existing WP2 latent cache (CPU only).

Plan section 4 keeps drive power OUT of the recursive token and gives it a
supervised auxiliary head instead, so the state-only baseline stays exactly
15-D for RQ2. The power target is read straight from the episode store, so
this needs no encoder pass -- the 4 GB z2 cache is reused as-is.

    power (T, 1) float32   motorshaft torque x speed, kW
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

TORQUE_FIELD = "engine_motorshaft_torque_nm"
SPEED_FIELD = "trans_motorshaft_speed_radps"


def episode_dir(traverse_root: Path, key: str) -> Path:
    store, episode = key.split("__", 1)
    return traverse_root / store / episode


def add_power(job: tuple[str, str, str]) -> str | None:
    cache_dir, traverse_root, key = job
    target = Path(cache_dir) / f"{key}.npz"
    with np.load(target) as data:
        payload = {k: data[k] for k in data.files}
    if "power" in payload:
        return None
    with np.load(episode_dir(Path(traverse_root), key) / "states.npz", allow_pickle=True) as states:
        fields = [str(f) for f in states["fields"]]
        table = states["table"]
    torque = table[:, fields.index(TORQUE_FIELD)]
    speed = table[:, fields.index(SPEED_FIELD)]
    payload["power"] = (torque * speed / 1000.0).astype(np.float32)[:, None]
    if payload["power"].shape[0] != payload["z1"].shape[0]:
        return f"{key}: power {payload['power'].shape[0]} != z1 {payload['z1'].shape[0]}"
    np.savez(target, **payload)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--traverse-root", default="artifacts/traverse")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    cache = Path(args.cache)
    keys = [p.stem for p in sorted(cache.glob("*.npz"))]
    print(f"adding power to {len(keys)} cached episodes", flush=True)
    jobs = [(str(cache), args.traverse_root, k) for k in keys]
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, err in enumerate(pool.map(add_power, jobs, chunksize=32), start=1):
            if err:
                errors.append(err)
            if i % 1000 == 0:
                print(f"  [{i}/{len(keys)}]", flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} episodes failed, first: {errors[0]}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
