"""Build the per-episode reference-route cache for the WP3 tracker.

Every cache key in the WP2 latent cache maps to a collected episode whose
``meta.json`` carries the route the collection driver followed (plan section
6.2: random smooth splines, near-obstacle passes, oracle routes). Meander
episodes carry no route and are skipped. The tracker's imagination env reads
these as the reference the policy must follow, so no new route generation is
needed for v1 (plan section 10: "random splines + oracle routes").

Writes ``<out>/<key>.npz`` with waypoints (N,2), speeds (N,), headings (N,),
stations (N,) and ``<out>/routes_manifest.json`` listing keys by family.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
    ap.add_argument("--stores", default="artifacts/traverse")
    ap.add_argument("--out", default="artifacts/traverse/wp3_routes")
    args = ap.parse_args()

    keys = json.loads((Path(args.cache) / "cache_manifest.json").read_text())["episodes"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    families: dict[str, list[str]] = {}
    missing, counts = [], Counter()
    for key in keys:
        store, ep = key.split("__", 1)
        meta_path = Path(args.stores) / store / ep / "meta.json"
        if not meta_path.exists():
            missing.append(key); continue
        meta = json.loads(meta_path.read_text())
        route = meta.get("route")
        fam = meta.get("family_actual") or meta.get("family")
        counts[fam] += 1
        if not route:
            continue
        np.savez(out / f"{key}.npz",
                 waypoints=np.asarray(route["waypoints"], np.float32),
                 speeds=np.asarray(route["speeds"], np.float32),
                 headings=np.asarray(route["headings"], np.float32),
                 stations=np.asarray(route["stations"], np.float32),
                 start=np.asarray([*meta["layout"]["start_xy"], meta["layout"]["start_yaw"]], np.float32),
                 approach=np.asarray(meta.get("oracle_approach_pose") or [np.nan] * 3, np.float32),
                 t0_s=np.float32(meta.get("t0_s", 0.0)))
        families.setdefault(fam, []).append(key)
    manifest = {"cache": args.cache, "families": families,
                "n_routes": sum(len(v) for v in families.values()),
                "family_counts_all": dict(counts), "missing_meta": missing}
    (out / "routes_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"routes: {manifest['n_routes']} / {len(keys)} keys; families "
          f"{ {k: len(v) for k, v in families.items()} }; missing meta {len(missing)}")


if __name__ == "__main__":
    main()
