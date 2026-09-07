#!/usr/bin/env python
"""Arena families for the learning comparison (plan §28 step 1): several independent terrain instances.

One arena per generator seed, its parameters drawn from the seed within the family ranges the pilot found
decision-rich (§10.3 / §10.9): slope cap 25–32°, roughness 0.15–0.28 m at 2–3 m correlation, craters 2–4 m sigma,
hills 2.5–4.5 m high. Splits of the learning comparison are BY ARENA (train / val / sealed test), so every arena
is a separate instance, never a re-heading through one feature. The BMP orientation is Chrono's convention as
calibrated on arena_v1 (copied, as for arena_v2/v3).

  PYTHONPATH=src python scripts/traverse_wp7_arenas.py --seeds 101 102 103 104 105 106 107
"""
from __future__ import annotations

import argparse, json, math, shutil, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nedm.traverse.terrain import ArenaSpec, TerrainMap, write_arena


def family_spec(seed: int, difficulty: float = 1.0) -> tuple[ArenaSpec, dict]:
    """``difficulty`` scales the slope cap and the roughness amplitude (1.0 = the f101-f107 family; fresh sealed arenas
    for plan §31 step 3 use > 1 so that the fastest-candidate heuristic fails on more layouts)"""
    rng = np.random.default_rng(seed)
    cap_deg = min(40.0, float(rng.uniform(25.0, 32.0)) * difficulty)
    rough = float(rng.uniform(0.15, 0.28)) * difficulty; corr = float(rng.uniform(2.0, 3.0))
    spec = ArenaSpec(seed=seed, slope_cap=math.tan(math.radians(cap_deg)),
                     n_hills=int(rng.integers(5, 8)), hill_sigma_range_m=(4.0, 7.0), hill_height_range_m=(2.5, 4.5),
                     n_craters=int(rng.integers(5, 8)), crater_sigma_range_m=(2.0, 4.0), crater_depth_range_m=(1.2, 2.5),
                     roughness_amplitude_m=rough, roughness_corr_m=corr)
    return spec, {"seed": seed, "difficulty": difficulty, "slope_cap_deg": cap_deg, "roughness_m": rough, "roughness_corr_m": corr,
                  "n_hills": spec.n_hills, "n_craters": spec.n_craters}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--root", default="assets/traverse")
    ap.add_argument("--orientation-from", default="assets/traverse/arena_v1")
    ap.add_argument("--prefix", default="arena_f")
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--family-json", default="", help="write the family params here (default <root>/<prefix>family.json)")
    args = ap.parse_args()
    orient = json.loads((Path(args.orientation_from) / "arena_meta.json").read_text())["orientation"]
    orient = {**orient, "note": f"Chrono's BMP convention, copied from {Path(args.orientation_from).name} (calibrated there)"}
    fam = {}
    for seed in args.seeds:
        spec, params = family_spec(seed, args.difficulty)
        out = Path(args.root) / f"{args.prefix}{seed}"
        write_arena(spec, out)
        meta_p = out / "arena_meta.json"; meta = json.loads(meta_p.read_text())
        meta["orientation"] = orient; meta["family"] = params
        meta_p.write_text(json.dumps(meta, indent=2))
        tmap = TerrainMap.from_dir(out); st = meta["slope_stats"]
        fam[out.name] = params
        print(f"{out.name}: cap {params['slope_cap_deg']:.1f} deg rough {params['roughness_m']:.2f} m @ {params['roughness_corr_m']:.1f} m | "
              f"{params['n_hills']} hills {params['n_craters']} craters | height [{meta['height_min_m']:.2f}, {meta['height_max_m']:.2f}] | "
              f"slope max {math.degrees(math.atan(st['max'])):.1f} p99 {math.degrees(math.atan(st['p99'])):.1f} deg flat<5deg {st['flat5deg_fraction']:.2f}")
    Path(args.family_json or (Path(args.root) / f"{args.prefix}family.json")).write_text(json.dumps(fam, indent=1))


if __name__ == "__main__":
    main()
