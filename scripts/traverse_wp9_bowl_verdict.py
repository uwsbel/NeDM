#!/usr/bin/env python
"""B1 verdict: did anything cross the crater? -- energy/crater plan section B1.

The runner's ``status`` alone cannot answer the B1 question. A run is ``completed`` when it reaches the
route end, and on the bowl arenas a vehicle can reach the route end by turning round inside the bowl,
driving back out of the deliberately-driveable 15 deg entry ramp and going round the outside. That is not
a crossing, and the plan's gate is about crossing.

So this classifies each direct entry by WHERE it first regains the plain after entering the bowl:

    entered      = radius < R_top (inside the steep rim)
    regained     = first frame after entry with terrain height > -0.30 m and radius > R_top
    crossed      = that frame is in the exit sector      (|azimuth| < 60 deg, the 47 deg wall)
    ramp exit    = that frame is in the entry sector     (|azimuth - 180| < 40 deg, the 15 deg ramp)

A ramp exit is a permitted escape direction: plan B1 requires "entry driveable and the exit wall
demanding", so leaving the way you came in is a detour, not a failure of the trap. The claim the gate
supports is therefore about the EXIT WALL, and it is stated for the tested envelope only -- never
"physically impossible under all actions".

  PYTHONPATH=src python scripts/traverse_wp9_bowl_verdict.py --run artifacts/traverse/wp9_bowl_b1
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nedm.traverse.terrain import TerrainMap  # noqa: E402

PLAIN_Z_M = -0.30       # above this the vehicle is back on the surrounding plain
EXIT_SECTOR_DEG = 60.0  # +- about the exit direction (azimuth 0)
ENTRY_SECTOR_DEG = 40.0  # +- about the entry ramp (azimuth 180)


def classify(pose: np.ndarray, tmap: TerrainMap, r_top: float) -> dict:
    r = np.hypot(pose[:, 0], pose[:, 1])
    az = np.degrees(np.arctan2(pose[:, 1], pose[:, 0]))
    entered = r < r_top
    if not entered.any():
        return {"outcome": "never entered"}
    i0 = int(np.argmax(entered))
    z = np.asarray([float(tmap.height(x, y)) for x, y in pose[i0:, :2]])
    out = np.where((z > PLAIN_Z_M) & (r[i0:] > r_top))[0]
    deepest = float(z.min())
    if len(out) == 0:
        return {"outcome": "never got out", "entered_s": i0 * 0.05, "deepest_m": deepest}
    j = i0 + int(out[0])
    a = float(az[j])
    if abs(a) < EXIT_SECTOR_DEG:
        outcome = "crossed the exit wall"
    elif abs(abs(a) - 180.0) < ENTRY_SECTOR_DEG:
        outcome = "left via the entry ramp"
    else:
        outcome = f"left via the flank (azimuth {a:.0f} deg)"
    return {"outcome": outcome, "entered_s": i0 * 0.05, "regained_s": j * 0.05,
            "regained_azimuth_deg": a, "deepest_m": deepest}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="artifacts/traverse/wp9_bowl_b1")
    ap.add_argument("--arena-root", default="assets/traverse")
    ap.add_argument("--arena-prefix", default="arena_bowl")
    ap.add_argument("--json", default=None, help="write the machine-readable verdict here")
    args = ap.parse_args()

    run = Path(args.run)
    rows = {}
    for line in (run / "rows.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "status" in r:
            rows[(r["key"], r["candidate"])] = r

    tmaps, r_top = {}, {}
    for arm in ("deep", "shallow", "flat"):
        d = Path(args.arena_root) / f"{args.arena_prefix}_{arm}"
        tmaps[arm] = TerrainMap.from_dir(d)
        r_top[arm] = float(tmaps[arm].meta["features"][0]["R_top_m"])

    per_run, outcomes = [], collections.Counter()
    status_by_arm = collections.defaultdict(collections.Counter)
    contact_by_arm = collections.Counter()
    total_by_arm = collections.Counter()
    for f in sorted(glob.glob(str(run / "records" / "*.npz"))):
        key, cand = Path(f).stem.split("__")
        arm = key.split("_")[1]
        row = rows.get((key, cand), {})
        if cand == "cam":
            continue
        total_by_arm[arm] += 1
        if float(row.get("max_contact_n", 0.0)) > 1.0:
            contact_by_arm[arm] += 1
        status_by_arm[arm][row.get("status", "?")] += 1
        if arm != "deep" or "detour" in cand:
            continue
        with np.load(f, allow_pickle=True) as d:
            pose = d["pose"]
        c = classify(pose, tmaps["deep"], r_top["deep"])
        c.update(key=key, candidate=cand, status=row.get("status"),
                 family="schedule" if cand.startswith("sch") else "tracker")
        outcomes[c["outcome"]] += 1
        per_run.append(c)

    n = sum(outcomes.values())
    verdict = {
        "run": str(run),
        "n_runs": len(rows),
        "deep_direct_entries": n,
        "outcomes": dict(outcomes),
        "crossed_the_exit_wall": outcomes.get("crossed the exit wall", 0),
        "status_by_arm": {a: dict(c) for a, c in status_by_arm.items()},
        "asset_contact_runs": {a: f"{contact_by_arm[a]}/{total_by_arm[a]}" for a in total_by_arm},
        "geometry": {a: {"R_top_m": r_top[a], "depth_m": abs(tmaps[a].feature_relief_m(tmaps[a].features[0]))}
                     for a in tmaps},
        "sectors": {"plain_z_m": PLAIN_Z_M, "exit_sector_deg": EXIT_SECTOR_DEG, "entry_sector_deg": ENTRY_SECTOR_DEG},
        "per_run": per_run,
    }

    print(f"{run}: {len(rows)} runs, {n} direct entries into the deep bowl\n")
    for k, v in outcomes.most_common():
        print(f"  {k:34s} {v}")
    print("\n  the runs that got out:")
    for c in per_run:
        if c["outcome"] not in ("never got out", "never entered"):
            print(f"    {c['key']}__{c['candidate']:12s} ({c['status']:9s}) {c['outcome']} at t={c.get('regained_s', 0):.1f}s")
    print("\n  status by arm:")
    for a in ("deep", "shallow", "flat"):
        print(f"    {a:8s} {dict(status_by_arm[a])}   asset contact {contact_by_arm[a]}/{total_by_arm[a]}")
    gate = outcomes.get("crossed the exit wall", 0) == 0
    print(f"\nGATE (no direct entry crosses the exit wall within the tested envelope): "
          f"{'PASS' if gate else 'FAIL'} -- {outcomes.get('crossed the exit wall', 0)}/{n} crossed")
    print("Stated as: no escape over the wall was observed within this tested envelope. NOT 'impossible "
          "under all actions'; reverse gear was never in the control envelope.")

    if args.json:
        Path(args.json).write_text(json.dumps(verdict, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
