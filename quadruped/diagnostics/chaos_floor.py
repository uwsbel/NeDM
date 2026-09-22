#!/usr/bin/env python3
"""How predictable is the robot at all? The floor any surrogate is measured against.

Shards 0-15 of the v2 corpus reused the v1 seeds, so each such episode ran twice in
Chrono from the same start with the same commands, pushes and injected noise. The dynamics
code is identical between the two (v2 moved only WHEN a row is read), so the twins differ
only by non-associative GPU accumulation in the granular solver -- a perturbation at the
level of float rounding. How fast they drift apart is how fast Chrono diverges from ITSELF,
and no model, however good, can predict the exact pose further out than that.

Scored with the surrogate's own metric: planar RMSE over the horizon divided by the
reference run's maximum displacement from the start (train.rollout_errdist). Measured from
t = 0 of each episode, because that is the only instant the twins share a state; later
starts have already diverged. Twins are kept only if their starting positions agree to
5 mm, which drops episodes whose bed was planned differently between the two runs.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HORIZONS = (0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
DT = 0.01


def load_pos(path):
    with path.open() as fh:
        r = csv.DictReader(fh)
        xy = [(float(row["pos_x_m"]), float(row["pos_y_m"]), float(row["time_s"])) for row in r]
    return np.array(xy)


def main():
    d1, d2 = Path(sys.argv[1]), Path(sys.argv[2])
    m1 = json.loads((d1 / "manifest.json").read_text())
    m2 = json.loads((d2 / "manifest.json").read_text())
    off1 = {p["shard"].replace("go2_crm_v1_", ""): p["episode_offset"] for p in m1["merged_from"]}
    off2 = {p["shard"].replace("go2_crm_v2_", ""): p["episode_offset"] for p in m2["merged_from"]}
    common = sorted(set(off1) & set(off2), key=lambda s: int(s[1:]))
    print(f"shards in both: {common}")
    n1, n2 = d1.name, d2.name
    errs = {h: [] for h in HORIZONS}
    kept = dropped = 0
    for s in common:
        for k in range(50):
            f1 = d1 / "episodes" / f"{n1}_{off1[s] + k:04d}_s0.csv"
            f2 = d2 / "episodes" / f"{n2}_{off2[s] + k:04d}_s0.csv"
            if not (f1.exists() and f2.exists()):
                continue
            a, b = load_pos(f1), load_pos(f2)
            if abs(a[0, 2]) > 1e-6 or abs(b[0, 2]) > 1e-6:
                continue        # s0 must start at t = 0
            if math.hypot(a[0, 0] - b[0, 0], a[0, 1] - b[0, 1]) > 0.005:
                dropped += 1
                continue
            kept += 1
            n = min(len(a), len(b))
            ra = a[:n, :2] - a[0, :2]
            rb = b[:n, :2] - b[0, :2]
            for h in HORIZONS:
                m = int(round(h / DT))
                if n <= m:
                    continue
                err = np.sqrt(((ra[1:m + 1] - rb[1:m + 1]) ** 2).sum(1).mean())
                dist = np.sqrt((ra[1:m + 1] ** 2).sum(1)).max()
                if dist > 1e-6:
                    errs[h].append(err / dist)
    print(f"twin episodes kept {kept}, dropped {dropped} (start positions differ > 5 mm)\n")
    print(f"{'horizon':>8s} {'n':>5s} {'median':>8s} {'mean':>8s} {'p90':>8s}")
    for h in HORIZONS:
        e = np.array(errs[h])
        if len(e):
            print(f"{h:7.1f}s {len(e):5d} {np.median(e):8.3f} {e.mean():8.3f} {np.percentile(e, 90):8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
