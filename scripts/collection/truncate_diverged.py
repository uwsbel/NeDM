#!/usr/bin/env python3
"""Truncate episodes at the first physically impossible joint jump.

The collector's `diverged` flag did not catch these: episode
go2_crm_s10100000_lateral_010 has its rr_calf joint spiral to -33 rad with body velocity
swinging to -4 m/s from t=7.13 s, and its metadata says diverged=false. Preprocess caught
it instead, as "channels jump by more than pi, add them to CIRCULAR_STATE_FIELDS" -- which
would have been exactly the wrong fix, since a leg joint moving 3.5 rad in one 10 ms step
is not a wrapped angle, it is a blown-up solve.

Truncate rather than drop: the pre-divergence rows are good data and there are far more of
them than not. Rows after the first jump are discarded because everything downstream of a
solver blow-up is meaningless, including any later rows that look plausible again.

Rate measured on this corpus: 2 of 388 episodes, spread across soil bins rather than
concentrated in the soft ones, so this is a background rate and not an artefact of the
soil sweep.

    truncate_diverged.py <dataset_root> [--apply]
"""
import csv, glob, json, math, os, sys

root = sys.argv[1]
apply = "--apply" in sys.argv
JOINTS = [f"joint_{l}_{s}_pos_rad" for l in ("rr", "rl", "fr", "fl")
          for s in ("hip", "thigh", "calf")]
touched = []
for p in sorted(glob.glob(f"{root}/episodes/*.csv")):
    rows = list(csv.DictReader(open(p)))
    if not rows:
        continue
    cut = None
    for j in JOINTS:
        if j not in rows[0]:
            continue
        prev = float(rows[0][j])
        for i in range(1, len(rows)):
            v = float(rows[i][j])
            if abs(v - prev) > math.pi:
                cut = i if cut is None else min(cut, i)
                break
            prev = v
    if cut is None:
        continue
    touched.append((os.path.basename(p), len(rows), cut))
    if not apply:
        continue
    fields = list(rows[0].keys())
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows[:cut])
    jp = p[:-4] + ".json"
    if os.path.exists(jp):
        m = json.load(open(jp))
        m["rows"] = cut
        m["diverged"] = True
        m["diverged_at_s"] = round(cut * float(m.get("dt_s", 0.01)), 3)
        m["truncated_by"] = ("truncate_diverged.py: first joint-position step above pi, "
                             "which a leg joint cannot physically take")
        json.dump(m, open(jp, "w"), indent=2)

for name, had, cut in touched:
    print(f"  {name}: {had} rows -> {cut} ({had-cut} discarded)")
print(f"  {'TRUNCATED' if apply else 'would truncate'} {len(touched)} episodes")
