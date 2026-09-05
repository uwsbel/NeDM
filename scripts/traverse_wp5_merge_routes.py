#!/usr/bin/env python
"""Merge several planner route files into one Chrono batch, driving each distinct route once.

Picks from different planner variants often coincide (e.g. the round-0 pessimistic pick and a CEM pick
that found nothing better). Identical routes are driven once under a joint candidate name
``a+b+c``; ``traverse_wp5_summarise_picks.py`` splits the Chrono rows back per pick name.
"""
import hashlib, json, sys
from pathlib import Path

out, files = Path(sys.argv[1]), [Path(f) for f in sys.argv[2:]]
merged, n_in, n_out = {}, 0, 0
for f in files:
    for key, cands in json.loads(f.read_text()).items():
        by_hash = merged.setdefault(key, {})
        for c in cands:
            n_in += 1
            h = hashlib.md5(json.dumps([c["waypoints"], c["speeds"]]).encode()).hexdigest()
            if h in by_hash:
                by_hash[h]["candidate"] += "+" + c["candidate"]
            else:
                by_hash[h] = dict(c)
result = {key: list(v.values()) for key, v in merged.items()}
n_out = sum(len(v) for v in result.values())
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result))
print(f"{n_in} picks over {len(result)} layouts -> {n_out} distinct routes -> {out}")
