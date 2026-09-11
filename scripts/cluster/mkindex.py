"""Slice the cluster episode index into a subset or an array shard.

The scorer takes one --index and scores every episode in it, so selecting which
episodes a job runs is done by writing a smaller index, not by a scorer flag.
Two uses:

  --limit N            first N episodes, for smoke tests
  --shard K --of M     deterministic round-robin shard K of M, for a job array
                       that splits ONE policy across M nodes

Round-robin rather than contiguous blocks, because the episodes are grouped by
command family in index order: a contiguous split would give one shard all the
pivots and another all the weaves, so a per-shard mean would be a mean over a
different distribution and shard wall-times would differ by the family cost.

episode_id is NOT rewritten -- it is derived from the last four path components
of csv_path by the scorer, and those are preserved from the NVIDIA fleet's
layout so results pair one-to-one across machines.
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--shard", type=int, default=None)
ap.add_argument("--of", type=int, default=None)
ap.add_argument("--split", default="val")
a = ap.parse_args()

d = json.load(open(a.index))
eps = [e for e in d["episodes"] if e.get("split") == a.split] or d["episodes"]

if (a.shard is None) != (a.of is None):
    raise SystemExit("--shard and --of must be given together")
if a.shard is not None:
    if not (0 <= a.shard < a.of):
        raise SystemExit("--shard must be in [0, --of)")
    eps = [e for i, e in enumerate(eps) if i % a.of == a.shard]
if a.limit is not None:
    eps = eps[:a.limit]

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump({"episodes": eps}, open(a.out, "w"), indent=1)
print("wrote %s  (%d episodes)" % (a.out, len(eps)))
