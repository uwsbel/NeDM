"""Concatenate per-shard scorer outputs back into one policy-level record file.

`mkindex.py --shard K --of M` splits one policy's episodes across M scorer
processes; this puts the M result files back together so the output is
indistinguishable from an unsharded run and `compare_to_fleet.py` can read it.

The only thing that can go wrong here is a duplicate or a missing episode_id,
and both are silent: a duplicate makes the mean count an episode twice, a
missing one makes it a mean over a different set than the reference. So both are
checked, and --expect makes the count an assertion rather than a thing you were
supposed to notice in the output.

    python scripts/cluster/merge_score_shards.py \
        --out out/crmtrack_go2_cts_150k_euler.json --expect 80 \
        out/shard_*.json
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("shards", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--expect", type=int, default=None,
                help="episode count the merged file must have")
a = ap.parse_args()

recs = []
for p in a.shards:
    part = json.load(open(p))
    print("  %-60s %3d records" % (p, len(part)))
    recs.extend(part)

ids = [r["episode_id"] for r in recs]
dupes = sorted({i for i in ids if ids.count(i) > 1})
if dupes:
    raise SystemExit("FATAL: %d duplicate episode_id(s) across shards, e.g. %s -- "
                     "the shards overlap and the mean would double-count"
                     % (len(dupes), dupes[:3]))
if a.expect is not None and len(recs) != a.expect:
    raise SystemExit("FATAL: merged %d records, expected %d -- a shard is missing or "
                     "one of them died, and a mean over the wrong set is not a result"
                     % (len(recs), a.expect))

recs.sort(key=lambda r: r["episode_id"])
json.dump(recs, open(a.out, "w"), indent=1)

ok = [r for r in recs if "mae_vx" in r]
print("wrote %s  (%d records, %d scored)" % (a.out, len(recs), len(ok)))
if ok:
    print("  mean mae_vx %.6f" % (sum(r["mae_vx"] for r in ok) / len(ok)))
