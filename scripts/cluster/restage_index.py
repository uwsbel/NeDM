"""Repoint a CRM episode index at a staged sidecar tree on another machine.

The scorer never opens an episode CSV. `score_crm_tracking.py` reads
`csv_path[:-4] + ".json"` and `csv_path[:-4] + ".config.json"` beside it, and
uses `csv_path` itself only as a dictionary key. So porting the scorer to a new
machine needs a few MB of sidecars, not the ~200 GB corpus -- but the directory
structure above them has to be reproduced exactly, because `episode_id` is
derived from the last four path components:

    uid(csv_path) = "_".join(components[-4:-2] + [stem])

and an id that differs by one component pairs with nothing. A paired comparison
against the fleet is the entire point of running the scorer on a second machine,
so getting those four components wrong does not fail loudly -- it produces a
clean-looking file that shares zero episode ids with the reference.

Hence: keep the tail of the path, replace the head. Do not rewrite the index by
hand, and do not rewrite `episode_id` in the index either (the scorer ignores
that field and recomputes the id from the path).

    python scripts/cluster/restage_index.py \
        --index  stage/score_subset_index.orig.json \
        --root   /srv/home/kasha2/nedm/stage/episodes \
        --out    stage/score_subset_index.euler.json

Verifies every sidecar the scorer will open actually exists under the new root,
and refuses to write an index that would send it looking for a missing file --
a missing sidecar surfaces as an unexplained per-episode exception 80 times over
otherwise.
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True, help="index whose csv_path values point at the source machine")
ap.add_argument("--root", required=True, help="staged sidecar root on THIS machine")
ap.add_argument("--out", required=True)
ap.add_argument("--keep", type=int, default=5,
                help="path components to preserve below --root. Must be >= 4, "
                     "because episode_id is built from the last four; 5 is the "
                     "default so the staged tree also keeps the datasets/ level "
                     "and reads the same as the fleet's layout.")
ap.add_argument("--split", default=None, help="keep only this split (default: keep all)")
a = ap.parse_args()

if a.keep < 4:
    raise SystemExit("--keep must be at least 4: episode_id is derived from the last "
                     "four components of csv_path and a shorter tail cannot reproduce it")

d = json.load(open(a.index))
eps = d["episodes"]
if a.split:
    eps = [e for e in eps if e.get("split") == a.split]

root = os.path.abspath(a.root)
missing = []
for e in eps:
    tail = os.path.normpath(e["csv_path"]).split(os.sep)[-a.keep:]
    new = os.path.join(root, *tail)
    e["csv_path"] = new
    for suffix in (".json", ".config.json"):
        side = new[:-4] + suffix
        if not os.path.exists(side):
            missing.append(side)

if missing:
    print("MISSING %d sidecar(s) under %s; first 10:" % (len(missing), root))
    for p in missing[:10]:
        print("   " + p)
    raise SystemExit("refusing to write an index that points at files that are not there")

os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
json.dump({"episodes": eps}, open(a.out, "w"), indent=1)
print("wrote %s  (%d episodes, all sidecars present)" % (a.out, len(eps)))


def uid(cp):
    q = os.path.normpath(cp).split(os.sep)
    stem = os.path.splitext(q[-1])[0]
    return "_".join(q[-4:-2] + [stem]) if len(q) >= 4 else stem


ids = [uid(e["csv_path"]) for e in eps]
if len(set(ids)) != len(ids):
    raise SystemExit("FATAL: episode ids are not unique after restaging -- pairing "
                     "would be silently wrong")
print("episode_id sample: %s" % ids[0])
