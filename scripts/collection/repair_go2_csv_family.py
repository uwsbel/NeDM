"""Make the CSV's scenario_family column agree with the episode JSON.

THE DEFECT. The first repair pass corrected scenario_family in the per-episode
JSON and the dataset index -- it had been a CONSTANT, one value per terrain for
eight families -- but the CSVs carry the same column and were left alone. So
980 of 1120 episodes have a CSV that says `go2_flat_constant_command` while
their JSON says `arc`, `weave`, `pivot` and so on. Only the `constant` episodes
agree, because their old value already equalled the corrected one:

    flat  968 - 847 mismatches = 121 = the constant episodes
    crm   152 - 133 mismatches =  19 = the constant episodes

That arithmetic is what identifies the cause; a mismatch count equal to
total-minus-constant is this and not corruption. Defect and the arithmetic check
both predicted by dorm-pc from its own half before this side had looked.

NOTHING DOWNSTREAM IS AFFECTED, AND THAT IS WHY IT SURVIVED. preprocess.py:280
reads scenario_family from the episode ENTRY, not the CSV, so the processed
caches, the rollout family selection, the reference sets and the trained model
all carry the correct eight families -- verified. The stale column is a trap for
the next reader rather than a live fault, which is exactly the class of defect
this dataset has been full of.

WHAT IS PROVEN, NOT ASSUMED. Every physics column is hashed before and after with
the id columns stripped, and the two hashes must match per episode. The strip
list now has FOUR entries, not three: scenario_family joins episode_id,
scenario_name and split, because a column being rewritten cannot also be part of
the invariant that proves nothing else was rewritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

# FOUR columns. scenario_family is added because this pass rewrites it.
ID_COLUMNS = ["episode_id", "scenario_name", "split", "scenario_family"]


def physics_digest(path: Path) -> str:
    """SHA-256 of every column except the id columns, header included."""
    h = hashlib.sha256()
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        keep = [c for c in (reader.fieldnames or []) if c not in ID_COLUMNS]
        h.update("|".join(keep).encode())
        for row in reader:
            h.update("|".join(row[c] for c in keep).encode())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--apply", action="store_true",
                    help="Without this it only reports. The dry run COMPARES; it does "
                         "not report an empty change list without looking, which is a "
                         "mistake this repo has made and retracted.")
    a = ap.parse_args()

    total = changed = 0
    for root in a.roots:
        episodes = root / "episodes"
        metas = [p for p in sorted(episodes.glob("*.json"))
                 if not p.name.endswith(".config.json")]
        n_mismatch = 0
        for meta_path in metas:
            meta = json.loads(meta_path.read_text())
            csv_path = episodes / f"{meta['episode_id']}.csv"
            if not csv_path.is_file():
                continue
            total += 1
            with csv_path.open(newline="") as fh:
                reader = csv.DictReader(fh)
                fields = list(reader.fieldnames or [])
                rows = list(reader)
            if "scenario_family" not in fields:
                continue
            want = meta["scenario_family"]
            if all(r["scenario_family"] == want for r in rows):
                continue
            n_mismatch += 1
            if not a.apply:
                continue
            before = physics_digest(csv_path)
            for r in rows:
                r["scenario_family"] = want
            tmp = csv_path.with_suffix(".csv.tmp")
            with tmp.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            after = physics_digest(tmp)
            if before != after:
                tmp.unlink()
                raise SystemExit(
                    f"{csv_path.name}: physics columns changed ({before[:12]} -> "
                    f"{after[:12]}). Refusing to write. This pass must touch ONE column.")
            tmp.replace(csv_path)
            changed += 1
        print(f"{root}: {len(metas)} episodes, {n_mismatch} CSVs disagreed with their JSON")

    print(f"\n{total} episodes examined, {changed} rewritten"
          + ("" if a.apply else "  (dry run -- pass --apply)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
