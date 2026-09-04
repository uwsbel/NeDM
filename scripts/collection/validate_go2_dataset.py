"""Consumer-side gates for the Go2 rigid/CRM datasets.

Every metadata defect found during the Go2 collection was invisible to reading the
collector and obvious the moment something consumed the data. This script is that
consumer, run deliberately and early: it reads a merged dataset the way preprocess,
the trainer, and the RL reference builder read it, and asserts the properties those
three actually depend on.

The gates are ordered by the stage they protect:

  G1  schema parity     every key a consumer reads is a key the collector writes
  G2  identity          episode ids are unique across the merged index
                        (preprocess.py:398 also raises on this, but only after
                        the collection it would invalidate has already run)
  G3  family cardinality  scenario_family distinguishes what command_family does
  G4  family agreement  scenario_family is consistent with terrain + command_family
  G5  index/json parity the two copies of a shared field agree
  G6  split             both splits exist and the ratio is in band
  G7  val coverage      each terrain's validation split contains every family
  G8  ROLLOUT COVERAGE  the episodes the trainer would SELECT cover every family
  G1b HMMWV parity      (warn) fields the HMMWV index carries and ours does not
  G9  boundary flag     terminated_near_boundary is present and matches status
  G10 rows              csv exists and its row count matches the index

G7 and G8 are deliberately both here and they are not the same check. A validation
split can cover all eight families perfectly while the trainer still draws all
twelve of its rollout episodes from one of them -- producing coverage and consuming
it are different guarantees. G8 is the one that matters, because rollout error at
the selection horizon is what every deployed checkpoint is chosen on.

Both run PER TERRAIN, because that is how the trainer evaluates: rollout_eval
loops `for domain in domains` (trainer.py:721) and selects once per processed
dataset. Checking the merged pool would demand twelve distinct families out of
sixteen and fail on data the trainer would handle correctly.

G1 exists because of a narrower rule than "audit every .get() with no default".
preprocess.py:233 reads `frames_path` with exactly that pattern and is harmless,
because it checks the absence and raises. A missing key is dangerous only where
ABSENCE AND A LEGITIMATE VALUE COLLAPSE TO THE SAME BRANCH: None is not a path, so
frames_path cannot collapse, while None and False are both "keep this episode", so
terminated_near_boundary collapses perfectly. A wrong value is at least present to
be noticed; this one is not. G1 therefore lists the keys whose absence is
indistinguishable from a meaningful value, not every key a consumer touches.

THE GATE READS THE RAW INDEX; THE TRAINER SELECTS FROM THE PROCESSED CACHE. Those
are the same episodes in the same order, so G8's predicted selection is the
trainer's actual one -- but only under two conditions, because the round-robin
does `family_episodes.pop(0)` and is order-sensitive within a family, not merely
membership-sensitive:

  1. PASS --dataset-root IN THE SAME ORDER preprocess receives its roots.
     preprocess.py:380-396 appends per root, then per episode within each root,
     with no sort and no shuffle anywhere; split partitioning at :405 preserves
     relative order. So order is carried exactly, and only the root order is
     yours to get wrong.
  2. DO NOT USE --max-episodes-per-split. preprocess.py:409 truncates each split
     with a bare slice, and the gate cannot see that truncation.

Outside those two, "the raw set would select 8/8 per terrain" and "the trainer
selected 8/8 per terrain" are the same claim. Inside them they are not, and only
the second one selects a checkpoint -- so re-ask the question against
split_metadata["scenario_families"] after preprocess, where the answer is measured
rather than predicted.

Pure stdlib on purpose: this must run on a box with no torch and no CUDA, before
anything expensive is scheduled.

Run:  python scripts/collection/validate_go2_dataset.py --dataset-root <dir> [--dataset-root <dir> ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
failures: list[str] = []
warnings: list[str] = []


def report(name: str, ok: bool, detail: str, *, warn_only: bool = False,
           inconclusive: str = "") -> None:
    """Print one gate. `inconclusive` suppresses a PASS that cannot be trusted.

    A gate whose input is already known-broken must not report PASS: an earlier
    failure can make a later check read clean off data it never actually saw. That
    is the same shape as the defects this script exists to catch, so it is spelled
    out rather than left to the reader to notice.
    """
    if inconclusive:
        print(f"[????] {name}: not meaningful -- {inconclusive}")
        warnings.append(f"{name}: inconclusive ({inconclusive})")
        return
    tag = PASS if ok else (WARN if warn_only else FAIL)
    print(f"[{tag}] {name}: {detail}")
    if not ok:
        (warnings if warn_only else failures).append(f"{name}: {detail}")


# Keys the downstream consumers read off an index entry. Sourced by grepping
# ep[...] / ep.get(...) in preprocess.py, trainer.py, dataset.py and the two RL
# reference builders, minus the ones those modules inject themselves (_dataset_*)
# and the runtime arrays they attach (states/actions/rollout).
CONSUMER_INDEX_KEYS = {
    "episode_id",                 # everywhere
    "scenario_name",
    "scenario_family",            # trainer.py:798, build_combined_*:135
    "split",                      # build_combined_*:129
    "csv_path",
    "rows",                       # build_combined_*:131
    "terrain_label",
    "terminated_near_boundary",   # build_combined_*:133 -- absent => filter never fires
}

# scenario_family must vary with command_family. These are the eight command
# families the driver schedules; the check is cardinality, not membership, so a
# renamed family does not fail the gate spuriously.
MIN_EXPECTED_FAMILIES = 2

# Keys the HMMWV index entry carries. Nothing downstream reads warmup_s off an
# index today, so its absence is a warning rather than a gate -- but the two
# schemas were meant to match, and a field missing for no reason is how the next
# one hides. terminated_near_boundary was invisible for exactly that long.
HMMWV_INDEX_KEYS = {
    "episode_id", "scenario_name", "scenario_family", "split", "csv_path",
    "rows", "duration_s", "warmup_s", "terminated_near_boundary",
}


def load_index(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index_path = root / "dataset_index.json"
    if not index_path.exists():
        raise SystemExit(f"no dataset_index.json under {root}")
    index = json.loads(index_path.read_text())
    for ep in index["episodes"]:
        ep["_root"] = str(root)
    return index, index["episodes"]


def episode_json(ep: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(ep["_root"]) / "episodes" / f"{ep['episode_id']}.json"
    return json.loads(path.read_text()) if path.exists() else None


def select_rollout_episodes(episodes: list[dict[str, Any]], max_episodes: int) -> list[dict[str, Any]]:
    """Mirror of Trainer._select_rollout_episodes (src/nedm/training/trainer.py:795).

    Vendored rather than imported because trainer.py pulls in torch and this gate
    must run before torch is available. If that function changes, this mirror is
    stale -- which is why G8 prints the selection it produces rather than only a
    verdict, so a drift shows up as a number that stops matching the trainer's log.
    The mirror does not have to be trusted; it has to be checkable.

    THE MIRROR COVERS THE SELECTION ALGORITHM ONLY, and deliberately omits the
    trainer's family-distribution logging, which sits between the loop and the
    return and cannot affect what is selected. So the two are no longer textually
    identical and that difference is expected: diff the bucket/sort/round-robin/
    pop/remove-when-empty body, not the whole function. A drift warning that fires
    on a change which cannot cause drift stops being read.
    """
    by_family: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        by_family.setdefault(episode["scenario_family"], []).append(episode)
    families = sorted(by_family)
    selected: list[dict[str, Any]] = []
    family_index = 0
    while len(selected) < max_episodes and families:
        family = families[family_index % len(families)]
        family_episodes = by_family[family]
        if family_episodes:
            selected.append(family_episodes.pop(0))
        if not family_episodes:
            families.remove(family)
            family_index -= 1
        family_index += 1
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, action="append", required=True,
                        help="dataset root (repeat to validate the merged set across boxes)")
    parser.add_argument("--rollout-episodes", type=int, default=12,
                        help="max_episodes the trainer selects on (must match the training config)")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-tolerance", type=float, default=0.06)
    parser.add_argument("--check-rows", action="store_true",
                        help="open every CSV and count rows (slow; off by default)")
    args = parser.parse_args()

    episodes: list[dict[str, Any]] = []
    for root in args.dataset_root:
        _, eps = load_index(root)
        episodes.extend(eps)
        print(f"  loaded {len(eps):5d} episodes from {root}")
    print(f"  {len(episodes)} episodes total across {len(args.dataset_root)} root(s)\n")
    if not episodes:
        raise SystemExit("no episodes")

    metas = {ep["episode_id"]: episode_json(ep) for ep in episodes}
    # Also read every episode JSON straight off disk. `metas` is keyed by episode
    # id, so colliding ids silently collapse -- which is precisely the case G2
    # exists to catch, and it would leave every family statistic below computed
    # from whichever episode happened to be written last.
    on_disk = [json.loads(p.read_text())
               for root in args.dataset_root
               for p in sorted((root / "episodes").glob("*.json"))]

    # ---- G1 schema parity -------------------------------------------------
    present = set(episodes[0])
    absent = sorted(k for k in CONSUMER_INDEX_KEYS if k not in present)
    report("G1 schema parity", not absent,
           "index entries carry every key the consumers read" if not absent
           else f"index entries are MISSING {absent} -- a consumer reads these with "
                f".get(), so absence becomes a silent default, not an error")

    drift = sorted(HMMWV_INDEX_KEYS - present - set(absent))
    report("G1b HMMWV parity", not drift,
           "index entry matches the HMMWV schema it was written against" if not drift
           else f"{drift} present in the HMMWV index and absent from ours, with no live consumer -- "
                f"harmless today, and the same gap that hid terminated_near_boundary",
           warn_only=True)

    # ---- G2 identity ------------------------------------------------------
    ids = Counter(ep["episode_id"] for ep in episodes)
    dupes = [i for i, n in ids.items() if n > 1]
    report("G2 identity", not dupes,
           f"{len(ids)} unique ids over {len(episodes)} episodes" if not dupes
           else f"{len(dupes)} COLLIDING ids ({dupes[:4]}...) -- episodes overwrite each other")

    # ---- G3 family cardinality -------------------------------------------
    fams = {ep.get("scenario_family") for ep in episodes}
    # The invariant after repair is exact: one scenario_family per distinct
    # (terrain, command_family) pair actually collected. Counting command families
    # alone would understate it, since the same family on two terrains is two
    # populations to the trainer.
    pairs = {(m.get("terrain_label"), m.get("command_family")) for m in on_disk}
    ok3 = len(fams) == len(pairs) and len(fams) >= MIN_EXPECTED_FAMILIES
    blocked = f"G2 failed: {len(dupes)} colliding ids collapse the per-episode records" if dupes else ""
    report("G3 family cardinality", ok3, inconclusive=blocked, detail=
           f"{len(fams)} scenario_family values for {len(pairs)} (terrain, command_family) pairs" if ok3
           else f"scenario_family has {len(fams)} value(s) but {len(pairs)} (terrain, command_family) "
                f"pairs were collected -- family-stratified selection degenerates toward list order")


    # ---- G4 family agreement ---------------------------------------------
    bad4 = [ep["episode_id"] for ep in episodes
            if (m := metas.get(ep["episode_id"])) and m.get("command_family")
            and m["command_family"] not in (ep.get("scenario_family") or "")]
    report("G4 family agreement", not bad4,
           "scenario_family embeds command_family on every episode" if not bad4
           else f"{len(bad4)} episodes whose scenario_family does not name their command_family "
                f"(e.g. {bad4[0]})")

    # ---- G5 index/json parity --------------------------------------------
    shared = ["scenario_family", "split", "rows", "terrain_label"]
    bad5 = [(ep["episode_id"], k) for ep in episodes if (m := metas.get(ep["episode_id"]))
            for k in shared if k in m and m[k] != ep.get(k)]
    report("G5 index/json parity", not bad5,
           f"index and episode JSON agree on {shared}" if not bad5
           else f"{len(bad5)} disagreements between the two copies (e.g. {bad5[0]}) -- "
                f"preprocess reads the index, dataset.py surfaces the JSON")

    # ---- G6 split ---------------------------------------------------------
    splits = Counter(ep.get("split") for ep in episodes)
    val_frac = splits.get("val", 0) / len(episodes)
    ok6 = splits.get("val", 0) > 0 and abs(val_frac - args.val_ratio) <= args.val_tolerance
    report("G6 split", ok6,
           f"train={splits.get('train',0)} val={splits.get('val',0)} ({val_frac:.1%})" if ok6
           else f"val fraction {val_frac:.1%} outside {args.val_ratio:.0%}+/-{args.val_tolerance:.0%} "
                f"(counts {dict(splits)}) -- a zero here means no validation set exists at all")

    # ---- G7 val coverage --------------------------------------------------
    # PER TERRAIN, because that is how the trainer evaluates. rollout_eval loops
    # `for domain in domains` (trainer.py:721) and calls the selection once per
    # processed dataset, so flat and crm each get their own twelve drawn from
    # their own families. Checking the merged pool would demand twelve distinct
    # families out of sixteen -- a requirement the trainer never actually faces.
    by_terrain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_terrain[ep.get("terrain_label") or "?"].append(ep)

    missing7: list[str] = []
    for terr, terr_eps in sorted(by_terrain.items()):
        terr_fams = {ep.get("scenario_family") for ep in terr_eps}
        terr_val_fams = {ep.get("scenario_family") for ep in terr_eps if ep.get("split") == "val"}
        gap = sorted(terr_fams - terr_val_fams)
        print(f"        {terr:6s} {len(terr_val_fams)}/{len(terr_fams)} families in val"
              + (f", missing {gap}" if gap else ""))
        missing7.extend(gap)
    report("G7 val coverage", not missing7,
           "every terrain's validation split covers every family it collected" if not missing7
           else f"{len(missing7)} family/terrain combinations absent from validation: {missing7[:4]}")

    # ---- G8 ROLLOUT COVERAGE ---------------------------------------------
    # The gate that matters, and the one G7 cannot stand in for. A validation
    # split can cover every family while the round-robin still fails to reach
    # them inside max_episodes. Scored against the families that EXIST on that
    # terrain, not the families that survived into val -- scoring against the
    # pool it drew from would report full coverage of an impoverished pool.
    bad8: list[str] = []
    for terr, terr_eps in sorted(by_terrain.items()):
        terr_fams = {ep.get("scenario_family") for ep in terr_eps}
        terr_val = [ep for ep in terr_eps if ep.get("split") == "val"]
        selected = select_rollout_episodes(list(terr_val), args.rollout_episodes)
        sel_fams = Counter(ep["scenario_family"] for ep in selected)
        reachable = min(args.rollout_episodes, len(terr_fams))
        ok = len(selected) > 0 and len(sel_fams) >= reachable
        print(f"        {terr:6s} {len(selected):2d} selected spanning "
              f"{len(sel_fams)}/{len(terr_fams)} families (need {reachable})"
              + ("" if ok else "   <-- FAIL"))
        for fam, n in sorted(sel_fams.items()):
            print(f"          {fam:44s} {n}")
        if not ok:
            bad8.append(f"{terr}: {len(sel_fams)}/{len(terr_fams)} in {len(selected)} episodes"
                        + (" (nothing to select from)" if not selected else ""))
    report("G8 rollout coverage", not bad8,
           f"every terrain's {args.rollout_episodes} selected episodes span all its families"
           if not bad8
           else "the episodes every checkpoint is SELECTED on do not span the collected "
                f"families: {bad8}")

    # ---- G9 boundary flag -------------------------------------------------
    if "terminated_near_boundary" in present:
        bad9 = [ep["episode_id"] for ep in episodes if (m := metas.get(ep["episode_id"]))
                and bool(ep.get("terminated_near_boundary")) != (m.get("status") == "bed_boundary")]
        rate = sum(1 for ep in episodes if ep.get("terminated_near_boundary")) / len(episodes)
        report("G9 boundary flag", not bad9,
               f"flag matches status on every episode, boundary rate {rate:.1%}" if not bad9
               else f"{len(bad9)} episodes whose flag disagrees with their status (e.g. {bad9[0]})")
    else:
        report("G9 boundary flag", False,
               "field absent from the index, so build_combined_flat_crm_rl_references.py:133 "
               "filters nothing -- derive it from the episode JSON status in the repair pass")

    # ---- G10 rows ---------------------------------------------------------
    if args.check_rows:
        bad10 = []
        for ep in episodes:
            path = Path(ep["_root"]) / ep["csv_path"]
            if not path.exists():
                bad10.append((ep["episode_id"], "missing csv"))
                continue
            with path.open(newline="", encoding="utf-8") as fh:
                n = sum(1 for _ in csv.reader(fh)) - 1
            if n != ep["rows"]:
                bad10.append((ep["episode_id"], f"{n} rows on disk vs {ep['rows']} in index"))
        report("G10 rows", not bad10,
               f"all {len(episodes)} CSVs present with matching row counts" if not bad10
               else f"{len(bad10)} mismatches (e.g. {bad10[0]})")
    else:
        print("[SKIP] G10 rows: pass --check-rows to open every CSV")

    per_family = Counter(ep.get("scenario_family") for ep in episodes)
    print("\n  episodes per family:")
    for fam, n in sorted(per_family.items()):
        print(f"    {fam:44s} {n:5d}")

    print()
    for w in warnings:
        print(f"WARN {w}")
    if failures:
        print(f"\n{len(failures)} GATE(S) FAILED -- do not preprocess:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
