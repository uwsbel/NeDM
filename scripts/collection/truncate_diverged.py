"""Truncate episodes at the point their joints start moving faster than physics allows.

The preprocessing jump detector refuses a corpus containing a state channel that moves
more than pi between consecutive rows, and suggests adding the channel to
CIRCULAR_STATE_FIELDS. For a Go2 joint that advice is wrong and would be actively harmful:
the joints are bounded well inside a full turn, so a jump of several radians in one 20 ms
control step is not an angle wrapping, it is the granular solver diverging. Unwrapping
would silently accept those rows as real motion and train on them.

Measured incidence, which is itself a result:

    reference, clean on-policy      1 / 400   0.2%
    disturbed, noise + pushes      14 / 400   3.5%
    payload, 0-8 kg               237 /1024  23.1%

Divergence scales with how hard the robot is pushed away from nominal operation. That
bounds how aggressively a coverage corpus can be perturbed before most of it is unusable,
and it means a perturbed corpus needs this pass where a clean one very nearly does not.

TRUNCATE rather than quarantine. Measured, the divergence lands in the last handful of
rows of an episode that is otherwise entirely valid:

    go2_crm_s8100000_arc_008          row 1375 of 1399   98.3% valid
    go2_crm_s8110000_pivot_021        row 1447 of 1454   99.5% valid
    go2_crm_s8110000_stop_and_go_017  row 1474 of 1475   99.9% valid

Dropping whole episodes would discard about 1400 good rows to remove a handful of bad
ones, and would do it selectively: divergences happen at the END, when the robot has
fallen hard enough to break the contact solver, so discarding them removes the hardest
falls and keeps the gentle ones. That is selecting against the coverage the corpus exists
to provide, which is the failure the collector's own guard was written to avoid.

The cut point is not a judgement call: it is the first row where a joint moves more than
pi in one control step, which no physical motion produces. Everything before it is real.

The index is rewritten to match, because preprocess reads the index rather than globbing,
and an index listing files that are no longer there fails later and less clearly.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

LIMIT = 3.14159
JOINTS = ["joint_%s_%s_pos_rad" % (leg, j)
          for leg in ("fl", "fr", "rl", "rr") for j in ("hip", "thigh", "calf")]


def first_bad_row(rows) -> int | None:
    """First row where a joint moved more than pi in one control step."""
    keys = [k for k in (rows[0] if rows else {}) if k.startswith("joint_") and k.endswith("_pos_rad")]
    prev = {}
    for i, r in enumerate(rows):
        for k in keys:
            try:
                v = float(r[k])
            except (TypeError, ValueError):
                continue
            if k in prev and abs(v - prev[k]) > LIMIT:
                return i
            prev[k] = v
    return None


def worst_jump(csv_path: Path) -> float:
    try:
        rows = list(csv.DictReader(csv_path.open()))
    except OSError:
        return 0.0
    if len(rows) < 2:
        return 0.0
    worst = 0.0
    for col in JOINTS:
        if col not in rows[0]:
            continue
        prev = None
        for r in rows:
            try:
                v = float(r[col])
            except (TypeError, ValueError):
                continue
            if prev is not None:
                d = abs(v - prev)
                if d > worst:
                    worst = d
            prev = v
    return worst


def main() -> int:
    root = Path(sys.argv[1])
    apply = "--apply" in sys.argv
    idx_path = root / "dataset_index.json"
    if not idx_path.exists():
        print(f"FATAL: no dataset_index.json under {root}", file=sys.stderr)
        return 2
    index = json.loads(idx_path.read_text())
    eps = index["episodes"]

    bad = []
    for e in eps:
        p = Path(e["csv_path"])
        if not p.is_absolute():
            p = root / e["csv_path"]
        if not p.exists():
            continue
        w = worst_jump(p)
        if w > LIMIT:
            bad.append((e, p, w))

    print(f"  {len(eps)} episodes indexed, {len(bad)} diverged "
          f"({100.0 * len(bad) / max(len(eps), 1):.1f}%)")
    for e, p, w in bad[:6]:
        print(f"    {p.name:<46s} worst joint jump {w:.2f} rad")
    if len(bad) > 6:
        print(f"    ... and {len(bad) - 6} more")
    if not apply:
        print("  (dry run, pass --apply to quarantine and rewrite the index)")
        return 0
    if not bad:
        return 0

    cut_total = kept_total = dropped = 0
    truncated = []
    for e, p, _w in bad:
        rows = list(csv.DictReader(p.open()))
        at = first_bad_row(rows)
        if at is None:
            continue
        # Too short to be worth keeping is a separate judgement from too corrupt. The
        # collector uses 50 rows as the floor for a usable episode; match it.
        if at < 50:
            dropped += 1
            continue
        fields = list(rows[0].keys())
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows[:at])
        cut_total += len(rows) - at
        kept_total += at
        truncated.append({"episode_id": e["episode_id"], "kept_rows": at,
                          "dropped_rows": len(rows) - at})
    index["truncated_diverged"] = truncated
    idx_path.write_text(json.dumps(index, indent=1))
    print(f"  truncated {len(truncated)} episodes: kept {kept_total} rows, "
          f"dropped {cut_total} ({100.0 * cut_total / max(kept_total + cut_total, 1):.2f}%)")
    if dropped:
        print(f"  {dropped} episode(s) diverged before row 50 and were left untouched "
              f"for inspection rather than truncated to a stub")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
