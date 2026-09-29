"""Recount the traversing study's headline numbers from the compact result tables.

Reads every CSV in ``traversing/results/`` and the expectations in ``traversing/results/expected_counts.json``,
prints n / goal / unsafe / safe_goal for every evaluated arm of every table, then checks each expectation
(counts exactly, means and medians within the stated tolerance) and exits non-zero listing any mismatch.

    python traversing/scripts/recount_milestones.py               # per-arm tables, numeric checks, verdict
    python traversing/scripts/recount_milestones.py --check-only  # mismatches and the verdict only

Each arm cell holds one outcome code: S = reached the goal, no unsafe event; s = reached the goal with an unsafe
event; U = goal not reached, unsafe; F = goal not reached, not unsafe; - = arm not run on this task.
n counts every cell except '-', goal = S + s, unsafe = s + U, safe_goal = S.

Wide tables have one row per task and one column per arm (every column whose cells are all outcome codes).
Long tables have an ``outcome_code`` column; their arms are identified by the file's ``summary_group_by`` columns.
Standard library only (Python 3.8+); see traversing/results/README.md for what every table and column means.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
CODES = {"S", "s", "U", "F", "-"}


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def counts(cells):
    """n / goal / unsafe / safe_goal of a list of outcome codes ('-' = not run, left out of n)."""
    unknown = set(cells) - CODES
    if unknown:
        raise ValueError(f"cells that are not outcome codes: {sorted(unknown)}")
    run = [c for c in cells if c != "-"]
    return {"n": len(run), "goal": sum(c in ("S", "s") for c in run),
            "unsafe": sum(c in ("s", "U") for c in run), "safe_goal": run.count("S")}


def select(rows, filters):
    """Rows whose cells equal every filter value (exact text match)."""
    for key in filters:
        if rows and key not in rows[0]:
            raise KeyError(f"filter column {key!r} not in table")
    return [r for r in rows if all(r[k] == v for k, v in filters.items())]


def arm_summary(rows, group_by):
    """{arm label: counts}. Long tables (with outcome_code): one entry per distinct group_by value;
    wide tables: one entry per column whose cells are all outcome codes."""
    if "outcome_code" in rows[0]:
        groups = {}
        for r in rows:
            groups.setdefault(" / ".join(r[c] for c in group_by), []).append(r["outcome_code"])
        return {label: counts(cells) for label, cells in groups.items()}
    arms = [c for c in rows[0] if all(r[c] in CODES for r in rows)]
    return {c: counts([r[c] for r in rows]) for c in arms}


def evaluate(rows, e):
    """The statistic an expectation asks for, as a dict with the same keys as e['expected']."""
    sel = select(rows, e.get("filters", {}))
    kind = e["kind"]
    if kind == "outcome_counts":
        return counts([r[e["column"]] for r in sel])
    if kind == "count_where":
        return {"value": len(sel)}
    values = [float(r[e["column"]]) for r in sel]
    if not values:
        raise ValueError("no rows match the filters")
    if kind == "mean":
        return {"value": statistics.fmean(values)}
    if kind == "median":
        return {"value": statistics.median(values)}
    raise ValueError(f"unknown kind {kind!r}")


def describe(e):
    filt = ", ".join(f"{k}={v}" for k, v in e.get("filters", {}).items())
    return f"[{e['claim_id']}] {e['kind']}({e['column']}){' where ' + filt if filt else ''}"


def check(rows, e):
    """(computed, list of problems) for one expectation; an empty list means it matches."""
    try:
        got = evaluate(rows, e)
    except (KeyError, ValueError) as err:
        return None, [str(err)]
    tol = e.get("tolerance", 0)
    problems = [f"{k}: got {got[k]:g}, expected {v:g}" for k, v in e["expected"].items()
                if abs(got[k] - v) > tol]
    # means/medians also carry the study's own full-precision value; the CSV must reproduce it too
    if "study_value" in e and abs(got["value"] - e["study_value"]) > tol:
        problems.append(f"value: got {got['value']:g}, study value {e['study_value']:g}")
    return got, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-only", action="store_true", help="print only mismatches and the final verdict")
    parser.add_argument("--results", type=Path, default=RESULTS, help="folder with the CSVs and expected_counts.json")
    args = parser.parse_args(argv)

    spec = json.loads((args.results / "expected_counts.json").read_text(encoding="utf-8"))["files"]
    csvs = sorted(p.name for p in args.results.glob("*.csv"))
    mismatches, n_checked = [], 0
    for name in sorted(set(csvs) - set(spec)):
        print(f"note: {name} has no expectations in expected_counts.json")

    for name, fspec in spec.items():
        path = args.results / name
        if not path.exists():
            mismatches.append(f"{name}: file missing")
            continue
        rows = load_csv(path)
        if len(rows) != fspec["rows"]:
            mismatches.append(f"{name}: {len(rows)} data rows, expected {fspec['rows']}")

        if not args.check_only:
            summary = arm_summary(rows, fspec.get("summary_group_by", ["arm"]))
            width = max(len(label) for label in summary)
            print(f"\n{name}  ({len(rows)} rows)")
            print(f"  {'arm':<{width}}  {'n':>5} {'goal':>5} {'unsafe':>6} {'safe_goal':>9}")
            for label, c in summary.items():
                print(f"  {label:<{width}}  {c['n']:>5} {c['goal']:>5} {c['unsafe']:>6} {c['safe_goal']:>9}")

        n_ok = 0
        for e in fspec["expectations"]:
            got, problems = check(rows, e)
            n_checked += 1
            n_ok += not problems
            if problems:
                mismatches.append(f"{name} {describe(e)}: " + "; ".join(problems))
            if not args.check_only and e["kind"] != "outcome_counts" and got is not None:
                tol = e.get("tolerance", 0)
                bound = f"+- {tol:g}" if tol else "exact"
                extra = f", study {e['study_value']:.7g}" if "study_value" in e else ""
                print(f"  {describe(e)} = {got['value']:.7g}  (expected {e['expected']['value']} {bound}{extra})"
                      f"  {'ok' if not problems else 'MISMATCH'}")
        if not args.check_only:
            print(f"  {n_ok}/{len(fspec['expectations'])} expectations match")

    print()
    if mismatches:
        print(f"FAIL: {len(mismatches)} mismatch(es) over {n_checked} expectations")
        for m in mismatches:
            print("  " + m)
        return 1
    print(f"OK: all {n_checked} expectations match across {len(spec)} tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
