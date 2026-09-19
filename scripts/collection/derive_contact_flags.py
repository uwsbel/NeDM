"""Populate the foot contact flags from the foot forces, with hysteresis.

collect_go2_smoke.py writes foot_fl/fr/rl/rr_in_contact but never fills them, so every
row of every CRM episode carries NaN in those four columns while foot_*_force_fz_n beside
them carries the real normal force. The 40-D contact-conditioned preset therefore cannot
be trained at all, which is what it did for eighty epochs before the preprocessing guard
existed to say so.

The information is not missing from the corpus, only from the column. A foot is in
contact when it is carrying load, and the force channel records exactly that.

HYSTERESIS, NOT A SINGLE THRESHOLD. A bare comparison chatters: the normal force crosses
any fixed level several times per stance as the soil yields under the foot, so a
single-threshold flag would encode solver noise as contact events. Engage at 25 N and
release at 5 N, the same pair the preprocessing contact-mode uses, which were chosen for
this corpus because the rigid-terrain default of 60 N classifies the median CRM stance
sample as swing.

WHY THIS ARM IS STILL WORTH RUNNING even though forcez carries the same forces
continuously: the two encode different things. forcez hands the model a magnitude it must
learn to threshold; this hands it the threshold already taken. If the coarser channel
transfers as well, the model did not need the magnitude, and that is a statement about
what the abstraction has to carry.

Idempotent: an episode whose flags are already finite is skipped.
"""
import csv
import sys
from pathlib import Path

FEET = ["fl", "fr", "rl", "rr"]
ENGAGE_N = 25.0
RELEASE_N = 5.0


def main() -> int:
    root = Path(sys.argv[1])
    apply = "--apply" in sys.argv
    paths = sorted(root.glob("episodes/*.csv")) or sorted(root.glob("*/episodes/*.csv"))
    if not paths:
        print(f"FATAL: no episodes under {root} in either layout", file=sys.stderr)
        return 2

    n = done = 0
    for csv_p in paths:
        rows = list(csv.DictReader(csv_p.open()))
        if not rows:
            continue
        n += 1
        flag_cols = [f"foot_{f}_in_contact" for f in FEET]
        force_cols = [f"foot_{f}_force_fz_n" for f in FEET]
        if any(c not in rows[0] for c in flag_cols + force_cols):
            continue

        def _finite(v):
            try:
                return v not in (None, "") and float(v) == float(v)
            except ValueError:
                return False

        if all(_finite(r[flag_cols[0]]) for r in rows[:20]):
            continue          # already populated
        done += 1
        if not apply:
            continue

        for fi, f in enumerate(FEET):
            state = 0
            for r in rows:
                try:
                    fz = float(r[force_cols[fi]])
                except (TypeError, ValueError):
                    fz = 0.0
                # Latch: rise past ENGAGE to enter contact, fall below RELEASE to leave.
                if state == 0 and fz >= ENGAGE_N:
                    state = 1
                elif state == 1 and fz < RELEASE_N:
                    state = 0
                r[flag_cols[fi]] = float(state)

        with csv_p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"  {n} episodes scanned, {done} needing contact flags"
          f"{' -- DERIVED' if apply else ' (dry run, pass --apply)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
