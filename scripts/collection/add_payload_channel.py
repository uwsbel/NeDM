"""Add the carried-mass channel from the per-episode sidecar.

WHY THE SURROGATE NEEDS IT. The 36-D state says nothing about what the robot is
carrying, so two episodes with the same pose, the same joint state and the same action
evolve differently and the surrogate has no way to tell them apart. It cannot learn
both, so it learns the average over the payload distribution, and an average is exactly
the kind of invented structure a policy optimiser climbs. This is the same argument the
force channels answered for contact, and it gets the same treatment: a separate
surrogate class, measured against the identical corpus without the channel, so the
question "does conditioning on load help" has a control rather than an anecdote.

WHY A CONSTANT COLUMN IS THE HONEST FORM. The payload is attached before the episode
starts and never changes, so a per-row column carries exactly the information the
sidecar already recorded, in the place preprocess can reach. Nothing is inferred.

NO RE-COLLECTION NEEDED, same as add_gravity_channels.py: payload_kg and robot_mass_kg
are already written into every episode sidecar by collect_go2_smoke.py since 299a57f.

Idempotent and safe on a corpus still being collected: an episode that already carries
the column is skipped, and an episode whose sidecar is missing is reported rather than
defaulted to zero. Defaulting would be the dangerous choice, because unloaded is a
legitimate value in this corpus (the 0 kg bin is a fifth of it), so a missing sidecar
would silently become a confident claim of no payload.
"""
import csv
import json
import sys
from pathlib import Path

PAYLOAD_FIELDS = ["payload_kg", "robot_mass_kg"]


def main() -> int:
    root = Path(sys.argv[1])
    apply = "--apply" in sys.argv
    n = done = nosidecar = 0
    # Both layouts, for the reason add_gravity_channels.py records: the collector
    # writes <root>/<scenario>/episodes/*.csv, a consolidated root is <root>/episodes/*.csv,
    # and a single-pattern glob matches nothing on the other one while exiting 0.
    paths = sorted(root.glob("episodes/*.csv")) or sorted(root.glob("*/episodes/*.csv"))
    if not paths:
        print(f"FATAL: no episodes under {root} in either layout", file=sys.stderr)
        return 2
    for csv_p in paths:
        side = csv_p.with_suffix(".json")
        if not side.exists():
            nosidecar += 1
            continue
        rows = list(csv.DictReader(csv_p.open()))
        if not rows:
            continue
        n += 1
        if PAYLOAD_FIELDS[0] in rows[0]:
            continue
        meta = json.loads(side.read_text())
        if "payload_kg" not in meta:
            nosidecar += 1
            continue
        done += 1
        if not apply:
            continue
        vals = {f: float(meta.get(f, 0.0)) for f in PAYLOAD_FIELDS}
        fields = list(rows[0].keys()) + PAYLOAD_FIELDS
        for r in rows:
            r.update(vals)
        with csv_p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"  {n} episodes scanned, {done} needing payload columns, "
          f"{nosidecar} without a usable sidecar"
          f"{' -- ADDED' if apply else ' (dry run, pass --apply)'}")
    return 1 if nosidecar and not n else 0


if __name__ == "__main__":
    raise SystemExit(main())
