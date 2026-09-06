"""Add the three projected-gravity components, computed from the stored quaternion.

WHY NOT RECONSTRUCT FROM roll/pitch. The policy consumes projected gravity, which
is the body-frame direction of gravity and the channel telling it which way is
down. Rebuilding it from the logged roll and pitch does NOT reproduce Chrono's
value: measured error 0.0288 mean / 0.0716 max per component, against components
of order 1. The logged angles come from GetCardanAnglesZYX, so the mismatch is
not an XYZ-versus-ZYX confusion -- three different compositions (Ry*Rx, Rx*Ry, and
a sign variant) were all wrong by the same 0.019, which rules out an ordering slip
and points at Chrono's Cardan convention differing from the textbook composition.

Rather than chase which, take the quantity from the quaternion, where it is EXACT
by construction (verified: 0.0e+00) and yaw-invariant by construction rather than
by argument.

NO RE-COLLECTION NEEDED. quat_e0..e3 are already in every CSV of both halves, so
this is a derivation over existing data -- 17 GB of collection does not have to be
redone for it.

Uses ImportedGo2Policy._projected_gravity itself rather than reimplementing the
formula, so the dataset and the policy cannot drift apart.
"""
import csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from nedm.quadruped.imported_policy import ImportedGo2Policy

GRAV_FIELDS = ["grav_body_x", "grav_body_y", "grav_body_z"]


class _Q:
    def __init__(self, e0, e1, e2, e3):
        self.e0, self.e1, self.e2, self.e3 = e0, e1, e2, e3


def main() -> int:
    root = Path(sys.argv[1])
    apply = "--apply" in sys.argv
    n = done = 0
    for csv_p in sorted(root.glob("episodes/*.csv")):
        rows = list(csv.DictReader(csv_p.open()))
        if not rows:
            continue
        n += 1
        if GRAV_FIELDS[0] in rows[0]:
            continue
        done += 1
        if not apply:
            continue
        fields = list(rows[0].keys()) + GRAV_FIELDS
        for r in rows:
            g = ImportedGo2Policy._projected_gravity(
                _Q(*(float(r[f"quat_e{i}"]) for i in range(4))))
            for name, value in zip(GRAV_FIELDS, g):
                r[name] = float(value)
        with csv_p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"  {n} episodes scanned, {done} needing gravity columns"
          f"{' -- ADDED' if apply else ' (dry run, pass --apply)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
