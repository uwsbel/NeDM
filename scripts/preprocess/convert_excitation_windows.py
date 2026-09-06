"""Render an excitation `windows.csv` into the per-episode layout everything else expects.

WHY A CONVERTER RATHER THAN A NEW READER. The excitation collector emits ONE flat
CSV with a `window` column for episode boundaries, no `episodes/` tree and no
per-episode sidecars. Every other tool here -- preprocessing, the gate, the
verdict harness -- enumerates episodes by globbing `episodes/*.csv` against a
`dataset_index.json`. A glob against a tree that does not exist returns an EMPTY
LIST and reports zero episodes rather than raising, which is indistinguishable
from a dataset that genuinely has none.

Converting once, here, means the well-tested preprocessing path is untouched and
the excitation data looks like every other dataset to every downstream consumer.

TWO THINGS THIS FIXES THAT ARE NOT LAYOUT.

grav_body_x/y/z ARE ABSENT. They are in `quadruped_contact_conditioned`, the
40-channel preset every current result uses, so a name-keyed loader stops on them.
They are not a collector difference: no collector records them. They are a
post-hoc backfill computed from the quaternion by add_gravity_channels.py, which
ran over the policy corpora and has not run here.

DERIVED BY CALLING THE CANONICAL IMPLEMENTATION, NOT BY REIMPLEMENTING IT.
add_gravity_channels.py calls ImportedGo2Policy._projected_gravity; so does this.
Deriving it inline instead agrees only to ~3e-8, because that function returns
float32 while an inline float64 expression does not -- the FORMULA is identical
character for character and the DTYPE is not. Measured: max|diff| 2.979e-08
against the stored columns, which is float32 epsilon and not a disagreement.

That residual is small enough to be harmless and exactly large enough to look like
a finding, which is why the second implementation should not exist at all. One
call site, one definition, no threshold to argue about.

kp and kd have ZERO VARIANCE (20.0 and 0.5 -- gain randomisation was considered
and dropped). Carried through as columns for provenance but they must never enter
a state preset: per-channel normalisation would divide by a std of zero.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from nedm.quadruped.imported_policy import ImportedGo2Policy  # noqa: E402

# Structure, not physics. None of these belongs in a state vector.
NON_PHYSICS = ("window", "burst", "phase", "chrono_build", "kp", "kd")
# Positional duplicates of joint_<leg>_<joint>_target_rad. Verified elementwise
# identical (max|diff| 0.00e+00, Chrono order RR/RL/FR/FL). Dropped so no consumer
# can read the action positionally, which is the only way to get the order wrong.
POSITIONAL = tuple(f"target_{i}" for i in range(12))


class _Q:
    """The 4-field shape _projected_gravity expects, matching add_gravity_channels."""

    def __init__(self, e0, e1, e2, e3):
        self.e0, self.e1, self.e2, self.e3 = e0, e1, e2, e3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("windows_csv", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    a = ap.parse_args(argv)

    name = a.dataset_name or a.windows_csv.parent.name
    eps_dir = a.out_dir / "episodes"
    eps_dir.mkdir(parents=True, exist_ok=True)

    with a.windows_csv.open(newline="") as fp:
        header = next(csv.reader(fp))
    keep = [c for c in header if c not in NON_PHYSICS and c not in POSITIONAL]
    derive_grav = "grav_body_x" not in header
    out_cols = keep + (["grav_body_x", "grav_body_y", "grav_body_z"] if derive_grav else [])
    # `phase` is dropped from the state but MUST survive: the phase split is what
    # lets batch_fraction control the informative fraction rather than the dataset.
    out_cols = out_cols + ["phase"]
    print(f"{len(header)} input columns -> {len(out_cols)} output "
          f"(dropped {len(header) - len(keep)}, derived grav: {derive_grav})")

    episodes: list[dict] = []
    # TWO VARIABLES ON PURPOSE. The first version compared row["window"] against the
    # variable it then overwrote with the FORMATTED episode id, so the comparison was
    # "0" != "go2_exc_b40_c3_w000000" on every row: a new file per row, each open
    # truncating the last, leaving one row per episode and 1.2M index entries.
    # It passed a row-count check -- 1,249,840 entries of 1 row sums to 1,249,840 --
    # which is why the check that caught it was "how many rows in episode 0".
    cur_window, cur_id, writer, fh, nrows = None, None, None, None, 0

    def close() -> None:
        nonlocal fh, nrows, cur_id
        if fh is None:
            return
        fh.close()
        episodes.append({
            "episode_id": cur_id,
            "scenario_name": cur_id,
            "scenario_family": f"{name}_excitation",
            "split": "val" if (len(episodes) % int(round(1 / a.val_fraction))) == 0 else "train",
            "csv_path": f"episodes/{cur_id}.csv",
            "rows": str(nrows),
            "terrain_label": "flat",
            "warmup_s": "0.0",
        })

    with a.windows_csv.open(newline="") as fp:
        for row in csv.DictReader(fp):
            if row["window"] != cur_window:
                close()
                cur_window = row["window"]
                cur_id = f"{name}_w{int(cur_window):06d}"
                fh = (eps_dir / f"{cur_id}.csv").open("w", newline="")
                writer = csv.DictWriter(fh, fieldnames=out_cols)
                writer.writeheader()
                nrows = 0
            if derive_grav:
                g = ImportedGo2Policy._projected_gravity(
                    _Q(*(float(row[f"quat_e{i}"]) for i in range(4))))
                row["grav_body_x"], row["grav_body_y"], row["grav_body_z"] = (
                    float(g[0]), float(g[1]), float(g[2]))
            writer.writerow({c: row[c] for c in out_cols})
            nrows += 1
    close()

    index = {
        "dataset_name": name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_count": len(episodes),
        "episodes": episodes,
        "converted_from": str(a.windows_csv),
        "note": ("Converted from a flat windows.csv. grav_body_* derived from the "
                 "quaternion where absent, matching read_arrays. kp/kd dropped "
                 "(zero variance). target_* dropped as positional duplicates of the "
                 "named target columns, verified elementwise identical."),
    }
    (a.out_dir / "dataset_index.json").write_text(json.dumps(index, indent=2) + "\n")
    ntr = sum(1 for e in episodes if e["split"] == "train")
    print(f"wrote {len(episodes)} episodes ({ntr} train / {len(episodes)-ntr} val) to {a.out_dir}")

    # SELF-CHECK AGAINST summary.json, IF IT IS THERE.
    #
    # A CONSERVATION CHECK ON A TOTAL CANNOT SEE A REDISTRIBUTION. The first version
    # of this script opened a new file per row, each truncating the last -- one row
    # per episode, 1.2M index entries -- and the row-count check PASSED, because
    # 1,249,840 entries of one row sums to exactly 1,249,840. The headline
    # verification confirmed the corruption.
    #
    # So the total is checked, and then the DISTRIBUTION is checked separately:
    # episode count against the source's own figure, and rows-per-episode against
    # its stated constant. A sum is invariant under exactly the failures that move
    # things around rather than losing them, and either margin alone leaves a hole --
    # the right rows-per-episode with the wrong episode count passes one and fails
    # the other.
    summary_path = a.windows_csv.parent / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        rows_total = sum(int(e["rows"]) for e in episodes)
        per_ep = {int(e["rows"]) for e in episodes}
        problems = []
        if "rows" in summary and rows_total != int(summary["rows"]):
            problems.append(f"rows {rows_total} != summary {summary['rows']}")
        if "kept" in summary and len(episodes) != int(summary["kept"]):
            problems.append(f"episodes {len(episodes)} != summary kept {summary['kept']}")
        if "rows_per_episode" in summary:
            want = int(summary["rows_per_episode"])
            if per_ep != {want}:
                problems.append(f"rows/episode {sorted(per_ep)[:4]} != stated {want}")
        if problems:
            raise SystemExit("conversion disagrees with summary.json: " + "; ".join(problems))
        print(f"  self-check OK: rows {rows_total}, episodes {len(episodes)}, "
              f"rows/episode {per_ep.pop()} -- all agree with summary.json")
    else:
        print("  summary.json absent: totals and distribution NOT cross-checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
