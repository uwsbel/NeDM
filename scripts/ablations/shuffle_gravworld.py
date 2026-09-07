"""Write grav_shuf_*_mps2: grav_world permuted ACROSS episodes.

THE DIMENSIONALITY CONTROL. quadruped_joint_gravworld_pose adds three channels to
quadruped_joint_grav_pose, so a gain over it confounds "tilt is observable" with
"three more inputs". This produces an arm with the same 39 channels whose three
extra columns carry the same marginal distribution, the same scale and no valid
correspondence to the episode they sit in.

A PERMUTATION rather than a constant: a constant has zero variance, and the input
normalisation would divide by a near-zero std and treat the channel unlike any
real one, which is a second difference rather than a control.

Gravity is CONSTANT WITHIN an episode -- it is set once at construction -- so the
permutation is over episodes, not over rows. Permuting rows would leave the
episode mean intact and hand the model the information back.
"""
import argparse, csv, glob, json, os, random, sys

SRC = ["grav_world_x_mps2", "grav_world_y_mps2", "grav_world_z_mps2"]
DST = ["grav_shuf_x_mps2", "grav_shuf_y_mps2", "grav_shuf_z_mps2"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root holding */episodes/*.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    csvs = sorted(glob.glob(f"{a.root}/*/episodes/*.csv"))
    if not csvs:
        print(f"no episodes under {a.root}", file=sys.stderr); return 2

    # One gravity triple per episode, read from the first data row. Verified
    # constant within the episode before use -- if it varies, the premise that a
    # cross-episode permutation destroys the information is wrong.
    triples, varying = [], []
    for c in csvs:
        rows = list(csv.DictReader(open(c)))
        if not rows: continue
        if any(f not in rows[0] for f in SRC):
            print(f"FATAL: {c} has no grav_world columns", file=sys.stderr); return 2
        first = tuple(float(rows[0][f]) for f in SRC)
        if any(abs(float(r[f]) - first[i]) > 1e-9
               for r in rows[::max(1, len(rows) // 20)] for i, f in enumerate(SRC)):
            varying.append(c)
        triples.append((c, first, len(rows)))
    if varying:
        print(f"FATAL: gravity varies within {len(varying)} episode(s), e.g. {varying[0]}",
              file=sys.stderr)
        return 2
    print(f"{len(triples)} episodes, gravity constant within each")

    # A derangement where possible: no episode may keep its own triple, or those
    # rows are the real channel and the control is contaminated.
    idx = list(range(len(triples)))
    rng = random.Random(a.seed)
    for _ in range(1000):
        rng.shuffle(idx)
        if all(i != j for i, j in enumerate(idx)):
            break
    else:
        print("FATAL: could not derange", file=sys.stderr); return 2
    fixed = sum(1 for i, j in enumerate(idx) if i == j)
    print(f"derangement: {fixed} episodes keep their own triple (must be 0)")

    if a.dry_run:
        print("dry run, nothing written"); return 0

    for i, (c, _, _) in enumerate(triples):
        donor = triples[idx[i]][1]
        rows = list(csv.DictReader(open(c)))
        hdr = list(rows[0].keys())
        for d in DST:
            if d not in hdr: hdr.append(d)
        for r in rows:
            for d, v in zip(DST, donor):
                r[d] = f"{v:.10g}"
        tmp = c + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=hdr); w.writeheader(); w.writerows(rows)
        os.replace(tmp, c)
    print(f"wrote {DST} into {len(triples)} episodes, permuted across episodes, seed {a.seed}")

    # Verify from disk rather than from the loop that just wrote it.
    same = 0
    for c, own, _ in triples:
        r = next(csv.DictReader(open(c)))
        if all(abs(float(r[d]) - own[i]) < 1e-9 for i, d in enumerate(DST)):
            same += 1
    print(f"re-read check: {same} of {len(triples)} episodes have shuf == own (must be 0)")
    return 0 if same == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
