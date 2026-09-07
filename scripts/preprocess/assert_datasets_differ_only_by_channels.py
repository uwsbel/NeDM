"""Assert two processed datasets differ ONLY in their channel set.

WHY. `go2_mix34_base_replicate` was built to isolate one variable -- the state
preset -- by pairing a 34-D surrogate against a 36-D one. Its dataset,
`go2_corrected_34d_excl`, was processed 2026-09-05 and the 36-D dataset
2026-09-06, and they differ in the circular-unwrap fix: the 36-D one records
`circular_unwrapped: [roll_rad, pitch_rad]` and a `processing_provenance` commit,
the 34-D one has neither field at all.

The unwrap is not cosmetic -- isolated, it moved `err/signal` from 0.609 to 0.518.
So that pairing would have confounded the channel set with the wrap fix, which is a
confound this project has already written down once about an earlier comparison.

The config for that run asserted no 36-D dataset reference survived. That was true
and insufficient: it checked what the run pointed AT, not whether the two things
being compared were otherwise identical. This checks the latter.

Absence is not agreement: if one dataset lacks the field that records a property,
that is a FAILURE, not a match. You cannot assert two things agree on a property
one of them does not record.
"""
import argparse, hashlib, json, sys
import numpy as np
from pathlib import Path

# target_fields is DERIVED from state_fields (one delta_ per channel), so it must
# differ by exactly the added channels -- checking it for equality would fail a
# perfectly matched pair. It is checked below against the expected additions instead.
MUST_MATCH = ("raw_dataset_roots", "dt_s", "contact_mode", "circular_unwrapped",
              "action_fields", "rollout_fields")


def load(d):
    return json.loads((Path(d) / "metadata.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--hash-rows", type=int, default=200000,
                    help="rows per split to hash over the shared channels")
    ap.add_argument("--expect-added", nargs="*", default=[],
                    help="state fields b may have that a does not; anything else fails")
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    fails = []

    for k in MUST_MATCH:
        av, bv = a.get(k, "<ABSENT>"), b.get(k, "<ABSENT>")
        if av == "<ABSENT>" or bv == "<ABSENT>":
            fails.append(f"{k}: {av!r} vs {bv!r} -- ABSENT is not a match, it is an unrecorded property")
        elif av != bv:
            fails.append(f"{k}: {av!r} != {bv!r}")

    # A DIFFERING COMMIT IS NOT AUTOMATICALLY A DIFFERING DATASET. Two datasets built
    # weeks apart can be byte-identical in behaviour if nothing on the preprocessing
    # path changed between them. Comparing the bare commit string rejects that case,
    # which would force a pointless rebuild -- but ACCEPTING a differing commit
    # without checking is how the unwrap confound got in. So: differing commits are
    # allowed only when the preprocessing path is provably unchanged between them.
    # An ABSENT commit is still a failure: nothing can be verified against it.
    pa = a.get("processing_provenance", {}).get("commit", "<ABSENT>")
    pb = b.get("processing_provenance", {}).get("commit", "<ABSENT>")
    if "<ABSENT>" in (pa, pb):
        fails.append(f"processing_provenance commit: {pa} vs {pb} -- ABSENT cannot be verified")
    elif pa != pb:
        import subprocess
        paths = ["src/nedm/training/preprocess.py", "src/nedm/training/constants.py",
                 "src/nedm/training/dataset.py"]
        r = subprocess.run(["git", "diff", "--stat", pa, pb, "--"] + paths,
                           capture_output=True, text=True)
        if r.returncode != 0:
            fails.append(f"commits {pa[:8]} vs {pb[:8]} differ and git could not compare them: "
                         f"{r.stderr.strip()[:120]}")
        elif r.stdout.strip():
            fails.append(f"commits {pa[:8]} vs {pb[:8]} differ AND the preprocessing path "
                         f"changed between them:\n      " + r.stdout.strip().replace("\n", "\n      "))
        else:
            print(f"  commits differ ({pa[:8]} vs {pb[:8]}) but the preprocessing path is "
                  f"IDENTICAL between them -- accepted")

    # EPISODE IDS, not counts. Two different corpora can agree on counts.
    for split in ("train", "val"):
        ea = a["splits"].get(split, {}).get("episode_ids")
        eb = b["splits"].get(split, {}).get("episode_ids")
        if ea is None or eb is None:
            ca = a["splits"].get(split, {}).get("episode_count")
            cb = b["splits"].get(split, {}).get("episode_count")
            ta = a["splits"].get(split, {}).get("transition_count")
            tb = b["splits"].get(split, {}).get("transition_count")
            if (ca, ta) != (cb, tb):
                fails.append(f"{split}: ({ca} ep, {ta} trans) != ({cb} ep, {tb} trans)")
            else:
                print(f"  NOTE {split}: episode_ids not recorded; matched on "
                      f"count+transitions ({ca} ep, {ta} trans) -- weaker than ids")
        elif sorted(ea) != sorted(eb):
            fails.append(f"{split}: episode id sets differ")

    # target_fields must differ by exactly delta_<added channel>, no more.
    ta_, tb_ = a.get("target_fields", []), b.get("target_fields", [])
    t_added = [f for f in tb_ if f not in ta_]
    t_removed = [f for f in ta_ if f not in tb_]
    if t_removed:
        fails.append(f"target_fields REMOVED: {t_removed}")
    expected_t = sorted(f"delta_{f}" for f in args.expect_added)
    if sorted(t_added) != expected_t:
        fails.append(f"target_fields added {sorted(t_added)}, expected {expected_t}")

    # IDENTITY, NOT SIZE. episode_ids are not recorded, so counts are the weakest
    # available split check and two different corpora can agree on them. The shared
    # channels themselves are the strong check: same raw root, same split seed, same
    # order, same unwrap => the shared columns hold the SAME NUMBERS row for row.
    shared = [f for f in a["state_fields"] if f in b["state_fields"]]
    for split in ("train", "val"):
        try:
            A = np.load(f"{args.a}/{split}_states.npy", mmap_mode="r")
            B = np.load(f"{args.b}/{split}_states.npy", mmap_mode="r")
        except FileNotFoundError as e:
            fails.append(f"{split}_states.npy missing: {e}")
            continue
        n = min(len(A), len(B), args.hash_rows)
        ia = [a["state_fields"].index(f) for f in shared]
        ib = [b["state_fields"].index(f) for f in shared]
        ha = hashlib.sha256(np.ascontiguousarray(A[:n][:, ia])).hexdigest()[:16]
        hb = hashlib.sha256(np.ascontiguousarray(B[:n][:, ib])).hexdigest()[:16]
        if len(A) != len(B):
            fails.append(f"{split}_states row count {len(A)} != {len(B)}")
        if ha != hb:
            d = np.abs(np.asarray(A[:n][:, ia], dtype=np.float64)
                       - np.asarray(B[:n][:, ib], dtype=np.float64))
            worst = shared[int(d.max(axis=0).argmax())]
            fails.append(f"{split}: {len(shared)} shared channels DIFFER over {n} rows "
                         f"({ha} vs {hb}); largest disagreement in {worst!r} "
                         f"(max {d.max():.6g})")
        else:
            print(f"  {split}: {len(shared)} shared channels identical over {n} rows  [{ha}]")

    added = [f for f in b["state_fields"] if f not in a["state_fields"]]
    removed = [f for f in a["state_fields"] if f not in b["state_fields"]]
    if removed:
        fails.append(f"b REMOVES state fields, not just adds: {removed}")
    if sorted(added) != sorted(args.expect_added):
        fails.append(f"state_fields added {sorted(added)}, expected {sorted(args.expect_added)}")

    print(f"\n  A {Path(args.a).name}  ({len(a['state_fields'])}-D)")
    print(f"  B {Path(args.b).name}  ({len(b['state_fields'])}-D)")
    if fails:
        print("\n  DIFFER BY MORE THAN THE CHANNEL SET:")
        for f in fails:
            print(f"    - {f}")
        print("\n  FAILED -- pairing these confounds the channel set with the differences above.")
        return 1
    print(f"\n  identical except state_fields: b adds {sorted(added)}")
    print("  PASSED -- the channel set is the only variable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
