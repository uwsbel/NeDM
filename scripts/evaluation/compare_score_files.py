"""Compare two scorings of the SAME policy on the same episodes.

Used to decide whether Chrono scoring can be spread across the fleet. The corpus
note says episodes replay bit-identically only on their collecting machine, which
would make a cross-machine arm comparison confound arm with machine. Rather than
assume it either way, score one policy on two boxes and measure the disagreement.

The two boxes here also carry DIFFERENT pychrono builds (_core.so md5 3b0bd530 on
sbel, d1d0bd0a on north), so a difference found is a difference of build-and-machine
together, not of machine alone -- and a null is correspondingly stronger.

Reports the paired disagreement, not just the two rates: two runs can land on the
same completion count while disagreeing on which episodes completed, and only the
paired form can tell those apart.
"""
import argparse, json, sys

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
o = ap.parse_args()


def load(p, tag):
    recs = json.load(open(p))
    d = {}
    for r in recs:
        k = r.get("episode_id") or (r["seed"], r["pitch"], r.get("roll"), r.get("family"))
        d[k] = r
    if len(d) != len(recs):
        raise SystemExit(f"FATAL: {p} has {len(recs)} records but {len(d)} distinct keys")
    return d


A, B = load(o.a, o.label_a), load(o.b, o.label_b)
common = set(A) & set(B)
if not common:
    raise SystemExit("FATAL: the two files share no episodes")
if len(common) != len(A) or len(common) != len(B):
    print(f"  NOTE: {len(A)} vs {len(B)} episodes, {len(common)} in common")

ka = sum(A[k]["completed"] for k in common)
kb = sum(B[k]["completed"] for k in common)
print(f"  {o.label_a}: {ka}/{len(common)} = {100*ka/len(common):.1f}%")
print(f"  {o.label_b}: {kb}/{len(common)} = {100*kb/len(common):.1f}%")

disagree = [k for k in common if A[k]["completed"] != B[k]["completed"]]
print(f"\n  PAIRED: episodes where the two disagree on completion: "
      f"{len(disagree)} of {len(common)}  ({100*len(disagree)/len(common):.2f}%)")

rows_same = sum(1 for k in common if A[k]["rows"] == B[k]["rows"])
print(f"  episodes with IDENTICAL row counts: {rows_same} of {len(common)} "
      f"({100*rows_same/len(common):.2f}%)")

if rows_same == len(common):
    print("\n  Replay is bit-identical across these two boxes on every episode.")
    print("  Scoring can be distributed; machine cannot confound an arm contrast.")
elif not disagree:
    print("\n  Completion agrees on every episode though row counts differ, so the")
    print("  verdict is machine-invariant even where the trajectory is not.")
else:
    d = [k for k in disagree]
    print(f"\n  Replay is NOT machine-invariant: {len(d)} completion flips.")
    print("  Score every arm on ONE machine, or treat machine as a blocking factor.")
    for k in d[:5]:
        print(f"    {str(k)[:52]:54s} {o.label_a} rows {A[k]['rows']:6d} -> "
              f"{o.label_b} rows {B[k]['rows']:6d}")
