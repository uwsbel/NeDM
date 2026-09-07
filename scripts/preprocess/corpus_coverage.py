"""Per-channel coverage of one corpus against another, all channels in one pass.

WHY. Two coverage holes in the excitation corpus were found on the same day, hours
apart, both consequences of one design decision (--branch-from-policy runs the
policy until it is walking, holds vx, and perturbs joints, so the robot is never
still and never turns):

    standing states        walking 0.10%    excitation 0.00%
    high yaw >1 rad/s      walking 56.8%    excitation 0.56%

Each was found by someone asking about that specific channel. There is no reason to
think those are the only two. This asks every channel at once.

THREE NUMBERS, because occupancy is not density:
  inside   fraction of B's rows landing inside A's p5-p95 -- is B on-distribution?
  covers   fraction of A's p5-p95 SPAN that B ever visits -- BINARY OCCUPANCY
  tail     P_B(|x| > A's p90 magnitude) / P_A(same) -- how often B reaches A's
           high-magnitude regime, RELATIVE. 1.0 = as often; 0.01 = 100x rarer.

`covers` alone is misleading and the first version of this file used it alone. It
reported yaw_rate at 0.980 -- apparently full coverage -- against an independent
measurement showing the excitation corpus reaches |yaw| > 1 rad/s in 0.56% of rows
against walking's 56.8%, a 101x gap. Both are correct: a handful of tail rows lights
up every bin, so binary occupancy cannot see a density hole. `tail` can.
"""
import argparse, json
import numpy as np
from pathlib import Path


def load(root, split="train", max_rows=400_000):
    root = Path(root)
    meta = json.loads((root / "metadata.json").read_text())
    S = np.load(root / f"{split}_states.npy", mmap_mode="r")
    n = min(len(S), max_rows)
    step = max(1, len(S) // n)
    return meta["state_fields"], np.asarray(S[::step][:n], dtype=np.float64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("reference"); ap.add_argument("candidate")
    ap.add_argument("--bins", type=int, default=50)
    a = ap.parse_args()
    fa, A = load(a.reference)
    fb, B = load(a.candidate)
    shared = [f for f in fa if f in fb]
    ia = {f: fa.index(f) for f in shared}
    ib = {f: fb.index(f) for f in shared}
    rows = []
    for f in shared:
        x, y = A[:, ia[f]], B[:, ib[f]]
        lo, hi = np.percentile(x, 5), np.percentile(x, 95)
        if hi <= lo:
            rows.append((f, float("nan"), float("nan"), lo, hi)); continue
        inside = float(((y >= lo) & (y <= hi)).mean())
        edges = np.linspace(lo, hi, a.bins + 1)
        covers = float((np.histogram(y, bins=edges)[0] > 0).mean())
        thr = np.percentile(np.abs(x), 90)
        pa = float((np.abs(x) > thr).mean())
        tail = float((np.abs(y) > thr).mean()) / pa if pa > 0 else float("nan")
        rows.append((f, inside, covers, tail, lo, hi))
    rows.sort(key=lambda r: (r[3] if r[3] == r[3] else 1e9))
    print(f"\n  reference {Path(a.reference).name}   candidate {Path(a.candidate).name}")
    print(f"  {len(shared)} shared channels, sorted by TAIL ascending (density holes first)\n")
    print(f"  {'channel':34s} {'inside':>8s} {'covers':>8s} {'tail':>9s}   ref p5..p95")
    for f, ins, cov, tl, lo, hi in rows:
        flag = "  <-- HOLE" if tl == tl and tl < 0.2 else ""
        print(f"  {f:34s} {ins:8.3f} {cov:8.3f} {tl:9.4f}   [{lo:7.3f},{hi:7.3f}]{flag}")
