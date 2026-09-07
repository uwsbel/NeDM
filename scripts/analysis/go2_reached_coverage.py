#!/usr/bin/env python3
"""Coverage of REACHED states, read straight from collector CSVs.

Same three numbers as scripts/preprocess/corpus_coverage.py, which is the
authority for the methodology and explains why `covers` alone is misleading:
a handful of tail rows lights up every bin, so binary occupancy cannot see a
density hole and `tail` can.

This exists because the coverage question has to be asked of REACHED states
rather than commanded ones. Commanding vx does not produce vx -- realised
fraction 4-5% below 0.1 m/s, see go2-command-realisation-deadband.md -- so a
commanded-coverage audit reports success on a corpus that stayed narrow. Reading
the state columns the collector actually wrote is the only version of the
question worth asking, and it avoids a preprocessing step that would have to be
run identically on both corpora to be comparable.
"""
import argparse, csv, glob, os
import numpy as np

CHANNELS = ("vel_body_x_mps", "vel_body_y_mps", "vel_body_z_mps",
            "yaw_rate_radps", "ang_vel_body_x_radps", "ang_vel_body_y_radps",
            "ang_vel_body_z_radps")


def load(pattern, cols, max_rows=400_000):
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit(f"no CSVs matched {pattern}")
    out = {c: [] for c in cols}
    n = 0
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                for c in cols:
                    v = r.get(c)
                    out[c].append(float(v) if v not in (None, "") else np.nan)
                n += 1
                if n >= max_rows:
                    break
        if n >= max_rows:
            break
    return {c: np.array(v) for c, v in out.items()}, n, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="glob for the reference (walking)")
    ap.add_argument("--candidate", required=True, action="append",
                    help="glob for a candidate; repeatable, label=glob")
    ap.add_argument("--bins", type=int, default=50)
    a = ap.parse_args()

    A, na, fa = load(a.reference, CHANNELS)
    print(f"reference: {na} rows from {fa} files")
    cands = {}
    for spec in a.candidate:
        label, _, pat = spec.partition("=")
        B, nb, fb = load(pat, CHANNELS)
        cands[label] = B
        print(f"candidate {label}: {nb} rows from {fb} files")

    print(f"\n  {'channel':24s} " + "".join(f"{l:>26s}" for l in cands))
    print(f"  {'':24s} " + "".join(f"{'inside':>8s}{'covers':>8s}{'tail':>10s}"
                                   for _ in cands))
    for c in CHANNELS:
        x = A[c][np.isfinite(A[c])]
        if x.size == 0:
            continue
        lo, hi = np.percentile(x, 5), np.percentile(x, 95)
        thr = np.percentile(np.abs(x), 90)
        pa = float((np.abs(x) > thr).mean())
        line = f"  {c:24s} "
        for label, B in cands.items():
            y = B[c][np.isfinite(B[c])]
            if y.size == 0 or hi <= lo:
                line += f"{'--':>26s}"; continue
            inside = float(((y >= lo) & (y <= hi)).mean())
            edges = np.linspace(lo, hi, a.bins + 1)
            covers = float((np.histogram(y, bins=edges)[0] > 0).mean())
            tail = float((np.abs(y) > thr).mean()) / pa if pa > 0 else float("nan")
            mark = "*" if tail < 0.2 else " "
            line += f"{inside:8.3f}{covers:8.3f}{tail:9.4f}{mark}"
        print(line)
    print("\n  * = tail below 0.2, i.e. the candidate reaches the reference's "
          "high-magnitude regime\n    more than 5x more rarely. That is a density "
          "hole; `covers` cannot see it.")


if __name__ == "__main__":
    main()
