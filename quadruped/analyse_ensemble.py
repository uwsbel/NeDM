#!/usr/bin/env python3
"""Pool the active-domain ensembles and decide the box size.

Reads one or more `active_domain_ensemble.py` outputs and reports, per arm, the mean
paired difference against the unapproximated solve with its standard error and t.

Two things this does that the in-run summary does not.

FIRST, it applies the `walked()` gate retroactively. The runs that produced these files
predate that gate, so they contain cases where the robot fell through the bed -- one
reference came back at mean_z = -0.303 m, a third of a metre below the soil bottom, and
`run_case` accepted it because the number was finite. Those are not gaits and averaging
them in would be averaging in a fiction. Cases dropped are named, not silently removed.

SECOND, it reports each machine separately as well as pooled. Pooling is defensible --
the claim that CRM results are machine-dependent was withdrawn (docs/EVALUATION.md) --
but "defensible" is not "verified", and a pooled number that disagrees with both of its
parts should be visible rather than hidden.

The null arm is the control throughout. It is a second run of the reference under
identical settings, so whatever it shows is what identical inputs produce. An arm is
evidence of bias only if it stands clear of the null.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

KEYS = ("mean_vx", "mean_vy", "mean_wz", "mean_z", "mean_up", "speed")
MIN_MEAN_Z, MAX_MEAN_Z, MIN_MEAN_UP = 0.38, 0.80, 0.90


def walked(b):
    return (MIN_MEAN_Z <= b["mean_z"] <= MAX_MEAN_Z) and b["mean_up"] >= MIN_MEAN_UP


def mean_sd(v):
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(v) / n
    if n < 2:
        return m, 0.0, n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    return m, sd, n


def load(path):
    d = json.loads(Path(path).read_text())
    cases, dropped = [], []
    for c in d.get("cases", []):
        if not walked(c["none"]):
            dropped.append((c["case"], "reference not a gait",
                            f"z={c['none']['mean_z']:.3f} up={c['none']['mean_up']:.3f}"))
            continue
        kept_arms = {}
        for arm, rec in c.get("arms", {}).items():
            if not walked(rec["behaviour"]):
                dropped.append((c["case"], f"arm {arm} not a gait",
                                f"z={rec['behaviour']['mean_z']:.3f}"))
                continue
            kept_arms[arm] = rec
        cases.append({**c, "arms": kept_arms})
    return cases, dropped


def report(label, cases):
    arms = sorted({a for c in cases for a in c["arms"]},
                  key=lambda s: (s != "null", s))
    print(f"\n{'=' * 76}\n{label}   ({len(cases)} cases)\n{'=' * 76}")
    if not cases:
        print("  nothing survived the gate")
        return
    for arm in arms:
        # AGAINST THE NULL, NOT AGAINST ZERO. Both the arm and the null are paired
        # differences from the same reference run, so the quantity that isolates the
        # active domain is (arm - null) within each case. Testing the arm against zero
        # instead would charge it for whatever two identical runs already differ by --
        # which is zero on some cases and not on others, so it cannot be assumed away.
        usable = [c for c in cases if arm in c["arms"] and "null" in c["arms"]]
        n = len(usable)
        tag = "  (control: identical inputs, so this is the noise)" if arm == "null" else ""
        print(f"\n  arm {arm}   n={n}{tag}")
        for k in KEYS:
            raw = [c["arms"][arm]["delta"][k] for c in usable]
            adj = [c["arms"][arm]["delta"][k] - c["arms"]["null"]["delta"][k]
                   for c in usable]
            m, sd, nn = mean_sd(adj)
            se = sd / math.sqrt(nn) if nn > 1 else 0.0
            t = m / se if se > 0 else float("nan")
            rm, _, _ = mean_sd(raw)
            flag = ""
            if arm != "null" and se > 0 and abs(t) >= 2.0:
                flag = "   <-- clear of the null"
            print(f"    {k:9s} {m:+.5f}  sd {sd:.5f}  se {se:.5f}  t {t:+6.2f}"
                  f"   (raw vs ref {rm:+.5f}){flag}")
        # The bound is as useful as the estimate: with n this small, "no detectable bias"
        # is only meaningful alongside what size of bias would have been detected.
        if arm != "null":
            adj = [c["arms"][arm]["delta"]["mean_vx"] - c["arms"]["null"]["delta"]["mean_vx"]
                   for c in usable]
            _, sd, nn = mean_sd(adj)
            if nn > 1:
                se = sd / math.sqrt(nn)
                print(f"    -> resolution: a mean_vx bias larger than {2 * se:+.4f} m/s "
                      f"would have shown at 2 se")
        cost = [c["arms"][arm]["ms_per_step"] for c in cases if arm in c["arms"]]
        if cost:
            cm, _, _ = mean_sd(cost)
            print(f"    cost      {cm:.2f} ms/step")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()

    pooled, all_dropped = [], []
    for f in a.files:
        cases, dropped = load(f)
        all_dropped += [(Path(f).name, *d) for d in dropped]
        report(f"{Path(f).name}", cases)
        pooled += cases

    if all_dropped:
        print(f"\n{'=' * 76}\nDROPPED ({len(all_dropped)})\n{'=' * 76}")
        for src, case, why, detail in all_dropped:
            print(f"  {src}  {case}  {why}  {detail}")

    if len(a.files) > 1:
        report("POOLED", pooled)
        print("\nPooling assumes the machines sample the same distribution. That claim was"
              "\nwithdrawn as unproven rather than established, so check the pooled numbers"
              "\nagainst the per-file ones above before quoting them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
