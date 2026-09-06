"""Fail loudly if a processed dataset still contains wrapped circular channels.

WHY THIS EXISTS. pitch_rad spans the full +-pi and jumps by exactly 2*pi where the
angle wraps -- 2159 such jumps in the 40-channel training set. Every model here is
a DELTA model, so each wrap becomes a +-2*pi target it must fit as an ordinary
real. Removing those spikes was measured to carry the WHOLE of the 0.5 s
action-sensitivity gain improvement (paired d(|log gain|) -0.4158, CI excluding
zero), while the 113x loss-weight change that accompanied it did nothing
detectable.

So this is not hygiene. A dataset that reaches training with the wraps intact
inherits a defect we have already proved is expensive, and it does so silently --
the training runs, the loss falls, and only the gate notices.

Run it on every processed dataset before anything trains on it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# A WRAP IS SPECIFICALLY ~2*pi. The first version of this check used 1.0 rad on the
# reasoning that a real per-step attitude change is ~0.03 rad -- and it FAILED the
# known-good unwrapped dataset, which legitimately contains 6 steps up to 2.8766 rad.
# Those are genuine large attitude changes in near-diverged episodes: np.unwrap leaves
# them alone precisely because they are BELOW pi and therefore not wraps. Testing
# "large" when the property is "wrapped" is the same magnitude-for-property conflation
# that the preprocessing guard itself got wrong on joint velocities.
WRAP_TOL_RAD = 5.0   # 2*pi = 6.283; no physical single-step change comes close
SUSPICIOUS_RAD = 1.0  # reported as a diagnostic, never as a failure


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    a = ap.parse_args(argv)

    md = json.loads((a.dataset_dir / "metadata.json").read_text())
    fields = md["state_fields"]
    declared = md.get("circular_unwrapped")

    print(f"dataset: {a.dataset_dir}")
    if declared is None:
        print("  metadata has NO 'circular_unwrapped' key -- built before the marker "
              "existed, or by a checkout predating c699338. Cannot confirm intent; "
              "the numeric check below is the only evidence.")
    else:
        print(f"  metadata declares circular_unwrapped = {declared or '[] (none)'}")

    bad = 0
    for split in a.splits:
        p = a.dataset_dir / f"{split}_targets.npy"
        if not p.exists():
            print(f"  {split}: no targets array, skipped")
            continue
        T = np.load(p, mmap_mode="r")
        for name in ("roll_rad", "pitch_rad", "yaw_rad", "body_slip_rad"):
            if name not in fields:
                continue
            i = fields.index(name)
            mx, cnt, susp = 0.0, 0, 0
            for s in range(0, T.shape[0], 1_000_000):
                c = np.abs(np.asarray(T[s:s + 1_000_000, i]))
                mx = max(mx, float(c.max()))
                cnt += int((c > WRAP_TOL_RAD).sum())
                susp += int((c > SUSPICIOUS_RAD).sum())
            status = "OK" if cnt == 0 else "WRAPPED"
            if cnt:
                bad += 1
            note = f"  ({susp} above {SUSPICIOUS_RAD} rad, large but not wraps)" if susp else ""
            print(f"  {split:5s} {name:14s} max|target| {mx:8.4f}  "
                  f"wraps(>{WRAP_TOL_RAD}): {cnt:6d}   {status}{note}")

    if bad:
        print("\nFAIL: wrapped circular channels reached the targets. Rebuild this "
              "dataset with a checkout containing c699338, or the delta model will "
              "be trained on +-2*pi discontinuities.")
        return 1
    print("\nPASS: no circular channel carries a wrap-sized target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
