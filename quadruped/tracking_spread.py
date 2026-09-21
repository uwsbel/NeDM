#!/usr/bin/env python3
"""How much does the tracking number move between runs that should be identical?

The study's headline is a tracking gap: the policy holds ~95% of commanded forward speed
on rigid ground and much less on CRM, and closing that gap is what fine-tuning is for. So
far that gap has been quoted from SINGLE runs -- 95% and 64% measured on sbel, then 97%
and 73% measured on a3 from the same policy and the same command.

Nine points of movement in the CRM number is either a machine difference or realisation
variance, and the two have very different consequences. If it is variance, then a single
3 s run cannot support the headline and the gap needs an error bar before it goes in a
paper. Stage 1 of the active-domain work already showed this system is chaotic, so
variance is the likely explanation and it needs measuring rather than assuming.

The perturbation is the spawn position. It is the most innocuous knob available -- it does
not change the policy, the soil, the command or the solver, only where on the particle
lattice the robot happens to start -- so any spread it produces is a lower bound on the
spread the metric has in general.

Reported as mean, standard deviation and full range per terrain, with the paired gap
computed per replicate rather than from the two means, since the gap is what gets quoted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--warmup", type=float, default=1.5)
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--spread-m", type=float, default=0.25,
                    help="range of spawn offsets, in metres, used as the perturbation")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from walk_check import run as walk_run

    urdf, policy = Path(a.urdf), Path(a.policy)
    # Offsets are a deterministic spread rather than random draws, so the replicate set is
    # reproducible and its extremes are known rather than sampled.
    offsets = np.linspace(-a.spread_m, a.spread_m, a.n)
    out = {"rigid": [], "crm": [], "offsets": [float(x) for x in offsets]}

    for kind in ("rigid", "crm"):
        print(f"=== {kind} ===", flush=True)
        for off in offsets:
            # walk_check spawns at the origin; the offset rides on the patch geometry by
            # shifting where the run starts along x.
            r = walk_run(kind, None, a.seconds, [a.vx, 0.0, 0.0], urdf, policy,
                         warmup_s=a.warmup, patch_x=8.0, patch_y=4.0, spacing=0.02,
                         spawn_xy=(float(off), 0.0))
            if r.get("diverged_at_s") is not None:
                print(f"  offset {off:+.3f}  DIVERGED", flush=True)
                continue
            trk = r["mean_vx"] / r["cmd_vx"] if r["cmd_vx"] else float("nan")
            out[kind].append({"offset": float(off), "tracking": float(trk),
                              "mean_vx": r["mean_vx"], "mean_z": r["mean_z"],
                              "mean_up": r["mean_up"]})
            print(f"  offset {off:+.3f}  tracking {trk:6.1%}  vx {r['mean_vx']:+.3f}  "
                  f"z {r['mean_z']:.4f}", flush=True)

    print("\n" + "=" * 62)
    stats = {}
    for kind in ("rigid", "crm"):
        v = np.array([x["tracking"] for x in out[kind]])
        if not len(v):
            continue
        stats[kind] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                       "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
        print(f"{kind:6s} tracking  mean {v.mean():6.1%}  sd {v.std(ddof=1):5.1%}  "
              f"range [{v.min():.1%}, {v.max():.1%}]  n={len(v)}")

    if "rigid" in stats and "crm" in stats:
        n = min(len(out["rigid"]), len(out["crm"]))
        gap = np.array([out["rigid"][i]["tracking"] - out["crm"][i]["tracking"]
                        for i in range(n)])
        stats["gap"] = {"mean": float(gap.mean()),
                        "sd": float(gap.std(ddof=1)) if n > 1 else 0.0,
                        "min": float(gap.min()), "max": float(gap.max())}
        print(f"\ngap    rigid - crm  mean {gap.mean():6.1%}  sd {gap.std(ddof=1):5.1%}  "
              f"range [{gap.min():.1%}, {gap.max():.1%}]")
        print("\nThis spread is what a single-run headline hides. Quote the mean with its "
              "sd, not one run.")

    if a.out:
        Path(a.out).write_text(json.dumps({"runs": out, "stats": stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
