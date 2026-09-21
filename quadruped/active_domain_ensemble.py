#!/usr/bin/env python3
"""Does the active domain bias BEHAVIOUR? Decided over an ensemble, against no
approximation at all.

`active_domain_study.py` established that a single trajectory cannot answer this. The
gait is chaotic: perturb the soil model slightly and the trajectories separate at a rate
that has nothing to do with how wrong the model is, and final-position error orders
non-monotonically in box size with the velocity difference changing sign.

What a corpus actually needs is not that individual trajectories match. It is that the
DISTRIBUTION of behaviour is unbiased -- that the robot is not systematically faster, or
riding systematically higher, because the soil outside a box was frozen. Chaotic scatter
averages down over an ensemble; a systematic shift does not. So: many cases drawn from the
real command distribution, each run at each active-domain setting, paired differences
taken within a case, and the MEAN paired difference reported with its standard error.

The reference is `none` -- no active domain at all. A comparison against a merely larger
box cannot distinguish "0.5 m agrees with 2.0 m" from "0.5 m and 2.0 m share a bias," and
the patch-cost benchmark showed the unapproximated solve runs at 37x real time, which is
affordable for a calibration even though it is not affordable for a corpus.

Two consequences of the reference being unapproximated, both of which constrain the setup:

  - With no active domain, EVERY particle is simulated, so the reference cost is
    proportional to bed size. The bed is therefore held at the established 8 x 4 x 0.2
    rather than widened, and commands are bounded so the robot stays on it.
  - The robot must stay on the bed for the whole window or the comparison is between a
    walking robot and a falling one. Spawn is placed at the near edge for the commanded
    direction, exactly as the collector does it.
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

from quadruped.active_domain_study import PATCH_X, PATCH_Y, run_case  # noqa: E402

# Bounded so a 3 s window plus warmup keeps the robot on an 8 x 4 bed with a metre of
# margin. These are narrower than the collection ranges in excitation.yaml: the point is
# a representative spread of gaits to average over, not coverage of the command space.
VX_RANGE = (-0.9, 1.2)
VY_RANGE = (-0.45, 0.45)
WZ_RANGE = (-1.0, 1.0)
START_INSET = 1.0


def make_cases(n, seed=20260920):
    """Commands drawn from a bounded version of the real distribution, with the spawn
    placed at the near edge for the direction commanded -- the collector's own rule."""
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        vx = float(rng.uniform(*VX_RANGE))
        vy = float(rng.uniform(*VY_RANGE))
        wz = float(rng.uniform(*WZ_RANGE))
        sx = 0.0 if abs(vx) < 1e-6 else -np.sign(vx) * (PATCH_X / 2 - START_INSET)
        sy = 0.0 if abs(vy) < 1e-6 else -np.sign(vy) * (PATCH_Y / 2 - START_INSET)
        cases.append((f"c{i:02d}", (vx, vy, wz), (float(sx), float(sy))))
    return cases


def behaviour(traj):
    """The aggregate quantities a corpus is actually made of.

    Deliberately not trajectory shape. These are what the study reports and what a
    fine-tuned policy is scored on, and each is an average over the window, so gait phase
    scrambling cancels rather than accumulating.
    """
    return {
        "mean_vx": float(np.mean(traj[:, 3])),
        "mean_vy": float(np.mean(traj[:, 4])),
        "mean_wz": float(np.mean(traj[:, 6])),
        "mean_z": float(np.mean(traj[:, 2])),
        "mean_up": float(np.mean(traj[:, 7])),
        "speed": float(np.mean(np.hypot(traj[:, 3], traj[:, 4]))),
    }


KEYS = ("mean_vx", "mean_vy", "mean_wz", "mean_z", "mean_up", "speed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260920,
                    help="case-generation seed; a second machine runs a "
                         "different seed so the two sets pool into one n")
    ap.add_argument("--active", default="0.5,1.0")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--free-flow-s", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ads = [float(x) for x in a.active.split(",")]
    urdf, policy = Path(a.urdf), Path(a.policy)
    cases = make_cases(a.n, seed=a.seed)

    # THE NULL ARM IS A SECOND REFERENCE RUN, identical in every setting, and it is
    # carried through the whole analysis exactly like a real arm.
    #
    # It exists because determinism turned out to be case-dependent rather than universal.
    # Two cases reproduced bit-identically; a third, with a lateral command, differed
    # between two identical runs by 0.019 m and 0.010 m/s in mean vx. That last number is
    # a quarter of the effect being looked for, so without a control there is no way to
    # tell "0.5 m biases velocity by 0.04" from "any two CRM runs differ by 0.04".
    #
    # With the null arm the test becomes a comparison rather than an assumption: an arm is
    # only evidence of bias if its mean paired difference stands out against the null
    # arm's, which is built from the same cases and the same estimator.
    ARMS = ["null"] + [str(x) for x in ads]
    rows = {k: [] for k in ARMS}
    per_case = []

    print(f"{a.n} cases, reference = none (unapproximated), "
          f"arms = {ARMS} ('null' is a repeat reference run), "
          f"{a.seconds}s window\n", flush=True)

    for name, cmd, spawn in cases:
        ref = run_case(cmd, None, a.seconds, a.warmup, urdf, policy, spawn, a.free_flow_s)
        if ref["diverged_at_s"] is not None:
            print(f"{name}: reference diverged, dropped", flush=True)
            continue
        bref = behaviour(ref["traj"])
        rec = {"case": name, "cmd": cmd, "spawn": spawn, "none": bref, "arms": {}}
        line = (f"{name} cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) "
                f"none: vx {bref['mean_vx']:+.3f} z {bref['mean_z']:.4f}")
        for arm in ARMS:
            ad = None if arm == "null" else float(arm)
            r = run_case(cmd, ad, a.seconds, a.warmup, urdf, policy, spawn, a.free_flow_s)
            if r["diverged_at_s"] is not None:
                line += f" | {arm} DIVERGED"
                continue
            b = behaviour(r["traj"])
            d = {k: b[k] - bref[k] for k in KEYS}
            rows[arm].append(d)
            rec["arms"][arm] = {"behaviour": b, "delta": d,
                                "ms_per_step": r["ms_per_step"]}
            line += f" | {arm}: dvx {d['mean_vx']:+.3f} dz {d['mean_z']:+.5f}"
        per_case.append(rec)
        print(line, flush=True)

    print("\n" + "=" * 78)
    print("PAIRED MEAN DIFFERENCE vs no active domain (mean +/- standard error)")
    print("=" * 78)
    summary = {}
    for ad in ARMS:
        d = rows[ad]
        if not d:
            continue
        n = len(d)
        summary[ad] = {"n": n}
        label = ("null (reference run twice -- this is the noise, not an effect)"
                 if ad == "null" else f"active domain {ad} m")
        print(f"\n{label}   (n = {n})")
        for k in KEYS:
            v = np.array([x[k] for x in d])
            m = float(np.mean(v))
            se = float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            # t is the number that decides it: a systematic bias survives averaging, a
            # chaotic difference does not, and |t| < 2 means this ensemble cannot
            # distinguish the approximation from the real thing.
            t = m / se if se > 0 else float("nan")
            summary[ad][k] = {"mean": m, "se": se, "t": t}
            flag = "  <-- systematic" if abs(t) >= 2.0 else ""
            print(f"  {k:9s} {m:+.5f} +/- {se:.5f}   t = {t:+6.2f}{flag}")

        # DOES THE BIAS GROW WITH SPEED? Stage 1 hinted that it does: on the slow case
        # the velocity differences were mixed in sign, while on the fast case every box
        # smaller than the reference ran faster. The mechanism would be that a quicker
        # robot disturbs soil further ahead of itself, so a fixed box clips more of the
        # region that matters. If real, the box has to be sized for the fastest command
        # in the corpus, not the average one -- and the collection ranges go to 1.5 m/s.
        spd = np.array([c["none"]["speed"] for c in per_case
                        if ad in c["arms"]])
        dv = np.array([c["arms"][ad]["delta"]["mean_vx"] for c in per_case
                       if ad in c["arms"]])
        if len(spd) >= 4 and np.std(spd) > 1e-9:
            rho = float(np.corrcoef(spd, dv)[0, 1])
            slope = float(np.polyfit(spd, dv, 1)[0])
            summary[ad]["bias_vs_speed"] = {"pearson_r": rho, "slope": slope,
                                            "n": int(len(spd))}
            print(f"  bias vs speed: r = {rho:+.3f}, slope = {slope:+.4f} "
                  f"(m/s of bias per m/s of speed)")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"cases": per_case, "summary": summary}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
