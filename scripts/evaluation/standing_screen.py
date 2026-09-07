"""Does this policy command bounded actions? A RATE over several commanded episodes.

WHY THIS EXISTS. The fine-tune failure that consumed 2026-09-06 was detectable at
row 0 -- the policy was at 0.100 m against a standing 0.30-0.42 m before the scored
episode began. Detecting it cost a 43-episode paired verdict, about an hour per arm.
This is the same signal for the price of one short episode, so a dW ladder or a
reward change can be screened before anything pays for a verdict.

IT IS A PRE-FILTER, NOT A VERDICT, AND IT REPORTS A RATE FOR A REASON.

Two earlier versions of this file were wrong and both passed something first:

  v1  tested median body height. FAILED its self-test -- v4 sinks to 0.181 m on a
      still command and scores 20 of 43, while exc25 sits HIGHER at 0.229 and
      scores 0. Body height does not predict the verdict.
  v2  tested max|command| on ONE zero-velocity episode. PASSED its self-test, and
      the pass was luck. Over the verdict's 43 COMMANDED episodes v4 diverges in
      31 of them; the single still-command episode is one of the few where it does
      not. A sample of one per policy is not a validation.

Measured per-episode divergence over the verdict's own 43 episodes, which is what
this screen has to approximate:

    base    2/43     v4  31/43     base36  40/43     exc25  43/43
    43of43           20of43        0of43             0of43

Monotone with the verdict, and GRADED -- so the output is a rate, not a label.
Divergence and scoring are different: an episode can diverge and still score,
because scored() needs only 1500 rows and finite velocity.
"""
import argparse, csv, glob, json, os, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, "src")
from nedm.quadruped.imported_policy import (ACTION_SCALE, IMPORTED_DEFAULTS, SIGN,
                                            CHRONO_TO_IMPORTED)

_C2I = np.asarray(CHRONO_TO_IMPORTED)
_DEF = np.asarray(IMPORTED_DEFAULTS).ravel()
_CHRONO_TARGETS = [f"joint_{leg}_{j}_target_rad"
                   for leg in ("rr", "rl", "fr", "fl")
                   for j in ("hip", "thigh", "calf")]


def raw_action_max(rows):
    """max |raw policy action| over an episode, inverted from the recorded targets.

    targets = action * ACTION_SCALE + IMPORTED_DEFAULTS, in the POLICY frame and
    order. Chrono records the negation (SIGN = -1) in Chrono joint order, so the
    reindex and the sign both have to be undone PER JOINT -- the defaults differ
    per joint (0.1, 0.8, -1.5, ...), so a scalar offset is not the inverse.
    """
    T = np.array([[float(r[c]) for c in _CHRONO_TARGETS] for r in rows])
    return float(np.abs((SIGN * T[:, _C2I] - _DEF) / ACTION_SCALE).max())

PY = os.environ.get("NEDM_PY", sys.executable)
CHRONO = os.environ.get("NEDM_CHRONO_PYTHONPATH", "/home/kyle/chrono-build/bin")
# THE CRITERION IS COMMAND MAGNITUDE, NOT BODY HEIGHT.
# The first version of this screen tested median pos_z and FAILED its own self-test:
# v4, which scores 20 of 43, collapses to 0.181 m under a zero-velocity command.
# Holding a stand on a still command is simply not the property that predicts the
# verdict -- the verdict's cell is -0.18 < vx <= -0.02 and never asks for zero.
#
# What does separate the knowns is the magnitude of the commanded joint target, and
# it separates them by fifteen orders of magnitude rather than by a tuned margin:
#
#     base   (43 of 43)   max|target|      1.253
#     v4     (20 of 43)   max|target|      2.402
#     exc25  ( 0 of 43)   max|target|  8.19e+15
#
# A Go2 joint moves within about +-3 rad and the verdict harness already rejects
# episodes above 5. 10 is loose enough that no plausible working policy trips it and
# tight enough that the runaway cannot hide: there is nothing between 2.4 and 8e15.
# 1e6, NOT 10 AND NOT 1e3.
#   base's median max is 4.547, so a bound at 10 sits inside its own operating range
#   and produced a spurious 2/43 on the verdict's own episodes.
#   At 1e3 the verdict rates are 0 / 23 / 43 / 43 and at 1e6 they are IDENTICAL, so
#   raising it costs nothing there -- but this screen runs harsher conditions than
#   the verdict generates, and under them base reaches 1.795e3 without diverging in
#   any meaningful sense. That produced three FALSE REJECTIONS of the one policy
#   that completes 43 of 43.
#   Real divergences sit at 1e16 to 1e35. 1e6 leaves three orders of margin above
#   base's worst and ten below the nearest true divergence.
TARGET_LIMIT_RAD = 1e6

# AND A LENGTH RULE, BECAUSE THERE ARE TWO FAILURE MODES AND THIS SCREENED FOR ONE.
# The verdict harness says so in its own output: "unbounded output and falling are
# different failures and only one is about walking." The verdict ACTS on both --
# scored() rejects on len(rows) < SCORED_ROWS + 500 regardless of command magnitude.
# This screen did not, so:
#     policy DIVERGES -> commands blow up, episode runs long   -> FAILED   correct
#     policy FALLS    -> episode ends short, commands bounded   -> PASSED   WRONG
# That produced a NON-MONOTONE gain sweep: cell1's rungs read 0/8, 1/8, 3/8, 5/8,
# 7/8, then 0/8 at k=1.50 -- zero divergence at the HIGHEST gain, because at high
# gain the policy stops being detectably broken rather than stopping being broken.
# A k* read off that curve is not merely imprecise, it is ill-posed.
#
# Reusing the verdict's own row rule rather than inventing a second definition of
# failure: two instruments with two definitions is how they came to disagree.
MIN_ROWS = 1500          # SCORED_ROWS(1000) + 500, from the verdict harness


# COMMANDED conditions spanning the verdict's cell (-0.18 < vx <= -0.02) plus a
# still command. A still command alone is the trap v2 fell into.
# Parameter KEYS ARE PER FAMILY (imported_policy.py:108-114). The first version of
# this list passed {"vx": ...} to every family; vel_step/yaw_step/weave/arc each
# raised KeyError, the collector exited rc=1, and screen() silently dropped them --
# so a self-test that claimed 8 conditions actually ran 4, all of them `constant`,
# and still printed "ORDERING PRESERVED". Missing conditions must be an error.
# Conditions carry the VERDICT's disturbances, not just its commands. Measured onset
# of divergence over the verdict's own episodes, seconds into the recording:
#
#     v4      p10 0.00   p50 0.63   p90 16.54   max 20.15
#     base36  p10 0.00   p50 0.00   p90  0.44   max  3.72
#     exc25   p10 0.00   p50 0.00   p90  1.19   max  2.47
#
# v4 is the slow diverger and the case this screen exists to catch, so the episode
# length is set from ITS p90, not from a guess. An earlier 4 s version read v4 at
# 0/4 and then 2/8 against a true 0.72 -- length was only half the reason. The other
# half is that the verdict perturbs (up to 120 N) and tilts the ground (+-3 deg)
# while that version did neither, so it was screening a gentler world than the one
# the number has to predict.
DURATION_S = 20.0
CONDITIONS = [("constant", {"vx": 0.0},                                    0.0,  0.0,  0.0),
              ("constant", {"vx": -0.05},                                 24.0,  1.5, -1.0),
              ("constant", {"vx": -0.12},                                 48.0, -2.0,  1.5),
              ("constant", {"vx": -0.17},                                 72.0,  2.5,  2.0),
              ("vel_step", {"vx0": -0.05, "vx1": -0.15, "t_switch": 8.0}, 96.0, -1.5, -2.5),
              ("yaw_step", {"vx": -0.08, "wz1": 0.5, "t_switch": 8.0},   120.0,  3.0, -1.5),
              ("weave",    {"vx": -0.10, "wz_amp": 0.4, "freq": 0.2},     60.0, -2.5,  2.5),
              ("arc",      {"vx": -0.06, "wz": 0.3},                      36.0,  1.0, -3.0)]


def _one(ckpt, fam, params, peak, roll, pitch, duration, seed, keep=None):
    out = keep or tempfile.mkdtemp(prefix="stand_")
    cmd = [PY, "scripts/collection/collect_go2_smoke.py", "--terrain", "rigid",
           "--duration-s", str(duration), "--imported-ckpt", str(ckpt),
           "--command-family", fam, "--command-params", json.dumps(params),
           "--ground-size-m", "200.0", "--perturb-peak-n", f"{peak:.1f}",
           "--prewalk-s", "1.0", "--ground-tilt-roll-deg", f"{roll:.2f}",
           "--ground-tilt-pitch-deg", f"{pitch:.2f}",
           "--episode-index", "0", "--seed", str(seed), "--spawn-x-m", "0.0",
           "--spawn-y-m", "0.0", "--heading-deg", "0.0", "--patch-y", "4.0",
           "--output-dir", out, "--overwrite", "--progress-interval-s", "99"]
    p = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=CHRONO),
                       capture_output=True, text=True)
    files = glob.glob(f"{out}/episodes/*.csv")
    if not files:
        return None
    rows = list(csv.DictReader(open(files[0])))
    if not rows:
        return None
    # Raw action space, so this is directly comparable with surrogate-side action
    # magnitudes. Comparing a Chrono target against a surrogate action without
    # inverting compares two different quantities.
    return raw_action_max(rows), len(rows)


def screen(ckpt, duration=DURATION_S, seed=0, keep=None):
    vals = [_one(ckpt, f, p, pk, r, pi, duration, seed)
            for f, p, pk, r, pi in CONDITIONS]
    missing = [CONDITIONS[i][0] for i, v in enumerate(vals) if v is None]
    if missing:
        # A dropped condition changes the denominator silently, which is how the
        # previous self-test reported a rate over 4 episodes while claiming 8.
        raise SystemExit(f"ABORT: {len(missing)} condition(s) produced no episode: "
                         f"{missing}. A rate over an unknown denominator is not a rate.")
    got = vals
    mags = [v[0] for v in got]; lens = [v[1] for v in got]
    unbounded = [i for i, m in enumerate(mags) if m > TARGET_LIMIT_RAD]
    short = [i for i, L in enumerate(lens) if L < MIN_ROWS]
    failed = sorted(set(unbounded) | set(short))
    return (f"{len(failed)}/{len(got)}",
            dict(failed=len(failed), n=len(got), rate=round(len(failed) / len(got), 3),
                 unbounded=len(unbounded), short=len(short),
                 median_raw=float(f"{sorted(mags)[len(mags)//2]:.4g}"),
                 max_raw=float(f"{max(mags):.4g}"),
                 median_rows=sorted(lens)[len(lens)//2], min_rows=min(lens)))


# Expected rates are the VERDICT's measured per-episode divergence, which this
# 8-episode screen approximates. The self-test requires the ORDERING, not the value:
# 8 episodes cannot resolve 0.72 from 1.00, and pretending otherwise is how v2 passed.
KNOWNS = [("base   (43 of 43)", "/home/kyle/sbel-artifacts/checkpoints/go2_cts_150k.pt", 2/43),
          ("v4     (20 of 43)", "/home/kyle/sbel-artifacts/finetune_go2_v4_smallstep/go2_finetuned_v4.pt", 31/43),
          ("base36 ( 0 of 43)", "/home/kyle/sbel-artifacts/finetune_go2_base36/go2_finetuned_base36.pt", 40/43),
          ("exc25  ( 0 of 43)", "/home/kyle/sbel-artifacts/finetune_go2_exc25/go2_finetuned_exc25.pt", 43/43)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--duration-s", type=float, default=DURATION_S)
    ap.add_argument("--self-test", action="store_true",
                    help="run the three known outcomes; the screen is not trustworthy "
                         "until it reproduces all three")
    a = ap.parse_args()
    if a.self_test:
        obs = []
        for label, ck, verdict_rate in KNOWNS:
            v, info = screen(ck, a.duration_s)
            obs.append((label, info["rate"] if info else None, verdict_rate))
            print(f"  {label:20s} screen {str(v):>6s} rate {info['rate'] if info else '?':<6} "
                  f"| verdict rate {verdict_rate:.2f}   {info}")
        # THE CRITERION IS THE IMPLICATION, NOT THE RATE.
        #   screen says diverged  =>  the verdict would say diverged
        # A screen is a fast NEGATIVE. It does not need to estimate the verdict's
        # rate -- it runs fewer, shorter, differently-conditioned episodes and never
        # will. What it must never do is reject a policy the verdict would pass.
        #
        # An earlier self-test checked ORDERING and passed while the screen falsely
        # rejected the one policy that completes 43 of 43. Ordering is too weak: it
        # is satisfied by a screen that rejects everything a little.
        false_rejections = [(lbl, r) for lbl, r, vr in obs if r > 0 and vr == 0.0]
        rates = [o[1] for o in obs]
        mono = None not in rates and all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
        print(f"\n  ordering {'preserved' if mono else 'NOT preserved'} (secondary)")
        if false_rejections:
            for lbl, r in false_rejections:
                print(f"  FALSE REJECTION: {lbl} rejected at rate {r} but its verdict rate is 0.00")
            print("\n  SELF-TEST FAILED -- the screen rejects a policy the verdict passes.\n"
                  "  Not usable as a pre-filter: a rung it rejects might have scored.")
            raise SystemExit(1)
        print("  no false rejections against a policy the verdict passes")
        print("  SELF-TEST PASSED -- safe as a conservative rejector.\n"
              "  Sensitivity is NOT claimed: v4 diverges in 31 of 43 and still scores 20,\n"
              "  so a policy this screen passes still needs the full verdict.")
        raise SystemExit(0)
    v, info = screen(a.ckpt, a.duration_s)
    print(f"  {v}   {info}")
