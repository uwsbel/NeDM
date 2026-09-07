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
import argparse, hashlib, csv, glob, json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
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

# A SELECTOR THAT SILENTLY DOES NOTHING IS WORSE THAN A WRONG ONE, because a wrong one
# eventually produces a visible contradiction and this produces a clean run on the other
# binary. `_one` sets PYTHONPATH=CHRONO on every collector subprocess, REPLACING whatever
# the caller exported. Where CHRONO holds no pychrono the import falls through to conda:
# no error, no warning, nothing different in the log.
#
# Measured on kyle-sbel 2026-09-07: the default path does not exist there (its source
# build is at ~/Documents/sbel/chrono-build/bin), so every sweep ran on conda's
# _core.so 8e9e3865 while the same session's verdict work ran on source 3b0bd530, and
# the three other boxes each ran their own local build. The physics build is a condition
# of the measurement and was recorded nowhere.
def _resolved_chrono():
    """(pychrono.__init__ path, _core.so md5) as a collector subprocess would see it."""
    out = subprocess.run(
        [PY, "-c", "import pychrono,os;f=os.path.join(os.path.dirname(pychrono.__file__),"
                   "'_core.so');print(pychrono.__file__);print(f)"],
        env=dict(os.environ, PYTHONPATH=CHRONO), capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"ABORT: cannot import pychrono under PYTHONPATH={CHRONO}\n"
                         f"{out.stderr.strip()}")
    init, so = out.stdout.strip().splitlines()[:2]
    md5 = hashlib.md5(open(so, "rb").read()).hexdigest() if os.path.exists(so) else None
    if not os.path.exists(os.path.join(CHRONO, "pychrono", "_core.so")):
        raise SystemExit(
            f"ABORT: NEDM_CHRONO_PYTHONPATH={CHRONO} contains no pychrono/_core.so, so "
            f"setting it has NO EFFECT and the run silently used\n  {init}\n"
            f"  md5 {md5}\n"
            f"Point NEDM_CHRONO_PYTHONPATH at the build you mean, or unset it to declare "
            f"that the ambient interpreter's pychrono is intended.")
    return init, md5

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


_SEEN_FAILURES = set()   # distinct collector errors already reported


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
        # The subprocess's stderr was captured and then discarded, so a fully failed
        # sweep reported "40 conditions produced no episode" and nothing about WHY.
        # On a3 every episode was dying on a missing URDF, because that box keeps its
        # assets at a path DEFAULT_ASSETS does not cover -- the exact failure
        # collect_go2_smoke's own comment predicts. Recovering that took a manual
        # re-run of the collector. Surface the reason, deduplicated so 40 identical
        # failures print once.
        why = (p.stderr or p.stdout or "").strip().splitlines()
        msg = why[-1] if why else f"exit {p.returncode}, no output"
        if msg not in _SEEN_FAILURES:
            _SEEN_FAILURES.add(msg)
            print(f"  [no episode] {fam}: {msg}", file=sys.stderr, flush=True)
        return None
    rows = list(csv.DictReader(open(files[0])))
    if not rows:
        return None
    # Raw action space, so this is directly comparable with surrogate-side action
    # magnitudes. Comparing a Chrono target against a surrogate action without
    # inverting compares two different quantities.
    return raw_action_max(rows), len(rows)


def screen(ckpt, duration=DURATION_S, seed=0, keep=None, concurrency=1, repeats=1):
    """repeats>1 pools REPEATS x len(CONDITIONS) episodes, seeds seed..seed+repeats-1.

    n=8 per rung gives a binomial sd of 0.177 at p=0.5, which is 71% of the entire
    0.85-1.00 rate span the crossing is interpolated inside -- so two arms that
    genuinely differ will very often return the same k*. Resolving dk* = 0.05 needs
    the rate pinned to 0.083, i.e. n ~= 36. Hence repeats.

    Each episode is an independent subprocess with its own temp dir and its own
    --seed, so concurrency cannot change any result; it is verified against serial
    rather than assumed.
    """
    _chrono_init, _chrono_md5 = _resolved_chrono()
    print(f"  pychrono {_chrono_md5[:8] if _chrono_md5 else '?'}  {_chrono_init}", flush=True)
    jobs = [(f, p, pk, r, pi, seed + rep)
            for rep in range(repeats) for f, p, pk, r, pi in CONDITIONS]
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            vals = list(ex.map(lambda j: _one(ckpt, j[0], j[1], j[2], j[3], j[4], duration, j[5]), jobs))
    else:
        vals = [_one(ckpt, f, p, pk, r, pi, duration, sd) for f, p, pk, r, pi, sd in jobs]
    missing = [jobs[i][0] for i, v in enumerate(vals) if v is None]
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
    # PER-EPISODE OUTCOMES, because two arms run the IDENTICAL job list -- same
    # family, params, peak, tilt and seed at every index. So their episodes are
    # matched, and comparing two rates with an unpaired two-proportion test is not
    # merely weaker, it is INVALID: it treats positively correlated observations as
    # independent and returns p-values that are too small. Measured 2026-09-07 on the
    # tilt bands -- unpaired gave 0.012 and 0.049 where McNemar gave 0.023 and 0.057.
    # Note McNemar's p is often LARGER, not smaller: concordant episodes carry no
    # information about the difference, so a matched design with low discordance has
    # a hard floor at 2/2**n_discordant. Record the counts and let the analysis pair.
    per_ep = [{"job": f"{jobs[i][0]}:{jobs[i][5]}", "failed": i in set(failed),
               "unbounded": i in set(unbounded), "short": i in set(short),
               "max_raw": mags[i], "rows": lens[i]} for i in range(len(got))]

    # PER-CONDITION BREAKDOWN. The pooled count is repeats x len(CONDITIONS) and its
    # first factor is a GRID DIMENSION, not a sample size: the repeats inside a
    # condition share family, params, perturbation peak and both tilts, and differ only
    # by seed. The independent unit is the CONDITION, so any statistic computed on the
    # pooled n -- a binomial standard error, a Fisher exact against another arm --
    # overstates its evidence by roughly sqrt(repeats).
    #
    # Recorded because two sessions made this error in opposite directions within an
    # hour on 2026-09-07: one inflated a coincidence by treating 40 as n, the other
    # deflated a p-value from 0.016 to 6e-11 the same way. Keeping the breakdown means
    # the correct unit is always available without a re-run.
    ncond = len(CONDITIONS)
    per_cond = [sum(1 for j, i in enumerate(range(len(got)))
                    if i % ncond == c and i in set(failed)) for c in range(ncond)]
    reps = len(got) // ncond
    return (f"{len(failed)}/{len(got)}",
            dict(failed=len(failed), n=len(got), rate=round(len(failed) / len(got), 3),
                 n_conditions=ncond, repeats_per_condition=reps,
                 per_condition_failed=per_cond,
                 condition_names=[c[0] for c in CONDITIONS],
                 pychrono=_chrono_init, pychrono_core_md5=_chrono_md5,
                 unbounded=len(unbounded), short=len(short),
                 median_raw=float(f"{sorted(mags)[len(mags)//2]:.4g}"),
                 max_raw=float(f"{max(mags):.4g}"),
                 median_rows=sorted(lens)[len(lens)//2], min_rows=min(lens),
                 per_episode=per_ep))


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
