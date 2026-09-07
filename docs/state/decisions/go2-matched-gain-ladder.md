# The four-arm ladder at matched gain, on two independent machines

**Status:** closed for the tracking half · **Date:** 2026-09-07 · **Machines:** a3-ubuntu, sliger-ubuntu

## What this is

Every fine-tuned arm plus the base controller, run on **one corpus, at one gain,
on one physics build**, on each of two machines that were never used for this
before. That comparison had not previously existed: earlier ladders mixed gains
between the treated and baseline arms, which is the confound that accounted for
every tracking result on sbel-pc (see `go2-asymmetric-protocol-audit.md`).

Each machine collected its own corpus (`go2_cell_a3`, `go2_cell_sliger`) and ran
its own episodes, so the two columns are independent replications, not two views
of one dataset. Builds differ: a3 is znver3/gcc 11, sliger is znver4/gcc 13.

## Result

Treated arm minus base controller, both at k=0.65, paired per episode, median with
a per-episode bootstrap interval (n = episodes, not windows):

| machine | arm | n | matched gain | as % of baseline error |
|---|---|---:|---|---:|
| a3 | v4 | 214 | -0.00015 [-0.00050, +0.00013] | -0.15% |
| a3 | armA | 214 | -0.00021 [-0.00090, +0.00027] | -0.20% |
| a3 | armB | 205 | -0.00047 [-0.00095, +0.00002] | -0.45% |
| sliger | v4 | 218 | -0.00012 [-0.00029, +0.00028] | -0.12% |
| sliger | armA | 218 | -0.00044 [-0.00086, -0.00002] | -0.45% |
| sliger | armB | 218 | -0.00045 [-0.00158, -0.00007] | -0.47% |

Median baseline error 0.10359 (a3), 0.09638 (sliger).

**Every arm is within half a percent of baseline error, and no arm is
distinguishable from any other.** Cross-machine, every arm's two intervals
overlap, so the measurement reproduces across corpora and builds.

## Two of the six intervals exclude zero. That is not a finding.

sliger's armA `[-0.00086, -0.00002]` and armB `[-0.00158, -0.00007]` exclude zero.
They are also **-0.45% and -0.47% of baseline error**. Quoted with the interval and
without the denominator this reads as a confirmed improvement; it is a real
difference of no consequence, and the same shape as cell3's
`[+0.00000, +0.00001]`. The acceptance criterion was **-0.020**, forty times
larger than anything measured here.

The consistent negative sign across all six cells is suggestive and **not** worth a
sign test: the three arms on a machine share one control term, so they are not
independent draws.

## The multiplier's own effect is strongly corpus-dependent

| machine | base@0.65 - base@1.0 | as % of baseline error |
|---|---|---:|
| a3 | +0.00064 [-0.00038, +0.00485] | +0.62% |
| sliger | +0.00339 [+0.00022, +0.01002] | +3.52% |

**5x apart, same nominal gain change, two corpora built to the same design.**
This is a third independent instance of the corpus-dependence sbel-pc found
between cell3 (+0.00969) and cell4 (+0.00075), and it is why an estimate of the
multiplier's effect measured on one corpus must not be carried to another.

## Matching removed variance, not only bias

The unmatched intervals on the same episodes span up to `+0.008`, roughly ten
times wider than the matched ones. The multiplier's effect is common-mode across
the pair, so matching cancels it and the residual is far better determined. **A
protocol fix that was made for correctness also bought an order of magnitude of
precision**, which is worth knowing when the next comparison is designed.

## What this closes and what it does not

Closed: the tracking half. Bounded at half a percent of baseline error, across
four arms, two corpora, two builds, direction unresolved.

Not touched: the stability half. `k*` is under a discriminative-validity test on
dorm-pc (v4 at 20/43 against armA at 0/43, n=40 per rung) after a power analysis
showed n=8 gave sampling noise equal to 71% of the crossing bracket.
