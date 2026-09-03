# Lessons: experiment design

Failures where the *experiment* could not have answered its question, whatever
the code did. These are the expensive ones, because the compute is spent before
anyone notices, and the output looks like a result.

## Ask what it would look like if the thing did nothing

**Cost:** four instances in one day · **Found:** 2026-09-03 · **Applies to:** every gate, test, and A/B

The single question that caught all four:

> **What would this look like if the mechanism under test did nothing at all?**

If the answer is "the same, or close enough that I could not tell," the
experiment is not an experiment yet. It is cheap to ask, it takes one line of
arithmetic or one careful read, and on 2026-09-03 it paid four times:

| failure | what "did nothing" looked like |
|---|---|
| G0a gate passing a fallen robot | base-z criterion satisfied by a robot on its back |
| `AttachFsiSphSystem` empty body | returns `-1`, compiles, links, no error |
| CUDA silently disabled | configure exits 0, two warnings, wrong modules built |
| enforcement A/B on collisions | **P(0/100) = 0.366 under the null** |

All four report success while doing nothing. None is detectable from exit
status, and none was caught by testing harder — only by asking what the null
would produce.

**The corollary: establish what counts as a distinguishable outcome BEFORE
spending the compute.** Fix pass criteria before running. Compute detectability
before launching. A criterion chosen after seeing output is not a criterion.

## Choose the metric before worrying about the sample size

**Cost:** ~100 episodes scored on a readout that could not see the effect · **Found:** 2026-09-03 · **Applies to:** any rare-event evaluation

**Expected:** enforcement reduces collisions, so count collisions.
**Happened:** at a 1% baseline, a perfect intervention still yields a
null-looking 0/100 with probability 0.366, and Fisher on 1/100 vs 0/100 gives
p = 1.000. The experiment could not distinguish perfect from useless.

The fix was not more runs. It was **the same 100 episodes read differently**.
Measured on `kyle-N7-B650E` against the observed unenforced distribution
(n=100, mean 3.067 m, sd 1.146 m), 2000 sims per point, α=0.05 two-sided:

| shift | in SD | power |
|---|---|---|
| 0.00 m | 0.00 | **0.05** ← calibration check, lands where it must |
| 0.20 m | 0.17 | 0.37 |
| **0.40 m** | 0.35 | **0.80** |
| 0.75 m | 0.65 | **1.00** |

The expected effect is ~0.80 m. **So on the continuous readout the effect is
essentially certain to be detected; on the binary one it was essentially certain
to be missed.** Same episodes, same compute, same physics, roughly an order of
magnitude difference in what the experiment can see.

**A rare binary event is usually the least informative function of a continuous
measurement you already have.** Collided/didn't discards every episode that came
close, and "came close" is most of the signal.

**Validate the machinery against a known answer first.** The same code scored
against its own data returned `U=5000, z=0.000, p=1, Cliff's δ=+0.000` — the
degenerate answer it must give, which also reveals the bootstrap's resolution
(median difference CI ±0.53 m) before any real comparison is attempted. And the
0.00-shift row returning exactly 0.05 says the test is calibrated rather than
merely optimistic.

**Pre-register the expected effect, with its provenance.** The 0.80 m
expectation comes from 12 seeds of *planning geometry*, not physics, and the
driven trajectory is not the planned one. It is an order-of-magnitude
expectation, not a prediction. **The power curve does not depend on it** — which
is the property that makes the pre-registration honest rather than decorative.

## Measure the noise floor before you compare anything to it

**Cost:** none, applied before the cross-API comparison · **Found:** 2026-09-03 · **Applies to:** every A-vs-B on a stochastic simulator

The cross-API check asks whether a Go2 run under the source build matches one
under conda 10.0.0. **The obvious version of that question has no answer.** GPU
SPH is not bit-reproducible — atomic accumulation order varies between runs on
identical hardware — and the system has intermittent contact, so trajectories
diverge from arbitrarily small differences. Two runs of the *same* build differ.

So "do the APIs agree" is unanswerable until we know **how much a build
disagrees with itself.**

**Procedure: run the conda arm N times first, unchanged.** The spread of those
runs is the noise floor. Only then run the source arm, and ask whether it falls
*inside* that spread. A source-vs-conda difference smaller than conda-vs-conda
is agreement; one substantially larger is a real API difference.

Without the noise floor there is no criterion at all, and whatever difference
appears gets argued about after the fact — which is the failure this file
exists to prevent.

**This is the same move as scoring a dataset against itself** to check the
statistics machinery returns the degenerate answer, and the same move as
asking what the null would produce. Establish what "no effect" looks like
*with your own instrument, on your own hardware*, before interpreting an effect.

**Corollary: separate the variables.** Compare physics **headless** first, so an
API difference cannot be confounded with a rendering difference. Only then turn
the camera on. Two changes landing at once produce a difference nobody can
attribute.

## Check that A and B ran on the same population before attributing anything

**Cost:** would have shipped a causal claim the data does not support · **Found:** 2026-09-03 · **Applies to:** any A/B where the treatment can reject a sample

The enforcement A/B returned p = 3.65e-06, Cliff's δ +0.379, a +1.216 m median
shift with a CI excluding zero, and a fall from 45% to 9% below the bound. Every
statistic was correct and the pre-registered power analysis held.

**On the 60 episodes where both arms ran the identical layout, the paired
difference was bit-identical: 0 of 60 improved.** The treatment changed nothing
it was applied to. The whole effect came from the treatment *rejecting* samples,
which caused the sampler to draw replacements, so the two arms were scored on
different and unequally difficult populations.

**A treatment that can reject a sample silently redefines the population it is
evaluated on.** The between-arm comparison then measures selection, not effect,
and it does so while producing entirely respectable statistics.

**Always run the paired comparison on the shared subset**, and report how many
samples the arms actually share. Here 40 of 100 differed, which was itself the
signal. If the treatment can refuse, assume it is re-selecting until shown
otherwise.
