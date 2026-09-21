# How the study's metrics have to be measured

**Updated:** 2026-09-20. Measured on a3, `tracking_spread.py`, 8 replicates per terrain,
3 s windows at 0.5 m/s commanded, corrected bed geometry (`89c898b8`).

## The tracking metric is not repeatable on CRM, and is on rigid

Eight runs per terrain, identical in every respect except where on the particle lattice
the robot starts. The spawn is offset over +/-0.25 m; the policy, soil, command, solver and
protocol are untouched.

| terrain | mean tracking | sd | range |
|---|---|---|---|
| rigid | 97.0% | **0.1%** | 96.8 - 97.1 |
| CRM | 70.2% | **3.4%** | 65.8 - 75.2 |
| gap (paired) | 26.8% | 3.4% | 21.6 - 31.3 |

**The rigid control is effectively deterministic. All of the variance is the granular
terrain.** That is the expected signature of contact-rich dynamics on a particle bed, and
it is consistent with the active-domain work, where the same chaos made trajectory error
order non-monotonically in box size.

Note the perturbation here is about as small as one can make while still changing
anything, so **3.4 points is a lower bound** on the metric's variance, not an estimate of
it. Command, soil realisation and episode length would all add more.

## What this invalidates

**The headline "95% rigid against 64% CRM" was a single run per terrain.** The CRM figure
sits below the entire range observed here, and the two numbers were measured on different
machines from the replicate set. The defensible statement is a gap of roughly 27 points
with a standard deviation of 3.4, not a gap of 31 points.

More importantly it sets a floor on what any fine-tuning result has to clear. With
sd = 3.4 points, a single-run improvement of 3 points is one standard deviation and means
nothing. The standard error on a mean of n replicates is 3.4/sqrt(n), so:

| replicates | standard error | smallest credible improvement (2 se) |
|---|---|---|
| 1 | 3.4 | 6.8 points |
| 4 | 1.7 | 3.4 points |
| 8 | 1.2 | 2.4 points |
| 16 | 0.85 | 1.7 points |

**Every reported CRM metric needs replicates.** Not because the simulation is unreliable,
but because the system is chaotic and one trajectory is one draw from a distribution.

This applies retroactively to the previous study's tracking-gain figure, which was also
quoted from single runs. That number is not withdrawn -- it has not been re-measured --
but it carries an unstated uncertainty of the same order, and it should not be quoted to
a precision the method cannot support until it has been replicated.

## Rules

1. **Any CRM number that goes in a document is a mean over replicates, with its sd and n.**
   A single CRM run is a diagnostic, not a result.
2. **Replicates vary the initial condition, not the seed of a sampler.** Spawn offset,
   command draw and soil realisation are the axes that matter; re-running an identical
   configuration reproduces it exactly on some cases and not others (see below), so it is
   not a replicate.
3. **Rigid may be quoted from fewer runs**, since its sd is 0.1 points, but state n.
4. **Improvements are reported as paired differences per replicate**, not as a difference
   of two means, so the chaotic component cancels where the pairing allows it.

## Determinism is case-dependent

Two identical CRM runs reproduce bit-identically on some cases and not on others. Three of
four active-domain cases had a noise floor of exactly zero; the fourth, a lateral command,
differed by 0.019 m in position and 0.0097 m/s in mean velocity between two runs with
identical inputs.

So "re-run it and see if you get the same answer" is not a valid check here -- it can
return a false confirmation. Any comparison of two configurations needs a **null arm**: a
repeat of the reference carried through the same analysis, so an effect is judged against
what identical inputs actually produce rather than against an assumption of zero.
