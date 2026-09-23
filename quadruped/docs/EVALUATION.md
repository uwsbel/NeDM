# How the study's metrics have to be measured

**Updated:** 2026-09-20, after a correction. Measured with `diagnostics/tracking_spread.py`, 3 s
windows at 0.5 m/s commanded, corrected bed geometry (`89c898b8`).

## The tracking metric is not repeatable on CRM, and is on rigid

Replicates are identical in every respect except where on the particle lattice the robot
starts. The spawn is offset over a range; policy, soil, command, solver and protocol are
untouched.

| machine | spawn spread | rigid | CRM | CRM sd |
|---|---|---|---|---|
| a3 | +/-0.25 m | 97.0% (sd 0.1) | 70.2% | 3.4 |
| north | +/-0.25 m | 97.0% (sd 0.1) | 78.3% | 3.5 |
| **north** | **+/-1.0 m** | 96.9% (sd 0.1) | **74.7%** | **5.7** |

**The rigid control is effectively deterministic. All of the variance is the granular
terrain.** That is the expected signature of contact-rich dynamics on a particle bed, and
it is consistent with the active-domain work, where the same chaos made trajectory error
order non-monotonically in box size.

## Settled: there is no machine effect, once sampling is wide enough

Both machines, same 8 spawn offsets over +/-1.0 m, same code, same soil:

| machine | CRM tracking | sd |
|---|---|---|
| a3 (RTX 5060 Ti) | 74.4% | 5.8 |
| north (RTX 5070 Ti) | 74.7% | 5.7 |
| **difference** | **+0.3 points** | se 2.88, **t = 0.10** |

A dead null. The earlier "9 points at 6 sigma" was entirely the perturbation being too
narrow: at +/-0.25 m the two machines read 70.2 and 78.3, and at +/-1.0 m they read 74.4
and 74.7. Nothing about the machines changed between those two measurements; only how
widely each one sampled its own distribution did.

So **CRM results are reproducible across machines**, and `diagnostics/machine_probe.py` was right all
along: the two produce bit-identical SPH state for two steps and then diverge at rounding
level, which is one computation amplified rather than two different ones.

Pinning the machine for a given comparison still costs nothing and is still the default,
but it is now a convention rather than a correction for a real effect.

## The correction, and what caused it

An earlier version of this document reported sd = 3.4 points and concluded, from a
9-point difference between a3 and north at 6 sigma, that **CRM results are
machine-dependent**. That conclusion is WITHDRAWN, and has since been positively refuted
by the measurement above rather than merely doubted.

Widening the spawn perturbation on a SINGLE machine from +/-0.25 m to +/-1.0 m moved
north's own mean by 3.6 points and grew its sd from 3.5 to 5.7. Redone against that, the
cross-machine difference is +4.5 points with se 2.34, **t = 1.9, not significant**. The
apparent machine effect was mostly an artifact of an under-dispersed replicate set: at
+/-0.25 m the eight replicates explore too narrow a neighbourhood, the sample sd
understates the true variance, and each machine happens to settle in a different part of
the distribution.

Two lessons worth more than the retracted number:

- **A replicate set that is too narrow does not look noisy, it looks precise.** The
  failure mode is an overconfident sd, and an overconfident sd manufactures significance.
  The tell was available and ignored: rigid had sd 0.1 while CRM had 3.4, so the
  perturbation clearly mattered enormously on CRM, which is exactly when its SIZE needs
  justifying rather than picking.
- **When a mechanism check and a statistic disagree, the mechanism usually wins.**
  `diagnostics/machine_probe.py` showed the two machines produce bit-identical SPH state for two steps
  and then diverge at rounding level -- the signature of the same computation amplified,
  not a different one. That was reported alongside a claim of systematic machine bias, and
  reconciled by inventing "a small systematic component on top". The probe was right and
  the reconciliation was motivated reasoning.

Note +/-1.0 m is still an arbitrary perturbation. **The true sd for the study is at least
5.7 and probably larger**, because a real corpus varies command, soil realisation and
initial state too. It should be measured against the actual episode distribution rather
than a spawn offset.

## What this means for every reported number

**The headline "95% rigid against 64% CRM" was a single run per terrain**, measured on
different machines. No single CRM run supports a headline. On the best current estimate
the gap is roughly 22 points with sd of about 5.7, and that sd is a floor.

Replicate budget, at sd = 5.7. The standard error on a mean of n replicates is 5.7/sqrt(n):

| replicates | standard error | smallest credible improvement (2 se) |
|---|---|---|
| 1 | 5.7 | 11.4 points |
| 4 | 2.9 | 5.7 points |
| 8 | 2.0 | 4.0 points |
| 16 | 1.4 | 2.9 points |
| 32 | 1.0 | 2.0 points |

**Every reported CRM metric needs replicates.** Not because the simulation is unreliable,
but because the system is chaotic and one trajectory is one draw from a wide distribution.

This applies retroactively to the previous study's tracking-gain figure, which was also
quoted from single runs. That number is not withdrawn -- it has not been re-measured --
but it carries an unstated uncertainty of the same order.

## Rules

1. **Any CRM number that goes in a document is a mean over replicates, with its sd and n.**
   A single CRM run is a diagnostic, not a result.
2. **Replicates must span the variation the claim generalises over.** A perturbation
   chosen for convenience understates the variance and manufactures significance. State
   the perturbation and its range next to the sd.
3. **Rigid may be quoted from fewer runs**, since its sd is 0.1 points, but state n.
4. **Improvements are reported as paired differences per replicate**, not as a difference
   of two means. The study's claim is a GAIN -- fine-tuned minus baseline -- and measuring
   both on the same machine in the same session cancels any common offset, which is what
   makes the claim robust even if an absolute level is not.
5. **Pin the machine for any comparison.** Not because machine dependence is established
   -- it is not -- but because it costs nothing and removes the question.

## Determinism is case-dependent

Two identical CRM runs reproduce bit-identically on some cases and not on others. Three of
four active-domain cases had a noise floor of exactly zero; the fourth, a lateral command,
differed by 0.019 m in position and 0.0097 m/s in mean velocity between two runs with
identical inputs.

So "re-run it and see if you get the same answer" is not a valid check here -- it can
return a false confirmation. Any comparison of two configurations needs a **null arm**: a
repeat of the reference carried through the same analysis, so an effect is judged against
what identical inputs actually produce rather than against an assumption of zero.

## A fall is not data, on CRM

Only the feet and calves are FSI-coupled; the trunk has no interaction with the soil and
there is no rigid ground on CRM. A robot that pitches onto its belly therefore descends
through the bed and keeps going. `validity.py` gains a `sinking` check for this, because
the existing floor at -0.5 m let a robot sitting at -0.40 m pass every check in the file.

## A push test needs a disturbance the arms share, and metrics that do not flatter

`evaluate.py --push-force N` shoves the trunk once per episode (`--push-at`, `--push-dirs`,
`--push-reps`), in the BODY frame, so "pushed from the left" means the same thing however
the robot is facing, and every arm gets the identical force at the identical time on the
identical episode. The bed is widened by 1 m on both axes, equally, so a sideways push is
not scored on a narrower bed than a forward one.

Two metrics were wrong on their first outing and are worth stating as rules:

  DRIFT IS NOT EXCURSION. Displacement measured sideways of the push direction read ~1 m
  even for a push straight ahead, because over the 3 s window the robot walks 1.5 m and
  this policy yaws while walking. Excursion is now distance from where the COMMAND says
  the robot should be, and the same measurement is taken over the 3 s before the push, so
  the push's own cost is the difference.

  A PER-ARM THRESHOLD COMPARES ARMS AGAINST DIFFERENT BARS. Recovery was "back inside this
  episode's pre-push error band", and a policy that tracks better has a tighter band, so
  the better tracker was held to the stricter test and looked slower. Both are now
  recorded: the per-episode band and a fixed 0.30 m/s bar.

Falls are the headline and are not averaged: a pair where either arm went down has no
recovery time to compare, so paired_eval counts them separately. Gate 4 is reported but
not enforced in push mode, because leaving the corpus region is what the push is for.
