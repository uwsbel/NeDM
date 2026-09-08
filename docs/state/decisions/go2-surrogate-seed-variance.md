# The surrogate training seed swings the result further than the arm does

**2026-09-08.** Six fine-tunes (arms A, B, C at two surrogate-training seeds each),
each scored in Chrono on the same 536 held-out episodes, at two displacement stages.

## Result

```
            dw4                          dw6
  arm    s1        s2      spread     s1        s2      spread
  A    54.1%     49.6%      4.5     23.1%     69.0%     45.9
  B    52.4%     17.4%     35.0     61.8%     34.5%     27.3
  C    33.2%     56.0%     22.8     41.0%     34.9%      6.1

  n = 536 episodes per cell. "spread" is the same arm, two seeds.
```

**The largest within-arm swing is 45.9 points.** Arm A at dw6 completes 23.1% on
seed 1 and 69.0% on seed 2. The only thing that differs between those two runs is
`training.seed` — 2026090701 against 2026090702. Same data, same architecture, same
objective, same fine-tune budget, same `--seed 0`, same `--target-dw`, same episodes.

**The largest between-arm gap at any single seed is 38.7 points** (dw6 s1, B over A).
**The seed moves the number further than the arm does**, so no arm ordering measured
here is identifiable.

## The sign reverses, and reverses with p < 0.0001

`B − A`, from the banded McNemar tables:

```
  dw6 seed 1    B beats A by +38.7 points, p < 0.0001 across five of six bands
  dw6 seed 2    A beats B by -34.5 points, p < 0.0001 across five of six bands
```

Both are correct arithmetic on 536 paired episodes. Both are wrong as statements about
the arms. **McNemar conditions on the pair of policies as fixed and asks whether the
episodes discriminate them.** The episodes do — emphatically. But the policy is a
deterministic function of a surrogate that is itself a random draw, and **the analysis
treats one draw as the population.** The episode is not the experimental unit. The
surrogate training run is, and n = 2.

Astronomically small p-values pointing in opposite directions are not a contradiction
to be resolved by picking a seed. **They are the signature of the wrong unit of
analysis**, and the tiny `floor` column in those tables — the smallest p the discordance
count could reach — was measuring how many episodes there are, not how much evidence.

## Two seeds per arm could never have worked, whatever the data said

With the seed as the unit and an exact permutation test over seed labels, the
smallest two-sided p reachable with n seeds per arm is `2 / C(2n, n)`:

```
  seeds per arm    smallest possible two-sided p
      2                0.3333
      3                0.1000
      4                0.0286   <- first count that can reach 0.05
      5                0.0079
      6                0.0022
```

**At two seeds per arm, no result can fall below p = 0.33.** Not with a larger
effect, not with more episodes, not with a better surrogate. Every A/B/C
comparison to date sat at that count, so at the level that matters it carried no
power at all, and the `p < 0.0001` it reported came entirely from counting 536
episodes as 536 independent replicates of an arm.

This also sets the requirement, rather than leaving it to taste: **four seeds per
arm is the floor**, and it is worth noting that the cost of an arm is now four
surrogate trainings, not one.

## It is seed sensitivity, not run-to-run noise

Checked before drawing any conclusion, because the two have the same signature and
opposite remedies. Two same-config repeats on north:

```
  A_s1_north_r1 vs A_s1_north_r2
    49 of 49 weight tensors identical,  max abs diff  0.000e+00
    identical val_loss to every printed digit (5.682633e-04)
    differ only in config.output_dir
```

**Training is deterministic on a fixed box.** The two checkpoints have different file
md5s *only* because the run directory name is stored inside them — and I briefly took
that md5 difference as evidence of nondeterminism before comparing the tensors. It is
not. So the variation above is entirely attributable to the seed, and the whole
downstream pipeline (fine-tune at `--seed 0`, export, Chrono replay) adds none of its own.

## This corroborates the gate reversal rather than being a second surprise

The open-loop action-sensitivity gate reversed the same way on the same seeds: B_s1
looked good and B_s2 scored 4.374, as bad as baseline, while C_s2 — the permuted control
whose channels carry no tilt information at all — scored 1.600. That was retracted at the
time as a single-seed finding. **Two independent measurements, one open-loop and one
closed-loop through Chrono, both reverse with the surrogate seed.** The gate result was
not a fluke of the gate.

## What this does and does not establish

**Establishes:** with two seeds per arm, the gravity-channel arms A, B and C cannot be
ordered by this evaluation, and no previously reported ordering between them survives.
The design flaw is in the unit of analysis, not in the harness, the corpus, the pairing,
or the McNemar computation — all of which check out (536 of 536 distinct keys at dw4 and
dw6, and the pre-`episode_id` gw_ files pair correctly on their composite fallback).

**Does not establish:** that the arms are equivalent. This is a power failure, not a null
result. A real arm effect smaller than ~45 points would be invisible here either way.

**Does not establish:** the size of the seed effect. Two draws bound it from below and
say nothing about its shape.

## What follows

Seeds are now the measurement, not a robustness check bolted on afterwards. Arm A is
being taken to six seeds (s3–s6 launched; `training.seed` 2026090703–06) to get the null
distribution of the between-seed difference. **Every arm comparison is then made against
that null, not against zero.** Arm D was already specified at four seeds for this reason;
that instinct was right and the reason is now measured rather than assumed.

**Scope:** rigid terrain, val-split episodes, dw4 and dw6 displacement stages, arms A/B/C.
Determinism verified on north only, two runs.
