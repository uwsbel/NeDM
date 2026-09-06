# Three W0 results recomputed on clean, paired data

Status: 2026-09-06. Supersedes the figures in
[w0-linear-baseline-contamination.md](w0-linear-baseline-contamination.md), which
established that W0's inputs carried diverged episodes. Two corrections are applied
here: `--max-abs-state 1e4` excludes diverged episodes, and the W4 arms are restricted
to the episodes they both admit.

## The arms were never paired, and asymmetrically so

```
  event      admits 255 of 400    85 diverged, 60 no detectable events
  fixed 0.10 admits 281           85 diverged, 34 no detectable events
  fixed 0.29 admits 281           same 281
  fixed 1.00 admits 281           same 281
```

The event arm's set is a strict **subset** of the fixed arm's: the intersection is 255,
exactly the event set. So the fixed arm was scored on 26 episodes the event detector
could not admit at all — by construction, the episodes where event detection fails.
Varying the split seed cannot address this, because the seed varies the split *within*
an arm.

The floor curve needs no such correction: all three horizons admit the identical 281
episodes, so its points were already on one population.

## 1. W4: the verdict does not survive

```
                          event    fixed 0.29    event wins
  as recorded  ridge      0.142       0.413         0/5      <- "fixed wins 5/5"
  clean+paired ridge      0.385       0.334         5/5
  clean+paired knn10      0.506       0.569         1/5
  clean+paired mlp        0.525       0.643         1/5
  clean+paired level      0.566       0.349         5/5
```

The recorded result — fixed-dt beating event indexing on every seed — was produced on
contaminated data with unpaired arms. On clean, paired data **the ridge comparison
reverses completely**, and the four metrics now disagree: ridge and level-ridge favour
event indexing 5/5, knn10 and the MLP favour fixed-dt 4/5.

**The correct conclusion is that W4 is INCONCLUSIVE, and it stays there.**

Two things make it inconclusive rather than decided the other way. First, the
pre-registration said "event beats fixed on >= 4 of 5 seeds" and never named the
estimator -- which was harmless while the estimators agreed and is load-bearing now.
A decision rule that does not name its statistic is not a pre-registration. Second,
**the MLP is a metric nobody had when W4 was decided**, because it was silently dead;
part of the disagreement is a measurement that did not exist. "The estimators
disagree" reads differently once one of them is new. A
comparison whose sign depends on the estimator is not a result, and reporting the two
metrics that now favour our preferred arm would repeat the original error with the
polarity flipped.

## 2. Body family at 0.29 s: the factor of five is a factor of 2.5 to 3.6

```
  published linear ridge            0.125     surrogate 0.723  ->  5.8x
  clean, body family (11 comps)     0.292                      ->  2.5x
  clean, body velocity only (6)     0.203                      ->  3.6x
```

Which component set the published 0.125 used is not recorded, so both are given. Either
way the surrogate's advantage on the body family is roughly half what was claimed.

## 3. Linear floor curve: higher, and no longer monotonic

```
  horizon    published ridge    clean ridge    knn10    mlp
    0.10 s        0.46             0.732       0.848   0.903
    0.29 s        0.39             0.356       0.470   0.477
    1.00 s        0.27             0.496       0.446   0.283
```

The bar a trained surrogate must clear is substantially higher at 0.10 s and 1.00 s.

**The curve is no longer monotonic in horizon, and the reason is the normaliser rather
than predictability.** Two tests, both free from data already in the summaries.

*Test 1, oscillation versus drift — refuted.* The hypothesis was that oscillatory
channels dip at 0.29 s while drift channels rise monotonically. Every group dips:

```
  group                             0.10    0.29    1.00
  joint velocity (12)              0.817   0.415   0.530     dip
  body angular rate (3)            0.610   0.422   0.540     dip
  relative position (3)  [drift]   0.491   0.262   0.448     dip
  joint position (12)              0.671   0.304   0.434     dip
  attitude (2)                     0.520   0.491   0.645     dip
  body linear velocity (3)         0.212   0.105   0.215     dip
```

The dip is universal, so it is not a property of channel type. Nor is it sample size,
which falls monotonically with horizon — 86,189 training pairs at 0.10 s, 29,639 at
0.29 s, 8,516 at 1.00 s — so the *least* data coincides with a *recovery* in score.

*Test 2, absolute error — this is the answer.* R^2 is normalised by the increment's own
spread, and that spread is not monotonic in horizon:

```
  horizon    sd(dx)    ridge R^2    residual RMSE
    0.10 s    0.306      0.732          0.157
    0.29 s    0.233      0.372          0.187
    1.00 s    0.282      0.490          0.203
```

**Residual RMSE grows monotonically with horizon, exactly as it must.** R^2 tracks
`sd(dx)`, which dips at 0.29 s and recovers at 1.00 s — consistent with gait
quasi-periodicity, where after roughly a gait cycle the oscillatory part of the state
has returned near its start and the net increment is small, while at three cycles
accumulated drift has made it large again. The earlier gait-phase argument failed
because it was applied to predictability; it belongs to the increment spread.

**Consequence for using this curve as a floor.** Comparing R^2 across horizons compares
against a moving normaliser: a surrogate at 1.00 s faces an apparently higher bar
(0.496) than at 0.29 s (0.356) not because prediction is easier but because the
increment spread is larger. Either quote the floor at a single horizon, or use residual
RMSE, which is monotonic and directly comparable.

## Note on cost

Fixing the MLP arm, which had been silently dead, made every W0 run substantially more
expensive. Ten runs with sklearn's default BLAS threading drove load to 137 on a
16-core box and none completed in fifty minutes. All runs here pin
`OMP_NUM_THREADS=1` and friends. Reviving dead code changes the cost profile of
everything downstream of it.
