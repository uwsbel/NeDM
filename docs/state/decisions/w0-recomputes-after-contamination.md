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

**The correct conclusion is that W4 is undecided, not that event indexing wins.** A
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

**The curve is no longer monotonic in horizon and I cannot explain that.** Predicting
further ahead should be harder, and 1.00 s scores above 0.29 s on ridge and knn10 while
the MLP still falls. The mean inter-event interval is 0.350 s, so 0.29 s is 0.83 gait
periods and 1.00 s is 2.86 — the fractional parts are nearly identical, so a simple
gait-phase argument does not account for it. Flagged rather than explained; a
non-monotonic predictability curve is either a real feature of the dynamics at these
horizons or an artefact of the fixed-dt construction, and I have not distinguished them.

## Note on cost

Fixing the MLP arm, which had been silently dead, made every W0 run substantially more
expensive. Ten runs with sklearn's default BLAS threading drove load to 137 on a
16-core box and none completed in fifty minutes. All runs here pin
`OMP_NUM_THREADS=1` and friends. Reviving dead code changes the cost profile of
everything downstream of it.
