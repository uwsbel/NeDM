# The W0 linear baseline was measured on contaminated data and is ~2x too low

Status: measured 2026-09-06. Bears on every comparison that uses the linear
baseline as a reference class for the surrogate's advantage.

## Identifying the dataset without the recorded command

W0's invocation glob was never recorded, so which dataset produced the published
figures could not be read off any artefact. It was recovered from a fingerprint
instead: the summaries record `n_episodes` and `n_dropped`, and W0's drop rule is
deterministic. Replaying that rule over the first 400 episodes of each candidate:

```
  go2_joint_off3000000   400 used   112 dropped   <- matches `rigid` and `rigid-banded`
  go2_stratified, go2_merged, go2_discrete        no joint-position columns at all
```

Only one candidate can even be scored, and it reproduces the drop count exactly.
Confirmed conclusively below, where re-running W0 on it reproduces the published
medians to three decimals.

## The measurement

`go2_joint_off3000000` carries diverged rows: 0.67% of rows, but **21% of the 400
episodes W0 used** contain at least one. Re-running W0 on exactly those 400, then on
the 316 that are divergence-free:

```
                              ridge     knn10    ridge(level)
  as published (400 eps)      0.182     0.282       0.360
  divergence-free (316 eps)   0.378     0.486       0.566
```

The `as published` row reproduces the recorded 0.182 / 0.282 exactly, which is what
identifies the dataset beyond doubt.

**The clean run uses LESS data and scores far better** — 256 usable episodes against
288. An improvement that survives a 11% reduction in training episodes is not a volume
artefact; contamination was suppressing the baseline.

The mechanism is visible in the per-component spreads: `sd(dx)` for
`joint_rr_hip_vel_radps` falls from 7.23 to 2.63 once diverged episodes are removed.
R^2 is not robust, and a handful of rows carrying 1e34 sets the residual for the fit.

## Why W0's own filter did not protect it

W0 applies no magnitude filter. It drops an episode only when no stance events are
detectable, which usually happens under divergence because divergence destroys foot
contact — protection as an incidental side effect of an unrelated rule. It removes
most diverged episodes and not all: of eight tested, six dropped and two kept.

## Consequence

Any statement of the form "the surrogate beats linear by N" that uses 0.182 or 0.282
as the linear reference is comparing against a baseline roughly half its true value,
and overstates the surrogate's advantage by about the same factor. The comparisons
need recomputing against 0.378 / 0.486, or against a fresh W0 run with a magnitude
filter applied.

This does not touch the excitation collection: every excitation dataset is 0.000%
diverged by the census, because that collector rejects diverged episodes.

## Unrelated bug found while running this

W0's MLP arm fails with `name 'sy' is not defined` and is silently skipped — the run
prints `(MLP skipped: ...)` and continues, so any MLP column in a W0 table predates
that regression or was never populated.
