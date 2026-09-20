# How far the NN-ROM can be rolled before it stops carrying information

**Status:** measured 2026-09-20, job 64723. Artifact `out/horizon_sweep.json`, regenerate
with `scripts/evaluation/horizon_sweep.py sel_baseline valw256e wcov_roll --episodes 32
--seed 0`.

## Why this had to be run

The question had been answered in working notes with "3.1% error at 0.3 s, usable to
1-1.5 s". No measurement behind that existed anywhere and it was withdrawn (`059b4302`)
after it had already reached a slide. Training logs rollout at 5 s and 10 s only, so
there was no grid to read an answer off. This produces one.

It calls the same `evaluate_rollouts()` the trainer uses, so the values are comparable
with the `rollout_crm_5.0s` and `10.0s` numbers already on record, and it pins
`rollout_eval.num_episodes` and the sampling seed, because that count is 12 on older
configs and 32 on newer ones and a median over 12 draws is not a median over 32. All
three arms below are rescored at 32.

## The measurement

`errdist` is planar position error divided by distance travelled, so a model predicting
NO MOTION AT ALL scores exactly 1.0 by construction. That is an analytic floor, not a
fitted baseline, and where a curve crosses it is where the model has stopped saying
anything useful about where the robot goes.

```
  arm            0.1s   0.2s   0.3s   0.5s   1.0s   2.0s   3.0s   5.0s  10.0s
  sel_baseline  0.014  0.019  0.034  0.067  0.150  0.646  1.057  1.535  2.322
  valw256e      0.013  0.014  0.032  0.073  0.164  0.370  0.853  1.887  3.036
  wcov_roll     0.043  0.099  0.166  0.241  1.536  4.457  4.278  3.695  2.458
```

## What it says

**At the branch length the fine-tune actually uses, 0.30 s, the deployed model is at
0.034** -- 3.4% of distance travelled. The withdrawn "3.1%" was close to right and is now
superseded by a figure with an artifact behind it.

**The model stays under the no-motion floor to about 3 s.** `sel_baseline` crosses 1.0
between 2 s and 3 s; `valw256e` between 3 s and 5 s. So the trustworthy horizon is
roughly an order of magnitude longer than the 0.30 s branch currently used, and the
branch is conservative rather than pushing the limit.

That matters for the stated goal of fine-tuning richer behaviour. Nothing here says a
longer branch would DELIVER a better policy -- the recorded finding is the opposite, that
horizon helped monotonically only up to 0.30 s -- but the model's fidelity is not what
stops it. Whatever limits branch length, it is not that the model has gone wrong by 0.5 s.

## An independent check on the coverage result

`wcov_roll`, trained on the disturbed corpus, is worse at EVERY horizon and collapses far
earlier: it crosses the no-motion floor between 0.5 s and 1.0 s, where the clean-corpus
arms hold to 3 s. At the 0.30 s branch it is 0.166 against 0.034, nearly five times the
error.

The coverage corpus was already known to deliver a worse policy (-30.2% against -40.9%
at matched selection). This says the model itself is also materially worse as a
multi-step predictor, measured in a way that has nothing to do with the fine-tune or the
Chrono verdict. Two independent measurements now point the same way, which is stronger
than either alone.

## Not established

Three arms, one seed, one corpus family. The crossing point is bracketed between grid
points rather than located, and nothing here connects horizon to delivered policy
quality, which would need branch-length arms fine-tuned and scored.
