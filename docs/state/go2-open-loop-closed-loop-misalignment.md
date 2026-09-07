# Open-loop data, open-loop validation, closed-loop use

Three things measured separately on 2026-09-07, put together. Each was established on its
own and none of them is a mechanism that died; the alignment between them is the finding.

```
   the CORPUS       25% of the surrogate's mix is the excitation corpus, whose
                    recorded window is 0.4 s of OPEN-LOOP random joint targets.
                    No policy is closing a loop in any excitation row.

   the VALIDATION   open-loop prediction error, flat out to 1.0 s.

   the FINE-TUNE    consumes CLOSED-loop rollout, valid to roughly 10 steps.

   the DEPLOYMENT   a closed loop.
```

**The corpus, the metric and the consumer are misaligned in the same direction.**

## The excitation corpus is not failing. It is succeeding on the wrong axis.

From `decisions/w1-rigid-scope.md`, same three checkpoints on both validation splits:

```
   surrogate            walking     excitation
   base_matchw (0%)     0.008933      0.741947
   exc25 (25%)          0.006657      0.123441      <- 6x better
   exc50 (50%)          0.011860      0.128190
```

The corpus buys exactly what it was collected to buy: a 6x improvement in predicting the
excitation distribution. **That purchase is on the open-loop axis.** The consumer of the
surrogate is a closed-loop rollout, and the walking corpus -- which does contain closed-
loop structure -- is the part being diluted to make room.

This reframes a result that has been carried as favourable. "The excitation corpus buys
what it was collected to buy" is true and is not evidence that it helps the fine-tune,
because what it buys is measured by an instrument aligned with the corpus and not with
the use.

## Why the yaw hole could never have been closed by sampling

`go2-excitation-yaw-hole.md` measured it: commanding `wz` doubles the high-yaw fraction
and leaves the corpus 29x short, because a commanded yaw rate has to survive 0.4 s of
open-loop joint noise and mostly does not.

The same structural fact explains both. **The excitation window contains no closed-loop
content, so neither the states a driven policy visits nor the action-consequence
structure a closed-loop rollout needs can appear in it, at any sampling density.**

## What this does and does not license

It does not establish that the mix causes the fine-tune's failure. Nothing here is a
matched comparison between mixes on a closed-loop endpoint; `base36` versus `armB`
addresses that and its tracking half is bounded at ~1% of baseline error with the
direction unresolved.

It does say the three-way alignment should be stated before any further corpus work, and
that "does the excitation corpus help" cannot be answered by an open-loop validation
number no matter how large the improvement.
