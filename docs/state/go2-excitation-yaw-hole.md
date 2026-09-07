# The yaw hole was never a sampling problem

The excitation corpus reaches |yaw| > 1 rad/s in 0.56% of rows against walking's 56.8%,
a 101x gap, and it has been treated as a coverage shortfall that better sampling could
close.

It cannot be. `collect_go2_excitation.py` branched on `family="constant"` with a `vx` key
only, so the command vector was `(vx, 0, 0)` in every episode ever collected. **`wz` was
commanded at exactly zero.** Fixing `set_time()` made the `vx` draw take effect and left
this untouched, so the corpus still could not reach turning states.

Verified rather than read off the source — three episodes each way:

```
   without --command-envelope   (0.2744, 0, 0)  (0.3261, 0, 0)  (-0.2746, 0, 0)
                                |wz| max 0.0000
   with --command-envelope      (0.1715, 0, -0.5252)  (0.3221, 0, +0.8006)
                                |wz| max 0.8006
```

`--command-envelope` (91d3546) branches on `arc(vx, wz)` across the policy's trained
ranges. Off by default, since it changes the command distribution of any corpus
collected with it.

## What this does to the coverage audit

The audit asks "does commanding the envelope close the holes". Until now the envelope
was not being commanded on one of its axes, so the audit could only ever have confirmed
a shortfall it was structurally guaranteed to find.

It must also judge **reached** states rather than commanded ones. Commanding `vx` does
not produce proportional `vx` — see `go2-command-realisation-deadband.md`, where the
realised fraction is 4-5% below 0.1 m/s — so a commanded-coverage audit would report
success on a corpus that stayed narrow. `corpus_coverage.py` already reads reached state
arrays, so it is the right instrument; the input is what was wrong.

Yaw is the channel where this matters least and most at once: `wz` is realised at
0.72-0.92 across its whole range, so unlike `vx` it will actually arrive once commanded.
The prediction is that the yaw hole closes and the forward-velocity holes do not.

## Registered prediction, before the audit corpus is collected

Written 2026-09-07 before any `--command-envelope` corpus exists beyond the three-episode
smoke test.

```
   yaw channels           HOLE CLOSES.    wz is realised at 0.72-0.92 across its whole
                          trained range, so once commanded it actually arrives.
   forward-velocity       HOLES PERSIST.  vx is realised at 4-5% below 0.1 m/s and 0.63
                          at the edge of the trained range, so commanding it does not
                          produce it.
```

**If both close, the deadband measurement is wrong somewhere** and the realisation pilot
needs re-examining before its conclusions are used further. If neither closes, commanding
the envelope is not the mechanism by which coverage is gained and the corpus design needs
a different intervention.

Recorded because the audit was previously a confirmation -- `wz` was never commanded, so
a yaw hole was guaranteed regardless of sampling -- and a question that can only return
one answer should not be run again without saying in advance what the other answers would
mean.

## Result: the prediction was wrong. Neither hole closes.

Two matched corpora, 300 windows x 12 workers each, 144,012 rows apiece, 100% kept.
Reached-state coverage against the walking corpus (`scripts/analysis/go2_reached_coverage.py`):

```
   channel                envelope inside/covers/tail    baseline inside/covers/tail
   vel_body_x_mps           1.000  0.580  0.0000*         1.000  0.680  0.0000*
   yaw_rate_radps           1.000  0.860  0.0000*         1.000  0.760  0.0000*
   ang_vel_body_z_radps     1.000  0.860  0.0000*         1.000  0.760  0.0000*
```

Yaw rate directly:

```
   corpus     n         p50 |yaw|   p90     p99     % > 1 rad/s
   walking    399,986      1.2469  2.4166  2.8613     60.20%
   envelope   144,000      0.2878  0.7061  1.1181      2.08%
   baseline   144,000      0.2283  0.6192  1.0187      1.13%
```

**Commanding `wz` moves yaw content and does not close the hole.** The high-yaw fraction
roughly doubles, 1.13% to 2.08%, and p90 rises 14% -- a real effect, in the predicted
direction, about 29x short of walking's 60.2%.

So the registered third branch applies: **commanding the envelope is not the mechanism by
which this coverage is gained.**

## Why, and what it means for the corpus design

The excitation *window* is not policy-driven locomotion. It is a 0.4 s burst of random
joint targets (`stand + ACTION_SCALE * U(-1,1)^12`) applied after a pre-roll. The command
only influences the **branch phase** that sets the initial state; during the recorded rows
the policy is not tracking anything. A yaw rate commanded at 1.0 rad/s therefore has 0.4 s
of open-loop joint noise in which to survive, and mostly it does not.

**The hole is a property of the excitation protocol, not of the command distribution.**
That is why every previous attempt to close it by sampling harder failed, and it predicts
that `--command-envelope` will not fix it either at any sampling density.

The forward-velocity hole persists exactly as predicted, for the additional reason that
`vx` is realised at 4-5% in the low band, so it was never going to arrive.

`--command-envelope` remains correct and worth keeping -- `wz` at exactly zero was a real
defect and the flag measurably widens the distribution -- but it is not the intervention
this corpus needs. Closing the yaw hole requires recording windows in which the policy is
*driving*, not windows of open-loop excitation seeded from a driven state.

## A configuration error worth recording

The first two attempts at this audit rejected 99.5% of windows on `joint_limit` and I
nearly reported that as the envelope pushing the robot into its limits -- a clean and
plausible finding. The control arm rejected at the same rate.

The cause was neither: I ran the collector at its module defaults, `WINDOW_ROWS = 170`,
while every corpus in `datasets/` was collected at `window_rows 40` with
`action_scale 0.3` and kept 900 of 900. A window had to survive 1.7 s of open-loop noise
instead of 0.4 s. Matching the recorded config gave 100% kept on both arms.

Same lesson as the anchor's RL config: **the module defaults are not what produced the
artifacts, and the run's own recorded config is the authority.**
