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
