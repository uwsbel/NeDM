# Forward velocity has a deadband, and the verdict cell sits inside it

Measured 2026-09-07 by `scripts/collection/drive_go2_realisation_pilot.py`, two arms
sharing seeds and perturbation draws, `vx` and `wz` swept separately across the policy's
trained range. Analysis in `scripts/analysis/go2_command_realisation.py`.

## The curve

Realised = mean over the last 10 s; ratio = realised / commanded. Control arm.

```
   vx  cmd    realised   ratio        wz  cmd    realised   ratio
     -0.50    -0.3126    0.63           -1.00    -0.9155    0.92
     -0.40    -0.2066    0.52           -0.80    -0.7252    0.91
     -0.30    -0.1347    0.45           -0.60    -0.5195    0.87
     -0.20    -0.0690    0.34           -0.40    -0.3310    0.83
     -0.10    -0.0054    0.05           -0.20    -0.1438    0.72
     +0.10    +0.0037    0.04           +0.20    +0.1453    0.73
     +0.20    +0.0428    0.21           +0.40    +0.3225    0.81
     +0.30    +0.0146    0.05           +0.60    +0.5070    0.84
     +0.40    +0.1634    0.41           +0.80    +0.6950    0.87
     +0.50    +0.3389    0.68           +1.00    +0.8748    0.87
```

**Yaw tracks. Forward velocity does not.** `wz` holds 0.72-0.92 across its whole trained
range with no deadband. `vx` realises 4-5% of command below 0.1 m/s and only reaches 0.63
at the edge of the trained range. The response is convex, not saturating: the ceiling is
not a limit that has been hit, it is the point where tracking finally starts working.

## Why it matters more than it looks

The verdict cell is `|cmd| in [0.02, 0.18]`. **That interval lies entirely inside the
deadband.** Measured directly on the two confirmatory corpora, not inferred from the
pilot:

```
   go2_fresh_cell3   n=300   median |cmd| 0.106   median |realised| 0.015   ratio 0.14
   go2_fresh_cell4   n=213   median |cmd| 0.091   median |realised| 0.027   ratio 0.30
```

So the baseline's median |velocity error| of 0.0935 m/s in cell3 is **very nearly the
command itself**: the error is dominated by the robot not moving, not by how well it
tracks. A criterion asking for a 0.020 m/s improvement is asking for a 22% reduction in
an error that is mostly deadband.

This does not invalidate the paired comparisons -- both arms face the same deadband on
the same episodes. It does mean the quantity being compared is largely "how much does
the deadband move" rather than "how well is the command tracked", and any mechanism
story told about tracking has to survive that.

**It also gives the family split a candidate mechanism.** Straight-line families command
only `vx`, which is barely realised; turning families command `wz`, which is realised at
0.8-0.9. The two strata differ in whether the robot does what it is told at all. That is
now the first thing to test, and it is testable: it predicts the split should track the
`wz` content of a family rather than its label.

Note cell4's higher ratio (0.30 vs 0.14). Perturbations push the trunk, and pushed motion
enters `|realised|` without being tracking. Disturbance partially fills the deadband,
which is a caveat for cell4's verdict, not a sign of better control.

## Perturbation effect: nothing detectable

```
   vx   slope  treated 0.192 [0.023, 0.384]   control 0.233 [0.045, 0.386]   d -0.041
   wz   slope  treated 0.804 [0.737, 0.854]   control 0.798 [0.723, 0.841]   d +0.006
```

Both differences sit well inside their intervals. At 60 N with these n, no effect of
perturbation on realisation is demonstrated.

## Limitation: the control arm has no independent replication

Its three "reps" per level came out **bit-identical**. With perturbation scaled to zero
and spawn, heading, tilt and prewalk all fixed, a control episode consumes no randomness,
so a different `--seed` produces the identical episode. The reps replicate nothing.

Intervals above are therefore clustered by command level (11 distinct, 5 in the slope
window), not by episode. Before this: resampling episodes gave `vx` slope [0.157, 0.273];
clustered it is [0.023, 0.384]. **The apparent precision was entirely duplication.**

The curve is reproducible, but "would a different initial condition give the same curve"
is untested. Varying spawn and heading per rep would answer it, and the forward/backward
asymmetry -- at |cmd| 0.30, backward realises 0.135 and forward 0.015, a 9x difference --
should not be claimed until it is. The non-monotone point at +0.30 (below the +0.20
value) is deterministic rather than noise, and is unexplained.
