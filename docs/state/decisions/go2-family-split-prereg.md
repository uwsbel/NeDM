# Pre-registration: does the arm-A family split replicate on fresh episodes?

Written 2026-09-07, **before the confirmatory episodes are collected**. The hypothesis
was found on 16 common-survivor episodes and tested on the 36 that contain them, so
every confirmation so far shares a sample with the discovery.

## The finding to be replicated

Arm A (`go2_finetuned_exc25`) at stable gain, full 36-episode kyle-sbel stratum, paired
difference `treated − baseline` in m/s, negative = treated better:

```
   k     ALL 36    straight (constant+vel_step)   turning (arc+weave+yaw_step)
 0.60   +0.00000   -0.03838 [-0.0593, -0.0104]    +0.03969 [+0.0196, +0.0710]
 0.65   +0.00006   -0.03846 [-0.0593, -0.0091]    +0.03950 [+0.0195, +0.0714]
 0.75   +0.00012   -0.03755 [-0.0592, -0.0104]    +0.03733 [+0.0138, +0.0696]
```

The aggregate null is two significant effects cancelling. `k` varies loop gain by 25%
and the split is invariant, so it is not an operating-point artefact — but all three
rows are the **same 36 baseline episodes**.

## Registered endpoints

**E1 — replication.** On fresh episodes, straight-line median paired difference is
negative with a 95% interval excluding zero, and turning is positive with an interval
excluding zero.

**E2 — powered claim.** The straight-line interval's upper bound is below **−0.020 m/s**,
the criterion this line was built against. Currently it is not: upper bounds are −0.0104,
−0.0091, −0.0104 at n=16. **The point estimate is roughly twice the criterion and the
effect is significantly non-zero; the interval does not exclude the criterion.** E2 asks
whether it does at adequate n.

**Sizing.** Half-width scales as `1/sqrt(n)`. At n=16 it is ~0.0245; excluding −0.020
from a point estimate of −0.038 needs < 0.018, so `n > 16 · (0.0245/0.018)² ≈ 30` per
stratum. Target **≥30 straight and ≥30 turning**, set by the collected condition mix
rather than left to the default 36 splitting 16/20.

## What would refute

- The split does not appear on fresh episodes → it was a property of those 36.
- Straight-line is significant but the interval still includes −0.020 at n≥30 → E1 holds,
  E2 fails, and the honest claim stays "significantly non-zero, about twice the criterion".
- The sign reverses in either stratum → the discovery was noise.

## Not registered, and not to be claimed from this run

Why the strata differ. The excitation corpus contains no turning — `family="constant"`,
`params={"vx": U(-0.8,0.8)}`, no `wz` key — and its rows are 101x poorer above 1.0 rad/s
of yaw than walking's (0.56% against 56.8%). That fits, and a fit is not a cause. The
discriminator is `base36` and `armB`, whose surrogates saw the same walking data without
excitation: if dilution is the mechanism their split should be weak or absent. Separate
run, separate registration.
