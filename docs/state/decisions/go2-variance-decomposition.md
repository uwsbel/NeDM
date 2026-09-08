# Where the noise lives: the surrogate seed, not the fine-tune

**2026-09-08.** Arm A spans 29.9% to 57.8% across three surrogate seeds at matched
epoch 80. That spread has two possible homes and they demand opposite responses:

- **the surrogate seed** -> replicate surrogate training, which is the current plan
- **the fine-tune being chaotic** -> more surrogate seeds buys nothing, and the policy
  optimisation step is itself the noise source

One surrogate held fixed (`A_s1` `last.pt`, md5 `868eff68`), four fine-tune seeds,
everything else identical.

## Result

```
  varying                            n   mean    sd    range   values
  fine-tune seed, surrogate FIXED    4   53.4   3.71    8.4    57.8 54.9 49.4 51.5
  surrogate seed, ft seed FIXED      3   42.9  14.05   27.9    57.8 40.9 29.9
```

The surrogate-seed figure contains the fine-tune's own noise, so subtract it:

```
  fine-tune variance                    13.74   (sd  3.7 points)
  surrogate-seed total variance        197.50   (sd 14.1 points)
  surrogate contribution, net           183.76   (sd 13.6 points)
  variance ratio, surrogate : fine-tune  13.4x
```

**The surrogate seed carries roughly 13x the variance of the fine-tune.** Replicating
surrogate training is the right response, and the four-seeds-per-arm plan is aimed at
the right target.

## The part that constrains what can ever be claimed

**Even with the surrogate held completely fixed, the fine-tune alone spreads results
over 8.4 points** (49.4 to 57.8). All four runs used the same surrogate, the same base
policy, the same episodes, the same budget, and stopped within 279-285 updates at
`||dW||` 4.002-4.007 -- the trajectories are as matched as this method can make them,
and the policies still differ by 8 points in Chrono.

So a single (surrogate, fine-tune) pair carries about 8 points of irreducible noise
before any arm difference is even considered. **Any arm effect smaller than that is
unmeasurable without replicating both levels**, and every single-run number this
project has ever reported -- including all of today's corrected ones -- sits inside a
band that wide.

## Caveats, stated rather than buried

- **n=4 and n=3.** These standard deviations are themselves imprecise; the variance
  ratio is an order-of-magnitude statement, not a measurement to two figures.
- **The three fine-tune-seed policies were scored on three different boxes** (sbel,
  north, a3), so some of the 8.4 points is cross-machine. The measured machine effect
  is 0.2 points on the rate, so it accounts for a negligible share -- but it is not
  zero and the boxes run different pychrono builds.
- **One surrogate, one arm.** Whether the fine-tune noise is this size for B, C and D
  is untested.

## What follows

Four surrogate seeds per arm is necessary and, on these numbers, still not obviously
sufficient: with a between-surrogate sd near 13.6, four seeds gives a standard error on
the arm mean of about 6.8 points. **An arm difference would have to be roughly 20 points
to clear that**, and the corrected B-A contrast is currently +1.5.

That is worth stating plainly: **the design as it stands can only detect very large arm
effects.** Reporting a null from it would be a statement about the design's resolution,
not about the arms.
