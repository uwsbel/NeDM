# k* does not discriminate the arms; the rate at nominal gain does

Third independent seed (303) of the discriminative-validity test, run on kyle-sbel with
`scripts/evaluation/gain_margin.py` at n=40 per rung, 8 rungs. Checkpoints verified by
md5 before use: v4 `3d856f9d...` (three local copies byte-identical), armA `ccc279a6...`
(matches its PROVENANCE file).

```
    k       v4          armA        difference
   0.50    0/40 0.00    0/40 0.00     +0.00
   0.70    2/40 0.05    0/40 0.00     -0.05
   0.85   16/40 0.40   20/40 0.50     +0.10
   0.90   24/40 0.60   26/40 0.65     +0.05
   0.95   14/40 0.35   40/40 1.00     +0.65
   1.00   23/40 0.57   40/40 1.00     +0.42
   1.20   37/40 0.93   40/40 1.00     +0.07
   1.50   40/40 1.00   40/40 1.00     +0.00

   k*      0.875        0.850         0.025
```

**The summary statistic separates the arms by 0.025. The underlying curves separate by
0.65 at k=0.95, Fisher exact p = 6.0e-11.** At nominal gain, 0.42 with p = 1.8e-06.

The rates at k=1.00 also reproduce the original observation that motivated this test:
v4 23/40 diverging against the recorded 23/43, armA 40/40 against 43/43.

## Why k* compresses the difference

`k*` is where the divergence rate crosses 0.5. Both arms cross in the same narrow band
because both are near 0.5 around k=0.85-0.90. **What differs is everything above the
crossing**, where armA saturates at 1.00 and v4 does not -- and a crossing point cannot
see that, because it is defined by one level set of a curve whose whole shape is the
signal.

**Worse, v4's curve is not monotone**: 0.40 at k=0.85, 0.60 at 0.90, 0.35 at 0.95, 0.57
at 1.00. A crossing interpolated through a non-monotone curve is not well defined -- a
different rung spacing would move it, and which of several crossings gets found depends
on where the samples happen to fall. `k*` is a sound summary only for a monotone
response, and v4's is not.

## What to use instead

Report the **divergence rate at nominal gain**, which is the quantity the deployment
cares about and where the arms differ by 0.42 to 0.65 with p ranging from 1e-6 to 6e-11.
`k*` remains meaningful as a statement about a single arm -- "unstable as deployed,
k* < 1" is true and useful for both -- but it should not be used to compare arms.

This is the same failure as reporting a median where the distributions differ in shape:
the summary is not wrong, it is answering a narrower question than the one being asked,
and its narrowness is invisible in the number.

## Note on the earlier power analysis

The coordinating session killed a six-sweep plan at n=8 because the binomial sd was 71%
of the rate span being interpolated across. That was correct and this run does not
contradict it. At n=40 the rungs are precise enough to show that **the imprecision was
not the only problem**: even measured well, k* is the wrong statistic for this
comparison.
