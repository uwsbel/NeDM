# What fine-tuning changes, and what it does not

Consolidated 2026-09-07, after the gain confound was removed from the tracking
comparisons. The project's measurements fall into two families and only one of them
survives contact with a matched-gain control.

## Tracking error: null

Every tracking comparison ran the treated arm at reduced gain against a baseline at
nominal. With both arms at the same gain, in cell4 at n=195:

```
   family split           +0.00003    interval width 0.003 around zero
   straight / turning     -0.00003 / -0.00001, both intervals spanning zero
   E2, the -0.020 line    decisively unmet; the effect is not 0.038 but 0.005,
                          and at matched gain not distinguishable from 0
   E3, family vs motion   permutation p 0.856, nothing to discriminate
```

cell5, the tracking-capable cell, at n=221:

```
   matched gain, ALL      +0.00154   [+0.00027, +0.00418]   +0.66% of baseline error
   straight               +0.00152   [+0.00009, +0.00343]
   turning                +0.00199   [+0.00004, +0.00736]
   split                  +0.00047
```

**The cell5 intervals exclude zero, and the exclusion survives multiplicity.** Nine
primary intervals have been looked at across this line (3 cells x straight/turning/
aggregate), so a single marginal exclusion on the third cell is the obvious objection.
Tested rather than argued:

```
   unadjusted        alpha 0.05      [+0.00027, +0.00418]   excludes 0
   Bonferroni /9     alpha 0.00556   [+0.00008, +0.00656]   excludes 0
   Bonferroni /12    alpha 0.00417   [+0.00007, +0.00689]   excludes 0
   sign test         136/221 positive, two-sided p = 7.3e-04
```

The sign test is the cleaner evidence: it does not depend on the interval construction
and its p clears even a /12 correction by a factor of six.

**So the sign is established and the magnitude is not.** The adjusted interval runs from
+0.00008 to +0.00689 -- from 0.03% to 3% of baseline error, a factor of 80. The fine-tune
is reliably worse at tracking-capable speed; how much worse is poorly determined and
small on any reading.

```
              ratio   baseline err   matched-gain effect        as % of error
   cell4      0.33      0.0886 m/s   -0.00001 [-0.0007,+0.0009]      -0.01%
   cell5      0.56      0.2335 m/s   +0.00154 [+0.0003,+0.0042]      +0.66%
```

Still no family split anywhere: +0.00003 and +0.00047.

## Stability: real, large, replicated

None of these used the asymmetric protocol.

```
   armA at nominal gain           0 of 36 survive; 36/36 diverge
   k* (uniform-gain margin)       armA 0.921, armB 0.925, base 1.450-1.473
   growth constant                1.40066 per step in simulation, against
                                  rho(J) = 1.4007 from the weights alone --
                                  four decimals, 96 episodes
   handover fragility             85/85 across three collections
```

Cross-machine, reported by the coordinating session from a3 and sliger, each with its
own baseline corpus and its own physics build, at nominal gain:

```
   base36  0 of 228        base 228 of 228
   armB    1 of 229        base 229 of 229
```

## The statement

**On every axis where anything is measurable at all, the fine-tune is worse --
negligibly on tracking, enormously on stability.**

```
   one-step accuracy (walking)   worse, monotone in dose, tight intervals
   tracking, matched gain        worse or nil, two orders below the criterion
   stability and recovery        worse, large, reproducible across machines
```

This is a better headline than "null on tracking, effect on stability" because it removes
the apparent tension: the two results point the same way and differ only in magnitude. It
also holds on the point estimates alone, so it does not rest on cell5's marginal
exclusion.

That accounts for the shape of everything measured: every tracking comparison has been
marginal, sign-unstable and sensitive to protocol, while every stability measurement has
been large, reproducible across machines, and predictable from the weights. The two were
being reported as one line of evidence when only one of them had any.

## A withdrawn quantification

An earlier version of this reported a "headroom" per cell -- baseline error split into a
deadband component `(1 - ratio) * |cmd|` and a residual that a tracking effect could act
on -- and gave cell4 66% deadband with 0.030 m/s of headroom.

**That decomposition is withdrawn.** It is circular: when under-realisation is the only
error source, `(1 - ratio) * |cmd|` *is* the error, so the residual is zero by
construction. cell5 made it visible by returning a deadband of 112% of the error and a
negative headroom. And it is backwards on its own terms -- failing to realise the command
is not an irreducible floor, it is exactly the deficiency a better policy would fix, so
subtracting it removes from the denominator the thing the treatment is meant to act on.

The defensible denominator is the whole baseline error, which is what the table above
uses. It is the more demanding comparison, not the more forgiving one.

## The order this was found in

The tracking effect was not shown to be small. It was shown to be **unattributable** —
present in the numbers, absent once the arms were matched. The distinction matters for
how the null should be written up: not "the fine-tune produces a small improvement we
lacked power to certify" but "the difference measured was between two protocols, and the
fine-tune's own contribution is +0.00003".
