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

Pending: the same control in cell5, the tracking-capable cell, where the deadband no
longer suppresses headroom. Until that lands the tracking null is established in one
cell, not two.

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

**Fine-tuning does not change how well the policy tracks. It changes whether the policy
stays stable.**

That accounts for the shape of everything measured: every tracking comparison has been
marginal, sign-unstable and sensitive to protocol, while every stability measurement has
been large, reproducible across machines, and predictable from the weights. The two were
being reported as one line of evidence when only one of them had any.

## The order this was found in

The tracking effect was not shown to be small. It was shown to be **unattributable** —
present in the numbers, absent once the arms were matched. The distinction matters for
how the null should be written up: not "the fine-tune produces a small improvement we
lacked power to certify" but "the difference measured was between two protocols, and the
fine-tune's own contribution is +0.00003".
