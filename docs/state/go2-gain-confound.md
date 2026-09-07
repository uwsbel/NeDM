# The family split is not a property of the fine-tune

Measured 2026-09-07 on cell4, n=195 paired episodes.
`scripts/analysis/go2_gain_decomposition.py`.

## The confound

The verdict harness runs the **treated** arm at `--action-mult k` and the **baseline**
at nominal gain — `arm_env` pops `NEDM_ACTION_MULT` for every other arm, so baselines are
always recorded at 1.0. Every paired difference it has ever reported is therefore
*fine-tune and gain*, not fine-tune alone. The multiplier exists for a real reason: arm A
diverges at nominal gain in 36 of 36. But "treatment at reduced gain against control at
full gain" is a two-variable comparison.

Running the **base** checkpoint as the treated arm at the same k supplies the third leg.
Per episode the three are exactly additive (residual 1.4e-17).

## The decomposition

```
                          straight n=74        turning n=121        split
   as reported   armA@0.75 - base@1.0
                          -0.00500             +0.00354             +0.00854
   multiplier    base@0.75 - base@1.0
                          +0.00012             +0.00435             +0.00423
   matched gain  armA@0.75 - base@0.75
                          -0.00003             -0.00001             +0.00003
```

Matched-gain 95% intervals: straight [-0.00163, +0.00126], turning [-0.00112, +0.00143],
aggregate [-0.00068, +0.00089].

**At matched gain the fine-tune does nothing, in either stratum, and produces no family
split.** The effect that has driven this thread is not a property of arm A.

The multiplier alone accounts for about half the observed split and its own family
structure is marginal (permutation p = 0.070). Medians of correlated sums do not
decompose additively (the two legs correlate at -0.24 by construction), so the observed
+0.00854 is not simply the multiplier's +0.00423 plus zero. What the matched-gain leg
establishes is the negative: **whatever produces the split, it is not the fine-tuning**,
because the fine-tuning compared like-for-like produces +0.00003 with an interval of
width 0.003 around zero.

## cell3: the multiplier accounts for the entire effect

The zero-disturbance corpus was the last result standing on the asymmetric protocol. I
had reported its aggregate +0.0097 with all five families positive and every interval
excluding zero, and described the uniformity as suggestive.

```
   as reported   armA@0.75 - base@1.0    +0.00971  [+0.00591, +0.01389]   10.4% of err
   multiplier    base@0.75 - base@1.0    +0.00969  [+0.00590, +0.01321]   10.4% of err
   matched gain  armA@0.75 - base@0.75   +0.00001  [+0.00000, +0.00001]    0.01% of err
```

**The multiplier accounts for +0.00969 of the +0.00971.** At matched gain the fine-tune's
effect is +0.00001, and the family split is +0.00000.

**The predicted mechanism was wrong.** Before running it I argued that if the confound
explained cell3 it would have to act through a family interaction rather than a common
shift, since the multiplier had moved cell4's aggregate by only +0.00075. It acted as a
near-exact common shift instead -- straight +0.00840 against the reported +0.00837,
turning +0.01141 against +0.01213 -- and it was an order of magnitude larger here than in
cell4. The disturbance-free corpus is where the multiplier does the most damage, which
neither of us predicted and which nothing measured beforehand implied.

The matched-gain interval is implausibly tight because a zero-disturbance episode is
deterministic; the two policies differ by a nearly constant amount. It excludes zero and
is 0.01% of baseline error, which is a real difference of no consequence.

## All three cells

```
              disturbance   as reported   multiplier   matched gain   % of err
   cell3      none            +0.00971     +0.00969      +0.00001       0.01%
   cell4      matched         -0.00026     +0.00075      -0.00001      -0.01%
   cell5      matched         +0.32792     +0.23904      +0.00154       0.66%
```

Every reported tracking effect on this box is accounted for by the action multiplier.

## E3's replication does not survive

E3 reproduced at 28% against 29% across two corpora and was reported as the one clean
replication. **Both corpora were scored with the same asymmetric protocol**, so the
replication does not touch the confound — E3 asks which *variable* explains an observed
split and would favour "family" whether the family structure came from the multiplier or
from the fine-tune.

Recomputed on each leg, with a permutation floor:

```
   as reported    family sep 0.00854  p 0.004   ->  FAMILY by 114%
   multiplier     family sep 0.00423  p 0.070   ->  MOTION by 318%
   matched gain   family sep 0.00003  p 0.856   ->  NOT DISCRIMINABLE
```

**On the fine-tune's actual effect, E3 has nothing to discriminate.**

## What this does not say

It does not say arm A is equivalent to base in general. It says that in this cell, on
this criterion, at k=0.75, with both arms at the same gain, no difference is detectable
at n=195 with intervals of width ~0.003. Arm A's divergence at nominal gain, its k* of
0.921, and the handover fragility replicated 85/85 are untouched by this — they are
measurements of stability, not of tracking error, and none of them used the asymmetric
protocol.

## The check that was missing

E3 originally ranked two separations without testing whether either was distinguishable
from chance. On matched-gain data both were ~0.00003 and it reported "MOTION separates
better by 3738%".

The first attempted fix bootstrapped the separation and required its lower bound above
zero. **A separation is an absolute difference, so its resampled distribution is
non-negative and that bound is always positive** — the floor passed everything and did
nothing, while looking like a fix. Permuting the group labels gives the distribution of
separations attributable to chance, which is the quantity a floor actually needs.
