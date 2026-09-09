# Gravity arms, corrected checkpoints: INTERIM at two seeds

**2026-09-08.** First results computed from `last.pt` (epoch 80, uniform across all
runs) after [`go2-checkpoint-selection-bug.md`](go2-checkpoint-selection-bug.md).
**Interim: two seeds per arm, which cannot certify any contrast.** Recorded now so
the corrected numbers are on the record beside the buggy ones, not to be cited as a
result.

## The correction moved the numbers, and not in one direction

```
  arm    best_val epoch   OLD rate   EP80 rate   change
  A_s1        26            54.1%      57.8%      +3.7
  A_s2         1            49.6%      40.9%      -8.8
  B_s1        73            52.4%      55.8%      +3.4
  B_s2         1            17.4%      45.9%     +28.5
  C_s1        26            33.2%      28.9%      -4.3
  C_s2        60            56.0%      37.1%     -18.9
```

**B_s2 moves +28.5 points** when its epoch-1 surrogate is replaced by epoch 80 --
the single clearest confirmation that the old numbers were reading checkpoint
maturity. But **the change is not monotone in epoch**: A_s2 and C_s2 both got
*worse* with a more-trained surrogate. A better surrogate is not automatically a
better teacher, which is the same shape as the earlier v6 result where the
conditioned surrogate produced a worse policy than the cruder one.

## Corrected comparison, seed as the unit

```
  arm    n   mean%    sd     per-seed
  A      2   49.3    12.0    57.8  40.9
  B      2   50.8     7.0    55.8  45.9
  C      2   33.0     5.8    28.9  37.1

  contrast   delta        p     floor    reading
  B - A      + 1.5    1.0000   0.3333    not separated
  C - A      -16.3    0.3333   0.3333    at the floor: maximally lopsided, uncertifiable
  C - B      -17.8    0.3333   0.3333    at the floor: maximally lopsided, uncertifiable
```

**Read the floor column before the p column.** C sits below A and B on *every* seed --
the permutation test is as lopsided as it can be -- and 0.3333 is still the smallest p
two seeds per arm can produce. The design, not the data, is what fails here.

## UPDATE, same day: a third seed of arm A broke the pattern

Arm A's third seed scored **29.9%** -- below *both* C seeds. One additional seed, and:

```
              n=2 (A: s1,s2)          n=3 (A: s1,s2,s3)
  arm A       49.3  sd 12.0           42.8  sd 14.1   (57.8  40.9  29.9)
  C - A      -16.3  p at floor        -9.8  p 0.4000
  B - A       +1.5                    +8.0  p 0.5000
```

**A's range is now 29.9-57.8 and contains C's range (28.9-37.1) entirely.** The
"C sits below everything on every seed" reading below was an artifact of *which two
seeds* arm A happened to have, and it did not survive the third.

This is the clearest demonstration available of why the two-seed design was
unreportable: the pattern was maximally lopsided, perfectly consistent across every
cell measured, and **wrong**. Nothing about the data at n=2 signalled that -- only the
floor did, which is why the floor is printed beside every contrast.

**Arm A's between-seed spread is 27.9 points at matched epoch 80.** That is the real
error bar on any single-seed claim about these arms.

## What the pattern suggests, and why it is not yet a finding

- **B - A is +1.5 points.** Real gravity channels bought essentially nothing over the
  baseline that has none.
- **C is 16-18 points below both.** C is the control whose gravity channels are
  permuted across episodes and carry no tilt information.

Taken at face value that reads: **three extra input channels HURT when they carry
noise and are NEUTRAL when they carry the true signal** -- so the model is not
extracting usable tilt information from them, merely tolerating them. That would be a
real, if negative, result about the arm design, and it is the opposite of what B was
built to show.

**It is not established.** Two seeds, p at the floor, and the earlier corrected/uncorrected
comparison shows single arms moving by up to 28 points from a checkpoint change alone.
A/B/C are being taken to four seeds (B_s3/B_s4, C_s3/C_s4 configs committed, datasets
staging) which is the first count at which any of this can reach p<0.05.

## THE BASELINE, which every number above must be read against

```
  policy                        val-split completion (536 episodes)
  BASE                                96.1%   (515/536)
  A_s1                                57.8%
  B_s1                                55.8%
  D_s1                                52.8%
  B_s2                                45.9%
  A_s2                                40.9%
  C_s2                                37.1%
  A_s3                                29.9%
  C_s1                                28.9%
```

**Every arm is a large degradation.** The best corrected fine-tune completes 57.8%
against base's 96.1% -- a 38-point loss. No arm is close.

**But base's 96.1% is a selection artefact and must not be quoted as a measurement.**
The val split consists of episodes on which the base controller ran to completion, so
base is being re-run on episodes chosen for base surviving them. This project already
documented that trap: *"the number cannot come out any other way"* up to determinism,
and the 21 failures are replay nondeterminism. **The defect is using base's rate as
evidence of anything**; the arms' rates on that same cell are a real measurement, and
what they measure is damage.

**What this cell structurally cannot show.** It is incapable of detecting an arm that
beats base where base FAILS -- no such episode can enter it. That is the complement,
the 522-episode base-failure set on which base completes 0 by construction, and it is
now being scored on the corrected policies. **The two rates are reported side by side
and never summed**: this one can only show degradation, that one can only show
capability, and neither is interpretable alone.

**So the honest reading of everything above:** the arm comparison is a comparison of
how much damage each arm does, not of which arm helps. Ranking A against B against C
was answering a question subordinate to the one that matters, which is why
fine-tuning inside the surrogate costs 38 points at all.

## val_loss is NOT comparable across these arms

```
  run     val_loss@80    state dim   note
  A_s1    5.652e-04       34         baseline, no gravity channels
  A_s2    5.699e-04       34
  B_s1    5.300e-04       39         + 3 real gravity channels
  B_s2    5.586e-04       39
  C_s1    5.272e-04       39         + 3 permuted gravity channels
  C_s2    5.435e-04       39
  D_s1    1.598e-03       43         + contact channels: a DIFFERENT problem
```

Two traps here, both of which would produce a confident wrong reading:

1. **B and C sit below A**, but they predict three extra channels that are *constant
   within an episode* and therefore nearly free to predict. A mean-over-channels loss
   falls when you add easy channels. That is arithmetic, not a better model.
2. **D looks 3x worse.** It predicts 43 channels including four contact booleans. It is
   not solving the same prediction problem, and its loss is not on the same scale as
   anything else in the table.

`checkpoint_metric = val_loss` is still correct, because selection happens **within a
run** where the channel set is fixed. **Never rank arms by it.** The Chrono completion
rate is the cross-arm metric, and it is the only one.

## Scope

Rigid terrain, 536 val episodes, `--target-dw 4.0` (all seven fine-tunes stopped within
0.007 of it, over 281-456 updates), fine-tune `--seed 0` throughout. Scoring distributed
across the fleet with the host recorded per file; the measured cross-machine effect is
0.2 points on the rate, and a3's pychrono build was not among the two probed.

---

## The matched random-perturbation control: the gradient is doing real work

Every corrected fine-tune stopped at `||dW||` ~= 4.00, which is **0.277% of the policy
norm** (1445.82). So the obvious deflationary explanation had to be excluded: perhaps
the policy is simply fragile to *any* move of that size, and the fine-tune's gradient is
worth no more than noise. That control had never been run, and without it every
displacement result in this project is uninterpretable.

Three random-direction perturbations of the base policy, sized in closed form to the
same displacement (realised `||dW||` = 4.007 against the fine-tunes' 4.000-4.007):

```
  BASE-FAILURE SET (522 episodes)
  BASE                     106/522   20.3%    <- measured, NOT 0 "by construction"
  RAND11  ||dW|| 4.007     112/522   21.5%
  RAND12  ||dW|| 4.007     101/522   19.3%
  ----------------------------------------
  C_s2                     211/522   40.4%
  B_s2                     311/522   59.6%
  B_s1                     346/522   66.3%
```

**A random 0.28% weight change does nothing: 19.3% and 21.5% against base's 20.3%.**
The fine-tuned arms reach 40-66% on the same episodes. The capability gain is therefore
attributable to the fine-tuning, not to the policy being differently-capable after any
perturbation of that magnitude.

This is the control coming back **negative**, which is what makes the treated numbers
mean anything. It also retires the deflationary reading of the whole displacement
programme: `||dW||` is not just a fragility knob.

**Still open:** the same control on the val split, which decides whether the
*degradation* is equally attributable. If random scores ~96% there like base, the
fine-tune causes both halves of the trade. If random also collapses to ~58%, then the
loss on easy episodes is fragility rather than fine-tuning, and only the gain is the
method's doing.

### The val-split half of the control: random does nothing there either

```
                        val split (536)      base-failure set (522)
  BASE                    96.1%  (515)          20.3%  (106)
  RAND11  ||dW|| 4.007     96.5%  (517)          21.5%  (112)
  RAND12  ||dW|| 4.007       -                   19.3%  (101)
  ------------------------------------------------------------
  C_s2                     37.1%                 40.4%
  B_s2                     45.9%                 59.6%
  B_s1                     55.8%                 66.3%
```

**The matched random perturbation is flat on BOTH cells.** It neither degrades the easy
episodes (96.5% against base's 96.1%) nor gains anything on the hard ones (21.5% against
20.3%). A 0.28% move in a random direction leaves this policy exactly where it was.

**So both halves of the trade are caused by the fine-tuning, not by weight fragility.**
That is the complete controlled statement:

> Fine-tuning inside the surrogate **trades ~40 points of easy-terrain reliability for
> ~46 points of hard-terrain capability**, and a displacement-matched random control
> moves neither.

**Caveat on the comparison, stated rather than glossed:** BASE was scored on sliger
(`_core.so` 60457362) and RAND11 on sbel (3b0bd530), so the 96.1-vs-96.5 comparison is
cross-machine. The measured cross-machine effect is 0.2 points, and the difference here
is 0.4 -- the conclusion "random does not degrade" is not sensitive to it, but a
same-machine BASE val-split run is queued rather than assumed.

### What this fixes about the earlier reading

Reporting the val split alone made this look like a method that simply damages policies,
and the degradation half is real. But the method is **not failing to learn** -- it learns
something the base controller does not have, on exactly the terrain the base controller
cannot handle, and the control confirms the gradient rather than the perturbation is
responsible.

The remaining defect is the one the failure-timing analysis identifies: **the objective
sees 0.1 s and the evaluation runs 41 s.** Fine-tuned policies fall at a median 2.3-3.6 s,
20-40x their training horizon, with zero survivals past 20 s. Every training branch
starts from a recorded base-policy state with the observation history warmed on recorded
observations, so the policy is never optimised on the distribution it induces. It finds
genuinely better actions for hard terrain and has no mechanism to notice the drift that
kills it seconds later.

**That names the next experiment**: iterate the data, not the model. Roll the fine-tuned
policy out, add the states it actually visits to the branch pool, refit. This attacks the
covariate shift directly, and unlike lengthening the branch it does not require a
surrogate certified past 0.1 s -- which is the constraint that made v6 fail.

---

## First four-seed comparison: arm D (contact-conditioned) against arm A (baseline)

```
  arm    n   mean%    sd     per-seed
  A      4   41.0    12.1    57.8  40.9  29.9  35.3
  D      4   52.8     8.6    52.8  62.7  41.8  54.1

  D - A  +11.9   p=0.1714   floor 0.0286   NOT separated

  by band:  A 76.7  43.3  31.9  28.3  51.5  42.7
            D 94.0  65.0  49.8  37.9  54.2  55.3     D higher in 6 of 6
```

D leads by 11.9 points and is higher in **every** band. It is still **not separated**:
p = 0.1714 against an attainable floor of 0.0286, so four seeds could have produced a
significant result and this data does not. Two of A's four seeds fall inside D's range.

**The 6-of-6 band sweep is not a second piece of evidence.** The bands share the same
four seeds, so they are not independent draws and a sign test over them would be
counting one experiment six times.

**What it would take.** Between-seed sd is ~12 and ~9, so the standard error on the
difference of means is about 7.3 points at n=4. A 11.9-point effect is 1.6 SE. Detecting
it at p<0.05 needs roughly **8-10 seeds per arm**, which is 4-6 more surrogate trainings
per arm at ~1.5 h each.

**So the honest status of the arm question:** contact conditioning looks like the best
arm and the direction is consistent, but it is not established, and the current design
resolves only effects of ~20 points. This is the same conclusion the variance
decomposition predicted before the data arrived, which is at least a check that the
power analysis was right.

## FINAL rigid comparison, three arms at four seeds

```
  arm    n   mean%    sd     per-seed
  A      4   41.0   12.1    57.8  40.9  29.9  35.3    baseline, no gravity channels
  B      4   54.6   16.2    55.8  45.9  76.9  39.9    + real gravity channels
  D      4   52.8    8.6    52.8  62.7  41.8  54.1    + contact conditioning

  contrast   delta        p     floor    reading
  B - A     +13.7    0.2571   0.0286    not separated
  D - A     +11.9    0.1714   0.0286    not separated
  D - B      -1.8    0.8571   0.0286    not separated
```

**Both information-carrying arms lead the baseline by 12-14 points, and neither is
established.** Four seeds could have separated these -- the floor is 0.0286 -- and did
not. B and D are indistinguishable from each other.

**Arm B's sd is 16.2**, the largest of the three, driven by a single seed at 76.9%
against its own 39.9%. That one run would have been a headline result on its own; the
seed-level design is the only reason it is reported as variance instead.

**Status of the rigid arm question: unresolved, direction consistent, not established.**
Closing it needs 8-10 seeds per arm (4-6 more trainings per arm at ~1.5 h each). The
project moved to CRM before spending that, on the reasoning that rigid offers ~12-point
effects against ~20-point resolution while CRM offers a 15-66% tracking deficit.

Arm C stands at three seeds (28.9, 37.1, 40.5); its fourth surrogate is trained but was
not fine-tuned or scored before the fleet moved to CRM collection.

## Base-failure set at four seeds: the method beats base, the arms do not separate

```
  reference (no seeds -- one deterministic policy each)
    BASE                          20.3%
    RAND  ||dW|| 4.007  (n=2)     19.3, 21.5      mean 20.4%

  arm   n   mean    sd     per-seed
  A     4   35.8   12.0    53.4  29.9  33.1  26.8
  B     4   51.9   21.6    66.3  59.6  62.1  19.7
  C     3   43.6   12.9    32.6  40.4  57.9
  D     4   53.1   17.7    54.6  63.6  27.6  66.5

  contrast   delta      p      floor
  B - A     +16.1   0.3143   0.0286    not separated
  D - A     +17.2   0.1714   0.0286    not separated
  D - B      +1.1   0.8286   0.0286    not separated
```

**Two different questions, and they have different answers.**

*Does fine-tuning inside the surrogate add capability where base struggles?* **Yes.**
**11 of the 12 fine-tuned seeds exceed base (20.3%) and both displacement-matched random
controls (19.3, 21.5).** The single exception, B_s4 at 19.7%, sits exactly at base level
rather than below it. Arm means run 16-33 points above the reference.

*Which arm is best?* **Unresolved**, exactly as on the val split. No pairwise contrast
separates at four seeds despite a floor of 0.0286, and D-B is +1.1 with p=0.83.

**A caution on the 11-of-12 count:** those are 3 arms x 4 seeds, not 12 independent
draws of one thing, so it is a description of consistency and not a hypothesis test. No
p-value is attached to it here. BASE and RAND have no seed dimension at all -- each is
one deterministic policy -- so an arm-versus-base contrast cannot be run through the
same permutation machinery, and the honest form is the per-seed listing above.

**Between-seed sd is 12-22 points on this cell**, larger than on the val split, so the
resolution here is worse rather than better despite the effect being larger.
