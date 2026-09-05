# Go2 fine-tune acceptance criterion

**Written 2026-09-04, before any fine-tuned checkpoint exists.** Every number below
is measured; the detectable effect is stated beside each threshold so a result that
lands inside the noise is recognisable as such rather than argued about afterwards.

Predicate and cell definitions: [go2-level3-preregistration.md] and
`go2-finetune-baseline-predicate.md`. This document adds only the decision rule.

## Scope, stated in the claim and not in a footnote

**Rigid terrain only.** All 3,503 merged episodes are `terrain_type: rigid` (1,762
sbel-pc + 1,741 dorm-pc, zero CRM), on Kyle's instruction to do rigid first. Level 3
measured the terrains behaving differently — transfer gap 4.8x rigid against 2.4x
soil, tuned P gain a factor of 2 apart — so **this criterion says nothing about
soil** and any sentence reporting it must carry "on rigid terrain".

## What is scored

**The median of PAIRED per-episode differences in absolute velocity error**, in the
backward low-command cell (`0.02 <= |cmd_vx| < 0.18`, backward), both arms run on
identical episodes: same command, same spawn, same heading, same tilt, same prewalk,
same perturbation, same seed. Only the checkpoint differs.

Three choices here, each forced by a measurement:

1. **Absolute velocity error, not the achieved/commanded ratio.** The low-bin
   denominator is 0.023–0.18 m/s, so a fixed 0.05 m/s error reads as 47% at the
   median command and 213% at the smallest. The ratio's CI in this cell is
   [-20.6%, +40.5%] — it cannot establish the sign, let alone an improvement. The
   ratio is still reported, for interpretability only.
2. **Median of the differences, NOT the difference of the medians.** This is the
   whole case for a paired design, and it is not a technicality: on the same 49
   pairs the two statistics **disagree about the sign of the effect**. Difference of
   medians +0.0138 m/s says the treatment was harmful; median of paired differences
   -0.0000 says it did nothing. Only the second has the shared episode difficulty
   removed. Pairing the data collection and then computing an unpaired summary
   discards the entire benefit while appearing to keep it.
3. **k = 1, no replication.** The plant is deterministic — 50/50 episodes
   bit-identical by sha256 across separate processes, with the perturbed arm as
   negative control at 0/49. There is nothing stochastic to average.

## Thresholds, with what each can detect

Sizing uses **sd of the paired difference = 0.0325 m/s**, measured on a
realisation-diverse set. **n is the SURVIVING pair count, not the attempted one.** A
pair dies if either arm fails, so at the measured 11.9% per-arm rate the pair loss is
`1 - 0.881^2 = 22.4%` — which matches the probe's observed 23% — and 64 attempted
yields about 49.

| | threshold | 95% half-width at n=49 |
|---|---|---|
| **Primary** | median paired difference **<= -0.020 m/s** and its 95% CI excludes 0 | **+-0.0114 m/s** |
| **Anchor** | wrong-way fraction increases **significantly**, McNemar p < 0.05 on discordant pairs | detects **~18 pts**, see below |
| **Guard** | treated per-episode sd **<= 1.5x** baseline sd | — |

- **-0.020 m/s is about 26% of the baseline's 0.077 m/s median under-shoot, and
  1.65x the half-width at the n we will actually have** (see the sizing correction
  below; 1.65 is the conservative end). It is detectable; -0.010 would not have been.

- **WHAT A FAIL MEANS, and it is not what a reader will assume.** A FAIL means **no
  improvement of at least 0.020 m/s was demonstrated — NOT that no improvement
  occurred.** A threshold is a decision boundary, not a detection guarantee, and an
  effect sitting near it is close to a coin flip. At the conservative half-width:

  | true improvement | passes about |
  |---|---|
  | 0.010 m/s | 5% |
  | 0.015 | 21% |
  | **0.020 (the threshold)** | **50%** |
  | 0.025 | 79% |
  | 0.030 | 95% |

  So a genuine 0.015 m/s improvement fails this criterion four times in five. **"The
  fine-tune failed" must be reported as "any improvement was below the threshold we
  committed to in advance", never as "fine-tuning does not work."** This is the same
  sentence the anchor carries about its 18-point floor, and it is needed for the same
  reason.
- **A SIGNIFICANT wrong-way regression is a FAIL regardless of the primary.**
  Decided before the numbers exist, because a treatment of the target size can move
  the two in opposite directions: a wider spread pushes more episodes across the zero
  crossing even when the centre improves. Both probes saw the spread widen — 1.17x
  here, 1.83x on dorm-pc's — and here wrong-way rose 55% to 65% while the median
  paired difference was exactly zero. A robot that reverses direction more often is
  worse whatever its median does.

  **What the anchor can and cannot detect, stated so a non-firing anchor is not
  over-read.** At the discordance rate measured on the probe (11 discordant pairs in
  49), McNemar needs a split of at least 10/1 to reach p < 0.05 — a net swing of 9
  pairs, or about **18 percentage points** of wrong-way fraction at n = 49. The rate
  is insensitive to the discordant count over the plausible range: 18 points at 15
  discordant, 20 points at 20.

  So **"the anchor did not fire" means "no regression of about 18 points or more",
  NOT "no regression occurred".** The null run is the worked example: it regressed
  10 points (8 discordant against 3, net 5 pairs) and correctly did not fire. A
  moderate regression is below this test's resolution at this n and must be reported
  from the point estimate and interval, which is why those are reported
  unconditionally.

  **The significance qualifier is load-bearing and a bright line here would be a
  bug.** An earlier draft read "must not increase". On a statistic carrying +-14
  points, a treatment whose true effect on wrong-way is zero moves the observed
  fraction up or down at chance, so that rule would reject a perfectly neutral
  fine-tune about half the time — noise with a verdict attached. The test is
  **McNemar on discordant pairs**, which uses the pairing and is therefore
  substantially more powerful than the unpaired +-14 suggests. The point estimate and
  its interval are reported either way, always, whether or not it fires.
- **The spread guard exists because "improved the median, doubled the
  inconsistency" is a real outcome we would otherwise have to adjudicate after
  seeing it.** 1.5x sits between the two proxy measurements.

## Pair handling and n

- A pair is dropped when the **treated** arm fails; the drop count is reported per
  bin. **Corrected from an earlier draft**, which modelled pair loss as
  `1 - 0.881^2 = 22.4%` on the assumption that either arm could fail. It cannot:
  **the baseline arm is the already-collected episode**, which by construction
  survived collection and passed the predicate, so only the treated arm is at risk
  and it runs on specs pre-selected for having worked once. **Measured on the dress
  rehearsal: 2 of 37, or 5%**, not 23%.

  That materially changes the n arithmetic. 37 attempted yields 35 surviving, not
  the 28 the old model predicted — above the 30 minimum rather than below it. The
  earlier claim that this half alone must return INCOMPLETE was wrong.

  (The 11.9% collection failure rate is still real and still structured — by family
  8.4% to 15.2%, and within `constant` backward-high 27.3% against backward-low
  8.3% — it simply does not apply twice.)
- **A bin with fewer than 30 surviving pairs is INCOMPLETE, never a pass.** At n=30
  the half-width is 0.0145 m/s and the -0.020 threshold is only 1.4 sigma.
- **n is reported beside every figure, per family.** Omitting it is what produced the
  26-point phantom disagreement between the two halves.

## Why the correlation used for sizing is 0.737 and not 0.959

Both numbers are correct for what they measured. dorm-pc's probe pinned spawn,
heading, tilt, prewalk and perturbation to fixed values, so only the command varied
and episode difficulty was nearly shared: r = 0.959, pairing reduces sd 56%. This
probe varied the realisation the way the collection does: **r = 0.737, pairing
reduces sd 48%.** The scoring set has realisation diversity, so **0.737 is the
applicable figure** and sizing on 0.959 would understate the noise. A number measured
on a simplified set does not transfer to the diverse one.

## What would make this criterion unwritable rather than failed

If the fine-tuned arm cannot be run on the same episodes — different spawn, different
perturbation schedule, a different collector commit — the pairing is void and the
whole basis above collapses to the unpaired case, where the half-width is 0.0207 m/s
and the -0.020 threshold sits at 1.0 sigma. **Then the honest report is "not
measurable at this n", not a null result.** Note that `hash()`-seeded spawn made
exactly this failure silent until 2026-09-04; both arms must be generated with spawn
passed explicitly, or from the sha256-seeded driver.


## This criterion has been tested against a known-null treatment, and it held

Random Gaussian weight perturbation is a treatment that **should not** improve
command tracking — it degrades a policy in arbitrary directions. Running it through
this exact criterion on 49 pairs, the primary correctly did not fire: median paired
difference **-0.0000 against a threshold of -0.020**.

That is a stronger property than "the thresholds are attainable". It means the
criterion has been shown to **reject something that ought to be rejected**, on the
real plant, through the same harness the verdict will use — the reachable-failure
property demanded of every other check today, applied to the decision rule itself.
Acceptance criteria are almost never negative-controlled, and an untested one can
only be validated by the result it produces, which is too late.

**The anchor was negative-controlled on the same run and also held.** The raw
fraction moved 55% to 65%, which the discarded bright-line rule would have called a
FAIL on a treatment that did nothing. Under the rule as written it does not fire:

```
  discordant pairs   baseline-only-wrong 3, treated-only-wrong 8, total 11
  exact two-sided McNemar          p = 0.227   -> does not fire, correctly
```

So both the primary and the anchor decline on a known-null treatment. That is the
property the bright-line version lacked, measured rather than argued.

**And the test is not vacuous**, which is the failure that made an earlier sign test
worthless here: at 11 discordant pairs the smallest attainable two-sided p is 0.0010,
so this test *can* reject at 0.05 — unlike the n = 5 sign test, whose floor of 0.0625
made rejection impossible regardless of the data. Check the attainable floor before
trusting any p-value in this study.

## The paired sd is not a constant: it scales with the treatment

A second machine measured the paired sd at 0.0667 against this document's 0.0325 and
the threshold appeared to sit below the noise. It does not, and the resolution is
that **the paired difference IS the treatment-by-realisation interaction, so its
variance scales with the size of the treatment.** A larger perturbation moves the
policy further from baseline and the two arms respond more differently to the same
realisation. There is no single "the noise floor" to quote.

Measured, spanning two machines and 3.4x in treatment magnitude:

| run | tracking shift, m/s | paired sd | predicted by fit |
|---|---|---|---|
| sbel-pc, rel-sigma 0.01 | 0.0213 | 0.0325 | 0.0330 (+2%) |
| sbel-pc, rel-sigma 0.02 | 0.0329 | 0.0436 | 0.0425 (-3%) |
| dorm-pc, alpha 0.02 | 0.0728 | 0.0667 | 0.0673 (+1%) |

```
  paired_sd = 0.307 * shift^0.58        all three within 3%
```

The two machines never disagreed. They measured the same relationship at different
points on it, and dorm-pc's number lands on the extrapolation of this one's.

### Which point sizes the threshold

**The one at the effect size being tested.** Sizing the detection of a 0.020 m/s
effect using the noise of a 0.073 m/s treatment demands that the criterion resolve a
threshold-sized effect through the noise of an effect 3.6x larger. At a
threshold-sized treatment:

```
  shift 0.020  ->  sd 0.0319  ->  half-width 0.0112  ->  threshold = 1.79 sigma
  sizing on 0.0667 instead     ->  half-width 0.0233  ->  threshold = 0.86 sigma
```

**The threshold stands at -0.020 m/s.** The 1.76x quoted in the table above is
confirmed at 1.79x by this route, which is an independent check on it.

### The self-consistency condition: keep the method, do not quote the number

Because the noise grows with the effect, "detectable" is implicitly defined rather
than fixed: an effect E is detectable when `E >= 1.96 * 1.25 * sd(E) / sqrt(n)`.
**That inequality is the right method and any future threshold on this plant should
be checked against it rather than against a single quoted sd.**

**But solving it requires the exponent, and three points against two parameters do
not determine one well.** The specific value is withdrawn: an earlier draft claimed
"E >= 0.0050, and our threshold is four times it", which reported the central fit as
though the exponent were known.

**No sensitivity analysis is recorded here, because both attempts at one were
invalid in the same way.** Each varied the exponent `b` while holding the coefficient
`a = 0.307` fixed. `a` and `b` are strongly correlated in a log-log fit, so moving one
alone swings the curve off the data entirely: `b = 0.121` at `a = 0.307` predicts an
sd of 0.193 at shift 0.0213 where **0.0325 was measured**. That is not a point in any
joint confidence region, it is an incoherent pair, and the dramatic ranges both
attempts produced were artifacts of perturbing one parameter of a two-parameter fit.

A draft of this document also recorded that exponents above 1 make the inequality
unsolvable so that "nothing is detectable at any magnitude". **That is wrong in
direction and is corrected here rather than deleted, so it is not re-derived.**
`E >= k * E^b` rearranges to `E^(1-b) >= k`; for `b > 1` the exponent `1-b` is
negative and the condition becomes a CEILING satisfied by all small E. Noise then
shrinks faster than the effect, and at `b = 1.037` a 0.020 effect would be detectable
at 10.8 sigma. **The dangerous end is the LOW exponent, not the high one** — at
`b = 0.121` the floor would be 0.079, four times our threshold. A wrong caution is
worse than no caution, because the next reader reasons from it.

The honest and narrower statement: **the fit is well determined near the measured
points, because it passes through them, and is not determined away from them.**

**None of this touches the threshold**, and the reason is the distinction the whole
document rests on. **-0.020 m/s sits essentially at the lowest MEASURED point,
0.0213 m/s, where sd = 0.0325 was measured directly.** The sizing is therefore
anchored on a measurement with no fit involved, and survives the power law being
wrong entirely. The fit explains why two machines disagreed; it is not load-bearing
for anything.

Every error in this section — the original floor claim, a 29-fold sensitivity range,
and a no-solution case that had the sign backwards — occurred in the interpolated
space between the measured points. The threshold never left the data. **Put
thresholds on measured points rather than between them, so a failing fit cannot take
the criterion with it.**

**One conservatism worth keeping.** These sds come from RANDOM weight perturbation,
which degrades a policy in arbitrary directions. A fine-tune moves the weights
purposefully, and a purposeful change of the same magnitude should produce less
interaction variance, not more. So the scaling law over-estimates the noise for a
real fine-tune, which is the safe direction for a threshold.

## Correction: size by bootstrapping the statistic that is scored

The half-widths above were computed as `1.96 * 1.25 * sd / sqrt(n)`, the normal-theory
standard error of a median. **That was the wrong tool twice over: it uses an sd, which
is not robust, to size a median, which is; and its 1.25 factor assumes normality,
which the paired differences badly violate.** They are strongly peaked — most episodes
barely change under a small treatment, a few change a lot — and for a peaked
distribution the median is far MORE precise than normal theory predicts, because the
density at the median is high.

Sizing therefore uses the **exact distribution-free order-statistic interval** for a
median: the interval between the k-th and (n+1-k)-th sorted paired differences, whose
coverage comes directly from the binomial. No normality, no fit, no resampling — the
fewest assumptions between the data and the claim, which is the same principle that
anchors the threshold on a measured point.

| treatment | sd | normal-theory | bootstrap | **exact order-statistic** | threshold |
|---|---|---|---|---|---|
| rel-sigma 0.01 (shift 0.0213) | 0.0325 | 0.0114 | 0.0027 | **0.0027** (k=18, cov 0.956) | **7.39 sigma** |
| rel-sigma 0.02 (shift 0.0329) | 0.0436 | 0.0153 | 0.0096 | **0.0122** (k=16, cov 0.974) | **1.65 sigma** |

**At the threshold-sized treatment the exact and bootstrap intervals agree to four
decimals**, both [-0.0038, +0.0016], so the sharpness is real and not a bootstrap
artifact. The cause is visible in the shape: **19 of 49 paired differences lie within
+-0.005 of zero** — a small perturbation leaves many episodes essentially unchanged,
producing a spike at zero and a very high density at the median. IQR 0.028 against an
sd of 0.0325 says the same thing.

**At the larger treatment the bootstrap IS optimistic by 27%**, 0.0096 against the
exact 0.0122, exactly the failure mode bootstrap median intervals are known for at
these n. The exact figure is the one quoted.

So: **half-width 0.0027 to 0.0122 m/s depending on treatment magnitude, making
-0.020 m/s between 1.65 and 7.39 sigma. The conservative end, 1.65 sigma, is the
number to quote.** Note that this is slightly WORSE than the normal-theory 1.76 at
the larger treatment — the old formula was not uniformly conservative, it was simply
wrong in an unpredictable direction, which is the actual reason to replace it.

**This also disposes of the heavy-tail concern rather than accommodating it.**
dorm-pc found its paired differences heavy-tailed — 5 of 30 above 0.1 m/s, and its sd
dropping 38% when those five are removed — and rightly observed that a threshold sized
on so unstable an sd is fragile whichever value is chosen. The answer is not a
different sd but not using one: the median is robust to exactly those five episodes,
and a bootstrap of it inherits that robustness. **A statistic that moves 38% when five
of thirty points are dropped should not be sizing anything.**

For the record, the tails are a property of the larger treatment and not of this
plant in the criterion's regime: at rel-sigma 0.01, **0 of 49** paired differences
exceed 0.1 m/s and trimming changes the sd by 0%; at rel-sigma 0.02, 1 of 46 and -5%.

### The cell contributes, but is not the main term

dorm-pc restricted its set to this document's cell: paired sd 0.0667 over
`|vx| 0.02-0.50` against 0.0552 over `0.03-0.17` (n=10, indicative only), and 0.0744
above the cell. **Cell accounts for about 17%; treatment magnitude carries the rest.**

Its realisation test is the decisive one and it points the same way: its spawn was
FIXED at (0, 0, heading 0) while this document's varies, so **its realisation
diversity is strictly lower and its sd is still larger** — which excludes realisation
diversity as the cause.

One residual specification difference, recorded so it is not rediscovered: dorm-pc
perturbed **all 14 parameter tensors including biases**, this document's probe
perturbed **weight tensors only**. That is why equal alpha gave unequal shift (0.0728
against 0.0329) and is a reminder that "alpha 0.02" does not name a treatment.


## An unevaluable anchor is not a satisfied anchor

The dress rehearsal produced only **5 discordant pairs** at n = 35, where the
smallest attainable two-sided McNemar p is **0.0625** — above 0.05, so the test
**cannot reject regardless of the data**. Silence there means "could not look", not
"no regression occurred", and counting it as a satisfied rule would be exactly the
n = 5 sign-test error that began this line of work.

The harness therefore reports **NOT EVALUABLE** and returns **INCOMPLETE** when the
primary is satisfied but the anchor is vacuous. **At least 6 discordant pairs are
required** (`2/2^6 = 0.031 < 0.05`). This is the same rule the level-3 report already
applies: a `[SKIP]` sets INCOMPLETE, never a pass.

Note the interaction, which is not obvious: **a treatment close to null produces few
discordant pairs and therefore a vacuous anchor.** The anchor becomes evaluable
precisely when the treatment is large enough to matter — 11 discordant pairs at the
threshold-sized perturbation, against 5 at the near-null one. So a vacuous anchor is
weak evidence that the treatment did little, but it is not licence to ignore the rule.

## KNOWN GAP IN COMMIT e0d039a — read before running the pushed harness

**The harness as pushed in `e0d039a` has no physical-admissibility predicate.** If
you are running that commit, it will admit episodes that are physically impossible.
The fix exists but is not yet pushed; Kyle's authorisation was scoped to that one
push, and a second is pending.

**Surviving collection is not the same as being physically real.** The eligibility
filter was survival-based, on the argument that a collected episode "passed the
predicate by construction". That argument is wrong: an episode can blow up to
absurd-but-**finite** values and pass every finiteness check in the pipeline. Found
in this half: `rigid_constant_70`, max |joint angle| **137 rad** and max |joint
target| **2e34**, against a Go2 joint range of about +-3 rad. It was eligible.

It is also the episode that failed the replay check earlier the same night, read at
the time as an unrelated collector regression. An episode carrying a joint target of
2e34 failing to reproduce bit-for-bit was not a coincidence, and the two findings are
one.

**The threshold is not a judgement call.** Over 1,481 episodes the population is
cleanly bimodal: p99 of max|joint angle| is 3.38 rad and p99 of max|joint target| is
3.64, then it jumps to 136 and 4.6e34 with nothing between. Every bound from 4 to 10
rad excludes the identical 10 episodes (0.7%). The predicate is 5.0 rad, applied over
the **retained episode** rather than only the scored window — a run that went absurd
earlier is not rehabilitated by ending calmly. Eligible in this half: 37 -> 36.

**Do NOT reconcile this with the training-cache exclusion.** dorm-pc excludes 500 of
3,503 by per-joint 2x-URDF bounds for the training cache. That is a different
decision with a different objective: excluding aggressively from TRAINING protects
the normalisation and the learned dynamics, while excluding aggressively from
EVALUATION costs representativeness, since a physically real but difficult episode is
exactly what should be scored. The two numbers are not meant to match.

**What must match is this predicate across both halves of the evaluation set.** That
is the only version of the consistency requirement that bites, and it is what would
make the merged baseline incoherent if violated.


## Power recomputed at the merged n. The threshold was NOT revisited.

**Corrected count.** An earlier draft said ~116 attempted and ~107 surviving. That
double-counted: dorm-pc's consolidated set holds BOTH machines' data, so its "80"
already contains this half's 36. The real total is **80 attempted, 36 here and 44
there**, and the design is **stratified rather than merged** (see below), so the
n that matters is per-machine: **~33 and ~40 surviving** at the measured 8% loss.
Both clear the 30-pair minimum independently.

**Two different quantities, and only one of them may move.** The **threshold** stays
at -0.020 m/s: it states what size of improvement is worth calling an improvement,
it was fixed in advance, and changing it because more data arrived would be
goalpost-moving in the hardest form to spot, because it would look like rigour. The
**power** describes what the criterion resolves at a given n and must be restated
when n changes.

### Every half-width is quoted with its treatment magnitude

**A half-width is not a property of the criterion alone.** The paired difference is
the treatment-by-realisation interaction, so its spread scales with the treatment:
`paired_sd = 0.307 * shift^0.58`. Quoting a half-width without naming the treatment
is what made two correct measurements look contradictory for an hour.

| treatment | measured | n = 33 | n = 107 |
|---|---|---|---|
| shift 0.0213 (threshold-sized, rel-sigma 0.01) | hw 0.0076 at n=33 | 0.0076 | 0.0042 |
| shift 0.0329 (larger, rel-sigma 0.02) | hw 0.0122 at n=46 | 0.0144 | 0.0080 |

The second row is the conservative sizing and is what the thresholds are quoted
against. Both rows are correct; they describe different treatments.

### Below the threshold these are FALSE-PASS rates, not power

| true improvement | | n = 33 (this box) | n = 40 (dorm-pc) | n = 73 (combined) | direction with n |
|---|---|---|---|---|---|
| 0.010 m/s | false-pass | 8.7% | 6.7% | **2.1%** | falls — better |
| 0.015 | false-pass | 24.8% | 22.7% | **15.6%** | falls — better |
| **0.020** | **the threshold** | **50%** | **50%** | **50%** | fixed — a coin flip at exactly its own size |
| 0.025 | power | 75.2% | 77.3% | **84.4%** | rises — better |
| 0.030 | power | 91.3% | 93.3% | **97.9%** | rises — better |

(Conservative sizing, shift 0.0329. At the threshold-sized treatment every row is
sharper: 9.9% and 1.0% false-pass, 90.1% and 99.5% power at n=33.)

**Above the threshold the number is power and rising with n is the criterion
improving. Below it the number is the rate at which the criterion passes an effect
SMALLER than the one we declared meaningful — a false-pass rate — and falling with n
is also the criterion improving.** Both columns move in the direction of a better
test, in opposite numerical directions. A reader seeing 24.8% fall to 11.0% will
otherwise read it as lost sensitivity.

The 50% row does not improve with n and never will: an effect exactly at a decision
boundary is a coin flip by construction. **A FAIL still means "no improvement of at
least 0.020 m/s was demonstrated", not "no improvement occurred."** The harness now
computes that sentence from the run's own half-width rather than carrying a fixed
number, because the hardcoded version was true at n = 33 and understated the
criterion at n = 107.


## The verdict is stratified by machine, not merged

**Episodes are only bit-reproducible on the machine that collected them.** Different
Chrono builds differ in the last digits and a chaotic plant amplifies it: a replay of
an s3000000 episode on the other box differs in 147 columns from row 0, physics
included — `joint_rr_hip_target_rad` -0.15132537 against -0.15141966. Separation is
perfect by machine, 14 of 14 foreign pairs differing and 6 of 6 native pairs
identical.

The **baseline arm is unaffected**, being the recorded file rather than a replay.
The **treated arm must run where its baseline was collected**, or the two arms differ
by BUILD as well as by checkpoint — in a design whose entire claim is that only the
checkpoint differs.

So each box scores its own episodes and the paired differences are combined
afterwards. Machine cancels within each pair, which makes this an ordinary stratified
paired design and loses nothing: 33 and 40 surviving pairs both clear the 30 minimum.

**Report the paired difference PER MACHINE as well as pooled.** If the treatment
effect differs between boxes that is a machine-by-treatment interaction, and it must
stay visible rather than being averaged away. The harness writes a machine-tagged
per-episode summary (`--summary-json`) for exactly this, and **refuses to run at all**
when eligible episodes come from another host, rather than silently dropping them.

**Provenance note for the dataset:** this collection is not reproducible across
machines. For training that is benign — the seed offsets are disjoint, so no spec
appears twice and the surrogate learns across a mixture of two very slightly
different plants. For anything replay-based it is a hard constraint, and the next
person to assume a simulation reproduces across machines will assume it silently.
