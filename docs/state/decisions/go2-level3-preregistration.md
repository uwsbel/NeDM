# Level 3 pre-registration — Go2 policy transfer to Chrono

Written **2026-09-04, before the policy checkpoint exists** (`go2_nn_tracking_v01`
is at iteration ~320/2000). Committed so git timestamps it ahead of the number.

---

# THE OPERATIVE RULE, IN ONE PLACE

Everything below this block is the record of how the rule was arrived at, across
eleven amendments. **This block is the rule.** If the two ever disagree, the
amendments are authoritative and this summary is the bug.

**What is measured.** The replay baseline drives Chrono with the reference's own
recorded commands — open-loop, unable to react. The policy is closed-loop. The
reported quantity is `policy_error / floor_error` **per reference**, same
reference, terrain, horizon and start distribution, differing only in open- versus
closed-loop.

**Primary arm.** 6.00 s (120 policy steps), rigid terrain, the **source** Chrono
build, `go2_flat_crm_ref40.npz` — the policy's own training reference file — at
pre-roll 0.0 so the window is exactly the one the policy tracked. Reference
indices **16, 17, 10, 19, 12, 13, 6, 15**: the highest-motion member of each flat
family. This is **performance on the most-moving reference in each family**, a
deliberate high draw, **not a representative sample**. Pooled floor **0.0655 m**.

**Two checkpoints, fixed in advance.** SELECTED (the save-interval checkpoint
nearest the logged in-model `position_error_m` minimum) and FINAL (`model_2000`).
Both are evaluated and **both are published**. Selection on any Chrono result is
forbidden.

**Verdict, on the per-reference ratio:**

    BEAT     median <= 0.90  AND  policy better on >= 6 of 8
    FAIL     median >  1.15   OR  policy worse  on >= 6 of 8
    PARITY   otherwise

**The count governs a median/count disagreement.** A median inside the BEAT
threshold with fewer than 6 improvements is PARITY, not BEAT.

**Co-reported, not substituted:** the median **paired absolute difference** (sign
test plus the same majority condition; no magnitude threshold in metres, on
purpose). If it agrees with the ratio, report the agreed verdict. If it
disagrees, the verdict is **SPLIT** and both are reported — a disagreement
between a scale-free and a scale-dependent statistic is itself the finding.

**The slope is NOT independent evidence.** The through-origin slope is a
floor-weighted mean ratio; on this reference set it agrees with the median ratio
to ~11%. Report it with its intercept and leave-one-out range, and count two
statistics where a reader might count three.

**The checklist runs on every verdict, whichever way it went** — asymmetric
scrutiny is a bias with a direction. **A skipped check is not a pass**; it makes
the result INCOMPLETE.

**Supplementary and labelled as such:** the least-moving eight (the other end of
the bracket — report as "on the most-moving X, on the least-moving Y"), the val
arm (0/8 training overlap; generalisation, compared by RATIO only, never by raw
error), and the 10.00 s horizon (an 80-step extrapolation beyond anything the
policy saw).

**Known limits:** this is a transfer test, not a generalisation test (8/8 primary
references were tracked during PPO); the training distribution is slow (27 of 40
references travel under 0.50 m); seven of eight references issue a constant or
single-step command; and the reward was balanced at the random-policy operating
point, which is why the policy drifted after iteration 733.

---

## The reading, stated before the measurement

The replay baseline is **open-loop**: Chrono driven by the reference's own
recorded command sequence, unable to react to anything. The policy is
**closed-loop**: it observes pose error against the reference and corrects. The
policy therefore has strictly more information, and can attack the very artefacts
the replay cannot — the cold start, the fresh contact history, the mid-episode
spawn.

**So the floor is a baseline to beat, not a noise term to subtract.** We had both
been reading it as noise. Three outcomes, named now:

| ratio | reading |
|---|---|
| policy < floor | Closed-loop correction works in Chrono. The policy actively compensates for harness artefacts an open-loop replay cannot. Strong transfer. |
| policy ≈ floor | Transfers, but adds nothing beyond replaying the commands. Weaker, still a pass. |
| policy > floor | Corrections are miscalibrated for Chrono — worse than replaying the recorded commands. The failure level 3 exists to catch, invisible without the floor. |

## Thresholds

Reported **per reference**, on `policy_mean_position_error / floor_mean_position_error`
for the same reference, terrain, horizon and start distribution — differing only
in open- versus closed-loop, so everything scale-dependent cancels.

- **Beat**: median per-reference ratio ≤ 0.90 **and** policy < floor on ≥ 6 of 8
- **Parity**: median in (0.90, 1.15], or a split verdict (beats on 4–5 of 8)
- **Fail**: median > 1.15, **or** policy > floor on ≥ 6 of 8

The count condition sits beside the median because the two can disagree, and
which one to quote is exactly the choice that must not be made afterwards.

### Why these are effect-size bounds, not noise bounds

**The replay is bit-identical across runs** — same 8 references, same config,
twice, max |diff| 0.000e+00 m. The policy path is deterministic too (inference
uses the mean action). So the ratio has no measurement noise, and the band is not
a confidence interval: it is a judgement about what size of difference is worth
calling a result. 10% is set as the smallest difference that would change what we
write.

## Named secondary prediction

`arc` is the reference to watch. Its floor is 0.177 m while five families sit
under 0.04, and unlike pivot/stop_and_go/lateral its floor barely moves between
mid-episode and episode-start starts (0.177 → 0.129) — so it is dominated by
genuine plant divergence rather than cold start, which is what feedback can
attack. **If the policy beats the floor anywhere, it should be there.** If it
beats everywhere except arc, that is evidence the beat is cold-start recovery
rather than plant compensation.

## Conditions fixed in advance

- **Primary**: 6.00 s (120 steps), random-start references, rigid terrain, source
  Chrono build. Matched to the policy's training horizon *and* its training start
  distribution.
- Supplementary and labelled as such: 10.00 s (an 80-step extrapolation beyond
  anything the policy saw), and episode-start references (which lower the floor
  and shift the evaluation distribution in the same move).
- Floors are already measured and recorded in
  `artifacts/rl_eval/README_go2_chrono_floor.md`. Primary pooled floor **0.0457 m**.

## Two caveats that are NOT escape hatches

**Every recorded command is inside the policy's action range.** Checked: over the
8 references, cmd_vx ∈ [−0.496, +0.192], cmd_vy ∈ [−0.138, 0.000], cmd_wz ∈
[−0.440, +0.508], against bounds ±0.5 / ±0.5 / ±1.0 — 0.00% of samples outside.
So a "policy > floor" result cannot be explained by commands the policy was
unable to issue. It would mean miscalibration, as stated above.

**This is a transfer test, not a generalisation test — and the overlap is
total, not "likely".** Measured, not assumed: **8 of 8** evaluated references are
episodes the policy actually tracked during PPO (the eval draws a 1178-step
window and PPO used an 1100-step window of the same episodes). "Likely" was the
word that should have triggered the check, and the true sentence is stronger than
the hedge.

So a second, supplementary arm now measures generalisation rather than
disclaiming it.

## The generalisation arm (supplementary)

`go2_flat_valref20_seg1178.npz`, 20 val-split references, **0 of 8 overlapping**
the policy's training references — confirmed, not assumed. Same builder, same
seed, same 8-family spread, one argument changed.

    PRIMARY        6.0 s, random-start, TRAIN refs   transfer, as registered
    SUPPLEMENTARY  6.0 s, random-start, VAL refs     generalisation

Train stays primary because level 3 asks whether a policy trained inside the
frozen model still works in Chrono, which is transfer. But a reader assumes
generalisation unless told, and answering the question beats disclaiming it.

**Val floor, 6.00 s, random-start, measured:**

| family | val floor | train floor |
|---|---|---|
| stop_and_go | 0.1254 | 0.0615 |
| arc | 0.1133 | 0.1771 |
| pivot | 0.0685 | 0.0372 |
| constant | 0.0683 | 0.0049 |
| yaw_step | 0.0618 | 0.0044 |
| weave | 0.0362 | 0.0277 |
| vel_step | 0.0245 | 0.0161 |
| lateral | 0.0237 | 0.0363 |
| **POOLED** | **0.0652** | **0.0457** |

**THE TWO ARMS ARE NOT ABSOLUTELY COMPARABLE, AND THE REASON IS NOT "VAL IS
HARDER".** The command distributions differ:

    train refs   cmd_vx [-0.496, +0.192]   barely any forward motion
    val refs     cmd_vx [-0.394, +0.472]   substantial forward motion

Forward is the plant's worst regime — a dead zone below ~0.35 m/s and ~4x worse
tracking than backward at the same magnitude. So the val references sit in
harder territory for the replay *and* for the policy, and the 43% higher floor is
mostly that, not novelty. **Compare ratios across arms, never the raw errors.**
Each arm is scored against its own floor, which is exactly what the ratio is for.

Action-range check repeated on the val references: cmd_vx [−0.394, +0.472],
cmd_vy [0.000, +0.078], cmd_wz [−0.532, +0.699], **0.00% outside the policy's
bounds**. The escape hatch is closed on both arms.

## One thing the numbers already say about expected effect size

Those 8 references command cmd_vx in [−0.496, +0.192] — barely any forward
motion, mostly backward or inside the measured forward dead zone below ~0.35 m/s.
The policy's authority to correct is limited by the plant in exactly this
operating region, so a modest beat is the physically expected outcome and a large
one would deserve scrutiny rather than celebration.

---

# Amendment, same day, BEFORE the final checkpoint exists

## Disclosure first

I ran the full level-3 path on **model_300** — an unconverged intermediate
checkpoint, 15% of training — as a plumbing test, to prove the policy loads and
the ratios compute before the real checkpoint lands. **I saw those numbers.**
Everything below is written knowing them, which is exactly why it is written as a
disclosure and an addition rather than as an edit to the criteria.

## The registered ratio has a denominator problem, and it is our own errdist mistake in a third costume

From the plumbing run:

| family | floor | policy | ratio | abs diff |
|---|---|---|---|---|
| arc | 0.1771 | 0.0418 | **0.24** | −0.1353 |
| stop_and_go | 0.0615 | 0.0218 | 0.35 | −0.0397 |
| pivot | 0.0372 | 0.0237 | 0.64 | −0.0135 |
| lateral | 0.0363 | 0.0266 | 0.73 | −0.0096 |
| vel_step | 0.0161 | 0.0187 | 1.16 | +0.0026 |
| weave | 0.0277 | 0.0470 | 1.69 | +0.0193 |
| constant | 0.0049 | 0.0187 | 3.79 | +0.0138 |
| yaw_step | 0.0044 | 0.0197 | **4.51** | +0.0154 |

**The two worst ratios have the two smallest floors.** The floors span 40x across
the sample, so dividing by them turns a **+1.5 cm** absolute regression into a
ratio of 4.51 while a **−13.5 cm** absolute improvement becomes 0.24. A
median-of-ratios treats the 1.5 cm as nearly 20x more significant than the
13.5 cm. That is a normalisation whose denominator varies enormously across the
sample — the same defect as `errdist` dividing by a pooled `mean_dist` over
families with 2.7x different path lengths, which this project has already been
caught by once.

## What I am NOT doing, and why

The obvious fix is to exclude references whose floor is below some threshold —
and a principled threshold exists (the floor should exceed the policy's own
in-model tracking error, ~0.018 m, or the replay is already better than the
policy's best case for reasons unrelated to transfer).

**That fix would drop `constant` and `yaw_step` — precisely the two references
where the policy currently looks worst — and would improve the apparent verdict.
So I am not making it.** A criterion changed after seeing preliminary numbers, in
the direction that flatters the subject, is not a fix; it is the thing
pre-registration exists to prevent, performed by the person who wrote the
pre-registration.

## What I am doing instead

**The registered criteria are UNCHANGED.** Median per-reference ratio ≤ 0.90 and
≥ 6 of 8 beats for a beat; median > 1.15 or ≥ 6 of 8 worse for a fail. That
verdict stands as written and will be reported first.

**Added, co-reported, not substituted:** the median **paired absolute
difference** (policy − floor, metres), which has no denominator and therefore no
denominator instability, and the count of references where each statistic favours
the policy.

- If the ratio verdict and the absolute-difference verdict **agree**, report the
  agreed verdict.
- If they **disagree**, the verdict is **SPLIT**, and both are reported with this
  table, because a disagreement between a scale-free and a scale-dependent
  statistic is a real finding about the policy — it means the policy improves the
  hard references and regresses the easy ones, or the reverse — and collapsing it
  to one number destroys the finding either way.

Neither statistic is privileged as "the real one". They measure different things
and the honest report says so.

## An additional prediction, registered before its evidence

From the command ranges already measured:

    train  cmd_vx [-0.496, +0.192]   forward span 0.192
    val    cmd_vx [-0.394, +0.472]   forward span 0.472   2.5x more forward range

A forward position error is corrected by commanding forward velocity, and small
forward corrections land inside the measured dead zone — commanded 0.30 achieves
0.030 on rigid, about 10% — while backward corrections track ~4x better. **So the
policy's correction authority is weakest exactly where the val arm spends more of
its time.**

- **Prediction:** `ratio_val > ratio_train`. Less beat on the val arm.
- **Falsifier:** `ratio_val ≤ ratio_train` means the beat is not dead-zone-limited
  and the mechanism is something else.

Registered because otherwise a worse val ratio has "it does not generalise" as
its default explanation, when the plant already predicts it — the same
misattribution as reading the 43% raw floor gap as a generalisation penalty when
it was a command distribution. The physical explanation should have to compete
with the generalisation one, not inherit the result by default.

And if the ratios come out **equal** despite 2.5x more forward range, that is the
ratio being robust across the plant's own worst nonlinearity, which is a stronger
result than equality across two arbitrary draws. Prediction due to the
coordinator; falsifier and the equality reading added here.

---

# Amendment 2: the structure of a split verdict

Both of us have now seen the model_300 plumbing numbers, so **neither of the
predictions below is blind**. They are registered flagged as such. The mechanisms
are independent of those numbers and would be defended either way, but the record
should say when a prediction was written, not only what it says.

## The registered monotonicity prediction, and why it is WEAK

A feedback controller helps where there is error to correct and can only add where
there is none, so improvement should be monotone in floor size.

  **Registered:** Spearman(floor, policy − floor) < −0.5, negative differences on
  high-floor references and positive on low-floor ones, crossover near the
  policy's own in-model tracking error.

Verified on the plumbing run: rho = **−0.857**, exact permutation p = **0.0107**
(n = 8, all 40 320 orderings enumerated, not approximated).

**But this statistic is close to mechanical and must not be reported as strong
evidence.** If the policy's error were CONSTANT across references then
policy − floor = c − floor and rho would be exactly −1 by construction. Measured
spreads:

    floor   0.0044 to 0.1771   40x
    policy  0.0187 to 0.0470   2.5x

So most of rho(floor, policy − floor) is the floor's own variance, not a fact
about the policy. A strong negative rho here mainly says "the policy's error is
more uniform than the replay's", which is a weaker claim than "the policy corrects
large errors and adds small ones".

## The informative form, registered with principled thresholds

The real question is whether the policy INHERITS reference difficulty. Two
statistics that are not near-tautological:

  **(a)** Spearman(floor, policy). Plumbing value **+0.595, exact p = 0.132** —
  not significant at n = 8. Note the power: with eight references, |rho| must
  exceed about 0.74 to reach p < 0.05, so this arm cannot support a confident
  claim either way and should be reported with its p, not as a bare correlation.

  **(b)** OLS slope of policy error on floor error. **0 means the policy's error
  is independent of how hard the reference is for open-loop replay; 1 means it
  inherits the difficulty entirely.** Thresholds set as natural thirds of that
  interval, not from data:

      slope < 1/3    policy error largely independent of reference difficulty
                     -> "corrects large errors and adds small ones" SUPPORTED
      1/3 to 2/3     partial
      slope > 2/3    the policy inherits reference difficulty
                     -> the beat is not correction, it is easier references

  Plumbing value: slope = **0.109**.

**If (b) is below 1/3 while the paired difference and the ratio disagree, the
honest headline is not "split verdict" but "the policy corrects large errors and
adds small ones"** — a characterisation, with the ratio and the paired difference
each measuring one half of it. If rho(floor, policy − floor) is near zero instead,
the two statistics disagree for reasons NOT explained by floor size and the split
verdict genuinely is ambiguity rather than structure. Framing due to the
coordinator; the weakness critique and (a)/(b) added here.

## Scope: which statistic governs where

**The paired absolute difference has no denominator and therefore cannot be
compared ACROSS arms.** Train and val floors differ by 43% and their command
distributions differ, so centimetres are not commensurable between them.

    within an arm    -> paired absolute difference governs
    between arms     -> the ratio is the only commensurable statistic

The forward-motion prediction (ratio_val > ratio_train) is therefore a RATIO claim
and stays one. Scope note due to the coordinator.

---

# Amendment 3: report the leave-one-out range, not just the slope

The slope of policy error on floor error is the statistic that replaced the
near-tautological correlation, so it is worth knowing whether it rests on one
point. With the predictor spanning 40x, it does:

| dropped | leverage (share of Sxx) | slope without it |
|---|---|---|
| **arc** | **77.7%** | +0.0936 |
| yaw_step | 7.7% | +0.1019 |
| constant | 7.5% | +0.0995 |
| vel_step | 3.9% | +0.1004 |
| weave | 1.4% | +0.1292 |
| stop_and_go | 1.1% | +0.1148 |
| lateral | 0.4% | +0.1091 |
| pivot | 0.3% | +0.1077 |

Full-sample slope **+0.1089**; leave-one-out range **+0.0936 to +0.1292**.

**The estimate is leveraged and the verdict is robust, and those are different
properties.** One reference carries three quarters of the predictor's variance,
so the point estimate is arc's to move — but every leave-one-out slope stays
inside the "< 1/3" band, so the CONCLUSION does not depend on arc.

  **Registered:** report the leave-one-out slope range alongside the point
  estimate, always. If the range straddles a threshold, that straddle IS the
  finding and no point estimate may be quoted without it.

This is the same discipline as the pooled line naming its family mix: a statistic
whose value is dominated by one member of the sample must say so on the line
where it appears. Leverage analysis due to the coordinator; verified here
independently (arc 77.7%, range 0.0936-0.1292).

---

# Amendment 4: the verdict is computed by a script that predates the result

`scripts/evaluation/report_go2_level3.py` implements every threshold and
statistic above and emits the verdict. Written and smoke-tested **before the
final checkpoint exists**, which is the point: if the verdict comes out of a
script that predates the number, nobody chooses how to compute it after seeing
the number — including whoever wrote the pre-registration, which amendment 1
exists to document as a live risk rather than a hypothetical one.

It refuses two things the pre-registration forbids: it will not compare paired
absolute differences across arms, and it will not print a slope without its
leave-one-out range.

## One rule that had to be made concrete, and is therefore registered here

The registration named the **median paired absolute difference** and the count,
but never said what makes it a BEAT. Making that concrete now, before the result:

    BEAT    median difference < 0 and policy better on >= 6 of 8
    FAIL    median difference > 0 and policy worse  on >= 6 of 8
    PARITY  otherwise

**A sign test plus the same majority condition, with NO magnitude threshold in
metres.** A magnitude threshold would have to be invented, and any value chosen
now would be chosen by someone who has seen the plumbing table — the same
objection as the excluded-reference "fix" in amendment 1, and refused for the
same reason. The ratio already carries the effect-size judgement (the registered
10%); the paired difference carries direction and consistency only. That is a
real limitation of the co-reported statistic and is stated rather than papered
over.

---

# Amendment 5: the eval eight are reselected for motion, from the policy's own pool

## The problem: the benchmark was mostly measuring station-keeping

Path length over the 6 s primary horizon, the eight references originally used:

    weave 1.790   arc 0.483   vel_step 0.276   stop_and_go 0.156
    lateral 0.134   yaw_step 0.122   constant 0.108   pivot 0.008

**Pivot travelled eight millimetres in six seconds**, against a floor of 0.0372 m
— 4.6x the entire distance covered. Six of eight moved less than 30 cm. A
tracking error several times larger than the trajectory is not tracking error,
it is drift from a stationary reference, and on six of eight references this
measured station-keeping.

It also explains the denominator instability in amendment 1: **the floors span
40x because the paths span 200x.** That was the symptom.

## The check that mattered: is the TRAINING set degenerate too?

The eval eight are drawn from the same pool the policy trains on, selected by the
same motion-blind `select_reference_episode_indices`. If the other references
looked the same, the policy would have spent 2000 iterations learning to hold
position, and the level-3 result would measure the wrong competence entirely.

Measured on all 40 training references, over the exact window the policy tracks
(rows 127–727, the 128-step context offset, 6.0 s):

    min 0.007   median 0.372   max 2.102 m
    below 0.05 m:  3/40      below 0.25 m: 14/40      below 0.50 m: 27/40

**Not degenerate.** The policy trains against a 300x range of reference motion
with a median of 0.372 m. The original eval eight (median 0.143 m) were an
unlucky draw from a pool whose median is 2.6x higher, not a representative
sample of it. Check proposed by the coordinator; it is the check that decides
between "reselect the eval set" and "the training set is wrong", and it came out
on the benign side.

**Truncation bias ruled out.** `select_reference_episode_indices` filters on
`episode_lengths >= segment_nn_steps`, and high-motion episodes are likelier to
hit the bed boundary and be truncated — which would bias the pool before any
draw. Measured: 2 of 769 flat and 6 of 118 CRM episodes excluded. Too few to
shift a median. The low-motion draw was chance, not a bias, and the note says so
because "we suspected a bias" and "we ruled one out" are different records.

## The change

The eval eight are now named indices into **`go2_flat_crm_ref40.npz` — the
policy's own training reference file** — chosen as the highest-motion member of
each flat family:

| family | old idx / path | new idx / path |
|---|---|---|
| lateral | 2 / 0.135 | **10 / 2.102** |
| constant | 1 / 0.108 | **17 / 1.730** |
| weave | 6 / 1.792 | 6 / 1.792 |
| arc | 0 / 0.477 | **16 / 0.540** |
| pivot | 3 / 0.007 | **19 / 0.461** |
| vel_step | 5 / 0.283 | **13 / 0.288** |
| stop_and_go | 4 / 0.151 | **12 / 0.283** |
| yaw_step | 7 / 0.110 | **15 / 0.277** |

    median path 0.143 -> 0.500 m (3.5x)      worst 0.007 -> 0.277 m

**The 8/8 train overlap is PRESERVED and is now exact.** Using the training
reference file rather than a re-drawn one means the eval tracks the same
episodes *and the same windows* the policy tracked, which the seg-1178 rebuild
did not. The primary still measures "transfer on references the policy actually
tracked"; only which of its own references, and that dissolves the open question
in the previous amendment. Route proposed by the coordinator.

## The new floor, and why it fixes amendment 1

    arc 0.0757  constant 0.1066  lateral 0.0775  pivot 0.0506
    stop_and_go 0.0970  vel_step 0.0717  weave 0.0338  yaw_step 0.0113
    POOLED 0.0655 m

The floor RISES (0.0457 -> 0.0655) because these references move. More important,
**the floor spread collapses from 40x to 9.4x**, which is the denominator
instability of amendment 1 fixed at its cause rather than worked around by a
statistic.

## The three flags

1. **Made after seeing the model_300 plumbing numbers.** Not blind.
2. **The direction it cuts is UNKNOWN, and that is a change from the earlier
   proposal.** A path-length *filter* on the original eight would have moved the
   verdict against the policy (median ratio 1.16, worse on 2 of 3) and I said so
   before it could be checked. But these are eight *different* episodes on which
   the policy's performance has never been measured, so the direction is
   genuinely unknown. That makes the change blinder than the filter, not less
   blind — but the claim "it cuts against the policy" no longer applies and must
   not be carried over.
3. **The justification is independent of both:** a tracking benchmark must track
   something that moves.

The original eight are reported as a secondary arm so nothing is hidden.

---

# Amendment 6: how the two eval sets are labelled and reported

## The reselected eight are MAX-MOTION PER FAMILY, not a representative sample

The original eight were an unlucky low draw. The reselected eight are a
deliberate high draw — the highest-motion member of each flat family. **Both are
non-representative.** The second is the right choice for a tracking benchmark,
and it means something specific:

  **"performance on the most-moving reference in each family"** — NOT typical
  performance.

Report it in those words. "We selected for motion" invites a reader to assume a
representative sample, and it is not one.

## The two sets are a BRACKET, not a result and an apology

    median reference path over 6 s:   original 0.143 m     reselected 0.500 m

They are the low and high ends of the same pool, so together they bracket the
range. Report as **"on the most-moving references X, on the least-moving Y"**,
which is a stronger statement than either alone and costs nothing beyond runs
already done. Neither is a footnote to the other. Framing due to the coordinator.

## PROHIBITED: comparing the new floor to the old signal

The floor rose 0.0457 → 0.0655 because the references move. The policy's
**0.0167 m is a training mean over all forty references** — a different
population entirely. `0.0655 / 0.0167 = 3.9x` is a new denominator over an old
numerator across different reference sets, and it is the same error caught four
times tonight in a fresh costume.

**No floor-to-signal ratio may be quoted until the policy has been measured on
the reselected eight specifically.** Its error there will almost certainly be
higher than 0.0167, because those are the eight that move most. Prohibition due
to the coordinator, who spotted it before it was written down.

## Registered limitation: the training distribution is slow

**27 of 40 training references travel less than 0.50 m over the tracked 6 s
window** (median 0.372 m, max 2.102 m). Not degenerate — that was checked and
settled in amendment 5 — but not ambitious either. This belongs in the writeup
beside the single-gait and thin-forward-command limitations rather than being
discovered by a reader.

---

# Amendment 7: policy checkpoint selection — registered at iteration 1086 of 2000

**Written while the run is still going and before any Chrono evaluation of any
checkpoint.** The trajectory has turned and the rule for which checkpoint gets
evaluated must exist before the answer does.

## What the trajectory shows

    iter  250   pos_err 0.0229   noise std 0.210
    iter  500           0.0171             0.150
    iter  733           0.0145  <- BEST    ~0.105
    iter 1000           0.0184             0.080
    iter 1086           0.0178             0.070

    best 0.0145 at iteration 733; latest 0.0178 at 1086
    DRIFT SINCE BEST +0.0033 m, +22.8%, over 353 iterations
    action noise std 0.690 -> 0.070

**Both late-training failure modes are present at once.** The coordinator
predicted that a reward flat below ~0.08 m gives little pressure to HOLD position
as opposed to improve it, and that the thing to watch for is drift or entropy
collapse rather than a plateau. It is drift AND entropy collapse: the policy
peaked at iteration 733 and has degraded 23% since, while exploration collapsed
by an order of magnitude. The final checkpoint will very likely be worse than one
from a third of the way in.

This is the same shape as the dynamics model's `val_loss` versus `rollout_sel`:
the last checkpoint is not the best checkpoint, and which one ships is a decision
that must be made by a rule rather than by default.

## The rule

**Selection metric: the logged in-model `/tracking/position_error_m`, minimised
over checkpoints that exist** (`save_interval` 100). Ties broken toward the
earlier checkpoint.

**Both checkpoints are evaluated in Chrono and both are reported:**

    SELECTED   the save-interval checkpoint nearest the logged minimum
    FINAL      model_2000

If they disagree, both numbers are published with the trajectory. Reporting only
the selected one would hide that selection was worth 23%; reporting only the
final one would ship a knowingly worse policy for the sake of appearing
unselected.

## Two limits of this rule, stated rather than buried

1. **The selection metric is a TRAINING metric.** `/tracking/position_error_m` is
   a mean over the 40 references the policy is training against, inside the
   model. It is not held out. So "SELECTED" means "best on its own training
   objective", and the Chrono number is what tests whether that transferred.
   This is weaker than the dynamics model's `rollout_sel`, which is computed on a
   val split, and the difference is real.

2. **Selection is NOT permitted on any Chrono result.** Evaluating several
   checkpoints in Chrono and keeping the best would be selecting on the test set.
   The Chrono evaluation runs exactly twice — SELECTED and FINAL — and the choice
   of which two was fixed by this amendment before either was run.

## Why the run is not being stopped early

Stopping now would itself be a decision made after seeing the trajectory, and it
would truncate the record that makes the drift visible. Letting it finish costs
about an hour of GPU and yields the complete curve plus the FINAL checkpoint the
comparison needs.

---

# Amendment 8: why the policy drifted — the reward was balanced at the wrong operating point

Post-hoc, formed after seeing the drift. Two tests on data not used to form the
hypothesis, and one of them corrects the hypothesis rather than confirming it.

## Test 1: did the policy trade position for smoothness? PARTIALLY

Raw, over iterations 733 → 1106:

    action_rate  0.0659 -> 0.0410     corr(iteration, action_rate) = -0.980
    pos_err      0.0145 -> 0.0159     corr(iteration, pos_err)     = +0.255

That is the predicted signature. **But `action_rate` is measured on SAMPLED
actions, and exploration noise collapsed over the same window** (σ 0.100 →
0.070), which reduces it on its own: for iid noise E[(ΔA)²] carries a term ~2σ².
Regressing it out:

    action_rate ~ 4.435 σ² + 0.019
    corr(iteration, residual) = -0.334      residual 0.0030 -> 0.0007

So roughly **90% of the action_rate decline is entropy collapse, not a trade**.
A real residual trend survives, but the dominant process in the window is the
noise schedule. The prediction is partially confirmed, and would have been
reported as strongly confirmed if the confound had not been removed.

## Test 2: where does the tracking loss actually sit? THIS IS THE FINDING

Decomposed across training from the logged terms:

    iter   position      yaw     state   total   act_rate pen   position share
       0    0.24653  0.30555   0.75521  1.30729     0.15786          18.9%
     414    0.00207  0.00124   0.03300  0.03631     0.02108           5.7%
     828    0.00193  0.00127   0.02451  0.02771     0.01180           7.0%
    1110    0.00159  0.00090   0.02057  0.02306     0.00832           6.9%

**At convergence the STATE term is 89% of the tracking loss and position is 6.9%.
The action-rate penalty is 5.2x the position term.**

The reward was solved to be balanced — and the registered measurement said so:

    under a random policy: position 0.515  yaw 0.528  state 0.281, spread 1.88x

**That balance holds at initialisation and not at convergence.** Position and yaw
errors fall ~15x during training while the state error falls ~6x, so the state
term takes over and position becomes nearly free to drift. Nothing in the reward
pushes back on a 0.0145 → 0.0178 excursion because that excursion costs 0.0007
against a total of 0.023.

## What this says about the sigma question, and what it does not

The coordinator's original concern was that `position_sigma_m = 0.55` is too
loose and the policy would plateau at 0.08–0.15 m. **The registered falsifier did
not fire** — the policy reached 0.0145 m, far past that band. A different
consequence of the same flat region appeared instead: it reached 0.0145 and then
fell back, with no gradient to hold it.

So this is a **partial hit, not a hit**: the mechanism showed up, its stated
falsifier stayed silent, and the writeup should say which.

And the prescription is different from the one the concern implied. "Tighten
position_sigma" treats a symptom. The cause is that **all three sigmas were
solved at the random-policy operating point** — a choice this document registered
explicitly and which is now shown to be the wrong operating point to balance at,
because the errors it balances shrink at very different rates. A reward balanced
where training STARTS is not balanced where it ENDS, and it is the endpoint that
determines what the policy converges to.

That is a limitation of a choice made here, found by measuring it. It belongs in
the writeup as such and not as an incidental observation.

## Epistemic status

Hypothesis formed after seeing the drift. Both tests use logged data not examined
for this purpose beforehand, so this is a post-hoc hypothesis with independent
confirmation from data not used to form it — materially stronger than a post-hoc
reading, and weaker than a pre-registration. Framing due to the coordinator, whose
arithmetic put position at ~0.1% of the tracking loss; measured it is 6.9%, the
comparison having been against the action-rate penalty rather than the tracking
loss. The conclusion survives the correction and is strengthened: the penalty is
5.2x position.

## Amendment 8b: the integral signature, tested

The proposed mechanism: `state` is the INSTANTANEOUS velocity error and
`position` is its INTEGRAL, so a small persistent velocity bias is nearly free
instantaneously and unbounded in the integral. Weighting 89% on the instantaneous
term and 6.9% on its integral is the wrong direction for exactly that failure.

Predicted signature: state error flat while position grows. Measured over
iterations 733 → 1132:

    position_error_m     0.0145 -> 0.0162    +11.7%   corr(iteration) = +0.210
    state_error_norm     0.3471 -> 0.3253     -6.3%   corr(iteration) = -0.876

**Stronger than predicted.** State error is not flat — it is still FALLING, with
a tight correlation, while position rises. The two move in OPPOSITE directions.
The policy is actively improving instantaneous velocity matching while losing
position, which is what a reward weighted 89/7 toward the instantaneous term
should produce.

This also connects to a plant property measured hours earlier for an unrelated
purpose: the drift exponent was **α = 0.98, ballistic** — a held bias, not
diffusive at 0.5. Persistent-bias drift is this plant's own signature, and the
reward under-weights the only term that integrates it. Mechanism and the α
connection due to the coordinator.

**A third consequence for the prescription.** An integral term and an
instantaneous term of the same underlying quantity cannot be balanced by their
magnitudes at ANY single operating point, because the integral's magnitude scales
with the horizon and the instantaneous one does not. Balancing at the convergence
point instead of the initialisation point would be unbalanced again at a
different episode length. Still a hypothesis about the fix; the measurement above
is not.

## Amendment 8c: the observation-scale mechanism does NOT bind, and why

A second mechanism was proposed: `hmmwv_tracking_env.py:811` hardcodes
`pose_error[:, 0] / 10.0`, so a Go2 position error of 0.0162 m arrives as 0.00162
among channels normalised to order 1 — a ~600x scale disparity, in a literal
inside a parent method that no config diff could surface.

**The literal is real. The consequence is not.** `empirical_normalization` is
`True` in the run's `train_cfg`, so rsl_rl wraps observations in
`EmpiricalNormalization` and the network receives `(x − mean) / std`. From
`model_1100.pt`, over 100,001,792 samples:

    channel      running mean   running std
    dx/10           -0.000079      0.008669
    dy/10            0.000040      0.006679
    dyaw/pi         -0.001086      0.043541

    neighbouring state_error stds: 1.34, 1.17, 1.03, 1.10, 1.12

The running std IS tiny, exactly as predicted — **and that is the divisor**. The
`/10.0` and the Go2's small scale are divided straight back out, so the channel
reaches the network at unit variance like every other.

Confirmed on the trained weights rather than argued from the mechanism. Actor
first-layer mean |w| by input channel:

    dx/10  0.1210     dy/10  0.1095     dyaw/pi 0.0622
    median over all 231 inputs: 0.0656
    -> 1.84x, 1.67x and 0.95x the median

**The network does not merely see position error; it weights it nearly twice the
average channel.** So the incentive finding stands alone and is not compounded by
an observability problem.

### What survives

- **The `/10.0` is dead code in effect, not correct code.** It is a constant
  rescale that the normaliser exactly undoes up to an offset. It is harmless
  *because* `empirical_normalization` is on — a coupling documented nowhere. Turn
  that flag off and the hazard binds immediately.
- **The hiding place is real even though this instance is neutralised.** A literal
  inside an inherited parent method cannot appear in a config diff, and we
  verified the training config key-for-key against the anchor and overrode
  `default_env_cfg` where it differed. Neither check could have reached a
  `/10.0`. That is a genuinely new location for this failure class.
- **The controls framing is correct and applies to the reward finding**, which is
  measured and unaffected: a tracking controller weighting instantaneous error
  13x its integral exhibits steady-state error, which is the textbook reason PID
  has an I term. We built that reward and observed exactly that. This makes the
  result legible as a principled one rather than a tuning accident.

Prescription 4 — "the integral term must arrive in the observation at a scale the
network can act on" — is **withdrawn for this run**. It would apply to a run
without empirical normalisation.

---

# Amendment 9: the slope is largely the ratio in disguise, and the gloss was written against the wrong reference set

## The intercept test, and it does not go our way

If `policy_i = k·floor_i` exactly, the regression returns slope `k` and intercept
0 — and then **the slope IS the ratio, leveraged, not a second reading of it.**
The through-origin slope is exactly a floor-weighted mean ratio, so comparing it
to the median ratio decides which case we are in.

Measured on the dress rehearsal (model_1100, reselected eight):

    OLS slope 0.8860      intercept -0.0103 m
    through-origin slope 0.7561   vs   median ratio 0.6795
    difference: 11%

**They agree to 11%. The slope is largely the ratio in disguise on this reference
set.** The intercept is −0.010 m against policy errors spanning 0.011–0.135 m, so
it is small but not negligible; the two statistics are not identical, but they are
not independent either.

**Consequence, stated plainly: the joint verdict is weaker than it looked.** The
pre-registration treated the ratio and the slope as separate readings. They are
not, on a reference set where policy error scales with floor. The statistics that
remain genuinely independent of the ratio are the **paired absolute difference**
(no denominator) and the **count**. Registered because a reader would otherwise
count three agreeing statistics where there are two. Raised by the coordinator
before the real number existed.

The report now prints the intercept and the through-origin/median-ratio
comparison on every run. The 15% flag for "one statistic twice" is a **display
threshold, not a verdict criterion** — no registered verdict depends on it.

## The slope gloss was written while looking at a station-keeping set

Amendment 2 registered: *"slope > 2/3 → the policy inherits reference difficulty
→ the beat is not correction, it is easier references."* That was written against
the ORIGINAL eight, where six of eight moved under 30 cm.

On references that actually move, **some inheritance of difficulty is expected** —
a harder trajectory is harder for any controller, open- or closed-loop. A slope
near 1 there may mean "the difficulty is real and affects both" rather than "the
policy got an easy draw."

The evidence that the gloss was population-dependent is our own: `spearman(floor,
policy)` moved from +0.595 (p = 0.13, slope 0.109) on the station-keeping eight to
+0.810 (p = 0.022, slope 0.886) on the moving eight. Same code, same statistic,
inverted reading — because on a station-keeping set policy error is uniform for
want of anything to track, which looks identical to "independent of difficulty".

**The thresholds stay.** 0 and 1 have data-independent meanings and that was the
whole reason for setting them from endpoints rather than from data. **The
interpretive gloss is corrected**, now, before the number arrives — because
afterwards whichever reading suits the result will look like the natural one:

    the slope measures THE FRACTION OF REFERENCE DIFFICULTY THE POLICY INHERITS.
    On a moving reference set a nonzero value is expected. The diagnostic content
    is in the ratio and the count, not in the slope alone.

Correction proposed by the coordinator; the supporting evidence is the structural
inversion measured here.

---

# Amendment 10: a fixed checklist, run on every verdict

## The disposition it replaces was a bias with a direction

I registered "treat a BEAT as needing more scrutiny than a PARITY or FAIL". The
instinct was right and the construction was wrong: **asymmetric scrutiny
systematically depresses the reported rate of whichever verdict is audited
harder**, and it feels like rigour the entire time. That is the mirror image of
the failure this document exists to prevent, and harder to notice because it
looks like caution.

A FAIL is just as easily an artifact. The CRM bed bug would have produced one. A
mismatched pre-roll would have produced one — and the unmatched least-moving arm
would have produced a wrong bracket in whichever direction the mismatch happened
to point. Correction due to the coordinator.

## The checklist, fixed now, run on any verdict

Implemented in `report_go2_level3.py` and executed automatically:

1. **policy and floor ran the same horizon** — no reference aborted or silently
   substituted
2. **the policy issued non-degenerate commands** — a policy emitting a constant
   produces entirely ordinary-looking position errors, so this cannot be read off
   the errors; the eval now records the command actually delivered to the plant
3. **per-reference numbers reconcile with the pooled figure**
4. **arms matched** — a reference appearing in two arms must have an identical
   floor, which is the only thing in either table that can detect a file,
   pre-roll or horizon mismatch
5. **arms agree in direction** — primary, least-moving and generalisation
6. **the two checkpoints agree in direction** — SELECTED and FINAL

If any fails, investigate — **regardless of which way the verdict went**. If all
pass, report the verdict as it stands, **including a BEAT**. It is also checkable
by someone else, which "I will be more suspicious" is not.

## A SKIPPED CHECK IS NOT A PASS

The first implementation skipped silently when its data was absent: the
non-degeneracy check found no `action_span` in an older eval JSON and simply did
not run, and the arms-matched check did not print at all with a single arm. Both
reported ALL CHECKS PASS.

That is the failure class this project has catalogued roughly fifteen times — a
check whose success path is reachable without the thing it checks having
happened — reproduced inside the checklist written to catch it, within minutes of
writing it. A skipped check now sets the overall result to **INCOMPLETE** and
prints `[SKIP]` with the reason and the remedy.

---

# Amendment 11: what the references actually command (a characterisation, not a change)

Found while verifying that the eval's new command recording worked: the replay
floor's command span came out **exactly 0.0** on the probe references. That is not
a bug in the replay — it is the data.

Commands issued over the 6 s evaluation window:

| | primary (most-moving) | bracket (least-moving) |
|---|---|---|
| arc / constant / lateral / pivot | 1 distinct command each | 1 each |
| stop_and_go / vel_step / yaw_step | 2 (a single step) | 1–2 |
| weave | **301** | 301 |
| **single constant for the whole 6 s** | **4 of 8** | **5 of 8** |

Over the full 11 s reference, `arc`, `constant`, `lateral` and `pivot` have **one**
distinct command; `stop_and_go`, `vel_step` and `yaw_step` have **two**; only
`weave` is richly time-varying (550 distinct).

## This is the collection design, and it makes the benchmark a CLEANER test, not a weaker one

The excitation families are piecewise-constant by construction — an "arc" is a
constant forward velocity plus a constant yaw rate. Nothing is broken.

But it changes what a reader should understand the task to be. **Seven of eight
references are constant or single-step, so the policy is not being asked to
follow a rich schedule.** The task is closer to "hold a commanded velocity and
correct the drift" than "follow a trajectory".

**And that is precisely the comparison this level-3 design tests.** With a
constant command the open-loop replay has NO mechanism to correct accumulated
drift — it can only reissue the same command — while the closed-loop policy can.
So the floor and the policy differ in exactly the one capability under
examination, with the command schedule held constant between them. A richer
schedule would have added a second difference (schedule-following ability) on top
of the one being measured.

It also explains why the plant's own drift signature matters: **α = 0.98,
ballistic, a held bias.** Under a constant command that bias integrates without
bound in open loop. Whether the policy corrects it is the whole question.

Recorded as a characterisation. **No criterion, threshold or arm changes.**

---

# Amendment 12: RETRACTION — the "drift after iteration 733" was a window artifact

Amendment 8 reported that the policy "peaked a third of the way in and has
degraded 23% since", measured at iteration 1086 over the window 733–1086. **The
completed run does not support that**, and the finding is withdrawn.

    window 733-1132 (as reported)   corr(iteration, pos_err) = +0.210
    window 733-END  (complete)      corr(iteration, pos_err) = -0.227

Same start point, longer window, **the sign flips**. Over 733→1999 the OLS slope
is −6.5e-07 m/iteration — −0.0008 m across the whole window — against a
per-iteration standard deviation of 0.0010 m. The series **oscillates**; it does
not drift. And the minimum is at iteration **1980**, not 733: the run ended
essentially at its best.

**I fitted a trend to a window that ended where I happened to be looking.** The
window was not chosen to flatter anything — it was chosen by the clock — but a
window chosen by when you measured is not evidence about a trend, and 400
iterations of a noisy series will support whichever sign the endpoints happen to
have. Same family as "an absence claim needs a time bound as well as a scope
bound", one level up: **a trend claim needs its window justified, not merely
stated.**

## Amendment 8b is withdrawn with it

The "integral signature" — state error falling while position rose, in opposite
directions — was measured over the same window. On the complete run:

    733-1132   corr(it, pos) +0.210   corr(it, state) -0.876   opposite
    733-END    corr(it, pos) -0.227   corr(it, state) -0.938   SAME direction

Position and state both improve over the full run. The opposition was the same
window artifact.

## What survives, and it is the part that was never a trend

**The reward decomposition at convergence stands**, because it is a snapshot of
the converged values rather than a claim about their evolution:

    at convergence: state 89% of the tracking loss, position 6.9%,
    action-rate penalty 5.2x the position term

and so does the conclusion drawn from it — **the reward was balanced at the
random-policy operating point and that balance does not hold at convergence**,
which is arithmetic on two measured operating points, not a fitted trend.

What is withdrawn is the claim that this *caused observable degradation*. It did
not. The reward is mis-balanced at convergence AND the policy did not drift; both
are true, and I connected them on 400 iterations of noise.

## Consequence for the checkpoint rule: SELECTED and FINAL coincide

The registered rule selects the save-interval checkpoint nearest the logged
minimum. That minimum is at iteration 1980, and the nearest saved checkpoint is
`model_1999` — which is also the last one written. **SELECTED == FINAL ==
model_1999**, so the two arms of the rule collapse to one evaluation and the
question "was selection worth anything?" answers itself: **no, on this run.**

The rule still did its job. It was written when the evidence pointed the other
way, it named the choice in advance, and it resolved to "no difference" without
anyone deciding that after the fact.
