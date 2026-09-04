# Level 3 pre-registration — Go2 policy transfer to Chrono

Written **2026-09-04, before the policy checkpoint exists** (`go2_nn_tracking_v01`
is at iteration ~320/2000). Committed so git timestamps it ahead of the number.

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
