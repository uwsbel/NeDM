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
