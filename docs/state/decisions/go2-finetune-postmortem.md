# Fine-tuning the imported policy inside the NRD: closed, and why

> ## REOPENED 2026-09-07 -- THIS DOCUMENT'S CONCLUSION IS NOT SUPPORTED AS WRITTEN
>
> This closed the line with **"over-optimisation inside a locally-valid model, not the
> objective."** The objective was missing its largest term.
>
> `go2_reward_terms.NOT_COMPUTABLE` drops `correct_base_height` at weight **-10.0**,
> ten times `tracking_lin_vel`, for the stated reason *"no pos_z_m in the 34-D state"*.
> That was true for v4's 34-D surrogate. It is **false for every 36-D and 40-D
> surrogate**, which carry `pos_z_m` -- but the dict is a module-level constant that
> does not depend on the loaded state, so the term was dropped from runs that could
> compute it. `collision` is likewise computable on the 40-channel corpus.
>
> **Measured since, one instrument, one root, 43 episodes each:**
>
> | policy | surrogate | surviving |
> |---|---|---|
> | unmodified | -- | **43 of 43**, paired difference exactly 0 |
> | v4 | 34-D, walking | **20 of 43** |
> | base36 | 36-D, walking only | **0 of 43** |
> | arm A | 36-D, walking + excitation | **0 of 43** |
>
> Two things follow. **The excitation data is not the cause** -- base36's surrogate
> never saw an excitation row and collapses identically. And **v4's 20 of 43, which
> this document read as the configuration that works, is the same failure in milder
> form** against a baseline that completes all 43.
>
> The failure is not a fall in the scored episode. The policy **cannot hold a stand**:
> it is at 0.100 m before the first recorded row, and the commanded joint targets then
> diverge exponentially to 1e30 because Chrono clamps torque but nothing clamps the
> command. Inside its own surrogate the same policy never falls -- the surrogate
> predicts height *rising* to 0.758 m while the action reaches 3e18.
>
> **Working hypothesis, not established:** 34-D cannot represent body height, so
> gradient ascent has no pathway to it; 36-D models it and nothing in the reward
> constrains it. Adding a channel to the state without adding it to the reward made the
> surrogate strictly worse to optimise inside.
>
> **The diagnostic printed on every run from the beginning** -- `reward: 10 computable
> terms, 4 omitted; largest omitted is correct_base_height at -10.0` -- including v4's.
> It is now `wrongly_omitted(state_fields)`, checked against the loaded surrogate.
>
> Superseding text waits on the fix experiments. Nothing below is edited.


**Decided:** 2026-09-05 · **Status: closed.** Superseded by
[`quadruped-joint-level-plan.md`](quadruped-joint-level-plan.md).

**Reads on top of**, and does not restate:
[`go2-action-sensitivity-gate.md`](go2-action-sensitivity-gate.md) (the gate and its
declared criterion),
[`go2-finetune-acceptance-criterion.md`](go2-finetune-acceptance-criterion.md) (the two
rules, declared first),
[`go2-finetune-displacement-result.md`](go2-finetune-displacement-result.md) (the `dW`
series), and
[`go2-contact-mode-coverage.md`](go2-contact-mode-coverage.md) (which modes the data
actually contains). This document is the **closure record**: what the line adds up to,
what is withdrawn, and why it is not reopened.

**Choice:** stop fine-tuning `go2_cts_150k` inside the joint-level surrogate. Do not
run further attempts, do not run the targeted rare-contact-mode collection that was
staged to support them.

**What would reopen it:** a surrogate that passes the action-sensitivity gate at a
horizon long enough to cover a gait cycle. Nothing available does, and the reason is
structural rather than a matter of more data — see *Two failures, not one* below.

## The verdict, on the rules declared before the numbers existed

n = 36 paired episodes, backward-low command cell, **rigid ground only**.

| measure | result | needed to pass |
|---|---|---|
| tracking error, paired difference | **+0.0120 m/s** | ≤ −0.020 |
| episodes travelling the wrong way | **47% → 81%** | no increase |
| McNemar on that rise | p = 0.0042 | — |
| episodes completed | **17 of 43** | original does 38 of 43 |

**Fails both declared rules, and the tracking figure has the wrong sign.** Three
independent measures agree: it survives less often, it does not steer better, and it
reverses direction more often.

**One check makes the result stronger.** The episodes the fine-tune failed to finish were
the ones the *original* found easier, replicated independently on both machines. So
comparing survivors **understates** the harm. The true effect is worse than the table.

## The six attempts

`dW` is displacement of the policy weights from their starting point.

| arm | `dW` | mean\|a\| | completed | fell | diverged |
|---|---|---|---|---|---|
| **BASELINE — original policy** | 0.000 | 1.033 | **38** | 5 | 0 |
| v1 — tracking only, short | 8.898 | 5.244 | 0 | 43 | 0 |
| v5 — tracking only, long | 19.301 | — | 0 | 43 | 0 |
| v2 — upstream 8 terms | 12.392 | 4.725 | 0 | 0 | 43 |
| v3 — upstream 10 terms, long | 19.287 | 1.239 | 0 | 0 | 43 |
| v4 — upstream 10 terms, short | 8.903 | 1.188 | **17** | 4 | 22 |
| v6 — v4 + longer horizon + new model | — | — | **0** | 43 | 0 |

The baseline checkpoint's weight norm is 1445.82, so v4 moved **0.62%** and v3 **1.33%**.
Numbers from [`go2-finetune-displacement-result.md`](go2-finetune-displacement-result.md).

**What v1-v5 establish.** Two conditions matter in sequence. The *objective* decides
which failure mode you get: tracking-only **falls** at both step sizes, the upstream
objectives **diverge** instead. The *step size* then decides whether anything survives:
the same ten-term objective gives 0 of 43 long and 17 of 43 short.

**v5 was run to falsify the earlier claim that damage scales with `dW`, and it did.** The
prediction on record was that v5 would diverge like v3 at ~133 rows; it fell like v1 and
ran to full length, 3884 rows. What survives is the narrower statement above.

**What attempt 6 establishes: nothing.** It changed two variables at once, and worse, its
25-step branch length was justified by a surrogate score that turned out to be a
channel-selection artifact. Corrected, that horizon is one the model *fails*. The run was
never licensed. See
[`../lessons/experiment-design.md`](../lessons/experiment-design.md#the-same-selection-rule-run-on-two-inputs-is-not-the-same-selection).

**And attempt 4 is not an improvement.** 17 against 38 is a smaller degradation, not a
gain. If shrinking the step monotonically approaches the original policy, the limit of
this procedure is the policy we started with.

## RETRACTED

Three figures reported as findings were artifacts of the same channel-selection defect.
**They were never committed to this repository** — they were reported in session and in a
working document on dorm-pc, and this is their first written record. They must not be
cited, and if they appear in any uncommitted tree that document is wrong:

| retracted | corrected | consequence |
|---|---|---|
| corr 0.876 @ 0.5 s | **0.310 [0.179, 0.431]** | a gate **failure**, not a pass |
| 0.944 | — | withdrawn |
| transition split +0.793 | **+0.245** vs −0.069 | the effect survives at one third the size |

The transition-split finding is the one that survives: **contact conditioning helps
specifically at contact transitions.** It is real, it is a third the size first reported,
and it is not enough.

## The gate, on matched channel sets

Action-sensitivity gate, two Chrono arms differing only in policy weights.

| model | horizon | apparatus | gain | corr | cos | verdict |
|---|---|---|---|---|---|---|
| pre-noise, 34 ch | 0.1 s | — | — | **0.668** [0.583, 0.738] | — | **PASS** |
| conditioned, 40 ch | 0.1 s | 0.13 | 0.874 | 0.616 [0.522, 0.695] | 0.978 | PASS |
| conditioned, 40 ch | 0.5 s | 0.55 | 1.088 | 0.310 [0.179, 0.431] | 0.798 | **FAIL** |
| unconditioned | 0.1 s | 0.08 | 0.910 | 0.495 [0.382, 0.593] | 0.960 | PARTIAL |

**Read the first two rows together.** The surrogate attempt 4 actually used is the
*pre-noise* one, and it scores **higher** than the conditioned model at the only horizon
anything passes. Conditioning's real gain, 0.181 → 0.310, lives at 0.5 s, where the gate
fails. **There is no horizon at which any available model is both better than attempt 4's
and passing**, which is what closed the line — not the verdict above it.

## Two failures, not one, and only one is the surrogate's

**Failure A — autoregressive horizon.** Trustworthy to 0.1 s, five steps at 50 Hz. A trot
cycle is 0.3–0.5 s, so the model cannot cover one gait cycle. Closed loop leaves the
training distribution at ~1.7 s.

**Failure B — the policy is stateful, and the reduced state does not close over it.**
`go2_cts_150k` carries a five-step observation history and a 32-dim student latent. Fed
**true** states with a perfect view of the world, it still drifts off its own recorded
behaviour once it accumulates its own actions — *worse than predicting the mean action*.

> **Even a perfect dynamics model would not reproduce the recorded trajectory.**

Failure B is not a surrogate defect and no surrogate improvement touches it. It
contaminates every trajectory-matched before/after comparison in this line, including the
table at the top of this document, and it is the reason the next plan does not use
trajectory reproduction as its success metric.

**The framework's design rule is written for the plant.** When the controller has memory,
the coordinates the transition depends on include the controller's own state. That is a
gap in §4.2, found by this case study, and it is carried forward as a contribution rather
than a defect.

## What survives

1. **Contact conditioning helps at contact transitions**, +0.245 against −0.069 elsewhere.
2. **Objective before step size**, established by a falsification test that worked.
3. **Joint-level predicts better one-step than body-level and worse closed-loop**, which
   is a direct empirical case for the manuscript's own claim that rollout, not one-step
   error, must drive model selection.
4. The plant and collection defects found along the way: ground-pitch tilt (+20 points of
   usable yield), the torque perturbation channel, the NaN boundary-exit bug, and the
   `hash()` seeding non-reproducibility.

## Scope, stated with the verdict

One command regime (backward, low speed), **rigid ground**, one imported policy, one
surrogate family. It is a decisive negative about what was measured. It is not a claim
about forward commands, other speeds, soil, or about fine-tuning in general.
