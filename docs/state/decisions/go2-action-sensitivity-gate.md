# Go2 action-sensitivity gate — result

**Run 2026-09-05, criterion declared before any number existed** (in
`gate_action_sensitivity.py`'s docstring and relayed to the coordinator ahead of the
run). Verdict: **FAIL**. Step 4, fine-tuning the imported policy inside the surrogate,
does not proceed on this checkpoint.

## What was asked

Fine-tuning a policy inside a frozen surrogate only means anything if a change in the
policy's actions produces the same change in trajectory that it would in Chrono. A
surrogate can have excellent one-step accuracy and still be action-blind: it predicts
the next state well because the history already determines it. The gate measures that
directly.

Two Chrono arms, **bit-identical until a branch instant** and differing only after it:
arm A is the already-collected episode, arm B swaps the policy's *weights* mid-episode.
Both action sequences are then replayed open-loop through the surrogate from arm A's
own history, so the only difference between the two surrogate rollouts is the actions.

```
  d_chrono(h) = state_B(h) - state_A(h)      Chrono's response to the action change
  d_model(h)  = model_B(h) - model_A(h)      the surrogate's response to the same change
```

Reported per horizon as **gain** (median of the norm ratio), **corr** of magnitudes
across episodes, and **cosine** between the two response vectors within each episode.
Cosine is a pass condition, not a diagnostic: gain and corr are both norms, and a
response of the right size pointing the wrong way would pass a magnitude-only gate.
For trainability direction dominates — a magnitude error rescales the gradient step,
which optimisation absorbs; a direction error sends the policy the wrong way.

## Result: body velocity, the primary family

| horizon | σ=0.01 ratio | gain | corr | cos | σ=0.05 ratio | gain | corr | cos |
|---|---|---|---|---|---|---|---|---|
| 0.1 s | 0.10 | **1.067** | 0.384 | **0.959** | 0.09 | **1.010** | 0.705 | **0.988** |
| 0.5 s | 0.84 | 3.290 | 0.035 | 0.535 | 0.26 | 2.776 | 0.127 | 0.326 |
| 1.0 s | 4.61 | — | — | — | 1.69 | — | — | — |

`ratio` is the **apparatus check**: the surrogate's own open-loop error against the
between-arm signal it is being asked to reproduce. Above 1.0 the gain is a ratio of two
noise terms and the horizon is **not measurable**, which is reported as INCOMPLETE
rather than as a number.

**At 0.1 s the surrogate is excellent, across all five channel families and both
perturbation sizes** — gains 0.79–1.07, cosines 0.85–0.99. It is *not* action-blind.
**By 0.5 s its response is 2.8–3.3x too large and directionally wrong. By 1.0 s its own
error buries the signal.**

```
  usable horizon of this surrogate   ~0.1 s
  one gait cycle                      0.567 s
```

**A 10–30x gap, and that is the whole result.** The failure is rollout quality, not
action sensitivity.

## Why 1.0 s could not be rescued

The declared INCOMPLETE branch fired once, at σ=0.05 — a policy change **5x larger than
the acceptance criterion's threshold-sized effect**. It did not become measurable
(4.61 → 1.69). The signal scales **sub-linearly**: 5x the perturbation gave 2.7x the
divergence, an exponent of **0.62**. sbel-pc independently measured **0.58** on another
machine for another purpose, so this is a property of the plant, not of either
apparatus. It means enlarging the treatment to buy measurability has sharply
diminishing returns.

**So: at 1.0 s, the effect of a policy change five times larger than the one we care
about is still buried under the model's own error.** A fine-tune optimising there would
be chasing model noise.

## Corrections made to the criterion, recorded not silently applied

1. **The apparatus veto was pooled over all 34 channels** and therefore dominated by
   joint velocities (magnitude 5–15) while judging body velocity (0.04). It read 0.96
   at 1.0 s — just inside the veto — where the primary family's true ratio was **4.61**.
   The guard failed at the one thing it was built for, silently, in the direction of
   permitting a measurement. Now per-family.
2. **"Both horizons evaluable, else INCOMPLETE"** conflated *cannot confirm success*
   with *cannot conclude anything*. Corrected to: PASS needs all horizons evaluable and
   passing; FAIL needs any horizon evaluable and failing; INCOMPLETE only when none is.
   Both forms return FAIL for this run — the original by a wrong route.

The first run is reported under the original rule and is not restated.

## Scope — a PASS would not have been clearance

- **Open-loop.** The fine-tune runs closed-loop, where the policy reacts to the
  surrogate's own predictions. Necessary, not sufficient.
- **One instant, one kind of state**: post-128-row, mid-gait, after prewalk. A
  fine-tuned policy differs from step 0 everywhere.
- **Rigid terrain only**, val-split episodes, and only the episodes this machine
  collected — see `PROVENANCE_NOTE.md` on cross-machine reproducibility.

## What follows

Retrain targeting rollout stability, justified by the training curve independently of
this gate: `val_loss` improved 9x monotonically while `rollout_sel` degraded 15x, and
the selected checkpoint was **epoch 16 of 80** against the previous run's 67.
Noise injection first, σ declared at 0.05, one value one run.

**Expected to buy 2–5x of horizon against a 10–30x gap**, so the diagnostic bar
(rollout_sel stops diverging, selected epoch past 50) is likely and the useful bar
(usable horizon past 0.5 s) is not expected from this change alone.

If the horizon does not extend, the fallback is **short-branch model-based RL**: branch
many 5-step (0.1 s) rollouts from recorded real states rather than rolling forward for
seconds. This gate's 0.1 s numbers are the positive measurement that would justify it —
a trustworthy window of exactly that width — rather than a consolation.

## Predicted stopping point of the noise retrain — declared before its gate ran

A model corrects errors of the size it was trained to see. The injected noise is
`U(0, 0.05)` scaled by each channel's std, so the reachable horizon is bounded by where
the rollout error is still inside that range. Rollout error of the ep16 checkpoint, as
a fraction of a standard deviation:

| family | 0.1 s | 0.5 s | 1.0 s | 2.0 s |
|---|---|---|---|---|
| **body_vel** | **0.0065** | **0.0446** | 0.2338 | 0.8851 |
| body_rate | 0.0591 | 0.1873 | 0.5644 | 7.8709 |
| joint_pos | 0.0293 | 0.1066 | 0.3518 | 1.8701 |
| joint_vel | 0.0360 | 0.1465 | 0.4398 | 1.4393 |
| gravity | 0.0052 | 0.0393 | 0.1187 | 1.5864 |

For the primary family the error at 0.5 s is **0.0446 std, just inside the 0.05
ceiling**, and at 1.0 s it is **0.2338 std, 4.7x outside it**.

**PREDICTION: the retrain extends the usable horizon for body velocity to roughly
0.5 s and stops there.** Not because noise injection is exhausted as a method, but
because the range chosen covers the errors at 0.5 s and not those at 1.0 s. Other
families are already outside the range at 0.5 s, so their reachable horizon sits
between 0.1 and 0.5 s.

**This makes a partial success interpretable in advance**: reaching 0.5 s and stopping
is the noise range doing exactly what it was sized for, **not** evidence that exposure
bias was only half the story. Sigma is not being changed on this run. If a wider range
is warranted afterwards it is a second declared experiment with its own bar, not a knob
turned toward a nicer answer.

### Two readings of "inside the range", declared as discriminable

The noise is `U(0, 0.05)`, so the **maximum** is 0.05 std but the **typical** draw is
0.025, and a sample as large as body_vel's 0.5 s error (0.0446) occurs in only ~11% of
samples. Whether learning to correct an error requires seeing it *often* or merely
*sometimes* gives different predictions from the same mechanism:

```
  extends to ~0.5 s and stops    max-noise reading; rare exposure suffices
  extends to ~0.3 s and stops    typical-noise reading; exposure must be frequent
  extends past 0.5 s             the noise range is not the binding constraint
  no extension at all            exposure bias was not the mechanism
```

0.3 versus 0.5 is the discrimination the data supports; finer than that it does not.
A power law through the four body_vel points has the 0.5 s value sitting 63% above the
fit while the others are within 21%, so the crossing horizons carry real uncertainty.

### The stronger test is the ORDERING across families, which needs no fit

Only body_vel and gravity have 0.5 s errors inside the injected range at all. Ranked by
0.5 s error in std units — smallest is furthest inside the range and should improve
most:

| rank | family | 0.5 s error (std) | inside range? |
|---|---|---|---|
| 1 | gravity | 0.0393 | yes |
| 2 | **body_vel** | 0.0446 | yes, barely |
| 3 | joint_pos | 0.1066 | no, 2.1x outside |
| 4 | joint_vel | 0.1465 | no, 2.9x outside |
| 5 | body_rate | 0.1873 | no, 3.7x outside |

**PREDICTION: improvement at 0.5 s follows this order — gravity and body_vel most,
then joint_pos, joint_vel, body_rate least.**

**If all five families extend uniformly, the noise range is not the binding constraint
and both readings above are wrong**, regardless of what horizon body_vel reaches. This
is a prediction about *pattern* rather than magnitude and is much harder to satisfy by
accident, so it carries more weight than the horizon number.

---

## Result of the noise retrain — both predictions above are FALSIFIED

`go2_corrected_34d_noise`, identical except `input_noise_sigma = 0.05`.

### The diagnostic bar was met decisively

| | no noise | noise |
|---|---|---|
| selected epoch | 16 of 80 | **75 of 80** |
| best `rollout_sel` | 0.5868 | 0.5045 |
| final `rollout_sel` | 8.8339 | **0.5796** |
| max over the run | 11.6566 | **1.9657** |
| final `val_loss` | 0.002806 | 0.007559 |

`rollout_sel` stopped diverging entirely. **Exposure bias was the mechanism.** The 2.7x
worse `val_loss` is what noise costs and is not a regression.

### Prediction 1 falsified: the horizon went far past the injected range

body_vel apparatus ratio (below 1.0 = measurable):

| horizon | before | after |
|---|---|---|
| 0.1 s | 0.10 | 0.06 |
| 0.5 s | 0.84 | 0.28 |
| 1.0 s | **4.61** | **0.42** |
| 2.0 s | **15.32** | **0.89** |

Predicted saturation at ~0.5 s (max-noise reading) or ~0.3 s (typical-noise reading).
**It reached 2.0 s.** The usable horizon went from ~0.1 s to ~1–2 s, a 10–20x extension
against an expected 2–5x. The noise range was **not** the binding constraint.

### Prediction 2 falsified — the ordering test, which was the stronger one

| | predicted rank | actual improvement at 0.5 s |
|---|---|---|
| gravity | 1 | 4.09x (rank 1) |
| body_vel | 2 | 2.99x (rank 3) |
| joint_pos | 3 | 2.40x (rank 4) |
| joint_vel | 4 | 2.36x (rank 5) |
| body_rate | **5** | **3.17x (rank 2)** |

`body_rate` was predicted last — furthest outside the injected range at 0.187 std — and
came second. Only `gravity` matched. The spread is narrow (2.36–4.09x), i.e. **broadly
uniform rather than ordered by distance from the range**, which is exactly the
falsifying outcome declared in advance and does not depend on any curve fit.

**So the quantitative theory — that a model corrects errors of the size it was trained
to see, bounded by the injected range — is wrong.** Noise injection improved rollout
stability broadly and roughly uniformly. No replacement mechanism is offered here;
constructing one to fit five points after the fact is what the prediction existed to
prevent.

### The gate still FAILS, but the failure collapsed to one condition in one family

| family | 0.5 s gain / corr / cos | 1.0 s gain / corr / cos |
|---|---|---|
| **body_vel** | 1.495 / **0.143** / 0.677 | 1.689 / **0.137** / 0.859 |
| joint_pos | 0.990 / 0.997 / 0.951 | 1.002 / 0.960 / 0.976 |
| joint_vel | 0.966 / 0.991 / 0.966 | 0.985 / 0.718 / 0.963 |
| gravity | 1.192 / 0.621 / 0.949 | 1.281 / 0.779 / 0.882 |

body_vel gain is now **inside** [0.5, 2.0] where it was 3.29 and 14.71, and cosine
passes at both horizons. **Only `corr` fails, and only for body velocity.**

**The correlation failure is real, not underpowered.** Across 23 episodes the body_vel
between-arm response spans 0.011 to 0.184 at 0.5 s, CV 0.787 — comparable to body_rate
(0.689) and gravity (0.929). The episodes genuinely differ in how much they respond, so
`corr` is determined. The surrogate gets the **direction** and the **typical magnitude**
of the body-velocity response right but does not track **which** episodes respond more.

**The useful bar is not met** — it was declared as gain, corr and cosine all passing at
0.5 s. Two of three, in one family, is not that bar. Step 4 does not proceed.

---

## Is action-sensitivity predictable from the reduced state? — declared before running

`corr = 0.143` says the surrogate does not know **which** situations are more
action-sensitive: it predicts a roughly constant body-velocity response where reality
varies ~17x across episodes. That matters for policy learning because a model that
under-predicts responsiveness where the plant is lively and over-predicts where it is
sluggish yields a **systematically mis-weighted gradient** — the policy learns a
compromise wrong in both regimes.

The question that decides what to do about it: **is the response magnitude predictable
at all from what the model can see?**

**Target.** `log10 ||d_chrono||` for body_vel at 0.5 s. Log because it is strictly
positive and spans ~17x, so a multiplicative model is the natural one.

**Features, restricted to what the surrogate actually receives:**

- **F1, instant** — the 34 state channels at the branch row, plus commanded
  `vx, vy, wz`. 37 features.
- **F2, window** — F1 plus the mean and sd of each state channel over the preceding
  128 rows, which is the window the model conditions on. 105 features.

F2 exists because the model sees a 128-step history, not an instant; scoring only F1
would understate the information available to it.

**Method.** Ridge, standardised features, alpha by inner CV, **out-of-sample R² by
5-fold outer CV**. In-sample R² with 105 features and ~140 samples is ~1 by
construction and will not be reported as evidence. Baselines: predicting the mean
(R² = 0), and the surrogate's own `d_model` as a predictor of `d_chrono`
(R² ≈ corr² ≈ 0.02).

**DECISION RULE, fixed now:**

```
  out-of-sample R2 >= 0.30   PREDICTABLE. The information is in the reduced state and
                             the model is not using it. Next move is a training or
                             capacity change.
  out-of-sample R2 <= 0.10   NOT PREDICTABLE. Sensitivity is set by something outside
                             the reduced state, so corr may be unachievable under this
                             projection. That is a finding about the ABSTRACTION, not
                             about training, and the next move is the state definition.
  0.10 < R2 < 0.30           AMBIGUOUS. Report as such; do not force a direction.
```

This is the first measurement in the study that speaks directly to how the reduced
state is chosen, rather than to how well a model fits a fixed one.

### The decision rule above was VACUOUS on one branch. Re-declared, calibrated.

Before touching real data, the regression was run on synthetic data with a **known**
answer. It is correct under the null and **severely biased downward under signal**:

| n | p | true 0.0 | true 0.2 | true 0.4 | true 0.6 |
|---|---|---|---|---|---|
| 140 | 105 | −0.023 | 0.043 | 0.145 | 0.293 |
| 140 | 37 | −0.036 | 0.071 | 0.249 | 0.464 |
| **600** | **37** | **−0.001** | **0.133** | **0.342** | **0.560** |
| 600 | 105 | −0.009 | 0.118 | 0.305 | 0.520 |

At the originally planned n=140, p=105, **a true R² of 0.2 reads as 0.04**. So the
declared "R² ≤ 0.10 → NOT PREDICTABLE → change the state definition" branch was
**unreachable**: an observed 0.10 was consistent with a true R² near 0.35. That branch
would have licensed a research-direction change on a number that could not support it —
the same vacuity as an anchor test whose smallest attainable p exceeds its own
threshold.

The bias is downward, so the **upper** branch was always sound: an observed R² ≥ 0.30
implies a true R² comfortably above it. The rule looked symmetric and was not.

*For reference: in-sample R² on **pure noise** at n=140, p=105 is **0.757**. That is why
only out-of-sample numbers are reported.*

**RE-DECLARED RULE**, primary analysis at **n ≈ 600, p = 37 (F1)**, chosen because
recovery there is usable across the range. F2 is reported as secondary.

```
  observed R2 >= 0.30   -> true R2 >~ 0.39   PREDICTABLE. Information is in the
                                             reduced state; fix training/capacity.
  observed R2 <= 0.05   -> true R2 <~ 0.12   Reduced state carries little of it;
                                             the finding is about the ABSTRACTION.
  0.05 < R2 < 0.30                           AMBIGUOUS. Report the implied true
                                             range and force no direction.
```

**Additional null control:** the same pipeline run on **permuted targets** with the real
feature matrix, which must return R² ≈ 0. This catches structure in the real `X` that
synthetic Gaussian features would not reproduce.

Every reported R² will be quoted with the calibration row beside it, so the number is
interpretable rather than merely printed.

### F0 — a small feature set declared on PHYSICAL grounds, before running

`n/p ≈ 10` is needed for stable out-of-sample recovery, so p=105 would need n≈1000–2000
and the dimensional problem is the real one, not the episode count. A small set is also
the only version whose *result means something*: "R² = 0.4 across 105 features" is not
actionable, while "response magnitude is governed by contact count and gait phase" is a
statement about the physics and speaks to what the reduced state must retain.

**Chosen for a reason each, before any of them was scored** — this is not selection on
the outcome:

| feature | why it should govern action-sensitivity |
|---|---|
| `cmd_speed` = hypot(cmd_vx, cmd_vy) | a faster gait has less stance margin per step |
| `cmd_yaw` = \|cmd_wz\| | turning loads the legs asymmetrically |
| `body_speed` = hypot(vel_body_x, vel_body_y) | momentum resists a perturbation |
| `n_contact` = feet with `foot_*_in_contact` | **the direct authority channel**: a leg in flight cannot transmit a torque change to the body |
| `tilt` = hypot(grav_body_x, grav_body_y) | attitude away from vertical sets how much a change couples into translation |
| `rate_mag` = ‖roll, pitch, yaw rates‖ | an already-rotating body responds differently |
| `joint_speed` = ‖12 joint velocities‖ | limbs near their velocity limit have less headroom |
| `phase_sin`, `phase_cos` | limit-cycle phase from `fr_thigh` position and velocity, each standardised over the preceding window, as `atan2(v̂, p̂)`. **Gait phase is the standard reason identical commands respond differently** |

**9 features. At n≈600 that is n/p ≈ 66**, comfortably inside the stable regime, and it
is the PRIMARY analysis. F1 (37) and F2 (105) are reported as secondary regularised
models answering "is the information anywhere" while F0 answers "is it where physics
says it should be".

**If none of these predicts the response, that is a far more interesting null than a
shrunken R² over everything** — it would say sensitivity is not governed by the obvious
mechanical quantities at all.

**F0 calibration — recovery is near-unbiased at p=9, so both branches become reachable:**

| n | p | true 0.0 | true 0.1 | true 0.2 | true 0.4 |
|---|---|---|---|---|---|
| **600** | **9** | −0.008 | **0.077** | **0.177** | **0.379** |
| 300 | 9 | −0.013 | 0.064 | 0.165 | 0.375 |
| 150 | 9 | −0.023 | 0.039 | 0.132 | 0.342 |

**FINAL RULE for F0 at n≈600**, with the limit stated as a number rather than left open:

```
  observed R2 >= 0.30   -> true R2 ~ 0.32   PREDICTABLE. The information is in the
                                            reduced state; fix training/capacity.
  observed R2 <= 0.05   -> true R2 <~ 0.07  NOT PREDICTABLE by these quantities.
                                            The finding is about the ABSTRACTION.
  0.05 < R2 < 0.30                          AMBIGUOUS; report the implied true range.

  STATED LIMIT: this design cannot distinguish a true R2 below ~0.07 from zero.
  Below that it reports "indistinguishable from no relationship", not "no relationship".
```

Contrast with the abandoned p=105 design, where an observed 0.10 was consistent with a
true 0.35 — a factor of 3.5 of unstated slack, on the branch that would have licensed
changing the state definition.

### Result: n = 605 pairs

| features | p | in-sample | **out-of-sample R²** | calibration says true R² ≈ |
|---|---|---|---|---|
| **F0 physical** | 9 | 0.115 | **0.067** | **~0.09** |
| F1 instant | 37 | 0.191 | −0.024 | ~0 |
| F2 window | 105 | 0.334 | 0.014 | ~0 |

**Permutation null on the real feature matrix: −0.011** (5 shuffles, max −0.003). The
pipeline returns nothing when there is nothing, on the real `X`.

**Verdict under the declared rule: AMBIGUOUS** — 0.067 sits in the 0.05–0.30 band, at
its very bottom, and the implied true R² of ~0.09 is close to the stated resolution
limit of ~0.07. Per the rule, no direction is forced.

**The substantive statement, which is sharper than the label:** roughly **90% of the
variance in action-response magnitude is not explained** by commanded speed, commanded
yaw, body speed, contact count, tilt, angular rate, joint speed or gait phase. The
richer feature sets do *worse*, not better — F1 and F2 contain everything in F0 as raw
channels and recover nothing, which is the dimensionality penalty, not extra signal.

Standardised F0 coefficients, largest first — the two leading terms have physically
sensible signs, and both are small:

```
  cmd_speed   +0.105     faster gait -> larger response
  n_contact   -0.049     more feet planted -> smaller response
  rate_mag    +0.035     tilt -0.033   cmd_yaw +0.028   body_speed -0.018
  phase_sin   -0.009     joint_speed -0.009   phase_cos +0.006
```

Gait phase — the quantity most often invoked to explain why identical commands respond
differently — is **among the weakest terms here**.

### What this does NOT establish

**It bounds what LINEAR models on these features extract; it does not bound what the
surrogate could extract.** The surrogate conditions on a 128×34 history plus actions,
and F2's window summaries are a crude proxy for that. So this does **not** show that
`corr ≥ 0.5` is unachievable, and it must not be used to argue the gate's corr
condition away.

What it does say: the obvious mechanical quantities, chosen a priori, carry little of
it. The clean follow-up that would separate *"not in the state"* from *"not linearly in
these summaries"* is a nonlinear model on the full window — and that is the question
that speaks to how the reduced state should be chosen.

---

## Nonlinear diagnostic — authorised by Kyle, declared before running

### A conv net was tried first and REJECTED by its own calibration

A 7,009-parameter 1D CNN over the full 128×46 window (states + actions) **memorised
instead of generalising**, on a *noise-free* target:

```
  fold 1:  ep 30  train R2 +0.818   inner-val -0.170   TEST -0.288
           ep 200 train R2 +0.996   inner-val -0.240   TEST -0.462
```

Test R² was negative from epoch 1 while training R² climbed to 0.996. At n=615 that
capacity has nothing to stop it. **Two bugs were found and fixed on the way** — the
first version took one full-batch step per epoch (~60 gradient steps total) and could
not recover a *planted* R² of 0.40 — and it still failed after the fix.

**Part of that failure was my calibration target's fault, which is worth recording:**
the three channels the planted signal used carry only **0.5–0.6%** of their variance in
the across-episode mean, against a median of **41%** across channels. I planted a
pathological needle by accident. But the memorisation is real regardless, so the CNN is
not the right instrument at this n.

### RBF kernel ridge, calibrated on the real F0 distribution

Capacity is set entirely by `(gamma, alpha)`, both chosen on training folds only.
Planting a **nonlinear function of F0 itself** — "if the answer were a nonlinear
function of these physical quantities, would we find it?" The planted signal's own
linear-model R² is 0.663, so a third of it is genuinely nonlinear.

| planted R² | recovered |
|---|---|
| 0.00 | −0.006 |
| 0.10 | 0.014 |
| 0.20 | 0.062 |
| 0.40 | **0.199** |
| 0.60 | 0.339 |

**RULE, declared now:**

```
  observed R2 >= 0.20  -> true R2 >~ 0.40   SIGNAL. The information IS in the state,
                                            nonlinearly. Fix the surrogate (capacity,
                                            training, context input) -- NOT the state.
  observed R2 <= 0.02  -> true R2 <~ 0.10   No signal above the floor. A finding about
                                            the ABSTRACTION, with the floor stated.
  0.02 < R2 < 0.20                          INCONCLUSIVE. Say so; do not read it as
                                            the second case.

  STATED FLOOR: this design cannot distinguish a true R2 below ~0.10 from zero.
```

The linear model is re-run on **this exact cached set and these exact folds**, so the
linear/nonlinear comparison is not confounded by a different episode list.

### Result, n = 615, identical folds for every row

| model | features | out-of-sample R² |
|---|---|---|
| linear ridge | F0 (9) | 0.072 |
| **kernel ridge (nonlinear)** | **F0 (9)** | **0.103** |
| linear ridge | F2 (92) | 0.043 |
| kernel ridge (nonlinear) | F2 (92) | 0.099 |

**Permutation null, kernel ridge on the real F0: −0.010** (max −0.005). At this
capacity that control is load-bearing, and it passes.

**VERDICT: INCONCLUSIVE.** 0.103 sits in the 0.02–0.20 band. Per the rule declared
before running, this is *not* read as "the information is not in the state".

**What can be said:**

- **Nonlinear structure is present but modest.** Kernel ridge beats linear on both
  feature sets, by +0.031 (F0) and +0.056 (F2), consistently and in the same direction.
- **The two feature sets agree** (0.103, 0.099) despite one having 9 features and the
  other 92, so the answer is not an artifact of a particular representation.
- By the calibration, an observed 0.103 implies a **true R² ≈ 0.25** — so roughly
  **three quarters of the variance in action-response magnitude remains unexplained**
  by any model tried, linear or nonlinear, on these summaries.

**An observation, flagged as a hypothesis and NOT as a conclusion:** the gate's
`corr ≥ 0.5` condition corresponds to R² = 0.25 between the model's response and
Chrono's — which is approximately where the *total predictability of the response from
the observable state* appears to sit. If that correspondence held, the bar would sit
right at the edge of what is extractable at all. It is **not** established: the two R²s
are different quantities (predicting `log‖d_chrono‖` from window summaries, versus the
correlation of two response magnitudes), the estimate carries wide uncertainty, and
per-channel mean and sd remain a lossy proxy for a 128×46 window.

**This does not license lowering the corr bar**, and is not offered as a reason to.
Finding structure would argue for fixing the surrogate; not finding it would argue for
changing the state. Neither argues for changing the criterion.

### What would resolve it

The floor is set by n and by the lossiness of the summaries. Resolving the inconclusive
band needs either **more pairs** (the design is ~2x short of separating a true 0.25 from
the 0.10 floor) or **a better window representation** than per-channel mean and sd.
Both are measurements on data already collected or cheaply extendable; neither is a
change of direction.

---

## The conditioning set was WRONG. Corrected, and it changes the answer.

`corr²(d_model, d_chrono) ≤ max_f R²(f(model's inputs), d_chrono)` is a genuine bound,
not an analogy: `d_model` is a deterministic function of the model's inputs, and `corr²`
is the variance explained by a *linear* function of `d_model`, hence by a restricted
function of those inputs.

**But the diagnostic above did not instantiate it.** From `gate_action_sensitivity.py`:

```
  MA = roll(tr, SA, AA, ...)      # shared history, arm A actions
  MB = roll(tr, SA, AB, ...)      # shared history, ARM B actions
```

`d_model` depends on the shared history **and on arm B's post-branch actions**. The
features used only arm A's *pre-branch* window — a strictly smaller information set. So
0.103 could not bound `corr²`, and the "ceiling" reading of it was invalid.

### Also found: 5 episodes with a DIVERGED arm B

Action differences of 1e7–1e9 rad and `d_chrono` of 1.2–2.6 m/s against a median of
0.042. The gate never applied an admissibility predicate to arm B. Medians are robust
to these but **`corr` is Pearson and is not**, and `corr` is what drove the FAIL.

**Checked: none of the 5 was among the gate's 16 episodes.** The reported FAIL is
uncontaminated. They are excluded from everything below (n = 610).

### Result on the fair conditioning set

| features | p | linear | nonlinear |
|---|---|---|---|
| F0 pre-branch only | 9 | 0.075 | 0.101 |
| **FA action-difference only** | 14 | 0.181 | **0.190** |
| **F0+FA (fair set)** | 23 | 0.212 | **0.207** |
| F0+FA+F2 (richest) | 115 | 0.177 | 0.182 |

Permutation null on the fair set: **−0.004**. Calibration at p=23, n=610:
planted 0.2 → 0.148, 0.4 → 0.309, 0.6 → 0.505, null → −0.005.

**CORRECTION TO MY OWN DECLARED GLOSS.** The rule said "observed ≥ 0.20 → true ≳ 0.40",
but that mapping was calibrated at p=9. At p=23 an observed 0.207 implies a **true
R² ≈ 0.27**, not 0.40. The qualitative branch (SIGNAL) is unchanged; the number attached
to it was wrong and is corrected rather than quietly kept.

**VERDICT: SIGNAL.** Predictability doubled once the missing term was included.

### Does the bound bind? No — but only just, and it is a lower bound

```
  total predictability (lower bound)   true R2 ~ 0.27   -> corr ceiling >~ 0.52
  gate requires corr >= 0.5            <=> R2 >= 0.25
  surrogate currently achieves         corr 0.137       <=> R2 ~ 0.019
```

Because these summaries are lossy, 0.27 is a **lower bound**, so the true ceiling is at
least 0.52. **The gate's corr bar is therefore NOT unreachable in principle** — the
third possibility is not supported. But the margin over the bar is slim on a lower
bound with wide uncertainty, and that is worth stating rather than rounding away.

**The actionable finding:** the surrogate captures about **7% of the predictability that
demonstrably exists** (0.019 of 0.27). And the single most predictive feature set is the
action-difference magnitude alone (0.190 of the 0.207) — **which the surrogate is fed
directly**. It has the most informative input and is not using it. That is a surrogate
deficiency, not a state-abstraction one, and it points at capacity/training rather than
at redefining the state.

---

## Localising the corr failure — and a defect in the gate itself

### The gate's corr condition was UNDERPOWERED at the n it ran

`corr` is a Pearson correlation tested against a threshold of 0.5. A correlation needs
far more episodes than a median does, and the gate ran at **n = 16**:

| | r | n | 95% CI | |
|---|---|---|---|---|
| gate as run, body_vel 0.5 s | 0.143 | 16 | **[−0.380, +0.596]** | **cannot reject corr ≥ 0.5** |
| gate as run, body_vel 1.0 s | 0.137 | 16 | [−0.385, +0.592] | cannot reject corr ≥ 0.5 |
| independent measurement, 0.5 s | **0.246** | **610** | **[+0.170, +0.319]** | **rejects corr ≥ 0.5** |

**The FAIL was driven by a condition the gate could not test.** The interval spanned the
threshold, so the observed 0.143 was not distinguishable from a passing 0.5. This is the
same vacuity as an anchor whose smallest attainable p exceeds its own threshold — built
into the one condition that decided the verdict, by the same person who built the
apparatus veto to prevent exactly this elsewhere. **n ≥ 60 would have sufficed** when the
true value is ~0.25.

**The verdict survives**, because the independent n=610 measurement puts corr at 0.246
with a CI that excludes 0.5. **But the gate's own evidence for it was inadequate**, and
the conclusion is now carried by the larger measurement rather than by the gate.

**Fix applied:** the gate now refuses to evaluate `corr` when its 95% CI reaches the
threshold, reporting INCOMPLETE for that horizon rather than FAIL — the same discipline
as the apparatus veto, which the first version simply never asked of `corr`.

### Where the failure actually is: magnitude scaling

n = 610, after arm-B admissibility:

| quantity | Pearson | Spearman |
|---|---|---|
| corr(‖d_chrono‖, ‖ΔA‖) — **Chrono** | 0.200 | **0.384** |
| corr(‖d_model‖, ‖ΔA‖) — **surrogate** | 0.058 | **0.258** |
| corr(‖d_model‖, ‖d_chrono‖) | 0.246 | 0.419 |

median ‖d_chrono‖ 0.0415, ‖d_model‖ 0.0540, gain 1.120; ‖ΔA‖ spans 32.6×.

**The answer is intermediate, and closer to the first case.** Chrono's response scales
with the action difference (Spearman 0.384); the surrogate's scales **too, but weakly**
(0.258 rank, and only 0.058 on Pearson — so what scaling it has is ordinal rather than
proportional). It is not flat, so "ignores action magnitude entirely" is wrong; it is
markedly under-responsive to *how different* the actions are, while getting direction
right (cosine 0.68–0.86) and typical magnitude right (gain 1.12).

That is a specific, nameable defect: **the surrogate compresses the dynamic range of its
action response.** It points at saturation, or at the response being partly dominated by
the model's own rollout error, rather than at a missing input.

---

## Gate re-run at full n — why, declared before running

**This is apparatus repair, not a second attempt at a better answer.**

**The corr outcome is known and is not in question.** An independent measurement at
n=610 puts corr at 0.246 with a 95% CI of [0.170, 0.319], excluding 0.5. The surrogate
fails that condition and the substantive verdict does not change.

**The re-run exists because gain and cosine were also measured at n=16 and are also
unestablished.** "Gain is now inside [0.5, 2.0]" has been quoted as the good news from
the noise retrain; it rests on the same 16 samples as the FAIL, and it deserves the same
scrutiny. A verdict in which two conditions passed on 16 samples is not a verdict,
whichever way it points.

**Change to the rule, applied to ALL conditions rather than to corr alone:** each of
gain, corr and cosine is now decided by its **95% interval**, not its point estimate —
exact order-statistic CIs for the medians (gain, cosine), Fisher-z for corr.

```
  PASS           the whole interval is inside the passing region
  FAIL           the whole interval is outside it
  INDETERMINATE  the interval straddles the threshold
```

A horizon FAILs if any condition FAILs, PASSes only if all three PASS, and is otherwise
INCOMPLETE.

**Note on the n=610 figure:** it swept all admissible pairs, 472 of which are
*train-split* episodes the surrogate was fitted on, so its rollouts there are
optimistically accurate and corr is biased **upward**. The rejection of corr ≥ 0.5 is
therefore conservative. The gate itself uses **val-split only** (157 pairs available),
which is the unbiased number.

### Re-run result: n = 200 val episodes, every condition with an interval

```
  0.5s  gain 1.159 [1.077, 1.296] PASS | corr 0.181 [ 0.043, 0.312] FAIL | cosine 0.688 [0.555, 0.849] PASS
  1.0s  gain 1.439 [1.167, 1.633] PASS | corr 0.074 [-0.066, 0.210] FAIL | cosine 0.633 [0.428, 0.799] INDETERMINATE
```

**VERDICT: FAIL** — same as before, but now established rather than asserted.

**The specific thing this re-run existed to check survived.** "Gain is now inside
[0.5, 2.0]" was the good news from the noise retrain and rested on 16 samples. At n=200
its interval is **entirely inside the band at both horizons** — [1.077, 1.296] and
[1.167, 1.633]. It is a real result, not a small-sample artifact. Cosine also passes
outright at 0.5 s, and is indeterminate at 1.0 s.

**The failure is isolated to `corr`, and now decisively:** upper bounds of 0.312 and
0.210 against a threshold of 0.5, where at n=16 the interval had reached 0.596 and could
not reject it.

**The predicted contamination direction was confirmed.** corr on val-only episodes is
**0.181** at 0.5 s, against **0.246** on the mixed train+val set — lower, as expected,
because the surrogate was fitted on the train episodes and its rollouts there are
optimistically accurate. The earlier figure was biased upward and the rejection was
conservative, as stated in advance.

Other families at n=200 remain strong — joint_pos and joint_vel hold gain ≈ 1.0 with
corr 0.92–0.997 and cosine 0.96–0.98 through 1.0 s. The deficiency is specific to body
velocity, and within that, specific to episode-level magnitude structure.

### The 0.1 s row — and why it cannot be decided here

body_vel, n = 200 val, retrained surrogate, rel-sigma 0.01:

| horizon | apparatus | gain | corr | cosine | all three |
|---|---|---|---|---|---|
| **0.1 s** | 0.08 | **0.910 [0.868, 0.976] PASS** | **0.495 [0.382, 0.593] INDETERMINATE** | **0.960 [0.931, 0.985] PASS** | **INDETERMINATE** |
| 0.5 s | 0.33 | 1.159 [1.077, 1.296] PASS | 0.181 [0.043, 0.312] FAIL | 0.688 [0.555, 0.849] PASS | FAIL |
| 1.0 s | 0.72 | 1.439 [1.167, 1.633] PASS | 0.074 [−0.066, 0.210] FAIL | 0.633 [0.428, 0.799] INDETERMINATE | FAIL |
| 2.0 s | 1.84 | 3.554 FAIL | 0.079 FAIL | 0.555 INDETERMINATE | FAIL |

**The corr point estimate at 0.1 s is 0.495 against a threshold of 0.500** — within 1%
of the bar, with an interval straddling it. Two of three conditions pass cleanly and
decisively; the third is exactly at the line.

**This is not a sample-size problem that more data fixes.** If the true value is 0.495,
**no n resolves it** — the interval converges onto the threshold. Deciding it requires
the truth to be meaningfully away from 0.5:

```
  true corr 0.45  ->  n ~ 1600 to establish FAIL
  true corr 0.495 ->  unreachable at any n
  true corr 0.55  ->  n ~ 1600 to establish PASS
  true corr 0.60  ->  n ~ 200 (already sufficient)
```

**And the ceiling here is n ≈ 306.** That is every locally-reproducible val episode;
251 are already branched, leaving headroom of 55. At n = 306 the interval at r = 0.495
is roughly [0.41, 0.58] — still straddling.

**So the short-branch route is NOT justified on the gate's own terms, and cannot be made
so with the data available on this machine.** That is a different answer from "it
fails": the gate is simply unable to decide its own condition at this horizon.

The `s3000000` half's val episodes could raise n, but they are only reproducible on
`kyle-sbel`, so their arm B would have to be generated there — see the cross-machine
section above.

**The threshold is not being relitigated.** 0.495 is not action-blind, and that is a
factual observation about where the point estimate sits, not an argument for moving a
bar that was declared in advance.

---

## Hypothesis: the noise injection CAUSED the corr defect

Noise injection trains `f(s + e)` against a target of `s[t+1] − (s + e)`, so the
predicted next state is `(s + e) + f(s + e) = s[t+1]`. **The model is explicitly trained
to annihilate state perturbations** — which is precisely why it fixed the rollout
divergence, because that is contraction, and contraction is stability.

In the gate, the two arms differ only in actions, and after one step that difference
**is** a state difference. A model trained to contract state differences suppresses the
growth of the between-arm difference while preserving its direction, since contraction
is roughly isotropic. **That is the dynamic-range compression, and it is the same
property that bought the stability.** If so the trade-off is intrinsic: one knob, two
effects, opposite signs.

**PARTIAL FIT, stated before testing.** The mechanism predicts *under*-response, but the
measured gain is **1.159 at 0.5 s and 1.439 at 1.0 s** — the model over-responds in
median magnitude. So contraction does not explain gain. What it does explain is
compression of the *spread across episodes*: episodes with larger action differences
have their extra response suppressed most, flattening the relationship and lowering
`corr` while leaving the median ratio intact. The defect is in `corr` and in Pearson
scaling (0.058 vs Chrono's 0.200), which is what the mechanism actually predicts.

**FREE TEST, run before spending an hour of training.** Both checkpoints exist. Measure
`corr` at 0.1 s on the **same 200 val episodes**:

```
  pre-noise corr MATERIALLY HIGHER  -> mechanism supported
  pre-noise corr <= post-noise      -> mechanism NOT supported; noise was not the cause
```

**CONFOUND, acknowledged in advance:** the pre-noise checkpoint is epoch 16 of 80 and
badly undertrained — it is the model whose `rollout_sel` diverged. A difference could be
training stage rather than noise. So a *positive* result is suggestive rather than
conclusive, while a *null* result argues against the mechanism more cleanly, since the
undertrained model has no reason to be worse at this.

### Result: the contraction mechanism is CONFIRMED, and it reframes the options

Same ~200 val episodes, same perturbation, both checkpoints:

| horizon | PRE-NOISE (ep16) app / gain / corr / cos | POST-NOISE (ep75) app / gain / corr / cos |
|---|---|---|
| **0.1 s** | 0.14 / 1.077 / **0.668** / 0.962 | 0.08 / 0.910 / **0.495** / 0.960 |
| 0.5 s | **1.12 (swamped)** / 2.304 / 0.202 / 0.504 | 0.33 / 1.159 / 0.181 / 0.688 |
| 1.0 s | **3.27 (swamped)** / 6.059 / 0.115 / 0.315 | 0.72 / 1.439 / 0.074 / 0.633 |

**corr at 0.1 s: pre-noise 0.668 [0.583, 0.738] against post-noise 0.495 [0.382, 0.593].**
A difference of +0.173 with barely-overlapping intervals. **Noise injection cost action
sensitivity, exactly as the contraction argument predicts.** The trade-off is real and
now measured rather than argued: one knob, stability up, sensitivity down.

### And the pre-noise model PASSES THE GATE at 0.1 s

```
  PRE-NOISE at 0.1 s, apparatus 0.14
    gain    1.077 [1.007, 1.152]  PASS
    corr    0.668 [0.583, 0.738]  PASS
    cosine  0.962 [0.939, 0.977]  PASS
    ==> GATE AT 0.1 s: PASS
```

**All three declared conditions, on intervals, with no threshold moved.** The post-noise
model is INDETERMINATE at the same horizon.

**This opens the short-branch route on the gate's own terms — using the checkpoint we
had written off.** A 5-step (0.1 s at 50 Hz) branch from a recorded real state needs a
model trustworthy over exactly 0.1 s, and never runs long enough for its rollout
instability to matter, because the state is reset from real data before it can diverge.
The property that model lacks is the one short-branch does not use; the property it has
is the one short-branch requires.

**The noise retrain fixed the wrong thing for this method.** It bought stability over
1–2 s, which short-branch does not need, and spent action sensitivity at 0.1 s, which
short-branch does need.

**Caveats, unchanged:** the pre-noise checkpoint is epoch 16 of 80 and undertrained — its
PASS is measured rather than inferred, but it is not a well-trained model. It is
emphatically NOT usable beyond 0.1 s (apparatus 1.12 at 0.5 s). Open-loop, one branch
instant, rigid terrain, val episodes on this machine only. A PASS here remains necessary
and not sufficient for closed-loop training.

**This also re-targets the bounded σ = 0.025 experiment**, which now has a quantitative
prediction rather than a directional one: corr at 0.1 s should land between 0.495 and
0.668 if contraction scales with noise magnitude.

---

## Defence of the checkpoint choice — written BEFORE the fine-tune produces any number

Adopting the pre-noise checkpoint after a retrain made the deciding condition *worse*
invites the reading that we picked whichever model scored best after seeing both. The
defence, recorded now so it cannot look constructed later:

1. **The criterion was fixed before either checkpoint was gated** — thresholds, primary
   family, horizons, apparatus veto and INCOMPLETE branch were all declared and relayed
   to the coordinator before any gate ran.
2. **Both checkpoints were measured identically**: same 200 val episodes, same
   perturbation, same folds, same intervals, same code.
3. **The mechanism was predicted before the comparison, by the coordinator, not
   after.** Precisely: *it was predicted that noise injection compresses action
   sensitivity and that reducing it should raise `corr`.* **Nobody predicted the
   pre-noise model would clear 0.5.** The weaker claim is the true one and is the one
   that should be stated, because the stronger one is checkable and false.
4. **The pre-noise PASS is a measurement against a pre-registered bar**, not a selection
   between checkpoints — it passes all three conditions on intervals with no threshold
   moved.

**What would undermine this defence**: evaluating in Chrono more than once. See the
one-shot rule below.

---

## Fine-tune feasibility check, run BEFORE building anything

### Correction: the surrogate state does NOT use a mixed convention

I stated mid-investigation that `roll_rate_radps` and `yaw_rate_radps` were Euler rates
while only the y-channel was body-frame. **That is wrong.** Measured over 500 rows:

```
  max |roll_rate_radps - ang_vel_body_x_radps| = 0.000e+00
  max |yaw_rate_radps  - ang_vel_body_z_radps| = 0.000e+00
```

They are the *same columns* under misleading names. The 34-D state carries true
body-frame angular velocity, so **the policy's 45-D observation is fully reconstructible
from the surrogate state** — `ang_vel(3) | gravity(3) | command(3) | dof_pos(12) |
dof_vel(12) | prev_actions(12)`, every block available. That axis of feasibility is
confirmed.

### THE POLICY IS RECURRENT, and that changes the short-branch design

`go2_cts_150k.pt` is a `legged_gym` `_TorchPolicyExporter` with **460,972 parameters and
hidden state carried across calls**. Feeding it the same observation twice in a row
gives outputs differing by **2.52**:

```
  same input twice in a row        max|diff| 2.523
  same input after a different one max|diff| 1.386
```

So a single reconstructed observation cannot reproduce the policy's action — the action
depends on the whole call history since reset. Replaying sequentially with the state
carried and teacher-forced `prev_actions` converges but does not close:

```
  first 10 control steps  median max|diff| 0.374
  last 50 control steps    median max|diff| 0.120     (~4% of action magnitude ~3.2)
```

**RETRACTED — this paragraph was wrong.** It read: *"the residual is structural; the
hidden state built before t = 2.37 s was never logged and cannot be recovered."* That
reasoned from a plausible mechanism (an RNN hidden state) instead of from the file.

**The policy's state is a 5-STEP OBSERVATION HISTORY, not an RNN hidden state.** Measured
here — feeding one observation repeatedly, the output stabilises at call 6, so the
history length is exactly 5 — and independently documented at
`src/nedm/quadruped/imported_policy.py:24-29`: a concurrent teacher-student model whose
5-step history feeds a `student_encoder` to a 32-dim latent, actor consuming
`[obs 45, latent 32] = 77`. The first layer is `(512, 225)` and `225 = 5 x 45`.

**So the state depends on nothing before those five steps.** The unlogged startup period
is irrelevant, five recorded observations reconstruct the state by construction, and
**the branch reset is clean** — which is what was authorised. The warm-up is 5 forward
passes, not ~64.

The 0.120 residual is a separate and smaller problem: **observation reconstruction
error accumulated over a long teacher-forced replay**, not inaccessible state.

### What this means for short-branch fine-tuning

- **Each branch needs a 5-step warm-up** — the five recorded observations preceding the
  branch point, which determine the policy's state exactly. Five forward passes, not
  sixty-four.
- Any residual is observation-reconstruction error, characterisable and possibly
  reducible, **not** an unrecoverable initial condition.
- **It does not corrupt the Chrono evaluation**, where the policy runs from a real spawn
  and builds its own hidden state naturally. The mismatch affects the fidelity of the
  training environment's initial conditions only.

**The design authorised is intact.** The branch IS "reset to a recorded state and roll
5 steps", with a 5-observation warm-up that determines the policy state exactly. My
earlier escalation over-stated the obstacle by reasoning from a mechanism rather than
checking the module.

### Smoke test result — gradient path INTACT, one residual unexplained

All four checks pass, in the order that could kill it fastest:

```
  1  GRADIENTS       14/14 parameter tensors, non-zero grads, backward OK
                     -> PATH 1: straight through the TorchScript export.
                        The rebuild-from-state-dict fallback is NOT needed.
  2  WARM-UP         5 recorded observations, policy state saturated
  3  BRANCH          5 steps in the frozen surrogate, observation rebuilt each
                     step from PREDICTED state (gravity, body rates, dof pos/vel,
                     prev_actions), joint targets fed back as surrogate actions
  4  ONE UPDATE      loss 0.053412, 14/14 params non-zero grad,
                     14/14 weight tensors changed after one Adam step
```

Surrogate frozen at 4,840,994 params; policy trainable at 460,972.

**The residual, and its resolution.** I first reported 0.284 median (~9%) at fresh branch
points against 0.120 for continuous replay, and could not account for the gap. **The gap
was my measurement error, not a property of the system.**

Control acts on ODD recorded rows (50 Hz control against 100 Hz recording). The
fresh-branch sweep sampled **even** rows, so it compared the policy's prediction against
a *stale* logged action carried over from the preceding control step. Re-measured on odd
rows:

```
  fresh 5-step warm-up   median 0.1167
  continuous replay      median 0.1167
  per-row max difference  0.000000     -- BIT-IDENTICAL
```

So warm-up **content** was not the issue either — the two paths agree exactly. The
flatness across 4/5/6/8/12 warm-up calls was also correct rather than suspicious: with a
5-slot FIFO, `nwarm=4` and `nwarm=12` end with the *same last five pushes*.

**True residual: 0.117 median, ~3.6% of action magnitude, uniform across the episode**
(0.121 over rows 301-599, 0.115 over 600-900). Its origin is still unexplained, but it is
consistent, small, and confined to the training environment.

**What that means for the design:** the training environment's policy sees observations
that differ from Chrono's by roughly 3.6% in action terms at branch start. That is a
fidelity limit of the training environment, stated rather than engineered away. It does
not touch the Chrono evaluation, where the policy builds its own state from a real spawn.

**It is characterised but not understood**, and I would rather record that than assert a
mechanism, having already been wrong once tonight by reasoning from a plausible mechanism
instead of from the module.

---

## Fine-tune ran. It shows strong signs of MODEL EXPLOITATION. Verdict NOT run.

Short-branch fine-tune completed under the declared config: 1500 updates, batch 64,
5-step (0.1 s) branches from recorded states, frozen pre-noise surrogate, verdict cell
excluded, checkpoint on a surrogate-internal metric only.

```
  val tracking MSE   0.1096 -> 0.032765     3.3x reduction
  best checkpoint    update 1500 -- THE LAST ONE
```

**Caveat 1: the run was still improving when the fixed budget cut it off.** The budget
was declared in advance and is being honoured rather than extended; the run is simply
undertrained, and that is a consequence of a budget I chose badly.

**Caveat 2, and it is the serious one — the policy's actions inflated 5x:**

| | mean \|raw action\| | max |
|---|---|---|
| baseline | 1.033 | 5.850 |
| **fine-tuned** | **5.244** | **20.180** |

The objective was velocity-tracking MSE inside the frozen model **with no action
magnitude penalty and no joint-limit constraint**. A 3.3x improvement in predicted
tracking bought with 5x larger actions is the classic signature of a policy exploiting
its model rather than learning the task. **That is a specification error in my config,
not a property of the method.**

**The physical-admissibility check was the WRONG COMPARISON, not a broken one.**
Resolved: the policy emits PD setpoints, not achieved positions, and a setpoint beyond a
joint limit is ordinary PD behaviour. The baseline has 49.8% of its setpoints outside
URDF limits and walks fine. The convention in the check was correct all along —
`act()` is `SIGN * out` with `out[CHRONO_TO_IMPORTED] = action * ACTION_SCALE +
IMPORTED_DEFAULTS`, which is what the check did. **No percentage from it means what I
implied it meant**, and the action-magnitude comparison remains the sound signal.

**THE CHRONO VERDICT HAS NOT BEEN RUN**, deliberately. Under the one-shot rule the
harness gets a single evaluation, and spending it on a checkpoint already showing a 5x
action inflation would burn the measurement on a known specification error.

**This is not "disliking the number and retrying" — no number exists.** The rejection
signal comes entirely from the policy and the surrogate side, uses no Chrono
information, and is a precondition of the same kind as the arm-B admissibility predicate.
The honest statement is that I under-specified the objective: an action-magnitude
constraint should have been declared with the rest of the config and was not.

---

## Written BEFORE the verdict was run

**The exploitation signature was identified from the policy and surrogate side alone,
before any Chrono evaluation.** Mean |raw action| 1.033 baseline against 5.244
fine-tuned, max 5.850 against 20.180, measured on recorded observations with no Chrono
information of any kind. That is what makes the corrected run below a **specification
error found on its own evidence**, not a config abandoned because its verdict was
disliked.

**HOW THE RESULT MUST BE STATED, whatever it is:**

```
  correct   "a policy fine-tuned WITHOUT an action-magnitude penalty exploits the
             surrogate and [does / does not] transfer to Chrono"
  WRONG     "fine-tuning does not work"
```

The distinction belongs in the result line itself, because the second reading is the one
that survives being quoted.

**If it transfers anyway, that is the more interesting result** and gets stated plainly
rather than explained away: it would mean 5x actions with 20 rad peaks are survivable on
this plant, which is a fact about the plant worth knowing.

## Corrected run — config declared before the first verdict exists

```
  ACTION PENALTY   loss += 0.01 * mean(a^2) on the raw policy output.
                   At baseline action scale (mean |a| 1.03) this contributes ~2% of the
                   initial tracking loss -- negligible. At the 5x inflation actually
                   observed it contributes ~50%, so it actively resists the failure
                   mode measured, rather than being a token regulariser.
  ACTION BOUND     HARD: the raw policy output is clamped to +-6.0, which is the
                   baseline policy's own observed maximum (5.850). An action outside the
                   range the baseline ever produces can never be evaluated.

                   NOT a joint-limit clamp, and the earlier draft of this config had
                   that wrong. The policy emits PD SETPOINTS, not achieved positions --
                   `act()` returns `SIGN * out` where `out[CHRONO_TO_IMPORTED] = action *
                   ACTION_SCALE + IMPORTED_DEFAULTS` -- and a setpoint outside a joint
                   limit is normal PD behaviour meaning "push toward the stop". The
                   BASELINE policy has 49.8% of its setpoints outside URDF limits and
                   walks perfectly well. So my "broken admissibility check" was not
                   broken: it was the wrong comparison, and clamping to joint limits
                   would have fought correct behaviour. Corrected before any verdict
                   existed.
  BUDGET           1500 -> 6000 updates. Justified by a SURROGATE-SIDE observation that
                   predates any Chrono result: the best checkpoint was update 1500 of
                   1500, still improving, so the first budget was simply too short.
  EVALUATION       one shot, same harness, same episodes, same exclusions.
```

Every number above is fixed while the first verdict does not exist.

---

## VERDICT, run once, on the checkpoint fine-tuned WITHOUT the original penalties

```
  eligible baseline episodes (this host only)   43
  replay check                                  5/5 physics identical
  treated arm                                   43/43 ran to completion
  surviving pairs                               0 / 43
  treated status                                fell, 43 of 43 (100%)
  median fell_at_s                              1.52 s
  harness label                                 INCOMPLETE (< 30 surviving pairs)
```

**THE RESULT, stated in the form fixed before it was run:**

> **A policy fine-tuned WITHOUT the original objective's penalties exploits the
> surrogate and does NOT transfer: 43 of 43 Chrono episodes fell, at a median of
> 1.52 s.**

**NOT** "fine-tuning does not work." The config omitted the twelve penalty terms the
policy was originally trained under, and that omission is the subject of the result.

The harness returns INCOMPLETE rather than FAIL, and that is the correct label — a
paired difference cannot be computed when no treated episode is scorable. The
substantive finding is stronger than a FAIL: not a degraded tracking error, a total loss
of locomotion.

**The exploitation signature was identified BEFORE this ran**, from the policy and
surrogate side alone — mean |raw action| 1.033 → 5.244, max 5.850 → 20.180. The Chrono
result confirms an prediction already on the record rather than discovering it.

### Throughput — what 99 seconds of fine-tuning actually bought

```
  1  batch                 64 branches per update, 5 steps per branch
  2  total env steps       480,000 train + 79,360 val = 559,360
                           (3.11 hours of simulated robot time at 50 Hz)
  3  equivalent Chrono     measured 41.25 s episode in 15.0 s wall = 137 control
                           steps/s  ->  4,079 s = 1.13 HOURS
  4  actual / speed-up     99 s  ->  41x
```

**41x, against the vehicle study's 7,404x.** The gap is structural and worth stating:
short-branch training pays a 128-step context window per 5-step branch, so the
surrogate's per-step cost is amortised over five steps instead of thousands. The
surrogate still makes this practical — 99 s against 1.13 h — but the factor is an order
of magnitude below the long-rollout case, and quoting the vehicle number for this regime
would be wrong.

**Was 1,500 updates a meaningful budget?** No — it was too small, and that is a
limitation of the run rather than of the method. The best checkpoint was update 1500 of
1500 and still improving. That observation is surrogate-side and predates this verdict,
which is what justifies raising the budget in the corrected config.

### Speed-up, with the reference terrain named — both numbers

```
  vs RIGID Chrono   137 control steps/s   1.13 h   ->    41x
  vs CRM SOIL        6.50 control steps/s 23.9 h   ->   870x
```

Both measured from collection metadata on this plant (rigid: 41.25 s episode in 15.0 s
wall, n=300; CRM: 16.00 s episode in 123 s wall, n=152). **Soil is 21x more expensive
than rigid here** — measured, not assumed; the 66x figure I was given would have given
~2,700x and overstates it by a factor of three.

**A speed-up without its baseline is not a number.** The case study is CRM soil, so 870x
is the figure for the terrain the study is about, and 41x is the figure for the terrain
this particular run used. **Quoting either without naming its reference is wrong**, and
this is the second ratio tonight that needed its denominator stated.

**One honest caveat:** this fine-tune and this surrogate are rigid-only. The 870x is
what the same 559,360 environment steps *would* cost in soil, not a soil experiment that
was run. It is the right number for the framework's claim and the wrong number to
describe this run.

## Reward terms: only 8 of 14 are computable, and the LARGEST is not

The surrogate rolls out all 34 of its state channels — verified, every `delta_*` target
present. But the 34-D state does not contain everything the original objective needs:

| computable (8) | NOT computable (6) | why |
|---|---|---|
| tracking_lin_vel | **correct_base_height (−10.0)** | no `pos_z_m` channel |
| tracking_ang_vel | lin_vel_z (0.0, inert) | no `vel_body_z_mps` |
| ang_vel_xy | torques (−1e-4) | torque not predicted |
| dof_acc | dof_power (−2e-5) | torque not predicted |
| action_rate | collision (−1.0) | thigh/calf/base contacts not recorded |
| action_smoothness | feet_regulation (−0.05) | foot vel/height not in the 34-D state |
| dof_pos_limits (−2.0) | | |
| hip_to_default (−0.05) | | |

**`correct_base_height` carries the largest converged weight (−10.0, five times
`dof_pos_limits`) and the reduced state cannot express it.** `pos_z_m` is recorded in
the CSVs; it is simply not in the `quadruped_joint_grav` preset.

**This is a concrete instance of the abstraction question the predictability diagnostic
circled.** The reduced state was chosen to model dynamics, and it cannot represent the
dominant term of the objective the policy was trained under. A policy fine-tuned without
it has no incentive to hold body height — and the observed failure was falling, which is
precisely a height and posture failure.

So the corrected run, as scoped, **optimises 8 of 14 terms and omits the largest**. That
is predictable in advance and is recorded before it runs rather than offered afterwards.

---

## v2 (upstream 8-term objective) — exploitation PERSISTS, and now we know why

```
  checkpoint     mean |raw action|    max        val reward
  BASELINE                  1.033    5.850            --
  v1 (fell 43/43)           5.244   20.180        (tracking only)
  v2 (8 terms)              4.725   11.152       best @ update 2200 of 6000
```

v2 converged properly — best at update 2200, not the last update, so the raised budget
was sufficient. **But the action inflation is barely reduced: 4.725 against v1's 5.244,
still 4.6x the baseline.**

**The mechanism is now identifiable rather than mysterious.** Of the 8 implemented terms,
none penalises action MAGNITUDE:

- `action_rate` and `action_smoothness` penalise *changes* and *second differences*, not
  size. A large but smooth action stream is free.
- `dof_pos_limits` penalises resulting *joint positions*, not commanded targets.
- **The two terms that penalise effort directly — `torques` (−1e-4) and `dof_power`
  (−2e-5) — are exactly the ones I classified as not computable.**

### Torque is computable after all, analytically

`PD_KP = 20.0, PD_KD = 0.5`, so `tau = Kp(target − q) − Kd·q̇` — a deterministic function
of the action and the joint state the surrogate already predicts. **No state extension is
needed for the effort terms.** Verified against the recorded `joint_*_torque_nm` over 600
frames:

```
  max|diff| median 0.507 Nm against a recorded |tau| of ~11.7 Nm   ->  4.34% relative
```

**Not exact** — likely torque is sampled at a different point in the substep loop, or is
clamped — and that 4.34% is recorded as a stated approximation rather than smoothed over.
For a penalty whose job is to resist large effort, a 4% error in magnitude is immaterial.

**Scale check, so the fix is not assumed to work:** at baseline effort, `torques`
contributes about 0.17 against a tracking maximum of 1.5 (~11%). At v2's 4.6x inflated
actions the target error scales with it, so `sum(tau²)` rises ~21x and the penalty
reaches ~3.6 — dominating the objective. **This term would actively resist the observed
failure**, which is what `action_rate` and `dof_pos_limits` demonstrably do not.

### My "not computable" classification was wrong on two of six terms

`torques` and `dof_power` were called uncomputable because torque is not a surrogate
state channel. It does not need to be — it is a function of quantities that are. The
classification asked "is this a channel?" when it should have asked "is this a function
of the channels?" That error cost a full training run and its diagnosis.

### PREDICTION for v2, recorded before v3 exists and before v2 is ever evaluated

**v2 will fall in Chrono, at a rate close to v1's 43/43.**

The predictor is measured, not theorised: mean |raw action| is **4.725** for v2 against
**5.244** for v1, on the same episodes by the same procedure, and v1 fell 43 of 43 at a
median of 1.52 s. Same quantity, nearly the same value.

**v2's evaluation is DEFERRED, not cancelled, and the checkpoint is kept.** If v3
succeeds, v2 becomes genuinely informative — the middle point establishing that action
magnitude *predicts* transfer, rather than that one extreme fails. It would then be run
on the same harness as its own single evaluation.

**The risk being accepted, stated plainly:** *"we expect it to fail, so we will not
measure it"* is a slippery argument. What makes it safe here is that the predictor is a
measured quantity rather than a theory, and that the checkpoint survives. **If v3's
signature also comes back high, its verdict is NOT skipped on the same reasoning** — two
skips in a row is a pattern rather than an instance.

### The two tracks answer different questions and neither is the other's fallback

```
  v3 (torques + dof_power)   attacks ACTION INFLATION.  Needs no state extension.
  36-D (pos_z, vel_body_z)   enables correct_base_height at -10.0, a POSTURE term.
```

`pos_z` was never the fix for action inflation, and torque was never a reason to extend
the state. Both are worth having, for unrelated reasons.

## v3 (10 terms, effort added) — exploitation FIXED, and it still fails

**The diagnosis was right and the fix worked**, on the quantity it targeted:

```
  checkpoint      mean|a|   max|a|   mean|Δa|   mean|target−q|   mean|τ|
  BASELINE          1.033    5.850      0.312           0.2218     4.184
  v1 tracking       5.244   20.180      --                  --        --
  v2 8-term         4.725   11.152      0.203           1.1360    22.764
  v3 10-term        1.239    3.033      0.150           0.2308     4.829
```

v3's max action (3.033) is **below the baseline's** (5.850). It is smoother than
baseline, its PD tracking error matches baseline, its torque matches baseline. **On every
open-loop statistic it is baseline-like.**

### VERDICT on v3, run once: 0 of 43 surviving pairs

But the failure MODE changed:

```
  v1   43/43 status 'fell',      full-length episodes (3884 rows), fell at 1.52 s
  v3   29/43 recorded, all 'diverged', median 133 rows (~1.33 s); 14 diverged
       before the collector's 50-row minimum and were discarded outright
```

**v1 fell over. v3 destabilises the solver.**

### What this establishes, and it is a finding about the method

**The exploitation signature was a valid screen for v1 and v2 and it has now reached its
limit.** v3 passes every open-loop check and fails closed-loop anyway, so the remaining
failure is not action-magnitude exploitation.

The most likely cause is structural rather than a further missing reward term: **branches
are 5 steps long, so nothing in the training objective can see an instability that
develops over more than 0.1 s.** The surrogate is certified over exactly that window —
that is why the branch length was chosen — and a policy optimised inside it is optimised
for 5-step behaviour. Chrono runs it for thousands of steps.

This is precisely what the gate's own scope note said and it is worth quoting against
itself: *"OPEN-LOOP. A PASS here is necessary but NOT sufficient for closed-loop
training, and a later closed-loop failure does not contradict it."* That was written
before any fine-tune existed. **The closed-loop failure it anticipated is the one now
measured.**

**What this does NOT establish:** that fine-tuning inside a surrogate cannot work. It
establishes that *short-branch* fine-tuning over a surrogate valid for 0.1 s does not
produce a policy that survives closed-loop Chrono, with three distinct objectives tried
and the action-magnitude failure mode eliminated.

## v2's deferred verdict, and what the three runs together show

v2 evaluated on the same harness, one shot: **0 of 43 surviving pairs.**

```
  ckpt   ||dW||   ||dW||/||W||   mean|a|   Chrono mode   rows survived
  v1      8.898        0.62%       5.244   FELL                  3884
  v2     12.392        0.86%       4.725   diverged               307
  v3     19.287        1.33%       1.239   diverged               133
```

**The discriminator resolves against magnitude.** v2's action magnitude is nearly v1's
(4.725 against 5.244) but it fails like v3, not like v1. So the failure MODE is not
driven by action magnitude, and the v1/v3 difference is something else.

**That something else appears to be weight-space displacement.** The three runs are
monotonic in `||dW||` and monotonically worse in Chrono survival — and **v3 moved
furthest while having the best action statistics of the three.** Better open-loop
behaviour, worse closed-loop survival.

The reading this supports: optimising inside a surrogate certified only over 0.1 s moves
the policy off the manifold where it works, and the damage scales with how far it moves,
largely independent of what objective drove the movement. Gradients through a model valid
for 0.1 s are trustworthy only for very small steps; beyond that you are optimising model
error.

**Stated with the caution it deserves:** three points, and displacement is confounded
with objective, since each run used a different reward. The ordering is clean and the
mechanism is plausible, but this is a hypothesis with a suggestive fit, not an
established law. **It makes a testable prediction** — a fine-tune constrained to a much
smaller `||dW||` should survive longer — and v1, the smallest displacement, is the only
one whose episodes ran to full length rather than being cut short by divergence.

**This supersedes "the objective was wrong" as the leading explanation.** Three
objectives were tried, spanning tracking-only to 10 of 14 upstream terms with the
action-magnitude failure eliminated, and all three failed with severity tracking
displacement rather than objective quality.

## Displacement test — declared before the checkpoint exists

The three runs confounded displacement with objective: each used a different reward AND
ended at a different `||dW||`. **The clean test is the same objective at a different
displacement.**

```
  TAKE          v3's objective, unchanged (10 upstream terms)
  STOP AT       ||dW|| = 8.9, matching v1's 8.898
  EVALUATE      one shot, same harness, same 43 host-local pairs

  PREDICTION, recorded now: if displacement drives the failure, this checkpoint
  survives materially better than v3's 133 rows -- toward v1's 3884, which is the
  survival v1 achieved at this same displacement under a WORSE objective.

  If it fails like v3 anyway, displacement was a coincidence of three points and the
  account is withdrawn.
```

v3's run wrote no intermediate checkpoints — only `best.pt` — so this is a re-run of the
same objective with displacement-based stopping rather than metric-based. The objective
is known-good; only the stopping point changes.

**Why this is a better use of the next evaluation than the half-noise route:** it tests
the leading hypothesis directly, rather than testing a remedy for it. And it separates
two readings that are otherwise the same claim measured differently — *"longer horizon
permits larger trustworthy steps"* against *"smaller steps stay inside a short-horizon
model's validity"*. **If a small-step v3 survives, horizon was never the constraint.**

### Displacement test result — the account is CONFIRMED

v4: v3's objective exactly, stopped at `||dW|| = 8.903` to match v1's 8.898.

```
  ckpt   ||dW||   mean|a|   recorded   status                              rows median   PAIRS
  v1      8.898     5.244      43/43   fell 43                                   3884       0
  v2     12.392     4.725      43/43   diverged 43                                307       0
  v3     19.287     1.239      29/43   diverged 29                                133       0
  v4      8.903     1.188      42/43   COMPLETED 17, fell 4, diverged 21         2126      20
```

**Same objective as v3. Same action statistics as v3. Only the step size differs — and
surviving pairs went from 0 to 20, with 17 episodes completing outright.**

The prediction recorded before the checkpoint existed was that v4 would survive
materially better than v3's 133 rows, toward v1's 3884. Measured: median 2126 rows.
**Confirmed.**

**What this establishes.** The failure was **over-optimisation inside a locally-valid
model**, not the objective and not the method. Three objectives all failed at large
displacement; the best objective at constrained displacement produces a policy that walks
in Chrono on 17 of 43 episodes. Gradients through a surrogate certified over 0.1 s are
usable — for small enough steps.

**It also resolves the two readings.** *"Longer horizon permits larger trustworthy
steps"* and *"smaller steps stay inside a short-horizon model's validity"* were the same
claim measured differently. **A small-step fine-tune survives on the existing 0.1 s
surrogate, so horizon was never the binding constraint — step size was.**

**The verdict is still INCOMPLETE**, at 20 surviving pairs against a declared minimum of
30, so no paired difference is computed and no claim is made about whether tracking
improved. **That minimum is not negotiable and is not being renegotiated.** What is
established is qualitative and large: the policy survives, where three previous
checkpoints did not.

Reaching 30 needs either more eligible host-local episodes — 43 is all this machine has
in the cell — or a policy that survives a higher fraction. sbel-pc's 37 would clear it
combined, run on its own machine per the stratification rule.

---

## Amended stopping rule for the half-noise run, and a standard applied to my own claims

**The rule as declared was unsound on one branch.** It read: *"half-noise passes corr at
0.5 s → proceed; it does not → the trade-off is intrinsic, write up."*

**One noise magnitude failing to separate stability from sensitivity is evidence about
that magnitude, not proof the trade-off cannot be broken by any means.** Amended:

```
  passes  -> proceed as before
  fails   -> "noise magnitude ALONE does not separate stability from sensitivity in
             the range tested." NOT "the trade-off is intrinsic."
```

That leaves open the variables nobody has varied — multi-step loss and contact
conditioning among them.

**The general error:** establishing that X does not cause a residual *removes X*. It is
not evidence the residual is irreducible. And once "floor" is written down it stops the
looking — elsewhere in this project an "11.9% irreducible solver-divergence floor" was
quoted four times before anyone tested a second variable, and tilt turned out to move it
to 4.2%.

**Checked against my own documents.** The 11.9% figure was never adopted here. The word
"floor" appears three times, all referring to the *predictability regression's*
detectability limit — *"this design cannot distinguish a true R² below ~0.07 from zero"*
— which is a statement about the design's resolution with the reasoning given, not a
claim that nature has a floor there. That scoping is the distinction and it holds.
