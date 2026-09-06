# Training policies inside a learned model

From `docs/vision/double_pen/rl_implementation_notes.md`. These are about the
*interface* between RL and a frozen surrogate, and generalize past the pendulum.

## Negative-everywhere rewards make termination an exploit

**Cost:** two killed training runs · **Found:** 2026-08-26

**Expected:** a spin/instability guard protects the model's validity domain.
**Happened:** by iteration 100 the policy had a spin-termination rate of 0.99,
mean episode length 6 steps, 93% saturated actions. **Cause:** the reward was
negative almost everywhere (≈ −5/s), and terminating cost nothing — so
"saturate the actuator and trip the guard" returned ≈ −24 versus ≈ −150 for
surviving the episode. The exploit is about the **horizon**, not the actions, so
action-magnitude and action-rate penalties would not have closed it.

**Two fixes, both work:**

1. Charge the failure its remaining distance cost:
   `w_d · (d_fail/L) · (remaining transitions)` — makes early termination
   value-equivalent to standing still, and changes no other transition's reward.
2. Better: make all reward terms **non-negative** (exponential reach reward,
   no failure penalty at all). This is what the arm study did and what finally
   worked — 85% plateau versus 13–17% under the shaped reward.

## Keep goals inside the data-covered region

The plan's `θ ~ U(0, 2π)` goal distribution put ~50% of goals where the NRD had
~1% of its training data (upper half: 65–100 rows per 10° bin versus 1000–7700
for the lower half). Success there was **0%** and stayed 0%. Restricting to the
lower half is what let the run get off the ground.

Check goal/reference coverage against the actual training distribution *before*
training, and log success split by region so a ceiling is visible rather than
mysterious.

## Reset from recorded context windows, and fingerprint the bank

Episodes reset from a bank of recorded 16-step `[z1, z2, a]` windows, encoded
once by the frozen encoder. A bank is **tied to the checkpoint that encoded it**
— the env refuses a bank whose `z2_mean` fingerprint differs from the model's.
Keep that check; a silently mismatched bank is very hard to diagnose.

## Tolerance can sit below the model's own error

One-hold (0.1 s) NRD tip error was RMSE 6.9 mm / p90 9.6 mm against a **1 cm**
success tolerance — the same magnitude. And the tip moved a median 34 mm per
20 ms step, so a 2 cm disc is easily stepped over by a pointwise 50 Hz check.
**Report success-vs-tolerance curves from the closest approach**, in both
simulators, so the sensitivity to an arbitrary threshold is explicit rather than
hidden.

## Confine imagined rollouts to the validated horizon

Study 3's tracker trains on randomized **1–3 s fragments** initialized from real
context windows, because that is where open-loop fidelity was validated;
continuous long-horizon tracking is evaluated in Chrono only. Hierarchy alone
does not do this for you — it has to be designed in.

## Frame-anchoring a drifted state is worse than full autonomy

Feeding *true* latents alongside a *drifted* `z1` gave 161.9 mm at 3 s versus
24.0 mm fully autonomous. The pairing is an input combination the model never saw
in training. Do not "help" a rollout with partially real inputs.

## A stateful policy is part of the dynamical system, and the reduced state does not cover it

**Cost:** every trajectory-matched comparison in the Go2 fine-tune line · **Found:** 2026-09-05

The framework's design rule (`§4.2`) says the reduced state retains the coordinates the
**transition** depends on. That is written for the *plant*. When the controller carries
memory, the system being rolled out is `(plant state, controller state)`, and the second
half is neither modelled nor validated.

`go2_cts_150k` carries a five-step observation history and a 32-dim student-encoder
latent. Fed **true** states from Chrono, with a perfect view of the world, it still drifts
off its own recorded behaviour once it accumulates its own actions — measurably **worse
than predicting the mean action**.

**Consequence:** trajectory reproduction is not an available success metric for a stateful
controller. A perfect dynamics model would fail it. Any before/after comparison that
scores "does the rollout match the recording" is measuring the controller's own
divergence, and the surrogate's contribution is not separable from it.

**Two ways out, in order of preference:**

1. **Score task outcomes, not trajectories** — completion, tracking error against the
   command, terminal state distribution. These are well defined under a divergent but
   correct rollout.
2. **Close the state over the controller** — include the controller's memory in `z`, or
   use a memoryless controller. The HMMWV policy is effectively memoryless, which is why
   this never surfaced in that case study.

**Check before designing the comparison:** roll the controller in the *high-fidelity*
simulator from a recorded initial condition and measure divergence from the recording. If
it diverges there, no surrogate result can be read against the recording.
