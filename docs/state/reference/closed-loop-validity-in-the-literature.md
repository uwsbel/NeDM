# What NeRD, DHAL and HALO certify, and what they then optimise

Written 2026-09-07 from the three specs in `papers/`, each extracted with its released
code. Tags follow those specs: `[P]` paper, `[C]` code, `[M]` measured off a figure.

The question: **does any of them certify CLOSED-LOOP validity — a policy feeding its own
actions back into the learned model — as opposed to one-step or teacher-forced error?**

## Summary

| | trained on | validity reported | optimises a policy inside the model? |
|---|---|---|---|
| **NeRD** | one-step, teacher-forced, horizon = 1 `[C]` | open-loop state error: **Cartpole and Ant only**; ANYmal has none at any horizon | **yes**, PPO entirely inside, up to 1000 steps |
| **DHAL** | one-step prediction `[P]` | one-step only; never rolled out | no — it learns a policy, not a world model |
| **HALO** | **multi-step, 5–7 steps, full BPTT** `[C]` | multi-step rollout to 6 propagated Poincaré steps | no — `g_ρ` models a *fixed* closed-loop map |

**No paper certifies what we need.** NeRD optimises closed-loop over 1000 steps while
reporting no open-loop state error for its quadruped at any horizon. HALO trains and
reports multi-step, but its model has no control input at all — the policy is baked in, so
there is no loop to close. DHAL never rolls out.

## NeRD — the gap is real, and its stability has an architectural explanation

Training is horizon 1 `[C]`: the transformer consumes a length-10 window of **ground-truth**
states and emits 10 independent one-step predictions; `loss` averages them. "The model is
never unrolled during training and no gradient ever flows through a predicted state. No
curriculum. No scheduled sampling. No DAgger."

It then trains PPO **entirely inside the model**, ANYmal at horizon 1000 `[P]`.

So NeRD has exactly our gap — certify one-step, optimise closed-loop — and it works. Why it
works is the transferable part, and it is not network accuracy:

1. **An analytic simulator closes the loop every step.** The predicted state is written back
   into Warp, `eval_fk` recomputes maximal coordinates, and the *analytic* collision detector
   recomputes contact `(p₁, n⃗, d)` from that state. Over-predicted sinkage produces corrected
   contact depths that push back. The spec: *"a physical feedback path that a pure
   autoregressive world model does not have, and it is the main reason a one-step-trained
   network survives 1000 autoregressive steps. The paper never frames it this way."*
2. **Base-frame re-anchoring resets the input distribution every step.** With
   `anchor_frame_step: every` the base pose input is identically (0, I) at every timestep, so
   *"the network's input distribution at step 1000 is statistically identical to step 1 —
   drift cannot leave the training manifold by translation, only by dynamics error."*

**Our surrogate has neither.** No analytic re-query of contact from the predicted state, no
per-step re-anchoring. NeRD's 1000-step survival is therefore not evidence that a purely
learned model survives closed-loop rollout; it is evidence that a learned model *wrapped in
an analytic corrector* does.

And NeRD's own quadruped evidence is weaker than it appears: the only ANYmal number is
closed-loop **reward** agreement (−0.02 %, −0.07 %) on a reward built from saturating
exponentials, where a trained policy sits at the peak and the metric is first-order
insensitive to velocity error. The one non-saturating reward in the table (Ant Spinning,
`R = ω_y + p_up`) disagrees by **+17.21 %**.

## HALO — trains multi-step, but there is no loop to close

`L_pred` is genuine BPTT through 5 (paddle-ball) or 7 (hopper, G1) compositions of the latent
map, decoded at every intermediate step, no `stop_gradient` `[C]`. The rollout test reaches
**6 propagated Poincaré steps** ≈ 3.6 s of G1 walking `[M]` — and there is no results table
anywhere in the paper, only curves.

But: *"No control input. The RL policy is baked in; `g_ρ` models the closed-loop map only."*
HALO learns the autonomous dynamics of an already-closed loop. Nothing feeds actions back,
so it cannot answer whether a *policy being optimised* destabilises the model. Its multi-step
training is still the one directly copyable idea here.

## DHAL — one-step only, and not a world model

DHAL learns a locomotion policy with hard one-hot mode routing. Its prediction error is
reported one-step only; the spec flags rollout-horizon error as a gap: *"a hard gate that
flickers can compound badly over a multi-step autoregressive rollout — a regime DHAL never
tests."*

## What this means for us

1. **The open-loop / closed-loop conflation is not ours alone.** NeRD certifies one-step and
   optimises at horizon 1000. Naming the gap is a genuine contribution, and it is
   *sharper* than "nobody measures this" — NeRD measures the wrong thing convincingly, on a
   reward metric that cannot detect the failure.
2. **The fix NeRD used is architectural, not statistical.** If closed-loop validity is the
   binding constraint, more data or better one-step accuracy is the wrong lever. An analytic
   contact re-query and per-step re-anchoring are what bought NeRD its horizon.
3. **HALO supplies the training recipe** — multi-step BPTT through 5–7 compositions — and
   its horizon is short, which is consistent with short-horizon closed-loop validity being
   the norm rather than a failure specific to us.
4. **A closed-loop divergence-versus-horizon curve appears in none of the three.** If ours is
   ~10 steps, that is not obviously worse than what these papers demonstrate; it is the
   number none of them report.
