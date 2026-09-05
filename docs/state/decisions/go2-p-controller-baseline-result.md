# The proportional baseline beats the learned policy on both terrains

**A hand-tuned proportional controller matches or beats the learned policy
everywhere measured, on tracking error and on command smoothness.**

    terrain   policy   P ctrl   P better on   pooled    P action rate
    rigid     0.0637   0.0165   8 of 8        3.9x      0.49x (smoother)
    soil      0.0494   0.0297   6 of 8        1.66x     0.35x (smoother)

Same references, same 6 s horizon, same action bounds, same floor methodology,
same Chrono build. `cmd = clip(K * local pose error)` — no learning, no
surrogate, no plant model.

## The asymmetries run against the baseline and it wins anyway

- It is tuned on references **held out** from the evaluation; the policy trained
  on all forty including the eight it is scored on.
- Its rigid gains sit at a **grid boundary and were still improving** when the
  search stopped, so the rigid margin is a lower bound.
- Its soil search was restricted to **isotropic gains over 24 rollouts**, against
  64 combinations on rigid, because CRM costs ~1.6 min per rollout.

## What survives is a narrowing, not a reversal

The margin falls from 3.9x on rigid to 1.66x on soil. **The learned policy's
relative position does improve where contact is harder — it just does not improve
enough to win.** That is the defensible claim.

## WITHDRAWN: "the framework earns its cost where the contact model is hard"

An earlier version of this result reported the advantage *reversing* on soil, with
the policy winning 1.18x. That was measured with **P gains tuned on rigid
references**. Tuning on soil gives an interior optimum at k=16 — unlike rigid,
where the search ran to the boundary — and k=32, the value used in the published
comparison, is **3.3x worse** than k=16 on the held-out soil set.

The reversal was a tuning artifact. It was published to the status page and has
been withdrawn rather than annotated.

**The shape of the error is worth keeping.** A boundary optimum was caught on
rigid and the grid extended twice; the same check was simply not applied to soil
until prompted. *A boundary optimum is a visible defect. A wrong-terrain optimum
looks like a number.*

## What this does and does not say about the framework

It does **not** say the surrogate is inaccurate: the transfer gap is separate and
measured (4.8x pooled on rigid, 2.4x on soil).

It says **this task is too easy to demonstrate the framework's value.** Both
controllers sit on top of the same imported walking policy, so this compares two
outer loops over an identical inner loop. The inner loop does locomotion, balance
and gait; the outer loop is a 3-DOF velocity correction, and three gains solve
that. Kyle's layering objection is correct, and these numbers are the evidence
for it.

A fair test of the framework would be one where the outer loop's job is not three
gains: direct joint control, or terrain the imported walking controller cannot
handle on its own.

## The cost claim is unaffected and stands on its own

    surrogate   262,144,000 policy steps in 4.331 h = RTF 841x
    Chrono CRM  RTF 0.1136 (total sim / total wall over 152 episodes)
    the same work in Chrono: 1,336 days = 3.66 years single-stream
    cost ratio  7,404x

The framework's claim is comparable tracking at a cost where the Chrono-trained
alternative is not runnable. The first half of that is what these numbers put in
question, on this task.
