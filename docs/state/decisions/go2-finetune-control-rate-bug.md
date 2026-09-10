# The fine-tune ran the policy at 100 Hz when control is 50 Hz

**2026-09-10.** Found by an open-loop audit of the surrogate, after two days of CRM
results that looked like a scientific finding and were very likely an instrument fault.

## The defect

The corpus records at 100 Hz (`record_step_s` 0.01); the controller runs at **50 Hz**.
The fine-tune's warm-up already honoured that -- it indexes `S_all[e][b - 2*k]`, stride
2 -- and `sample()` deliberately picks an odd `b` with the comment *"control acts on ODD
rows"*.

**The branch loop did not.** It called the policy once per SURROGATE step, i.e. once per
100 Hz record row. `BatchedGo2Policy` carries a 5-slot observation buffer and the branch
is 5 steps, so those five calls **completely flushed it** with double-rate samples. Every
gradient was taken through a controller that does not exist at deployment, and the
error scales with how far the optimiser moves the actor.

Two consequences beyond the gradient itself:

- **The branch was 0.05 s, not the 0.1 s** the docstring claimed and the
  action-sensitivity gate certified. The gate converts horizons as `int(round(h/dt))`
  with dt = 0.01, so its "0.1 s" was 10 surrogate steps -- twice the branch.
- **An off-by-one in the model input.** `hs` was not extended before slicing while the
  action tensor was, so the action window dropped its oldest entry and every context
  token but the last was paired with the action one step later. Measured cost in forward
  accuracy alone: **+6-9% velocity RMS, +21-25% on pos_z**, 8 of 9 paired bootstrap
  intervals excluding 1.0. `preprocess.py` and the gate's `roll()` both pair
  index-for-index, so **the fine-tune was optimising a different instrument than the one
  that was certified.**

## Why the controls could not catch it

Three displacement-matched random perturbations were flat against base on both rigid
cells and on CRM (0.1576, 0.1629 against base 0.1632). That control is what licensed
every treated number -- and it is **structurally blind to this bug**, because a random
weight perturbation never enters the branch loop. The signature we could not explain --
controls flat, every gradient-trained arm 4-6x worse with achieved velocity driven to
zero -- is exactly what this defect predicts.

**A control that cannot fail in the same way as the treatment does not validate the
treatment's machinery.** It validated the policy weights and the scorer; it never
touched the optimiser's inner loop.

## What the audit also established

Measured on 3,520 open-loop windows from 147 val episodes, paired across variants:

- **The surrogate is genuinely good at the branch horizon**: 2.9-3.3x better than
  persistence on velocity and 18-20x on `pos_z` at h=5, non-overlapping intervals.
  Error grows LINEARLY out to 25 steps -- integrated unbiased one-step error, not
  compounding instability. Parity with persistence arrives at ~0.5 s for velocity.
  **The model was never the problem at the horizon being optimised.**
- **The terrain channels buy nothing.** The 52-D terrain surrogate is slightly WORSE
  than the 36-D baseline at every horizon on every channel, and `terrain_s1` scored
  0.8551 in Chrono -- no better than the arms without it. Sinkage, slip and surface
  displacement are real signals in the data and do not help this predictor.
- **Ensemble disagreement DOES track error** (Spearman +0.61 on velocity, +0.87 on the
  all-channel quantity the penalty uses). Pessimism failed not because the signal is
  absent but because it is **3-10% of the objective at `--pessimism 1.0`** and the
  3-member spread **understates true error by 2.7x**.

## Status of everything measured before this

**The CRM arm results are not a finding about deformable terrain.** They were computed
through this branch. The same code produced the rigid results, so those need re-checking
too -- including the conclusion that fine-tuning "trades" capability, and the D-C
contrast that reached p=0.0286.

The corrected fine-tune is re-running. Whether the bug was the cause is now an empirical
question with a clean reference: base 0.1632, controls flat, broken arms 0.68-1.07.
