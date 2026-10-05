# One architecture for the core, collision and contact networks (version 5): plan

Written 2026-10-04 before any version-5 result. Code: `src/nedm/contact_nrd/model_v5.py`, `train_v5.py`.
Configs: `configs/contact_nrd/v5/`. Runs: `runs_v5/` on AMD.

## Why (the user's request, 2026-10-04)

- One fixed design and one configuration for the core, the collision network and the contact network. Ball and
  pool use the same file. Only the data changes.
- Inputs: the state and its history only, plus learned identity codes for the pairs and the fixed bodies.
- The user expects a Transformer that uses history and attention correctly to do better than a history-1 MLP,
  and suspects a defect in how the context window and attention are used.

## What the audit found (workflow `contact-nrd-arch-audit`, findings checked by a second agent)

| Finding | Status |
|---|---|
| History order, padding, causal mask, readout: no wiring bug | confirmed |
| The core always had context 1: attention over one token does nothing; the "Transformer core" is a residual MLP | confirmed |
| History networks trained only on true histories; in rollout their own small errors enter the history (pool median at 2 s: 8.92 mm own history, 1.34 mm true differences) | confirmed (mechanism partly) |
| A per-group linear path read all history columns and carried most of that damage | confirmed |
| The pool history-8 Transformer score 0.18 is from a run cancelled at 30k of 160k updates; Transformer runs had 3.3x lower learning rate and no rollout refinement | confirmed |
| Static padding at episode start (copies of the first state) is far outside the training data | confirmed |
| Per-system hand-set scales and settings | confirmed |
| The simulated physics is memoryless in the full state; the pool state hides z and vz | plausible / confirmed |

## Version-5 design (identical for every system)

- Tokens: current state x(t) and step differences (separate input maps), learned lag, role (self, partner,
  other), pair and fixed-body codes. Missing history is masked, never filled.
- Mixers (the only change between arms): `mlp` (control), `time`, `entity` (history 2), `spacetime`.
- Width 128, 4 heads, feed-forward 512, 4 blocks, float32 trunk, float64 state and linear paths.
- Output: zero-initialised head + per-query ridge linear path on the current states; scales from data rules.
- Training: core (clean, then noise + history dropout); collision (only the sampled pair is evaluated);
  contact (noise, history dropout, pushforward of 1-3 own steps on half the samples after 25 % of training);
  rollout refinement for every arm; no futility rule. Noise sigma = RMS of the core's own one-step error.
- Loss scales: contact kappa = [1e-2, 5e-4, 3e-3] x RMS of the contact change (position, velocity, spin); rollout
  4x that; core in units of its output scale.

## Stages and decision rules

1. **Screen** (short schedule: core 15k + 15k, collision 10k, contact 40k, refinement 300 + 300): four mixers x two
   systems, and REF = the old best state-only recipe (history-1 MLP, frozen v2 core) on the same short schedule.
   Ratio = validation score / REF score for each system. Drop an arm if the geometric mean of the two ratios is
   above 1.25 or either ratio is above 1.6.
2. **Ablations on the best Transformer**: noise scale (0 and 3), history 1, a 20 ms model step (the user accepts
   50 Hz), seed 62. One noise value is then fixed for both systems.
3. **Full runs**: best two designs x two systems x seeds 61, 62, 63.
4. **Fresh test and targeting**: the validation pick and the 3-seed ensemble on new shots, paired bootstrap
   against the earlier models; targeting in Chrono.

## Milestone

One configuration file with no system-specific keys that, on both systems:
- pool: validation-picked seed scores 0.0171 or less, and the 3-seed median is 0.0197 or less;
- ball: validation-picked seed has fresh path p95 0.93 mm or less, with 100 % events right;
- hygiene: all rollouts finite; own-history vs true-history gap 1.2x or less at the 2-s median.

Stretch goal: pool validation 0.011 or less (1.25x the pair-feature model).

## Note

A separate study by another session runs at the same time on the same cluster
(`contact_nrd_codex_unified_20261004`). It is not touched by this work. This work uses the 8-GPU node.

## Amendment 1 (2026-10-04, 16:30 CDT, during the screen)

The first screen run showed the ball contact error growing after own-history training started (spacetime ball
path p95 1.53 mm at 10k updates, 2.93 mm at 20k). Cause: during the 1-3 own steps the model used its own
switches; a bounce one step early or late gave a current state far from the truth, paired with the true next
state. Fix, applied to every arm before any result was read: the own steps use the true switches, and an own
history is used only if its current state is within 3 loss scales of the truth (`push_max_dev`). The contact
stage was restarted from the trained cores and collision networks; a no-own-history control (spacetime) was
added. The MLP-control ball run had already finished its contact stage with the old rule; it is rerun in stage 1c.

## Screen result (validation; short schedule; after refinement)

| Arm | Pool score | Ball score | Ratio to REF (pool / ball) |
|---|---|---|---|
| REF (old recipe, history-1 MLP, frozen v2 core) | 0.0443 | 0.00161 | 1 / 1 |
| spacetime, history 4 | 0.0134 | 0.00177 | 0.30 / 1.10 |
| spacetime, history 4, no own-history training | 0.0154 | 0.00177 | 0.35 / 1.10 |
| time, history 4 | 0.0185 | 0.00169 | 0.42 / 1.05 |
| entity, history 2 | 0.0215 | 0.00325 | 0.49 / 2.02 (dropped: ball ratio > 1.6) |
| mlp, history 4 | 0.0282 | rerun | 0.64 / - (dropped on pool rank and the old ball result) |
