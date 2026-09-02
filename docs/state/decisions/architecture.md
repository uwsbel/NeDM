# Standing architectural decisions

## `z2` is appended to `z1`, never substituted

**Status:** standing · The core commitment of the vision line.

`z2 = E_φ(x)`, `z = concat(z1, z2)`, and the model predicts both `ẑ1` and `ẑ2`.
`z1` must first be a sufficient Markov state for the modeled mechanics under
fixed system parameters; `z2` then carries the high-dimensional sensor
representation and its evolution. **Rejected:** replacing the physical state with
a learned latent (the world-model default). **Why:** the paper's whole claim is
about choosing an explicit reduced state; a pure latent gives that up.

## The decoder is off during policy rollouts

Decoding costs an order of magnitude (Study 1: 293 k → 22 k transitions/s) and
buys nothing a policy needs. It exists for diagnostics and cross-modal
consistency checks. **Consequence for benchmarking:** never present a
decoder-off NRD against an always-rendering simulator as the headline number —
Study 3 §12.4 mandates a three-row like-for-like protocol.

## No reward, success, or task loss in the core NRD objective

`L_NRD = λ1·L_z1 + λ2·L_z2 + λ3·L_recon + λ4·L_rollout + λ5·L_constraints`.
Task-specific reward belongs to the downstream RL or planning phase. If imagined
RL needs a learned reward estimator, train it afterward as a separate task
adapter. **Why:** keeps one dynamics model reusable across downstream tasks;
prevents the surrogate from becoming task-specialized.

Note the deliberate boundary: Study 3's auxiliary warm-up heads (occupancy,
vehicle heatmap, elevation) are **representation-shaping losses from analytic
ground truth**, not task losses, and are consistent with this rule.

## Checkpoints are selected on open-loop rollout error

`checkpoint_metric: rollout_sel` everywhere. The file is still named
`best_val.pt` but is the rollout-selected epoch. The two metrics rank
checkpoints differently. **Why:** one-step loss does not predict rollout
stability, which is what a policy actually consumes.

## Study 3 is hierarchical, not end-to-end

Planner (from `z2`, once per episode) + tracker (from partial `z1`, 20 Hz),
rather than one end-to-end RL agent. **Why:** it confines learned-model rollouts
to short validated horizons and replaces the main exploitation surface —
long-horizon RL against model error — with supervised imitation of a privileged
oracle. The end-to-end agent survives only as a clearly-labeled bracket run
after G7, either matched-budget or explicitly called a low-budget bracket, never
an unlabeled strawman.

## Study 3 v1 permits privileged information, but declares it

Obstacle layout and the dynamics model's localization are **vision-only**; the
planner's start pose and the goal are **privileged**, with a stated upgrade
ladder. **Why:** v1 is a feasibility test of the full stack; purity upgrades are
a ladder, not a gate. The honesty requirement is that the contract table in §1
is published with the result.

## Study 3 uses one fixed authored terrain map

**Consequence:** RQ2 is a **localization** claim on a known map, not a general
terrain-from-depth claim. The plan was reworded to say so and carries an
ablation grid plus a privileged-`(x,y,ψ)` upper bound to bound it. **What would
reopen it:** multi-map generalization, which is explicitly deferred.
