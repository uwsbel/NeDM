# Contact NRD: one learned-simulator design for systems with switching contacts

This folder documents the contact NRD (neural reduced-order dynamics) studies of 2026-10-02 to 2026-10-04.
Detailed records: `design.md`, `results.md` (pair-feature design), `state_only_*.md`, `minimal_state_*.md`
(state-only inputs), `unified_v5_plan.md` and `unified_v5_results.md` (the converged design). Code:
`src/nedm/contact_nrd/`, with the system collectors and evaluators in `src/nedm/bouncing_ball/` and
`src/nedm/pool_ball/`. Configs: `configs/contact_nrd/` (`v5/` is the converged design). Cluster scripts:
`scripts/contact_nrd/cluster/`.

## Two kinds of systems, one family of models

An NRD predicts the next state from the current state (and history) and the action:
`s(t + dt) = s(t) + core(...) [+ contact terms]`.

| System type | Contact behaviour | Networks | Studies |
|---|---|---|---|
| Vehicles on terrain, robot arms, double pendulum | No switching: the contact mode does not change, or the contact is always there | **Core only**: state history + action -> state change | HMMWV traversing, tracked vehicle, arm reaching (earlier studies) |
| Bouncing ball, pool, an arm that pushes an object | Contacts switch on and off; each impact puts a jump into one step | **Core + collision network + contact network** | Ball, pool (this folder); SO101 arm pushes a T (next) |

The earlier studies are therefore the special case of this design with no collision and no contact network.
A contact that is always present (a vehicle on the ground, a pool ball on the cloth) belongs to the core.

## The converged design (version 5)

```
for each moving body i:     d_core_i   = core(history of body i)
for each pair (i, j):       g_ij       = 1[ collision(pair tokens) >= 0 ]            (on / off)
for each pair that is on:   d_ij, d_ji = contact(pair tokens)                         (state changes)
next state = state + d_core + sum over pairs of g_ij * d_ij
```

| Part | Value (identical for every system; only the data changes) |
|---|---|
| Inputs | The last 4 states of every moving body (the current state and three step differences) plus learned codes for the pair, the fixed partner body, the role (self, partner, other) and the time lag. No geometry, gravity, radii or relative features. |
| Network (core, collision and contact) | Transformer encoder over these tokens: 4 blocks, width 128, 4 heads, feed-forward 512; approximately 0.8 M parameters each. A per-pair linear path on the current state is added. |
| Missing history | Masked out of attention (rollout start), never filled with copies |
| Model step | 20 ms |
| Training | Core on free steps; collision network on exact contact labels; contact network on contact steps with noise on the history and on the model's own 1-3 previous steps (with the true switches); rollout refinement. Every scale comes from a data rule. |
| Config | `configs/contact_nrd/v5/v5f_st_20ms_s63.json` (the same file for ball and pool) |

## Results on new test shots (models frozen before the shots were collected)

| System, metric | Version 5 | Pair-feature model | Earlier best state-only |
|---|---|---|---|
| Pool, ball B error at 2 s, median / p95 (4,800 shots) | **0.77 / 3.88 mm** | 1.63 / 7.07 mm | 1.90 / 12.3 mm |
| Pool, shots over 1 cm | **96** | 148 | 300 |
| Ball, path error median / p95 (1,800 shots) | **0.52 / 0.95 mm** | 0.63 / 1.11 mm | 0.53 / 0.94 mm |
| Pool targeting in Chrono, within 1 cm (median miss) | **100 / 100 (0.90 mm)** | 99 / 100 (1.67 mm) | 93 / 100 (1.59 mm) |
| Ball targeting in Chrono, within 3 mm (median miss) | 50 / 50 (**0.43 mm**) | 50 / 50 (0.59 mm) | 50 / 50 (0.59 mm) |

Paired bootstrap: on pool, version 5 beats the pair-feature model (ratio 0.48 at the median and 0.55 at p95);
on the ball it equals the earlier best state-only model and beats the pair-feature model.

## What made the difference

| Finding | Effect |
|---|---|
| The earlier "Transformer core" always got one time step, so attention did nothing | Now every network attends over 4 time steps and over the bodies |
| History networks were trained on true histories only; in a rollout their own errors broke them (pool median 1.3 -> 8.9 mm) | Noise on the history and training on the model's own steps: own-history and true-history rollouts now agree (ratio 1.00) |
| Missing history was filled with copies of the first state | Masked; a rollout that starts 3 steps before a contact has no penalty |
| Comparisons between designs were not equal (a stopped run, a lower learning rate, no refinement) | One recipe for every design: attention is 2-3x better than an MLP in the same recipe |
| History 8 / history 1 / history 4; 10 ms / 20 ms | History 4 and 20 ms are best |

## How to run (AMD cluster)

```bash
# train (one config for both systems; --data picks the system)
python3.12 -m nedm.contact_nrd.train_v5 --data <root>/data/pool_train --config configs/contact_nrd/v5/v5f_st_20ms_s63.json --output-dir <run>
# test on a fresh cohort, with paired verdicts against other models and ensembles
python3.12 -m nedm.contact_nrd.certify_shared --system pool --data <fresh>/pool_raw --unified name=<run> ... --headline name --comparison other --output out.json
# targeting through the model, then Chrono replay
sbatch scripts/contact_nrd/cluster/target.sbatch   # CN_CHECKPOINT=<run>/best.pt ...
# history robustness checks
python3.12 -m nedm.contact_nrd.probe_v5 --data <root>/data/pool_train --run <run> --output probe.json
```

Data, checkpoints and result files stay on the cluster:
`/work1/dannegrut/harry/experiments/contact_nrd_20261002/` (`data/`, `runs_v5/`, `results_v5/`, `fresh_v5/`).

## Limits

- Fixed starting layouts: a local fit of nearby training launches still targets pool more precisely
  (median 0.23 mm). A model pays off once starting positions vary.
- The ball launches are a narrow family; generalisation outside it was not tested.
- The pool core gives a small velocity change for a ball at rest (median 8e-5 m/s per step on rest inputs).
- A few constants of the recipe (loss weights, noise scale, history length) were chosen once for both systems.
