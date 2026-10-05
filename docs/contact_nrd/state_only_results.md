# State-only collision and contact networks: results (2026-10-03/04)

Plan, decision rule and every change: `state_only_plan.md`. Code: `model_v3.py`,
`train_v3.py`; configs `configs/contact_nrd/v3/`; cluster runs `runs_v3/`.

## What the networks see (deployed model's core reused, frozen)

| | Collision network | Contact network |
|---|---|---|
| Ball | [2 rows x 9]: the ball's state, one row per wall (floor, wall), + a learned wall code | same rows, 3 x 1024 tanh, per-wall scale/shift -> the ball's change [9] |
| Pool, `routed` | [10 rows x 18]: [this ball, other ball] for A-B (both orders), [this ball, zeros] for each cushion | same rows -> this ball's change [9]; A and B share each response |
| Pool, `joint` | [9 rows x 18]: the whole state of both balls for every pair + a learned pair code | same rows -> both balls' change [18], masked to the pair's members |

- With history (`k8`), every row holds 8 steps: the current state plus 7 differences.
- No gravity, geometry, relative positions, radii or pair frames enter.
- The inputs pass through data-fitted per-group scaling (fitted on contact or
  near-contact steps) and a fixed soft limit, 6·tanh(x/6).

## Pool (validation, 2,400 shots; selection score: lower is better)

| Arm | Score | Target-worthy B p95 at 2 s | B median at 2 s | B path p95 | Events right |
|---|---|---|---|---|---|
| **Deployed (pair features)** | **0.0088** | **6.4 mm** | 1.63 mm | 3.9 mm | 99.1 % |
| `joint`, history 1, seed 61 / 62 / 63 | 0.0296 / 0.0303 / 0.0290 | 21.9 / 22.2 / 21.3 | 2.28 / 2.37 / 2.20 | 13.4 / 14.3 / 13.2 | 98.0 / 98.1 / 97.9 % |
| `joint` + refinement forced (exploratory) | 0.0281 | 20.9 | 2.35 | 12.8 | 98.5 % |
| `routed`, history 1 | 0.0458 | 34.8 | 2.87 | 19.7 | 97.7 % |
| `routed`, history 8, MLP | 0.184 | 131 | 27.7 | 94.7 | 90.2 % |
| `routed`, history 8, Transformer (no refinement, 4 h limit) | 0.0844 | 56.2 | | 48.8 | 92.5 % |
| `joint` + mild whitening | 0.0575 | 40.1 | 5.05 | 30.6 | 95.9 % |
| `joint` + mirror copies (exploratory) | 0.0916 | 66.6 | 5.35 | 44.9 | 94.9 % |
| `routed` + mirror copies (exploratory) | 0.134 | 97.5 | 5.23 | 67.2 | 94.8 % |
| `joint` + rigid-motion copies of A-B contacts (exploratory) | 0.104 | 80.2 | 10.6 | 42.0 | 95.2 % |

**Where the gap is.** One-step error with the true switches, validation contacts, RMS
position / velocity:

| Pair | Deployed | `joint` |
|---|---|---|
| A-B | 0.037 mm / 1.7 mm/s | 0.096 mm / 4.9 mm/s |
| ball-cushion | 0.04-0.05 mm / 2.7-5.0 mm/s | 0.06-0.11 mm / 3.1-6.4 mm/s |

The collision network is not the gap. From the state alone it misses 0-4 and
falsely fires 0-8 times per ~2,200 contacts per pair, on par with the deployed one.

**Fresh pool cohort** (4,800 new shots, collected after the freeze, seed 1068567611; the pool study's own evaluator):

| Model | B at 2 s median | B at 2 s p95 | Target-worthy p95 | B path p95 | Events right | Shots over 1 cm |
|---|---|---|---|---|---|---|
| **Deployed (pair features)** | **1.64 mm** | **7.6 mm** | **6.5 mm** | **4.4 mm** | **99.2 %** | **155** |
| State-only `joint`, seed 63 (validation pick) | 2.20 | 21.5 | 18.9 | 12.7 | 98.6 % | 475 |
| State-only `joint`, seeds 61 / 62 | 2.31 / 2.22 | 21.3 / 20.9 | 17.3 / 18.2 | 12.1 / 13.1 | 98.5 / 98.3 % | 438 / 498 |
| `joint` + refinement (exploratory) | 2.30 | 20.2 | 17.4 | 11.8 | 98.8 % | 437 |
| `routed`, history 1 | 2.87 | 29.4 | 28.5 | 18.6 | 98.3 % | 598 |

Verdict against the deployed model, paired bootstrap:
- median ratio 1.35 (1.30-1.39) and p95 ratio 2.83 (2.55-3.21): **worse**;
- wrong events 66 vs 40 (p = 0.004);
- shots over 1 cm 475 vs 155 (p = 1e-53).

## Ball

Validation (900 shots; the ball study's score, lower is better):

| Arm | Score | Path p95 | End p95 | Events right |
|---|---|---|---|---|
| **Deployed (pair features)** | 0.00209 | 1.08 mm | 2.01 mm | 100 % |
| `routed`, history 1, seed 61 / 62 / 63 | 0.00283 / 0.00276 / 0.00267 | 1.43 / 1.42 / 1.36 | 2.79 / 2.69 / 2.62 | 100 % |
| `routed`, history 8, MLP | 0.00201 | 1.04 | 1.94 | 100 % |
| `routed`, history 8, Transformer (stopped by the 4 h limit at 150k of 160k updates, no refinement) | **0.00169** | **0.89** | **1.60** | 100 % |

Fresh ball cohort (1,800 new shots, seed 1321133335; the ball study's own evaluator):

| Model | Path median | Path p95 | End p95 | Worst end | Shots over 1 cm | Events right |
|---|---|---|---|---|---|---|
| Deployed (pair features) | 0.62 mm | 1.12 mm | 2.14 mm | 3.9 mm | 0 | 100 % |
| `routed`, history 1, seed 63 (the plan's pick) | 0.70 | 1.39 | 2.63 | 20.8 | 1 | 100 % |
| `routed`, history 1, seeds 61 / 62 | 0.72 / 0.69 | 1.47 / 1.50 | 2.76 / 2.82 | 9.8 / 18.8 | 0 / 7 | 100 % |
| `routed`, history 8, MLP | 0.59 | 1.07 | 1.91 | 17.1 | 1 | 100 % |
| **`routed`, history 8, Transformer** | **0.53** | **0.93** | **1.66** | **3.4** | **0** | **100 %** |

Paired bootstrap against the deployed model (path error; 95 % ranges):

| Arm | Median ratio | p95 ratio | Verdict |
|---|---|---|---|
| history 1, seed 63 (the plan's pick) | 1.13 (1.09-1.16) | 1.24 (1.14-1.31) | matches at the median; p95 inconclusive (just over 1.25) |
| history 8, Transformer | 0.85 (0.83-0.87) | 0.83 (0.78-0.86) | **beats**; endpoint p95 0.78 (0.73-0.82) |
| history 8, MLP | 0.95 (0.92-0.97) | 0.95 (0.89-0.98) | **beats** |

The plan chose the arm by pool. On the ball, that arm is the history-1 model.
Picking the ball's own validation-best arm (the history-8 Transformer) is a ball-only
selection made on validation, and that model then beat the deployed model on fresh shots.

## Answer

| | State only, history 1 | State only, history 8 + Transformer | Pair features (deployed) |
|---|---|---|---|
| Ball, fresh path p95 | 1.39 mm | **0.93 mm** | 1.12 mm |
| Pool, B at 2 s p95 | 21.5 mm (fresh; joint) | 56 mm (validation, target-worthy; routed; not fresh-tested) | **7.6 mm** (fresh) |
| Pool, contact detection | as good as deployed | as good as deployed | — |

- **Ball: the state is enough, and history helps.** The history-8 Transformer is the
  most accurate ball model of the study.
- **Pool: the state is enough to detect contacts, not (yet) to predict their effect
  as well.**
  - Error at 2 s is 1.35x the deployed model at the median and 2.8x at p95.
  - Most of the gap is the A-B collision response (3x one-step error).
  - None of these closed it: history, Transformer, whitening, mirror copies, rigid-motion copies of A-B contacts, refinement.
- **Likely reason (pool).** The pair frame makes the A-B response independent of
  where and in which direction the collision happens. From absolute states the
  network must learn that from 19k collisions, all from one fixed start. Copies under
  symmetry spread its capacity over states the test shots never visit.

## Practical lessons from getting state-only inputs to train

1. Scale each group's inputs on its own contact (contact network) or near-contact
   (collision network) steps, as version 2 did. Global scaling squeezes the
   centimetres that matter.
2. Do not whiten. It magnifies directions that are constant at contact (spin locked to
   velocity), and rollouts then explode.
3. Add a fixed soft limit, 6·tanh(x/6). Without it, the 10 % far steps reach -100 to
   -400 sigma and the loss blows up.
4. Routed rows must output one shared 9-number change for the affected ball (a review
   finding fixed before the counted runs).

## Follow-up (10-04): contact network trained on contact steps only (no far steps)

Validation, seed 61, end of contact training (refinement skipped by the futility rule in all three pool runs):

| Arm | Score | Target-worthy B p95 at 2 s | B median at 2 s | B path p95 | Events right |
|---|---|---|---|---|---|
| Deployed (pair features) | 0.0088 | 6.4 mm | 1.63 mm | 3.9 mm | 99.1 % |
| pool `joint`, with far steps, soft limit (before) | 0.0296 | 21.9 | 2.28 | 13.4 | 98.0 % |
| pool `joint`, contact steps only, soft limit | 0.0275 | 20.5 | 2.54 | 12.2 | 98.2 % |
| **pool `joint`, contact steps only, no soft limit** | **0.0212** | **15.5** | 1.93 | 9.6 | 98.1 % |

| Ball arm | Score | Path p95 | End p95 | Events right |
|---|---|---|---|---|
| Deployed (pair features) | 0.00209 | 1.08 mm | 2.01 mm | 100 % |
| `routed`, history 1, with far steps (before) | 0.00283 | 1.43 | 2.79 | 100 % |
| `routed`, history 1, contact steps only (best at 100k of 160k updates) | 0.0023 | 1.22 | 2.15 | 100 % |

- Dropping the far steps helps both systems. The ball learns about 10x faster
  (0.0052 at 10k updates, against 0.078).
- Without far steps, the soft limit is not needed, and leaving it out is better on pool.
- The pool gap to the deployed model shrinks from 3.4x to 2.4x on the validation score.
