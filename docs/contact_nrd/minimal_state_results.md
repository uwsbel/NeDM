# Minimal state and per-network inputs: results (2026-10-04 night)

Plan and changes: `minimal_state_plan.md`. Runs: `runs_v4/` on AMD.

All numbers are validation, unless marked fresh. Score = the pool study's selection score
(lower is better). "2 s p95" = B's error at 2 s on target-worthy shots, 95th percentile.

## Pool

| Arm | Positions | Collision sees | Contact sees / gives | Score | 2 s p95 |
|---|---|---|---|---|---|
| Deployed pair features (reference) | predicted | pair features | pair features | 0.0088 | 6.4 mm |
| p0 state, seeds 61 / 62 / 63 | predicted | state | state / position, velocity, spin | 0.0197 / 0.0203 / 0.0171 | 14.0 / 14.7 / 12.5 |
| p1 integrated, no correction (idea 1, literal) | integrated | state | state / velocity, spin | 0.097 | 75.7 |
| p2 integrated + position correction | integrated | state | state / + position correction | 0.0188 | 13.4 |
| p3 collision from positions, 2 steps | integrated | positions now + last step | state | 0.036 | 26.8 |
| p3b collision from positions, 1 step | integrated | positions now | state | 11.3 (fails) | 10,000 |
| p4 contact from velocity and spin | integrated | state | velocity, spin | 0.165 (diverged) | 130 |
| p5 both (idea 2, full) | integrated | positions, 2 steps | velocity, spin | 0.178 (diverged) | 138 |
| d0 = p0 + A-B difference (diagnostic) | predicted | + difference | + difference | 0.0201 | |
| d2 = p2 + A-B difference (diagnostic) | integrated | + difference | + difference | 0.0163 | |

Ensembles of three seeds (collision logits and contact changes averaged):

| | Single best | Ensemble of 3 | Ensemble 2 s p95 / median |
|---|---|---|---|
| Pair features (deployed recipe) | 0.0088 | **0.0075** | 5.3 / 1.50 mm |
| State only (p0) | 0.0171 | 0.0135 | 9.8 / 1.70 mm |

## Ball

| Arm | Score | Path p95 | End p95 | Events right |
|---|---|---|---|---|
| Deployed pair features | 0.00209 | 1.08 mm | 2.01 mm | 100 % |
| b0 state (no far steps, no soft limit) | 0.00225 | 1.20 | 2.09 | 100 % |
| b2 integrated + correction (Chrono sub-step rule) | 0.00253 | 1.36 | 2.34 | 100 % |
| b1 integrated, no correction | 0.069 | 56.5 | 25.7 | 100 % |
| b4 contact from velocity and spin | 0.24 | 63 | 93 | 98.7 % |
| b5 both | diverged | | | |

## What this says about the ideas

- **Idea 1 (minimal state, integrated positions).**
  - With a position correction from the contact network it works as well as predicting positions; it is not better. Pool: 0.0188 against 0.0197 / 0.0203 / 0.0171 for three seeds. Ball: 1.36 against 1.20 mm.
  - Without the correction it fails in both systems (pool 0.097, ball 56 mm). An impact part-way through a 10 ms step moves the ball in a way the velocities at the step's ends cannot tell.
  - For free motion, integration is exact: pool 1.3e-7 m per step, against 1.9e-6 m learned. It needs the simulator's own sub-step rule for the ball.
- **Idea 2 (less information per network).** It hurts in every case.
  - The collision network needs approach speed as well as position. From positions alone it fires 50-320 false alarms per ~2,200 contacts. With the previous step's positions (speed implied) it works, but the rollouts are worse.
  - The contact network needs positions, which give the contact geometry and the impact time within the step. With velocity and spin only, it diverges in both systems.
- **What did help pool:**
  - dropping the far steps;
  - refinement (no futility cut): about 0.027 to 0.020;
  - ensembling three seeds: 0.017-0.020 to 0.0135.
  - The same ensembling helps the pair-feature model (0.0088 to 0.0075). Like for like, the state-only model stays about 1.8-1.9x behind.

## More levers tried on pool (validation)

| Lever | Result |
|---|---|
| 4x training shots (76,800) | single seeds 0.0179 / 0.0253 / 0.0188; ensemble 0.0162. No reliable gain: seed spread is larger than the data effect |
| 4x shots with integrated positions | 0.0173 / 0.0220 / 0.0208; ensemble 0.0158 |
| 8-step history through causal Transformers | 0.18, much worse |
| Weight decay 1e-2 / contact width 512 / 2048 | 0.0206 / 0.0180 / 0.0197, all within seed noise |
| A-B difference as extra input (diagnostic) | 0.0154 with integrated positions; no gain with predicted positions |

## Fresh pool cohort

4,800 new shots, seed 202610045, fixed before any checkpoint was chosen;
manifest in `artifacts/contact_nrd/minimal_state/MANIFEST_fresh_v4.txt`.

| Model | B at 2 s median | B at 2 s p95 | Target-worthy p95 | B path p95 | Events right | Shots over 1 cm |
|---|---|---|---|---|---|---|
| Pair features, single (deployed) | 1.64 mm | 7.58 mm | 7.10 mm | 4.20 mm | 99.1 % | 161 |
| Pair features, ensemble of 3 | **1.48** | **5.87** | **5.51** | **3.25** | 99.1 % | **77** |
| State only, single (best seed by validation) | 1.94 | 12.74 | 12.08 | 7.39 | 98.8 % | 319 |
| State only, ensemble of 3 | 1.69 | 10.17 | 9.84 | 6.06 | 98.9 % | 246 |
| Integrated positions (idea 1), single | 1.91 | 12.29 | 12.41 | 7.19 | 98.7 % | 313 |
| Integrated positions (idea 1), ensemble of 3 | 1.59 | 10.05 | 9.94 | 6.04 | 98.9 % | 243 |
| A-B difference diagnostic, single | 1.73 | 12.11 | 10.52 | 7.20 | 99.0 % | 314 |
| *(before this round) state only, single, on the previous fresh cohort* | *2.20* | *21.5* | *18.9* | *12.7* | *98.6 %* | *475* |

Paired bootstrap (B at 2 s; ratio state-only / pair features; 95 % ranges):

| Comparison | Median | p95 | Shots over 1 cm |
|---|---|---|---|
| state single vs pair single | 1.18 (1.14-1.21) matches | 1.68 (1.49-1.90) worse | 319 vs 161 |
| integrated single vs pair single | 1.16 (1.12-1.20) matches | 1.62 (1.46-1.84) worse | 313 vs 161 |
| state ensemble vs pair ensemble | 1.14 (1.11-1.18) matches | 1.73 (1.57-1.89) worse | 246 vs 77 |
| integrated ensemble vs pair ensemble | 1.08 (1.05-1.12) matches | 1.71 (1.48-1.88) worse | 243 vs 77 |
| state ensemble vs pair single | 1.03 (0.99-1.06) matches | 1.34 (1.20-1.49) inconclusive | 246 vs 161 |

## Answer

- **The milestone was not reached.** No state-only pool model came within 1.25x of the
  pair-feature model on validation (best single 0.0171 against 0.0088).
- **The typical shot now matches; the tail does not.**
  - On fresh shots the state-only model's median error at 2 s is within 6-18 % of the pair-feature model, like for like.
  - Its 95th percentile is 1.6-1.7x, with about twice as many shots over 1 cm.
- **Progress since the last state-only result.** The single model's p95 went from 21.5 to
  12.7 mm and its median from 2.20 to 1.94 mm, from dropping the far steps and the
  soft limit and allowing refinement.
- **Idea 1 (minimal state, integrated positions).**
  - It needs a position correction for impacts within a step, and with it ties predicting positions (fresh median 1.91 vs 1.94 mm, p95 12.3 vs 12.7 mm).
  - It is exact for free motion, but the remaining error lives in contacts, so it does not close the gap.
- **Idea 2 (less information per network).** It fails in every variant: the collision
  network needs approach speed, and the contact network needs the contact geometry
  and timing that only positions give.
- **What remains.** The tail error sits in the A-B collision response. The pair-feature
  model shares that response across every position and direction on the table. From
  the state alone, neither 4x the data nor input copies under symmetry gave the network
  that sharing.

## Targeting through the state-only models (2026-10-04, after the fresh test)

Same targets, settings and code as the deployed targeting (`target_ball.json`,
`target_pool.json`). Each chosen launch is replayed in Chrono. Results:
`results_target_v4/` on AMD (jobs 450857-450859, mi3501x).

| System, model | Levenberg-Marquardt: hits, median, worst | Gradient descent: hits, median, worst |
|---|---|---|
| Ball (within 3 mm), pair features (deployed) | 50 / 50, 0.59 mm, 1.89 mm | 50 / 50, 0.51 mm, 1.79 mm |
| Ball, state only, 8-step Transformer | 50 / 50, 0.59 mm, 1.96 mm | 50 / 50, 0.43 mm, 2.11 mm |
| Ball, state only, history 1 | 50 / 50, 0.52 mm, 2.25 mm | 50 / 50, 0.50 mm, 1.65 mm |
| Pool (within 1 cm), pair features (deployed) | 99 / 100, 1.67 mm, 10.2 mm | 89 / 100, 2.01 mm, 54.4 mm |
| Pool, state only, best single (p0, seed 63) | 93 / 100, 1.59 mm, 94.5 mm | 88 / 100, 1.91 mm, 67.5 mm |

- Ball: the state-only models target as well as the deployed model.
- Pool, Levenberg-Marquardt: 6 targets hit only by the pair-feature model, 0 only by the state-only model (exact paired test p = 0.031). All 7 state-only misses are targets where B hits a cushion before t; in 5 of them the Chrono replay has no B cushion hit. Targets without a cushion hit: 50 / 50 for both.
- Pool, gradient descent: no difference (8 against 7 one-sided hits, p = 1.0).
