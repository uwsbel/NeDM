# One architecture for the core, collision and contact networks (version 5): results

Plan, audit and amendments: `unified_v5_plan.md`. Code: `src/nedm/contact_nrd/model_v5.py`, `train_v5.py`,
`probe_v5.py`, `ensemble.py` (EnsembleV5). Configs: `configs/contact_nrd/v5/` (the final configuration is
`v5f_st_20ms_s6{1,2,3}.json`; one file for both systems). Runs: `runs_v5/` on AMD. Results: `results_v5/` on AMD.
Diagram: `~/.claude/diagrams/v5-unified-design.png`.

## Answer

- **Converged.** One design and one configuration file train the core, the collision network and the contact
  network for both systems; only the data differs. Design: a Transformer encoder (4 blocks, width 128, 4 heads,
  feed-forward 512) over tokens of the last 4 states of every moving body plus learned pair and fixed-body codes;
  20 ms model step.
- **Pool, fresh 4,800 shots:** the state-only model now beats the pair-feature model. B at 2 s: median 0.77 mm and
  p95 3.88 mm, against 1.63 mm and 7.07 mm (paired ratios 0.48 and 0.55, both "beats"). It also beats the
  pair-feature ensemble of 3 (1.46 / 5.50 mm). The earlier best state-only model had 1.90 / 12.3 mm.
- **Ball, fresh 1,800 shots:** path error median 0.52 mm, p95 0.95 mm. It equals the earlier best state-only model
  (8-step Transformer, 0.53 / 0.94 mm, paired p95 ratio 1.02, 95 % range 0.98-1.05) and beats the pair-feature
  model (0.63 / 1.11 mm).
- **Targeting in Chrono:** pool 100 / 100 within 1 cm (median miss 0.90 mm; pair features 99 / 100, 1.67 mm);
  ball 50 / 50 within 3 mm (median 0.43 mm; pair features 0.59 mm).
- **Why it works (from the audit and the ablations):** no wiring bug; the earlier history models failed because
  they never saw their own errors in training, a linear path read every history column, and the comparisons were
  unfinished. With noise, own-history training, masked missing steps and one recipe, attention wins clearly over
  an MLP in the same recipe (pool 0.0134 against 0.0282 on the short schedule). History length 4 is slightly
  better than 1 on pool and equal on the ball; history 8 is worse on pool. The 20 ms step is better than 10 ms.

## Screen and ablations (validation score, lower is better; short schedule)

| Arm | Pool | Ball |
|---|---|---|
| Old recipe (history-1 MLP, frozen v2 core), same schedule | 0.0443 | 0.00161 |
| Space-time Transformer, history 4 (seeds 61 / 62) | 0.0134 / 0.0154 | 0.00177 / 0.00183 |
| Same, no own-history training | 0.0154 | 0.00177 |
| Same, history 1 / history 8 | 0.0159 / 0.0296 | 0.00175 / 0.00172 |
| Same, noise scale 0 / 3 | 0.0164 / 0.0158 | 0.00172 / 0.00173 |
| Same, 20 ms step | 0.0120 | 0.00169 |
| Time Transformer, history 4 | 0.0185 | 0.00169 |
| Entity Transformer, history 2 | 0.0215 | 0.00325 |
| MLP, history 4 (same recipe) | 0.0282 | 0.00506 |

## Full runs (validation score; seeds 61 / 62 / 63)

| Arm | Pool | Ball |
|---|---|---|
| Space-time, 20 ms | 0.00518 / 0.00580 / **0.00494** | 0.00174 / 0.00174 / **0.00168** |
| Space-time, 10 ms | 0.00703 / 0.00836 / 0.00840 | 0.00183 / 0.00169 / 0.00177 |
| Earlier: pair features (deployed), state-only best seed | 0.0088, 0.0171 | 0.00209, 0.00169 |

Validation picks: seed 63 on both systems.

## Fresh cohorts (new seeds; collected after every model was frozen)

Pool: 4,800 shots, seed 202610055 (`fresh_v5/pool_raw`). B's error at 2 s, mm.

| Model | Median | p95 | Shots over 1 cm | Events right |
|---|---|---|---|---|
| **Version 5, 20 ms, seed 63 (pick)** | **0.77** | **3.88** | **96** | 99.5 % |
| Version 5, 20 ms, seeds 61 / 62 | 0.68 / 0.76 | 4.17 / 4.09 | 109 / 115 | 99.4 / 99.3 % |
| Version 5, 10 ms, seeds 61-63 | 1.05-1.08 | 4.93-5.63 | 90-99 | 99.0-99.3 % |
| Version 5, 20 ms, ensemble of 3 | 0.47 | 2.69 | 54 | 99.6 % |
| Pair features (deployed), single / ensemble of 3 | 1.63 / 1.46 | 7.07 / 5.50 | 148 / 82 | 99.2 / 99.1 % |
| Earlier state-only (history-1 MLP), seeds 61-63 / ensemble | 1.90-2.02 / 1.68 | 12.3-16.5 / 10.5 | 300-426 / 260 | 98.2-99.1 / 99.0 % |

Paired bootstrap, pick against each arm (ratio, 95 % range): pair features median 0.48 (0.46-0.49) beats, p95 0.55
(0.48-0.63) beats; pair-feature ensemble 0.53 / 0.71 beats; earlier state-only seed 63 0.41 / 0.31 beats; the
version-5 10 ms seeds beat at both quantiles; seeds 61 / 62 of the same recipe "match"; the 20 ms ensemble is
better than the single pick (1.66 / 1.44, "worse" for the single).

Ball: 1,800 new shots, seed 1321133355 (`fresh_v5/ball_merged`). Path RMS error, mm.

| Model | Median | p95 | Shots over 1 cm |
|---|---|---|---|
| **Version 5, 20 ms, seed 63 (pick)** | **0.52** | **0.95** | 0 |
| Version 5, 20 ms, seeds 61 / 62 | 0.52 / 0.54 | 0.96 / 1.02 | 0 |
| Version 5, 10 ms, seeds 61-63 | 0.53-0.56 | 0.91-1.01 | 0 |
| Version 5, 20 ms / 10 ms, ensemble of 3 | 0.51 / 0.52 | 0.93 / 0.91 | 0 |
| Earlier best state-only (8-step time Transformer) | 0.53 | 0.94 | 0 |
| Earlier state-only, history 1 | 0.65 | 1.26 | 1 |
| Pair features (deployed) | 0.63 | 1.11 | 0 |

Paired verdicts of the pick: earlier best state-only "matches" (median 0.98, 0.96-1.00; p95 1.02, 0.98-1.05); pair
features "beats" (0.83 / 0.86); history-1 state-only "beats" (0.80 / 0.76). The 20 ms path metric uses 20 ms
samples and the 10 ms models 10 ms samples.

## Targeting through the picks (same targets and settings as every earlier targeting run; Chrono replay)

| System, method | Hits | Median miss | Worst miss | Earlier pair features | Earlier state-only |
|---|---|---|---|---|---|
| Pool (within 1 cm), Levenberg-Marquardt | 100 / 100 | 0.90 mm | 5.5 mm | 99 / 100, 1.67 mm | 93 / 100, 1.59 mm |
| Pool, gradient descent | 97 / 100 | 0.84 mm | 20.6 mm | 89 / 100, 2.01 mm | 88 / 100, 1.91 mm |
| Ball (within 3 mm), Levenberg-Marquardt | 50 / 50 | 0.43 mm | 2.3 mm | 50 / 50, 0.59 mm | 50 / 50, 0.59 mm |
| Ball, gradient descent | 50 / 50 | 0.47 mm | 2.4 mm | 50 / 50, 0.51 mm | 50 / 50, 0.43 mm |

With the fixed starting layouts and dense training grids, a local linear fit of training launches is still more
precise on pool (median 0.23 mm); a model pays off once starting positions vary.

## Hygiene checks (validation shots; `probe_v5.py`)

| Check | Pool | Ball | Bar |
|---|---|---|---|
| Own history / true history, median error at the target time | 1.00 | 1.00 | 1.2 or less |
| Mid-start 3 steps before a contact, no history: median error after 0.5 s (cold / warm) | 0.29 / 0.30 mm | 0.31 / 0.31 mm | 1.5x or less |
| Next-state response to older-history noise of the core's error size, relative to that noise | 0.05 | 0.007 | small |
| Finite rollouts | 100 % | 100 % | 100 % |

## Milestone

| Criterion | Result | Met |
|---|---|---|
| One configuration, no system-specific keys | `v5f_st_20ms_s6x.json` (only the seed differs) | yes |
| Pool validation pick 0.0171 or less; 3-seed median 0.0197 or less | 0.00494; 0.00518 | yes |
| Ball fresh path p95 0.93 mm or less, 100 % events | 0.952 mm single (ensemble 0.929 mm); equal to the earlier best on the same cohort (paired); 100 % | equal, not under the literal bar for the single model |
| Hygiene | all pass | yes |
| Stretch: pool validation 0.011 or less | 0.00494 | yes |

## Limits

- One recipe still has fixed constants that were chosen once for both systems (kappa, noise scale 1, history 4,
  own-history settings); they were not tuned per system.
- The validation pick uses the minimum over many checks (small optimistic bias); the fresh cohorts are the claim.
- The pool core output is not exactly zero for a ball at rest (median 8e-5 m/s per step on rest inputs); the
  rollout errors include this effect.
- Ball launches are a narrow 2-parameter family; generalisation outside it was not tested in this round.
- A separate study on the same cluster (`contact_nrd_codex_unified_20261004/v5_independent_qualification`) runs
  its own qualification of these models; its results are not part of this document.

## Compute

Approximately 25 node-hours on AMD (screen, ablations, 12 full runs, certification, targeting, probes) plus
approximately 2 CPU node-hours for the fresh cohorts.
