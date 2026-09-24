# Raising soil goal-reaching from a moving start: 83.9 % to 97.5 %

Written 2026-09-22 16:40, finalised 2026-09-24. Branch `crm_improve_v1`, root
`artifacts/traverse/crm_improve_20260922/` (K2). Baselines and suites come from the previous effort
`artifacts/traverse/generalist_20260921/` (K1, read-only). All numbers are closed-loop Chrono drives on the f104
arena, 800 paired start-goal pairs (600 fresh + 200 reused), CEM 4x64 unless a gradient arm is named, picks hashed
before driving, paired group bootstrap with a terrain-clustered interval as robustness.

## Answer in one table (soil, goal reached, 800 groups)

| planner decides | label-free model | oracle domain tag | soil specialist |
|---|---|---|---|
| after a 3 s approach (K1 protocol) | 83.9 % | 84.4 % | 83.0 % |
| after a 1 s approach | 93.2 % (history), 94.8 % (transformer) | 95.4 % | 93.6 % |
| after a 0.5 s approach | 94.1 % (K1 history), 95.8 % (short-window history) | 96.1 % | – |
| 0.5 s approach + gradient refinement | **97.5 %** (short-window history), 97.2 % (transformer) | – | – |
| from a standing start (K1 A3) | 93.8 % | 96.1 % | 95.8 % |

Rigid ground under the same short approach: 99.5-100 % for every arm, 100.0 % for the final configuration.

The final configuration is label-free (no domain tag, no world identifier at deployment): a shared CNN-GRU risk model
with a two-second history encoder trained with short-anchor rows, deciding 0.5 s after the start from the recorded
motion, its route then refined by gradient descent on the ensemble's risk. Soil 97.5 %, rigid 100.0 %, and 17 % faster
than the 3 s baseline (median time ratio 0.83).

## Why the old number was low

The 3 s straight approach, not the model, cost most of the 16 points.

- 86 of 800 groups failed under every arm; in those groups the decision point sits 2.0 m from the first steep cell
  (against 8.0 m in groups every arm completes), and 69 % of the next 10 m is steeper than 15 degrees. Terrain
  position alone separates all-fail from all-success groups with AUC 0.855; vehicle condition (speed, sinkage, slip,
  pitch) only 0.60.
- 68 of those 86 groups were completed by the same planner family from a standing start, where it turns off the
  straight line immediately (2.7 m sideways and 35 degrees off after 6 m of travel).
- The failure rate added by the approach grows with the approach terrain: 0 points on flat approaches, +12 points at
  12-17 degrees, +21 at 17-25, +38 above 25.
- A 1 s approach travels about 0.7 m and a 0.5 s approach about 0.1 m, so the vehicle still stands on the flat start
  pad when it decides, while a 0.25 s history window already identifies the world (AUC 0.999).

## What was tested and what it changed

1. **Approach length (the fix).** Same suite, same models, only the decision time moves: soil 83.9 % to 93-95 % at
   1 s and to 94-96 % at 0.5 s; every arm improves by 8.8 to 12.2 points of failure with the clustered interval
   excluding zero and 7-9 of 9 terrain clusters improving. Rigid stays at the ceiling.
2. **Handover speed step (refuted).** Making every candidate route start at the vehicle's current speed made soil
   *worse*: 81.8 % against 83.9 % (paired p 0.02). The new follower starts near zero throttle, so the vehicle loses
   momentum at the foot of a climb. The speed step correlates with failure between groups, but within a group, at the
   same decision state, the effect disappears once the route's terrain is accounted for (odds ratio 0.90 [0.65, 1.27]).
3. **Data balance (no effect).** The mixed training file is already row-balanced (58k rigid, 57k soil). Raising the
   soil share of each batch to 75 % did not help offline (soil within-group AUC 0.965 at a standing start against
   0.974 for the balanced model). The soil-only specialist and the shared model are within a point of each other
   closed loop at every protocol.
4. **Architecture (helps at 1 s, not at 0.5 s).** A transformer whose tokens are the 96 route stations *and* the 40
   history frames (attention over time and space) is the best soil model at the earliest decision offline (0.984
   against 0.974 for the CNN-GRU at a standing start) and closed loop at 1 s (94.8 % against 92.8 % for the CNN-GRU
   trained on the same rows, and statistically level with the oracle tag). At 0.5 s the CNN-GRU wins instead
   (95.8 % against 94.2 %). The transformer is also three times cheaper to train and drives 20 % faster.
5. **History window length.** Domain identification saturates: a 0.25 s window at the 0.5 s decision already reaches
   AUC 0.999 (hand features 0.995), and even 0.1 s reaches 0.991. Part of that separation is a difference between the
   two simulator set-ups (engine idle speed alone gives 0.990 at 0.1 s), so the probe is not proof that the model
   senses soil; the closed-loop gap between the label-free and oracle arms is the honest measure, and it is 0.3 to
   0.7 points at 0.5-1 s.
6. **Gradient route refinement (second fix).** Refining the CEM pick by 60 Adam steps on the frozen ensemble's risk
   (the night-2 method, re-implemented for history-conditioned models and moving decision states) adds 1.8 points on
   soil for the CNN-GRU (97.5 % against 95.8 %, p 0.016) and 3.0 points for the transformer (97.2 % against 94.2 %),
   with the clustered interval excluding zero, and it costs 2.6-3.4 s per decision.
7. **Moving-state training data (not a data problem).** 7,182 soil and 7,182 rigid continuation drives were collected
   from 3 s decision states of the 1,200 training groups (six continuations per state; in 85 % of soil states the six
   disagree on the outcome, so the rows carry the decision signal). Retraining with them changes nothing offline: soil
   within-group AUC 0.987 against 0.988 without them. On the new validation rows themselves (continuations from the
   same moving state, 367 soil pairs) the model trained *without* them already ranks the routes at 0.989, identical to
   the model trained with them. The remaining 3 s gap is therefore the protocol (where the vehicle stands when it
   decides), not missing training states. The earlier cross-scoring found weaker within-state ranking (0.64) only
   among the near-optimal picks of six planners, where the routes are nearly equally risky.

## Cost

39.0 billed node-hours of the 100 allowed (my own jobs, from the accounting records), dominated by roughly
20,000 Chrono soil episodes.

## Artefacts

`scout/` (D1 failure anatomy, D2 cross-scoring, W window probe, SYNTHESIS), `s2/` (short-approach picks, drives,
results per protocol), `s4/` (gradient picks, drives, results), `e2/` (handover test), `a5data/` (approach routes,
decision states, continuation tasks), `datasets/` (short-anchor and decision-frame rows), `offline_v1/`,
`deploy_v1/`, `NOTES_*.md` and `VERIFY_*.md` per module, `LOG.md`, `PLAN.md`. New code: `scripts/ci_train.py`,
`ci_planner.py`, `ci_grad.py`, `ci_a5data.py`, `ci_short_anchors.py`, `ci_window_probe.py`.
