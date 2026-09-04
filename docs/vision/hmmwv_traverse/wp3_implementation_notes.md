# WP3/WP4 implementation notes — tracker in imagination and planner rollout scoring

**Date:** 2026-09-04 (afternoon session) · **Modules:** `nedm/traverse/nrd_model.py`,
`nedm/traverse/tracker_env.py` · **Scripts:** `traverse_wp3_routes.py`,
`traverse_wp3_train_tracker.py`, `traverse_wp3_eval_tracker.py`,
`traverse_wp4_score_candidates.py` · **Artifacts:** `artifacts/traverse/wp3_routes/`,
`wp3_tracker_{v1,v1_s2,hist4}/`, `wp3_tracker_eval.json`, `wp4_scores_v1/`

Session goal: a planner rollout inside the NRD — candidate plans tracked by a learned
policy inside the frozen WP2 model and scored on predicted physics. Achieved in
pipeline form and calibrated against recorded episodes; Chrono untouched.

## What was borrowed from the state-only HMMWV tracking study

`src/nedm/rl/hmmwv_tracking_env.py` + `scripts/training/train_hmmwv_rl_tracking.py`
(the published Study Case I policy, 20/20 Chrono transfers). Kept verbatim in spirit:

- tracking reward `exp(-loss)` — bounded, strictly positive, so early termination
  only forfeits reward (the dpend "negative-everywhere" exploit cannot occur);
  no termination penalty; failures do not bootstrap, time-outs do;
- soft action-rate (0.2) and throttle×brake (0.05) penalties **plus** a hard
  steering-rate clamp of 0.1 per 20 Hz step applied *during training* — the
  study's dominant Chrono failure was steering reversals the model tolerated;
- loose failure bounds (cross-track 6 m, roll 0.6, pitch 0.4 rad) so the policy
  learns recovery instead of being reset;
- PPO block unchanged: lr 3e-4, adaptive KL 0.01, entropy 3e-3, 5 epochs × 8
  minibatches, [512,256,128] ELU, init noise 0.7, empirical obs normalization,
  2048 envs × 64 steps; rsl_rl 2.2.4 `OnPolicyRunner`. (Plan §10 names the arm
  preset lr 1e-4/kl 0.005; the HMMWV values were used because that is the study
  that transferred.)
- model context = 16 frames, pose integrated outside the network.

Changed per plan §10: references are **routes without timing** (the `PlanCandidate`
format), so the reward is geometric — cross-track (σ 1 m, w 2), heading
(σ 0.35 rad, w 0.8), speed vs the profile (σ 1 m/s, w 0.5); the observation is
the 38-D deployment set `[along/cross/heading error, 10 preview points × (dx, dy,
v_ref), vx, yaw rate, last action]`; `action_repeat = 1` at 0.05 s; episodes are
**1–3 s fragments** initialized from real recorded 16-frame context windows at
random progress along the episode's own route (frames after the vehicle parks
are excluded), so every imagined rollout stays inside the WP2-validated horizon.

Reference routes: the collection driver's own route per episode
(`traverse_wp3_routes.py`, 7627 of 9518 episodes; meander episodes have none) —
random smooth splines, near-obstacle passes, oracle routes, exactly plan §10's
"random splines + oracle routes". No new route generation.

## Environment facts

- Bank per split on GPU: normalized z1/actions, poses, fp16 scene maps
  (`map_v2`), padded routes. 5308 train episodes load in 21 s on MI350.
- Each step: tracker → clamp → dynamics model (16-frame history of state, crop
  token, action) → next state → pose → **re-crop the static map at the new pose**
  → next token. Power head integrated to energy (kJ).
- Throughput: 164k env-steps/s incl. PPO on one MI350; 1000 iterations = 14 min.
- Scripted pure-pursuit controller on the env's own observation serves as the
  baseline and as the smoke test.

## Tracker result (held-out layouts, 2048 3 s fragments, inside the model)

| policy | mean \|ct\| | p95 \|ct\| | mean per-fragment max \|ct\| | speed err | fail |
|---|---|---|---|---|---|
| pure pursuit (scripted) | 0.245 m | 0.725 m | 0.464 m | 0.343 m/s | 1.3 % |
| PPO 38-D obs, seed 1 | **0.150 m** | 0.543 m | 0.416 m | 0.279 m/s | 0.7 % |
| PPO 38-D obs, seed 2 | 0.146 m | 0.524 m | 0.428 m | 0.292 m/s | 1.4 % |
| PPO + 4-frame state history | 0.161 m | 0.549 m | 0.424 m | 0.268 m/s | 0.7 % |

Seeds agree; the state-history ablation buys nothing, so the deployment
observation set is sufficient (plan §10 obs ablation, in-model). The learned
policy beats the scripted controller by ~40 % on mean cross-track but only ~25 %
on the tail (p95), and part of the tail is inherited: fragments start wherever
the recorded vehicle was. **These numbers are measured inside the model the
policy was trained in** and are not G6; the Chrono evaluation is the gate and has
not been run. The steering clamp is trained in, which is the mitigation Study
Case I needed for transfer.

## Planner rollout scoring (Planner-C, plan §9.5) — `traverse_wp4_score_candidates.py`

For a held-out episode: the privileged oracle produces up to 6 candidate routes
to the house by parameter sweeps (default, energy weight 0 / 4, cruise 4 / 9 m/s,
inflation 3 m; duplicates dropped), plus the **recorded route** as a calibration
candidate with ground truth (time to the route end and engine energy from the
logged torque × speed). Every candidate is tracked inside the model from the
episode's real first 0.8 s, up to 20 s, and scored on time, energy (power head),
cross-track, safety bounds and footprint collision against the layout's obstacle
discs (privileged, allowed by the v1 charter).

**Calibration, 32 held-out oracle-family episodes, trained tracker:**

| recorded route | real vehicle | imagined | corr | MAE |
|---|---|---|---|---|
| time to goal | 11.14 s | 10.03 s | **0.997** | 1.11 s |
| engine energy | 150.7 kJ | 107.0 kJ | 0.77 | 43.7 kJ |
| mean cross-track | 0.074 m | 0.028 m | — | — |
| completed | 32/32 | 32/32 | | |

Time-to-goal is predicted almost perfectly in rank and with a consistent ~10 %
optimism. Energy is **underestimated by ~30 %** and the shortfall is the model's:
replaying the recorded driver inputs open-loop through the model gives 115 vs
149 kJ (corr 0.92) on the 21/32 episodes that stay on route open-loop. The power
head was trained on recorded states; imagined states drift (and the imagined
vehicle is slightly faster/lighter), so absolute energy needs a calibration
factor or a head retrained on closed-loop rollouts before Planner-C uses it in
absolute terms. For ranking, the bias is largely common-mode.

**Candidate table (32 layouts):**

| candidate | n | completed | collided | time | energy | mean ct | max ct |
|---|---|---|---|---|---|---|---|
| recorded (= oracle default) | 32 | 100 % | 0 | 10.03 s | 107 kJ | 0.028 | 0.136 |
| shortest (energy w 0) | 26 | 100 % | 0 | 10.21 | 113 | 0.054 | 0.321 |
| energy-averse (w 4) | 18 | 100 % | 0 | 10.24 | 108 | 0.045 | 0.259 |
| slow (cruise 4 m/s) | 31 | 97 % | 0 | 14.90 | 99 | 0.023 | 0.145 |
| fast (cruise 9 m/s) | 30 | 100 % | 0 | 9.32 | 122 | 0.033 | 0.151 |
| wide berth (inflation 3 m) | 17 | 100 % | 0 | 9.95 | 106 | 0.064 | 0.376 |

The scorer behaves sensibly — faster profiles cost energy, the slow profile saves
it, tighter geometry raises tracking error — and the composite (time + kJ/10)
picks fast / shortest / oracle about equally. **Only the recorded route has
ground truth**, so the ranking among alternatives is plausible, not validated;
that needs the same candidates driven in Chrono (WP5).

## Chrono evaluation (G6, the real gate) — `traverse_wp3_chrono_eval.py` on newton

The fragment-trained policy drives **continuously** in Chrono on 31 held-out
oracle-family layouts (recorded routes, 25 s cap; one run of 96 lost to a
newton reboot). Layout rebuilt from `meta.json`, no rendering, the policy sees
its training observation rebuilt from Chrono's true pose/speed (privileged in
v1), same squash and steering clamp. Bracket: the collection `ChPathFollowerDriver`.

| controller | completed | contact | rollover | time to end | energy | mean \|ct\| | p95 \|ct\| | max \|ct\| | speed err |
|---|---|---|---|---|---|---|---|---|---|
| scripted follower | 31/31 | 0 | 0 | 11.09 s | 158 kJ | 0.075 m | 0.255 m | 0.314 m | 0.42 m/s |
| **PPO tracker, seed 1** | **31/31** | **0** | **0** | 11.22 s | 184 kJ | **0.029 m** | **0.071 m** | 0.126 m | 0.36 m/s |
| PPO tracker, seed 2 | 31/31 | 0 | 0 | 11.34 s | 208 kJ | 0.042 m | 0.116 m | 0.440 m | 0.34 m/s |

**G6 passes on the transfer question.** A policy trained only on 1–3 s
imagined fragments tracks 11 s routes in the real simulator with zero
divergences, zero asset contact, and cross-track **2.5x lower than the scripted
driver** (paired: better on 28/31 layouts, seed 1). Its held-out p95 lateral
error of 0.07–0.12 m is far inside the oracle's interim 0.9 m inflation margin
(§7.4), so the planner margin can shrink. Steering never exceeded the clamp.
Two caveats: the tracker spends 15–30 % more engine energy than the follower
(more aggressive throttle to hold the profile — see speed error), and seed 2 has
one 0.44 m excursion. In-model numbers (0.15 m mean) were *pessimistic*
relative to Chrono (0.03 m): the model rollouts are harder than reality here,
the opposite of the exploitation failure the earlier study saw.

Imagined vs Chrono on the same routes with the same policy (n=31): time-to-end
corr **0.978**, imagined 10 % fast (9.98 vs 11.22 s); energy corr 0.76,
imagined 107 vs Chrono 184 kJ — Chrono under the PPO tracker is itself 20 %
above the recorded driver's 151 kJ, so the power head's shortfall vs a
closed-loop Chrono drive is ~40 %.

## Chrono validation of the candidate ranking — `wp4_chrono_candidates_v1` (2026-09-04, newton)

The same 185 (episode, candidate) pairs the scorer imagined (`wp4_scores_tracker_v1`)
were driven in Chrono by the PPO tracker (seed 1), plus the scripted follower as
bracket; 370 runs, 18 min on 10 procs. Comparison script:
`scripts/traverse_wp4_compare_chrono.py`.

- Chrono: **185/185 candidates completed, zero asset contact, zero rollover** for
  the tracker (all six sweep families incl. `fast` at 9 m/s and `wide_berth`).
- Time-to-end: imagined vs Chrono corr **0.989**, MAE 1.17 s, imagined 10 % fast
  (10.76 vs 11.92 s). Per-episode rank agreement across candidates (Spearman) 0.92;
  picking the fastest candidate in imagination matches Chrono on **28/31** layouts
  (mean regret 0.02 s, max 0.50 s). Caveat: the "fast" sweep wins most layouts in
  both, so this is partly a trivial agreement.
- Energy: corr 0.65, Chrono **1.62x** the imagined value (176.5 vs 108.9 kJ). With
  the combined objective time + energy/10 the imagined pick matches Chrono on only
  **10/31** layouts (Spearman 0.30), mean regret 0.97 s-equivalent, max 4.7. The
  candidates are near-tied on that objective, and the power head's scale error
  plus its 0.65 correlation is enough to reshuffle them. **Energy cannot enter
  plan selection until the power head is recalibrated** (open item 2).
- Media: `artifacts/traverse/media/planner_imagination_{grid,single}.mp4`
  (`scripts/traverse_wp4_render_imagination.py`): six held-out layouts, every
  candidate tracked inside the model, Chrono time shown alongside at the end.

## Where this leaves the gates

- **G6 (tracker):** built, in-model numbers above; Chrono continuous tracking
  not run. Its p95 lateral error (0.52–0.55 m in-model) is below the oracle's
  interim 0.9 m margin, pending Chrono.
- **G5 / Planner-C:** the rollout scorer exists and its time prediction is
  calibrated on held-out layouts; energy needs calibration; costmap head +
  search (Planner-B) still not built — candidates come from the privileged oracle.
- **ẑ₂ branch:** closed on evidence this session (WP2 notes addendum); the scorer
  indexes the t=0 scene map along the imagined trajectory, as §19 prescribed.

## Open

1. ~~Chrono evaluation of the tracker (real G6) and of the same candidate set~~
   done (sections above): tracker transfers, time ranking validated, energy not.
2. Power-head calibration (fit a scale on the replay residual, or retrain on
   closed-loop rollouts) before energy enters plan selection in absolute terms.
3. Static-scene assumption still unmeasured.
4. Fragment starts inherit the recorded driver's error; a start-perturbation
   curriculum would test recovery explicitly.
5. Test split untouched throughout.
