# Broader-terrain FDM / MPPI experiment

This campaign tests whether the existing RGB-D reference-conditioned FDM can support approximately 240 m HMMWV traversals. Data collection, training and protected physical evaluation are complete. The method transfers partially: selected RGB-D planning safely completes 4/6 unseen arenas versus 3/6 for the matched blank-image control, but the complete time/risk/energy milestone is not met. Actual Chrono replay videos, forecasts and telemetry are linked below.

## Data and compute

- 36 heightmap arenas, each 240 m square: 24 training, six validation, six protected test. Six families cover rolling hills, ridge passes, cross-slopes, valleys, rough ground and mixed obstacles. Every arena has flat launch and arrival pads.
- 15 route/speed references per arena, with 2/4/6 m/s commands and up to 180 s per attempt. All outcomes are retained. The verified train/validation cohort has 450 trajectories and 30 global RGB-D maps: 109 goal arrivals, 318 timeouts and 23 rollovers. Of those arrivals, 103 meet the strict safety schema.
- Each of the 30 training/validation arenas has safe fixed-reference controls at all three speeds. All measured safe completions used outer ±44 m reference offsets, so this benchmark establishes feasibility within a restricted route library, not arbitrary path planning.
- Headless Chrono physics and telemetry were separated from the one-time map render. The 450-route campaign completed on AMD in 39 min 05 s with 48 workers and zero collection errors. Both forecast packs were prepared in about 2 min 21 s.
- All eight training arms completed 5,000 optimizer updates on AMD: 4 s and 12 s forecasts, RGB-D and matched blank-image controls, seeds 11 and 29. There are 53,830 training and 13,625 validation anchors, with scene-disjoint splits. Within each seed, sample draws and rotation augmentations match across arms.

The existing AMD Chrono build, PID driver and collection infrastructure were reused. Older datasets were audited, but their different sensor/controller contracts and missing substep channels prevented a silent merge. The main `/home/harry/NeDM` checkout and its ongoing work were left untouched; this effort uses the `traverse_mppi` worktree.

## Model and measurements

`fixed measured RGB-D + causal vehicle history + proposed reference → finite FDM forecast → risk/time/work cost → MPPI reference → native Chrono PID`

For the deployed 12 s model, inputs are RGB-D `[4,512,512]`, history `[16,24]`, proposed command features `[60,5]` and global context `[8]`. Trained outputs are trajectory `[60,4]` (XY and sin/cos yaw), cumulative work `[60,1]`, three event logits `[60,3]` (converted to probabilities for scoring) and signed roll/pitch `[60,2]`. The model is frozen during planning; MPPI optimizes proposed references, and the native PID is not learned.

The model uses registered RGB plus measured depth-derived elevation. Candidate-aligned visual patches retain local terrain detail. It forecasts motion, positive mechanical work, contact, bounded stall, rollover and signed roll/pitch over a finite horizon, without recursively feeding predicted states or images back into the model. Time and work beyond the finite forecast are extrapolation heuristics, not learned whole-route targets.

The data retain 198 rich fields: engine/shaft torque, speed and power; integrated positive/signed/negative mechanical work; per-wheel slip, spin, force/moment, load proxy and deflection; body pose, velocity, acceleration, roll/pitch and rates; controller inputs; and typed suspension spring/shock channels. Ordinary state logging is 20 Hz; mechanical work and contact/attitude peaks cover every 2 ms physics step. Mechanical work is not fuel consumption. RigidTerrain with TMEASY supports vehicle/tire dynamics but does not model soil sinkage or rut formation.

The map is a one-time overhead observation, not an onboard exploration camera. The NN receives no authored terrain heights or obstacle lists. Terrain truth is still used by the simulator and for PID path altitude, and is disclosed in each run protocol.

## Forecast evidence

At the fixed 5,000-update budget, the 12 s RGB-D models show a useful pre-failure vision signal:

| Measurement | RGB-D seeds 11 / 29 | Matched blank seeds 11 / 29 |
|---|---:|---:|
| Pre-onset contact AUROC | 0.914 / 0.924 | 0.776 / 0.756 |
| Pre-onset bounded-stall AUROC | 0.915 / 0.922 | 0.729 / 0.725 |
| Pre-onset 12 s final-position error | 10.14 / 9.12 m | 11.89 / 11.93 m |
| All-anchor 12 s final-position error | 6.00 / 6.28 m | 6.62 / 6.37 m |

Pre-onset contact and bounded-stall ranking improve on all six validation arenas for both seeds. Shuffling scene images worsens the same RGB-D checkpoints. Aggregate motion and energy advantages are modest; the 4 s models do not show an aggregate RGB-D advantage in average final-position error. The 4 s/12 s architectures have different parameter counts, so this is not a capacity-matched horizon comparison.

The originally declared primary checkpoint is seed 11 validation-best at update 4,000, SHA `65945dd3ca79621f5cf2467e0a19f66cdb943faf798565be0affe657132df33f`. Its event calibration is weaker than the fixed-final checkpoint: at the current thresholds, pre-onset contact/stall recall is 63.4%/51.1%. Rollover validation has only four positive routes and cannot support a strong safety claim.

## Why longer planning remains difficult

An initial-state diagnostic evaluates all 90 validation references with the exact deployed scorer. 78 references are safe over their first 12 s, but only 21 complete the whole route safely. Even replacing learned forecasts with measured first-12-second outcomes leaves the unchanged cost choosing unsafe full routes in three of six scenes. This diagnostic truth never enters the deployed planner.

Forecast mistakes also matter: the primary checkpoint assigns contact probabilities 0.083, 0.056 and 0.152 to selected routes that physically contact within 12 s, below its 0.35 threshold. Thus both prediction/calibration and the long-route cost approximation contribute.

Initial fixed-library choices identify safe full routes in 3/6 scenes for the primary checkpoint, 5/6 for the fixed-final checkpoint, and 0/6 for the matched blank checkpoint under the zero-energy cost. These are joins against previously executed exact references, not MPPI rollout success. The subsequent physical plan-once comparison verifies 5/6 safe goals for the final checkpoint at zero energy weight and 4/6 at the default 0.02 coefficient. The original best checkpoint gives 3/6 and 2/6. All 48 plan-once trials and audits completed.

On the ridge validation scene, the initial MPPI choice is a slight refinement of a measured-safe +44 m reference. At the first replan (1 s), it switches to the opposite route family despite the original remaining available and passing its risk limits. Subsequent switches and refinements shrink the detour, leave the original corridor by 3.85 s, and eventually produce a bounded-motion interval ending at 26.65 s while the planner is requesting a stop. This is an abstention/stop-and-go failure, not proof of a spontaneous terrain stall. The original strict bounded-motion schema is retained, with planner-induced stops distinguished in supplemental diagnostics. This is evidence for a route-commitment ablation, not a proven counterfactual fix.

## Physical evaluation

The first full receding-horizon grid exposed a software defect: 15 of 24 runs stopped when a fresh Hermite proposal contained a cusp; the other nine completed simulation but timed out. Invalid proposals are now rejected individually and logged, with strict corruption checks on retained references. The entire declared grid was rerun from immutable online_v8 with unchanged model and costs: all 24 processes and artifact checks passed. The zero/default-energy arms each safely completed 4/6 arenas; higher energy weights 0.2 and 0.5 completed 2/6 and 1/6. The default weight has three paired safe goals, but consumed 6.677% more mechanical work with 0.133% more time. No weight passed the declared energy gate. Weight 0.02 is retained as the declared fallback, with no efficiency improvement claim. The invalid earlier grid remains archived.

At 22:56:01 UTC, the final model, costs, source and 30 comparison tasks were frozen, then protected test data were unsealed for evaluation only. The validation rule chooses plan-once with the fixed-final seed 11 checkpoint, retaining energy coefficient 0.02 despite a failed energy gate. The final plan-once validation energy comparison saves only 0.316% work on four safe pairs and reduces safe completion from 5/6 to 4/6, so it is not an energy-efficiency success. All test scenes and failures are reported below. Energy savings require paired safe arrivals at the same goal; stalled or shortened runs cannot earn an efficiency win.

## Evidence

- Campaign plan: `fdm_diverse_campaign_20260909.md`
- Campaign root: `artifacts/traverse/fdm_diverse_v1_20260909`
- Terrain previews and profiles: `geometry/`
- Complete collection audit: `full_collection_audit_412066.json`, `full_route_support.json`
- Forecast reports: `reports/full_learning_pair_v1/`, `reports/full_learning_analysis_v2/`
- Exact initial-reference diagnostic: `reports/initial_reference_v1/`
- Failed-grid integrity audit: `reports/online_full_validation_grid_v2_cohort_02/`

The protected phase compares five arms on all six test scenes: selected RGB-D time/risk and energy-aware, matched blank time/risk and energy-aware, and the original best-checkpoint receding planner. The frozen fixed-reference baseline is −44 m offset at 6 m/s, chosen using validation; best-of-15 test routes are only a hindsight feasibility bound.

## Protected results

All 30 physical trials completed and passed the checksum, source/runtime, measured-anchor and raw-telemetry audits. No model, threshold, cost or policy was changed after the protected freeze.

| Frozen arm | Safe full goals / six arenas | Unsuccessful trials |
|---|---:|---:|
| Selected RGB-D, time/risk, plan once, final checkpoint | 4/6 | 2 |
| Selected RGB-D, energy-aware, plan once, final checkpoint | 4/6 | 2 |
| Matched blank, time/risk, plan once, final checkpoint | 3/6 | 3 |
| Matched blank, energy-aware, plan once, final checkpoint | 3/6 | 3 |
| Original RGB-D best checkpoint, receding, time/risk | 4/6 | 2 |
| Predeclared fixed −44 m / 6 m/s reference control | 3/6 | 3 |

The RGB-D plan-once model succeeds on cross-slopes, mixed obstacles, ridge passes and the valley network. It fails on rolling hills and rough mosaic. The blank model succeeds on mixed obstacles, rolling hills and rough mosaic: RGB-D wins three scenes, loses two and ties one. A net gain of one arena is useful evidence, but not strong proof of broad superiority. Original receding planning also reaches 4/6; plan-once is not a universal improvement.

All six arenas have at least one physically safe route among the 15 reference controls, with 21/90 safe reference completions overall. The remaining learned-planner failures therefore cannot be attributed simply to every route being impossible. This best-of-15 information is a hindsight feasibility bound, not an input to the planner or a deployable baseline.

Across the four paired safe RGB-D completions, the energy term reduces mean per-scene mechanical work by **0.3256%**, with **0.4782% less time**. This is far below the predeclared 5% work-saving target. The matched blank model reduces work by 0.4489% and increases time by 1.8854% on three safe pairs. Tiny changes in either arm do not establish useful energy-aware optimization.

Protected initial-state forecasts show some visual value: 12 s final-position error is 5.07 m for RGB-D versus 7.12 m for blank; work MAE is 65.7 versus 112.7 kJ. Contact ranking AUROC is 0.935 versus 0.753. At the frozen 0.35 threshold, RGB-D detects six of nine known-positive future-contact cases forecast from launch and misses three. At the frozen 0.5 bounded-stall threshold, both models miss all five future bounded-stall positives forecast from launch despite strong RGB-D ranking AUROC. Risk calibration remains a real limitation.

Across 4,551 overlapping pre-first-failure test windows (not independent scene trials), RGB-D contact/stall recall is 73.8%/41.6%, versus 51.5%/19.1% for blank. Counting already-failed windows would inflate RGB-D recall to 95.5%/89.0%, so those aggregate numbers are not used as evidence of advance warning. The RGB-D pre-first-failure final-position error is 8.52 m versus 9.95 m for blank; shuffling scene maps worsens the RGB-D error to 13.02 m. This supports real scene use while keeping its planning and calibration limitations explicit.

## Milestone assessment and next target

- **Diverse reusable physical data: met.** The cohort has 540 reference trajectories and 36 maps with separate training/validation/test scenes, flat launch pads and rich vehicle telemetry. All training runs used AMD.
- **Complex traversal count: limited pass.** The 4/6 protected safe-completion threshold declared before full training in [evaluation_plan_v1.json](../../../artifacts/traverse/fdm_diverse_v1_20260909/evaluation_plan_v1.json) is reached. The restricted path library, small scene count and modest advantage over blank/fixed-side controls limit the claim.
- **Vision-conditioned prediction: positive but incomplete.** Matched image controls and held-out forecasts show useful information; this does not produce reliable calibration or dominance on every scene.
- **Energy-aware benefit: not met.** The measured work reduction is about 0.33%, not 5%; increasing the coefficient on validation often reduced safe completion.
- **Reliable joint risk/time/energy planner: not established.** Initial missed hazards, finite-horizon extrapolation and route-switching/abstention behavior remain unresolved.

The next focused experiment should supervise whole-reference remaining time/work and failure outcomes, and collect varied speed profiles and bounded path perturbations around feasible references. The current dataset mainly covers three cruise speeds on five geometric families; arbitrary MPPI refinements are only geometrically constrained, not certified to be in the training distribution. Any further tuning should use training/validation or a newly sealed cohort, never these already opened test results as fresh validation.

## Delivered evidence

Core freeze: `artifacts/traverse/fdm_diverse_v1_20260909/protected_test_freeze_v1.json` (SHA `29684ce59cdd68e9b1f7310dcd2e63c55d021f6ff6712b83f99d286722efeb33`). Protected physical report: `reports/online_protected_test_v1_cohort_01/`. Protected forecasting is finalized in [the offline report](../../../artifacts/traverse/fdm_diverse_v1_20260909/reports/protected_test_offline_final_v2/report.md), including exact recomputation of all 90 reference outcomes and six saved-prediction summaries (angular reductions agree within 0.000028 degrees). [The full 30-trial outcome figure](../../../artifacts/traverse/fdm_diverse_v1_20260909/reports/protected_test_visuals_v2/all_30_trials.png) retains every failure and every safe energy pair. The actual Chrono replays below also passed exact physical and decision parity checks.

## Actual Chrono replays

The [predeclared media selection and cadence amendment](../../../artifacts/traverse/fdm_diverse_v1_20260909/demo_selection_rule_v2.json) select the first paired safe scene and the first failed energy-aware scene in lexicographic order. The complete 30-trial matrix above retains the full result set.

| Replay | Actual Chrono video | Forecast, candidate references and actual trace |
|---|---|---|
| Cross-slopes, time/risk; safe goal in 41.25 s | [5 Hz video](../../../artifacts/traverse/fdm_diverse_v1_20260909/demos/protected_v1/cross_time/report/selected_trial/actual_chrono.mp4) | [Overview](../../../artifacts/traverse/fdm_diverse_v1_20260909/demos/protected_v1/cross_time/report/selected_trial/overview.png) |
| Cross-slopes, energy-aware; safe goal in 41.20 s | [5 Hz video](../../../artifacts/traverse/fdm_diverse_v1_20260909/demos/protected_v1/cross_energy/report/selected_trial/actual_chrono.mp4) | [Overview](../../../artifacts/traverse/fdm_diverse_v1_20260909/demos/protected_v1/cross_energy/report/selected_trial/overview.png) |
| Rolling hills, energy-aware; contact/blockage and timeout at 180 s | [Full-duration 1 Hz video](../../../artifacts/traverse/fdm_diverse_v1_20260909/demos/protected_v1/rolling_failure_1hz/report/selected_trial/actual_chrono.mp4) | [Overview](../../../artifacts/traverse/fdm_diverse_v1_20260909/demos/protected_v1/rolling_failure_1hz/report/selected_trial/overview.png) |

All three passive-camera replays reproduce all 241 stored physical arrays and every planning decision exactly; only wall-clock timings are excluded from outcome equality. The videos contain 208, 207 and 181 actual native Chrono frames, respectively, with their measured timestamps preserved. The failure clip samples at 1 Hz over the full 180 s physical rollout. Its measured failure includes obstacle contact and bounded blockage, with no planning abstention; the camera shows the final vehicle position against a cube obstacle.

The plots distinguish the full commanded reference, the 12 s learned forecast and the complete actual trace. Each selected configuration makes one launch MPPI decision, followed by native PID execution. [Measured work, power, slip, roll and pitch](../../../artifacts/traverse/fdm_diverse_v1_20260909/demos/protected_v1/cross_energy/report/telemetry.png) remain available alongside each replay. Camera rendering ran on AMD; verified plotting and timestamp-preserving video encoding ran locally.
