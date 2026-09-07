# Energy-efficient imagination planning and a controlled crater test

Date: 2026-09-07. Status: proposed experiments; this document launches no jobs.
Requested by the user after the section 33 closeout. These are two independent questions:

1. Does NRD imagination choose lower-work routes than an inexpensive cost predictor under the same mission constraints?
2. Can the existing NRD learn an obvious terrain-dependent trap when we deliberately simplify and overfit the problem?

The second is a model/representation diagnostic. Passing it would not establish superiority over an A* planner with a terrain traversability cost.

## Evidence and scope

- Notes section 9: the full sampling-plus-NRD pipeline beat plain A* by about 19% in the chosen cost and 41% in reported energy, taking about two seconds longer. Much of this gain was available to a cheap scorer; the original candidate sets were not completely matched.
- Sections 10.5/10.8: the genuinely shared-bank NRD advantage over regression was modest, with a feature-grouped interval including zero.
- Sections 11 and 13.12: cost-ranking results vary by arena. The favorable cost-only analysis on old sealed arenas is exploratory, not a fresh confirmation.
- Section 13.13: the fresh feasibility test did not establish an imagination advantage. Preserve that negative result.
- The newer [individual traces](../../../artifacts/traverse/wp7_stall_diag/rollouts/README.md) show an identical hill crossing failing at commanded 4 m/s and passing at 6 m/s, and a launch failure escaping with stronger throttle. They support separating an obvious trap from momentum/contact threshold cases. The other conversation itself was not accessible; this plan uses its available local trace artifacts.

All previously inspected arenas f101-f111 are development data for any new claim. Keep historical test results intact and label any re-analysis exploratory. Collection and Chrono remain on Newton; training remains on the AMD cluster. Do not overwrite the ongoing rollout/video work.

## A. Establish the energy-planning claim

### A0. Freeze the physical quantity and mission before comparing models

The current runner integrates engine output torque times motorshaft speed. This is signed mechanical shaft work, not measured fuel consumption. Audit the selected engine/transmission, timing, units, negative-power intervals, and the agreement between the recorded power series and integrated work before new claims. Chrono's transmission motorshaft-speed accessor returns the speed communicated to the engine; its name alone is not evidence of a shaft mismatch.

Report positive shaft work `W+ = integral max(torque * speed, 0) dt`, negative shaft work separately, and legacy signed work as a secondary measure. Call the result mechanical-work efficiency unless a validated fuel-consumption model is added. Audit cached records before assuming this change leaves rankings unchanged; recollect only if the needed quantities are absent.

Primary mission: minimize positive shaft work while reaching the same goal by a fixed deadline, with the same contact/clearance requirements and final speed requirement. Include launch and braking to the terminal state. This prevents rewarding a route merely for finishing slower or carrying unaccounted kinetic energy through the goal.

For the existing-data pilot, define deadlines from a fixed nominal reference path and commanded-speed rule, using inputs available to all planners. Select the reference rule and any deadline sweep on development data, then freeze them. Keep legacy `time + signed_work/10` secondary. Do not select an energy weight or terrain after observing which makes NRD win.

### A1. Isolate ranking with the existing recordings

Use identical route-and-speed candidate banks and identical execution controllers. First provide all scorers the same Chrono-confirmed feasible subset to isolate ranking; this is explicitly an oracle diagnostic, not the deployed planner. Do not also give them the true deadline-compliant subset: deadline prediction remains part of their decision.

Compare:

| Arm | Information and purpose |
|---|---|
| Plain A* with its declared speed profile | Reproduce the original system baseline. |
| A* route/profile sweep plus calibrated analytic work estimate | Strong classical baseline including grade, rolling resistance, acceleration/braking and an appropriate powertrain efficiency approximation. Fit any coefficients on training data only. |
| Same bank plus direct learned time/work predictor | Primary inexpensive competitor: route profile, terrain, start state, requested speeds; train on the same episodes as NRD. Include nominal acceleration and an adequate sequence encoder. |
| Same bank plus NRD and the fixed tracker in imagination | Predict time and cumulative work along the evolving physical state. |
| Existing NRD-plus-geometry-floor configuration | Preserve the deployable baseline, but ablate the floor and give comparison methods access to the same floor information. A hybrid win is a hybrid claim. |

Use the same available terrain source for the primary comparison. If a prior heightmap is used for projection, disclose it and allow it for both methods. Report camera-decoded and privileged-geometry versions separately. Neither arm receives recorded future actions or states when selecting a route.

Report actual work, completion/deadline compliance, paired work differences on common compliant layouts, and regret against the best Chrono candidate meeting the deadline. Report every failure/abstention separately; never count a short failed drive as an energy saving. Add per-route time/work error, prediction error at the selected candidate, and inference time. Use paired uncertainty grouped by physical terrain feature/layout, with arena-level variation shown separately.

Deliverable: one all-arena table and the time/work trade-off curves, including unfavorable arenas. Identify whether any advantage comes from path geometry, speed transitions, or cost prediction. Avoid another broad dynamics-training grid.

### A2. A bounded planning pilot, only if A1 supports it

Develop approximately 30 missions across at least four development terrain instances, with physically motivated choices: short steep versus longer gentle routes, curved versus straight routes, and ordered hill/turn/descent sequences requiring acceleration and braking. Choose the family using vehicle/task requirements and Chrono outcomes, without using which model wins as a filter.

Generate and freeze a common candidate bank per mission with per-layout random seeds. Cross-score every candidate, including the union of candidates from any adaptive search. Replay selected paths with the same actual controller in Chrono. Include a direct cost model with access to sequence order and initial velocity so the NRD comparison is demanding.

First judge with a common feasibility gate, then restore the actual perception/feasibility pipeline and report total mission performance. The oracle-gated result alone cannot support a complete planner claim.

Proposed practical continuation gate, fixed before this pilot: at least 5% paired work reduction against the strongest inexpensive baseline at comparable deadline/completion performance, with the paired interval supporting a reduction and benefit across multiple terrains. This 5% is a proposed engineering relevance threshold, not an existing result. A miss means report the result and stop scaling this branch.

### A3. Fresh confirmation and the contribution of imagination

After development, freeze models, inputs, mission rules, candidate counts, scoring, fallback, and seeds. Choose the fresh evaluation size from development variance before collecting; aim for at least six independent terrain seeds and roughly ten missions per seed, increasing only by a predeclared precision rule. Look once.

Run the identical-bank comparison first. Then compare candidate methods at a matched planning-time budget and report amortized and per-query cost. The existing approximately 19-second planner is not a demonstrated 1 Hz planner.

Show two or three representative predicted-versus-Chrono cumulative-work/speed traces, selected by a declared rule, alongside aggregate results. The scientific claim requires better decisions than the direct cost predictor, not merely more realistic-looking trajectories. Energy-aware planning itself already exists; the additional contribution must be that simulating evolving vehicle state improves the route/profile choice.

## B. A deliberately overfit, unambiguous crater

### B0. Define what the simplified model should learn

The user's hypothesis is plausible as a learnability test: a large, visible terrain feature with consistently unsuccessful escape attempts should be easier than a marginal climb whose outcome changes with momentum and contact.

But `delta_v = 0` preserves velocity. NRD must learn the actual transition: approach/entry motion, loss of uphill progress, and failure to escape under the tested controls. A trapped vehicle may roll back or rock and its wheels/engine may keep spinning; do not set every state delta or the whole crater's speed to zero. A deep crater is not necessarily a stationary obstacle at its near edge. Learn the stopping/turnaround region observed in Chrono.

The current model uses `RGB-D -> frozen spatial feature map -> ego crop at predicted pose -> local token`, with state/action history predicting physical-state deltas. The local token is re-indexed from a static map; success here would establish local visual conditioning, not prediction of future images or recurrent global z2.

### B1. Verify a trap in Chrono before collecting training data

Start with one smooth, broad bowl and a driveable surrounding plain. Minimize small roughness and remove houses/obstacles from the crossing so failure is dominated by the crater. Make entry driveable and the exit wall demanding without routinely triggering rollover or invalid terrain/contact behavior. A shallow/passable version and a flat version share the approach corridor. Add an exterior detour.

The existing generator both caps crater depth from `slope_cap` and smooths steep gradients. Increasing requested depth alone can leave the actual crater driveable. Inspect the final quantized height field used by Chrono; a dedicated controlled bowl may be needed. Keep it height-field compatible and visibly resolved in RGB-D and the local feature crop. Check depth saturation, normalization, camera coverage and crop indexing; use one training normalization convention.

Proposed geometry search cap: three bowl settings, up to 96 short verification runs total. Sweep approach headings, speeds 2/4/6/8 m/s, and both the existing tracker and independently specified throttle/steering schedules, including sustained high throttle. Document the allowed gears, steering, speeds and 40-second post-entry observation horizon. Test escape directions supported by that control envelope; if reverse is excluded, explicitly exclude reverse from the claim. Repeat selected boundary cases with small initial-condition perturbations.

Gate: direct entries do not escape in any verified trial; the shallow/flat controls and an exterior detour succeed. A rollover, obstacle contact, aborted simulation, or off-route cutoff does not count as the desired stall. If it only stops for one controller or speed, revise the geometry within the cap. Say "no escape observed within this tested envelope," never "physically impossible under all actions."

### B2. Collect a tiny corpus with real interventions

After B1, a starting corpus is approximately 96 episodes: three surface conditions (deep/shallow/flat), four headings, four speed levels, and two driving/control families, plus a few verified detours. Reuse B1 recordings where their schema is adequate. This is a small controlled family built around one arena, not a broad new terrain collection.

First select 24 complete training episodes, balanced across the three surfaces and including moving-success examples with sustained throttle. Intentionally overfit these exact episodes. Hold out whole episodes/control schedules for the next stage; do not split overlapping windows from one trajectory between training and evaluation.

For schedules used as a control intervention, actually execute each schedule in Chrono on each surface. Do not perturb action inputs and keep the old physical targets: that is an augmentation assumption, not a measured counterfactual. Adaptive-controller episodes remain useful training data, but their recorded future actions are not available at planning time.

Record state, actions, pose, power, initial conditions, true/camera terrain, valid length, and termination reason. Keep at least 10 seconds of actual post-trapping attempts when possible; disable the stall early-abort for this diagnostic. Do not turn missing frames into synthetic stationary data. Match initial states on the common flat approach; do not demand physically impossible identical states inside different terrain.

### B3. Overfit before generalization or planning

Keep the 17-D state and current architecture initially. Train cropper and NRD with the encoder frozen. Use balanced moving/approach/trapped windows, a 4-second rollout/progress loss and valid recorded targets; test complete rollouts as well as local errors. One main configuration, followed by a second seed if it passes, is sufficient. If it fails, use the diagnostic arms below instead of sweeping dozens of hyperparameters.

Proposed overfit gate: all 24 training episodes have the correct escape/completion outcome from their starting state under their known control schedule; successful controls keep moving; trapped episodes do not drift into an imaginary escape. Compare the location/amount of lost progress against Chrono. Score near-zero speed only during actually stationary recorded intervals, using absolute body speed, and keep rocking/non-escape as a separate outcome. Freeze numerical trajectory tolerances from repeatability and map resolution before judging the model.

This same-episode test deliberately uses recorded controls to check reproduction; it is not foresight evidence. Next evaluate whole held-out schedules on the same arena, including externally specified actions that never consulted the eventual outcome. Then put the fixed tracker in imagination and test from before entry. Show stage-specific counts; a pass while seeded inside a stall is weaker than a pass from the approach.

### B4. Establish whether the visual terrain feature matters

Use matched training capacity and data for three diagnostic arms:

| Input | Interpretation |
|---|---|
| State/action history + normal local visual token | The intended NRD model. |
| State/action history with local terrain token removed | Detect whether the visual input is unnecessary or ignored. |
| State/action history + privileged local height patch | Separate visual encoding/crop failure from inability of the dynamics model to fit this regime. |

Compare matched flat-approach states/actions with the deep versus shallow/flat scene. Use actually simulated terrain variants as the reference. Change terrain-associated inputs consistently, including the projection heightmap; report a token-only swap separately as a network-sensitivity probe. A mislabeled scene, wrong episode map, absolute coordinate, or already-stuck initial state must not supply the answer.

After exact overfit, relocate/rotate the crater and vary start distance in a small test to check whether failure follows the local feature rather than the original coordinates or elapsed drive time. Failure on relocation narrows the result to memorization; it does not erase a successful overfit diagnostic.

Decision tree: privileged geometry fits but visual NRD fails -> inspect representation/cropping, then allow one encoder fine-tune. Neither fits even the 24 episodes -> inspect targets/integration and physical-state sufficiency; do not proceed to planning. Both fit but removing vision changes nothing -> improve matched pre-entry controls before claiming visual conditioning.

### B5. A small planning demonstration, conditional on B3/B4

Give the same planner a direct path through the crater and an exterior detour. Keep both in the candidate bank for the diagnostic; an upstream geometric filter must not remove the direct path before NRD can evaluate it. The NRD rollout, without a manually assigned crater stopping rule, should predict failed direct traversal and successful detour, leading to a successful Chrono detour.

Also run a geometry/traversability-aware A* baseline. On an intentionally obvious trap it may solve the task just as well. That is an expected outcome: the contribution of B is demonstrating learned visual terrain-to-motion behavior, not winning against a hand-coded obstacle rule. It supplies a useful prerequisite if we later return to marginal, speed-dependent terrain.

## Execution order and limits

First do A0/A1 (metric audit and existing-data cost analysis) and B1 (bounded physical verification). Only B1 success triggers the tiny B2/B3 overfit exercise; only positive A1 evidence triggers a larger energy pilot. The two questions can progress independently and neither changes the previous negative feasibility result.

No new MPC, policy optimization inside imagination, large data campaign, or broad parameter grid is part of these first steps. Record a decision after each gate. Proposed thresholds and resource counts are defaults to freeze in the execution manifest before results, not permission requests or results already achieved.

## Implementation pointers and external context

- `src/nedm/traverse/terrain.py`: crater construction, depth cap and slope repair.
- `src/nedm/traverse/map_crop.py`, `nrd_model.py`: local visual token and delta-state dynamics.
- `scripts/traverse_wp3_chrono_eval.py`: actual controller loop, recording, power and termination.
- `scripts/traverse_wp2_train_map.py`: event sampling, sequence loss and validation.
- `scripts/traverse_wp7_cheap_predictor.py`: matched direct route/time/work baseline.
- `scripts/traverse_wp8_stall_rollouts.py`: individual physical traces; current uncommitted work should be preserved.
- [Chrono transmission API](https://api.projectchrono.org/10.0.0/classchrono_1_1vehicle_1_1_ch_transmission.html) and [engine API](https://api.projectchrono.org/9.0.0/classchrono_1_1vehicle_1_1_ch_engine.html): shaft interface semantics; verify against the installed version for execution.
- [Energy-Optimal Ground Vehicle Trajectory Planning on Deformable Terrains](https://impact.ornl.gov/en/publications/energy-optimal-ground-vehicle-trajectory-planning-on-deformable-t/): energy-aware off-road planning precedes this study; the comparator must include energy estimation.
