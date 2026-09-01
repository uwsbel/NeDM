# Critical Review: NRD HMMWV Traversal Study Plan

**Reviewed plan:** `NRD_hmmwv_traversal_study_plan.md`  
**Review date:** 2026-08-31  
**Recommendation:** Conditional go for WP0 and a small oracle-stack prototype; do not begin the 200-episode pilot until the blocking issues below are resolved.

## 1. Overall judgment

The proposed study is a strong research direction, but the plan is not yet implementation-ready or fully experimentally identified.

- Research idea: approximately 8/10
- Experimental identification: approximately 6/10
- Implementation readiness: approximately 5/10

The hierarchy, held-out layout splits, oracle brackets, staged collection, alignment gates, and direct Chrono transfer tests are all good choices. The main problems are unresolved start/goal interfaces, an unreachable goal specification, a planner that does not yet encode HMMWV feasibility, confounded terrain-vision claims, and substantial mismatches with the current data and tracker infrastructure.

## 2. Critical blockers

### 2.1 The planner has no defined source for its A* start and goal cells

The plan says task-relevant information exists only in the overhead image, but v1 supplies a relative goal vector and explicitly defers the `z2` localizer. The costmap head outputs only occupancy and terrain cost; it does not output vehicle position, yaw, or goal position.

A relative goal vector cannot be placed on a global image-coordinate map without a known vehicle pose and a precise coordinate convention.

Resolve this by choosing one explicit v1 contract:

1. **Recommended privileged-v1 contract:** provide `start_pose_world` and `goal_pose_world` to the planner. State honestly that obstacle layout comes from vision while localization and goal placement remain privileged in v1.
2. **Vision-localized contract:** add vehicle-center, vehicle-yaw, and goal heatmap heads and make their errors acceptance gates.

The recommended choice is consistent with the plan's feasibility-first charter and avoids quietly depending on an interface that does not exist.

### 2.2 The house-center goal is physically unreachable

The house is approximately 5 x 4 m, its center is the goal, it has collision enabled, and its footprint is hard-blocked and inflated in the oracle costmap. The HMMWV cannot physically reach within 2 m of the house center.

Replace the house-center goal with a collision-free approach pose or approach ring outside the house. The oracle can choose the final approach pose using terrain slope, clearance, and vehicle heading. Success should be measured relative to that approach pose, not the house center.

### 2.3 The energy costmap is inconsistent with the energy model

The plan defines energy as a function of slope along the direction of travel and speed, but the predicted map contains one scalar energy value per cell. Energy through a cell is directional: uphill, downhill, and cross-slope traversal are not equivalent.

Use directional edge costs, for example:

\[
c(i \rightarrow j) = w_d d_{ij} + w_e\,\hat e(\nabla h \cdot \hat t_{ij}, v_{ij}, a_{ij}).
\]

The plan also uses 8-connected A* followed by spline smoothing. That does not guarantee:

- a feasible HMMWV turning radius;
- full-footprint collision clearance;
- acceptable cross-slope traversal;
- collision freedom after spline smoothing.

At minimum, collision-check the full vehicle footprint and curvature after smoothing, rejecting or repairing invalid paths. Prefer a heading-state lattice or Hybrid A* for the privileged oracle. Ordinary grid A* can remain a useful naive baseline.

### 2.4 The claimed terrain-from-depth result is confounded by the fixed map

The terrain is fixed, and the terrain-cost label is identical for every episode. A model can therefore locate the roof marker, use its pixel position as an absolute-position code, and retrieve memorized position-dependent dynamics or terrain cost without using depth geometry at all.

On one fixed map, RQ2 should be described as:

> Does visual localization on a known terrain map improve position-dependent dynamics prediction over a state-only model?

It should not be presented as evidence that the model learned general terrain geometry from depth unless depth-specific ablations support that conclusion.

Required ablations should include:

- RGB-only input;
- depth-only input;
- depth zeroed or shuffled;
- `z2` shuffled between layouts;
- vehicle-location information masked or shuffled;
- state plus privileged `(x, y, yaw)` or a privileged local-terrain patch as an oracle upper bound.

### 2.5 A frozen global reconstruction latent may not preserve the needed information

The proposed 128-D global latent must simultaneously preserve:

1. small obstacle positions and types;
2. vehicle position and yaw;
3. static terrain/layout information;
4. features useful for physical-state prediction.

Study 1 showed that a plain reconstruction objective can erase a small moving foreground object while retaining a deceptively good aggregate image loss. The HMMWV scene creates the same risk for small rocks and the vehicle marker.

Encoder fine-tuning should therefore depend on downstream occupancy and localization performance, not only on whether latent prediction plateaus. Add analytically generated auxiliary representation losses during warm-up:

- obstacle occupancy or class masks;
- vehicle-center heatmap and yaw;
- class/foreground-weighted RGB reconstruction;
- elevation reconstruction with its own normalization and loss.

A safer architecture would factor the representation into a mostly static `z_layout` and a dynamic `z_vehicle`, or preserve a spatial feature map rather than compressing the entire arena immediately into one global vector. If the single-latent architectural claim is retained, it must be tested against one of these less constrained baselines.

### 2.6 The proposed tracker does not match the current reference/reward interface

`PlanCandidate` contains geometry and a speed profile, but the current HMMWV tracking reward compares the full predicted state against recorded reference states. Arbitrary planner routes do not provide reference roll, tire forces, spindle speeds, or drive power.

Define a new geometric tracking reward based on:

- cross-track error;
- heading error;
- speed error;
- action rate and simultaneous throttle/brake penalties;
- rollover and safety penalties.

Full-state reference error should be removed or restricted to fields, such as forward speed, that the planner actually specifies.

There are also two concrete specification errors:

- The proposed observation is 38-D, not approximately 39-D: `3 + 10*3 + 1 + 1 + 3 = 38`.
- The existing tracker defaults to `action_repeat=5`. With an NRD timestep of 0.05 s, retaining that value would produce 4 Hz policy control. Token-rate control requires `action_repeat=1`, or a separately specified model/control-rate pair.

Finally, the hierarchy does not by itself reduce tracker imagination to a 1-3 s model horizon. During a 20-60 s imagined episode, `z1`, `z2`, and pose are still recursively propagated for the whole episode. Train the tracker on randomized 1-3 s fragments initialized from real context windows, then evaluate continuous tracking in Chrono.

### 2.7 Storage and infrastructure costs are underestimated

The full tier contains approximately 0.4-1.6 million frames, not merely `10^5-10^6`:

- `1000 episodes * 20 s * 20 Hz = 400,000` frames;
- `2000 episodes * 40 s * 20 Hz = 1,600,000` frames.

At 256 x 256 with RGB uint8 and depth uint16, each frame is 327,680 bytes. Raw storage is therefore approximately:

- 122 GiB for the smallest full tier;
- 488 GiB for the largest full tier.

That excludes segmentation, manifests, and duplication into processed caches. The current preprocessing path stores uncompressed NumPy memmaps, so static-background compressibility does not help unless a new storage representation is implemented.

Before pilot collection, define:

- separate RGB and depth arrays and their dtypes;
- compression/chunking and random-window access strategy;
- whether processed caches duplicate or reference raw frames;
- measured loader throughput at the intended training batch size;
- peak disk requirement during preprocessing.

The current infrastructure also needs more extension than the work package suggests:

- `ConvEncoder`, `ConvDecoder`, and frame conversion are hard-coded for three-channel uint8 RGB;
- preprocessing supports only train/validation splits, not 70/15/15;
- the joint NRD autonomous rollout evaluator is still oriented around double-pendulum metrics and lacks HMMWV pose integration;
- segmentation masks and layout manifests need split-aligned loading and evaluation support.

WP2 should be treated as a real RGB-D dataset/model subsystem extension, not a small `--frames` change.

## 3. Additional scientific and implementation concerns

### 3.1 Drive power should probably be an auxiliary output

Powertrain shaft torque multiplied by shaft speed is signed mechanical power, not necessarily consumed fuel or electrical energy. It may be negative during braking and omits idle losses, drivetrain losses, and other consumption terms.

Record the raw quantities from day one, but consider predicting power with a separate supervised auxiliary head rather than recursively feeding it as a physical-state variable. Report positive drive work and braking work separately unless the chosen powertrain model supplies a more defensible consumption measure.

The pilot energy regression should include at least speed, longitudinal acceleration, slope along heading, and steering/curvature. A model using only slope and speed is likely to assign acceleration energy to terrain.

### 3.2 Planning margins must include tracking and perception uncertainty

Obstacle inflation should not be an arbitrary fixed margin. It should include:

- half the vehicle footprint;
- costmap/pixel localization uncertainty;
- the tracker's held-out p95 lateral tracking error;
- a small fixed safety allowance.

This links G5 and G6: the final planner cannot be declared safe until the tracker error envelope is known. After smoothing, every candidate must be collision-checked using the same inflated footprint.

### 3.3 Segmentation cannot be pilot-only if it defines final metrics

G1 and G5 use per-class recall and small-object evaluation against segmentation masks. Masks must therefore be recorded for all validation and test layouts used by those gates, not just optionally for the pilot tier.

### 3.4 One-step `z2` persistence is an unusually strong baseline

At 20 Hz, the global scene is almost static and the vehicle moves only a small number of pixels. Aggregate latent prediction can look good while vehicle localization is wrong.

G2 should include task-space metrics in addition to latent MSE/cosine:

- predicted vehicle-center and yaw error;
- per-class object permanence;
- local terrain feature error at the predicted vehicle position;
- improvement over a static-layout plus constant-velocity vehicle baseline.

### 3.5 The end-to-end RL baseline must not be a strawman

A "modest budget" baseline is acceptable only if it is labeled as a low-budget bracket. A headline hierarchical-versus-end-to-end comparison should match environment steps, wall-clock compute, observation privilege, and reward information. Otherwise, defer the end-to-end baseline until the hierarchical stack passes G7.

### 3.6 Throughput comparisons must be like-for-like

Report at least three numbers:

1. Chrono physics without rendering versus NRD transition-only;
2. Chrono plus RGB-D rendering versus data collection;
3. complete tracker-environment policy steps per second, including reference and reward work.

Comparing decoder-off NRD against Chrono with continuous rendering can overstate the relevant training speedup because the reduced-observation tracker does not itself consume a rendered image at every policy step.

### 3.7 Acceptance gates need fixed quantitative thresholds

Terms such as "with margin," "near oracle," "bounded RMSE," and "low collision rate" are not executable acceptance criteria. Use the pilot to set thresholds, freeze them before full-data training, and leave the test split untouched until final evaluation.

Report confidence intervals, not only point estimates. Approximately 100 scenarios will give wide intervals for success and collision rates, so use paired bootstrap intervals and publish failure categories.

## 4. What the plan gets right

The following design choices should be retained:

- hierarchical planning and tracking rather than relying entirely on long-horizon model-based RL;
- held-out splits at the layout level;
- true-map oracle and straight-line comparison brackets;
- smoke, pilot, and full collection tiers;
- explicit camera/state alignment and settle-at-height tests;
- direct Chrono transfer as the decisive control evaluation;
- `PlanCandidate` and `score_plan` interfaces that preserve a later Planner-C path;
- decoder-off throughput measurement;
- the deployed 15-D HMMWV state plus a separately considered power quantity;
- the existing yaw-first dead-reckoning convention.

The installed `nedm` PyChrono environment also exposes both `ChDepthCamera` and `ChSegmentationCamera`, so the basic sensor choice is technically available.

## 5. Recommended revised execution order

### WP0A — Freeze the v1 information contract

Before scene implementation:

1. Make start pose and absolute goal/approach pose privileged in v1.
2. Rewrite the top-level claim so obstacle layout, rather than all navigation information, is the load-bearing visual signal.
3. Define every planner and tracker coordinate frame.
4. Replace the house-center goal with an approach pose.

### WP0B — Oracle vertical slice in Chrono

Before collecting an RGB-D dataset:

1. Build the true height/occupancy map.
2. Generate a direction-aware, footprint-safe path.
3. Produce a `PlanCandidate` with geometry and speed.
4. Track it in Chrono using privileged state.
5. Validate curvature, clearance, slope, and goal reachability.

This isolates planner/tracker feasibility from perception and NRD quality. If this stack fails, more image data will not help.

### WP0C — Sensor and storage smoke test

Collect approximately ten layouts and verify:

- RGB/depth co-registration;
- p95 as well as median alignment;
- depth-to-elevation conversion at the image edges;
- small-object visibility;
- compression ratio and random-window load rate;
- exact raw and processed peak disk use.

### WP1 — Perception pilot before joint dynamics

Using the smoke data, prove that the representation supports:

- held-out obstacle occupancy;
- vehicle center and yaw;
- elevation reconstruction;
- start/goal raster placement under the chosen privilege contract.

Only after these probes pass should the 200-layout pilot be collected.

### WP2 — HMMWV-specific NRD infrastructure

Implement:

- versioned RGB-D schema and loaders;
- HMMWV autonomous rollout evaluation with pose integration;
- task-space `z2` metrics;
- state-only, RGB-only, depth-only, and shuffled-latent ablations;
- rollout-selected checkpoints at 0.5-1.0 s horizons.

### WP3 — Planner evaluation

Compare:

1. true map plus feasible oracle search;
2. true occupancy with no energy term;
3. predicted occupancy with memorized terrain cost;
4. full predicted map;
5. straight-line and naive 8-connected A* baselines.

Evaluate the final smoothed footprint path on the true map, not just the raw grid path.

### WP4 — Short-fragment tracker training

Train on 1-3 s fragments sampled from same-layout context windows and references. Define `action_repeat=1` at 0.05 s unless a separate control rate is explicitly selected. Use the measured p95 Chrono tracking error to update planner inflation margins.

### WP5 — Protected full-stack evaluation

Freeze models, thresholds, scenario generation, and comparison budgets before evaluating the test layouts. Report paired confidence intervals and distinguish:

- perception failure;
- no-path/planner failure;
- smoothing/feasibility failure;
- tracker divergence;
- collision despite nominal clearance;
- timeout or energy inefficiency.

## 6. Final recommendation

Proceed with the scene and a small oracle-based vertical slice. Do not begin the 200-episode pilot or full RGB-D collection until these mandatory redlines are resolved:

1. define the start/goal privilege and coordinate contract;
2. replace the unreachable house-center goal;
3. make planner costs directional and paths HMMWV-feasible;
4. specify the geometric tracker reward and short-fragment training protocol;
5. establish the RGB-D storage/schema design;
6. add the ablations required to separate visual localization, obstacle perception, and actual use of depth.

With those changes, the study becomes a credible fixed-map feasibility experiment and a good foundation for the later localization, multi-map, and Planner-C upgrades.
