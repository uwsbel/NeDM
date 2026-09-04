# NRD Study 3 Plan: HMMWV Terrain Traversal with an Overhead RGB-D Camera

**Purpose:** First NRD study where vision is load-bearing — a hierarchical planner/tracker stack on a fixed bumpy arena
**Simulator:** Project Chrono (HMMWV vehicle stack) with Chrono::Sensor RGB + depth cameras
**Builds on:** `docs/vision/NRD_overall_project_plan.md` (Phase 3, pulled forward ahead of Phase 2 tabletop manipulation), Study 1 (`docs/vision/double_pen/`), and the state-only NeDM HMMWV stack
**Status:** v1.4 — revised 2026-09-04 pm (§20: tracker + planner rollout built; ẑ₂ decision now evidence-based); v1.3 2026-09-04 am (§19); v1.2 2026-09-03 (§18); v1.1 2026-08-31 after `NRD_hmmwv_traversal_study_plan_review.md`; §16 = original decision log, §17 = review resolutions
**v1 charter:** Feasibility of the full stack (NRD + planner + tracker) on ONE fixed terrain map, trained and collected locally. Privileged information is allowed anywhere it unblocks v1; deployment-purity upgrades are a ladder, not a v1 gate.

## 1. Study objective, information contract, and positioning

Study 1 validated synchronized \(z_1\)/\(z_2\) prediction on a system where vision was informationally redundant. This study makes vision load-bearing, with an **explicit v1 information contract** (review §2.1):

| Quantity | v1 source | Later ladder |
|---|---|---|
| Obstacle layout (per-episode) | **vision only** (\(z_2\) → costmap head) | — (already pure) |
| Terrain geometry | vision + memorization (fixed map; see RQ2 caveat) | multi-map generalization |
| Vehicle position/yaw **for the dynamics model** | **vision only** (\(z_1\) has no pose by design) | — (already pure) |
| Vehicle start pose **for the planner** | privileged `start_pose_world` | \(z_2\) localizer head |
| Goal | privileged `goal_pose_world` (approach pose, §3.2) | goal heatmap from vision (house recognition) |
| Reward/termination geometry (training only) | privileged layout + poses | — |

The load-bearing visual claim is therefore: **per-episode obstacle layout is available only through the camera, and the dynamics model can localize only through the camera.** Planner-side localization and goal placement are honestly privileged in v1.

The control architecture is **hierarchical**, not end-to-end RL:

```
                 ┌─ overhead RGB-D ─→ E_phi ─→ z2 ─┐
                 │                                  ├─→ PLANNER (once per episode, replan later)
 start/goal pose (privileged v1) ───────────────────┘        │
                 │                          reference path + speed profile (PlanCandidate)
                 │                                           ↓
 partial z1 ─────┴────────────────────────────→ TRACKER (20 Hz) ─→ [steer, throttle, brake]
```

- **Planner (from \(z_2\)):** decodes traversal structure (occupancy + elevation gradients) from the sensor latent, searches it with direction-aware edge costs, validates HMMWV feasibility, and emits geometry + a slope-modulated speed profile ("Planner-B"). k-best candidates are then scored by NRD rollout ("Planner-C" scoring, §9.5); differentiable fine-tuning stays deferred.
- **Tracker (from partial \(z_1\)):** a reduced-observation policy with a **geometric tracking reward** (§10), trained inside frozen NRD imagination on **short randomized fragments** (1–3 s) so imagination stays within the validated horizon; continuous tracking is evaluated in Chrono.

Why the hierarchy: it confines learned-model rollouts to short, validated horizons (tracker fragments; the v1 planner rolls out only the short scoring segments of §9.5, never a full traverse) and replaces the main exploitation surface (long-horizon RL against model error) with supervised imitation of a privileged oracle. Feedback-policy transfer tolerates proportional drift (Study 1: 87%→87% transfer despite a 3.3× model-error gap; tracked ROM: ~8% rollout error, 100/100 goals), but we do not rely on that for training-time horizons. The end-to-end RL agent survives only as a clearly-labeled bracket, run after G7 (§11).

## 2. Research questions

1. **Perception:** does \(z_2\) from one fixed overhead RGB-D camera carry decodable obstacle occupancy and terrain structure on **held-out asset layouts** of the fixed map?
2. **Localization-conditioned dynamics (reworded per review §2.4):** does **visual localization on a known terrain map** improve position-dependent dynamics prediction over a matched state-only model? On one fixed map this is a localization claim, *not* a general terrain-from-depth claim; the ablation grid (§8.3) separates localization, obstacle perception, and actual use of depth.
3. **Planning from vision:** can costmap-decode + feasible search imitate an energy-aware privileged oracle, measured by path-length/energy ratios and full-footprint collision rates on held-out layouts?
4. **Control:** can a tracker observing only \([\text{pose err}, \text{preview}, v_x, \dot\psi, \text{last actions}]\) (38-D), trained on short imagined fragments with a geometric reward, track planner references continuously in Chrono?
5. **Composition:** does plan-once → track reach **approach poses** on held-out layouts in Chrono, beating the straight-line bracket and approaching the oracle-plan bracket, with pre-frozen thresholds and paired confidence intervals?
6. **Throughput:** like-for-like NRD-vs-Chrono numbers under the three-row protocol of §12.4.

## 3. Scene specification

### 3.1 Arena and terrain (fixed, authored)

- Arena: **80 × 80 m** rigid heightmap patch (`rigid_heightmap` machinery, single authored map, not the noise library).
- Heightmap: 512 × 512 8-bit BMP (~0.16 m/px), range ≈ ±2.0 m; composed craters (depth 1–2 m, r 4–8 m), hills (1.5–3 m, r 8–15 m), roughness ±0.15 m; slopes capped ≈ 20°. WP0 verifies drivability; the generator is kept for later randomization.
- Spawn/goal sampling on low-slope cells; traverse distances 20–60 m; boundary truncation reused. Spawns are off-origin: the collector settles the vehicle at local terrain height (flat-origin gotcha handled explicitly).

### 3.2 Assets and the goal (review §2.2 fix)

- Rocks 8–15 (0.8–2 m), trees 5–10, one **house** (~5 × 4 × 3 m, high-contrast roof). All assets immovable, collision enabled.
- **The goal is an approach pose, not the house center.** The oracle selects a collision-free approach pose on a ring ≈ 7 m from the house center (house half-diagonal ~3.2 m + vehicle half-length ~2.4 m + margin), preferring low slope, clearance, and a sensible final heading. Success = within 2 m of the approach pose. The house-center "goal" exists only as the semantic target; every metric is measured against the approach pose.
- Layout constraints: minimum separations, low-slope house cell, and an **oracle reachability check** (§7) — unreachable layouts are resampled.
- Visual design: distinct class colors; bright roof marker on the HMMWV for alignment tests.
- **Class masks are analytic, not sensor-recorded** (deviation from review §3.3, same guarantee at lower cost): per-class masks are rasterized from the layout manifest + camera model for **every** episode in every split, so G1/G5 metrics are always available. `ChSegmentationCamera` is used once at smoke tier to validate the analytic projection (target: IoU ≈ 1 up to anti-aliasing).

### 3.3 Overhead camera (fixed, whole-arena)

- One fixed camera at arena center, ≈ 100 m altitude, nadir view, HFOV ≈ 47° covering ~88 m; near-orthographic with a fixed per-pixel ray-correction map in the manifest.
- **RGB:** `ChCameraSensor` 256 × 256 (≈ 0.31 m/px → rock 3–6 px, house 16–20 px, vehicle ≈ 6 × 16 px). **Depth:** `ChDepthCamera` 256 × 256, `maxDepth` ≈ 120 m; converted to height-above-datum with ray correction, normalized by arena height range. Depth-to-elevation accuracy is explicitly checked **at image edges** at smoke tier.
- Sync: Study 1 manual-trigger pattern; instantaneous shutter; zero lag.
- **Coordinate frames contract (review WP0A):** four frames — world (Chrono), arena raster (costmap grid), image (pixels), ego (vehicle) — with fixed, versioned affine/ray maps between world↔raster↔image recorded in the manifest. `PlanCandidate` is always world-frame; the tracker consumes ego-frame errors.

### 3.4 Vehicle, actions, timing

- HMMWV per `src/nedm/hmmwv_data.py`; actions `[driver_steering, driver_throttle, driver_braking]`, standard clamps, steering rate limiting in env and eval.
- Token/control interval **\(\Delta t = 0.05\) s (20 Hz)**, camera at 20 Hz, Chrono substepping underneath; WP0 throughput probe confirms.

## 4. Physical state \(z_1\) and the power channel

**\(z_1\) is exactly the deployed 15-D HMMWV preset** (`tire_normal_force_omega`): \([v_x, v_y, \phi, \theta, \dot\phi, \omega_{b,y}, \dot\psi, F_z^{fl..rr}, \omega_{\mathrm{spindle}}^{fl..rr}]\). No terrain one-hot. Delta targets, dead-reckoned pose via the existing `_integrate_pose` convention; \((x,y,\psi)\) never a model input.

**Drive power is recorded but is NOT a recursive state channel** (review §3.1): the collector logs powertrain shaft torque and speed every step; the NRD gets a separate supervised **auxiliary power head** on the shared backbone features (usable during imagined rollouts for Planner-C scoring and energy metrics, without injecting a noisy signed channel into the token). Energy reporting separates positive drive work from braking work. This also keeps the state-only baseline exactly matched at 15-D for RQ2.

Deliberate non-Markovness stands: without position, \(z_1\) cannot determine the bump underfoot; that information can only flow through \(z_2\) (RQ2).

## 5. Sensor state \(z_2\) and representation safeguards

- Input \(x_t \in \mathbb R^{4\times256\times256}\) (RGB + normalized elevation). Encoder: Study 1 `ConvEncoder` extended to 4 channels + one extra stride-2 stage; mirrored decoder, diagnostic only.
- **\(z_2\) is an ego-indexed crop of the encoder's spatial map, not a global pooled vector (v1.3, 2026-09-04).** The scene map \(M = E_\phi^{(2)}(x)\in\mathbb R^{64\times64\times64}\) is the **stage-2** backbone tap `backbone[:4]` — NOT the final 256x16x16 map, which at 5.44 m/cell would make an 8x8 crop span 43 m, half the arena. Stage 2 gives 1.36 m/cell over the camera's 87 m field of view, encoded **once per episode** from a vehicle-free pixel-wise median of 9 frames — the layout is static and the camera fixed, so nothing is gained by re-encoding per frame, and the moving vehicle medians away. At each step an \(8\times8\) window spanning \(\pm5\) m (1.25 m sample spacing, matched to the 1.36 m cell size) is resampled from \(M\), ego-aligned to the vehicle's yaw, projected \(64\to16\) channels and flattened to \(z_2\in\mathbb R^{256}\). The resampling is differentiable (`grid_sample`), so the dynamics loss trains the projection. World→image uses the §3.3 camera contract with terrain height at each sample point (verified to 1e-6 m against the quantized heightmap). \(z_2\) statistics fitted at joint-init (Study 1 normalization footgun).
- **\(z_2\) is indexed, not predicted forward.** During imagination the crop is retaken from the static \(M\) at the pose dead-reckoned from predicted \(z_1\); there is no \(\hat z_{2,t+1}\) head. Measured (WP2 Batch C, 2 seeds): indexing closes **75 %** of the privileged-terrain gap at 1 s and **51 %** at 5 s (leak-free scene maps, 2 seeds), where the pooled global vector closed 39 % / −6 %; the autoregressive variant of the same token collapses to −57 % / −145 %. The global pooled latent is retained only as a diagnostic and as the WP1/G1 comparison point.
- **Auxiliary representation losses during AE warm-up are mandatory** (review §2.5; labels are analytic and free): (a) occupancy/class-mask prediction, (b) vehicle-center heatmap + yaw, (c) foreground/class-weighted RGB reconstruction, (d) elevation reconstruction with its own normalization. These are representation-shaping losses from analytic ground truth, not task/reward losses — consistent with the master-plan boundary.
- **Encoder fine-tuning criterion:** downstream occupancy-probe and localization-probe performance, not latent-prediction plateau alone.
- **Information-bottleneck staging** (partial adoption of review §2.5): the single global latent remains the v1 spine, but the perception pilot (WP1) also probes occupancy/localization **from the encoder's pre-pooling spatial feature map**, quantifying what global pooling destroys. Pre-declared fallback if the single-latent probes miss their bars: keep a low-res spatial feature map (or a factored \(z_{\mathrm{layout}}/z_{\mathrm{vehicle}}\)) as the planner-facing representation while the global \(z_2\) continues to serve the dynamics token.
- v1 fuses RGB and depth in one encoder; two-stream encoders remain a deferred ablation.

## 6. Dataset plan

### 6.1 Tiers and storage (review §2.7 numbers adopted)

| Dataset | Episodes | Duration | Frames | Raw frame bytes |
|---|---:|---:|---:|---:|
| Smoke | 10 | 20 s | 4k | ~1.3 GiB |
| Pilot | 200 | 20–40 s | 80–160k | 25–49 GiB |
| Full | 1000 (default 30 s) | 20–40 s | 0.4–1.6 M | **122–488 GiB** |

Per frame at 256²: RGB uint8 (196,608 B) + depth uint16 (131,072 B) ≈ 0.31 MiB. Consequences, fixed **before** pilot collection (WP0C):

- separate RGB (uint8) and depth (uint16, mm) arrays with a versioned schema;
- **chunked, compressed storage** (episode-chunked zarr/zstd or equivalent) — the static background should compress heavily, but the current uncompressed memmap path does not exploit that, so this is a new storage representation, measured (compression ratio, random-window loader throughput at training batch size, peak disk during preprocessing) at smoke tier;
- an explicit decision whether processed caches **reference** raw frame stores or duplicate them (default: reference);
- levers if the budget is blown: 30 s episodes, 128² frames, or camera at 10 Hz with 20 Hz states (multi-rate is the last resort).

### 6.2 Driver mixture

60% spline-following pure-pursuit over random smooth routes (3–8 m/s, slope-modulated profile) · 20% random meander · 10% near-obstacle passes (half with contact) · 10% **oracle-route following** (reusing the WP0B oracle + scripted tracker — the vertical slice becomes the collection driver). All steering-rate-limited. Per-episode randomization: layout, spawn, goal, seed; terrain fixed.

### 6.3 Recorded fields

States (15-D) + actions + rollout pose + powertrain torque/speed, RGB + depth frames, per-episode layout manifest (asset types/poses/sizes, goal + approach pose, spawn), analytic class masks (derivable on demand from manifests — store the manifests, rasterize in loaders), contact events, solver status; global manifest with camera model, frame-transform maps, heightmap hash, energy-fit coefficients.

### 6.4 Alignment and splits

- Automated alignment test: project GT roof marker and asset centers through the camera model vs blob centroids; report **median and p95** (targets ≤ 2 px median, ≤ 4 px p95 at 256²).
- Splits by episode = layout, **70/15/15 train/val/test** (the current preprocess supports only train/val — extending it is a WP2 deliverable, §17). Test layouts untouched until final evaluation. "Overfit one map" = memorize terrain, generalize across layouts.

## 7. Privileged oracle: feasible, direction-aware, margin-audited (review §2.3, §3.2)

1. **Energy model (pilot-fitted):** \(P \approx f(v,\; a_{\mathrm{long}},\; \text{slope along heading},\; \kappa)\) — acceleration and curvature included so acceleration energy is not attributed to terrain. Fit diagnostics reported; planning evaluates the quasi-static slice (\(a_{\mathrm{long}}=0\), profile speed).
2. **Direction-aware search:** edge costs \(c(i\!\to\!j) = w_d\,d_{ij} + w_e\,\hat e(\nabla h \cdot \hat t_{ij},\, v_{ij})\) on an 8-connected grid, with a cross-slope traversal cap. Uphill, downhill, and cross-slope edges cost differently by construction.
3. **Feasibility validation (mandatory):** after spline smoothing + speed profile \(v_{\mathrm{ref}}(s) = \min(v_{\mathrm{cruise}}, \sqrt{a_{\mathrm{lat,max}}/|\kappa|}, v_{\mathrm{slope}})\), every path is checked for full-footprint swept collision clearance, curvature ≤ vehicle limit, and cross-slope bounds; invalid paths are repaired or rejected. If WP0B shows grid-A*+smoothing rejects too often, upgrade the oracle to a heading-state lattice / Hybrid A* (pre-declared escalation, not v1 default — deviation from review §2.3's "prefer" phrasing, on evidence-first grounds).
4. **Inflation margin is a budget, not a constant:** half footprint + costmap/pixel localization uncertainty + **tracker held-out p95 lateral error (from G6)** + fixed safety allowance. G5 planner safety is not final until G6 supplies the tracker envelope; WP4 feeds the measured p95 back into the margin and plans are re-validated.
5. **Approach-pose selection** (§3.2) and layout reachability filtering.
6. **Diverse candidates:** k-best via cost-weight sweeps + stochastic tie-breaks (imitation variety; required by Planner-C).

Oracle touches privileged data only at training/eval. Deployment inference: camera → \(z_2\) → predicted map → same search + validation stack.

## 8. NRD model, baselines, ablation grid

### 8.1 Model

Token \(u_t = [z_{1,t}(15),\, z_{2,t}(256),\, a_t(3)]\) = 274-D, NeDM backbone (`ContinuousTransformer`, 6L/256/8H), ctx 16 @ 0.05 s. Heads: \(\Delta z_1\) (15, residual in normalized target space) and the auxiliary power head (1, §4). **No \(\hat z_{2,t+1}\) head in v1.3** — \(z_2\) is re-indexed from the static scene map each step (§5), so there is no sensor latent to roll forward. Losses per master plan; multi-step curriculum available. Stages: AE warm-up (with §5 auxiliary losses) → scene-map cache → \(z_2\) stats → joint training, frozen encoder first.

Reference implementation: `scripts/traverse_wp2_encode_map.py` (per-episode scene map), `src/nedm/traverse/map_crop.py` (`MapCropper`), `scripts/traverse_wp2_train_map.py --map-mode index`. The pooled-\(z_2\) path (`traverse_wp2_train.py`) is kept as the matched baseline for the §8.3 grid.

### 8.2 Infrastructure deliverables (review §2.7 — WP2 is a subsystem, not a flag)

RGB-D (4-ch, mixed-dtype) support in encoder/decoder/frame pipeline; train/val/test preprocessing; **HMMWV pose-integrated autonomous rollout evaluation inside the NRD trainer** (the dpend trainer's tip-metric analog, using `_integrate_pose`); task-space \(z_2\) metrics (below); **rollout-based checkpoint selection at 0.5–1.0 s horizons** (repo lore: judge on rollout, not val_loss).

### 8.3 Baselines and ablation grid (review §2.4, §3.4)

- Matched **state-only** \([z_1,a]\) model (RQ2 comparator).
- **Privileged upper bound:** state + \((x,y,\psi)\) (and optionally + a privileged local terrain patch) — the ceiling on what localization can buy.
- Input ablations: RGB-only, depth-only, depth-zeroed/shuffled.
- Integrity ablations: \(z_2\) shuffled between layouts; vehicle region masked.
- \(z_2\)-prediction baselines: persistence AND **static-layout + constant-velocity-vehicle** (unusually strong here; aggregate latent metrics alone are uninformative).
- Task-space \(z_2\) metrics: predicted vehicle-center/yaw error, per-class object permanence, local terrain error at the predicted vehicle position.

## 9. Costmap head and planner (Planner-B spine)

### 9.1 Training data — auto-generated pairs

Input: \(z_2\) of a **random timestep** of episode \(e\) (vehicle-position augmentation → vehicle-invariance; loss down-weighted under the vehicle footprint). Labels rasterized from the episode manifest + fixed heightmap + calibrated energy model:

- channel 1 — **obstacle occupancy** (inflated per §7.4): varies per episode, must be read from \(z_2\);
- channels 2–3 — **elevation gradients** \((\partial h/\partial x, \partial h/\partial y)\) (replaces the v1.0 scalar cost, enabling §7.2 directional edge costs at inference): constant across episodes, memorizable — permitted by the charter and quantified by the §8.3 shuffles.

Every episode = one labeled layout (200 pilot / 1000+ full); frames are augmentation. Privileged only at training; deployment is camera → \(z_2\) → head → search.

### 9.2 Inference

\(z_2\) → conv-transpose head → occupancy + gradient rasters → directional-edge A* (start/goal from the privileged v1 contract, §1) → smoothing + speed profile → **same feasibility validation as the oracle (§7.3)** → `PlanCandidate`. Costmap-then-search over direct waypoint regression because optimal paths are multi-modal; Planner-A (regression) is retained as the demonstrating ablation.

### 9.3 Planner evaluation ladder (review WP3)

On held-out layouts, all evaluated by running the final smoothed, footprint-checked path against the **true** map: (1) true map + feasible oracle; (2) true occupancy, no energy term; (3) predicted occupancy + memorized terrain; (4) full predicted map; (5) straight-line and naive 8-connected A* baselines. Metrics: collision rate, path-length ratio, energy ratio, no-path/rejection rate, occupancy IoU/AUC + per-class small-object recall (analytic masks).

### 9.4 Plan cycle

Plan once at episode start; replan trigger on large tracking deviation as the first upgrade (first consumer of predicted \(z_2\) inside imagination).

### 9.5 Planner-C rollout scoring (promoted into v1 — see §18)

**Why this is in v1:** if the planner only ever consumes z₂ encoded from
*recorded* frames, the NRD's z₂-prediction branch has no downstream consumer on
the planning side, and "NRD rolls out z₂" is unearned there. On a fixed map
with immovable assets and one whole-arena overhead camera, the layout is fully
observed at t=0 and never changes, so any planner loop built on "consume the
newly predicted z₂" is the identity map — architecture cannot fix that; only a
dynamic or partially-observed scene can (deferred, §16).

What *does* change under imagination is the vehicle's state and the terrain it
sits on. So v1 makes the NRD load-bearing for planning through **candidate
scoring, not candidate generation**: the k-best plans from §9.2 are driven by
the WP3 tracker inside the frozen NRD from the episode's real start context and
scored on time, energy (auxiliary power head, §4), tracking cost, feasibility
(roll/pitch/cross-track bounds) and footprint collision. Plan generation stays
supervised; plan *selection* consumes the rollout.

**Mechanism (v1.4, replaces "roll z₂ forward"):** z₂ is *indexed* — the
t = 0 scene map is re-cut around the dead-reckoned pose at every imagined step
(§5) — while z₁ and the power head are predicted. The scorer is
`scripts/traverse_wp4_score_candidates.py`; the rollout is **closed-loop**
(tracker in the loop), which the WP3/4 notes show is what keeps a 10–20 s
imagined traverse on the route (open-loop replay of recorded actions drifts
0.47 m and misses the end in a third of episodes; with a controller the imagined
time-to-goal correlates 0.99 with the recorded one). Energy from the power head
is biased ~25 % low over 10 s and is used for relative ranking only until
recalibrated (open item).

### 9.6 Planner-C interfaces (unchanged, mandatory)

`PlanCandidate {waypoints, v_profile, meta}` world-frame; `score_plan(candidate, model, context) → {time, energy, collision_prob, feasibility}` (v1: geometric scorer; C: NRD-rollout scorer over **short imagined tracking segments**, energy from the auxiliary power head); differentiable head end-to-end from \(z_2\); k-best candidates; context-bank reset windows recorded per episode.

## 10. Tracker (review §2.6 adopted in full)

*Status 2026-09-04: built and trained in imagination (`nedm/traverse/tracker_env.py`,
`scripts/traverse_wp3_train_tracker.py`); Chrono evaluation (the actual G6) not yet run.
Design carries the state-only HMMWV tracking study's transfer lessons; see
`wp3_implementation_notes.md`.*

- **Environment:** NRD imagination env (dpend pattern: joint \([z_1,z_2]\) rollout from recorded context windows) + reference machinery. References: random splines + oracle routes (§6.2 distribution).
- **Geometric tracking reward** (replaces full-state reference error, which planner routes cannot supply): cross-track error, heading error, **speed error vs the reference profile**, action-rate penalty, simultaneous throttle+brake penalty, rollover/safety penalties charged with remaining cost (no cheap exits). Full-state reference terms are dropped except fields the planner specifies (speed).
- **Short-fragment training:** episodes are randomized **1–3 s fragments** initialized from real same-layout context windows at varied reference-progress states (starts, straights, turns, slope entries). This keeps every imagined rollout inside the validated horizon — the hierarchy does not do this by itself. Continuous long-horizon tracking is evaluated in Chrono (and, diagnostically, in NRD to measure drift).
- **Observation (38-D):** \([\text{pose err}(3), \text{preview } 10{\times}3, v_x, \dot\psi, \text{last actions}(3)]\). Asymmetric critic (full state) optional. Pose privileged in v1 Chrono eval per the §1 contract.
- **`action_repeat = 1` at \(\Delta t=0.05\) s** (20 Hz policy = token rate); any other control rate must be separately specified.
- Obs-subset ablation (full-state obs vs deployment set vs minimal) on closed-loop tracking cost.
- Output: **held-out p95 lateral error → §7.4 inflation margin** (G6 → G5 linkage).
- PPO presets from the arm/tracking family (lr 1e-4, kl 0.005, entropy 1e-3, noise 0.3).

## 11. End-to-end RL baseline (review §3.5)

Deferred until the hierarchical stack passes G7. Then either (a) matched-budget (env steps, wall-clock, observation privilege, reward information) if the comparison is a headline claim, or (b) run cheap but labeled explicitly as a low-budget bracket. Never an unlabeled strawman.

## 12. Evaluation

### 12.1 Acceptance gates

Numeric thresholds are **set from the pilot tier and frozen before full-tier training**; the test split stays untouched until final evaluation; results carry paired-bootstrap confidence intervals (review §3.7).

| Gate | Requirement |
|---|---|
| G0a Contract + oracle slice | §1 contract frozen; **oracle vertical slice** (WP0B): oracle plans tracked in Chrono by a scripted controller reach approach poses on ≥95% of sampled layouts, zero collisions, feasibility validation passing |
| G0b Sensor + storage | Depth verified (incl. image edges); alignment ≤ 2 px median / ≤ 4 px p95; analytic masks validated vs `ChSegmentationCamera`; compression ratio + loader throughput + peak disk measured |
| G1 Representation | AE recon + warm-up auxiliary heads: held-out-layout occupancy probe IoU, vehicle-center/yaw probe error, per-class recall above pilot-frozen bars (perception pilot on smoke data first, formal numbers on pilot tier) |
| G2 One-step | \(z_1\)/\(z_2\) beat persistence AND the static-layout + constant-velocity baseline on task-space metrics (§8.3), held-out layouts |
| G3 Localization-conditioned dynamics | Joint NRD beats matched state-only on dead-reckoned pose rollout error; ablation grid separates localization / obstacles / depth; privileged \((x,y,\psi)\) row bounds the claim |
| G4 Cross-modal | Decoded vehicle blob tracks dead-reckoned pose; per-class object permanence over autonomous \(z_2\) rollouts |
| G5 Planner | §9.3 ladder: full predicted map within frozen margins of the feasible oracle on length/energy; footprint-checked collision rate at frozen bar; margins include the G6 tracker envelope |
| G6 Tracker | Fragment-trained policy tracks oracle references **continuously** in Chrono: RMSE + p95 lateral error under frozen bars, no divergences; p95 fed back to §7.4 |
| G7 Full stack (v1 milestone) | Protected evaluation (models, thresholds, scenarios, budgets frozen): plan-once → track on test layouts; success@2 m of approach pose; beats straight-line bracket; within frozen margin of oracle-plan bracket; failure taxonomy reported (perception / no-path / feasibility / tracker divergence / collision-despite-clearance / timeout-energy) |

### 12.2 Scenario suites

S1 open field · S2 rock field · S3 crater on the line · S4 hill saddle · S5 house among distractors. ~100 test-layout scenarios, paired NRD↔Chrono where applicable, one process per scenario. Report CIs and per-category failures, not just point rates.

### 12.3 Comparison rows

(1) oracle-plan + tracker · (2) hierarchy (B) · (3) straight-line + tracker · (4) end-to-end RL (post-G7, per §11).

### 12.4 Throughput (review §3.6 — like-for-like)

Three rows, each vs batch size: (1) Chrono physics (no rendering) vs NRD transition-only; (2) Chrono + RGB-D rendering vs NRD data-generation mode; (3) full tracker-environment policy steps/s including reference/reward work. No decoder-off vs always-rendering comparisons presented alone.

## 13. Risks and responses

| Risk | Response |
|---|---|
| Global latent drops small assets / vehicle marker | Mandatory warm-up auxiliary losses; spatial-feature-map probe quantifies pooling loss; pre-declared spatial/factored fallback (§5) |
| Costmap head memorizes terrain, ignores assets | Occupancy channel is layout-specific; G1/G5 held-out-layout IoU; layout-shuffle ablation |
| Vehicle blob imprints on costmap | Random-timestep sampling + loss masking |
| RQ2 over-claimed on fixed map | Reworded claim + §8.3 grid + privileged upper bound (review §2.4) |
| Smoothed paths infeasible/colliding | §7.3 validation on oracle AND predicted maps; lattice/Hybrid A* escalation if rejection rates high |
| Planner margins ignore downstream error | §7.4 margin budget incl. G6 p95; G5 not final until G6 |
| Tracker imagination beyond validated horizon | Short-fragment training (§10); continuous eval only in Chrono |
| Reward interface mismatch (full-state refs) | Geometric tracking reward (§10) |
| Storage blowout / loader bottleneck | §6.1 schema + compression + measured throughput before pilot; declared levers |
| Dead-reckoned pose drift in long imagined use | Fragments in training; privileged pose at v1 eval; localization ladder later |
| Authored map too hard/easy; energy fit confounded | WP0 drivability sweep; regression incl. accel + curvature |
| GPU box constraints | Local rendering, one job at a time, small batches |

## 14. Work packages (revised order per review §5)

- **WP0a — Contract freeze + oracle vertical slice:** §1 contract, §3.3 frames contract, approach-pose goal; true-map oracle (directional costs, feasibility validation, approach selection) + scripted pure-pursuit tracking **in Chrono with privileged state** — no rendering, no learning. If this fails, image data will not help. *(G0a)*
- **WP0b — Scene + sensor + storage smoke:** camera pair, sync, alignment (median+p95), depth-at-edges, analytic-mask validation, drivability sweep, rendering FPS, storage schema + compression + loader benchmarks on ~10 layouts. *(G0b)*
- **WP1 — Perception pilot (smoke data, before pilot collection):** AE + auxiliary heads on the smoke tier; occupancy/localization probes from \(z_2\) and from the spatial feature map; provisional bars. **Pilot collection (200 layouts) starts only after these probes pass.**
- **WP2 — NRD subsystem:** RGB-D schema/loaders, 70/15/15 preprocessing, joint + state-only training, HMMWV pose rollout eval in the NRD trainer, task-space \(z_2\) metrics, rollout-selected checkpoints, §8.3 grid. *(G1–G4 formal)*
- **WP3 — Tracker** (was WP4; reordered in v1.2): imagination env + geometric reward + fragment training; obs ablation; continuous Chrono eval; p95 lateral error → §7.4 inflation margin. *(G6)*
- **WP4 — Planner** (was WP3): costmap head + directional search + validation; §9.3 ladder; §9.5 NRD rollout scoring of k-best candidates; Planner-A ablation optional. *(G5, with margins already supplied by WP3)*
- **WP5 — Protected full stack:** freeze everything, full-tier collection decision, test-layout suites, comparison rows, throughput protocol, report, go/no-go for Planner-C and scaling. *(G7)*

## 15. Expected outputs

Terrain generator + fixed map; oracle stack (feasible search + validation + approach poses) with calibrated energy model; RGB-D collector + versioned compressed frame store + layout manifests with analytic masks; joint NRD + matched state-only + ablation-grid checkpoints; costmap head + planner; fragment-trained tracker; protected G7 evaluation with CIs and failure taxonomy; three-row throughput benchmark; predicted-vs-true costmap and route-choice figures.

## 16. Decision log (2026-08-31 discussion)

1. One fixed authored terrain map (generator kept for later randomization).
2. \(z_1\) contains no absolute pose; pose dead-reckoned (existing convention); \(z_1\) = deployed 15-D preset (power demoted to auxiliary head in v1.1).
3. Single fixed overhead camera covering the whole arena; RGB + depth fused in one 4-channel encoder for v1.
4. Hierarchy over end-to-end RL (now a post-G7 bracket): Planner-B spine; **Planner-C interfaces mandatory from day one** (world-action-model path stays open).
5. Planner output = geometry + slope-modulated speed profile.
6. Tracker observation = reduced subset (38-D in v1.1).
7. Goal supplied as a privileged pose in v1 (approach pose in v1.1); house-by-vision deferred.
8. v1 is a feasibility test: privileged information acceptable throughout.
9. Local collection/training; milestone = trainability on the one map with layout-level generalization.

**Deferred:** localization ladder, receding-horizon replanning, vision-only goal, two-stream encoders, terrain randomization, Planner-C execution, multi-map scaling.

**Naming:** Phase/Study 3 per the master plan, executed before Phase 2.

## 17. Review resolutions (v1.0 → v1.1, from `NRD_hmmwv_traversal_study_plan_review.md`)

**Adopted:** §2.1 privileged start/goal contract + claim rewrite (§1) · §2.2 approach-pose goal (§3.2) · §2.3 directional edge costs, footprint/curvature validation, gradient-channel labels (§7, §9.1) · §2.4 RQ2 rewording + ablation grid (§2, §8.3) · §2.5 auxiliary warm-up losses, fine-tune criterion, spatial-probe comparison + declared fallback (§5) · §2.6 geometric reward, 38-D obs, `action_repeat=1`, short-fragment training (§10) · §2.7 storage math, schema/compression requirements, WP2 re-scoped incl. 70/15/15 preprocessing and HMMWV rollout eval (§6.1, §8.2) · §3.1 power as auxiliary head, richer energy regression, drive/brake work split (§4, §7.1) · §3.2 margin budget incl. tracker p95 (§7.4) · §3.4 task-space G2 metrics + static+CV baseline (§8.3, G2) · §3.5 e2e baseline post-G7/matched-or-labeled (§11) · §3.6 three-row throughput (§12.4) · §3.7 frozen thresholds, untouched test split, CIs, failure taxonomy (§12.1) · §5 revised WP order incl. WP0 oracle vertical slice and perception-pilot-before-collection (§14).

**Deviations (with rationale):** review §3.3 — class masks are rasterized **analytically** from layout manifests for all splits (sensor segmentation only validates the projection at smoke tier); same metric guarantee, no extra render pass. Review §2.3 — Hybrid A*/lattice is a pre-declared **escalation** triggered by WP0b/WP0a rejection rates, not the v1 default; directional costs + post-hoc feasibility validation are mandatory either way. Review §2.5 — factored/spatial representations are a pre-declared **fallback** gated on the WP1 probes rather than a parallel v1 build; the spatial-feature probe runs regardless to quantify the pooling cost.

## 20. v1.4 change (2026-09-04, afternoon): ẑ₂ decision confirmed; WP3 + Planner-C built

1. **The ẑ₂ prediction head stays out of v1, now on a fair test.** With the
   map projection frozen and the token loss normalized (the two defects of the
   Batch C run), autoregressive prediction of the local crop still falls below
   persistence within 0.5 s and returns z1 accuracy to state-only level, two
   seeds (`wp2_implementation_notes.md`, addendum). Indexing remains the v1
   mechanism; the branch is reinstated when the scene or the camera moves (§16).
2. **Pose drift explains ~40 % of the 5 s shortfall**, not all of it; the rest
   is accumulated state error. Recorded there too.
3. **§10 tracker implemented** (`nedm/traverse/tracker_env.py`,
   `traverse_wp3_train_tracker.py`): geometric reward, 38-D deployment
   observation, 1–3 s fragments from recorded context windows, HMMWV-study PPO
   recipe with the steering clamp trained in. In-model held-out mean cross-track
   0.15 m, p95 0.54 m, vs 0.245 / 0.73 m for scripted pure pursuit; the state-
   history obs ablation changes nothing. **G6 still requires the Chrono run.**
4. **§9.5 Planner-C scorer implemented** (`traverse_wp4_score_candidates.py`):
   k candidates from oracle parameter sweeps, tracked inside the NRD from the
   real start context, scored on time / energy / tracking / safety / collision.
   Restated mechanism, as §19 required: the scene map is encoded once and
   **indexed along the imagined trajectory**; predicted z1 supplies time and
   feasibility. Calibration on 32 held-out layouts: time-to-goal corr 0.997
   (10 % optimistic); **energy underestimated ~30 % by the power head** even
   under replayed recorded inputs — calibrate before absolute use.
5. **G4 respecified:** cross-modal consistency for v1 is "the indexed token at
   the imagined pose supports z1 prediction as well as at the true pose" —
   measured: identical at 1 s, 0.06 z1-MAE apart at 5 s. The decodability form
   of G4 is retired with the ẑ₂ head.

## 19. v1.3 change (2026-09-04): z2 becomes a spatial index

WP2 established that the single failure behind every disappointing z2 result —
elevation channel unused, latent undecodable after 0.5 s (G4), only 39 % of the
privileged-terrain ceiling — is **global pooling**, not the encoder, the data,
the capacity or the modality. A 256-number summary of an 80x80 m arena cannot
carry a height field. Replacing it with a 256-number encoding of the 10 m under
the vehicle closes 75 % of that ceiling at 1 s and 51 % at 5 s, and fixes the pose-channel
regression outright (0.117 m, better than state, than pooled z2, and than the
privileged ceiling itself).

Consequences recorded here, detail in `wp2_implementation_notes.md`:

1. §5 and §8.1 rewritten: \(z_2\) = ego-indexed crop, no \(\hat z_2\) head.
2. §9.5 Planner-C rollout scoring must be restated. "Roll z2 forward and score"
   is not the mechanism — on a static map the scorer should **index** the t=0
   scene map along each candidate path. Predicted \(z_1\) still supplies time,
   energy and feasibility; obstacle geometry comes from the spatial map, which
   was already the G1 recommendation for the planner.
3. G4 as written (cross-modal decodability of a predicted latent) no longer has
   an object in v1.3 — nothing is predicted forward. It should be respecified
   against the *indexed* token or retired; the WP1 probes already cover whether
   the encoded map decodes the scene.
4. §18.3's deferral of the learned receding-horizon planner is reinforced: the
   static-scene argument now has direct dynamics-side evidence.

## 20. v1.4 changes (2026-09-04 pm)

1. **The ẑ₂ head decision is now evidence-based, not a one-run over-reach** (the
   v1.3 wording was challenged and re-tested). A fair two-stage test — projection
   frozen so the target is stationary, loss on normalized targets, two seeds,
   persistence baseline — predicts the next crop well one-step (4 % residual
   variance) but falls **below persistence within 0.5 s** when fed back, and
   drags z₁ to state-only level (0.33–0.35 vs 0.25 for indexing at 1 s). Feeding
   the crop the *true* pose at 5 s recovers only ~40 % of the long-horizon gap, so
   most of it is accumulated state error, not "reading the wrong place". §5's
   "index, not predict" stands; the branch returns with a moving scene or an
   ego camera (§16 ladder). Detail: `wp2_implementation_notes.md` addendum.
2. **§9.5 restated** (above): closed-loop tracker rollout over an indexed scene
   map; G4 as written is retired and replaced by the scorer calibration against
   recorded routes (imagined vs recorded time / energy / cross-track on the same
   route), which tests the thing the planner actually consumes.
3. **WP3 tracker built** (§10 status line) and **Planner-C scorer built** (WP4's
   §9.5 deliverable) ahead of the costmap head; candidates currently come from
   the privileged oracle's cost sweep (§7.6), which is the v1 charter's allowed
   privileged source until the WP4 costmap head exists.
4. Checkpoint selection in the WP2 map trainers is at the 5 s horizon, not
   §8.2's 0.5–1.0 s; consistent across arms, recorded so the "@1 s" tables are
   read correctly.

## 18. v1.2 changes (2026-09-03 plan sync)

1. **WP order: tracker before planner** (§14). §7.4 already makes the planner's
   inflation margin depend on the tracker's held-out p95 lateral error (G6→G5),
   so the tracker was always upstream in the real dependency graph; the v1.1
   order forced G5 to be re-validated after the fact.
2. **Planner-C rollout scoring promoted from post-v1 into v1** (§9.5), to make
   predicted z₂ load-bearing on the planning side and not only in the tracker's
   imagination env. Plan generation stays supervised (Planner-B); selection
   among k-best candidates is scored by NRD rollout.
3. **Learned receding-horizon planner `P_k = π_H(z₂, goal)` considered and
   deferred.** On a static, fully-observed map, `z₂,k+1 ≈ z₂,k` in everything
   the planner cares about, so the loop reduces to replanning on updated pose
   and cannot beat encoding frame 0 once. It becomes the right design once the
   scene has moving obstacles or a partial/ego view — both already on the §16
   deferred ladder. Revisit at the WP5 go/no-go.
4. **Planner-facing representation stays the §5 spatial-map fallback pending
   WP1 v7**, which tests whether z₂'s ~0.4 BEV ceiling is position-blindness in
   the attention pooling rather than capacity or data
   (`wp1_implementation_notes.md`). If v7 clears the bar, the planner reads z₂
   directly and item 3's blocker becomes the only one left.
