# RGB-D route-choice implementation and measured results

**Latest, 2026-09-09:** The [smooth-hill revision and four regenerated videos](mppi_smooth_hill_revision_20260909.md)
are complete. The rounded hill produces an aligned wheel-slip stall at a 4 m/s
command. MPPI reaches safely in 8.10 s, versus 14.50 s for the shown slower
detour and 7.80 s for the known faster base reference. Collision-risk calibration
and explicit prediction of the later hill stall remain unresolved. All four
recordings pass shared-observation, frozen-reference and telemetry checks.

**2026-09-09 update:** The [new joint rock/berm four-route video diagnostic](mppi_fourway_render_20260909.md)
is complete. MPPI selects the fastest safe route among the four executions
(6.85 s), and all four progress forecasts are within 0.316 m at four seconds.
The explicit contact head nevertheless misses the rock collision at 2.072 s;
terrain stall confirmation occurs beyond the four-second forecast horizon.
The linked report includes the frozen prediction overview and four actual
Chrono videos. Earlier route-choice success is not a calibrated hazard claim.

The requested milestone is a trained RGB-D-conditioned scorer that guides MPPI
to a fast, goal-reaching route without collision or stall. The geometry-only
pilot did not meet that requirement. Actual RGB-D is now implemented and
trained, and the first paired Chrono route-choice experiment is complete.
**The initial Chrono route-choice demonstration now succeeds. One combined
RGB-D model selects safe routes in all eight held-out development scenes and
its MPPI references physically reach safely in all three demonstration cases.**

```text
Current overhead RGB-D + measured history/pose + candidate path and speed
  -> trained spatial/route-patch encoder + history GRU + forward command GRU
  -> predicted motion and risk -> time/progress/risk cost -> MPPI -> Chrono PID
```

The fixed primary is `rgbd_focused_pilot_v2/patch_rgbd_s11/last.pt`, 5,000 AMD
updates, source snapshot `83d62378c36e`. Both RGB-D seeds choose eight safe
recorded references out of eight validation scenes; matched blank-input patch
models choose six and five. Primary mean time regret is 0.0125 s relative to
the fastest safe recorded sibling. These are eight scene-level trials, not
hundreds of independent overlapping windows. The [complete validation and
image-control report](../../../artifacts/traverse/fdm_rgbd_runs/rgbd_focused_pilot_v2/report.md)
includes every model and abstention.

| Same combined primary model | Measured MPPI completion | Fastest safe sampled reference | Direct-route outcome |
|---|---:|---:|---|
| Visible rock | 6.60 s, safe | 7.00 s | Contact and stopping |
| Hill crossing | 6.90 s, safe | 7.10 s | Safe but slower |
| Separate terrain-blockage diagnostic | 6.60 s, safe | 7.00 s | Both speeds stall and time out |

Actual observations, paths and goal-progress traces: [primary demonstration](../../../artifacts/traverse/fdm_rgbd_demo_v1/combined_measured_route_choices.png)
and [6 m/s capped comparison](../../../artifacts/traverse/fdm_rgbd_demo_v1/combined_capped_measured_route_choices.png).
PDF copies are saved beside both figures.

Every selected execution has zero asset contact, strict sustained stall and
bounded blockage. The terrain's two straight routes make only 5.75 m of goal
progress in 10 s, while the selected route makes 33.53 m and reaches the goal.
All compared routes start from identical measured RGB-D/state. Normal and
blank-image selected terrain references both succeed (6.60/6.65 s), so that
single demonstration does not establish image necessity. The broader
eight-scene controls supply the visual-use evidence.

These time gains combine path and speed optimization: the original MPPI
selections reach about 6.8 m/s, above the focused dataset's 6 m/s cruising
command. They are within declared optimizer limits and physically tested, but
are not guaranteed to remain in training support. A separately declared
6 m/s capped control also reaches every goal without collision or stall:

| Same model, MPPI capped at 6 m/s | Measured completion | Difference from fastest sampled safe route |
|---|---:|---:|
| Rock | 6.90 s | -0.10 s |
| Terrain blockage | 6.95 s | -0.05 s |
| Hill | 7.40 s | +0.30 s |

Thus time improvement is not universal: under the collected speed ceiling,
the hill refinement is slower than its best sampled detour. The cap comparison
is an explicitly post-hoc development ablation, with its own [protocol and
actual outcomes](../../../artifacts/traverse/fdm_rgbd_demo_v1/combined_capped_physical_comparison.json).
Speed bounds also do not prove support for every deformed path.

The evidence is for one initial MPPI selection followed by fixed-reference PID
execution. Repeated online replanning, arbitrary-terrain generalization and
calibrated safety guarantees remain unvalidated. Terrain avoidance is earned
through predicted loss of progress; the four-second event head cannot establish
a confirmed blockage that first matures after five to nine seconds.

## What is implemented

Current four-channel RGB-D and measured recent vehicle history condition a
forward GRU over each proposed path/speed sequence. The network predicts
residual body velocities, integrated motion, work and failure probabilities.
Goal/time/risk costs remain outside the network, and MPPI refines several
geometric route families. This is an HMMWV adaptation of the reference's
information flow, not a copy of its quadruped architecture or weights.

The first RGB-D encoder retains an indexed spatial feature grid before its
projection. The follow-up also extracts observed RGB-D patches along each
candidate reference, feeding their learned features directly to the forward
GRU. Joint image/world-pose quarter turns discourage fixed-map memorization.
Neither version receives authored terrain heights or obstacle locations.

The images are from the existing fixed overhead camera. Measured world pose
and a supplied goal remain inputs; no camera-only localization claim is made.
The standard Chrono PID driver retains its original terrain-height lookup
solely to construct its 3D tracking curve. That controller privilege is
excluded from the neural scorer, geometric route generator and MPPI filtering.

An optional execution-only `--path-height-source current_depth` mode builds
the PID's path heights from the saved observed depth point cloud, rejecting
missing coverage without any terrain-query fallback. The first focused rock
reference also succeeded in this mode in 6.55 s, with matched initial state
and pixels. This is a separate flat-terrain controller ablation, not evidence
for general height reconstruction through occlusion or steep terrain. See the
[controller comparison](../../../artifacts/traverse/fdm_rgbd_demo_v1/depthheight_controller_comparison.json).

Implementation: [model](../../../src/nedm/traverse/fdm_rgbd_model.py),
[sensor inputs](../../../src/nedm/traverse/fdm_rgbd_data.py),
[planner](../../../src/nedm/traverse/fdm_rgbd_planner.py),
[separate observation/planning/execution entry points](../../../scripts/traverse_fdm_rgbd_chrono.py).

## First RGB-D training and physical comparison

The reused pack has 20,000 training and 5,000 validation current-frame images
from the frozen 1,000/250 episode identities. All eight modality/seed variants
trained on AMD: job `409527` completed the 100-update smoke in 27 s, and job
`409531` completed the 3,000-update pilot in 79 s. Source snapshot
`78cf98ea8610`, full source/data hashes and checkpoint provenance are retained.
No optimizer updates ran locally.

The fixed-budget RGB-D models had 4 s endpoint errors of 0.866/0.750 m, versus
0.606/0.639 m for matched blank-image models. The seed-11 selected checkpoint
was almost invariant to shuffled or blank images. Seed 29 changed under image
interventions, but did not show consistent useful visual improvement. High
pooled event AUC did not establish anticipation of unseen hazards. See the
[modality and prospective-failure report](../../../artifacts/traverse/fdm_rgbd_runs/rgbd_pilot_v1/report.md).

The physical selection protocol fixed `rgbd_s11/last.pt` at 3,000 updates,
candidate families and cost settings before examining physical outcomes.
Every reference began from the same measured state and RGB-D observation.
Twelve base references and two selected MPPI refinements were executed on AMD.

| Development scene | Selected RGB-D + MPPI outcome | Measured alternatives |
|---|---|---|
| Visible rock, 8 m ahead | Contact, timeout and sustained stopping; 4.81 m goal progress | Both straight routes fail; all four detours reach safely, fastest in 7.00 s |
| Hill crossing | Reaches safely in 6.90 s | Straight 6 m/s: 8.55 s; fastest unrefined detour: 7.10 s; all six routes succeed |

The hill demonstrates a measured time improvement, but does not test stall
avoidance because none of its alternatives stalls. The rock exposes a clear
perception/risk failure: the model assigned roughly 1e-6 contact probabilities
and selected the blocked route. These are development scenes and fixed-route
executions, not untouched-terrain or continuous-replanning validation.

[Physical outcomes](../../../artifacts/traverse/fdm_rgbd_demo_v1/physical_comparison.json),
[measured route/time figure](../../../artifacts/traverse/fdm_rgbd_demo_v1/measured_route_choices.png),
[predeclared protocol](../../../artifacts/traverse/fdm_rgbd_demo_v1/selection_protocol.json).

## Measured gap and focused follow-up

The focused obstacle comparison is complete: 16 training and four validation
scenes, six recorded sibling references per scene, 908/222 current-image
windows. Eight models trained for 5,000 updates on AMD in job `409566` (274 s),
after the eight-model smoke `409565` (38 s). All arms used identical data,
per-seed samples and image/world rotations. The selected model and fixed update
count were declared before evaluating its physical demonstration.

| Fixed-last model | Safe selections / four scenes | Mean time regret against fastest safe sibling |
|---|---:|---:|
| RGB-D with candidate patches, seed 11 (primary) | 4/4 | 0.000 s |
| RGB-D with candidate patches, seed 29 | 4/4 | 0.025 s |
| Blank-input patch model, seed 11 | 4/4 | 0.350 s |
| Blank-input patch model, seed 29 | 3/4 | 0.250 s on its safe completions |

The primary model's same-checkpoint blank-image intervention reduces safe
selections to one, with three abstentions. Shuffling images changes two route
choices and increases endpoint error by 5.232 m, but still selects four safe
routes; it does not demonstrate a safety degradation from shuffling. These are
small, designed development cohorts, and the table ranks recorded families
without MPPI refinement. See the [complete comparison](../../../artifacts/traverse/fdm_rgbd_runs/rgbd_focused_pilot_v1/report.md).

On the original visible-rock development observation, the primary model now
rejects both straight paths with predicted contact probability 1.0 and selects
a fast detour. Those extreme probabilities are not a calibration claim.
Chrono job `409579` (27 s) independently executed the two saved MPPI references:

| Development scene | Actual selected route | Fastest safe unrefined sibling |
|---|---:|---:|
| Visible rock | Safe goal in 6.55 s | 7.00 s |
| Hill crossing | Safe goal in 6.85 s | 7.10 s |

Both have zero contact, strict sustained stall and bounded blockage. All
initial measured state/RGB/depth values match the scored observation exactly,
with pose storage roundoff below 9e-7 m. Blanking the rock observation in the
same model makes it abstain. The [measured figure](../../../artifacts/traverse/fdm_rgbd_demo_v1/focused_measured_route_choices.png)
shows the actual paths and goal progress. First-pilot outputs remain intact.

At anchor zero, the reused 1,000 training and 250 validation episodes contain
**zero contact or low-progress positives within the prediction horizon**.
Across the first two anchor seconds, there is only one positive training
window and no positive validation window. Most later positives start after
contact or stopping is already observable. This is a concrete missing-data
condition alongside the architecture's weak use of images.

Fresh standard-PID sibling collection therefore varies obstacle location,
size, pose and route; safe-route identity changes with the image. Separate
terrain constructions are physically probed for failure and safe alternatives.
All siblings of a scene share a declared train/validation split; the first
demonstration scenes remain excluded from training. Recorded images correspond
to each actual current anchor, accompanied by full 20 Hz telemetry.

The first audited six siblings already contain four contact-positive launch
windows, one sustained-stall-positive launch window and two contact-free
routes. The follow-up preserves low-net-progress diagnostics and separately
trains a named sustained-stall target: any fully future contiguous 2 s interval
with |body vx| below 0.3 m/s under throttle above 0.3, excluding parking. This
captures moving and then stopping even when the whole 4 s net displacement is
larger than the old low-progress threshold.

[Original launch-condition support audit](../../../artifacts/traverse/fdm_rgbd_focused_v1/support_audit/early_anchor_support.md),
[first focused-scene support](../../../artifacts/traverse/fdm_rgbd_focused_v1/support_audit/first_scene_support.json).

## Terrain-blockage follow-up

The combined pack retains every original focused-obstacle tensor/image row
exactly and adds 72 separately declared terrain siblings. It has 24 training
and eight validation scenes (144/48 episodes, 1,341/440 windows). On the terrain
cohort, 24 routes become blocked and all 48 alternatives reach safely. The
failed route identity changes with terrain location. The new diagnostic scene
is excluded from this pack, and only its initial observation was generated
before freezing the model-selection protocol.

All eight variants completed the combined-data smoke (`409580`) and the
5,000-update pilot (`409581`, 286 s). The primary diagnostic selections were
saved before any of its eight physical executions (`409589`, 31 s). The two
original-scene regressions completed in `409591` (28 s), and all three capped
references completed in `409592` (28 s). No local optimizer updates occurred.

Bounded-motion semantic version 2 is a separate label: a fully future 2 s
window whose measured XY endpoints have diameter at most 0.25 m, with applied
throttle above 0.3 throughout and no parking or goal arrival. It catches
oscillation in place that the original instantaneous-velocity criterion can
miss. The original net-progress and strict-stall targets remain available.
Tests found no false blockage among the 73 successful obstacle/mesa-probe
routes, and no normal initial launch was flagged. This criterion is intended
for the tested 4/6 m/s commands, not deliberate crawling.

There is a material horizon limit: terrain blockage confirmation occurs at
5.05–9.50 s, so no terrain example has a confirmed bounded-motion event by the
initial four-second horizon. However, actual four-second goal progress is
4.96–6.31 m for the later-blocked routes versus 11.12–20.05 m for safe routes.
The model can therefore earn a route choice through motion/progress prediction;
that would not establish early explicit prediction of a later confirmed stall.
See the [horizon audit](../../../artifacts/traverse/fdm_rgbd_focused_v1/support_audit/terrain_horizon_support_v2.json)
and [new diagnostic protocol](../../../artifacts/traverse/fdm_rgbd_stall_demo_v1/selection_protocol.json).

## Calibration and checks

Legacy OptiX depth uses its measured ray scale 1.2; the Vulkan backend uses 1.0.
An analytical box/wall check verified Vulkan Euclidean ranges. Two identical
validation scenes replayed across renderers differed by about 0.50/255 RGB MAE
and 0.020–0.023 m registered elevation MAE on commonly valid pixels. Valid-depth
masks and physics runtimes still differ and are recorded explicitly.

Checks cover current-image causality, depth registration, RGB/depth gradients,
modality isolation, cached/direct predictions, interval labels/censoring,
post-arrival parking, consistent stations, cusp rejection and MPPI abstention
when all candidates exceed declared risk limits. The risk thresholds remain
operating choices, not demonstrated calibration guarantees. The comparison's
fixed limits are 0.35 contact probability and 0.5 progress/stall probability.

[Renderer comparison](../../../artifacts/traverse/fdm_rgbd_demo_v1/renderer_replay/comparison.json),
[planner checks](../../../artifacts/traverse/fdm_rgbd_planner_checks.json),
[execution record](../../../artifacts/traverse/fdm_rgbd_execution_20260908.json).

All source edits and outputs remain isolated in `/home/harry/NeDM-traverse_mppi`
on branch `traverse_mppi`, with an independent AMD experiment directory. The
active `/home/harry/NeDM` checkout and its jobs have not been modified.
