# Smooth-hill correction to the four-route Chrono demonstration

The original middle berm had a short, wall-like face. This revision replaces it
with a broad rounded hill and tests an aligned uphill wheel-slip stall. The
vehicle, tire friction, powertrain, native PID, rock position, start and goal are
unchanged. The hill moves to the scene midpoint so the two detours separate
before the rising ground.

## View the corrected scene

- [Old versus new hill profile](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/terrain_profile/terrain_profile_comparison.png)
  ([PDF](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/terrain_profile/terrain_profile_comparison.pdf)).
- [Frozen prediction overview: 567 scored candidates and four highlighted routes](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/visualization/prediction_overview.png).
- [Video 1: MPPI selected route](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/visualization/route_01_selected_best.mp4).
- [Video 2: Rock-crossing reference](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/visualization/route_02_collision_collision.mp4).
- [Video 3: Smooth hill, 4 m/s climb](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/visualization/route_03_stall_stall.mp4).
- [Video 4: Clear 4 m/s detour](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/visualization/route_04_progress_low_progress.mp4).
- [Four videos side by side](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/visualization/four_route_comparison.mp4).
- [Media and evidence bundle](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/fourway_smooth_media.zip).

## What changed physically

| Terrain property | Original berm | Smooth hill |
|---|---:|---:|
| Relief above nearby ground | 1.90 m | 6.49 m |
| Distance over which elevation rises from 10% to 90% | 0.58 m | 7.57 m |
| Peak uphill grade over a 0.5 m span | 70.36 degrees | 41.78 degrees |
| World center | (-8, 0) m | (0, 0) m |

The replacement is a Gaussian hill with design height 6.5 m and standard
deviation 4.5 m. Its toe and crest are rounded. Profile measurements use the
actual stored heightfield, with the finite-difference span disclosed; individual
grid-scale grade estimates are noisier. The profile is scene verification,
not an input to the neural model.

In the final recorded execution, the 4 m/s straight command climbs the hill and then
shows sustained effortful low progress while facing uphill at approximately
34 degrees pitch. Four tires carry force and exhibit slip under full throttle,
with no braking, asset collision or sampled chassis-contact resultant. Bounded
motion is confirmed at 11.45 s; the strict low-speed criterion is confirmed at
17.20 s. Small intermittent movements remain, so this is not a claim of exactly
zero motion throughout the recording.

During 20–25 s, median pitch is 34.49 degrees nose uphill and yaw is 0.79
degrees, with full throttle, zero braking and first gear. Front-wheel
circumferential speeds are 4.97 and 8.50 m/s while their horizontal hub speeds
are only 0.10 and 0.08 m/s. This supports a wheel-slip, low-progress failure on
the incline. The [final mechanism measurements](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/final_stall_mechanism.json)
retain raw tire-force values and their interpretation limits.

The same hill's straight 6 m/s command reaches the goal in 8.75 s. This is a
terrain-and-command interaction, not a claim that the hill blocks every speed.
The 6 m/s and 4 m/s clear detours reach in 7.80 s and 14.50 s in the geometry
controls. A more stationary alternative was rejected because it ended after
rollback with saturated steering on shallower ground; it was a less direct
illustration of the requested uphill stall.

## Final predictions and Chrono outcomes

All four recorded runs are complete. Goal progress is reduction in Euclidean
distance to the goal, measured at the same four-second forecast endpoint.

| Route | Predicted 4 s progress | Actual 4 s progress | Full Chrono outcome |
|---|---:|---:|---|
| MPPI selected | 17.04 m | 14.61 m | Safe goal at 8.10 s |
| Rock crossing | 8.73 m | 6.61 m | Contact at 2.09 s; 25 s timeout |
| Smooth hill, straight 4 m/s | 9.65 m | 11.71 m | Bounded stop at 11.45 s; strict stall at 17.20 s; 25 s timeout |
| Clear detour, 4 m/s | 11.60 m | 10.74 m | Safe goal at 14.50 s |

The selected route is the fastest safe completion among these four videos.
However, the original clear 6 m/s reference reaches in **7.80 s**, so the selected
route is **0.30 s slower than that known safe control**. This does not establish
improved or globally optimal travel time.

The clear 6 m/s and straight 6 m/s controls exactly match frozen base candidates
5 and 1, respectively, and share the final measured initial RGB-D/state/history.
Their retained [clear-route outcome](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/probes/siblings/smooth_strength_gaussian_h6p5_s4p5_cx0/family_05/outcome.json)
and [straight-route outcome](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/probes/strength/smooth_strength_gaussian_h6p5_s4p5_cx0/family_01/outcome.json)
are safe. The model also assigns the straight 6 m/s control **43.91% contact
risk** and rejects it, providing a hazard false positive alongside the
rock-route false negative below.

Explicit hazard prediction remains incomplete. The rock route's predicted
contact probability is **1.14%**, below the declared 35% threshold, despite
contact inside four seconds. The hill route's predicted bounded-stop probability
is **0.00435% by four seconds**, while its actual bounded stop is confirmed at
11.45 s. The latter is outside the initial learned horizon; this experiment
does not establish early prediction of the later hill stall.

Maximum four-second progress error is **2.43 m** and endpoint errors range from
0.88 to 2.49 m. The model reverses the hill-versus-slower-detour progress order
at four seconds. The original berm scene's stronger accuracy and complete
progress ordering therefore do not carry over to this scene. Time-to-goal cost
beyond four seconds remains an analytical extrapolation.

See the [full numerical comparison](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/comparison.md)
and [machine-readable comparison](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/comparison.json).

## Experimental boundary

The fixed RGB-D checkpoint is unchanged:
`rgbd_focused_pilot_v2/patch_rgbd_s11/last.pt` (5,000 earlier AMD updates).
No retraining occurred. All 37 geometry-development executions are retained,
including successful climbs, bad-launch cases and non-monotonic outcomes across
nearby hill heights. These are purpose-selected development scenes, not an
unbiased benchmark or a universal grade threshold.

The final planner scores the exact original six geometric command references
from the shared measured RGB-D/state, then performs bounded MPPI refinement.
The three diagnostic references are unchanged commands from the physical
probes, preventing small changes in their starts from confounding the comparison.
All 567 forecasts and the four references were frozen before the final recorded
executions. The learned horizon remains four seconds; longer dashed route tails
in the overview are commands, not learned future motion.

The simulator supplies a separate camera and passive wheel/contact diagnostics
for visualization and validation. Those measurements are not additional neural
inputs. Tire-force vectors, native tire slip, pose, effort and actual height
profiles support the failure analysis. Terrain-normal query projections are
approximate; they are not claimed as exact calibrated tire-contact-frame loads.
Twenty-hertz chassis-contact sampling cannot exclude every brief contact between
samples, although no chassis-contact resultant was sampled in the final hill run.

The final four-run AMD job `411859` completed successfully in 8 min 19 s. All
**730 source camera frames** match their recorded timestamps, poses, states and
actions exactly. The starting measured RGB-D/state and first visual frame are
identical across the four runs. Frozen source and reference hashes pass; all
three diagnostic runs reproduce their prior probe trajectories exactly with
the passive video observer enabled. The MPPI-selected command is new and is
validated against its frozen reference and shared initial observation.
See [execution validation](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/actual_validation.json)
and [media checks](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/visualization/media_qa.json).

All physics and rendering run on AMD. Changes and artifacts remain in the
`traverse_mppi` worktree; original scene media and the busy main checkout are
preserved. This remains one initial MPPI decision followed by PID execution,
not continuous online replanning.

Evidence: [scene protocol](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/scene_protocol.json),
[all geometry probes](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/geometry_probe_index.json),
[independent stall-mechanism review](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/probe_refine_recommendation.md),
[frozen model selection](../../../artifacts/traverse/fdm_rgbd_fourway_smooth_20260909/prediction/selection.json).

Implementation: [hill generation](../../../scripts/traverse_fdm_rgbd_smooth_hills.py),
[passive diagnostic observer](../../../src/nedm/traverse/fdm_slope_probe.py),
[combined video/diagnostic wrapper](../../../scripts/traverse_fdm_rgbd_smooth_video.py),
[profile rendering](../../../scripts/traverse_fdm_rgbd_terrain_profile.py),
[planning and frozen reference selection](../../../scripts/traverse_fdm_rgbd_fourway_plan.py).
