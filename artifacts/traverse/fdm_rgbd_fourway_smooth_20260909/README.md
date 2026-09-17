# Smooth hill: frozen RGB-D MPPI forecasts and four Chrono executions

The original short, wall-like berm is replaced by a broad Gaussian hill. Its
10–90% rising transition is 7.57 m; peak uphill grade over a 0.5 m span is
41.78 degrees. The vehicle, friction, native PID and model checkpoint are
unchanged. The hill center moves to (0, 0) so the detours separate before it.

## View

- [Prediction overview: 567 candidates, four highlighted routes](visualization/prediction_overview.png)
- [Old versus new physical hill profile](terrain_profile/terrain_profile_comparison.png)
- [All four videos side by side](visualization/four_route_comparison.mp4)

| Individual video | Measured consequence |
|---|---|
| [MPPI selected](visualization/route_01_selected_best.mp4) | Safe goal at 8.10 s |
| [Rock crossing](visualization/route_02_collision_collision.mp4) | Contact at 2.09 s, timeout at 25 s |
| [Smooth hill, straight 4 m/s](visualization/route_03_stall_stall.mp4) | Uphill wheel-slip stall; bounded stop at 11.45 s, strict stall at 17.20 s, timeout at 25 s |
| [Clear detour, 4 m/s](visualization/route_04_progress_low_progress.mp4) | Safe goal at 14.50 s |

Videos contain actual Chrono camera frames and measured telemetry. Colored
solid paths show frozen initial four-second neural forecasts; dashed tails are
commands beyond that horizon. Cyan traces show only measured motion through the
displayed timestamp. Individual videos include an explicitly labeled one-second
final-frame hold; the comparison holds completed runs at their last measured time.

## What this establishes and what remains unresolved

The rounded hill produces low progress with wheel slip under full throttle,
zero braking and first gear, while the HMMWV remains pointed uphill. No asset
collision or chassis-contact resultant was sampled in that run. Brief contacts
between samples cannot be excluded. The same straight route at 6 m/s succeeds
in 8.75 s, so this is a terrain-and-command interaction.

The selected route is fastest among the four videos, but a known clear 6 m/s
base reference reaches in 7.80 s, 0.30 s faster. No global optimality or improved
travel time is established. The model assigns the rock route only 1.14% contact
risk despite collision inside its four-second horizon. The hill's confirmed
stall occurs beyond that horizon; explicit early prediction is not established.
Four-second progress errors reach 2.43 m, including a hill/slower-detour ranking
reversal. Route names are experimental hypotheses, not positive risk predictions.

This is a purpose-selected development scene and one initial MPPI decision
followed by fixed PID execution. No retraining or repeated online replanning was
performed. All physics/rendering ran on AMD. The busy main checkout and original
scene media were preserved; work is isolated on branch traverse_mppi.

## Evidence

- [Prediction versus measured outcomes](comparison.md), [full values](comparison.json)
- [All four execution checks](actual_validation.json)
- [Final hill mechanism measurements](final_stall_mechanism.json)
- [Scene protocol](scene_protocol.json), [execution freeze](execution_protocol.json)
- [Model selection](prediction/selection.json), [frozen artifact hashes](prediction/frozen_artifacts.json)
- [Geometry development index](geometry_probe_index.json)
- [Video provenance](visualization/visualization_provenance.json), [decode and visual checks](visualization/media_qa.json)

All 730 source camera frames exactly match their recorded telemetry. All four
start with identical measured RGB-D/state. The three diagnostic executions
exactly reproduce their development-probe trajectories with passive rendering.
The final AMD job 411859 completed in 8 min 19 s. This portable bundle contains
media and summary evidence; provenance also names full raw artifacts retained
in the experiment directory, which are not all included in the ZIP.
