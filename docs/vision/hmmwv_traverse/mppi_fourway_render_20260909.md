# One scene, four routes: frozen MPPI predictions and Chrono video validation

**Smooth-hill revision:** [The requested rounded hill and four new Chrono videos
are complete](mppi_smooth_hill_revision_20260909.md). The results below describe
the original berm scene and remain preserved as a separate experiment.

The learned scorer selected the fastest safe route among the four executed
references. It also predicted their four-second progress ranking correctly.
**Explicit hazard prediction remains incomplete:** it missed a rock collision
inside its forecast horizon; the terrain stall was confirmed after that horizon.

The scene deliberately combines a rock, a steep berm and a clear detour. This is
a designed diagnostic using the existing fixed model, not an unbiased benchmark.

## View the results

- [Prediction overview: 592 scored candidates and four highlighted routes](../../../artifacts/traverse/fdm_rgbd_fourway_20260909/visualization/prediction_overview.png)
  ([PDF](../../../artifacts/traverse/fdm_rgbd_fourway_20260909/visualization/prediction_overview.pdf)).
- [Video 1 — MPPI selected: safe goal in 6.85 s](../../../artifacts/traverse/fdm_rgbd_fourway_20260909/visualization/route_01_selected_best.mp4).
- [Video 2 — Rock crossing: collision, stopping, timeout](../../../artifacts/traverse/fdm_rgbd_fourway_20260909/visualization/route_02_collision_collision.mp4).
- [Video 3 — Berm crossing: terrain stall without asset collision](../../../artifacts/traverse/fdm_rgbd_fourway_20260909/visualization/route_03_stall_stall.mp4).
- [Video 4 — Slower clear route: safe goal in 9.65 s](../../../artifacts/traverse/fdm_rgbd_fourway_20260909/visualization/route_04_progress_low_progress.mp4).
- [All four videos side by side](../../../artifacts/traverse/fdm_rgbd_fourway_20260909/visualization/four_route_comparison.mp4).
- [Download the media and comparison bundle](../../../artifacts/traverse/fdm_rgbd_fourway_20260909/fourway_media.zip).

In the overview, solid colored lines are actual learned forecasts through four
seconds; dashed tails are commanded references beyond that horizon. Faint lines
are the 592 unique candidates scored during planning. The videos show actual
Chrono camera frames, measured telemetry and a cyan trace of positions already
visited. A labeled one-second final-frame hold makes the endpoint readable.

## Predicted versus measured

Goal progress is the reduction in Euclidean distance to the goal, compared at
exactly four seconds for every route. Goal completion uses the declared 2.5 m
arrival radius.

| Route | Predicted progress at 4 s | Chrono progress at 4 s | Full Chrono consequence |
|---|---:|---:|---|
| 1. MPPI selected | 17.25 m | 17.18 m | Goal at 6.85 s; no collision or stall |
| 2. Rock crossing | 6.45 m | 6.76 m | Rock contact at 2.072 s; stopped; 15 s timeout |
| 3. Berm crossing | 5.12 m | 5.05 m | Bounded stop confirmed at 7.55 s; no asset contact; 15 s timeout |
| 4. Slower clear route | 11.97 m | 11.66 m | Goal at 9.65 s; no collision or stall |

The maximum progress error is 0.316 m; predicted four-second endpoint errors
range from 0.101 to 0.403 m. Route 4 is a safe 4 m/s command with less progress
over the same duration. It is not a failure to reach the goal.

The rock route's predicted contact probability is only **0.0103%**, despite
contact at 2.072 s: an explicit within-horizon risk miss. The model discourages
this route through predicted poor progress. The terrain route's predicted
confirmed-stop probability is **0.0590% by four seconds**; its actual bounded stop
is first confirmed at 7.55 s. That later event is outside the initial forecast,
so this run does not establish explicit early stall prediction. Strict sustained
stall is confirmed at 9.50 s. Bounded stop and strict stall are separate criteria.

The four labels describe the chosen experimental cases; they are not four
categorical neural predictions. “Selected” means lowest learned cost among the
scored candidates. Its observed 6.85 s is fastest among these four executions,
not proof of a global optimum. The analytic time-to-goal cost extrapolates beyond
four seconds; it is not a learned full-route arrival-time forecast.

## Reproducibility and scope

The fixed checkpoint is
`rgbd_focused_pilot_v2/patch_rgbd_s11/last.pt` (5,000 AMD updates; source snapshot
`83d62378c36e`). No retraining occurred for this scene. All four routes and
predictions were saved and hashed before executing any of them. Candidate speeds
are capped at the focused training command ceiling of 6 m/s; that bound alone
does not guarantee support for every deformed path.

Each run starts from exactly the same measured RGB-D and vehicle context.
The neural scorer receives current overhead RGB-D, measured history/pose and
candidate commands. Authored scene geometry is excluded from scoring and MPPI
filtering. The standard Chrono PID driver uses terrain heights for its 3D
tracking reference. The separate oblique camera is only for the videos.

AMD job `411816` executed the four physics/rendering runs in parallel in
4 min 41 s. All 470 source frames match recorded telemetry, and initial camera
frames are identical across runs. Frozen reference/source hashes and observation
equality checks passed. Camera instrumentation also passed a no-camera physics
parity check. Work remains isolated in `NeDM-traverse_mppi`, branch
`traverse_mppi`; the main working tree and its running jobs were not modified.

This validates one initial MPPI selection followed by fixed-reference PID
execution. It does not yet validate repeated online replanning or calibrated
collision/stall avoidance across unseen terrain.

Detailed evidence: [comparison](../../../artifacts/traverse/fdm_rgbd_fourway_20260909/comparison.json),
[execution checks](../../../artifacts/traverse/fdm_rgbd_fourway_20260909/actual_validation.json),
[frozen predictions](../../../artifacts/traverse/fdm_rgbd_fourway_20260909/prediction/selection.json),
[media provenance](../../../artifacts/traverse/fdm_rgbd_fourway_20260909/visualization/visualization_provenance.json).

Implementation: [prediction recorder](../../../src/nedm/traverse/fdm_rgbd_diagnostics.py),
[planning and freeze script](../../../scripts/traverse_fdm_rgbd_fourway_plan.py),
[Chrono video capture](../../../scripts/traverse_fdm_rgbd_video.py),
[comparison](../../../scripts/traverse_fdm_rgbd_fourway_compare.py),
[media composition](../../../scripts/traverse_fdm_rgbd_fourway_compose.py).
