# MPPI + RGB-D forward model: four-route Chrono diagnostic

Open [the prediction overview](visualization/prediction_overview.png), then the
[side-by-side video](visualization/four_route_comparison.mp4) or each individual:

1. [MPPI selected](visualization/route_01_selected_best.mp4): safe goal in 6.85 s.
2. [Rock crossing](visualization/route_02_collision_collision.mp4): collision at 2.072 s, stopping, timeout.
3. [Berm crossing](visualization/route_03_stall_stall.mp4): terrain stall without asset collision, timeout.
4. [Slower clear route](visualization/route_04_progress_low_progress.mp4): safe goal in 9.65 s.

The overview uses one measured current RGB-D observation. Gray lines show 592
unique scored candidates; solid colored lines show the four-second learned
forecasts, while dashed tails are commanded routes beyond that horizon.
Video scene frames are actual Chrono renders. Cyan traces show measured motion
so far. Playback includes a labeled final-frame hold; physical time is in the HUD.

| Route | Predicted goal progress at 4 s | Actual at 4 s |
|---|---:|---:|
| MPPI selected | 17.25 m | 17.18 m |
| Rock crossing | 6.45 m | 6.76 m |
| Berm crossing | 5.12 m | 5.05 m |
| Slower clear route | 11.97 m | 11.66 m |

The progress ranking is correct, and the selected route is fastest among these
four executions. Explicit event prediction is incomplete: predicted rock contact
probability was only 0.0103%, a missed collision inside the four-second horizon.
Terrain bounded stopping was confirmed at 7.55 s, outside that horizon; this
does not establish explicit early prediction of the later stall.

All four references and forecasts were frozen before execution. Starting RGB-D
and measured state matched exactly. The existing fixed model was reused without
retraining; all four Chrono runs executed in parallel on AMD (job 411816).
This purpose-built scene validates one initial MPPI decision followed by PID
tracking, not repeated online replanning or general safety guarantees.

See [the detailed comparison](comparison.md), [machine-readable results](comparison.json),
[execution checks](actual_validation.json), and
[media provenance](visualization/visualization_provenance.json).
