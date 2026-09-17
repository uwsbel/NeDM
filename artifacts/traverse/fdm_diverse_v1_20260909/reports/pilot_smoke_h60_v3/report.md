# Diverse FDM evidence report

This report evaluates saved forecasts on the frozen val pack. It does not establish closed-loop MPPI success or full-route energy efficiency.

Pack: `/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_diverse_v1_20260909/packs/pilot_h60_v2`. Scenes: 1; episodes: 3; overlapping windows: 180.

Fixed-budget final checkpoints and validation-selected best checkpoints are shown separately. Unobserved tails are masked; unsupported event heads are not counted as valid risk predictors.

## Fixed final update

| Run | Step | Horizon | Complete windows | FDE (m) | Work MAE (kJ) | Roll MAE (deg) | Pitch MAE (deg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| rgbd_s11 | 100 | 4s | 171 | 5.576 | 68.934 | 8.451 | 5.220 |
| rgbd_s11 | 100 | 8s | 159 | 10.716 | 145.428 | 7.186 | 3.825 |
| rgbd_s11 | 100 | 12s | 147 | 16.820 | 235.428 | 7.619 | 6.577 |
| blank_s11 | 100 | 4s | 171 | 5.558 | 65.402 | 8.549 | 4.531 |
| blank_s11 | 100 | 8s | 159 | 10.720 | 139.100 | 8.945 | 3.434 |
| blank_s11 | 100 | 12s | 147 | 16.804 | 226.191 | 8.094 | 6.484 |
| rgbd_s29 | 100 | 4s | 171 | 5.957 | 62.183 | 8.251 | 5.216 |
| rgbd_s29 | 100 | 8s | 159 | 11.620 | 137.685 | 8.611 | 4.060 |
| rgbd_s29 | 100 | 12s | 147 | 17.922 | 232.723 | 12.067 | 4.880 |
| blank_s29 | 100 | 4s | 171 | 5.946 | 61.552 | 7.338 | 5.187 |
| blank_s29 | 100 | 8s | 159 | 11.663 | 136.696 | 8.077 | 5.373 |
| blank_s29 | 100 | 12s | 147 | 17.957 | 232.134 | 11.064 | 7.823 |

## Validation-selected best

| Run | Step | Horizon | Complete windows | FDE (m) | Work MAE (kJ) | Roll MAE (deg) | Pitch MAE (deg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| rgbd_s11 | 100 | 4s | 171 | 5.576 | 68.934 | 8.451 | 5.220 |
| rgbd_s11 | 100 | 8s | 159 | 10.716 | 145.428 | 7.186 | 3.825 |
| rgbd_s11 | 100 | 12s | 147 | 16.820 | 235.428 | 7.619 | 6.577 |
| blank_s11 | 100 | 4s | 171 | 5.558 | 65.402 | 8.549 | 4.531 |
| blank_s11 | 100 | 8s | 159 | 10.720 | 139.100 | 8.945 | 3.434 |
| blank_s11 | 100 | 12s | 147 | 16.804 | 226.191 | 8.094 | 6.484 |
| rgbd_s29 | 100 | 4s | 171 | 5.957 | 62.183 | 8.251 | 5.216 |
| rgbd_s29 | 100 | 8s | 159 | 11.620 | 137.685 | 8.611 | 4.060 |
| rgbd_s29 | 100 | 12s | 147 | 17.922 | 232.723 | 12.067 | 4.880 |
| blank_s29 | 100 | 4s | 171 | 5.946 | 61.552 | 7.338 | 5.187 |
| blank_s29 | 100 | 8s | 159 | 11.663 | 136.696 | 8.077 | 5.373 |
| blank_s29 | 100 | 12s | 147 | 17.957 | 232.134 | 11.064 | 7.823 |

## Support and image controls

- rgbd_s11: supported contact/rollover/progress heads = `[True, False, True]`; progress target `bounded_motion`.
  - last image shuffle unavailable: Fewer than two distinct validation scene images.
  - best image shuffle unavailable: Fewer than two distinct validation scene images.
- blank_s11: supported contact/rollover/progress heads = `[True, False, True]`; progress target `bounded_motion`.
  - last image shuffle unavailable: Fewer than two distinct validation scene images.
  - best image shuffle unavailable: Fewer than two distinct validation scene images.
- rgbd_s29: supported contact/rollover/progress heads = `[True, False, True]`; progress target `bounded_motion`.
  - last image shuffle unavailable: Fewer than two distinct validation scene images.
  - best image shuffle unavailable: Fewer than two distinct validation scene images.
- blank_s29: supported contact/rollover/progress heads = `[True, False, True]`; progress target `bounded_motion`.
  - last image shuffle unavailable: Fewer than two distinct validation scene images.
  - best image shuffle unavailable: Fewer than two distinct validation scene images.

Hash-verified raw prefixes support pre-onset/already-observed-event strata for 180 windows. Detailed horizon, per-scene, hazard, image-control and matched-arm results are in `report.json`.

For physical MPPI evaluation, compare completion/failure/abstention first. Compare time and mechanical work only for paired safe full-goal completions with the same start, goal and time cap. A stalled vehicle can consume less total energy and must not win an efficiency comparison by failing early.

Signed attitude endpoint predictions and solver-step peak attitude are different targets; both comparisons are retained. Reported full-route work/time extrapolations remain planning heuristics until measured in Chrono.
