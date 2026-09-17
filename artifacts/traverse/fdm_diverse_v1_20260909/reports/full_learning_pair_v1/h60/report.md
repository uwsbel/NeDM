# Diverse FDM evidence report

This report evaluates saved forecasts on the frozen val pack. It does not establish closed-loop MPPI success or full-route energy efficiency.

Pack: `/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909/packs/full_pair_v1/h60`. Scenes: 6; episodes: 90; overlapping windows: 13625.

Fixed-budget final checkpoints and validation-selected best checkpoints are shown separately. Unobserved tails are masked; unsupported event heads are not counted as valid risk predictors.

## Fixed final update

| Run | Step | Horizon | Complete windows | FDE (m) | Work MAE (kJ) | Roll MAE (deg) | Pitch MAE (deg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| rgbd_s11 | 5000 | 4s | 13330 | 3.238 | 33.153 | 4.153 | 4.623 |
| rgbd_s11 | 5000 | 8s | 12970 | 4.440 | 64.027 | 4.784 | 3.609 |
| rgbd_s11 | 5000 | 12s | 12610 | 5.997 | 99.277 | 4.202 | 3.798 |
| blank_s11 | 5000 | 4s | 13330 | 2.987 | 30.261 | 3.401 | 4.192 |
| blank_s11 | 5000 | 8s | 12970 | 4.667 | 67.157 | 5.437 | 3.497 |
| blank_s11 | 5000 | 12s | 12610 | 6.623 | 102.166 | 6.113 | 3.625 |
| rgbd_s29 | 5000 | 4s | 13330 | 2.966 | 30.575 | 4.471 | 3.295 |
| rgbd_s29 | 5000 | 8s | 12970 | 4.423 | 56.558 | 5.476 | 4.144 |
| rgbd_s29 | 5000 | 12s | 12610 | 6.276 | 88.245 | 4.841 | 3.843 |
| blank_s29 | 5000 | 4s | 13330 | 2.736 | 30.691 | 3.917 | 3.578 |
| blank_s29 | 5000 | 8s | 12970 | 4.337 | 61.280 | 3.789 | 3.427 |
| blank_s29 | 5000 | 12s | 12610 | 6.374 | 98.737 | 4.829 | 3.488 |

## Validation-selected best

| Run | Step | Horizon | Complete windows | FDE (m) | Work MAE (kJ) | Roll MAE (deg) | Pitch MAE (deg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| rgbd_s11 | 4000 | 4s | 13330 | 3.026 | 34.749 | 3.446 | 3.703 |
| rgbd_s11 | 4000 | 8s | 12970 | 4.472 | 67.583 | 3.977 | 4.449 |
| rgbd_s11 | 4000 | 12s | 12610 | 5.975 | 102.198 | 5.107 | 4.418 |
| blank_s11 | 2000 | 4s | 13330 | 3.544 | 33.746 | 4.331 | 3.010 |
| blank_s11 | 2000 | 8s | 12970 | 5.498 | 70.415 | 6.075 | 3.781 |
| blank_s11 | 2000 | 12s | 12610 | 7.755 | 109.003 | 5.843 | 3.914 |
| rgbd_s29 | 4000 | 4s | 13330 | 2.965 | 30.846 | 3.716 | 4.324 |
| rgbd_s29 | 4000 | 8s | 12970 | 4.284 | 64.620 | 4.765 | 3.855 |
| rgbd_s29 | 4000 | 12s | 12610 | 6.243 | 98.614 | 4.755 | 3.774 |
| blank_s29 | 2000 | 4s | 13330 | 3.186 | 29.733 | 5.371 | 3.487 |
| blank_s29 | 2000 | 8s | 12970 | 4.729 | 62.741 | 5.157 | 3.550 |
| blank_s29 | 2000 | 12s | 12610 | 6.818 | 99.753 | 7.231 | 4.859 |

## Support and image controls

- rgbd_s11: supported contact/rollover/progress heads = `[True, True, True]`; progress target `bounded_motion`.
- blank_s11: supported contact/rollover/progress heads = `[True, True, True]`; progress target `bounded_motion`.
- rgbd_s29: supported contact/rollover/progress heads = `[True, True, True]`; progress target `bounded_motion`.
- blank_s29: supported contact/rollover/progress heads = `[True, True, True]`; progress target `bounded_motion`.

Hash-verified raw prefixes support pre-onset/already-observed-event strata for 13625 windows. Detailed horizon, per-scene, hazard, image-control and matched-arm results are in `report.json`.

For physical MPPI evaluation, compare completion/failure/abstention first. Compare time and mechanical work only for paired safe full-goal completions with the same start, goal and time cap. A stalled vehicle can consume less total energy and must not win an efficiency comparison by failing early.

Signed attitude endpoint predictions and solver-step peak attitude are different targets; both comparisons are retained. Reported full-route work/time extrapolations remain planning heuristics until measured in Chrono.
