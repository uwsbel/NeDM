# Diverse FDM evidence report

This report evaluates saved forecasts on the frozen val pack. It does not establish closed-loop MPPI success or full-route energy efficiency.

Pack: `/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909/packs/full_pair_v1/h20`. Scenes: 6; episodes: 90; overlapping windows: 13625.

Fixed-budget final checkpoints and validation-selected best checkpoints are shown separately. Unobserved tails are masked; unsupported event heads are not counted as valid risk predictors.

## Fixed final update

| Run | Step | Horizon | Complete windows | FDE (m) | Work MAE (kJ) | Roll MAE (deg) | Pitch MAE (deg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| rgbd_s11 | 5000 | 4s | 13330 | 2.841 | 34.496 | 3.854 | 3.029 |
| blank_s11 | 5000 | 4s | 13330 | 2.521 | 27.444 | 4.279 | 2.875 |
| rgbd_s29 | 5000 | 4s | 13330 | 2.735 | 38.126 | 5.140 | 2.967 |
| blank_s29 | 5000 | 4s | 13330 | 2.582 | 29.259 | 3.653 | 3.107 |

## Validation-selected best

| Run | Step | Horizon | Complete windows | FDE (m) | Work MAE (kJ) | Roll MAE (deg) | Pitch MAE (deg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| rgbd_s11 | 5000 | 4s | 13330 | 2.841 | 34.496 | 3.854 | 3.029 |
| blank_s11 | 3000 | 4s | 13330 | 2.721 | 30.446 | 3.305 | 3.512 |
| rgbd_s29 | 3000 | 4s | 13330 | 3.044 | 44.377 | 4.629 | 3.857 |
| blank_s29 | 5000 | 4s | 13330 | 2.582 | 29.259 | 3.653 | 3.107 |

## Support and image controls

- rgbd_s11: supported contact/rollover/progress heads = `[True, True, True]`; progress target `bounded_motion`.
- blank_s11: supported contact/rollover/progress heads = `[True, True, True]`; progress target `bounded_motion`.
- rgbd_s29: supported contact/rollover/progress heads = `[True, True, True]`; progress target `bounded_motion`.
- blank_s29: supported contact/rollover/progress heads = `[True, True, True]`; progress target `bounded_motion`.

Hash-verified raw prefixes support pre-onset/already-observed-event strata for 13625 windows. Detailed horizon, per-scene, hazard, image-control and matched-arm results are in `report.json`.

For physical MPPI evaluation, compare completion/failure/abstention first. Compare time and mechanical work only for paired safe full-goal completions with the same start, goal and time cap. A stalled vehicle can consume less total energy and must not win an efficiency comparison by failing early.

Signed attitude endpoint predictions and solver-step peak attitude are different targets; both comparisons are retained. Reported full-route work/time extrapolations remain planning heuristics until measured in Chrono.
