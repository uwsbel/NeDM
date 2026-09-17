# Diverse FDM evidence report

This report evaluates saved forecasts on the frozen val pack. It does not establish closed-loop MPPI success or full-route energy efficiency.

Pack: `/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_diverse_v1_20260909/packs/pilot_h60_v2`. Scenes: 1; episodes: 3; overlapping windows: 180.

Fixed-budget final checkpoints and validation-selected best checkpoints are shown separately. Unobserved tails are masked; unsupported event heads are not counted as valid risk predictors.

## Fixed final update

| Run | Step | Horizon | Complete windows | FDE (m) | Work MAE (kJ) | Roll MAE (deg) | Pitch MAE (deg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| rgbd_s11 | 1000 | 4s | 171 | 7.050 | 86.242 | 6.361 | 4.087 |
| rgbd_s11 | 1000 | 8s | 159 | 12.573 | 174.058 | 7.135 | 3.882 |
| rgbd_s11 | 1000 | 12s | 147 | 18.825 | 261.283 | 5.710 | 4.460 |
| blank_s11 | 1000 | 4s | 171 | 6.091 | 77.733 | 6.570 | 4.270 |
| blank_s11 | 1000 | 8s | 159 | 11.082 | 159.157 | 4.947 | 3.870 |
| blank_s11 | 1000 | 12s | 147 | 16.335 | 238.664 | 5.432 | 4.742 |
| rgbd_s29 | 1000 | 4s | 171 | 7.266 | 82.757 | 5.523 | 5.144 |
| rgbd_s29 | 1000 | 8s | 159 | 13.336 | 165.411 | 8.120 | 4.200 |
| rgbd_s29 | 1000 | 12s | 147 | 19.436 | 249.567 | 7.811 | 3.875 |
| blank_s29 | 1000 | 4s | 171 | 5.573 | 51.770 | 6.990 | 8.099 |
| blank_s29 | 1000 | 8s | 159 | 9.884 | 107.273 | 9.723 | 6.179 |
| blank_s29 | 1000 | 12s | 147 | 14.631 | 190.813 | 8.404 | 5.014 |

## Validation-selected best

| Run | Step | Horizon | Complete windows | FDE (m) | Work MAE (kJ) | Roll MAE (deg) | Pitch MAE (deg) |
|---|---:|---:|---:|---:|---:|---:|---:|
| rgbd_s11 | 500 | 4s | 171 | 6.161 | 76.779 | 5.861 | 4.512 |
| rgbd_s11 | 500 | 8s | 159 | 10.863 | 159.434 | 6.031 | 3.959 |
| rgbd_s11 | 500 | 12s | 147 | 15.427 | 238.453 | 6.457 | 5.011 |
| blank_s11 | 250 | 4s | 171 | 5.072 | 64.747 | 6.794 | 5.656 |
| blank_s11 | 250 | 8s | 159 | 9.789 | 134.758 | 7.906 | 3.982 |
| blank_s11 | 250 | 12s | 147 | 15.649 | 215.708 | 8.118 | 9.525 |
| rgbd_s29 | 500 | 4s | 171 | 6.145 | 64.615 | 5.459 | 4.856 |
| rgbd_s29 | 500 | 8s | 159 | 11.130 | 134.750 | 6.958 | 3.800 |
| rgbd_s29 | 500 | 12s | 147 | 15.800 | 215.956 | 6.801 | 4.085 |
| blank_s29 | 500 | 4s | 171 | 5.869 | 53.436 | 6.985 | 6.318 |
| blank_s29 | 500 | 8s | 159 | 10.070 | 117.002 | 8.582 | 4.993 |
| blank_s29 | 500 | 12s | 147 | 14.505 | 195.990 | 9.346 | 4.327 |

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
