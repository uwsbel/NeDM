# Focused RGB-D experiment

Offline frame-0 selection among six recorded sibling routes per held-out development scene. No MPPI refinement or new execution occurs in this report.

Progress target: `bounded_motion`, semantic version 2. Target-safe goal requires measured goal completion without contact or the named progress event; the original recorded safe-goal field is retained separately in JSON.

| Run | Checkpoint | Step | Target-safe goal | Contact | Strict stall | Bounded motion | Abstain | Safe time (s) | Safe regret (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global_rgbd_s11 | last | 5000 | 5/8 | 2 | 3 | 3 | 0 | 7.200 | 0.360 |
| global_rgbd_s11 | best | 500 | 6/8 | 1 | 2 | 2 | 0 | 7.192 | 0.250 |
| global_blank_s11 | last | 5000 | 5/8 | 2 | 2 | 3 | 0 | 7.040 | 0.230 |
| global_blank_s11 | best | 500 | 6/8 | 1 | 1 | 2 | 0 | 7.133 | 0.233 |
| patch_rgbd_s11 | last | 5000 | 8/8 | 0 | 0 | 0 | 0 | 6.894 | 0.012 |
| patch_rgbd_s11 | best | 1000 | 8/8 | 0 | 0 | 0 | 0 | 6.894 | 0.012 |
| patch_blank_s11 | last | 5000 | 6/8 | 1 | 1 | 2 | 0 | 7.100 | 0.125 |
| patch_blank_s11 | best | 1000 | 6/8 | 1 | 1 | 2 | 0 | 7.133 | 0.233 |
| global_rgbd_s29 | last | 5000 | 6/8 | 2 | 2 | 2 | 0 | 7.042 | 0.125 |
| global_rgbd_s29 | best | 500 | 6/8 | 1 | 1 | 2 | 0 | 7.133 | 0.233 |
| global_blank_s29 | last | 5000 | 6/8 | 1 | 2 | 2 | 0 | 7.208 | 0.192 |
| global_blank_s29 | best | 500 | 6/8 | 1 | 1 | 2 | 0 | 7.133 | 0.233 |
| patch_rgbd_s29 | last | 5000 | 8/8 | 0 | 0 | 0 | 0 | 7.219 | 0.337 |
| patch_rgbd_s29 | best | 1000 | 8/8 | 0 | 0 | 0 | 0 | 6.956 | 0.075 |
| patch_blank_s29 | last | 5000 | 5/8 | 1 | 2 | 3 | 0 | 7.210 | 0.120 |
| patch_blank_s29 | best | 500 | 6/8 | 1 | 1 | 2 | 0 | 7.133 | 0.233 |

Time regret is defined only for actually safe completed selections, relative to the fastest actually safe completed sibling in that scene. Failure and abstention receive no invented travel time.

Image interventions (same fixed-last checkpoint):

| Run | Control | Target-safe goal | Contact | Strict stall | Bounded motion | Abstain | Choices changed | FDE change (m) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| global_rgbd_s11 | normal | 5/8 | 2 | 3 | 3 | 0 | 0 | — |
| global_rgbd_s11 | shuffle | 4/8 | 3 | 4 | 4 | 0 | 6 | -0.010 |
| global_rgbd_s11 | blank | 3/8 | 4 | 4 | 5 | 0 | 5 | -0.034 |
| global_blank_s11 | normal | 5/8 | 2 | 2 | 3 | 0 | 0 | — |
| global_blank_s11 | shuffle | 5/8 | 2 | 2 | 3 | 0 | 0 | 0.000 |
| global_blank_s11 | blank | 5/8 | 2 | 2 | 3 | 0 | 0 | 0.000 |
| patch_rgbd_s11 | normal | 8/8 | 0 | 0 | 0 | 0 | 0 | — |
| patch_rgbd_s11 | shuffle | 7/8 | 1 | 1 | 1 | 0 | 3 | 4.312 |
| patch_rgbd_s11 | blank | 7/8 | 0 | 0 | 1 | 0 | 5 | 6.463 |
| patch_blank_s11 | normal | 6/8 | 1 | 1 | 2 | 0 | 0 | — |
| patch_blank_s11 | shuffle | 6/8 | 1 | 1 | 2 | 0 | 0 | 0.000 |
| patch_blank_s11 | blank | 6/8 | 1 | 1 | 2 | 0 | 0 | 0.000 |
| global_rgbd_s29 | normal | 6/8 | 2 | 2 | 2 | 0 | 0 | — |
| global_rgbd_s29 | shuffle | 5/8 | 2 | 3 | 3 | 0 | 3 | 0.760 |
| global_rgbd_s29 | blank | 6/8 | 2 | 2 | 2 | 0 | 2 | 0.028 |
| global_blank_s29 | normal | 6/8 | 1 | 2 | 2 | 0 | 0 | — |
| global_blank_s29 | shuffle | 6/8 | 1 | 2 | 2 | 0 | 0 | 0.000 |
| global_blank_s29 | blank | 6/8 | 1 | 2 | 2 | 0 | 0 | 0.000 |
| patch_rgbd_s29 | normal | 8/8 | 0 | 0 | 0 | 0 | 0 | — |
| patch_rgbd_s29 | shuffle | 5/8 | 1 | 3 | 3 | 0 | 6 | 4.390 |
| patch_rgbd_s29 | blank | 0/8 | 1 | 1 | 1 | 7 | 8 | 5.336 |
| patch_blank_s29 | normal | 5/8 | 1 | 2 | 3 | 0 | 0 | — |
| patch_blank_s29 | shuffle | 5/8 | 1 | 2 | 3 | 0 | 0 | 0.000 |
| patch_blank_s29 | blank | 5/8 | 1 | 2 | 3 | 0 | 0 | 0.000 |

Terrain frame-0 forecasts, fixed-last normal images:

Measured validation outcomes: 8 eventual blocked routes and 16 safe completed routes. Mean four-second goal progress is 5.802 m versus 14.513 m. Bounded-event positives within four seconds: 0/24.

| Run | Predicted 4s progress blocked / safe (m) | Bounded probability blocked / safe | Choices changed without risk terms |
|---|---:|---:|---:|
| global_rgbd_s11 | 12.778 / 12.702 | 0.000191 / 4.67e-05 | 0/4 |
| global_blank_s11 | 11.467 / 12.847 | 0.0078 / 0.000359 | 0/4 |
| patch_rgbd_s11 | 5.753 / 14.498 | 6.89e-05 / 3.35e-06 | 0/4 |
| patch_blank_s11 | 10.925 / 11.872 | 0.0912 / 0.00637 | 0/4 |
| global_rgbd_s29 | 12.732 / 13.059 | 4.22e-06 / 1.93e-06 | 0/4 |
| global_blank_s29 | 10.293 / 12.256 | 0.126 / 0.00209 | 0/4 |
| patch_rgbd_s29 | 5.288 / 13.934 | 0.00012 / 1.21e-06 | 0/4 |
| patch_blank_s29 | 12.378 / 12.727 | 0.0314 / 0.0159 | 1/4 |

These groups use eventual measured outcomes. Terrain frame-0 four-second bounded-event labels and their positive count are reported separately; later blockage is outside that supervised horizon when no positive is yet observed.

Limits:

- Only 8 held-out development scenes; their overlapping windows and six siblings are not independent trials.
- Validation was used for best-checkpoint selection. Fixed-last is a separate predeclared update count; neither is an untouched test evaluation.
- Failure-focused validation prevalence is designed, so Brier/ECE and risk thresholds are not natural deployment calibration.
- Four-second target labels and measured full-route events have different time coverage; later failures may be outside the model horizon.
- This ranks six actually recorded reference families; it does not evaluate MPPI-refined unseen routes or receding-horizon replanning.
- Rollover has no positive training or validation support and is excluded from operational cost.
