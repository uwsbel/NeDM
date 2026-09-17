# Focused RGB-D experiment

Offline frame-0 selection among six recorded sibling routes per held-out development scene. No MPPI refinement or new execution occurs in this report.

Progress target: `sustained_stall`, semantic version 1. Target-safe goal requires measured goal completion without contact or the named progress event; the original recorded safe-goal field is retained separately in JSON.

| Run | Checkpoint | Step | Target-safe goal | Contact | Strict stall | Bounded motion | Abstain | Safe time (s) | Safe regret (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global_rgbd_s11 | last | 5000 | 3/4 | 1 | 1 | — | 0 | 7.400 | 0.250 |
| global_rgbd_s11 | best | 500 | 2/4 | 1 | 1 | — | 1 | 7.325 | 0.375 |
| global_blank_s11 | last | 5000 | 3/4 | 1 | 1 | — | 0 | 7.400 | 0.250 |
| global_blank_s11 | best | 500 | 2/4 | 1 | 1 | — | 1 | 7.325 | 0.375 |
| patch_rgbd_s11 | last | 5000 | 4/4 | 0 | 0 | — | 0 | 7.000 | 0.000 |
| patch_rgbd_s11 | best | 1000 | 4/4 | 0 | 0 | — | 0 | 7.025 | 0.025 |
| patch_blank_s11 | last | 5000 | 4/4 | 0 | 0 | — | 0 | 7.350 | 0.350 |
| patch_blank_s11 | best | 500 | 3/4 | 1 | 1 | — | 0 | 7.300 | 0.300 |
| global_rgbd_s29 | last | 5000 | 3/4 | 1 | 1 | — | 0 | 7.400 | 0.250 |
| global_rgbd_s29 | best | 500 | 2/4 | 1 | 1 | — | 1 | 7.325 | 0.375 |
| global_blank_s29 | last | 5000 | 2/4 | 1 | 1 | — | 0 | 7.225 | 0.400 |
| global_blank_s29 | best | 500 | 2/4 | 1 | 1 | — | 1 | 7.325 | 0.375 |
| patch_rgbd_s29 | last | 5000 | 4/4 | 0 | 0 | — | 0 | 7.025 | 0.025 |
| patch_rgbd_s29 | best | 500 | 4/4 | 0 | 0 | — | 0 | 7.025 | 0.025 |
| patch_blank_s29 | last | 5000 | 3/4 | 1 | 1 | — | 0 | 7.400 | 0.250 |
| patch_blank_s29 | best | 500 | 2/4 | 1 | 1 | — | 1 | 7.325 | 0.375 |

Time regret is defined only for actually safe completed selections, relative to the fastest actually safe completed sibling in that scene. Failure and abstention receive no invented travel time.

Image interventions (same fixed-last checkpoint):

| Run | Control | Safe goal | Contact | Stall | Abstain | Choices changed | FDE change (m) |
|---|---|---:|---:|---:|---:|---:|---:|
| global_rgbd_s11 | normal | 3/4 | 1 | 1 | 0 | 0 | — |
| global_rgbd_s11 | shuffle | 3/4 | 1 | 1 | 0 | 0 | -0.000 |
| global_rgbd_s11 | blank | 3/4 | 1 | 1 | 0 | 0 | 0.022 |
| global_blank_s11 | normal | 3/4 | 1 | 1 | 0 | 0 | — |
| global_blank_s11 | shuffle | 3/4 | 1 | 1 | 0 | 0 | 0.000 |
| global_blank_s11 | blank | 3/4 | 1 | 1 | 0 | 0 | 0.000 |
| patch_rgbd_s11 | normal | 4/4 | 0 | 0 | 0 | 0 | — |
| patch_rgbd_s11 | shuffle | 4/4 | 0 | 0 | 0 | 2 | 5.232 |
| patch_rgbd_s11 | blank | 1/4 | 0 | 0 | 3 | 3 | 4.930 |
| patch_blank_s11 | normal | 4/4 | 0 | 0 | 0 | 0 | — |
| patch_blank_s11 | shuffle | 4/4 | 0 | 0 | 0 | 0 | 0.000 |
| patch_blank_s11 | blank | 4/4 | 0 | 0 | 0 | 0 | 0.000 |
| global_rgbd_s29 | normal | 3/4 | 1 | 1 | 0 | 0 | — |
| global_rgbd_s29 | shuffle | 3/4 | 1 | 1 | 0 | 0 | 0.000 |
| global_rgbd_s29 | blank | 3/4 | 1 | 1 | 0 | 0 | 0.002 |
| global_blank_s29 | normal | 2/4 | 1 | 1 | 0 | 0 | — |
| global_blank_s29 | shuffle | 2/4 | 1 | 1 | 0 | 0 | 0.000 |
| global_blank_s29 | blank | 2/4 | 1 | 1 | 0 | 0 | 0.000 |
| patch_rgbd_s29 | normal | 4/4 | 0 | 0 | 0 | 0 | — |
| patch_rgbd_s29 | shuffle | 4/4 | 0 | 0 | 0 | 1 | 5.249 |
| patch_rgbd_s29 | blank | 0/4 | 1 | 1 | 3 | 4 | 6.624 |
| patch_blank_s29 | normal | 3/4 | 1 | 1 | 0 | 0 | — |
| patch_blank_s29 | shuffle | 3/4 | 1 | 1 | 0 | 0 | 0.000 |
| patch_blank_s29 | blank | 3/4 | 1 | 1 | 0 | 0 | 0.000 |

Limits:

- Only 4 held-out development scenes; their overlapping windows and six siblings are not independent trials.
- Validation was used for best-checkpoint selection. Fixed-last is a separate predeclared update count; neither is an untouched test evaluation.
- Failure-focused validation prevalence is designed, so Brier/ECE and risk thresholds are not natural deployment calibration.
- Four-second target labels and measured full-route events have different time coverage; later failures may be outside the model horizon.
- This ranks six actually recorded reference families; it does not evaluate MPPI-refined unseen routes or receding-horizon replanning.
- Rollover has no positive training or validation support and is excluded from operational cost.
- Frame0 validation has zero sustained_stall-positive 4-second labels; event discrimination cannot be validated at those anchors.
