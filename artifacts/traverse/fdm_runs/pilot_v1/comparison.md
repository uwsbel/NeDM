# First finite-horizon PID predictor results

All training ran on AMD. These are offline validation results on one terrain; no new closed-loop MPPI outcome was measured.

Nominal reference-following prediction: **1.305 m** mean endpoint error at four seconds.

| Arm | Seed | Updates | 4 s endpoint error, m | Contact AUC at 4 s | Low-progress AUC at 4 s | Work MAE at 4 s, kJ |
|---|---:|---:|---:|---:|---:|---:|
| history | 11 | 1000 | 0.919 | 0.982 | 1.000 | 16.549 |
| history | 29 | 1000 | 0.944 | 0.984 | 1.000 | 19.261 |
| no_history | 11 | 1000 | 0.952 | 0.987 | 1.000 | 19.194 |
| no_history | 29 | 1000 | 0.959 | 0.989 | 1.000 | 20.389 |
| no_terrain | 11 | 1000 | 0.933 | 0.981 | 1.000 | 18.016 |
| no_terrain | 29 | 1000 | 0.974 | 0.979 | 1.000 | 20.302 |
| profile | 11 | 1000 | 0.877 | 0.980 | 0.999 | 15.020 |
| profile | 29 | 1000 | 0.909 | 0.981 | 0.999 | 15.019 |

Rows above use the fixed-budget final checkpoints. The companion JSON also contains validation-selected checkpoints and calibration/support counts.

All arms with the same seed used identical sampled windows. The profile baseline has 35,680 parameters; the three other arms have 211,680 each.

Validation contains 250 episodes, including only 11 contact-positive and 10 low-progress-positive episodes. Its 5,000 overlapping windows are not independent trials. Rollover prediction is unsupported and disabled.

The geometry input is privileged BMP terrain plus authored asset footprints. The no_terrain arm removes only explicit terrain columns; speed references and state history can retain terrain information.

The next decision gate is matched short-horizon failure judgment and physical execution under the same driver, followed by CEM versus MPPI using the same scorer.
