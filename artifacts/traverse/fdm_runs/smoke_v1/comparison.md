# First finite-horizon PID predictor results

All training ran on AMD. These are offline validation results on one terrain; no new closed-loop MPPI outcome was measured.

Nominal reference-following prediction: **1.305 m** mean endpoint error at four seconds.

| Arm | Seed | Updates | 4 s endpoint error, m | Contact AUC at 4 s | Low-progress AUC at 4 s | Work MAE at 4 s, kJ |
|---|---:|---:|---:|---:|---:|---:|
| history | 11 | 100 | 1.282 | 0.916 | 0.993 | 26.553 |
| history | 29 | 100 | 1.259 | 0.932 | 0.995 | 25.755 |
| no_history | 11 | 100 | 1.271 | 0.907 | 0.988 | 25.806 |
| no_history | 29 | 100 | 1.248 | 0.917 | 0.989 | 26.115 |
| no_terrain | 11 | 100 | 1.278 | 0.917 | 0.994 | 26.016 |
| no_terrain | 29 | 100 | 1.257 | 0.937 | 0.997 | 25.371 |
| profile | 11 | 100 | 1.156 | 0.849 | 0.994 | 23.944 |
| profile | 29 | 100 | 1.159 | 0.819 | 0.990 | 22.817 |

Rows above use the fixed-budget final checkpoints. The companion JSON also contains validation-selected checkpoints and calibration/support counts.

All arms with the same seed used identical sampled windows. The profile baseline has 35,680 parameters; the three other arms have 211,680 each.

Validation contains 250 episodes, including only 11 contact-positive and 10 low-progress-positive episodes. Its 5,000 overlapping windows are not independent trials. Rollover prediction is unsupported and disabled.

The geometry input is privileged BMP terrain plus authored asset footprints. The no_terrain arm removes only explicit terrain columns; speed references and state history can retain terrain information.

The next decision gate is matched short-horizon failure judgment and physical execution under the same driver, followed by CEM versus MPPI using the same scorer.
