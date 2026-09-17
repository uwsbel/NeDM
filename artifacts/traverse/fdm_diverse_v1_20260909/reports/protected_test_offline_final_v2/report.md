# Protected offline FDM evaluation

All 90 fixed references on six protected scenes were evaluated after the final freeze. Saved predictions were checksum-verified; all summaries were independently recomputed within recorded floating-point precision. Inference was not repeated.

| Model | Step | 12 s anchor-zero FDE (m) | 12 s work MAE (kJ) |
|---|---:|---:|---:|
| rgbd | 5000 | 5.069 | 65.727 |
| blank | 5000 | 7.118 | 112.684 |

The predeclared -44 m, 6 m/s reference completes safely on 3/6 scenes. Hindsight over all15 references finds safe controls on 6/6 scenes; this is feasibility, not a deployed policy.

Full event calibration, per-scene, causal-prefix and image controls are in report.json. Mechanical work is not fuel energy; overlapping windows are not independent samples. Offline forecasts do not establish receding-horizon MPPI completion.
