# Online scientific report and verified pilot artifacts

The new [`render_traverse_fdm_diverse_online_report.py`](../../../scripts/render_traverse_fdm_diverse_online_report.py) creates static scientific figures and timestamp-aware video from completed online outputs. It never opens an authored terrain, obstacle list or future sensor image. Its XY map bins the original RGB-D hit points for display; it leaves unobserved bins blank.

At the requested decision index, it reconstructs history using only the measured trajectory prefix. The exact `history_sha256`, recorded state/pose, observation SHA and source snapshot hashes must match the executed protocol. Optional selected-route forecast replay additionally requires the exact checkpoint SHA and step and runs on CPU without an optimizer. This is a conditional forecast for the selected reference; later online replans can change actual motion, so its difference from the displayed physical continuation is not a matched fixed-reference prediction error.

All geometric proposals are shown, including rejected proposals, with coincident speed variants counted rather than represented as separate spatial paths. Orange is the selected command/reference, dashed magenta is the finite FDM forecast, and blue is actual Chrono motion. Saved family scores are shown before refinement and are never reordered or filtered by physical outcomes. Telemetry separately shows time, progress, velocity, positive mechanical work, power, native tire slip and signed roll/pitch. Native slip is dimensionless and can become large near a zero-speed denominator; its full range is retained on a symlog axis. The separate slip-speed plot uses the recorded circumferential-minus-horizontal-hub speed in m/s and labels its slope approximation.

## Verified development smoke

AMD job **412081 remains FAILED**: its final stdout `json.dumps(summary)` could not serialize an ndarray route, after complete physical artifacts had been written. [Independent validation](../../../artifacts/traverse/fdm_diverse_v1_20260909/reports/online_smoke_v5_validation/validation.json) confirms all four runs have complete actual endpoints, solver-rate work/contact outcomes, matching observation/source hashes and verified history hashes at every decision. Headless and video 4 s runs have all eleven physical arrays and the entire trajectory file SHA exactly equal. This validates implementation artifacts; it does not turn the job into a successful job or establish traversal generalization.

Final reviewed figures:

- [20 s overview, time/risk mode](../../../artifacts/traverse/fdm_diverse_v1_20260909/reports/online_smoke_v5_visuals_final/time_risk_20s/overview.png)
- [Measured telemetry](../../../artifacts/traverse/fdm_diverse_v1_20260909/reports/online_smoke_v5_visuals_final/telemetry.png)
- [Native slip and separate physical slip-speed diagnostic](../../../artifacts/traverse/fdm_diverse_v1_20260909/reports/online_smoke_v5_visuals_final/slip_diagnostics.png)
- [Depth-derived visible elevation](../../../artifacts/traverse/fdm_diverse_v1_20260909/reports/online_smoke_v5_visuals_final/observed_elevation.png)
- [Saved candidate scores](../../../artifacts/traverse/fdm_diverse_v1_20260909/reports/online_smoke_v5_visuals_final/time_risk_20s/candidate_scores.png)
- [Actual 4 s Chrono video, energy/risk mode](../../../artifacts/traverse/fdm_diverse_v1_20260909/reports/online_smoke_v5_visuals_final/energy_risk_video_4s/actual_chrono.mp4)

The video uses all nine original physical PNG frames and their actual timestamps, including the terminal endpoint. Encoded timestamps agree within **3.1 × 10⁻¹³ s**; the container lasts **4.001 s** because its measured final still is held for 1 ms. No motion interpolation or predicted vehicle imagery is inserted. Source frame hashes and encoder commands are saved with the video.

The 20 s time/risk run made **91.15 m net goal progress**, but began rolling backward late in the run. Its native slip peak is **4187.746058660296 at 17.85 s**, with measured forward body velocity **−0.04490 m/s** at that peak. The maximum separately derived horizontal slip speed reaches **50.72 m/s**. The goal was not reached before the timeout. The unequal 20 s time/risk and 4 s energy/risk recordings are smoke diagnostics, not an energy-efficiency comparison.

## Reproduction

From the isolated worktree, use a new output directory:

```bash
/home/harry/miniconda3/envs/nedm/bin/python scripts/render_traverse_fdm_diverse_online_report.py \
  --run time_risk_20s=artifacts/traverse/fdm_diverse_v1_20260909/online_pilot_smoke_v5/rolling_longer \
  --run energy_risk_video_4s=artifacts/traverse/fdm_diverse_v1_20260909/online_pilot_smoke_v5/rolling_video \
  --observation artifacts/traverse/fdm_diverse_v1_20260909/pilot_w8/observations/diverse_v1_train_rolling_hills_00 \
  --checkpoint artifacts/traverse/fdm_diverse_v1_20260909/runs/pilot_learning_h60_v3/patch_rgbd_s11/best.pt \
  --code-root artifacts/traverse/fdm_diverse_v1_20260909/snapshots/online_v5 \
  --decision-index 0 --encode-video --out /tmp/fdm_online_report_new
```
