# Fresh near-assets observation audit

All available checks passed: **True**. Three fresh observations from AMD job 412331 were independently downloaded; all 18 file hashes match the remote outputs.

| Scene | Speed m/s | Roll / pitch ° | XY settle drift m | Initial asset clearance m | Depth/BMP p95 m | Marker error px |
|---|---:|---:|---:|---:|---:|---:|
| mixed_obstacles_00 | 0.0186 | 0.002 / -0.570 | 0.0016 | 12.819 | 0.0435 | 1.970 |
| rough_mosaic_00 | 0.0247 | 1.656 / 0.407 | 0.0057 | 11.836 | 0.0896 | 2.042 |
| rolling_hills_00 | 0.0203 | -0.514 / -0.103 | 0.0123 | 11.594 | 0.0373 | 2.281 |

All four wheel normal forces are positive. State/pose/history match the native anchor artifact exactly, and history and model RGB-D encoding recompute exactly. Source case/map hashes match frozen scene design. All six common physics/input source files match both online_v9 and campaign_v2; all recorded sources match campaign_v2. The full runtime inventory is identical across all three observations.

Each frame is 1024² RGB-D and the model input is 4×512×512. All arena corners are in frame. Target rock roofs have 38, 22, 34 depth hits (mixed, rough, rolling). The source confirms the common 0.8 s settle; recorded task time zero follows that settle.

The observations support proceeding with the frozen prediction-before-physics comparison. No initial state/camera defect was detected; this is not a prediction of route safety.

Limits: depth residuals here compare against the bilinear BMP heightfield, not a fresh native Chrono terrain query. The sparse roof marker is a coarse RGB registration check. No settle time-series or initial chassis-contact force was stored, so the report does not assert perfect mechanical equilibrium or zero transient terrain contact. Initial asset separation is geometric and exceeds 11 m in all cases. No scene reselection, new physics or model inference was performed.

[Machine-readable audit](observation_audit.json) includes every check, exact state, provenance and download hashes.

## Route-validity erratum

The earlier **63/63 geometry-valid** count used the default curvature cap **0.125 m⁻¹**. The frozen online policy uses **0.025 m⁻¹**, retaining **9/21 per scene, 27/63 total**: offsets 0 and ±6 m at three speeds. Offsets ±12 and ±18 m are rejected by that unchanged policy. The predeclared scene specification remains unchanged. This correction does not affect the camera or settled-state checks above.
