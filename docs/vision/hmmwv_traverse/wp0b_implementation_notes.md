# WP0b implementation notes — sensor smoke (alignment + depth: PASSED)

**Date:** 2026-09-01 · **Script:** `scripts/traverse_wp0b_sensor_smoke.py`
**Artifacts:** `artifacts/traverse/wp0b_sensor_smoke/` on newton (summary.json, alignment_overlay.png)

## Results (10 layouts, 256², settled t=0 frames)

| Check | Result | Bar |
|---|---|---|
| Alignment median (201 targets: marker/roof/canopy/rock) | **0.97 px** | ≤ 2 px |
| Alignment p95 | **3.06 px** (max 6.3) | ≤ 4 px |
| Per-class median | roof 0.17 · canopy 0.88 · rock 1.07 · marker 1.94 px | — |
| Depth→elevation, whole image | **6.3 mm** median | — |
| Depth→elevation, image edge (r > 0.45·W) | **16.9 mm** median / 24 mm p95 | checked explicitly (§3.3) |
| Depth convention | **ray** (planar loses: 3.8 m median) | frozen |
| Depth ray-correction scale | **1.200** (fitted vs calibrated heightmap) | manifest value |
| 20 Hz sim+render throughput | 1.30 frames/s ⇒ 0.065× realtime | input to §6.1 planning |

## Findings

1. **ChDepthCamera ray-scale bug (upstream-worthy):** this Chrono build's
   depth camera casts rays with tangents exactly **1.20×** wider than the
   constructor HFOV implies; the RGB `ChCameraSensor` honors the HFOV
   (validated by the 0.97 px alignment). Uncorrected, depth→elevation error
   was 1.5 m median / 3 m at edges with a radial signature. The correction is
   `CameraModel.depth_to_world(..., ray_scale)` in `nedm/traverse/camera.py`;
   the smoke script fits it against the calibrated heightmap (plan §3.3's
   "per-pixel ray-correction map", realized as one scalar).
2. **Alignment must be measured under near-zenith light:** with the 55°
   collection sun, color-centroids shift ~1 px toward the lit side (median
   inflated to ~3 px). The probe renders with `light_elevation_deg=80`
   (RenderSpec knob); collection keeps 55°.
3. **Vehicle marker materials matter:** the original orange marker with 0.4×
   emissive saturated to sand-white and was undetectable. Now blue
   (`VEHICLE_MARKER_RGB`), 0.1× emissive. Blob detection references are
   *rendered* colors (`DETECT_RGB` in the smoke script), not material diffuse.
4. **Open item — marker detection reliability:** 6/10 layouts detected the
   marker (it is only ~5×3 px at 256²). Fine for G0b (camera model is proven
   by 201 asset targets), but G1's vehicle-center probes will want either a
   larger marker footprint or detection tuned on collection-light frames.
5. **Throughput reality check:** physics dominates (headless gate runs
   ~15–25× slower than realtime; rendering adds little). At 0.065× realtime,
   a 30 s pilot episode costs ~8 min single-process; 200 episodes ≈ 26 h ⇒
   pilot collection must run multi-process on newton (measure GPU contention
   with ~4–8 render processes before committing).

## Still owed for G0b

- Analytic class-mask rasterizer + one-shot `ChSegmentationCamera` validation.
- Storage schema (episode-chunked compressed store), compression ratio and
  random-window loader throughput benchmarks, peak-disk measurement.
