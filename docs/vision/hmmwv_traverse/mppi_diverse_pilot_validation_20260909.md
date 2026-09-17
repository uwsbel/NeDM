# AMD pilot geometry and passive-collection validation

**The enlarged RGB-D camera, headless collection and corrected rich-telemetry observer passed their actual AMD checks.** The frozen `campaign_v2` collector can proceed to the diverse cohort. These are collection-integrity results, not evidence that the trained planner has generalized.

[Actual four-camera preview](../../../artifacts/traverse/fdm_diverse_v1_20260909/pilot_checks_v1/actual_camera_preview.png) · [Camera/physics audit summary](../../../artifacts/traverse/fdm_diverse_v1_20260909/pilot_checks_v1/validated_summary.json) · [Corrected-observer validation](../../../artifacts/traverse/fdm_diverse_v1_20260909/pilot_checks_v2/validation.json)

## Actual terrain and RGB-D geometry

AMD job **412058** ran four independent strict geometry audits against the completed pilot observations. Each audit built a terrain-only Chrono patch and compared 4,000 world height samples and 4,096 unobstructed observed depth rays. Assets and the vehicle were excluded by geometric footprint masks, not by measured error. Frozen BMPs and orientation metadata remained unchanged.

| Scene | Depth elevation versus Chrono, median / p95 | BMP versus Chrono height, p95 | RGB marker error |
|---|---:|---:|---:|
| Training rolling hills | 2.20 / 6.38 mm | 5.56 cm | 2.23 raw pixels |
| Training ridge passes | 2.32 / 7.01 mm | 7.15 cm | 2.24 raw pixels |
| Training rough mosaic | 2.02 / 7.68 mm | 9.28 cm | 2.27 raw pixels |
| Validation mixed obstacles | 2.24 / 6.71 mm | 4.28 cm | 2.27 raw pixels |

Both launch and arrival pads passed actual Chrono flatness checks in every scene. All arena corners fit inside the 1024² raw frame with at least **147 pixels** of margin. The saved 512² model encoding exactly matches recomputation from raw RGB and floating-point metric depth. Camera height is 400 m, FOV 47°, and the elevation scale is 40 m.

The millimeter-scale depth residual measures consistency with the actual Chrono surface at the measured ray hit. The larger BMP-query residual includes the existing `TerrainMap` sampling/interpolation convention relative to the simulator mesh. These are separate checks and should not be combined into a claim of millimeter-accurate terrain authoring. The RGB marker is one independent registration point; it does not by itself validate every object boundary.

## Renderer and observer parity

The rolling-hills `family_14` route was executed with the same frozen case, route, physics settings, runtime and controller. The requested cap was 60 s; all counterparts reached the goal after **41.35 s**, with **827 recorded intervals plus the terminal state**.

Two independent counterparts were compared to the original headless rich-telemetry pilot:

- A single passive RGB-D render enabled, with the original rich observer.
- Headless execution with the rich observer disabled.

All **11 numeric physical arrays** matched exactly, including state, pose, applied controls, engine-interface power, positive work per interval, contact and terminal values. Contact event lists and physical outcomes were also identical. The common outcome was a safe goal, **288.52 kJ** of positive engine-interface mechanical work and no asset contact. This work measurement is not fuel energy.

The rich observer's N+1 state rows and N interval rows passed alignment checks. Every interval had solver-step power integration and matching post-step risk sampling. Pose, controls, power and interval work agreed with the standard trajectory recording.

Job 412058's six physical/audit subprocesses all exited successfully. Its wrapper initially exited with an analysis-only `KeyError`: `route_sha256` was looked up in simulator provenance instead of the outcome file. The submitted wrapper and error are retained. Correcting that lookup and rerunning only the comparison produced the passing parity report; no physical result was regenerated or changed. Slurm reported **2 min 32 s** for the job; the longest subprocess was the renderer-enabled execution at **150.9 s**.

## Corrected suspension getters

The original frozen pilot observer called an opaque SWIG suspension-vector getter and emitted unowned-allocation warnings. The corrected observer uses typed HMMWV double-wishbone scalar getters. AMD job **412065** independently checked the new `campaign_v2` observer against the original physical trajectory.

The job completed in **1 min 2 s**, exit 0. Only `src/nedm/traverse/fdm_rich_telemetry.py` changed among the collection dependency hashes. All 11 numeric physical arrays remained exact, and the entire `trajectory.npz` file had the same SHA-256 in both runs:

`18072cf90901430cc7c3c76cddf8d0e3b18892cabbc760d091078e1cb244ee4f`

All **24 spring/shock fields** were finite across **828 samples**, and the collector emitted **zero SWIG memory-allocation warnings**. The signals are nontrivial: front-left spring force spans approximately 4.5–21.0 kN and shock force −16.5–4.7 kN on this route. The rich recording now contains **198 sample fields**. The only fields with missing values are terminal desired-speed/parking commands, where no subsequent commanded interval exists.

This supports staged collection: one verified global RGB-D observation per scene can be joined to independent headless route executions. The proof is bounded to this runtime and controlled parity route; the full cohort still needs per-run provenance, observation-anchor joins and telemetry validation.
