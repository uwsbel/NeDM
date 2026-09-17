# Large terrain cohort for RGB-D FDM / MPPI

The first enrichment cohort contains **36 independent 240 m × 240 m heightfields** and **540 geometric route commands**. It extends the existing 80 m traversal arenas by a factor of three in each horizontal dimension. Geometry is generated and checked; physical passability and prediction performance must be measured separately on AMD.

[Terrain family preview](../../../artifacts/traverse/fdm_diverse_v1_20260909/geometry/terrain_families_preview.png) · [Route height profiles](../../../artifacts/traverse/fdm_diverse_v1_20260909/geometry/route_height_profiles.png) · [Geometry audit](../../../artifacts/traverse/fdm_diverse_v1_20260909/geometry/geometry_audit.json)

## Scene design

| Family | Terrain variation | Main intended comparison |
|---|---|---|
| Rolling hills | Overlapping smooth hills, broad relief, one steeper smooth hill | Short climbing route versus longer low-relief route |
| Ridge passes | Three crosswise ridges with offset rounded passes | Traversing a pass versus climbing the ridge |
| Cross slopes | Elongated banked hills with varying orientation | Side-slope exposure, pitch, roll and work |
| Valley network | Curved valleys, banks and intervening hills | Descending and climbing versus staying on a gentle route |
| Rough mosaic | Two spatial roughness scales and smooth hills/basins | Wheel slip, suspension motion and energy at different speeds |
| Mixed obstacles | Smooth hills, basin, rocks and trees | Collision avoidance together with terrain risk |

Every scene has two **10 m radius flat pads**, each blending into the surrounding terrain over another **12 m** with a quintic smootherstep. The first pad is the launch; the opposite pad is the goal and can support future reverse-direction tasks. Obstacle placement excludes both pad neighborhoods. All 72 pad checks confirm exactly constant quantized height and zero queried slope within an 8 m radius. The positive-control path is also smooth through the launch exit.

Five geometric paths connect the pads, with lateral offsets **0, −22, +22, −44 and +44 m**. Each has **2, 4 and 6 m/s** commands with the existing terminal deceleration convention. Paths are **228.25–247.93 m** long. Their nominal constant-cruise durations span **38–124 s**; full physics collection has a **180 s cap** to retain delayed failures. Reaching the horizon is a timeout, not goal completion. The four-scene pilot may use an explicitly recorded 60 s override to measure throughput.

One randomly chosen outer path follows an authored gentle corridor; its relief varies between scenes. This is a **construction control, not a physical success label**. Rocks and trees are kept outside that corridor. The other four paths cross different hills, grades and obstacles. Commands are generated from start, goal, offset and speed alone and are never filtered using terrain or obstacle truth. All 540 pass kinematic and arena-bound checks with `arena_half_extent_m=120`.

The corridor avoids an entirely infeasible cohort and supplies matched slower/faster controls. It also constrains the first enrichment cohort: all routes still come from five related path shapes. Successful unseen-arena results would support terrain transfer within this reference distribution, not unrestricted navigation or arbitrary global-route topology.

## Split and protection

There are **24 training scenes** (four independent seeds per family), **six validation scenes** (one per family) and **six sealed test scenes**. Every sibling route and time window inherits the whole scene split. All 36 BMP hashes differ.

Three sealed scenes test new layouts of known families. The other three add a fixed composition of intersecting oblique ridges and a curved trough that is absent from training and validation. This is a composition holdout, not a wholly unseen terrain physics model. Test outcomes must remain unopened until model weights, normalization, risk thresholds and MPPI cost weights are frozen.

The preview shows one **training** scene per family. The pilot selects training rolling hills, training ridge passes, training rough mosaic and validation mixed obstacles. It contains no sealed test scene.

## Height and camera conventions

The terrain grid is **512²**, giving **0.46875 m per pixel**. Smooth feature widths start around 5 m, and coherent roughness correlation lengths are 0.9–2.8 m. The same 8-bit grayscale BMP drives both Chrono `RigidTerrain` and `TerrainMap`; no unquantized float surface is used as the physical truth.

World grid columns increase along +X and rows increase along +Y. Pixel centers are `−size/2 + (i+0.5) × size/pixels`. BMP rows are written in reverse order; the inherited `rot90=0, flipud=true` transform restores world order when `TerrainMap` loads them. Only the transform is inherited from `arena_v1`; its old calibration residual statistics are **not** reused as new accuracy measurements. Runtime calibration checks must report errors without rewriting these frozen assets.

Across the cohort, measured relief is **7.84–18.21 m**, and quantization steps are **3.14–7.20 cm**. Quantized peak grades span **44.9–69.3°**, while scene p99 grades span **29.0–47.8°**, measured over a 0.9375 m centered difference. Steep terrain and corridor side transitions are intentionally retained; there is no global slope cap or guaranteed universal stall threshold. Only 0.05–2.33% of cells exceed 40° in individual scenes. Report actual Chrono pose, wheel slip, work and signed progress before assigning a failure mechanism.

The original 100 m-high / 47° camera does not cover this cohort. Collection requires a separately declared, registered **global RGB-D** observation, currently planned at **400 m height, 47° FOV, 1024² raw pixels**, reduced to a 512² model input. Approximate coverage at zero elevation is 348 m. Depth must stay in floating-point metric units, and encoding uses a declared 40 m elevation scale; the old narrow uint16 depth window is inappropriate here.

This global observation is an aerial map-like sensor. Enlarging the terrain creates long traversals and multiple terrain decisions, but it does not establish onboard limited-visibility exploration: the global camera intentionally sees the full arena. Terrain and asset manifests, geometry controls and physical diagnostic outputs remain outside the learned scorer input.

## Files and collection interface

The generator is [traverse_fdm_rgbd_diverse_arenas.py](../../../scripts/traverse_fdm_rgbd_diverse_arenas.py). It refuses to overwrite an existing manifest or arena. The first cohort is already generated; do not rerun into these paths.

- Arena assets: `assets/traverse/arena_fdm_diverse_v1_<split>_<family>_<index>/`.
- Complete manifest: `artifacts/traverse/fdm_diverse_v1_20260909/cases/cases.json`.
- Split manifests: `train_manifest.json`, `val_manifest.json`, `test_manifest.json` in the same folder.
- Development pilot: `pilot_manifest.json` in the same folder.
- Standard case files: `<scene_id>.json`.
- Route files: `routes/<scene_id>/family_00.json` through `family_14.json`.

Each manifest record includes `scene_id`, `split`, `family`, `seed`, `evaluation_stratum`, `case`, `routes`, `case_sha256`, `route_sha256`, `arena_bmp_sha256` and `arena_meta_sha256`. Case and route paths are relative to the manifest directory; arena paths are relative to the worktree root. Existing runner fields (`layout`, `goal_xy`, `goal_radius_m`, `horizon_s`, `settle_reference`) are preserved. Added collection metadata distinguishes goal completion from timeout and requests retention of sustained-failure telemetry.

Geometry audit checks cover inverse orientation, exact reload of quantized heights, independent BMP identities, all flat pads, and all reference shapes and limits. These checks do not replace the AMD runtime height / camera calibration or actual route outcomes.
