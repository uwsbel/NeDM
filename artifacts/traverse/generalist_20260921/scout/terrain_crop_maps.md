# Scout: fixed depth map, arena assets, ego-aligned terrain crop (generalist plan A/B)

Read-only, 2026-09-21, worktree `/home/harry/NeDM-traverse_mppi`. Numbers marked *measured here* were computed in this
session from stored artifacts (numpy only; the BMP was parsed by hand because PIL is absent from `python3`).

## 1. World (x, y) -> pixel -> elevation, and its accuracy against Chrono

Three rasters of the same f104 terrain exist; they differ in frame and accuracy.

| Raster | File | Index convention | Accuracy vs Chrono `RigidTerrain.GetHeight` (4,096 audit points) |
|---|---|---|---|
| BMP heightmap (`TerrainMap`) | `assets/traverse/arena_f104_50h_v1/arena_000.bmp` + `arena_meta.json` (80 m, 512 px, 0.15625 m/px, heights -1.8..3.9 m, 8-bit step 0.02235 m, orientation `flipud: true`, calibrated rmse 0.0081 m) | `height_grid[iy, ix]`, row 0 = -y after `_apply_orientation` (`terrain.py:231-235,267,281-283`); bilinear on cell centres `fx = (x+40)/0.15625 - 0.5` (`terrain.py:286-301`) | rmse 0.0098 m; 0.0033 m if queried at `(x, y) * 511/512` (*measured here*, matches `sensor_v2/LOG.md:34-38`) |
| N2 static depth image `rgbd[3]` | `artifacts/traverse/crm_f104_v1/map_root/static_map_v1 -> ../maps/arena_f104_50h_v1/observation.npz` (4,512,512) | row 0 = +y; `row = 255.5 - y/mpp`, `col = 255.5 + x/mpp`, `mpp = 2*110*tan(23.5deg)/512 = 0.18683 m`; elevation = `rgbd[3] * 10 m`, invalid = -2 (`scripts/f104_n2_dataset.py:17-34`; encoding `src/nedm/traverse/fdm_diverse_data.py:20-44`) | rmse 0.050 m, MAE 0.026 m, p95 0.124 m, max 0.255 m (*measured here*); it is a flat-ground lookup, error grows as `r*z/H` |
| v2 metric grid `z` | `artifacts/traverse/crm_f104_v1/grids/arena_f104_50h_v1/grid.npz` (512x512, coverage 1.0, 2.8 px/cell, `grid.json`) | same index convention as `TerrainMap` (row 0 = -y, cell centres, `scripts/sensor_dataset_v2.py:43-50`); built by intrinsic back-projection `z = H - range/sec` (`scripts/sensor_map_v2.py:44-52`) | rmse 0.0050 m, p95 0.010 m, max 0.022 m (*measured here*); vs `TerrainMap` 0.0093 m, vs `TerrainMap` at 511/512 scale 0.0037 m |

The 511/512 relation: Chrono (rigid `RigidTerrain.cpp:313-314`, CRM `chrono_fsi/sph/ChFsiProblemSPH.cpp:810-811`) puts
the 512 samples on the patch edges (spacing 80/511) while `TerrainMap` and the v2 grid index cell centres (80/512), so
`x_terrainmap = (511/512) x_chrono`, at most 0.078 m at the arena edge (`sensor_v2/REPORT.md:23-25`). The v2 grid is a
measurement of Chrono's mesh and therefore already sits in Chrono's frame; only `TerrainMap` queries need the scale.
Negative control: y-mirroring the N2 lookup gives 1.18 m rmse (*measured here*), so the row sign above is verified.
The camera contract (nadir at (0,0,110), image up = +y, taps flip OptiX rows) is `src/nedm/traverse/camera.py:8-11`,
`scene.py:199-210,339-342`, `scripts/crm_capture_map_local.py:45-59`. `gen_planner.set_map` (`scripts/gen_planner.py:35-42`)
feeds the N2 sampler from `flipud(TerrainMap.height_grid)` at mpp 0.15625 and matches `TerrainMap` to 0.0 m
(*measured here*); the same trick with `flipud(grid z)` reproduces the v2 sampler to 5e-15 m (*measured here*).

## 2. Validity of the same map for CRM

Same BMP (sha `5d5bc683...` in both `arena_meta.json` and `maps/.../observation.json`), same height range, same
node-on-edge convention and y flip in `CRMTerrain.Construct` (`scripts/crm_collect.py:150-156`;
`ChFsiProblemSPH.cpp:881,939-951`); soil top = BMP height snapped to the 0.08 m particle lattice, 0.24 m deep.
**The task's "soil surface = BMP + 0.5 m" is wrong**: the +0.5 m in the `crm_collect.py:7-8` docstring is the
path-follower Bezier z (`scripts/traverse_fdm_rgbd_diverse_chrono.py:106`), never the terrain; the start drop is +0.75 m
(`crm_collect.py:177`). *Measured here* from `crm_extra.npz` on 40 goal-reached `collect_v1` runs: tyre bottom minus BMP
height at the first recorded frame (after the 0.8 s settle) median -0.027 m (p5 -0.18, p95 +0.13); over whole episodes
median +0.037 m, p5 -0.43, p95 +0.60 (chassis-xy vs wheel-xy on slopes plus sinkage). Max wheel sinkage below the BMP
(`outcome.json crm.max_wheel_sinkage_below_bmp_m`, `crm_collect.py:285-291,371-374`) median 0.31 m in a 404-outcome sample
(220 breakthroughs, hazard-enriched, not a rate). So deformation is real (up to the 0.24 m layer under the wheels) but
local to the vehicle's own track, and soil is fresh every process (`crm_collect.py:12`, `crm_f104_v1/REPORT.md:27`);
ruts never persist between episodes, and the persistent-rut multi-goal case was not done (`REPORT.md:151`). The static
map is thus a valid *undeformed prior* for both domains; sinkage is hidden state for the history encoder. It cannot be
refreshed live under CRM: OptiX does not render SPH from Python (`crm_f104_v1/scout/map_capture_corridor.md:437`).

## 3. Ego-aligned local elevation crop: spec and what exists

Existing numpy: `scripts/traverse_wp2_add_terrain.py:42-53` `terrain_patch(tmap, pose)` is exactly this crop at
K=8, +/-6 m, heights minus the map height at the vehicle centre, divided by 2 m, `du` forward / `dv` left; it is the
`terrain (N,T,64)` field of the NRD cache (`src/nedm/traverse/nrd_data.py:55`), sourced from `TerrainMap` (BMP), not depth.
Existing torch: `MapCropper.sample_points` (`src/nedm/traverse/map_crop.py:80-85`) + `_terrain_height`
(`map_crop.py:60-78`, `grid_sample` bilinear, border padding, `align_corners=False`, normalised
`gx = (x+40)/80*2-1`, which equals the `TerrainMap` cell-centre index to the bit) already sample any `(B,1,512,512)`
elevation grid whose rows increase with +y. Night-2's `TMap.sample` / `t_station_tensor` (`scripts/f104_n2_grad.py:63-99`)
is a differentiable *route-corridor* sampler in the N2 image convention (row 0 = +y, `align_corners=True`); it does not
produce an ego square, but its point sampler could be reused.

Proposed `ego_elevation_crop(elev, pose, k=16, half_m=4.0, mpp=0.15625, half_extent=40.0)`:
inputs `elev` = v2 `grid.npz['z']` (row 0 = -y) as `(1|B,1,512,512)`, `pose (B,3)` or `(B,L,3)` world x, y, yaw.
Offsets `du, dv = meshgrid(linspace(-half_m, half_m, k), indexing='ij')` (axis 0 forward, axis 1 left);
`px = x + du cos - dv sin`, `py = y + du sin + dv cos`; heights by bilinear cell-centre sampling (numpy:
`sensor_dataset_v2.sample`; torch: `_terrain_height`); centre `h0` = same sampler at (x, y); output
`(h - h0)` in metres (scale by 2 m as the existing patch does) plus `valid = (|px| < 40) & (|py| < 40)` since 8 m crops
within 4 m of the edge read border-clamped values. Reference to the map height at the centre rather than measured chassis
z keeps CRM sinkage out of the crop. In imagination, `pose` comes from `integrate_pose` (`src/nedm/traverse/nrd_model.py:36-42`)
exactly as the tracker env re-crops each step (`src/nedm/traverse/tracker_env.py:275-289,357`); with 0.5 m spacing the
0.156 m grid is not aliased. Recorded poses are `trajectory.npz['pose'] (T,3)` float64 at 50 ms (*inspected*).

## 4. What `MapCropper` assumes

It is not a raw crop. It projects ego points into the WP1 overhead image with `CameraModel()` defaults (256x256 px,
hfov 47 deg, camera 100 m; `map_crop.py:43-48`, `camera.py:28-32`), samples the frozen encoder's stage-2 feature map
(64 ch at 64x64, 1.36 m/cell; `map_crop.py:3-7`, tracker cache `(E,64,64,64)` at `tracker_env.py:171`), K=8 over +/-5 m,
then a learned 1x1 conv + linear to 256-D (`map_crop.py:56-57,103-107`). The heightmap buffer (`map_crop.py:49-51`) only
supplies z for the perspective scale. Its intrinsics do not match the f104 capture (1024 px, 110 m), so `forward` cannot
be pointed at the f104 image or feature map as-is, but the two primitives named in section 3 can be used directly on a
raw elevation grid, bypassing `proj`/`fc`.

## 5. The risk-model corridor and one shared geometry source

N2/CRM_N2 corridor `X (5,96,32)` = [elev_rel, grade, cross, speed, valid]: 96 stations `linspace` over the route length
(spacing L/95), 32 laterals over +/-6 m along the left normal (index 0 = right), `elev_rel` relative to station-0 centre,
gradients clipped +/-2, invalid filled with the reference (`f104_n2_dataset.py:12,45-60`); context is 17-D anchor state
+ goal offset (3) + start yaw + route length (`f104_n2_dataset.py:115-117`). CRM_N2 was trained by this script on
`map_root/static_map_v1` (`crm_f104_v1/REPORT.md:37`, `scripts/crm_train.py:5`), i.e. on the flat-ground lookup. The
12-channel v2 corridor (`sensor_dataset_v2.py:65-98`) and its footprint-masked / range-limited twins
(`scripts/vehicle_corridor.py:15-27,30+`, `scripts/nav_mask_datasets.py:19-45`) read the v2 grid. Batched torch twin of the
N2 corridor: `f104_n2_grad.py:80-99`, verified against numpy in `selfcheck` (`:436-450`); no torch twin of `tensor12`
exists (`nav_online.py` is numpy). Sharing one source: set `DS.G.update(rgbd=[0,0,0,flipud(z)/10], mpp=0.15625,
ctr=255.5, npx=512, elev_scale=10)` from `grid.npz` so `station_tensor`, `TMap`, and the ego crop all read the same
depth-derived, Chrono-frame grid (equivalence verified to 5e-15 m).

## Gaps / unknowns

- Whether the deployed N2/CRM_N2 checkpoints tolerate the corrected grid was only checked for `set_map` (0.035 m mean,
  logit corr 0.985, `gen_planner.py:9-11`); the capture-vs-grid difference is larger (rmse 0.051 m, max 0.28 m, *measured*).
- CRM effective contact surface vs the BMP was inferred from spindle heights at the chassis xy, not from particle
  positions; the +/-0.15 m first-frame spread includes slope geometry.
- Pilot CRM runs lack `command_reference.npz` / `anchor_state.npz` (*inspected*); `collect_v1` runs have them.
- Whether a 16x16/0.5 m crop is informative beyond the 8x8/1.7 m one is untested; the BMP's 0.156 m resolution supports it.

## What must change for the plan

- Read "preserve the local depth-derived terrain crop" (B.1) as "sample the static v2 grid at the recorded pose": no
  per-frame depth or crop exists in any arena recording; the NRD cache's terrain field is the BMP patch.
- Drop "soil surface = BMP + 0.5 m"; the surface is the BMP (lattice-snapped); +0.5 m is the follower path z.
- Build A's corridors and B's crops from `grid.npz` (not `rgbd`); if a frozen N2 checkpoint must be reused, keep it on
  its own flat lookup and treat the swap as a distribution shift.
- Query `TerrainMap` at `(x, y)*511/512` wherever it is still used as truth; the v2 grid needs no scale.
- A's "geometry context" is mostly anchor vehicle state (17-D); decide explicitly whether it stays in the deployable branch.
