# Scout: reference depth image -> world grid -> corridor tensor (f104, for the CRM port)

Read-only reconnaissance, 2026-09-16. Worktree `/home/harry/NeDM-traverse_mppi` (branch `traverse_mppi`). Nothing was
rendered, simulated or submitted; every number below marked "measured here" was computed in memory from existing
artifacts. The proposed local capture script in section 9 is UNTESTED (writing/running it was outside the read-only rule).

## 0. Bottom line

1. The model-input map for f104 already exists and is bit-reproducible: capture
   `artifacts/traverse/fdm_f104_50h_20260909/static_map_v1/` (== `sensor_v1/maps/arena_f104_50h_v1/`, byte-identical
   `observation.npz`, sha256 `faffac47...e200267`) -> world grid
   `artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/grids/arena_f104_50h_v1/{grid.npz,grid.json}`.
   Measured here: running today's `scripts/sensor_map_v2.grid_from_capture` on `static_map_v1` reproduces the stored
   `grid.npz` arrays exactly (`np.array_equal` True for z, range_m, sec, rgb, cover).
2. The arena BMP is unchanged for CRM, so the UNDEFORMED reference map is the same image. Reusing the existing grid is the
   zero-risk path for the frozen checkpoints. A fresh OptiX render on luffy is expected to match it to < 1e-4 m in depth
   (OptiX vs AMD lavapipe on the same scene: max 4.6e-5 m over 721k terrain pixels, identical valid mask,
   `render_latency_v1/optix_local/compare_optix_vs_amd_1024.json`).
3. All existing captures were rendered on AMD with Chrono Vulkan-RT on lavapipe (CPU). No OptiX map capture exists yet.
   The two capture scripts cannot run locally as written (section 8); a ~60-line local twin is given in section 9.
4. The reference image is rendered from a `veh.RigidTerrain` mesh of the BMP. That stays valid under CRM as a
   render-only proxy (separate ChSystem, no SPH). The OptiX camera does NOT see SPH particles from Python (section 10).

## 1. File map

| Role | Path | Key lines |
|---|---|---|
| f104-only capture (original, job 412395) | `scripts/capture_traverse_f104_map.py` | L14 SLURM assert, L26-27 arena gate, L30-37 RigidTerrain, L39-42 height audit, L43-45 camera dict, L46-53 lights/cameras/render, L54 encode, L56-58 saves, L59-61 corner check, L62-69 json |
| any-arena capture (sensor_v1, job 420753) | `scripts/sensor_capture_map.py` | L16 SLURM assert, L28-31 `gen_arenas.json` gate + heights from meta, rest identical |
| its array job | `scripts/sensor_capture.sbatch` | L9-12 env, L15 arena list, L18-19 command |
| depth -> metric world grid | `scripts/sensor_map_v2.py` | L26-28 constants, L31-69 `grid_from_arrays`, L72-77 `grid_from_capture`, L80-92 `save`, L95-105 CLI |
| 12-channel corridor sampler + dataset builder | `scripts/sensor_dataset_v2.py` | L21-22 constants, L27-40 `init_grid`/`set_grid`, L43-50 `sample`, L53-62 `corridor_points`, L65-98 `tensor12`, L114-154 `labels_only`, L169-196 CLI |
| vehicle-present frame capture | `scripts/vehicle_capture.py` | L14-17 CAMERA, L44-47 RenderSpec+build_scene, L56-62 settle, L74-77 render+save |
| footprint exclusion + masked corridor | `scripts/vehicle_corridor.py` | L15-16 footprint, L19-27 `exclusion_mask`, L30-67 `tensor12_excluded` |
| live loop (render -> grid -> mask -> corridors -> ensemble) | `scripts/nav_online.py` | L46-49 CAMERA, L50 MARGIN_M=1.5, L295-312 `Navigator.observe`, L363-374 corridors, L422-474 `corridors12_batch` |
| runner that renders in-sim | `scripts/nav_runner.py` | L125-133 RenderSpec (`with_rgb` off for depth-only ckpts), L191-204 `render_all`, L477 backend label |
| local launcher (sys.path trick) | `scripts/nav_local.py` | whole file (19 lines) |
| planner hooks | `scripts/gen_planner.py` | L35-42 `set_map` (authored heightmap), L251-256 `set_sensor_map` (v1 capture), L306-318 `set_grid_map`/`corridors12` (v2 grid), L321-323 `GridRiskModel` |
| N2 (night2_v1) corridor + map loader | `scripts/f104_n2_dataset.py` | L17-24 `init_map`, L27-34 `sample_map`, L45-61 `station_tensor` |
| v1 10-channel sampler (superseded) | `scripts/sensor_dataset.py` | L27-36 `init_map`, L60- `tensor10` |
| camera taps, camera pose, scene | `src/nedm/traverse/scene.py` | L199-210 taps (row flip), L217-227 `RenderSpec`, L339-342 `overhead_camera_pose`, L374-396 RigidTerrain patch, L418-452 sensor manager + cameras |
| pinhole model | `src/nedm/traverse/camera.py` | L27-50 |
| 4x512x512 encoding | `src/nedm/traverse/fdm_diverse_data.py` | L20-44 `encode_global_rgbd` |
| privileged heightmap | `src/nedm/traverse/terrain.py` | L231-235 `_apply_orientation`, L260-301 `TerrainMap` (L286-289 cell-centre convention) |
| render timing harness (ran on OptiX locally) | `scripts/render_latency_bench.py` | whole file |

## 2. Camera and scene contract (identical in all five places that define it)

Defined in `capture_traverse_f104_map.py` L43-45, `sensor_capture_map.py` L47-49, `vehicle_capture.py` L14-17,
`nav_online.py` L46-49, `render_latency_bench.py` L67:

| Parameter | Value |
|---|---|
| image | 1024 x 1024 |
| camera position | (0, 0, 110.0 m) world, nadir, fixed to the terrain body |
| hfov | `math.radians(47.)` = 0.8203047484373349 rad (square image, so vfov equal) |
| focal length | f = (W/2)/tan(hfov/2) = 1177.5193841849634 px; cx = cy = 511.5 |
| max depth | 180.0 m; a ray that misses returns exactly 180.0 on both backends (OptiX `optix/shaders/miss.cu` L94) |
| depth meaning | Euclidean range along the pixel ray, metres, float32 (`depth_measurement='Euclidean_ray_range_m'`, `depth_ray_scale=1.0`) |
| ground footprint at z=0 | 2*110*tan(23.5 deg) = 95.66 m -> 0.0934 m/px; the 80 m arena fills 69.9-70.0% of pixels (rows 82-942, cols 80-943) |
| ranges seen on f104 | 107.21 m (hilltop near nadir) to 124.14 m (corner) |
| lights | ambient (0.35, 0.35, 0.38); one directional ChColor(1, .95, .85), elevation 45 deg, azimuth 120 deg. `RenderSpec` default elevation is 55: captures and nav pass 45 explicitly |
| background | solid `SKY_RGB = (0.53, 0.71, 0.92)` |
| terrain colour | `patch.SetTexture(grass_texture.jpg, 40, 40)` if `<source-root>/chrono/data/sensor/textures/grass_texture.jpg` exists, else `ChColor(.42, .5, .32)`. Every AMD capture used the flat colour (the cluster source trees have no `chrono/`); the local worktree has the texture (`chrono -> /home/harry/NeDM/chrono`) |
| sensor trigger | update rate 500 Hz in the capture scripts, `1/step_size` = 500 Hz in `build_scene`; `SetLag(0)`, `SetCollectionWindow(0)`; one `manager.Update()` = one render, consumed through `_Tap.take()` (LaunchedCount must advance by exactly 1) |
| depth camera | `sens.ChDepthCamera(body, rate, pose, W, H, hfov, max_depth)`; do NOT push a `ChFilterDepthAccess` (scene.py L449-450) |
| RGB camera | `sens.ChCameraSensor(...)` + `ChFilterRGBA8Access()`; alpha dropped by the tap |

Pose (`scene.py` L339-342): `rot = QuatFromAngleZ(pi/2) * QuatFromAngleY(pi/2)`, position (0, 0, H). Chrono cameras look
along their +x; this gives x_cam = -Z (down) and image up = +Y (north-up map).

### Orientation conventions (three different ones are in play)
1. Image arrays after the taps (`scene.py` L201, L208 do `[::-1]` because both OptiX and Vulkan buffers are bottom-up):
   row 0 = +y (north), col 0 = -x (west). `u = cx + f*x/(H-z)`, `v = cy - f*y/(H-z)` (`camera.py` L8-11, L46-50).
2. World grid from `sensor_map_v2` and `TerrainMap.height_grid`: `grid[row, col]`, row = y bin increasing toward +y
   (row 0 = -y), col = x bin increasing toward +x; cell centre = -40 + (i + 0.5) * 0.15625.
3. Raw BMP: image row 0 = +y in Chrono. `arena_meta.json` orientation is `{rot90: 0, flipud: true}`, i.e.
   `TerrainMap.from_dir` flips the raw array so its row 0 = -y (`terrain.py` L231-235, L281-283). `meta['features']` are in
   the un-flipped generation frame: read them through `TerrainMap.features` only.
   `gen_planner.set_map` L38 does `np.flipud(tm.height_grid)` to get back to convention 1 for the N2 sampler.

Measured here, negative controls on the stored f104 grid vs `TerrainMap`: y-mirrored 1.12 m rmse, transposed 1.08 m rmse,
correct 0.0099 m. `static_map_audit_v1/report.md`: mirrored image y gives 2.73 m p95 vs 0.0088 m.

### The 511/512 offset (`sensor_v2/LOG.md` L34-38, `REPORT.md` L23-25, `docs/progress.md` L460)
Chrono `RigidTerrain` puts the 512 BMP samples at the patch edges (spacing 80/511); `TerrainMap` puts them at cell
centres (spacing 80/512, `terrain.py` L286-289). Result: `x_terrainmap = (511/512) * x_chrono`, a pure radial scale, up
to 0.078 m at the arena edge. The sensor measures Chrono's frame. Measured here on the stored f104 grid (262,144 cells):

| reference | rmse | MAE | p95 | max | bias |
|---|---|---|---|---|---|
| `tm.height(X, Y)` | 0.0099 m | 0.0071 m | 0.0208 m | 0.056 m | +0.0018 m |
| `tm.height(X*511/512, Y*511/512)` (Chrono frame) | 0.0038 m | 0.0026 m | 0.0081 m | 0.029 m | 0.0000 m |

(8-bit quantisation step is 0.02235 m, uniform rms 0.0065 m; the grid averages ~2.8 pixels per cell.)
CRM NOTE: `ChFsiProblemCartesian::Construct(heightmap, ...)` uses the SAME conventions as RigidTerrain
(`/home/harry/chrono/src/chrono_fsi/sph/ChFsiProblemSPH.cpp` L810-811 `dx = length/(nx-1)`; L939-951 flips y so image
top-left = +y). So the BMP goes in un-flipped for CRM too (consistent with `flip_bmp: False` in
`artifacts/traverse/crm_f104_v1/smoke/*/smoke_report.json`), and the 511/512 relation to `TerrainMap` is unchanged.

## 3. Capture outputs (the map-capture file format)

Directory per arena, written by either capture script:

| File | Content |
|---|---|
| `observation.npz` (compressed) | `rgb` uint8 (1024,1024,3); `depth_m` float32 (1024,1024), metres of ray range, 180.0 = miss; `rgbd` float32 (4,512,512) |
| `observation.json` | `schema` (`fdm_f104_static_rgbd_v1` or `f104_family_static_rgbd_v1`), `camera` dict (section 2 plus `model_image_size: 512`, `elevation_scale_m: 10.0`), BMP/meta/observation sha256, `native_geometry` {length_m, width_m, height_min/max, sample_count 4096, p95_abs_error_m, rmse_m}, provenance (source_root, source_manifest sha, runtime fingerprint, slurm_job_id) |
| `native_height_audit.npz` | `xy` (4096,2) uniform in +-38 m (rng seed 104), `native_height_m` = `RigidTerrain.GetHeight(ChVector3d(x, y, 20.))`, `bmp_bilinear_height_m` = `TerrainMap.height`; audit only, never model input. Gate: p95 |err| < 0.05 m (f104: p95 0.02049, rmse 0.00982, which is the 511/512 offset plus mesh triangulation) |
| `rgb.png` | the RGB frame |

`rgbd` (`fdm_diverse_data.encode_global_rgbd` L20-44): per-pixel `z = H - depth/sec`; 2x BOX downsample to 512; channels
0-2 RGB/255, channel 3 = `clip(z/10, -1, 1)`, or -2 where any of the 4 source pixels is invalid. Image frame
(row 0 = +y), 95.66/512 = 0.18684 m/px, i.e. a FLAT-GROUND pixel->world mapping. This is what N2 consumes.

Existing captures (all AMD, Vulkan-RT on lavapipe via `nrd_use_lavapipe`, Chrono build `/work1/dannegrut/harry/nrd/chrono-build`):

| Capture | Job | Location |
|---|---|---|
| f104 original | 412395, source `pilot_source_v1` | local `artifacts/traverse/fdm_f104_50h_20260909/static_map_v1/`; cluster `$C/static_map_v1/` |
| f104 + g228/g203/g217/g216/g231 | array 420753 (f104 = 420754), partition mi2101x, 33 s each, source `gen_v1/source` | local `.../sensor_v1/maps/<arena>/`; cluster `$C/sensor_v1/maps/<arena>/` |

with `C=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909`. The f104 re-capture is byte-identical to
`static_map_v1` (verified here: rgb and depth arrays equal, same npz sha256).

As run on AMD (`scripts/sensor_capture.sbatch`; the array range is not in the file, 0-5 was used):
```
source /work1/dannegrut/harry/nrd/env.sh; nrd_pychrono; nrd_use_lavapipe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=16
"$NRD_PYTHON" -P -u "$S/sensor_capture_map.py" --source-root "$C/gen_v1/source" \
  --arena "$C/gen_v1/source/assets/traverse/$A" --out "$S/maps/$A" --chrono-data "$CHRONO_BUILD/data"
```

## 4. Back-projection to the world grid (`scripts/sensor_map_v2.py`)

Constants L26-28: `MPP = 80/512 = 0.15625`, `HALF = 40.0`, `N = 512` (the BMP's own resolution; hard-wired, L36-37 raises
for any other arena size).

`grid_from_arrays(depth, rgb, cam, arena_size_m=80)` L31-69:
- guards L34-35: requires `cam['depth_ray_scale'] == 1` and `cam['depth_measurement'] == 'Euclidean_ray_range_m'`
  (the `backend` string is not checked by any map consumer);
- `f = (w/2)/tan(hfov/2)`; `ray_x = (u - (w-1)/2)/f`; `ray_y = -(v - (h-1)/2)/f`; `sec = sqrt(1 + ray_x^2 + ray_y^2)`;
- valid pixel: finite, `> 0`, `< max_depth_m - 1e-6`;
- `axial = depth/sec`; `x = ray_x*axial`; `y = ray_y*axial`; `z = H - axial` (v1 used `ray*H`: up to ~1.5 m misplacement);
- bin `xi = int((x+40)/MPP)`, `yi = int((y+40)/MPP)`, only pixels with |x|,|y| < 40; per-cell MEANS via `np.bincount`;
- returns dict: `z` (512,512) f32 m; `range_m` (512,512) f32 m (mean measured ray range); `sec` (512,512) f32;
  `rgb` (3,512,512) f32 in [0,1] (divides by 255 only if max > 1.5); `cover` (512,512) f32 = pixel count;
  `camera`; `meta` {mpp, half_extent_m, n, camera_height_m, focal_px, ...}. Empty cell: z/range/sec NaN, cover 0, no fill.

`save()` L80-92 writes `<out>/grid.npz` (keys `z, range_m, sec, rgb, cover`) + `<out>/grid.json` (camera + meta +
coverage stats). `sensor_dataset_v2.init_grid` reads only `mpp, half_extent_m, n, camera_height_m` from the json, so the
older stored json (keys `row0`, `mean_pixels_per_cell`) and the current writer's json are both accepted.

CLI (`--maps` is the PARENT directory; every subdirectory holding an `observation.npz` is converted):
```
python3 scripts/sensor_map_v2.py --maps <dir of capture dirs> --out <grids dir>
# -> <grids dir>/<capture dir name>/grid.npz + grid.json
```
Stored f104 grid: coverage 1.0, 2.796 pixels/cell (1-4), z in [-1.7776, 3.8961] m, range 107.21-124.14 m,
sec 1.000001-1.12493. Cost: 0.16 s offline here; 54-56 ms quoted for the live path.

## 5. Corridor sampler (`scripts/sensor_dataset_v2.py`)

`N_STATION, N_LATERAL, HALF_WIDTH_M = 96, 32, 6.0` (L21). Output `X`: float32 `(12, 96, 32)` = (channel, station,
lateral), stored float16 in datasets (`X12` of shape (n_routes, 12, 96, 32)).

`corridor_points(wp, stations)` L53-62: stations re-derived from waypoint arc length if missing or non-monotone; 96
stations `linspace(s[0], s[-1], 96)` (spacing depends on route length); tangent from `np.gradient`; left normal
`(-ty, tx)`; 32 lateral offsets `linspace(-6, +6, 32)` (step 0.3871 m), index 0 = right of travel, 31 = left; centre
index 16 is at +0.19 m.

`sample(img, x, y)` L43-50: bilinear on cell centres, `fx = (x+40)/mpp - 0.5`, indices clipped to [0, n-2].

`tensor12(wp, sp, st)` L65-98, channel order = `CHANNELS` L22:

| # | Name | Definition |
|---|---|---|
| 0 | `z_rel` | `zf - z0`; `z0` = z at station 0, lateral 16 (fallbacks: nanmean of station 0, of everything, 0.0); invalid samples filled with `z0` |
| 1 | `grade` | `clip(np.gradient(zf, ds, axis=0), -2, 2)`, `ds = route_len/95` |
| 2 | `cross` | `clip(np.gradient(zf, dl, axis=1), -2, 2)`, `dl = 12/31` |
| 3 | `speed` | commanded speed (m/s) interpolated on arc length, broadcast across laterals |
| 4 | `valid` | `inside (|x|,|y| < 40 - 1e-6) & (bilinear cover > 0.999) & isfinite(z)` |
| 5 | `range_abs` | bilinear `range_m`; invalid -> 110.0 (camera height) |
| 6 | `range_rel` | `range_abs - r0`, `r0` from a VALID cell of station 0 (L84-91) |
| 7 | `sec1` | bilinear `sec - 1`; invalid -> 0 |
| 8-10 | `R,G,B` | bilinear colour in [0,1] |
| 11 | `cover1` | `clip(cover, 0, 8)/8` |

Returns `(X, route_len_m)`. Model variants (`scripts/sensor_train_v2.py` L18-22): `H` = [z_rel, grade, cross, speed,
valid]; `H0` = [z_rel, speed, valid]; `Dabs` = [range_abs, sec1, speed, valid]; `Drel` = [range_rel, sec1, speed, valid].
No deployed variant reads R, G, B or cover1. Checkpoints: `sensor_v2/matched/matched_{H,H0,Dabs,Drel}_s{0,1,2}.pt`; each
stores its `channels` and `norm` (mu/sd per continuous channel); `GridRiskModel.score` selects channels by name and
appends an all-ones channel (`gen_planner.py` L286-300, L321-323).

Masked variant (`vehicle_corridor.tensor12_excluded` L30-67, batch twin `nav_online.corridors12_batch` L422-474):
excluded cells get cover 0, so they behave exactly like unobserved cells; `z0`/`r0` = MEAN over the valid samples of the
first station with >= 4 valid samples (`MIN_VALID_PER_STATION`), not the centre sample. Footprint half-extents
2.6 x 1.3 m + margin (`MARGIN_M = 1.5` in nav; measured worst vehicle+shadow reach 0.50 m, `VEHICLE_PILOT.md`).
Finding: vehicle-present frame + mask == vehicle-free frame + mask (0.0000 m on every valid sample, same pick 20/20).

Dataset builder CLI, as run (`$C/sensor_v2/ds_v2.sbatch`; numpy only, no Chrono):
```
# f104 rows, labels reused from the N2 label file (order asserted equal to --ids)
"$NRD_PYTHON" -u sensor_dataset_v2.py --grid $C/sensor_v2/grids/arena_f104_50h_v1 \
   --ids $C/sensor_v1/ids_station_ds_all.json --run-dirs $C/sensor_v1/run_dirs_abs.txt \
   --labels $C/sensor_v1/labels_station_ds_all.npz --out ds_v2_f104_train.npz --workers 16
# any arena, labels recomputed by labels_only()
"$NRD_PYTHON" -u sensor_dataset_v2.py --grid $GRID --runs "$C/gen_v1/data/runs/${A}_*" "$C/gen_v1/test/runs/${A}_*" \
   "$C/sensor_v1/test/runs/${A}_*" "$C/sensor_v1/test2/runs/${A}_*" --out ds_v2_gen_$A.npz --workers 16
```
Per run dir it reads `command_reference.npz` (`reference_waypoints`, `reference_speeds`, `reference_stations`) and, in
`--runs` mode, `trajectory.npz`, `outcome.json`, `case.json`, `anchor_state.npz`. Output keys: `X12`, `channels`, `ctx`
(row[21] overwritten with route length), `id, group, split, source, profile, fail, unsafe, status, event_idx, route_len,
min_vx, back_s, arena`. Gotchas: `arena` = `basename(run_dir).split('_')[0]`; `source = 'designed' if '/data/' in path`.

Planner-side consumers: offline picks `gen_planner.set_grid_map(griddir)` -> `corridors12(cands)`
(`scripts/sensor_pools_v2.py --grid <griddir> ...`); online `Navigator.observe(rgb, depth, pose)` ->
`grid_from_arrays` -> `V2.set_grid` -> `corridors12_batch(cands, exclude)`.

## 6. The three model-input contracts (choose which one the CRM port preserves)

| Model | Map file consumed | Sampler | Geometry |
|---|---|---|---|
| N2 (night2_v1, `night2_v1/final/N2_s*.pt`) | `<root>/static_map_v1/observation.npz['rgbd']` + `observation.json['camera']` (`f104_n2_dataset.init_map` L17-24; the path component `static_map_v1` is hard-coded) | `station_tensor` L45-61 -> (5,96,32): elev_rel, grade, cross, speed, valid | flat-ground image lookup, `row = ctr - y/mpp`, `col = ctr + x/mpp`, mpp 0.18684 m; elevation = `rgbd[3]*10`. Misplaces points by up to ~1.4 m (r*z/H) and N2 was TRAINED on that, so keep it bug-compatible for N2 |
| N2 in gen_v1 missions | authored heightmap via `gen_planner.set_map(arena_dir)` L35-42 (no capture; mpp 0.15625) | same `station_tensor` | agrees with the capture path to 0.035 m mean, logit corr 0.985 (gen_planner docstring) |
| sensor_v1 D/E0/... | `maps/<arena>/observation.{npz,json}` via `sensor_dataset.init_map` / `gen_planner.set_sensor_map` | `tensor10` | same flat-ground lookup, plus raw depth at 1024 px |
| sensor_v2 / nav_v1 matched_H, matched_Dabs | `grids/<arena>/grid.{npz,json}` or a live `grid_from_arrays` dict | `tensor12` / masked variants | correct back-projection, metric grid |

One capture directory serves all of them (`rgbd` for N2/v1, `depth_m` + `rgb` for v2).

## 7. OptiX depth-camera FOV bug and the fix

- Bug (confirmed 2026-09-08, `docs/chrono_optix_depth_fov_bug.md`): OptiX `ChDepthCamera` used
  `f = (W/2)(pi/2)/hFOV` instead of `(W/2)/tan(hFOV/2)`. At 47 deg: 980.43 px instead of 1177.52 px (scale 0.8326, depth
  covers ~55 deg). OptiX RGB was correct; Vulkan depth was correct; identical only at 90 deg. `CameraModel.pixel_rays(ray_scale)`
  and the `OptiX_legacy` backend option in `scripts/traverse_fdm_rgbd_chrono.py` L484 are the old workaround
  (`depth_ray_scale` ~1.20). `sensor_map_v2` refuses any `depth_ray_scale != 1`.
- Fix: commit `9e1a0448b` "Sensor: fix the FOV of the OptiX depth, normal and segmentation cameras (#819)" in
  `/home/harry/chrono` (branch `project/nrd`, clean tree). Verified here: it is an ancestor of HEAD; pychrono `.so`
  files built 2026-09-16 14:51; build has OptiX 9.1 (`OptiX_INSTALL_DIR=/home/harry/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64`),
  `CMAKE_CUDA_ARCHITECTURES=120`, FSI-SPH ON (CUDA), SENSOR ON, VSG OFF, IRRLICHT ON.
- Verification of the fixed build (`render_latency_v1/README.md`): vs the AMD frame of the same scene, median 7.6e-6 m,
  max 4.6e-5 m, identical valid mask (70.02%), best-fit image scale 1.0000005; after back-projection the grids agree to
  a median of 5.7e-6 m. 1024^2 depth frame: 6.9 ms (AMD lavapipe 2.7 s).
- RGB-D collected with older builds (newton) is still misregistered.

## 8. Running locally on luffy (RTX 5090, OptiX)

Recipe (`scripts/nav_local.py`, `nav_v1/RUNNING.md` L62-70):
```
PYTHONPATH=/home/harry/chrono/build/bin /usr/bin/python3.12 <script> ... --chrono-data /home/harry/chrono/data
```
- The build is compiled against system Python 3.12 + numpy 1.26.4; conda's numpy 2 cannot import it. Verified here
  (import only): `numpy 1.26.4`, `PIL 10.2.0`, `pychrono` from `/home/harry/chrono/build/bin/pychrono/`, and
  `veh.CRMTerrain`, `pychrono.fsi`, `sens.ChOptixSensor`, `sens.ChDepthCamera` all present; `nedm.traverse.{scene,
  terrain,camera,fdm_diverse_data}` and `sensor_map_v2`, `sensor_dataset_v2` import cleanly under that interpreter.
- If torch is needed in the same process: `import numpy` FIRST (system 1.26), THEN
  `sys.path.append('/home/harry/miniconda3/envs/nedm/lib/python3.12/site-packages')` (`nav_local.py` L12-16). A capture
  needs no torch; `sensor_map_v2.py` / `sensor_dataset_v2.py` need only numpy and run under any Python.
- First OptiX use costs ~8.2 s of shader compile during scene/sensor setup (`Shader compile time: 8.19`), then first frame
  0.03-0.05 s. `_Tap.take()` default timeout is 10 s; pass a larger one for the first frame.
- A render at simulated t = 0 without any physics step does fire (`ChOptixEngine.cpp` L238:
  `ChTime > NumLaunches/UpdateRate - 1e-7`); the AMD capture scripts rely on the same behaviour under Vulkan.
- Batch env used locally (`nav_local_batch.py` L22-23): `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

Why `capture_traverse_f104_map.py` / `sensor_capture_map.py` do not run locally as-is:
1. `assert os.environ.get('SLURM_JOB_ID')` (L14 / L16) and `os.environ['SLURM_JOB_ID']` in the json;
2. `runtime_fingerprint()` raises unless `CHRONO_BUILD` is set, and labels the backend `Vulkan_RT_lavapipe`
   (`traverse_fdm_rgbd_diverse_batch.py` L62-91);
3. `sha(source/'source_manifest.json')`: that file exists only in the frozen cluster source trees, so the json step
   raises FileNotFoundError AFTER `observation.npz` was written, and the output dir then blocks a rerun
   (`if a.out.exists(): raise`);
4. `sensor_capture_map.py` additionally gates on `scripts/gen_arenas.json` (f104 is listed, sha `5d5bc683...04ee8ed`);
5. the camera dict hard-codes `'backend': 'Vulkan_RT_lavapipe'` (cosmetic; nothing reads it).

## 9. Exact procedure: ONE undeformed f104 reference image on luffy with OptiX -> map files -> validation

Step 0 (no rendering; already valid): the consumable files exist.
```
artifacts/traverse/fdm_f104_50h_20260909/static_map_v1/observation.{npz,json}            # N2 / v1 contract
artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/grids/arena_f104_50h_v1/grid.{npz,json} # v2 contract
```

Step 1: create the local twin (proposed path `scripts/crm_capture_map_local.py`; UNTESTED; every Chrono call is copied
from `sensor_capture_map.py` L34-65, only the cluster-provenance lines are replaced):
```python
#!/usr/bin/env python3
"""Local OptiX twin of scripts/sensor_capture_map.py: one static, vehicle-free overhead RGB-D capture of an arena.
Same camera, lights, rigid-mesh terrain and encoding; cluster-only provenance (SLURM id, source manifest, runtime
fingerprint) replaced by local provenance. The rigid mesh is a render-only proxy of the UNDEFORMED surface."""
import argparse, hashlib, json, math, platform, subprocess, sys
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path('/home/harry/NeDM-traverse_mppi'))
    ap.add_argument('--arena', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--grass-texture', action='store_true',
                    help='default is the flat colour (.42,.5,.32), which is what every AMD capture rendered')
    a = ap.parse_args()
    repo = a.repo.resolve(); sys.path.insert(0, str(repo / 'src')); sys.path.insert(0, str(repo / 'scripts'))
    import pychrono as chrono, pychrono.vehicle as veh, pychrono.sensor as sens
    from nedm.traverse.scene import _rgb_tap, _depth_tap, overhead_camera_pose, SKY_RGB
    from nedm.traverse.terrain import TerrainMap
    from nedm.traverse.camera import CameraModel
    from nedm.traverse.fdm_diverse_data import encode_global_rgbd
    from PIL import Image
    assert hasattr(sens, 'ChOptixSensor'), 'this pychrono is not an OptiX build'
    arena = a.arena.resolve(); meta = json.loads((arena / 'arena_meta.json').read_text())
    assert meta['size_m'] == 80.
    hmin, hmax = float(meta['height_min_m']), float(meta['height_max_m'])
    if a.out.exists(): raise ValueError('Use a new map output directory')
    a.out.mkdir(parents=True)
    system = chrono.ChSystemSMC(); system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    terrain = veh.RigidTerrain(system); mat = chrono.ChContactMaterialSMC()
    mat.SetFriction(.9); mat.SetRestitution(.01); mat.SetYoungModulus(2.e7)
    patch = terrain.AddPatch(mat, chrono.CSYSNORM, str(arena / meta['bmp']), 80., 80., hmin, hmax)
    texture = repo / 'chrono/data/sensor/textures/grass_texture.jpg'
    if a.grass_texture and texture.is_file(): patch.SetTexture(str(texture), 40., 40.)
    else: patch.SetColor(chrono.ChColor(.42, .5, .32))
    terrain.Initialize(); system.GetCollisionSystem().BindAll()
    tmap = TerrainMap.from_dir(arena)
    rng = np.random.default_rng(104); xy = rng.uniform(-38., 38., (4096, 2))
    native = np.asarray([terrain.GetHeight(chrono.ChVector3d(float(x), float(y), 20.)) for x, y in xy])
    interp = tmap.height(xy[:, 0], xy[:, 1]); err = native - interp
    p95 = float(np.quantile(np.abs(err), .95)); assert p95 < .05, 'BMP/world orientation or height range mismatch'
    camera = {'width': 1024, 'height': 1024, 'hfov_rad': math.radians(47.), 'cam_height_m': 110., 'depth_ray_scale': 1.,
              'max_depth_m': 180., 'backend': 'OptiX_luffy_RTX5090_fork_fovfix_9e1a0448b',
              'depth_measurement': 'Euclidean_ray_range_m', 'model_image_size': 512, 'elevation_scale_m': 10.,
              'observation_mode': 'One static vehicle-free global terrain RGB-D map of the UNDEFORMED surface (rigid-mesh proxy)'}
    manager = sens.ChSensorManager(system); manager.scene.SetAmbientLight(chrono.ChVector3f(.35, .35, .38))
    manager.scene.AddDirectionalLight(chrono.ChColor(1., .95, .85), math.radians(45.), math.radians(120.))
    bg = sens.Background(); bg.mode = sens.BackgroundMode_SOLID_COLOR
    bg.color_zenith = chrono.ChVector3f(*SKY_RGB); manager.scene.SetBackground(bg)
    pose = overhead_camera_pose(camera['cam_height_m']); body = patch.GetGroundBody()
    rgb_cam = sens.ChCameraSensor(body, 500., pose, 1024, 1024, camera['hfov_rad'])
    rgb_cam.SetLag(0.); rgb_cam.SetCollectionWindow(0.); rgb_cam.PushFilter(sens.ChFilterRGBA8Access()); manager.AddSensor(rgb_cam)
    dep_cam = sens.ChDepthCamera(body, 500., pose, 1024, 1024, camera['hfov_rad'], camera['max_depth_m'])
    dep_cam.SetLag(0.); dep_cam.SetCollectionWindow(0.); manager.AddSensor(dep_cam)
    rgb_tap, dep_tap = _rgb_tap(rgb_cam), _depth_tap(dep_cam)
    manager.Update(); rgb = rgb_tap.take(timeout_s=120.); depth = dep_tap.take(timeout_s=120.)
    rgbd = encode_global_rgbd(rgb, depth, camera); assert np.isfinite(rgbd).all()
    np.savez_compressed(a.out / 'observation.npz', rgb=rgb, depth_m=depth, rgbd=rgbd)
    Image.fromarray(rgb).save(a.out / 'rgb.png')
    np.savez_compressed(a.out / 'native_height_audit.npz', xy=xy, native_height_m=native, bmp_bilinear_height_m=interp)
    cm = CameraModel(width=1024, height=1024, hfov_rad=camera['hfov_rad'], cam_height_m=110.)
    corners = [cm.world_to_pixel(x, y, z) for x in (-40., 40.) for y in (-40., 40.) for z in (hmin, hmax)]
    assert all(0 <= u < 1024 and 0 <= v < 1024 for u, v in corners), 'Arena not fully visible'
    git = lambda d: subprocess.run(['git', '-C', str(d), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    (a.out / 'observation.json').write_text(json.dumps({
        'schema': 'f104_family_static_rgbd_v1', 'arena': arena.name, 'camera': camera,
        'arena_bmp_sha256': sha(arena / meta['bmp']), 'arena_meta_sha256': sha(arena / 'arena_meta.json'),
        'observation_sha256': sha(a.out / 'observation.npz'),
        'native_geometry': {'length_m': 80., 'width_m': 80., 'height_min_m': hmin, 'height_max_m': hmax,
                            'sample_count': 4096, 'p95_abs_error_m': p95, 'rmse_m': float(np.sqrt(np.mean(err ** 2)))},
        'vehicle_or_goal_markers_rendered': False, 'rgbd_input_contains_authored_height_samples': False,
        'all_corners_visible': True, 'native_height_samples_are_audit_only': True,
        'capture_script_sha256': sha(__file__), 'host': platform.node(), 'pychrono': chrono.__file__,
        'chrono_git': git('/home/harry/chrono'), 'repo_git': git(repo),
        'terrain_colour': 'grass_texture' if (a.grass_texture and texture.is_file()) else 'flat_0.42_0.5_0.32'},
        indent=2, allow_nan=False) + '\n')
    print(json.dumps({'map': str(a.out), 'native_height_p95_m': p95,
                      'valid_depth_fraction': float(np.mean(depth < camera['max_depth_m'] - 1e-6))}))

if __name__ == '__main__':
    main()
```

Step 2: render (from the worktree root; ~10 s, dominated by the OptiX shader compile):
```
cd /home/harry/NeDM-traverse_mppi
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/harry/chrono/build/bin /usr/bin/python3.12 -u scripts/crm_capture_map_local.py \
  --arena assets/traverse/arena_f104_50h_v1 \
  --out artifacts/traverse/crm_f104_v1/maps/arena_f104_50h_v1
```
(No `--chrono-data`: a terrain-only scene loads no Chrono data files. The output dir must not exist.)

Step 3: convert to the map the v2 dataset builder and planner consume (`--maps` = parent dir):
```
python3 scripts/sensor_map_v2.py --maps artifacts/traverse/crm_f104_v1/maps --out artifacts/traverse/crm_f104_v1/grids
# -> artifacts/traverse/crm_f104_v1/grids/arena_f104_50h_v1/{grid.npz,grid.json}
# expected print: "arena_f104_50h_v1: coverage 100.00%  mean pixels/cell 2.80  z range [-1.78, 3.90]"
```
Consumers then take `--grid artifacts/traverse/crm_f104_v1/grids/arena_f104_50h_v1` (`sensor_dataset_v2.py`,
`sensor_pools_v2.py`) or `gen_planner.set_grid_map(<that dir>)`. For the N2 contract, note `f104_n2_dataset.init_map(root)`
appends the literal `/static_map_v1/`, so the capture dir must be named (or symlinked as) `<root>/static_map_v1`.

Step 4: validation (prints only; no writes):
```
cd /home/harry/NeDM-traverse_mppi && python3 - <<'EOF'
import sys, numpy as np
sys.path.insert(0, 'src')
from pathlib import Path
from nedm.traverse.terrain import TerrainMap
NEW = 'artifacts/traverse/crm_f104_v1'; OLD = 'artifacts/traverse/fdm_f104_50h_20260909'
tm = TerrainMap.from_dir(Path('assets/traverse/arena_f104_50h_v1'))
g = np.load(f'{NEW}/grids/arena_f104_50h_v1/grid.npz'); z = g['z']
xs = -40 + (np.arange(512) + .5) * (80 / 512); X, Y = np.meshgrid(xs, xs)
def rep(ref, name):
    e = (z - ref)[np.isfinite(z)]
    print(f'{name}: rmse {np.sqrt((e**2).mean()):.4f} mae {np.abs(e).mean():.4f} p95 {np.quantile(np.abs(e), .95):.4f} max {np.abs(e).max():.4f}')
    return float(np.sqrt((e**2).mean()))
print('coverage', float((g['cover'] > 0).mean()), 'px/cell', float(g['cover'].mean()))
a = rep(tm.height(X, Y), 'vs BMP (TerrainMap, cell centres)       expect rmse 0.0099 mae 0.0071 p95 0.021')
b = rep(tm.height(X * 511 / 512, Y * 511 / 512), 'vs BMP in Chrono frame (511/512)      expect rmse 0.0038 mae 0.0026 p95 0.008')
c = rep(tm.height(X, -Y), 'NEGATIVE CONTROL y-mirrored           expect ~1.12')
assert (g['cover'] > 0).all() and a < 0.015 and b < 0.006 and c > 0.5
new = np.load(f'{NEW}/maps/arena_f104_50h_v1/observation.npz'); old = np.load(f'{OLD}/static_map_v1/observation.npz')
ok_n, ok_o = new['depth_m'] < 180 - 1e-6, old['depth_m'] < 180 - 1e-6
d = np.abs(new['depth_m'] - old['depth_m'])[ok_n & ok_o]
print('valid-mask disagreement px', int((ok_n != ok_o).sum()), '(expect 0-few)  valid frac', float(ok_n.mean()), '(expect 0.69905)')
print(f'OptiX - AMD depth: median {np.median(d):.2e}  max {d.max():.2e}   (expect ~8e-6 / < 1e-4 m)')
go = np.load(f'{OLD}/sensor_v2/grids/arena_f104_50h_v1/grid.npz')
print('grid z max |new - stored AMD grid|', float(np.nanmax(np.abs(z - go["z"]))), '(expect < 1e-4 m)')
print('rgbd elevation max |new - old| (m)', float(np.abs(new['rgbd'][3] - old['rgbd'][3]).max() * 10))
assert np.median(d) < 1e-4 and d.max() < 1e-2
EOF
```
The "expect" figures for the BMP comparison were measured here on the stored AMD grid; the OptiX-vs-AMD figures come from
the vehicle-scene comparison (`compare_optix_vs_amd_1024.json`). If the 1024^2 depth differs by ~17% in image scale
(grid z off by metres near the edges) the wrong Chrono build was imported. The capture's own gates: native-vs-BMP
p95 < 0.05 m (AMD: 0.0205, rmse 0.0098), all 8 arena corners inside the image, `rgbd` finite.
Colour is NOT expected to match pixel-for-pixel across renderers (irrelevant for H / H0 / Dabs / Drel / N2).

## 10. RIGID-terrain-specific places, and what CRM changes

| Where | What is rigid-specific | CRM consequence |
|---|---|---|
| `capture_traverse_f104_map.py` L30-37, `sensor_capture_map.py` L34-41 | builds `veh.RigidTerrain` + SMC material from the BMP in its own `ChSystemSMC` | Keep as-is: it is a render-only proxy of the undeformed surface and never touches the CRM simulation. Material values are irrelevant to the image |
| same, L39-42 / L43-46 | audit uses `RigidTerrain.GetHeight` (Bullet raycast; needs `BindAll()` and a probe start ABOVE the surface, z=20) | Still valid for the proxy. It says nothing about the CRM particle surface |
| `scene.build_scene` L374-396, L434, L443 | terrain is `RigidTerrain`; both cameras are attached to `patch.GetGroundBody()`; `TraverseScene.patch_body` | A CRM scene has no patch body. Cameras need some fixed body. `build_config()['terrain']['type'] = 'rigid_heightmap'` (L95-104) is also rigid-only |
| `vehicle_capture.py` L47-62, `nav_runner.py` L133, `render_latency_bench.py` L69 | live frames come from `build_scene` (rigid mesh + HMMWV meshes) and `terrain.Synchronize/Advance` of a RigidTerrain | In a CRM system the OptiX camera sees only bodies with visual shapes. SPH particles are invisible: `ChSensorManager.AttachFsiSphSystem` IS exposed in this pychrono, but `ChFsiSphRenderOptions` is NOT wrapped, and its defaults (`render_particle_spacing = 0`, empty `sprite_shapes`; `/home/harry/chrono/src/chrono_sensor/ChFsiSphRender.h` L41-45) render nothing. C++ reference: `src/demos/sensor/demo_SEN_CRM_Rendering.cpp`. So a live frame would show the vehicle over a void (all terrain pixels = 180.0 -> invalid -> every corridor sample invalid) unless a visual-only rigid proxy mesh of the BMP is added, in which case the frame shows the UNDEFORMED surface + vehicle (ruts never appear) |
| `nav_online.mask_leak` L394-419, `nav_runner` `--path-heights map`, follower z = `tmap.height + .5`, start z = `tmap.height + .75` | `TerrainMap` used as ground truth for the surface | CRM's surface is the BMP bilinear height snapped to the particle grid: `Iz = round(z/spacing)` (`ChFsiProblemSPH.cpp` L881), xy particle pitch `80/round(80/spacing)`; top-particle centres therefore sit within +-spacing/2 of the BMP (smoke: aabb z max 3.92 at 0.08 m vs 3.90), the effective contact surface is about half a spacing above them, and soil settles/compacts. The reference image carries none of this. A constant vertical offset cancels in `z_rel`; it does NOT cancel in `range_abs` (Dabs), but there the reference image is what the model was trained on, so keep the image unchanged |
| `sensor_map_v2.py` L26-28, L36-37 | grid fixed to 80 m / 0.15625 m | Fine for the full f104 arena. A cropped CRM domain (smoke `crop_0.08`) still uses the full-arena image: corridor points outside the soil patch would read valid terrain that does not exist in the simulation |
| Labels (`sensor_dataset_v2.labels_only` L114-154) | no terrain query; uses vx (state col 0), throttle (action col 1), pose, `outcome.json` status, thresholds tuned on rigid ground (`vx < -0.10 & thr > 0.3`, stall = `|vx| < 0.3 & thr > 0.3` for 20 frames = 1.0 s, S0 = 20 frames skipped) | Format unchanged; on soft soil wheel slip/sinkage will change the base rates of "stall" and "rollback" (other scout's subsystem, flagged here because this file holds the labeller) |

## 11. Gotchas

1. `sensor_map_v2.py --maps` takes the parent directory, not a capture directory; a capture dir passed directly yields
   no output and no error (the loop just finds no subdirectory with `observation.npz`).
2. Both capture scripts refuse an existing `--out`; a failed local run of the ORIGINAL scripts leaves a half-written dir.
3. `f104_n2_dataset.init_map(root)` hard-codes `root + '/static_map_v1/observation.*'`.
4. Two raster resolutions exist: N2/v1 sample a 0.18684 m image-frame raster; v2 samples the 0.15625 m world grid. Feeding
   N2 from `gen_planner.set_map` (heightmap, 0.15625 m, no flat-ground displacement) is a third variant that was
   accepted in gen_v1 (0.035 m mean difference, logit corr 0.985).
5. `grid_from_arrays` decides RGB scaling by `rgb.max() > 1.5`; the depth-only nav path passes an all-zero uint8 image,
   which is left as zeros (harmless; ~10 ms wasted).
6. `TerrainMap` heights are NOT Chrono's frame (511/512). Any "sensor vs heightmap" acceptance threshold should use the
   scaled query, or expect a 0.010 m rmse floor with up to 0.056 m outliers on steep cells near the edge.
7. `meta['features']` coordinates are y-mirrored relative to the simulated field; use `TerrainMap.features`.
8. Local vs AMD colour differs if the grass texture is used (local worktree has it, cluster trees do not). Only matters
   for models reading R, G, B (none deployed). `nav_runner` refuses `--rgb off` when a checkpoint lists colour channels.
9. `build_scene` comment L9-11: decorations must go on the EXISTING terrain body because a new body perturbs the rigid
   solver; irrelevant for a separate render-only system, relevant if a proxy body is added into the CRM system mid-study
   (add it from the first run so all data share it).
10. OptiX frame lag while driving: the vehicle silhouette sits ~0.16 m behind the reported chassis position (0.10 m parked;
    `render_latency_v1/README.md` open items). Covered by the 1.5 m mask margin; irrelevant for a vehicle-free capture.
11. Chrono Vulkan-RT path on AMD: 2.7 s/frame and, in the fork's current source, no Python depth camera in a Vulkan-only
    build (bindings exist only as uncommitted SWIG files in `/home1/harry/chrono` on the cluster). If CRM runs on the AMD
    cluster, do the reference render once on luffy (or reuse the stored grid) and ship `grid.npz`/`grid.json`.
12. AMD Chrono results are not reproducible across nodes; renders are (lavapipe copied to luffy gave bit-identical depth).
    The map capture is therefore safe to produce on either machine; physics arms are not interchangeable.
13. Stored `grid.json` files predate the current writer (keys `row0`, `mean_pixels_per_cell`); both schemas load.
14. `depth_m` background is exactly 180.0 and is excluded by `depth < max_depth - 1e-6`; lowering `max_depth_m` below
    ~125 m would silently invalidate arena corners (max range on f104 is 124.14 m).
