# Scout report: how ONE f104 episode is simulated (rigid), and what CRM must change

Scope: newest single-goal episode runner (gen_v1 / sensor_v2 drives). Read-only recon, 2026-09-16.
Repo: `/home/harry/NeDM-traverse_mppi` (branch `traverse_mppi`). Cluster campaign: `amd:/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909`.
All line numbers are for the local files; the hashes below prove which local files equal the frozen cluster source.

## 0. Call chain and file identity

```
sbatch gen_array.sbatch                      scripts/gen_array.sbatch (16 lines)
 -> gen_runner.py  (one SLURM array task = one shard, ThreadPool of subprocesses)   scripts/gen_runner.py (41 lines)
   -> gen_collect.py  (one process = one episode; audit + stop-policy wrapper)      scripts/gen_collect.py (326 lines)
     -> importlib-loads FROZEN  <source-root>/scripts/traverse_fdm_rgbd_diverse_chrono.py, text-patches run_chrono() with 3 hooks, exec()s it
       -> nedm.traverse.scene.build_config / build_scene        src/nedm/traverse/scene.py
         -> nedm.hmmwv_data.create_hmmwv / capture_row           src/nedm/hmmwv_data.py
       -> nedm.traverse.terrain.TerrainMap (BMP height oracle)   src/nedm/traverse/terrain.py
       -> nedm.traverse.fdm_rich_telemetry.RichTelemetry (frame_observer)
```

`gen_collect.py` is byte-for-byte `scripts/collect_traverse_f104.py` except the arena gate (diff = lines 246-249: f104-only sha/height-range check replaced by the `scripts/gen_arenas.json` allowlist `{arena_dir_name: bmp_sha256}`; f104 = `5d5bc683...04ee8ed`).

sha256, local vs cluster `gen_v1/source/` (frozen) — identical unless noted:

| file | sha256 (first 12) | local == frozen |
|---|---|---|
| scripts/traverse_fdm_rgbd_diverse_chrono.py | 2996c567f248 | yes |
| src/nedm/traverse/scene.py | frozen b4f4f6b8d037 = `git HEAD`; **working tree cb164313d2c5 (uncommitted `RenderSpec.with_rgb` edit)** | **NO (working tree)**; physics identical, but `gen_collect` will reject the working-tree file ("Frozen source file mismatch", gen_collect.py:242) |
| src/nedm/hmmwv_data.py | 3e6fd41c6f9c | yes |
| src/nedm/traverse/{terrain,layout,fdm_data,fdm_diverse_data,fdm_rich_telemetry}.py, src/nedm/training/constants.py | fc9869f5…, 014c717f…, a4263180…, 5c5cf1a0…, 86fa14ba…, 8760562e… | yes |
| gen_collect.py / gen_runner.py / gen_array.sbatch | b6ba06226… / 9f741f767… / a131720d7… | yes (cluster copies live in `gen_v1/`, not in `source/`) |

Frozen source root layout on cluster: `gen_v1/source/{scripts,src,assets/traverse/arena_*,source_manifest.json,source_manifest_parent.json}`. `source_manifest.json` = `{"schema":"f104_production_source_v1","files":{relpath: sha256,...}}`; `gen_collect` requires the 9 `SOURCE_FILES` (gen_collect.py:27-30) to match it (line 241-242) and re-checks after the run (line 301). **A CRM port needs a new frozen source root + manifest (add `src/nedm/hmmwv_crm.py` to `SOURCE_FILES`).**

## 1. Commands as actually used

Submit (from `sacct -o SubmitLine`; `GEN_TASKS`, `GEN_OUT` exported in the calling shell, `--export=ALL`):
```
cd $campaign/gen_v1
GEN_TASKS=$PWD/tasks_test.json GEN_OUT=$PWD/test  sbatch -p mi2101x -J gen_test   -c 16  --array=0-47 -o $PWD/test/arr_%A_%a.out --export=ALL gen_array.sbatch   # job 420022
GEN_TASKS=...tasks_data_mi3008x.json GEN_OUT=$PWD/data sbatch -p mi3008x -J gen_data_a -c 192 --array=0   -o $PWD/data/arrA_%A_%a.out --export=ALL gen_array.sbatch # 420037
... -p mi2508x -c 128 --array=0 (420038);  -p mi2101x -c 16 --array=0-29 (420039)
```
`gen_array.sbatch`: `#SBATCH -A dannegrut -N 1 -n 1 -t 03:00:00`; `source /work1/dannegrut/harry/nrd/env.sh; nrd_pychrono; nrd_use_lavapipe; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1; export FDM_RUNTIME_FINGERPRINT=$campaign/pilot_runtime_412394.json; export GEN_ROOT=$campaign/gen_v1; export PYTHONPATH="$GEN_ROOT/source/scripts:$GEN_ROOT/source/src:..."; export GEN_WORKERS=${GEN_WORKERS:-$((SLURM_CPUS_PER_TASK-2))}; "$NRD_PYTHON" -P -u "$GEN_ROOT/gen_runner.py"`.

Per-episode command built by gen_runner.py:25-27:
```
$NRD_PYTHON -P -u $GEN_ROOT/gen_collect.py --source-root $GEN_ROOT/source --case $GEN_ROOT/<t.case> --route $GEN_ROOT/<t.route> \
   --out $GEN_OUT/runs/<t.id> --chrono-data /work1/dannegrut/harry/nrd/chrono-build/data --horizon-s 120
```
stdout/stderr -> `$GEN_OUT/logs/<id>.log`; `subprocess.run(..., timeout=3600)` (gen_runner.py:29).

Task row (`tasks_*.json`, list of dicts): `{'id','group','arena','case': path rel. to GEN_ROOT,'route': rel.,'shard': int,'run': bool}`; rows with `shard == SLURM_ARRAY_TASK_ID and run` are executed (gen_runner.py:15). Shard = `md5(group) % n_shards` (gen_pools.py:80) so **all arms of one start/goal group share a node** (Chrono is deterministic per node only). `run=False` marks a duplicate route already run under another arm id.

`gen_collect.py` CLI (parser lines 212-228): `--source-root --case --route --out --chrono-data` (required), `--source-manifest-sha256 --case-sha256 --route-sha256` (optional pins), `--horizon-s 120` (must be a multiple of 0.05, <=120; line 235), `--minimum-elapsed-s 24 --confirm-s 2 --recovery-tail-s 8` (cannot be lowered; line 236), `--disable-early-stop`, `--check-only`. It then hard-sets (lines 281-284): `command="collect", backend="Vulkan_RT_lavapipe", depth_ray_scale=1, record_rgbd_stride=0, path_height_source="truth", render_parity=False, rich_telemetry=True, frame_observer=F104Telemetry`. So `render=None`: **no sensor manager, no cameras** in these episodes.

Requires env `FDM_RUNTIME_FINGERPRINT` -> JSON `{"runtime_sha256": {path: sha}}` containing a `_vehicle.so` key and a `/vehicle/hmmwv/` key (gen_collect.py:271-274). Current file `pilot_runtime_412394.json` (108 entries) fingerprints **`chrono-build` (no FSI)**; CRM needs a new fingerprint of `chrono-build-fsi`.

## 2. Inputs

**case.json** (e.g. `gen_v1/cases_test_f104/cases/f104_g1_test_group_0000.json`): `id`, `split` in {train,val,test} (required, line 244), `arena` = path relative to source root (`assets/traverse/arena_f104_50h_v1`), `layout` = `EpisodeLayout` json `{episode_id, seed, assets: [] (must be empty, line 250), house_xy, house_yaw, start_xy, start_yaw}`, `goal_xy`, `goal_radius_m` 2.5, `horizon_s` 120, `arena_half_extent_m` 40, `settle_reference` (a route; only used by `observe`), metadata.

**route.json** (`read_route`, chrono.py:79-95): `{"waypoints": [[x,y],...] (N>=3, float m, world), "speeds": [N] m/s in [0,10], "stations": [N] m (must equal cumulative chord length to 1e-4, strictly increasing), "headings": [N] rad (not used by the sim), "meta": {...}}`; may be wrapped as `{"route": {...}}`. Waypoint spacing ~0.5 m. Hard requirement (chrono.py:193): `|wp[0]-start_xy| <= 0.25 m` and `|wp[-1]-goal_xy| <= 0.25 m`.

**arena**: `arena_meta.json` (`size_m` 80, `pixels` 512, `height_min_m` -1.8, `height_max_m` 3.9, `bmp` `arena_000.bmp`, `orientation {rot90:0, flipud:true}`; 8-bit gray, quantization 0.02235 m; max slope 0.675 = 34 deg, p99 0.545).

## 3. Scene construction (RIGID-specific block)

`run_chrono` chrono.py:156-172:
- :161 `tmap = TerrainMap.from_dir(arena)` — numpy bilinear oracle of the BMP (pixel-CENTRE convention, res = 80/512).
- **:162 start pose height: `build_config(arena, (*start_xy, tmap.height(*start_xy) + 0.75), start_yaw)`** -> chassis init z = BMP height + 0.75 m, yaw only (no pitch/roll alignment to slope; vehicle is dropped and settles).
- :163-165 `--chrono-data` overrides `chrono_data_root` / `vehicle_data_root` (= `<data>/vehicle`).
- :171 `scene = build_scene(config, layout, tmap, arena, plan=None, render=None)`.

`build_config` scene.py:54-106: `step_size_s=0.002`, `tire_step_size_s=0.001`, `record_step_s=0.05`; vehicle `HMMWV_Full`, contact `SMC`, engine `SHAFTS`, transmission `AUTOMATIC_SHAFTS`, `AWD`, `PITMAN_ARM`, **tire `TMEASY`**, **`chassis_collision: "HULLS"`**; terrain `rigid_heightmap`, 80x80 m, heights [-1.8,3.9], friction 0.9, restitution 0.01, Young 2e7 Pa.

`create_hmmwv` hmmwv_data.py:290-336: `veh.HMMWV_Full()`, `SetContactMethod(SMC)`, `SetInitPosition(ChCoordsysd((x,y,z), QuatFromAngleZ(yaw)))`, engine/transmission/drive/steering/tire types, `SetTireStepSize`, `SetChassisCollisionType(HULLS)`, `Initialize()`, all vis NONE, `SetCollisionSystemType(BULLET)`. Own ChSystem (`hmmwv.GetSystem()`); default solver/timestepper (not overridden on the rigid path).

`build_scene` scene.py:345-464 — **RigidTerrain lines**:
- **:375 `terrain = veh.RigidTerrain(system)`**; :376-379 SMC patch material; **:380-388 `terrain.AddPatch(mat, CSYSNORM, bmp, 80, 80, -1.8, 3.9)`**; :389-394 texture/colour; **:395 `terrain.Initialize()`**; **:396 `patch_body = patch.GetGroundBody()`**.
- :402 `_add_assets` (no-op for f104, `assets == []`); :410-413 asset collision-family mask against `patch_body.GetCollisionModel().GetFamily()` (rigid-only concept).
- :415-452 sensors only if `render is not None`; cameras are **attached to `patch_body`** (:434, :443) — CRM has no patch; use `terrain.GetGroundBody()` or a fixed dummy body.
- Roof marker visual box on chassis (:363-372) — harmless.
- `calibrate_orientation` scene.py:113-172 uses `RigidTerrain.GetHeight` (:143) — not called per episode.

Other rigid-terrain queries in the episode path:
- **gen_collect.py:88 `scene.terrain.GetHeight(ChVector3d(x,y,20.))`** in `StopPolicy.validate_native_height` (50 points: 7x7 grid on [-36,36] + start; pass if p95 |err| <= 0.08 m and max <= 0.15 m vs `tmap.height`; hard `require`, writes `native_height_check.json`). Runs at frame 0 (after settle; before the first Advance, Bullet ray queries return 0 — comment lines 79-80).
- **gen_collect.py:202 `scene.terrain.GetNormal(vehicle.GetSpindlePos(axle, side))`** in `F104Telemetry._snapshot` (rich telemetry only).
- **hmmwv_data.py:513 `tire.ReportTireForce(terrain)`** inside `capture_row(..., include_tires=True)` (called at chrono.py:229 and :328) — this is the source of state columns 7-10 (`tire_*_force_wheel_fz_n`). Also :514-516 `GetLongitudinalSlip/GetSlipAngle/GetCamberAngle`, :542 `tire.GetDeflection()`.
- fdm_rich_telemetry.py:175 `tire.ReportTireForce(scene.terrain)`, :171-174 native tire getters (wrapped in `_read` with NaN fallback, so they degrade instead of crashing).
- chrono.py:106 path z = `tmap.height(x,y) + 0.5` (numpy oracle, not a Chrono call — survives CRM unchanged).

## 4. Driver (contract to preserve exactly)

`make_driver` chrono.py:98-112; constants `DRIVER` dict :35-39 (copied verbatim into `outcome.json["driver"]`):
- Path: route waypoints subsampled to **>= 2.0 m station spacing** (always keeps the last), each point `ChVector3d(x, y, tmap.height(x,y) + 0.5)`; `chrono.ChBezierCurve(points)` (open).
- `veh.ChPathFollowerDriver(vehicle, bezier, "route", float(route["speeds"][0]))`.
- Steering: `GetSteeringController().SetLookAheadDistance(5.0)`, `SetGains(0.8, 0.0, 0.0)`.
- Speed: `GetSpeedController().SetGains(0.6, 0.05, 0.0)`.
- `driver.Initialize()` is called **before** the 0.8 s settle.
- Speed profile feed (chrono.py:214-218), **once per 50 ms frame**: `pos` = chassis `GetFrameRefToAbs().GetPos()` xy; `wp = nearest_index(xy, pos, wp)` = monotone argmin over `xy[wp : wp+60]` (:115-117); `at_end = wp >= len(xy)-2 and |pos - xy[-1]| < 3.0`; `driver.SetDesiredSpeed(0 if frame<0 or at_end else speeds[wp])`.
- Per physics substep (:221-227, :287): `driver.Synchronize(t)`; `inputs = driver.GetInputs()`; **external steering rate limit 2.0 /s**: `m_steering = clip(m_steering, prev - 2*dt, prev + 2*dt)`, and **forced to 0 during settle** (`frame < 0`); `terrain.Synchronize(t)`; `hmmwv.Synchronize(t, inputs, terrain)`; ... `driver.Advance(dt)`. The limit is `2*dt` per substep, so it is step-size invariant; the PID integrators are advanced with `dt` too. Throttle/brake are the unmodified PID output (during settle target = 0 m/s).

## 5. Time stepping, settle, horizon

- `DT = 0.05` s record/control frame (chrono.py:28); physics `dt = config.simulation.step_size_s = 0.002` (:195); `substeps = round(DT/dt) = 25` (:196).
- **Settle = `SETTLE_S = 0.8` s** (:29), not 1 s: `frame` starts at `-16` (:197); frames < 0 run the same loop with desired speed 0, steering 0, nothing recorded, no termination checks, no observer calls.
- Horizon: `--horizon-s 120` -> `frames = 2400` (:200-201); `while frame < frames` (:213). Default status `"timeout"` (:208).
- **Rigid advance (lines to change for CRM): chrono.py:287 `driver.Advance(dt)`, :288 `terrain.Advance(dt)`, :289 `hmmwv.Advance(dt)`.** Also the terminal re-synchronize at :326-327 (`terrain.Synchronize`, `hmmwv.Synchronize`, no advance).

## 6. Termination (order = priority)

Evaluated once per 50 ms frame for `frame >= 0`, after the 25 substeps, on `terminal_pose` = chassis ref (x, y, yaw=`GetCardanAnglesZYX().z`) (:306-307):
1. **rollover** (:314): `|vehicle.GetRoll()| > 60 deg or |vehicle.GetPitch()| > 60 deg` -> break.
2. **goal_reached** (:308, :317): `|terminal_xy - goal_xy| <= goal_radius_m (2.5)`; `goal_time = (frame+1)*DT` -> break.
3. Injected hook (gen_collect.py:174-178, inserted before `        frame += 1\n` at 12-space indent, i.e. inside `if frame >= 0:`) -> `StopPolicy.check(frame, record_pose, record_action, record_parked, terminal_pose, wp, desired_speed)` (gen_collect.py:120-154):
   - **terrain_bounds_exit**: `max(|x|,|y|) > 40.0` (always active, even with `--disable-early-stop`).
   - **prolonged_blockage_terminated**: last 40 intervals (2 s) have no `parked`, all `action[:,1] (throttle) > 0.3`, and the max pairwise XY distance over the 40 starts + current endpoint `<= 0.25 m`; window only counts once `elapsed >= 24 s`; first qualifying window starts confirmation; any non-qualifying frame cancels it; stop when `elapsed >= first + 2 s (confirm) + 8 s (tail)` -> earliest possible stop 34 s.
4. **timeout** at 120 s.
Post-hoc outcome flags (no termination): `sustained_near_stop` = >= 2 s contiguous `|vx| < 0.3 and throttle > 0.3 and not parked` (:311-313); `bounded_blockage_v1` windows (:342-349); `low_net_progress_4s_windows` (:339-341); asset contact > 1 N (:293-297; assets empty here).

## 7. What is recorded

Row `i` is sampled at substep 0 of frame `i` **after Synchronize, before Advance** (:228-275): "state/action/pose row i precedes interval [i, i+1)". Rate 20 Hz. No extra terminal row; `terminal_state`/`terminal_pose` are measured after the last interval without advancing (:325-332).

`trajectory.npz` (chrono.py:350-354; verified on `gen_v1/data/runs/g203_data_group_0000__route_00`, N=351):
| key | shape dtype | content |
|---|---|---|
| `state` | (N,17) f32 | `STATE_FIELD_PRESETS["tire_normal_force_omega_pt"]` (constants.py:54): 0 `vel_body_x_mps`, 1 `vel_body_y_mps`, 2 `roll_rad`, 3 `pitch_rad`, 4 `roll_rate_radps`, 5 `ang_vel_body_y_radps`, 6 `yaw_rate_radps`, 7-10 `tire_{fl,fr,rl,rr}_force_wheel_fz_n` (world-Z component of `ReportTireForce`), 11-14 `tire_{fl,fr,rl,rr}_spindle_omega_radps` (spindle ang vel . spin axis, + = forward), 15 `engine_motor_speed_radps`, 16 `engine_motorshaft_torque_nm` |
| `action` | (N,3) f32 | **[steer, throttle, brake]** = `inputs.m_steering` (after rate limit), `m_throttle`, `m_braking` (:235) |
| `pose` | (N,3) f64 | `pos_x_m, pos_y_m, yaw_rad` of chassis REF frame (:234) |
| `terminal_pose` (3,) f64, `terminal_state` (17,) f32, `terminal_parked` () bool | | endpoint |
| `power_kw` (N,) f64 | | engine motorshaft torque x transmission motorshaft speed / 1000 at the sample |
| `positive_work_kj_per_interval` (N,) f64 | | substep-integrated `max(P,0)*dt` |
| `contact_n` (N,) f64 | | max asset contact force in the interval (0 here) |
| `parked` (N,) bool | | `at_end` flag per frame |
| `state_fields` (17,) <U27, `dt_s` () f32 = 0.05 | | |

`anchor_state.npz` (:241-242, frame 0): `state (17,) f32, pose (3,) f64, history (16,24) f32` (`build_history`: t=0 state repeated 16x, synthetic previous action `(0,0,1)`, ego xy, sin/cos yaw), `goal_xy (2,) f64, goal_radius_m f32`.
`command_reference.npz` (gen_collect.py:297-300): `interval_start_s (N,)`, `desired_speed_mps (N,)` (**read back from `rich_telemetry.npz["command_desired_speed_mps"][:N]` — so rich telemetry is NOT optional**), `reference_waypoints (M,2)`, `reference_stations`, `reference_speeds`, `reference_headings`.

Episode directory (cluster `gen_v1/test/runs/<id>/`): `anchor_state.npz, case.json (copy), reference.json (route copy), collection_request.json (contract + hashes + thread env), native_height_check.json, initial_state_validation.json, trajectory.npz, outcome.json, collection_meta.json, contact_events.json, simulation_provenance.json, command_reference.npz, f104_episode.json, episode_complete.json` (+ `collection_failure.json` on exception). `rich_telemetry.npz/.json` and `rich_intervals.npz` are written then **deleted by gen_runner.py:31-33** on rc==0. Labels need only `trajectory.npz, outcome.json, command_reference.npz, case.json, anchor_state.npz` (that subset is what was synced locally).

`outcome.json` keys (chrono.py:361-386): `status` in {goal_reached, rollover, terrain_bounds_exit, prolonged_blockage_terminated, timeout}, `goal_reached`, `safe_goal_reached`, `goal_time_s`, `elapsed_s`, `frames`, `final_goal_distance_m`, `goal_progress_m`, `net_displacement_m`, `path_length_m`, `longest_consecutive_effortful_near_zero_speed_s`, `sustained_near_stop`, `bounded_blockage_v1(_windows)`, `low_net_progress_4s_windows`, `positive_work_kj`, `wall_s`, `driver`, `camera`, hashes.

Launch gate at frame 0 (`on_anchor`, gen_collect.py:101-118, hard `require`): finite; body horizontal speed <= 1 m/s; |roll|,|pitch| <= 20 deg; yaw error <= 10 deg; start xy error <= 1 m. Example settled values: speed 0.107 m/s, xy error 0.06 m.

## 8. RNG / determinism / resumability

- **No RNG anywhere in the episode path** (`layout.seed` is unused with empty assets). Outcome is a deterministic function of (case, route, node); Chrono is reproducible per node only -> keep paired arms in one shard/job.
- Skip-if-done: gen_runner.py:23 returns `'cached'` iff `<out>/episode_complete.json` exists (written last, gen_collect.py:316, contains sha256 of every artifact).
- **Not idempotent for failures:** gen_collect.py:270 refuses to run if `outcome.json`, `collection_request.json` or `trajectory.npz` already exist ("Preserve existing output; use a new directory"). A crashed/killed episode leaves `collection_request.json` (+ `collection_failure.json`), so a plain resubmit returns rc!=0 for it until the directory is moved away by hand (gen_v1 did this: `missions_run/superseded_before_fix/`). No mid-episode checkpointing.
- gen_runner prints `FAIL <id> rc=..` per failure; a `subprocess.TimeoutExpired` (3600 s) is NOT caught -> it propagates out of `ex.map` and kills the shard.

## 9. Rigid throughput actually achieved (for scale)

- Job 420022 shard 0 (mi2101x, `-c 16` -> 14 workers): 134 episodes in 1102 s wall. Episode `f104_g1_test_group_0000__n2`: 9.55 s sim, `wall_s_including_finalization` 26.8 s. Data episode on the 192-core node: 17.55 s sim, `wall_s` 82 s. => per-process RTF ~0.2-0.36 with one process per core, 2 ms step, TMEASY, 522k-triangle Bullet heightmap. gen_v1 total: 6,639 + 9,000 episodes + 600 missions in < 1 h wall across 4 partitions.

## 10. Existing CRM code in this repo (soil/SPH parameters previously used)

Single source of truth: `src/nedm/hmmwv_crm.py` (`configure_crm_terrain`, :82-160), used by `scripts/collection/collect_hmmwv_crm_smoke.py`, `collect_hmmwv_crm_dataset.py`, `src/nedm/rl/hmmwv_chrono_crm_tracking_env.py`, `scripts/throughput/probe_sim_fps.py`. Config: `configs/hmmwv_crm_eval.json` (identical values in `collect_hmmwv_crm_smoke.build_collector_config` :125-214 and `prepare_hmmwv_crm100_generation.py` :60-116; production launcher `scripts/collection/run_hmmwv_crm2000_collection.sh`).

| parameter | value used (hmmwv_crm_2000 dataset, CRM eval) |
|---|---|
| step (MBD = CFD) | `step_size_s = tire_step_size_s = 5e-4`; `terrain.SetStepSizeCFD(step)` (:108); record 0.01 s |
| SPH spacing | `initial_spacing_m = 0.08` (Chrono C++ demo uses 0.04; design doc proposed 0.05) ; `d0_multiplier = 1.0` |
| bed | flat box `Construct(ChVector3d(150,150,0.25), center (0,0,0), BoxSide_ALL & ~BoxSide_Z_POS)` (:150-158) — **flat only; heightmap never used in this repo** |
| soil | density 1700 kg/m3, cohesion 5000 Pa, friction `mu_fric_s = mu_fric_2 = 0.8`, `mu_I0 = 0.04`, Young 1e6 Pa, Poisson 0.3, `average_diam = 0.005 m` |
| SPH | `IntegrationScheme_RK2`, `free_surface_threshold 2.0`, `artificial_viscosity 0.5`, `ShiftingMethod_NONE` (ppst push/pull 1.0), `ViscosityMethod_ARTIFICIAL_BILATERAL`, `BoundaryMethod_ADAMI`, consistent gradient/laplacian OFF, `num_proximity_search_steps 4` |
| active domain | `SetActiveDomain(ChVector3d(2.0, 2.0, 1.0))`, `SetActiveDomainDelay(0.1)` |
| MBD solver | `Type_BARZILAIBORWEIN`, `Type_EULER_IMPLICIT_LINEARIZED`, `system.SetNumThreads(12, 1, 1)`, Bullet (:100-103) — note this REPLACES the default solver the rigid f104 runs use |
| vehicle | HMMWV_Full, SMC, SHAFTS / AUTOMATIC_SHAFTS / AWD / PITMAN_ARM, **tire `RIGID_MESH`**, no `chassis_collision` key (-> NONE), init z = 0.7 m above a z=0 bed, warm-up 0.2 s |
| wheel FSI coupling | `terrain.RegisterVehicle(vehicle)` (:109); per wheel `terrain.AddRigidBody(spindle, ChBodyGeometry{coll_meshes=[TrimeshShape(VNULL,QUNIT, GetVehicleDataFile("hmmwv/hmmwv_tire_coarse_closed.obj"), VNULL)]}, False)` (:139-146) -> BCE markers |
| stepping | `driver.Synchronize; terrain.Synchronize(t); hmmwv.Synchronize(t, inputs, terrain); driver.Advance(dt); terrain.Advance(dt)` — **no `hmmwv.Advance`** (smoke :516-560; env :47-57) |
| tire channels | `capture_crm_tire_fields` (:163-202): `terrain.GetFsiBodyForce(spindle)` / `GetFsiBodyTorque(spindle)`; `force_wheel_fz = F . world_up`; `spindle_omega = spindle.GetAngVelParent() . spin_axis`; camber/deflection = 0. Same key names as `tire_field_names()`; `capture_crm_row` = `capture_row(include_tires=False)` + these |
| bounds | terminate when within `boundary_margin_m = 5` of the bed edge |
| resume | `collect_hmmwv_crm_dataset.py --resume`: episode sidecar `episodes/<name>.json` = completion marker (:337) |

Particle count: 150x150x0.25 @ 0.08 -> 1876 x 1876 x 4 = ~14.1 M SPH particles. **f104 arena 80x80, depth 0.25, spacing 0.08, uniform depth -> 1001 x 1001 x 4 = ~4.0 M** (+ ~3 M bottom BCE at 3 layers + walls); at 0.04 spacing -> 2001^2 x 7 = ~28 M.

**Achieved CRM throughput: NOT recorded anywhere I could find** (no number in `docs/progress.md`, `docs/hmmwv_crm_data_collection_pipeline.md`, `docs/collection_pipelines_amd_port.md`, no `artifacts/throughput*`, no CRM run.log on this machine). Only qualitative notes: `probe_sim_fps.py` docstring ("CRM SPH + 12 threads has frozen this box under larger/parallel loads"; defaults warm-up 0.6 s + 2.0 s measured), env docstring ("minutes-per-reference, not seconds"), dataset card (2,000 episodes of 12-18 s, collected serially in one tmux on the workstation). To measure: `$ENV scripts/throughput/probe_sim_fps.py --case hmmwv_crm --sim-seconds 3 [--crm-threads N]` (reports steps/s, ms/step, RTF, `crm_particles`); `terrain.GetRtfCFD()` / `GetRtfMBD()` also exist.

## 11. Cluster CRM runtime facts (checked today)

- `env.sh` sets `CHRONO_BUILD=$NRD_ROOT/chrono-build` (**no FSI**); `nrd_pychrono` puts `$CHRONO_BUILD/bin` on PYTHONPATH. The FSI superset build is `/work1/dannegrut/harry/nrd/chrono-build-fsi` (`bin/pychrono/{core,vehicle,sensor,fsi,fea,robot,postprocess}`; HIP for gfx90a/gfx942/gfx950; Chrono `main` @ f54254fa, 2026-09-07). For CRM: `export CHRONO_BUILD=$NRD_ROOT/chrono-build-fsi` before `nrd_pychrono`, and pass `--chrono-data $NRD_ROOT/chrono-build-fsi/data` (gen_runner.py:27 hard-codes `chrono-build/data`; both data trees contain `vehicle/hmmwv/hmmwv_tire_coarse_closed.obj`).
- API drift vs local pychrono 10.0.0 (conda `nedm`), verified by reading the SWIG wrappers:
  | local 10.0.0 | cluster main | status in this worktree's `hmmwv_crm.py` |
  |---|---|---|
  | `fsi.ElasticMaterialProperties()` | `fsi.SoilProperties()` (same 8 fields + `rheology_model`, `mcc_*`) | **NOT handled here** (:111); handled in `/home/harry/NeDM` (branch nrd_vision) via `getattr` |
  | `terrain.SetElasticSPH(m)` | `terrain.SetCrmSPH(m)` | **NOT handled here** (:120); handled in `/home/harry/NeDM` |
  | `terrain.SetActiveDomainDelay(s)` | **`terrain.SetFreeFlowDuration(s)`** (no `SetActiveDomainDelay` in cluster build) | **NOT handled in either checkout** (:149) — `docs/collection_pipelines_amd_port.md` calls it a "2-line rename"; it is 3. CRM has evidently never been stepped on the cluster (doc only claims import + symbol presence). |
  | heightmap ctor | identical in both: `Construct(heightmap_file: str, length, width, ChVector2d height_range, depth, uniform_depth: bool, ChVector3d pos, int side_flags)` | unused so far |
- `CRMTerrain::GetHeight` returns **0.0**, `GetNormal` returns vertical, `GetCoefficientFriction` 0 (CRMTerrain.h:79-86) — placeholders.
- `sinfo` (nodes, cpus/node; GRES column is null so GPU counts are not reported by SLURM): mi2101x (24, 16), mi2104x (21, 128, exclusive per AMD port doc), mi2508x (10, 128), mi3008x (2, 192), mi3001x (8, 16), mi3501x (8, 24, default), mi3508x (4, 256), mi3258x (1, 256); listed time limit 4-00:00:00 on all but `devel` (30 min). (Memory note says a 4 h cap applies on MI350 — treat the effective limit as unverified.)

## 12. RIGID -> CRM change list (every place found)

| # | where | rigid behaviour | CRM requirement |
|---|---|---|---|
| 1 | scene.py:87 `tire_model: TMEASY` | handling tire that queries terrain height/normal/friction | must be `RIGID_MESH` (or RIGID); TMEASY on CRM sees height 0 everywhere |
| 2 | scene.py:93 `chassis_collision: HULLS` | chassis hulls collide with the terrain mesh (belly/high-centre contact, crater-rim strikes) | SPH soil only feels bodies registered with `AddRigidBody`. **Only wheels are registered today -> chassis passes through soil.** Decide explicitly: register the chassis hull as an FSI body (cost: more BCE, larger active domain) or accept no belly contact. `chassis.GetContactForce()` (rich telemetry `chassis_contact_resultant_n`) will read 0 regardless; use `GetFsiBodyForce(chassis)` if registered. |
| 3 | scene.py:58-59 step 2 ms / tire 1 ms | 25 substeps per 50 ms frame | 5e-4 used before -> 100 substeps/frame; `DT/dt` must stay an integer (chrono.py:196). Python per-substep overhead (observer `on_substep`/`on_post_substep`, driver sync) x4. |
| 4 | scene.py:375-396 | `RigidTerrain` + `AddPatch(bmp)` + `Initialize`, `patch_body` | `veh.CRMTerrain(system, spacing)` ... `Construct(str(bmp), 80., 80., ChVector2d(-1.8, 3.9), depth, True, ChVector3d(0,0,0), BoxSide_ALL & ~BoxSide_Z_POS)`; must run AFTER `create_hmmwv` (needs spindles, reconfigures solver/threads) and BEFORE any stepping; `TraverseScene.patch_body` -> `terrain.GetGroundBody()`; add `wheels` list to the scene for FSI force capture. |
| 5 | scene.py:410-413 | collision-family mask vs terrain patch | drop (no patch collision model); irrelevant with `assets == []` |
| 6 | chrono.py:162 | init z = `tmap.height(start)+0.75` | keep formula but re-validate: CRM surface is grid-quantised (`Iz = round(z/spacing)`, 0.08 m steps) and the wheels sink; prior CRM runs used +0.7 over a flat bed. Launch gate (speed<=1, tilt<=20 deg, xy<=1 m after 0.8 s) must still pass on slopes up to 34 deg. |
| 7 | chrono.py:287-289 | `driver.Advance; terrain.Advance; hmmwv.Advance` | **delete `hmmwv.Advance(dt)`** (CRMTerrain::Advance -> DoStepDynamics -> VehicleAdvance callback). Keep both Synchronize calls (:226-227, :326-327). |
| 8 | chrono.py:229, :328 + hmmwv_data.py:513 | `capture_row(include_tires=True)` -> `ReportTireForce` | replace with `capture_crm_row(hmmwv, terrain, wheels, ...)` so state[7:11] = FSI `force . world_up` and state[11:15] = spindle omega with the SAME key names; still append `engine_motor_speed_radps`, `engine_motorshaft_torque_nm` (:231-232). Distribution shift to expect: FSI Fz is noisy/impulsive vs TMEASY; any normalisation stats in the hazard model trained on rigid will not match. |
| 9 | chrono.py:189 `vehicle.GetTire(...).GetRadius()` | TMEASY radius | works for RIGID_MESH too; `collect_wheel_runtime` already stores it |
| 10 | gen_collect.py:83-99 `validate_native_height` | 50 x `terrain.GetHeight` vs BMP oracle, hard fail | **fails by construction on CRM (GetHeight == 0).** Replace with an SPH-surface audit (e.g. particle positions from `GetFluidSystemSPH()`/`SaveInitialMarkers`/`GetSPHBoundingBox`, tolerance >= spacing) or a settled-wheel-height audit; keep writing `native_height_check.json` because `f104_episode.json` reads `native_height_report["passed"]` (:308). |
| 11 | gen_collect.py:196-207 `F104Telemetry` | `terrain.GetNormal` under hubs | returns (0,0,1) on CRM -> `*_force_projected_terrain_normal_n` silently becomes world-Z force; compute the normal from `tmap.gradient` instead or drop the field |
| 12 | fdm_rich_telemetry.py:171-177 | `tire.ReportTireForce`, native slip/deflection getters | wrapped in `_read` (NaN + capability log) so they will not crash, but `*_force_world_*` would be wrong/zero; feed FSI forces. `unavailable_by_design.soil_sinkage_m` and the "Rigid terrain plus TMeasy" limitation strings (:322-330) are now false statements. |
| 13 | gen_collect.py:166-190 `adapted_function` | text-patches `run_chrono` at 3 exact anchor strings (counts 1, 2, 1) | a CRM fork of `run_chrono` must keep these strings verbatim: `"    hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain\n"`, `'                    if args.command == "observe":\n'` (x2), `'        frame += 1\n'` — or the wrapper fails closed. |
| 14 | gen_collect.py:27-30, :241-242, :271-274 | source manifest of 9 files; runtime fingerprint of `chrono-build` | new frozen source root + manifest incl. `hmmwv_crm.py`; new fingerprint JSON for `chrono-build-fsi` (must contain a `_vehicle.so` key and a `/vehicle/hmmwv/` key; add `_fsi.so`) |
| 15 | gen_collect.py:123 bounds `> 40.0` | vehicle drives off the rigid patch edge | CRM bed has BCE side walls at +-40 m; previous CRM runs stopped 5 m inside the edge. Measured on the 200 gen_v1 f104 test cases: start/goal |x|,|y| <= 33.9 m, but route waypoints reach 37.9 m (2.1 m from the wall, inside the old 5 m CRM margin); decide whether to keep 40.0 (label parity) or add a margin. |
| 16 | gen_runner.py:27, :29, :37 | hard-coded `chrono-build/data`; `timeout=3600`; `GEN_WORKERS = cpus-2` CPU processes | point at the FSI data dir; **raise/remove the 3600 s timeout** (uncaught `TimeoutExpired` kills the shard; a 120 s-horizon CRM episode = 240k SPH steps); workers = number of GPUs, each pinned (`HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` per worker — nothing in the code selects a device today). |
| 17 | gen_array.sbatch | `OMP_NUM_THREADS=1`, `-t 03:00:00`, no GPU request, `nrd_use_lavapipe` | `configure_crm_terrain` calls `system.SetNumThreads(chrono_threads=12,1,1)`; reconcile with `OMP_NUM_THREADS=1` (the AMD doc warns SLURM/OMP settings silently change thread counts). Longer wall limit. lavapipe not needed when `render=None`. |
| 18 | hmmwv_crm.py:100-103 | rigid runs use Chrono default solver/timestepper | CRM path switches to Barzilai-Borwein + Euler implicit linearized -> vehicle dynamics differ from rigid even before soil effects; record in provenance (`simulation_provenance.json` has `physics_dt_s` only). |
| 19 | hmmwv_crm.py:111, :120, :149 | pychrono 10.0.0 names | add `SoilProperties`/`SetCrmSPH`/`SetFreeFlowDuration` fallbacks (see sec. 11) |
| 20 | scene.py:434, :443 | cameras attached to `patch_body` | only matters for `observe`/render paths (not gen_collect); SPH particles are not visible to the sensor module anyway — overhead depth maps must keep coming from the rigid/BMP render |

## 13. Gotchas specific to a heightmap CRM bed (from reading `ChFsiProblemCartesian::Construct(heightmap…)`, chrono main)

- Pixel convention: `dx = length/(nx-1)` — BMP samples at patch EDGES, same as `RigidTerrain` (so the known `TerrainMap` 511/512 radial offset vs Chrono, <= 0.078 m at the edge, carries over unchanged; CRM and rigid share the horizontal frame). Image top row -> +y (code flips y after generation), i.e. the same orientation `arena_meta.orientation {rot90:0, flipud:true}` was calibrated for — **but unverified empirically for CRM; `GetHeight` cannot be used to check it.**
- Surface height is quantised to the SPH grid: `Iz = round(z / spacing)` -> 0.08 m staircase at the previous spacing (BMP quantum is 0.022 m). The old audit tolerance (p95 <= 0.08 m) is at the quantisation limit.
- `uniform_depth=False` fills down to a flat floor near `z = -depth` measured from z=0 (`nz = Iz + Nz`): with f104 heights down to -1.8 m this yields `nz <= 0` -> **holes**, and metres-deep soil under hills. Use `uniform_depth=True` (constant soil thickness following the surface, bottom BCE follows the terrain) or shift `height_range`/`pos.z`.
- Bottom BCE is only generated when `BoxSide_Z_NEG` is in `side_flags` (it is, with `ALL & ~Z_POS`).
- Soil friction 0.8 -> repose ~38.7 deg + 5 kPa cohesion vs max terrain slope 34 deg (p99 28.6 deg): particles outside the active domain are frozen, inside it slopes near the cap may creep under the wheels. `SetActiveDomainDelay/SetFreeFlowDuration(0.1 s)` lets the full 4 M-particle bed move for the first 0.1 s of the 0.8 s settle.
- Active domain `(2.0, 2.0, 1.0)` m is per registered FSI body (the 4 spindles); z extent 1.0 m is fine on slopes only because it is body-relative.
- First-ever heightmap CRM + first-ever CRM stepping on the cluster build: budget a smoke test before anything else (construct, print `GetNumSPHParticles()`, settle 0.8 s, check sum Fz ~ weight 25.2 kN as `collect_hmmwv_crm_smoke.summarize_force` does, `min pos_z` sanity).
- Rollover/goal/blockage/parked logic, route format, driver, action columns, pose/yaw definitions, 20 Hz sampling, `anchor_state`/`history` convention and the output file set need NO change and should be kept byte-compatible so the existing label/dataset scripts run unmodified.

## 14. Key paths

- /home/harry/NeDM-traverse_mppi/scripts/gen_runner.py, gen_collect.py, gen_array.sbatch, gen_arenas.json, collect_traverse_f104.py
- /home/harry/NeDM-traverse_mppi/scripts/traverse_fdm_rgbd_diverse_chrono.py (run_chrono :147-399, make_driver :98-112)
- /home/harry/NeDM-traverse_mppi/src/nedm/traverse/scene.py (build_config :54-106, build_scene :345-464), terrain.py (TerrainMap :260-336), fdm_rich_telemetry.py, fdm_data.py (build_history :79-93)
- /home/harry/NeDM-traverse_mppi/src/nedm/hmmwv_data.py (create_hmmwv :290-336, capture_row :438-544), src/nedm/training/constants.py (:54)
- /home/harry/NeDM-traverse_mppi/src/nedm/hmmwv_crm.py; src/nedm/rl/hmmwv_chrono_crm_tracking_env.py; configs/hmmwv_crm_eval.json; scripts/collection/{collect_hmmwv_crm_smoke.py,collect_hmmwv_crm_dataset.py,prepare_hmmwv_crm100_generation.py,run_hmmwv_crm2000_collection.sh}; scripts/throughput/probe_sim_fps.py; docs/hmmwv_crm_data_collection_pipeline.md
- /home/harry/NeDM/src/nedm/hmmwv_crm.py (has the SoilProperties/SetCrmSPH fallbacks), /home/harry/NeDM/docs/collection_pipelines_amd_port.md, /home/harry/NeDM/docs/chrono_amd_cluster.md
- /home/harry/NeDM-traverse_mppi/assets/traverse/arena_f104_50h_v1/{arena_000.bmp,arena_meta.json}
- /home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_f104_50h_20260909/gen_v1/{PLAN.md,REPORT.md,LOG.md,tasks_test.json,tasks_data.json,cases_test_f104/,test_f104/routes/,data/runs/}
- amd:/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/{gen_v1/{source,gen_collect.py,gen_runner.py,gen_array.sbatch,test/runs,data/runs},pilot_runtime_412394.json}
- amd:/work1/dannegrut/harry/nrd/{env.sh,chrono-build,chrono-build-fsi,smoke_fsi.py,smoke_fsi_api2.py}; Chrono source amd:/home1/harry/chrono (src/chrono_vehicle/terrain/CRMTerrain.{h,cpp}, src/chrono_fsi/sph/ChFsiProblemSPH.cpp:786-970)
