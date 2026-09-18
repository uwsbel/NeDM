# f104 rigid pipeline recon — LABELS, DATASET BUILD, TRAINING (for the CRM port)

Read-only recon, 2026-09-16. Repo `/home/harry/NeDM-traverse_mppi` (branch `traverse_mppi`), campaign artifacts
`A = artifacts/traverse/fdm_f104_50h_20260909`, cluster campaign `C = /work1/dannegrut/harry/experiments/fdm_f104_50h_20260909`.
ALL f104/gen/sensor/nav scripts are UNTRACKED in git (`git status` shows `?? scripts/...`); nothing below is tied to a commit.
`[RIGID]` = rigid-terrain-specific, needs a decision for CRM. `[GOTCHA]` = trap.

---------------------------------------------------------------------------------------------------------------------

## 0. TL;DR — what is deployed, where it lives

| Use | Model | Input | Files (local) | Files (cluster) |
|---|---|---|---|---|
| **gen_v1 single-goal tests + 5-goal missions** (and night-2 closed loop, demo videos) | **N2**: GRU, geometry-only ctx, **5 seeds** | 5-ch height corridor `X` (+const ch) from arena heightmap via `gen_planner.set_map` | `A/night2_v1/final/N2_s{0..4}.pt` (+`N2_meta.json`) md5 s0 `432515a4…` | `C/gen_v1/models/N2_s{0..4}.pt` (same md5) |
| tilt-aware variant (tested once, 300 hazard groups) | **N2T**: same, label = unsafe OR tilt>30 deg | same | `A/night2_v1/final/N2T_s{0..4}.pt` | NOT on cluster |
| designed-data-only ablation | N2A | same | `A/night2_v1/final/N2A_s{0..4}.pt` | no |
| **nav_v1 (most recent deployed; sensor-driven waypoint navigation)** | **matched_Dabs**: GRU, geometry-only ctx, **3 seeds** | 4-ch direct depth `[range_abs, sec1, speed, valid]` (+const) from the back-projected world grid, vehicle footprint masked | `A/sensor_v2/matched/matched_Dabs_s{0,1,2}.pt` md5 s0 `9d9c1a88…` | `C/sensor_v2/matched/` and copy `C/nav_v1/models/` (same md5) |
| nav/sensor_v2 height control arm | **matched_H** 3 seeds | `[z_rel, grade, cross, speed, valid]` from the same grid | `A/sensor_v2/matched/matched_H_s{0,1,2}.pt` md5 s0 `1bea5a4c…` | same two dirs |
| also present | matched_H0, matched_Drel (3 seeds each); sensor_v1 `final/{D,E0}_s{0..4}.pt`; night-1 `night_v1/final/{H1,A1}_full_s{0..4}.pt` (trained on the WRONG throttle column) | | | `C/nav_v1/matched_r20/r20_{Dabs,H}_s{0,1,2}.pt` = retrain with 20 m sensing radius (not deployed in the main campaign) |

All of these are the SAME network (`scripts/gen_riskmodel.py:Net`, arch `gru`), same loss, same optimiser; only input
channels / training rows differ. N2 trained on f104 only (31,851 rows). matched_* trained on f104 + g228 + g203 + g217
(54,180 rows), held-out g216 + g231.

---------------------------------------------------------------------------------------------------------------------

## 1. Episode files the labeller consumes (per run dir)

Needed: `trajectory.npz`, `outcome.json`, `command_reference.npz`, `case.json`, `anchor_state.npz` (progress.md:420; the
rest of the rich telemetry was deleted in the 09-15 cleanup).

`trajectory.npz` (20 Hz, `dt_s`=0.05 float32; N frames):
- `state (N,17) float32`, fields (`state_fields`): `vel_body_x_mps, vel_body_y_mps, roll_rad, pitch_rad, roll_rate_radps,
  ang_vel_body_y_radps, yaw_rate_radps, tire_{fl,fr,rl,rr}_force_wheel_fz_n, tire_{fl,fr,rl,rr}_spindle_omega_radps,
  engine_motor_speed_radps, engine_motorshaft_torque_nm`
- `action (N,3) float32` = **[steering, throttle, braking]** (verified: col0 in [-1,0.64], col1 in [0,1], col2 in [0,0.03])
- `pose (N,3) float64` = [x, y, yaw] (NOT z)
- also `terminal_pose, terminal_state, power_kw, contact_n, parked (N,) bool, terminal_parked, positive_work_kj_per_interval`
`command_reference.npz`: `reference_waypoints (M,2)`, `reference_stations (M,)`, `reference_speeds (M,)`,
`reference_headings (M,)` (+ `interval_start_s`, `desired_speed_mps` per frame). Waypoint step 0.5 m.
`outcome.json`: `status` in {`goal_reached`, `prolonged_blockage_terminated`, `timeout`, `terrain_bounds_exit`, `rollover`}
(priority rollover > goal > bounds > blockage > timeout; `scripts/gen_collect.py:54-154` StopPolicy). goal radius 2.5 m.
`case.json`: `id` (group id), `split`, `goal_xy`, `layout.start_xy`, `layout.start_yaw` (+ arena, family, ...).
`anchor_state.npz`: `state (17,) float32` (settled state at t=0), `pose (3,)`, `history (16,24)`, `goal_xy`, `goal_radius_m`.

`[RIGID]` state cols 7-10 are rigid-contact tire normal loads, cols 11-16 spindle/engine. The deployed models ignore
all 17 (ctx variant `none`), BUT every builder hard-codes ctx = `concat(anchor_state.state[17], 5 geometry terms)` and
the trainers select `GEOM = [17,18,19,20,21]`. CRM collector must still write a 17-long `anchor_state.state`
(zeros/NaN-free placeholders are fine for the deployed variant) or the ctx column indices shift silently.
`[RIGID]` `gen_collect.StopPolicy.validate_native_height` calls `scene.terrain.GetHeight` (RigidTerrain) and aborts the
episode if it disagrees with the BMP (collector subsystem, but it gates whether an episode exists to label).

---------------------------------------------------------------------------------------------------------------------

## 2. Labelling contract (authoritative copy: `scripts/f104_n2_dataset.py:one()` lines 91-123)

Identical copies: `scripts/sensor_dataset_v2.py:labels_only()` lines 114-154 (verified identical on 40 routes, sensor_v2
LOG) and the night-1 local indexer `scripts/f104_night_index.py:one()` lines 51-85.

Constants: `DT=0.05`, `SETTLE_S=1.0` -> `s0 = 20` frames skipped (f104_n2_dataset.py:13,101). `[GOTCHA]` DT is
hard-coded, NOT read from `trajectory['dt_s']`; "20 frames" = 1 s only at 20 Hz.

```
vx  = state[:,0]          # body-frame forward speed, m/s
thr = action[:,1]         # THROTTLE = column 1   (line 100)
fail  = outcome.status != 'goal_reached'
back  = (vx < -0.10) & (thr > 0.3)                               # rolling backwards under throttle
rb    = first frame >= s0 with  back | (vx < -0.30)              # first backward frame (run length 1)
ns    = first frame >= s0 starting >=20 consecutive frames of (|vx| < 0.3) & (thr > 0.3)   # >=1 s effortful near-stop
ev    = min(rb, ns) if any else (N-1 if fail else None)          # terminal frame for failures with neither
back_s = back[s0:].sum()*DT ;  min_vx = vx[s0:].min()
clean = (not fail) and back_s < 0.05 and min_vx > -0.30          # back_s<0.05 == ZERO backward frames
unsafe = not clean ;  if clean: ev = None                        # clean runs are censored even if they crawled
```
So **unsafe = failed OR any post-settle frame with (vx<-0.10 & throttle>0.3) OR min vx <= -0.30**. The near-stop test
never makes a run unsafe by itself; it only localises the event earlier for runs that are already unsafe.
In `station_ds_all.npz`: unsafe 32.4%, fail 16.8%; `event_idx>=0` iff `unsafe==1` (checked: 0 violations either way).

**Event-station assignment** (lines 111-113, 122): project every pose onto the planned reference polyline
(`project()`, lines 71-79: closest-segment arc length), take the **high-water mark of arc-length progress up to and
including the event frame** `hwm = s[:ev+1].max()` (not the pose at the event frame), `frac = hwm / polyline_length`,
`event_idx = clip(round(frac*95), 0, 95)`; `-1` = censored. Event deciles on the training set: p10 21, p50 45, p90 71.

**Throttle-column fix (2026-09-11):** `action = [steering, throttle, braking]`. Night-1 (`f104_night_index.py` before
09-11) read column 0 (steering) as throttle; `A/night_v1/episodes_v0_steering_col.json` is that version and the
night-1 models `night_v1/final/{H1,A1}_full_s*.pt` were trained on it. Everything from `station_ds_fix.npz` onward
(N2, N2T, sensor_v1, sensor_v2, nav) uses column 1. `station_ds_fix.npz` was made by re-running the fixed
`f104_night_index.py` and overwriting `event_idx/unsafe/fail` in `night_v1/station_ds.npz` (inline python, no script).
Same convention in the collector's blockage rule (`gen_collect.py:129`, `actions[-40:][:,1] > .3`).

**Tilt variant N2T** (`A/night2_v1/PLAN.md:127-139`, LOG 03:57): `max_tilt = degrees(|state[20:, 2:4]|).max()` (roll,
pitch, after the 1 s settle; `scripts/f104_n2_tilt_label.py` for night-1 routes -> `night2_v1/tilt_night1.json`; cluster
`C/tilt_cluster.py` for waves A1/A2 -> `C/tilt_A1.json`, `C/tilt_A2.json`). `unsafe_tilt = unsafe OR max_tilt > 30`.
Built by an INLINE python snippet (no script in repo): copies `station_ds_all.npz`, sets `unsafe=new`, keeps
`unsafe_plain`, adds `max_tilt`, and for the 2,700 routes flagged only by tilt sets **`event_idx = 95`** (event pinned to
the route END, not located). Output `station_ds_tilt.npz` was deleted locally 09-15 (rebuildable). Train threshold 30 deg,
test threshold 35 deg, deliberately. Trained with
`python scripts/f104_n2_deploy.py --ds A/night2_v1/station_ds_tilt.npz --tag N2T --seeds 5`.

`[RIGID]` label risks on CRM (decide before collecting, the thresholds are part of the contract):
- Sinkage/slip stalls: on soft soil `|vx|<0.3 & thr>0.3` for >=1 s will be common and end as
  `prolonged_blockage_terminated` -> fail -> unsafe with a `near_stop` event. Same rule, very different base rate.
- `vx<-0.10 & thr>0.3` with wheel slip: vx is chassis body speed so slip itself is not flagged, but soil creep/settling
  after t=0 can give small negative vx; the 1 s settle (plus the collector's 0.8 s pre-settle) was tuned on rigid
  ground. Check `settle_neg` / `vx[:40]` distributions on a CRM pilot before trusting `s0=20`.
- Tilt on deformable soil includes sinkage-induced pitch; N2T's 30 deg may need re-checking.
- Labels are per-episode and terrain is reset each episode; ruts are never in the model input.

---------------------------------------------------------------------------------------------------------------------

## 3. Model-input contract (corridor tensor) — `f104_n2_dataset.station_tensor()` lines 45-60

Identical copies: `f104_night_dataset.py:33-48`; planner-side call `gen_planner.corridors()` lines 141-146.

- `N_STATION=96`, `N_LATERAL=32`, `HALF_WIDTH_M=6.0` (corridor +-6 m, lateral spacing 12/31 = 0.387 m).
- `resample_route(wp, stations, 96)`: 96 points uniform in `reference_stations` (falls back to polyline arc length if
  stations are non-monotone). Tangent = `np.gradient(pts)`, left normal = (-ty, tx); lateral offsets
  `linspace(-6, 6, 32)` along the normal (index 0 = right side, 31 = left).
- `sample_map(gx, gy)` (lines 27-34): bilinear on `G['rgbd']` (4,n,n), `row = ctr - y/mpp`, `col = ctr + x/mpp`
  (row 0 = +y), elevation = `rgbd[3] * elev_scale` (scale 10.0), valid iff all 4 corner texels `> -1.999`.
- Channels `X (5,96,32)`:
  0 `elev - e0` [m], e0 = elevation at station 0 centre sample (col 16) (nan-mean of station 0 if invalid); invalid cells filled with e0
  1 along-path grade `np.gradient(fill, ds, axis=0)` clipped +-2, `ds = route_len/95`
  2 cross slope `np.gradient(fill, dl, axis=1)` clipped +-2, `dl = 12/31`
  3 **commanded speed** [m/s] at the station (interp of `reference_speeds` over polyline arc length), broadcast over lateral
  4 valid mask (0/1)
- stored `float16`. Trainer adds a 6th constant-ones channel ("known" flag, legacy of night-1 speed-dropout) -> `cin=6`.
- `ctx (22,) float32` = `[anchor_state.state(17), goal_dx, goal_dy, |goal-start|, start_yaw, route_len]`
  (lines 115-117). **Deployed = geometry-only**: cols `[17..21]`. Planner side: `gen_planner.geom_ctx(start_xy,
  goal_xy, start_yaw, L)` lines 149-151 (nav_v1 passes the CURRENT pose as start).
- Normalisation (in trainer, frozen into each checkpoint): channels 0-3 standardised with fit-set per-channel mean/sd
  over all rows/stations/laterals; ch 4 and the ones-channel raw; ctx standardised per column.
  N2: `mu=[0.6108, 0.00675, 0.00469, 3.3179]`, `sd=[1.0153, 0.1782, 0.1863, 1.7417]`,
  `ctx_mu=[7.22, 2.75, 44.32, -0.094, 45.34]`, `ctx_sd=[32.02, 31.35, 10.18, 1.649, 10.19]`.

**Two map sources feed the same sampler** (module global `f104_n2_dataset.G`):
1. `init_map(root)` lines 17-24: `root/static_map_v1/observation.{json,npz}` = one vehicle-free overhead depth render
   (`rgbd (4,512,512)`, cam 110 m, hfov 0.8203 rad -> `mpp = 2*110*tan(hfov/2)/512 = 0.1868 m`, covers +-47.8 m, off-terrain
   texels = -2 sentinel -> valid=0). **All N2/N2T training tensors (station_ds_*) came from this.** `[RIGID]` it is a
   Chrono sensor render of the RIGID mesh, flat-ground pixel placement (v1 sampling, up to ~1.5 m misplacement on slopes).
2. `gen_planner.set_map(arena_dir)` lines 35-42: `TerrainMap.from_dir` heightmap, `flipud`, `R[3]=h/10`, `mpp = size_m/n
   = 0.15625`. **Used by gen_v1 pools/missions and `gen_build_dataset.py`.** Agreement with (1) on f104: 0.035 m mean
   height error, logit corr 0.985, identical pick 32/40.
   `[GOTCHA]` with set_map the valid channel is ALWAYS 1 (index clipping replicates the edge; no off-arena sentinel).
   `[GOTCHA]` TerrainMap puts BMP samples at cell centres (80/512) while Chrono RigidTerrain puts them at patch edges
   (80/511): `x_terrainmap = (511/512) x_chrono`, up to 0.078 m at the edge (sensor_v2 LOG). `[RIGID]` re-check how
   CRMTerrain lays the same BMP out before assuming either registration.

For CRM on the SAME BMP the corridor tensor of a given route is identical to the rigid one (input is pre-traversal
geometry + commanded speed); only labels change. That is what makes "frozen rigid model vs CRM-trained model on the same
X" a clean comparison.

### sensor_v2 corridor (`scripts/sensor_dataset_v2.py`), used by matched_* and nav_v1
- World grid from `scripts/sensor_map_v2.py` (`--maps <captures> --out <dir>`): depth back-projected with intrinsics to a
  512x512 metric grid, `mpp 0.15625`, +-40 m, per-cell `z, range_m, sec, rgb, cover`; `grid.npz` + `grid.json`
  (`A/sensor_v2/grids/arena_*`, also `C/sensor_v2/grids/`). Grid row index = y ascending (cell centres `-half+(i+0.5)mpp`).
- `tensor12()` lines 65-98, `CHANNELS` line 22: `[z_rel, grade, cross, speed, valid, range_abs, range_rel, sec1, R, G, B,
  cover1]`; same 96x32x+-6 m geometry (`corridor_points` 53-62); valid = inside arena & all 4 cells covered & finite;
  invalid: z->z0, range->cam_h (110), sec->1.
- Variants (`sensor_train_v2.py:18-25`): `H=[z_rel,grade,cross,speed,valid]`, `H0`, `Drel`, **`Dabs=[range_abs,sec1,speed,valid]`**,
  `DabsC`, `RGBDabs`. Continuous channels standardised, `valid` raw, + ones channel (`cin` = 6 for H, 5 for Dabs).
  Dabs norm: `mu=[112.217, 0.02613, 3.3576]`, `sd=[2.3074, 0.01710, 1.7131]`.
- nav_v1 deployment input = `scripts/vehicle_corridor.py:tensor12_excluded()` (footprint 2.6 x 1.3 m half-extents + 1.5 m
  margin marked invalid; refs from first station with >=4 valid samples); `nav_mask_datasets.py` rewrites stored datasets
  the same way (`--radius-m 20 --margin 1.5`) for the r20 retrain.
- `[RIGID]` the grid is a render of the rigid visual mesh. With CRM the depth camera would have to see the SPH surface
  (or keep rendering the BMP mesh as a proxy) — perception subsystem decision; the training code does not care.

---------------------------------------------------------------------------------------------------------------------

## 4. Dataset builders

### 4.1 `scripts/f104_n2_dataset.py` (numpy only, cluster-runnable; THE labeller)
CLI: `--root <campaign root containing static_map_v1/>  --runs '<glob>:<source>' [...]  --out <npz>  --workers 16`.
As run (cluster, `C/n2_ds.sbatch`, `C/n2_ds2.sbatch`; env `source /work1/dannegrut/harry/nrd/env.sh; nrd_pychrono`):
```
"$NRD_PYTHON" -P -u $C/f104_n2_dataset.py --root $C --runs "$C/production_v3/runs/*:designed"  --out $C/night2_v1/station_ds_A1.npz --workers ${SLURM_CPUS_PER_TASK:-32}
"$NRD_PYTHON" -P -u $C/f104_n2_dataset.py --root $C --runs "$C/production_v4/runs/*:on_policy" --out $C/night2_v1/station_ds_A2.npz --workers ${SLURM_CPUS_PER_TASK:-32}
```
(14,348/14,381 and 9,309/9,520 labelled; a run missing any of the 5 files is silently dropped, `one()` returns None.)
npz schema: `X (n,5,96,32) f16`, `ctx (n,22) f32`, `id, group, split, source, status` (object), `profile i8`
(`int(id.split('_route_')[1]) % 4` = speed profile 0..3 = constant_2/4/6/smooth_2_6_2 for designed routes; `-1` otherwise),
`fail i8, unsafe i8, event_idx i16, route_len f32, min_vx f32, back_s f32`. `group = case['id']`, `split = case['split']`.

### 4.2 `scripts/f104_n2_merge.py` -> `A/night2_v1/station_ds_all.npz` (1.12 GB, LOCAL ONLY)
Concats `station_ds_fix.npz` (night-1, 12,542) + A1 (14,348) + A2 (9,309) = **36,199 rows / 2,700 groups**; forces
`split='train'` for night-2 waves; adds `wave`; keys `X, ctx, id, group, split, source, profile, fail, unsafe, event_idx,
route_len, wave`. split counts train 33,840 / test 1,595 / val 764. Sources: designed 26,069, on_policy 9,309, MPPI arms 821.
The intermediates (`station_ds_{fix,A1,A2,tilt}.npz`, `night_v1/station_ds.npz`) were deleted locally 09-15; A1/A2 remain at
`C/night2_v1/`. Cluster equivalent of the merged labels: `C/sensor_v1/labels_station_ds_all.npz` (all keys but X) +
`ids_station_ds_all.json` + `run_dirs_abs.txt` (41,684 run dirs), and the v2 tensors `C/sensor_v2/ds_v2_f104_train.npz`.

### 4.3 `scripts/f104_night_index.py` + `f104_night_dataset.py` (night-1, local, historical)
Index -> `A/night_v1/episodes.json` (adds `event_kind` rollback/near_stop/terminal, `event_s`, `event_frac`, `settle_neg`,
`max_dev_pre_event`, ...); dataset adds old-model features. Hard-coded paths into `traj_pull/` (deleted) and
`ProcessPoolExecutor(14)`: **not re-runnable now**; superseded by 4.1 for any new data.

### 4.4 `scripts/gen_build_dataset.py` (new arenas; heightmap corridors)
`python scripts/gen_build_dataset.py --runs A/gen_v1/data/runs A/gen_v1/test/runs --out A/gen_v1/station_ds_gen_v1.npz --workers 14`
(as run, local, from repo root). Calls `gen_planner.set_map('assets/traverse/arena_<tag>')` per arena then
`f104_n2_dataset.one((dir, None, source))` unchanged. `[GOTCHA]` arena tag = `basename.split('_')[0]` (`f104` ->
`arena_f104_50h_v1`, else `arena_<tag>`); source = `'designed' if '/data/' in root else 'gen_test_arm'`; relative asset
path -> must run from repo root. Output keys as 4.1 + `arena`; dtypes are numpy defaults (int64/float64/`<U`), X f16.
15,639 rows; **never used for training N2** (only analysis); the sensor_v2 equivalents were used for matched_*.

### 4.5 `scripts/sensor_dataset_v2.py` (v2 grid corridors)
Two modes (as run, `C/sensor_v2/ds_v2.sbatch`, `sbatch -p mi2101x -J ds_v2 --array=0-5`, ~10-30 s per arena on 16 cores):
```
# f104 training rows: tensors rebuilt, LABELS REUSED from station_ds_all (asserts id order)
"$NRD_PYTHON" -u sensor_dataset_v2.py --grid $C/sensor_v2/grids/arena_f104_50h_v1 --ids $C/sensor_v1/ids_station_ds_all.json \
   --run-dirs $C/sensor_v1/run_dirs_abs.txt --labels $C/sensor_v1/labels_station_ds_all.npz --out ds_v2_f104_train.npz --workers 16
# per arena: labels computed by labels_only()
"$NRD_PYTHON" -u sensor_dataset_v2.py --grid $GRID --runs "$C/gen_v1/data/runs/${A}_*" "$C/gen_v1/test/runs/${A}_*" \
   "$C/sensor_v1/test/runs/${A}_*" "$C/sensor_v1/test2/runs/${A}_*" --out ds_v2_gen_$A.npz --workers 16
```
Schema: `X12 (n,12,96,32) f16`, `channels`, `ctx (n,22)` (col 21 = v2 route length), labels as 4.1 + `arena`
(f104 mode also `route_len12`). Files: `C/sensor_v2/ds_v2_f104_train.npz` (2.7 GB) + `ds_v2_gen_{f104,g203,g216,g217,g228,g231}.npz`
(0.23-0.37 GB each); range-limited copies in `C/nav_v1/ds_r20/`. None of these are local.

---------------------------------------------------------------------------------------------------------------------

## 5. Architecture and loss (`scripts/gen_riskmodel.py`, identical to `f104_n2_train.py:Net` 27-69)

```
x (B,cin,96,32) -> 4x [Conv2d 3x3 pad1, stride (1,1) then (1,2),(1,2),(1,2)] + BatchNorm2d + GELU, ch cin->32->64->64->96
   (station axis never strided: 96 kept; lateral 32->32->16->8->4)
 -> cat[mean over lateral, max over lateral] (B,192,96) -> Linear(192,96)+GELU per station
 ctx (B,5) -> Linear(5,32)+GELU, broadcast to 96 stations;  pos = linspace(0,1,96)
 -> cat (96+32+1=129) -> Conv1d(129,96,k=5,pad=2)+GELU+Dropout(0.1) -> BiGRU(96->64, 1 layer, bidirectional) -> Linear(128,1)
 -> hazard logits (B,96)
```
256,161 trainable params (256,677 tensors incl. BN buffers). `layers=2`/`heads` only matter for `arch='tx'`.
Loss `survival_nll(haz, ev)` (`f104_night_train.py:63-72`, copy `sensor_train.py:57-64`): with `h=softplus(logit)`,
event rows: `sum_{s<ev} h_s + softplus(-logit_ev)`; censored rows (`ev=-1`): `sum_{s<=95} h_s`. Mean over batch. No class
weights, no augmentation.
Route score `route_logit = log(sum_s softplus(logit_s) + 1e-6)`; ensemble = **mean of route logits over seeds**, then
`P(unsafe) = 1 - exp(-exp(z))`; planner = `argmin z` over ~256 candidates (no time term). Scores are a ranking only
(badly overconfident: median predicted 0.008% vs realised 0.33%).

Hyperparameters (all variants; `f104_n2_train.train_one` 95-128, `sensor_train.train_one` 103-133, `sensor_train_v2.train_one` 92-127):
AdamW lr 2e-3, wd 1e-4; OneCycleLR(max_lr 2e-3, total_steps = 30 * (n_fit // 256), default pct_start 0.3);
batch 256, drop-last, fresh `torch.randperm` each epoch; grad-clip 5.0; **30 epochs**; seeds `0..k-1` via
`torch.manual_seed(seed); np.random.seed(seed)`; last-step weights saved (no early stopping / no model selection);
data kept in pinned host memory, one batch moved per step. NOT bit-reproducible (cuDNN/MIOpen GRU).
Dev fold (`f104_night_train.dev_group`): `md5(group) % 5 == 0`; `Data.fit = split=='train' & ~dev`;
`Data.dev = split=='train' & dev & source=='designed'` (f104_n2_train.py:72-92).
Selection metric: `G_unsafe` = AUC within (group, speed profile) cells; `W_` within group; `P_` pooled.

Checkpoint format (`torch.save` dict, load with `torch.load(p, map_location=dev, weights_only=False)`):
- N2/N2T/N2A: `state, arch='gru', layers=2, ctx_variant='none', ctx_cols=[17..21], norm={mu(4),sd(4)}, ctx_mu(5), ctx_sd(5), cin=6, nctx=5`
- sensor_v1/v2: same + `channels`, `norm={channels, cont_index, mu, sd}`, `variant`, (`version='v2'`); no `ctx_variant`.
Loaders: `gen_planner.RiskModel` (lines 154-184; normalises `x[:, :4]` with `norm.mu/sd`, appends ones; glob from arg /
`$GEN_MODELS` / default N2) — also loads any 5-ch H/E checkpoint correctly; `SensorRiskModel` / `GridRiskModel`
(268-323; select channels by name from the 10-/12-ch tensor, normalise `cont_index`).

---------------------------------------------------------------------------------------------------------------------

## 6. Training runs as actually executed

### 6.1 N2 / N2A / N2T — trained LOCALLY on the workstation RTX 5090 (conda env `nedm`, torch 2.12.0+cu130), repo root
```
/home/harry/miniconda3/envs/nedm/bin/python scripts/f104_n2_train.py --archs gru --ctxs none --seeds 1 --epochs 3 --out /tmp/smoke2.json   # smoke
/home/harry/miniconda3/envs/nedm/bin/python scripts/f104_n2_train.py            # sweep gru,tx,mlp x full,chassis,none, 3 seeds -> night2_v1/sweep.json (ds default station_ds_fix.npz)
/home/harry/miniconda3/envs/nedm/bin/python scripts/f104_n2_final.py --curve 1.0 --seeds 3 --out /tmp/n2_curve_op   # scaling curve -> scaling.json (does NOT save models; --final-seeds is unused)
/home/harry/miniconda3/envs/nedm/bin/python scripts/f104_n2_deploy.py --tag N2A --seeds 5                            # run BEFORE wave A2 was merged: 24,531 designed-only rows (today: add --sources designed)
/home/harry/miniconda3/envs/nedm/bin/python scripts/f104_n2_deploy.py --tag N2  --seeds 5                            # 31,851 rows  <- DEPLOYED
/home/harry/miniconda3/envs/nedm/bin/python scripts/f104_n2_deploy.py --ds A/night2_v1/station_ds_tilt.npz --tag N2T --seeds 5
```
`f104_n2_deploy.py` defaults: `--ds A/night2_v1/station_ds_all.npz --ctx none --arch gru --epochs 30`; fit = `D.fit | D.dev`.
Runtime: 31-41 s/seed alone (N2A 31 s @24.5k rows; N2 71-79 s while sharing the GPU with the scaling job); 3,720 steps.
`[GOTCHA]` `fit = D.fit | D.dev` and `D.dev` is designed-only, so the 1,989 dev-fold ON-POLICY/other-source rows are in
neither: deployed N2 used 31,851 of 33,840 train-split rows. Reproduce this quirk (or not) knowingly.
`[GOTCHA]` output dir is HARD-CODED: `A/night2_v1/final/{tag}_s{s}.pt` + `{tag}_meta.json` (f104_n2_deploy.py:25-32). Running it
with `--tag N2` on a CRM dataset OVERWRITES the frozen rigid checkpoints. Use a parametrised copy / a new tag.
`[GOTCHA]` `N2_meta.json` G_unsafe (~0.997) is in-sample (dev fold is inside the fit set).
`[GOTCHA]` import chain `f104_n2_deploy -> f104_n2_train -> f104_night_train -> train_f104_multihead`, all via
`sys.path.insert(0,'scripts')` (cwd = repo root). `f104_n2_train.train_one` pins the fit tensor on EVERY call (line 102).

### 6.2 sensor_v1 (cluster MI350X, partition `mi3501x`, 24 CPUs, self-contained: `sensor_train.py` + `gen_riskmodel.py`)
`C/sensor_v1/sensor_train.sbatch`:
```
source /etc/profile; module load pytorch/2.10.0; source /work1/dannegrut/harry/venvs/nedm/bin/activate
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID} MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}; mkdir -p /tmp/miopen_${SLURM_JOB_ID}
python3.12 -u sensor_train.py --ds $C/sensor_ds_f104.npz --out $C/$RUN_OUT --mode $RUN_MODE --variants $RUN_VARIANTS --seeds $RUN_SEEDS $RUN_EXTRA
```
submitted as
```
RUN_OUT=smoke RUN_MODE=sweep  RUN_VARIANTS=RGBD RUN_SEEDS=1 RUN_EXTRA='--max-steps 100' sbatch -p mi3501x -t 00:25:00 -J sensor_smoke -o $C/sensor_v1/logs/%x_%j.out --export=ALL sensor_train.sbatch
RUN_OUT=sweep RUN_MODE=sweep  RUN_VARIANTS=$v   RUN_SEEDS=3 RUN_EXTRA='' sbatch -p mi3501x -t 01:30:00 -J sweep_$v  -o ... --export=ALL sensor_train.sbatch   # sleep 15-20 s between submits
RUN_OUT=final RUN_MODE=deploy RUN_VARIANTS=D    RUN_SEEDS=5 RUN_EXTRA='' sbatch -p mi3501x -t 01:30:00 -J deploy_D -o ... --export=ALL sensor_train.sbatch
```
`--mode deploy` = "exactly the night-2 deployment rows" (`fit | dev`, 31,851), 5 seeds, saves `<out>/<variant>_s<seed>.pt`.
Runtime MI350X: **130-170 s/seed** (3,720 steps) -> 13-15 min per 5-seed job incl. load; sweep jobs 5-7 min.
Variant `E` = `[elev_rel,grade,cross,speed,valid]` = the N2 input; dev G_unsafe 0.978 vs N2-protocol 0.983 (pipeline check).

### 6.3 sensor_v2 matched (cluster MI350X) — produced matched_H / matched_Dabs (nav_v1's models)
`C/sensor_v2/train_v2.sbatch` (same env block + MIOPEN lines; `-c 24 -t 03:00:00`):
```
python3.12 -u sensor_train_v2.py --files $C/ds_v2_f104_train.npz $C/ds_v2_gen_f104.npz $C/ds_v2_gen_g228.npz $C/ds_v2_gen_g203.npz \
  $C/ds_v2_gen_g217.npz $C/ds_v2_gen_g216.npz $C/ds_v2_gen_g231.npz --out $C/$RUN_OUT --variants $RUN_VARIANTS --seeds $RUN_SEEDS --tag $RUN_TAG $RUN_EXTRA
```
submitted as (one job per variant, 12 s apart)
```
for v in H H0 Drel Dabs; do RUN_OUT=matched RUN_VARIANTS=$v RUN_SEEDS=3 RUN_TAG=matched RUN_EXTRA='--save' sbatch -p mi3501x -t 02:00:00 -J m_$v -o $C/sensor_v2/logs/%x_%j.out --export=ALL train_v2.sbatch; sleep 12; done
RUN_OUT=matched RUN_VARIANTS=H RUN_SEEDS=3 RUN_TAG=matched RUN_EXTRA='--save --seed-list 2' sbatch -p mi3501x -t 01:00:00 -J m_H2 ... train_v2.sbatch   # after the OOM, see 7
```
Defaults `--train-arenas f104,g228,g203,g217 --eval-arenas g216,g231 --epochs 30`; fit = ALL rows of the train arenas
(every source, every split: 54,180), eval 9,845 (3,600 designed). Output `<out>/<tag>_<variant>_s<seed>.pt` +
`<tag>_<variants>.json`. Runtime MI350X: **Dabs 206-231 s/seed, H 287-365 s/seed** (6,330 steps); 3-seed job 10-13 min.
Ensemble read-out: `eval_v2.sbatch` -> `sensor_eval_v2.py --ck-dir $C/matched --variants $EV_VARIANTS --out matched_ensemble.json`.
nav r20 retrain: `sbatch -p mi3501x -o logs/trainr20_%j.out nav_train_r20.sbatch` (`--variants Dabs,H --seeds 3 --tag r20 --save`,
`PYTHONPATH="$N/code:$N/source/src"`), 23 min for 6 models.

---------------------------------------------------------------------------------------------------------------------

## 7. Cluster GPU gotchas (all hit during this campaign)

1. **MIOpen cache collision**: 3 of 5 concurrent jobs died with `miopenStatusInternalError` in ~12 s. Fix = per-job
   `MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}` and `MIOPEN_CUSTOM_CACHE_DIR` (same dir), `mkdir -p` it. Also stagger submits.
2. **Pinned host buffer once per variant**: `m_H` was OOM-`Killed` twice at seed 2 (host mem 99.6% on the 24-CPU MI350 node)
   because `pin_memory()` was called per seed and pinned pages were not returned. Fix in `sensor_train_v2.py:85-89,
   111-114,149`: `fit_buffers()` builds the pinned tensor ONCE per variant, passes it to every seed, plus
   `del opt, sch; gc.collect(); torch.cuda.empty_cache()`; and `--seed-list` to resume a single seed. `sensor_train.py`
   and `f104_n2_train.py` still pin per seed (fine at 31,851 rows x 6 ch = 2.3 GB; not at 54k rows x several variants).
   Do not put the corridor tensor on the GPU wholesale (night-2 merged set ~27 GB as f32).
3. Env: `module load pytorch/2.10.0` (rocm 7.1.0) + **`python3.12`** + venv `/work1/dannegrut/harry/venvs/nedm`; never clobber
   PYTHONPATH (`export PYTHONPATH="...:${PYTHONPATH:-}"`). Dataset building uses the Chrono env instead
   (`source /work1/dannegrut/harry/nrd/env.sh; nrd_pychrono; "$NRD_PYTHON"`), with `OMP/OPENBLAS/MKL_NUM_THREADS=1`.
4. Partition `mi3501x` (1x MI350X, 24 CPUs), 4 h walltime cap; pytorch/2.10.0 crashes on MI210 nodes (`mi2101x` is for
   CPU/Chrono work only). Account `-A dannegrut`.
5. AMD Chrono is deterministic per node only: paired arms of a group must share one node/job (affects label comparability
   between rigid and CRM runs too: do not pair episodes across nodes).

---------------------------------------------------------------------------------------------------------------------

## 8. Recipe: train the SAME architecture from scratch on a CRM dataset

Prereq: CRM run dirs with the 5 files of section 1, `case.split == 'train'` (rows with any other split are silently
excluded from fit), group ids stable (dev fold is `md5(group)%5==0`), dir names `f104_<...>` (and containing `_route_NN`
if you want `profile` = NN % 4 for the matched-speed metric).

Step 1 — tensors + labels (heightmap corridors, identical X to what gen_v1 scores with):
```
cd /home/harry/NeDM-traverse_mppi
python scripts/gen_build_dataset.py --runs <CRM>/data/runs --out <CRM>/station_ds_crm.npz --workers 14     # path must contain '/data/' -> source='designed'
```
(or, to match N2's TRAINING map exactly, `python scripts/f104_n2_dataset.py --root A --runs '<CRM>/data/runs/*:designed'
--out ...` which samples `A/static_map_v1`; both need only numpy.) Check the printed unsafe/fail rates and the
`labelled k/n` line (dropped runs are silent).

Step 2a — local / exact N2 code path: copy `scripts/f104_n2_deploy.py` with the output dir parametrised (DO NOT run the
original with `--tag N2`), then `python <copy> --ds <CRM>/station_ds_crm.npz --ctx none --arch gru --seeds 5 --epochs 30 --tag CRM_N2`.
Dev-fold metrics first: `python scripts/f104_n2_train.py --ds <CRM>/station_ds_crm.npz --archs gru --ctxs none --seeds 3 --out <CRM>/sweep.json`.

Step 2b — cluster (policy: real training on AMD), self-contained files `sensor_train.py` + `gen_riskmodel.py`:
`sensor_train.py` reads key `X10` and picks variant `E` = channel indices 0-4, so save the CRM npz with `X10 = X`
(5 channels is enough) or patch the one line (`d['X10']` at sensor_train.py:77); then
```
RUN_OUT=crm_final RUN_MODE=deploy RUN_VARIANTS=E RUN_SEEDS=5 RUN_EXTRA='' sbatch -p mi3501x -t 01:30:00 -J crm_E -o <logs>/%x_%j.out --export=ALL sensor_train.sbatch   # edit C= and --ds in the sbatch
```
-> `E_s{0..4}.pt`, loadable by `gen_planner.RiskModel` (norm.mu/sd are the 4 continuous channels) — same rows rule
(`fit|dev`), same hyperparameters as N2. ~2.5 min/seed per 32k rows on MI350X.
`[GOTCHA]` every trainer computes its read-out metrics BEFORE `torch.save`, and `auc()` raises on an EMPTY mask
(`y.min()` of a zero-size array): `sensor_train.py` needs >=1 row with `split=='test' & source=='designed'`
(line 82/128), `f104_n2_train.py`/`f104_n2_deploy.py` need a non-empty designed dev fold (`md5(group)%5==0`),
`sensor_train_v2.py` a non-empty eval arena. A CRM set with only `split='train'` rows crashes `sensor_train.py` after
the first seed has trained and before anything is saved — give some groups `split='test'` or guard the metric call.
`sensor_train_v2.py` also accepts an `X`-keyed 5-ch file (line 55, variant `H`) but needs an `arena` key and a NON-EMPTY
eval arena (metrics on an empty eval set raise) — only useful if CRM data spans >1 arena.

For a depth-input (nav-style) CRM model: `sensor_dataset_v2.py --grid A/sensor_v2/grids/arena_f104_50h_v1 --runs '<CRM>/data/runs/f104_*' --out ds_v2_crm.npz`
then `sensor_train_v2.py --variants Dabs,H --seeds 3 --save` with arenas set accordingly.

## 9. Recipe: score routes with the FROZEN rigid model (comparison arm)

On candidate routes (what gen_v1 does; `scripts/gen_pools.py:34-48`):
```python
import sys; sys.path.insert(0, 'scripts'); import numpy as np, gen_planner as P
P.set_map('assets/traverse/arena_f104_50h_v1')
m = P.RiskModel('artifacts/traverse/fdm_f104_50h_20260909/night2_v1/final/N2_s*.pt')     # or env GEN_MODELS; cluster: $C/gen_v1/models/N2_s*.pt
X, L = P.corridors(cands)                                     # cands: dicts with waypoints, speeds, stations
z, p = m.score(X, P.geom_ctx(start_xy, goal_xy, start_yaw, L)); pick = int(np.argmin(z))
```
On a built dataset (rigid model vs CRM labels): `d = np.load(ds, allow_pickle=True); z, p = m.score(d['X'].astype('float32'), d['ctx'][:, 17:22])`
then `f104_night_train.metrics(z[mask], d, mask)` for P/W/G AUCs, `f104_n2_final.pick_quality` or
`sensor_train_v2.route_choice` for top-pick unsafe rate. Each checkpoint carries its own normalisation — never
re-standardise with CRM statistics when scoring the frozen model.
nav_v1 model: `P.set_grid_map('<A>/sensor_v2/grids/arena_f104_50h_v1'); X12, L = P.corridors12(cands);
P.GridRiskModel('<A>/sensor_v2/matched/matched_Dabs_s*.pt').score(X12, ctx5)` (online: `nav_online.Navigator`, footprint mask on).
Cluster env var names: `GEN_SRC`, `GEN_MODELS`, `GEN_RULE` (gen_mission.sbatch:14), `NAV_MODELS` (nav_runner.py:122).

---------------------------------------------------------------------------------------------------------------------

## 10. Remaining RIGID-specific / porting checklist for this subsystem
- Label thresholds (vx -0.10/-0.30, thr 0.3, near-stop 0.3 m/s x 1 s, settle 1.0 s, tilt 30/35 deg) were tuned on rigid
  ground; keep them for comparability but measure their base rates on a CRM pilot first (section 2).
- `DT=0.05` hard-coded in 3 labellers; CRM collector must keep 20 Hz logging and the `[steering, throttle, braking]` order.
- `anchor_state.state` must stay 17-long; tire-force fields have no rigid-contact meaning under CRM.
- `fail = status != 'goal_reached'`: keep the status vocabulary; a new CRM-only terminal status (e.g. SPH domain exit,
  solver blow-up) counts as fail -> unsafe with a TERMINAL event at the high-water mark unless filtered out beforehand.
- Corridor source: static overhead depth map (training) vs TerrainMap heightmap (gen_v1 deployment) vs v2 grid (nav_v1) —
  three registrations (flat-ground v1 0.187 m/px; 511/512 offset; back-projected). Pick ONE for CRM train+deploy.
- `set_map` never marks off-arena samples invalid; `init_map` does. CRM arenas with a smaller active SPH domain than the
  80 m BMP will not be reflected in `valid` unless added.
- On-policy data mattered on rigid (within-group AUC .933 -> .969 on planner-proposed routes): a CRM set of designed
  routes only reproduces N2A, not N2.
- `f104_n2_deploy.py` hard-coded output path (overwrite hazard); `f104_night_index.py` no longer runnable; tilt dataset
  builder exists only as an inline snippet (section 2); N2T and `station_ds_all.npz` are local-only; v2 datasets are cluster-only.
