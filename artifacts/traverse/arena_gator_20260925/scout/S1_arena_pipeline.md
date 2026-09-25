# S1 scout: the arena pipeline (for task A "more arenas" and task B "Gator on f104")

Written 2026-09-25. Worktree `/home/harry/NeDM-traverse_mppi`, branch `crm_improve_v1`. Read-only: no repository file was
changed, nothing was submitted, no Chrono episode was run. Four things were run outside the repository, into
`/tmp/s1_arena_check/` only: the arena generator (f104 and g203 regenerated), one case generation (60 groups on g203),
one local OptiX map capture of g203 (a render, not a drive) and its metric grid. Read-only `ssh amd` listings and the
allocation balance were also checked. "Measured here" marks numbers computed in this session; "not verified" marks
anything taken on trust.

## 0. Bottom line

1. **f104 is one seed of a deterministic generator.** `scripts/traverse_wp7_arenas.py --seeds 104` (terrain model in
   `src/nedm/traverse/terrain.py`) reproduces `assets/traverse/arena_f104_50h_v1/arena_000.bmp` byte-for-byte
   (sha256 `5d5bc683...`), and seed 203 reproduces `arena_g203` byte-for-byte (measured here, 0.8 s for both).
   New arenas of the same kind cost seconds. Difficulty is one knob (`--difficulty`: slope cap and roughness
   amplitude); hill/crater counts, heights and sizes are fixed ranges in the code.
2. **The five gen_v1 sibling arenas (g228, g203, g217, g216, g231)** are the 5 of 40 seeds (201-240) closest to f104
   on 8 terrain statistics. They carry **rigid data and evaluation**: 9,000 designed training drives (150 start/goal
   groups x 12 routes per arena) plus 6,639 test drives, all raw run folders still local. **No soil (CRM) drive has
   ever been made on any arena other than f104.** Four more siblings (g213, g204, g234, g223) were used only as
   unseen evaluation arenas in the navigation study; 31 of the 40 seeds were never used.
3. **Only three per-arena inputs exist:** the heightmap folder (BMP + `arena_meta.json`), one overhead depth image of
   the bare arena (the risk model's only terrain input), and the start/goal case files. Everything else (route
   proposals, controller, labels, trainer, network) does not depend on the arena.
4. **The collectors take the arena from `case.json`**, relative to the cluster source tree. Rigid collectors
   additionally require the arena's BMP sha256 in an allowlist `gen_arenas.json` next to the collector. The CRM
   collector has no allowlist. So a new arena needs: its folder copied into the cluster source trees, one allowlist
   line (rigid), and cases.
5. **The corridor tensor is computed once, at dataset-build time, from one map per builder process** (`--root` /
   `--map-root`). The trainer (`ci_train.py`) never touches a map, so rows from different arenas can be concatenated
   freely as long as each arena's rows were built with that arena's map. There is **no arena column and no check
   anywhere that the map matches the case's arena** (grep: zero hits in planners and builders): planning or labelling
   with the wrong map would fail silently.
6. **The frozen f104 wiring is mostly defaults, not hard gates.** Real blockers for a new arena: the evaluation-group
   blacklist is f104-only (`ci_train.py:54`, `ga_build_mixed.py:29`), the CRM task builder and the closed-loop
   task/approach tooling have f104 paths as constants (`crm_tasks.py:17-19`, `ci_a5data.py:76-97`), and the
   feature-clustered statistics take one arena. Section 2.9 lists every place.
7. **CRM needs nothing beyond the new BMP+meta** in the CRM source tree: the soil is built from the heightmap with
   the same orientation and pixel convention as rigid ground, particle count depends on area only (4.0 M soil + 3.0 M
   boundary markers at 0.08 m for any 80 m arena). Caveats: soil friction 0.8 means a 38.7 deg friction angle and the
   f104-family arenas reach 33-36 deg (harder ones 42-44 deg), and the soil orientation was only verified on f104.
8. **The model's map input is a flat-ground lookup of the depth image, and its error grows with relief:** rmse
   0.050 m on f104, 0.062 m on g203, 0.079 m on g231 against Chrono's own terrain (measured here). This is an
   arena-dependent input distortion that task A should decide about up front (keep it for continuity, or build
   corridors from the metric grid for all arenas including f104; section 2.1).
9. **Per-arena effort:** local preparation is minutes of compute (generator < 1 s, OptiX map 22 s measured, grid
   0.4 s, 60 groups of cases 2.9 s). The cost is cluster collection: matching f104's soil data (91.5 simulated hours,
   15,235 drives) took 2 h 39 min on ~111 GPUs and ~37 billed node-hours for the whole CRM night; rigid data is
   cheap (well under an hour of wall time per arena, not measured for 24,000 drives). Allocation today: 639.0 of 1,500
   node-hours used, queue empty (checked here).
10. **One-time engineering before the first new arena: ~2-3 hours of agent work** (parameterised copies of the CRM task
   builder and the short-approach tooling, blacklist patterns for new evaluation groups, a map/case consistency
   assertion, per-arena and arena-clustered statistics). After that ~15-30 minutes of preparation per arena.

## 1. How f104 was made; the sibling arenas; controllable generation; asset inventory

### 1.1 Provenance of f104

- Generated by `scripts/traverse_wp7_arenas.py --seeds 101 102 103 104 105 106 107` (the "f101-f107 family" of the
  earlier learning-comparison study; family parameters in `assets/traverse/arena_ffamily.json`) in the
  `~/NeDM-mppi-claude` worktree, output `assets/traverse/arena_f104`.
- Copied byte-exact into `assets/traverse/arena_f104_50h_v1` by `scripts/generate_traverse_f104_collection.py:83-107`,
  which adds `source_provenance` to the metadata (source path, sha256 of BMP and meta). The two folders' BMPs are
  identical (`cmp`, measured here); the metadata differs only by that block.
- Orientation block (`rot90 0, flipud true`, rmse 0.0081 m over 400 samples) was calibrated once on `arena_v1` with
  `src/nedm/traverse/scene.py:113` (`calibrate_orientation`) and copied into every later arena (`--orientation-from`).
  Every f104-family arena carries the same block.

### 1.2 The generator (all arenas of the family)

`traverse_wp7_arenas.family_spec(seed, difficulty)` (`scripts/traverse_wp7_arenas.py:23-35`) draws from
`np.random.default_rng(seed)`:

| Parameter | Drawn range (difficulty 1.0) | f104 value |
|---|---|---|
| slope cap | uniform 25-32 deg, times difficulty, capped at 40 deg | 30.87 deg (tan 0.598) |
| roughness amplitude | uniform 0.15-0.28 m, times difficulty | 0.240 m |
| roughness correlation length | uniform 2.0-3.0 m | 2.22 m |
| hills | 5-7, sigma 4-7 m, height 2.5-4.5 m | 5 hills, 3.0-4.1 m high, sigma 4.4-6.5 m |
| craters | 5-7, sigma 2-4 m, depth 1.2-2.5 m, rim 0.25 x depth at 1.6 sigma | 5 craters, 1.4-2.0 m deep, sigma 2.8-3.6 m |

`terrain.generate_height_field` (`src/nedm/traverse/terrain.py:132-175`): feature centres at least 8 m from the edge
and pairwise at least 1.2 (sigma_i + sigma_j) apart (`:87-113`); each hill/crater amplitude is capped at
0.95 x slope cap x sigma / e^-0.5 so a lone feature respects the cap (`:149-165`); blurred Gaussian noise at the
roughness amplitude (`:167-171`); then a local slope-limited diffusion where the slope exceeds the cap (`:116-129`).
`write_arena` (`:195-228`) quantises to an 8-bit grayscale BMP, 512 x 512 px over 80 x 80 m (0.15625 m/px), height
range rounded outward to 0.05 m (f104: [-1.8, 3.9] m, step 0.02235 m per grey level), and writes
`arena_meta.json` with size, pixels, height range, seed, orientation, feature list, slope statistics and the family
parameters.

**Gotcha (terrain.py:238-257, 303-316):** `meta["features"]` are in the generation frame, mirrored in y relative to
what Chrono simulates. Always read them through `TerrainMap.features`. The case generators do (`gen_cases.py:37`,
`f104_n2_cases.py:31`), as does the clustered-CI script (`n2_cluster_ci.py:16`).

**Determinism (measured here):**
```
PYTHONPATH=src python scripts/traverse_wp7_arenas.py --seeds 104 203 --root /tmp/s1_arena_check --prefix arena_x \
    --orientation-from assets/traverse/arena_f104_50h_v1 --family-json /tmp/s1_arena_check/fam.json
# arena_x104/arena_000.bmp sha256 5d5bc683...8ed == arena_f104_50h_v1; arena_x203 46b4fb2c...231 == arena_g203; 0.8 s
```

### 1.3 Controllable difficulty

- Exposed: `--difficulty` multiplies the slope cap (max 40 deg) and the roughness amplitude. Precedent: f108-f111 at
  1.25 (`scripts/traverse_wp8_sealed_prep.sh:10`, `assets/traverse/arena_fsealed2_family.json`): slope caps
  32.6-39.4 deg, maximum slopes up to 44 deg, flat fraction as low as 0.12.
- Not exposed: hill/crater counts, heights, depths and widths are constants in `family_spec`. Changing them means a
  new wrapper that builds its own `ArenaSpec` (`terrain.py:69-84`) and calls `write_arena`; no repository edit is
  needed for that.
- Selecting "similar to f104": gen_v1 ranked 40 seeds by distance = root-mean-square of scaled differences over
  8 statistics (slope cap, roughness, correlation length, hill and crater counts, 99th-percentile slope, flat
  fraction, height range; scales in `gen_v1/arena_similarity.json`). The script that computed it is not in the
  repository; the formula reproduces all 40 stored distances to 0.0024 (measured here), so it is easy to recompute.
- Constraints for any new arena: `size_m == 80` is asserted by the rigid collectors (`gen_collect.py:248`,
  `gen_collect_ext.py:501`), the map capture (`crm_capture_map_local.py:28`) and the metric grid
  (`sensor_map_v2.py:36-37`); heights must stay inside +-10 m (`elevation_scale_m` clip in
  `fdm_diverse_data.py:39`), comfortably true for this family.

### 1.4 The gen_v1 sibling arenas

Where: `assets/traverse/arena_g{203,216,217,228,231}` locally and in the cluster source trees
`/work1/dannegrut/harry/experiments/{fdm_f104_50h_20260909/gen_v1,generalist_20260921,crm_improve_20260922}/source/assets/traverse/`
(the CRM root `crm_f104_20260916/source` holds only f104). How: `traverse_wp7_arenas.py --seeds 201..240 --prefix
arena_g`, difficulty 1.0, the 5 closest to f104 kept (`gen_v1/PLAN.md` section A). Note `assets/traverse/arena_gfamily.json`
now lists only g213/g204/g234/g223 (it was overwritten when those four were generated for the navigation study); each
arena's own `arena_meta.json` has its family block.

| arena | seed | slope cap | roughness @ corr. | hills (height) | craters (depth) | height range m | max / p99 slope deg | flat < 5 deg | distance to f104 |
|---|---|---|---|---|---|---|---|---|---|
| f104 | 104 | 30.9 | 0.240 @ 2.22 | 5 (3.0-4.1 m) | 5 (1.4-2.0 m) | [-1.80, 3.90] | 34.0 / 28.6 | 0.25 | 0 |
| g228 | 228 | 29.6 | 0.209 @ 2.50 | 5 (3.6-4.3) | 5 (1.2-2.4) | [-1.95, 4.40] | 34.4 / 28.1 | 0.32 | 0.81 |
| g203 | 203 | 30.5 | 0.195 @ 2.09 | 6 (2.5-4.3) | 5 (1.3-2.3) | [-2.00, 4.15] | 35.4 / 29.1 | 0.28 | 0.81 |
| g217 | 217 | 31.0 | 0.244 @ 2.73 | 6 (2.5-3.8) | 6 (1.2-2.5) | [-2.00, 3.80] | 33.1 / 27.9 | 0.22 | 0.85 |
| g216 | 216 | 31.4 | 0.245 @ 2.35 | 5 (2.8-4.1) | 6 (1.9-2.4) | [-2.05, 4.45] | 36.1 / 30.0 | 0.27 | 0.86 |
| g231 | 231 | 28.8 | 0.263 @ 2.15 | 5 (3.2-4.4) | 6 (1.7-2.3) | [-2.05, 4.55] | 33.7 / 27.9 | 0.21 | 0.94 |
| g213 | 213 | 29.6 | 0.242 @ 2.40 | 5 (2.9-4.2) | 6 (1.9-2.3) | [-2.30, 4.50] | 31.9 / 28.4 | 0.29 | 0.98 |
| g204 | 204 | 28.8 | 0.256 @ 2.62 | 6 (2.9-4.0) | 5 (1.2-2.4) | [-2.35, 4.10] | 33.7 / 27.1 | 0.28 | 1.01 |
| g234 | 234 | 30.6 | 0.209 @ 2.28 | 7 (2.9-4.2) | 5 (1.2-2.2) | [-2.10, 4.25] | 35.5 / 29.2 | 0.23 | 1.09 |
| g223 | 223 | 31.2 | 0.203 @ 2.68 | 6 (2.8-4.2) | 5 (1.2-2.4) | [-2.00, 4.45] | 35.9 / 29.8 | 0.32 | 1.10 |

Next unused seeds in the ranking: g201 (1.14), g227 (1.21), g218 (1.22), g232 (1.23), g224 (1.27), g208 (1.30) ...;
never generated, captured, driven or inspected.

What has been run on them (all rigid ground; no CRM anywhere but f104):

| Study | Arenas | What | Where |
|---|---|---|---|
| gen_v1 test (09-15) | f104 + 5 siblings | 200 hill/crater groups per arena, 6 arms (model, hand rule, straight line, each speed free and at 2 m/s); 6,639 drives | `fdm_f104_50h_20260909/gen_v1/test/runs` (6,639 local), `REPORT.md` |
| gen_v1 data (09-15) | 5 siblings | 150 groups x 12 designed routes per arena = 9,000 drives, all strata, 90/5/5 split; **training data, first used offline in sensor_v2** | `gen_v1/data/runs` (9,000 local, 764 MB), tensors `gen_v1/station_ds_gen_v1.npz` |
| gen_v1 missions | 5 siblings | 20 five-goal missions per arena | `gen_v1/missions_run` |
| sensor_v1 / sensor_v2 (09-15) | f104 + 5 siblings | one overhead capture per arena (`sensor_v1/maps/*`, `sensor_v2/grids/*`); two 1,200-group Chrono tests; sensor_v2 matched offline training split by whole arena (train f104+g228+g203+g217, held out g216+g231) with `scripts/sensor_train_v2.py` (`--train-arenas/--eval-arenas`, :53-63, :134); 200-group Chrono pilot on g216/g231 | `sensor_v1/`, `sensor_v2/REPORT.md` sections 5-6 |
| night-2 planner study (09-17/18) | g216, g231 | sampler / CEM arms, 961 + 941 drives | `crm_night2_v1/REPORT.md` section 5 |
| navigation (09-16) | 6 development + g213/g204/g234/g223 unseen | continuous missions, 4 each on the unseen arenas; depth rendered in-simulation (no static capture exists for these four) | `nav_v1/` |

State of the stored gen_v1 tensors: `station_ds_gen_v1.npz` has 15,639 rows (g217 2,932, g228 2,911, g216 2,909,
g203 2,898, g231 2,886, f104 1,103; source designed 9,000 / test arm 6,639) with an `arena` column, but **its
corridors come from the authored heightmap** (`gen_build_dataset.py:14-16,41`, via `gen_planner.set_map`), not the
depth image all current models use, and it has no history windows. Measured here on 30 g203 routes: depth-image vs
heightmap corridor elevation differs by rms 0.072 m, p95 0.18 m. For today's shared model the rows must be rebuilt
from the raw run folders (all present locally, including `positive_work_kj_per_interval` and `parked` in
`trajectory.npz`, which the re-anchoring builder needs).

### 1.5 Arena assets under `assets/traverse/` (103 entries)

| Group | Arenas | Size | Relevance |
|---|---|---|---|
| f104 family, difficulty 1.0 | f101-f107 (`arena_ffamily.json`), f104 = f104_50h_v1 | 80 m | f101-f107 carry older-line data (`wp7_collect_f10x`, a different collector and task format; not interchangeable) |
| harder family, difficulty 1.25 | f108-f111 (`arena_fsealed2_family.json`) | 80 m | older-line sealed arenas; max slope 42-44 deg |
| gen_v1 siblings | g203, g216, g217, g228, g231 | 80 m | rigid data + tests (1.4) |
| navigation unseen | g204, g213, g223, g234 | 80 m | evaluation only |
| early authored arenas | arena_v1 (seed 7), v2_steep, v3_rough | 80 m | orientation calibrated on arena_v1 |
| bowls, probes, smooth/mesa probes, fourway, stall demo, focus terrain | `arena_bowl_*`, `arena_fdm_probe_*`, `arena_fdm_smooth_*`, `arena_fdm_fourway_*`, `arena_fdm_stall_demo_v1`, `arena_fdm_focus_terrain_*` | 80 m | single-feature research arenas of other lines |
| diverse terrain types | `arena_fdm_diverse_v1_{train,val,test}_*` (rolling hills, ridge passes, cross slopes, valley network, rough mosaic, mixed obstacles) | **240 m** | rejected by every f104-pipeline size assert |

## 2. Every per-arena input of the pipeline

### 2.1 Maps and frames

Three rasters of the same terrain are in use:

| Raster | Produced by | Frame | Used for |
|---|---|---|---|
| BMP heightmap via `TerrainMap.from_dir` | generator | `height_grid[iy, ix]`, row 0 = -y after the orientation flip (`terrain.py:231-235,281-284`); bilinear on cell centres `fx = (x+40)/0.15625 - 0.5` (`:286-301`) | case gates (start slope, footprint), rigid native-height audit, path-follower height (BMP + 0.5 m), CRM start drop (+0.75 m), CRM launch check and sinkage/breakthrough rule, gen_v1 heightmap corridors (`gen_planner.set_map`, `gen_planner.py:35-42`, flips back to image frame) |
| Overhead depth image `static_map_v1/observation.npz['rgbd']` (4 x 512 x 512) | `crm_capture_map_local.py` (local OptiX) or `sensor_capture_map.py` (AMD, needs SLURM + allowlist, :16, :29-30); f104 only: `capture_traverse_f104_map.py:26-27` | image frame: row 0 = +y, col 0 = -x; `row = 255.5 - y/mpp`, `col = 255.5 + x/mpp`, **mpp = 2*110*tan(23.5 deg)/512 = 0.18683 m, a flat-ground mapping**; elevation = channel 3 x 10 m, invalid = -2 (`f104_n2_dataset.py:17-34`; encoding `fdm_diverse_data.py:20-44`) | **the risk model's corridor, in both worlds, for every current model**: dataset builders and planners call `f104_n2_dataset.init_map(<root>)`, which reads `<root>/static_map_v1/observation.{json,npz}` (`:18-19`) |
| Metric grid `grid.npz` | `sensor_map_v2.py --maps <dir of captures> --out <dir>` (`:95-105`) | like TerrainMap (row 0 = -y, cell centres) but measured from Chrono's surface by back-projecting each depth pixel | approach-route grade (`ga_approach.py:61,187`, `ci_a5data.py:82,202`), tracker terrain crops (`gb_crop.py`) |

Camera contract (identical in all capture scripts): 1024 x 1024, nadir at (0, 0, 110 m), hfov 47 deg, max depth 180 m,
Euclidean ray range, ambient + one directional light, flat terrain colour; the terrain is a `RigidTerrain` mesh of the
BMP, i.e. an undeformed proxy (also for CRM). The capture asserts native-height p95 < 0.05 m and that all corners are
visible (`crm_capture_map_local.py:41-44,64-66`).

**511/512 offset.** Chrono puts the 512 BMP samples on the patch edges (spacing 80/511), TerrainMap on cell centres
(80/512), so `x_terrainmap = (511/512) x_chrono`, up to 0.078 m at the edge (`sensor_v2/LOG.md:35-36`,
`sensor_v2/REPORT.md:24`, `docs/progress.md:462`). CRM's heightmap construction uses Chrono's convention too.
Measured here from each capture's `native_height_audit.npz` (4,096 points of `RigidTerrain.GetHeight`):

| arena | TerrainMap rmse / p95 | TerrainMap at 511/512 scale |
|---|---|---|
| f104 | 0.0098 / 0.0205 m | 0.0033 / 0.0084 m |
| g203 | 0.0100 / 0.0209 m | 0.0032 / 0.0080 m |
| g231 | 0.0114 / 0.0235 m | 0.0033 / 0.0081 m |

Same size on every arena; it matters only where TerrainMap is used as truth.

**The flat-ground lookup error grows with relief (measured here, same audit points):**

| arena | depth-image corridor lookup vs Chrono terrain: rmse / p95 / max |
|---|---|
| f104 | 0.050 / 0.124 / 0.255 m |
| g203 | 0.062 / 0.159 / 0.360 m |
| g231 | 0.079 / 0.202 / 0.464 m |

A point at height z is read from where flat ground at that pixel would be, displaced radially by about r z / H, so
arenas with high hills far from the centre are distorted more. Every current model was trained on this lookup, so
keeping it keeps continuity; but for "does training on more arenas generalise" it is an arena-specific input
distortion that a one-arena model can memorise. The metric grid removes it and plugs into the same sampler:
`DS.G.update(rgbd=[0,0,0,flipud(z)/10], mpp=0.15625, ctr=255.5, npx=512, elev_scale=10)` reproduces the grid sampler to
5e-15 m (`generalist_20260921/scout/terrain_crop_maps.md` section 1, not re-verified here). Switching means
rebuilding every dataset (f104 included) and retraining every model: cheap offline, but a decision for the plan.

**OptiX (local) and AMD captures are interchangeable (measured here).** f104: the local OptiX capture
(`crm_f104_v1/maps/arena_f104_50h_v1`) vs the AMD lavapipe capture (`fdm_f104_50h_20260909/static_map_v1`): identical
valid mask, max elevation difference 3.0e-5 m. g203: a fresh local capture (22 s wall, 8 s of it shader compile,
native p95 0.021 m, 70.0 % valid pixels) vs `sensor_v1/maps/arena_g203` (AMD): identical valid mask, max elevation
difference 3.6e-5 m, depth 5.3e-5 m. Metric grids from the two captures differ by at most 0.0072 m in z (a few cells
change bin). Existing captures: f104 (both backends) and the five gen_v1 siblings (AMD, `sensor_v1/maps/`); none for
g204/g213/g223/g234 or new seeds.

Map roots used by the current pipeline: `artifacts/traverse/crm_f104_v1/map_root/static_map_v1 ->
../maps/arena_f104_50h_v1` (local), `$C/static_map_v1` with `C=/work1/dannegrut/harry/experiments/crm_f104_20260916`,
and the rigid campaign root `fdm_f104_50h_20260909/static_map_v1`. A new arena needs its own root folder holding a
`static_map_v1` link to its capture.

### 2.2 Start/goal cases

| Script | Arena | Used for |
|---|---|---|
| `generate_traverse_f104_collection.py` | f104 only (copies the BMP, :83-107) | the original 1,500 `f104_v1_group_*` (first production wave) |
| `f104_n2_cases.py` | `--arena` (default f104, :27), but globs `f104_v1_group_*.json` to avoid (:33) and labels `family: f104` (:108) | the 1,200 `f104_v2_group_*` training groups (`cases_night2`), shared by rigid production_v3/v4 and the CRM collection |
| `gen_cases.py` | **any arena** (`--arena`, `--strata all|feature`, `--avoid` dirs with a 2 m margin in 4-D start+goal space, `--prefix`, `--seed`, `--wave`) | every later case set: gen_v1 data/test on all six arenas, `f104_crm_eval_group_*` (200, feature, seed 20260926104), `f104_pair_group_*` (600, all strata, seed 20260921104) |
| `f104_n2_onpolicy.py` | map-free (`--cases`, default f104 dirs :57-58) | 8 routes per group drawn from the planner's proposal distribution (`cases_night2_onpolicy`) |

Geometry (`gen_cases.py:50-101`): start on a 1 m lattice within +-34 m where slope < 7 deg, jittered +-0.35 m;
strata round-robin over the arena's features (hill/crater crossing straight through or with a lateral miss of
+-0.65 sigma, goal 10-23 m beyond the feature) plus "long traverse" and "roughness transfer" (goal >= 42 m); route
length 24-85 m; footprint check at the start (`generate_traverse_f104_collection.footprint / start_ok`); all 3
lateral offsets of the designed family must validate (curvature 0.10 /m, +-36 m). Each group gets 12 designed
routes (offsets 0/-4/+4 m x profiles constant 2/4/6 and 2-6-2 m/s). Split = sha256 of the rounded start/goal,
90/5/5 train/val/test (`:111-112`). Speed measured here: 60 groups on g203 in 2.9 s (11,311 attempts); the 200-group
test sets needed 41,730-46,341 attempts (gen_v1 logs).

**Blacklist gotcha.** Evaluation case sets get the same hash split, so most of their groups say `split: train`
(the f104 CRM evaluation set: 189 of 200, `generalist_20260921/scout/planner_closed_loop.md:30`). Exclusion from
training is by group-name pattern only: `SUITE_GROUPS` in `ci_train.py:54` and `BLACKLIST` in `ga_build_mixed.py:29`
(also imported by `ci_short_anchors.py`, asserted equal in `ci_a5data.py:138-142`), all three patterns f104-only
(`f104_crm_eval_group_*`, `f104_g1_test_group_*`, `f104_pair_group_*`). The gen_v1 sibling test groups
(`g2xx_test_group_*`) are NOT blacklisted; the 6,639 test drives inside `station_ds_gen_v1.npz` (source
`gen_test_arm`) would be fitted if that file were used as-is. New evaluation sets need new patterns in those lists
(or a builder that forces `split: test`).

### 2.3 Route candidates

- Proposal: `scripts/f104_n2_sampler.py` imports only numpy: 9 designed anchors + random 3-mode sine lateral offsets
  and 4 speed knots around the base route (`route_00` of the case, or `gen_planner.base_route(pose, goal)` for a moving
  decision). No terrain input.
- Validator: `gen_planner.CFG` (`gen_planner.py:30`: max 6 m/s, curvature 0.125 /m, arena half-extent 40 m) and
  `fdm_mppi.validate_reference`. Same for every 80 m arena.
- Scoring: corridors from the global `DS.G` map (`f104_n2_iter.py:122-123`), CEM rounds (`f104_n2_iter.plan_iter`),
  speed-continuous families (`ci_planner.py`), gradient refinement with an arena penalty 10 relu(|xy| - 37 m)^2
  (`ci_grad.py:22,653`) reading the same map through `DetMap(DS.G)` (`:562`).

### 2.4 Collection tasks and collectors

Rigid:
- `gen_collect.py` (single route, gen_v1 flavour) and `gen_collect_ext.py` (adds branch/policy modes; used by the
  generalist and soil-improvement efforts) take the arena from `case["arena"]` relative to `--source-root`
  (`gen_collect.py:245-249`, `gen_collect_ext.py:498-502`) and require `size_m == 80` and the BMP sha256 in
  `gen_arenas.json` **next to the collector script** (the local `scripts/gen_arenas.json` lists f104 + the 5 gen_v1
  siblings; the cluster copies in `gen_v1/`, `generalist_20260921/source/scripts/`, `crm_improve_20260922/source/scripts/`
  hold the same 6, checked here). The frozen loop (`traverse_fdm_rgbd_diverse_chrono.py:159-171`) and `scene.py`
  (`build_config :54-105`, `AddPatch :380-386`) read size and height range from the arena's meta. The source-manifest
  gate checks only the 9 frozen source files (`gen_collect.py:27-30,237-242`), not assets or the allowlist; the local
  copies of those 9 files equal the `crm_improve_20260922/source/source_manifest.json` hashes (checked here), so a
  fresh cluster source tree can be rsync'd from this worktree.
- `collect_traverse_f104.py` hard-gates the exact f104 BMP and height range (`:246-247`); superseded by gen_collect.
- Runners: `gen_runner.py` (via `gen_array.sbatch`, root fixed to `fdm_f104_50h_20260909/gen_v1`, :11-16) and
  `gen_runner_g.py` (via `gen_array_g.sbatch`, root from `GEN_ROOT`). Task rows `{id, case, route, shard[, mode, extra]}`;
  the rows' `arena` field is a label nobody reads (grep: no runner reads it).
- `FDM_RUNTIME_FINGERPRINT=$campaign/pilot_runtime_412394.json` (`gen_array_g.sbatch:17`) and the check that it
  contains `/vehicle/hmmwv/` (`gen_collect.py:274`) are vehicle-, not arena-, specific.

CRM (soil):
- `crm_collect.py` / `crm_collect_ext.py`: arena from `case["arena"]` under `--source-root` (`crm_collect.py:172`,
  `crm_collect_ext.py:109-125`), **no allowlist**; imports `gen_collect` (StopPolicy, hashing) and the frozen
  `make_driver`/`read_route` from the source tree.
- `crm_worker.py`: `source` = `$CRM_ROOT/source` (fixed), case/route paths relative to `CRM_ROOT` (absolute paths also
  work, pathlib join), `CRM_CONFIG` relative to `CRM_ROOT`, collector overridable by `CRM_COLLECTOR`.
- `crm_collect.sbatch:11` and `crm_launch.sh:5` fix `CRM_ROOT=/work1/dannegrut/harry/experiments/crm_f104_20260916`,
  whose `source/assets/traverse/` holds **only** `arena_f104_50h_v1` (checked here).
- `crm_tasks.py:17-19` builds tasks for exactly `f104_v2_group_{0000..1199}`, 12 designed + 8 on-policy routes per
  group, tiered by a per-group shuffle, ids equal to the rigid twin ids, `episode_seed = md5(id)[:8]`.

### 2.5 Labelling and dataset builders (chain behind the current shared model)

The final shared model (`crm_improve_20260922/deploy_v1/deploy_a1_haux_gru_s*.pt`) was trained on
`mixed_reanchor_plus_branch_both.npz` + `short_anchor.npz` + `anchor_k60.npz` (217,814 fit rows, rigid 109,492 /
soil 108,322; `deploy_v1/deploy_a1_haux_gru.json`). Each stage and its arena coupling:

| Stage | Script | Arena coupling |
|---|---|---|
| labels + standing-start corridor | `f104_n2_dataset.py --root <map root> --runs <glob>:<source>` | one map per call (`init_map`, :17-24); labels arena-free (:91-123); ctx = anchor state (17) + goal dx, dy, dist, start yaw, route length; no arena column |
| energy/time twin tensors | `n2_energy_targets.py` (twin_*.npz) | arena-free (imports `project`) |
| moving anchors (k = 0 and every 2 s) | `n2_reanchor_dataset.py --root --ids --runs --out` | one map per call; needs `positive_work_kj_per_interval` |
| history windows + domain + privileged | `ga_build_mixed.py --rigid --crm --rigid-runs --crm-runs --out` | defaults are f104 files and run roots (:31-34); blacklist f104 (:29); needs both a rigid and a soil file (asserts equal key sets) |
| short anchors 0.5/1/1.5 s | `ci_short_anchors.py --ref <mixed npz> --root <map root> --rigid-runs --crm-runs` | defaults f104 (:39-40); one map per call; splits and groups asserted equal to the reference file |
| branch continuations | `ga_branch_anchors.py` (f104 run roots, routes, patterns :54-60), `ga_branch_dataset.py --map-root` (default f104, :293) | optional (gave no gain) |
| tracker dynamics cache | `gb_build_cache.py` (:38-45 f104 roots, `f104_v2_group_` split rule) | tracker only, not needed for the planner |
| gen_v1-style arena tensors | `gen_build_dataset.py` | arena from the run-name prefix (:35, :41), **heightmap corridor** (:14-16) |

Raw files a builder needs per run: `trajectory.npz`, `outcome.json`, `command_reference.npz`, `case.json`,
`anchor_state.npz`, plus `crm_extra.npz` for soil (1.8 GB for the 15,235 f104 soil runs locally; 764 MB for the 9,000
gen_v1 rigid runs). Build speed: `ci_short_anchors.py` made 89,998 rows from 30,048 episodes in 83 s.

### 2.6 Training (`ci_train.py`)

- **Rows from different arenas can be mixed.** `--ds` is repeatable; files are concatenated in memory, missing keys
  dropped and listed, ids must be unique (`:256-285`). X must be `(5, 96, 32)` (`:324`): the corridor is whatever the
  builder sampled; `ci_train` never re-samples a map. So "the corridor per row from its own map" holds exactly when
  each arena's file was built with that arena's `--root`.
- Splits: from each row's `split`; in holdout mode a dev fold is md5(group) % 5 == 0 (`ga_train.py:103`);
  `--split-eval val|test` evaluates rows with that split from **all** files. There is no arena column or arena
  filter: to hold out an arena offline, leave its file out of `--ds` or relabel its split (else its `train` rows would
  be fitted). Precedent for arena-level splits: `sensor_train_v2.py:53-63,134` (`--train-arenas/--eval-arenas` on an
  `arena` column).
- Data-quantity control: `--data-frac-crm/--data-frac-rigid` and `--subsample` keep a seeded fraction of training
  **groups** pooled across files, not per arena; matched-size "1 arena vs 3 arenas" comparisons need per-arena
  subsetting outside the trainer.
- Domain is rigid/soil only (from `domain` or the `@crm/@rigid` id suffix, `:282-284`); the auxiliary head predicts
  the world, not the arena.
- Cost: ~800 s per seed on an MI350X for 218k rows x 30 epochs (`deploy_a1_haux_gru.json`); ~3x rows -> ~40 min per
  seed, inside the 4 h partition cap.

### 2.7 Planners

`ga_planner.py`, `ci_planner.py` (wraps it, `--family free|cont|cont_head`), `ci_grad.py`: all take `--cases <dir>`
and `--map-root <root>` and call `DS.init_map` once per process (`ga_planner.py:467`, `ci_planner.py:628-629`,
`ci_grad.py:557`), so **one arena per invocation**. `--arena-tag` (default f104, `ga_planner.py:435`, `ci_grad.py:544`)
only labels rigid task rows. Reference-pick defaults point at f104 (`ga_planner.py:472`, crm world only and only if the
folder exists). Decision states come from `--poses` (recorded pass-1 runs). **No check that the map root's BMP
(`observation.json: arena_bmp_sha256`) matches `case["arena"]`** (grep: no such comparison in any planner or builder).

### 2.8 Closed-loop evaluation and statistics (the recipe behind 97.5 % / 100 %)

From `crm_improve_20260922/s2/run_s2_new.sh`, `s4/run_grad_picks.sh`, `s2/analyze_s2.sh`:
1. Suite: 800 f104 groups (600 `f104_pair_group` + 200 `f104_crm_eval_group`) assembled by `ga_suite.py` (constants
   :36-55, rigid rows labelled `arena='f104'` :284); cluster copies of the cases at `$C/generalist/suite/cases` (soil)
   and `generalist_20260921/a5/cases` (rigid).
2. Approach routes: `ci_a5data.py --stage approach-routes` (straight line or route_00 at 3 m/s, whichever is flatter
   over the first 12 m on the metric grid; `--grid`, `--arena` exist but set names, twin split file, cluster case
   folders and `ARENA_TAG='f104'` are constants, :76-97; `detect_set` rejects group names that are neither f104 suite
   nor f104 twin, :149-154).
3. Pass 1: drive the approach for 0.5 s in both worlds (`--stage pass1-tasks`), then `--stage analyze` -> poses and
   40-frame history windows.
4. Picks: `ci_planner.py --family free --cases <suite> --map-root <f104 root> --models <deploy glob> --world W --arms B
   --poses <poses>`; gradient refinement `ci_grad.py` with the same inputs.
5. Pass 2 rows: `ga_a5_pass2_tasks.py` (branch at frame 10 onto the pick; rigid rows get `arena='f104'` hard-coded at
   :42, a label only).
6. Drives: soil via `crm_collect.sbatch` with `CRM_COLLECTOR=generalist_20260921/source/scripts/crm_collect_ext.py`,
   rigid via `gen_array_g.sbatch` with `GEN_ROOT=crm_improve_20260922`.
7. Statistics: `ga_a3_index.py` (arena-free), `ga_analyze.py` (paired group bootstrap, McNemar, time ratio; optional
   feature-clustered interval through `n2_cluster_ci.py`, which clusters by nearest feature of ONE `--arena`,
   :13-19; default f104 at `ga_analyze.py:196`). Pooling several arenas in `ga_analyze` works (group ids are unique)
   but arena-level clustering and per-arena tables exist only in `gen_analyze_test.py` (arena list hard-coded :15-16).

### 2.9 Every place f104 or its map is wired in

"Blocks" = a new arena fails or is silently wrong without a change; "default" = overridable by a CLI flag;
"label" = cosmetic; "self-test" = test code only.

| File:line | What | Kind |
|---|---|---|
| `ci_train.py:54` | evaluation-group blacklist, f104 patterns only | blocks (leak risk) |
| `ga_build_mixed.py:29`; `ci_a5data.py:93` (asserted equal :138-142) | same blacklist | blocks (leak risk) |
| `ga_build_mixed.py:31-34` | default reanchor files and f104 run roots | default |
| `ci_short_anchors.py:39-40` | default reference file and f104 map root | default |
| `crm_tasks.py:17-19` | `f104_v2_group_*` x 1,200, `cases/night2...` paths | blocks (needs a copy) |
| `crm_collect.sbatch:11`, `crm_launch.sh:5`, `crm_dataset.sbatch:11-14`, `crm_train.sbatch:14` | CRM root; `static_map_v1` at that root = f104 | blocks for datasets (map), default for drives (arena comes from the case) |
| `$C/source/assets/traverse/` (cluster) | only f104 present | blocks CRM drives until the arena folder is copied |
| `scripts/gen_arenas.json` + cluster copies next to each rigid collector | 6-arena allowlist | blocks rigid drives |
| `gen_collect.py:25` | `F104_BMP_SHA256` constant, unused | none |
| `collect_traverse_f104.py:246-247`, `capture_traverse_f104_map.py:26-27`, `generate_traverse_f104_collection.py:83-85` | exact-f104 gates | f104-only tools (use gen_collect / crm_capture_map_local / gen_cases) |
| `sensor_capture_map.py:16,29-30` | SLURM-only + allowlist | alternative: local `crm_capture_map_local.py` |
| `crm_capture_map_local.py:13,28,41` | repo default, 80 m assert, audit RNG seed 104 | harmless |
| `f104_n2_dataset.py:18-19` | map must live at `<root>/static_map_v1/` | convention |
| `f104_n2_cases.py:27-29,33,108`; `f104_n2_onpolicy.py:16,57-58` | f104 defaults, f104_v1 avoid glob, family label | default (prefer `gen_cases.py`) |
| `gen_build_dataset.py:41` | 'f104' prefix -> `arena_f104_50h_v1`, else `arena_<prefix>` | convention |
| `ga_branch_anchors.py:54-60`; `ga_branch_dataset.py:293-294` | f104 run roots / map root / schema reference | default / f104-only (branch rows optional) |
| `gb_build_cache.py:38-45`; `gb_crop.py:179-180` | tracker cache and crop defaults | tracker only |
| `ga_planner.py:435,472`; `ci_grad.py:544` | arena tag, reference picks | label / default |
| `ci_planner.py:52,669,688,816,826`; `ci_grad.py:681,725,852`; `f104_n2_grad.py:40,438-458`; `planner_arms.py:87,95-96` | f104 map root / cases in docs and self-tests | self-test / default |
| `ga_approach.py:61-64,547` | metric grid, arena, stand-in runs, CRM root | default |
| `ci_a5data.py:76-97,324,746,1058` | suite/twin case dirs, twin split file (asserts 1,200 groups 1089/56/55 :167), cluster case paths, `ARENA_TAG='f104'` | blocks (needs a copy for other case sets) |
| `ga_a5_pass2_tasks.py:42` | rigid `arena='f104'` | label |
| `ga_suite.py:36-55,284` | f104 suite constants | f104 suite builder |
| `ga_analyze.py:196`; `n2_cluster_ci.py:13` | cluster-CI arena default | default (single arena) |
| `gen_analyze_test.py:15-16` | arena list | per-study constant |
| `crm_collect.py:218`, `crm_collect_ext.py:181,612-636` | `"crm_f104"` telemetry label and schema names | label |
| `gen_runner.py` / `gen_array.sbatch:11-16` | gen_v1 root | default (use `gen_array_g.sbatch` + `GEN_ROOT`) |

## 3. CRM soil on a new arena

### 3.1 How the soil is built

`crm_collect.build_crm` (`scripts/crm_collect.py:69-140`): `veh.CRMTerrain(system, spacing)`; soil = density 1,700
kg/m3, Young 1 MPa, Poisson 0.3, mu_I0 0.04, friction 0.8, grain 5 mm, cohesion 5 kPa; SPH = RK2, d0 1.2, free-surface
0.8, artificial viscosity 0.5, PPST shifting, Adami walls; the four HMMWV rigid-mesh tyres
(`hmmwv/hmmwv_tire_coarse_closed.obj`) are FSI bodies; active domain 2 x 2 x 1 m per body;
`terrain.Construct(<arena BMP>, size, size, (height_min, height_max), depth 0.24, True, origin, sides)` with side walls and
floor (`:128-130`). The run loop (`run`, `:156-...`): `tire_model=RIGID_MESH`, no chassis-soil contact, drop from
TerrainMap height + 0.75 m, path follower on BMP + 0.5 m, 0.8 s settle, launch check (speed <= 1 m/s, roll/pitch
<= 25 deg, yaw <= 10 deg, xy <= 1 m, chassis 0-1.2 m above the BMP; `:412-430`), extra terminal status
`soil_breakthrough_terminated` when a wheel sinks more than depth + 0.06 m below the BMP for 0.25 s.

Configuration: the file used for all soil drives is `$C/configs/crm_main.json` on the cluster = local
`artifacts/traverse/crm_f104_v1/configs/crm_main.json` (spacing 0.08 m, depth 0.24 m, step 1 ms, active domain
[2, 2, 1] m, 4 MBS threads). **There is no `configs/crm_main.json` in the repository's `configs/`** (only
`hmmwv_crm_eval.json` and a transformer config); the defaults inside `crm_collect.py:36-46` use a 0.5 ms step, and the
worker forwards `--crm-config` only when the collector's file name contains `crm_collect` (`crm_worker.py:91-99`).

### 3.2 What a different arena needs

Only the BMP + `arena_meta.json`, present under `$CRM_ROOT/source/assets/traverse/<arena>/`, and cases whose
`arena` field points there. Height range, size and orientation come from the metadata; the soil surface follows the
BMP exactly like `RigidTerrain` (node-on-edge, same y flip; `crm_f104_v1/scout/map_capture_corridor.md` section 2 cites
`ChFsiProblemSPH.cpp:810-811,939-951`). Things to watch:

- Particle count depends on area and depth, not relief: 4,008,004 soil particles + 3,006,003 boundary markers for f104
  at 0.08 m (`crm_f104_v1/smoke/full_0.08/smoke_report.json`), so throughput per GPU should be the same on any 80 m
  arena (real-time factor ~0.3 MI210, ~0.5 MI350X at 1 ms; drives are bit-identical across AMD GPU types).
- Orientation was checked on the f104 lattice only (rmse = lattice quantisation, y-mirrored 1.23 m).
  `scripts/crm_smoke.py --arena <dir> --check-surface --spacing 0.16` repeats that check (~1 min on the local GPU;
  it is a Chrono run, so not done here).
- Slope vs soil strength (reasoning, not tested): friction 0.8 = friction angle 38.7 deg. f104-family arenas at
  difficulty 1.0 reach 32-36 deg maximum slope (f104 34.0), so the margin is 3-7 deg plus 5 kPa cohesion; harder
  arenas (difficulty 1.25: 42-44 deg) exceed the friction angle and may slump or behave differently.
- The static depth map stays a rigid-mesh render of the undeformed surface (OptiX cannot draw SPH particles from
  Python); no per-episode rendering is involved.
- The breakthrough and launch checks use TerrainMap heights (the 511/512 scale, <= 0.078 m horizontally): same small
  bias on every arena.

## 4. Effort per new arena and the step list

### 4.1 Effort

| Step | Where | Cost |
|---|---|---|
| generate BMP + meta | local | < 1 s (measured) |
| similarity/difficulty check | local | seconds (formula verified) |
| overhead depth map (OptiX) | local RTX 5090, `/usr/bin/python3.12` with `PYTHONPATH=/home/harry/chrono/build/bin` | 22 s (measured); AMD alternative 33 s per arena (earlier array job, not re-measured) |
| metric grid | local | 0.4 s (measured) |
| CRM orientation smoke | local GPU | ~1 min (not run) |
| training cases 1,200 groups + evaluation suite + on-policy routes | local | minutes (60 groups took 2.9 s; 1,200 not measured) |
| rigid collection, 1,200 groups x 20 routes | cluster CPU partitions | well under 1 h wall (gen_v1's 15,639 drives ran 03:55-04:43 across partitions; not measured for 24,000) |
| soil collection matching f104 (15,235 drives, 91.5 h) | ~111 AMD GPUs | 2 h 39 min wall; the whole CRM night cost ~37 billed node-hours (`crm_f104_v1/REPORT.md:17`); soil drives dominate at roughly 2 billed node-hours per 1,000 drives (39 for ~20,000 in `crm_improve`) |
| sync + label + tensors | local | < 30 min (builders are minutes; transfer ~2 GB per arena per world) |
| training ensemble | MI350X | ~15-40 min per seed |
| closed-loop evaluation per unseen arena | cluster | soil pass 1 + pass 2 for ~200-400 groups x arms: a few hundred to ~1,600 soil drives (~1-4 node-hours); rigid minutes |

One-time engineering before the first new arena (new scripts only, following the earlier "no edits to specialist
scripts" convention): a parameterised soil task builder (copy of `crm_tasks.py`); a short-approach task/analysis tool
that accepts any case set and cluster paths (copy of the `ci_a5data.py` stages); blacklist patterns for new evaluation
groups (`ci_train.py:54`, `ga_build_mixed.py:29`, `ci_a5data.py:93`) or a builder that forces `split: test`; an
assertion that `observation.json: arena_bmp_sha256` equals the case arena's BMP sha256 in every planner/builder call;
an `arena` column in the dataset files and a per-arena subset tool for matched-size training; per-arena and
arena-clustered statistics (port from `gen_analyze_test.py`). Estimate 2-3 hours of agent work, then ~15-30 minutes
of preparation per arena.

Budget today: 639.0 of 1,500 node-hours used (42.6 %), 0 jobs queued (`slurm_balance2.py`, `squeue`, checked here).
Soil data equal to f104's on two new arenas would take roughly 60-75 node-hours and 5-6 hours of GPU-pool wall time
if run one after the other (the pool, ~36 simulated hours per wall hour, is the bottleneck).

### 4.2 Step list for one new arena (example `arena_g201`; `R=/home/harry/NeDM-traverse_mppi`,
`PY=/home/harry/miniconda3/envs/nedm/bin/python`, `export PYTHONPATH=src:scripts`)

1. **Generate.** `$PY scripts/traverse_wp7_arenas.py --seeds 201 --root assets/traverse --prefix arena_g
   --orientation-from assets/traverse/arena_f104_50h_v1 --family-json <campaign>/family_g201.json`
   (keep `--family-json` off the shared `arena_gfamily.json`). The arena must sit under the repository root because
   `gen_cases.py:126` stores `arena.relative_to(ROOT)`. Regenerate once more into /tmp and compare sha256
   (determinism), and compute its distance to f104 with the section 1.3 formula.
2. **Map.** `PYTHONPATH=/home/harry/chrono/build/bin /usr/bin/python3.12 scripts/crm_capture_map_local.py --arena
   assets/traverse/arena_g201 --out <campaign>/maps/arena_g201` (asserts native p95 < 0.05 m, corners visible); then
   `mkdir -p <campaign>/map_roots/g201 && ln -s ../../maps/arena_g201 <campaign>/map_roots/g201/static_map_v1`;
   `$PY scripts/sensor_map_v2.py --maps <campaign>/maps --out <campaign>/grids` (metric grid for approach grades).
3. **Soil orientation smoke** (Chrono, local GPU): `crm_smoke.py --arena assets/traverse/arena_g201 --check-surface
   --spacing 0.16`; expect surface rmse at lattice quantisation, flipped far worse.
4. **Cases.** Training pool: `$PY scripts/gen_cases.py --arena assets/traverse/arena_g201 --groups 1200 --strata all
   --prefix g201_v2_group --seed <new> --wave <campaign>_train --out <campaign>/cases_train_g201`.
   Evaluation suite(s): `gen_cases.py ... --prefix g201_eval_group --strata feature|all --avoid
   <campaign>/cases_train_g201/cases --seed <other>`. On-policy routes: `$PY scripts/f104_n2_onpolicy.py --cases
   <campaign>/cases_train_g201/cases --out <campaign>/cases_train_g201_onpolicy --n 8`.
   Add `g201_eval_group_*` to the blacklists (or force `split: test`).
5. **Tasks.** Soil: a copy of `crm_tasks.py` with the new prefix and case/route paths (12 designed + 8 on-policy per
   group, tiered, `episode_seed = md5(id)`). Rigid: the same ids, rows `{id, case, route, shard=md5(group)%K,
   mode: native}` for `gen_runner_g.py`; identical ids give rigid/soil twins as on f104.
6. **Stage on the cluster** (G = a new root, e.g. `/work1/dannegrut/harry/experiments/arena_gator_20260925`):
   rsync `src/ scripts/ assets/traverse/arena_g201` into `$G/source/` plus a copy of
   `crm_improve_20260922/source/source_manifest.json` (the 9 frozen files match it); append `"arena_g201": <sha256>` to
   `$G/source/scripts/gen_arenas.json`; copy `arena_g201` into `$C/source/assets/traverse/`
   (`C=/work1/dannegrut/harry/experiments/crm_f104_20260916`, the soil worker's fixed source tree); rsync cases and
   routes under `$G` and under `$C/<campaign>/`.
7. **Collect.** Rigid: `sbatch -p <cpu partition> -c <cpus> --array=0-(K-1) --export=ALL,GEN_ROOT=$G,
   GEN_TASKS=$G/tasks/tasks_rigid_g201.json,GEN_OUT=$G/rigid_g201,GEN_COLLECTOR=$G/source/scripts/gen_collect_ext.py
   $G/source/scripts/gen_array_g.sbatch`. Soil: the `crm_launch.sh` pattern (one sbatch per GPU partition,
   `CRM_TASKS=...,CRM_OUT=$C/<campaign>/collect_g201,CRM_CONFIG=configs/crm_main.json`); max 50 queued array tasks
   per user.
8. **QA and sync.** `crm_qa.py` for validated hours; count `collection_failure.json`; rsync only `trajectory.npz,
   outcome.json, command_reference.npz, case.json, anchor_state.npz, crm_extra.npz, episode_complete.json`.
9. **Tensors per arena and world** (map root = the arena's):
   `f104_n2_dataset.py --root <campaign>/map_roots/g201 --runs "<runs>/*_route_*:designed" "<runs>/*_op_*:on_policy"
   --out station_g201_<world>.npz` -> `n2_reanchor_dataset.py --root <same> --ids station_g201_<world>.npz --runs
   <runs> --out reanchor_g201_<world>.npz` -> `ga_build_mixed.py --rigid reanchor_g201_rigid.npz --crm
   reanchor_g201_crm.npz --rigid-runs <rigid runs> --crm-runs <soil runs> --out mixed_g201.npz` ->
   `ci_short_anchors.py --ref mixed_g201.npz --root <same> --rigid-runs ... --crm-runs ... --out short_anchor_g201.npz`
   (and `--ks 60` for the k60 file). For a rigid-only arena `ga_build_mixed.py` needs a small variant (it requires both
   worlds).
10. **Train.** `ci_train.py --ds <f104 files> <g201 files> ... --arch gru --cond hist_aux --mode deploy` (the deploy
    arguments in `deploy_v1/deploy_a1_haux_gru.json`); for matched-size comparisons subset per arena first.
11. **Evaluate on an unseen arena.** Its own map root, grid and suite; approach routes and pass-1 tasks with the
    generalised short-approach tool; pass-1 drives both worlds; decision states; `ci_planner.py` and `ci_grad.py` with
    `--map-root <its root> --cases <its suite> --poses <its poses>`; `ga_a5_pass2_tasks.py`; drives;
    `ga_analyze.py --cluster-ci --cases <suite> --arena assets/traverse/<arena>` per arena, plus a pooled table with
    arena-level resampling.

### 4.3 Decisions task A should make up front

- Which arenas: the 5 gen_v1 siblings already have rigid data (150 groups each, 1/8 of f104's 1,200) and have been
  inspected in several studies; g213/g204/g234/g223 have been used only as unseen evaluation arenas; g201, g227, g218,
  g232 ... have never been used. A clean "unseen arena" test needs arenas that no model or person tuned on.
- Equal data: f104's soil data is 1,200 groups / 15,235 drives / 91.5 h; rigid twins 15,024 drives. "More arenas"
  helps only if compared at matched total rows (or matched per-arena groups), otherwise it is also "more data".
- Map geometry: keep the flat-ground depth lookup (continuity with every result so far; arena-dependent distortion
  0.05-0.08 m rmse) or move all arenas, f104 included, to the metric grid (section 2.1).
- Statistics: with 2-3 training arenas and 1-2 unseen arenas, the arena is the unit that generalises; report per
  arena and resample arenas/features, not only groups.

## 5. Notes for task B seen during this scout (vehicle coupling)

The vehicle, not the arena, is hard-wired in: `src/nedm/hmmwv_data.py` (`create_hmmwv`, `capture_row`, `WHEEL_SPECS`;
one of the 9 hash-checked frozen source files), `scene.build_config` vehicle block, the frozen runner's `scene.hmmwv`
hooks (`gen_collect.py:170-171,199`), the runtime-fingerprint check for `/vehicle/hmmwv/` (`gen_collect.py:274`), the
soil tyre mesh `hmmwv/hmmwv_tire_coarse_closed.obj` (`crm_collect.py:45`, `crm_main.json`), and the 17-column state
layout (`STATE_FIELD_PRESETS["tire_normal_force_omega_pt"]`) that the history encoder reads (12 state columns incl. four
spindle speeds and engine speed). The f104 arena inputs (BMP, depth map, cases, routes) carry over to another vehicle
unchanged; the route validator's limits (6 m/s, 0.125 /m curvature) are HMMWV-tuned values in `gen_planner.CFG`.

## 6. Not verified here

- How the 40 seeds 201-240 were generated and ranked (the ranking script is not in the repository; its output and
  formula are consistent).
- Rigid collection wall time for 24,000 drives on one arena; AMD capture time for a new arena; case generation time
  for 1,200 groups.
- The metric-grid substitution equivalence (5e-15 m) and the soil heightmap convention lines in `ChFsiProblemSPH.cpp`
  (both quoted from earlier scout reports).
- Soil behaviour on slopes near or above the 38.7 deg friction angle.
- That `crm_smoke.py --check-surface` passes on a non-f104 arena (not run: it is a Chrono simulation).
