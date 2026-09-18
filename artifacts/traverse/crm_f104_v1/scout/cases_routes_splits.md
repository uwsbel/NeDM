# Scout report: case generation, routes, speeds, splits (f104 rigid pipeline -> CRM port)

Scope: read-only reconnaissance, 2026-09-16. Repo `/home/harry/NeDM-traverse_mppi` (branch `traverse_mppi`).
Campaign artifacts: `A = artifacts/traverse/fdm_f104_50h_20260909` (local), `C = /work1/dannegrut/harry/experiments/fdm_f104_50h_20260909` (cluster).
Everything below was read from source; items marked VERIFIED were additionally checked numerically tonight (generator dry run into a temp dir under /tmp, since deleted; margin arithmetic on the local case files).
Side effects of this scout: this file only. (`git status` also shows `.gitignore` modified — a `/chrono` line, mtime 23:20:42 — that was NOT done by this scout.)

Python: the generators import `nedm.traverse.fdm_diverse_planner`, which imports torch at module top. Base conda python has no torch.
Use `/home/harry/miniconda3/envs/nedm/bin/python` (numpy 2.4.6, torch 2.12). Run from the repo root (several scripts use relative `src`, `scripts`, `artifacts/...` paths).

---------------------------------------------------------------------------------------------------------------

## 0. Bottom line for tonight

1. The case/route layer is pure geometry on the BMP (`TerrainMap`), no Chrono, no physics outcome. Nothing in it is rigid-specific except the *assumptions* listed in section 8. All generators run unchanged for CRM.
2. Cheapest and cleanest training pool: **reuse the frozen night-2 pool verbatim** — `A/cases_night2/cases` (1,200 groups `f104_v2_group_0000..1199`, 12 designed routes each) plus `A/cases_night2_onpolicy/routes/<group>/op_00..07.json` (8 planner-proposal routes each). Same groups have rigid outcomes (paired rigid-vs-CRM labels for free) and every existing held-out set keeps its verified margin against them. Take an index prefix (multiple of 36 groups) to fit the CRM compute budget.
3. Held-out sets that already exist and are VERIFIED disjoint (section 5): fresh184 (>= 6.01 m), ext523 (>= 4.00 m), haz300 (>= 3.37 m), and three 200-group hill/crater pools at >= 2.0 m (`f104_g1_test_group`, `f104_s1_group`, `f104_s2_group`). For brand-new groups use `scripts/gen_cases.py --avoid ... --margin-m 2.0` (command in section 10; dry-run VERIFIED: 24 groups in 3.5 s).
4. Planner evaluation on held-out single start/goals: `scripts/gen_pools.py` (heightmap corridors, N2-format model) or `scripts/sensor_pools_v2.py` (world-grid corridors, matched_H / matched_Dabs format). Both build a 256-candidate proposal pool + a 256-candidate fixed-2 m/s pool per group with md5-seeded RNG and write one route JSON per distinct pick.
5. Route files have a VARIABLE number of waypoints (0.5 m spacing, `max(33, ceil(L/0.5)+1)`, 58-171 points for 28-85 m). "96 stations" is only the model-input resampling (`scripts/f104_n2_dataset.py:12`, `N_STATION, N_LATERAL, HALF_WIDTH_M = 96, 32, 6.0`).

---------------------------------------------------------------------------------------------------------------

## 1. File formats (contracts)

### 1.1 Route JSON (every route everywhere: designed, on-policy, planner pick)
```
{"waypoints": [[x,y],...]   float64, metres, world frame (x right, y up; arena is [-40,40]^2), N x 2
 "speeds":    [v,...]       m/s, N, commanded speed at each waypoint (PID speed reference, spatial not temporal)
 "stations":  [s,...]       metres, N, cumulative arc length, stations[0]=0, must equal cumsum(|diff(waypoints)|) to 1e-4
 "headings":  [psi,...]     rad, N, atan2(gradient(y), gradient(x)); must agree with tangents within 0.15 rad
 "meta":      {...}}        free-form (see below)
```
Enforced by:
- `src/nedm/traverse/fdm_diverse_planner.py:16-38 check_reference_contract` (>= 3 points, finite, no zero-length segment, station consistency atol 1e-4, no segment-to-segment turn > 45 deg, heading error <= 0.15 rad).
- `scripts/traverse_fdm_rgbd_diverse_chrono.py:79-95 read_route` (collector side: N >= 3, strictly increasing station, speeds in [0, 10], stations match geometry atol 1e-4; accepts a `{"route": {...}}` wrapper).
- `scripts/traverse_fdm_rgbd_diverse_chrono.py:193-194`: `waypoints[0]` within 0.25 m of `layout.start_xy` AND `waypoints[-1]` within 0.25 m of `goal_xy`, else ValueError. Every generator below pins both ends.
- Driver construction (`same file :98-112 make_driver`): waypoints decimated to >= 2 m station spacing (last point always kept), Bezier z = `tmap.height(x,y)+0.5`, `ChPathFollowerDriver` target speed initialised to `speeds[0]`, look-ahead 5 m, steering gains (0.8,0,0), speed gains (0.6,0.05,0). RIGID FLAG: path z comes from the BMP ("truth" height).

`meta` for designed routes: `candidate="rgbd_geometric_family", lateral_offset_m, cruise_speed_mps, fdm_station=0.0, scene_id, collection_stratum, feature_index, feature_geometry_only{kind,x_m,y_m,sigma_m,amplitude_m}, feature_lateral_miss_m, pair_group_id, route_seed, minimum_recovery_tail_s=8.0, recovery_label_scope, speed_profile_id, route_index`.
`meta` for on-policy routes: `candidate="n2_wide", max_lateral_m, mean_speed_mps, scene_id, route_index, wave="night2_A2_on_policy", proposal="f104_n2_sampler.sample_one", route_seed`.
`meta` for planner picks: `candidate="gen_<arm>" | "v2_<arm>" | "sensor_v1_<arm>", scene_id, pool ("proposal"|"fixed2"), cand_index`.
GOTCHA: `validate_reference` reads `meta.fdm_station` if present (`fdm_mppi.py:124-127`) to trim the executed prefix; designed routes carry `fdm_station: 0.0`. Sampler outputs have no such key and are trimmed at the waypoint nearest the anchor pose instead.

### 1.2 Case JSON (one per start/goal group) — sample (VERIFIED output of `gen_cases.py`, `settle_reference` elided)
```json
{
 "id": "f104_crm_dry_group_0000",
 "split": "train",
 "arena": "assets/traverse/arena_f104_50h_v1",
 "family": "f104_terrain_family",
 "evaluation_stratum": "hill_entry_cross_exit",
 "layout": {"episode_id": "f104_crm_dry_group_0000", "seed": 20260920104, "assets": [],
            "house_xy": [-33.56962337248009, 11.148938911978654], "house_yaw": 1.9746414491006974,
            "start_xy": [-19.26122743361476, -22.33406712812794], "start_yaw": 1.9746414491006974},
 "goal_xy": [-33.56962337248009, 11.148938911978654],
 "goal_radius_m": 2.5,
 "horizon_s": 120.0,
 "arena_half_extent_m": 40.0,
 "settle_reference": {"waypoints": "...", "speeds": "...", "stations": "...", "headings": "...", "meta": "... (= route_00 inlined)"},
 "family_parameters": {"speeds": [2.0, 4.0, 6.0], "offsets": [0.0, -4.0, 4.0],
                       "speed_profiles": ["constant_2", "constant_4", "constant_6", "smooth_2_6_2"]},
 "collection_contract": {"maximum_duration_s": 120.0, "record_sustained_failure_until_horizon": false,
                         "minimum_recovery_tail_s": 8.0, "stop_policy_owned_by_runner": true,
                         "count_actual_completed_task_seconds_only": true, "exclude_settle_and_failed_jobs_from_hours": true},
 "geometry_initialization": {"height_range_m": 0.169, "max_grade_deg": 6.56, "fitted_grade_deg": 1.27,
                             "plane_residual_max_m": 0.095, "sample_count": 91, "rectangle_m": [6, 3],
                             "method": "Bilinear quantized BMP; native settling pending"},
 "wave": "crm_dry",
 "role": "Terrain-only fixed-map data enrichment; no physics outcome filtered",
 "split_scope": "Deterministic start/goal group split 90/5/5; all 12 paired references together."
}
```
Units: metres, radians, seconds. `start_yaw` always points from start to goal (`atan2` of the chord). `house_xy/house_yaw` are legacy fields of `EpisodeLayout` (`src/nedm/traverse/layout.py:80-115`) set to goal/yaw; `assets` must be `[]`.
Fields the collector actually consumes (`scripts/gen_collect.py:244-250`, `scripts/traverse_fdm_rgbd_diverse_chrono.py:158-200,308`): `id`, `split` (must be train|val|test), `arena` (path relative to the source root; BMP sha must be in `scripts/gen_arenas.json`; `size_m == 80`), `layout.*` (all 7 keys, `assets == []`), `goal_xy`, `goal_radius_m` (default 2.5), `horizon_s` (overridden by `--horizon-s 120`), `settle_reference` (only for the `observe` command). The vehicle is spawned at `(start_xy, tmap.height(start_xy)+0.75)` with `start_yaw` (RIGID FLAG, section 8).
Fields the dataset builder consumes (`scripts/f104_n2_dataset.py:113-122`): `id` -> npz `group`, `split`, `goal_xy`, `layout.start_xy`, `layout.start_yaw`; geometry context `ctx5 = [goal_x-start_x, goal_y-start_y, |rel|, start_yaw, route_len]` (float32; `gen_planner.geom_ctx`, `scripts/gen_planner.py:149-151`).
`layout.seed` is NEVER read by physics or drivers (grep of `scene.py` and the runner: no use). It is an id only.

### 1.3 Manifest `<out>/cases/cases.json`
`{"schema":1, "campaign":..., "wave":..., "arena_dir":..., "strata":..., "seed":..., "prefix":..., "records":[{scene_id, split, family, evaluation_stratum, case (file name), routes [12 paths relative to <out>/cases], case_sha256, route_sha256[12], arena, route_lengths_m[12], horizon_s, start_xy, goal_xy}, ...]}`.
(`generate_traverse_f104_collection.py` records additionally carry `arena_bmp_sha256, arena_meta_sha256, geometry_initialization, feature_index, reference_geometry_checks, split_group_start_cell_8m` and it also writes `cases/pilot_manifest.json` (first `--pilot-groups`=8 groups) and `design/collection_design.json`.)

### 1.4 Directory layout written by the group generators
```
<out>/cases/cases.json
<out>/cases/<prefix>_<index:04d>.json
<out>/cases/routes/<prefix>_<index:04d>/route_00.json ... route_11.json
```
Route ordering: `for offset in [0,-4,+4]: for profile in [constant_2, constant_4, constant_6, smooth_2_6_2]` -> `route_index = 4*offset_index + profile_index`. So `route_00` = straight, 2 m/s (this is the BASE route every planner pool deforms), `route_02` = straight 6 m/s, `route_04..07` = -4 m, `route_08..11` = +4 m. The dataset builder recovers the speed profile as `int(run_id.split('_route_')[1]) % 4` (`f104_n2_dataset.py:119`) -> **run directory names must end in `_route_<NN>`** for designed routes.

### 1.5 Episode / run id conventions actually used
- designed, campaign + night 2: `<group>_route_<NN>` (e.g. `f104_v1_group_0007_route_02`)
- on-policy: `<group>_op_<NN>` (e.g. `f104_v2_group_0000_op_00`; `C/production_v4/tasks.json`, 9,600 rows)
- gen_v1 data: `<group>__route_<NN>` (double underscore; still matches `'_route_'`)
- planner picks: `<group>__<arm>` (first arm that selected that candidate; identical picks are driven once)
- retries: `<id>__retry_<NNN>` (`run_traverse_f104_shard.py:133`)
Ids are globally unique through the prefix (`f104_v1_group`, `f104_v2_group`, `f104_t2_group`, `f104_t3_group`, `f104_g1_test_group`, `f104_s1_group`, `f104_s2_group`, `gXXX_*`). Seeds are NOT globally unique (section 9, item 12).

### 1.6 Task rows for the simple array runner (`scripts/gen_runner.py`, env `GEN_ROOT`, `GEN_TASKS`, `GEN_OUT`, `SLURM_ARRAY_TASK_ID`)
`{"id": run id, "group": group id, "arena": tag, "case": path relative to GEN_ROOT, "route": path relative to GEN_ROOT, "shard": int, "run": true}`.
Shard rule used for gen_v1 data (VERIFIED on `A/gen_v1/tasks_data.json`): `shard = int(md5(group).hexdigest(), 16) % 60` — all routes of a group on one shard/node (Chrono on the AMD cluster is deterministic per node only).

### 1.7 Mission JSON (`gen_missions.py`, `nav_missions.py`)
`{"id", "arena" (relative), "goal_radius_m": 2.5, "goals": [[x,y],...], "layout": {episode_id, seed, assets: [], house_xy (= goals[0]), house_yaw, start_xy, start_yaw (= bearing to goals[0])}, "legs": [{length_m, turn_deg, hill, crater},...], "ground_m", "hazard_legs"[, "n_goals", "total_length_m", "max_turn_deg"]}`.
Runners read only `id, arena, goals, goal_radius_m, layout` (`scripts/nav_runner.py:114-118`, `scripts/gen_mission_runner.py:86-97`).

---------------------------------------------------------------------------------------------------------------

## 2. Start/goal group generators (three near-identical copies)

| script | use | differences |
|---|---|---|
| `scripts/generate_traverse_f104_collection.py` (243 lines) | original frozen campaign (`f104_v1_group_*`, seed 20260909104) | copies the BMP from `--source` (default `/home/harry/NeDM-mppi-claude/assets/traverse/arena_f104`, asserts size 80 / heights [-1.8, 3.9]); `--out`/`--arena` must be inside the repo (`:94-95`); writes pilot manifest + design summary; hard-coded id prefix `f104_v1_group` (`:176`); dedupe only by rounded key |
| `scripts/f104_n2_cases.py` (142 lines) | night-2 wave A1 (`f104_v2_group_*`) and the unsaved test pools t2/t3 | `--prefix`, `--existing` dir whose `f104_v1_group_*.json` rounded keys are skipped (`:32-36`, exact-key match only, no metric margin); imports `footprint, start_ok, speed_profile, sha, dump` from the first script |
| `scripts/gen_cases.py` (160 lines) | gen_v1 / sensor_v1 / sensor_v2 pools, any arena | `--strata all|feature`, `--avoid DIR...` + `--margin-m` (4-D metric margin against avoided dirs AND among the new groups, `:97-101`), `--wave`; `family = "f104_terrain_family"`. **This is the one to use.** |

### 2.1 CLI
```
generate_traverse_f104_collection.py --source DIR --out DIR --arena DIR --groups 500 --pilot-groups 8 --seed 20260909104 --horizon-s 120
f104_n2_cases.py --groups 1200 --seed 20260912104 --prefix f104_v2_group --arena assets/traverse/arena_f104_50h_v1 --out A/cases_night2 --existing A/cases
gen_cases.py     --groups 1200 --seed 20260912104 --prefix f104_v2_group --arena DIR --out DIR --strata {all,feature} --avoid DIR [DIR...] --margin-m 2.0 --wave gen_v1
```
`--arena` is resolved and must be under the repo root (`gen_cases.py:126` `args.arena.relative_to(ROOT)`). `--out` gets `cases/` appended. `--avoid` globs `<dir>/*.json` NON-recursively and silently skips files without `layout/goal_xy` (so `cases.json`, `groups.json` are harmless) -> pass the directory that directly holds the group JSONs (`.../cases/cases` for generator output, flat dir for `cases_*_final`).
`dump()` (`generate_traverse_f104_collection.py:30-44`) refuses to overwrite a file whose text differs (`FileExistsError`), and is a no-op when identical -> re-running with the same args is idempotent; re-running with changed args into the same dir fails fast.

### 2.2 Algorithm (identical in all three; line numbers from `gen_cases.py`)
- Inputs: `TerrainMap.from_dir(arena)` (`src/nedm/traverse/terrain.py:278`); `tmap.features` = hills/craters **in world coordinates after the calibrated orientation transform** (f104 has `orientation.flipud = true`, so feature y is the NEGATIVE of `arena_meta.json` y; never aim at the meta features directly — comment at `generate_...:111-112`). f104: 5 hills (sigma 4.4-6.5 m, amplitude 3.0-4.1 m) + 5 craters (sigma 2.8-3.6 m, amplitude 1.4-2.0 m).
- Start candidates (`:50-53`): 1 m grid on [-34, 34]^2 where `tmap.slope < tan(7 deg)`; chosen uniformly, jittered U(-0.35, 0.35) m per axis (`:62`).
- RNG: ONE `np.random.default_rng(args.seed)` consumed sequentially (`:54`) -> the group list is a deterministic function of (seed, arena, avoid set, margin, strata). With identical settings a longer run reproduces a shorter run as an exact prefix (this is how groups 0500-1499 of `cases_reserve_v1` extend the first 500).
- Strata round-robin (`:60-61`): `feature_index = (index + stalled//1000) % n_strata`, `n_strata = len(features)+2` (= 12 on f104; `--strata feature` -> 10). After 1,000 consecutive rejected attempts the target stratum advances (gates are never relaxed). `max_attempts = max(20000, groups*1000)`; shortfall raises RuntimeError.
  - feature strata (`:63-80`): start must be 9-36 m from the feature centre; aim point = centre + miss * normal, `miss = [0, +0.65, -0.65][(index // 12) % 3] * sigma_m`; goal = start + length * direction with `length = min(distance to the +-35 m box along the ray, |target-start| + U(10, 23))`, rejected if `< |target-start| + 5` (goal always >= 5 m past the feature). Stratum name `<kind>_entry_cross_exit` (miss 0) or `<kind>_cross_slope`.
  - the two extra strata (`:81-87`): goal = another flat candidate, chord >= 42 m; named `long_traverse` (index 10) and `roughness_transfer` (index 11) — generated IDENTICALLY, only the label differs.
  - all: chord length 24-85 m (`:88`). Realised on f104: 28-85 m (the footprint/validator gates remove the shortest).
- Launch footprint gate (`generate_...:47-64`): 6 m x 3 m rectangle aligned with yaw, 13 x 7 samples; `height_range <= 0.65 m`, `max_grade <= 12 deg`, `fitted plane grade <= 6 deg`, `max plane residual <= 0.20 m`. Stored in `geometry_initialization`.
- Dedupe (`:94-96`): key = start+goal rounded to 0.1 m. `gen_cases.py` adds the 4-D Euclidean margin (`:97-101`).
- Route feasibility (`:102-109`): the 9 base routes (3 offsets x 3 constant speeds) from `propose_route_families([*start, yaw], goal, speeds=[2,4,6], offsets=[0,-4,4], step_m=.5)` must ALL pass `check_reference_contract` and `validate_reference(route, [], MPPIConfig(arena_half_extent_m=36., max_speed_mps=6., max_curvature_inv_m=.10), [*start, yaw])`.
- Episode count: exactly 12 routes per group.

### 2.3 Designed route geometry (`src/nedm/traverse/fdm_diverse_planner.py:248-279 propose_route_families`)
- Cubic Hermite from start to goal, `t = linspace(0,1, max(33, ceil(L/0.5)+1))`, start tangent = `L * [cos yaw, sin yaw]`, end tangent = `goal - start`. With yaw = chord bearing this is a STRAIGHT line.
- Lateral family: `xy = base + offset * sin^2(pi t) * normal`, normal = left normal of the chord; offsets 0, -4, +4 m (zero offset and zero slope at both ends).
- Constant speed: `v = min(cruise, sqrt(4 * (L - s)))` = terminal deceleration cone at 2 m/s^2 (`sqrt(2 a d)`, a = 2); ends at v = 0 exactly at the goal; starts at the full cruise speed although the vehicle is at rest (PID ramps up).
- Speed profiles (`generate_...:67-78 speed_profile`): `constant_2/4/6` = above; `smooth_2_6_2`: with x = s/L, `v = 2 + 4*smoothstep((x-.12)/.25) - 4*smoothstep((x-.58)/.25)` (2 m/s until 12 %, 6 m/s from 37 % to 58 %, back to 2 m/s by 83 %), then the same terminal cone. GOTCHA: only the constant-speed bases are validated; the smooth profile's acceleration is never checked (on a 28 m route the 2->6 ramp needs ~2.3 m/s^2 > the 1.5 m/s^2 limit). Harmless (PID tracks what it can) but it is not a "validated" reference.
- Speed range overall: commanded 0-6 m/s; cruise levels 2, 4, 6.

### 2.4 Route validator (`src/nedm/traverse/fdm_mppi.py:121-167 validate_reference`, config `:18-38`)
Checks on the part of the route ahead of the anchor pose: three-point (circumcircle) curvature `planner_s._curvature_max` (`src/nedm/traverse/planner_s.py:65-72`) on the route's OWN 0.5 m samples `<= max_curvature_inv_m`; speed within [min, max]; `d(v^2)/(2 ds)` within `[-max_decel 2.0, +max_accel 1.5]` m/s^2; swept footprint corners (half length 2.6 + 0.1 m, half width 1.3 + 0.1 m, resampled at 0.25 m) inside `|x|,|y| <= arena_half_extent_m`; obstacles unused (`[]`).
Two configurations are in use:
| where | curvature | min turning radius | arena half extent (footprint corners) | speeds |
|---|---|---|---|---|
| designed base routes (group generators) | 0.10 1/m | 10 m | 36 m | max 6 |
| planner proposals, on-policy routes, missions (`CFG` in `gen_planner.py:30`, `f104_n2_onpolicy.py:17`, `f104_n2_cand.py:25`, `gen_missions.py:19`) | 0.125 1/m | 8 m | 40 m (the terrain edge) | 0-6 |
Start/goal points: grid +-34 m (+0.35 jitter); feature-strata goals clipped to +-35 m; mission points +-32 m (`gen_missions.py:40`) / +-33 m (`nav_missions.py:43`). The collector aborts an episode when the chassis leaves +-40 m (`gen_collect.py:123`, status `terrain_bounds_exit`).
VERIFIED on 40 hazard groups x 256 proposals: route centreline beyond +-34 m in 1.5 % of candidates, beyond +-36 m in 0.3 %, maximum 38.3 m. CRM FLAG: those candidates drive within 2-4 m of the patch edge (section 8).

### 2.5 Group split (train/val/test BY GROUP)
`gen_cases.py:111-112` (same in the other two):
```python
key = tuple(np.round(np.r_[start, goal], 1))
split_key = int.from_bytes(hashlib.sha256(json.dumps(key).encode()).digest()[:4], "big") % 100
split = "test" if split_key < 5 else "val" if split_key < 10 else "train"      # 90/5/5, all 12 routes share it
```
Realised: v1 first 500 -> 448/30/22; v2 1,200 -> 1,089/56/55; whole 2,700-group training pool -> 2,431/141/128.
Second layer used by the trainers (`scripts/f104_night_train.py:26-27`, used in `f104_n2_train.py:77-82`): **dev fold** = `int(md5(group_id).hexdigest(), 16) % 5 == 0` among `split == "train"`; fit = train and not dev; the dev metric uses designed routes only. Both are name/coordinate hashes, NOT geometric separation (the night-2 audit notes new groups land within 0.2-0.5 m of old ones).
GOTCHA: held-out pools made by the same generators STILL carry the 90/5/5 hash in `split` (e.g. `f104_g1_test_group`: 180/9/11). The collector requires the field; the dataset builder copies it into the npz; trainers select `split == 'train'`. Never merge a held-out pool's npz into a training npz without overriding `split`.

---------------------------------------------------------------------------------------------------------------

## 3. Planner-proposal routes: `scripts/f104_n2_sampler.py` (108 lines, numpy only)

Constants (`:23-24`): `A_ACC, A_DEC = 1.5, 2.0` m/s^2; `V_MIN, V_MAX = 0.5, 6.0` m/s.
- `lateral_profile(f, L, rng, sigma=5.0, modes=3, kappa_max=0.125, budget=0.55)` (`:58-63`): `lat(f) = sum_{j=1..3} a_j sin(j pi f)`, `a_j ~ N(0, sigma)/j`, clipped to `+- 0.55 * kappa_max * L^2 / (j pi)^2` (curvature budget). Zero at both ends -> start pose and goal stay pinned.
- `speed_knots(f, k, vals)` (`:35-40`): smoothstep interpolation between k knots placed at `linspace(0,1,k)`; ends are FREE (not forced to zero).
- `shape(xy, station, speed, lat, dv)` (`:43-55`): offsets each base point along the LOCAL left normal; recomputes stations; `v = clip(speed + dv, 0.5, 6.0)`; terminal cone `v <= sqrt(2*2.0*(L - s))`; forward pass `v_j <= sqrt(v_{j-1}^2 + 2*1.5*ds)`; backward pass with 2.0; headings from gradients. `v[0]` is NOT forced to 0 (0.5-6 m/s from rest).
- `sample_one(base, rng, knots=4, lat_sigma=5.0, lat_clip=10.0, sp_sigma=1.5, sp_clip=4.0, base_speed=None, modes=3, kappa_max=0.125)` (`:66-78`): lateral clipped to +-10 m; speed delta knots `clip(N(0, sp_sigma), +-4)` added to the base speeds (or to a constant `base_speed`). meta `candidate='n2_wide'`.
- `anchors(base, offsets=(0,-4,4), speeds=(2,4,6))` (`:81-93`): 9 designed-style routes, `lat = off * sin^2(pi f)` along the local normal, constant speed + cone; meta `candidate='n2_anchor', lateral_offset_m, cruise_speed_mps`. Order: offset-major, so index 0 = straight 2 m/s, 2 = straight 6 m/s.
- `propose(base, anchor_pose, rng, n=256, validate=None, cfg=None, **kw)` (`:96-108`): valid anchors first, then rejection sampling until n, at most `8*n` tries; returns `(list, tries)`.
All candidates inherit the base route's waypoint COUNT (so per-group arrays stack). Base = the group's `route_00` with its own speeds (2 m/s + cone).
Measured (VERIFIED tonight, 40 hazard groups): acceptance 77 % mean (min 50 %); max lateral p50/p90/max = 4.1/7.5/10.0 m; route-mean speed p5/p50/p95 = 0.98/2.01/3.47 m/s. Curvature is the only rejection reason in practice.

### 3.1 On-policy training routes: `scripts/f104_n2_onpolicy.py` (81 lines)
CLI: `--cases A/cases_night2/cases --out A/cases_night2_onpolicy --n 8 --seed 4242`.
Per group (order = `records` order of `<cases>/cases.json`): `rng = default_rng(seed + group_order_index)` (`:27`); route k uses `lat_sigma = [7,7,5,5,5,5,3,3][k]`, `sp_sigma = [2.5,1,2.5,1.5,1.5,1,2.5,1.5][k]`, `base_speed=2.0` (`:32-34`); must pass `validate_reference(CFG)` and `check_reference_contract`; up to `40*n` tries.
Output: `<out>/routes/<group>/op_<k:02d>.json` and `<out>/routes.json` = list of `{group, route (path as typed, relative to cwd), index, max_lateral_m, mean_speed_mps, speed_at_85pct}`. VERIFIED dry run: 24 groups -> 192 routes in 2 s; lateral p50/p90/max 4.2/8.1/9.8 m, mean speed p5/p50/p95 1.05/1.96/3.47.
GOTCHAS: `--n > 8` -> IndexError (schedule lists have 8 entries). The seed depends on the group's POSITION in cases.json -> regenerating for a subset or re-ordered manifest gives different routes. To reuse the existing 9,600 night-2 routes, use the files (`A/cases_night2_onpolicy/routes/`, cluster copy `C/cases_night2_onpolicy_v1/routes/`), do not regenerate.
Rigid outcome rates for orientation (VERIFIED from `A/night2_v1/station_ds_all.npz`, 36,199 routes, X float16 (N,5,96,32), ctx (N,22)): designed unsafe 23.6 % / fail 10.9 %; by profile constant_2 56.2 % / 27.2 %, constant_4 19.8 % / 8.1 %, constant_6 1.4 % / 0.7 %, smooth_2_6_2 16.9 % / 7.4 %; on-policy 56.4 % / 33.2 %. On rigid ground, slow = stuck on grades, fast = momentum carries over. Expect CRM to move these numbers a lot (sinkage, slip, launch dig-in).

### 3.2 Offline candidate packs for fixed test sets (night 2)
- `scripts/f104_n2_cand.py` (no CLI; constants `ROOT, CASES=A/cases_test_final, OUT=A/night2_v1/testcand, N=256`; reads `A/night2_v1/fresh_test_groups.json`). Three sets per group: `night1` (old 3-knot deformer `f104_night_speedcand.deform_smooth_speed`, lateral sigma 2.6 clip 6, speed sigma 1.2), `night2` (`S.propose`), `fixed2` (`S.sample_one(lat_sigma=5, sp_sigma=0, base_speed=2.0)`). RNG: `md5(g)`, `md5(g+'n2')`, `md5(g+'f2')`, first 8 hex digits (`:46,54,57`). Output `<OUT>/<g>__<set>.npz`: `X` float16 (n,5,96,32), `geom_ctx` float32 (n,5), `route_len`, `wp` (n,N,2), `sp`, `st`, `hd`, `anchor` bool, `cruise`, `offset`, `max_lateral`, `mean_speed`, `split`.
- `scripts/f104_n2_haz.py`: same for `A/night2_v1/haz_test_groups.json` -> `A/cases_haz_final` (stages ONLY `<g>.json` + `route_00.json`, `:22-30`), tags `'h1','h2','h3'`, night-1 budget `40*N` tries. `scripts/f104_n2_ext.py`: `ext_test_groups.json` -> `A/cases_ext_final`, fixed-2 only, `N=192`, tag `'ext'`.
- These call raw `validate_reference`, which RAISES on degenerate (folded) routes; `gen_planner.safe_validate` (`scripts/gen_planner.py:94-99`) converts that to a rejection. Prefer the gen_planner path.

---------------------------------------------------------------------------------------------------------------

## 4. The packaged planner: `scripts/gen_planner.py` (323 lines)

- `CFG` (`:30`), `ELEV_SCALE=10.0`, `N_CAND=256`. Env overrides: `GEN_SRC` (src tree), `GEN_MODELS` (checkpoint glob), `GEN_RULE` (hand-rule json).
- Map sources (mutually exclusive, module-global state): `set_map(arena_dir)` (`:35-42`, BMP heightmap, `np.flipud(tm.height_grid)/10` into channel 3 of a fake RGB-D — RIGID FLAG: undeformed BMP), `set_sensor_map(mapdir)` (`:251-256`, captured overhead RGB-D, 10-channel corridors `corridors10`), `set_grid_map(griddir)` (`:306-310`, sensor_v2 back-projected world grid, 12-channel `corridors12`).
- `base_route(pose, goal, scales=(1.0,1.5,2.0,0.7,2.5), radii=(12,10,9))` (`:102-121`): first choice is exactly the frozen generator's `route_00` (`propose_route_families(..., speeds=[2.], offsets=[0.], step_m=.5)[0]`, speeds `constant_speed(st, 2.)`). If it fails `CFG` (vehicle facing away from the goal, mission legs only): Hermite with start-tangent scale 1.5, 2.0, 0.7, 2.5 (`_hermite :50-58`), then arc-then-line (Dubins CS) at radius 12/10/9 m toward the goal side (`_arc_line :61-91`), then the long-way arc at 12/10/9/8.5 m; first valid wins; else the invalid first is returned. Single start/goal cases never reach the fallbacks (start yaw faces the goal).
- `proposal_pool(base, pose, rng, n=256)` (`:124-127`) = `S.propose(..., validate=safe_validate, cfg=CFG)`; `fixed2_pool` (`:130-138`) = up to `6*n` tries of `S.sample_one(lat_sigma=5, sp_sigma=0, base_speed=2.0)`.
- `corridors(cands)` (`:141-146`) -> `X` float32 (n,5,96,32) channels `[elev - e0, grade_along, grade_cross, speed, valid]` via `f104_n2_dataset.station_tensor` (`scripts/f104_n2_dataset.py:45-60`; 96 stations uniformly in arc length, 32 lateral samples across +-6 m), and `L` route lengths.
- `RiskModel` (`:154-184`): ensemble from `GEN_MODELS` or `A/night2_v1/final/N2_s*.pt`; checkpoint keys `cin, nctx, arch, layers, state, norm{mu,sd}, ctx_mu, ctx_sd`; `score(X, ctx5) -> (mean logit, 1-exp(-exp(z)))`. `SensorRiskModel` / `GridRiskModel` (`:268-323`) add `channels`, `norm.cont_index`; channel name lists at `:270` and `:323`.
- `anchor_index(cands, offset, cruise)` (`:202-208`); `plan(pose, goal, rng, model, rule, mode)` (`:211-241`), modes `n2 | s | rule | straight6 | straight2`.

---------------------------------------------------------------------------------------------------------------

## 5. Inventory of existing f104 start/goal pools (VERIFIED against the local files)

| pool | location (local) | prefix | seed | groups | routes on disk | note |
|---|---|---|---|---|---|---|
| campaign initial | `A/cases/` | `f104_v1_group_0000-0499` | 20260909104 | 500 | 12 designed | split 448/30/22; strata 111 crater_cross / 56 crater_entry / 140 hill_cross / 70 hill_entry / 82 long / 41 roughness |
| campaign reserve | `A/cases_reserve_v1/` | `f104_v1_group_0500-1499` | same stream | 1,000 | 12 designed | flat dir (no `cases/` level) |
| night-2 A1 | `A/cases_night2/cases/` (cluster: `C/cases_night2_v1/`, FLAT) | `f104_v2_group_0000-1199` | 20260912104 | 1,200 | 12 designed | split 1,089/56/55 |
| night-2 A2 | `A/cases_night2_onpolicy/routes/` (cluster `C/cases_night2_onpolicy_v1/`) | same groups | 4242 + order | 1,200 | 8 on-policy (`op_00-07`) | 9,600 routes |
| test pool t2 | NOT on disk (only the selected finals) | `f104_t2_group` | 20260912777 | >= 797 | — | made with `f104_n2_cases.py`; the command line was never saved |
| test pool t3 | NOT on disk | `f104_t3_group` | 20260912888 | "4,000" (max selected index 3198) | — | same |
| fresh184 | `A/cases_test_final/` (+`groups.json`), list `A/night2_v1/fresh_test_groups.json` | t2 (41) + t3 (143) | — | 184 | 12 designed | rule: >= 6 m 4-D from all 1,500 old + A1 groups, >= 3 m apart. VERIFIED min 6.01 m (median 6.96) to the 2,700-group pool, min mutual 3.01 m. Only 15 % hill/crater; chord median 55 m |
| ext523 | `A/cases_ext_final/`, list `ext_test_groups.json` | t2 + t3 | — | 523 | `route_00` ONLY | >= 4 m, >= 3 m apart, excludes fresh184. VERIFIED 4.00 m; 43 % hill/crater |
| haz300 | `A/cases_haz_final/`, list `haz_test_groups.json` | t2 (63) + t3 (237) | — | 300 | `route_00` ONLY | hill/crater strata only, rule >= 2.0 m (declared 4 m, shipped 2 m), >= 3 m apart. VERIFIED realised min 3.37 m |
| gen_v1 f104 test | `A/gen_v1/cases_test_f104/cases/` | `f104_g1_test_group` | 20260915104 | 200 | 12 designed | `--strata feature --margin-m 2.0`, VERIFIED min 2.02 m to the pool and >= 2.0 m to every other held-out set |
| sensor_v1 test | `A/sensor_v1/cases_f104/cases/` | `f104_s1_group` | 20260917104 | 200 | 12 designed | same recipe, VERIFIED 2.01 m |
| sensor_v1 test 2 | `A/sensor_v1/cases2_f104/cases/` | `f104_s2_group` | 20260918104 | 200 | 12 designed | same recipe, VERIFIED 2.01 m |
| 5-goal missions | `A/gen_v1/missions/f104/` | `f104_mission_000-099` | 20260915500 | 100 | — | `gen_missions.py` |
| nav missions | `A/nav_v1/missions/` | `f104_nav_000-003` | 7104 | 4 (of 30 over 10 arenas) | — | `nav_missions.py` |

Cluster copies of the three finals: `C/cases_test_final_v1`, `C/cases_haz_final_v1`, `C/cases_ext_final_v1`; gen_v1 pools under `C/gen_v1/cases_test_f104` etc.
Margin metric everywhere: Euclidean distance in 4-D `(start_x, start_y, goal_x, goal_y)`, metres.
History of the held-out margin: night-2 primary test >= 6 m (kept mostly long flat traverses -> every arm at the failure floor, underpowered; the night-2 audit also found the margin buys no terrain novelty on one arena) -> extension >= 4 m -> hazard test >= 2 m and hill/crater strata only -> from gen_v1 on: `gen_cases.py --strata feature --margin-m 2.0 --avoid <everything used before>`.
The scripts that SELECTED fresh184 / ext523 / haz300 out of t2/t3 were inline and never saved (not in `scripts/`, not in `~/NeDM-archive/traverse_mppi_removed_f104_scripts_2026-09-15.tar.gz`). `gen_cases.py --avoid` supersedes them.
Cross-set overlaps (VERIFIED): ext523 sits 0.42 m from a fresh184 group and 0.34 m from a haz300 group at the closest (they came from the same pools with different rules) — do not treat those three as mutually independent samples; the three 200-group pools are >= 2.0 m from everything.

---------------------------------------------------------------------------------------------------------------

## 6. Offline pick builders for closed-loop tests

### 6.1 `scripts/gen_pools.py` (116 lines) — heightmap corridors + N2-format model + hand rule
```
python scripts/gen_pools.py --cases <dir with <group>.json and routes/> --arena <arena dir> --out <dir> --arena-tag f104 [--workers 12] [--shards 24]
```
Per case: base = `routes/<g>/route_00.json`; `proposal_pool` with `rng = default_rng(int(md5(g+'gen_night2').hexdigest()[:8],16))`, `fixed2_pool` with tag `'gen_fixed2'` (`:30-31,42-43`). Arms (`:20`, picks `:74-79`): `n2` argmin model logit over the proposal pool; `rule` argmin hand-rule score over the same pool; `straight6` / `straight2` = anchor (offset 0, cruise 6 / 2); `n2_fixed2`, `rule_fixed2` over the fixed-2 pool. Identical (pool, index) picks are written once under the first arm's name.
Writes `<out>/routes/<g>__<arm>.json`, `<out>/picks/<g>.json` (per-arm `route_id, pool, index, risk, logit, rule_score, rule_rank, model_rank, mean_speed, length_m`, plus pool-level spearman and risk quantiles), `<out>/tasks_cluster.json` rows `{id, group_id, arena, arm, case (relative to --cases), shard = md5(g) % shards, run (True only for the first arm of a duplicate pick)}`.
GOTCHAS: (i) instantiates `RiskModel()` and `HandRule()` unconditionally — needs N2-format checkpoints (`GEN_MODELS`) and `A/gen_v1/hand_rule.json` (`GEN_RULE`) even if only the straight arms are wanted. (ii) its task rows have NO `route` key and use `group_id`; `gen_runner.py` needs `route` and paths relative to `GEN_ROOT`. The driven list `A/gen_v1/tasks_test.json` (keys `arena, case, group, id, route, run, shard`; e.g. case `cases_test_f104/cases/<g>.json`, route `test_f104/routes/<rid>.json`) was produced by an unsaved merge step. `sensor_pools*.py` write runner-ready rows directly.
gen_v1 f104 realised: 200 groups -> 1,103 distinct drives (n2 200, n2_fixed2 200, straight2 200, rule 190, rule_fixed2 172, straight6 141 listed under their own name).

### 6.2 `scripts/sensor_pools_v2.py` (83 lines) — world-grid corridors + matched_H / matched_Dabs-format models
```
python scripts/sensor_pools_v2.py --cases <dir> --grid A/sensor_v2/grids/arena_f104_50h_v1 --out <dir> --arena-tag f104 \
   --models H=A/sensor_v2/matched/matched_H_s*.pt,Dabs=A/sensor_v2/matched/matched_Dabs_s*.pt \
   --case-prefix <case dir as seen from the runner root> --route-prefix <route dir as seen from the runner root> [--workers 12] [--shards 8]
```
Seeds `md5(g+'v2_proposal')`, `md5(g+'v2_fixed2')` (`:30-31`). Arms: `straight6`, each model NAME (speed free), `NAME_fixed2` (`:48`). Writes `<out>/routes/`, `<out>/picks/`, `<out>/tasks.json` rows `{id, group, arena, case=<case-prefix>/<file>, route=<route-prefix>/<rid>.json, shard, run: True}` (only first-of-duplicate rows). Example row: `A/sensor_v2/tasks_pilot.json[0]`.
`scripts/sensor_pools.py` (101 lines) is the sensor_v1 twin: `--map <captured RGB-D dir>`, tags `'s1_proposal','s1_fixed2'`, arms `n2, straight6, n2_fixed2` + the named sensor models, default 24 shards.
NOTE: every study used a different md5 tag, so the same group gets a different 256-candidate pool in each study. For a rigid-vs-CRM comparison on identical candidate pools reuse the rigid study's script+tag (e.g. `gen_pools.py` / `'gen_night2'` on `f104_g1_test_group`).

---------------------------------------------------------------------------------------------------------------

## 7. Multi-goal missions

- `scripts/gen_missions.py` (95 lines): `--arena DIR --tag f104 --n 100 --seed 20260915500 --out DIR`. Exactly 5 goals; points on the 1 m grid within +-32 m with slope < 7 deg (+U(-0.35,0.35) jitter); legs 20-35 m; heading change at each goal <= 110 deg; each leg's `gen_planner.base_route(pose_with_previous_heading, goal)` must validate under `CFG`; >= 3 of 5 leg chords cross a hill zone (`h > median ground + 2.0 m`) or crater zone (`h < median - 0.7 m`) sampled at 60 points (`hazard_flags :22-27`; f104 median ground 0.1626 m); hazard-crossing next goals preferred with p = 0.7; start passes the launch footprint gate. id `<tag>_mission_<NNN>`, `layout.seed = seed + index`. (Pre-registered 25-45 m / 60 deg was infeasible: 0 of 200,000 attempts.)
- `scripts/nav_missions.py` (119 lines): `--arena --tag --n --seed --out [--goals-min 5 --goals-max 8 --leg-min 22 --leg-max 35 --turn-max-deg 120 --total-min 150 --total-max 250]`; grid +-33 m; next goal > 12 m from every earlier point when possible (no fold-back); hazard legs >= `max(3, n_goals//2)`; id `<tag>_nav_<NNN>`. GOTCHA: that hazard rule makes 1- and 2-goal missions impossible (never accepted) — for single-goal missions convert cases (snippet in 10c).
- `scripts/nav_tasks.py` (41 lines): `--missions DIR --out tasks.json --arms W R2 R1 --shards 10 [--save-frames ids]`; arm table `:5-18` (`W` plan once per waypoint, `R2`/`R1` periodic 2 s / 1 s, `*L` latency-charged, `*_20` 20 m sensing radius, `R1rand`, `R1S`); ALL arms of a mission share shard `i % shards`.

---------------------------------------------------------------------------------------------------------------

## 8. RIGID-terrain-specific assumptions in this subsystem (what CRM must decide)

1. **Arena edge margin.** Designed routes keep the swept footprint inside +-36 m, but planner/on-policy candidates are only constrained to +-40 m = the patch edge (`CFG.arena_half_extent_m=40`), missions to +-32/33 m points. 1.5 % of proposal candidates have a centreline beyond +-34 m (max 38.3 m). A CRM patch needs boundary walls / particle containment and the soil near a wall is not representative. Options: (a) keep `CFG` (exact contract) and accept/flag near-edge episodes post hoc via `max|xy|`; (b) tighten `arena_half_extent_m` to 36 in the four `CFG` definitions — this CHANGES the proposal distribution and breaks candidate-pool identity with the rigid studies. If the CRM domain is an active box moving with the vehicle over a rigid shell, the question disappears.
2. **Launch footprint gate and spawn.** `start_ok` thresholds (0.65 m range, 12 deg, 6 deg, 0.20 m) were chosen so a rigid-ground spawn at `height+0.75 m` settles within the collector's launch check (`gen_collect.py on_anchor`: speed <= 1 m/s, |roll|,|pitch| <= 20 deg, yaw error <= 10 deg, xy error <= 1 m after a 0.8 s settle). On CRM the vehicle sinks and the settle is longer; the gate itself can stay, the settle time/thresholds are the collector's problem. `geometry_initialization.method` literally says "native settling pending".
3. **Standing-start speed references.** `constant_4`, `constant_6` and many proposals command 4-6 m/s at station 0 from rest, and proposals may start anywhere in 0.5-6 m/s. On rigid ground the PID just saturates throttle; on soft soil a full-throttle launch can dig in. This is a label-distribution effect, not a contract break — keep the routes identical, expect more launch failures, and consider looking at launch entrenchment separately in the labels.
4. **Hazard definition by geometry only.** Strata (hill/crater aim points, `miss = 0.65 sigma`), mission hazard zones (+2.0 m / -0.7 m about the median) and the 7 deg flat-candidate rule are BMP geometry. On CRM, flat soft ground is itself a hazard; `long_traverse`/`roughness_transfer` groups (2 of 12 strata) are the only ones that sample it on purpose. If CRM failures turn out to be dominated by sinkage on flats, raise their share by generating an extra `--strata all` pool and sub-selecting those two strata (there is no CLI flag for a flat-only stratum).
5. **Validator limits are kinematic rigid-ground numbers** (kappa 0.10/0.125 1/m, accel 1.5, decel 2.0 m/s^2, 6 m/s cap). Keep them unchanged for the port (they define the route distribution the models are trained/queried on).
6. **Path z for the Bezier driver** = BMP height + 0.5 m (`make_driver`). Geometrically still right for CRM's initial surface; ruts do not matter to the 2-D steering controller. No change needed, but it is "truth height", as on rigid.
7. **Map source for planner corridors**: `set_map` reads the undeformed BMP; `set_grid_map`/`set_sensor_map` read captures of the rigid mesh. For CRM the initial surface equals the BMP, so heightmap corridors remain valid as the pre-traverse map. Sensor captures of a particle surface are another scout's topic.
8. **Native-terrain audit** (`gen_collect.py:85-88`, `scene.terrain.GetHeight` vs BMP at a 7x7 grid on +-36 m) and the allowlist `scripts/gen_arenas.json` (`arena_f104_50h_v1` sha256 `5d5bc683...ee8ed`) are rigid-collector gates that reference case fields; a CRM collector must keep accepting the same case JSON (fields in 1.2).
9. **Determinism / pairing.** Shard rule "all routes/arms of a group on one node" exists because rigid Chrono on the AMD nodes is deterministic per node only. SPH determinism is unknown — keep the same group-to-shard rule so paired comparisons stay same-node.
10. **Budget.** Rigid: 18,000 episodes = 66 h simulated (13 s mean per episode; horizon 120 s; a stall costs >= 24 + 2 + 8 s) collected in 13.6 wall minutes. CRM will be orders of magnitude slower per simulated second, so the pool must be sub-selected (section 10a) rather than driven whole.

---------------------------------------------------------------------------------------------------------------

## 9. Gotchas (numbered for reference)

1. All f104/gen/nav/sensor scripts are UNTRACKED in git (171 untracked files in `scripts/`). Copies exist on the cluster: `C/gen_v1/source/{scripts,src,assets}`, `C/gen_v1/planner/`, `C/nav_v1/code/`. Do not `git clean`.
2. Use the `nedm` conda env (torch needed by import); run from the repo root.
3. `generate_traverse_f104_collection.py` defaults point at another worktree (`/home/harry/NeDM-mppi-claude/...`) and it hard-codes the `f104_v1_group` prefix and campaign name — do not use it for new pools.
4. `--avoid` is non-recursive; generator output lives one level down (`<out>/cases/`). Cluster copy of night-2 cases is flat (`C/cases_night2_v1/<g>.json`), local is `A/cases_night2/cases/<g>.json`.
5. `tmap.features` are orientation-corrected (f104: y sign flipped vs `arena_meta.json`).
6. The miss cycle divides by `len(features)+2` even with `--strata feature` (`gen_cases.py:71`). With `--strata all` the full (stratum x miss) pattern repeats every 36 groups; with `--strata feature` (10 strata, miss still stepping every 12) it repeats only every 180. Take index prefixes in multiples of 36 from `--strata all` pools to stay balanced (the 1,000-rejection stratum skip can shift the phase slightly).
7. `long_traverse` and `roughness_transfer` are the same generator branch with different labels.
8. `split` inside held-out pools is the same 90/5/5 hash (item in 2.5).
9. `smooth_2_6_2` is never acceleration-validated; `constant_*` start at cruise speed from rest.
10. `f104_n2_onpolicy.py`: `--n <= 8`; seed tied to manifest order; `routes.json` paths are as typed (relative to cwd).
11. `f104_n2_cand/haz/ext.py` have no CLI (module constants), use raw `validate_reference` (can raise), and `cases_haz_final` / `cases_ext_final` contain only `route_00` (no designed routes 01-11).
12. `layout.seed = base_seed + index` is not globally unique: v2 (20260912104..13303) overlaps t2 (20260912777..) and t3 (20260912888..). Nothing reads it. If the CRM collector needs a per-episode RNG seed (particle jitter etc.), derive it from the run id, e.g. `int(md5(run_id).hexdigest()[:8], 16)`.
13. `gen_pools.py` rows are not runner-ready (no `route` key); `sensor_pools_v2.py` rows are.
14. `nav_missions.py` cannot emit missions with fewer than 3 hazard legs (so none with < 3 goals).
15. Planner pool RNG tags differ per study (3.2, 6.1, 6.2) — record the tag you use.
16. `station_tensor` resamples to 96 stations over the route's own length, so station spacing varies per route (0.3-0.9 m); `event_idx` labels are in that 0-95 index space.
17. Proposal pools can come back with fewer than 256 candidates (budget `8*n` tries) and `fixed2_pool` can be empty; pick code handles `None`.
18. A 6 m held-out margin is a selection effect, not a novelty guarantee: it keeps long flat traverses (15 % hazard vs ~83 % in the pool). Use 2 m + `--strata feature` for a powered planner test; report both if a "far" set is wanted (fresh184 exists).

---------------------------------------------------------------------------------------------------------------

## 10. Recommended recipe for tonight (copy-paste)

```bash
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python
A=artifacts/traverse/fdm_f104_50h_20260909
K=artifacts/traverse/crm_f104_v1
ARENA=assets/traverse/arena_f104_50h_v1
# every f104 start/goal ever used (training pool + all held-out sets); non-recursive dirs
AVOID="$A/cases $A/cases_reserve_v1 $A/cases_night2/cases $A/cases_test_final $A/cases_haz_final $A/cases_ext_final \
       $A/gen_v1/cases_test_f104/cases $A/sensor_v1/cases_f104/cases $A/sensor_v1/cases2_f104/cases"
```

### (a) Training pool: start/goal groups x varied routes and speeds
PRIMARY (reuse verbatim, zero generation): night-2 groups `f104_v2_group_NNNN`, 20 routes per group =
12 designed (`$A/cases_night2/cases/routes/<g>/route_00..11.json`: offsets 0/-4/+4 m x constant 2/4/6 + smooth 2-6-2) +
8 on-policy (`$A/cases_night2_onpolicy/routes/<g>/op_00..07.json`: sine-basis detours to +-10 m, free speed 0.5-6 m/s).
Sub-select by index prefix in multiples of 36 groups (round-robin strata x 3 miss values), e.g. first 288 groups = 5,760 episodes; if that is too many, consider thinning routes (e.g. offset 0 x 4 profiles + `op_00..03` = 8 per group) before dropping groups (the night-2 scaling curve was measured over the fraction of GROUPS: 25 % -> G_unsafe .936, 50 % -> .957; routes-per-group was never ablated, so this is a judgement call). Keep the group's hash `split` and the md5 dev fold; they need no file changes.
Task list (runner-ready rows; ids unique; whole group on one shard):
```python
import json, hashlib, glob, os
A='artifacts/traverse/fdm_f104_50h_20260909'; NG, SH = 288, 48
rows=[]
for i in range(NG):
    g=f'f104_v2_group_{i:04d}'; sh=int(hashlib.md5(g.encode()).hexdigest(),16)%SH
    case=f'{A}/cases_night2/cases/{g}.json'
    for r in sorted(glob.glob(f'{A}/cases_night2/cases/routes/{g}/route_*.json')):
        k=os.path.basename(r)[6:8]; rows.append(dict(id=f'{g}_route_{k}', group=g, arena='f104', case=case, route=r, shard=sh, run=True))
    for r in sorted(glob.glob(f'{A}/cases_night2_onpolicy/routes/{g}/op_*.json')):
        k=os.path.basename(r)[3:5]; rows.append(dict(id=f'{g}_op_{k}', group=g, arena='f104', case=case, route=r, shard=sh, run=True))
json.dump(rows, open('artifacts/traverse/crm_f104_v1/tasks_train.json','w'), indent=1)
```
(Paths here are repo-relative; rewrite the `case`/`route` prefixes to whatever root the CRM runner uses. Run ids reuse the rigid ids on purpose — keep CRM outputs under a separate campaign root so `<id>` pairs 1:1 with the rigid run of the same name.)
ALTERNATIVE (fresh groups never driven on anything; dry-run VERIFIED, ~0.15 s per group):
```bash
$PY scripts/gen_cases.py --groups 288 --seed 20260923104 --prefix f104_crm_train_group --arena $ARENA \
    --out $K/cases_train --strata all --margin-m 2.0 --wave crm_f104_v1_train --avoid $AVOID
$PY scripts/f104_n2_onpolicy.py --cases $K/cases_train/cases --out $K/cases_train_onpolicy --n 8 --seed 20260923
```
-> `$K/cases_train/cases/{cases.json,<g>.json,routes/<g>/route_00..11.json}` and `$K/cases_train_onpolicy/{routes.json,routes/<g>/op_00..07.json}`.
Cost of the alternative: the existing held-out sets are only guaranteed >= 2 m from these new groups if they are in `--avoid` (they are, above), and the rigid pairing is lost.

### (b) Held-out validation / test groups (offline ranking metrics, 12 designed routes each)
- Same-distribution hold-out inside the pool: groups with `split in {val, test}` (v2: 56 + 55 groups; first 288 groups contain ~5 % each) and the md5 dev fold (`int(md5(g).hexdigest(),16) % 5 == 0`, 20 % of train groups). No generation; just make sure these groups are in the driven subset.
- Geometrically separated hold-out, existing: `$A/gen_v1/cases_test_f104/cases` (200 hill/crater groups, >= 2.02 m VERIFIED) for a powered test; `$A/cases_test_final` (184 groups, >= 6.01 m VERIFIED, mostly long flat traverses) as the "far" set.
- Fresh (recommended if the budget allows, so nothing about them was ever looked at):
```bash
$PY scripts/gen_cases.py --groups 108 --seed 20260924104 --prefix f104_crm_val_group  --arena $ARENA \
    --out $K/cases_val  --strata all --margin-m 2.0 --wave crm_f104_v1_val  --avoid $AVOID $K/cases_train/cases
$PY scripts/gen_cases.py --groups 108 --seed 20260925104 --prefix f104_crm_test_group --arena $ARENA \
    --out $K/cases_test --strata all --margin-m 2.0 --wave crm_f104_v1_test --avoid $AVOID $K/cases_train/cases $K/cases_val/cases
```
(drop the `$K/cases_train/cases` argument if the primary reuse option is taken). Remember gotcha 8: override `split` to `val`/`test` when these land in an npz, or keep them in separate npz files.

### (c) Held-out single-goal missions for planner evaluation
Groups: hill/crater strata only, 2 m margin (the recipe of gen_v1 / sensor_v1 / sensor_v2):
```bash
$PY scripts/gen_cases.py --groups 200 --seed 20260926104 --prefix f104_crm_eval_group --arena $ARENA \
    --out $K/cases_eval --strata feature --margin-m 2.0 --wave crm_f104_v1_eval \
    --avoid $AVOID $K/cases_train/cases $K/cases_val/cases $K/cases_test/cases
```
or reuse `$A/gen_v1/cases_test_f104/cases` (`f104_g1_test_group`, rigid results for six arms exist in `$A/gen_v1/test_results.json` -> direct rigid-vs-CRM comparison of the same picks).
Picks once a CRM-trained model exists (N2-format checkpoints):
```bash
GEN_MODELS="$K/models/N2crm_s*.pt" $PY scripts/gen_pools.py --cases $K/cases_eval/cases --arena $ARENA \
    --out $K/eval_f104 --arena-tag f104 --workers 12 --shards 24
```
or, for matched_H / matched_Dabs-format checkpoints:
```bash
$PY scripts/sensor_pools_v2.py --cases $K/cases_eval/cases --grid $A/sensor_v2/grids/arena_f104_50h_v1 --out $K/eval_f104_v2 \
    --arena-tag f104 --models H=$K/models/matched_H_s*.pt,Dabs=$K/models/matched_Dabs_s*.pt \
    --case-prefix cases_eval/cases --route-prefix eval_f104_v2/routes --shards 8
```
Model-free baselines available before any CRM model exists: drive `route_02` (straight 6 m/s) and `route_00` (straight 2 m/s) of each eval group — they are the `straight6` / `straight2` arms up to the anchors' local-normal construction (identical for offset 0).
If the online mission runner (plan at the actual settled pose, `gen_mission_runner.py` / `nav_runner.py`) is preferred over offline picks, convert cases to one-goal missions:
```python
import json, glob, os
K='artifacts/traverse/crm_f104_v1'; os.makedirs(f'{K}/missions_eval', exist_ok=True)
for p in sorted(glob.glob(f'{K}/cases_eval/cases/f104_crm_eval_group_*.json')):
    c=json.load(open(p))
    m=dict(id=c['id']+'_m1', arena=c['arena'], goal_radius_m=c['goal_radius_m'], goals=[c['goal_xy']],
           layout={**c['layout'], 'episode_id': c['id']+'_m1'}, n_goals=1, stratum=c['evaluation_stratum'])
    json.dump(m, open(f"{K}/missions_eval/{m['id']}.json",'w'), indent=1)
```
Multi-goal missions, if wanted later: `$PY scripts/gen_missions.py --arena $ARENA --tag f104crm --n 50 --seed 20260927500 --out $K/missions5` (or reuse `$A/gen_v1/missions/f104/f104_mission_000..099.json`, rigid results in `$A/gen_v1/mission_results.json`).

### Reuse verbatim vs adapt
| reuse verbatim | adapt / do not use |
|---|---|
| `gen_cases.py`, `f104_n2_onpolicy.py`, `f104_n2_sampler.py`, `gen_planner.py` (route side), `gen_missions.py`, `nav_missions.py`, `nav_tasks.py`, `sensor_pools_v2.py`, all existing case/route files | `generate_traverse_f104_collection.py` (wrong defaults, hard-coded prefix), `f104_n2_cases.py` (superseded by `gen_cases.py`), `f104_n2_cand/haz/ext.py` (no CLI, hard-coded night-2 paths, raw validator), `gen_pools.py` task rows (add `route`, rename `group_id`) |
