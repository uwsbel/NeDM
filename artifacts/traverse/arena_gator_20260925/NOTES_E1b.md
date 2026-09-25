# NOTES E1b: spread test arenas, new suites, declared subsets, spread headroom rows, map-lookup error (2026-09-25, 02:25-02:50)

PLAN section 7 items 1, 2, 4 and 6. K3 = this folder, G3 = `/work1/dannegrut/harry/experiments/arena_gator_20260925`.
Everything was made locally, and only new files were copied to the cluster. **No job was submitted**, and no soil or rigid
episode of any suite was driven. The only local Chrono runs were the soil surface smokes and 6-second rigid launch smokes.
[verifier E1b: the 6-second launch smokes did drive route_00 of group 0000 of each spread suite for 6 s (all stopped on
the time limit, no outcome; output only in `smoke/rigid_e1b`); no suite episode was driven to an outcome.]
No existing script was edited. `scripts/gen_arenas.json` gained 4 entries, as the task asked.

## 1. Spread arenas: the rule and its output (written first)

`scripts/ag_spread_select.py` reads the E1 ranking of seeds 241-280 (`arenas/selection.json`), splits ranks 5-40 into
the quarters 5-13, 14-22, 23-31 and 32-40, and takes the arena with the smallest BMP sha256 in each quarter. Before it
chose, it re-read every candidate's BMP hash from the generated files in `/tmp/ag_e1/gen_a` and found them equal to
`selection.json`. It wrote `arenas/selection_spread.json` before anything else was generated. That file also holds a
replacement rule, declared with the choice: an arena that fails a gate is replaced by the next-smallest hash in its
quarter. No replacement was needed, because every gate below passed.

| quarter | chosen | rank | distance to f104 | nearest training arena (distance) | distance to f104 / g203 / g228 |
|---|---|---|---|---|---|
| 5-13 | **g258** | 6 | 1.110 | g228 (0.879) | 1.110 / 1.396 / 0.879 |
| 14-22 | **g268** | 17 | 1.317 | g228 (0.917) | 1.317 / 1.133 / 0.917 |
| 23-31 | **g263** | 23 | 1.458 | g203 (1.194) | 1.458 / 1.194 / 1.237 |
| 32-40 | **g241** | 38 | 1.791 | g228 (1.341) | 1.791 / 1.813 / 1.341 |

For comparison, the near arenas (distance to f104, then nearest training arena and its distance): g260 0.653, g228
0.606; g271 0.739, f104 0.739; g251 0.779, f104 0.779; g247 0.906, g228 0.721. Distances between arenas use the same
8 statistics and gen_v1 scales as the ranking (`ag_arena_rank.distance`).

Arena statistics (slopes in degrees; the soil friction angle is 38.7 deg):

| arena | slope cap | roughness m @ corr. m | hills / craters | height range m | max / p99 slope | flat < 5 deg | BMP sha256 | map observation sha256 |
|---|---|---|---|---|---|---|---|---|
| g258 | 27.5 | 0.264 @ 2.76 | 5 / 5 | [-1.90, 4.60] | 31.8 / 27.3 | 0.27 | `140478df62eedc9a` | `d4659e4ece39c7c9` |
| g268 | 31.7 | 0.218 @ 2.91 | 6 / 6 | [-2.00, 4.80] | 36.1 / 28.7 | 0.31 | `2d52cdb7877efec2` | `1ec5c8cf1d428cfe` |
| g263 | 27.4 | 0.165 @ 2.08 | 6 / 6 | [-1.80, 3.80] | 32.9 / 26.4 | 0.31 | `31bce710ad19f30a` | `68b28c7c366eac64` |
| g241 | 26.5 | 0.196 @ 2.74 | 5 / 6 | [-1.80, 4.00] | 32.1 / 25.0 | 0.40 | `028a3290941c30f9` | `cb86b83b49c02cf9` |

g268's steepest cell (36.1 deg) is 2.6 deg below the friction angle. The E1 arenas were 2.7-6.8 deg below.

## 2. Preparation of the four arenas

All the gates below pass. Details are in `arenas/e1b_checks.json`, written by `scripts/ag_e1b_checks.py`.

**Copies and determinism.** The four arenas were copied to `assets/traverse/arena_{g258,g268,g263,g241}`, which hold only
`arena_000.bmp` and `arena_meta.json`. I regenerated them once more into `/tmp/ag_e1b/gen_b` with the E1 command. The BMP
and metadata sha256 are identical across the first generation, the regeneration, the copies and `selection.json`. All
four were added to `scripts/gen_arenas.json`, which now has 14 entries, each equal to its BMP.

**Maps.** I made one capture per arena with `scripts/crm_capture_map_local.py` (local OptiX; 21-22 s each) into
`maps/arena_<a>/`.
- The native-height p95 is 0.022-0.023 m, against a limit of 0.05 m. All corners are visible and the RGB-D is finite
  (4x512x512). The valid fraction is 0.698-0.701.
- The camera block is identical to the f104 soil map's. So are the capture-script hash and the Chrono commit
  (a92c6f72).
- The observation's BMP and metadata hashes match the assets.

**Map roots and grids.** The map roots are `map_roots/<a>/static_map_v1 -> ../../maps/arena_<a>`, and each loads with
`f104_n2_dataset.init_map`. The metric grids are `grids/arena_<a>/` (`sensor_map_v2.py`, run on the four new maps only).
Every grid covers 100 % of the arena. Against the BMP at Chrono's 511/512 scale the rmse is 0.0036-0.0043 m (the E1
arenas: 0.0038-0.0041). With y flipped it is 1.06-1.42 m, so the grids are oriented correctly.

**Soil surface smoke.** I ran `crm_smoke.py --check-surface --spacing 0.16` under `flock /tmp/luffy_crm.lock`, with f104
first as the control, into `smoke/<a>/` and `smoke/e1b_control_f104/`.
- The surface rmse as-is is 0.0473 (g258), 0.0476 (g268), 0.0472 (g263) and 0.0471 m (g241). The f104 control is
  0.0468 m, the same value as E1's.
- The y-mirrored control is 1.17-1.54 m.

**Settle check at production spacing.** I also ran the 0.08 m settle check (depth 0.24 m, step 1 ms, 3 s), as E1 did,
into `smoke/prod_spacing_0.08/<a>/`.
- Each arena had 4,008,004 soil particles and 3,006,003 boundary markers.
- The HMMWV stays 0.46-0.82 m above the BMP surface and drives off: 2.7-3.3 m/s at 3 s and 2.6-4.0 m from the start.
- It ran at 0.56-0.57x real time.

**Rigid launch smoke.** This is new in E1b. The unmodified `gen_collect_ext.py --local` drove route_00 of group 0000 on
each arena with a 6 s horizon, so that no outcome is learnt. Results are in `smoke/rigid_e1b/`.
- The allowlist check passes, and the launch check passes (states finite, roll and pitch below 0.07 rad).
- The native-height check passes (p95 0.019-0.024 m against a limit of 0.08 m).
- Each wall time was 20-25 s.

## 3. Suites (declared in `cases/README.md` before generation)

`gen_cases.py --groups 250 --strata feature --prefix <a>_test_group --seed 2026092504<n> --wave arena_gator_suite`. I
regenerated each suite into `/tmp/ag_e1b/cases_re`, and all four are byte-identical.

| set | groups | strata (hill entry / crater entry / hill cross / crater cross) | features with no group | route_00 length p5/p50/p95 m | lock (first 16) |
|---|---|---|---|---|---|
| test_g258 | 250 | 40 / 44 / 83 / 83 | hill 1 (crater 7 has 1 group) | 29.1 / 39.6 / 50.5 | `a3c3e97464b47b1e` |
| test_g268 | 250 | 42 / 42 / 81 / 85 | hill 0 | 30.4 / 41.1 / 54.1 | `4e321b706c824567` |
| test_g263 | 250 | 42 / 42 / 84 / 82 | crater 9 | 29.3 / 37.3 / 50.5 | `752923119303b2a6` |
| test_g241 | 250 | 43 / 48 / 72 / 87 | none (crater 7 has 7 groups) | 29.7 / 40.0 / 50.6 | `4fffe2441b24d60e` |

- **Locks.** The locks and manifests are `suites/test_<a>.{SUITE_LOCKED.sha256,manifest.json}`, written by
  `ag_suite_lock.py` into a scratch folder and moved in. `suites/ALL_LOCKED_E1b.sha256` = `e2909946...` covers the 8 new
  files. E1's `ALL_LOCKED.sha256` (`c2d0022c...`) was deliberately left unchanged, because `ag_suite_lock.py` would
  otherwise have rewritten it. `ag_suite_lock.py --check` passes on all 13 locks.
- **Ids.** All 4,850 group ids across the K3 case sets are unique.
- **Uneven coverage.** As on the E1 suites, the generator's skip-after-1,000-failures rule leaves one feature without
  groups on three of the four arenas. That leaves fewer (arena, nearest feature) clusters for the bootstrap.

**The blacklist does NOT cover these suites (important).** E1's patterns name each arena: `g260_test_group_*` and so on,
7 in all. They catch **0 of the 1,000 new suite ids** in all three builder lists (`ci_train.SUITE_GROUPS`,
`ga_build_mixed.BLACKLIST`, `ci_a5data.SUITE_PATTERNS`). The manifests record 0 hits.

The task said the E1 patterns would cover `<arena>_test_group`, but they do not. The rule that existing scripts are not
edited stands, so I did not add the lines. Instead:
- New `scripts/ag_blacklist.py` holds all 14 patterns and provides `patch_builders()`. That call extends the three lists
  in place for the current process, and ci_a5data's equality check still holds; I checked that
  `ci_train.blacklisted('g258_test_group_0001')` goes from False to True. It also provides `assert_clean(ids)` and a
  command-line check. A copy is on the cluster as `G3/source/scripts/ag_blacklist.py` (a new file).
- The fix that needs permission to edit is one more pattern list in each of the three files:
  `['g258_test_group_*', 'g268_test_group_*', 'g263_test_group_*', 'g241_test_group_*']`, the same style as E1's lines.
  Until then, E4 must either call `ag_blacklist.patch_builders()` or check its outputs with `assert_clean`.
- The current risk is low. The training arenas are f104, g203 and g228, and E4 builders select by arena prefix. But the
  spread rigid rows would land in the same kind of output folder as training rows.

## 4. Declared subsets (`scripts/ag_declared_subsets.py`; order = md5 of the group id as a hex string, lowest first)

| file | contents |
|---|---|
| `suites/soil_unseen_subset.json` | 125 groups per unseen test arena, 8 arenas (near g260 g271 g251 g247; spread g258 g268 g263 g241) = 1,000 groups; per arena 8-11 features covered |
| `suites/f104_indist_200.json` | 200 of the 450 hill/crater groups of the 600 `f104_pair_group_*` (source `generalist_20260921/cases/pair_v1/cases`; the case files and route_00 are byte-identical in the 800-group suite folder `A_adapt/suite/cases`). Strata: hill cross 65, hill entry 30, crater cross 73, crater entry 32 |
| `e3/spread_headroom/spread_headroom_groups.json` | 75 groups per spread arena = 300 |
| `suites/SUBSETS_LOCKED_E1b.sha256` | `05039510...` over the three files |

The three subsets are nested because they use the same order: each spread arena's 75 headroom groups lie inside its 125
soil groups. So the straight 6 m/s drives of the headroom scan are also the straight 6 m/s arm of the soil evaluation on
those groups (same route, same id `<g>__straight6`). E6 can reuse them or re-drive them, but should decide which.

## 5. Spread headroom rows (the same frozen model, planner call and straight 6 m/s construction as the g217 dev check)

- **Models.** The frozen K1 soil specialist `crm_f104_v1/train_v1/deploy/CRM_N2_s{0..4}.pt`; all 5 hashes equal
  `e3/dev_headroom/models.sha256`.
- **Planner call.** Per arena, after `ag_map_check.py` passed (the map root's BMP equals the cases' arena):
  `ga_planner.py --cases K3/cases/test_<a>/cases --map-root K3/map_roots/<a> --models '...CRM_N2_s*.pt' --world crm
  --arms B --ref-picks /nonexistent_no_reference --task-root K3 --groups @e3/spread_headroom/groups_<a>.txt --out
  e3/spread_headroom/picks_crm_Scrm_<a>`. That is CEM 4 x 64 (tag `n2iter_cem4x64`), from the case pose at rest. Each
  arena took 33-62 s on the 5090.
- **Subset check.** The same call on 3 g217 dev groups reproduced E3a's 150-group dev picks byte for byte, so taking a
  subset of groups does not change the picks.
- **Straight 6 m/s.** Built by `scripts/ag_spread_headroom.py` with the exact code of `ag_dev_headroom.py`. It first
  rebuilt all 150 g217 dev straight6 routes and found them byte-identical to E3a's.
- **Rows.** `e3/tasks/spread_headroom_rows.json` (sha256 `8ea0fe10...`) has **597 rows, not 600.** In 3 groups
  (g258_test_group_0039, g268_test_group_0047, g263_test_group_0053) the specialist's pick is the straight 6 m/s route,
  so it is driven once and the row's `arms` lists both. The rows have the soil_v1 dev-row format: tier -1, kind
  `spread_headroom`, `vehicle: hmmwv`, ids `<g>__Scrm_B` / `<g>__straight6`, episode_seed = md5(id)[:8], and paths
  relative to G3. The ids and seeds are unique, and none clashes with soil_v1.json.
- **Dry run.** `ag_soil_tasks.py --head tasks_dev.json spread_headroom_rows.json --check-superset soil_v1.json` built a
  16,663-row superset in `/tmp`, with every check passing. That is how E3 can put them into the next soil file.
- **Locks.** Per-arena `picks_crm_Scrm_<a>/PICKS_LOCKED.sha256`, and the combined
  `e3/spread_headroom/SPREAD_PICKS_LOCKED.sha256` = `d1d12ca79ca04ed3...` over all 597 route files. Both were written
  before any drive.
- **Model-side hint only (not an outcome).** The ensemble's own mean predicted failure for its chosen route is 0.013
  (g258), 0.020 (g268), 0.006 (g263) and 0.086 (g241), against 0.004 on the g217 dev arena. Whether g241 has real
  headroom is for the drives to show.

## 6. Rigid rows for the spread test groups

`e3/tasks/rigid_spread_designed_rows.json` (+ `.meta.json`, sha256 `99b50c56...`) holds 12,000 rows: the 12 designed
routes of all 250 groups per spread arena.
- The format is the `test_designed` block of `rigid_hmmwv_v1.json`. The builder `scripts/ag_rigid_spread_rows.py`
  reproduces that block's 12,000 near-arena rows exactly (list equality) when given the near arenas and offset 12.
- Shards are 14 and 15, so the rows can be appended to rigid_hmmwv_v1 or run alone with `--array=14-15`.
- No id or shard clashes with rigid_hmmwv_v1.

**They cannot run from `G3/source` yet.** The rigid collector checks `scripts/gen_arenas.json`, and the cluster copy
(read-only, 10 entries) does not list g258/g268/g263/g241. The rigid job 436075, still waiting, reads that file, so it
must not be replaced while that job can still start tasks. E3 should either update it after 436075 has finished, or run
these rows from a second source tree that has the 14-entry file. The soil collector does not read the allowlist, so the
headroom rows can run from `G3/source` now.
[verifier E1b: "update it after 436075" is only allowed once no job that reads `G3/source` is running. Tonight's rule
forbids copying a file that is not new into `G3/source` while such jobs run, and at 03:00 the soil arrays 436080/436092
and the Gator pilot jobs also run from `G3/source`. Until they have all ended, the second source tree is the allowed way.]

## 7. Map-lookup error per arena (the covariate; `arenas/map_lookup_error.json`, `scripts/ag_map_lookup_error.py`)

This is the error of what the models see: the flat-ground lookup of the capture's elevation channel, minus Chrono's
native height, on the capture's 4,096 audit points. It is E1's measure; the script reproduces E1's 8 values exactly
(difference 0). "Features" means points within 2 sigma of a hill or crater centre, using the positions in Chrono's frame
(`TerrainMap.features`); `arena_meta` stores them y-mirrored.

| arena | role | rmse m | p95 m | max m | rmse on features | rmse elsewhere | distance to f104 | nearest training (distance) |
|---|---|---|---|---|---|---|---|---|
| f104 | training | 0.050 | 0.124 | 0.255 | 0.079 | 0.009 | 0 | g228 (0.811) |
| g203 | training | 0.062 | 0.159 | 0.360 | 0.096 | 0.008 | 0.815 | g228 (0.750) |
| g228 | training | 0.072 | 0.181 | 0.382 | 0.111 | 0.009 | 0.811 | g203 (0.750) |
| g217 | dev | 0.066 | 0.154 | 0.427 | 0.088 | 0.009 | 0.854 | f104 (0.854) |
| g260 | near | 0.060 | 0.130 | 0.481 | 0.095 | 0.008 | 0.653 | g228 (0.606) |
| g271 | near | 0.065 | 0.146 | 0.483 | 0.101 | 0.010 | 0.739 | f104 (0.739) |
| g251 | near | 0.068 | 0.169 | 0.395 | 0.105 | 0.011 | 0.779 | f104 (0.779) |
| g247 | near | 0.071 | 0.165 | 0.452 | 0.103 | 0.012 | 0.906 | g228 (0.721) |
| g258 | spread | **0.085** | 0.210 | 0.514 | 0.130 | 0.010 | 1.110 | g228 (0.879) |
| g268 | spread | **0.094** | 0.204 | 0.619 | 0.127 | 0.009 | 1.317 | g228 (0.917) |
| g263 | spread | 0.059 | 0.149 | 0.336 | 0.094 | 0.009 | 1.458 | g203 (1.194) |
| g241 | spread | 0.062 | 0.149 | 0.401 | 0.106 | 0.009 | 1.791 | g228 (1.341) |

For the training arenas, "nearest training" means the nearest other training arena. The error sits almost entirely on
the hills and craters (0.08-0.13 m there, about 0.01 m elsewhere). g258 and g268 have the largest errors of all 12 arenas,
and they also have the largest height ranges (6.5 and 6.8 m). The two farthest arenas, g263 and g241, have errors like f104's
siblings. So map error and distance from training are separate covariates here and are not simply confounded.

## 8. Staged on the cluster (new paths only; checked after the copy)

- `G3/source/assets/traverse/arena_{g258,g268,g263,g241}`: the BMP hashes match.
- `G3/cases/test_{g258,g268,g263,g241}`.
- `G3/maps/arena_<a>`, `G3/map_roots/<a>` (the links resolve), `G3/grids/arena_<a>`.
- `G3/e3/spread_headroom/`: picks, straight6 routes, group lists, locks.
- `G3/e3/tasks/spread_headroom_rows.json`, `G3/e3/tasks/rigid_spread_designed_rows.json(.meta.json)`.
- `G3/suites/`: a new folder with the subsets, their lock, and the 4 new suite locks and manifests.
- `G3/source/scripts/ag_blacklist.py`.

Every case and route path in both row files exists on G3, and all 597 route files match `SPREAD_PICKS_LOCKED`. Every
target path was checked to be absent before the copy, and rsync ran with `--ignore-existing`.

Not touched on the cluster: `G3/source/scripts/gen_arenas.json` (see section 6), `G3/cases/README.md` and
`G3/cases/case_checks.json` (older copies, not new files), and every task file in `G3/tasks`. No job ids, so
`submissions.tsv` is unchanged.

## 9. New and changed files

New scripts:
- `scripts/ag_spread_select.py`
- `ag_declared_subsets.py`
- `ag_spread_headroom.py`
- `ag_rigid_spread_rows.py`
- `ag_map_lookup_error.py`
- `ag_e1b_checks.py`
- `ag_blacklist.py`

Changed: `scripts/gen_arenas.json` (4 entries added) and `cases/README.md` (E1b section appended).

New artefacts in K3:
- `arenas/{selection_spread,map_lookup_error,e1b_checks}.json`
- `maps/arena_<a>` and `maps/logs/capture_<a>.log`
- `map_roots/<a>`, `grids/arena_<a>` and `grids/sensor_map_v2_e1b.log`
- `smoke/{<a>,e1b_control_f104,prod_spacing_0.08/<a>,rigid_e1b}`
- `cases/test_<a>` and `cases/logs/test_<a>.log`
- `suites/test_<a>.*`, `suites/ALL_LOCKED_E1b.sha256`, `suites/soil_unseen_subset.json`, `suites/f104_indist_200.json`
  and `suites/SUBSETS_LOCKED_E1b.sha256`
- `e3/spread_headroom/`
- `e3/tasks/spread_headroom_rows.json`, `e3/tasks/rigid_spread_designed_rows.json(.meta.json)`

New assets: `assets/traverse/arena_{g258,g268,g263,g241}/`.
