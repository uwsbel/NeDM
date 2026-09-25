# VERIFY E1b: spread test arenas, suites, subsets, headroom rows, rigid rows, map-lookup error (2026-09-25, 02:50-03:06)

This is an independent re-check of module E1b (the builder's notes are `NOTES_E1b.md`). Where I could, I recomputed
results with my own code or through a different code path from the builder's. My scratch files are in `/tmp/ag_v1b/`.
I ran no Chrono simulation and submitted no job. On the cluster I only listed and hashed files.

**Verdict: pass.** Every claim I checked holds, and every file that can be regenerated was regenerated
byte-identically. The one real gap is the one the builder reported: the three existing builder blacklists do not cover
the new suites. That needs the user's permission to fix and has a working stop-gap (section 6). I made two wording
additions to `NOTES_E1b.md`, listed in section 11.

## 1. Selection rule, and whether it was recorded before use

- I regenerated seeds 241-280 into `/tmp/ag_v1b/gen` with the declared command. The family file is byte-identical to
  `arenas/family_g241_g280.json`.
- I recomputed the 8-statistic distance with my own code. The full ranking of the 40 arenas equals `selection.json`
  entry by entry: same order, same distances, same BMP and metadata hashes.
- **Rule** (PLAN 7.1): split ranks 5-40 into quarters (5-13, 14-22, 23-31, 32-40) and take the smallest BMP sha256
  in each quarter. My own sort picks g258 (rank 6), g268 (17), g263 (23) and g241 (38). That is the same choice, with
  the same hash order inside every quarter as `selection_spread.json`.
- Re-running `ag_spread_select.py` on my regeneration gives the same file. Apart from the time stamp, the only
  difference is the folder name it records.
- **Recorded first.** `selection_spread.json` was written at 02:25:08, and its script hash is unchanged. The first
  copies to `assets/traverse` came at 02:25:26, the maps at 02:26-02:27, the grids at 02:27 and the cases at 02:28.
  The replacement rule is in the file, and no replacement was made.
- **Distances to the training arenas, recomputed:**

| arena | to f104 | nearest training arena (distance) |
|---|---|---|
| g258 | 1.110 | g228 (0.879) |
| g268 | 1.317 | g228 (0.917) |
| g263 | 1.458 | g203 (1.194) |
| g241 | 1.791 | g228 (1.341) |

  The near arenas are 0.653-0.906 from f104 and 0.606-0.779 from their nearest training arena. So the builder's
  remark holds: measured against the nearest training arena, the near and spread arenas sit closer together than
  they do measured against f104.
- **Arena statistics table** (notes section 1): every value re-read from the metadata matches. g268's steepest cell
  (36.1 deg) is 2.6 deg below the soil friction angle.

## 2. Arenas byte-exact, and the allowlist

- In `assets/traverse/arena_{g258,g268,g263,g241}`, `arena_000.bmp` and `arena_meta.json` are byte-identical to my
  regeneration.
- No other arena from seeds 241-280 was copied. `assets/traverse` holds the 9 older siblings plus the 8 new arenas.
- `scripts/gen_arenas.json` has 14 entries, and each equals the sha256 of its BMP. Against HEAD the old 6 entries are
  unchanged, and the change adds lines only.
- `git diff` shows only E1's one-line additions in `ci_train.py`, `ga_build_mixed.py` and `ci_a5data.py`. E1b edited
  no existing script. All E1b scripts are new `scripts/ag_*` files.

## 3. Maps, map roots, grids

I checked all 12 arenas with my own code.

- The observation, BMP and metadata hashes all match.
- The camera block, the capture-script hash and the Chrono commit (a92c6f72) are identical to the f104 soil map's.
- The RGB-D array is 4x512x512 and finite.
- **Spread arenas:**
  - Native-height p95 is 0.0216-0.0234 m (limit 0.05).
  - All 8 arena corners project inside the image (pixels 64-959).
  - The valid depth fraction is 0.699-0.701.
  - Each map root is a relative link to its own capture and loads through `f104_n2_dataset.init_map`
    (512 px, 0.18683 m per pixel).
  - On the cluster the links resolve too.
- **Grids.** I read each grid at the 4,096 audit points and compared it with Chrono's own height there.
  - Error on the four spread arenas: 0.0049-0.0053 m; on the E1 arenas: 0.0050-0.0054 m.
  - With y flipped: 1.12-1.49 m, so the orientation is right.
  - Coverage is 100 %.

## 4. Soil smokes

- **The logs.** Each log loads its own arena's BMP. They report surface error 0.0473 / 0.0476 / 0.0472 / 0.0471 m
  (g258 / g268 / g263 / g241) against 0.0468 m for the f104 control. With y mirrored the error is 1.17-1.54 m.
- **Independent recheck.** I recomputed the surface fit from the particle dumps (`/tmp/ag_e1b/crm_smoke`). I took the
  top particle of every column and compared it with Chrono's native height at the audit points: 0.0377 / 0.0376 /
  0.0384 / 0.0394 m, against 0.0383 m on f104. With y mirrored it is 1.17-1.52 m. So the soil surface follows the
  terrain on the new arenas exactly as it does on f104.
- **Settle check at 0.08 m spacing:**
  - 4,008,004 soil particles and 3,006,003 boundary markers on each arena.
  - The chassis stays 0.46-0.82 m above the ground.
  - At 3 s the vehicle is doing 2.7-3.3 m/s and is 2.6-4.0 m from the start.
  - It ran at 0.56-0.57x real time.
- **Rigid launch smoke:**
  - On all 4 arenas the launch check passed (roll and pitch at most 0.068 rad) and the native-height check passed
    (p95 0.019-0.024 m).
  - The run stopped on the 6 s limit.
  - These smokes drove route_00 of each suite's group 0000 for 6 s, with no outcome. I added this to the notes'
    first paragraph.

## 5. Suites: counts, regeneration, separation

- **Regeneration.** I regenerated the four suites from the seeds declared in `cases/README.md` (`2026092504<n>`,
  `--strata feature`, wave `arena_gator_suite`). All 3,251 files per suite are byte-identical.
- **Declared before generation.** The README was last written at 02:28:07, and the suite folders were created at
  02:28:13. The README's heading says "declared 02:30", which is off by two minutes. I left the README unedited so
  that its file time stays as evidence.
- **Counts:** 250 groups per suite, 12 designed routes each, own arena in every case. The case and route hashes in
  `cases.json` match the files. Strata (hill entry / crater entry / hill cross / crater cross):

| suite | strata | features with no group | route_00 length p5 / p50 / p95 (m) |
|---|---|---|---|
| g258 | 40 / 44 / 83 / 83 | hill 1 (crater 7 has 1 group) | 29.1 / 39.6 / 50.5 |
| g268 | 42 / 42 / 81 / 85 | hill 0 | 30.4 / 41.1 / 54.1 |
| g263 | 42 / 42 / 84 / 82 | crater 9 | 29.3 / 37.3 / 50.5 |
| g241 | 43 / 48 / 72 / 87 | none (crater 7 has 7 groups) | 29.7 / 40.0 / 50.6 |

  All of this equals the notes.
- **Ids.** All 4,850 group ids across the K3 case sets are unique.
- **2 m separation** (start + goal):
  - Inside each spread suite, the closest pair of groups is 2.056-2.186 m apart.
  - No training pairs exist on these arenas, so a train/test separation does not apply there.
  - The 200 f104 in-distribution groups are at least 2.004 m from every one of the 1,200 f104 training groups
    (median 3.23 m).
- **Feature frames.** I confirmed the builder's note that the metadata stores hill and crater positions mirrored in
  y. At the mirrored centres, hills are on average 2.0-2.6 m high (craters counted as depth). At the stored positions
  the value is about 0.

## 6. Blacklists

- **The three builder lists catch none of the new ids** (`ci_train.SUITE_GROUPS`, `ga_build_mixed.BLACKLIST`,
  `ci_a5data.SUITE_PATTERNS`, 10 patterns each). That is 0 of the 1,000 new group ids and 0 of 14,000 episode-style ids
  (`_route_NN`, `__Scrm_B`, `__straight6`). The builder's report is correct.
- **What does catch them.** `ag_blacklist.ALL` (14 patterns), `ag_build_ds.suite_hit` (E4's dataset builder: its
  generic `*_test_group_*` rule, applied as a hard stop) and the three lists after `ag_blacklist.patch_builders()`
  each catch 1,000 of 1,000 groups and 14,000 of 14,000 episode ids. After the patch the four pattern sets are equal.
- **No training id is caught** by any of these. I tested 147,600 ids:
  - the g203 and g228 training groups with their designed and on-policy episode ids;
  - the 1,200 f104 groups and their 24,000 soil training ids, with and without `@crm`;
  - the `gator__` / `gatorR8__` ids;
  - the 48,000 rigid training rows.
- `ag_blacklist.ALL` also catches every one of the 2,450 K3 suite groups (near, spread, held-out, dev).
- The cluster copy `G3/source/scripts/ag_blacklist.py` equals the local file (`82975af9...`).
- **Still open, needs the user's permission:** one more pattern line in each of the three existing scripts. Until
  then, the protection is E4's `ag_build_ds.py`, which refuses any `*_test_group_*` id.
  - `ci_train.py`'s own check at training time would not stop a spread-suite row.
  - Any training file not built by `ag_build_ds.py` should be checked with `ag_blacklist.assert_clean` on its group
    ids before E5 trains on it.

## 7. Lock files

- Every `*LOCKED*.sha256` in `suites/` (16 files) passes `sha256sum -c` from the repo root. The first-line combined
  hash, which I recomputed with my own code, matches in every file.
- `ALL_LOCKED_E1b.sha256` (`e2909946...`) covers exactly the 8 new locks and manifests.
- E1's `ALL_LOCKED.sha256` is unchanged (`c2d0022c...`, 18 files, no spread file).
- The four per-arena `PICKS_LOCKED.sha256` files and `SPREAD_PICKS_LOCKED.sha256` (`d1d12ca7...`) verify. The latter
  lists exactly the 597 route files on disk.
- `SUBSETS_LOCKED_E1b.sha256` (`05039510...`) verifies.
- **Timing.** The subset files were written at 02:32:59, before the first headroom pick at 02:33. Their lock came
  later, at 02:42, but the content is unchanged, and the subsets are reproducible from the ids alone (section 8).
- **Cluster copies.** The 12 files in `G3/suites` are byte-equal to the local ones. The trees `G3/cases/test_<a>`,
  `G3/maps/arena_<a>`, `G3/grids/arena_<a>` and `G3/e3/spread_headroom` give the same tree hash as the local ones.

## 8. Declared subsets (md5 rules recomputed)

The rule is: order the group ids by md5 as a lowercase hex string and take the lowest.

- **`soil_unseen_subset.json`:** my 125 lowest per arena, for all 8 unseen arenas, equal the file (1,000 groups). The
  strata and features covered (8-11 per arena) match.
- **`f104_indist_200.json`:**
  - The pool is 450 hill/crater groups of the 600 `f104_pair_group_*` in `generalist_20260921/cases/pair_v1`. The
    other 150 are long-traverse and roughness groups.
  - My 200 lowest equal the file. Strata: hill cross 65, hill entry 30, crater cross 73, crater entry 32.
  - All 200 case files and route_00 files are byte-identical in the 800-group suite folder.
- **Headroom groups:** my 75 lowest per spread arena equal `spread_headroom_groups.json` and `groups_<a>.txt`. They
  lie inside that arena's 125 soil groups.

## 9. Spread headroom rows

- **Same model as the dev check.** The 5 `CRM_N2_s*.pt` hashes equal `e3/dev_headroom/models.sha256`. Each arena's
  planner summary matches the g217 dev summary on every setting: world crm, arm B, CEM 4 x 64 tag, deployed tag,
  model glob, members, float16 scoring, corridors, pose and base sources.
- **Picks reproduced.** I re-ran `ga_planner.py` with the declared command on all four arenas. The picks and routes
  are byte-identical, and so are the four `PICKS_LOCKED` hashes. The builder's subset evidence (3 g217 groups
  planned alone) is also byte-identical to the 150-group dev picks.
- **Straight 6 m/s through a different code path.** I ran E3a's own dev script `ag_dev_headroom.py`, not the new
  spread script, on the 75 headroom cases per arena. It rebuilt all 297 straight 6 m/s routes byte for byte. It found
  the same three groups where the pick equals the straight route (g258_test_group_0039, g268_test_group_0047,
  g263_test_group_0053).
- **Rows** (`spread_headroom_rows.json`, `8ea0fe10...`):
  - 597 rows (149 / 149 / 149 / 150), tier -1, kind `spread_headroom`, vehicle `hmmwv`, no `extra` (as the HMMWV
    soil_v1 rows).
  - Every field equals the rows the dev script writes, apart from the added `vehicle` and the case path, which points
    to `cases/test_<a>`.
  - `episode_seed = int(md5(id)[:8], 16)`. Ids and seeds are unique and do not clash with soil_v1, the Gator pilot
    or the drift file.
  - The route-content hash equals the row's `sha256` in 597 of 597 rows.
  - The dry-run superset in `/tmp/ag_e1b/soil_dryrun.json` keeps every soil_v1 row unchanged and in order.
- **Cluster files.** Every case and route path used by these rows and by the rigid rows (13,597 files) exists on G3
  with the same sha256 as locally. No spread-suite run exists yet in `G3/soil_v1`, `G3/rigid_v1` or
  `G3/pilot_gator`, so the picks were locked before any drive.
- **Model's own predicted failure for its pick** (from the planner logs, not an outcome): g258 0.013, g268 0.020,
  g263 0.006, g241 0.086; the g217 dev arena 0.004.
- **Overlap with the soil evaluation** (builder's point 3). This is confirmed and matters for E6. The headroom
  straight 6 m/s routes are the evaluation's straight 6 m/s arm on those 75 groups: same deterministic construction,
  same id. If E6 writes them with the same id into `soil_v1`, the workers treat them as done and skip them. E6 should
  decide this on purpose.

## 10. Rigid designed rows, and the map-lookup error table

**Rigid rows.** I rebuilt the rows with my own code from the rule I read off the near-arena block of
`rigid_hmmwv_v1.json`:
- id `<g>_route_NN`, tier = route index, `shard = offset + md5(group) % 2`, `episode_seed = md5(id)[:8]`, split from
  the case, kind `test_designed`.
- With offset 12 this rule reproduces all 12,000 near rows field for field.
- With offset 14 it reproduces all 12,000 spread rows field for field, in the same order (shards 14/15 =
  6,072 / 5,928). The file hash is `99b50c56...`, and no id clashes with rigid_hmmwv_v1.

**Where the rigid rows can run.** The builder's warning is confirmed: `gen_collect_ext.py` refuses any arena that is
not in `gen_arenas.json` next to itself, and `G3/source/scripts/gen_arenas.json` is read-only with 10 entries (no
spread arena). The allowlist is not one of the manifest-checked frozen files, so a second source tree with the
14-entry file would pass the source gates.

**Map-lookup error.** I re-implemented the flat-ground lookup: bilinear reading of the elevation channel, the same
pixel convention, and the valid mask from the four neighbouring pixels. I also computed the feature regions myself,
mirroring the metadata positions in y.
- All 12 arenas reproduce `arenas/map_lookup_error.json` to 0 difference: rmse, p95, max, and the error on and off
  the features.
- The 8 E1 values in `maps/flat_lookup_error.json` are reproduced too.
- Spread arenas: g258 0.085, g268 0.094, g263 0.059, g241 0.062 m (f104 0.050).
- The error sits on the hills and craters (0.08-0.13 m there, 0.008-0.012 m elsewhere).
- The builder's reading that map error and distance from training are separate covariates is fair as a description
  (the two farthest arenas have ordinary errors), but with 8 test arenas it is a qualitative statement only.

## 11. Corrections made, and points for later modules

**Corrections (wording only, marked `[verifier E1b: ...]` in `NOTES_E1b.md`):**
1. Top paragraph: the 6-second rigid launch smokes did drive route_00 of each spread suite's group 0000, stopped on
   the 6 s limit with no outcome.
2. Section 6: "update `gen_arenas.json` after 436075 has finished" is only allowed once no job that reads `G3/source`
   is running. Tonight's rule forbids copying a file that is not new into that tree while such jobs run, and at 03:00
   the soil arrays 436080/436092 and the Gator pilot jobs also run from it. Until they have all ended, the second
   source tree is the allowed route.

**Points for later modules (no fix needed in E1b):**
1. **Blacklist (section 6):** ask the user to allow the one-line additions to the three existing scripts. Until then,
   build every training file with `ag_build_ds.py` or check it with `ag_blacklist.assert_clean`.
2. **Rigid spread rows (E3):** use a second source tree with the 14-entry allowlist, unless every `G3/source` job has
   ended.
3. **Test-group rows carry hash splits** (mostly `train`: 10,956 of the 12,000 rigid spread rows). Exclusion is by id
   pattern only, as E1's verifier noted for the near suites. No builder should select rows by split.
4. **Soil evaluation (E6):** decide whether to reuse the 297 headroom straight 6 m/s drives or re-drive them under a
   new id (section 9).
5. **Fewer bootstrap clusters.** Three of the four spread suites have one feature with no group, and g258 has one
   crater with a single group. That leaves fewer (arena, nearest feature) clusters.
6. **Soil friction margin.** g268 has the smallest margin to the soil friction angle of any test arena: its steepest
   cell is 2.6 deg below it.
