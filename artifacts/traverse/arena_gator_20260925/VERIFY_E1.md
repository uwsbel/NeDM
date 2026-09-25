# VERIFY E1: arenas, maps, soil surface check, cases, blacklists, locks (2026-09-25, 01:31-01:38)

Independent re-check of module E1 (builder notes: `NOTES_E1.md`). I recomputed everything from the files with my own
code, not by re-running the builder's check scripts (except the lock checker, which I also ran). My scratch files are in
`/tmp/ag_v1/` (regenerated arenas and case sets, check scripts). No Chrono simulation was run and nothing was
submitted. The cluster was only listed.

**Verdict: pass.** Every claim I checked holds. One wording slip in the notes was corrected. Four caveats for later
modules are listed in section 8. Nothing here blocks E3 or E4.

## 1. Arena generation, ranking and choice

- I regenerated seeds 241-280 and 201-240 plus f104 into `/tmp/ag_v1` with the declared command. The family
  parameter file is byte-identical to `arenas/family_g241_g280.json`. f104 regenerates byte-identically, and so do the
  9 siblings already in `assets/traverse` (g203 g228 g217 g216 g231 g213 g204 g234 g223).
- **Distance, recomputed with my own code.** I used the 8 statistics from each arena's metadata and the gen_v1 scales.
  The f104 reference vector equals the stored one exactly. All 40 stored gen_v1 distances (seeds 201-240) are
  reproduced (max difference 0.0) in the same order.
- **Ranking of 241-280:** g260 0.6529, g271 0.7387, g251 0.7788, g247 0.9061, then g277 1.100 and g258 1.110. There is
  a clear gap after rank 4. My full ranking equals `selection.json` entry by entry (max difference 0.0, same order).
  The 4 selected arenas are the 4 closest. All are at difficulty 1.0 and 80 m.
- **Copies.** `assets/traverse/arena_g{260,271,251,247}` hold only `arena_000.bmp` and `arena_meta.json`. Both files are
  byte-identical to my fresh regeneration. No other arena of 241-280 was copied.
- **No earlier use.** None of seeds 241-280 appears in git history or in any earlier artefact json/md file.
- **Arena statistics table** (notes section 1): every value re-read from the metadata matches.

## 2. Allowlist (`scripts/gen_arenas.json`)

The 6 old entries are unchanged against HEAD. The 4 new entries (`arena_g260/g271/g251/g247`) equal the sha256 of the
BMPs in `assets/traverse`, and all 10 entries match their BMPs. The rigid collectors look the entry up by arena folder
name, which is the key used. The file is still valid JSON; it gained a trailing newline, which is harmless.

## 3. Maps, map roots, grids

- **7 captures** (g203 g228 g217 g260 g271 g251 g247). On all 7, the observation, BMP and metadata hashes in
  `observation.json` match the files. The camera block is identical to the f104 soil map
  (`crm_f104_v1/maps/arena_f104_50h_v1`), including the same OptiX backend string. The capture-script sha256 is the
  same as f104's and the Chrono commit is a92c6f72. The RGB-D array is 4x512x512 and finite.
- **Assertions re-derived from the saved audit points:**
  - Native-height p95: 0.0204-0.0237 m (limit 0.05).
  - All 8 arena corners at min/max height project inside the 1024 px image.
- **Valid fraction:** 0.697-0.700 of the 512 px map on every arena, against 0.697 on f104.
- **Against the older AMD captures** (`fdm_f104_50h_20260909/sensor_v1/maps`) for g203, g228 and g217:
  - The valid masks are identical.
  - Max elevation difference is 3.4-3.7e-5 m and max depth difference 5.3e-5 m.
  - The camera blocks differ only in the backend name and a descriptive text field.
  - RGB is identical to within 1 grey level, except 2 pixels on g217. The model reads only the elevation channel and
    the valid mask, so this does not matter.
- **Map roots:** every `map_roots/<a>/static_map_v1` link resolves to its own capture. `f104_n2_dataset.init_map`
  loads each one (512 px, 0.18683 m per pixel).
- **Flat-lookup error of the model's map input,** recomputed on the same 4,096 audit points: f104 0.0499, g203 0.0616,
  g228 0.0723, g217 0.0663, g260 0.0596, g271 0.0654, g251 0.0677, g247 0.0708 m rmse. This matches
  `maps/flat_lookup_error.json`.
- **Metric grids:**
  - All 7 have 100 % coverage.
  - Against the BMP at Chrono's 511/512 scale the rmse is 0.0038-0.0041 m. With the y axis flipped it is 1.08-1.43 m,
    so the grids are oriented correctly.
  - For g203, g228 and g217 against `sensor_v2/grids`: the covered cells are identical, z rmse 2.6-4.8e-5 m, max
    0.007-0.014 m.
  - The per-cell pixel counts differ in 10-32 cells, which is why the covered-cell mask is identical but the raw count
    arrays are not.

## 4. Soil surface check (coarse lattice) and production-spacing settle runs

- All 16 logs load their own arena's BMP.
- **Coarse surface check (0.16 m):**
  - Surface rmse as-is: f104 0.0468, g203 0.0470, g228 0.0470, g217 0.0471, g260 0.0467, g271 0.0469, g251 0.0474,
    g247 0.0472 m.
  - With the y axis mirrored: 1.15-1.56 m.
  - 201,601 surface columns each. This matches the notes table.
  - The vehicle does fall through the coarse lattice on every arena, f104 included, so this is a property of the
    lattice, not of the new arenas.
- **Production spacing (0.08 m, 1 ms step):**
  - 4,008,004 soil particles and 3,006,003 boundary markers on all 8 arenas, running at 0.54-0.57x real time.
  - The chassis stays 0.36-0.79 m above the BMP surface throughout.
  - The vehicle reaches 3.1-4.3 m/s at 3 s on six arenas and creeps on g271 and g251.
  - I checked the builder's explanation for the creeping: along the smoke's heading from (-30, -30), the first 6 m rise
    1.32 m on g271 (12.4 deg) and 1.43 m on g251 (13.4 deg), against 0.07 m on f104.

## 5. Cases

- **Byte-exact regeneration.** I regenerated all 9 case sets and both on-policy route sets from the seeds declared in
  `cases/README.md`, with the declared prefixes, strata, waves and avoid folders. All 11 are byte-identical to the
  stored files (every case, route and `cases.json`).
- **Counts:**

| set | groups | designed routes | strata | splits in the case files (train/val/test) | arena field |
|---|---|---|---|---|---|
| train g203 | 1,200 | 12 each | all | 1,083 / 48 / 69 | `assets/traverse/arena_g203` |
| train g228 | 1,200 | 12 each | all | 1,063 / 60 / 77 | `assets/traverse/arena_g228` |
| held-out g203 / g228 | 150 / 150 | 12 each | hill/crater only | (blacklisted by id) | own arena |
| dev g217 | 150 | 12 each | hill/crater only | | own arena |
| test g260 / g271 / g251 / g247 | 250 each | 12 each | hill/crater only | | own arena |

- **On-policy routes:** exactly 8 per group for all 1,200 groups of each pool (9,600 per arena). Every path in
  `routes.json` exists, and the largest waypoint coordinate is 38.4 m (inside the 40 m bound). The medians are max
  lateral 4.24 / 4.29 m and mean speed 2.04 / 2.03 m/s, as in the notes.
- **Ids:** all 3,850 group ids are unique and consecutive per set. The case and route sha256 values in every
  `cases.json` match the files.
- **2 m separation** (start + goal, 4-D):
  - Held-out g203 against all 1,200 g203 training pairs: minimum 2.0125 m, median 3.16 m, 0 pairs under 2 m.
  - Held-out g228 against g228: minimum 2.0051 m, median 3.215 m, 0 under 2 m.
  - The test arenas and the dev arena have no training pairs in this study, so the condition holds there trivially.
  - Within each set, pairs are at least 2.0 m apart.
- **Soil prefix counts (notes section 6.1), confirmed:**
  - The first 606 groups hold 559 training groups on g203 and 520 on g228.
  - The smallest prefix with 545 training groups is 591 groups on g203 and 634 on g228.
  - The smallest prefix with 363 training groups is 394 on g203 and 426 on g228.

## 6. Blacklists

- One line was appended to each of `ci_train.py`, `ga_build_mixed.py` and `ci_a5data.py`. I imported all three
  modules:
  - The three pattern lists are equal as sets (10 patterns: the 3 old f104 patterns plus the 7 new ones).
  - `ci_a5data`'s own equality check passes.
- **Suite ids are caught.** All 1,450 new suite group ids are caught by all three lists, and so are their episode ids
  (`<group>_route_NN`, `_op_NN`).
- **Nothing else is caught.**
  - No training-pool group id and none of their 24,000 episode ids (plain or with an `@crm` suffix) is caught.
  - None of the 8,142 older group ids in earlier `cases.json` files is caught by the new patterns.
  - The 1,200 f104 training groups are not caught.

## 7. Lock files

- Every `*LOCKED.sha256` in `suites/` passes `sha256sum -c` from the repo root with 0 failures. I also recomputed the
  first-line combined hash with my own code, and it matches for all 10 files, including
  `ALL_LOCKED = c2d0022c5a31f44ef55129895e4eedea403e59abfe5635beb3dd98f5daf8c334`.
- The builder's `ag_suite_lock.py --check` exits 0.
- **File counts:** each lock lists exactly the expected number of files.
  - Suites: 1 + 13 x groups.
  - Training pools: plus 1 + 8 x groups on-policy files, 25,202 each.
  - The lists match every json on disk in those folders.
- **Manifests:**
  - Each manifest's lock hash equals its lock file.
  - Each manifest's map BMP hash matches its arena.
  - Blacklist hits are 150 or 250 per suite and 0 for the training pools.
- `PLAN.md` still matches `PLAN.sha256`.

## 8. Corrections made and caveats for later modules

**Correction (small, made directly).** In `NOTES_E1.md` section 2, "every other arena is 20-45 % worse than f104" on
the flat-lookup error is really 19-45 % (g260 is 1.19x). I added an annotation in place. No other file was changed by
me.

**Caveats (no fix needed in E1):**

1. **Soil prefix (confirmed).** Use at least the first 634 g228 groups (591 on g203) if the matched designs need 545
   soil training groups per new arena. The alternative is for E4 to accept 520 on g228. This is a plan deviation that
   E3 must record.
2. **Side effect of the blacklist line in `ci_a5data.py`.** Its "which case set is this" check now accepts a folder of
   new suite groups as the f104 "suite" set. Before, it stopped with an error. The tool would then write pass-1 rows
   pointing at the f104 cluster case folder and grade approaches on the f104 grid unless `--grid` is given. The plan
   uses a standing start and does not call this tool, so nothing is affected now. If a moving-start read-out is ever
   added on the new arenas, use a parameterised copy, not this script. The line cannot simply be removed, because the
   script asserts that its list equals the builders' blacklist.
3. **Uneven feature coverage in the suites.** This follows from the generator's skip-after-1,000-failures rule and
   matters for the planned (arena, nearest feature) cluster bootstrap in E6: fewer effective clusters on some arenas.
   - g251: two hills have only 1 and 2 groups, and two of its six craters have none. The suite has 163 hill groups
     against 87 crater groups.
   - g260: one crater has no groups.
   - g271: one hill has 3 groups.
   - g247: every feature is covered.
   - Held-out g203: one hill and one crater have none.
   - Held-out g228: one crater has 1 group.
   - Dev g217: 3 of its 12 features have none.
4. **Staging (confirmed empty).** The cluster root exists and is empty. The cluster soil source tree has only f104.
   - The map roots are relative symlinks (`map_roots/<a>/static_map_v1 -> ../../maps/arena_<a>`), so copy `maps/`
     together with `map_roots/` (or dereference the links) when staging.
   - On-policy `routes.json` paths are relative to the repo, so the task builder must map them to cluster paths.

Old gen_v1 cases lie within 2 m of 5-7 held-out groups per arena and of some dev groups on g217. This matters only if
gen_v1 rows were ever added to training, and the plan does not add them. The frozen f104-only soil model used for the
g217 headroom check never saw g217.
