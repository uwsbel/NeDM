# NOTES E1: arenas, maps, soil check, cases, blacklists (2026-09-25, 01:18-01:30)

Everything below was made locally in the worktree (branch `arena_gator_v1`). K3 = this folder. Nothing was submitted
to the cluster. The cluster root G3 `/work1/dannegrut/harry/experiments/arena_gator_20260925` was created (empty).
No file was copied into any cluster source tree.

## 1. Test arenas (PLAN 1.3)

- Generated seeds 241-280 at difficulty 1.0 with `scripts/traverse_wp7_arenas.py --prefix arena_g
  --orientation-from assets/traverse/arena_f104_50h_v1` into `/tmp/ag_e1/gen_a` (40 arenas, 13.5 s). Family
  parameters: `arenas/family_g241_g280.json`.
- New script `scripts/ag_arena_rank.py` recomputes the gen_v1 distance (root mean square of 8 scaled differences to
  f104: slope cap, roughness, roughness correlation length, hill count, crater count, 99th-percentile slope, flat
  fraction, height range; scales from `fdm_f104_50h_20260909/gen_v1/arena_similarity.json`). Check: seeds 201-240
  regenerated into `/tmp/ag_e1/gen_ref`; all 40 stored distances reproduced exactly (max difference 0.0000, same
  order; computed from the metadata, not from the rounded features in the json). The f104 statistics equal the stored
  reference vector. The 9 regenerated seeds that already exist in `assets/traverse` (g203 g228 g217 g216 g231 g213
  g204 g234 g223) are byte-identical to them.
- Ranking of 241-280 and the choice are in `arenas/selection.json` (the script picked the 4 smallest distances; nothing
  else was looked at). A replacement rule was written with the choice (a chosen arena that fails a mechanical
  preparation gate is replaced by the next one in the ranking); no replacement was needed.

| rank | arena | distance to f104 |
|---|---|---|
| 1 | **g260** | 0.653 |
| 2 | **g271** | 0.739 |
| 3 | **g251** | 0.779 |
| 4 | **g247** | 0.906 |
| 5 | g277 | 1.100 |
| 6 | g258 | 1.110 |

Surprise (harmless): three of the four new arenas are closer to f104 than the closest gen_v1 sibling (g228, 0.811).

- Copied (only these four) to `assets/traverse/arena_g{260,271,251,247}` (the repo root is needed because
  `gen_cases.py` writes the arena path relative to it). Regenerated once more into `/tmp/ag_e1/gen_b`: BMP and
  metadata sha256 identical to the copies and to the first generation (recorded in `selection.json: determinism`).

Arena table (`max / p99 slope` in degrees; soil friction angle is 38.7 deg, so every arena stays 2.7-6.8 deg below it):

| arena | seed | distance | slope cap | roughness m @ corr. m | hills / craters | height range m | max / p99 slope | flat < 5 deg | BMP sha256 | map observation sha256 |
|---|---|---|---|---|---|---|---|---|---|---|
| f104 | 104 | 0 | 30.9 | 0.240 @ 2.22 | 5 / 5 | [-1.80, 3.90] | 34.0 / 28.6 | 0.25 | `5d5bc683b6a8d9e9` | `53e23f99bf2c0141` (crm_f104_v1) |
| g203 | 203 | 0.815 | 30.5 | 0.195 @ 2.09 | 6 / 5 | [-2.00, 4.15] | 35.4 / 29.1 | 0.28 | `46b4fb2c6d6c5dba` | `f799f9f6e99ce9bd` |
| g228 | 228 | 0.811 | 29.6 | 0.209 @ 2.50 | 5 / 5 | [-1.95, 4.40] | 34.4 / 28.1 | 0.32 | `c861bd7a7990e225` | `28bf7593d730f97f` |
| g217 | 217 | 0.854 | 31.0 | 0.244 @ 2.73 | 6 / 6 | [-2.00, 3.80] | 33.1 / 27.9 | 0.22 | `22bc058bb40e331d` | `10e9b7b9cd0f92b7` |
| g260 | 260 | 0.653 | 31.5 | 0.240 @ 2.70 | 5 / 5 | [-1.65, 4.40] | 35.8 / 28.7 | 0.30 | `a746091bac79f8b8` | `360d0bf32185d3b2` |
| g271 | 271 | 0.739 | 30.5 | 0.260 @ 2.60 | 6 / 5 | [-1.55, 4.30] | 33.6 / 27.4 | 0.23 | `de115241c7f4e9b4` | `3d12787f3c09898e` |
| g251 | 251 | 0.779 | 28.9 | 0.265 @ 2.22 | 5 / 6 | [-1.90, 3.95] | 31.9 / 27.4 | 0.20 | `29375393dd33ec89` | `70ebc9c74276d99c` |
| g247 | 247 | 0.906 | 30.3 | 0.246 @ 2.60 | 5 / 6 | [-1.80, 4.60] | 36.0 / 27.2 | 0.29 | `2931c436bdee76a9` | `7704d2d66a14f006` |

Full hashes: `scripts/gen_arenas.json` (BMP), `maps/arena_<a>/observation.json` (map).

## 2. Maps (PLAN 1.2)

- `scripts/crm_capture_map_local.py` (local OptiX, same backend string, Chrono commit `a92c6f72` and capture-script
  sha256 as the f104 soil map) for g203 g228 g217 g260 g271 g251 g247 into `maps/arena_<a>/` (21 s each, logs in
  `maps/logs/`). All built-in assertions pass: native-height p95 0.020-0.024 m (< 0.05), all corners visible, finite
  RGB-D; each `observation.json` BMP and metadata hash equals the arena in `assets/traverse`.
- Against the existing AMD (Vulkan lavapipe) captures `fdm_f104_50h_20260909/sensor_v1/maps/`: g203, g228, g217 have
  identical valid masks, max elevation difference 3.4-3.7e-5 m, max depth difference 5.3e-5 m (expectation < 1e-4 m
  met). Metric grids against `sensor_v2/grids`: identical coverage, rms z difference 3-5e-5 m, max 0.007-0.014 m (a few
  cells change bin, as the scout saw). Details `maps/map_checks.json`.
- Map roots `map_roots/<a>/static_map_v1 -> ../../maps/arena_<a>` (relative links); every root loads with
  `f104_n2_dataset.init_map`. f104 keeps `artifacts/traverse/crm_f104_v1/map_root`.
- Metric grids `grids/arena_<a>/grid.{npz,json}` via `scripts/sensor_map_v2.py --maps maps --out grids` (100 %
  coverage, 2.8 pixels per cell).
- Flat-ground lookup error of the model's map input against Chrono's terrain (the PLAN 1.2 caveat), same 4,096 audit
  points per arena (`maps/flat_lookup_error.json`): rmse / p95 / max = f104 0.050 / 0.124 / 0.255 m; g203 0.062 /
  0.159 / 0.360; g228 0.072 / 0.181 / 0.382; g217 0.066 / 0.154 / 0.427; g260 0.060 / 0.130 / 0.481; g271 0.065 /
  0.146 / 0.483; g251 0.068 / 0.169 / 0.395; g247 0.071 / 0.165 / 0.452. Every other arena is 20-45 % worse than f104
  on this measure. [verifier E1: 19-45 % (g260 is 1.19x f104's rmse); see VERIFY_E1.md]

## 3. Soil surface check

`crm_smoke.py --arena <dir> --check-surface --spacing 0.16` (task command, defaults otherwise: depth 0.24 m, step
0.5 ms, 4 s), local 5090, under `flock /tmp/luffy_crm.lock`, f104 run first as the control. Reports and logs in
`smoke/<a>/`; the marker dumps stayed in `/tmp/ag_e1/crm_smoke` (18 MB each).

| arena | surface rmse as-is (bias) | y-mirrored control | wall |
|---|---|---|---|
| f104 (control) | 0.0468 m (+0.0022) | 1.234 m | 13 s |
| g203 | 0.0470 m (+0.0025) | 1.550 m | 13 s |
| g228 | 0.0470 m (+0.0015) | 1.562 m | 13 s |
| g217 | 0.0471 m (+0.0009) | 1.222 m | 13 s |
| g260 | 0.0467 m (+0.0028) | 1.205 m | 13 s |
| g271 | 0.0469 m (-0.0000) | 1.148 m | 12 s |
| g251 | 0.0474 m (-0.0004) | 1.236 m | 11 s |
| g247 | 0.0472 m (+0.0024) | 1.417 m | 15 s |

The soil surface follows the BMP on every arena exactly as on f104
(rmse = lattice quantisation at 0.16 m; 201,601 surface columns each).

Surprise: in this coarse smoke the HMMWV falls through the thin particle layer on every arena, f104 included (chassis
3-25 m below the ground after 4 s). That is the 0.16 m / 0.24 m-deep lattice, not the arena. So I added a second,
production-setting settle check (spacing 0.08 m, depth 0.24 m, 1 ms step, 3 s; `smoke/prod_spacing_0.08/<a>/`): on all
8 arenas 4,008,004 soil particles + 3,006,003 boundary markers, the vehicle sits 0.36-0.79 m above the BMP surface and
drives off (3.1-4.3 m/s at 3 s on six arenas; on g251 and g271 the default start at (-30, -30) faces a 12-13 deg
climb over the first 6 m of the BMP and it creeps at 0.1-0.5 m/s; on f104 the same stretch is flat), 0.54-0.57x real time on the 5090. One short smoke per arena; not a collection.

## 4. Cases (declared seeds in `cases/README.md`; checks in `cases/case_checks.json`)

| set | folder | groups | strata | splits written in the cases (train/val/test) | lock sha256 (first 16) |
|---|---|---|---|---|---|
| g203 training pool | `cases/train_g203/cases` + `cases/train_g203_onpolicy` | 1,200 (14,400 designed + 9,600 on-policy routes) | all | 1,083 / 48 / 69 | `3ee462f7d1fbe9e1` |
| g228 training pool | `cases/train_g228/cases` + `cases/train_g228_onpolicy` | 1,200 (14,400 + 9,600) | all | 1,063 / 60 / 77 | `d08c30adeec6848f` |
| g260 test | `cases/test_g260/cases` | 250 | feature | (ids blacklisted) | `e437916bf8336835` |
| g271 test | `cases/test_g271/cases` | 250 | feature | | `21499c3e5556715e` |
| g251 test | `cases/test_g251/cases` | 250 | feature | | `98506e43dd0f4244` |
| g247 test | `cases/test_g247/cases` | 250 | feature | | `5aa8a3f0a355fd3e` |
| g203 held-out | `cases/heldout_g203/cases` | 150 | feature | | `728de5919b5d35fe` |
| g228 held-out | `cases/heldout_g228/cases` | 150 | feature | | `b2fc3bb173eb9800` |
| g217 dev | `cases/dev_g217/cases` | 150 | feature | | `f782bf42b5cbdef2` |

- Timings: training pools 75 s (g203) and 103 s (g228), suites 9-35 s each, on-policy routes 2 s per arena.
- On-policy routes (`f104_n2_onpolicy.py --n 8`): confirmed map-free (it reads only `route_00` and the start pose; the
  validator gets no obstacles and no terrain; `f104_n2_sampler` imports only numpy) and it uses
  `MPPIConfig(max_speed 6, curvature 0.125, arena_half_extent_m=40)`. 8 routes for every group; distribution matches
  f104's (max lateral median 4.24 / 4.29 m vs 4.35, mean speed median 2.04 / 2.03 vs 2.03 m/s).
- Held-out suites are >= 2.005 m (g228) and >= 2.012 m (g203) from every training pair of their arena in start+goal
  space (median 3.2 m). All 3,850 new group ids are unique.
- Strata of the training pools (round-robin, with the generator's skip-after-1,000-failed-attempts rule):
  g203 hill cross 309, crater cross 366, hill entry 155, crater entry 186, long 92, roughness 92;
  g228 330 / 275 / 170 / 140 / 185 / 100 (f104's pool: 330 / 266 / 170 / 136 / 198 / 100).
- Locks: `suites/<set>.SUITE_LOCKED.sha256` (suites) and `suites/<set>.CASES_LOCKED.sha256` (training pools incl.
  on-policy routes), written by the new `scripts/ag_suite_lock.py` (same scheme as `ga_suite.py`: line 1 = one sha256
  over relative path + content of every file, sorted; then one `sha256  path` line per file, paths relative to the
  repo root). Re-check: `python scripts/ag_suite_lock.py --check --out artifacts/traverse/arena_gator_20260925/suites`
  or from the repo root `tail -n +2 <lock> | sha256sum -c --quiet`. `suites/<set>.manifest.json` summarises each set
  (arena, BMP hash, map root and its observation hash, strata, seed, blacklist hits). `suites/ALL_LOCKED.sha256`
  = `c2d0022c5a31f44ef55129895e4eedea403e59abfe5635beb3dd98f5daf8c334` over all locks and manifests. Checked OK after
  writing.

## 5. Blacklists and allowlist (edits to existing files, additive, one appended line each)

- `scripts/ci_train.py` after line 54: `SUITE_GROUPS += [...]`
- `scripts/ga_build_mixed.py` after line 29: `BLACKLIST += [...]`
- `scripts/ci_a5data.py` after line 93: `SUITE_PATTERNS += tuple([...])` (its equality assertion with
  `ga_build_mixed.BLACKLIST` passes; checked)
- patterns: `g260_test_group_*`, `g271_test_group_*`, `g251_test_group_*`, `g247_test_group_*`,
  `g203_heldout_group_*`, `g228_heldout_group_*`, `g217_dev_group_*`. Every suite id matches in all three lists; no
  training id and none of 9,050 older case files match the new patterns.
- `scripts/gen_arenas.json`: added `arena_g260`, `arena_g271`, `arena_g251`, `arena_g247` (BMP sha256; the existing
  six unchanged and re-checked against the assets). The file gained a trailing newline.
- Not changed (not in the task): the regex blacklists in `gb_build_cache.py:44` and `ga_branch_anchors.py:59`
  (tracker / branch rows, f104-only tools).

New files: `scripts/ag_arena_rank.py`, `scripts/ag_suite_lock.py`, `assets/traverse/arena_g{260,271,251,247}/`.

## 6. Things later modules need to know

1. **Soil tier plan, g228 (PLAN 1.4 says "first 606 groups (>= 545 training groups)")**: that holds for g203 (559
   training groups in the first 606) but NOT for g228 (520). The smallest prefix with 545 training groups is 591 groups
   on g203 and **634 on g228** (363 training groups: 394 / 426). E3 should use at least the first 634 groups of g228
   (or 634 on both), or E4 must accept 520 g228 soil training groups.
2. Rigid training groups per arena are 1,083 (g203) and 1,063 (g228), not 1,089 as on f104 (the hash split lands
   differently); PLAN 2.1's "rigid A3 = 3 x 1,089" becomes 1,089 + 1,083 + 1,063 unless subset.
3. Nothing is staged on the cluster: the soil source tree `crm_f104_20260916/source/assets/traverse/` has only f104;
   the gen_v1 / generalist / crm_improve trees have g203/g228/g217 but none of g260/g271/g251/g247, and their
   `gen_arenas.json` copies list only the old six. Copy the arena folders and the updated `gen_arenas.json` (E3).
4. `routes.json` of the on-policy folders stores repo-relative paths
   (`artifacts/traverse/arena_gator_20260925/cases/train_<a>_onpolicy/routes/...`), as f104's did; the task builder
   must map them to cluster paths. Each case's `arena` field is `assets/traverse/arena_<a>` (relative to the source
   root).
5. `gen_cases.py` labels every `cases.json` `"campaign": "fdm_f104_50h_20260909/gen_v1"` and every case `family:
   f104_terrain_family`; cosmetic.
6. Test and held-out cases carry hash splits (mostly `train`); exclusion is by id pattern only (section 5). The old
   gen_v1 test ids (`g203_test_group_*` etc.) remain unblacklisted, as before; 5-7 held-out groups per arena lie within
   2 m of an old gen_v1 case on the same arena, which only matters if gen_v1 rows were ever added to training.
7. The planners and builders still do not check that a map root belongs to the case's arena; each map root's
   `observation.json: arena_bmp_sha256` is the value to compare against (the suite manifests record the pairing).
