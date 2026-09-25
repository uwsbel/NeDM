# VERIFY E4: training files, subsets and the first rigid training (independent check, 2026-09-25 03:03-03:15)

**Verdict: PASS.** Every claim I re-tested holds. The rebuilt f104 file, the three rigid subset files and the running
training job are what the notes say. I found no defect in the data or the tools, so I changed no file of E4. Section 9
lists seven notes for the next modules. None of them blocks the running training.

All my scripts and results are in `verify_e4/`. I wrote my own checks. The only builder code I ran is the code under
test: `ag_build_ds.py` (small builds and refusal tests) and `ag_subset.py` (list-only runs). I also called the
unchanged earlier per-episode functions directly, without going through the builder. Local python is the `nedm`
environment. The cluster checks ran on the login node (numpy only) and inside job 436135 (one `srun --overlap` step,
see section 8).

## 1. Row counts and row contents for a sample of runs (`verify_e4/v2_twin_rows.py`)

I drew random runs, seeded:
- rigid: 150 of the 8,976 episodes that are new in this rebuild, plus 50 of the 15,024 twin episodes;
- soil: 80 of the 211 new episodes, plus 50 twins.

Three checks per run, starting from its raw folder:
- **My own anchor count.** I wrote my own projection onto the route and my own event rule, from the re-anchoring
  tool's description and its default settings. The rule: the standing start, plus at most 3 moving starts. A moving
  start is a multiple of 40 frames before the event, with at least 12 m of route left, less than 1 m off the route,
  and not parked.
- **The unchanged tools on the raw folder.** I called the re-anchoring tool's per-episode function and
  `ga_build_mixed`'s history cutter in process.
- **Array comparison with the file's rows of that episode.** Corridors, context, energy and time targets, fail /
  unsafe labels, event index, split, group, status, id, history window, mask, privileged context.

| world | runs | same row count (own count = file = tool) | same anchor frames | rows identical | history identical |
|---|---|---|---|---|---|
| rigid | 200 | 200 | 200 | 200 | 200 |
| soil | 130 | 130 | 130 | 130 | 130 |

**Whole-file totals** (from the record, consistent with the table above):
- rigid: 93,397 rows from 24,000 episodes (3.89 per episode);
- soil: 58,268 rows from 15,235 episodes (3.82 per episode).
- The soil split counts equal the soil check's own record (`crm_f104_v1/collect_v1/qa.json`): 13,821 / 713 / 701
  episodes. It validated 15,235 of 15,236 runs, and the one it flagged is not in the local folder.
- The rigid and soil run folders hold exactly the episodes in the file, no more and no fewer (`v1`).

**Small builds of my own** (`verify_e4/v4_small_builds.py`). I ran `ag_build_ds.py` on 40 random rigid and 30
random soil f104 runs, three times: rigid alone, soil alone, and both worlds. Every array of the 157 + 112 + 269 rows
equals the big file's rows of those episodes. The big file has no further rows on them. This shows two things:
- rows depend only on their own episode;
- the one-world path gives exactly the two-world file's rows, independently of the builder's own check.

The current local builder (`5e2eae03...`) reproduces the big file on these samples. That settles the builder's point 5:
the record shows the builder's hash at build start, and the later edits changed nothing.

## 2. Arena, vehicle and tier columns (`verify_e4/v1_columns_subsets.py`, `v3_cluster_tests.py`)

- **Tier.** Every one of the 151,665 rows carries the tier of its id in the f104 soil task file
  (`crm_f104_20260916/tasks_train.json`, the order the soil collection ran in): 0 rows without a task row, 0
  mismatches. The tier is also constant within each episode.
  - Soil standing-start tier counts: tiers 1-11 on 1,200 groups; tier 0 on 1,199 (the group without it is
    `f104_v2_group_0211`, the QA-flagged run); tier 12 on 836.
  - Rigid: 20 tiers x 1,200.
- **Arena and vehicle.** The columns are `f104` and `hmmwv` only. I read the case and outcome records of all 39,235
  runs:
  - every case names `assets/traverse/arena_f104_50h_v1`;
  - no outcome record has a vehicle block;
  - case group and case split equal the file's group and split on every episode.
- **Cluster test files** (recomputed on the cluster from the raw runs):
  - g203 soil partial build: 298 episodes, all with a row in `tasks/soil_v1.json`, tier mismatches 0, cases name
    `arena_g203`, no vehicle block.
  - Gator pilot build: every soil episode's tier equals its pilot task row and the f104 order. Every rigid episode's
    tier equals the f104 order. Cases name f104. Every outcome record says `gator`.
  - The re-anchored row count equals the file's row count in both builds.
  - A random 25 episodes per world and build, recomputed with the unchanged tools, are array-identical
    (`verify_e4/v7_cluster_rows.json`). That covers a new arena's map (g203) and Gator runs as well.
  - One difference (note 9.3): the pilot's rigid task file numbers its tiers by route index, not by the per-group
    order.

## 3. Blacklist: no suite id in any training file

I checked the f104 file and the three subset files, locally and (by sha256) on the cluster.

**Patterns (25 in all):**
- the 14 study patterns of `ag_blacklist.py`, including the 4 spread-arena suites that the builder lists lack;
- `ga_build_mixed`'s list;
- the builder's generic patterns;
- broader ones of my own (`*test*`, `*eval*`, `*heldout*`, `*dev_group*`, `*_t2_group_*`).

Result: 0 hits among the 25,200 distinct names (24,000 episodes + 1,200 groups; every row id is its episode name
plus the anchor and world suffix).

**Suite names.** I collected the 3,435 group names in the suite case folders:
- the 800-group f104 suite, the 600 pair groups, the 200 soil evaluation groups, and the f104 final-test cases;
- the 8 unseen test suites, the g203/g228 held-out groups and the g217 dev groups.

None of them is a group of any training file. Every group matches `f104_v2_group_NNNN`, and every episode matches
`..._route_NN` / `..._op_NN`. The two cluster test files have no suite hit either.

**Beyond the ids (content, a note, not a leak).** No f104 suite case has both its start and its goal within 1 m of
a training case's start and goal. But 90 of the 800 suite groups have both within 2 m (`v8_suite_geometry.json`). So
do 37 of the 200 groups of the in-distribution reference (`suites/f104_indist_200.json`). The f104 suites predate
tonight and were not built with the 2 m avoidance rule that E1 used for the g203/g228 held-out groups. See note 9.7.

## 4. Map check

- The map root `crm_f104_v1/map_root` resolves to `maps/arena_f104_50h_v1` (observation `53e23f99...`). The BMP
  hash it records, `5d5bc683...`, equals the sha256 I computed of `assets/traverse/arena_f104_50h_v1/arena_000.bmp`,
  locally and in `G3/source`.
- The cluster copy `G3/e4/map_roots/f104/static_map_v1` is byte-equal (all three files) to the local map and to
  `crm_f104_20260916/static_map_v1`.
- That this is the right capture follows from sections 1 and 6. The tools, run with this map root, reproduce both
  the file rows and the night-2 rows exactly.
- **Refusal tests, re-run by me on 40 f104 rigid runs** (`/tmp/ve4`, exit 1 in all four cases):

| case | outcome |
|---|---|
| g203 map with f104 runs | refused: "map/arena check failed ... BMP 5d5bc683 != map 46b4fb2c" |
| HMMWV runs asked for as the Gator (id prefix forced empty) | refused: "40 runs whose vehicle block does not match --vehicle gator" |
| Gator asked for with the default prefix | refused: nothing selected |
| arena g203 asked for over f104 runs | refused: nothing selected |

  The last two stop with the message "runs name other arenas: {}". The build does stop, but the message is
  misleading when nothing was selected (note 9.1).

## 5. The one-world variant

Covered by section 1: my own one-world rigid and soil builds equal the two-world file's rows. The builder's own check
(`e4/f104_hmmwv/thin_variant_check.json`: rigid 93,397 and soil 58,268 rows, every array equal) agrees.

## 6. The f104 file reproduces the earlier shared-model file for the twin ids

Compared with `generalist_20260921/A_adapt/datasets/mixed_reanchor.npz` (`v2`):
- **Rows.** All 115,868 rows of the earlier file are in the new one. The new file has exactly those rows on those
  episodes, no extra anchor.
- **Small arrays.** Identical on every row: id, episode, group, split, domain, anchor frame, fail, unsafe, event
  index, status, source, profile, route length, speed at the anchor, remaining metres, time to event.
- **Large arrays.** Identical on a random 4,000-row sample *and* on all 115,868 rows: corridors `X` (float16),
  context, energy and time targets, history window, mask, privileged context.

## 7. The subset tool's rules (`v1`, `v5_subset_heavy.py`, `v6_subset_rules.py`)

**The three written files**, recomputed with my own code: lowest md5 of the group id among f104's 1,089 rigid
training groups, and the dev fold = md5 % 5 == 0 (224 groups).

| file | rows (mine = file) | same id set | fitted, deploy | fitted, holdout (rows / groups) | dev fold / val rows |
|---|---|---|---|---|---|
| M1 | 93,397 | yes | 84,787 | 67,368 / 865 | 17,419 / 4,358 |
| LC545 | 59,606 | yes | (50,996, not to be used) | 33,577 / 431 | 17,419 / 4,358 |
| LC272 | 42,641 | yes | (34,031, not to be used) | 16,612 / 213 | 17,419 / 4,358 |

- **Groups and nesting.** The manifests' selected groups equal mine. The sets nest (272 inside 545 inside 1,089).
- **Order and arrays.** Rows keep the input order. All 28 arrays of each file equal the source rows.
- **Hashes.** Local and cluster sha256 are equal (`3174999e...`, `2ab94685...`, `4d10dced...`).
- **The trainer agrees.** The logs of job 436135 report the same counts: fit 84,787 (deploy), 67,368 / 33,577 /
  16,612 (holdout), dev 17,419 and val 4,358 in every run.

**Presets without real data yet.** I made a synthetic three-arena file from the f104 columns:
- f104 as is;
- "g203" = f104 renamed, with soil tiers above 9 removed;
- "g228" = renamed, with a seventh of the groups dropped and at most 540 training groups (fewer than 545).

I then compared `ag_subset.py`'s selection with my own on nine cases:
- M2 soil;
- M3 soil and rigid;
- A3 soil;
- two-arena leave-one-out (273 + 272);
- M3 with a tier cut 0-9;
- H with a Gator-prefixed id list (90 % of the soil episodes);
- LC545.

Results:
- Selected groups, rows, training rows per arena and fitted rows in holdout mode are equal in every case.
- One-arena g228 at 545 is refused ("545 requested, 540 available").
- M3 picks the same groups in both worlds, and M3 lies inside M2.

## 8. The training jobs run with the recorded arguments

- **Job list.** `e5/jobs/rigid_f104_v1.tsv` has the same sha256 (`1c6c93fa...`) locally, on the cluster, in
  `submissions.tsv` and in the job log header.
- **Code and data.** `ci_train.py` `7a4f2d67...`, `ga_train.py` `dc73d0bb...` and the three data files are equal on
  both sides.
- **The five running processes.** I inspected them inside the allocation (`srun --overlap`, step 436135.1, under 1 s;
  recorded in `submissions.tsv`).
  - Their command lines are exactly the five lines of the job list, with `--out` / `--tag` as in `e5/README.md`.
  - None has `NEDM_VEHICLE` in its environment. Each has `OMP_NUM_THREADS=4`, and the working directory is
    `G3/source`.
- **Job settings.** `scontrol` shows mi3501x, 1 node, 24 CPUs, limit 3 h 50 min, script `G3/source/scripts/ag_train.sbatch`
  (sha256 = local).
- **Recipe.** It matches PLAN 1.1: GRU, no history input, one world, geometry context, 5 seeds x 30 epochs, batch 256,
  trainer default learning rate and weight decay (as the deploy_a1 run, whose args have `lr`/`wd` = None).
  - No early stopping or model selection on the validation rows: fixed epochs with a one-cycle schedule.
  - With `--split-eval val` the test-split rows in the files are loaded but never used.
- **Smoke 436133.** Both runs exit 0. The round trip max difference is 0.0 on 2,048 rows.
- **Progress at 03:13.** LC272 seeds 0-1 and LC545 seed 0 are done. LC272 s0 scores within-group AUC 0.956 from a
  standing start and 0.961 from moving starts, as reported.

## 9. Notes for the next modules (nothing here needs a change to the files already built)

1. **Misleading stop message.** When no run is selected, `ag_build_ds.py` stops with "runs name other arenas: {}"
   instead of "nothing selected". It stops correctly. I left it alone so as not to change the tool's hash mid-study.
2. **The map check compares the arena, not the capture.** f104 has two captures of the same BMP: `crm_f104_v1`, and
   the older `fdm_f104_50h_20260909/static_map_v1` (observation `faffac47...`, corridors differ by up to 0.002). The
   check cannot tell them apart. Every record stores the observation sha256, so the choice is auditable. For f104
   builds, use `crm_f104_v1/map_root` locally or `G3/e4/map_roots/f104` on the cluster (both `53e23f99...`).
3. **Gator rigid tiers.** The pilot's rigid task file sets `tier` = route index (route_00 = tier 0), not the
   per-group order of the HMMWV pool. The pilot test build ran without task files, so its tiers follow the f104
   order, which is correct. A production Gator rigid task file with route-index tiers would make `ag_build_ds.py
   --tasks` stop (tier assertion). That is safe but blocking. Write the per-group-order tier (as `ag_tasklib`
   does), or build without task files and select with the validated-id list.
4. **Holdout-only files are guarded only by text.** The learning-curve files are guarded only by their manifest text.
   Deploy mode on them would fit the dev fold, and nothing in the trainer prevents that.
5. **Rigid M1 assumes the new arenas' rigid pools finish all 20 tiers.** Rigid collection 436075 was still waiting
   ("Priority") at 03:13; it has not started since 01:53. If g203/g228 rigid stop short, PLAN 7.8 requires rigid M1
   to be retrained on the matched tier range.
6. **Job 436135 time limit.** In the worst case each step stays at 0.23 s after the smaller runs finish. Then the
   two deploy runs (49,650 steps each) end at about 06:05, still inside the 06:42 limit. The builder's 05:15 is the
   likely case.
7. **In-distribution suite closeness (for the report's disclosure).** 37 of the 200 f104 in-distribution reference
   groups start and end within 2 m of some training group's start and goal (minimum 1.52 m). It is in-distribution
   by design, but it is closer than the g203/g228 held-out rule.

The builder's six listed problems are confirmed:
- soil f104 tier 12 on 836 groups;
- g228 has 520 soil training groups (g203 559), from `soil_v1.json` and the case splits;
- the test-arena cases carry split `train` in 221-233 of 250 groups per arena;
- the cluster `ag_build_ds.py` (`6a4da5a8...`) differs from the local one only by the two options and their help text
  (diff checked);
- the record hash (point 5): section 1;
- the f104 map is not in `G3/map_roots`: section 4.

## 10. Files

`verify_e4/`:
- `v1_columns_subsets.{py,json,out}`: tiers, arena/vehicle, blacklist, subset recomputation;
- `v2_twin_rows.{py,json,out}`: twin comparison and per-run rows;
- `v3_cluster_tests.{py,json}`: cluster test files, columns;
- `v4_small_builds.{py,json}` and `small_build_*_record.json`: my small builds;
- `v5_subset_heavy.{py,json,out}`: all arrays of the subsets;
- `v6_subset_rules.{py,json}`: synthetic presets;
- `v7_cluster_rows.{py,json}`: cluster rows recomputed;
- `v8_suite_geometry.json`.

Scratch builds and link folders are in `/tmp/ve4`.
