# VERIFY U1: independent check of the Polaris unseen-arena evaluation build (2026-09-28, 15:05-15:20 CDT)

Verdict: **PASS.** Nothing blocks the full launch. Six minor points are listed in section 9.

Short names: K3 / G3 = the Gator study (local / cluster, read-only); K4 / G4 = this study.

What I did not do:
- I submitted no cluster job. The launcher ran once as a dry run, which submits nothing and writes nothing.
- I wrote nothing into G3 or G4. My cluster access was read-only: reads, sha256 checks, and one copy of 16 G4 and
  8 G3 run folders to this machine.
- I edited no existing file. My only writes are this file and the new folder `verify_U1/` (my scripts and
  their outputs).
- I did not set NEDM_VEHICLE.
- No local soil run.
- One GPU process at a time, each under `timeout 600`. No other GPU process was running during mine; the whole
  re-plan took about 1.5 min.

## 1. Spec written before any drive, and it matches PLAN 9.2

- **sha256.** `e6/analysis/spec_unseen_v1.json` = `bdc2a7385b5dc7fa...` (recomputed). It equals the `.sha256` file
  beside it. The file is read-only.
- **Order of events** (every G4 drive of these arms comes after the spec):

  | event | time (CDT) |
  |---|---|
  | spec written | 14:03:04 |
  | pilot job 442088 submitted (first job to touch these arms) | 15:01:03 |
  | first run folder on G4 | 15:01:41 |

  - The only G4 run folders of these arms are the 16 pilot drives. Their logs and claims exist, and nothing else
    anywhere under G4.
  - `soil_v1/failed` holds only an old M113 id.
- **PLAN is unchanged since the spec.** The spec records PLAN.md sha `dc278146...`, which is the current PLAN.md
  sha.
- **Against PLAN 9.2**, all match:
  - "Reached safely" = the goal is reached, the drive is not unsafe, and there is no belly flag. This is the
    `unsafe_belly` label.
  - The rate is pooled over the 1,000 declared pairs and given per arena.
  - Intervals: pair level (Wilson, pair bootstrap, cluster bootstrap) and over the 8 arenas (arena bootstrap, with
    a t interval beside it).
  - Bar: "meets" = pooled rate of at least 90 %. "Clearly above" = the lower end of the two-sided 95 % arena
    interval is at least 90 %.
  - Each arena's rate is reported against 90 %.
  - Paired tests use exact McNemar and the cluster bootstrap over (arena, nearest feature), as K3.
  - The HMMWV planners are context only.
  - Validity follows PLAN 4.3(c).
- **K3's settings are unchanged:** cluster key `cluster`, boot 4000, seed 0, alpha 0.05, margin 2 points,
  min_groups 50.
- **Declared subset.** File sha `c991e542...`. The group-list sha `9ef445de...` recomputes, with 125 pairs per
  arena and 1,000 in total.

## 2. Picks: right checkpoints, standing start, right map, exactly the declared pairs

- **Checkpoints.** `e5/deploy/polaris_full_soil/SHA256SUMS -c`: 5/5 OK.
  - The spec and all 16 pick manifests name the same five hashes (69b33482, 2949a162, 0fd15fc4, fc8d8606, dd15bf19).
  - This is the same ensemble as the f104 milestone planner `e6/picks/f104/polaris_full_grad`, with the same
    gradient settings: 60 steps, 17 starts, pessimistic keep, abstain at 0.3.
  - Every checkpoint was trained on `polaris_full_f104_soil.npz` (f104 soil only, 52,021 rows, CRM domain filter).
- **Standing start.**
  - CEM manifests: "case layout pose at rest (standing start, all-masked history)".
  - The gradient command has no `--poses`.
- **Map root.** I ran `scripts/ag_map_check.py --map-root K3/map_roots/<a> --cases K3/cases/test_<a>/cases`
  myself. It exits 0 on all 8 arenas; each map's arena is `arena_<a>` and the BMP hashes match.
  - The manifests' own map checks agree, with the same root per arena that K3's HMMWV picks used.
  - The CEM command equals K3's M1a_free command apart from the models and the output folder.
- **Pairs.** In all 16 folders, `groups.txt` and the pick records hold exactly the declared 125 groups of that arena.
- **Locks.** I recomputed every route lock with my own code (file name plus file-content hash, sorted).
  - 16/16 equal `PICKS_LOCKED.sha256`.
  - The set lock recomputes to ALL = `255958f013a3c017...` from the 16 lines "route lock, manifest sha256, folder".
  - 620 picks changed and 380 abstained, consistent with the per-arena counts.
- **Builder's own checks, all 8 arenas:**
  - the CEM re-plan was identical on 125/125 in each arena;
  - gradient stage B equals the recorded CEM pick on 125/125 (z values equal too);
  - the gradient re-plan of 10 groups was identical;
  - no planning call timed out or needed a second attempt.
- **My own re-plan** (`verify_U1/replan.sh`, `selection.json`, `replan_compare.json`).
  - I chose groups outside the builder's 10-group re-plan set, with my own seed:
    - CEM: 2 per arena;
    - gradient: 1 per arena, always a pick the refinement changed.
  - **CEM: 16/16 route files identical byte for byte, and the pick records are equal.**
  - **Gradient: 8/8 identical.** Both the B and G route files match byte for byte, and the pick and manifest
    records are equal apart from timings. Stage B matched the recorded CEM pick 8/8.
  - Planning a subset gives the same result as planning the whole arena.

## 3. Drive rows (`e6/tasks/soil_eval_polaris_unseen_v1.json`, sha `44edd114...`)

- **2,618 rows:**
  - vehicle `polaris` on every row, with `extra = --vehicle polaris`;
  - tier -10, run true;
  - 0 duplicate ids and 0 duplicate seeds.
- **Coverage.** Every declared group has exactly one drive for each of the three arms (1,000 × 3). There are no
  groups outside the declared set.
- **Merged drives.** 378 rows serve both the gradient and CEM arms (the refinement abstained). 2 rows serve all three
  arms: g263_0032 and g263_0056, where the CEM pick is the straight 6 m/s route and the refinement abstained.
- **Routes.** Each arm's route equals its pick folder's recorded route: the same file, or equal route content for the
  merged rows. 0 mismatches.
  - The collector reads only waypoints, speeds, stations and headings, so content equality is sufficient.
- **Straight 6 m/s route.** The route content equals the route of K3's stored HMMWV straight6 drive on
  **1,000/1,000** pairs (K3 index `route_sha256`).
- **Cases.** Every case is K3's `cases/test_<a>/cases/<g>.json`, with `arena = assets/traverse/arena_<a>`.
- **The file glob.** The build used `g2[4-7]*`. It matches the 8 test arenas and excludes K3's training arenas g203 and
  g228.
- **On G4.** I checked all 3,626 case and route files of the new rows plus the 8 HMMWV reference rows with
  `sha256sum -c` on G4: **all match**.

## 4. Arena staging on G4 (only new files)

- **What the collector reads for an arena** (checked in the code): `crm_collect.py` builds the arena path as
  `source-root / case["arena"]`, then reads only:
  - `arena_meta.json` (through `TerrainMap.from_dir`, `scene.build_config` and `build_crm`);
  - the BMP it names.
- **G4/source now holds 8 new arena folders**, each with exactly those 2 files.
  - The files are read-only.
  - All 16 are equal (sha256) to G3's copies.
- **Dispatcher record.** `DISPATCHER.sha256` on G4 grew from 103 to 119 lines, and its first 103 lines are identical to
  the 00:55 real-staging record. So files were only added.
  - `sha256sum -c` passes, and no G4 source file is newer than the record.
- **Local record.** The 16 lines appended to `stage_records/added_files.tsv` were written by `ov_stage.sh --add`
  itself (its designed record). This is the one existing file the build touched.

## 5. Pilot (job 442088): HMMWV reference rows reproduce K3; the Polaris drives are good

- **Groups.** The pilot used the lowest-md5 group of each arena, so the choice did not depend on outcomes.
- **Reference rows.** Case and route are the same as K3's stored soil rows: sha equal, 8/8.
  - The row seed differs from K3's, but the collector only records the seed. It does not use it.
- **My comparison** (the 16 G4 folders and 8 G3 folders copied to this machine):

  | check | result |
  |---|---|
  | end state and goal time | same on 8/8 |
  | `case.json` and `reference.json` | identical on 8/8 |
  | all trajectory arrays | identical on 7/8 |

  - **g251_0000 is the exception.** It first differs at row 256 of 324, i.e. from 12.8 s of a 16.2 s drive.
    - End poses are 1.7 cm apart. The final goal distances are 2.4925 m and 2.4921 m.
    - The outcome is the same: goal reached at 16.2 s.
    - Pilot node k003-004; stored run on k004-002. g241_0179 ran on the same pair of nodes and came out identical, so
      this is the known node-level non-determinism, not a staging fault.
- **Polaris rows:**
  - 8/8 reached the goal safely: no belly flag, launch check passed, QA ok.
  - The vehicle record says polaris.
  - The request files' route and case sha256 equal the local files.
  - Index fields are correct: set `unseen`, the right near/spread role, and cluster labels equal K3's on all
    3,000 index rows.
- **Weakness, as the builder says:** all 8 stored reference drives reached the goal, so the end-state test alone is
  weak. The identical arrays are the stronger evidence.

## 6. Analysis tool (`scripts/ov_unseen_analyze.py`, sha `ce15dc28...`, read-only)

- **Known K3 numbers.** I recomputed them from K3's index with my own code:
  - f104-only HMMWV planner (mean of its two seeds) 87.95 %;
  - its two seeds alone 88.5 / 87.4 %;
  - three-arena planner 90.3 %;
  - all-data planner 90.6 %;
  - straight route 60.0 %.

  These equal K3's REPORT table (87.95 / 90.3 / 90.6 / 60.0), and the tool's context block gives the same values. The
  builder's self-test also passes when re-run into /tmp.
- **Mocks** (`verify_U1/make_mock.py`, `check_mock.py`). I planted outcomes on the real 1,000 pairs: goal not
  reached, unsafe with the goal reached, and belly flag only. I ran the tool on them through the command line, then
  recomputed everything independently.
  - Rates, Wilson intervals, per-arena rates and each arena's position against 90 % are exact.
  - The arena and cluster bootstraps agree within 0.1 point.
  - McNemar discordant counts and the exact one- and two-sided p-values are exact, and so is Holm.
  - Belly-only drives count as not reached safely.
  - The verdicts are right: mock A (95.0 %, arena interval [92.5, 97.1]) gives "clearly above"; mock B (90.4 %,
    [84.4, 96.4]) gives "meets".
  - **Edge cases also correct:**

    | case | verdict |
    |---|---|
    | 1 missing drive | "incomplete" |
    | the same drive listed as failed twice | counted as a failure and a crash, "not physically trustworthy" |
    | belly flags on 11 % of drives | "not physically trustworthy" |
    | 50/1,000 launch failures | "not physically trustworthy" |
    | 49/1,000 launch failures | valid |
    | one non-finite drive | "not physically trustworthy" |

## 7. Superset task file (`tasks/soil_v7_polaris.json`, sha `3330fd3e...`, equal on G4)

- **Contents.**
  - All 22,886 rows of v6b (`05b62f9a`, the newest file on G4) are present, unchanged.
  - Added: the 2,618 new rows plus the 8 reference rows at tier -10. These are identical to their sources except for
    an added `tier_set_by` field. The 16 pilot rows equal their v7 rows.
  - 25,512 rows; 0 duplicate ids; no new row's seed clashes with any other row.
- **What a launch would drive now** (read on G4 at 15:15): only the **2,610 open Polaris rows at tier -10**.
  - The v6b tier -11 rows have all finished since the builder's 15:08 count (54 open then, 0 now).
  - No Gator or M113 row is open.
  - No open row has a claim, a run folder or a failed-attempt record.
- **Launcher dry run** of the exact launch command: it passes.
  - My queue is now **0**, not 8, so the launch makes 12 tasks in total.
  - Settings: episode timeout 2,400 s, claim margin 1,000 s, mi3501x at 3 h (within the 4 h cap).
  - Free now: mi2104x 12 idle nodes, mi3501x 7 idle.
- **Cost estimate.** About 2,610 × 60-70 s on 36 GPUs, i.e. about 1.3-1.5 h. Billing: 8 × 1.5 × 0.4 + 4 × 1.5 × 0.125,
  about 5.5 node-hours. This agrees with the builder's 1.5 h / 6.

## 8. Rules

- Files: every builder file is new; see section 4 for the one record line-append by the staging tool.
  - The other files changed after 13:54 (f104 stage-2 results, v6b, sync folder, polaris_full_grad) belong to the
    orchestrator.
- The builder ran one cluster job, the pilot (2 tasks, 131 s).
- NEDM_VEHICLE was never set: the vehicle is explicit on every row.
- Nothing was written into G3.

## 9. Minor points (none blocks the launch)

1. **The analysis tool was changed after the pilot's 8 outcomes existed.**
   - File time 15:04:28; the pilot drives finished 15:03.
   - The builder says only the interim label for partial runs changed. The old copy (`c4345148`) is gone, so I could
     not diff it.
   - I checked the final read-out against the spec independently (section 6), and the spec does not pin a tool hash.
   - Suggest writing the tool sha `ce15dc28859802ea...` into LOG as frozen before the full drives.
2. **v7 was built and staged before the pilot result.** Staged 15:01:23; the pilot was submitted at 15:01:03 and
   finished at 15:03:14. That is out of the brief's order, but harmless: nothing was launched and the pilot passed.
3. **Two tests in the Holm family.** It holds planner vs straight route and planner vs CEM-only. PLAN 9.2 names only the
   straight-route comparison. This is conservative (at most a factor 2 on the straight-route p), and the brief asked for
   both.
4. **Fragile inputs to the row build.**
   - The glob `g2[4-7]*` is correct today but would break if new arenas matched it. Coverage is checked at 1,000/1,000.
   - `--existing` pointed at v6, not v6b. That has no effect: no identical-route reuse is possible across arenas, and
     seeds were checked clash-free in v7.
5. **Manual failed-drive list.** At read-out, the list of drives that failed twice must be made by hand (NOTES_U1
   section 6). If it is forgotten, the verdict says "incomplete", which is the safe default.
6. **Scientific caveat, not a build defect.** The model predicts about 1e-5 risk everywhere. So the refinement's 620
   changes rest on tiny differences, and planner vs straight route will be decided by the straight route's own failures.
