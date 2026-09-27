# VERIFY gator_full: task B stage 2 (all-tier Gator and HMMWV soil planners, 800-group Gator test), independent checker, 2026-09-26 21:27-21:45

**Verdict: PASS.** Every number I checked in `RESULTS_gator_full.md` reproduces from raw files with my own code:
- the training ids;
- the files, run arguments and checkpoints;
- the spec and pick-lock timing;
- the closed-loop rates and all four declared tests.

I fixed the wording in 11 places (section 8). Two of them matter for how the results read:
- the "more HMMWV data makes the HMMWV planner worse on the Gator" claim had no training-noise caveat;
- the offline AUC criterion rests on far fewer groups than the text implied.

No result or decision changes. No cluster job was submitted. The only cluster action was a read-only extraction on the login
node into a new folder, `G3/tools/verify_bf/`.

K3 = this folder, G3 = `/work1/dannegrut/harry/experiments/arena_gator_20260925`.
- My scripts are `scripts/ag_vbf_cluster.{py,sh}` (`1d6f1c9b` / `36a3108d`), `ag_vbf_data.py` (`d68079de`),
  `ag_vbf_rescore.py` (`ba59bcd7`), `ag_vbf_outcomes.py` (`edf54695`) and `ag_vbf_belly.py` (`760d63c4`).
- Their outputs are in `verify_gator_full/`: `data.json`, `rescore.json` (+ log), `outcomes.json` (+ log), `belly.json`
  and `cluster_out/` (light columns, run records, sha256 lists, sacct).
- None of them imports a study tool. The exceptions are `ci_train.load_ci_model` / `score`, used to run the checkpoints,
  the `TerrainMap` feature list, used as a data source for the terrain clusters, and `ci_planner.py`, used for the
  re-planning sample in 4.

Model labels:
- **G_full** = Gator-trained, all tiers.
- **H_full** = HMMWV-trained, all tiers.
- **G** = Gator-trained, tiers 0-6 (stage 1).
- **H** (= M1a) = HMMWV-trained, tiers 0-6 (stage 1, the "f104 only" ensemble 1).
- **straight 6** = the straight route at 6 m/s (no model).

## 1. Training ids and files (`ag_vbf_data.py` on light columns pulled read-only from the cluster files)

**Gator ids, all validated.** Five sets are the same 15,235 ids:
- the Gator task rows of `soil_v3.json` at tiers 0-12;
- the validated-id list `e5/ids_bf/gator_soil_validated_all.txt` (sha256 `97d76976`, equal on both sides);
- the drives in the Gator per-arena file `ci_f104_gator_crm.npz` (`0bd3ee63`);
- the drives in the G_full (Gator-trained, all tiers) training file (`20ce0ed0`);
- G_full's rows, which match the per-arena file row for row.

Every row carries vehicle `gator`, the soil domain and arena f104. Each drive's tier equals its task-file tier (0
mismatches). Drives per tier: 1,199 at tier 0, 1,200 at each of tiers 1-11 and 836 at tier 12, as reported.

**H_full (HMMWV-trained, all tiers) uses exactly the same ids.**
- Its drives are the validated ids without the `gator__` prefix: 0 extra, 0 missing.
- Its rows are the 58,268 soil rows of the E4 f104 file (`350d58f4`), identical and in the same order.
- H_full and F104all (the full HMMWV f104 soil file) have the same sha256 (`5e9c2961`).
- Every row carries vehicle `hmmwv`, and each twin drive has the same tier as its Gator drive.
- The two vehicles' files hold the same groups in the same splits: 1,089 train, 56 val, 55 test.
- So "H restricted to the Gator-validated ids = the full HMMWV file" is right.

**Fitted rows, recomputed.** Deploy counts every training row. Holdout also leaves out the dev fold (md5 of the group id
mod 5 = 0).

| file | deploy | holdout | standing-start training rows |
|---|---|---|---|
| G_full | 50,822 | 40,367 | 13,821 |
| H_full | 52,923 | 42,027 | 13,821 |

All match the report and the checkpoints' stored `train_rows`.

**Other file checks.**
- **Suite groups.** None of the 800 suite groups and no suite-name pattern appears in either training file. Only
  `f104_v2_group_*` names are present.
- **Stage-1 G file** (`b3938949`). Its 8,399 drives are exactly the tier 0-6 part of G_full.
- **Statuses of the Gator drives.** 13,231 blocked, 1,805 goal, 198 broke through and 1 timeout: 88.15 % failed,
  against 68.05 % for the HMMWV.
- **Collection read-out** (report 1.1), recomputed from the light columns:
  - 3,263 drives failed only for the Gator and 201 only for the HMMWV;
  - by speed profile, Gator / HMMWV: 2 m/s 91.3 / 86.2 %, 4 m/s 83.8 / 56.3 %, 6 m/s 82.7 / 33.5 %,
    2-6-2 m/s 84.2 / **54.5** % (1,247 of 2,286; the report said 54.6, fixed), planner routes 92.2 / 83.9 %;
  - by tier: 87.3-89.3 %.
- **Groups where every route fails.** For the Gator, every route of the group fails in 693 of the 1,200 groups; for the
  HMMWV, in 180.

## 2. Run arguments (job outputs, lane logs, sacct)

**Recipe.** The job lists `e5/jobs/soil_bf_{G,H}.tsv` equal `soil_s1_B.tsv` except for the data path, tags and output
folders: `--arch gru --cond none --domain-filter crm --ctx geom --split-eval val --bs 256 --epochs 30 --seeds 5
--seed0 0`, with `--roundtrip-check` on the deploy runs. The job outputs record the same arguments.

**Code and platform.**
- The trainer is `ci_train.py` `7a4f2d67`, the same hash as stage 1 (job outputs of 436353 / 436354).
- torch 2.10.0+rocm7.1 on both stages.
- The GPU differs: MI350X now (k007-005-v2/-v3), MI250X at stage 1. The report declares this.

**Lane logs.**
- All 4 runs exit 0.
- Fitted rows 50,822 / 40,367 (G_full) and 52,923 / 42,027 (H_full).
- 5,940 and 6,180 steps per deploy seed (= floor(rows / 256) x 30).
- Round-trip max difference 0.0 on all 10 deploy seeds.

**sacct.**
- Jobs 439361 / 439362: 19:28:03-19:47:57 and 19:28:06-19:47:18, COMPLETED.
- Drive jobs 439414 (7 x mi3501x), 439415 (5 x mi2104x) and 439426 (10 x mi2101x): all 22 tasks COMPLETED,
  20:01:16-21:20:17.

**Billing, with my own sum over sacct** (same partition weights as NOTES_S1 2):
- task B stage 2: **5.119** billed node-hours;
- the session since 09-25 00:00: **113.29**;
- no job after 439362 other than these, and no partition without a weight.

## 3. Checkpoints (`ag_vbf_rescore.py`, local 5090, TF32 off)

**Hashes.** All 10 deploy files (5 G_full, 5 H_full) have the same sha256 in four places:
- local `SHA256SUMS`;
- the pick lock `LOCK_crm_bfull.json`;
- the model manifest `soil_bf_models.json`;
- the cluster training folder.

The 10 holdout files also equal their cluster copies.

**Loading and scoring.**
- All 30 checkpoints load, including the stage-1 G and H holdout runs.
- The stored recipe is gru / none / geom / crm / 30 epochs / split-eval val, seeds 0-4, the declared data path and the
  declared fitted rows.
- On the val rows of each model's own vehicle, the ensemble logits reproduce the trainer's stored logits to <= 1.7e-5
  (members <= 4.8e-5), with identical labels.

**My own offline AUC reproduces every value in report section 3 to 4 decimals.** This is the within-group AUC of
"unsafe" from a standing start, own pair counting and own row selection.

| model | on Gator drives: dev+val / val | on HMMWV drives: dev+val / val |
|---|---|---|
| G_full holdout | 0.9462 / 0.9502 | 0.8136 / 0.8246 |
| G_full deploy (val) | 0.9502 | 0.8217 |
| H_full holdout | 0.8266 / 0.8474 | 0.9793 / 0.9825 |
| H_full deploy (val) | 0.8474 | 0.9913 |
| G holdout | 0.9365 / 0.9486 | 0.8109 / 0.8290 |
| H holdout | 0.8233 / 0.8574 | 0.9736 / 0.9694 |

- Pick-fail shares also match: 0.632 / 0.643 / 0.743 / 0.721, random 0.878, best possible 0.582.
- No holdout model had fitted any scored group (0 rows dropped).
- **Coverage (new, now in the report).** The 2,976 Gator pairs come from only **116 of the 280** groups: 163 groups are
  all-unsafe for the Gator and 1 is all-safe. On val, the 603 pairs come from **22 of the 56** groups. Criterion (3)
  (0.9502 >= 0.95) therefore rests on 22 groups.

## 4. Spec, picks and their timing

**Spec.** `spec_soil_v1_Bfull.json` has sha256 `a1d75180...`, equal to the `.sha256` file and to the value quoted in the
19:29 LOG line.
- The file is read-only with mtime 19:29:22.
- The first checkpoint was written at 19:32, the pick routes at 19:54:40, and the first new drive request at 20:01:17.
- The results file embeds the spec, and the embedded copy is identical to the frozen file.
- `ag_analyze.py` (`42561c33`) is unchanged since the 09-25 commit.
- Evidence limit: the timing rests on file mtimes, the job records and the LOG. There is no external timestamp.

**Pick locks.** I re-derived all locks with my own code (name + content hash of every route file, sorted):
- G_full `04cd65c3` and H_full `a55070c1`;
- the set lock ALL `52f510b7` (file 19:59:55), whose per-directory lines carry the manifest hashes;
- the stage-1 set lock ALL `2c766a52` (09-25 14:34), for the G, M1a (= H) and straight 6 directories.

All equal. Both planner runs record `rerun_check.identical = true`, with the same torch 2.12.0+cu130 and the same
`ag_picks.py` hash as the stage-1 picks. Map check ok.

**Re-planning sample.** In a new process I ran `ci_planner.py` with the deployed checkpoints on the 12 lowest-md5 suite
groups. G_full gave 12 / 12 and H_full 12 / 12 route files byte-identical to the locked ones. So the locked picks come
from these checkpoints.

**Task file.** `soil_v4.json` is `d73fe77b` both locally and on G3.
- Its first 43,901 rows are soil_v3, unchanged.
- It adds 2,318 new rows: 1,541 Gator rows at tier -9 and 777 HMMWV rows at tier -8.
- All 2,318 were driven, all have a completion marker, and all requests are later than the lock (first 20:01:17, last
  outcome 21:19:29).

## 5. Outcomes recomputed from raw run records (`ag_vbf_outcomes.py`)

**Method: match drives to picks by route, without the study's mapping files.**
- On the login node, `ag_vbf_cluster.py` read every run folder of an f104 suite group (6,666 folders). For each it
  recorded the vehicle block, the sha256 of `reference.json` and of its route content (waypoints, speeds, stations,
  headings), the collection request, the outcome and the launch check.
- For each of the 8 arms x 800 groups, I looked for a drive of the same group and vehicle whose route content equals the
  locked pick.

**Result of the matching.**
- **6,400 / 6,400 matched**: 6,282 distinct drives, 0 ambiguous, and my drive is the index's drive in 6,400 / 6,400
  cases.
- Most drives' `reference.json` is byte-identical to the pick file. The rest (G_full 1, H_full 6 per vehicle,
  straight 6 15-22) are reuses whose route file differs only in `meta`.
- Reuse by content, as reported:
  - G_full: 38 of its drives already existed (37 stage-1 G drives, 1 straight 6);
  - H_full on the Gator: 21 (15 from H, 6 from straight 6);
  - H_full on the HMMWV: 23.

**Provenance of the 6,282 drives:**
- all complete, and all passed the launch check with finite states;
- collector `cb6792be` and config `crm_main_step1ms_spacing008` on all;
- case sha256 equals the suite case on all;
- the request's route hash equals `reference.json` on all;
- the 3,918 Gator drives carry wrapper `b52e1fa6` and `ag_vehicle` `072716ee`, and the 2,364 HMMWV drives carry no
  vehicle block;
- every local `outcome.json` is byte-equal to the cluster copy (6,282 / 6,282);
- fail (goal not reached) from the raw outcome equals the index's label on 6,400 / 6,400.

**Terrain clusters.** I recomputed them from the case start/goal and the f104 features: 9 clusters of 57-128 groups, the
same as the index.

**Rates.** Goal not reached over 800 groups, by stratum, and on the 200 in-distribution groups:

| arm (plain label) | vehicle | fail | statuses (blocked / broke through / goal) | hill / crater / other | in-dist. 200 | median time | mean pick risk |
|---|---|---|---|---|---|---|---|
| G_full, Gator-trained, all tiers | Gator | 32.625 % | 254 / 7 / 539 | 41.1 / 27.9 / 20.7 | 35.0 | 18.95 s | 0.181 |
| G, Gator-trained, tiers 0-6 | Gator | 34.75 % | 264 / 14 / 522 | 44.3 / 28.6 / 22.7 | 36.0 | 18.4 | 0.184 |
| H_full, HMMWV-trained, all tiers | Gator | 56.375 % | 440 / 11 / 349 | 55.9 / 62.5 / 46.0 | 62.5 | 20.85 | 0.0045 |
| H, HMMWV-trained, tiers 0-6 | Gator | 51.875 % | 405 / 10 / 385 | 53.8 / 56.8 / 38.0 | 60.0 | 21.85 | 0.0021 |
| straight 6 | Gator | 85.75 % | 679 / 7 / 114 | - | 89.0 | 17.0 | - |
| H_full | HMMWV | 4.375 % | 7 / 27 / 765 + 1 timeout | 6.5 / 2.5 / 2.7 | 2.5 | 17.1 | 0.0045 |
| H | HMMWV | 5.5 % | 7 / 36 / 756 + 1 timeout | 8.6 / 2.1 / 4.0 | 3.5 | 17.7 | 0.0021 |
| straight 6 | HMMWV | 32.25 % | 30 / 227 / 542 + 1 rollover | - | 34.5 | 9.9 | - |

**Declared family.** My own bootstraps use seed 12345 and 4,000 draws: the cluster bootstrap resamples the 9 terrain
clusters, and the group bootstrap resamples the 800 groups. Holm is applied over my one-sided cluster p.

| test | diff | my cluster 90 % (report) | my group 95 % (report) | won / lost, McNemar | my cluster p1 / Holm | clusters better / worse / equal |
|---|---|---|---|---|---|---|
| F1 G_full vs H_full, Gator (primary) | -23.75 | [-30.0, -16.6] ([-30.4, -16.5]) | [-27.0, -20.5] ([-27.0, -20.4]) | 206 / 16, 3.1e-43 | 0.0002 / 0.001 | 9 / 0 / 0 |
| F2 G_full vs straight 6, Gator | -53.125 | [-58.2, -47.3] ([-58.4, -47.3]) | [-56.8, -49.5] | 433 / 8, 1.2e-116 | 0.0002 / 0.001 | 9 / 0 / 0 |
| F3 H_full vs straight 6, Gator | -29.375 | [-34.2, -24.5] ([-34.1, -24.6]) | [-32.9, -25.9] | 262 / 27, 1.6e-49 | 0.0002 / 0.001 | 9 / 0 / 0 |
| F4 G_full vs G, Gator | -2.125 | [-3.7, -0.5] ([-3.7, -0.5]) | [-4.4, +0.1] ([-4.5, +0.1]) | 52 / 35, 0.086 | 0.022 / 0.022 (tool 0.018) | 5 / 2 / 2 |

- **All four reject under Holm in my re-run**, as in the report.
- F4's group bootstrap:
  - one-sided p 0.037 (report 0.039);
  - group 90 % interval [-4.0, -0.25];
  - cluster 95 % interval [-3.9, -0.1].
- So "F4 passes at 90 % by both bootstraps, at 95 % only by the cluster one" holds.
- Dropping any one cluster leaves F4 between -1.6 and -2.6 points.
- Identical picks:
  - F1: 0 groups;
  - F2: 2;
  - F3: 17;
  - F4: 37.

**Secondary contrasts, all reproduced.**

| contrast | diff | interval | won / lost or detail |
|---|---|---|---|
| H_full vs H on the Gator | +4.5 | cluster 90 % [+2.7, +6.3], group 95 % [+1.9, +7.1] | 44 won / 80 lost, McNemar 0.0016; 0 / 8 / 1 clusters |
| H_full vs H on the HMMWV | -1.125 | [-2.2, -0.2] / [-2.6, +0.4] | 24 / 15, McNemar 0.20 |
| H_full vs straight 6 on the HMMWV | -27.875 | - | 233 / 10 |
| G_full vs H | -19.25 | - | 169 / 15 |
| G vs H | -17.125 | - | 159 / 22 |
| G_full on the Gator vs H_full on the HMMWV | +28.25 | - | - |
| H_full, Gator vs HMMWV | +52.0 | - | 1 / 417 |

**Headroom, time and body-in-soil.**
- **Headroom closed:** G_full 62.0 % [58.4, 65.8], G 59.5 %, H_full 34.3 %, H 39.5 %, H_full on the HMMWV 86.4 %
  [81.8, 90.5], H on the HMMWV 82.9 %.
- **Median time ratios on joint successes:**
  - G_full / H_full 0.893 (333 pairs);
  - G_full / G 1.021 (487);
  - H_full / H on the HMMWV 0.979 (741).
- **Body-in-soil flag on the Gator drives, own code:** G_full 1.5 %, G 2.1 %, H_full 2.5 %, H 3.1 %, straight 6
  11.1 %, equal to the report when frames are counted at 0.05 s.
  - Straight 6 would be 11.5 % if the frame spacing is taken from `elapsed / (frames - 1)`, because 3 drives have a run
    of exactly 1.00 s.
  - This is a convention at the threshold, and the report keeps the tool's convention.

**Not recomputed independently.** The secondary "unsafe" contrast (-23.4) needs the backward-motion clause from the
trajectories. The simulated hours of the new drives are 11.43 h Gator and 4.03 h HMMWV, summed from the outcome
`elapsed_s`, which matches.

## 6. Scope and rules

- Only new scripts were written after the resume (`ag_bf_*`, and my `ag_vbf_*`). The unchanged tools the report cites
  have 09-25 mtimes, and there are no tracked changes under `scripts/` or `src/`.
- No file changed after 19:15 in any other `artifacts/traverse` folder or in the earlier cluster roots (0 files).
- The extra arm (H_full driven by the HMMWV) was declared in the frozen spec before any drive, and the report declares
  it as a deviation.

## 7. What remains uncertain (in the report now)

1. **F4 (G_full vs G, -2.1 points) is marginal.** It passes the declared rule only through the 9-cluster bootstrap. The
   group interval at 95 % touches zero, and the McNemar p is 0.086. The two ensembles differ in data and also in GPU
   type, and there is no second Gator ensemble to measure training noise.
2. **H_full vs H on the Gator (+4.5 points)** has the same limitation. It is one ensemble against one, on two platforms,
   with training noise unmeasured. Out of distribution this noise could be several points: the f104-only HMMWV
   ensembles differed by up to 6.4 points on single 125-group arenas. The first version of the report stated it as a
   plain effect of data.
3. **Criterion (3)** (0.950 >= 0.95) rests on 22 val groups with both outcomes. At that size the seed-noise scale is
   up to 0.04 (S1).

The primary result (F1, -23.8 points; 206 vs 16 pairs; all 9 clusters) is far outside any of these uncertainties.

## 8. Wording fixes made in `RESULTS_gator_full.md`

1. Section 0, first bullet: the headline "More data helps the Gator planner a little" now reads "By the declared rule,
   ...; the evidence is marginal". A sub-bullet was added: one ensemble per model, different GPU types, training noise
   unknown.
2. Section 0: "The gap grew because the HMMWV planner got worse" now reads "grew by 6.6 points, mostly because ...
   (+4.5); the Gator planner's own gain (-2.1) is the rest".
3. Section 0: the headline "More HMMWV data makes the HMMWV planner ... worse on the Gator" is now descriptive ("With
   more HMMWV data, ... did ..."). A caveat was added that the intervals cover pair sampling only, training noise is
   unmeasured, and the platforms differ. "So the extra data makes ..." became "A likely reading is ...".
4. Section 0 and 4.5, criterion (3): added that only 22 of the 56 val groups (116 of the 280 dev + val groups) have
   both outcomes.
5. Section 0: added plain labels next to the G_full / H_full codes.
6. Section 1.1: HMMWV 2-6-2 m/s 54.6 -> 54.5 % (1,247 / 2,286 = 54.55 %; the tool had rounded 0.5455 a second time).
7. Section 2: job 439362 ran 19:28-19:47, not 19:29-19:48. Speed is now given as the per-seed range (0.031-0.041 and
   0.027-0.036 s/step) instead of one seed's value.
8. Section 3: "2,976 pairs in 280 groups" now reads "from only 116 of the 280 groups (163 all unsafe, 1 all safe); val
   603 pairs from 22 groups; tiers 0-6: 832 pairs in 96 groups". The "random / best possible pick" row is labelled as
   pick-fail shares, not AUC.
9. Section 4.3: "the cluster interval is narrower than the group interval" now compares them at the same 90 % level.
   The training-noise note adds the per-arena spread (up to 6.4 points) and "none of the intervals includes training
   noise".
10. Section 4.4: a note below the table says the two H_full vs H rows are single ensembles on different GPU types.
11. Section 5 (1, 2) and 6:
    - "Doubling the data" now reads "1.8 times (8,399 -> 15,235 drives)".
    - "in most groups every route fails" now reads "in 693 of the 1,200 groups (58 %; HMMWV 180)".
    - "more HMMWV data sharpens that (confident wrongness)" is replaced, because the predicted risk rose from 0.21 to
      0.45 % (it did not become more confident). The text now says its picks suit the Gator even less.
    - "gain grew therefore" now reads "mostly for this reason".
    - "Tonight's soil drives" now reads "This study's soil drives", with a note that pairing 09-26 with 09-25 drives
      relies on that reproducibility.
