# VERIFY S1: soil stage 1 (training data and models at tiers 0-6), independent verifier, 2026-09-25 14:05-14:17

**Verdict: PASS.** The soil stage-1 report and `NOTES_S1.md` hold up. Tier coverage, the subset selections, the H = Gator-id
identity, the vehicle records, the suite scan, the run arguments, the synced checkpoints and the offline scores were
all recomputed with my own code; every number I checked matches. I fixed three small things (section 9); none of them
changes a result. No cluster job or job step was submitted; the only cluster-side action was a read-only extraction
on the login node (`G3/tools/verify_s1/`, new files only).

K3 = this folder, G3 = `/work1/dannegrut/harry/experiments/arena_gator_20260925`. My scripts and small outputs are in
`verify_s1/`; the extracted light columns and run records stay on the cluster in `G3/tools/verify_s1/out/`.

## 1. Tier coverage (`verify_s1/recompute_s1.py`, output `recompute_s1.json`)

**Method.** I pulled the light columns (id, group, split, tier, vehicle, arena, episode, anchor frame) of every soil
stage-1 file and compared each drive's tier with its row in the soil task file (`tasks/soil_v2.json`, sha256
`acf1e98c...`, equal on both sides).

| file | soil rows | drives | tiers present | groups (train / val / test) | against the task rows of tiers 0-6 |
|---|---|---|---|---|---|
| f104, HMMWV (the E4 file, 0-12, cut by the subset step) | 32,152 at tiers 0-6 | 8,399 | 0-6 after the cut | 1,089 / 56 / 55 | = the Gator's 8,399 ids (below) |
| g203, HMMWV | 15,987 | 4,240 | 0-6 only | 559 / 24 / 23 | 4,242 rows (606 groups x 7); 2 missing = the 2 soil-QA rejects |
| g228, HMMWV | 16,261 | 4,242 | 0-6 only | 520 / 38 / 48 | 4,242 of 4,242 |
| f104, Gator | 30,827 | 8,399 | 0-6 only | 1,089 / 56 / 55 | 8,399 of 8,399 |

- Tier of every drive = its task-file tier: 0 mismatches on all three new files.
- The Gator task tier equals the HMMWV twin's f104 tier for all 15,235 Gator rows, and the f104 HMMWV ids of tiers 0-6
  are exactly the Gator ids of tiers 0-6 (8,399 both).
- **"7 routes per group everywhere" has three exceptions**, all explained:
  - two g203 groups have 6: `g203_v2_group_0241_route_03` and `_0308_route_07` were rejected by the soil QA check (as reported);
  - f104 group 0211 has 6 for both vehicles: its tier-0 route (`f104_v2_group_0211_route_11`) is the one id the
    original HMMWV collection flagged (`collect_v1/qa.json`: unstalled break). So it was never a training id, and the
    Gator was never asked to drive it. That is why the total is 8,399 and not 8,400.
- All 18 subsets and evaluation files hold tiers 0-6 only, soil rows only and one vehicle each. f104 is cut the same way
  in M1, H, LC272, LC545, M2, M3, A3, the LOAO files and its evaluation files, on the training rows and the val / test rows.

## 2. Subsets recomputed (same script)

**Method.** Starting from the per-arena files, I applied my own code (nothing imported from `ag_subset.py`):
- cut each file to tiers 0-6;
- within each arena, rank the training groups by the md5 of the group id and keep the first N;
- keep every val / test row of the arenas in the design;
- for the learning-curve and evaluation files, also keep the dev fold (md5 % 5 == 0).

The md5 ranking is the same before and after the tier cut.

**Result: 18 of 18 files are identical to my selection, including the row order.**

| file | plain label | training groups per arena | rows | fitted, deploy | fitted, holdout (groups) | training drives |
|---|---|---|---|---|---|---|
| M1 | f104 only | f104 1,089 | 32,152 | 29,210 | 23,205 (865) | 7,622 |
| H | HMMWV on the Gator-validated ids | f104 1,089 | 32,152 | 29,210 | 23,205 (865) | 7,622 |
| M2 | f104 + g203, same total | 545 + 545 | 33,227 | 29,048 | 22,941 (861) | 7,628 |
| M3 | three arenas, same total | 363 x 3 | 35,573 | 29,071 | 23,284 (873) | 7,621 |
| A3 | three arenas, all data | 1,089 + 559 + 520 | 64,400 | 57,898 | 46,274 (1,733) | 15,173 |
| G | Gator-trained, f104 | f104 1,089 | 30,827 | 28,057 | 22,284 (865) | 7,622 |
| LC272 / LC545 | f104 learning curve (holdout only) | 272 / 545 + dev fold | 14,673 / 20,544 | - | 5,726 (213) / 11,597 (431) | 3,059 / 4,585 |
| LOAO1 g203 / g228 | one new arena alone | 545 / all 520 | 15,615 / 16,261 | - | 11,344 (430) / 11,409 (426) | 3,813 / 3,640 |
| LOAO2 f104+g203 / f104+g228 / g203+g228 | two arenas | 273 + 272 | 18,676 / 19,911 / 18,042 | - | 11,428 / 11,839 / 11,786 | 3,813 / 3,815 / 3,813 |
| evaluation files | dev fold + val + test: f104, f104 on Gator ids, g203, g228, Gator f104 | 224 / 224 / 117 / 94 / 224 dev-fold groups | 8,947 / 8,947 / 4,327 / 4,852 / 8,543 | - | - | - |

- The sha256 values of all 22 files on the cluster are the ones in `NOTES_S1` (`verify_s1/cluster_out/sha256_npz.txt`).
- The local copies of the 5 evaluation files are byte-identical to the cluster copies.
- The row counts per arena, the standing-start training rows (g203 3,911, g228 3,640, Gator 7,622) and the rows per drive
  (HMMWV f104 3.83, g203 3.77, g228 3.83, Gator 3.67) are as reported.

## 3. H uses exactly the Gator-validated ids

- **The id list.** `e5/ids_soil/gator_soil_validated_t0-6.txt` has 8,399 unique ids, all with the `gator__` prefix. It is
  byte-identical to the cluster copy (sha256 `ed248f52...`).
- **The drives of H** = the validated ids without the prefix, exactly: 0 extra and 0 missing in either direction.
- **The drives of G** = the validated ids, exactly.
- **H is the M1 file.** The two files are byte-identical (sha256 `48a29d16...` both), with the same ids in the same order.
- **Splits match.** Every H drive has the same split as its Gator twin, so the two vehicles' files hold the same
  1,089 / 56 / 55 groups.
- **So H = the M1a soil ensemble is correct** (same file, recipe, seeds 0-4 and GPU type). The manifest
  (`e5/deploy/soil_s1_models.json`) points H to `e5/deploy/M1a_soil`.

## 4. Vehicle records (`verify_s1/vs1_runs.py` on every run folder, read only)

**Gator runs**
- The soil output folder holds 9,412 Gator run folders.
  - All 9,362 finished ones carry the Gator block (`vehicle.name == 'gator'`): the 8,399 of tiers 0-6 and 963 of tier 7.
  - The other 50 are tier-7 drives still running; they have no outcome yet.
- All 8,399 training runs record the same set-up:
  - collector wrapper `b52e1fa6` and vehicle switch `072716ee`;
  - spawn +0.35 m;
  - calibrated soil wheels (front 0.19575 m, rear 0.2275 m);
  - chassis not coupled to the soil.
- None of them uses the pilot's +0.08 m wheel. The 144 pilot runs imported into this folder are calibrated Gator runs
  like the rest.

**HMMWV runs**
- None of the 11,431 finished HMMWV runs in the same folder has a vehicle block. They cover g203, g228, the dev arena,
  the spread-arena headroom runs, the drift checks and the bit-identity checks.
- None of the 15,235 original f104 HMMWV runs (`collect_v1`) has one either.

**Every drive of every training file**
- has its run folder, a completion marker and a passed launch check, and the arena in its case matches;
- g203 / g228 / f104 HMMWV drives: no vehicle block; Gator drives: the Gator block.

**The status counts** equal NOTES_S1 section 3 and 7.1:
- g203: goal 1,116, breakthrough 2,542, blockage 577, timeout 5;
- g228: 1,154 / 2,594 / 489 / 5;
- Gator: blocked 7,284, goal 1,014, broke through 101.

## 5. Gator collection read-out re-computed (own code; `vs1_belly.py` for the body-in-soil flag)

| criterion (PLAN 7.7) | limit | reported | mine |
|---|---|---|---|
| ids validated | >= 95 % | 100 % | 8,399 / 8,399 with completion marker, launch check passed, Gator block |
| crashed / non-finite | < 1 % | 0 | 0 QA flags, 0 NaN in the clearance series |
| launch-check failures | < 5 % | 0 | 0 |
| body in soil (> 0.05 m under the surface for > 1 s in a row) | <= 10 % | 8.3 % | 8.3 % (frame spacing 0.05 s), 8.4 % with each run's recorded spacing (0.0500-0.0503 s; 704 drives); 9.7 % counting > 1 s in total; 0 flagged drives reach the goal |
| goal not reached, Gator vs HMMWV on the same ids | report | 87.9 % vs 67.9 % | 87.93 % vs 67.91 %; 1,801 Gator-only and 120 HMMWV-only failures |
| simulated hours | report | 77.7 h vs 50.3 h | 77.65 h vs 50.33 h |

## 6. Run arguments from the logs (`verify_s1/check_runs_s1.py`, output `check_runs_s1.json`)

**What I checked.** 19 runs: 16 in job 436354, LOAO1 g228 in step 436353_0.0, and G deploy + G holdout in step
436353_0.1. For each run I read the trainer's own summary (arguments + one record per member), its job-list line and
its log.
- **Recipe:** GRU, no history input, soil rows only, geometry context, val split, batch 256, 30 epochs, 5 seeds.
  Every member records lr 0.002 and weight decay 1e-4; no data fraction or subsampling. **0 deviations.**
- **Data file and seeds:** the right file for each run, members 0-4 (M1b and M3b: 5-9), and the round-trip check on
  every deploy run.
- **Round trip:** every deploy member records a checkpoint round-trip difference of exactly 0.0.
- **Fitted rows:** equal to my recount on every run, e.g. A3 deploy 57,898, G holdout 22,284, LOAO1 g228 11,409.
  No rigid row was fitted.
- **Job lists and logs:** each job-list line is exactly the recipe line. Every log ends `exit: 0`, and both job
  summaries end `all done rc=0`. The local logs are byte-identical to the cluster logs (22 files).
- **Scheduler records (`sacct`):**
  - 436354 COMPLETED, 11:27:05-11:59:26 on k004-004;
  - 436353_0.0 COMPLETED, 6 min 7 s;
  - 436353_0.1 COMPLETED, 13 min 28 s;
  - probe 436349: the scheduler says COMPLETED, but its own log shows the pytorch test failing ("HIP error: file not
    found", `exit: 1`), i.e. it "failed as expected" as reported;
  - probe 436350: log `exit: 0`.
- **Code versions and vehicle switch:** trainer `ci_train.py` `7a4f2d67`, `ga_train.py` `dc73d0bb`. The job script
  `ag_train_gpus.sbatch` `d031c7ff` and the overlap launcher `ag_s1_overlap_train.sh` `f5734474` are the same locally and
  on the cluster. Both unset `NEDM_VEHICLE`. All jobs and steps are in `submissions.tsv`.
- **Timing:** every data file and its manifest were written before its lane started (the M2 / M3 / A3 / LOAO lanes
  waited for them), and none was modified afterwards.

## 7. Synced checkpoints

- **Hashes.** `sha256sum -c SHA256SUMS` passes in all 8 deploy folders (H after the fix in section 9). The 49 deploy files
  of the 7 ensembles equal the cluster's sha256. The local training mirror `e5/train/soil_s1/` is identical to the
  cluster on all 126 files present; G's deploy files exist locally only in `e5/deploy/G_soil`, which matches.
- **Loading** (`verify_s1/rescore_s1.py`, local 5090, TF32 off). Every deploy checkpoint loads. On the val rows of its
  own arenas it reproduces the trainer's stored member logits to at most 5.3e-5 (ensemble 1.9e-5), with identical
  labels and no fitted rows among them. This is the usual difference between two GPU types.

## 8. Offline AUC re-scored with my own code (`rescore_s1.py`, output `rescore_s1.json`)

**Method.**
- My own within-group AUC: (unsafe, safe) route pairs inside each start/goal group, pooled, ties counted as half.
- My own choice of rows: standing start, dev fold + val.
- My own fitted-group guard, taken from each run's logits file (0 rows dropped: holdout runs never fit the dev fold).

**Result: 14 of 14 values equal `soil_s1_A_auc.json` / `soil_s1_B_auc.json` to 4 decimals**, for dev fold + val, for val
alone, and for the "safest route was unsafe" pick rate.

| model (holdout version) | f104 | g203 | g228 |
|---|---|---|---|
| f104 only (M1) | 0.9748 | 0.9428 | 0.9387 |
| three arenas, same total (M3) | 0.9658 | 0.9451 | 0.9599 |
| three arenas, all data (A3) | 0.9784 | 0.9622 | 0.9741 |
| f104 only, 272 / 545 groups (learning curve) | 0.9537 / 0.9621 | - | - |

| task B model (holdout version), f104 | on Gator drives: dev + val (val only) | on HMMWV drives: dev + val |
|---|---|---|
| G, Gator-trained | 0.9255 (0.9521) | 0.8218 |
| H = f104-only HMMWV model | 0.8125 (0.8085) | 0.9748 |

- **Saturation.** On the 280 scored groups, every standing-start Gator route fails in 182 groups (65 %, as reported;
  HMMWV 24 %). Every route succeeds in 2 groups, and 96 groups have both outcomes.
- **Pick rates.** Random pick 0.879 against best possible 0.650 on Gator drives; G 0.686, H 0.746. G closes 84 % of that
  gap and H 58 %, as reported.

## 9. Small problems fixed

1. **Wrong number in `NOTES_S1.md` 7.3.** It said "only 42 % of the Gator's standing-start rows are in groups with both
   outcomes, so the AUC rests on 832 of 1,960 rows". But 832 is the number of (unsafe, safe) route pairs, not of rows.
   The correct figures are 96 of 280 groups and 672 of 1,960 rows (34 %). Corrected in place, with a note. The same
   42 % figure is in the 14:01 LOG line, which is append-only, so the correction is logged instead.
2. **H's hash list could not be checked where it sits.** `e5/deploy/H_soil/SHA256SUMS` listed bare file names, so
   `sha256sum -c` failed inside `H_soil/` (the checkpoints live in `M1a_soil/`). I rewrote it with `../M1a_soil/` paths,
   the same convention as the rigid `e5/deploy/H/`. The hashes are unchanged and it now passes. No tool reads this
   file by file name (`soil_s1_models.json` points H to `M1a_soil` directly).
3. **Wording in the H README.** `H_soil/README.md` said the hashes were "repeated below", but they are in `SHA256SUMS`.
   The wording now says so.

## 10. Caveats for the next steps (no action needed for stage 1)

1. **The Gator's AUC criterion (PLAN 3 (3), >= 0.95).** It is met only on the 56 val groups (0.952 holdout, 0.957 deploy),
   not on dev fold + val (0.925, 280 groups). The Gator's AUC rests on 832 route pairs in 96 groups, so it is less
   certain than the HMMWV numbers (1,902 pairs).
2. **The body-in-soil flag is uneven.** It is 8.3 % overall, inside the limit, but 10.4 % on the constant 4 m/s designed
   routes and 11.2 % on the constant 6 m/s ones (planner routes 6.0 %). The overall rule is met; the per-profile
   excess should be named next to any Gator soil result by speed.
3. **Val-only read-outs on one arena are small.** On g203, val-only scores rest on 136 pairs in 17 groups. The f104-only,
   three-arena and all-data models all score 0.912 there. NOTES_S1 correctly uses dev fold + val for the comparison.
4. **GPU type.** Every soil model (task A and G, and H = M1a) was trained on MI250X, so the soil contrasts do not cross
   GPU types (unlike rigid).
5. **Scope.** These are offline scores on the training arenas only. They say nothing yet about closed-loop goal reaching
   on the unseen test arenas.
6. **State at 14:15.**
   - Queue: 34 tasks.
   - Running: 436080 x 6, 436208 x 3, 436210 x 3, 436215 x 4, 436353, 436460 x 6, 436461.
   - Pending: 436416 x 3, 436460 x 2, 436490 x 5.
   - 0 failed soil ids.
   - Account ledger: 700.5 (hourly file).

## 11. Files

- **Cluster, read-only, on the login node:** `verify_s1/vs1_extract_light.py`, `vs1_runs.py`, `vs1_run.sh`,
  `vs1_belly.py`. Outputs are in `G3/tools/verify_s1/out/`: light columns, run records, belly flags, and
  `sha256_npz.txt` / `sha256_train.txt`. The two hash lists are also copied to `verify_s1/cluster_out/`.
- **Local:**
  - `verify_s1/recompute_s1.py` / `.json`: sections 1-5;
  - `check_runs_s1.py` / `.json`: section 6;
  - `rescore_s1.py` / `.json`: sections 7-8;
  - `cluster_logs/`: cluster copies of the training logs.
