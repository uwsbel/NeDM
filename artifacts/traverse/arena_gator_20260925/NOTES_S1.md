# NOTES S1: soil stage 1 - training data and models at tiers 0-6 (2026-09-25, 10:55-)

Soil track, step 1 of the resumed workflow (PLAN 7.10). K3 = this folder, G3 =
`/work1/dannegrut/harry/experiments/arena_gator_20260925`. No existing script was edited, nothing in earlier artefact
folders or cluster roots was written, and nothing in `G3/source` was touched (new tools live in `G3/tools/s1`).
Every job is in `G3/e3/submissions.tsv` and has a line in `LOG.md`.

Written 12:05 (task A) and completed 14:05 (task B, capacity).

## 0. Summary

- **Task A soil models are trained and deployed** on tiers 0-6 (7 routes per group) of every arena, as PLAN 7.10
  asks: f104-only (M1, two 5-seed ensembles), f104 + g203 at the same total (M2), three arenas at the same total (M3,
  two ensembles), and three arenas with all their data (A3). All 16 training runs of job 436354 exited 0 in 32 min;
  every checkpoint reloads with identical scores.
- **Matched data really is matched:** M1, M2 and M3 each fit about 29,100 rows from about 7,620 training drives;
  A3 fits 57,898 rows.
- **Offline check (holdout versions, same held-out rows for every model):** ranking the routes of a group from a
  standing start, the within-group AUC of "unsafe" is 0.975 / 0.943 / 0.939 for the f104-only model on f104 / g203 /
  g228, 0.966 / 0.945 / 0.960 for the three-arena model at the same total and 0.978 / 0.962 / 0.974 for the
  three-arena model with all data. So adding arenas helps offline on the arenas it adds (g228: +0.021 at matched
  data, +0.035 with all data), costs nothing on f104 when the data is added (A3) and a little at matched data (M3
  -0.009 on f104). These are offline scores on training arenas, not closed-loop results on unseen arenas.
- **Compute:** every MI350X node was taken by another user until at least 14:55 (and 11 more of their jobs are queued
  ahead of ours), and pytorch still fails on the MI210 nodes. The soil models were therefore trained on an 8-GPU
  MI250X node (`mi2508x`), one training run per GPU (new job script `scripts/ag_train_gpus.sbatch`). The same node type
  also runs soil collection (checked on 8 Gator drives: all valid, same speed per GPU as an MI210), so one extra
  8-GPU soil job (436353) was added for the Gator tiers.
- **Task B (Gator):** all 8,399 Gator soil ids of tiers 0-6 validated (0 launch failures, body-in-soil flag 8.3 %,
  limit 10 %); the Gator fails 87.9 % of them against 67.9 % for the HMMWV. G (Gator-trained) is trained; H (HMMWV-trained
  on exactly those ids) is byte-for-byte the M1 file, so H = the M1a ensemble. Offline, G ranks Gator routes with AUC
  0.925 (dev fold + val) / 0.952 (val), H only 0.81 on the same Gator drives.
- **Collection:** the 18 old MI210 tasks are replaced on soil_v2 by 16 MI210 + 12 MI210 (4-GPU nodes) + 8 MI250X
  GCDs; 60 soil GPUs were running at 14:01; tiers 7-12 need about 86 simulated hours (about 5 wall hours).

## 1. State found at 10:55 and what changed the plan

- Soil collection (all jobs write `G3/soil_v1`), per arena and vehicle (`scripts/ag_soil_tier_status.py`, new):
  g203 and g228 tiers 0-5 complete, tier 6 at 398 / 393 of 606; the Gator tiers 0-3 complete, tier 4 at 471 of 1,200,
  tiers 5-6 almost empty (3,105 Gator rows, about 29 simulated hours, left in tiers 0-6).
- The 18 soil_v1 MI210 tasks (436080) read the old task file, so they can only drive HMMWV rows; the Gator rows are
  driven only by the soil_v2 jobs (436210 and 436215, 7 nodes x 4 MI210). The Gator end of tiers 0-6 was therefore
  about 14:30, not 12:40. With the extra 8-GPU job (section 2) it moved to about 13:35.
- MI350X: all 7 working `mi3501x` nodes were started by one other user at 10:52 for 4 h, with 11 more of their 4-h
  jobs queued at priority 90 against our 25-70 (fair share 20 vs 86). `mi3001x` (MI300X) is also full.
- GPU probe jobs (`scripts/ag_gpu_probe.sh`): pytorch 2.10.0 (the training recipe) still fails on an MI210 compute
  node (devel 436349: "HIP error: file not found" on the first matrix product) and works on an MI250X node
  (mi2508x 436350: 8 devices, matrix product, convolution and GRU forward/backward all fine).

## 2. Compute used

| job | partition | what | state |
|---|---|---|---|
| 436349 | devel (MI210) | pytorch probe | failed as expected (MI210 unusable for training) |
| 436350 | mi2508x | pytorch probe | ok (8 MI250X GCDs) |
| 436353 | mi2508x, 8 h | soil collection on `soil_v2`, frozen dispatcher `ag_crm_collect.py` (b52e1fa6), output `soil_v1` (8 workers, one per GCD) | running since 11:05 on k004-003 |
| 436354 | mi2508x, 3 h | soil training job A (16 runs, 8 lanes) | COMPLETED 11:27-11:59 on k004-004 |
| 436353_0.0 | step inside 436353 | LOAO1_g228 soil holdout as an `srun --overlap` training step on GCD 0 next to a soil worker (`scripts/ag_s1_overlap_train.sh`), a test of the fall-back for task B training | done 12:08-12:14, exit 0; 0.047 s per step after the first seed (0.043 alone); the soil worker on that GCD slowed by about a third for those 6 min, the other 7 not at all |
| 436416 | mi2104x x 3, 10 h | soil collection on `soil_v2` (same recipe as 436353), submitted 12:11 so it is in line when mi2104x nodes free (15:30-16:25) | pending |
| 436353_0.1 | step inside 436353 | G soil deploy + holdout (task B), GCDs 0-1 shared with soil workers | done 13:38-13:52, both exit 0 |
| 436460 | mi2101x x 8, 10 h | soil collection on `soil_v2` (replacement) | tasks 0-5 running from 13:43-13:56, 6-7 pending |
| 436461 | mi2508x x 1, 10 h | soil collection on `soil_v2` (replacement, 8 MI250X GCDs) | running since 13:29 on k004-002 |
| 436490 | mi2101x x 5, 10 h | soil collection on `soil_v2` (replacement for 436080_11-17) | pending (first in line) |

- Billing: the ledger moves by about 10 x the partition weight per node-hour (measured 11:00 -> 12:00: harry
  +6.49 against 0.65 weighted node-hours running), i.e. mi2101x 0.1, mi2104x 0.4, mi3501x 0.125 and mi2508x 0.8 billed
  per node-hour. This module's own jobs: 436353 / 436461 about 0.8 per hour each while they run, 436354 about 0.45, probes < 0.1;
  the 10 h replacements at most 16 x 1 + 3 x 4 + 8 (mi2101x / mi2104x / mi2508x) = about 36 if they all run their full
  limit. Ledger (account-wide, hourly): 639.0 at the session start, 677.8 at 10:53, 700.5 at 14:00 (harry +5.6 to +6.5
  per hour this afternoon, all soil collection jobs included).

## 3. Soil files for the new arenas (tiers 0-6)

Built on the login node from `G3/tools/s1` (a copy of this worktree's `scripts/` and `src/`; `ag_build_ds.py`
`5e2eae03...`, the same file as E4/E5a used) with `scripts/ag_s1_build.sh g203|g228`:

```
ag_build_ds.py --world crm --crm-runs $G3/soil_v1/runs --tasks-crm $G3/tasks/soil_v2.json --tiers 0-6 \
  --arena g203 --vehicle hmmwv --map-root $G3/map_roots/g203 --source-root $G3/source --out $G3/e4/soil_s1/g203_hmmwv --workers 16
```

| arena | drives selected | rejected | groups (train / val / test) | rows | training rows (standing start) | status of the drives | sha256 |
|---|---|---|---|---|---|---|---|
| g203 | 4,240 of 4,242 | 2 by the soil QA check (breakthrough without a stall: `g203_v2_group_0241_route_03`, `_0308_route_07`) | 559 / 24 / 23 | 15,987 | 14,750 (3,911) | goal 1,116, breakthrough 2,542, blockage 577, timeout 5 | `5bb738f2...` |
| g228 | 4,242 of 4,242 | 0 | 520 / 38 / 48 | 16,261 | 13,938 (3,640) | goal 1,154, breakthrough 2,594, blockage 489, timeout 5 | `83f8ab83...` |

- Every drive passed the map check (arena BMP = map root), carries no vehicle block (HMMWV), is a training-pool
  group, and has a task row of tier 0-6. 3.8 rows per drive (as f104).
- f104 HMMWV: the existing file `G3/e4/f104_hmmwv/ci_f104_hmmwv_both.npz` (E4, verified), cut to tiers 0-6 in the
  subset step, training AND val/test rows (`--eval-rows filtered`), so every arena is read at the same tier range.
- Light-column check of every file (`scripts/ag_s1_check_files.py`): tiers 0-6 only, soil rows only, one vehicle,
  one standing-start row per drive, no duplicate ids.

## 4. Subsets (`scripts/ag_s1_subsets.sh f104|new`; `ag_subset.py --world crm --tiers 0-6 --eval-rows filtered`)

Lowest md5 of the group id within each arena (nested sets). Files in `G3/e4/soil_s1/subsets/`, manifests copied to
`K3/e4/soil_s1/subsets/`; table `K3/e4/soil_s1/subsets_table.json`.

| file | design | groups per arena | training drives | fitted rows, deploy | fitted rows, holdout | sha256 |
|---|---|---|---|---|---|---|
| `M1_f104_hmmwv_soil` | M1 (f104 only) | f104 1,089 | 7,622 | 29,210 | 23,205 | `48a29d16...` |
| `M2_hmmwv_soil` | M2 (2 arenas, same total) | f104 545 + g203 545 | 7,628 | 29,048 | 22,941 | `ab556193...` |
| `M3_hmmwv_soil` | M3 (3 arenas, same total) | 363 x 3 | 7,621 | 29,071 | 23,284 | `10c38ddc...` |
| `A3_hmmwv_soil` | A3 (3 arenas, all data) | f104 1,089 + g203 559 + g228 520 | 15,173 | 57,898 | 46,274 | `04d9daf4...` |
| `LC545_f104_hmmwv_soil` | f104 learning curve (holdout only) | f104 545 (+ dev fold) | 4,585 | - | 11,597 | `8351a63e...` |
| `LC272_f104_hmmwv_soil` | f104 learning curve (holdout only) | f104 272 (+ dev fold) | 3,059 | - | 5,726 | `e1dc1fe7...` |
| `LOAO1_g203_hmmwv_soil` | one new arena alone | g203 545 | 3,813 | 14,378 | 11,344 | `7fe5b1bf...` |
| `LOAO1_g228_hmmwv_soil` | one new arena alone | g228 520 (all; 545 do not exist) | 3,640 | 13,938 | 11,409 | `5fc99353...` |
| `LOAO2_f104_g203_hmmwv_soil` | two arenas | 273 + 272 | 3,813 | 14,497 | 11,428 | `c0b908c0...` |
| `LOAO2_f104_g228_hmmwv_soil` | two arenas | 273 + 272 | 3,815 | 14,646 | 11,839 | `18c5f4af...` |
| `LOAO2_g203_g228_hmmwv_soil` | two arenas | 273 + 272 | 3,813 | 14,482 | 11,786 | `77c74fc1...` |

Evaluation files for the offline read-out (`G3/e5/eval_soil/`, copies in `K3/e5/eval_soil/`; dev fold = training
groups with md5 % 5 == 0, plus val and test rows; no other training rows):
`EV_f104_hmmwv_crm` (224 dev-fold + 56 val groups, 8,947 rows, `1bd5233d...`), `EV_g203_hmmwv_crm` (117 + 24, 4,327,
`061ddb4c...`), `EV_g228_hmmwv_crm` (94 + 38, 4,852, `182bfa42...`).

## 5. Training (job 436354, `e5/jobs/soil_s1_A.tsv`)

- Recipe (the recorded soil recipe, identical to the rigid runs except the world): `ci_train.py --arch gru --cond none
  --domain-filter crm --ctx geom --split-eval val --bs 256 --epochs 30 --seeds 5` (trainer defaults lr 2e-3, wd 1e-4),
  deploy runs with `--roundtrip-check`. Trainer = `G3/source/scripts/ci_train.py` `7a4f2d67...` (the file behind every
  model of this study), `ga_train.py` `dc73d0bb...`.
- Job script `scripts/ag_train_gpus.sbatch` (new; the lane format of `ag_train_lanes.sbatch`, each lane pinned to one
  GPU with `HIP_VISIBLE_DEVICES`), `NEDM_VEHICLE` unset, mi2508x (8 MI250X GCDs, 128 cores), 3 h limit:
  `sbatch -p mi2508x -c 128 -t 03:00:00 -J ag_s1_train_A --export=ALL,AG_JOBS=$G3/e5/jobs/soil_s1_A.tsv $G3/tools/s1/scripts/ag_train_gpus.sbatch`.
- 11:27-11:59 on k004-004: 16/16 runs `exit: 0`; 0.04 s per step per GCD (M1 seed about 2.5 min, A3 seed about
  6.5 min); round trip max difference 0.0 on all 30 deploy seeds.
- Platform note: every soil model (task A and G) was trained on MI250X; the first rigid models (M1, M2) were trained on
  MI350X, the later rigid ones (E5a job 436352) on MI250X. Within soil every model uses the same platform and recipe.

| model | plain label | tag | seeds | data | output (G3) | local deploy copy |
|---|---|---|---|---|---|---|
| M1a | f104 only, ensemble 1 | `M1a_soil_deploy` | 0-4 | M1 | `e5/train/soil_s1/M1_soil` | `e5/deploy/M1a_soil/` |
| M1b | f104 only, ensemble 2 | `M1b_soil_deploy` | 5-9 | M1 | same | `e5/deploy/M1b_soil/` |
| M2 | f104 + g203, same total | `M2_soil_deploy` | 0-4 | M2 | `e5/train/soil_s1/M2_soil` | `e5/deploy/M2_soil/` |
| M3a | three arenas, same total, ensemble 1 | `M3a_soil_deploy` | 0-4 | M3 | `e5/train/soil_s1/M3_soil` | `e5/deploy/M3a_soil/` |
| M3b | three arenas, same total, ensemble 2 | `M3b_soil_deploy` | 5-9 | M3 | same | `e5/deploy/M3b_soil/` |
| A3 | three arenas, all data | `A3_soil_deploy` | 0-4 | A3 | `e5/train/soil_s1/A3_soil` | `e5/deploy/A3_soil/` |
| G | Gator-trained, f104 (task B; section 7) | `G_soil_deploy` | 0-4 | G | `e5/train/soil_s1/G_soil` | `e5/deploy/G_soil/` |
| H | HMMWV-trained on the Gator-validated ids (task B) = M1a | (M1a) | 0-4 | H = M1 (same bytes) | - | `e5/deploy/H_soil/README.md` -> `M1a_soil/` |
| holdout versions | offline read-out only | `<M1,M2,M3,A3,LC545,LC272,LOAO*,G>_soil_holdout` | 0-4 | as named | `e5/train/soil_s1/offline_soil` | `e5/train/soil_s1/offline_soil/` |

Each deploy folder has the 5 checkpoints, the trainer summary, the training logits (needed by the fitted-group guard of
the offline scorer) and `SHA256SUMS`; `scripts/ag_deploy_sync.sh` (E5a) copied them and required the cluster sha256 to
be equal.

## 6. Offline within-group AUC (task A)

`bash scripts/ag_s1_offline.sh A` = `scripts/ag_offline_auc.py` (unchanged) on the local 5090 with TF32 off
(`NVIDIA_TF32_OVERRIDE=0`); where the scorer's rows equal the trainer's own held-out rows the two agree exactly
(max |difference| 0.0). Output `e5/offline/soil_s1_A_auc.json`, table `e5/offline/soil_s1_A_auc.md`.

"AUC" = within-group AUC of the unsafe label (can the model rank the routes of one start/goal pair), standing-start
rows; "pick" = share of groups where the route the model ranks safest was unsafe. Holdout-mode ensembles on the dev
fold + val rows (never fitted by any of them) are the fair comparison; deploy ensembles can only be read on the val
rows (their own dev fold was fitted), which are few (24 / 38 groups on g203 / g228).

| model (holdout version) | f104: AUC start / moving, pick | g203 | g228 |
|---|---|---|---|
| f104 only, 272 groups (LC272) | 0.954 / 0.953, 0.289 | 0.924 / 0.942, 0.390 | 0.913 / 0.915, 0.409 |
| f104 only, 545 groups (LC545) | 0.962 / 0.962, 0.279 | 0.939 / 0.945, 0.369 | 0.917 / 0.923, 0.386 |
| f104 only, 1,089 groups (M1) | 0.975 / 0.971, 0.271 | 0.943 / 0.954, 0.355 | 0.939 / 0.935, 0.348 |
| f104 + g203, same total (M2) | 0.957 / 0.966, 0.286 | 0.950 / 0.957, 0.355 | 0.938 / 0.937, 0.364 |
| three arenas, same total (M3) | 0.966 / 0.964, 0.268 | 0.945 / 0.958, 0.383 | 0.960 / 0.958, 0.356 |
| three arenas, all data (A3) | **0.978** / 0.976, 0.257 | **0.962** / 0.965, 0.333 | **0.974** / 0.971, 0.333 |
| g203 only, 545 (LOAO1_g203) | 0.945 / 0.947, 0.307 | 0.939 / 0.954, 0.369 | 0.912 / 0.915, 0.379 |
| g228 only, all 520 (LOAO1_g228; trained 12:08 in step 436353_0.0) | 0.936 / 0.933, 0.318 | 0.946 / 0.956, 0.369 | 0.943 / 0.942, 0.364 |
| f104 + g203, 273 + 272 | 0.960 / 0.957, 0.289 | 0.945 / 0.954, 0.355 | 0.933 / 0.932, 0.379 |
| f104 + g228, 273 + 272 | 0.956 / 0.957, 0.293 | 0.952 / 0.960, 0.355 | 0.948 / 0.939, 0.356 |
| g203 + g228, 273 + 272 | 0.933 / 0.940, 0.329 | 0.951 / 0.962, 0.355 | 0.945 / 0.944, 0.348 |
| groups scored | 280 | 141 | 132 |

Deploy ensembles on the val rows (standing start AUC; f104 56, g203 24, g228 38 groups): M1a 0.977 / 0.897 / 0.915,
M1b 0.985 / 0.904 / 0.919, M2 0.939 / 0.919 / 0.897, M3a 0.972 / 0.941 / 0.963, M3b 0.964 / 0.904 / 0.952,
A3 0.985 / 0.919 / 0.945. The two M1 ensembles differ by up to 0.008 and the two M3 ensembles by up to 0.037 on
these small val sets: that is the seed-noise scale for single-arena val read-outs.

Reading (offline only, training arenas only):
- On the arenas a model has not seen, the f104-only model loses 0.03-0.04 AUC (0.975 on f104 vs 0.94 on g203/g228),
  and more f104 data closes part of that (0.913 -> 0.939 on g228 from 272 to 1,089 groups), so f104-only is still
  data-limited on other arenas.
- At the same total, three arenas beat one arena on g228 (+0.021) and tie on g203 (+0.002), at -0.009 on f104.
- The arenas differ in how hard their routes are to rank: f104 is the easiest (even the g203-only model scores 0.945
  there, more than on g203 itself), g228 the hardest. The two-arena models at 545 groups score 0.93-0.95 on the arena
  they did not see, against 0.945-0.960 for the arenas they did see.
- A3 has the most data and the most arenas and is best everywhere. Whether this carries to closed-loop goal reaching
  on the unseen test arenas is what the soil evaluation measures.

## 7. Task B: the Gator on f104 soil, tiers 0-6

### 7.1 Collection (PLAN 7.7 criteria), complete at 13:33

`scripts/ag_s1_gator_soil_qa.py` on all 8,399 Gator soil ids of tiers 0-6, against the HMMWV `collect_v1` drives of the
same ids (`e5/ids_soil/gator_soil_qa_t0-6_summary.json`):

| criterion | limit | Gator, tiers 0-6 |
|---|---|---|
| ids validated (soil QA check) | >= 95 % | **100 %** (8,399 / 8,399) |
| crashed / NaN | < 1 % | 0 |
| launch-check failures | < 5 % | 0 |
| body in soil (lowest hull point > 0.05 m under the surface for > 1 s in a row) | <= 10 % | **8.3 %** (9.6 % counting > 1 s in total); every flagged drive is a failure anyway, so "flagged counts as failure" changes nothing |
| goal not reached | informative labels: overall 10-90 % and 3 of 4 designed profiles inside | **87.9 %** (HMMWV 67.9 %); by designed profile 2 m/s 90.8 % (just outside), 4 m/s 83.1 %, 6 m/s 82.5 %, 2-6-2 m/s 84.5 % (3 of 4 inside); planner routes 92.1 % |
| discordant pairs | report | 1,801 routes fail only for the Gator, 120 only for the HMMWV |
| simulated hours | report | 77.7 h vs 50.3 h for the same ids (1.54 x) |

- Statuses: Gator 7,284 blocked, 1,014 goal, 101 broke through; HMMWV 4,599 broke through, 2,695 goal, 1,096 blocked.
- Per tier the failure rate is flat (87.3-88.8 %), so tiers 0-6 are a fair sample of the full design.
- The labels are near saturation: in 65 % of the 280 scored f104 groups (dev fold + val) every Gator route of tiers
  0-6 fails (HMMWV: 24 %).

### 7.2 Files and models

- Gator file `G3/e4/soil_s1/f104_gator/ci_f104_gator_crm.npz` (`ag_s1_build.sh gator`: `--vehicle gator`, map root =
  the crm_f104_v1 capture): 8,399 drives, 0 rejected, all carry the Gator vehicle record; 30,827 rows (3.67 per drive,
  HMMWV 3.83), 28,057 training rows (7,622 standing start), groups 1,089 / 56 / 55, sha256 `dfd1975c...`.
- Validated ids `e5/ids_soil/gator_soil_validated_t0-6.txt` (8,399; `ag_e5a_ids.py --world crm` on the tier 0-6 Gator
  rows).
- **G** = `G_f104_gator_soil` (every validated training group, 1,089): 28,057 fitted rows in deploy mode, 22,284 in
  holdout mode, sha256 `b3938949...`.
- **H** = the HMMWV f104 file restricted to exactly the validated ids (`--preset H`, applied to every row). Because
  the Gator validated every id, it is **byte-identical to the M1 soil file** (sha256 `48a29d16...` both;
  `e5/ids_soil/H_vs_M1_soil.json`: same ids, same order). H is therefore not trained again: H = the M1a soil ensemble,
  and its holdout twin is the M1 soil holdout run (`e5/deploy/H_soil/README.md`), as E5a did for rigid.
- Training: no GPU node was free at 13:38 (all MI250X, MI300X and MI350X nodes busy), so G was trained as an
  `srun --overlap` step inside our own soil job 436353 (step 436353_0.1, GCDs 0-1 shared with two soil workers,
  `scripts/ag_s1_overlap_train.sh 436353 $G3/e5/jobs/soil_s1_B.tsv 32`): G deploy (seeds 0-4) and G holdout,
  13:38-13:52, both exit 0, 0.047-0.049 s per step, round trip exact. Deploy copy `e5/deploy/G_soil/`.

### 7.3 Offline read-out (`bash scripts/ag_s1_offline.sh B`, `e5/offline/soil_s1_B_auc.{json,md}`)

Each model scored on the Gator drives and on the HMMWV drives of the same f104 ids (tiers 0-6). The scorer reproduces
the trainer's own numbers exactly on each model's own-vehicle rows (the trainer check also lists other-vehicle row
sets that merely have the same row count; those differences are expected).

| model | on Gator drives: AUC start / moving, pick | on HMMWV drives: AUC start / moving, pick |
|---|---|---|
| G, Gator-trained (holdout version, dev fold + val, 280 groups) | 0.925 / 0.936, 0.686 | 0.822 / 0.855, 0.429 |
| G (deploy, val 56 groups) | 0.957 / 0.935, 0.661 | 0.829 / 0.833, 0.375 |
| H = M1a, HMMWV-trained (holdout version, 280 groups) | 0.812 / 0.833, 0.746 | 0.975 / 0.971, 0.271 |
| H (deploy, val 56 groups) | 0.798 / 0.819, 0.786 | 0.977 / 0.971, 0.268 |
| random pick / best possible pick (280 groups) | 0.879 / 0.650 | 0.693 / 0.239 |

- "Pick" = share of groups where the route ranked safest was unsafe. On the Gator drives G closes 84 % of the gap
  between a random and the best possible pick (0.879 -> 0.686, best 0.650), the HMMWV-trained model 58 %.
- PLAN 3 criterion (3), offline within-group AUC on held-out groups >= 0.95: **met on the val groups (0.952 holdout,
  0.957 deploy), not on the larger dev fold + val set (0.925)**. Only 96 of the 280 groups (672 of the 1,960
  standing-start rows, 34 %) have both outcomes for the Gator (182 groups: every route fails, 2: none fails), so the
  AUC rests on 832 (unsafe, safe) route pairs. (Corrected by the verifier, VERIFY_S1.md: the first version said
  "42 % of the rows ... 832 of 1,960 rows"; 832 is the trainer's pair count, not a row count.)
- Each vehicle's model ranks the other vehicle's outcomes clearly worse (0.81-0.82 against 0.93-0.98): the two
  vehicles fail on different routes, which is what the closed-loop comparison of G and H on the Gator will measure.

## 8. Soil collection capacity (step 4)

- The 18 soil_v1 MI210 tasks (436080) stop claiming 13:41 (tasks 0-10), 14:09 (11) and 14:52-15:03 (12-17); they read
  the old file and could only drive HMMWV rows. Their capacity is replaced on `soil_v2` with the frozen dispatcher,
  same output folder, `ag_soil_launch.sh` (queue check, submissions recorded), all with a 10 h limit:
  436416 (mi2104x x 3 = 12 MI210, submitted 12:11, scheduler start about 15:34), 436460 (mi2101x x 8, 13:30),
  436461 (mi2508x x 1 = 8 MI250X GCDs, 13:29, started 13:29), 436490 (mi2101x x 5, 14:02), plus 436208 (mi2101x x 3,
  pending since 04:47, started 13:42). 436460_0-5 started 13:43-13:56; 436460_6-7, 436490 and 436416 were pending at
  14:05 (first in line on mi2101x). That is 16 MI210 + 12 MI210 + 8 GCDs requested for the 18 released MI210.
- At 14:01, 60 soil GPUs were running for this effort (16 mi2101x, 28 on mi2104x, 16 MI250X GCDs), 0 failed ids,
  0 collection failures, 0 retired workers. Left in tiers 7-12: g203 2,648, g228 2,622, Gator 6,046 rows, about 86
  simulated hours (about 5 wall hours at this capacity).
- The soil_v2 jobs 436210 / 436215 (28 MI210) stop claiming about 17:30-18:10 and 436353 about 18:50; they need
  replacing then as well.
- **For the soil evaluation step:** jobs read their task file once at start, so the soil_v2 jobs above will never drive
  evaluation rows. A soil_v3 (superset of soil_v2, same folder, e3/README recipe) needs its own jobs; capacity can be
  moved to it by cancelling soil_v2 tasks (a killed episode is lost and re-taken after 25 min), at the risk of other
  users taking the freed nodes; `STOP_CLAIMS` would stop every job in the folder, soil_v3 ones included.

## 9. Files

- New scripts: `scripts/ag_soil_tier_status.py`, `ag_gpu_probe.sh`, `ag_train_gpus.sbatch`, `ag_s1_build.sh`,
  `ag_s1_subsets.sh`, `ag_s1_offline.sh`, `ag_s1_auc_table.py`, `ag_s1_manifest_table.py`, `ag_s1_check_files.py`,
  `ag_s1_gator_soil_qa.py` (all also in `G3/tools/s1/scripts`).
- Cluster: `G3/tools/s1` (tool tree + logs), `G3/e4/soil_s1` (built files, subsets), `G3/e5/eval_soil`,
  `G3/e5/train/soil_s1`, `G3/e5/ids`, `G3/e5/jobs/soil_s1_*.tsv`, `G3/e5/logs/*_436354.*`.
- Local: `e4/soil_s1/` (records, manifests, tool logs), `e5/eval_soil/`, `e5/deploy/*_soil/` and
  `e5/deploy/soil_s1_models.json` (every soil deploy ensemble: folder, checkpoints with sha256, seeds, data file and
  its sha256, fitted rows), `e5/ids_soil/` (validated Gator ids, Gator collection read-out, H vs M1),
  `e5/train/soil_s1/offline_soil/`, `e5/offline/soil_s1_{A,B}_auc.{json,md}`, `e5/jobs/soil_s1_*.tsv`
  (`soil_s1_B_withH.tsv` was prepared for the case H != M1 and not used), `e5/logs/`.
- Also new: `scripts/ag_s1_overlap_train.sh` (training as an overlap step inside our soil job).
