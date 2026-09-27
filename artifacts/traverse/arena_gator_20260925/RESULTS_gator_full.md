# RESULTS: task B stage 2. The Gator and HMMWV soil planners retrained on all collected data (2026-09-26, 19:20-21:40)

PLAN 7.10 said a stage-2 retrain of G and H on more tiers would happen if the collection finished. It did: the Gator
drove all 15,235 HMMWV `collect_v1` soil ids (tiers 0-12) by 23:33 on 09-25. This file reports the retrain and its
closed-loop test on the 800-group f104 suite. K3 = this folder, G3 = `/work1/dannegrut/harry/experiments/arena_gator_20260925`.
Every step has a line in `LOG.md` (19:24-21:2x). No existing script was edited and nothing in an earlier artefact folder
or cluster root was written. The new scripts are `scripts/ag_bf_*` and the cluster tool tree is `G3/tools/bf`.

Model labels used below:

| code | plain label | data |
|---|---|---|
| **G_full** | Gator-trained, all tiers | the Gator's own drives of all 15,235 f104 ids (tiers 0-12) |
| **H_full** | HMMWV-trained, all tiers | the HMMWV drives of exactly the ids the Gator validated (all 15,235), which is the full HMMWV f104 soil file |
| G | Gator-trained, tiers 0-6 (stage 1) | the Gator's drives of the 8,399 ids of tiers 0-6 |
| H (= M1a) | HMMWV-trained, tiers 0-6 (stage 1; the same file as "f104 only", ensemble 1) | the HMMWV drives of those 8,399 ids |
| straight 6 | straight route at 6 m/s (no model) | - |

## 0. Answer in brief

- **By the declared rule, more data helps the Gator planner a little; the evidence is marginal.** The Gator-trained
  planner on all tiers fails to reach the goal on **32.6 %** of the 800 start/goal pairs. The tiers 0-6 version fails
  on 34.8 %. The difference is **-2.1 points**. The declared rule counts this as an improvement (Holm-adjusted p 0.018),
  but:
  - the group-level 95 % interval is [-4.5, +0.1];
  - the pairs split 52 won / 35 lost (exact McNemar two-sided p 0.086);
  - 5 of the 9 terrain clusters get better and 2 get worse;
  - there is only one ensemble per model (and the two were trained on different GPU types), so how much two trainings
    on the same data differ is unknown for the Gator.
- **The Gator-trained planner beats the HMMWV-trained one on the Gator by more than before.**
  - All tiers: G_full (Gator-trained, all tiers) 32.6 % against H_full (HMMWV-trained, all tiers) **56.4 %**, i.e.
    **-23.8 points** (cluster 90 % [-30.4, -16.5]; 206 pairs won, 16 lost; Holm-adjusted p 0.001).
  - Tiers 0-6 (stage 1): G 34.8 % against H 51.9 % (-17.1 points).
  - The gap grew by 6.6 points, mostly because the HMMWV planner got worse on the Gator (+4.5 points); the Gator
    planner's own gain (-2.1) is the rest.
- **With more HMMWV data, the HMMWV-trained planner did slightly better on the HMMWV and worse on the Gator.**
  - On the HMMWV: 4.4 % against 5.5 % (-1.1 points, group 95 % [-2.6, +0.4]; weak).
  - On the Gator: 56.4 % against 51.9 % (**+4.5 points**, 95 % [+1.8, +7.3]; 80 pairs lost, 44 won; 8 of 9 clusters
    worse).
  - These intervals cover only the choice of start/goal pairs. Each side is one 5-seed ensemble, trained on different
    GPU types. How far two trainings on the same HMMWV data differ when the Gator drives their picks was not measured,
    so part of the +4.5 may be training noise.
  - A likely reading is that the extra HMMWV data makes the planner more specific to the HMMWV. The HMMWV-trained
    all-tier planner reaches the goal on 95.6 % of the pairs when the HMMWV drives it. That is the level of the frozen
    HMMWV reference in PLAN 1.1 (95.8 %).
- **Both all-tier planners beat the straight route on the Gator** (85.8 % fail): by 53.1 points (G_full, Gator-trained)
  and 29.4 points (H_full, HMMWV-trained).
- **Task B "works" criteria for the all-tier Gator planner:**
  1. It beats straight 6 m/s on the Gator: yes.
  2. It beats H_full (HMMWV-trained, all tiers) or is within 2 points of it: yes, it is better by 23.8 points.
  3. Offline within-group AUC >= 0.95 on held-out groups: only just, and only on the 56 val groups (0.950). Only 22 of
     those 56 groups have both a safe and an unsafe route, so the value rests on 603 route pairs. On the dev fold + val
     (280 groups, 116 of them with both outcomes) it is 0.946, up from 0.936 for the tiers 0-6 model.
  4. Headroom closed: the Gator planner removes 62 % [58, 66] of the straight route's failures. The HMMWV planner
     removes 86 % [82, 91] on the HMMWV.
- **The Gator collected the same data on every tier:** 15,235 / 15,235 ids validated, 0 launch failures, 0
  crashed or NaN, body-in-soil flag 8.5 % (limit 10 %). The Gator fails on 88.2 % of these routes, the HMMWV on
  68.1 %, so the Gator's labels are close to saturated.
- **Cost:** 5.12 billed node-hours (training 0.08, drives 5.04). The session total is 113.3 (sacct since 09-25 00:00;
  soft cap 150).

## 1. Data (all tiers)

### 1.1 Collection read-out

`scripts/ag_s1_gator_soil_qa.py` (unchanged) compares the 15,235 Gator training rows of soil_v3 with the HMMWV
`collect_v1` drives of the identical ids. Output: `e5/ids_bf/gator_soil_qa_all_summary.json`. The other track's
`RESULTS_soil.md` 3.3 found the same numbers independently.

| criterion (PLAN 7.7) | limit | Gator, tiers 0-12 |
|---|---|---|
| ids validated | >= 95 % | **100 %** (15,235 / 15,235; all complete, all carry the Gator vehicle block) |
| crashed / NaN | < 1 % | 0 |
| launch-check failures | < 5 % | 0 |
| body in soil (lowest hull point > 0.05 m under the surface for > 1 s in a row) | <= 10 % | **8.5 %** (9.8 % counting > 1 s in total) |
| goal not reached | report | **88.2 %** (HMMWV 68.1 %); 3,263 routes fail only for the Gator, 201 only for the HMMWV; flat by tier (87.3-89.3 %) |
| by designed speed profile, Gator / HMMWV | report | 2 m/s 91.3 / 86.2 %, 4 m/s 83.8 / 56.3 %, 6 m/s 82.7 / 33.5 %, 2-6-2 m/s 84.2 / 54.5 %, planner routes 92.2 / 83.9 % |
| simulated hours | report | 141.1 h against 91.5 h for the same ids |
| statuses (Gator) | report | 13,231 blocked, 1,805 goal, 198 broke through, 1 timeout |

### 1.2 Files

All files were built on the cluster login node from `G3/tools/bf`, a copy of this worktree's `scripts/` and `src/`
(`ag_build_ds.py` `5e2eae03`, the same file as S1, E4 and E5a).

- **Gator file** (`scripts/ag_bf_build.sh gator`), written to `G3/e4/soil_bf/f104_gator/ci_f104_gator_crm.npz`
  (sha256 `0bd3ee63`). Arguments: `ag_build_ds.py --world crm --vehicle gator --tasks-crm soil_v3.json --tiers 0-12`,
  with the crm_f104_v1 map root.
  - 15,235 drives selected, 0 rejected, map check ok.
  - 55,826 rows; 13,821 standing-start training rows.
  - Tiers: 0-11 have 1,200 each, except tier 0 with 1,199 (group 0211's tier-0 id was never a training id, VERIFY_S1 1).
    Tier 12 has 836.
- **Subsets** (`scripts/ag_bf_subsets.sh`: `ag_subset.py --world crm --tiers 0-12 --eval-rows filtered`). Manifests
  are in `e4/soil_bf/`.

| file | plain label | training groups | rows | fitted rows, deploy / holdout | sha256 |
|---|---|---|---|---|---|
| `G_full_f104_gator_soil` | Gator-trained, all tiers | 1,089 | 55,826 | **50,822** / 40,367 (stage-1 G: 28,057 / 22,284) | `20ce0ed0` |
| `H_full_f104_hmmwv_soil` | HMMWV-trained on the Gator-validated ids, all tiers | 1,089 | 58,268 | **52,923** / 42,027 (stage-1 H: 29,210 / 23,205) | `5e9c2961` |
| `F104all_f104_hmmwv_soil` | full HMMWV f104 soil file, no id filter (comparison only) | 1,089 | 58,268 | - | `5e9c2961` |
| `EV_f104_gator_full_crm` / `EV_f104_hmmwv_gatorids_full_crm` | dev fold + val + test rows, all tiers (offline scoring) | 224 dev-fold groups | 15,459 / 16,241 | - | `01bbbee3` / `375a3887` |

- **H_full is the full HMMWV f104 soil file itself.** The Gator validated every id, so restricting the HMMWV file to
  the validated ids changes nothing. The two files are byte-identical (sha256 `5e9c2961` both; same ids, same order;
  `e5/ids_bf/H_full_vs_F104all_soil.json`), and they hold every soil row of the E4 f104 file (58,268). H_full was
  still trained here, because no earlier model had this recipe on this file.

## 2. Training

- **Recipe.** The same as the stage-1 soil models and S1 job B; the job lists `e5/jobs/soil_bf_{G,H}.tsv` differ from
  `soil_s1_B.tsv` only in the data path.
  - `ci_train.py --arch gru --cond none --domain-filter crm --ctx geom --split-eval val --bs 256 --epochs 30 --seeds 5 --seed0 0`;
  - deploy runs add `--roundtrip-check`;
  - each model also has a holdout run, used for the offline read-out.
  - Trainer: `G3/source/scripts/ci_train.py` `7a4f2d67`. Job script: `scripts/ag_train_gpus.sbatch` (S1).
- **Jobs.** Two lanes each, on one MI350X (`mi3501x`, `-t 3h`):

| job | what | ran | runs and exit | speed |
|---|---|---|---|---|
| 439361 | G_full deploy + holdout | 19:28-19:48, 19.9 min | both exit 0 | 0.031-0.041 s/step per seed, 5,940 steps per deploy seed |
| 439362 | H_full deploy + holdout | 19:28-19:47, 19.2 min | both exit 0 | 0.027-0.036 s/step per seed, 6,180 steps per deploy seed |

  - Checkpoint round trip: max difference 0.0 on all 10 deploy seeds.
- **Copies.** Deploy ensembles are in `e5/deploy/G_full_soil` and `e5/deploy/H_full_soil`, copied with
  `scripts/ag_deploy_sync.sh`; `SHA256SUMS` are equal on both sides. Manifest: `e5/deploy/soil_bf_models.json`.
  Holdout runs are in `e5/train/soil_bf/offline_soil_bf/`.
- **Platform deviation.** No MI250X node was free, so the all-tier models were trained on MI350X. The stage-1 soil
  models were trained on MI250X. The recipe and trainer file are the same, but G_full vs G and H_full vs H compare
  across platforms. G_full and H_full share the platform.

## 3. Offline ranking (`scripts/ag_bf_offline.sh`; `e5/offline/soil_bf_B_auc.{json,md}`)

**Method.** `scripts/ag_offline_auc.py` (unchanged) ran on the 5090 with TF32 off. Where the rows are a model's own
training-vehicle rows, the scorer equals the trainer's numbers exactly (difference 0.0).

**What is scored.** The within-group AUC of "unsafe" from a standing start: can the model rank the routes of one
start/goal pair? "Pick fails" is the share of groups whose lowest-risk route was unsafe.

**Which rows.** Holdout versions are scored on the dev fold + val (280 groups; never fitted by any holdout model).
Deploy versions are scored on val only (56 groups).

All tiers:

| model | Gator drives: AUC dev+val / val, pick fails | HMMWV drives of the same ids: AUC dev+val / val, pick fails |
|---|---|---|
| G_full, Gator-trained, all tiers (holdout / deploy on val) | **0.946** / 0.950 (deploy 0.950), 0.632 | 0.814 / 0.825, 0.350 |
| G, Gator-trained, tiers 0-6 | 0.936 / 0.949 (deploy 0.952), 0.643 | 0.811 / 0.829, 0.336 |
| H_full, HMMWV-trained, all tiers | 0.827 / 0.847 (deploy 0.847), 0.743 | **0.979** / 0.983 (deploy 0.991), 0.204 |
| H, HMMWV-trained, tiers 0-6 (M1 holdout / M1a) | 0.823 / 0.857 (deploy 0.849), 0.721 | 0.974 / 0.969 (deploy 0.971), 0.200 |
| no model: pick fails for a random pick / for the best possible pick (dev+val; not AUC values) | 0.878 / 0.582 | 0.688 / 0.136 |

The Gator AUC on all tiers rests on 2,976 (unsafe, safe) route pairs. They come from only 116 of the 280 groups: in
163 groups every route is unsafe for the Gator, and in 1 every route is safe. On the 56 val groups the 603 pairs come
from 22 groups. At tiers 0-6 it rested on 832 pairs in 96 groups. (HMMWV drives: 6,922 pairs from 236 of the 280
groups.) On the tiers 0-6 rows alone (the S1 evaluation files):
- G_full scores 0.944 on dev+val and 0.936 on val; G scores 0.925 and 0.952.
- H_full on HMMWV drives scores 0.983 and 0.982; H scores 0.975 and 0.972.

**Reading.**
- More tiers raise the Gator model's ranking on the larger held-out set by 0.010 (0.936 -> 0.946). The lowest-risk pick
  fails slightly less often (0.643 -> 0.632).
- On the 56 val groups the two Gator models tie (0.950 vs 0.949-0.952). That is inside the seed-noise scale S1 found
  for val-only read-outs (up to 0.04).
- Each vehicle's model still ranks the other vehicle's outcomes clearly worse (0.81-0.85).

## 4. Closed loop on the 800-group f104 suite

### 4.1 Protocol (declared before any drive)

- **Spec.** `e6/analysis/spec_soil_v1_Bfull.json`, sha256 `a1d75180...`, written by `scripts/ag_bf_spec.py`. It was
  frozen at 19:29, while the models were still training and before any pick or drive existed.
  - Same tool (`ag_analyze.py`) and settings as `spec_soil_v1.json`: margin 2 points, alpha 0.05, 4,000 bootstrap
    draws, seed 0, clusters = the nearest terrain feature to the start-goal midpoint (9 on f104), at least 50 groups.
  - fail = goal not reached.
  - **Family, Holm at 0.05 over the one-sided cluster-bootstrap p:**
    - F1 (primary): G_full vs H_full on the Gator;
    - F2: G_full vs straight 6 on the Gator;
    - F3: H_full vs straight 6 on the Gator;
    - F4: G_full vs stage-1 G on the Gator.
  - Group bootstrap and exact McNemar are reported beside every contrast.
  - **Disclosed in the spec:** the stage-1 task B outcomes were already known when it was written (the stage-1 drives had
    been indexed for sizing at 19:28). Nothing about G_full or H_full was known.
- **Picks** (`scripts/ag_bf_picks.sh`).
  - `ag_picks.py --world crm --mode free`, i.e. CEM 4 x 64 from the case pose at rest, on all 800 suite groups. Same
    suite cases and f104 map root as the stage-1 picks; map check passed.
  - Each directory was planned twice, and both runs were identical (`--rerun-check`).
  - Locks: G_full `04cd65c3`, H_full `a55070c1`. Set lock `e6/picks/LOCK_crm_bfull.sha256`, ALL `52f510b7`, written at
    19:59:55, before any drive.
  - H_full's picks serve both vehicles. The planner does not know the vehicle; stage-1 H was used the same way.
- **Rows** (`scripts/ag_bf_rows.sh` = `ag_eval_tasks.py build --world crm` with soil_v3 in `--existing`).
  - Every stage-1 arm is listed again and resolves by route content to exactly its stage-1 drive (800 / 800 run ids
    each).
  - Tier -9 (Gator): 1,541 new drives. G_full 762 and H_full 779; 38 and 21 of their picks equal a route already
    driven, and those drives are reused.
  - Tier -8 (secondary, H_full on the HMMWV): 777 new drives.
  - `e3/tasks/soil_v4.json`: 46,219 rows = soil_v3 unchanged as its prefix + 2,318 new rows; sha256 `d73fe77b`;
    written by `scripts/ag_bf_soil_v4.py`. It is staged read-only in `G3/tasks`, and 3,894 staged files are hash-equal
    on G3.
- **Drives.**
  - Launched at 20:01 with the e3/README recipe (frozen dispatcher `ag_crm_collect.py` `b52e1fa6`, output
    `G3/soil_v1`, `NEDM_VEHICLE` unset). Jobs:
    - 439414: mi3501x x 7;
    - 439415: mi2104x x 5 (20 MI210);
    - 439426: mi2101x x 10.
  - All 2,318 rows were done at 21:20; every job COMPLETED after 1 h 17-19 min. 0 failed ids, 0 worker errors.
  - New simulated time: 11.4 h Gator, 4.0 h HMMWV.
- **Index and checks.**
  - The index files of all 6,282 run folders were synced into `e6/runs_soil` (`scripts/ag_bf_sync.sh`).
  - Index `e6/index/soil_eval_bfull.json`: 6,400 rows (8 arms x 800), 0 missing.
  - Drive QA (`ag_s2_extras.py`, unchanged, run on the cluster on a G3-path copy of the index;
    `e6/analysis/soil_extras_Bfull.json`): 0 QA flags, 0 launch failures and 0 non-finite runs over 6,282 runs.

### 4.2 Rates (800 groups)

| arm (plain label) | vehicle | goal not reached | goal reached | blocked / broke through / goal | body-in-soil flag | median time (s) | mean predicted risk of the picks |
|---|---|---|---|---|---|---|---|
| **G_full**, Gator-trained, all tiers | Gator | **32.6 %** | 67.4 % | 254 / 7 / 539 | 1.5 % | 18.9 | 0.181 |
| G, Gator-trained, tiers 0-6 (stage 1) | Gator | 34.8 % | 65.3 % | 264 / 14 / 522 | 2.1 % | 18.4 | 0.184 |
| **H_full**, HMMWV-trained, all tiers | Gator | **56.4 %** | 43.6 % | 440 / 11 / 349 | 2.5 % | 20.9 | 0.0045 |
| H, HMMWV-trained, tiers 0-6 (stage 1) | Gator | 51.9 % | 48.1 % | 405 / 10 / 385 | 3.1 % | 21.9 | 0.0021 |
| straight 6 m/s | Gator | 85.8 % | 14.2 % | 679 / 7 / 114 | 11.1 % | 17.0 | - |
| H_full, HMMWV-trained, all tiers | HMMWV | 4.4 % | 95.6 % | 7 / 27 / 765 (+1 timeout) | - | 17.1 | 0.0045 |
| H, HMMWV-trained, tiers 0-6 | HMMWV | 5.5 % | 94.5 % | 7 / 36 / 756 (+1 timeout) | - | 17.7 | 0.0021 |
| straight 6 m/s | HMMWV | 32.2 % | 67.8 % | 30 / 227 / 542 (+1 rollover) | - | 9.9 | - |

Goal not reached by terrain stratum (descriptive, not declared):

| arm | vehicle | hill (370) | crater (280) | other: long traverse, roughness (150) |
|---|---|---|---|---|
| G_full | Gator | 41.1 % | 27.9 % | 20.7 % |
| G | Gator | 44.3 % | 28.6 % | 22.7 % |
| H_full | Gator | 55.9 % | 62.5 % | 46.0 % |
| H | Gator | 53.8 % | 56.8 % | 38.0 % |
| H_full | HMMWV | 6.5 % | 2.5 % | 2.7 % |
| H | HMMWV | 8.6 % | 2.1 % | 4.0 % |

The 200 declared in-distribution groups show the same order. On the Gator: G_full 35.0 %, G 36.0 %, H_full 62.5 %,
H 60.0 %, straight 89.0 %. On the HMMWV: H_full 2.5 %, H 3.5 %.

### 4.3 Declared family (Holm at 0.05; `e6/analysis/results_soil_v1_Bfull.{json,txt}`)

diff = rate(test) - rate(ref), in points of "goal not reached"; negative means the test arm fails less.

| test | plain question | diff | cluster 90 % | group 95 % | pairs won / lost (McNemar two-sided p) | Holm-adjusted p | decision |
|---|---|---|---|---|---|---|---|
| F1 PRIMARY | Gator-trained vs HMMWV-trained, all tiers, both on the Gator | **-23.8** | [-30.4, -16.5] | [-27.0, -20.4] | 206 / 16 (3e-43) | 0.001 | **improves** |
| F2 | all-tier Gator planner vs straight 6, on the Gator | -53.1 | [-58.4, -47.3] | [-56.6, -49.4] | 433 / 8 (1e-116) | 0.001 | improves |
| F3 | all-tier HMMWV planner vs straight 6, on the Gator | -29.4 | [-34.1, -24.6] | [-33.0, -25.8] | 262 / 27 (2e-49) | 0.001 | improves |
| F4 | Gator-trained, all tiers vs tiers 0-6, on the Gator | **-2.1** | [-3.7, -0.5] | [-4.5, +0.1] | 52 / 35 (0.086) | 0.018 | improves (marginal, see below) |

Each of F1-F3 improves in all 9 clusters, and 0 to 17 groups have identical picks.

On F4: 37 of the 800 G_full picks are the same route as G's. 5 clusters get better, 2 get worse and 2 are unchanged
(sign p 0.45). The group-level one-sided p is 0.039.

The decision comes from the cluster bootstrap over 9 clusters, as in the stage-1 spec. With so few clusters the cluster
interval is narrower here than the group interval at the same level (90 %: cluster [-3.7, -0.5], group about
[-4.0, -0.3] in the verifier's re-run). F4 therefore passes at 90 % by both bootstraps, but at 95 % only by the cluster
one.

No second ensemble of G was trained, so the training-noise floor for the Gator is unknown. On the unseen arenas, the
two f104-only HMMWV ensembles differed by 1.1 points pooled over 1,000 groups (cluster 90 % [-3.2, +0.9]; RESULTS_soil),
at a failure rate of about 12 %, and by up to 6.4 points on single arenas of 125 groups. The Gator failure rates here
are roughly 3 to 5 times higher. None of the intervals above includes training noise.

Read F4 as a small, plausible gain, not a firm one.

### 4.4 Secondary contrasts (unadjusted)

| contrast | diff | intervals | pairs lost / won by the test arm |
|---|---|---|---|
| HMMWV-trained, all tiers vs tiers 0-6, **on the Gator** | **+4.5** (56.4 vs 51.9) | cluster 90 % [+2.6, +6.3], group 95 % [+1.8, +7.3] | 80 / 44 (p 0.0016); 8 of 9 clusters worse |
| HMMWV-trained, all tiers vs tiers 0-6, on the HMMWV | -1.1 (4.4 vs 5.5) | cluster 90 % [-2.2, -0.2], group 95 % [-2.6, +0.4] | 15 / 24 (p 0.20) |
| all-tier HMMWV planner vs straight 6, on the HMMWV | -27.9 | cluster 90 % [-41.5, -16.1] | 10 / 233 |
| Gator-trained vs HMMWV-trained, all tiers, on the Gator, **unsafe** | -23.4 | cluster 90 % [-29.9, -16.4] | 16 / 203 |
| all-tier Gator planner vs the stage-1 HMMWV planner, on the Gator | -19.3 | cluster 90 % [-24.1, -13.8] | 15 / 169 |
| stage-1 G vs stage-1 H on the Gator (reproduces the stage-1 primary) | -17.1 | cluster 90 % [-21.6, -12.4] | 22 / 159 |
| Gator-trained all tiers on the Gator vs HMMWV-trained all tiers on the HMMWV | +28.2 | group 95 % [+25.0, +31.5] | 235 / 9 |
| HMMWV-trained all tiers: Gator vs HMMWV (same routes) | +52.0 | group 95 % [+48.4, +55.5] | 417 / 1 |

The first two rows compare single ensembles trained on different GPU types (MI350X all tiers, MI250X tiers 0-6). Their
intervals cover the choice of start/goal pairs only, not the difference between two trainings on the same data (see
the training-noise note in 4.3).

**Headroom closed** is the share of straight 6's failures a planner removes:
- on the Gator: G_full **62 %** [58, 66], G 59 % [56, 63], H_full 34 % [30, 38], H 40 % [36, 43];
- on the HMMWV: H_full **86 %** [82, 91], H 83 % [78, 88].

**Time to the goal on joint successes** (median paired ratio):
- G_full / H_full 0.89 [0.88, 0.91], 333 pairs (the Gator planner is faster);
- G_full / G 1.02 [1.01, 1.03], 487 pairs;
- H_full / H on the HMMWV 0.98 [0.96, 0.99], 741 pairs.

### 4.5 Task B "works" criteria, all-tier models (PLAN 3; spec note)

| criterion | rule | all tiers | stage 1 (tiers 0-6) |
|---|---|---|---|
| (1) G beats straight 6 m/s on the Gator | F2 improves | **yes**: -53.1 points | yes: -51.0 |
| (2) G beats H on the Gator or is within 2 points | upper one-sided 95 % bound of F1 < +2 | **yes**: better by 23.8 (bound -16.5) | yes: better by 17.1 |
| (3) offline within-group AUC >= 0.95 on held-out groups | >= 0.95 | **val only**: 0.950 on val (56 groups, 22 with both outcomes), 0.946 on dev fold + val (280, 116 with both outcomes) | val only: 0.952 / 0.925 |
| (4) headroom closed vs the HMMWV's | reported | 62 % on the Gator against 86 % for H_full on the HMMWV | 59 % against 83 % |

## 5. Reading

1. **The Gator can be taught; more of its own data probably adds a little.** Giving the Gator 1.8 times the training
   data (8,399 -> 15,235 drives, tiers 0-6 -> 0-12) lowered its failure on the suite by about 2 points (65.3 % ->
   67.4 % goal reached). Offline, the held-out ranking rose by 0.01. Both gains are small, and the closed-loop one is
   within what a second training might change. The Gator's labels are near saturation: 88 % of its training routes
   fail, and in 693 of the 1,200 groups (58 %) every one of its routes fails (HMMWV: 180 groups, 15 %). That limits
   what more of the same routes can teach.
2. **HMMWV data does not transfer better when there is more of it.** The all-tier HMMWV planner is slightly better on
   the HMMWV and 4.5 points worse on the Gator (single ensembles; training noise not measured, see 4.4).
   - Its picks carry a mean predicted risk of 0.45 %, yet they fail on 56 % of the pairs when the Gator drives them
     (stage-1 H: 0.21 % predicted, 52 % failed).
   - It is confidently wrong about the Gator. With more HMMWV data its picks suit the Gator even less, although its
     predicted risk did not fall.
   - The Gator-trained planner's gain over it therefore grew, from 17 to 24 points, mostly for this reason.
3. **The Gator result still falls short of the HMMWV's.** The Gator planner on the Gator fails on 28 points more pairs
   than the HMMWV planner on the HMMWV (32.6 % vs 4.4 %). It closes 62 % of the straight route's failures against
   86 %. Part of this is the vehicle: with its stock rear-wheel drive it stalls on climbs, rear wheels spinning
   (NOTES_E2, E3b1). The soil wheel model also matters: in the pilot, wheels 0.08 m larger lowered the Gator's failure
   by 13 points, just under the 15-point rule for "results depend on the wheel model".

## 6. Deviations and caveats

- **An extra arm was declared.** H_full on the HMMWV was not in the requested contrasts. It was declared in the frozen
  spec as a secondary, later-tier arm, before any drive, and cost 777 drives (4.0 simulated h, about a quarter of the
  drive billing).
- **Training platform:** MI350X for the all-tier models, MI250X for stage 1 (section 2).
- **Spec timing:** it was written after the stage-1 task B outcomes were known (disclosed in the spec). The new arms
  were unknown.
- **Statistics:** 9 clusters on f104. See F4 in 4.3 for what this does to marginal results. Only one ensemble per
  model.
- **Reuse:** stage-1 drives (G, H, straight 6 m/s on both vehicles) were reused by route content, not re-driven.
  This study's soil drives reproduce across nodes and GPU types (bitid and drift checks, NOTES_E3a/E3b2/E6a 5.3), so
  pairing a 09-26 drive with a 09-25 drive of the same group is sound. Of the new picks, 59 (Gator) and 23 (HMMWV)
  equal an already-driven route and use that drive.
- **Collection caveats carry over from stage 1:** the calibrated soil wheel cylinders, the chassis not coupled to the
  soil, the flat-ground map lookup (f104 rmse 0.050 m), and a body-in-soil flag of 8.5 % on the training drives
  (1.5-3.1 % on the planner drives, 11.1 % on straight 6 m/s).

## 7. Compute (sacct; `scripts/ag_bf_billed.py`: partition weights of NOTES_S1 2, reproduces the 108.2 at 19:18)

| job | partition | what | elapsed | billed |
|---|---|---|---|---|
| 439361 | mi3501x x 1 | G_full training (deploy + holdout) | 19.9 min | 0.04 |
| 439362 | mi3501x x 1 | H_full training (deploy + holdout) | 19.2 min | 0.04 |
| 439414 | mi3501x x 7 | soil drives (soil_v4) | 1 h 17 min each | 1.13 |
| 439415 | mi2104x x 5 (20 MI210) | soil drives | 1 h 18 min each | 2.61 |
| 439426 | mi2101x x 10 | soil drives | 1 h 18 min each | 1.30 |
| **total** | | | | **5.12** |

The session total is 113.3 billed node-hours (since 09-25 00:00), under the ~130 target and the 150 soft cap. The
queue is empty.

## 8. Files

- **Scripts (new):** `scripts/ag_bf_build.sh`, `ag_bf_subsets.sh`, `ag_bf_offline.sh`, `ag_bf_picks.sh`,
  `ag_bf_spec.py`, `ag_bf_rows.sh`, `ag_bf_soil_v4.py`, `ag_bf_sync.sh`, `ag_bf_status.sh`, `ag_bf_billed.py`.
  Copies are in `G3/tools/bf/scripts`.
- **Cluster:**
  - `G3/e4/soil_bf/` (Gator all-tier file, subsets);
  - `G3/e5/eval_soil_bf/`, `G3/e5/ids_bf/`;
  - `G3/e5/train/soil_bf/` (deploy + holdout runs);
  - `G3/e5/jobs/soil_bf_{G,H}.tsv`, `G3/e5/logs/*_43936[12]*`;
  - `G3/tasks/soil_v4.json` (read-only);
  - `G3/tools/bf/logs/`.
- **Local:**
  - models: `e5/deploy/{G_full_soil,H_full_soil}/` (with `SHA256SUMS`), `e5/deploy/soil_bf_models.json`,
    `e5/train/soil_bf/offline_soil_bf/`;
  - data records: `e4/soil_bf/` (manifests, build record, tool logs), `e5/eval_soil_bf/` (evaluation files, 0.55 GB),
    `e5/ids_bf/` (validated ids, collection read-out, H_full vs full-file comparison);
  - offline read-out: `e5/offline/soil_bf_B_auc.{json,md}`;
  - jobs and logs: `e5/jobs/soil_bf_{G,H}.tsv`, `e5/logs/*_43936[12]*`;
  - picks: `e6/picks/crm_bfull/f104/{G_full_free,H_full_free}/`, `e6/picks/LOCK_crm_bfull.{json,sha256}`;
  - drive rows: `e6/tasks/soil_eval_bf{1,2}.json` (+ mapping / meta / staging / build and stage logs),
    `e3/tasks/soil_v4.json` (+ meta, check file);
  - drives and analysis: `e6/runs_soil/` (index files of the runs), `e6/index/soil_eval_bfull.json`,
    `e6/analysis/spec_soil_v1_Bfull.{json,sha256}`, `e6/analysis/results_soil_v1_Bfull.{json,txt}`,
    `e6/analysis/soil_extras_Bfull.json`.
