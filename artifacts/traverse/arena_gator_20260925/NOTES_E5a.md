# NOTES E5a: rigid training for tasks A and B (2026-09-25, 05:25-05:50, resumed 10:58-12:50)

Module E5a of PLAN.md (sections 2.1, 3, 7.5, 7.6, 7.7). K3 = this folder, G3 = `/work1/dannegrut/harry/experiments/arena_gator_20260925`.
The first E5a agent was cut off by the session shutdown at 05:50 before it wrote notes. Section 1 reconstructs its work from
LOG.md and the cluster files. Section 2 onwards is the resumed work (10:58-12:50), written as the "resumed" record.

New scripts:
- first agent: `scripts/ag_offline_auc.py`, `scripts/ag_train_lanes.sbatch`, `scripts/ag_e5a_ids.py`;
- resumed: `scripts/ag_train_lanes_multi.sbatch`, `scripts/ag_deploy_sync.sh`, `scripts/ag_offline_auc_notf32.py`,
  `scripts/ag_auc_table.py`.

No existing script was edited. Nothing in `G3/source` was touched; new cluster files are only in `G3/tools/e5a`, `G3/e4`,
`G3/e5`.

## 0. Summary

**Every rigid model of tasks A and B is trained.** Each is a 5-seed ensemble with the same recipe as the f104 models:
- GRU network, no history input, rigid rows only, geometry context;
- 30 epochs, batch 256, trainer default learning rate and weight decay;
- standing-start plus re-anchored rows.

| model | what it is | training groups per arena | rows in the file | rows fitted | job, GPU |
|---|---|---|---|---|---|
| M1a | f104 only (today's recipe) | f104 1,089 | 93,397 | 84,787 | 436135, MI350X |
| M1b | f104 only, second ensemble (seeds 5-9) | f104 1,089 | 93,397 | 84,787 | 436135, MI350X |
| M2 | two arenas, same total | f104 545 + g203 545 | 101,916 | 84,310 | 436234, MI350X |
| M3a | three arenas, same total | f104 363 + g203 363 + g228 363 | 112,726 | 84,481 | 436352, MI250X |
| M3b | three arenas, same total, second ensemble (seeds 5-9) | same as M3a | 112,726 | 84,481 | 436352, MI250X |
| A3 | three arenas, all their data | f104 1,089 + g203 1,083 + g228 1,063 | 278,735 | 250,490 | 436352, MI250X |
| G | Gator-driven data on f104 (task B) | f104 1,089 (Gator runs) | 93,551 | 84,922 | 436352, MI250X |
| H | HMMWV data on the ids the Gator validated (task B) | identical to M1a (section 3.3) | 93,397 | 84,787 | = M1a |

- **Where they are.** The deploy ensembles are copied to `e5/deploy/<model>/`, each with a `SHA256SUMS` file whose
  hashes equal the cluster copies: M1a, M1b, M2, M3a, M3b, A3, G. `e5/deploy/H/README.md` points to M1a.
- **Offline ranking quality** (section 5): the within-group AUC of "unsafe" (how well the model orders the routes of one
  start/goal pair), on standing-start rows of the validation groups of each training arena. These rows were never fitted
  by any model.

| model | f104 | g203 | g228 | f104, Gator-driven rows |
|---|---|---|---|---|
| M1a, f104 only | 0.983* | 0.948 | 0.946 | 0.927 |
| M1b, f104 only, second ensemble | 0.983* | 0.948 | 0.937 | 0.927 |
| M2, f104 + g203 | 0.976* | 0.977* | 0.942 | 0.923 |
| M3a, three arenas, same total | 0.976* | 0.969* | 0.970* | 0.920 |
| M3b, three arenas, second ensemble | 0.974* | 0.968* | 0.969* | 0.923 |
| A3, three arenas, all data | 0.986* | 0.988* | 0.980* | 0.930 |
| G, Gator-trained | 0.817 | 0.855 | 0.778 | 0.972* |
| H (= M1a), HMMWV-trained | 0.983* | 0.948 | 0.946 | 0.927 |

Validation groups: f104 56, g203 48, g228 60, Gator f104 56. * = the model was trained on other groups of this arena with
this vehicle.

**Readings.** These are offline scores; the closed-loop evaluation (E6) decides.
1. **Arenas in the training set are ranked about 2-4 points better than arenas outside it.**
   - The f104-only model scores 0.983 on f104 but 0.946-0.948 on g203 and g228.
   - The three-arena model at the same total data scores 0.969-0.976 on all three arenas. It pays for this with 0.7
     points on f104, where it has a third of the f104 data.
   - With all the data of the three arenas (A3) the scores rise to 0.980-0.988, f104 included (0.986).
2. **Two arenas do not rank an unseen third arena better than the better single arena.** This is the leave-one-arena-out
   test at about 545 groups (section 5.2; holdout mode; 270-280 scored groups):

   | unseen arena | trained on arena 1 | trained on arena 2 | trained on both |
   |---|---|---|---|
   | g228 | f104: 0.937 | g203: 0.920 | 0.938 |
   | g203 | f104: 0.949 | g228: 0.943 | 0.950 |
   | f104 | g203: 0.931 | g228: 0.949 | 0.952 |

   The unseen-arena scores stay 2.2-3.5 points below the in-arena score (0.973-0.976).
3. **More f104 data barely moves the unseen arenas.** On g203 and g228, the f104 learning curve (213 / 431 / 865 fitted
   groups) is 0.942 / 0.949 / 0.951 and 0.924 / 0.937 / 0.939 (dev + val rows).
4. **Task B, offline.** On the Gator's own f104 validation routes, G ranks at 0.972 (0.969 in holdout mode on 280
   groups).
   - That meets the declared offline criterion (>= 0.95).
   - The HMMWV-trained H ranks the same Gator routes at 0.927.
   - Each vehicle's model ranks the other vehicle's outcomes poorly: G scores 0.78-0.86 on HMMWV rows.
   - The Gator's rigid labels differ a lot from the HMMWV's. From a standing start 11.7 % of the Gator's validation
     routes are unsafe, against 35.9 % of the same routes driven by the HMMWV.

**Cost.** E5a's jobs came to about 2.3 billed node-hours:
- 436234 and 436235: 3.84 h each on mi3501x, about 0.96 billed. About 0.86 of that was spent waiting for data that the
  cut-off agent never wrote.
- 436351: about 0.02.
- 436352: 1 mi2508x node for 1 h 37 min, about 1.3.

The account ledger read 682.4 at 11:00 and 688.9 at 12:44 (session start 639.0). It is account-wide, so it includes the
soil track's jobs.

## 1. Before the shutdown (first E5a agent, 05:25-05:50; reconstructed)

- **Tool tree `G3/tools/e5a`** (05:25): a copy of the worktree `scripts/` and `src/`.
  - The builders used from it are byte-equal to the local files: `ag_build_ds.py` `5e2eae03`, `ag_subset.py` `3891a006`,
    `ag_tasklib.py` `1363371e`, `ag_e5a_ids.py` `91951703`, `ag_offline_auc.py` `7f653f89`,
    `ag_train_lanes.sbatch` `d233f129`.
  - Test build of g203 tier 0: 1,177 episodes, 4,528 rows (`G3/e4/tests/e5a/g203_t0`).
  - Driver scripts: `G3/tools/e5a/e5a_build.sh <g203|g228|gator>` (one-world rigid build on the login node) and
    `e5a_subsets.sh <g203|g228|gator>` (the subsets of one stage, in parallel). Copies are in `e5/tools_e5a_logs/`.
- **g203 rigid file** (05:42): 24,000 episodes selected, 0 rejected, map check ok, 92,225 rows. Then its subsets: M2,
  LOAO1_g203, LOAO2_f104_g203, and the evaluation file `EV_g203_hmmwv_rigid.npz` (dev fold + val + test of g203).
- **Jobs 436234 / 436235** (05:47, mi3501x, two lanes each; job lists `e5/jobs/rigid_A_v1{a,b}.tsv`).
  - Finished by 06:11: M2 deploy (4.8 min per seed, 0.029 s per step), M2 holdout, LOAO1_g203 holdout, LOAO2_f104_g203
    holdout.
  - The M3 / A3 lines waited the lanes' 2-hour limit for subset files, then failed at 07:47-08:12 on the missing files.
    Their logs end with `FileNotFoundError ... M3_hmmwv_rigid.npz` / `A3_hmmwv_rigid.npz`.
  - The three g228 leave-one-arena-out lines never started. Both jobs hit their 3 h 50 min limit at 09:37.
- **g228 rigid file** (06:04-06:11): built by the first agent (24,000 episodes, 0 rejected, 93,113 rows). Its subset stage
  was never run.

## 2. Resumed (10:58-12:50): what was found and done

1. **Checked state.**
   - Present on G3: the rigid files of g203 and g228 and the subsets of g203.
   - Missing: the subsets involving g228, the Gator f104 rigid file, G and H.
   - All rigid data was complete (436075 done; 425 of 425 pool shards).
2. **Built the missing data on login1** with the first agent's drivers, unchanged:
   - `e5a_subsets.sh g228`, 10:58-11:10: M3, A3, LOAO1_g228, LOAO2_f104_g228, LOAO2_g203_g228, EV_g228;
   - `e5a_build.sh gator`, 10:58-11:08;
   - then `e5a_subsets.sh gator`, 11:08-11:12: validated ids, G, H, EV_f104_gator, and the H-vs-M1 comparison.
3. **Changed where the training ran.** At 11:00 every MI350X node was taken by one other user: 7 running and 11 pending
   4-hour jobs, with a scheduler priority of 91 against our 25 (fairshare). Every MI300X node was taken as well.
   - Two 8-GPU MI250X nodes (mi2508x) were idle.
   - The cluster skill warns that MI210 compute nodes fail on the first matrix product with pytorch 2.10. MI250X is not
     that case: smoke job **436351** (2 lanes on 2 GPUs, 1 epoch, 1 min) exited 0 on both lines, with an exact
     checkpoint round trip. One epoch took 30.9 s against 19.9 s on the MI350X.
   - New job script `scripts/ag_train_lanes_multi.sbatch` = `ag_train_lanes.sbatch` plus two changes: lane k runs on GPU
     k-1 (`HIP_VISIBLE_DEVICES`), and the node's cores are shared between the lanes (16 threads each).
4. **Job 436352** (11:05, mi2508x, 3 h 50 min limit, one lane per GPU). Job list `e5/jobs/rigid_AB_v2_mi250.tsv`
   (sha256 `f7a927d6...`):

   | lane / GPU | runs | finished | per seed |
   |---|---|---|---|
   | 1 | A3 deploy, seeds 0-4 | 12:42 | 17.9 min |
   | 2 | A3 holdout | 12:24 | 14.2 min |
   | 3 | M3a deploy, seeds 0-4 | 11:38 | 6 min |
   | 4 | M3b deploy, seeds 5-9 | 11:38 | 6 min |
   | 5 | M3 holdout, then LOAO2_g203_g228 holdout | 11:32, 11:45 | 4.8 / 2.4 min |
   | 6 | LOAO1_g228 holdout, then LOAO2_f104_g228 holdout | 11:19, 11:31 | 2.4 min |
   | 7 | G deploy | 11:43 | 6 min |
   | 8 | G holdout | 11:37 | 4.8 min |

   - Speed: every run held 0.036-0.037 s per step, about 1.25 times the MI350X two-lane time per step (0.029 s). Each lane waits for its
     data manifest; the G lanes waited 381 s for the Gator subsets.
   - Every run exited 0, and every deploy run's checkpoint round trip was exact (maximum difference 0.0 on 2,048 rows).
   - The longest lane (A3 deploy, about 29,300 steps per seed) needed about 1 h 32 min of the 3 h 50 min limit.
5. **Synced the ensembles** to `e5/deploy/` (section 4), **scored them offline** (section 5), and logged every step in
   LOG.md and `G3/e3/submissions.tsv` (436351 and 436352).

## 3. Data files and subsets (sha256 of every npz: `e4/cluster_records/SHA256SUMS_npz`)

### 3.1 Per-arena rigid files (`ag_build_ds.py --world rigid`, completion marker, launch check, map check, vehicle check)

| file | episodes selected / rejected | rows | groups train / val / test | status mix |
|---|---|---|---|---|
| `G3/e4/g203_hmmwv/ci_g203_hmmwv_rigid.npz` | 24,000 / 0 | 92,225 | 1,083 / 48 / 69 | goal 18,174, blockage 4,718, timeout 892, arena exit 172, rollover 44 |
| `G3/e4/g228_hmmwv/ci_g228_hmmwv_rigid.npz` | 24,000 / 0 | 93,113 | 1,063 / 60 / 77 | goal 19,164, blockage 3,696, timeout 922, arena exit 173, rollover 45 |
| `G3/e4/f104_gator/ci_f104_gator_rigid.npz` | 24,000 / 0 | 93,551 | 1,089 / 56 / 55 | goal 22,420, blockage 1,144, timeout 387, rollover 46, arena exit 3 |
| `G3/e4/f104_hmmwv/ci_f104_hmmwv_both.npz` (E4) | 24,000 / 0 (rigid) | 93,397 rigid | 1,089 / 56 / 55 | goal 19,230, blockage 4,120, timeout 529, arena exit 85, rollover 36 |

Build records: `e4/cluster_records/<arena>_<vehicle>/*_record.json`.

**Gator tiers (VERIFY_E4 note 9.3).** The Gator build used `--tasks-rigid rigid_v2.json`. Its Gator rows carry the
collect_v1 per-group tier of the same id, not the pilot's route index (`scripts/ag_rigid_tasks_v2.py` line 54). The
builder's tier assertion therefore passed on all 24,000.

### 3.2 Training subsets (`ag_subset.py`, lowest md5 of the group id within each arena; val / test groups kept whole)

| file | groups per arena | rows | fitted, deploy | fitted, holdout (groups) | val rows |
|---|---|---|---|---|---|
| M1 (`M1_f104_hmmwv_rigid`) | f104 1,089 | 93,397 | 84,787 | 67,368 (865) | 4,358 |
| M2 (`M2_hmmwv_rigid`) | f104 545 + g203 545 | 101,916 | 84,310 | 66,886 (864) | 8,070 |
| M3 (`M3_hmmwv_rigid`) | 363 on each of f104, g203, g228 | 112,726 | 84,481 | 67,909 (875) | 12,710 |
| A3 (`A3_hmmwv_rigid`) | 1,089 + 1,083 + 1,063 | 278,735 | 250,490 | 199,167 (2,571) | 12,710 |
| LC272 / LC545 (E4, holdout only) | f104 272 / 545 | 42,641 / 59,606 | - | 16,612 (213) / 33,577 (431) | 4,358 |
| LOAO1_g203 / LOAO1_g228 | 545 on one arena | 50,855 / 53,090 | - | 33,309 (433) / 33,976 (436) | 3,712 / 4,640 |
| LOAO2_f104_g203 / _f104_g228 / _g203_g228 | 273 + 272 | 59,819 / 61,722 / 61,803 | - | 33,699 (435) / 34,374 (441) / 34,773 (449) | 8,070 / 8,998 / 8,352 |
| G (`G_f104_gator_rigid`) | f104 1,089 (Gator) | 93,551 | 84,922 | 67,489 (865) | 4,359 |
| H (`H_f104_hmmwv_rigid`) | f104 1,089 (HMMWV, Gator-validated ids) | 93,397 | 84,787 | 67,368 (865) | 4,358 |

- LOAO1_f104 is the LC545 holdout run: the same 545 f104 groups, as the first agent recorded.
- Evaluation files for the offline scores (`--groups <arena>=0 --keep-dev-fold`): dev fold + val + test of one arena and
  vehicle. Local and cluster sha256 are equal:
  - `e5/eval/EV_f104_hmmwv_rigid.npz` `ed473bc6...` (21,777 dev + val rows);
  - `EV_g203_hmmwv_rigid.npz` `270f87f4...` (20,741);
  - `EV_g228_hmmwv_rigid.npz` `d1f3cc1c...` (21,515);
  - `EV_f104_gator_rigid.npz` `279f2c46...` (21,792).
- Matched groups are not matched rows (PLAN 7.9). M1 / M2 / M3 fit 84,787 / 84,310 / 84,481 rows, within 0.6 %.

### 3.3 H is M1a

- **Every id validated.** `scripts/ag_e5a_ids.py` found all 24,000 Gator rigid ids validated
  (`e5/ids/gator_rigid_validated.json`): 0 not validated, and the native-height check passed on all 24,000. Against the
  task B criteria (PLAN 7.7) that is 100 % validated, 0 crashed or NaN runs, and 0 % launch-check failures.
- **Identical file.** The H file built from those ids is byte-identical to the M1 file (both sha256 `3174999e...`;
  `e5/ids/H_vs_M1.json`: same id set, same row order).
- **So H was not trained.** The same file, recipe, seeds 0-4 and GPU type give the M1a ensemble.
  - The H deploy ensemble is `e5/deploy/M1a/`.
  - The holdout-mode twin of H is `e5/train/offline_rigid/M1_rigid_holdout_s*.pt`.

## 4. Deploy ensembles (`e5/deploy/<model>/`: `<tag>_s<k>.pt`, `<tag>.json`, `<tag>_logits.npz`, `SHA256SUMS`)

| model | tag | seeds | member sha256 (first 8) | trainer val W_unsafe, standing start / moving (all arenas of the model pooled) |
|---|---|---|---|---|
| M1a | `M1a_rigid_deploy` | 0-4 | see SHA256SUMS | 0.983 / 0.987 |
| M1b | `M1b_rigid_deploy` | 5-9 | see SHA256SUMS | 0.983 / 0.988 |
| M2 | `M2_rigid_deploy` | 0-4 | `204e5e07 bd2a1c38 9d04ed22 feb25752 5bb80347` | 0.976 / 0.979 |
| M3a | `M3a_rigid_deploy` | 0-4 | `990bce54 036ab007 c3e1ab8d befbb669 e5f759d8` | 0.972 / 0.973 |
| M3b | `M3b_rigid_deploy` | 5-9 | see SHA256SUMS | 0.970 / 0.974 |
| A3 | `A3_rigid_deploy` | 0-4 | see SHA256SUMS | 0.984 / 0.985 |
| G | `G_rigid_deploy` | 0-4 | see SHA256SUMS | 0.972 / 0.975 |
| H | = M1a | 0-4 | `e5/deploy/H/SHA256SUMS` (paths into `../M1a/`) | = M1a |

- **Copy tool.** `scripts/ag_deploy_sync.sh <model> <train subdir> <tag>` copies the files and writes `SHA256SUMS`. It
  stops unless the cluster's sha256 list is identical.
- **Training logits.** They are copied with the checkpoints, because `ag_offline_auc.py` reads the fitted groups from
  them (its guard against scoring rows a model was trained on).
- **All outputs.** Every training output (including the holdout runs) is also mirrored in `e5/train/`, and the run logs
  in `e5/logs/`.

## 5. Offline within-group AUC

### 5.1 Method

- **Tool.** `scripts/ag_offline_auc_notf32.py` = `scripts/ag_offline_auc.py` with TF32 switched off (as
  `ag_score_offline.py` does). It runs on the local 5090 and scores each ensemble on the four evaluation files. The
  ensemble score is the mean of the member logits.
- **Metric.** Within-group AUC of `unsafe`: route pairs inside one start/goal group, the ranking the planner uses.
  Standing-start rows (the protocol of this study) and moving-start rows are reported separately.
- **Row sets.**
  - `val`: val-split groups, never fitted by any model.
  - `dev+val`: adds the dev-fold training groups (md5(group) % 5 == 0). These are scored only where the model fitted
    none of them: holdout-mode runs anywhere, and deploy runs on arenas outside their training set.
  - The guard drops rows of fitted groups by group name. Group names carry no vehicle, so G is never scored on HMMWV runs
    of routes it saw with the Gator, and the other way round.
- **Consistency.** Where the evaluation rows are exactly the trainer's own rows (one-arena models on their own arena),
  the scorer reproduces the trainer's W_unsafe exactly, with difference 0.0 on every such row set.
  - The raw `trainer_check` blocks also pair the Gator evaluation with the HMMWV trainer rows, because both have 1,120
    standing-start rows (56 groups x 20 routes). Those pairs compare different rows and are not a check.
  - `scripts/ag_auc_table.py` therefore lists the check per evaluation file (`trainer_check_max_diff_by_eval`).
- **Files.**
  - JSON per scoring call: `e5/offline/auc_rigid_*.json`, logs in `e5/offline/logs/`.
  - Merged table and summary: `e5/offline/auc_table.md`, `e5/offline/auc_summary.json`, including the members' W_unsafe
    range, W_fail, and the failure of the lowest-risk route.
  - `auc_rigid_prelim_1115.json` is a first pass of the same computation (TF32 off) on 3 evaluation files. Its numbers
    equal the final ones.

### 5.2 Table (standing-start W_unsafe; val groups: f104 56, g203 48, g228 60, Gator f104 56; dev+val 270-280 groups)

| model | what it is | mode | fitted groups | f104 val | f104 dev+val | g203 val | g203 dev+val | g228 val | g228 dev+val | Gator f104 val | Gator f104 dev+val |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M1a | f104 only | deploy | 1,089 | 0.983 / 0.987 | fitted | 0.948 / 0.957 | 0.953 | 0.946 / 0.953 | 0.943 | 0.927 / 0.929 | fitted |
| M1b | f104 only, second ensemble | deploy | 1,089 | 0.983 / 0.988 | fitted | 0.948 / 0.957 | 0.954 | 0.937 / 0.949 | 0.941 | 0.927 / 0.932 | fitted |
| M2 | f104 + g203, same total | deploy | 1,090 | 0.976 / 0.983 | fitted | 0.977 / 0.975 | fitted | 0.942 / 0.955 | 0.946 | 0.923 / 0.924 | fitted |
| M3a | three arenas, same total | deploy | 1,089 | 0.976 / 0.978 | fitted | 0.969 / 0.969 | fitted | 0.970 / 0.972 | fitted | 0.920 / 0.927 | fitted |
| M3b | three arenas, same total, second ensemble | deploy | 1,089 | 0.974 / 0.978 | fitted | 0.968 / 0.972 | fitted | 0.969 / 0.970 | fitted | 0.923 / 0.928 | fitted |
| A3 | three arenas, all data | deploy | 3,235 | 0.986 / 0.989 | fitted | 0.988 / 0.985 | fitted | 0.980 / 0.982 | fitted | 0.930 / 0.932 | fitted |
| G | Gator-trained, f104 | deploy | 1,089 | 0.817 / 0.819 | fitted | 0.855 / 0.862 | 0.844 | 0.778 / 0.807 | 0.805 | 0.972 / 0.975 | fitted |
| M1_holdout | f104 only | holdout | 865 | 0.981 / 0.985 | 0.983 | 0.946 / 0.952 | 0.951 | 0.936 / 0.947 | 0.939 | 0.923 / 0.928 | 0.913 |
| M2_holdout | f104 + g203, same total | holdout | 864 | 0.975 / 0.980 | 0.976 | 0.971 / 0.972 | 0.979 | 0.935 / 0.949 | 0.942 | 0.922 / 0.925 | 0.906 |
| M3_holdout | three arenas, same total | holdout | 875 | 0.974 / 0.979 | 0.975 | 0.963 / 0.966 | 0.975 | 0.964 / 0.968 | 0.968 | 0.921 / 0.928 | 0.902 |
| A3_holdout | three arenas, all data | holdout | 2,571 | 0.986 / 0.988 | 0.986 | 0.983 / 0.982 | 0.985 | 0.980 / 0.981 | 0.983 | 0.925 / 0.930 | 0.912 |
| LC272_holdout | f104 only, 272 groups | holdout | 213 | 0.967 / 0.971 | 0.963 | 0.931 / 0.935 | 0.942 | 0.929 / 0.941 | 0.924 | 0.911 / 0.917 | 0.888 |
| LC545_holdout | f104 only, 545 groups (the f104 one-arena set) | holdout | 431 | 0.970 / 0.978 | 0.974 | 0.942 / 0.945 | 0.949 | 0.939 / 0.948 | 0.937 | 0.914 / 0.922 | 0.900 |
| LOAO1_g203_holdout | g203 only, 545 groups | holdout | 433 | 0.930 / 0.940 | 0.931 | 0.964 / 0.970 | 0.976 | 0.911 / 0.933 | 0.920 | 0.925 / 0.931 | 0.899 |
| LOAO1_g228_holdout | g228 only, 545 groups | holdout | 436 | 0.947 / 0.952 | 0.949 | 0.938 / 0.947 | 0.943 | 0.968 / 0.972 | 0.973 | 0.918 / 0.925 | 0.894 |
| LOAO2_f104_g203_holdout | f104 + g203, 273 + 272 groups | holdout | 435 | 0.973 / 0.978 | 0.972 | 0.959 / 0.960 | 0.970 | 0.931 / 0.948 | 0.938 | 0.911 / 0.922 | 0.895 |
| LOAO2_f104_g228_holdout | f104 + g228, 273 + 272 groups | holdout | 441 | 0.966 / 0.972 | 0.968 | 0.949 / 0.952 | 0.950 | 0.963 / 0.966 | 0.966 | 0.907 / 0.920 | 0.896 |
| LOAO2_g203_g228_holdout | g203 + g228, 273 + 272 groups | holdout | 449 | 0.954 / 0.956 | 0.952 | 0.959 / 0.963 | 0.971 | 0.958 / 0.961 | 0.962 | 0.913 / 0.922 | 0.892 |
| G_holdout | Gator-trained, f104 | holdout | 865 | 0.838 / 0.834 | 0.836 | 0.875 / 0.881 | 0.857 | 0.826 / 0.837 | 0.836 | 0.969 / 0.974 | 0.969 |

- Val cells give standing start / moving start. Dev+val cells give the standing start only; "fitted" means the model
  trained on some of those dev-fold groups.
- Row counts: val 4,358 / 3,712 / 4,640 / 4,359 rows; dev+val 21,777 / 20,741 / 21,515 / 21,792 rows (f104 / g203 / g228
  / Gator f104).
- The full numbers (moving starts, W_fail, lowest-risk-route failure, member ranges) are in `e5/offline/auc_summary.json`,
  the compact table in `e5/offline/auc_table.md`.

### 5.3 What the table says (offline only)

- **Training-noise floor.** The two f104-only ensembles differ by 0.000 on f104, g203 and the Gator rows, and by 0.009
  on g228. M3a and M3b differ by 0.001-0.003. Single members spread 0.02-0.03 on the unseen arenas. Differences under
  about 0.01 between ensembles are within training noise.
- **In-arena vs unseen-arena gap.** For every one-arena model with at least 431 fitted groups, the arena it was trained
  on scores 0.973-0.983 and the other two 0.920-0.951 (holdout mode, dev + val).
  - With two arenas at the same total, the unseen third scores 0.938 / 0.950 / 0.952 (g228 / g203 / f104). That is the
    better single arena's score (0.937 / 0.949 / 0.949) plus 0.001-0.003.
  - At matched data, variety helps an arena only once it is in the training set.
- **Matched designs M1 -> M2 -> M3 on the three training arenas** (val, deploy):
  - f104: 0.983 -> 0.976 -> 0.976;
  - g203: 0.948 -> 0.977 -> 0.969;
  - g228: 0.946 -> 0.942 -> 0.970.
  - M3 is within 0.008 of the best matched-data model (M1, M2, M3) on every training arena. M1 is 2.1-2.4 points behind
    M3 on the two arenas it did not see.
  - Whether that carries over to the eight never-seen test arenas is E6's closed-loop question. The leave-one-arena-out
    rows above suggest a smaller effect there.
- **A3 against M3** (three arenas with all their data, against the same arenas with a third of it):
  - A3 scores 0.986 / 0.988 / 0.980 on f104 / g203 / g228 (val), against M3a's 0.976 / 0.969 / 0.970, i.e. +1.0 to +1.9
    points. In holdout mode (dev + val, 270-280 groups) it scores 0.986 / 0.985 / 0.983 against 0.975 / 0.975 / 0.968.
  - On f104 A3 matches or slightly exceeds the f104-only M1a (0.986 against 0.983, within the noise floor), so adding the
    other two arenas' data costs f104 nothing.
  - Against M1, A3 adds both arenas and data. Its gain on g203 / g228 (+4.0 / +3.4 points over M1a) is about the sum of
    variety at matched data (M1 -> M3, +2.1 / +2.4) and data (M3 -> A3, +1.9 / +1.0).
- **Task B:** see section 0, reading 4.
  - G's startup W_fail (goal not reached) on the Gator's validation routes is 0.946. Its lowest-risk route per group is
    never unsafe on the 56 validation groups; a random route is unsafe 11.7 % of the time.
  - H on the same routes: W_unsafe 0.927, lowest-risk route unsafe 1.8 % (1 of 56 groups).

## 6. Deviations, caveats

1. **GPU type.**
   - M1a, M1b, M2 and the earlier holdout runs (M1, M2, LC272, LC545, LOAO1_g203, LOAO2_f104_g203) were trained on
     MI350X.
   - M3a, M3b, A3, G and the remaining holdout runs were trained on MI250X (mi2508x), because no MI350X or MI300X node was
     free for hours.
   - Same code, recipe and data; floating-point results differ between GPU types, so the ensembles are not
     bit-reproducible across the two groups.
   - The comparisons are between separately trained ensembles in any case. The M1a / M1b and M3a / M3b pairs (0.000-0.009
     apart) show the size of training noise.
2. **mi2508x billing** is 0.8 billed per node-hour (8 GPUs, whole node), against 0.125 for one MI350X. Running all 10
   runs at once on one node cost about 1.3 billed.
3. **H is M1a** (section 3.3). With 100 % of the Gator ids validated, "H trained on exactly the Gator-validated ids" is
   the M1 file byte for byte.
4. **The trainer's own val numbers for multi-arena models pool the arenas.** The per-arena numbers are in section 5.
5. **Holdout-mode runs fit about 80 % of the nominal groups** (the dev fold is left out), so the leave-one-arena-out
   and learning-curve rows compare about 430-450 and 213 / 431 / 865 fitted groups.
6. **Offline rows exist only for the three training arenas (HMMWV) and for f104 driven by the Gator.** The eight unseen
   test arenas have no training-format rows here; E6a's `ag_build_evalonly.py` + `ag_score_offline.py` cover them.
7. **The first agent's 436234 / 436235 idled about 3.4 h each** (about 0.86 billed together), waiting for subset files that
   the shutdown prevented. The lane script's 2-hour wait was the cause; the new job was submitted only when its data was
   finished or minutes from finishing.

## 7. Exact commands

Cluster, on login1 (numpy environment of the collectors; the drivers are `G3/tools/e5a/e5a_{build,subsets}.sh`):

```bash
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925; T=$G3/tools/e5a
setsid nohup bash $T/e5a_subsets.sh g228 > $T/logs/subsets_g228.out 2>&1 < /dev/null &     # 10:58-11:10
setsid nohup bash $T/e5a_build.sh gator  > $T/logs/build_gator.out  2>&1 < /dev/null &     # 10:58-11:08
setsid nohup bash $T/e5a_subsets.sh gator > $T/logs/subsets_gator.out 2>&1 < /dev/null &   # 11:08-11:12 (after the build)
# training (queue checked first: squeue -u $USER -h -r | wc -l <= 46)
env -u NEDM_VEHICLE sbatch -p mi2508x -c 128 -t 00:20:00 -J ag_train_smoke_multi \
  --export=ALL,AG_JOBS=$G3/e5/jobs/smoke_multi_mi250.tsv $T/scripts/ag_train_lanes_multi.sbatch     # 436351
env -u NEDM_VEHICLE sbatch -p mi2508x -c 128 -t 03:50:00 -J ag_train_rigid_AB_v2 \
  --export=ALL,AG_JOBS=$G3/e5/jobs/rigid_AB_v2_mi250.tsv $T/scripts/ag_train_lanes_multi.sbatch     # 436352
grep end $G3/e5/logs/ag_train_rigid_AB_v2_436352.out; grep -h "ENSEMBLE .* rigid startup" $G3/e5/logs/*_436352.log
```

Local (repo root):

```bash
K3=artifacts/traverse/arena_gator_20260925; export PYTHONPATH=src:scripts; unset NEDM_VEHICLE
PY=/home/harry/miniconda3/envs/nedm/bin/python
bash scripts/ag_deploy_sync.sh M3a M3_rigid M3a_rigid_deploy          # likewise M1a M1b M2 M3b A3 G
E="--eval f104=$K3/e5/eval/EV_f104_hmmwv_rigid.npz --eval g203=$K3/e5/eval/EV_g203_hmmwv_rigid.npz \
   --eval g228=$K3/e5/eval/EV_g228_hmmwv_rigid.npz --eval f104_gator=$K3/e5/eval/EV_f104_gator_rigid.npz"
$PY scripts/ag_offline_auc_notf32.py --model M3a="$K3/e5/deploy/M3a/M3a_rigid_deploy_s*.pt" \
    --model M3b="$K3/e5/deploy/M3b/M3b_rigid_deploy_s*.pt" --model G_holdout="$K3/e5/train/offline_rigid/G_rigid_holdout_s*.pt" \
    $E --check-trainer --out $K3/e5/offline/auc_rigid_M3ab_Gholdout.json   # one call per group of models, see e5/offline/*.json argv
$PY scripts/ag_auc_table.py $K3/e5/offline/auc_rigid_{deploy_M1_M2,M3ab_Gholdout,G,A3,holdout_*}.json \
    --out-md $K3/e5/offline/auc_table.md --out-json $K3/e5/offline/auc_summary.json
```

## 8. For the next modules

- Rigid picks (E6b) use `e5/deploy/<model>/<tag>_s*.pt`:
  - task A: M1a, M1b, M2, M3a, M3b, A3;
  - task B: G, and M1a as H.
- Every checkpoint is a `ci_train` rigid model (`domain_filter` rigid), so `ag_picks.py`'s world check passes.
- The seed-floor contrast is M1a vs M1b and M3a vs M3b.
- Offline scores on the unseen test arenas (evaluation-only files from the designed test drives) are E6's job
  (`ag_build_evalonly.py`, `ag_score_offline.py`). The rigid test drives are complete (436075 shards 12-13, pool shards
  2000-2124).
- No soil model is part of E5a. The soil stage-1 training (tiers 0-6, PLAN 7.10) is the soil track's.
