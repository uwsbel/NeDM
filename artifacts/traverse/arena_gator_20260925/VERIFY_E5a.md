# VERIFY E5a: rigid training for tasks A and B (independent verifier, 2026-09-25 12:45-13:40)

**Verdict: PASS.** The training report and `NOTES_E5a.md` hold up. The subsets, the run arguments, the checkpoints and
the offline scores were all recomputed with my own code, and every number I checked matches. Two wording notes and three
caveats are listed in section 7. No file of the builder was changed, and no cluster job or job step was submitted.

K3 = this folder, G3 = `/work1/dannegrut/harry/experiments/arena_gator_20260925`. My scripts and outputs are in
`verify_e5a/`. The one cluster-side action was a read-only extraction on the login node (`G3/tools/verify_e5a/`, new
files only): the light per-row columns and the sha256 of every E5a data file.

## 1. Subsets recomputed from the built files (`verify_e5a/recompute_subsets.py`, output `recompute_subsets.json`)

**Method.** I took the four per-arena rigid files (f104 HMMWV, g203, g228, f104 Gator) and applied the declared rule with
my own code, importing nothing from `ag_subset.py`:
- within each arena, rank the training-split groups by md5 of the group id and keep the first N;
- keep every val and test row of the arenas in the design;
- for the learning-curve and evaluation files, also keep the dev fold (md5 % 5 == 0).

Then I compared the result with the files on the cluster.

**Result:** 17 of 17 files are identical to my selection, including the row order.

| file | what it is | groups per arena (training) | rows | rows fitted, deploy | rows fitted, holdout (groups) | val rows |
|---|---|---|---|---|---|---|
| M1 | f104 only | f104 1,089 | 93,397 | 84,787 | 67,368 (865) | 4,358 |
| H | HMMWV on the Gator-validated ids | f104 1,089 | 93,397 | 84,787 | 67,368 (865) | 4,358 |
| M2 | two arenas, same total | f104 545 + g203 545 | 101,916 | 84,310 | 66,886 (864) | 8,070 |
| M3 | three arenas, same total | 363 on each | 112,726 | 84,481 | 67,909 (875) | 12,710 |
| A3 | three arenas, all data | 1,089 + 1,083 + 1,063 | 278,735 | 250,490 | 199,167 (2,571) | 12,710 |
| G | Gator drives on f104 | f104 1,089 | 93,551 | 84,922 | 67,489 (865) | 4,359 |
| LC272 / LC545 | f104 learning curve (holdout only) | 272 / 545 + dev fold | 42,641 / 59,606 | - | 16,612 (213) / 33,577 (431) | 4,358 |
| one arena at 545 groups | g203 only / g228 only | 545 | 50,855 / 53,090 | - | 33,309 (433) / 33,976 (436) | 3,712 / 4,640 |
| pairs of arenas at 545 groups | f104+g203 / f104+g228 / g203+g228 | 273 + 272 | 59,819 / 61,722 / 61,803 | - | 33,699 (435) / 34,374 (441) / 34,773 (449) | 8,070 / 8,998 / 8,352 |
| evaluation files | dev fold + val + test of f104, g203, g228, Gator f104 | 224 / 222 / 218 / 224 dev-fold groups | 26,029 / 26,025 / 27,514 / 26,062 | - | - | 4,358 / 3,712 / 4,640 / 4,359 |

**H is the M1 file.**
- The sha256 values I computed on the cluster are equal for both files (`3174999e25828668...`), and my selection gives
  the same 93,397 rows for both.
- The Gator file holds all 24,000 rigid f104 ids (`gator__` prefix, vehicle `gator` on every row). Those are exactly the
  24,000 HMMWV rigid f104 episodes, with the same split for every group.

**Tiers.**
- Every group of each per-arena file has 20 episodes with tiers 0-19, each once (1,200 groups x 4 files).
- The tier of every g203 / g228 / Gator episode equals its task-file tier: 0 missing, 0 mismatched against
  `rigid_hmmwv_v1.json` / `rigid_v2.json`.
- All subsets span tiers 0-19.

**Suite and blacklist scan.**
- I scanned every id, group and episode string of all 21 files: the 4 per-arena files, the 13 subsets and the 4
  evaluation files.
- They were checked against 3,250 suite group ids (8 test arenas x 250, 2 x 150 held-out, 150 dev, the 800-group f104
  suite, the 200 in-distribution f104 groups) and the patterns `*_test_group_*`, `*_heldout_group_*`, `*_dev_group_*`,
  `*pair_group_*`, `*crm_eval_group_*`, `*g1_test_group*`.
- Result: **0 hits in every file.**
- Positive controls (suite ids with and without the `gator__` prefix and the `@rigid` suffix) are caught, and training
  ids are not.

**File hashes.** The sha256 of the four evaluation files on the cluster equal the local copies and NOTES_E5a:
`ed473bc6`, `270f87f4`, `d1f3cc1c`, `279f2c46`.

## 2. Each run's arguments (`verify_e5a/check_runs.py`, output `check_runs.json`)

**What I checked.** I read all 19 runs behind the report: 7 deploy ensembles and 12 holdout-mode runs. For each one I
checked the trainer's own summary (`<tag>.json`: parsed arguments and one record per member), the job-list line and
the run log. The recipe checked on every run:
- GRU, no history input, rigid rows only, geometry context;
- val split for evaluation, batch 256, 30 epochs, 5 seeds;
- trainer default learning rate and weight decay: every member records 0.002 / 1e-4;
- no data fraction or subsampling;
- the right `--ds` file and mode, seed0 0 (or 5 for M1b and M3b);
- `--roundtrip-check` on every deploy run.

**Results.**
- **Recipe:** 0 deviations on any run.
- **Seeds:** members are seeds 0-4 (M1b and M3b: 5-9).
- **Fitted rows:** each run's fitted count equals my recomputed count exactly, for example A3 deploy 250,490, G holdout
  67,489 and the g203+g228 pair 34,773. No soil row was fitted.
- **Round trip:** every deploy member records a checkpoint round-trip difference of exactly 0.0 on the training node.
- **Job lists:** each run's job-list line is exactly the recipe line. The M3 / A3 lines also appear in the timed-out
  lists of 436234 / 436235.
- **Exit codes:** every run's log ends with `exit: 0` (436135: 5 runs, 436234 / 436235: 4 runs, 436352: 10 runs).
  436352's own summary lists all 10 with exit 0 and `all done rc=0`.
- **Vehicle switch:** both lane scripts `unset NEDM_VEHICLE`, and `submissions.tsv` records 436351 / 436352 (and 436234 /
  436235) with their job-list and script hashes.

**Accounting** (`sacct`):

| job | partition | state | elapsed |
|---|---|---|---|
| 436135 | mi3501x | COMPLETED | 1:22:59 |
| 436234 / 436235 | mi3501x | TIMEOUT | 3:50:18 / 3:50:15 |
| 436351 | mi2508x | COMPLETED | 1:09 |
| 436352 | mi2508x | COMPLETED | 1:36:51 (the report's "1 h 37 min") |

**Cost.** The partition weights are 0.0125 (mi3501x) and 0.08 (mi2508x). The ledger bills 10 times the weight per
node-hour, i.e. 0.125 and 0.8. I inferred the factor from `allocation_usage.json`: other allocations there show billing
of about 0.1 per node-hour, which is 10 times the mi2101x weight of 0.01. The builder used the same convention.
- 436234 + 436235 cost 0.96 billed.
- 436351 cost 0.015.
- 436352 cost 1.29.
- Total 2.27, as reported.

## 3. Synced checkpoints

**Hashes.**
- `sha256sum -c` passes in all 8 deploy folders: 7 ensembles plus H's pointer file, 56 files in all.
- Every listed hash equals the sha256 of the same file in the cluster training folders.
- The whole local mirror `e5/train/` (133 files, deploy and holdout) is identical to `G3/e5/train/`.

**Loading and outputs** (`verify_e5a/rescore.py`, output `rescore.json`, local 5090, TF32 off):
- Every deploy checkpoint loads with `ci_train.load_ci_model`.
- On the val rows of each model's own arenas, it reproduces the trainer's stored member logits to at most 6.7e-5 (ensemble
  at most 2.7e-5).
- The stored labels are identical.
- This cross-GPU difference is the expected float noise. The exact (0.0) round trip is the trainer's check on its own
  node (section 2).

## 4. Offline AUC re-scored with my own code

**Method.** My own within-group AUC: route pairs (unsafe, safe) inside each start/goal group, pooled over groups, ties
counted as half. It is applied to the ensembles' logits from `ci_train.score`.

**Deploy ensembles, val groups** (7 models x 4 row sets, standing and moving starts; `rescore.json`):
- All 56 values equal `e5/offline/auc_summary.json` to 4 decimals.
- Examples, standing start:

| model | f104 | g203 | g228 | f104, Gator drives |
|---|---|---|---|---|
| M1a, f104 only | 0.9831 | 0.9482 | 0.9464 | 0.9269 |
| M3a, three arenas, same total | 0.9757 | 0.9693 | 0.9705 | 0.9200 |
| A3, three arenas, all data | 0.9857 | 0.9877 | 0.9804 | 0.9295 |
| G, Gator-trained | 0.8167 | 0.8550 | 0.7777 | 0.9721 |

**Leave-one-arena-out holdout runs, dev + val groups** (`verify_e5a/rescore_loao.py`, output `rescore_loao.json`): all 12
values behind the report's "two arenas do not help the unseen third" reading reproduce to 4 decimals.
- Unseen g228: trained on f104 0.9365, on g203 0.9200, on both 0.9379, on g228 itself 0.9733.
- Unseen g203: trained on f104 0.9493, on g228 0.9428, on both 0.9503, on g203 itself 0.9763.
- Unseen f104: trained on g203 0.9314, on g228 0.9486, on both 0.9520, on f104 itself 0.9738.

**Unsafe rates from a standing start, val routes.**
- f104 HMMWV 35.9 %, Gator on the same routes 11.7 %.
- The standing-start val routes of the two vehicles are the same 1,120 routes.
- G's lowest-risk route is never unsafe on the 56 Gator val groups. For H (= M1a) it is unsafe in 1 of 56 groups (1.8 %).

## 5. The report's derived statements

All of them follow from the verified table:
- the 2-4-point gap between seen and unseen arenas;
- two arenas lie +0.003 / +0.001 / +0.001 above the better single arena, and 2.2 / 2.6 / 3.5 points below the in-arena
  score for f104 / g203 / g228;
- A3 lies +1.0 / +1.9 / +1.0 above M3a on val;
- the ensemble-pair noise is at most 0.009 (M1a against M1b on g228);
- the task B numbers: G 0.972 against the declared 0.95, H 0.927 on the Gator's routes, G 0.78-0.86 on HMMWV drives.

## 6. What I did not re-verify

- The per-arena builds themselves, i.e. the runs to rows step (E4's verifier covered the builder on f104). For g203 /
  g228 / Gator I checked only the outcome side: 24,000 episodes per file, 20 per group, tiers equal to the task files,
  0 rejected in the build records, and the Gator validation record (24,000 / 24,000, native-height check passed on all).
- The 11:00 queue situation on the MI350X / MI300X nodes (not reconstructible now).

## 7. Notes and caveats

1. **Wording in the training report.** "Every training run exited 0" is true for every run behind a model. The first
   attempts of M3a / M3b / A3 deploy and M3 / A3 holdout in the timed-out jobs 436234 / 436235 exited 1 (their data
   files did not exist yet) and were re-run in 436352. The report does describe this; `NOTES_E5a.md` states it
   correctly.
2. **GPU type crosses the primary comparisons.**
   - The f104-only reference (M1a), and H = M1a, were trained on MI350X.
   - The three-arena models (M3a, M3b, A3) and G were trained on MI250X.
   - So every declared contrast (three arenas against f104 only; G against H) compares ensembles from different GPU
     types. The second-ensemble pairs measure training noise within one GPU type only: M1a / M1b on MI350X, M3a / M3b on
     MI250X, at most 0.009 offline.
   - This is disclosed in NOTES_E5a section 6.1. It should also be named next to the closed-loop results.
3. **Gator rigid outcomes.** The Gator's 24,000 rigid drives include 46 rollovers and 3 arena exits (0.2 %). They are
   driving outcomes, validated like any other run. "0 crashed or NaN" refers to software failures.
4. **Scope of the offline scores.** They cover only the val (and dev-fold) groups of the three training arenas and f104
   Gator drives. The eight unseen test arenas are still to be scored from the finished designed test drives
   (`ag_build_evalonly.py` + `ag_score_offline.py`); the E6a partial read-out from 05:35 is stale.
5. **Ledger.** `slurm_balance2.py` read 688.9 at about 13:00 (file time 12:00). It is account-wide, as the report says.

## 8. Files

- `verify_e5a/ve5a_extract_light.py`, `ve5a_run.sh`: the cluster extraction (login node, read only); outputs in
  `G3/tools/verify_e5a/out/` (light columns + `sha256.txt`).
- `verify_e5a/recompute_subsets.py` / `.json`: section 1.
- `verify_e5a/check_runs.py` / `.json`: section 2.
- `verify_e5a/rescore.py` / `.json` and `rescore_loao.py` / `.json`: sections 3-4.
