# NOTES E6a: evaluation tooling for both tasks (2026-09-25, 05:20-06:05)

Module E6a of PLAN.md (sections 2.2-2.3, 3, 7.1-7.4, 7.6-7.7; REVIEW_R2 amendment 7). K3 = this folder,
G3 = `/work1/dannegrut/harry/experiments/arena_gator_20260925`. Six new scripts, no existing script edited, nothing
written to the cluster, no job or job step submitted (queue 40/50 at 06:00, ledger 650.8 unchanged by this module).
The one cluster-side action was reading: the finished rigid f104 ensembles (job 436135) were copied to `e5/train/`,
and the test-group rigid runs that existed at 05:33 were copied to `/tmp/ag_e6a/rigid_runs` for the offline self-test.

## 0. Summary

- **Picks** (`scripts/ag_picks.py`): one arena, one world, one ensemble, one mode (`free`, `fixed2`, `straight6`,
  `straight2`), standing start, map check first, lock file, and an optional second planning run that must come out
  identical. Self-test: rigid M1a on 20 g260 test groups in modes free / fixed2 / straight6. The locks were identical in
  all three executions of each mode (`5c98f545...`, `80d14eee...`, `c19a0910...`). The straight 6 m/s code rebuilt all
  447 earlier straight routes (150 on the dev arena, 297 on the spread arenas) byte for byte. The planner takes about
  0.4 s per group on the 5090.
- **Drive rows** (`scripts/ag_eval_tasks.py`):
  - Rigid and soil rows are built from pick directories. Arms with the same route are driven once, and a mapping file
    records which drive each arm uses. The vehicle is always given on the row. Every rigid row of a group goes into one
    shard. A staging list names every file the cluster needs.
  - `smoke` drives rows locally. `stage` copies the files and checks their hashes on the cluster.
  - Self-test: 60 rigid rows in 3 shards. All 6 local smoke drives (every arm of 2 groups) completed, and so did a Gator
    / HMMWV pair on the f104 suite (the Gator run carries its vehicle record).
  - The soil rows reuse the spread-arena headroom drives: 8 of 8 straight 6 m/s routes map to existing soil_v2 rows.
- **Outcome index and statistics** (`scripts/ag_eval_index.py`, `scripts/ag_analyze.py`):
  - Outcomes per drive: fail, unsafe, unsafe without the backward clause, backward-only, tilt over 30 deg, time.
  - Two terrain-feature clusterings per group.
  - Paired group bootstrap, cluster bootstrap with one-sided p-values, and exact McNemar. Holm over the declared family
    of four, the ±2-point "no meaningful difference" rule, per-arena intervals with a random-effects pooled estimate,
    and near against spread.
  - The effect is also regressed on the distance to the nearest training arena and on the map error.
  - Generalisation gaps (raw, and against the straight arm), dose response, non-inferiority, closed headroom, and the
    time ratio on joint successes.
  - The statistics self-test on planted effects passed every check. The plumbing run on the smoke drives went through;
    its numbers mean nothing.
- **Offline scores** (`scripts/ag_build_evalonly.py`, `scripts/ag_score_offline.py`):
  - Evaluation-only row files (split `evalonly`) are built from the designed-route drives on test groups. The trainer
    refuses them ("no training rows").
  - The scorer reproduces the trainer's stored logits to 2e-5, with TF32 off.
  - First, partial read-out (196 of 250 g260 groups): the f104-only rigid model ranks the routes of an unseen arena
    less well from a standing start. Within-group ranking of unsafe (AUC) is 0.926, against 0.983 on f104 validation
    groups. The second ensemble gives 0.917, and the learning curve is flat (272 / 545 / 865 fitted groups:
    0.907 / 0.928 / 0.920). Preliminary; rebuild when the collection is complete.

## 1. The tools

### 1.1 `scripts/ag_picks.py`

- **Modes.**
  - `free`: `ci_planner.py --family free --world <w> --arms B`, i.e. CEM 4 x 64 from the case pose at rest.
  - `fixed2`: `ci_planner.py --family free --fixed2 --world rigid --arms B` (REVIEW_R2 amendment 7; refused for soil).
  - `straight6` / `straight2`: the offset-0 anchor at 6 or 2 m/s of the night-2 proposal pool. It is built with exactly
    `ag_dev_headroom.py`'s code (`gen_planner.proposal_pool` from route_00 and the layout pose, rng
    md5(group + 'crm_proposal'), `anchor_index(0, v)`) and written with the same file dump.
    `--selfcheck-straight` rebuilds the 150 dev and 297 spread headroom routes and requires identical bytes (passed).
- **Groups.** `all`, `@file`, a json list, or a declared-subset file: `soil_unseen_subset.json` (per arena) or
  `f104_indist_200.json`. `--first N` keeps the N lowest md5(group id), the declared order of every subset here.
- **Checks before planning.**
  - `ag_map_check.py` on the declared groups' cases, and the arena must be the one named.
  - Every checkpoint must be `ci_train` and trained on this world (`domain_filter`), unless `--allow-domain-mismatch`.
- **After planning.**
  - Every pick has its route file, its content hash equals the recorded one, and the route starts within 0.25 m of the
    pose and ends within 0.25 m of the goal.
  - `PICKS_LOCKED.sha256` uses ga_planner's scheme.
- **Rerun check.** `--rerun-check` plans again into a scratch folder and requires identical route files and pick
  records.
- **Manifest.** `ag_picks.json` holds the checkpoints with sha256, the map observation hash, the groups and their hash,
  the command, the per-group picks, the lock, the rerun result and the numerics (GPU, TF32 state).

### 1.2 `scripts/ag_eval_tasks.py`

`build --world rigid|crm --arm NAME=<glob of pick dirs>[@hmmwv|@gator] ... --out F [--existing task files]`:

- **Picks and ids.**
  - One pick directory per arena per arm, all of the requested world.
  - Arms whose routes have the same content hash are one drive; the row's `arms` lists them all.
  - Run id: `<g>__<first arm>` for the HMMWV and `gator__<g>__<first arm>` for the Gator. Arm names may not contain
    `__`. No id may look like a training id.
- **Episode seed.** md5(id)[:8]. A seed that collides with an `--existing` row, or with an earlier new row, is salted
  (md5(id#k)) and the row gets `episode_seed_salt`. With about 40k soil rows, a chance collision of 32-bit seeds is
  likely enough to plan for. Seeds are provenance only.
- **Vehicle, always on the row.** HMMWV: `['--vehicle', 'hmmwv']`. Gator: `['--vehicle', 'gator',
  '--runtime-fingerprint', G3/runtime/gator_runtime_fingerprint.json]`. So the collector must be a dispatching wrapper,
  and a job that ran plain `crm_collect.py` / `gen_collect_ext.py` on these rows would stop at the unknown argument
  instead of driving the wrong vehicle.
- **Rigid rows.**
  - Every row of a group, every arm and both vehicles, sits in one shard. Groups are sorted by md5 in blocks of
    `--groups-per-shard` (8); shard = `--shard-base` (3000) + block.
  - Case and route paths are absolute G3 paths, so the rows run from `G3/source` or from `G3/r2/source`.
  - Existing rigid runs are never reused (Chrono rigid is deterministic per node only).
- **Soil rows.**
  - The soil_v2.json row format, with tier -1 by default, `kind: eval`, `arms`, `sha256`, `vehicle`, and paths relative
    to G3.
  - A route already present as a row of an `--existing` file is not re-driven: same group, vehicle and case, and the
    same route content, re-hashed from the file. The mapping points to that row.
  - An id clash with a different drive stops the build. `--no-reuse` re-drives everything.
- **Outputs.**
  - `F` (rows), `F.mapping.json` (group -> arm -> run id, reused-from, case, mode, predicted P).
  - `F.staging.tsv` (local file -> G3 path -> sha256). Files outside K3, e.g. the f104 800-group suite cases, go to
    `G3/ext/<repo path>`.
  - `F.meta.json` (counts, identical-route savings, reuse, coverage, shards, and the collector contract).

`smoke --tasks F --ids ... --out D`: drives rigid rows locally with exactly `gen_runner_g.py`'s collector arguments plus
`--local`, on the local Chrono build, one thread each. For Gator rows the cluster fingerprint path is dropped, because
the local gate is bypassed.

`stage --tasks F [--execute]`: `rsync --ignore-existing` of the staging list (new paths only), then a sha256 check of
every listed file on the cluster. Without `--execute` it is a dry run plus the hash check.

### 1.3 `scripts/ag_eval_index.py`

`--mapping <F.mapping.json> ... --runs <run folders> --out index.json` gives one row per (world, vehicle, group, arm).

**Labels.** They come from `ga_analyze.safe_labels` (= `f104_n2_analyze.labels`, after the 1 s settle):
- `fail`: goal not reached.
- `unsafe`: not (goal reached, and < 0.05 s rolling backwards under throttle, and minimum forward speed > -0.30 m/s).
- `unsafe_noback`: `fail` itself, the NOTES_E3b1 definition, kept as its own column.
- `backward_only`: unsafe but not failed.
- `tilt30`, `elapsed`, and status.

**Per group.**
- The evaluation stratum.
- `cluster`: the terrain feature nearest to the start-goal midpoint, from `TerrainMap.features` in the Chrono frame (the
  `n2_cluster_ci.py` rule).
- `cluster_design`: the case's design feature (route_00 `feature_index`, the rule behind the subsets'
  `groups_per_feature`). The two agree on 876 of the 1,000 unseen soil groups and on 176 of the 200 f104
  in-distribution groups. The index order is the same (checked: identical positions).
- The arena role (near / spread / training / dev) and the group set (unseen / indist_f104 / f104_suite / heldout / dev).
- The map-lookup error and the distance to the nearest training arena.
- A Gator row must carry the Gator vehicle record, and an HMMWV row must not (asserted).
- Missing drives are kept as `missing: true`.

### 1.4 `scripts/ag_analyze.py`

`--index ... --spec spec.json --out results.json`. The report is also written as `results.txt`. The declared default
spec is written by `--write-template` (`e6a/spec_template.json`).

**Arms and vehicles.**
- Composite arms (`combine`) average the second ensembles per group (0, 0.5 or 1; REVIEW_R1 4).
- Every contrast names a world and a vehicle. Vehicle `any` pools both vehicles, e.g. task B's H on the HMMWV against H
  on the Gator; arm names are unique per world.

**Per contrast** (diff = rate(test) - rate(ref) in points; negative means the test arm fails less):
- **Paired group bootstrap:** 95 % and 90 % intervals and a one-sided p = (1 + #{bootstrap diff >= 0}) / (B + 1).
- **Exact McNemar:** two-sided (`ga_analyze.mcnemar`) and one-sided.
- **Cluster bootstrap** over (arena, feature) clusters: the primary clustering plus the other one as a robustness line,
  with cluster wins and losses.
- **Per arena:** diff and 95 % interval, arena sign counts, a DerSimonian-Laird pooled estimate over arenas with tau
  and I², near against spread (with the difference of the two), and the per-arena effect against distance and map
  error (slope, Pearson, Spearman; 8 points, descriptive).
- **Non-inferiority:** the one-sided upper 95 % bound is below the margin.
- **Identical picks** and the median time ratio on joint successes (`ga_analyze.time_ratio`).

**Decision rule (PLAN 7.3).**
- Holm at 0.05 over the family's one-sided **cluster** p-values: "improves" if Holm rejects.
- Otherwise "no meaningful difference" if the 90 % cluster interval lies within ±2 points, else "inconclusive".
- Fewer than 50 paired groups: "too few groups".
- Contrasts outside the family are labelled "secondary, unadjusted"; task B's primary is "declared primary, outside the
  Holm family".

**Other blocks.**
- `gaps`: rate(unseen) - rate(in distribution) with independent bootstraps; the difference in differences against the
  straight arm, pooled and per arena; and the per-arena share of the straight route's failures that the model removes.
- `dose`: rates, consecutive paired steps, and the bootstrap slope in points per added arena.
- `headroom` (task B criterion 4).
- `time_ratio`, and rates per world, vehicle, set and arm.

**The declared default spec** (`e6a/spec_template.json`; arm names are placeholders for E6b):
- **Family:** soil M3 vs M1 and A3 vs M1 on fail; rigid fixed-2 M3 vs M1 and A3 vs M1 on unsafe. All four are on the 8
  unseen arenas, with M1 / M3 as the averages of the a/b ensembles.
- **Secondary:** M2 vs M1, the seed floor M1a vs M1b, rigid speed-free no-harm, f104 no-harm, and the in-arena effect on
  the g203/g228 held-out groups.
- **Task B:** G vs H on the Gator (declared primary), G vs straight 6 m/s on the Gator, H on the Gator vs H on the
  HMMWV, and the rigid G vs H.
- Gaps against straight 6 (soil) and straight 2 (rigid), dose M1 -> M2 -> M3, and headroom for G and H.

### 1.5 `scripts/ag_build_evalonly.py`

This builds trainer-format rows for suite drives, for offline scoring only.
- **Where and when it will write.** It requires `--allow-suite-ids` and an output inside `K3/e4/evalonly/` (or
  `G3/e4/evalonly/` on the cluster); anything else is refused (tested).
- **Selection.** Ids matching `<arena>_(test|heldout|dev)_group_NNNN_route_NN`, optionally only the `--tasks` rows of
  kind `test_designed` for this arena. Runs without a completion marker are left out and counted.
- **Pipeline.** It reuses `ag_build_ds.py` unchanged: its per-run checks, the map/arena check (a wrong map is refused:
  tested with the g260 map on g268 runs), `f104_n2_dataset.py`, `n2_reanchor_dataset.py`, and `ci_one_world`.
  `ga_build_mixed.blacklisted` is switched off in this process only.
- **Guards against training.**
  - Every row's split is set to `evalonly`, and the case split is kept as `case_split`.
  - Every group must be a suite group (`ag_blacklist.is_suite`).
  - A marker file `EVAL_ONLY_NOT_FOR_TRAINING.txt` is written next to the output.
  - `ci_train.py` on such a file stops with "no training rows" (tested).
- **Reproducibility.** A rebuild from the same runs is byte-identical (g268 file `a9a59dc6...` twice).
- **On the cluster.** It can also run on the login node from a copy in `G3/tools`, with
  `PYTHONPATH=$G3/tools:$G3/source/scripts:$G3/source/src`: the dataset tools and the map check then come from
  `G3/source`.

### 1.6 `scripts/ag_score_offline.py`

- **Scoring.** Every checkpoint is loaded with `ci_train.load_ci_model` and scored with `ci_train.score`: the planner's
  own path, i.e. the raw corridor, geometry context columns 17-21, and the history window only for history models. The
  ensemble score is the mean of the member logits.
- **Rows.** Every row, or `--splits`, or `--startup-only`.
- **Metrics per file and per arena**, on startup / moving-start / all rows, with `ga_train.regime_metrics`:
  - within-group AUC of unsafe and fail (`W_*`), the same within speed-profile cells (`G_*`), and pooled AUC;
  - the failure of the lowest-risk route per group against random and oracle;
  - Brier and ECE.
- **Checks and extras.** `--check-logits` compares with a trainer `_logits.npz`; `--save-logits` keeps the per-row
  logits.
- **TF32 is off by default.** With the 5090's default TF32 convolutions the member logits differ from the trainer's by up
  to 0.015. Without TF32 they differ by 4.6e-5 (ensemble 1.8e-5, on the 4,358 f104 validation rows).

## 2. Self-test results (`e6a/selftest/`, script `e6a/selftest/run_selftest.sh`, about 2 min)

| step | result |
|---|---|
| picks, g260, 20 groups (the 20 lowest-md5 groups of g260's declared soil subset) | free: mean speed 3.61 m/s, predicted P 4e-6; fixed2: 1.99 m/s, P 1.3e-4; straight6: 5.54 m/s. Every mode was planned twice with identical files, and the locks were equal across three separate executions. |
| straight-route self-check | 447/447 earlier dev and spread straight 6 m/s routes byte-identical |
| rigid rows | 60 rows (no identical picks across arms here), shards 3000-3002 (24 / 24 / 12 rows), 80 staged files: 20 case files already on G3 with equal hashes, 60 route files not staged (a self-test, not meant for the cluster) |
| local smoke (every arm of g260_test_group_0064 and _0088) | 6/6 complete, all goal reached (M1a free 22.6 / 18.0 s, fixed2 21.8 / 17.3 s, straight6 8.1 / 6.4 s); about 45 s wall each |
| soil rows (8 g258 groups: straight6 + an M1a soil-world plumbing arm) | 8 new rows, 8/8 straight6 arms mapped to the soil_v2 headroom rows `<g>__straight6`, no id or seed clash with soil_v2's 31,901 rows |
| task B rows (4 f104 in-distribution groups, straight6 for both vehicles) | both vehicles of a group in one shard; Gator rows carry `--vehicle gator --runtime-fingerprint ...`; local pair `gator__f104_pair_group_0133__S6_gator` / `f104_pair_group_0133__S6_hmmwv` both goal reached (6.05 / 6.40 s), Gator vehicle record present, HMMWV none |
| index | 60 + 8 rows, 8 driven, 60 missing (not driven); sets and clusters filled |
| analysis (plumbing) | every block runs; with 2 groups every decision is "too few groups", as it should be |
| synthetic statistics check (8 arenas x 125 groups + 200 in-distribution; planted: soil M3 fixes 45 % of M1's failures, soil A3 = M1, rigid M3 = M1, rigid A3 fixes 60 %) | P1 improves (-4.5 points, cluster 90 % [-5.9, -3.2], Holm-adjusted p 0.002), P2 and P3 "no meaningful difference" (exactly 0, p = 1), P4 improves (-10.4); Holm and McNemar match hand-computed values; the composite arm is the per-group mean; the planted gap is found (+5.8 points, 95 % [+2.2, +9.2]); dose response monotone with a negative slope; 90 % cluster-interval coverage of a noisy null 0.90 over 40 re-draws |
| offline consistency | f104 validation rows: ensemble 1.8e-5, members 4.6e-5 max abs logit difference from the trainer's file |

## 3. First offline read-out (partial, preliminary)

The evaluation-only file `e4/evalonly/g260_rigid_test_partial0535/` was built from 2,268 of the 3,000 g260 designed test
drives that existed at 05:33: 196 groups, 8,579 rows (sha256 `48b0f9e3...`). Another 129 runs were still in progress and
were left out. The g268 file holds 4 groups (48 drives).

Rigid, standing start (the startup rows):

| model | g260 unseen: within-group AUC unsafe / fail | g260: lowest-risk route fails | f104 validation groups (56): AUC unsafe / fail |
|---|---|---|---|
| M1a (deploy, 1,089 groups) | **0.926** / 0.860 | 1.0 % (random 7.0 %, best possible 0 %) | **0.983** / 0.906 |
| M1b (second ensemble) | 0.917 / 0.859 | 1.0 % | |
| learning curve, holdout mode (fitted 213 / 431 / 865 groups) | 0.907 / 0.928 / 0.920 | 1.0 / 0.5 / 1.5 % | |

Reading, with caveats (partial arena, one near arena, offline ranking only):
- The f104-only rigid model ranks an unseen arena's designed routes worse than f104's own held-out routes.
- More f104 data does not close that gap beyond about 431 fitted groups.
- This is the kind of gap the task A question is about. It is not yet a closed-loop result: the rigid designed routes
  fail only 7 % of the time on g260, against 19 % on f104.

## 4. Exact commands

Local environment (every command from the repo root):

```bash
cd /home/harry/NeDM-traverse_mppi
export PYTHONPATH=src:scripts OMP_NUM_THREADS=6; unset NEDM_VEHICLE
PY=/home/harry/miniconda3/envs/nedm/bin/python
K3=artifacts/traverse/arena_gator_20260925
```

Self-test (all of section 2): `bash $K3/e6a/selftest/run_selftest.sh`.

Production recipe for E6b (arm names are examples; the arm name is what the analysis spec uses):

```bash
# picks: one call per (arena, world, model, mode); several in parallel on the 5090 are fine
for a in g260 g271 g251 g247 g258 g268 g263 g241; do
  $PY scripts/ag_picks.py --arena $a --world rigid --mode fixed2 --cases $K3/cases/test_$a/cases --groups all \
     --models "$K3/e5/train/M1_rigid/M1a_rigid_deploy_s*.pt" --model-tag M1a --out $K3/e6/picks/rigid/$a/M1a_fixed2 --rerun-check
  $PY scripts/ag_picks.py --arena $a --world rigid --mode straight2 --cases $K3/cases/test_$a/cases --groups all --out $K3/e6/picks/rigid/$a/straight2
done
# soil: --world crm --mode free --groups $K3/suites/soil_unseen_subset.json (125 per arena); in distribution:
#   --arena f104 --cases artifacts/traverse/generalist_20260921/A_adapt/suite/cases --groups $K3/suites/f104_indist_200.json
#   (map root default crm_f104_v1/map_root); held-out: --arena g203 --cases $K3/cases/heldout_g203/cases; task B: the
#   same f104 suite with --groups all (800), G and H models, straight6.
# rows (all arms of a world in ONE build, so every rigid arm of a group shares a shard)
$PY scripts/ag_eval_tasks.py build --world rigid --arm M1a_fx2="$K3/e6/picks/rigid/*/M1a_fixed2" --arm straight2="$K3/e6/picks/rigid/*/straight2" \
   [--arm ...] --existing $K3/e3/tasks/rigid_v2.json --out $K3/e6/tasks/rigid_eval_v1.json [--shard-base 0]
$PY scripts/ag_eval_tasks.py build --world crm --arm M1_free="$K3/e6/picks/crm/*/M1a_free" --arm straight6="$K3/e6/picks/crm/*/straight6" \
   [--arm G_free="$K3/e6/picks/crm/f104/G_free@gator" ...] --existing $K3/e3/tasks/soil_v2.json --out $K3/e6/tasks/soil_eval_v1.json
$PY scripts/ag_eval_tasks.py smoke --tasks $K3/e6/tasks/rigid_eval_v1.json --ids <a few ids> --out /tmp/ag_smoke   # optional
$PY scripts/ag_eval_tasks.py stage --tasks $K3/e6/tasks/rigid_eval_v1.json            # dry run + remote hash check
$PY scripts/ag_eval_tasks.py stage --tasks $K3/e6/tasks/rigid_eval_v1.json --execute  # copy (new paths only) + check
# after the drives: sync the run folders, then
$PY scripts/ag_eval_index.py --mapping $K3/e6/tasks/rigid_eval_v1.json.mapping.json $K3/e6/tasks/soil_eval_v1.json.mapping.json \
   --runs /tmp/ag_eval/rigid_eval_runs /tmp/ag_eval/soil_runs --out $K3/e6/index/eval_v1.json
$PY scripts/ag_analyze.py --write-template $K3/e6/spec_v1.json   # then set the arm names, and freeze the spec (sha256) before looking
$PY scripts/ag_analyze.py --index $K3/e6/index/eval_v1.json --spec $K3/e6/spec_v1.json --out $K3/e6/analysis/results_v1.json
# offline: evaluation-only files once the rigid test drives are complete (near: 436075 shards 12-13; spread: pool shards 2000-2124)
rsync -a --include='g*_test_group_*/***' --exclude='*' amd:/work1/dannegrut/harry/experiments/arena_gator_20260925/rigid_v1/runs/ /tmp/ag_eval/rigid_test_runs/
for a in g260 g271 g251 g247; do $PY scripts/ag_build_evalonly.py --arena $a --world rigid --runs /tmp/ag_eval/rigid_test_runs \
   --tasks $K3/e3/tasks/rigid_hmmwv_v1.json --map-root $K3/map_roots/$a --out $K3/e4/evalonly/${a}_rigid_test --allow-suite-ids; done
for a in g258 g268 g263 g241; do $PY scripts/ag_build_evalonly.py --arena $a --world rigid --runs /tmp/ag_eval/rigid_test_runs \
   --tasks $K3/e3/tasks/rigid_v2.json --map-root $K3/map_roots/$a --out $K3/e4/evalonly/${a}_rigid_test --allow-suite-ids; done
$PY scripts/ag_score_offline.py --models "$K3/e5/train/M1_rigid/M1a_rigid_deploy_s*.pt" --tag M1a \
   --ds $K3/e4/evalonly/g*_rigid_test/ci_*_evalonly.npz --out $K3/e6/offline/M1a_rigid_test.json
```

## 5. Decisions, caveats and what E6b must know

1. **Running the rigid rows.** The rows are shaped like `gen_runner_g.py` rows.
   - **Pool steps.** The frozen `ag_rigid_pool.py` knows only shards 1000-1999 (Gator pool) and 2000-2999 (spread HMMWV).
     Eval shards (3000 and up) need a new pool driver (a new file) with one class:
     - `GEN_ROOT = G3/r2` and `GEN_COLLECTOR = G3/r2/source/scripts/ag_gen_collect_ext.py`;
     - `FDM_RUNTIME_FINGERPRINT` = the f104 HMMWV fingerprint;
     - Gator rows override it with their own argument.
   - **Array submission.** Alternatively build with `--shard-base 0` and submit an array with the same variables (array
     indices must stay under 1,001; mind the queue cap).
   - **Collector files.** In both trees they are identical to the frozen hashes (`ag_vehicle.py` `072716ee`,
     `ag_gen_collect_ext.py` `11c28916`, `gen_collect_ext.py` `b9f36a02`, `gen_runner_g.py` `b47c9fd5`,
     `ag_rigid_runner.py` `8b35e312`). The r2 tree is needed for the spread arenas.
   - **Output folder.** Use a new rigid output folder, e.g. `G3/rigid_eval`.
2. **Soil rows go into a soil_v3 superset** of soil_v2, as in `e3/README.md` section 0: a new path, the same output
   folder `soil_v1`, and the frozen dispatcher `ag_crm_collect.py`. The explicit `--vehicle hmmwv` makes the dispatcher
   run the unchanged `crm_collect.main`, which E2 showed is bit-identical. A job using plain `crm_collect.py` would
   refuse these rows.
3. **Reuse of the headroom drives (the E1b verifier's open point): decided yes, by default.**
   - The straight 6 m/s arm on the spread arenas (and on the dev arena g217) maps to the existing `<g>__straight6`
     rows: 297 + 150 drives are not repeated. Soil drives of tonight's tree are identical across nodes and GPU types
     (bitid 3/3, drift re-runs).
   - `--no-reuse` re-drives them under new ids.
   - Rigid never reuses.
4. **TF32.** The planner runs with torch's defaults, i.e. TF32 convolutions on the 5090. That is how every earlier local
   pick of this study was made, and it is recorded in each manifest. Picks are deterministic on this machine. The
   offline scorer switches TF32 off, to reproduce the trainer's numbers.
5. **"Unsafe without the backward-motion clause" equals "fail"** by construction (the NOTES_E3b1 definition). The
   index keeps it as a column; `backward_only` is the part that only the backward clauses add.
6. **Two clusterings.**
   - The primary is the nearest feature to the start-goal midpoint (the earlier studies' `n2_cluster_ci.py` rule, via
     `TerrainMap.features`).
   - The design feature of the case is reported as `cluster_alt`. They differ on 12 % of the groups.
   - Spec `cluster_key` switches them. Keep the default unless the plan owner decides otherwise before the drives.
7. **The p-value** is a percentile-bootstrap p, (1 + #{bootstrap diff >= 0}) / (B + 1), so its floor is
   1 / (B + 1) (2.5e-4 at B = 4000). The Holm-adjusted floor with four tests is 1e-3.
8. **The in-distribution caveat (VERIFY_E4 note 7) still holds.** 37 of the 200 f104 in-distribution groups start and
   end within 2 m of some training group.
9. **The evaluation-only files are partial now.** They were built from the runs that had a completion marker at 05:33.
   Rebuild them with the commands in section 4 when shards 12-13 (near) and 2000-2124 (spread) are done.
   `/tmp/ag_e6a/rigid_runs` is a scratch copy.
10. **Held-out and dev drives.** No designed-route drives exist yet on the held-out or dev groups (none were planned). The
    builder accepts them (`heldout` / `dev` id patterns) when they do.
11. **A self-test glob caught a real pitfall.** A glob such as `picks/rigid/*/straight6` takes every arena that has such
    a folder, e.g. an f104 in-distribution folder next to the test arenas. That is intended for production, but it means
    stray pick folders get driven. Keep only real pick folders under `e6/picks`. The `.meta.json`
    `groups_by_arena` / `arm_coverage_groups` counts show what went in.

## 6. Files

- **New scripts:** `scripts/ag_picks.py`, `scripts/ag_eval_tasks.py`, `scripts/ag_eval_index.py`, `scripts/ag_analyze.py`,
  `scripts/ag_build_evalonly.py`, `scripts/ag_score_offline.py`. Their sha256 at hand-off are in section 7.
- **Self-test** (`e6a/selftest/`):
  - `run_selftest.sh` and `logs/`;
  - `picks/` (rigid g260 x 3 modes, rigid f104 straight6, soil g258 straight6 + plumbing arm);
  - `tasks/` (rows, mapping, staging, meta); `smoke_rigid/`, `smoke_taskB/` (local runs);
  - `index/`; `analysis/` (specs, results, reports); `analysis_synthetic/` (the statistics check); `offline/` (scores).
- **Declared spec template:** `e6a/spec_template.json`.
- **Evaluation-only files:** `e4/evalonly/g260_rigid_test_partial0535/`, `e4/evalonly/g268_rigid_test_partial0535/`.
- **Fetched from the cluster (read only):** `e5/train/{M1_rigid,offline_rigid,smoke}/` (job 436135 outputs).

## 7. Tool hashes at hand-off

The self-test outputs above were produced by exactly these files: the final `run_selftest.sh` pass, checked against the
hashes recorded in each output.

| script | sha256 (first 16) |
|---|---|
| `scripts/ag_picks.py` | `35aad37ce38d04bb` |
| `scripts/ag_eval_tasks.py` | `394eb8bf08434ac9` |
| `scripts/ag_eval_index.py` | `e2e00cbd3727e03b` |
| `scripts/ag_analyze.py` | `42561c33694fc8f6` |
| `scripts/ag_build_evalonly.py` | `a61d742626dcf08d` |
| `scripts/ag_score_offline.py` | `8bdaf224d8d2e43a` |

The six g260 smoke drives ran at 05:27, from a task file that is byte-identical to the final one (`8b5b36e9...`). The
`smoke` code has not changed since, so the final pass found them complete and did not re-drive them.
