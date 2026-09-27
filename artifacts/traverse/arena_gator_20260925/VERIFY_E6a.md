# VERIFY E6a: evaluation tools (independent verifier; started 05:52 and cut off at about 06:01 by the session shutdown, resumed 12:45-13:40)

**Verdict: PASS**, with one fix that was already in place.

- **The fix.** The first version of `ag_eval_tasks.py` gave soil Gator rows a `--runtime-fingerprint` argument, and the
  frozen soil dispatcher refuses that argument. Every soil Gator drive would have stopped at start-up.
  - The first verifier fixed it at 05:57, before the shutdown; there is no written record of it.
  - I confirmed the fix, and confirmed that it changes nothing else (section 5).
  - I recorded it in `NOTES_E6a.md` sections 1.2 and 7.
- **Checks.** Every check the brief asked for passes: pick determinism, the planner call, straight routes, one shard per
  group, the statistics on known answers, and the offline scorer against the trainer.
- **For the evaluation module.** Section 8 lists four points it must respect.

K3 = this folder, G3 = `/work1/dannegrut/harry/experiments/arena_gator_20260925`. The first verifier's partial work is in
`verify_e6a/{picks,tasks,synthetic,offline}` (05:54-06:01). The resumed work is in `verify_e6a/r2/`.

- **Tool versions.** The current tool hashes equal NOTES_E6a section 7 for all six scripts, except `ag_eval_tasks.py`,
  which is now `6fda2fcc...` (the fix). That was the version in use by 05:58, and it is the version committed in
  801b3a425.
- **Side effects.** No cluster job or step was submitted, and nothing was written to G3 by this module.

## 1. Pick determinism (re-run of 20 groups, three models the builder never used)

Each case was planned in two separate processes (r1, r2) and, inside r1, a second time by `--rerun-check`. The 20
groups are the 20 lowest-md5 groups of the declared soil subset of the arena.

| picks | lock r1 | lock r2 | rerun inside r1 | pick records r1 vs r2 |
|---|---|---|---|---|
| g263, rigid, three-arena model (M3a), speed free | `828b28ff` | `828b28ff` | identical | identical (40 records) |
| g263, rigid, three arenas all data (A3), fixed 2 m/s | `3b8acec2` | `3b8acec2` | identical | identical |
| g241, soil, three arenas all data (A3 soil), speed free | `36463724` | `36463724` | identical | identical |

Record fields exclude wall time.

**Earlier re-runs by the first verifier.**
- f104-only (M1a) picks on the builder's 20 g260 groups gave the builder's locks exactly: free `5c98f545`, fixed 2 m/s
  `80d14eee`, straight 6 m/s `c19a0910`.
- Two concurrent runs of the second f104-only ensemble (M1b) agreed: 12 g271 groups speed free (`ef2d858f`), 12 g241
  groups fixed 2 m/s (`960f9491`).

## 2. The planner is called as REVIEW_R2 says works

**The command.** Every planner-mode manifest and `planner.log` shows
`ci_planner.py --family free [--fixed2] --world <w> --arms B`, with the arena's `--cases` and `--map-root`. It adds a
non-existent `--ref-picks` and `--task-root K3`; neither affects the route.

**Direct call.** I ran `ci_planner.py` directly with REVIEW_R2 amendment 7's exact form (plus `--arena-tag <arena>
--shards 16`) on the same groups and models as section 1. Route files are byte-identical, and pick records identical,
for 60 of 60 groups (rigid free, rigid fixed 2 m/s, soil free). Directory: `r2/direct/`.

**Summaries.**
- Arm B only.
- Tags: `n2iter_cem4x64` for speed free, `n2iter_cem4x64_fixed2` for fixed 2 m/s.
- The `fixed2` flag and the world are right, and the model kind is `ci_train`.
- The start is the standing start for 20 of 20 groups.

**Speeds.** Fixed 2 m/s routes start at 2.0 m/s and never exceed it (interior minimum 1.41: the stop at the goal).
Speed-free routes range from 0.5 to 6.0 m/s.

**Refusals.** Each of these stops before planning, and no output folder is left:
- fixed 2 m/s in the soil world;
- a soil model in the rigid world ("trained with --domain-filter crm");
- a wrong map root for the cases (the BMP sha256 differs from the map's).

## 3. Straight routes equal `ag_dev_headroom.py`'s

**Against the dev-arena tool's own code.** I re-ran `ag_dev_headroom.py` itself on the dev arena g217 (output
`r2/devh_rerun/`).
- It reproduces its E3a lock `73a80be2...` and all 150 stored straight 6 m/s route files.
- Those 150 files are byte-identical to `ag_picks.py --mode straight6` on the same 150 groups (`r2/picks/g217_straight6`).

**Built-in self-check.** `--selfcheck-straight` rebuilt 447 of 447 earlier straight 6 m/s routes byte for byte: dev 150,
spread arenas 74 + 74 + 74 + 75.

**Against the planner module's own straight modes.** I compared `gen_planner.plan(mode='straight2' | 'straight6')` with
`ag_picks.py`'s straight routes.
- 250 of 250 g260 routes at 2 m/s and 250 of 250 g241 routes at 6 m/s have the same route content hash.
- The case's route_00 geometry equals `gen_planner.base_route(pose, goal)` on all 500 groups.

## 4. Rigid rows of a group share a shard

**Test build.** My build (`r2/tasks/rigid_v.json`) has 790 rows over three arenas and 520 groups:
- speed free and fixed 2 m/s on g263;
- a duplicate fixed 2 m/s arm (the r2 picks) to test identical-route merging;
- straight 6 m/s on g241 for both vehicles;
- straight 2 m/s on g260.

It was built with `--existing` set to both rigid task files.

**Results.**
- Every group sits in exactly one shard, including the Gator and HMMWV rows of the same group.
- There are at most 8 groups per shard.
- Shards 3000-3064 are not used by any existing rigid file.
- Every arm's mapping resolves to a row in its group's shard.
- The 20 identical routes are driven once (a row serving both arms).
- There are no id or seed clashes with the 96,200 existing rigid rows.
- Paths are absolute G3 paths.
- `extra` is `--vehicle hmmwv` for HMMWV rows, and for Gator rows the same Gator form as `rigid_v2.json`
  (`--vehicle gator --runtime-fingerprint G3/runtime/gator_runtime_fingerprint.json`).

**The first verifier's 104-row, 44-group build** (6 arms, duplicate and Gator arms included) gives the same result.

**Cluster trees.** Both cluster source trees (`G3/source`, `G3/r2/source`) hold the collector files with the frozen
hashes, and so does the worktree: `ag_vehicle.py` `072716ee`, `ag_gen_collect_ext.py` `11c28916`, `gen_collect_ext.py`
`b9f36a02`, `gen_runner_g.py` `b47c9fd5`, `ag_rigid_runner.py` `8b35e312`, `ag_crm_collect.py` `b52e1fa6`,
`crm_collect.py` `cb6792be`.

## 5. Soil rows and the fix

**Test build.** My build (`r2/tasks/soil_v.json`) has 38 rows:
- the soil model's picks on 20 g241 groups;
- straight 6 m/s for both vehicles, with `--existing soil_v2.json`.

**Results.**
- 22 arm-group pairs point to existing soil_v2 drives:
  - the 20 HMMWV straight 6 m/s spread-headroom drives;
  - one model pick identical to straight 6 m/s;
  - one model pick identical to the frozen model's headroom pick. That reuse is correct by content: same group,
    vehicle, case and route.
- Gator rows are never reused from HMMWV drives.
- Tier -1, relative paths, `extra` `--vehicle hmmwv` / `--vehicle gator`.
- `ag_crm_collect.py --vehicle hmmwv` calls the unchanged `crm_collect.main`; E2 showed that path to be bit-identical.

**The fix.**
- The frozen soil dispatcher raises `ValueError` on `--runtime-fingerprint` (`ag_crm_collect.py` line 93).
- `verify_e6a/tasks/soil_v_before_fix.json` shows the first version's soil Gator rows with that argument.
- The current code adds it only in the rigid world.

**Scope of the fix.** Rebuilt with the current tool, all three of the builder's self-test task files (rigid, soil, task
B rigid) are byte-identical to the originals, which were built by the hand-off version `394eb8bf`; their mappings are
identical too. So the fix changes soil Gator rows only.

## 6. Statistics on synthetic tables with known answers

**(a) Exact answers** (`r2/exact/exact_answers.py`, my own table; every check passed).

The table has 8 unseen arenas x 100 groups in 16 clusters of 50. Every cluster carries the same planted difference, so
every cluster-bootstrap replicate equals the planted value.

| planted test arm (against the same reference) | difference | cluster 90 % interval | one-sided p | Holm-adjusted p | decision |
|---|---|---|---|---|---|
| fixes 5 of 15 reference failures per cluster | -10.0 | [-10, -10] | 1/4001 | 4/4001 | improves |
| 1 fixed + 1 broken per cluster | 0.0 | [0, 0] | 1 | 1 | no meaningful difference |
| 1 broken per cluster (+2 points, the margin itself) | +2.0 | [2, 2] | 1 | 1 | no meaningful difference (margin inclusive) |
| 2 broken per cluster | +4.0 | [4, 4] | 1 | 1 | inconclusive |

**Other checks in the same table:**
- per-arena differences and the random-effects pooled estimate equal the planted value, and the near minus spread
  difference is 0 with interval [0, 0];
- McNemar counts and one-sided p are exact;
- the non-inferiority test is strict: +2.0 fails against a margin of 2, the null passes;
- the two-ensemble average (mean of two arms) gives exactly -5.0;
- 40 held-out groups give "too few groups";
- Holm on three hand-computed sets gives [0.048, 0.048, 0.048, 0.5], [0.004, 0.06, 0.06, 0.06] and an unsorted
  [0.4, 0.052, 0.4, 0.06], with the right rejections.

**(b) Monte-Carlo checks** (the first verifier's `verify_e6a/synthetic/`). The table is regenerated byte-identically
(16,120 rows, 1,240 groups, index `2f661260`). `ag_analyze.py` was run again with the current code, and `check_synthetic.py`
recomputes every read-out with its own code: ALL CHECKS PASS.
- Differences are exact.
- Cluster intervals agree within bootstrap noise (at most 0.1 point).
- Holm (the tool's adjusted p equal Holm applied to its own p), McNemar and per-arena values are exact.
- DerSimonian-Laird is exact.
- The decisions match in all 12 contrasts:
  - a borderline effect, one-sided p 0.03, is not rejected in a family of four;
  - a tight null gives "no meaningful difference";
  - a wide null and a clear harm give "inconclusive";
  - cluster-correlated effects give a cluster interval 3 times wider than the group interval;
  - the two-ensemble average;
  - 40 groups give "too few".

**(c) Coverage** (`verify_e6a/synthetic/coverage.py`, re-run with the same result). On cluster-correlated outcomes (80
clusters, 300 re-draws), the cluster bootstrap's 90 % interval covers the true difference 90.3 % of the time, and a plain
group bootstrap only 77.7 %. This supports the declared cluster bootstrap.

**(d) Builder's self-test.** The builder's own statistics self-test (`ag_analyze.py --synthetic-selftest`) passes.

## 7. Offline scorer against the trainer's own evaluation (`r2/offline/`)

**Method.** `ag_score_offline.py` (TF32 off) was run on rows whose trainer metrics exist. I compared its metrics with the
trainer summary JSON: within-group, within-profile and pooled AUC, pick / random failure, Brier, calibration error, n,
number of groups.

| model, rows | metrics compared | largest difference | logits vs the trainer's file |
|---|---|---|---|
| Gator-trained (G) deploy, Gator f104 val rows (standing, moving, all) | 10 x 3 | 1.1e-8 | ensemble 2.7e-5, members 5.4e-5 |
| g228-only holdout run, g228 val rows | 10 x 3 | 2.3e-8 | ensemble 1.3e-5, members 4.7e-5 |

The first verifier's check (second f104-only ensemble, f104 val) gave ensemble 1.6e-5 and members 4.9e-5.

**Independent scoring.** My own AUC code in VERIFY_E5a section 4 reproduces 68 values of E5a's offline table. That table
was made by `ag_offline_auc_notf32.py`, which uses the same `ci_train.score` path.

## 8. Other checks, and what the evaluation module must respect

- **Clustering.** The nearest-feature clustering is consistent with the case design. My own lookup on g263 / g260 / g241
  (`TerrainMap.features`, start-goal midpoint) agrees with the design feature on 315 of 375 groups (84 %); NOTES_E6a
  reports 87.6 % over all 1,000. If the two frames did not match, agreement would sit near chance (about 10 %).
- **Vehicle block.** The index's vehicle-block test reads the same `vehicle.name` field in rigid and soil Gator
  outcomes.
- **Evaluation-only files.** They carry split `evalonly` and suite groups only.

Points for E6b:
1. **A clearly harmful arm is labelled "inconclusive".** The plan defines three labels only. When the whole 90 %
   interval lies above +2 points, the report must say in words that the test arm is worse.
2. **Rigid pairing needs the whole shard on one node.** Chrono rigid is deterministic per node only. The frozen pool
   driver keeps the completed rows of a shard, and on a stale-claim takeover it drives the rest on the new node, which
   would split a group across nodes. The new eval pool driver should do one of two things:
   - re-drive the whole shard on a takeover or a second attempt that changes node;
   - or record the host per run, and have the index flag groups driven on more than one host.
3. **One build call per world.** All rigid arms and both vehicles of a group must go into one build call: the one-shard
   rule holds within one build. Keep `e6/picks` free of stray folders (NOTES_E6a 5.11).
4. **Stale read-outs.** The 05:35 offline read-out on g260 (partial) is stale. The rigid test drives are complete, so the
   evaluation-only files must be rebuilt before any offline number on the test arenas is reported.

## 9. Files

- `verify_e6a/r2/picks/`, `logs/`: section 1 and 3 pick runs.
- `r2/direct/`: direct planner calls.
- `r2/devh_rerun/`: `ag_dev_headroom.py` re-run.
- `r2/tasks/`: rigid and soil builds.
- `r2/exact/`: exact-answer statistics.
- `r2/synthetic_rerun/`, `r2/coverage_rerun.json`, `r2/builder_selftest/`.
- `r2/offline/`: scorer against the trainer.
- First verifier, 05:52-06:01: `verify_e6a/picks/`, `tasks/` (including `soil_v_before_fix.json`), `synthetic/`,
  `offline/`, `build_*_v.log`, `offline_M1b.log`.
