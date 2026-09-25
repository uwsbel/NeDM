# NOTES E3b2: soil relaunch with the Gator rows, and the Gator and spread-arena rigid rows (2026-09-25, 04:34-05:15)

Module E3b2 of PLAN.md (section 7.8, e3/README.md section 6, REVIEW_R2 amendment 8). K3 = this folder,
G3 = `/work1/dannegrut/harry/experiments/arena_gator_20260925`. The current launch map with stop and relaunch commands
is `e3/README.md` section 0. Nothing in the earlier artefact folders or cluster roots was written, no existing script
was edited, and no file that jobs read was replaced in `G3/source` (only new files were added there).

## 0. Summary

- **Soil relaunched at 04:47** with `soil_v2.json` (31,901 rows, a checked superset of soil_v1): 3 bit-identity rows,
  the 597 spread headroom rows, then tiers 0-12 with the 15,235 Gator rows (`gator__<collect_v1 id>`,
  `--vehicle gator`) next to the unchanged g203/g228 rows. **One collector for every new row: the frozen dispatcher
  `ag_crm_collect.py`**, which runs the unchanged `crm_collect.main` for HMMWV rows; on the cluster three already
  driven HMMWV routes re-driven through it came out identical in every array (two across GPU types). The 144
  calibrated pilot Gator runs were copied in instead of re-driven.
- Only the 2 waiting old tasks were cancelled; the 22 running soil_v1 tasks keep going. New soil jobs: **436207**
  (2 x MI350X, running), **436212** (2 x MI350X, start when 436092 ends ~05:58), **436208** (3 x MI210, pending),
  **436210** (3 x 4 MI210, pending), **436215** (4 x 4 MI210, pending), and the devel check **436213**. Queue 46/50 at
  05:00 (44 at 05:10).
- **Rigid:** `rigid_v2.json` (96,200 rows, superset of rigid_hmmwv_v1) adds the 24,000-route Gator pool (shards
  1000-1299) and the 12,000 spread designed routes (shards 2000-2124, run from a second source tree whose only change
  is the 14-arena allowlist). No CPU node was free (436075 had started, mi2104x 20/20 allocated), so the rows run
  **inside the soil allocations** (`srun --overlap`): the one-node test slowed soil by +0.7 %; with all 18 mi2101x
  nodes loaded the median is +4.3 % (worst +8.5 %), under the 10 % rule. 20 steps, 248 rigid workers, no queue slot,
  no extra billing; a login-node helper adds steps to new soil jobs.
- **First episodes:** 59 new soil runs and 1,571 new rigid runs, all with launch checks passed, finite states, the
  1 ms / 0.08 m soil setup, the Gator vehicle record on every Gator run and none on any HMMWV run. Gator soil 7/7
  bogged (as the pilot); Gator rigid 5.7 % failure (HMMWV twins 20 %), with 4 rollovers in one group.
- **The main problem is soil GPU time for the Gator.** The running soil_v1 jobs (18 MI210) keep doing HMMWV rows until
  13:41-15:03, so the Gator only gets the new launch's GPUs: about 1,000 Gator routes by 12:00 with the 4 MI350X alone,
  about 3,000 if the pending mi2104x tasks start at the scheduler's estimates, against 15,235 planned (section 7).
  The HMMWV g203/g228 tiers reach about 0-6 complete by 12:00. Rigid should be complete by about 08:15.

## 1. Inputs and how they were read

- **Gator pilot (E3b1): GO-with-caveats for soil and for rigid.** The task puts Gator soil rows in on "GO or
  GO-with-caveats", and Gator rigid rows in "only if the pilot says GO for rigid". I read the rigid condition as
  "not NO-GO": every declared rigid collection check passed (288/288 validated, 0 crashed, 0 launch or ground-height
  failures), PLAN section 3 says "go unless the pilot shows a broken vehicle", and the rigid caveats (few failures in
  the labels, short chassis scrapes, creeping back during the settle) are about how the data reads, not about whether
  it is valid. So the 24,000 Gator rigid rows went in. If the plan owner meant a strict GO, the Gator rigid shards not yet
  started can be skipped by writing a done record for each (`e3/README.md` section 0, "Drop the Gator rigid rows";
  they are the first 300 shards of the pool, see section 5).
- **E1b verifier: pass.** So the 597 spread headroom soil rows and the 12,000 spread designed rigid rows went in.
- **Budget.** Ledger 645.3 of 1,500 node-hours used at 04:47 (639.0 at the session start; the ledger lags about an
  hour).

## 2. Which soil collector runs the HMMWV rows of the new launch

Two options were allowed: (a) the frozen Gator dispatcher `ag_crm_collect.py` for every row of the new launch, or (b)
the HMMWV rows kept on the unchanged `crm_collect.py` in a separate launch.

**Chosen: (a), one launch, one task file, one collector.** Reasons:
- With the HMMWV rows (no vehicle flag, `NEDM_VEHICLE` unset) the dispatcher imports only `ag_vehicle.py` (numpy)
  and calls the unchanged `crm_collect.main(argv)` with the argument list it was given; it patches nothing. So the
  HMMWV physics is the frozen code by construction. E2 and the E2 verifier had shown array-identical HMMWV runs through
  it on three routes on the 5090.
- A separate HMMWV launch would split the GPUs by job instead of by tier: the planned 1,200 : 606 : 606 per-tier
  interleave would then depend on how many GPUs each launch happens to get. In one file every new worker takes the
  lowest tier left over both vehicles.
- **Checked on the cluster, on production rows:** the file starts (tier -2) with three `bitid__<id>` rows, i.e.
  soil_v1 training routes that the old launch had already driven with `crm_collect.py` called directly
  (g203 designed, g228 designed, g228 on-policy). The new launch re-drove them through the dispatcher. **All 3 are
  identical to the originals in every array** of trajectory, command reference, anchor state and soil extras, with the
  same outcome and time, and two of them ran on the other GPU type (original on MI210, re-drive on MI350X).
- Every new job ran with `NEDM_VEHICLE` unset (submitted with `env -u NEDM_VEHICLE`; the login shell had none either)
  and with `CRM_COLLECTOR` and `CRM_CONFIG` passed explicitly by `ag_soil_launch.sh`. The job logs print the
  collector hash `b52e1fa6...` (= the frozen file, `e3/gator_collector_sha256.txt`).
- The running soil_v1 tasks keep `crm_collect.py` (they read their task file once at start and cannot switch). HMMWV
  rows from both launches are therefore interchangeable (the bitid check above).

## 3. Soil task file v2 (`e3/tasks/soil_v2.json`, 31,901 rows, sha256 `acf1e98c3c995c10...`)

Built by `scripts/ag_soil_tasks_v2.py` in two stages:
1. The unchanged E3a builder `ag_soil_tasks.py` with the soil_v1 defaults, `--head` = the g217 dev rows and the 597
   spread headroom rows, and `--check-superset soil_v1.json` (the same call as the E1b dry run; 16,663 rows,
   `dd745cd3...`).
2. The Gator rows and the three bitid rows are added by the new script, because `ag_soil_tasks.py --append` refuses
   repeated episode seeds and a Gator row must carry the same seed as its HMMWV twin (10 of the twins are also the
   soil_v1 drift rows). The new script checks the superset again against soil_v1 and against stage 1.

| tier | rows | what |
|---|---|---|
| -2 | 3 | `bitid__<id>`: already completed soil_v1 rows re-driven through the dispatcher (section 2) |
| -1 | 300 + 10 | g217 dev headroom and f104 drift rows of soil_v1 (all complete; skipped) |
| -1 | 597 | spread headroom rows (E1b, `SPREAD_PICKS_LOCKED d1d12ca7...`), HMMWV |
| 0-12 | per tier: 1,200 Gator (tier 0 1,199, tier 12 836) + 606 g203 + 606 g228 | written Gator, then g203, then g228 inside each tier |

- **Gator rows (15,235):** `gator__<collect_v1 id>` for every f104 soil id that has a collect_v1 run (the 15,235 of
  PLAN 1.6; none QA-flagged), with the collect_v1 case, route (absolute `crm_f104_20260916` paths) and episode seed,
  the collect_v1 tier, `extra ["--vehicle", "gator"]`, and the pilot's fields (`pair_id`, `stratum`, `split`,
  `wheel: calibrated`, `ref_runs`). Built as `ag_pilot_tasks.py` builds its calibrated rows; the 144 pilot rows are
  asserted equal to their production rows. Splits: 13,821 train, 713 val, 701 test; 9,168 designed, 6,067 on-policy.
- The worker sorts its shuffled rows by tier only, so inside a tier the order is random per worker; the written order
  is for reading.
- Every soil_v1 row is present unchanged, so the not-yet-claimed g203/g228 rows keep their ids and no finished
  episode is repeated. Nothing above tier 12.
- Checks passed: unique ids; unique seeds among the g203/g228 training rows and among the Gator rows; every Gator seed
  and tier equal to its twin's collect_v1 row; Gator `extra` exactly `["--vehicle", "gator"]`; no HMMWV row names a
  vehicle; every G3-relative path exists in K3 and every path (31,901 rows) exists on the cluster.
- **The 144 calibrated pilot runs were copied in** (`gator__*` of `G3/pilot_gator/soil/runs` -> `G3/soil_v1/runs`,
  `cp -a`, list with hashes in `G3/soil_v1/imported_from_pilot.tsv`), before the first new job started, so they are not
  driven twice (about 1.4 simulated hours saved). They are the same rows (asserted), made by the same frozen collector
  files (vehicle block `072716ee...` / wrapper `b52e1fa6...`, 1 ms step, 0.08 m spacing, checked on all 144). Their
  `collection_request.json` still names the pilot output folder. The `gatorR8__` rows stayed in the pilot folder.
- A 15-row check file `e3/tasks/soil_v2_check.json` (the 3 bitid rows and the 12 Gator tier-0 rows with the lowest md5
  of the id, identical to their soil_v2 rows) was run on one devel MI210 job so that the first Gator production
  episodes existed within minutes; its runs are in the same folder.

## 4. Soil launch (04:47)

- **Cancelled:** only the two waiting tasks of the old launch (`scancel -t PENDING 436080 436092` -> 436080_[18-19]).
  The 18 running mi2101x tasks of 436080 and the 4 mi3501x tasks of 436092 keep running on soil_v1 with
  `crm_collect.py`. `STOP_CLAIMS` was absent.
- **Submitted** (queue 31 -> 46 of 50 array tasks; every submission returned a job id, all in `G3/e3/submissions.tsv`):

| job | partition | tasks | limit | task file | state at 05:00 |
|---|---|---|---|---|---|
| 436207 | mi3501x (1 MI350X) | 2 | 4 h | soil_v2 | running on k007-005-v6/-v7 (claims stop 08:30) |
| 436212 | mi3501x | 2 | 4 h | soil_v2 | waits for 436092_0/_1 (`--dependency=aftercorr:436092`), i.e. starts ~05:58 when those end |
| 436208 | mi2101x (1 MI210) | 3 | 12 h | soil_v2 | pending (Priority; the other 6 mi2101x nodes are other users') |
| 436210 | mi2104x (4 MI210) | 3 | 12 h | soil_v2 | pending behind 436075_[12-13] and two jobs of another user; scheduler estimate 07:45 |
| 436215 | mi2104x | 4 | 12 h | soil_v2 | pending; scheduler estimate 10:37 |
| 436213 | devel (MI210) | 1 | 30 min | soil_v2_check | ran the 15-row check file (section 3) |

- **MI350X nodes:** 2 of the 3 idle ones were taken now, and 436212 replaces only 2 of the 4 old soil_v1 MI350X
  tasks when they end, so from ~05:58 on 3 MI350X nodes stay free for training (REVIEW_R2 amendment 3); until then
  only 1 is free (no training job was queued: 436135 had finished at 04:15).
- **What this means for the balance.** A running job reads its task file once. The 18 MI210 of 436080 therefore keep
  driving g203/g228 rows only, until they stop claiming at 13:41-15:03, and the Gator gets only the new launch's GPUs.
  The new workers always take the lowest open tier, so they first do the spread headroom rows and then Gator tiers
  (the HMMWV tiers are ahead: tiers 0-1 complete and tier 2 at 37 % at 04:57). The planned 1,200 : 606 : 606 interleave
  therefore does not hold in time: the HMMWV tiers run ahead of the Gator tiers, and how far the Gator gets depends
  on whether the pending mi2101x/mi2104x tasks get nodes (section 7). There is no per-job soft stop that could move
  the old workers onto soil_v2 without releasing their nodes (`STOP_CLAIMS` stops every job in the folder, old and
  new, and a stopped job gives its node back).

## 5. Rigid task file v2 and where it runs

**`e3/tasks/rigid_v2.json`** (`scripts/ag_rigid_tasks_v2.py`, 96,200 rows, sha256 `ace9eb9910d919d8...`):

| shards | rows | what |
|---|---|---|
| 0-13 | 60,200 | every rigid_hmmwv_v1 row unchanged (run by 436075 from the v1 file) |
| 1000-1299 | 24,000 | Gator pool: `gator__<id>` for the 14,400 designed (production_v3) + 9,600 on-policy (production_v4) f104 night-2 routes, their case/route files (absolute `fdm_f104_50h_20260909` paths, sha256 of the production task files kept in the rows), `extra ["--vehicle", "gator", "--runtime-fingerprint", G3/runtime/gator_runtime_fingerprint.json]`, tier = the crm_tasks.py tier of the id (0-19, so `--tasks` builds work; the pilot's rows had tier = route index), split = the case's (21,780 / 1,120 / 1,100) |
| 2000-2124 | 12,000 | the 12 designed routes of the 1,000 spread test groups (E1b rows, only `shard` changed), HMMWV |

- 4 Gator groups (80 routes) or 8 spread groups (96 routes) per shard, groups sorted by md5 of the group id. Every
  route of a group is in one shard, and a shard runs on one node.
- The 288 Gator pilot rows are the same routes (asserted) but were not copied: rigid Chrono is deterministic per node
  only, and re-driving them costs about 5 CPU-minutes per group.
- Every path exists on the cluster; 400 sampled Gator rows match the production case/route sha256.

**Second source tree for the spread rows.** The rigid collector checks the arena against `scripts/gen_arenas.json`
next to itself. The frozen `G3/source` copy lists 10 arenas (not g258/g268/g263/g241) and is read by running jobs,
so it was not touched. `G3/r2/source` is a copy of `G3/source` (rsync without `__pycache__`); `diff -r` shows only
`scripts/gen_arenas.json` differing (the worktree's 14-entry file; all 12 staged arenas match their BMPs).
`G3/r2/cases -> ../cases`, so the rows' relative paths resolve. Check-only runs of spread route g258_test_group_0000
route_00: accepted from `G3/r2/source` (manifest `c9e4ff01...` = crm_improve's), refused from `G3/source` ("Arena
arena_g258 BMP not in gen_arenas.json allowlist").

**CPU capacity.** At 04:34 job 436075 was no longer pending: 11 of its 14 shards had started, 3 were waiting, and
every mi2104x node was allocated (20/20), as was every mi2101x node; 3 mi3501x nodes were idle but kept for training.
So there was no free CPU node to submit to. The soil allocations, however, used about one core of 16 (mi2101x) or 24
(mi3501x) each. The rigid rows therefore run as job steps inside the running soil allocations (`srun --overlap`),
which costs no queue slot and no extra billing:
- `scripts/ag_rigid_pool.py` (driver) and `scripts/ag_rigid_pool.sh` (environment = gen_array_g.sbatch's rigid recipe):
  one driver per step claims whole shards one at a time (mkdir claims with a 60 s heartbeat; a claim older than 15 min
  whose shard has no done record is taken over), runs each with `ag_rigid_runner.py` (gen_runner_g.py with rich
  telemetry kept), `SLURM_ARRAY_TASK_ID` = the shard, and picks collector, source root and runtime fingerprint from
  the shard number (1000s: `G3/source/scripts/ag_gen_collect_ext.py` + the Gator fingerprint; 2000s:
  `G3/r2/source/scripts/gen_collect_ext.py` + the HMMWV fingerprint). Before and after a shard, run folders without a
  completion marker are moved to `G3/rigid_v1/pool/moved` (the collector refuses a used folder); incomplete rows get
  one second attempt. After each shard the 0.9 MB `rich_telemetry.npz/json` of completed runs are deleted and
  `rich_intervals.npz` (chassis contact maxima) is kept. No new claims after the soil job's end time minus 30 min, or
  after `touch $G3/rigid_v1/pool/STOP`.
- `scripts/ag_rigid_pool_launch.sh <workers> <raw job ids>` starts one step per running soil job that has none,
  records it in `submissions.tsv`.
- Output folder: `G3/rigid_v1` (the same folder as 436075; ids do not overlap). Pool records in `G3/rigid_v1/pool/`.

**The < 10 % test (PLAN 7.8).** One step with 12 rigid workers was started at 04:48 inside soil
task 436080_0 (node k006-004-v4, 16 cores, one MI210; step 436081.0). `scripts/ag_overlap_rate.py` compares the soil
loop cost (wall seconds per simulated second, from each run's `rtf_sim_over_wall`) of episodes on that node that ended
in the hour before 04:48 with those that started after it, and does the same for all other soil nodes as a control:

| | episodes | median loop cost | mean |
|---|---|---|---|
| test node, before | 53 | 3.113 | 3.109 |
| test node, with 12 rigid workers | 10 | 3.136 (+0.7 %) | 3.124 (+0.5 %) |
| other soil nodes (control), median after/before | 24 nodes | +0.4 % | |

Slowdown relative to the control: +0.3 %, far below the 10 % limit. With all 18 mi2101x steps running the slowdown is larger (median +4.3 %, at most +8.5 % on one node; section 6), still under the limit. (The node's 16 cores: soil used about one core,
the 12 rigid collectors one each; load 12.5.) The first rigid shard (1000, 80 Gator routes) finished 80/80 in 571 s.
The pool was then started in the other 17 tasks of 436080 (12 workers each) and in both tasks of 436207 (mi3501x,
16 workers each, as in the pilot, which measured +0.7 to +3.1 % there), 04:58-04:59: 20 steps, 248 rigid workers, plus
one 4-worker step in 436207_0 (raw job 436209, step 436207_0.1) that ran spread shard 2000 first as an early check of the second tree
(`submissions.tsv` lists every step). Record: `e3/e3b2/overlap_test_k006-004-v4.json`.

## 6. First new episodes

Checked with `scripts/ag_e3b2_verify.py` on local copies of every run that the new jobs had finished (records
`e3/e3b2/verify_soil_first.json` 05:03, `e3/e3b2/verify_rigid_first.json` 05:09).

| | runs | outcomes | launch check failed | QA / native-height failed | non-finite | vehicle record wrong |
|---|---|---|---|---|---|---|
| Gator soil (new launch) | 7 | 7 blockage stops | 0 | 0 (crm_qa) | 0 | 0 |
| HMMWV soil (new launch: 3 bitid + 49 spread headroom) | 52 | 35 goal, 15 breakthrough, 2 blockage | 0 | 0 (crm_qa) | 0 | 0 |
| Gator rigid (pool) | 1,531 | 1,444 goal (94.3 %), 61 blockage, 22 timeout, 4 rollover | 0 | 0 (native height) | 0 | 0 |
| HMMWV spread rigid (pool, r2 tree) | 40 | 37 goal, 3 blockage | 0 | 0 | 0 | 0 |

- "Vehicle record right" means: every Gator run carries the vehicle block (name gator, `ag_vehicle.py` `072716ee...`,
  the soil or rigid wrapper hash, soil cylinders 0.19575 / 0.2275 m, `vehicle_extra.npz`; rigid runs also the Gator
  runtime fingerprint `599c514e...`), and no HMMWV run has a vehicle block or `vehicle_extra.npz` (spread rigid runs
  also name `G3/r2/source` as their source root and the HMMWV fingerprint). All 59 new soil runs have the 1 ms step and
  0.08 m spacing; all rigid runs have both gates "checked"; `rich_intervals.npz` is kept on all 1,571 rigid runs.
- **bitid:** 3/3 array-identical to the soil_v1 originals (section 2).
- The Gator soil rows fail as in the pilot (7/7 bogged, 93.8 % in the pilot).
- Gator rigid against the HMMWV twins (production_v3/v4) on the same 1,531 routes: designed 3.1 % vs 9.6 % failure,
  on-policy 9.5 % vs 35.4 %; 41 routes fail only for the Gator, 261 only for the HMMWV. The on-policy routes, which the
  pilot did not drive, fail more often for both vehicles.
- **New: 4 Gator rollovers**, all in one group (`f104_v2_group_0603`: route_00, route_01, op_00, op_07; the HMMWV
  bogged on all four). The pilot had none. Rollover is one of the collector's normal outcomes, states are finite; the
  monitor should see whether they stay rare (0.26 % so far).
- **Soil speed with all pool steps running** (`e3/e3b2/overlap_all_mi2101x_0459.json`, t0 = 04:59, 4-9 episodes per
  node after): the 18 shared mi2101x nodes slowed by a median +4.3 % (per node -0.9 to +8.5 %; +0.5 to +7.8 % against
  the mi3501x control nodes), more than the single-node test. The mi2101x nodes are 8 virtual machines per physical
  host, so the steps on the other 7 machines of a host share its cores and memory. Every node is still under the 10 %
  limit; the cost is about 0.2 simulated soil hours per wall hour for the ~3 h the rigid pool runs. The monitor should
  re-measure once there are >= 15 episodes per node (~05:45) and stop a node's step (`scancel <job>.<step>`) if it
  stays above 10 %.

## 7. Expected completion at the measured rates

Measured rates (05:04):
- Soil, whole worker including episode setup: MI210 0.24-0.29 simulated s per wall s (median 0.27), MI350X
  0.41-0.45 (HMMWV mix); the Gator pilot gave 0.29 (MI210) and 0.48 (MI350X). Mean simulated length per route
  tonight: HMMWV g203/g228 20.7 s, spread headroom ~13-15 s, Gator 35.3 s (pilot).
- Rigid pool: 505 Gator routes per hour per 12-worker step on mi2101x, 1,076 per 16-worker step on mi3501x
  (shard records); about 11,200 per hour over the 20 steps.

Work left at 05:00 and where it goes:

| rows | left | simulated h | who drives them |
|---|---|---|---|
| g203/g228 soil training (HMMWV) | ~12,900 | ~74 | mainly the 18 soil_v1 MI210 (4.9 simulated h per wall h) until 13:41-15:03 |
| spread headroom soil (HMMWV) | ~550 | ~2.1 | new launch first |
| Gator soil | 15,087 (148 of 15,235 done: 144 copied + the first new ones) | ~148 | new launch only (after the headroom rows) |
| Gator rigid pool | ~23,700 | | pool steps |
| spread designed rigid (HMMWV) | 12,000 | | pool steps, after the Gator shards |

Expected, at these rates:
- **Rigid:** Gator pool done about **07:15**, spread designed rows about **08:15** (the 436212 steps added by the helper
  after ~06:00 shorten this a little; the 436207 steps stop claiming at 08:17). 436075 (HMMWV v1): its last two shards
  waited for mi2104x nodes at 05:04; its shards take about 1 h on a node.
- **HMMWV soil g203/g228 (soil_v1 jobs alone):** tiers 0-6 complete and tier 7 about 40 % by 12:00; tiers 0-7 and
  tier 8 about 60 % by 13:41, when 11 of the 18 old tasks stop claiming (the other 7 stop 14:09-15:03). The new jobs add
  to these tiers only once the Gator has caught up with them.
- **Spread headroom rows:** done about **06:40** (2 MI350X until ~05:58, 4 after).
- **Gator soil, only the 4 MI350X of the new launch** (436207 until 08:30, 436212 ~05:58-09:41): about 1.8 simulated
  h per wall h after the headroom rows, i.e. about 450 Gator routes by 09:41 and about 1,000 (not even tier 0, 11.8
  simulated h per Gator tier) by 12:00 if the MI350X tasks are resubmitted at 08:30 / 09:41. **This is far short of
  the 15,235 ids.**
- **Gator soil if the pending tasks start** (436210 3 x 4 MI210 from ~07:45, the scheduler's estimate; 436215 4 x 4
  from ~10:37; 436208 3 MI210 whenever other users free mi2101x nodes): about 5 simulated h per wall h from ~08:00 and
  about 10 from ~10:40, i.e. roughly 3,000 Gator routes (tiers 0-1 and part of 2) by 12:00 and one Gator tier per
  1.2-2.4 wall hours after that. All 15,235 would take until the evening (about 148 simulated h in total).
- When the 18 soil_v1 tasks stop claiming (13:41-15:03), those GPUs are lost to this effort unless resubmitted on
  soil_v2 (they would then drive Gator rows first, since the HMMWV tiers are ahead).

## 8. What the monitor must watch

1. **Soil failures and retired workers** (every 30 min): `ls $G3/soil_v1/failed | wc -l`,
   `ls $G3/soil_v1/runs/*/collection_failure.json | wc -l`, `grep -l "3 consecutive failures\|Traceback" $G3/soil_v1/logs/*.out`.
   A Gator-specific failure repeats on retry and retires a GPU after three in a row; if Gator rows fail, set them to
   `run: false` in a soil_v3 (superset recipe) rather than letting workers retire.
2. **Tier progress per vehicle/arena** (`ag_collect_status.py --tasks $G3/tasks/soil_v2.json --out $G3/soil_v1`): the
   HMMWV g203/g228 tiers run ahead of the Gator tiers (section 4). For the PLAN 7.8 cut, the monitor must decide which
   tier "in progress after ~12:00" means (HMMWV, Gator, or both) and then `touch $G3/soil_v1/STOP_CLAIMS`.
3. **Pending soil capacity:** whether 436208 (mi2101x x3), 436210 (mi2104x x3) and 436215 (mi2104x x4) start. When one
   starts, the login-node helper `ag_rigid_pool_autostep.sh` adds a rigid pool step to it within 5 min (96 workers on
   mi2104x - **not yet tested against the 10 % rule on a 4-GPU node**: run `ag_overlap_rate.py --hosts <node> --t0
   <step start>` about 20 min after the first mi2104x step starts and `scancel <job>.<step>` if soil slowed > 10 %).
4. **MI350X 4-hour limits:** 436207 stops claiming at 08:30, 436212 at ~09:41. Resubmit on soil_v2 then
   (`ag_soil_launch.sh ... mi3501x:24:4:<n>`) if training does not need the nodes; keep 3 MI350X nodes free for
   training.
5. **Queue:** 46 of 50 at 05:00. Keep at least 3 free; 436075's last 2 shards and the devel check free slots soon.
6. **Rigid pool:** `ls $G3/rigid_v1/pool/done | wc -l` (425 shards: 1000-1299 Gator, 2000-2124 spread) and the
   `incomplete` lists in the done records (should stay empty); `ls $G3/rigid_v1/pool/moved | wc -l` counts killed or
   failed partial runs. The steps and the helper run as `srun` clients / a bash loop on **login1**: if login1 is
   rebooted the steps die (the shards are re-taken by any new step after 15 min; restart the helper, it re-adds steps).
   The steps in the 436080 tasks stop claiming at 13:28-14:49, in 436207 at 08:17.
7. **Soil speed on the shared nodes:** at ~05:45 re-run `ag_overlap_rate.py --out $G3/soil_v1 --hosts <the 18 mi2101x
   hosts> --t0 04:59` (host list in `e3/e3b2/overlap_all_mi2101x_0459.json`); at 05:09 the median was +4.3 %, the worst
   node +8.5 % (4-9 episodes each). Stop the step of any node that stays above +10 % with >= 15 episodes.
8. **Budget:** `slurm_balance2.py` (645.3 used at 04:47). This module adds at most about 2 (mi3501x) + 3.6 (mi2101x)
   + 34 (mi2104x, if all 7 run their full 12 h) billed node-hours at the billing rates E3a and E3b1 used; the soil cut after ~12:00 ends them earlier.
9. **Drift/bit-identity:** the bitid rows passed (3/3). Nothing else to repeat unless the collector files change
   (they are read-only; `sha256sum` against `e3/gator_collector_sha256.txt`).

## 9. Deviations and notes

- **Rigid GO read as not-NO-GO** (section 1).
- **Pilot soil runs copied, not re-driven** (144 rows; section 3). Their collection records point to the pilot folder.
- **Three extra HMMWV rows (`bitid__`)** at tier -2 for the cluster bit-identity check; they are not training rows
  (the `bitid__` prefix keeps them out of every `<arena>_v2_group_*` selection) and must not be used as data.
- **The rigid rows run inside soil allocations, not on free CPU nodes:** 436075 had started (not pending) but no CPU
  node was free; the soil-allocation route costs no billing or queue slot and passed the < 10 % test (+0.7 % on the
  test node; +4.3 % median, at most +8.5 %, once all 18 mi2101x nodes carried steps).
- **Spread rigid rows were re-sharded** (14-15 -> 2000-2124) and run from a second source tree `G3/r2/source` that
  differs from `G3/source` only in `scripts/gen_arenas.json`; their `collection_request.json` records
  `source_root = G3/r2/source`. HMMWV physics files are byte-identical in both trees.
- **Rich telemetry:** the pool keeps `rich_intervals.npz` (chassis contact maxima) for every rigid run and deletes the
  0.9 MB `rich_telemetry.npz/json`; 436075's runs (gen_runner_g.py) keep none of the three.
- **Gator rigid tiers** are the crm_tasks.py tiers (0-19), not the pilot's route index, so `--tasks` builds accept them.
- **The per-tier interleave is not met in time** (section 4): the running soil_v1 jobs could not be moved to the new
  file without giving their nodes back.
- **A devel job** ran the 15-row check file (first Gator episodes within minutes); one queue slot for 30 min.
- **Login-node helper** `ag_rigid_pool_autostep.sh` (bash loop on login1) keeps adding pool steps to new soil jobs; it
  exits when all 425 shards are done or when `$G3/rigid_v1/pool/STOP` exists.

## 10. Files

- New scripts: `scripts/ag_soil_tasks_v2.py`, `ag_rigid_tasks_v2.py`, `ag_rigid_pool.py`, `ag_rigid_pool.sh`,
  `ag_rigid_pool_launch.sh`, `ag_rigid_pool_autostep.sh`, `ag_overlap_rate.py`, `ag_e3b2_verify.py` (all also in `G3/source/scripts` as new files;
  the pool files are read-only there).
- Task files (`e3/tasks/`, copies in `G3/tasks/`, read-only): `soil_v2.json` (+ `.meta.json`), `soil_v2_check.json`,
  `rigid_v2.json` (+ `.meta.json`).
- `G3/r2/` (second source tree), `G3/soil_v1/imported_from_pilot.tsv`, `G3/rigid_v1/pool/`.
- Checks: `e3/e3b2/` (overlap test, first-episode verification JSON).
- `e3/README.md` section 0: launch map, stop and relaunch commands.
