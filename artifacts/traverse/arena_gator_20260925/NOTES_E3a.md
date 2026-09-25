# NOTES E3a: staging and launch of the HMMWV collections on g203/g228 and the g217 headroom drives (2026-09-25, 01:42-02:25)

K3 = this folder, G3 = `/work1/dannegrut/harry/experiments/arena_gator_20260925`. Launch and stop commands, and the
relaunch recipe for the Gator rows, are in `e3/README.md`. Nothing in the earlier artefact folders or the earlier
cluster roots was written; no arena folder was copied into an earlier cluster source tree (the soil jobs use the new
root G3 instead of `crm_f104_20260916`).

## 1. Cluster state at launch (why the soil pool is small)

At 01:42 the cluster was busier than REVIEW_R2 found at 01:20: another user had just taken the last 7 idle mi2104x
nodes for 6 h, so all 20 working mi2104x nodes were busy. Free GPUs: 11 mi2101x (MI210) + 7 mi3501x (MI350X, one more
drained). Every 8-GPU node was busy with queues of other users' jobs.

- Soil runs on 11 MI210 + 4 MI350X at first: about 5.5 simulated hours per wall hour from the real-time factors, 4.5-5 measured with episode setup (section 6). It rises to about 8.4 if the 9
  waiting mi2101x tasks get nodes (other users' waiting mi2101x jobs are mostly held by dependencies).
- The training rows need about 95 simulated hours (15,756 routes x ~21.6 s, the f104 mean). At 5.5-8.4 simulated
  hours per wall hour, the 12:00 cut reaches about 55-85 simulated hours, i.e. about tiers 0-6 to 0-10 of 0-12, unless
  mi2104x or 8-GPU nodes free up.
- The rigid array waits on mi2104x ("Priority"). Two other users' jobs wait there ahead of it; one running 4 h job ends
  about 04:15, most others run for many hours.

## 2. Staging (G3)

| item | where | check |
|---|---|---|
| source tree | `G3/source`: `src/`, `scripts/` rsync'd from this worktree (no `__pycache__`), `assets/traverse/arena_{f104_50h_v1,g203,g228,g217,g260,g271,g251,g247}` | 9 frozen files (`gen_collect.SOURCE_FILES`) equal the crm_improve_20260922 manifest hashes; `source_manifest.json` is a byte copy of crm_improve's (sha256 `c9e4ff01...`) |
| allowlist | `G3/source/scripts/gen_arenas.json` (E1's 10 entries) | every entry equals its staged BMP (g216, g231 listed but not staged) |
| collectors | `crm_collect.py` `cb6792be`, `crm_worker.py` `f856b998`, `gen_collect.py` `b6ba0622`, `gen_collect_ext.py` `b9f36a02`, `gen_runner_g.py` `b47c9fd5`, `gen_array_g.sbatch` `e3bcb4d8`, `gc_control.py`, `f104_n2_sampler.py` | each equal to the crm_improve_20260922 source copy |
| soil config | `G3/configs/crm_main.json` `90cd049e` | equal to `crm_f104_20260916/configs/crm_main.json` (collect_v1) and `crm_f104_v1/configs` |
| cases | `G3/cases/` = all of `K3/cases` (training pools, on-policy routes, suites; 69,275 files) | file count equal; every task path exists on the cluster |
| maps | `G3/maps`, `G3/map_roots` (relative links, resolve), `G3/grids` | for cluster-side dataset builds later (REVIEW_R2 amendment 10) |
| dev picks | `G3/e3/dev_headroom/` = `K3/e3/dev_headroom/` | |
| task files | `G3/tasks/soil_v1.json`, `G3/tasks/rigid_hmmwv_v1.json` (+ `.meta.json`) | |

Frozen (`chmod a-w`) on the cluster: the collectors and runners above, `ag_soil.sbatch`, `ag_soil_launch.sh`,
`gen_arenas.json`, every `src/nedm/**/*.py`, `configs/crm_main.json`, both task files. E2's cluster folders
`G3/e2_tools`, `G3/runtime` were left alone.

## 3. Task files

**Rigid `rigid_hmmwv_v1.json`** (`scripts/ag_rigid_tasks.py`, sha256 `168044497dd338de...`): 60,200 rows.
- g203 and g228: all 1,200 groups x 20 routes = 24,000 each (14,400 designed + 9,600 on-policy per arena), shards
  0-11 by md5(group). Ids `<group>_route_NN` / `<group>_op_NN`, `tier` = the soil tier of the same route,
  `episode_seed` = md5(id) (provenance only; rigid does not read it).
- 12,000 designed routes on the 1,000 test groups (REVIEW_R1 5c), shards 12-13.
- 200 f104 drift rows in shard 0 (REVIEW_R1 7b).
- Splits in the rows: train 53,878 / val 2,625 / test 3,697.

**Soil `soil_v1.json`** (`scripts/ag_soil_tasks.py`, sha256 `db0ba97f4b763ae6...`): 16,066 rows.
- 300 g217 dev rows (tier -1).
- 10 f104 drift rows (tier -1).
- g203 and g228 groups 0-605, tiers 0-12: 15,756 rows.
  - 1,212 rows per tier, one route per group per tier. Written tier by tier with the arenas alternating, and the
    workers sort by tier.
  - g203 has 4,746 designed + 3,132 on-policy rows; g228 has 4,725 + 3,153.
  - The groups hold 559 (g203) and 520 (g228) training-split groups.

Checks:
- `ag_tasklib.group_routes` reproduces all 24,000 rows of the f104 `tasks_train.json` (tier, route, episode_seed) when
  given the f104 group names.
- All 15,756 soil training ids have a rigid twin with the same id, tier, case and route.
- On-policy paths are taken from each `routes.json` and asserted equal to the `op_NN` file.
- Ids and episode seeds are unique.
- Every path exists locally and on the cluster.

**Dev headroom picks** (`e3/dev_headroom/`):
- Map check first (`scripts/ag_map_check.py`): the g217 map root's BMP hash equals the cases' arena. The negative
  control, the f104 map with the g217 cases, fails.
- Frozen K1 soil specialist `crm_f104_v1/train_v1/deploy/CRM_N2_s{0..4}.pt`, the model behind
  `generalist_20260921/A_adapt/suite/picks_crm_Scrm` (hashes in `models.sha256`).
- Planner command: `ga_planner.py --cases K3/cases/dev_g217/cases --map-root K3/map_roots/g217 --models '...CRM_N2_s*.pt'
  --world crm --arms B --ref-picks /nonexistent_no_reference --task-root K3`. This is CEM 4 x 64 (tag `n2iter_cem4x64`)
  from the case pose at rest, 62 s for 150 groups on the 5090. Picks lock `picks_crm_Scrm_g217/PICKS_LOCKED.sha256` =
  `2d9044ca...`.
- Straight 6 m/s (`scripts/ag_dev_headroom.py`): the offset-0 / 6 m/s anchor of the night-2 proposal pool, built as
  `crm_pools.py` does. It differs from the specialist's pick in all 150 groups, so there are 300 drives.
- Combined lock `DEV_PICKS_LOCKED.sha256` = `73a80be249aea107...`, written before any drive.
- Run ids are `<g>__Scrm_B` and `<g>__straight6`. ga_planner's own id `<g>__B` would clash with any later model's dev
  pick in the shared soil folder.

## 4. Review amendments applied here (staging, task files, collection)

REVIEW_R2:
- **1 (collector and config explicit):**
  - New `scripts/ag_soil.sbatch` (copy of crm_collect.sbatch with CRM_ROOT = G3). It refuses to start unless
    CRM_COLLECTOR and CRM_CONFIG are exported and the collector's name contains `crm_collect`. It prints the collector,
    worker, config and task-file sha256 into every job log.
  - The devel smoke confirmed the config arrives (step 0.001 s, spacing 0.08 m in `collection_request.json`).
  - Rigid submissions pass GEN_COLLECTOR / GEN_RUNNER / GEN_ROOT explicitly.
  - **Deviation:** HMMWV rows in this launch call the unchanged `crm_collect.py` directly, not the `ag_crm_collect.py`
    dispatcher. E2 is still editing `ag_crm_collect.py` / `ag_vehicle.py` and every episode re-reads the collector, so
    an unfrozen dispatcher could break HMMWV rows mid-run.
  - The Gator relaunch switches to the frozen dispatcher. E2 showed HMMWV rows through it are array-identical
    (`e2/bitid`), so rows from both launches mix.
- **2:**
  - Soil launched at 01:58 instead of 03:00.
  - No row above tier 12 in any soil file.
  - The cut procedure (drop the partly filled last tier on every arena at dataset time) is in `e3/README.md` section 5.
  - Declaring what gets dropped next is left to the plan owner.
- **3:**
  - Sized launch `scripts/ag_soil_launch.sh`, replacing `crm_launch.sh`:
    - one sbatch per partition, arrays sized by hand;
    - it refuses if queued + new tasks > 45;
    - every job id is checked and recorded in `G3/e3/submissions.tsv`.
  - 3 MI350X nodes were kept out of soil (for training and the Gator pilot).
- **4:**
  - Rigid runs on mi2104x only; soil on mi2101x + mi3501x only, so the two do not compete.
  - The rigid pools were not cut to 13 tiers: the task asked for all 20. The rows carry the soil tier, so E4 can cut
    rigid to tiers 0-12 for matched designs.
- **6:**
  - Map/arena check tool (`ag_map_check.py`), run before the dev picks.
  - Ids carry the arena prefix (`g203_v2_group_*`, `g228_v2_group_*`, `g217_dev_group_*`, `drift__*`), so builders can
    glob one arena at a time in the shared folders.
  - The Gator id prefix rule is written into the relaunch recipe.
- **8:** relaunch recipe written (`e3/README.md` section 6).
  - Priority rows use negative tiers (dev and drift at -1), and the planner's `tier = group index` was replaced.
  - `ag_soil_tasks.py --append ... --check-superset soil_v1.json` builds the extended file with every old row unchanged.
- **9:** the headroom check uses exactly the CRM_N2 ensemble and `ga_planner.py --world crm --arms B`.
- **10 (partly):** maps, map roots and grids are staged on the cluster, so datasets can be built there.
- **12:**
  - `scripts/ag_collect_status.py` reports per kind and arena.
  - All HMMWV rows of this effort run from the `G3/source` tree (recorded in every run's `collection_request.json`).

REVIEW_R1:
- **5c (rigid part):** 12,000 designed test-arena routes added to the rigid file, in their own shards so they can be
  cancelled alone.
  - The optional 2,400-drive soil subset was **not** added: soil capacity is the constraint. It can be appended as the
    last tier at the relaunch.
- **7a:** map/arena check done for the dev picks.
- **7b:** drift rows added: 10 f104 soil ids at tier -1 and 200 f104 rigid ids in shard 0.
  - Compare them with `scripts/ag_drift_check.py`. Soil must match array for array; rigid needs >= 95 % the same
    status.
  - The devel smoke's single rigid drift id already matched production_v3 (goal at 9.45 s).

Not applied here:
- REVIEW_R1 6's "run the dev check with tonight's M1". The task specifies the frozen K1 specialist. An M1 arm can be
  added later as more tier -1 rows with id tag `__M1_B`, with no id clash.
- Everything about Gator criteria, suites, statistics, training and evaluation order (R1 1-4, 8-11; R2 5, 7, 11).
- E1 verifier caveat 1 (g228 needs its first 634 groups for 545 training groups): the task says groups 0-605.
  - With 606 groups, g203 has 559 training groups and g228 has 520. That fully covers M2 (f104 + g203, 545 + 545) and
    M3 (363 per arena).
  - Only A3's g228 soil share (520 instead of about 545) and a one-arena g228 model at 545 (R1 5a) are short.
  - `ag_soil_tasks.py --groups-for g228=634` adds the 28 groups (364 drives, about 2.2 simulated h) at the relaunch
    if wanted. Their rows would sit in the same tiers and catch up.

## 5. Jobs, first checks

- 436073 (rigid smoke, devel): 4/4 complete.
  - g203 designed: goal. g228 on-policy: prolonged blockage at 77 s. g260 test designed: goal.
  - f104 drift: goal at 9.45 s, the same as production_v3.
- 436074 (soil smoke, devel): 2/2 complete, crm_config step 0.001.
  - g203: goal in 9.2 s, wall 33 s. g228 on-policy: soil breakthrough at 30.3 s, wall 100 s.
- 436075 rigid production: 14 tasks, waiting on mi2104x.
- 436080 soil, mi2101x x 20: 11 running.
- 436092 soil, mi3501x x 4: running.
- First soil episodes and the drift checks: section 6.

New or changed files:
- New scripts: `scripts/ag_tasklib.py`, `ag_rigid_tasks.py`, `ag_soil_tasks.py`, `ag_dev_headroom.py`, `ag_map_check.py`,
  `ag_drift_check.py`, `ag_collect_status.py`, `ag_soil.sbatch`, `ag_soil_launch.sh`.
- No existing script was edited.
- Local outputs: `e3/tasks/`, `e3/dev_headroom/`, `e3/README.md`.

Billed estimate:
- Soil: about 1.6 billed per wall hour now, up to about 2.5 with all 20 mi2101x tasks, so about 16-25 by 12:00.
- Rigid: about 15 mi2104x node-hours, about 6 billed.

## 6. First episodes and checks (02:00-02:25)

**State at 02:19.**
- 390 soil episodes complete.
- 0 failed ids, 0 `collection_failure.json`, 0 launch-check failures (75 checked at 02:03), 0 retired workers,
  0 tracebacks.
- The dev and drift rows (tier -1) finished at about 02:16; the workers are in training tier 0 (80/1,212 at 02:19).
- 39 of 50 queue slots in use: rigid 14 waiting, soil 15 running + 9 waiting, 1 devel drift job.
- At 02:19 mi2101x had 0 idle nodes, mi2104x 0, mi3501x 3 (kept free).

**Speed.**
- Wall time per simulated second, averaged per kind: 2.6-3.3.
- Per worker, including episode setup: 0.26 simulated s per wall s on MI210, 0.41 on MI350X. That is below the
  0.32 / 0.5 real-time factors because dev and early episodes are short (setup is a larger share).
- The pool therefore gives about 4.5-5 simulated h per wall hour: about 1.5 wall h per training tier (1,212 routes x
  ~21.6 s = 7.3 simulated h).
- By 12:00 that is about tiers 0-5 of 0-12, or about 0-9 if the 9 waiting mi2101x tasks start. The 4 mi3501x tasks
  stop claiming at about 05:40.

**Status mix of the first 80 training episodes (tier 0, both arenas).**

| rows | goal | soil breakthrough | blockage |
|---|---|---|---|
| g203 designed | 10 | 21 | 2 |
| g228 designed | 7 | 16 | 4 |
| on-policy, both arenas | 1 | 14 | 5 |

That is 78 % not goal reached, against 68 % on f104 collect_v1 (designed 58 %, on-policy 84 %). It is an early sample,
biased towards short episodes, and not yet comparable.

**Dev headroom, all 300 drives done** (g217, standing start, 150 hill/crater groups).

| arm | goal reached | other outcomes |
|---|---|---|
| frozen f104-only soil specialist, CEM 4x64 | **95.3 % (143/150)** | 6 breakthroughs, 1 blockage |
| straight 6 m/s | 73.3 % (110/150) | 38 breakthroughs, 2 blockages |

- Specialist by stratum: hill cross 46/50, hill entry 26/28, crater cross 44/44, crater entry 27/28.
- On f104 the same model fails 5.8-7.8 % of hill groups and 1.0-2.7 % of crater groups (REVIEW_R1 section 1). On g217
  it fails 7.7 % of hill groups (6/78) and 1.4 % of crater groups (1/72).
- So on this dev arena the f104-only soil model loses no points against its f104 level. That is the case PLAN 2.2
  names: "no gap at all". The decision on the primary is not made here.
- Straight 6 m/s gets 73 % here against 66.5 % on the 200 f104 eval groups: the arena is not harder for a model-free
  route.

**Soil drift check (REVIEW_R1 7b)** (`e3/drift_soil.json`, `scripts/ag_drift_check.py`).
- 10 f104 collect_v1 ids re-driven on tonight's tree: 10/10 have the same outcome.
- 9/10 are array-identical (trajectory, command reference, anchor state, soil extras).
- The 10th, `f104_v2_group_0885_op_07`, is goal reached in both, at 23.30 s against 23.15 s. Its state first differs
  at frame 7 (0.35 s, inside the braked settle), by about 1e-6 m in pose.
- Re-runs of that id tonight:
  - twice on another MI210 node (devel, job 436103);
  - once on an MI350X mi3501x node (job 436106).
- All re-runs are identical to tonight's run. Only the collect_v1 run, on mi3508x node k007-003, differs. Another
  collect_v1 id driven on an mi3508x node (k007-002) was reproduced exactly.
- Reading: tonight's tree reproduces itself exactly across MI210 and MI350X, and reproduces collect_v1 in 9 of 10 ids.
  The one difference is a single old run that nothing reproduces.
- It is not a systematic build or collector change, which would show from frame 0 in more runs. But the strict
  "byte-identical" rule of R1 7b is met by 9/10, not 10/10.
- 15 more collect_v1 ids are being re-driven on devel (job 436108, output `G3/e3/drift_more`, about 21 min) to size
  the rate. Check with:
  `rsync -a --include='drift__*/***' --exclude='*' amd:$G3/e3/drift_more/runs/ /tmp/ag_drift_more/ &&
  $PY scripts/ag_drift_check.py --mode soil --runs /tmp/ag_drift_more`.

The rigid drift rows (200) run inside rigid shard 0, whenever mi2104x frees up.
