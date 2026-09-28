# Scout S3: smoke-test design, collection recipe, cost and cluster (Polaris and M113 on f104 soil)

Read-only scout, 2026-09-27, 22:40-23:30 CDT. Worktree `/home/harry/NeDM-traverse_mppi`, branch `offroad_vehicles_v1`.
Short names used below:
- **K3** = `artifacts/traverse/arena_gator_20260925` (the study that added the Gator).
- **G3** = `/work1/dannegrut/harry/experiments/arena_gator_20260925` (its cluster root).
- **K4 / G4** = this study's local / proposed cluster root (`artifacts/traverse/offroad_vehicles_20260927`,
  `/work1/dannegrut/harry/experiments/offroad_vehicles_20260927`).
- **"soil"** = the deformable CRM ground: SPH particles, 0.08 m spacing, 1 ms step, 0.24 m deep (`configs/crm_main.json`).
- **"tier"** = the position of a route inside its start/goal group. Tier k holds one route of every group, and the
  workers drive tier 0 first, then tier 1, and so on.
- **"id"** = one route of one start/goal group, e.g. `f104_v2_group_0426_route_03`. A vehicle prefix goes in front
  (`gator__...`).

Nothing existing was edited. Nothing was submitted to the cluster; the cluster commands were `sinfo`, `squeue`,
`sshare`, `sacct`, `sacctmgr`, `scontrol`, `slurm_balance2.py`, `cat`/`ls`/`grep`, one `rsync` pull and one PyChrono
import check. I ran no local soil simulation.
Scratch files are in `K4/scratch/S3/`:
- `throughput.py`, `sacct_since0925.txt`, `workers_soil_v1.txt`: throughput and cost per partition;
- `cost_table.py`: the table in section 4;
- `smoke_samples.py`, `sample_A.json`, `sample_B.json`: the declared smoke samples with their stored outcomes;
- `drift_more/`, `drift_more_check.json`: a reproducibility check that had never been read out.

---

## 0. Answers in brief

- **The Gator pilot = 144 soil routes** (24 f104 groups x tiers 0-5, 4 groups per terrain type, lowest md5 of the
  group name), plus the same 144 routes with soil wheels 0.08 m larger.
  - Goal not reached: Gator 93.8 %, Gator with larger wheels 80.6 %, HMMWV 69.4 % on the same routes.
  - All drives valid, cost 0.85 billed.
  - Full collection on identical routes: Gator 88.2 % vs HMMWV 68.1 % not reached (3,263 routes fail only for the
    Gator, 201 only for the HMMWV).
- **Node caveat, soil version.** The "not reproducible across nodes" note is about rigid (CPU) Chrono. Soil (GPU)
  runs reproduce across GPU types, and repeat old runs with the same outcome:
  - 25/25 same outcome and 22/25 identical arrays against runs made 9 days earlier, including 15 re-drives from the
    Gator study that nobody had read out; I compared them tonight.
  - 3/3 identical when re-driven MI210 -> MI350X.

  Stored Gator outcomes are therefore a valid reference. I still recommend re-driving the Gator on the 144 primary
  routes in the same launch: +1.4 simulated hours (~0.5 billed). The stored runs then become a reproducibility check.
- **Smoke test.**
  - Two declared route samples with known Gator and HMMWV outcomes:
    - **A** = the 144 pilot routes: primary, paired route by route.
    - **B** = 96 pairs of the 800-pair f104 test suite (16 per terrain type), each driven along two routes: straight
      6 m/s, and the route picked by the HMMWV-trained planner. Secondary. B also hints at whether 90 % is within reach.
  - Before either: flat-soil driveability checks (settle, 2/4/6 m/s straight, a turn, braking), validity gates, and a
    decision rule declared before any drive.
- **Decision rule for "better than the Gator"** (per vehicle, Holm over the two). All of these on sample A:
  - it passes the validity gates;
  - it fails at least 10 points less than the Gator;
  - exact one-sided McNemar p (a paired test) under the Holm level;
  - the 95 % interval, resampling the 24 groups, lies below 0;
  - it fails less than the Gator did with its larger wheels (80.6 %).

  On sample B, straight 6 m/s must not contradict the result. The 144 routes detect a gain of 8-10 points or more
  (p <= 1e-3).
- **Cost at the Gator's rates.**
  - Soil costs 0.343 billed node-hours per simulated hour, pooled over every soil job of 09-25/26 (0.28 on MI350X
    alone).
  - The Gator needed 9.26 simulated hours per 1,000 ids, i.e. **3.2 billed per 1,000 ids**; the HMMWV 6.0 h and 2.1
    billed.
  - One vehicle, all 15,235 ids, Gator-length drives: **141 h, 48 billed** at equal speed; **212 h / 73 billed** at
    1.5x slower per simulated second; **423 h / 145 billed** at 3x.
  - A vehicle that reaches the goal as often as the HMMWV has shorter drives (x 0.65): 92 / 137 / 275 h and
    31 / 47 / 94 billed.
  - **Smoke test: 6-17 Gator-speed simulated hours, 2-6 billed, 2-6.5 wall hours on the 6 free MI350X.** Sample A
    alone takes 1.2-2.9 h.
- **The cluster is busy, and we are last in line.**
  - Our fair-share priority is about 11 points, against 76-653 for the users waiting now.
  - Only mi3501x has idle nodes (5 now, 6 after 23:15; 4-hour limit per task). Every MI210 node is taken, with about
    30 other tasks waiting on mi2101x. mi2104x frees 5 nodes around 01:45-02:15 and 9 around 05:30-06:30.
  - The account has used 758.2 of 1,500 node-hours (harry 276.9).
  - Projection (uncertain by about +-50 %): about 80 Gator-speed simulated hours by 08:00 and about 160 by noon on
    09-28. That is one vehicle's full set if it runs at Gator speed or better, or two vehicles' first stage (tiers 0-6).
    The M113's full set at 3x is not possible overnight.
- **Staging.** Collect tiers 0-6 first (8,399 ids, the Gator's stage 1). Train and evaluate the first planner on them
  while tiers 7-12 run. Put the evaluation drives ahead of tiers 7-12 in a newer task file, as the Gator study did. If
  both vehicles pass, run them one after the other, not tier-interleaved, using "run": false flags in successive task
  files (section 3.5).

---

## 1. The Gator pilot and the identical-routes comparison

### 1.1 Which routes

- **Rule.** Declared in `scripts/ag_pilot_tasks.py` before any pilot drive (`K3/NOTES_E3b1.md:67-76`,
  `K3/e3/tasks/pilot_gator.meta.json`). In each of the 6 terrain types of the f104 night-2 groups, take the 4
  training-split groups with the lowest md5 of the group name. Take their soil routes of tiers 0-5 from the HMMWV
  soil collection `collect_v1`. The terrain types are:
  hill cross-slope, crater cross-slope, hill entry/cross/exit, crater entry/cross/exit, long traverse and roughness
  transfer.
- **Groups** (numbers of `f104_v2_group_NNNN`):

  | terrain type | groups |
  |---|---|
  | crater cross-slope | 0426 0894 0594 0427 |
  | crater entry/cross/exit | 0188 0259 0295 0404 |
  | hill cross-slope | 1060 1144 0926 0916 |
  | hill entry/cross/exit | 0829 0362 0685 0868 |
  | long traverse | 0213 0177 0286 0621 |
  | roughness transfer | 1103 0563 0575 1187 |
- **Rows.** 144 in all: 86 designed routes (constant 2 m/s 18, constant 4 m/s 16, constant 6 m/s 28, smooth 2-6-2 m/s
  24) and 58 routes proposed by the planner's sampler.
  - They are the rows `gator__<id>` of `K3/e3/tasks/pilot_gator_soil.json` (sha `172ad6f8...`, tiers 0-5).
  - The same file holds the 144 wheel-sensitivity rows `gatorR8__<id>` (both soil cylinders 0.08 m larger), at
    tiers 10-15, so they ran after the calibrated rows.
- **Declared copy for this study:** `K4/scratch/S3/sample_A.json`. It holds the 144 ids with the Gator, larger-wheel
  Gator and HMMWV statuses and episode lengths, taken from `K3/e3/pilot_gator_eval.json`.

### 1.2 Where the outcomes are

| what | local | cluster |
|---|---|---|
| Gator, calibrated wheels (144) | `K3/e3/pilot_gator/soil/runs/gator__*` | `G3/pilot_gator/soil/runs/gator__*`; copied unchanged into `G3/soil_v1/runs/` (list with hashes `G3/soil_v1/imported_from_pilot.tsv`, `K3/NOTES_E3b2.md:101-102`) |
| Gator, wheels + 0.08 m (144) | `K3/e3/pilot_gator/soil/runs/gatorR8__*` | `G3/pilot_gator/soil/runs/gatorR8__*` |
| HMMWV, same ids (not re-driven; `K3/NOTES_E3b1.md:80`) | `artifacts/traverse/crm_f104_v1/collect_v1/runs/<id>` (15,235 runs) | `/work1/dannegrut/harry/experiments/crm_f104_20260916/collect_v1/runs/<id>` |
| per-route table of all three | `K3/e3/pilot_gator_eval.json` (`soil_rows`), text `K3/e3/pilot_gator_eval.txt` | |
| Gator vs HMMWV, all 15,235 ids | `K3/e5/ids_bf/gator_soil_qa_all_summary.json` (one row per id: tier, kind, split, profile, both statuses, both lengths, belly flag) | `G3/e5/ids_bf/` |

### 1.3 How the pilot was launched

`K3/NOTES_E3b1.md:86-92`; every job is also in `G3/e3/submissions.tsv`.

- **436122** (array 0-1): 2 x mi3501x (one MI350X and 24 cores each), 4 h limit, `scripts/ag_pilot.sbatch`.
  - Each node ran one soil worker on the GPU (the unchanged `crm_worker.py` with `ag_crm_collect.py`, the Gator
    switch).
  - It also ran 16 rigid Gator workers on the CPU cores. On the same nodes, rigid HMMWV twins were re-driven as
    `srun --overlap` steps (`ag_pilot.sbatch:27-55`).
- **8 devel jobs** (436123, 436127, 436130, 436131, 436146, 436165, 436166, 436187): one MI210 each, 30 min, some
  chained with `afterany`. They added soil workers on the same task file.
- **Frozen before the first drive.** `ag_vehicle.py`, `ag_crm_collect.py` and the Gator hull file (hashes in
  `K3/e3/gator_collector_sha256.txt`); `NEDM_VEHICLE` unset, since every row carries `--vehicle gator`.
- **Timing.** Wall time 02:31-04:29 (about 2 h) for 288 soil drives, 2.76 simulated hours. The calibrated rows ran
  first.
- **Cost.** About 0.85 billed, rigid work included (`K3/NOTES_E3b1.md:218`).

### 1.4 Pilot results (`K3/e3/pilot_gator_eval.txt`, `K3/NOTES_E3b1.md:10-37`)

| on the 144 routes | Gator calibrated | Gator wheels + 0.08 m | HMMWV |
|---|---|---|---|
| goal not reached | **93.8 %** (135) | 80.6 % (116) | **69.4 %** (100) |
| discordant routes vs HMMWV (fails only for the Gator / only for the HMMWV) | 36 / 1, exact McNemar p 5.5e-10 | 25 / 9 | |
| constant 2 / 4 / 6 m/s, smooth, proposals | 100 / 93.8 / 89.3 / 95.8 / 93.1 % | 83.3 / 56.2 / 78.6 / 83.3 / 86.2 % | 100 / 62.5 / 28.6 / 58.3 / 86.2 % |
| groups with any goal among their 6 routes | 6 / 24 | 10 / 24 | 18 / 24 |
| end states | 134 stopped by the blockage rule, 9 goals, 1 breakthrough | 116 blockage, 28 goals | 44 goals, 86 breakthroughs, 14 blockage |
| mean simulated length | 35.3 s | 33.7 s | 23.0 s |

- **Validity.** 144/144 valid; 0 crashed, non-finite, or failed launch checks. Belly-in-soil flag 4.2 % (limit 10 %).
- **Wheel sensitivity.** 13.2 points, just under the declared 15-point threshold (20 routes turn into goals, 1 the
  other way).
- **Speed.** Gator soil loop 1.92 wall s per simulated s on MI350X and 3.09 on MI210, the same per simulated second as
  the HMMWV (`K3/NOTES_E3b1.md:194-196`). The Gator costs more per id only because its drives last longer.

### 1.5 The identical-routes comparison, all 15,235 soil ids

Sources: `K3/REPORT.md:280-293` and `K3/e5/ids_bf/gator_soil_qa_all_summary.json`. The latter was built by
`scripts/ag_s1_gator_soil_qa.py` on the cluster from `G3/soil_v1/runs` against the `collect_v1` twins.

- **Goal not reached.** Gator 88.15 %, HMMWV 68.05 %: 3,263 routes fail only for the Gator, 201 only for the HMMWV.
  - By profile: 2 m/s 91.3 / 86.2; 4 m/s 83.8 / 56.3; 6 m/s 82.7 / 33.5; smooth 84.2 / 54.5; proposals 92.2 / 83.9 %.
  - Flat across tiers: 87.3-89.3 %.
- **End states.** Gator: 13,231 blockage stops, 1,805 goals, 198 breakthroughs, 1 timeout. HMMWV: 4,867 goals, 8,372
  breakthroughs, 1,985 blockage stops.
- **Simulated hours.** Gator 141.13 h (mean 33.35 s per id), HMMWV 91.51 h (21.62 s). Mean by end state: Gator
  blocked 35.0 s, Gator goal 21.6 s; HMMWV goal 16.6 s, HMMWV breakthrough 20.9 s.
- **What this means.** A vehicle that bogs costs the full 34 s blockage stop on most routes. A vehicle that reaches
  the goal or breaks through quickly costs about 2/3 of that.
- **Planner suite (800 pairs), goal reached.**
  - On the Gator: Gator-trained planner 67.4 %, HMMWV-trained planner 43.6 %, straight 6 m/s 14.3 %.
  - The HMMWV on its own planner: 95.6 % (`K3/REPORT.md:300-310`).

### 1.6 Reproducibility across nodes, for soil specifically

- **The memory note** (`amd-not-reproducible-across-nodes.md`) is about rigid Chrono on CPUs. There, runs differ from
  frame 2 between node types and marginal outcomes flip.
- **Soil runs on the GPU.** The evidence:
  - `K3/e3/drift_soil.json`: 10 `collect_v1` ids (driven 09-16) re-driven on 09-25 give 10/10 the same outcome and
    9/10 identical arrays. The one exception differs at frame 7 by about 1e-6 m and reaches the goal in both runs, at
    23.30 s vs 23.15 s.
  - `K3/NOTES_E3a.md:218-222`: three re-runs of that id, on two MI210 nodes and one MI350X, are identical to each
    other.
  - `K3/NOTES_E3b2.md:61-66`: 3/3 production rows re-driven through the Gator switch are identical in every array, 2
    of them after moving from MI210 to MI350X.
- **New tonight.** The 15 extra drift re-drives of the Gator study (`G3/e3/drift_more`, job 436108) were never
  compared. I pulled their index files and ran the unchanged `scripts/ag_drift_check.py` (output
  `K4/scratch/S3/drift_more_check.json`):
  - 15/15 have the same outcome and 13/15 have identical arrays.
  - `f104_v2_group_0465_route_05` breaks through at 12.3 s vs 13.1 s; `..._0900_route_02` reaches the goal at the
    same time with a slightly different trajectory.
- **Pooled:** 25/25 the same outcome (one-sided 95 % lower bound 88.7 %) and 22/25 identical arrays against runs from
  another date and other nodes.
- **Conclusion.** Stored soil outcomes are a sound reference, far sounder than stored rigid outcomes. A small share
  of long or marginal drives can still differ. Section 2.5 turns this into a recommendation.
- **No rebuild since the Gator runs.** The cluster soil build (`$NRD_ROOT/chrono-build-fsi`) is dated 2026-09-08
  (`libChrono_fsisph.so` 12:05, `_vehicle.so` 12:10), i.e. the one behind the Gator runs. It has `veh.M113`,
  `veh.TrackedVehicle`, `veh.CRMTerrain` and GPU code for gfx90a/gfx942/gfx950 (MI210/MI250X, MI300X, MI350X). Its
  data folder has `vehicle/Polaris/*.json` and `vehicle/M113/*.json` (checked by import on the login node). If M113
  support needs a rebuild, put it in a new build folder and re-drive the Gator controls with it (section 2.5).

---

## 2. Smoke-test design

### 2.1 Stage 0: rigid ground and flat soil (before any route sample)

1. **Rigid ground, local** (short, allowed): the vehicle scouts' checks on the real f104 heightmap (launch check,
   native-height check, states finite, a few designed routes). Nothing here needs the cluster.
2. **Flat-soil driveability, same production soil setting** (0.08 m spacing, 1 ms step, the `crm_main.json` soil
   properties, `crm_collect.build_crm`, a straight flat patch at least 60 m long). One scripted drive per vehicle, run
   for the **HMMWV, Gator, Polaris and M113 alike**, so the numbers can be compared:

   | phase | what | pass (declare before running) |
   |---|---|---|
   | settle | 0.8 s braked settle (as production), then 1.2 s at rest | launch check passes (`crm_collect.py:421`: speed <= 1 m/s, roll/pitch <= 25 deg, chassis 0-1.2 m above ground); vertical speed at the anchor < 0.05 m/s; creep reported (the Gator crept up to 0.40 m/s, rear brakes only) |
   | straight 2 / 4 / 6 m/s | 8 s at each speed, same follower gains | within 10 % of the commanded speed after 4 s at each step; lateral drift < 0.5 m; no breakthrough; slip and sinkage reported next to the Gator's and HMMWV's |
   | turn | constant curvature 0.10 /m (the designed-route curvature cap) at 3 m/s for a half circle | cross-track error < 0.5 m. This matters for the M113, which steers by driving its two tracks at different speeds |
   | braking | full brake from 6 m/s | stops without pitching over; no backward creep > 0.4 m/s; stopping distance reported |
   | every frame | | all 17 state fields finite; vehicle block written; belly clearance defined |

   - **Where to run it.** Locally: at most one short settle-and-drive per vehicle inside
     `flock /tmp/luffy_crm.lock`. The 5090 runs soil at about 1.9-2.0 wall s per simulated s (`K3/LOG.md:22`) plus
     20-60 s of setup, so four ~30 s scripted drives do not fit the 10-minute local budget if the M113 is slow. The
     full four-vehicle set therefore goes to **one mi3501x node with `-t 00:45:00`**; the short limit helps it start
     in a gap. devel (MI210) is full tonight.
   - **Cost.** About 0.1 billed.

### 2.2 Stage 1: paired route samples

| sample | rows per new vehicle | known outcomes | role |
|---|---|---|---|
| **A** = the 144 pilot routes (section 1.1; `K4/scratch/S3/sample_A.json`), ids `<veh>__<collect_v1 id>`, same case, route, episode seed and tier | 144 | Gator 93.8 %, Gator + 0.08 m wheels 80.6 %, HMMWV 69.4 % not reached | **primary**, paired route by route |
| **B** = 96 pairs of the 800-pair f104 suite: in each terrain type, the 16 groups with the lowest md5 of the group name (`K4/scratch/S3/sample_B.json`). Two routes each: straight 6 m/s (`e6/picks/crm/f104/straight6`) and the pick of the HMMWV-trained all-data planner H_full (`e6/picks/crm_bfull/f104/H_full_free`) | 192 | straight 6: Gator 83.3 %, HMMWV 34.4 %. H_full pick: Gator 51.0 %, HMMWV 3.1 %. For comparison, the Gator-trained planner on the Gator: 25.0 % (all 800: 85.75 / 32.25 / 56.4 / 4.4 / 32.6 %) | **secondary**: a planner-free check, and an early sign of whether "reach the goal on 90 %" is plausible |

- **Why A.** The rows already exist, and the group rule is declared and outcome-blind. The larger-wheel Gator gives a
  built-in check of wheel sensitivity, and every route has all three reference outcomes.
- **Why A is enough for the decision.** With the Gator reaching the goal on only 9 of 144 routes, suppose the new
  vehicle keeps those goals and adds new ones. The exact one-sided McNemar p is then 0.02 for +5 points, 9e-4 for
  +8, 2.6e-4 for +10 and 1.5e-6 for +15 (`smoke_samples.py`).
- **Why B.** The H_full pick is a route that a vehicle-agnostic planner already chose as safe for the HMMWV. On those
  routes the HMMWV reached the goal 95.6 % of the time and the Gator 43.6 %. How often a new vehicle reaches the goal
  on them tells how much room its own planner has. Sample B's rows reuse the stored task rows of `K3/e3/tasks/soil_v4.json`
  (case and route paths, tiers -9/-7), with the vehicle flag changed. Keep the Gator row's episode seed for
  traceability; soil uses the seed for provenance only (`crm_collect.py:393`).
- A stratified sample drawn from all 15,235 ids with all 13 routes per group would also measure how many groups a
  vehicle can pass at all, but it costs about 6.5 simulated minutes per group. Drop it tonight: A's 6 routes per
  group give that number too (Gator 6/24, HMMWV 18/24).

**Arms and tier order in one smoke task file.** The workers drive the lowest tier first and pick rows at random
within a tier (`crm_worker.py:77-79`), so both vehicles advance together.

| tier | rows |
|---|---|
| -4 | 3 HMMWV `bitid__` rows (already-driven `collect_v1` ids through the new switch: must be identical in every array, as `K3/NOTES_E3b2.md:61-66`) |
| -3 | quick look: A tiers 0-1 for Polaris and M113 (48 + 48) + the Gator re-drive of A tiers 0-1 (48) |
| -2 | rest of A: tiers 2-5 for Polaris, M113, Gator (96 x 3) |
| -1 | B for Polaris and M113 (192 x 2), + 16 Gator straight-6 control rows of B |
| 0 | optional: A with the Polaris soil wheels + 0.08 m (144), the counterpart of the Gator's larger-wheel rows. It only applies if the Polaris uses stand-in cylinders on soil. |

- **Jobs.** A new copy of `ag_soil_launch.sh` pointing to G4: `mi3501x:24:4:6` (the 6 free MI350X), plus the
  `-t 00:45:00` driveability job.
- **Reading and stopping early.** Read the quick look after about 30-45 min. If a vehicle is broken, or its soil loop
  costs more than 6x the Gator's per simulated second, write a superset task file with its remaining rows set to
  `"run": false`. That is the recipe of `K3/e3/README.md:205`; a running worker reads its file only once
  (`crm_worker.py:30`), so it applies to new tasks.
- **Two workers per GPU.** Start the second one as an overlap step (`ag_s2_soil_step.sh:24` renames the worker tag).
  On the MI250X node k004-002 the two workers together drove about **1.45x** as many simulated seconds as one worker
  alone (0.22 + 0.215 against 0.29-0.30 simulated s per wall s per GPU; `K4/scratch/S3/workers_soil_v1.txt`). This
  is not measured on MI350X. Test it on one node in the first hour and keep it only if the node's total rate is at
  least 1.2x.

### 2.3 Validity gates

The PLAN 7.7 criteria of the Gator study (`K3/PLAN.md:178-182`), plus the new-vehicle items:

1. At least 95 % of ids complete and passing `crm_qa.check`. That check requires a completion marker, a passed launch
   check, all 17 state fields finite, and no "explosion" or "breakthrough at speed" flag (`crm_qa.py:21-51`).
2. Less than 1 % crashed, non-finite or explosion.
3. Less than 5 % failed launch checks. A failed launch is a non-zero exit, and three in a row retire a GPU worker
   (`crm_worker.py:131`). The quick-look tier exists to catch this.
4. Belly-in-soil flag (lowest hull point more than 0.05 m below the undisturbed surface for more than 1 s) on at most
   10 % of drives, with hull points defined for each vehicle. The Gator used `ag_gator_belly.json`.
   - The body is not coupled to the soil for any vehicle.
   - For the heavy M113, this flag is the main physical-plausibility check.
5. The new-vehicle record is present in every new run and absent from HMMWV runs (as `ag_build_ds.py:345-347` checks
   for the Gator).
6. Bit-identity rows pass: HMMWV `bitid__` rows identical in every array; Gator control rows (section 2.5).

**Items to settle before the smoke. They decide whether the comparison is fair, and they are the vehicle scouts'
work:**
- **The breakthrough stop rule is wheel-based.** It uses `GetSpindlePos` against the tyre radius
  (`crm_collect.py:285-291`), and the M113 has no spindles.
  - Without an equivalent rule (road-wheel hubs against the road-wheel radius), the M113 would be exempt from one of
    the two ways routes fail. It would win partly for that reason: breakthroughs are 55 % of the HMMWV's routes.
  - Declare the equivalent and apply it to every vehicle. Record the largest sinkage for all.
- **The 17-field state includes 4 tyre normal forces and 4 wheel spin rates** (`src/nedm/training/constants.py:32-40`).
  The M113 needs a declared mapping, e.g. front and rear road-wheel loads per side and sprocket spin rates. A
  non-finite placeholder would fail validity gate 1 on every drive.
- **Wheel stand-ins.** Build any soil stand-in cylinder for the Polaris with the Gator's rule: calibrated so the
  settled sinkage matches the HMMWV's (`scripts/ag_soil_calibrate.py`; `ag_vehicle.py:46-48`). Keep follower gains,
  route limits and stop rules unchanged, as for the Gator (`K3/REPORT.md:344-349`). Declare any change made for the
  M113's steering.

### 2.4 Decision rule for "better than the Gator"

Declared before any smoke drive. Applied to each new vehicle; Holm over the two vehicles, one-sided at 0.05.

- **Primary endpoint:** goal not reached, on sample A, paired with the Gator on the same route. The reference is the
  same-launch Gator re-drive; section 2.5 says what happens if it disagrees with the stored runs.
- **"Better" requires all of:**
  1. validity gates 1-6 pass;
  2. failure rate lower by **at least 10 points** (point estimate);
  3. exact McNemar one-sided p at or below its Holm level;
  4. the 95 % interval of the difference from a bootstrap over the 24 groups lies wholly below 0;
  5. the failure rate is also **below the Gator's 80.6 % with wheels 0.08 m larger**, so the gain cannot be explained
     by a more generous wheel model alone.
- **Consistency on sample B (must not contradict):** on straight 6 m/s, the failure rate must be below the Gator's
  83.3 % (point estimate). If A says "better" and B says "worse", report mixed and do not collect without the user.
- **Why 10 points and rule 5.** The Gator's result moved 13 points with its wheel model. The Gator-trained planner
  reached only 67 %, and the 90 % bar needs a large step. A few points' gain would not change the answer to the
  user's question.
- **Reported, not gating:**
  - groups with any goal among their 6 routes (Gator 6/24, HMMWV 18/24);
  - goal reached on the H_full picks of B (Gator 49 %, HMMWV 97 %);
  - end-state mix;
  - simulated length per route;
  - wall cost per simulated second (this sets the collection cost, section 4);
  - belly flag.
- **Go to full collection** only for a vehicle that is "better" **and** whose projected collection fits the declared
  budget (section 4). With two vehicles "better", collect first the one with the larger margin on A (ties: the
  cheaper per id).

### 2.5 Should the Gator be re-driven in the same jobs?

**Yes for sample A (all 144 rows, prefix e.g. `gatorctl__`). No for sample B, except 16 control rows.**

- **Why not strictly needed.** Soil reproduces across GPU types and dates at the outcome level (25/25, section 1.6).
  Also, `crm_worker.py` hands routes to whichever worker claims them first, so "same job" cannot mean "same node" for
  soil anyway.
- **Why it is still worth doing:**
  - it costs +1.41 simulated hours, about 0.5 billed and about 30 min of the 6-GPU pool;
  - the primary test then compares arms driven by the same code, build and launch;
  - it re-proves that the new vehicle switch left the Gator path unchanged, if the Gator is driven through the new
    dispatcher.
- **Rule.** Compare the re-driven Gator rows with the stored runs:
  - If at least 95 % have the same end state and at least 90 % have identical arrays: use the re-drive as the
    reference and report the agreement.
  - If not: the Gator path or the build changed. Stop and find out before comparing, and re-drive all of B's Gator
    rows as well (+2.9 simulated hours).
- **Rigid ground is different.** Any paired rigid comparison must follow the memory rule: every arm in one shard on
  one node (`ag_eval_tasks.py` does this for rigid).

---

## 3. Collection recipe: "the same amount of data as the Gator"

### 3.1 What "the same amount" means

- **The ids.** The same 15,235 soil ids: every f104 soil id with a `collect_v1` run, all 1,200 night-2 groups,
  tiers 0-12. Tier 0 has 1,199 ids, tiers 1-11 have 1,200 each and tier 12 has 836.
  - 9,168 designed routes and 6,067 planner proposals.
  - Splits: 13,821 train, 713 val, 701 test (`K3/NOTES_E3b2.md:93`).
  - Each tier is a similar mix of profiles (about 180 of each designed profile and about 470 proposals per tier;
    `K4/scratch/S3/throughput.py` output).
- **The rows** are the Gator rows of `K3/e3/tasks/soil_v2.json` (`gator__<id>`, `extra ["--vehicle", "gator"]`), with
  only the prefix and the vehicle flag changed. They carry:
  - case and route as absolute `crm_f104_20260916` paths;
  - the episode seed and tier of the `collect_v1` row;
  - split and terrain type.

  The vehicle prefix is required: the id parsers need the trailing `_route_NN` / `_op_NN` (`K3/e3/README.md:214`).
- **Reuse.** The 144 smoke rows of sample A are the same rows. If they were driven by the same frozen collector
  files, copy them in as finished runs instead of re-driving them, as done for the Gator (`K3/NOTES_E3b2.md:101-102`).

### 3.2 Staging (G4), and why new copies of scripts are needed

- **A new cluster root G4.** It needs:
  - `source/`: `src/` and `scripts/` of this worktree, without `__pycache__`;
  - `configs/crm_main.json`: a byte copy of G3's (sha `90cd049e...`);
  - `tasks/`, `soil_v1/`: the output folder;
  - `runtime/`, `e3/submissions.tsv`.
- **Do not write into G3.** It is the previous study's record, and its source tree is read-only.
- **The existing tools know only the HMMWV and the Gator, and some are wired to G3.** Make new copies (e.g. `ov_*`);
  do not edit them:
  - `ag_vehicle.py:39` `VEHICLES = ("hmmwv", "gator")`;
  - `ag_build_ds.py:51` `VEHICLE_PREFIX`, `:272` choices, `:346` vehicle-record check hard-wired to `'gator'`;
  - `ag_eval_tasks.py:54` `VPREFIX`;
  - `ag_soil.sbatch:21` `CRM_ROOT` = G3; `ag_soil_launch.sh:9` `G3=`;
  - `ag_bf_billed.py:13` has no weight for mi3001x, mi3508x or mi3008x. Their ledger weights are 0.125, 1.2 and 1.0
    (10 x `TRESBillingWeights` in `/etc/slurm/slurm.conf`). The script prints but does not count partitions it does
    not know (`:30-39`).
- **The soil job script** must receive the collector and config explicitly, and the collector's name must contain
  `crm_collect` (`ag_soil.sbatch:14-17`). Otherwise the worker does not pass the soil config on, and the collector
  silently runs at its 0.5 ms default.
- **Freeze the new dispatcher before the first job** (`chmod a-w`, sha256 recorded): every episode re-reads it
  (`K3/e3/README.md:220-224`).
- **Submit with `env -u NEDM_VEHICLE`**; never set it.

### 3.3 Launch, watch, stop, relaunch (recipe of `K3/e3/README.md` sections 0, 5, 6)

- **Launch.** `ov_soil_launch.sh <tasks> $G4/soil_v1 <frozen collector> <partition>:<cpus>:<hours>:<n>`, with specs
  like `mi3501x:24:4:6`, `mi2104x:128:6:<n>`, `mi2101x:16:6:<n>`, `mi2508x:128:6:<n>`.
  - It refuses when queued plus new tasks would exceed 50 - 5 (`ag_soil_launch.sh:13-19`).
  - Each task stops claiming new drives 1,000 s before its limit (`ag_soil_launch.sh:25`).
  - Use 4-6 h limits: mi3501x is capped at 4 h, short limits help a job start in gaps before other jobs' reserved
    start times (backfill), and each new task reads the newest task file.
- **Watch** (as `K3/e3/README.md:43-56`):
  - `ag_collect_status.py` / `ag_soil_tier_status.py` copies: per-tier counts;
  - `ls soil_v1/failed | wc -l`;
  - `grep -l "3 consecutive failures\|Traceback" soil_v1/logs/*.out`;
  - `squeue -u harry -h -r | wc -l` at 47 or below.
- **Stop.**
  - Soft: `touch soil_v1/STOP_CLAIMS` stops every job writing to that folder. Workers finish their drive and exit
    (`crm_worker.py:84`). Delete the file before any relaunch.
  - Hard: `scancel`. A killed drive is re-taken after 25 min (`crm_worker.py:23`, `CRM_STALE_S` = 1500 s) and
    restarted from scratch.
- **Relaunch or add rows.**
  - A new task file must be a checked superset of the old one, under a new path, read-only, writing to the same
    output folder.
  - Cancel only pending tasks. Never edit a task file that a job has read (`K3/e3/README.md:100`).
  - Retries: up to 2 attempts per id (`crm_worker.py:24`).

### 3.4 How the Gator's rows were validated and indexed (repeat for each new vehicle)

- **Collection read-out.** `scripts/ag_s1_gator_soil_qa.py` (numpy only, on the login node) checks every row for a
  completion marker, `crm_qa.check`, the launch check, end state and length, and the belly flag from
  `vehicle_extra.npz`. It compares each row with its HMMWV twin by tier, profile and kind
  (`ag_s1_gator_soil_qa.py:1-12, 55-65`).
  - Outputs: `K3/e5/ids_soil/gator_soil_qa_t0-6_summary.json` (stage 1) and
    `K3/e5/ids_bf/gator_soil_qa_all_summary.json` (all tiers), plus the lists of validated ids
    (`gator_soil_validated_*.txt`).
  - Gator result: 15,235 of 15,235 validated, 0 launch failures, belly flag 8.5 % (`K3/LOG.md:112`).
- **Dataset.** `scripts/ag_build_ds.py --world crm --vehicle <v> --tiers 0-6|0-12 --tasks-crm <task file>`.
  - It selects rows by vehicle prefix and task row, and rejects negative tiers (`:330`).
  - It runs the map/arena check, the vehicle-record check, and the completion, launch and QA checks.
  - It then calls the unchanged builder chain (`f104_n2_dataset.py -> n2_reanchor_dataset.py -> ga_build_mixed.py`).
  - Gator result: 15,235 drives -> 55,826 rows (`K3/LOG.md:108`).
  - Then `ag_subset.py` (training subset) and the comparison file of HMMWV twins on exactly the validated ids.
- **Training.** `ci_train --arch gru --cond none --domain-filter crm --ctx geom`, 30 epochs, 5 seeds, 2 lanes per
  MI350X: about 20 min and 0.08 billed per model (`K3/LOG.md:108, 116`). MI210 cannot train (pytorch 2.10 HIP
  error).

### 3.5 Staged collection plan (tier order)

1. **Stage 1 = tiers 0-6** (8,399 ids), the same cut as the Gator's and HMMWV's stage-1 models. This allows a matched
   comparison: new vehicle at tiers 0-6 against the Gator at tiers 0-6 (65.3 % goal reached) and the HMMWV (94.5 %)
   (`K3/REPORT.md:304-309`).
   - Build and train as soon as tier 6 is complete for that vehicle. At dataset time, drop any tier that is only
     partly filled (`K3/e3/README.md:199-201`).
   - The Gator gained only +2.1 points from the rest of its data, and the HMMWV +1.1 points, so stage 1 already
     answers most of the question.
2. **Stage-1 evaluation** (800 pairs, standing start, the planner formulation declared as default). Build a superset
   task file with these drives at a negative tier, so new tasks drive them before tiers 7-12. This is the Gator
   study's soil_v3 pattern (`K3/LOG.md:100-101`).
   - Sample B's straight 6 m/s and H_full drives count toward the 96 pairs they cover.
3. **Tiers 7-12** (6,836 ids), then retrain and re-evaluate on all tiers.
4. **Two vehicles.** Do not interleave them tier by tier: that doubles each vehicle's time to stage 1. Keep the
   `collect_v1` tier values, which the dataset builder needs. Sequence the vehicles through task-file versions:
   - v1: vehicle 1 tiers 0-12, and vehicle 2 rows with `"run": false`;
   - v2: a superset that turns on vehicle 2's tiers 0-6 once vehicle 1's stage 1 is done;
   - v3: the remaining rows.

   The 4-h mi3501x turnover makes new files take effect within hours.

---

## 4. Throughput, cost and wall time

### 4.1 Measured throughput per node type (all soil jobs of 09-25/26)

Sources: 165 worker status files `G3/soil_v1/workers/*.json` joined with `sacct`; `K4/scratch/S3/throughput.py`.

| partition (node) | GPUs per node | simulated s per wall s per worker (setup included) | simulated h per node-hour | billed per node-hour | billed per simulated h |
|---|---|---|---|---|---|
| mi3501x (1 MI350X) | 1 | 0.42-0.48 (mean 0.444) | 0.44 | 0.125 | **0.282** |
| mi2104x (4 MI210) | 4 | 0.312 | 1.25 | 0.4 | 0.321 |
| mi2101x (1 MI210) | 1 | 0.272 | 0.27 | 0.1 | 0.367 |
| mi2508x (8 MI250X GCDs) | 8 | 0.29-0.30 unshared (0.27 incl. the shared node) | 2.17 | 0.8 | 0.369 (about 0.25 with two workers per GCD) |
| devel (MI210) | 1 | 0.29 | | 0.1 | |
| **all soil jobs** (300.7 simulated h, 103.0 billed) | | | | | **0.343** |

- The billing weights are 10 x `TRESBillingWeights` in `/etc/slurm/slurm.conf`. That matches `NOTES_S1:65-66` and
  `ag_bf_billed.py:13`.
- The HMMWV and the Gator cost the same per simulated second on the same GPU (`K3/NOTES_E3b1.md:203`).
- **Observed pool rates.**
  - 09-25 early: 4.5-5.5 simulated h per wall hour (11 MI210 + 4 MI350X; `K3/NOTES_E3a.md:14`).
  - Mid-day: about 15 with 46 MI210 (`K3/PLAN.md:196`).
  - About 17 with 60 GPUs (`K3/LOG.md:89`).
- **The Gator's full set** took 04:47-23:33 on 09-25 (about 19 wall hours), while sharing the pool with the HMMWV
  tiers and the evaluation drives (`K3/LOG.md:103`).

### 4.2 Billed node-hours per 1,000 soil ids

- **Gator:** 9.26 simulated h per 1,000 ids -> **3.18 billed** pooled (2.61 on mi3501x only). All 15,235 Gator ids
  -> about 48 billed. That agrees with its share of the 98.0 billed soil bucket (`K3/REPORT.md:434`).
- **HMMWV:** 6.0 simulated h per 1,000 ids -> 2.06 billed.
- **New vehicle:** multiply by its slowdown per simulated second, and by the ratio of its mean episode length to
  33.35 s. The smoke measures both.

### 4.3 Cost and wall time for one vehicle

From `K4/scratch/S3/cost_table.py`.
- "Gator-length" = 33.35 s per id; "HMMWV-length" = 21.6 s.
- Slowdown = wall cost per simulated second relative to the Gator.
- Billed at the pooled 0.343 (mi3501x only: x 0.82).
- Wall hours at three capacities, in Gator-speed simulated h per wall hour:
  - **L** = the 6 free MI350X: 2.66;
  - **M** = L + 6 mi2104x nodes: 10.2;
  - **H** = L + 15 mi2104x + 1 mi2508x: 23.6.
- Two workers per GPU, if confirmed, divide the wall hours by about 1.45.

| ids | episode length | slowdown | Gator-speed sim h | billed (pooled / mi3501x) | wall h at L / M / H |
|---|---|---|---|---|---|
| **all 15,235** | Gator-length | 1x | 141 | **48.4** / 39.8 | 53 / 13.8 / 6.0 |
| | | **1.5x** | 212 | **72.6** / 59.7 | 80 / 20.8 / 9.0 |
| | | **3x** | 423 | **145.2** / 119.4 | 159 / 41.5 / 17.9 |
| | | 6x | 847 | 290 / 239 | 318 / 83 / 36 |
| | HMMWV-length | 1x | 92 | 31.4 / 25.8 | 34 / 9.0 / 3.9 |
| | | 1.5x | 137 | 47.1 / 38.7 | 52 / 13.5 / 5.8 |
| | | 3x | 275 | 94.2 / 77.4 | 103 / 26.9 / 11.6 |
| **stage 1, tiers 0-6 (8,399)** | Gator-length | 1x / 1.5x / 3x | 78 / 116 / 233 | 26.6 / 39.9 / 79.9 | 29-3.3 / 44-4.9 / 88-9.9 |
| | HMMWV-length | 1x / 1.5x / 3x | 50 / 75 / 151 | 17.3 / 25.9 / 51.8 | 19-2.1 / 28-3.2 / 57-6.4 |
| tiers 0-3 (4,799) | Gator-length | 1x / 1.5x / 3x | 44 / 67 / 133 | 15.2 / 22.8 / 45.7 | 17-1.9 / 25-2.8 / 50-5.6 |

- The M113 is the one at risk of 3x or worse. Its single-pin track has 63-64 shoes per side
  (`vehicle/M113/track_assembly/*SinglePin*.json`). If every shoe is coupled to the soil, that is about 127 soil-coupled
  bodies instead of 4 wheels, and track contact may need a smaller step. The smoke's quick look must measure this
  before any collection plan is fixed.

### 4.4 Smoke-test cost (section 2.2 design)

| part | Gator-speed simulated h: best (HMMWV-length, both 1x) / middle (HMMWV-length, Polaris 1.5x, M113 3x) / worst (Gator-length, 1.5x / 3x) |
|---|---|
| flat-soil driveability job (4 vehicles) | about 0.3 / 0.4 / 0.6 (one 45-min mi3501x node, about 0.1 billed) |
| A: Polaris + M113 + Gator re-drive + 3 bit-identity rows | 3.3 / 5.6 / 7.8 |
| B: Polaris + M113 (192 each) + 16 Gator controls | 1.9 / 4.2 / 7.4 |
| optional: Polaris larger-wheel rows | 0.9 / 1.4 / 2.1 |
| **total** | **6.1 / 11.2 / 17.3 h -> 2.1 / 3.8 / 5.9 billed** (x 0.82 on mi3501x only) |
| wall time on the 6 MI350X: A alone / everything | 1.2 / 2.1 / 2.9 h; 2.3 / 4.2 / 6.5 h (about /1.45 with two workers per GPU) |

### 4.5 Evaluation and training (per vehicle and stage)

- **Evaluation.** 800 pairs x 3 arms (own planner, straight 6 m/s, H_full transfer) = 2,400 drives. At about 22 s
  each that is about 15 Gator-speed simulated hours x slowdown, i.e. **about 5 billed at 1x and 15 at 3x**. For
  comparison, the Gator's stage-2 evaluation was 2,318 drives, 15.4 simulated h and 5.04 billed, in 1 h 17 min on 22
  tasks (`K3/LOG.md:122`).
- **Training:** below 0.2 billed per stage.
- **Offline scoring and picks:** local.

### 4.6 The cluster tonight and what fits

State at 22:40 CDT on 09-27 (`sinfo`, `squeue`, `sshare`, `slurm_balance2.py`):

- **mi3501x:** 5 idle, 1 frees at 23:15, 1 completing, 1 drained; nobody waiting. **The only free GPUs.**
- **mi2104x:** 20/20 allocated (1 down). Frees:
  - 23:14, 23:40, 00:36;
  - **5 nodes 01:44-02:12**;
  - 03:19 and 04:38;
  - **8 nodes 05:26-06:30**.

  3 other tasks wait there.
- **mi2101x:** 24/24 allocated. 13 tasks end 23:05-23:12, but about 30 other users' tasks wait: jindongwang 6 x 4 h,
  d287zhan 15, mohammedalser 2 x 12 h, and others.
- **mi2508x:** 7 allocated, 1 held for a waiting 30 h job, 2 down; 3 other tasks wait.
- **mi3001x** (MI300X; soil code compiled for it but never run there): 6-7 allocated, 17 waiting.
- **Our priority is at the bottom.**
  - Priority = 2500 x fair-share + up to 500 for age (full after 3 days) + 5 for the partition.
  - harry's fair-share factor is 0.0044, i.e. about 11 points. Users waiting now have 76-653.
  - We get idle nodes and backfill gaps only.
- **Budget.** Account ledger 758.2 of 1,500 node-hours used (harry 276.9; four other users share the account). Nothing
  billed since 09-27 00:00. The previous session ended at 113.3 billed (`scripts/ag_bf_billed.py`).
- **Projection** if the smoke starts at about 23:15 and collection at about 01:30 (about +-50 %):
  - 23:00-02:00 at about 3 simulated h per wall hour;
  - 02:00-05:30 at about 9;
  - 05:30-12:00 at about 20;
  - cumulative collection capacity about 80 Gator-speed simulated hours by 08:00 and about 160 by 12:00 on 09-28.
- **What fits by about noon:**
  - one vehicle's full set if it runs at 1x-1.5x with HMMWV-length drives (92-137 h), or at 1x with Gator-length
    drives (141 h);
  - or both vehicles' stage 1, if both are near 1x;
  - not the M113's full set at 3x (275-423 h, about 1-2 more days, 94-145 billed).

---

## 5. Cluster risks

1. **Capacity, not budget, limits the night.** Mitigations:
   - Keep 6 mi3501x tasks rolling. They are capped at 4 h; queue successors with `--dependency=afterany:<id>`, which
     costs queue slots.
   - Submit mi2104x tasks early, with 4-6 h limits, so they start in gaps when nodes free around 02:00 and 05:30.
   - Consider two workers per GPU (section 2.2).
   - Expect about 5 simulated h per wall hour when others' jobs return (`arena-gator-state` memory).
2. **At most 50 queued or running tasks per user** (`K3/e3/README.md:47`; the launcher keeps 5 free,
   `ag_soil_launch.sh:13`). Chained 4-h tasks and devel jobs eat slots.
3. **Running workers read their task file once** (`crm_worker.py:30`). New rows, such as evaluation drives or the
   second vehicle, reach only tasks started later. Plan the task-file versions (section 3.5), or use overlap steps
   inside running allocations (`ag_s2_soil_step.sh`).
4. **A vehicle-specific crash retires GPUs.** Three failures in a row stop a worker (`crm_worker.py:131`), and each
   id gets 2 attempts. The quick-look tier and the `run: false` superset are the defence.
5. **Reproducibility.**
   - Soil: consistent at the outcome level across GPU types and dates (25/25 same outcome, 22/25 identical arrays).
     Keep the bit-identity and Gator control rows in every new launch that uses a new dispatcher or build.
   - Rigid: deterministic per node only (memory note). Paired rigid arms go in one shard on one node.
6. **The M113 may cost well over 3x per simulated second** (about 127 track shoes, a possibly smaller step). Before
   the smoke, declare a cost gate: above about 6x, no full M113 collection tonight.
7. **Fairness traps that could bias the comparison** (section 2.3): the wheel-based breakthrough rule; the tyre fields
   of the 17-field state; wheel stand-in sizes; hull points for the belly flag; follower gains tuned for the HMMWV.
8. **Training GPUs.** MI210 cannot train (pytorch 2.10 HIP error). Training needs MI350X or MI250X. On a full cluster,
   run it as an overlap step inside our own soil job, as done for the Gator (`K3/NOTES_S1.md:58, 226`).
9. **Session loss.** The last study lost its session twice while cluster jobs kept running. Log every job id and
   decision in `K4/LOG.md`. Keep login-node helper loops restartable: they die if login1 reboots
   (`K3/e3/README.md:40`).
10. **Ledger blind spots.** `ag_bf_billed.py` lacks mi3001x, mi3508x and mi3008x. The account ledger updates about
    hourly and is shared with 4 other users.
11. **MI300X (mi3001x) is untested for soil,** although compiled for it. Run a drift and bit-identity row there before
    trusting it with training data.

## 6. Recommendations

1. **Before the smoke, declare in the plan:**
   - samples A and B (files in `K4/scratch/S3/`);
   - the decision rule of section 2.4;
   - the validity gates of section 2.3;
   - the M113 breakthrough equivalent and its state-field mapping;
   - the Polaris wheel stand-in rule;
   - a cost gate (e.g. no full collection above 6x per simulated second, or above about 80 billed projected per
     vehicle);
   - a session soft cap of about 150 billed, as last time.
2. **Stage 0.** Local rigid checks; one tiny local soil settle-and-drive per vehicle under the flock; the full
   flat-soil driveability job on one mi3501x node for all four vehicles (45 min).
3. **Stage 1.** One smoke task file with tiers -4 to 0 (section 2.2), on the 6 free MI350X. Re-drive the Gator on
   sample A in the same launch. Read the quick look at about +45 min and the decision after sample A (about 1.2-2.9
   h). Let B finish while collection starts.
4. **Collect only a vehicle that passes.** Build it as the superset of the smoke file (tiers 0-12 at their
   `collect_v1` values) in the same output folder, so the 144 A rows are not driven twice. Start with tiers 0-6. Put
   stage-1 evaluation drives ahead of tiers 7-12. Train at tiers 0-6 (matched to the Gator's stage 1) and evaluate on
   the 800 pairs with the default planner formulation; the 90 % bar is judged there. Then finish tiers 7-12.
5. **If both pass,** sequence the vehicles through task-file versions (section 3.5). Expect only stage 1 of the second
   vehicle by noon on 09-28.
6. **Build new copies of the tools** (`ov_*`) for G4 and the new vehicle names (section 3.2). Add mi3001x, mi3508x
   and mi3008x to the billing copy.

## Appendix: files used

- **Previous study:**
  - reports and notes: `K3/REPORT.md`, `K3/PLAN.md`, `K3/LOG.md`, `K3/e3/README.md`, `K3/NOTES_E3a.md`,
    `NOTES_E3b1.md`, `NOTES_E3b2.md`, `NOTES_S1.md`;
  - pilot and outcomes: `K3/e3/pilot_gator_eval.{txt,json}`, `K3/e3/tasks/pilot_gator.meta.json`,
    `K3/e3/tasks/soil_v2.json`, `soil_v4.json`, `K3/e5/ids_bf/gator_soil_qa_all_summary.json`,
    `K3/e6/index/soil_eval_bfull.json`, `K3/e3/drift_soil.json`.
- **Code:** `scripts/crm_worker.py`, `crm_collect.py`, `crm_qa.py`, `ag_soil.sbatch`, `ag_soil_launch.sh`,
  `ag_pilot.sbatch`, `ag_crm_collect.py`, `ag_vehicle.py`, `ag_build_ds.py`, `ag_eval_tasks.py`,
  `ag_s1_gator_soil_qa.py`, `ag_s2_soil_step.sh`, `ag_bf_billed.py`, `ag_drift_check.py` (run read-only into scratch),
  `src/nedm/training/constants.py`.
- **Cluster (read-only):** `G3/soil_v1/workers/*.json`, `G3/e3/drift_more/runs` (index files pulled), `sacct` since
  09-25, `/etc/slurm/slurm.conf`, `sshare`, `slurm_balance2.py` and `/share/accounting/allocation_usage.json`, and the
  PyChrono import check in `$NRD_ROOT/chrono-build-fsi`.
