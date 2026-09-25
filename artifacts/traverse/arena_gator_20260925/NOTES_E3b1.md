# NOTES E3b1: Gator cluster pilots on f104, soil and rigid (2026-09-25, 02:25-04:40)

Module E3b1 of PLAN.md (sections 3 and 7.7). K3 = this folder, G3 = `/work1/dannegrut/harry/experiments/arena_gator_20260925`.
Every number below comes from `scripts/ag_pilot_eval.py` on the local copy of `G3/pilot_gator` (`e3/pilot_gator/`,
summary `e3/pilot_gator_eval.json`). Nothing in the earlier artefact folders or cluster roots was written; no existing
script was edited.

## 0. Verdicts

**Gator soil: GO-with-caveats.** Every declared collection check passes: 144/144 drives validated, no crashed or
NaN drive, no launch-check failure, the body-in-soil flag on 4.2 % of drives (limit 10 %), and the wheel-radius
sensitivity is 13.2 points (limit 15). The caveats change how any Gator soil result can be read:
1. **The labels are close to saturated.** The Gator fails 93.8 % of the pilot routes, against 69.4 % for the HMMWV on
   the identical ids (36 routes fail only for the Gator, 1 only for the HMMWV). Constant 2 m/s fails 100 %, and the
   straight 6 m/s route (route_02) reaches the goal on 0 of 8. Only one of the four designed speed profiles lies inside
   the 10-90 % band of REVIEW_R1 3(b). Only 3 of 24 groups have any designed route that reaches the goal (HMMWV 16).
   So a Gator soil planner would have very little to choose between, G against H would be compared near the floor,
   and "works" criterion (1) (beat straight 6 m/s) is probably not applicable (R1 3(d): straight 6 m/s at or below 5 %).
2. **The result is sensitive to the wheel model, just under the declared threshold.** With both soil cylinders 0.08 m
   larger, the failure rate drops to 80.6 %: 20 routes turn into successes and 1 the other way (exact McNemar
   p = 2e-5). Goals go from 9 to 28 of 144, and at 4 m/s the failure rate goes from 94 % to 56 %. At the larger radius
   the labels would be inside the band on all four profiles.
3. **Cost: 1.53 times the HMMWV's simulated hours per id.** The mean episode is 35.3 s against 23.0 s, because 74 %
   of Gator drives run to the 34 s blockage stop. The 15,235 soil ids would be about 149 simulated hours (HMMWV
   91.5). One Gator tier of 1,200 rows is about 11.8 simulated hours, against 7.3 for a g203 + g228 tier. With the
   PLAN 7.8 interleave (1,200 : 606 : 606), the Gator would take about 62 % of the soil GPU time.

The Gator is not broken: it bogs. The rear wheels dig to about 0.15 m and spin, and the drive ends on the
blockage rule. The HMMWV on the same routes more often digs through the soil layer (breakthrough). The PLAN 7.8
condition for putting Gator rows in the soil file (no failed or rejected launches) is met. Whether the full Gator soil
set is worth 1.5 times the HMMWV cost for about 6 % successes is a decision for the plan owner. The pilot already
answers the route-by-route comparison.

**Gator rigid: GO-with-caveats.** 288/288 drives validated, with no crash, NaN, launch-check or ground-height failure.
The Gator fails 3.5 % of the designed routes, against 13.9 % for the HMMWV (8 against 38 discordant routes).
- An HMMWV re-drive of the same 288 routes on the same nodes reproduced the collection's HMMWV runs exactly
  (288/288 identical state arrays), so the reference is solid.
- Unsafe: 4.5 % against 26.4 %; without the backward-motion clause, 3.5 % against 13.9 %.
Caveats:
1. **The labels are sparse.** Only constant 2 m/s (12.5 % failure) lies inside the 10-90 % band. The on-policy
   routes, which make up 40 % of the rigid pool, were not piloted. A speed-free rigid Gator planner will be near the
   ceiling; the fixed 2 m/s read-out is where differences can show.
2. **The chassis touches the ground on 32 of 288 routes** (HMMWV 9; only 3 routes shared). These are mostly
   single-frame scrapes on crater entries and exits. Two of the 10 Gator failures are high-centred stalls on a crater
   lip.
3. **The Gator creeps backwards during the settle,** because it brakes only its rear wheels. The anchor speed is
   below -0.2 m/s on 60 of 288 routes (lowest -0.40; HMMWV -0.08). This enters the models' start-state input, but it
   did not add spurious backward-motion labels: 3 Gator routes against 36 HMMWV routes are unsafe only through that
   clause.

## 1. What was run

**Collector files (task 1).**
- `G3/source/scripts/ag_vehicle.py` was the stale 01:30 copy (`3e828142...`, VERIFY_E2 item 7). Before replacing it I
  checked that nothing running imported it: every soil_v1 and drift job runs the unchanged `crm_collect.py` (job logs),
  the rigid job 436075 runs `gen_collect_ext.py` and was still waiting. It was replaced by the verified file
  (`072716ee...`) at 02:30. The other three E2 files on the cluster were already identical to the verified ones.
- Frozen (`chmod a-w`): `ag_vehicle.py`, `ag_crm_collect.py`, `ag_gen_collect_ext.py`, `ag_gator_belly.json`, and the
  two new pilot files `ag_rigid_runner.py`, `ag_pilot.sbatch`. Hashes of every file the Gator path loads are in
  `e3/gator_collector_sha256.txt` (read on the login node; local copies identical). They import nothing new besides
  `ag_vehicle.py` and `ag_gator_belly.json`.
- Every pilot episode records `ag_vehicle.py` = `072716ee7154...` in its vehicle block (checked on all Gator runs).
- A 3 s local soil smoke with the enlarged wheels (radius + 0.08 m) confirmed the override reaches the soil build
  (front 0.27575 m, rear 0.3075 m recorded) before anything was submitted.

**Rows (task 2)** (`scripts/ag_pilot_tasks.py`, `e3/tasks/pilot_gator.meta.json`).
- Group rule, declared in the script before any pilot row ran: in each of the 6 collection strata of the f104 night-2
  cases, the 4 training-split groups with the lowest md5 of the group id. 24 groups:
  crater cross-slope 0426 0894 0594 0427, crater entry 0188 0259 0295 0404, hill cross-slope 1060 1144 0926 0916,
  hill entry 0829 0362 0685 0868, long traverse 0213 0177 0286 0621, roughness 1103 0563 0575 1187 (group numbers of
  `f104_v2_group_NNNN`, lowest md5 first).
- Soil, 144 rows `gator__<id>`: the HMMWV `collect_v1` rows of tiers 0-5 of those groups (its `tasks_train.json`,
  sha `b4e2fdba...` = `collect_v1_inputs.sha256`), with their own case, route and episode seed. 86 designed routes,
  58 planner-proposal routes. Wheel radius sensitivity, 144 rows `gatorR8__<id>`: the same rows with both soil
  cylinders 0.08 m larger (front 0.27575, rear 0.3075 m), put in tiers 10-15 so every calibrated row ran first.
  File `pilot_gator_soil.json` sha `172ad6f8...`.
- Rigid, 288 rows `gator__<group>_route_NN`: the 12 designed routes of each group, the case and route files behind
  the HMMWV night-2 runs (`fdm_f104_50h_20260909/cases_night2_v1`), 2 shards by group hash (all routes of a group on
  one node). File `pilot_gator_rigid.json` sha `f41236d3...`.
- The HMMWV rows of the same ids were not re-collected on soil: they are the `collect_v1` runs (local copy
  `crm_f104_v1/collect_v1/runs`). On rigid the HMMWV reference is `production_v3`; in addition the 288 HMMWV routes
  were re-driven on the same nodes as the Gator (below), at no extra billing.

**Jobs (task 3)** (all in `G3/e3/submissions.tsv`).

| job | where | what |
|---|---|---|
| 436122 (array 0-1; task 0 = job 436124) | 2 x mi3501x (one MI350X + 24 cores each), 4 h limit | `scripts/ag_pilot.sbatch`: one soil worker on the GPU (unchanged `crm_worker.py`, collector `ag_crm_collect.py`, `configs/crm_main.json`) and 16 rigid Gator workers on the CPU cores (`ag_rigid_runner.py` = `gen_runner_g.py` that keeps the rich telemetry, collector `ag_gen_collect_ext.py`, Gator runtime fingerprint) |
| 436124.0, 436122.0 | job steps inside the two pilot allocations (`srun --overlap`) | HMMWV re-drive of the 288 rigid routes (`scripts/ag_pilot_hmmwv_rigid.sh`: unchanged `gen_collect_ext.py`, HMMWV fingerprint, rich telemetry kept), same shard = same node as the Gator rows, output `G3/pilot_gator/rigid_hmmwv` |
| 436123, 436127, 436130, 436131, 436146, 436165, 436166, 436187 (array 0-1) | devel (one MI210, 30 min each, claims stop after 20 min; 10 tasks, some chained with `afterany`) | extra soil workers on the same soil file: Gator timing on the MI210 and more throughput |

- The third idle mi3501x node was left free; another module's training job (436135) took it at about 02:50.
- Queue: at most 44 of 50 array tasks while these ran. `NEDM_VEHICLE` is unset in the job script; every row carries
  `--vehicle gator`.
- Output folder `G3/pilot_gator/{soil,rigid,rigid_hmmwv,logs}`.

## 2. The PLAN 7.7 criteria

| criterion (PLAN 7.7 / REVIEW_R1 3) | limit | Gator soil, calibrated wheels | Gator soil, radius + 0.08 m | Gator rigid |
|---|---|---|---|---|
| ids validated | >= 95 % | **100 %** (144/144) | 100 % (144/144) | **100 %** (288/288) |
| crashed / NaN / exploded | < 1 % | **0** (0 worker failures, 0 `collection_failure.json`, 0 non-finite, 0 quality flags) | 0 | **0** (0 runner failures, 0 non-finite) |
| launch-check failures | < 5 % | **0 %** | 0 % | **0 %** (native-height check 0 failures too) |
| body in soil (lowest hull point > 0.05 m under the surface for > 1 s in a row) | <= 10 % of drives | **4.2 %** (6/144; 5.6 % counting > 1 s in total) | 0 % | n/a (rigid: 31 routes with the hull below the surface for 1-547 frames = the ground contacts below) |
| wheel-radius sensitivity (overall failure change) | > 15 points = depends on the wheel model | **-13.2 points** (93.8 -> 80.6 %): under the limit, but 20 routes flip to goal and 1 the other way (McNemar p 2e-5) | | n/a |
| informative labels (R1 3(b): overall 10-90 %, and 3 of 4 designed profiles inside) | | overall 93.8 %: **no**; 1 of 4 profiles inside | 80.6 %: yes, 4 of 4 | overall 3.5 %: **no**; 1 of 4 profiles inside (HMMWV: 13.9 %, 2 of 4) |
| unsafe with / without the backward-motion clause | report | | | Gator 4.5 / 3.5 %, HMMWV 26.4 / 13.9 % |

Soil drives were validated with `crm_qa.check`. Rigid drives count as validated when they have the completion marker,
pass the launch and native-height checks, and have finite states. Every Gator run carries the Gator vehicle block
(`ag_vehicle.py` `072716ee...`) and the production physics step (0.001 s). No HMMWV re-drive has one.

## 3. Soil: Gator against the HMMWV on the identical 144 ids

Failure = goal not reached. The HMMWV rows are the `collect_v1` runs of the same ids. Soil runs are bit-identical
across GPU types, and tonight's tree reproduced `collect_v1` in 9 of 10 drift ids (NOTES_E3a section 6).

| routes | n | Gator calibrated | Gator + 0.08 m | HMMWV | Gator-only / HMMWV-only failures (calibrated) | exact McNemar p |
|---|---|---|---|---|---|---|
| all | 144 | **93.8 %** | 80.6 % | **69.4 %** | 36 / 1 | 5.5e-10 |
| constant 2 m/s | 18 | 100 % | 83.3 % | 100 % | 0 / 0 | 1 |
| constant 4 m/s | 16 | 93.8 % | 56.2 % | 62.5 % | 5 / 0 | 0.06 |
| constant 6 m/s | 28 | 89.3 % | 78.6 % | 28.6 % | 17 / 0 | 1.5e-5 |
| smooth 2-6-2 | 24 | 95.8 % | 83.3 % | 58.3 % | 9 / 0 | 0.004 |
| planner proposals | 58 | 93.1 % | 86.2 % | 86.2 % | 5 / 1 | 0.22 |

By stratum, calibrated Gator against the HMMWV:
- crater cross-slope 83 / 38 %;
- crater entry 100 / 96 %;
- hill cross-slope 96 / 83 %;
- hill entry 100 / 63 %;
- long traverse 88 / 54 %;
- roughness 96 / 83 %.

Groups with any goal among their 6 pilot routes: Gator 6/24 (10/24 at + 0.08 m), HMMWV 18/24.

**How the Gator fails.**
- Status mix: 134 blockage stops, 1 breakthrough, 9 goals. The HMMWV on the same ids: 86 breakthroughs, 14 blockages,
  44 goals.
- Median of each drive's largest wheel sinkage: 0.16 m (HMMWV 0.32 m, which is past its 0.30 m breakthrough rule).
  The Gator stops before it digs through.
- Median p95 slip ratio: 34 (HMMWV 96).
- Failed Gator drives get a median of 23.0 m towards the goal.
- On the 8 joint successes the Gator takes 1.08-1.67 times the HMMWV's time (median 1.24).

**Body in soil.** The median lowest hull point over a drive is -0.015 m, and the start clearance is 0.167 m.
- 6 drives are flagged, with 1.7-4.05 s in a row below -0.05 m. All of them are failures that were already bogged.
  The deepest, -0.16 m, is the one breakthrough. Two more pass only the cumulative version (0.65 and 0.85 s in a row).
- Counting the flagged drives as failures leaves the failure rate unchanged (93.8 %).
- At + 0.08 m the wheels ride higher (median largest sinkage 0.075 m, start clearance 0.204 m) and no drive is
  flagged.

**Simulated hours.** 1.41 h for the calibrated Gator against 0.92 h for the HMMWV on the same ids (1.53 times). The
whole soil pilot was 2.76 simulated hours.

## 4. Rigid: Gator against the HMMWV on the identical 288 designed routes

The HMMWV reference is `production_v3`. It was also re-driven tonight in the same allocation and on the same node as
each Gator group (unchanged `gen_collect_ext.py`, HMMWV fingerprint), and all 288 re-drives are identical to
`production_v3`: same status, identical state arrays. So the comparison below is the same whichever HMMWV copy is
used, and the cross-node caveat does not apply.

| profile | n | Gator fail | HMMWV fail | Gator unsafe | HMMWV unsafe |
|---|---|---|---|---|---|
| all | 288 | **3.5 %** | **13.9 %** | 4.5 % | 26.4 % |
| constant 2 m/s | 72 | 12.5 % | 34.7 % | 15.3 % | 63.9 % |
| constant 4 m/s | 72 | 0.0 % | 12.5 % | 1.4 % | 25.0 % |
| constant 6 m/s | 72 | 1.4 % | 2.8 % | 1.4 % | 2.8 % |
| smooth 2-6-2 | 72 | 0.0 % | 5.6 % | 0.0 % | 13.9 % |

- Paired failures: 8 Gator-only, 38 HMMWV-only, 2 both (McNemar p = 9e-6). Paired unsafe: 1 against 64.
- Statuses:
  - Gator: 278 goals, 5 blockage stops, 5 timeouts.
  - HMMWV: 248 goals, 32 blockage stops, 3 exits from the arena, 4 timeouts, 1 rollover.
- 9 of the 10 Gator failures are at constant 2 m/s (slow climbs that time out or stall). Two are high-centred on a
  crater lip (`0259_route_02`, `0295_route_00`), with the hull below the surface for 547 and 420 frames.
- **Ground contact.**
  - The chassis touches the ground on 32 routes: 16 in crater-entry groups; 20 of the 32 routes are in 4 groups
    (0259, 0295, 0685, 1103). 29 of the 32 contacts leave the hull below the surface for 0-6 frames, with peaks up to 228 kN (single contact spikes).
  - The HMMWV touches the ground on 9 routes (peak 359 kN), only 3 of them among the Gator's 32.
  - Lowest belly point over all routes: -0.056 m; median of each route's lowest point: +0.068 m.
- **Tilt.** Largest roll 40° (HMMWV 57°), largest pitch 33° (HMMWV 49°).
- **Anchor creep.** Median anchor speed -0.04 m/s, lowest -0.40 m/s (HMMWV -0.015 and -0.08). It is below -0.2 m/s on
  60 routes, all within the 1 m/s launch limit.
- Only 3 Gator routes are unsafe through the backward-motion clause alone, against 36 HMMWV routes. The S2 / R1 3(e)
  worry, false "slid backwards" labels from the rear-only brakes, did not show on these routes.
- **Simulated hours:** 1.42 h for the Gator against 1.92 h for the HMMWV. The median episode is 13.4 s against 15.2 s;
  on joint goals the Gator takes 0.97 times the HMMWV's time.

## 5. Speed and cost

| what | where | wall seconds per simulated second |
|---|---|---|
| Gator soil, episode loop (median over drives) | MI350X (mi3501x), 192 drives | **1.92** |
| | MI210 (devel), 96 drives | **3.09** |
| Gator soil, whole worker including setup and file writes | MI350X, 2 workers, 6,683 simulated s | **2.10** (0.48 simulated s per wall s) |
| | MI210, 9 workers, 3,251 simulated s | **3.43** (0.29) |
| HMMWV soil on the same ids (`collect_v1`, mixed GPU types), loop | | 3.05 |
| Gator rigid, per episode (16 single-thread workers on the 24-core EPYC 7V13 node, beside one soil worker) | CPU | **11.8 s + 0.88 s per simulated s**; median 1.76 |
| HMMWV rigid, same routes, same nodes, same load | CPU | 10.7 s + 1.43 s per simulated s; median 2.11 |
| HMMWV rigid, `production_v3` (126 processes per mi2104x node) | CPU | 28.5 s + 3.0 s per simulated s |

- On MI350X the soil loop runs at the same speed per simulated second for the Gator and the HMMWV (E2 found the same
  on the 5090). The Gator costs more per id only because its drives are longer.
- **Rigid work beside soil does not slow soil.** Soil episodes that overlapped the rigid workers ran at 1.996 and 1.917
  wall s per simulated s on the two nodes, against 1.936 and 1.904 for those that did not (+3.1 % and +0.7 %). That
  is well under the 10 % rule of PLAN 7.8, so rigid rows can run on the idle cores of soil allocations.
- **Production estimates from these rates.**
  - Gator soil, all 15,235 ids: about 149 simulated hours. On one MI350X that is about 310 GPU-hours; at tonight's
    ~5 simulated hours per wall hour, about 30 wall hours.
  - Gator rigid pool, 24,000 routes at about 20 simulated s each: about 200 CPU-hours at the pilot's per-worker rate.
    Beside soil that is about 13 mi3501x node-hours at 16 workers per node. At the HMMWV `production_v3` density (126
    per mi2104x node) it is about 3-5 node-hours.
- **Pilot cost.**
  - 2 mi3501x nodes x 1.95 h = 3.9 node-hours.
  - 10 devel jobs, about 20 min each = 3.5 node-hours.
  - The HMMWV rigid re-drive ran inside the mi3501x allocations.
  - About 0.85 billed in total, estimated with the partition billing weights (0.125 and 0.1 billed per node-hour,
    the rate E3a used). The ledger updates hourly and does not separate this module's jobs.

## 6. Deviations and notes

- **Timing on two GPU types only.** No MI300X or 8-GPU node was free. The MI210 timing comes from devel jobs, which
  stop taking new drives after 20 min.
- **Extra step, not requested:** the HMMWV rigid re-drive, done so that each group's two vehicles ran on one node.
  It cost nothing extra and settled the reference. Its runs live in `G3/pilot_gator/rigid_hmmwv`, ids `hmmwv__<id>`.
- **Rich telemetry.** The rigid runner copy keeps it (for the chassis contact forces). Locally,
  `rich_telemetry.npz` was left out of the sync; `rich_intervals.npz`, which holds the contact maxima, was copied.
- **Pilot rows were not appended to `soil_v1`.** The E3a relaunch recipe (`e3/README.md` section 6) is unchanged.
  The pilot ids (`gator__*`, `gatorR8__*`) live only in `G3/pilot_gator/soil`. If production Gator rows reuse the
  `gator__` prefix in `soil_v1`, the 144 pilot runs could be copied in as finished runs instead of re-driven. They
  were produced by the same frozen collector files.
- **A caveat for the rigid pilot.** It drove designed routes only. The Gator's failure rate on the 8 on-policy routes
  per group is unknown.
- **The sensitivity test has little room upwards.** At 93.8 % failure, the larger radius could raise the failure rate
  by at most 6 points, so the test measures mainly the downward change.

## 7. Files

- **Scripts** (new, `scripts/`):
  - `ag_pilot_tasks.py`: builds the task files.
  - `ag_pilot.sbatch`: soil and rigid in one job.
  - `ag_rigid_runner.py`: `gen_runner_g.py` that keeps the rich telemetry.
  - `ag_pilot_hmmwv_rigid.sh`: the same-node HMMWV re-drive.
  - `ag_pilot_eval.py`: all numbers in this note.
- **Task files** (`e3/tasks/`):
  - `pilot_gator_soil.json` (`172ad6f8...`), `pilot_gator_rigid.json` (`f41236d3...`), `pilot_hmmwv_rigid.json`
    (`0a122ceb...`), `pilot_gator.meta.json`;
  - `ref/crm_f104_tasks_train.json`, a copy of the collect_v1 task file.
- **Frozen collector hashes:** `e3/gator_collector_sha256.txt`.
- **Results:**
  - `e3/pilot_gator_eval.json`: summary plus one row per drive;
  - `e3/pilot_gator_eval.txt`: text report;
  - `e3/pilot_gator/`: local copy of `G3/pilot_gator` without `rich_telemetry.npz`.
- **Cluster:** `G3/pilot_gator/{soil,rigid,rigid_hmmwv,logs}`. The job ids are in `G3/e3/submissions.tsv` and in
  LOG.md.

