# CRITIC: completeness check of scouts S1-S5 (Polaris and M113 on f104 soil), 2026-09-27 23:30-23:50 CDT

Read-only. I read the five reports in full, spot-checked the most decision-relevant claims in code, scratch outputs
and the cluster (read-only `sinfo`/`squeue`/`sed`), and wrote only this file. No Chrono run, no cluster job, no
`NEDM_VEHICLE`.

Names used below:
- **soil / CRM**: Chrono's particle soil (0.08 m spacing, 0.24 m deep, 1 ms step in production).
- **sample A**: the Gator pilot's 144 soil routes (24 groups x tiers 0-5), where the Gator failed 93.8 %.
- **launch check**: the collector's test after the 0.8 s braked settle (speed, roll/pitch, chassis height 0-1.2 m).
- **re-framed Polaris**: S1's copy of the Polaris JSON with the chassis reference moved from the front axle to
  mid-wheelbase.
- **stock / corrected / open-diff driveline**: Chrono's shipped Polaris driveline with the power bug; S5's fix (final
  ratio 1.0, gearbox ratios x 0.25, Chrono's limited-slip split kept); Chrono's `Polaris_4WD.json` (open
  differentials, power-correct; S1 calls it "shafts").

## 0. Bottom line

1. **The Polaris is ready for a cluster smoke once one design choice is made: the driveline.** The scouts disagree on
   it. S1 recommends the stock driveline as the primary; S5 says stock results would be "an artefact". The only f104
   soil evidence (S1, 4/4 goals) uses the stock driveline. Declare one primary and drive both on sample A.
2. **The M113 has two blockers that no report puts together:**
   - **gearing:** its drive push is at most 0.35 of its weight, against about 14 % internal resistance (S2 §1.2,
     S5 §3.3);
   - **track-pad grip on soil:** it slides backwards when braked on 10° (S2 §3; S5's own raw data show the same).

   Add 7.7x cost, a 0.5 ms soil step, and a reference point 2 m ahead of the centre. A tonight M113-vs-Gator result
   would measure Chrono's M113 set-up, not tracks. All three scouts who looked (S2, S3, S5) agree: no full M113
   collection tonight.
3. **Two of S3's gates would trip by chance.**
   - S3's own drift data give 88 % identical arrays (22/25), and the drift check's own rule failed
     (`scratch/S3/drift_more_check.json`: `"pass": false`).
   - A "≥ 90 % identical arrays" rule (S3 §2.5) and "bitid rows identical in every array" (S3 gate 6) against stored
     runs are therefore likely to stop the night on noise.
   - Use outcome-level agreement against stored runs, and do array identity only same-node and back-to-back.
4. **S5's fix for the Polaris launch failure is not possible under the frozen-file rule. S1's fix is.**
   - The launch check is a class defined inside `crm_collect.main()` (`scripts/crm_collect.py:380,412-430`), so no
     wrapper can change it.
   - S1's re-framed JSON plus a +0.40 m spawn passes it (+0.385 m on f104 soil).
5. **The timeline in S3 is already stale.**
   - It assumed the smoke would start at 23:15. At 23:35 none of the new wrapper code exists.
   - Realistic: smoke at about 01:00-01:30, the sample-A decision at about 03:00-04:00, the Polaris stage-1 planner
     result (tiers 0-6) late morning, and all 15,235 ids by afternoon on 09-28.
   - Cluster capacity is better than S3 saw (section 2, last row).

## 1. Contradictions between the reports, and how to resolve them

| # | Topic | Says | Says | Resolution |
|---|---|---|---|---|
| C1 | Polaris driveline for the primary arm | S1 §6.2.3: stock as primary (plausible below 6 m/s, ~50 kW at the wheels at 6 m/s), open-diff as the sensitivity arm | S5 §0.4, §5.1: stock creates 16x power, "any result an artefact"; use the corrected one | Both are partly right; see 1.1. Declare before the smoke and drive **stock and corrected** on sample A (+~1 simulated h each). Open-diff is the third arm both scouts want. |
| C2 | Polaris launch check and spawn | S1: re-frame the reference to mid-wheelbase, spawn +0.40 m (verified, +0.385 m on f104) | S5 §5.4: spawn +0.10 m and "change the soil launch gate to a vehicle-specific reference" | S5's gate change needs an edit of a frozen collector (`crm_collect.py:412-430` is inside `main()`). **Use S1's re-framing.** It also puts the goal and exit rules on the body centre, as for the HMMWV. |
| C3 | Polaris soil-wheel radius | S1: 0.25 m (stock - 0.08), calibrated on the same local build as the Gator calibration | S5: borrowed r - 0.09 = 0.24 m, "not calibrated", try r - 0.07 / r - 0.09 | Adopt S1's 0.25 m. S5's soil numbers come from conda pychrono 10.0.0 (`scratch/S5/crm_smoke.py:15-22,91,95`, an old-API fallback), a different Chrono from the calibration build (local `a92c6f72`) and the cluster build (`f54254fa`). Its sinkage and slope numbers are not comparable with S1's. |
| C4 | M113 NSC contact as a remedy | S2 §2.4, §8.6: NSC is stable at 1-2 ms on rigid ground; listed as a pad-calibration candidate | S5 §3.3, §3.5: with NSC the track jams (0.4 m in 4 s) even on rigid ground | S2 tested only a braked settle with NSC, not driving. **Drop NSC.** |
| C5 | M113 on a 10° soil slope | S2 §3: braked, it slides back at 0.76 m/s, then recovers under throttle | S5 §3.5: "climbs slowly" | S5's raw record shows the same slide: `scratch/S5/crm_smoke_results.jsonl`, m113_bds 10° at 0.5 ms, `t0.v = -0.729` m/s after the settle. The two agree; S5's table omits it. |
| C6 | M113 "no final drive" | S5 §3.3: neither driveline has a final-drive reduction | code | The BDS driveline has a 0.5 bevel reduction (`chrono/src/chrono_models/vehicle/m113/driveline/M113_DrivelineBDS.cpp:31`; JSON `M113/driveline/M113_TrackDrivelineBDS.json:9`). S5's 0.35-of-weight push already includes it, so its conclusion stands; only the wording is off. |
| C7 | M113 separate vehicle step on soil | S2 §2.3: `RegisterVehicle` forces vehicle step = soil step | S5 recorded runs with `dt 1 ms, mbd_dt 0.5 ms` | S2 is right. With a registered vehicle, `AdvanceMBS` calls the callback, which calls `vehicle.Advance(step)` with the whole step (`chrono/src/chrono_vehicle/terrain/CRMTerrain.cpp:43`, `chrono_fsi/ChFsiSystem.cpp` AdvanceMBS; cluster source identical). `ChVehicle::Advance` does not sub-step (`ChVehicle.cpp:236-238`). S5's `SetStepsizeMBD` call (`crm_smoke.py:86-89`) was a no-op, so its "1 ms / 0.5 ms" rows really ran the M113 at 1 ms. |
| C8 | Gator re-drive and reproducibility | S1 §6.3.9: the cluster reproduces only per node, so keep paired arms in one job. S4 §3.2 arm 5: reuse stored Gator drives only if 3-5 rows are bit-identical | S3 §1.6: soil reproduces at the outcome level across nodes and dates (25/25 same outcome, 22/25 identical arrays) | S3's evidence covers soil; the memory note is about rigid ground on CPUs. S3 also notes (§2.5) that `crm_worker` hands rows to whichever GPU claims them, so "same job" cannot mean "same node" for soil. Both bit-level criteria (S3's ≥ 90 % arrays; S4's 3-5 rows bit for bit) will fail about 1 time in 2 by noise (0.88^5 ≈ 0.53). Use outcome agreement (≥ 95 % same end state). |
| C9 | Collect only with headroom? | S1 rec 5: collect all 15,235 only if soil failures leave > 10 % headroom | the user's instruction and S3 §2.4: collect if it is better than the Gator | A headroom gate is not in the user's request. Don't make it a go/no-go. Pre-declare how a ceiling result is reported (section 3c). |
| C10 | Local M113 pad work | S2 rec 2: about 30 min locally on tilted soil | local rule: CRM runs under the lock, ≤ 10 min in total | S2's 30-minute local plan breaks the local soil rule. Do it on one cluster node (devel or mi3501x, 30-45 min), or not at all. |

### 1.1 C1 in numbers (why neither driveline is clean)

**Same torque at low speed.** Below about 6 m/s both give about the same wheel torque:
- stock: first gear only, ~190 N·m / 0.267 / 0.25 ≈ 2.8 kN·m;
- corrected: ~185-397 N·m through 0.067-0.125, i.e. 2.8-3.2 kN·m.

So in the route speed band, stock is a plausible one-gear vehicle (S1 §0.4).

**What differs:**
- stock cannot upshift, and its engine and power columns are wrong by 16x;
- the corrected and open-diff variants have no torque converter, so wheelspin drives the gearbox up
  (S1 §3.3.2, S5 §3.2).

**Where it shows:**
- On rigid 25° the difference is large. Stock holds 5.86 m/s under speed control (S1 §5.1). Corrected reaches only
  2.4 m/s from rest (S5 §3.2). Open-diff rolls back (S1).
- On soil ramps the variants are close: 12.0 vs 11.8 m at 20° (S1 §5.3). S5's conda-build soil runs even put
  corrected ahead at 25°.

**The decision:** a driveline-sensitivity arm on sample A settles whether the answer depends on the model.
- The default planner reads no engine or power column: `ci_train.py` has no energy head (grep finds none), and the
  planner uses `--cond none --ctx geom`.
- So stock's broken telemetry affects only the reported energy numbers.

## 2. Claims I checked myself

| Claim | Where it matters | Check | Result |
|---|---|---|---|
| Soil launch check cannot be changed from a wrapper (S1) | Polaris spawn and launch check (C2) | `scripts/crm_collect.py:380` `def main`, `:412` `class CrmPolicy` inside it, `:421-422` height window 0-1.2 m | **Confirmed**; S5 rec 4 not doable |
| Stock Polaris driveline 16x power (S1, S5) | driveline choice (C1) | `chrono/src/chrono_vehicle/wheeled_vehicle/driveline/ChSimpleDriveline.cpp:106-116`: speed x ratio, torque / ratio | **Confirmed** (the engine sees speed x 0.25 instead of / 0.25) |
| S1's 4/4 Polaris goals on f104 soil | the only f104 evidence | `scratch/S1/f104_soil/*/*/outcome.json`: all 5 runs `goal_reached` | **Confirmed**, with caveats: 3 of 4 routes ran in the interim "z-only" frame, all used the stock driveline, and the 4 routes are hand-picked designed routes from `NOTES_E2.md:198-203` (HMMWV reached the goal on 3/4). Not a sample. |
| M113 forces a 0.5 ms soil step (S2) | M113 cost and fairness | C7 above | **Confirmed**. The HMMWV at 0.5 vs 1 ms flipped 3.5 % of outcomes (crm-f104-night memory). A 0.5 ms M113 against a 1 ms Gator is a small, declarable deviation. A Gator-at-0.5 ms control on sample A would remove it (the 1.4 simulated h of the Gator re-drive, at twice the wall cost). |
| Soil drift check (S3) | the Gator re-drive rules and bit-identity gates | `scratch/S3/drift_more_check.json` | 15/15 same status, 14/15 same length, 13/15 identical arrays, **`pass: false`** under its own rule ("every npz array identical"). S3 reports the counts but not the failed pass flag. |
| Standing start = 0.5 s history (S4) | default planner | `crm_improve_20260922/REPORT.md:10-17,66` | **Confirmed**: soil specialist 95.8 % from a standing start = short-window history 95.8 % at 0.5 s; gradient refinement +1.7-1.8 points (p 0.016) was measured **from the moving start**. From a standing start it has only night 2's +1.0 (n = 200, not significant). S4's HMMWV bridge arm is what tests it. |
| Per-row soil config in one task file | M113 at 0.5 ms next to 1 ms rows | `scripts/crm_worker.py:97` `config = t.get('config') or CONFIG` | **Works.** M113 rows can carry `"config": "configs/crm_m113.json"` in the same smoke task file. No scout noted this; S3's single-file smoke design depends on it. |
| Worker time limits vs a slow M113 | M113 on the cluster | `crm_worker.py:26` episode timeout 2,400 s; `:131` 3 failures in a row retire the worker; `ag_soil_launch.sh:25` stops claiming 1,000 s before the job limit | **Gap.** At 15-24 wall s per simulated s, the 34 s blockage stop takes 510-820 s. But a creeping or sliding M113 escapes the blockage rule (every position within 0.25 m for 2 s, `gen_collect.py:59`) and can run to the 120 s horizon: 1,800-2,900 s. It is killed at 2,400 s, counted as a failure, and three in a row retire the GPU. M113 jobs need `CRM_EPISODE_TIMEOUT_S` ≥ 3,600 and a claim margin ≥ 3,000 s. |
| Cluster state (S3, 22:40) | capacity | `sinfo`/`squeue` at 23:35 | Changed: mi3501x 7 idle, mi2104x 7 idle, mi2101x 11 idle, devel 4 idle. But 25 other users' mi2101x/devel tasks are waiting. **Another session of the user has 3 mi2104x jobs** (`rg_smoke`, `rg_aug_v1` running, `rg_aug_v1_cont` pending; submitted 23:27-23:32). They are not this study's, but they count against the same 50-task cap and the shared budget. |

## 3. What is still missing

### 3a. For a fair smoke tonight

1. **Driveline declaration** (C1). The corrected-driveline data folder exists only as a template
   (`scratch/S5/vehdata/Polaris/`, stock frame). S1's re-framed JSON (`scratch/S1/vehdata/Polaris_ov/`) uses the stock
   driveline. The combination "re-framed + corrected" has never been built or run.
   - A one-minute local rigid build check is needed: mass, spindle positions, settle at +0.40 m.
   - So is one short local soil settle under the lock.
2. **Gates that do not trip on noise** (section 0, item 3). Replace array-identity gates against stored runs with:
   - (i) outcome agreement ≥ 95 % against stored runs;
   - (ii) a same-GPU back-to-back check: the old dispatcher, then the new one, on 2-3 HMMWV and Gator rows, run by one
     short script outside the worker pool. Arrays must be equal.
3. **The Polaris belly-in-soil flag file** (S3 gate 4 needs hull points per vehicle). Only a recipe exists (S1 §6.1.2,
   from the visual chassis mesh). Also, `ag_s2_extras.py:46,68` knows only the Gator and the HMMWV (S4 §1.12).
4. **A first cluster step that proves the Polaris builds on a compute node** with the private vehicle-data folder.
   - Nobody has done it (S1 §7).
   - A missing JSON aborts the process with no failure record (S1 §4). The first of these then retires a GPU after
     three rows (`crm_worker.py:131`).
   - Put 2-3 Polaris rows in the smoke's quick-look tier before any other Polaris row.
5. **The M113, if it is run at all.** It needs:
   - a declared gearing fix (see 3d);
   - a pad design that holds a braked 10° slope;
   - its episode-timeout settings (section 2);
   - per-row `config`;
   - a reference-point decision.

   No scout has a pad design that holds (S2 §3; C5). Without one, S3's "better than the Gator" rule cannot be read for
   the M113.
6. **The breakthrough rule for the M113.** S2 §4 has a concrete equivalent: the lowest pad of a track quarter more
   than 0.30 m below the undisturbed surface for 5 frames. S3 §2.3 asks for it to be declared. It is designed, not
   built or tested.
7. **The M113's reference point is at the front sprocket.** Goals count about 2 m early, and 158 of 1,200 start
   footprints are unchecked (S2 §4). The JSON M113 (`M113/vehicle/M113_Vehicle_SinglePin_BDS.json`) could be
   re-framed the way S1 did the Polaris, and its gearing changed in the same data folder. No scout considered that
   (S5 mentions the JSON only for gearing). Untested.

### 3b. For collecting 15,235 soil ids per passing vehicle

1. **About 20 new `ov_*` copies**, none written yet. S1 §2, S3 §3.2 and S4 §1.12 each give a partial list. The union:
   - `ov_vehicle.py`, `ov_crm_collect.py`;
   - task-row builder (from `ag_soil_tasks_v2.py`), `ov_soil.sbatch` + `ov_soil_launch.sh` (G4 root);
   - QA read-out (from `ag_s1_gator_soil_qa.py:47`), `ag_s2_extras.py`;
   - `ag_build_ds.py:51,272,346`, `ag_eval_tasks.py:53-54,97-100,170`, `ag_eval_index.py:156`,
     `ag_bf_soil_v4.py:76`;
   - spec (`ag_bf_spec.py`), deploy/sync shell scripts;
   - the billing script (with mi3001x, mi3508x and mi3008x added, S3 §3.2);
   - a manifest writer for gradient picks (S4 §2.4).

   Budget 1-2 h of careful work, and freeze and hash everything before the first row.
2. **Only the declared primary Polaris arm's sample-A rows can be reused** as collection rows, and only if the frozen
   dispatcher and data folder are the same (S3 §3.1). Sensitivity arms need their own id prefixes (e.g.
   `polarisOD__`, `polarisSTK__`) so the dataset builder never mixes them.
3. **Cost.**
   - Polaris: about Gator speed per simulated second (S1 §5.4: real-time factor 0.49-0.56 vs the Gator's 0.51-0.55,
     local).
   - If it reaches the goal as often as the HMMWV: about 92 simulated h, about 31 billed (S3 §4.3).
   - It fits the budget (742 node-hours left, less whatever the `rg_*` jobs use).
   - It does not fit before 08:00. Plan stage 1 (tiers 0-6) first, as S3 says.
4. **The M113 full collection is out on three counts:**
   - the S3 cost gate (6x) is below S2's measured 7.7x;
   - S2's 250-450 billed estimate exceeds S3's 150-billed session cap;
   - the gearing and grip issues above.

### 3c. For training and evaluating a planner against the 90 % bar

1. **A pre-declared ceiling rule.** If the Polaris straight 6 m/s route already reaches the goal on about 88 % or
   more of sample B's 96 pairs, S4's "planner works" test F2 (planner beats the straight route) has little room.
   - Declare now how that case is reported. For example: "the 90 % bar is met by the straight route; the planner's gain
     is X points, with this interval".
   - Do not make it a reason to skip the collection (C9).
2. **Standing-start gradient refinement is lightly tested on soil** (section 2, S4 row). Keep S4's HMMWV bridge arm
   (H_full + gradient from a standing start, target about 97 %) in the spec. It is the only check that the default
   behaves as claimed.
3. **Gator comparison arms.** Re-drive `Gcem_gator` and `straight6_gator` (about 4 billed) rather than rely on a
   bit-identity test (C8). Or use outcome-level reuse with the agreement rate reported.
4. **"Same amount of data".** The primary model must be trained on exactly the 15,235 ids. S4's wide-shape rows (W1,
   2,400 more) must stay a separately labelled secondary model (S4 already says so).
5. **M113 only.** Even the history-free default needs a 17-column state and per-wheel soil fields, or the builders
   stop (`crm_qa.py:32`, S4 risk 4). S2 §4 designs the quarter mapping; not built.

### 3d. Unverified claims that matter tonight

- The Polaris on any cluster node, and whether its 0.25 m calibration holds on MI350X/MI210 (S1 §7).
- The re-framed + corrected Polaris combination: never run (3a.1).
- The M113's 15-24 wall s per simulated s on the cluster: projected from one local run (S2 §2.4).
- S5's gearing fix "gearbox ratios ÷ 4": untested. A single-number alternative is the BDS bevel ratio in
  `M113_TrackDrivelineBDS.json:9` (0.5 → 0.125). Also untested.
- The M113 pad grip: no design has held a braked slope.
- S3's capacity projection (±50 %), made before the 23:35 change in idle nodes and the `rg_*` jobs.

## 4. Recommendations, in order

1. **Declare now** (in PLAN.md before any drive):
   - the Polaris primary driveline;
   - the re-framed JSON + spawn +0.40 m + 0.25 m cylinders (S1);
   - sample A and B and the decision rule (S3 §2.4);
   - outcome-level gates instead of array-identity gates against stored runs;
   - a same-GPU bit-identity script;
   - the ceiling-reporting rule;
   - the M113 decision (below).
2. **Primary driveline.** My lean is **stock, labelled "Chrono's Polaris as shipped (known power bug; engine and power
   columns invalid)"**. It is the only variant with f104 soil evidence. It avoids the gearbox's upshift artefact. In
   the ≤ 6 m/s band its wheel torque is close to the corrected one's.
   - Paired arms on sample A: corrected and open-diff (+~2 simulated h, < 1 billed).
   - If stock and corrected differ by more than 10 points on sample A, report the result as driveline-dependent. Then
     let the user choose before tiers 7-12.
   - Choosing corrected as primary is also defensible, but then it needs a local build check first (3a.1).
3. **M113: do not include it in the smoke task file tonight.** Record that the stock M113 is gear-limited on rigid
   ground (S5 §3.3) and slides when braked on 10° soil with the only pad representation that couples at 0.08 m (S2 §3,
   S5 raw data). If the user wants an M113 number anyway:
   - one cluster devel/mi3501x job of at most 45 min;
   - the JSON M113 with a declared bevel-ratio fix and a ridged pad;
   - a braked-hold test on 10° and 15°;
   - only if it holds, 24 sample-A routes at 0.5 ms, with a Gator-at-0.5 ms control, per-row config and an episode
     timeout ≥ 3,600 s.
4. **Order of work:**
   - write and freeze the `ov_*` copies (Polaris only);
   - local rigid build and one soil settle under the lock;
   - stage G4;
   - one mi3501x job with 2-3 quick-look Polaris rows, then samples A and B (Gator re-drive included);
   - after sample A, collection tiers 0-6;
   - stage-1 training and the evaluation drives at a negative tier ahead of tiers 7-12 (S3 §3.5, S4 §3).
5. **Cluster hygiene:**
   - count the `rg_*` jobs against the 50-task cap;
   - submit with `env -u NEDM_VEHICLE`;
   - log every job id in `LOG.md`;
   - keep mi3501x as the backbone and add the mi2104x/mi2101x nodes that are idle now with 4-6 h limits.
