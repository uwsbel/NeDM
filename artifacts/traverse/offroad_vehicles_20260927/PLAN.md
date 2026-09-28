# PLAN: Chrono's Polaris and M113 on f104 soil (2026-09-27/28)

Branch `offroad_vehicles_v1` (from `arena_gator_v1` 051dfdba5). Local root K4 = `artifacts/traverse/offroad_vehicles_20260927/`.
Cluster root G4 = `/work1/dannegrut/harry/experiments/offroad_vehicles_20260927` (new; G3 =
`.../arena_gator_20260925` and every older root are read-only). Scout reports: `scout/S1-S5`, `scout/CRITIC.md`.
Written 2026-09-27 23:55, before any cluster drive of this study. Amendments go in section 9 with a timestamp.

## 0. Questions

- **Q1 (smoke).** On f104 soil, does Chrono's Polaris, and does Chrono's M113 tracked vehicle, traverse better than the
  Chrono Gator (same routes, same soil, same follower)?
- **Q2 (data).** For each vehicle that passes Q1, can it collect the same data as the Gator (the same 15,235 soil task
  ids, tiers 0-12)?
- **Q3 (planner).** Does a planner trained on that vehicle's data reach the goal on 90 % or more of the 800 f104 soil
  suite pairs when driven by that vehicle (the user's success bar)?

## 1. Fixed decisions (reasons in the scout reports)

1. **Soil, arena and loop unchanged:** f104, production CRM settings (`crm_main.json`: 0.08 m spacing, 1 ms step,
   0.24 m layer over the terrain-following floor), the frozen collector `crm_collect.py`, the frozen follower gains,
   route limits, labels and stop rules. Soil only; no rigid collection (the user asked about soil).
2. **Polaris = Chrono's JSON Polaris (MRZR, 1,378 kg, 4WD, 0.33 m tyres)** with only the changes it needs to run in our
   loop (S1 section 4, 6.1; CRITIC item 2):
   - chassis reference moved to mid-wheelbase (copy of the two top-level JSON files; physics-identical by test), spawn
     ground + 0.40 m in the re-framed model;
   - soil contact wheels = calibrated cylinders r = 0.25 m (stock 0.33 - 0.08), stock width (S1 6.2: settled sinkage
     matches the HMMWV's). Rigid-mesh tyre for the vehicle model as in the collectors;
   - **primary driveline = the stock one as shipped** (Chrono's `SimpleDriveline`, which has the reduction defect of
     S1 3.3 / S5: engine speed reported 1/16, gearbox stays in first; at 0-6 m/s it behaves like a one-gear ~50 kW
     4WD vehicle). Labelled as such everywhere.
   - Vehicle names: `polaris` (primary), and smoke-only sensitivity arms `polaris_pc` (power-corrected: driveline
     reduction 1.0 and gearbox ratios x 0.25, S5 templates, on the re-framed model), `polaris_4wd` (Chrono's shafts
     4WD driveline `Polaris_4WD.json`, open differentials), `polaris_w08` (primary with soil cylinders +0.08 m,
     r = 0.33).
   - Private vehicle-data folder (re-framed + variant JSONs) hashed into every vehicle record.
3. **M113 = pychrono `veh.M113`** (S2 section 6): SMC contact, single-pin shoes, brake-steering driveline (BDS),
   shafts engine + automatic shafts transmission, each of the 127 track shoes coupled to the soil as one flat pad box,
   **0.5 ms step for M113 rows** (1 ms is unstable with the tracked vehicle; per-row soil config), 17-column state with a
   declared quarter mapping (road-wheel loads / sprocket speeds), a breakthrough rule equivalent (lowest shoe-bottom
   point per quarter against the rigid floor, same 0.30 m threshold semantics), belly points, episode timeout >= 3,600 s.
   - Arms: `m113` (stock) and `m113_g4` (gearbox ratios / 4: the stock model's drive push is <= 0.35 of its weight,
     S5 / CRITIC item 4; a real M113 climbs ~31 deg).
   - **Bounded path (CRITIC items 4-10):** the M113 enters the smoke only if its module passes its own gates by 03:00
     (build on a compute node; the braked vehicle holds on 10 deg tilted soil, with pads or a declared grouser
     variant; one f104 episode passes QA). Otherwise it is reported as "not representable on our soil tonight" with the
     scout evidence. Its full collection is out of scope tonight regardless (cost 250-450 node-hours, S2 section 5);
     if it passes Q1, report that and propose a reduced collection to the user.
4. **Gator reference:** the existing `ag_vehicle` Gator, re-driven through the new dispatcher on sample A in the same
   launch (`gatorctl__` prefix).
5. **Task rows:** ids `<vehicle>__<collect_v1 id>` (M113 / sensitivity arms the same with their names), same case,
   route, episode seed, tier as the Gator rows of `K3/e3/tasks/soil_v2.json`; explicit `--vehicle <name>` in every
   row's extra; never `NEDM_VEHICLE` (submit with `env -u NEDM_VEHICLE`).
6. **New scripts only**, prefix `ov_`; every `ag_*` / older file stays frozen. The collector wrapper's name contains
   `crm_collect`. The new dispatcher is frozen (`chmod a-w`, sha256) before the first job; HMMWV and Gator rows through
   it must match the stored runs (section 3.4).
7. **Planner default = the vehicle's own soil model + CEM 4 x 64 + gradient refinement (`ci_grad`), from a standing
   start** (S4 section 2: the crm_improve gain came from gradient refinement, +1.7 points; deciding after a 0.5 s
   approach with the history model added ~0 when the ground type is known). Model: `ci_train --arch gru --cond none
   --domain-filter crm --ctx geom`, 5 seeds x 30 epochs, fresh normalisers, on the vehicle's own rows.

## 2. Smoke test (Q1)

### 2.1 Stage 0: flat soil driveability (one MI350X job, <= 45 min)
HMMWV, Gator, Polaris (all four arms) and M113 (if ready) on a flat soil patch with the production settings: settle,
straight 2 / 4 / 6 m/s, a 0.10 /m half circle at 3 m/s, full brake from 6 m/s; plus the braked hold on 10 deg and 15 deg
tilted soil. Pass rules as S3 2.1 (launch check, speed within 10 % after 4 s, cross-track < 0.5 m, finite states,
vehicle block). Reported for all; a vehicle failing stage 0 does not enter stage 1.

### 2.2 Stage 1: paired route samples (one task file, tiers in this order)
| tier | rows |
|---|---|
| -4 | 3 HMMWV + 3 Gator bit-identity rows through the new dispatcher (already-driven `collect_v1` ids) |
| -3 | quick look: sample A tiers 0-1 (48 routes) for `polaris`, `gatorctl`, and `m113`, `m113_g4` if admitted |
| -2 | rest of sample A (tiers 2-5, 96 routes) for `polaris`, `gatorctl` (+ M113 arms only if the quick look shows cost <= 8x the Gator per simulated second) |
| -1 | sample A (144) for `polaris_pc`, `polaris_4wd`, `polaris_w08`; sample B (96 suite pairs x {straight 6 m/s, H_full pick}) for `polaris` |

- **Sample A** = the 144 routes of the Gator pilot (24 groups x 6 routes; `scratch/S3/sample_A.json`): stored
  outcomes Gator 93.8 %, Gator + 0.08 m wheels 80.6 %, HMMWV 69.4 % goal not reached.
- **Sample B** = 96 of the 800 suite pairs (16 per terrain type, lowest md5; `scratch/S3/sample_B.json`) with the
  straight 6 m/s route and the HMMWV-trained H_full pick (stored: Gator 83.3 / 51.0 %, HMMWV 34.4 / 3.1 % not reached).

### 2.3 Validity gates (per vehicle arm)
As S3 2.3: >= 95 % of ids complete and pass `crm_qa.check`; < 1 % crashed / non-finite / explosion; < 5 % launch-check
failures; belly flag <= 10 % of drives (hull points per vehicle); vehicle record present; bit-identity rows judged at
the **outcome level** (CRITIC item 3): HMMWV / Gator rows and the `gatorctl` re-drive must have the same end state as
the stored runs on >= 95 % of rows (exact arrays reported, not gating).

### 2.4 Decision rule: "better than the Gator" (declared now, before any drive)
Primary endpoint: goal not reached on sample A, paired route by route with the same-launch Gator re-drive. A vehicle
(primary arm `polaris`; for the M113 the better of its two arms, Holm over 3 comparisons) is **better** if all hold:
1. its validity gates pass;
2. failure rate lower by >= 10 points (point estimate);
3. exact one-sided McNemar p <= its Holm level (alpha 0.05);
4. the 95 % bootstrap interval over the 24 groups of the difference lies below 0;
5. its failure rate is below 80.6 % (the Gator with 0.08 m larger wheels).
Consistency: on sample B's straight 6 m/s route its failure rate must be below the Gator's 83.3 %; if A and B disagree,
report "mixed" and do not collect. Reported, not gating: the three Polaris sensitivity arms (if `polaris` and
`polaris_pc` or `polaris_4wd` differ by > 10 points, the result is labelled driveline-dependent), groups with any goal,
the H_full-pick goal rate on B, end-state mix, simulated seconds per route, wall seconds per simulated second.
**Ceiling rule:** if `polaris` fails < 12 % of sample B's straight 6 m/s routes, the report says the straight route
already nearly meets the bar; the collection and planner still run (the user asked for them).

## 3. Collection (Q2) for a vehicle that passes

1. Rows = the Gator's 15,235 soil rows with the vehicle prefix / flag changed (S3 3.1); the 144 sample-A rows of the
   primary arm are reused when driven by the frozen dispatcher.
2. Order: tiers 0-6 (8,399 ids) first; then the stage-1 evaluation drives at a negative tier (section 4); then tiers
   7-12. Superset task files only (recipe `K3/e3/README.md`); `STOP_CLAIMS` for soft stops.
3. Launch: `ov_soil_launch.sh` (copy of `ag_soil_launch.sh` on G4), 4-6 h tasks, all partitions that run soil
   (mi3501x 4 h cap, mi2104x, mi2101x, mi2508x), <= 45 of my queued tasks (another session of the user's also has
   jobs; keep the total <= 50). Two workers per GPU tested on one node in the first hour, kept if >= 1.2x.
4. Validation (as `ag_s1_gator_soil_qa.py`): completion, QA, launch, belly flag, twin comparison with HMMWV and Gator
   by tier/profile. Criteria (PLAN 7.7 of K3): >= 95 % ids valid, < 1 % crashed, < 5 % launch-check failures, belly
   <= 10 %.

## 4. Planner and evaluation (Q3)

1. Data and training: `ov_build_ds.py` (vehicle-generalised copy of `ag_build_ds.py`) -> `ov_subset` -> `ci_train`
   (recipe of section 1.7), on MI350X. Stage 1 = tiers 0-6, stage 2 = all tiers.
2. Picks (standing start, 800-pair f104 suite `K3` suites; also reported on the 600 fresh pairs), locked by sha256
   before any drive; analysis spec frozen before any outcome:
   - `<v>_grad`: own model + CEM 4 x 64 + `ci_grad` default (declared default planner);
   - `<v>_cem`: own model, CEM 4 x 64 (the Gator's protocol);
   - `straight6`: straight route at 6 m/s (sample B's drives count for their 96 pairs);
   - Gator: `G_full_grad` (new drives); stored `G_full` CEM and `straight6_gator` drives reused if the sample-A
     re-drive passes the outcome-level rule;
   - HMMWV check: `H_full_grad` on the HMMWV (expected about 97 %; validates the standing-start gradient arm);
   - optional, only if time and budget allow: wide-shape search (probe arm A4b: 5 bends, 10 m, CEM 16 x 512) for
     the passing vehicle and the Gator; the Gator's probe picks (A4, 134 + 100 pairs).
3. **Criteria for "the planner works" (declared):** `<v>_grad` (a) beats `straight6` on the same vehicle (paired, Holm),
   (b) reaches the goal on >= 90 % of the 800 pairs and of the 600 fresh pairs ("meets the bar"; "clearly above" if the
   95 % group-bootstrap lower bound is >= 90 %), (c) valid drives: no crash / NaN, < 5 % launch failures, belly flag
   <= 10 %, and the bar still met when belly-flagged drives count as failures.
4. **Declared family (Holm at 0.05, one-sided, group bootstrap + exact McNemar beside it):** per passing vehicle
   `<v>_grad` vs `straight6`; `<v>_grad` vs `G_full_grad` (vehicle comparison, both with their own planner);
   `<v>_grad` vs `<v>_cem`; and `G_full_grad` vs `G_full` CEM. "No meaningful difference" only if the 90 % interval lies
   within +-2 points.

## 5. Budget, time, stop rules

- Soft cap 150, hard cap 180 billed node-hours (sacct since 09-27 22:00, `ov_bf_billed.py` with weights for every
  partition incl. mi3001x 0.125, mi3508x 1.2, mi3008x 1.0).
- Expected (S3 section 4): smoke 3-8; Polaris collection 30-50 (0.65-1x the Gator's 48); evaluation 10-15; training < 2.
- Timeline target: smoke launched by ~01:30; Q1 decision ~03:30-04:00; Polaris tiers 0-6 by late morning 09-28;
  stage-1 evaluation drives right after; tiers 7-12 and stage 2 in the afternoon.
- Stop a vehicle's rows (superset with `run: false`) if its quick look shows > 8x the Gator's wall seconds per
  simulated second, > 5 % launch failures, or any crash pattern.

## 6. Modules (all new files; contracts)

| module | owner files | contract |
|---|---|---|
| M1 Polaris + dispatcher | `scripts/ov_vehicle.py`, `scripts/ov_crm_collect.py`, `assets/traverse/vehicles/ov_polaris/` (JSONs, belly points, manifest), `scripts/ov_polaris_check.py` | `--vehicle {hmmwv,gator,polaris,polaris_pc,polaris_4wd,polaris_w08,m113,m113_g4}`; `gator` delegates to `ag_vehicle` (byte-identical rows); `m113*` delegate to `ov_m113` (lazy import); writes the `vehicle` block |
| M2 M113 | `scripts/ov_m113.py`, `configs/crm_m113.json` (copy with `step_s` 0.0005), `scripts/ov_m113_check.py` | `ov_m113.install(args)` patches the collector like `ag_vehicle` does, for `m113` / `m113_g4` only; nothing installed otherwise |
| M3 staging, tasks, launch, smoke | `scripts/ov_stage.sh`, `ov_soil.sbatch`, `ov_soil_launch.sh`, `ov_worker` copy only if the episode timeout must be per row, `ov_smoke_tasks.py`, `ov_driveability.py`, `ov_soil_tasks.py`, `ov_bf_billed.py`, `ov_collect_status.py`, `ov_smoke_analyze.py` (frozen decision rule of 2.4) | task rows per section 1.5; G4 layout as S3 3.2 |
| M4 planner + evaluation chain | `ov_build_ds.py`, `ov_subset.py` (if needed), `ov_train.sbatch`, `ov_grad_picks.py` (ci_grad from a standing start + the pick manifest the row builder reads), `ov_eval_tasks.py`, `ov_eval_index.py`, `ov_analyze.py` (or thin wrappers over the `ag_` ones with a vehicle map) | vehicle-generalised, no hard-coded `gator`; refuse test-group leaks as `ag_build_ds.py` does |

Each module ends with its own checks (listed in the scout checklists: S1 6.4, S2 section 8, S3 2.1/3.2, S4 1.11) and an
independent verifier before any cluster use.

## 7. Rules
Plain language in reports; older artefact folders read-only; local soil runs only under `flock /tmp/luffy_crm.lock`
and short; no dataset collection locally; training on the cluster; never set `NEDM_VEHICLE`; <= 50 queued tasks per
user (count the other session's jobs); mi3501x 4 h cap; commit attribution line; checkpoint commits + push on
`offroad_vehicles_v1` with per-folder allowlists in `.gitignore`.

## 8. Deliverables
`RESULTS_smoke.md`, `RESULTS_collection.md`, `RESULTS_planner.md`, `REPORT.md` (plain language, answers to Q1-Q3, the 90 %
bar, caveats: driveline defect, soil-wheel stand-ins, body not coupled to soil, M113 pad model and step), figures,
progress document entry, memory update.

## 9. Amendments

### 9.1 (2026-09-28 01:10, before any smoke drive; after modules M1-M4 passed verification and review R1)
Adopted from REVIEW_R1 and the module notes (NOTES_M1-M4, VERIFY_M1-M4):
1. **B1 - M113 in its own task file and jobs.** `tasks/smoke_m113_v1.json` (432 rows: `m113`, `m113_g4` and `gatorh` =
   the Gator at the M113's 0.5 ms step, on sample A; quick look = tiers 0-1 first) runs in separate jobs with the 4,000 s
   episode timeout. The Polaris decision never waits for it. The Polaris smoke file is `tasks/smoke_v2_polaris.json`.
2. **B2 - stage 0 excludes a new vehicle only** for launch failure, non-finite state, not reaching 2 m/s on flat soil, or
   (M113) not holding braked on 10 deg tilted soil. The 6 m/s speed and 0.5 m turn marks of 2.1 are reported against
   the HMMWV, not gating (the frozen follower cuts a 10 m radius by ~0.8 m for every vehicle). Stage 0 results: M1 job
   441578 (Polaris arms, HMMWV, Gator), M2 jobs 441588/441589 (M113). All admitted vehicles pass B2.
3. **M113 definition (M2).** Chrono's C++ M113 class built exactly as `veh.M113` builds it (identical by test), with one
   physics change: the brake is Chrono's shafts brake (a clutch per sprocket) because the stock simple brake never locks
   (the braked stock M113 rolls back even on rigid 10 deg). Flat pad boxes (no grousers: the braked M113 holds on 10 and
   15 deg soil like the Gator with them). `m113_g4` = the same with 4x lower gearbox ratios ("re-geared M113"). The
   M113 soil breakthrough stop is effectively inactive (pads rest 0.09-0.16 m above the surface): declared, its largest
   sinkage reported. Its reference point is the front sprocket (goals count ~2 m early): reported as a caveat.
4. **S1 - driveline robustness.** Q1 is called "robust" only if `polaris_pc` also meets criteria 2 and 5 of 2.4; if the
   primary and a driveline arm differ by > 10 points the result is "driveline-dependent" and I ask the user before
   tiers 7-12. The Polaris gradient picks are also driven with `polaris_pc` (secondary bar).
5. **S2 - Gator reuse.** Stored Gator drives are reused only if the `gatorctl` re-drive keeps at least 8 of the 9 stored
   Gator goals on sample A and >= 95 % of end states; otherwise the Gator CEM arm is re-driven under a new name.
   Gator straight-route controls on sample B: not added (the 144-route re-drive already tests reproducibility).
6. **S3, S4 - Q3 answer.** Q3 is answered by 4.3 (b) and (c) only; "beats the straight route" is reported separately.
   "Reaches the goal safely" = goal reached AND not unsafe (no roll-back on a climb) AND no belly-in-soil flag; the
   goal-only rate is reported beside it.
7. **S5 - stages.** Stage 2 (all 15,235 ids) answers Q3 and carries the Holm family; stage-1 arms are named `*_s1_*` and
   are interim.
8. **S6 - intervals.** "Clearly above the bar" uses the 9-cluster bootstrap (as the Gator study); the pair-level group
   bootstrap is reported beside it. Intervals are two-sided 95 % unless stated.
9. **S8 - tier order.** Sample B rows at tier -2 (with the rest of sample A); sensitivity arms at tier -1. The Gator and
   HMMWV gradient picks (M4) are locked (`e6/picks/LOCK_gradref_v1.sha256`, cf0396c4) and are driven only after the
   evaluation spec is frozen.
10. **S9 - wheel calibration** is reported (settled sinkage vs the HMMWV), not gating. Caveat: `polaris_w08` (0.33 m soil
    cylinders) did not move from its start on the one route tried; wheel size matters at a standing start with the stock
    driveline's low torque.
11. **Not adopted:** R1 S7's goal-from-centre-of-mass recount (it would count every M113 goal as a failure as worded);
    the M113 brake-cap fix (F1: never reached in the drives).
