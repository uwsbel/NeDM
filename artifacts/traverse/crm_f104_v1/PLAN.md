# CRM f104 night session — plan (written 2026-09-16 ~23:50 CDT, before the main collection and before any evaluation drive)

Goal: HMMWV traversal on CRM (Chrono SPH deformable soil) built from the familiar f104 arena BMP; collect 50-100
simulated hours, train the existing risk network from scratch on the CRM training split, and compare the resulting
planner with the frozen rigid-trained planner on held-out single-goal CRM missions.

Local root `artifacts/traverse/crm_f104_v1/`; cluster root `/work1/dannegrut/harry/experiments/crm_f104_20260916/`.

## 1. What is kept from the rigid pipeline (contracts)

- Case / route JSON, start-goal groups, designed routes (3 lateral offsets x 4 speed profiles) and planner-proposal
  routes: the night-2 pool `f104_v2_group_0000..1199` reused verbatim, 20 routes per group. Episode ids equal the
  rigid ids, so every CRM episode has a rigid twin (same route, same controller).
- Controller: `ChPathFollowerDriver`, steering 0.8/0/0 with 5 m look-ahead, speed 0.6/0.05/0, 2 /s steering rate limit,
  50 ms control and record period, 0.8 s braked settle, path height = BMP + 0.5 m, parking rule within 3 m of the end.
- Termination: goal radius 2.5 m, rollover 60 deg, arena bounds +-40 m, prolonged blockage (2 s window within 0.25 m at
  throttle > 0.3, not before 24 s, + 2 s + 8 s), 120 s horizon. Imported from `scripts/gen_collect.py`, not copied.
- Files per episode: `trajectory.npz` (17-field state, action = [steering, throttle, braking], pose), `outcome.json`,
  `command_reference.npz`, `anchor_state.npz`, `case.json`, `reference.json`, `episode_complete.json` (atomic marker).
- Labels: `scripts/f104_n2_dataset.py` unchanged (unsafe = goal not reached OR rolled backwards under throttle OR
  vx <= -0.30 after a 1 s settle; event station = high-water mark of route progress at the first event).
- Model input: 96 x 32 corridor (+-6 m), channels elevation-relative, along grade, cross slope, commanded speed,
  valid; geometry-only 5-d context. Network `scripts/gen_riskmodel.py` (CNN -> BiGRU -> per-station hazard), survival
  loss, AdamW 2e-3, OneCycle, batch 256, 30 epochs, 5 seeds.

## 2. What CRM changes (unavoidable)

- Terrain: `veh.CRMTerrain` constructed from the same BMP, same height range [-1.8, 3.9] m, 80 x 80 m, uniform soil
  depth 0.24 m, side walls + floor boundary markers. Orientation verified on the initial particle lattice against the
  BMP (rmse = lattice quantisation; y-mirrored control 1.23 m). Fresh terrain per process = soil reset per episode.
- Soil: the repo's earlier CRM setting = Chrono's vehicle demo: density 1700 kg/m3, cohesion 5 kPa, friction 0.8,
  Young 1 MPa, Poisson 0.3, mu(I) off-set 0.04, grain 5 mm; RK2, PPST shifting, artificial viscosity 0.5, ADAMI walls.
- Particle spacing 0.08 m (the repo's earlier CRM value; 4.0 M particles), active box 2 x 2 x 1 m per wheel.
- Tyres: rigid mesh tyres coupled to the soil as FSI bodies (TMEASY cannot see particles). The chassis is NOT coupled
  to the soil (no belly contact) - limitation, same as the repo's earlier CRM work and Chrono's demo.
- Solver: Barzilai-Borwein + implicit linearised Euler, physics step from the CRM config (rigid used 2 ms).
- The rigid native-height raycast audit is replaced by a settled-launch check (speed, tilt <= 25 deg, yaw, position,
  chassis height above the BMP in [0, 1.2] m).
- One added terminal status `soil_breakthrough_terminated`: any wheel more than soil depth + 0.06 m below its
  rolling height on the BMP for 0.25 s. Found in the pilot: a stalled vehicle at full throttle spins one wheel (open
  differentials), excavates the whole 0.24 m layer in ~10-15 s and then drops through the floor, which the old rules
  read as a "rollover". It is counted as a failure (bogged). QA will check that these are preceded by a stall.

## 3. Step-size decision rule (declared before the pilot comparison is complete)

Pilot: 24 groups x 6 routes at step 5e-4 (A) and 1e-3 (B), plus 12 groups x 4 routes at spacing 0.06 m (C).
Adopt 1e-3 for the main collection iff on the shared episodes (n >= 60): same terminal status >= 85 %, failure-rate
difference within +-8 points, median time-to-goal difference on joint successes within +-0.3 s. Otherwise 5e-4.
C is a resolution sanity check only (0.06 m is not affordable for 50 h tonight).

## 4. Collection

- Tasks: 1,200 groups x 20 routes = 24,000 candidate episodes, tiered (k-th random route of every group) so an early
  stop leaves every group with the same number of routes. Unique episode id and `episode_seed = md5(id)[:8]`.
- Splits by start-goal group: the generator's 90/5/5 hash (train/val/test), all 20 routes of a group together.
- Workers: one process per GPU, `mkdir` claims, heartbeat, stale take-over, resumable, `STOP_CLAIMS` file to stop.
- Validated driving hours = sum of `actual_elapsed_s` over episodes with a completion marker, a passed launch check,
  finite states, and not flagged by QA as invalid physics; the 0.8 s settle and crashed processes are excluded.

## 5. Training

`scripts/crm_train.py --mode deploy`: from scratch, every row of the training-split groups, 5 seeds, identical
hyper-parameters; reported on the held-out val+test groups (never fitted). A `--mode holdout` run (dev fold left out)
is a sanity read-out only. The frozen rigid ensemble `night2_v1/final/N2_s{0..4}.pt` is scored on the same held-out
CRM rows for the offline comparison.

## 6. Evaluation on held-out single-goal CRM missions (pre-registered)

- 200 fresh hill/crater start-goal groups `f104_crm_eval_group_*` (seed 20260926104), >= 2 m (4-D start+goal
  distance) from every start-goal pair ever used on f104, including all 1,200 CRM training groups.
- One 256-candidate proposal pool and one 256-candidate fixed-2 m/s pool per group (md5 seeds `crm_proposal`,
  `crm_fixed2`), corridors from the OptiX reference depth image. Every model scores the SAME tensors; picks are
  written to disk before any drive; identical picks are driven once.
- Arms: `crm` and `rigid` (argmin of the ensemble-mean route logit, speed free), `crm_fixed2` and `rigid_fixed2`
  (geometry only at 2 m/s), `crm_pess` and `rigid_pess` (argmin of the max over members), `straight6`, `straight2`.
- PRIMARY: goal not reached (`fail`), `crm` vs `rigid`, speed free; exact two-sided McNemar on discordant groups and
  a paired group bootstrap (4,000) for the difference. Co-primary under Holm: `unsafe`, `crm` vs `rigid`, speed free.
- Secondary: the same two outcomes for the fixed-2 arms and the pessimistic arms; each model vs `straight6`;
  median time to goal; tilt > 30 deg; rigid-twin outcome rates are reported as context only.
- Groups with every arm present are analysed. If fewer than 200 groups can be driven in the night, the driven
  subset is the first N groups by index (fixed before driving), never chosen by outcome.

## Amendment 1 (2026-09-16 23:48 CDT, after seeing 32 shared A/B pilot episodes, before the decision)

The step-size rule in section 3 said "same terminal status >= 85 %". That criterion is confounded by my own change:
the collector gained the `soil_breakthrough_terminated` status in the middle of the pilot, so the SAME failure (stalled,
dug in) is recorded as `rollover`, `soil_breakthrough_terminated` or `prolonged_blockage_terminated` depending on which
collector version and which termination fired first. Literal status agreement at n = 32 is 69 %, all disagreements but
two being between those three failure names. The labels use only goal reached / not reached, the backward-motion test
and the stall onset, none of which depend on the failure name. The rule is therefore applied to the BINARY outcome:
adopt 1e-3 iff (n >= 60 shared) goal-reached agreement >= 85 %, failure-rate difference within +-8 points, and median
time-to-goal difference on joint successes within +-0.3 s. Seen so far at n = 32: 93.8 % agreement, 50 % vs 50 %
failures, +0.00 s.

## Amendment 2 (2026-09-17 00:30 CDT, before any evaluation drive, no CRM model trained yet)

From an independent review of the scripts and the pilot data:
- On CRM `unsafe` is (almost) identical to `fail`: no pilot success rolled backwards (stalled vehicles dig in instead
  of sliding). A co-primary that duplicates the primary only doubles its p-value. The evaluation therefore has ONE
  primary outcome: goal not reached, `crm` vs `rigid`, speed free, exact McNemar, alpha 0.05. `unsafe` is secondary.
- Evaluation drives are ordered group by group; if the night ends early the analysed set is the longest prefix of
  groups (by index) with every arm driven.
- Offline read-out caveat: the 111 val/test groups are unseen by the CRM model but the frozen rigid ensemble was
  fitted on the rigid twins of those very routes, so the offline numbers favour the rigid model if anything. The
  clean comparison is the Chrono evaluation on the 200 fresh groups, unseen by both.
- Known physics caveats kept unchanged for the whole study (collection and evaluation use identical settings): the
  2 x 2 x 1 m active box is attached to the spinning wheel hub, so its vertical reach varies during a revolution
  (Chrono behaviour; Chrono's own demo uses a smaller 0.8 m cube); recorded tyre normal forces are single 1 ms
  samples (not used by any model tonight).
