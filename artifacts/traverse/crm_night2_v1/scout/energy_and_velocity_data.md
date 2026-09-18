# Scout: energy signal and re-anchored (moving-start) samples from recorded episodes

Read-only reconnaissance, 2026-09-17/18, worktree `/home/harry/NeDM-traverse_mppi` (branch `traverse_mppi`). No simulation or
cluster job was run. Numbers marked SCAN come from a numpy pass over local episodes (script `/tmp/energy_scan.py`, per-episode
tables saved next to this file in `energy_scan/{rigid_v2,rigid_v4_onpolicy,crm_v1}.npz`): 6,000 of the 11,412 local
`production_v2` designed-route episodes, 4,000 of the 9,600 `production_v4` planner-proposal episodes, 8,000 of the 15,235 CRM
episodes (random subsets, seed 0). Python: `/home/harry/miniconda3/envs/nedm/bin/python`.

---

## 0. TL;DR

1. **Energy signal.** `outcome.json:positive_work_kj` (W+) = sum of `trajectory.npz:positive_work_kj_per_interval` exactly; it is
   positive engine-output torque x motorshaft speed integrated at the physics substep (2 ms rigid, 1 ms CRM). The 20 Hz
   `power_kw` reproduces W+ to +-2 %, and `state[:,15]*state[:,16]/1000` (engine speed x torque) is *identical* to `power_kw`
   (ratio 1.000, r 0.99999), so the 17-D state carries the power. Negative (braking) work is 7 % of W+ on rigid, 1 % on CRM.
2. **Energy is dominated by stalls, not geometry.** Rigid designed routes: clean success median 241 kJ (5.6 kJ per metre of
   route); failure median 2,153 kJ (49 kJ/m), of which 92 % is spent *after* the first stall/rollback event and 48 % (median;
   p90 73 %) while strictly stalled (|vx| < 0.3 m/s at throttle > 0.3) at 60-70 kW. Among successes, route geometry
   (length, climb, v^2 L) explains only R^2 0.13 of W+ on rigid; elapsed time + stalled seconds take it to 0.90
   (corr(W+, time) 0.84, corr(W+, route length) 0.005). On CRM soil geometry matters more (R^2 0.29, 5.9 kJ/m of length,
   53 kJ per metre of climb; corr(W+, L) 0.43), successes cost 2x rigid per metre (11.7 kJ/m), and failures are truncated by the
   34 s blockage/breakthrough rules (median 727 kJ, 59 % stalled).
3. **Speed trade-off, paired within start-goal group (both routes clean).** Rigid: 6 m/s costs x1.30 (p10-p90 0.70-1.67) the
   W+ of 2 m/s at x0.37 the time; 4 vs 2: x1.13; smooth 2-6-2 vs 2: x1.22. CRM: 6 vs 2 x1.03 (0.86-1.35), 4 vs 2 x1.01 -
   on soil the speed penalty nearly vanishes because sinkage/rolling resistance scales with distance. Energy-min therefore
   favours slow routes on rigid, which are the routes that fail (2 m/s: 27 % fail rigid, 87 % CRM). An energy score can only
   be used behind a risk gate and against a deadline/time weight (the A0 lesson, section 2).
4. **The energy branch was already closed negative on 09-07 (A0/A1)**, and the failure mode was the optimiser's curse: a
   learned/imagined work predictor with 20-33 % per-route error, argmin'd over a bank, picks its own under-predictions and
   loses to a hand-written analytic work model by +1.5..+2.5 % W+ (CI excluding zero); the pre-registered 5 % gate failed in
   the wrong direction. Tonight must (i) use W+, never signed work; (ii) benchmark any energy head against the analytic
   baseline `scripts/traverse_wp9_analytic.py` refit on f104; (iii) report predicted/true at the pick vs over all candidates;
   (iv) compare energy only among paired safe arrivals at the same goal; (v) censor energy explicitly at termination.
5. **Velocity/re-anchoring.** Every existing sample starts from rest (ctx vx in [-0.2, 0.33] m/s rigid, [-0.09, 0.09] CRM), which
   is why night 2 found the 17-D state carries nothing (AUC .948/.954/.947 with full/chassis/no state). Cutting episodes every
   2 s before the event gives 5.4 (designed) / 10.8 (on-policy) / 6.5 (CRM) anchors per episode with vx spanning 0-6.3 m/s;
   positive rates 22 % / 49 % / 64 % (episode level 24 / 58 / 68 %). vx at the anchor is a very strong predictor by itself
   (rigid: 95 % positive below 0.5 m/s, 63 % at 0.5-1.5, 32 % at 1.5-3, 10 % at 3-4.5, 2.6 % above 4.5), but most of that is
   the "already stuck" mass (event within 2 s for 99 % of anchors below 0.5 m/s), which nav_v1 already showed no route choice
   fixes. Offline evaluation must therefore be stratified by vx and the value of the corridor measured *within* vx bins.
6. **Builder cost is trivial** (station tensor 0.64 ms, projection 2 ms/episode; 18k episodes scanned in 13 s on 16 workers);
   the cost is the output: 51k episodes x ~7 anchors x 30.7 kB (float16 X) = ~10.5 GB, and `f104_n2_train.Data` casts X to
   float32 in RAM (x4). Cap anchors (e.g. 4 per episode incl. t = 0 -> 6.3 GB) or build corridors on the fly in the loader.

---

## 1. Energy signals: what exists and how it is computed

### 1.1 Per-frame arrays (`trajectory.npz`, 20 Hz, one row per 50 ms control interval)

Rigid collector `scripts/traverse_fdm_rgbd_diverse_chrono.py` (`DT = .05` line 28, `SETTLE_S = .8` line 29; physics
`step_size_s` 2 ms -> 25 substeps per frame, line 195-196); CRM collector `scripts/crm_collect.py` (imports the rigid
collector's DT/SETTLE/driver/route reader; physics 1 ms from `artifacts/traverse/crm_f104_v1/configs/crm_main.json:5` -> 50
substeps; soil depth 0.24 m line 4).

| key | shape | definition | where |
|---|---|---|---|
| `state` | (n,17) float32 | `STATE_FIELD_PRESETS["tire_normal_force_omega_pt"]`: 0 `vel_body_x_mps`, 1 `vel_body_y_mps`, 2 `roll_rad`, 3 `pitch_rad`, 4 `roll_rate_radps`, 5 `ang_vel_body_y_radps`, 6 `yaw_rate_radps`, 7-10 tire fz (fl,fr,rl,rr), 11-14 spindle omega, **15 `engine_motor_speed_radps` = `engine.GetMotorSpeed()`**, **16 `engine_motorshaft_torque_nm` = `engine.GetOutputMotorshaftTorque()`** | rigid :231-232 (sampled at substep 0 of each frame); CRM `measure()` :217-223 |
| `action` | (n,3) | `[steering, throttle, braking]` (throttle = column 1) | rigid :235; CRM :226 |
| `pose` | (n,3) float64 | `[x, y, yaw]` at frame start | rigid :234 |
| `power_kw` | (n,) | `engine.GetOutputMotorshaftTorque() * transmission.GetOutputMotorshaftSpeed() / 1000` at substep 0 (signed) | rigid :236; CRM :245 |
| `positive_work_kj_per_interval` | (n,) | `sum over substeps of max(p, 0) * dt`, same product evaluated every substep (kJ) | rigid :285-286, 299-300; CRM :267-268, 273-274 |
| `parked` | (n,) bool | `at_end` = within 3 m of the last waypoint and past waypoint N-2 -> desired speed 0 | rigid :217-218 (appended :301) |
| `contact_n` | (n,) | asset contact (rigid); zeros on CRM | rigid :293-302; CRM :334 |
| `terminal_state`, `terminal_pose`, `terminal_parked`, `dt_s`, `state_fields` | | endpoint after the last interval | rigid :325-333 (CRM :313-316) |

`outcome.json:positive_work_kj` = the running sum of the per-interval work (rigid :380, CRM :366; `policy.check` at CRM :304); definition string at rigid
:384: "Positive engine output torque times transmission motorshaft feedback speed, substep-integrated; not fuel energy".
Also present: `status`, `goal_time_s`, `elapsed_s`, `path_length_m`, `net_displacement_m`, `goal_progress_m`,
`longest_consecutive_effortful_near_zero_speed_s`, `bounded_blockage_v1(_windows)`, `low_net_progress_4s_windows`.
CRM adds `crm.{max_wheel_sinkage_below_bmp_m, max_abs_slip_ratio_p95, ...}` and `crm_extra.npz` (`pos_z_m`,
`bmp_ground_z_m`, `quat`, `spindle_z_m`, `slip_ratio` (n,4), `fsi_force_wheel_fx_n` (n,4)).
`command_reference.npz`: `interval_start_s`, `desired_speed_mps` (n,), `reference_waypoints` (N,2), `reference_stations`,
`reference_speeds`, `reference_headings` (N ~ 82-92 at 0.5 m).

SCAN consistency: `sum(positive_work_kj_per_interval)/positive_work_kj` = 1.000 (min-max) on all three sets; 20 Hz
`sum(max(power_kw,0))*0.05 / W+` p10/50/90 = 0.981/1.000/1.007 (rigid), 0.994/0.999/1.003 (CRM);
`sum(max(state15*state16,0))*0.05/1000` / the same from `power_kw` = 1.000 at p10/50/90, r = 0.99999 with W+ on every set ->
`engine.GetMotorSpeed()` and `transmission.GetOutputMotorshaftSpeed()` coincide here (the A0 audit asked exactly this question,
`wp4_implementation_notes.md:2107-2110`). Negative work / W+: rigid designed p50 6.9 % (p90 16 %), on-policy 3.9 %, CRM 0.8 %.

Physical scope (from `docs/vision/hmmwv_traverse/fdm_f104_50h_collection_20260909.md:59` and
`wp4_implementation_notes.md:2107-2111`): mechanical work at the engine-transmission motorshaft, upstream of the torque
converter (positive tyre-contact work was 58 % of positive shaft work in the older study); the Chrono engine is a torque map plus
a losses map, no fuel model. Peak power in the data: 135.6 kW (rigid) / 136.2 kW (CRM) at full throttle.

### 1.2 Stop rules that bound a failed episode's energy (they define the censoring)

`scripts/gen_collect.py:54-150 StopPolicy` (reused verbatim by CRM, `crm_collect.py:404, 415`): a bounded-displacement window
(2 s, <= 0.25 m diameter, all throttle > 0.3, not parked) may only start counting after `minimum_elapsed_s` 24 s, then 2 s
confirmation, then an 8 s recovery tail -> earliest `prolonged_blockage_terminated` at 34 s; horizon 120 s (`timeout`);
`terrain_bounds_exit` at |x|,|y| > 40 m; `rollover` at |roll| or |pitch| > 60 deg (rigid :314-315, CRM :298-299). CRM adds
`soil_breakthrough_terminated` (any wheel > 0.24 + 0.06 m below the BMP surface for 5 frames, `crm_collect.py:281-292`),
which ends 55 % of CRM episodes (all preceded by a stall). Consequence: a rigid failure spins its wheels for >= 10-34 s at
~60-70 kW before the runner stops it, so failure energy is an artefact of the stop rule, not a property of the route.

### 1.3 SCAN: distributions (kJ unless noted; p10 / p50 / p90)

| set (n) | class | W+ | elapsed s | W+ / route length kJ/m | stalled s | E_stalled/W+ | E_slow(|vx|<1)/W+ | E before first event / W+ | mean kW moving / stalled |
|---|---|---|---|---|---|---|---|---|---|
| rigid designed (6,000; fail 11.3 %, unsafe 23.7 %) | success (5,322) | 150 / 241 / 908 | 7.7 / 12.4 / 27.5 | 3.2 / 5.6 / 20.3 | 0.25 / 0.3 / 1.45 | .05 / .10 / .16 | .10 / .17 / .35 | - | 19 / 69 |
| | unsafe-but-reached (742) | 684 / 1,228 / 2,526 | 19 / 31 / 56 | 15 / 28 / 61 | 1.0 / 2.4 / 11.3 | .06 / .12 / .29 | .20 / .37 / .63 | .08 / .18 / .38 | 38 / 62 |
| | fail (678) | 932 / 2,153 / 5,897 | 34 / 49 / 120 | 20 / 49 / 132 | 13.7 / 20.5 / 37 | .16 / .48 / .73 | .43 / .80 / .94 | .03 / .08 / .23 | 38 / 61 |
| rigid on-policy (4,000; fail 34 %, unsafe 58 %) | success (2,638) | 130 / 276 / 1,954 | 15 / 31 / 68 | 2.8 / 6.2 / 44 | 0.25 / 0.5 / 10.3 | .03 / .11 / .34 | .06 / .25 / .69 | - | 10 / 57 |
| | fail (1,362) | 565 / 1,698 / 4,504 | 34 / 53 / 120 | 12 / 38 / 98 | 14 / 22 / 41 | .28 / .63 / .89 | .62 / .90 / .98 | .03 / .08 / .23 | 19 / 52 |
| CRM (8,000; fail 68 %) | success (2,564) | 395 / 530 / 743 | 9.4 / 13.7 / 27 | 9.0 / 11.7 / 16.6 | 0.3 / 0.3 / 0.35 | .03 / .05 / .06 | .06 / .09 / .17 | - | 38 / 71 |
| | fail (5,436) | 509 / 727 / 1,042 | 12 / 21 / 34 | 11 / 17 / 25 | 4.7 / 8.6 / 23 | .41 / .59 / .75 | .52 / .69 / .83 | .28 / .45 / .63 | 27 / 50 |

"stalled" = |vx| < 0.3 m/s and throttle > 0.3 and not parked (the labeller's near-stop test, `f104_n2_dataset.py:105`; collector 'slow' rigid :311, CRM :295); "first
event" = the labeller's `ev` (rollback or 1 s near-stop after the 1 s settle, `:103-109`). E_parked = 0 everywhere (desired
speed 0 while parked). Failures' high-water-mark station (`hwm/L`): rigid 0.22 / 0.46 / 0.75, CRM 0.19 / 0.42 / 0.69.

Dependence on commanded speed (clean successes only). Unpaired medians hide a survivorship confound (slow routes that
succeed are the easy ones): rigid W+ by profile constant_2 / 4 / 6 / smooth = 268 / 234 / 246 / 233 kJ (p90 1,840 / 800 / 357 /
735), 6.1 / 5.4 / 5.6 / 5.5 kJ/m; CRM 450 / 527 / 563 / 513 kJ, 10.0 / 11.5 / 12.6 / 11.3 kJ/m. Paired within (group, lateral
offset), both clean: see TL;DR item 3 (rigid n = 196 pairs for 6 vs 2, CRM n = 49). Time ratios are fixed by the profile:
6 vs 2 x0.37, 4 vs 2 x0.52, smooth vs 2 x0.63.

Dependence on grade (route climb from the static map along the commanded waypoints; median climb 2.9 m rigid, 2.4 m CRM):
least-squares on clean successes, W+ ~ [L, climb, v^2 L, vmax^2, T, descent, stalled_s]:

| set | L only | L + climb | + v^2 L | + vmax^2 + T | + descent + stalled_s | corr(W+, climb) | corr(W+, T) | corr(W+, stalled_s) |
|---|---|---|---|---|---|---|---|---|
| rigid designed | R^2 0.000 | 0.067 | 0.129 | 0.887 | 0.898 (rmse 172 kJ) | 0.26 | 0.84 | 0.75 |
| rigid on-policy | 0.000 | 0.074 | 0.120 | 0.619 | 0.871 | 0.27 | 0.68 | 0.80 |
| CRM | 0.183 (5.9 kJ/m) | 0.277 (+53 kJ per m climb) | 0.287 | 0.406 | 0.416 (rmse 109 kJ) | 0.41 | 0.28 | 0.27 |

Reading: on rigid f104 the energy of a *successful* route is "time x ~20 kW + stalled seconds x ~60 kW" - i.e. a second,
noisier reading of the risk label - and the between-route headroom at equal risk is the +-30 % paired speed effect above. On
CRM the energy is closer to a terrain quantity (rolling resistance from sinkage, 2x the rigid cost per metre) and the head has
something to learn that the hazard head does not already encode. m g = 25 kJ per metre of climb; the fitted 53 kJ/m is
2.1x that (slip + drivetrain), plausible.

Cumulative energy along the route (successes, median fraction of the episode's W+ reached by first arrival at station
24 / 48 / 72 of 96): rigid 0.43 / 0.75 / 0.98; on-policy 0.25 / 0.65 / 0.98; CRM 0.36 / 0.65 / 0.92. The last quarter is almost
free on rigid (terminal deceleration cone, `f104_n2_sampler.shape` :49). For failures the energy accumulated by the censor
station (hwm) is only 7 % of the episode's W+ on rigid (42 % on CRM): 93 % of a rigid failure's energy is unobservable as a
"cost of driving that station" - it is the cost of being stuck.

GOTCHA (station arrival): the projected station of a *successful* episode never reaches L because the run stops inside the
2.5 m goal radius (`reached all 96 stations` = 0.04 % rigid, 0 % CRM). Define arrival at the last station := `goal_reached`.

### 1.4 Candidate energy targets (definitions)

Let `s(t)` = projected station of the pose on the route (`f104_n2_dataset.project` :71-79), `smax(t) = max_{t'<=t} s(t')`
(first-arrival envelope, so rollbacks do not re-count stations), `cumE[k] = sum_{i<k} positive_work_kj_per_interval[i]`.
Station grid `g_j = j L / 95`, `j = 0..95`; arrival frame `a_j = first k with smax[k] >= g_j` (for `goal_reached` episodes set
`a_95 = n`, and any `g_j` beyond `smax[-1]` also -> `n`).

* **T1 - route energy to goal** `E_goal = W+` (outcome), observed only for `goal_reached`; unsafe-but-reached episodes carry
  their struggle energy (x5 a clean success) - honest as a cost, but it makes the energy head a second risk head.
* **T2 - per-station first-arrival increments (recommended)** `dE_j = cumE[a_j] - cumE[a_{j-1}]` (kJ), observed for
  `j <= j_cens`, `j_cens = 95` for successes, `= event_idx` (the hazard head's station, `f104_n2_dataset.py:122`) otherwise;
  the loss is masked beyond `j_cens`. Two sub-variants: (a) **clean-driving energy**: censor at `event_idx` for *every* episode,
  including unsafe-but-reached ones, so the energy head learns "cost per station when the route is driven as commanded"
  and the hazard head owns the event (no double counting; the planner objective is then `P(event) * penalty + E_clean +
  w_t * time`); (b) censor only for `goal not reached`. Recommend (a) as the primary, (b) as a read-out.
* **Units: predict kJ per metre and multiply by the station spacing** `ds = L_remaining / 95`. With re-anchored samples (section
  3) `ds` ranges 0.08-0.7 m, so per-station kJ targets would vary 10x for identical terrain; kJ/m is invariant. Typical values:
  rigid 3-20 kJ/m (p10-p90), CRM 9-17 kJ/m; the launch station and climbs are the peaks. Use softplus outputs, Huber (or
  log1p) loss - the distribution is heavy-tailed (p90/p50 of W+ = 3.8 on rigid).
* **T3 - route-level surrogate for the planner** `E_route = sum_j softplus(e_j) * ds`, and a **time** estimate is free from
  the same arrival frames (`T_j = a_j * DT`; T at the last station = `goal_time_s`), which the A0 deadline needs.
* Do not define the target from `power_kw` sampled at 20 Hz when the substep-exact interval work exists; do not integrate past
  the terminal frame (the wp8 head bug, x2.95, `wp4_implementation_notes.md:2191-2194`).

---

## 2. Prior energy work: what was settled and what must not be repeated

Commit `6abf6ee` (2026-09-07, "Energy branch closed negative (A0/A1), 47 deg crater trap verified (B1), feature-mirror defect
fixed"; `git show --stat 6abf6ee`): scripts `traverse_wp9_{truth,analytic,arm_cheap,arm_nrd,a1_table,cheap_features}.py`,
artefacts `artifacts/traverse/wp9_energy/{truth_summary.json,a1_table.{json,txt}}`, narrative
`docs/vision/hmmwv_traverse/wp4_implementation_notes.md` section 14 (lines 2069-2260), plan
`docs/vision/hmmwv_traverse/energy_and_crater_experiment_plan.md` (A0-A3). Earlier: WP4 section 3 (:110-141, power head does not
transfer into imagination), section 7 (:394-429, the sampler exploiting a throttle-based energy estimate, ratio > 1000), section
8.2 (:457-470, CEM "8 % imagined gain" = the curse), 8.8 (:612-619, geometry floor `wp5_energy_floor/energy_floor.json`: geometry
R^2 0.60 / 0.51, sigma 36 kJ), `wp4_power_calib/power_calib.json` (linear power models: kinematic features R^2 0.48/step,
kinematic + actions 0.82, episode-energy corr 0.94-0.97 on recorded data).

* **A0 (`:2105-2127`)**: all earlier energy numbers were *signed* shaft work; switching to W+ changes the best candidate on
  33 % of layouts (27 % among deadline-compliant), so nothing historical can be reinterpreted. Over 5,745 runs on f101-f111:
  W+ 384.6 kJ, W- -33.3 kJ. Terminal kinetic energy was uncontrolled (median 6.8 % of W+). The mission the code never had:
  minimise W+ subject to reaching the same goal by a deadline `K (L_min/V_ref + V_ref/a)` (V_ref 5 m/s, a 1.5; K = 1.0 was the
  floor of the grid, i.e. no slack) and the same terminal speed. **Rule for tonight: W+ only; energy comparisons only among
  paired safe arrivals at the same goal under the same deadline; a failed or shortened drive never earns a saving** (also
  `mppi_online_cohort_reporting_20260909.md:7`, `fdm_diverse_results_20260909.md:56`).
* **A1 (`:2129-2198`)**, 385 layouts, one shared bank, one pick per arm, oracle-feasible regime: analytic model regret 9.55 kJ
  (5.5 %) < imagination fine-tuned 11.20 < direct learned predictor 13.13 < imagination frozen 13.23 < fastest 17.47. Paired
  against the analytic model the imagination costs +1.5 to +2.5 % *more* work (CIs exclude zero), at every deadline slack
  0.8-2.0; against the direct learned predictor it ties (-0.3 %, CI -1.25..+0.86). Measured cause = optimiser's curse:
  pred/true at the pick minus pred/true over all candidates = -0.002 (analytic), -0.015 (learned), -0.086 / -0.180
  (imagination). "On this family, work is mostly climb plus acceleration - geometry readable from a height map". The
  pre-registered gate (>= 5 % paired W+ reduction vs the strongest cheap baseline) failed in the wrong direction.
  **Do not repeat**: an energy predictor evaluated by its own argmin without (i) the analytic baseline
  (`traverse_wp9_analytic.py:13-30`: positive tractive work from grade + rolling + drag + acceleration, its square, low-/high-
  speed variants, time, cornering `m v^2 kappa`, cross-slope; NNLS on W+), (ii) the curse metric, (iii) MAPE per route, (iv) a
  time/deadline constraint. The learned direct predictor's own MAPE was 18 %, the analytic 16.7 %.
* Earlier online result (`fdm_diverse_results_20260909.md:54-56, 87, 98`): energy weights 0.2 / 0.5 in the MPPI cost reduced safe
  completion 4/6 -> 2/6 -> 1/6; the default 0.02 saved 0.33 % work on 4 safe pairs. "Energy-aware benefit: not met."
* Later in the same worktree the energy question was dropped from the f104 line entirely: the night-2 model scores risk only,
  `gen_planner.plan` picks argmin risk with no time term (`docs/progress.md:361`). The SCAN above shows why an energy head
  on rigid f104 would mostly re-learn the hazard: energy is time-plus-stall. The CRM soil is the one place (2x cost per metre,
  geometry R^2 0.29, speed penalty ~0) where an energy score is not just risk in disguise - but there the between-route
  spread at equal risk is only +-15 % (p10-p90 of the paired ratio 0.86-1.35), against a 23-point risk gap between planners.
  Pre-register a small, honest gate (e.g. >= 5 % paired W+ at equal goal-reached rate and <= +10 % time) before looking.

Documentation caveats to carry: review `NRD_hmmwv_traversal_study_plan_review.md:152-156` (signed power is not consumption;
an energy regression must include speed, longitudinal acceleration, slope along heading and curvature, else acceleration energy
is attributed to terrain); `mppi_feasibility_20260908.md:71` ("Positive shaft work must be integrated from positive power").

---

## 3. Velocity input and re-anchored (moving-start) samples

### 3.1 What the current pipeline does with vehicle state (and why it is blind to velocity)

* Dataset row (`scripts/f104_n2_dataset.py:91-127 one()`): `ctx = [anchor state (17), goal dx, dy, |d|, start_yaw, route_len]`
  (22-d, :117) where the anchor state is `anchor_state.npz:state` = the settled rest state at frame 0 (rigid :238-241). `X` =
  `station_tensor(wp, sp, st)` (:45-60): 96 stations uniformly in arc length over the *whole* route, 32 lateral samples over
  +-6 m, channels `[elev - e0, grade along (clip +-2), cross slope, commanded speed, valid]`; `e0` = elevation at station 0
  centre (:53); the speed channel is interpolated on the route's own cumulative station (:57-58). Label `event_idx =
  round(hwm/L * 95)` with `hwm = max projected station up to the event` (:111-113, :122), `-1` if clean.
* Trainer `scripts/f104_n2_train.py:21-23`: `CTX_COLS = {'full': 0..21, 'chassis': [0,1,2,3,4,5,6] + [17..21], 'none': [17..21]}`;
  the deployed N2 ensemble is `ctx='none'` (5-d geometry). `Data` (:73-92): fit = `split=='train' & ~dev_group(group)` with
  `dev_group = md5(group) % 5 == 0` (`f104_night_train.py:26-27`); dev fold = designed routes only; **X cast to float32 in
  RAM (:84)**; ctx standardised on the fit rows (:89-91). Loss `survival_nll` (`f104_night_train.py:63-72`): sum of softplus
  hazard over stations `< ev` (clean: `<= 95`) plus `softplus(-haz[ev])` at the event station.
* Model `scripts/gen_riskmodel.py:15-56`: ctx enters through `Linear(nctx, 32)` (:25) and is broadcast to every station (:48);
  `route_logit = log(sum softplus(haz))` (:10-13); heads :31/:37/:40.
* Night-2 result (`night2_v1/REPORT.md:16-20`): full 17-D state .948, chassis .954, none .947 (dev AUC, 3 seeds) - "every
  episode starts from the same settled rest state (vx, roll, pitch IQR ~0.07)". Confirmed by the datasets: rigid `ctx[:,0]`
  in [-0.20, 0.33] m/s, CRM [-0.09, 0.09]. **A velocity input cannot be learned from the existing rows.**
* Deployed replanner `scripts/nav_online.py`: `Navigator.decide(pose, goal, rng, v_now=0.0, ...)` (:333-392) builds fresh
  candidates from the *pose* (`P.base_route(pose, goal)`), scores them with `P.geom_ctx(pose[:2], goal, pose[2], L)` (5-d, no
  velocity; :376), optionally adds the trimmed current route (`trim_route` :269-280: nearest waypoint, re-stationed, minimum
  8 m remaining) and keeps it unless a candidate beats it by `switch_margin`. `reachable_speed` (:167-175) is OFF by default -
  the docstring (:24-31) records the trap: clipping candidate speeds to the acceleration cone from `v_now` pins the command to
  the measured speed (0.3 m/s after 5 s) because the follower reads its speed command at the vehicle's own station.
  `validate_reference` (`src/nedm/traverse/fdm_mppi.py:121-141`) cuts the route at `meta.fdm_station` or the nearest point to
  `anchor_pose` (:124-130) and checks curvature/speed/acceleration *between stations only* - a 6 m/s candidate is valid while
  the vehicle does 1 m/s.
* nav_v1 closed-loop evidence (`nav_v1/REPORT.md:243-268`): decisions taken above 2 m/s are followed by a slide at 1.2 %
  (10/802), the same as waypoint decisions (0.7 %); below 2 m/s 53 % (78/147). Its conclusion: "no targeted collection or
  retraining is warranted for the moving-start shift" with the whole arena visible. Tonight's velocity input is therefore a
  hypothesis the closed-loop data did not demand; the offline gain will be large and mostly trivial (section 3.4) unless
  evaluated within speed strata.

### 3.2 Precise definition of a re-anchored sample at frame k

Given an episode (arrays above), route `(wp, sp, st)` from `command_reference.npz`, goal `case.json:goal_xy`:

1. **Projection**: `s, dev, L = project(pose[:, :2], wp)`; `smax = maximum.accumulate(s)`; `s_k = smax[k]`, `rem = L - s_k`.
   Lateral deviation at anchors is small (p50/p90: 0.15/0.41 m rigid designed, 0.28/0.73 on-policy, 0.25/0.88 CRM), so start
   the remaining route at the **projection point** (interpolated), not at the vehicle: the follower tracks the route, and the
   candidates the planner would propose at deployment start at the pose anyway. (`trim_route` uses the nearest waypoint;
   the difference is <= 0.25 m.)
2. **Remaining route**: `i = searchsorted(st, s_k, side='right')`; `wp2 = [interp(s_k) ; wp[i:]]`, `sp2 = [interp(s_k, st, sp) ;
   sp[i:]]`, `st2` recomputed from `wp2`. The first commanded speed is the profile's speed at the anchor station, not `vx[k]`;
   this is the deployment situation (candidates command speeds the vehicle is not at). In the recorded data the gap
   `cmd - vx` at anchors is p10/50/90 = -0.42/0.09/1.11 m/s (rigid), -0.14/0.20/1.86 (CRM); 13-18 % of anchors have |gap| > 1
   m/s. COVERAGE GAP: a mid-route switch from a 2 m/s route to a 6 m/s one (gap 4 m/s) never occurs in the recordings.
3. **Corridor**: `X, L2 = station_tensor(wp2, sp2, st2)` -> (5,96,32); `e0` becomes the elevation under the vehicle; the station
   spacing becomes `rem/95` instead of `L/95`. Full routes have `ds` 0.33-0.69 m (L p5-p95 31-65 m); with a minimum remaining
   length of 12 m `ds` goes down to 0.13 m. The conv stack (3x3, stride 1 along stations; Conv1d k = 5; BiGRU) works in station
   units, so physical receptive fields shrink 2-5x on short remainders. Options: (A) drop-in - keep 96 over the remaining
   length and rely on `route_len` in ctx (this is what `trim_route` + `Navigator.decide` already do at deployment, so the
   deployed system is already in this regime, untested); (B) fixed physical spacing (e.g. 0.5 m, 96 stations = 48 m window,
   `valid = 0` and loss-masked beyond the goal; 25 % of full routes exceed 48 m -> use 128 stations or 0.7 m). Recommend
   A for tonight (no planner change; measure AUC as a function of remaining length), B as the arm to try if A degrades below
   ~20 m. Under either, express the energy target per metre (section 1.4).
4. **Context**: `ctx = [state[k] (17), goal - pose[k,:2], |goal - pose[k,:2]|, pose[k,2], L2]` - same 22-d layout, so
   `CTX_COLS['chassis']` (= vx, vy, roll, pitch, roll rate, pitch rate, yaw rate + 5 geometry) is the velocity variant with zero
   trainer changes; add `'vel': [0, 1, 6] + GEOM` (vx, vy, yaw rate) as the minimal variant. At deployment `gen_planner.geom_ctx`
   (5-d) must be extended with the measured state; `RiskModel.score` already standardises with the checkpoint's `ctx_mu/sd`.
   Anchor distributions (SCAN): |vy| p50/90 0.07/0.26 m/s, |yaw rate| 0.04/0.15 rad/s (rigid); 0.08/0.38, 0.04/0.16 (CRM);
   pitch at anchors is informative on CRM (the stall precursor is a 10-25 deg climb, `crm_f104_v1/REPORT.md` section 3).
5. **Label**: same event rule on the remaining part: `event_idx_k = round((s_ev - s_k)/rem * 95)` clipped to [0, 95] with
   `s_ev = smax[ev]`; `-1` if the episode is clean. Anchors with `k >= ev` are **dropped** (the vehicle is already in the event;
   the remaining-route question is moot and the survival mask would be empty). For failures with no rb/ns event
   (timeout/bounds/rollover, `ev = n-1` at `:105`) the rule gives the furthest station reached, as today.
6. **Energy target** per section 1.4 relative to the anchor: `E_j = cumE[a_j] - cumE[k]` on the remaining grid, censored at
   `event_idx_k` (variant a) - the observed part of a failed episode is then the clean driving up to the stall, which is exactly
   the increment the planner wants.
7. **Time target** (free): `T_j = (a_j - k) * DT`.

### 3.3 Sampling density, counts, class balance (SCAN, anchors every 2 s = 40 frames from k = 40, pre-event, remaining >= 5 m)

| set | anchors / episode (mean, median, max) | vx at anchor p5/25/50/75/95 (m/s) | positive rate (episode-level unsafe) | positives with event within 2 s / within first 10 % of remaining | anchors from failing episodes |
|---|---|---|---|---|---|
| rigid designed | 5.4 / 5 / 19 | 1.41 / 1.91 / 2.55 / 4.23 / 6.30 | 22.2 % (23.7 %) | 19 % / 23 % | 10 % (episode fail 11 %) |
| rigid on-policy | 10.8 / 9 / 59 | 0.29 / 0.50 / 0.91 / 1.91 / 3.50 | 49.4 % (57.6 %) | - / 22 % | 30 % (34 %) |
| CRM | 6.5 / 5 / 54 | 0.30 / 0.68 / 1.78 / 2.80 / 4.89 | 63.9 % (67.9 %) | 16 % / 23 % | 64 % (68 %) |

Positive rate by vx bin (rigid designed / on-policy / CRM): < 0.5 m/s: 95 / 63 / 85 % (share of anchors 1 / 25 / 17 %);
0.5-1.5: 63 / 52 / 78 %; 1.5-3: 32 / 39 / 64 %; 3-4.5: 10 / 28 / 40 %; > 4.5: 2.6 / 20 / 17 %. Time-to-event after positive
anchors p10/50/90: rigid 1.0 / 5.2 / 13 s (below 0.5 m/s: median 0.25 s, 99 % within 2 s; 1.5-3 m/s: 6.6 s), CRM 1.3 / 6.7 /
23 s (below 0.5 m/s: median 12.9 s - on soil the crawl persists before the dig-in). On-policy routes command 0.5-0.8 m/s in
their slow bins (`V_MIN = 0.5`, `f104_n2_sampler.py:24`), so vx < 0.5 there is "crawling as commanded", not stuck.

Recommendations: every 2 s (adjacent 1 s anchors are near-duplicates: the corridor shifts 2-6 m), always include k = 0 (the
existing row), cap at 6-8 anchors per episode (long slow on-policy episodes reach 59) chosen stratified by vx, minimum
remaining length 12 m. Keep the slow anchors in training (they teach the velocity dependence) but tag `vx_anchor`,
`time_to_event`, `rem`, and evaluate ranking within vx strata (>= 1.5 m/s and event > 2 s ahead) - otherwise the AUC gain is
the trivial stuck-vs-moving split. Weight anchors so that each episode contributes equally (failing episodes contribute
fewer anchors: they end at the event), or the dataset drifts toward easy, long, fast episodes.

Fundamental limit: at a given anchor state only *one* continuation was driven, so the recorded data can never show whether
route A beats route B *from that state*. Same-group comparisons at the same anchor time have different states. The velocity
input can be validated offline only through calibration/AUC across anchors; the ranking question is closed-loop
(replanning at 1-2 Hz on limited-range sensing, where nav_v1 says the gap is, `nav_v1/REPORT.md:367-372`).

### 3.4 Leakage rule

* All anchors of an episode -> the episode's split; all episodes of a start-goal group -> the group's `case.json:split`
  (already the rule: rigid 2,700 groups, CRM 1,200 groups, `dev_group` by md5 of the group id).
* **The CRM and rigid datasets disagree on 111 groups**: CRM's 1,200 groups are exactly the night-2 A1 groups, which are all
  `train` in `night2_v1/station_ds_all.npz`, while the CRM dataset re-designates 56 of them `val` and 55 `test`
  (`crm_f104_v1/REPORT.md` section 6 item 5 documents the consequence). 15,024 route ids are shared. If both worlds are trained
  or compared together, adopt the CRM split for those 1,200 groups (move the 111 groups' rigid rows out of training).
* Twin routes (same id, rigid vs CRM) share terrain and geometry; never put one twin in train and the other in test when the
  terrain-only channels are the input.
* Anchors from the same episode are strongly correlated (same terrain, same outcome) - never split them, and count
  episodes, not anchors, when quoting effective sample sizes.

### 3.5 Builder pseudo-code (numpy only, matches `f104_n2_dataset.one` conventions)

```python
N_STATION, DT, S0 = 96, 0.05, 20          # S0 = 1 s settle, as in f104_n2_dataset

def cut_route(wp, sp, st, s_k):
    i  = np.searchsorted(st, s_k, side='right')
    p0 = np.array([np.interp(s_k, st, wp[:, 0]), np.interp(s_k, st, wp[:, 1])])
    wp2 = np.vstack([p0, wp[i:]]); sp2 = np.r_[np.interp(s_k, st, sp), sp[i:]]
    st2 = np.r_[0., np.cumsum(np.linalg.norm(np.diff(wp2, axis=0), axis=1))]
    return wp2, sp2, st2

def anchors_for_episode(d, every_s=2.0, min_rem_m=12.0, cap=8):
    z = np.load(d + '/trajectory.npz'); o = json.load(open(d + '/outcome.json')); c = json.load(open(d + '/case.json'))
    wp, sp, st = load_route(d + '/command_reference.npz')
    S, A, P = z['state'].astype(float), z['action'].astype(float), z['pose'].astype(float)
    iw = z['positive_work_kj_per_interval'].astype(float); n = len(S); vx, thr = S[:, 0], A[:, 1]
    fail = o['status'] != 'goal_reached'
    back = (vx < -.10) & (thr > .3)
    rb = first_run(back | (vx < -.30), 1, S0); ns = first_run((np.abs(vx) < .3) & (thr > .3), 20, S0)
    ev = min(x for x in (rb, ns) if x is not None) if (rb is not None or ns is not None) else (n - 1 if fail else None)
    if (not fail) and back[S0:].sum() * DT < .05 and vx[S0:].min() > -.30: ev = None          # clean
    s, dev, L = project(P[:, :2], wp); smax = np.maximum.accumulate(s)
    cumE = np.r_[0., np.cumsum(iw)]                       # cumE[k] = positive work before frame k
    goal = np.asarray(c['goal_xy'], float)
    out = []
    for k in [0] + list(range(int(every_s / DT), n, int(every_s / DT))):
        if ev is not None and k >= ev: break              # already in the event: not a planning state
        s_k = smax[k]; rem = L - s_k
        if rem < min_rem_m: break
        wp2, sp2, st2 = cut_route(wp, sp, st, s_k)
        X, L2 = station_tensor(wp2, sp2, st2)             # (5,96,32); e0 = elevation under the vehicle
        ev_rel = -1 if ev is None else int(np.clip(round((smax[ev] - s_k) / rem * (N_STATION - 1)), 0, N_STATION - 1))
        grid = s_k + np.linspace(0., rem, N_STATION)
        a = np.searchsorted(smax, grid, side='left')      # first frame at/after each station
        if not fail: a[a >= n] = n                        # goal arrival counts as reaching the last stations
        E = cumE[np.minimum(a, n)] - cumE[k]; T = (np.minimum(a, n) - k) * DT
        dE = np.diff(E, prepend=0.)
        j_cens = N_STATION - 1 if ev is None else ev_rel  # variant (a): clean-driving energy; use 95 for successes in (b)
        obs = np.arange(N_STATION) <= j_cens
        ctx = np.r_[S[k], goal - P[k, :2], np.linalg.norm(goal - P[k, :2]), P[k, 2], L2].astype(np.float32)   # 22-d
        out.append(dict(X=X.astype(np.float16), ctx=ctx, event_idx=ev_rel, dE_per_m=(dE / (rem / (N_STATION - 1))).astype(np.float32),
                        T=T.astype(np.float32), obs=obs, k=k, vx=vx[k], rem=rem, tte=(-1 if ev is None else (ev - k) * DT),
                        id=os.path.basename(d), group=c['id'], split=c['split']))
        if len(out) >= cap: break
    return out
```

Selection of the capped subset (if `cap` binds): keep k = 0 and the last pre-event anchor, fill the rest uniformly over the
vx-sorted remainder. Save `X` as float16 and change `f104_n2_train.Data` to keep it float16 (convert per batch).

### 3.6 Runtime and size estimate (measured on this workstation)

`station_tensor` 0.64 ms (full or cut route), `project` 2.0 ms per episode (437 frames x 92 waypoints), `np.load` of a
trajectory 0.4 ms page-cached; the SCAN (equivalent work minus the per-anchor corridor) processed 18,000 episodes in 12.6 s on
16 workers. Per episode with ~7 anchors: ~2.5 ms + 7 x 0.8 ms ~ 8 ms -> 36k rigid + 15k CRM = 51k episodes ~ 7 min single
core, **~1 min on 16 workers** if the files are page-cached; a cold read of 51k x 3 files is the real cost (minutes, not
hours; the night-2 labeller on the cluster did 36k rows in a few minutes). Output: 51k x 6.7 anchors x 30.7 kB = ~10.5 GB
float16 (`station_ds_all.npz` is 1.1 GB for 36k rows); with cap 4 -> 6.3 GB; the trainer's float32 cast would need 25-42 GB
RAM -> keep float16, or store only `(episode, k)` and build corridors in the DataLoader (0.65 ms each is cheap).

---

## 4. Concrete recommendations for tonight

1. **Energy head**: per-station softplus output in kJ/m x `ds`, target T2 variant (a) (clean-driving energy, censored at the
   hazard event station, masked loss), Huber on log1p; route energy = masked sum; also emit per-station time. Train it jointly
   with the hazard head (shared trunk, two `Linear(2*width, 1)` heads) and as a separate net; report per-route MAPE and
   pred/true at the argmin vs over all candidates (the curse metric), against (i) the A1 analytic model refit on f104 W+ and
   (ii) the trivial baselines "energy = a + b x commanded time" and "kJ/m x length". On rigid expect the head to be mostly a
   time/stall predictor (R^2 of geometry 0.13); on CRM it has real content (11.7 kJ/m, 53 kJ per metre of climb).
2. **Where energy can and cannot pay**: the paired spread between clean routes at the same goal is +-30 % on rigid (slow is
   cheaper) and +-15 % on CRM (speed-neutral). Any energy-aware pick must be gated by risk first (A1's regime a'), must carry a
   time weight or deadline, and must be judged on paired safe arrivals only, with a pre-registered gate.
3. **Velocity**: build the anchored dataset with `ctx` variant `chassis` (or `[vx, vy, yaw_rate] + GEOM`), evaluate AUC within
   vx strata and versus remaining length; the deployed `Navigator.decide` already scores trimmed routes with shrunken station
   spacing, so option A is a test of the current deployment as much as of the new input. Add `vx` to `geom_ctx` at deployment.
4. **Split**: unify the 111 mismatched groups before any rigid+CRM comparison; count episodes, not anchors.
5. **Do not**: use signed work; integrate past termination; select on a learned energy without the curse check; let an
   energy weight lower goal-reached (the 0.2/0.5 weights did, 09-09); count a failed or shortened drive as a saving; quote the
   stuck-vs-moving AUC as the value of the velocity input.
