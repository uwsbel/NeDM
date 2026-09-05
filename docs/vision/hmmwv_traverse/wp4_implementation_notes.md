# WP4 implementation notes — sensor-based planning (Planner-B) and energy calibration

**Date:** 2026-09-04 (night session) · **Modules:** `nedm/traverse/planner_b.py`,
`nedm/traverse/power_calib.py` · **Scripts:** `traverse_wp4_train_maphead.py`,
`traverse_wp4_planner_ladder.py`, `traverse_wp4_power_calib.py`,
`traverse_wp4_power_diag.py`, `traverse_wp4_score_candidates.py` (extended),
`traverse_wp3_chrono_eval.py` (`--route-file`) · **Artifacts:**
`artifacts/traverse/wp4_maphead_{v1,v2}/`, `wp4_planner_ladder_*/`,
`wp4_power_calib/`, `wp4_scores_pred_{occ,full}/`, `wp4_chrono_pred_{occ,full}/`

Session goal, set after the user's review of the WP3 result: stop feeding the
planner privileged information. Candidates had come from the oracle A* over the
true heightmap and the true obstacle list; the collision check in the rollout
scorer used the true obstacle discs. Both are replaced here by quantities decoded
from the camera. Start pose and goal stay privileged (v1 contract, plan §3).

## 1. Map head — `traverse_wp4_train_maphead.py`

Input is the per-episode static scene feature map (64×64×64, encoder stage-2,
image coordinates) that the dynamics model already indexes — camera only, one
median image per episode. Output is a world-aligned 128×128 grid (0.625 m/cell)
with two channels: obstacle occupancy logits and normalized elevation. The
image→world warp uses the flat-ground pinhole model only (no heightmap); a
5-layer conv stack (3×3, 3×3, dilated 3×3, 3×3, 1×1) absorbs the residual
perspective shift (< 1.2 m at the arena edge). Labels: analytic footprint discs
from the layout manifest (`masks.bev_occupancy`) and the arena heightmap (identical
for every episode, hence memorized; kept for the ladder's "full predicted map"
rung). Loss: BCE (pos-weight 5, rock/tree pixels ×3) + elevation MSE. 6662
training layouts, 1427 held-out; AdamW, one-cycle LR.

| head | width | steps | val IoU | rock/tree px recall | false-positive area | elevation RMSE | detect rate rock / tree / house |
|---|---|---|---|---|---|---|---|
| v1 | 64 | 4000 (2 min, 5090) | 0.730 | 0.985 | 29 m²/layout | 0.131 m | 0.999 / 0.998 / 1.0 |
| **v2** | 128 | 8000 (9 min) | **0.819** | 0.990 | **17 m²/layout** | 0.076 m | 0.999 / 1.000 / 1.0 |

An asset counts as detected when ≥ 50 % of its footprint cells are predicted
occupied. 11 366 rocks and 15 655 trees on held-out layouts: **12 rocks and 4
trees missed** by v2. IoU is bounded by boundary cells at 0.6 m resolution, not
by misses; the WP1 probe's 0.878 was at the same 128 grid from a 16×16×256 map
with a deeper decoder. Trained locally: the job is two to nine minutes, below
the cluster round-trip.

## 2. Planner-B — `planner_b.py`

Every predicted-occupied cell (probability ≥ threshold, isolated single cells
dropped) becomes a disc obstacle of radius half the cell diagonal (0.44 m), and
the **unchanged** oracle pipeline — direction-aware A*, shortcut + Chaikin,
curvature repair, footprint validation, speed profile — runs on that disc list.
Terrain comes from the true heightmap (rung "predicted occupancy + memorized
terrain") or from the predicted elevation (rung "full predicted map"; Gaussian
blur σ = 1 cell before gradients). `plan_on_predicted_map` never reads the layout
manifest. Planning cost ≈ 0.14 s per plan on CPU (220 discs vs ~12 for the oracle).

### Ablation ladder — `traverse_wp4_planner_ladder.py`, 100 held-out oracle-family layouts

Every plan is judged on the **true** map with the oracle's footprint sweep
(uninflated true footprints), against the oracle's own plan for the same layout.
Interim tracker margin 0.9 m (inflation 2.0 + 0.9). v2 head, threshold 0.85,
curvature repair 40 passes (see below).

| rung | no path | collision | true-slope violation | 5th-pct true clearance | length ratio | energy-proxy ratio |
|---|---|---|---|---|---|---|
| oracle (true map, privileged) | 2 % | 0 | 0 | 0.48 m | 1.000 | 1.000 |
| **predicted occupancy + memorized terrain** | **2 %** | **0** | 0 | 0.56 m | 1.001 | 1.002 |
| **full predicted map** | **3 %** | **0** | 8.2 % | 0.71 m | 1.001 | 1.006 |
| straight line (naive bracket) | 0 | 42 % | 11 % | −0.84 m | 0.990 | 1.044 |

The camera-derived planner finds a plan on every layout the oracle does (98/100
vs 98/100), the plan is the oracle's to within 0.1 % in length and energy proxy,
with more clearance (false-positive cells add margin), and **no collision on any
layout**. Getting there took two findings:

1. *All* failures were `validation_rejected`, never "no A* path": with ~220 cell
   discs the inflated boundary is jagged, A* hugs it, and the oracle's 12
   curvature-repair passes leave the smoothed path above the 1/8 m curvature
   cap. 40 passes rescue 7 of 9 rejected layouts (no-path 9 % → 2 %); merging
   blobs into one enclosing disc each made it *worse* (16 %) because merged
   neighbours over-inflate corridors; extra Chaikin passes or +0.5–1 m inflation
   rescue 1–3 of 9.
2. Threshold and head quality both matter before that fix (v1 head at 0.5: 17 %;
   v1 at 0.95: 10 %; v2 at 0.5: 14 %; v2 at 0.85: 9 %): misses would show up as
   collisions and there are none, so the residual is false positives.

Predicted elevation adds 1 point of no-path and an 8 % rate of plans exceeding
the true slope caps by more than the validation slack (5.7 % without the σ = 1
cell blur, at 12 % no-path). Whether those plans are drivable is the Chrono
question in §5.

### Tracker margin (plan step 5): shrink only as a rescue

Reducing `tracker_p95_margin_m` from the interim 0.9 to 0.3 outright (Chrono p95
is 0.07–0.12 m) **raised** the oracle's no-path rate from 2 % to 18 % on the
oracle-family layouts. A* then threads closer to obstacles and the
shortcut/Chaikin smoother pushes the path into the uninflated footprint, so
validation rejects it. The smoother, not the margin, is what the inflation
protects. What does work is a fallback ladder (`oracle.plan_to_ring_fallback`):
plan at 0.9, and only if nothing validates retry at 0.6, then 0.3. On the
100-layout ladder every rung then reaches **100 % feasibility with zero
collisions** (oracle 98 layouts at 0.9 + 2 at 0.6; camera-only 98 + 1 + 1;
full predicted 97 + 3); on 300 val layouts of all families the oracle's two
no-path layouts become plans. The ladder is available in the ladder script
(`--margin-fallback`) and the scorer (`--margin-fallback`). **Chrono-validated**
(`wp4_chrono_fallback`, `traverse_wp4_export_fallback_plans.py`): over all 1154
routed held-out layouts the camera-only planner needed the fallback on 34 (29 at
0.6 m, 5 at 0.3 m; 4 layouts remain infeasible). 31 of those plans were driven by
the tracker in Chrono: **31/31 completed, zero contact**, min footprint clearance
0.55 m, max cross-track 0.30 m — the tracker's real error fits inside the reduced
margins with room to spare.

## 3. Energy — why the power head fails and what transfers

Symptom (WP3 notes): imagined energy 109 kJ vs Chrono 176 kJ under the same
tracker (1.62×, corr 0.65), while replaying *recorded* actions the head matched
recorded energy (149 vs 151). The tracker's actions are outside the recorded
driver's distribution and the model's throttle response is too easy — imagined
time is also 10 % fast — so the imagined tracker reaches the speed profile with
less throttle than Chrono needs.

`power_calib.py` fits linear power models on recorded data (2000 train
episodes, 1427 val) from several feature sets and `traverse_wp4_power_diag.py`
applies them to the **imagined** trajectories of the 185 scored candidates
(`--dump-trajectories`) and compares with their Chrono energies:

| power model (fit on recorded data) | recorded val: episode-energy corr | imagined → Chrono: ratio | corr | combined-cost pick agreement |
|---|---|---|---|---|
| learned power head (WP2) | — | 1.62 | 0.65 | 6/31 |
| kinematic (vx, ax, pitch, Fz, ω) | 0.94 | 2.36 | **−0.08** | 5/31 |
| speed only | 0.62 | 2.64 | −0.32 | 3/31 |
| throttle/brake × speed ("act") | 0.94 | **1.15** | **0.80** | 14/31 |
| kinematic + actions | 0.97 | 1.21 | 0.80 | 14/31 |

Kinematic models transfer *worse* than the head even though they fit recorded
data well: the imagined vx has 0.7 m/s MAE and the imagined tire loads and wheel
speeds have MAE ≈ 75 % of their recorded standard deviation (channel table in
the diag output), so features built on them are noise in imagination. Features
built on the *commanded* throttle and brake transfer. The scorer now reports
`energy_act_kj` (used for selection) beside the head's `energy_kj`; the residual
15 % under-estimate is the model's throttle-response bias and would need
tracker-driven Chrono episodes in the dynamics training set (DAgger-style) to
remove. Combined-objective pick agreement stays low because candidates are
near-tied on time + energy/10; time-only agreement is 28/31 (WP3 notes).

## 4. Camera-only candidates in imagination — `traverse_wp4_score_candidates.py --candidates predicted --collision predicted`

Same 32 held-out layouts as the WP3/WP4 scoring. Candidates from the six-way
parameter sweep on the predicted map (v2 head, threshold 0.85), tracked by the
PPO tracker inside the NRD, collision scored against the predicted cells; the
true discs are reported as a metric only.

| rung | candidates found (32 layouts) | layouts with ≥ 1 | completed in imagination | true-disc collision | min true clearance |
|---|---|---|---|---|---|
| predicted occupancy + memorized terrain | 145 | 32/32 | 99.3 % | **0** | 0.31 m |
| full predicted map | 147 | 32/32 | 99.3 % | **0** | 0.50 m |

The chain camera → occupancy → A* → tracker-in-imagination → scoring now uses
the true map nowhere except the reported metric and the physics of the recorded
start context. Routes are exported (`routes.json`) for Chrono.

## 5. Chrono validation of camera-only plans — `wp4_chrono_pred_{occ,full}` (newton)

Every camera-only candidate the scorer imagined was driven in Chrono by the PPO
tracker (`traverse_wp3_chrono_eval.py --route-file`, 10 procs). First batch =
the §4 candidates (cell discs, 12 repair passes); the final-planner candidates
(40 passes, 165 per rung) run next as `wp4_chrono_pred_{occ,full}_r40`.

| rung | Chrono runs | completed | contact | rollover | off route | mean ct | p95 ct | imagined vs Chrono time corr | energy (act model) ratio / corr |
|---|---|---|---|---|---|---|---|---|---|
| predicted occupancy + memorized terrain | 145 | **145/145** | **0** | 0 | 0 | 0.034 m | 0.10 m | 0.987 | 1.19 / 0.72 |
| full predicted map | 147 | **147/147** | **0** | 0 | 0 | 0.033 m | 0.10 m | 0.981 | 1.16 / 0.75 |
| final planner (40 repair passes): predicted occupancy + memorized terrain | 165 | **165/165** | **0** | 0 | 0 | 0.034 m | 0.11 m | 0.990 | 1.20 / 0.71 |
| final planner: full predicted map | 165 | **165/165** | **0** | 0 | 0 | 0.033 m | 0.10 m | 0.964 | 1.22 / 0.64 |

**Plans built from the camera alone are safe to drive**: 622 of 622 completed
with no asset contact and no rollover on 32 held-out layouts, tracked as tightly
as the oracle's plans (mean cross-track 0.033–0.034 vs 0.029 m). That includes
the full-predicted-map plans the ladder flagged as exceeding the true slope caps
(8 % of them): none rolled over or left the route, so the slope caps with their
15 % validation slack are conservative for this arena rather than the flag being
a driving failure. The imagined-vs-Chrono
calibration is unchanged from the oracle candidates (time corr 0.99, 10 % fast;
throttle-based energy 19 % low, corr 0.72), so the scorer's judgement of
camera-only plans is as trustworthy as its judgement of privileged ones. Pick
agreement on the combined objective stays weak (8/29; time-only 26/29) for the
same near-tie reason as before.

## Where this leaves the gates

- **G5 (planning from vision):** camera-derived occupancy + memorized terrain
  matches the oracle on every ladder metric at 100 held-out layouts; full
  predicted map is one point behind on feasibility and needs the slope question
  answered in Chrono. Start pose and goal remain privileged (v1 contract).
- **Planner-C scoring:** collision check now runs on the predicted map; energy
  uses the throttle-based calibrated model.
- **Energy:** 15–20 % low with corr 0.7–0.8; the fix is dynamics-model side
  (tracker-driven training episodes), not calibration side.
- **Margin:** 0.9 m stays as the default; the 0.9 → 0.6 → 0.3 fallback rescues
  the last 2–3 % of layouts and its plans drive clean in Chrono (31/31).

## Open

1. ~~Dynamics model retraining with tracker-driven Chrono episodes~~ done (§6.2):
   energy fixed (ratio 1.02–1.17), time bias −10 % remains → add powertrain state
   (engine speed, gear) to z1 and retrain.
2. ~~Chrono-validate the margin-fallback plans~~ done (31/31); a clearance-aware
   smoother would let the default margin drop toward the measured 0.1 m.
3. ~~Vehicle localisation from the camera~~ done (§6.1): 5 cm / 1.4°, tracker on
   camera pose 32/32 zero contact; goal and start pose from the camera too (§6.3–6.4).
4. Test split untouched throughout.

## 6. Follow-ups started 2026-09-05 (per the WP4 recommendation list)

### 6.1 Camera-based vehicle localisation for the tracker — `traverse_wp4_train_posehead.py`, `traverse_wp3_chrono_eval.py --localisation`

The tracker's pose in the Chrono evaluation was the simulator's. A pose head on
the frozen encoder's 64×64 stage-2 map (heatmap + soft-argmax + sub-cell
regression, yaw as sin/cos) is trained on the WP1 frame set (6662 train / 1427
val layouts, same split as WP1 v6). Pixel → world inverts the pinhole model at
the vehicle-centre height using the known arena heightmap (fixed terrain).

| head | frames seen | val centre error mean / median / p95 | val yaw error mean / p95 |
|---|---|---|---|
| WP1 v5 spatial probe (16×16 map, reference) | — | 0.80 m | 3.3–4.4° |
| cluster smoke, 100 steps | 6 k | 0.106 / 0.102 / 0.196 m | 15.4° / 21.2° |
| **v1, 15 k steps (MI350, 12 min)** | 960 k | **0.054 / 0.046 / 0.110 m** (19 200 frames) | **1.45° / 3.3°** |

In Chrono the tracker can take its pose from (a) the truth, (b) the per-frame
camera estimate, or (c) a complementary filter: odometry prediction from body
velocities and yaw rate (sensorable), camera correction with gain 0.3 on
position, and a heading measurement that blends the camera yaw with the
direction of travel between consecutive camera fixes when moving faster than
1.5 m/s. The filter is initialised from the camera alone during the 0.8 s
settle, so no privileged pose enters at any point. With the *smoke* head
(15° yaw error) the raw camera pose broke tracking (mean cross-track 0.6–2.5 m);
the filter with motion heading brought it back to 0.22–0.39 m — the design is
robust to a weak yaw channel. Results with the trained head: §6.1 table below.

Chrono, the tracker on the 32 held-out recorded routes (`wp4_chrono_loc_{camera,fused}`):

| pose the tracker sees | completed | contact | time to end | mean ct | p95 ct | max ct | localisation error xy mean / p95 | yaw |
|---|---|---|---|---|---|---|---|---|
| Chrono truth (v1 contract) | 31/31 | 0 | 11.22 s | 0.029 m | 0.071 m | 0.34 m | — | — |
| **camera, per frame** | **32/32** | **0** | 11.27 s | 0.041 m | 0.105 m | 0.35 m | 0.047 / 0.097 m | 1.28° |
| odometry + camera filter | 32/32 | 0 | 11.29 s | 0.042 m | 0.117 m | 0.39 m | 0.036 / 0.074 m | 1.42° |

**The tracker no longer needs the simulator's pose.** Fed only the per-frame
camera estimate it completes every route with zero contact and a mean
cross-track within 1 cm of the truth-fed runs; the filter lowers the
localisation error (3.6 vs 4.7 cm) but not the tracking error, so the plain
per-frame estimate is the deployment choice. Tracking is now measured against
the true pose while the controller sees only the camera. With this, the only
privileged inputs left in the whole chain are the start pose and the goal.

### 6.2 Tracker-driven training episodes for the dynamics model — `traverse_wp4_collect_tracker_episodes.py`

2000 train-split layouts driven in Chrono by the PPO tracker (routes: recorded
37 %, oracle 21 %, slow 22 %, fast 19 %), recorded as 400-frame cache rows (z1,
applied action, pose, power); the layout's existing scene map is reused through
``source_key``. The map trainer appends them with ``--extra-train-cache`` (train
split only; val/test untouched). Retrained model: `wp2_mapv2_dagger_amd`.

**Round 1** (snapshot of 776 tracker-driven episodes = +12 % training data;
40 k steps, MI350, 27 min; checkpoint selected at 5 s state error as before).
Same 32 held-out layouts, same tracker, imagined rollouts vs the Chrono batches:

| dynamics model | candidates | imagined vs Chrono time (corr / bias) | power head energy: ratio / corr | throttle-model energy: ratio / corr |
|---|---|---|---|---|
| `wp2_mapv2_index_amd` (collection driver only) | oracle sweep, 185 | 0.989 / −10 % | 1.62 / 0.65 | 1.15 / 0.80 |
| **`wp2_mapv2_dagger_amd` (+776 tracker episodes)** | oracle sweep, 185 | 0.987 / −10 % | **1.31 / 0.87** | 1.13 / 0.84 |
| `wp2_mapv2_index_amd` | camera-only, 165 | 0.990 / −9 % | — | 1.20 / 0.71 |
| **`wp2_mapv2_dagger_amd`** | camera-only, 165 | 0.989 / −10 % | — | 1.17 / 0.79 |

The learned power head is the clear winner of the extra data: under the
tracker's actions its energy correlation with Chrono rises from 0.65 to 0.87
and the under-estimate shrinks from 1.62× to 1.31×, so the head now beats the
throttle-based calibration on correlation (0.87 vs 0.84). The 10 % time bias did
**not** move: the imagined tracker still holds the speed profile more tightly
than the real one (Chrono speed error 0.36 m/s). On the recorded held-out
episodes the retrained model is marginally worse (5 s state error 0.441 vs
0.435, 5 s pose error 1.73 vs 1.67 m) — the price of 12 % out-of-distribution
data in the mix. Where the 10 % sits: against the speed profile's own implied duration
(153 oracle-sweep candidates, profile 11.60 s), the **imagined** tracker finishes
at 0.93× the profile time (it runs *above* the commanded speed inside the
model) while **Chrono** finishes at 1.04× (the real vehicle lags the profile
by 3–6 %, most on the fast sweep). Same policy, opposite sign: the model
accelerates more per unit throttle than Chrono does, which is the same defect
the energy gap showed. Round 1 fixed the power channel but not the speed
channel. **Rollout-consistency loss** (`traverse_wp2_train_map.py --rollout-steps 8`):
an 8-step autoregressive loss (predicted state fed back, map re-cropped at the
dead-reckoned pose — the imagination env's own step) added to the one-step
loss, fine-tuned from the round-1 model for 8 k steps (11 min). Held-out
recorded rollouts improve sharply: 5 s state error **0.335** (one-step models
0.435–0.441), and open-loop replay of the recorded actions now completes 31/32
routes (old model 21/32, which drifted off route). Against Chrono under the
tracker (`wp4_scores_tracker_ro8`): power-head energy ratio 1.24 / corr 0.82,
throttle-model 1.10 / 0.84, combined-objective pick agreement 16/31 (Spearman
0.55–0.58, the best so far) — **but the time bias is still −10 %.**

The replay test locates it: with the recorded driver's own actions the new
model finishes the recorded routes in 9.72 s where the recording took 11.04 s
(−12 %, corr 0.87), i.e. the model's longitudinal response to throttle is too
strong even under in-distribution actions, and the tracker inherits it. Neither
12 % tracker-driven data nor the rollout loss moved it, so it is a model-input
question rather than a data-mix one: z1 carries no powertrain state (engine
speed, gear), and the HMMWV's torque response lags throttle through the
transmission. The concrete next step is to add engine speed / gear to the state
(both are in the stores) — out of scope for this session.

**Round 2** (all 1991 tracker-driven episodes = +30 % training data, fine-tuned
from the rollout-loss model for 8 k steps with the rollout loss;
`wp2_mapv2_dagger2_ro8_amd`). Summary of the four dynamics models against the
same Chrono batch (185 oracle-sweep candidates, PPO tracker):

| dynamics model | held-out 5 s state err | time bias | power-head energy ratio / corr | throttle-model ratio / corr | combined-objective pick agreement | Spearman (combined) |
|---|---|---|---|---|---|---|
| one-step, collection data only | 0.435 | −10 % | 1.62 / 0.65 | 1.15 / 0.80 | 10/31 | 0.30 |
| + 776 tracker episodes | 0.441 | −10 % | 1.31 / 0.87 | 1.13 / 0.84 | 15/31 | 0.37 |
| + rollout loss (8 steps) | 0.335 | −10 % | 1.24 / 0.82 | 1.10 / 0.84 | 16/31 | 0.55 |
| **+ all 1991 tracker episodes, rollout loss** | **0.331** | −10 % | **1.17** / 0.74 | **1.02** / 0.79 | **20/31** | **0.63–0.66** |

For the purpose that matters — ranking candidate plans on time + energy — the
final model doubles the pick agreement of the original (20/31 vs 10/31) and
lifts the within-layout rank correlation from 0.30 to 0.66; its throttle-model
energy is unbiased (ratio 1.02). The time bias is untouched by any of it
(replay: −9 %, 25/32 routes completed open-loop). Recommended checkpoint for the
scorer: `--dynamics-checkpoint artifacts/traverse/wp2_mapv2_dagger2_ro8_amd/ckpt_best.pt`
(the tracker itself was trained in the original model and needs no change:
0.03 m in Chrono).

### 6.3 Goal from the camera — `planner_b.goal_from_map`

The house is the largest predicted blob; its centroid replaces the privileged
house position as the approach-ring centre. On 400 held-out layouts the centroid
is 0.09 m (mean) / 0.39 m (max) from the true house centre, and the blob radius
averages 3.50 m against the 3.5 m footprint. Ladder with camera goal + margin
fallback (`--goal predicted`, 100 layouts): no-path 0 %, collisions 0, plans
identical to the oracle's in length; plan endpoints lie within 1.0 m of the true
approach ring on 100 % of layouts (within the planner's own 0.75 m ring
tolerance on 86 %, the oracle's discretised endpoints already use 0.64 m of it)
and within the study's 2 m success radius on all. The scorer takes
`--goal predicted`. **With this, the start pose is the only privileged input
left in the chain** (and the tracker's pose head could supply it too).

### 6.4 Start pose from the camera; the all-sensor chain — `traverse_wp4_start_pose_from_camera.py`, scorer `--start-poses`

The pose head applied to the camera frame at the rollout start gives the start
pose within 0.040 m mean / 0.076 m max and 1.2° yaw on the 32 held-out
episodes. The scorer now plans from that estimate (start), toward the largest
blob (goal), around the predicted cells (obstacles), rolls out from the
estimate (dead reckoning), and scores collision on the predicted cells:
**no privileged quantity enters candidate generation, rollout or scoring**
(`wp4_scores_allsensor`: 173 candidates on 32 layouts, 99.4 % complete in
imagination, zero true-disc collisions, min true clearance 0.32 m). Those
routes are queued in Chrono with the tracker on camera pose
(`wp4_chrono_allsensor`), alongside the camera-planned routes with camera pose
(`wp4_chrono_loc_camera_pred_occ`).

Camera-planned routes (final planner, 165 candidates) tracked with the camera
pose in Chrono (`wp4_chrono_loc_camera_pred_occ`): **165/165 completed, zero
contact**, mean cross-track 0.045 m (0.034 m with the true pose on the same
routes), localisation 4.8 cm / 1.3°, time and energy identical to the true-pose
runs (12.06 vs 12.04 s, 178 vs 177 kJ). So the planner's map, the plan, the
tracker's pose and the rollout scoring all come from the camera, and the
result in the simulator is unchanged.

All-sensor candidates (camera start pose, camera goal, camera obstacles,
camera-pose tracking; `wp4_chrono_allsensor`): **173/173 completed, 0 contact**,
mean cross-track 0.045 m, p95 0.122 m, localisation 0.048 m / 1.30°,
time 12.48 s, energy 175 kJ. Imagined vs Chrono on the same 173
candidates: time corr 0.988 (11.03 vs 12.48 s), throttle-based
energy corr 0.686 at ratio 1.17. **Nothing privileged remains in the
deployed chain**; the true map is used only to judge the results.

## 7. Planner-S: sampling + imagination instead of search — `planner_s.py`, `traverse_wp5_sample_planner.py`

Prompted by the timing measurement (§6 follow-up: map decode 1 ms, five A*
candidates 0.7 s, imagining them 1.4–1.7 s on GPU *or* CPU): the imagined
rollout is batched and cheap, so the planner should not be bound by A*'s
handful of candidates. Planner-S samples 5000 smooth routes per layout
(Catmull-Rom through three scattered control points from a point 4 m ahead of
the camera start pose to a sampled point on the approach ring around the
camera goal; per-route scatter scale 4–14 % of the chord; cruise speed
3–9 m/s; the oracle's speed-profile ramps), rejects curvature > 1/8 m on the
dense curve, applies the oracle's curvature repair to borderline ones, sweeps
the footprint against the camera's obstacle cells with 0.2 m slack, and
imagines every survivor (mean 354 per layout, capped at 2000) together with the
A* candidates in one batched rollout with the tracker from the camera start
pose. Terrain feasibility is judged by the physics model (roll / pitch /
cross-track failure flags), not by slope caps. Wall time 6.4 s per layout on the
5090 (sampling 2.3 s, A* 0.8 s, imagining ~350 routes 3.5 s).

Chrono, 32 held-out layouts, tracker on camera pose, one pick per layout
(`wp5_chrono_sample_planner_v2`):

| pick (objective in imagination) | completed | contact | Chrono time | Chrono energy | Chrono cost time+E/10 | better than the A* pick | Chrono / imagined energy at the pick |
|---|---|---|---|---|---|---|---|
| A* best (time + throttle-model energy/10) | 32/32 | 0 | 15.47 s | 126.5 kJ | 28.12 | — | 1.66 |
| sampled best, same objective | 32/32 | 0 | 16.00 s | 112.4 kJ | 27.25 | 17/32 | **2.16** |
| **sampled best, pessimistic energy = max(power head, throttle model)** | **32/32** | **0** | **14.11 s** | **119.7 kJ** | **26.08** | **22/32** | 1.55 |
| sampled best, time only | 32/32 | 0 | 10.74 s | 173.6 kJ | 28.10 | 14/32 | 1.02 |

Three findings:

1. **Sampling + imagination beats A* + imagination in the real simulator** once
   the objective is made robust: the pessimistic pick is 9 % faster and 5 %
   cheaper in energy than the A* pick, wins on 22 of 32 layouts, and every one
   of the 128 sampled routes driven completed with zero contact. The routes are
   camera-only from start pose to goal, and 11 % of the imagined-OK samples
   beat the best A* candidate on the imagined objective, so the search space
   A* explores is genuinely small.
2. **Optimiser's curse is real and measurable.** With the plain objective the
   sampler homes in on routes where the throttle-based energy estimate is
   near zero (power-head / throttle-model ratio at the pick > 1000, median over
   all samples 0.94); Chrono energy is then 2.16× the imagined value and the
   Chrono advantage shrinks to 17/32. Taking the *maximum* of the two
   independent energy estimates removes the exploit (ratio 1.55, below even
   the A* pick's 1.66). Selecting from thousands of imagined rollouts needs a
   pessimistic or ensemble score; selecting from five did not.
3. **Time-only picks are accurately imagined** (energy ratio 1.02, time bias the
   usual 10–15 %): the model's mistakes are in energy attribution, not in
   which route is fast.

Open: (a) the 0.2 m prefilter slack and the tracker's 0.05 m camera-pose error
are both inside the 0.9 m planner margin the A* path uses, so the sampled
routes run closer to obstacles (min true clearance 0.39 m vs 0.35 m for A*
picks — comparable) — a proper safety margin for sampled routes should come
from the tracker's measured error, as §7.4 of the plan intends; (b) 5000 samples
with three control points is a first family; iterative resampling around the
best (CEM) would use the same budget better; (c) the 10 % time bias still needs
powertrain state in z1 (§6.2).

## 8. Overnight 2026-09-05: using the imagination budget, and making imagined energy accurate

Two goals set for the night: (1) push the sampling planner further, since imagined
rollouts are batched and cheap; (2) make the energy the imagination reports accurate,
because the round-2 sampler exploited the throttle-model energy (optimiser's curse) and
even the pessimistic pick was 1.55× under Chrono.

### 8.1 How much does the sample budget buy? — `wp5_sample_planner_v2/cands_*.npz`

Running best of the pessimistic cost (time + max(power head, throttle model)/10) over the
imagined-OK samples of round 2, relative to the final best per layout (30 layouts, median
297 imagined-OK samples per layout, range 4–1026):

| imagined samples | excess over final best (mean) | worst layout |
|---|---|---|
| 10 | +12.0 % | +28 % |
| 25 | +7.1 % | +28 % |
| 50 | +4.3 % | +21 % |
| 100 | +2.4 % | +11 % |
| 200 | +0.7 % | +7 % |
| 400 | +0.4 % | +2 % |

Random sampling of this route family saturates at ~200–400 imagined routes; doubling the
5000-sample budget would buy well under 1 %. The value of more rollouts is therefore not
more of the same samples — it is either local refinement or robustness to model error.

### 8.2 Cross-entropy refinement — `planner_s.resample_routes`, `traverse_wp5_sample_planner.py --cem-rounds`

Every candidate now carries its control polygon; a CEM round perturbs the interior control
points of the best K imagined routes by N(0, σ) m, slides the ring end point by the matching
angle, jitters cruise speed by N(0, σ_v), rebuilds (curvature / arena / camera-cell
prefilter) and imagines the children in one batch. σ = 1.5 m fails outright (10–25 % of
children survive the filters, none beats its parent: the round-0 optimum is a short route
at the right speed and metre-scale moves only lengthen it). σ = 0.4 m, σ_v = 0.25 m/s,
32 elites × 16 children × 3 rounds, shrink 0.7 works: on 31 layouts the imagined
pessimistic cost falls from 20.22 (round 0) to 18.67 (−7.7 %), almost entirely through
imagined energy (75.9 → 63.9 kJ; time 12.63 → 12.29 s), for +11 s per layout. Whether
that 8 % is real or the optimiser's curse climbing the model's energy errors is what the
Chrono batch `wp5_chrono_sample_planner_v3` decides (§8.6).

### 8.3 Offline energy benchmark — `traverse_wp5_energy_bench.py`

Every route the tracker has driven in Chrono (726, six batches; camera-localised batches
are re-imagined from the camera start estimate, the others from the recorded start) is
re-imagined with any set of dynamics checkpoints and the estimators are scored against the
Chrono energies: ratio (Chrono / imagined), correlation, MAE, within-layout rank
correlation and top-1 agreement of time + E/10 where a batch has ≥ 3 candidates per layout
(126 groups). No simulator time needed — this is the test bed for goal (2). Baseline
(`wp5_energy_bench_base`, deployed model `wp2_mapv2_dagger2_ro8_amd`):

| estimator | ratio | corr | MAE kJ | rank ρ | top-1 /126 | ratio on the sampled-planner picks (v1 / v2) |
|---|---|---|---|---|---|---|
| power head | 1.23 | 0.84 | 34 | 0.56 | 72 | 1.42 / 1.31 |
| throttle model | 1.12 | 0.81 | 37 | 0.48 | 71 | 1.77 / 1.36 |
| pessimistic max(head, throttle) | 1.08 | 0.82 | 31 | 0.51 | 75 | 1.40 / 1.23 |
| throttle model refit on the tracker-driven episodes | 1.19 | 0.81 | 37 | 0.44 | 65 | 1.99 / 1.45 |
| 4-model ensemble, max of all estimates | 0.98 | 0.78 | 31 | 0.48 | 66 | 1.10 / 1.03 |

Imagined time is −11 % against Chrono in every batch (corr 0.987). Two things stand out:
the under-estimate is worst exactly on the routes the sampler picked (the curse, measured:
1.3–1.8× where the population average is 1.1–1.2×), and only the ensemble maximum removes
it there (1.03–1.10) — at the price of a slight over-estimate everywhere else. Refitting
the throttle model on tracker-driven data does not help: the throttle the *imagined*
tracker applies is what is wrong, not the coefficients.

### 8.4 Fixing the model instead of the estimator — powertrain state (`tire_normal_force_omega_pt`)

Two hypotheses for the −10 % speed bias and the low energy were trained side by side on
the cluster (40 k steps, recorded data only, same recipe as `wp2_mapv2_index_amd`):

* **Loss weighting** (`--delta-scale`): per-step vx changes are ~0.03 of the state std, so
  the state loss barely sees them; weight each channel by 1/std of its normalized one-step
  delta (vx ×7.8, roll ×8, tire loads ×0.4). Result: **null** — held-out state error
  0.414 vs 0.435, but the closed-loop time bias under the tracker is −11.3 % (unchanged) and
  after the tracker-data + rollout-loss fine-tune (`wp2_mapv2_dscale_dag_ro8_amd`) still
  −11.7 %; ranking marginally better (ρ 0.62 vs 0.56).
* **Powertrain state**: engine speed and motorshaft torque appended to z1 (17-D; sidecar
  cache `wp2_z2_cache_v6_pt` built from the stores by `traverse_wp5_build_z1_sidecar.py`,
  frame-aligned; the trainer takes `--z1-extra-cache`, the tracker env concatenates it, the
  checkpoint records `z1_dim`; the deployed 38-D tracker is unaffected because its
  observation uses only vx and yaw rate). Their product *is* the recorded power, so a
  17-D model gives a third energy estimate for free: Σ engine speed × torque along the
  imagined state (`energy_state_kj`).

| model (recorded data only, 40 k steps) | time bias vs Chrono under the tracker | power head ratio / corr | throttle model ratio | state-power ratio |
|---|---|---|---|---|
| `wp2_mapv2_index_amd` (15-D) | −10.6 % | 1.58 / 0.53 | 1.20 | — |
| `wp2_mapv2_dscale_amd` (15-D, delta-scaled loss) | −11.3 % | 1.65 / 0.74 | 1.24 | — |
| **`wp2_mapv2_pt_amd` (17-D powertrain state)** | **−9.4 %** | **1.09 / 0.76** | 0.71 | **1.09** |
| `wp2_mapv2_pt_dscale_amd` (17-D + delta-scaled) | −8.3 % | 1.11 / 0.58 | 0.70 | 1.16 |
| + 8 k steps rollout loss (8 steps), recorded data: **`wp2_mapv2_pt_ro8_amd`** | **−8.7 %** | **0.94 / 0.74** | 0.72 | **0.97** |
| `wp2_mapv2_pt_dscale_ro8_amd` | −9.4 % | 0.96 / 0.70 | 0.73 | 0.95 |
| for reference: deployed `wp2_mapv2_dagger2_ro8_amd` (15-D, tracker data + rollout loss) | −11.3 % | 1.23 / 0.84 | 1.12 | — |

With the rollout loss the 17-D model's power head is unbiased under the tracker (0.94–0.97
overall) and — the point of the exercise — **on the sampled-planner picks it reads 0.93–1.00
where the deployed model read 1.31–1.42**: the systematic under-estimate the sampler was
exploiting is gone. What the recorded-only 17-D model still lacks is the tracker-driven
data's ranking quality (rank ρ 0.45–0.46, top-1 65–67/126 vs 0.56 / 72–75 for the deployed
model), hence the re-collection of the 1991 tracker-driven episodes with the powertrain
channels (`traverse_wp4_collect_tracker_episodes.py --preset tire_normal_force_omega_pt`,
newton, `wp2_z2_cache_dagger_v2`) and the fine-tune on them (§8.5). Open-loop replay of the
recorded driver's actions moves from −9 % (deployed model, 25/32 routes completed) to
+2 % (30/32) — the acceleration response is no longer too strong.

With identical training data the powertrain state takes the power head from a 58 %
under-estimate to 9 % under the tracker's actions, and shaves 1–3 points off the time
bias. The throttle model flips to a 30 % *over*-estimate: the imagined tracker now has to
push the throttle harder, as the real one does — the acceleration response was the
defect, and the engine state carries the information (gear, torque lag) that the tire
channels alone did not. The sampling planner's pessimistic term is therefore configurable
(`--pess-terms head state` for 17-D models; `head act` remains the 15-D default).

### 8.5 Fine-tune on the re-collected tracker-driven episodes — `wp2_z2_cache_dagger_v2`

The 1991 tracker-driven train-split episodes were re-collected on newton with the 17-D rows
(`--preset tire_normal_force_omega_pt`; 1991 written, 2.6 h at 18 workers) and the 17-D
rollout-loss models fine-tuned on them for 8 k steps (`--extra-train-cache … --rollout-steps 8`,
init from the recorded-only rollout-loss checkpoints). A 903-episode snapshot was used first
(`wp2_mapv2_pt_dagp_ro8_amd`, the model behind §8.11–8.12), the full set afterwards.
Frame-aligned against the held-out tracker episodes (§8.9 protocol; from rest §8.10):

| model | time bias (frame 16 / from rest) | power head ratio / corr (frame 16) | from rest | state power from rest |
|---|---|---|---|---|
| `wp2_mapv2_pt_dagp_ro8_amd` (903 episodes) | −0.1 % / −0.1 % | 0.98 / 0.87 | 0.97 / 0.91 | 0.95 / 0.92 |
| **`wp2_mapv2_pt_dag_ro8_amd` (all 1991)** | **+0.1 % / −0.2 %** | **0.96 / 0.90** | **0.96 / 0.90** | **0.95 / 0.90** |
| `wp2_mapv2_pt_dscale_dag_ro8_amd` (all 1991, delta-scaled) | −0.6 % / +2.2 % | 1.03 / 0.86 | 1.01 / 0.88 | 1.00 / 0.88 |

The full set adds correlation frame-aligned (0.87 → 0.90) and nothing from rest — the 903
episodes had already done the work. In the confounded batch benchmark (§8.3 protocol) the
tracker-data fine-tune looked like a regression (head 0.94 → 1.21, time −8.7 → −11.8 %);
§8.9 explains why that reading was wrong. **Recommended dynamics checkpoint for the planner:
`wp2_mapv2_pt_dag_ro8_amd` with the sidecar `wp2_z2_cache_v6_pt` and `--pess-terms head state`.**

### 8.6 Chrono: resampling, clearance penalty, ensemble — `wp5_chrono_sample_planner_v3`

Same 32 held-out layouts, tracker on camera pose, one Chrono run per distinct route
(`traverse_wp5_merge_routes.py` drives coinciding picks once;
`traverse_wp5_summarise_picks.py` splits the rows). Deployed 15-D dynamics model;
"ens" = pessimistic energy taken as the maximum over the four 15-D checkpoints.

| pick | completed | contact | Chrono time | Chrono energy | Chrono cost | beats the A* pick | Chrono / imagined energy at the pick | min clearance |
|---|---|---|---|---|---|---|---|---|
| A* best | 32/32 | 0 | 15.18 s | 129.6 kJ | 28.15 | — | 1.38 | 1.45 m |
| sampled, pessimistic, round 0 (the round-2 winner) | 31/31 | 0 | 14.31 s | 114.3 kJ | 25.75 | 24/31 | 1.51 | 0.99 m |
| + 3 CEM rounds | 31/31 | 0 | 14.01 s | 110.6 kJ | 25.07 | 25/31 | 1.73 | 0.98 m |
| **+ 3 CEM rounds + clearance penalty** | **31/31** | **0** | **13.87 s** | **110.0 kJ** | **24.88** | **26/31** | 1.70 | **1.04 m** |
| ensemble pessimism, round 0 | 32/32 | 0 | 14.14 s | 124.8 kJ | 26.62 | 19/32 | 1.38 | 1.03 m |
| ensemble + CEM + clearance | 32/32 | 0 | 14.24 s | 122.0 kJ | 26.45 | 21/32 | 1.35 | 1.08 m |

Reading: (1) **CEM refinement is real but discounted** — the imagined cost fell 7.7 %, the
Chrono cost 2.6 % (25.75 → 25.07); the rest was the sampler climbing the model's energy
errors (Chrono/imagined energy 1.51 → 1.73 at the pick). (2) The **clearance penalty is
free**: a further 0.2 cost points *and* a wider real margin (1.04 m vs 0.98 m), 26/31 layouts
better than A*. (3) **Ensemble pessimism hurts** with these members: the maximum over four
15-D models — one of them the weak collection-only model (energy corr 0.53) — is more
conservative but ranks worse (bench ρ 0.48 vs 0.51), and its Chrono picks are slower and
costlier than the single-model ones (26.62 vs 25.75). Pessimism over poor estimates is not a
substitute for a better estimate; §8.4 provides the better estimate.

### 8.7 Chrono: the powertrain-state model as the planner's imagination — `wp5_chrono_sample_planner_v4_pt`

Same layouts and picks, dynamics `wp2_mapv2_pt_ro8_amd` (17-D, recorded data + rollout loss,
no tracker-driven data), pessimistic energy = max(power head, state power):

| pick | completed | contact | Chrono time | Chrono energy | Chrono cost | beats the A* pick | Chrono / imagined energy at the pick |
|---|---|---|---|---|---|---|---|
| A* best | 32/32 | 0 | 14.61 s | 136.2 kJ | 28.24 | — | 1.12 |
| sampled, pessimistic, round 0 | 31/31 | 0 | 16.13 s | 101.0 kJ | 26.23 | 22/31 | 1.48 |
| + 3 CEM rounds | 31/31 | 0 | 16.15 s | 95.7 kJ | 25.72 | 22/31 | **2.17** |
| + 3 CEM rounds + clearance penalty | 31/31 | 0 | 16.13 s | 96.9 kJ | 25.81 | 23/31 | 2.15 |

Two lessons. (1) A model that is unbiased *on average* (§8.4: 0.97 over 726 routes, 1.12 on
the A* picks here) is still exploitable by a search over thousands of routes: the CEM pick's
imagined energy was 44 kJ against 96 kJ in Chrono. Population-level calibration does not
protect the argmin; the search needs a guard of its own. (2) The recorded-only 17-D model
chooses *slower* routes than the tracker-data 15-D model (16.1 s vs 13.9 s) and ends up
0.9 cost points worse (25.81 vs 24.88) despite the lower energy — its ranking of candidates
is weaker (bench ρ 0.45 vs 0.56) because it has never seen the tracker's actions. Hence the
fine-tune on the re-collected 17-D tracker episodes (§8.5) and the geometry floor (§8.8).

### 8.8 Geometry floor against the curse — `traverse_wp5_energy_floor.py`, `energy_floor.py`

Chrono energy of all 990 driven routes regressed on route geometry (length, length-weighted
v², positive climb, peak speed, re-acceleration): R² 0.60 in fit, 0.51 on a held-out batch,
σ 36 kJ. Geometry alone is *not* an energy estimator (the NRD's estimates correlate 0.74–0.84
with Chrono), but fit − 1.5σ is a floor no driven route of that geometry has gone below; the
planner adds it to the pessimistic maximum (`--energy-floor … --floor-sigmas 1.5`). At the
v4 CEM pick the floor would have read ~66 kJ against the model's 44 kJ.

### 8.9 The comparison was confounded: imagination starts 0.8 s into the drive — `traverse_wp5_aligned_bench.py`

Every imagined-vs-Chrono number so far (WP3 §, WP4 §3/§6, §8.3–8.7 above) compared a
Chrono run that starts **from rest at frame 0** with an imagined rollout that starts **from
the recorded context at frame 16** — where the recorded vehicle is already at 1.98 m/s,
0.6 m down the route and has spent 31.5 kJ launching (val split, 200 episodes; the tracker in
Chrono reaches frame 16 at 0.84 m/s having spent 10–17 kJ). The imagination therefore
inherits a launch it never pays for: ~0.8–1 s of time and ~20–30 kJ of energy on runs of
11–15 s and 110–175 kJ. That is most of the "−10 % time bias" and most of the "1.2×
energy under-estimate".

The frame-aligned test removes it: 96 held-out layouts are driven by the tracker in Chrono
with the 17-D rows and the route saved (`traverse_wp4_collect_tracker_episodes.py --split
val`, never used for training); every model imagines the same route **from that episode's
own frames 0–15** with the same tracker, and time-to-end and energy are compared **from
frame 16 in both** (67 episodes reach the route end; Chrono 11.16 s, 157.6 kJ from frame 16):

| model | time bias | time corr | power head: ratio / corr | throttle model | state power | max(head, throttle) |
|---|---|---|---|---|---|---|
| `wp2_mapv2_index_amd` (15-D, collection driver only) | +2.9 % | 0.978 | 1.21 / 0.69 | 1.02 | — | 1.00 |
| **`wp2_mapv2_dagger2_ro8_amd` (15-D, deployed)** | **+0.1 %** | **0.996** | **0.99 / 0.82** | 0.97 | — | 0.94 |
| `wp2_mapv2_dscale_dag_ro8_amd` | −0.4 % | 0.998 | 1.03 / 0.85 | 1.02 | — | 0.97 |
| `wp2_mapv2_pt_amd` (17-D, recorded only) | +1.3 % | 0.993 | 0.93 / 0.81 | 0.68 | 0.92 / 0.80 | 0.68 |
| `wp2_mapv2_pt_ro8_amd` | +4.8 % | 0.972 | 0.75 / 0.69 | 0.61 | 0.77 / 0.71 | 0.61 |
| **`wp2_mapv2_pt_dagp_ro8_amd` (17-D, + 903 tracker episodes, rollout loss)** | **−0.1 %** | **0.998** | **0.98 / 0.87** | 0.96 | **0.96 / 0.87** | 0.92 |

Corrected conclusions:

1. **The deployed model has no time bias and an unbiased power head** under the tracker
   (+0.1 %, 0.99). The two DAgger rounds did fix the original 21 % energy under-estimate
   (`index_amd` → `dagger2_ro8`); the rollout loss fixed the drift. Nothing about the
   longitudinal response is "too easy" — the earlier diagnosis was the start-up artefact.
2. **Powertrain state buys ranking, not bias**: with the tracker data the 17-D model matches
   the deployed one on bias and lifts the energy correlation from 0.82 to 0.87, with the
   state-derived power (engine speed × torque along the imagined state) as good as the head.
   Without tracker data the 17-D models *over*-estimate energy by 25–33 % and time by
   1–5 % — §8.4's "unbiased 0.94" was the confound cancelling an over-estimate. The 903-episode
   fine-tune did not "revert" the gains (§8.5); it corrected them.
3. **The optimiser's-curse ratios in §8.6–8.7 are inflated by ~0.25** (the launch): the
   round-0 pessimistic pick is ~1.25× not 1.5×, the CEM picks ~1.45× (15-D) and ~1.9×
   (recorded-only 17-D). The exploitation is real but smaller, and the geometry floor (§8.8),
   fitted on from-rest Chrono energies, sits ~20 kJ above what the imagination can report —
   it needs refitting on frame-16 energies before it is tightened.
4. **Deployment implication**: the imagination has always been seeded with the *recorded*
   first 16 frames of the episode being evaluated. A live vehicle has no recording — it sits
   at rest. The planner must imagine from a rest context; §8.10 tests whether the model can.

### 8.10 Imagining from rest — `rollout(..., rest_start=True)`, `traverse_wp5_aligned_bench.py --from-rest`

Context seeded with the episode's frame-0 state (settled, at rest, brake on) repeated 16 times,
tokens cropped at the start pose; ground truth is the *whole* Chrono episode from frame 0
(67 episodes, 11.96 s, 167.9 kJ including the launch):

| model | completed | time bias | time corr | power head: ratio / corr | state power | throttle model |
|---|---|---|---|---|---|---|
| `wp2_mapv2_index_amd` (collection driver only) | 0.55 | +46 % | 0.70 | 0.97 / 0.49 | — | 0.64 |
| `wp2_mapv2_dagger2_ro8_amd` (deployed) | 1.00 | −3.5 % | 0.992 | 0.99 / 0.84 | — | 0.94 |
| `wp2_mapv2_pt_ro8_amd` (17-D, recorded only) | 1.00 | +4.8 % | 0.978 | 0.76 / 0.75 | 0.78 / 0.77 | 0.60 |
| **`wp2_mapv2_pt_dagp_ro8_amd` (17-D + tracker episodes)** | **1.00** | **−0.1 %** | **0.996** | **0.97 / 0.91** | **0.95 / 0.92** | 0.90 |

The models trained with tracker-driven episodes launch from a parked context as the real
vehicle does (the collection-only model never learned to — it stalls or drifts on 45 % of
the layouts). The 17-D tracker-data model imagines a complete run from standstill with no
time bias and an energy estimate that is unbiased and correlates 0.91–0.92 with Chrono — the
best fidelity numbers of the study, and the configuration a live planner can actually use.
The sampling planner now takes `--from-rest` (start pose = camera estimate, context = rest).

### 8.11 Chrono v5: geometry floor, 15-D vs 17-D imagination (recorded-context start) — `wp5_chrono_sample_planner_v5`

Same 32 layouts; both planners use CEM (3 rounds) and the floor at fit − 1.5σ; "f15" imagines
with the deployed 15-D model (pessimism = max(head, throttle, floor)), "pdf" with the 17-D
tracker-data model (max(head, state, floor)). Imagination still from the recorded context.

| pick | Chrono time | Chrono energy | Chrono cost | beats A* | min clearance |
|---|---|---|---|---|---|
| A* best (f15 / pdf imagination) | 15.66 / 15.25 s | 124.7 / 124.6 kJ | 28.12 / 27.71 | — | 1.43 / 1.48 m |
| sampled, round 0 (f15 / pdf) | 14.90 / 14.92 s | 105.6 / 105.4 kJ | 25.46 / 25.46 | 25/31 / 25/31 | 1.02 / 0.94 m |
| CEM (f15 / pdf) | 14.65 / 14.57 s | 101.8 / 102.6 kJ | 24.83 / 24.84 | 25/31 / 24/31 | 1.01 / 0.94 m |
| **CEM + clearance (f15 / pdf)** | 14.82 / 14.69 s | 99.4 / 100.4 kJ | **24.76 / 24.73** | **26/31 / 25/31** | 1.07 / 1.03 m |

All 169 distinct routes completed with zero contact. The loose floor changes little against
§8.6 (24.88 → 24.76), and with the recorded-context start the two dynamics models pick
equally well — consistent with §8.9: both are calibrated once the start is accounted for, and
the 17-D model's extra correlation does not show through a search that still starts 0.8 s into
the drive. The Chrono rows now record the launch energy (`energy_first16_kj`); in these
camera-localised runs it is only ~2 kJ (the tracker launches gently on a pose estimate), so the
batch-level energy ratios remain dominated by the *recorded* context's 2 m/s head start and are
not a model-fidelity measure — use §8.9/§8.10.

### 8.12 Chrono v6: the deployable configuration — imagine from rest, 17-D tracker-data model — `wp5_chrono_sample_planner_v6`

Planner: camera map + camera start pose, **rest context** (`--from-rest`), dynamics
`wp2_mapv2_pt_dagp_ro8_amd`, pessimism = max(power head, state power, geometry floor), 5000
samples + 3 CEM rounds (+ clearance penalty variant), 18 s per layout. 100 distinct routes,
all completed, zero contact:

| pick | Chrono time | Chrono energy | Chrono cost | beats A* | Chrono / imagined energy | Chrono / imagined time | min clearance |
|---|---|---|---|---|---|---|---|
| A* best | 14.42 s | 129.0 kJ | 27.32 | — | **0.98** | **1.04** | 1.51 m |
| sampled, pessimistic, round 0 | 14.52 s | 110.5 kJ | 25.57 | 25/31 | 1.07 | 1.04 | 0.97 m |
| **+ 3 CEM rounds** | **14.08 s** | **103.3 kJ** | **24.40** | **26/31** | 1.18 | 1.05 | 1.03 m |
| + 3 CEM rounds + clearance penalty | 14.10 s | 105.8 kJ | 24.68 | 24/31 | 1.20 | 1.05 | 1.07 m |

This is the first batch in which the imagination is compared like for like (both start from
rest), and it is calibrated: the A* pick's imagined energy is within 2 % and its time within
4 % of Chrono. The search still finds the model's soft spots — the CEM pick's energy is 18 %
under — but that is down from 50–70 % (§8.6) and the pick is the best Chrono cost of the
study: 24.40 against 27.32 for A* (−11 %), 14.08 s and 103 kJ against 14.42 s and 129 kJ.

### Where §8 leaves things

* **Goal 1 (use the imagination budget)**: random sampling saturates by ~300 imagined routes;
  the budget is better spent on CEM refinement (+2–3 % Chrono cost) and on a rest-context
  imagination that is calibrated against the real run. Ensemble-max pessimism over weak
  members does not help; a clearance penalty is free safety.
* **Goal 2 (accurate imagined energy)**: the deployed model was already unbiased once measured
  properly; the real gains were (a) the frame-aligned benchmark and the from-rest start, which
  make imagined time / energy comparable to Chrono at all, and (b) the powertrain state with
  tracker-driven data, which raises the energy correlation to 0.91–0.92 from rest and lets the
  energy be read off the predicted state. The remaining error is the optimiser's curse at
  ~1.2× on the CEM pick; a tighter floor refitted on from-rest energies, or a pessimistic
  ensemble of *strong* members (the two 17-D tracker-data checkpoints), are the next levers.
* Deployment gap closed: the planner no longer needs a recording of the episode it plans —
  camera map, camera pose, rest context.
* Open: the single-process live Chrono demo (camera frame at t = 0 → map → plan from rest →
  drive), the test split, and a Chrono batch with the full-data checkpoint (§8.5; it matches
  the 903-episode model from rest, so §8.12 stands as the result).
