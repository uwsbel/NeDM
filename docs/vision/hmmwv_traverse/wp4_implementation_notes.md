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

## 9. Step one (2026-09-05): plain A* versus sampling + world model on the test split, with live inputs only

Goal, agreed with the user: both planners see the same camera map, camera start pose and rest
state, and hand the same tracker a route. The classical baseline (A* with rule-based slope caps and
a rule-based speed profile) has no dynamics and no notion of energy; the sampling planner pays
neural-network inference to roll the tracker through the learned physics model on thousands of
candidates. The claim to test is that the sampled route is executed as planned (predicted time and
energy match Chrono), costs less, and is no less safe. A matched-candidate control — the same
sampled routes and CEM procedure scored by a cheap route-geometry regression instead of the world
model — isolates what the dynamics model adds over "sampling + any scorer".

### 9.1 Closing the evaluation boundary — `traverse_wp5_live_inputs.py`, `wp5_live_cache_test`

The reviewer's critiques of §7–8 all concerned inputs a live vehicle would not have:

| leak | before | now |
|---|---|---|
| scene map | vehicle-free median over 16 frames spread across the recorded episode | the fixed camera's frame 0 (vehicle parked at the start), encoded with the same WP1 stem |
| start pose | camera estimate at frame 16 (0.65 m into the drive) even in from-rest mode | camera estimate at frame 0; 32 test layouts: 0.046 m mean / 0.13 m max, 0.97° yaw |
| rest tokens | ego crop at the recorded frame-0 pose | ego crop at the camera estimate |
| terrain in speed profile / A* slope caps / geometry floor | arena height field | map head's predicted elevation (`--terrain predicted`) |
| layouts | val split (used for checkpoint selection, map/pose head validation) | test split, never touched by any training or selection |
| clearance reporting | mean of per-route minima | mean **and worst** minimum, count under 0.3 m |
| no valid sampled route | nothing exported | `deploy` = clearance-penalised CEM pick, or plain A* as fallback |

**The single-frame map breaks the dynamics model unless the vehicle masks itself out.** The map
head is indifferent — decoded occupancy from the single frame matches the full median (IoU 0.876
vs 0.876 against the true layout, same goal error) — but the dynamics model's ego crop at the start
pose then contains the vehicle's own body, which it has never seen (the cache maps are vehicle-free
by construction). On 2 of 3 layouts probed, the imagined vehicle never moved off the start (20 s
horizon, progress 0.0 m, no failure flag) and on the third it completed at 306 kJ instead of 170.
The live builder now rasterises the vehicle box at the *camera-estimated* pose (same box and 3 px
dilation as the cache builder), fills the hole from its neighbours, and encodes that. With the
mask the three layouts complete from rest at 223 / 178 / 175 kJ against 194 / 182 / 170 with the
full median map, and the decoded occupancy is unchanged (IoU 0.877). This is a real deployment
requirement, not an evaluation detail: any live use of the crop must remove the ego vehicle.

### 9.2 Protocol

32 test-split layouts (oracle family), all inputs from §9.1, tracker `wp3_tracker_v1`, camera
localisation in Chrono, 30 s horizon. Arms, all exported by `traverse_wp5_sample_planner.py`:

* **plain A*** (`astar_plain`): default `PlannerParams` (7 m/s cruise, slope caps, 0.9 m margin
  with the 0.6 / 0.3 m fallback ladder, 40 curvature-repair iterations like the sweep). Exported
  whatever the imagination says about it.
* **A* sweep + world model** (`astar_best`): the six A* variants, best imagined pessimistic cost.
* **sampling + world model**: 5000 samples → ≤ 2000 imagined, 3 CEM rounds (32 elites × 16
  children, σ 0.4 m / 0.25 m/s, shrink 0.7), pessimism = max(power head, state power, geometry
  floor), dynamics `wp2_mapv2_pt_dag_ro8_amd` (17-D, full tracker data), picks `sampled_pess`,
  `cem_pess`, `cem_pess_clear`, `deploy`.
* **sampling + geometry scorer** (`geo_*`, `--scorer geometry`): identical candidate family and
  CEM procedure; time = profile-implied duration, energy = the Chrono-fitted geometry regression,
  feasibility = geometric footprint clearance. No dynamics model anywhere.

Prediction accuracy is read per arm from its own numbers: imagined time / energy for the
world-model picks, regression energy and profile time for the geometry picks, profile time for
plain A* (it has no energy estimate at all; the world model's imagined energy for the A* route is
reported as the world model's prediction of the baseline).

### 9.3 Chrono result — `wp5_chrono_compare_test` (32 test layouts, 282 distinct routes, all completed, zero contact)

Cost = Chrono time + Chrono energy / 10. "wins" and Δcost are paired against plain A* on the 31
layouts where plain A* found a route (on `ep_2314` it found none even at the 0.3 m margin; sampling
found 21–25 valid routes there and drove one). "E / pred" is Chrono energy over the arm's *own*
prediction (imagination for the world-model picks, geometry regression for the geometry picks; the
plain-A* row shows the world model's imagined energy for the A* route, since A* predicts none).
Clearance is to the true obstacle discs; "worst" is the smallest minimum over the 32 routes.

| planner | time | energy | cost | wins | Δcost ± SE | E / pred | t / pred | t / profile | speed err | clearance mean / worst | < 0.3 m |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **plain A*** (rule-based, no dynamics) | 11.11 s | 187.9 kJ | 29.91 | — | — | (0.92) | — | 1.04 | 0.38 m/s | 1.48 / 0.72 m | 0 |
| A* sweep + world model | 13.96 | 131.1 | 27.06 | 24/31 | −2.78 ± 0.64 | 0.96 | 0.99 | 1.02 | 0.32 | 1.44 / 0.63 | 0 |
| sampling + world model, round 0 | 13.51 | 115.6 | 25.08 | 30/31 | −4.97 ± 0.59 | 1.03 | 1.00 | 1.01 | 0.28 | 1.02 / 0.17 | 2 |
| + 3 CEM rounds | 13.27 | 108.9 | **24.16** | 30/31 | −5.81 ± 0.60 | 1.10 | 1.00 | 1.01 | 0.28 | 0.98 / 0.09 | 2 |
| **+ clearance penalty (= `deploy`)** | 13.22 | 111.2 | **24.35** | **30/31** | **−5.66 ± 0.62** | 1.11 | 1.00 | 1.01 | 0.29 | 1.03 / **0.41** | **0** |
| sampling + geometry scorer, round 0 | 14.43 | 126.0 | 27.03 | 24/31 | −2.82 ± 0.70 | 1.28 | 1.01 | 1.01 | 0.29 | 0.93 / 0.06 | 2 |
| + 3 CEM rounds | 14.40 | 119.5 | 26.35 | 24/31 | −3.48 ± 0.74 | 1.32 | 1.01 | 1.01 | 0.27 | 0.92 / 0.18 | 4 |
| + clearance penalty (= `geo_deploy`) | 14.82 | 113.7 | 26.19 | 26/31 | −3.64 ± 0.69 | 1.30 | 1.01 | 1.01 | 0.27 | 1.02 / 0.28 | 1 |
| A* sweep + geometry scorer | 15.03 | 131.6 | 28.19 | 22/31 | −1.61 ± 0.83 | 1.29 | 1.02 | 1.02 | 0.29 | 1.47 / 0.68 | 0 |

Head to head on the deployable pick, **world model vs geometry scorer over the same candidates:
26/32 layouts, −1.84 ± 0.44 cost**. Noise floor: the two arms' plain-A* routes differ only by
float noise in the predicted terrain (28 of 31 identical geometries, speeds within 0.01 m/s) and
were both driven — mean |Δcost| between the two copies is 1.00, mean |ΔE| 9.9 kJ, arm means
29.91 vs 29.69 — so single-layout differences under ~1 are noise; the arm-level differences above
are 6–10 standard errors.

Wall time per layout on the 5090: plain A* 0.1–0.3 s; A* sweep 0.6–2 s; sampling + world model
19.1 s (5000 samples, ≤ 2000 imagined, 3 CEM rounds, one 17-D model); sampling + geometry 9.1 s
(CPU, dominated by route building — the geometry scorer itself is negligible).

### 9.4 Reading

1. **Plain A* is beaten on 30 of 31 layouts, by 19 % in cost (29.9 → 24.35), with 41 % less
   energy (188 → 111 kJ) for 2.1 s more time, and it fails outright on one layout where
   sampling does not.** The A* speed profile is also the one the tracker holds worst (speed
   error 0.38 vs 0.28–0.29 m/s; Chrono takes 4 % longer than the profile promises against 1 %
   for the sampled routes): the rule-based profile asks for speeds the vehicle does not deliver.
2. **What the world model adds over "sampling + any scorer": 1.84 cost (7 %) on the same
   candidates, 26/32 layouts.** Both scorers pick shorter, slower, cheaper routes than A*; the
   geometry regression's picks are 1.6 s slower and no cheaper in energy than the imagined picks.
   The world model's advantage is not only the number — it is the only arm whose predictions are
   trustworthy at the pick.
3. **Prediction accuracy.** Imagined time is exact (1.00 on every pick, 0.99 at the A* sweep).
   Imagined energy is 0.96 at the A*-sweep pick, 1.03 at the round-0 sampled pick, 1.10–1.11 at
   the CEM picks (the optimiser's curse, unchanged from §8.12's 1.18–1.20 in size class but on
   an untouched split). The geometry regression, fitted on 865 Chrono runs, is *exploited* by
   the same search: 1.28–1.32 at its own picks — it says 87 kJ, Chrono says 114. The world model
   also predicts the plain-A* route it did not choose to within 8 % (0.92, over-estimating).
4. **Safety.** Zero contact on all 282 routes. Without the penalty both samplers drive to
   0.06–0.18 m of a true obstacle on their worst layout (the reviewer's point); with the penalty
   the world-model pick's worst clearance is 0.41 m (none under 0.3) at +0.19 cost. The geometry
   pick with the same penalty still has one route under 0.3 m (0.28), because its clearance is
   geometric while the imagined one includes the tracker's actual deviation. Max roll / pitch
   are 15–18° in every arm — this terrain does not stress dynamic feasibility, as anticipated.
5. **Live inputs cost nothing** once the vehicle masks itself: the single-frame camera map plus
   frame-0 pose give the same decoded occupancy as the recorded medians and a from-rest
   imagination calibrated to 1.00 in time. The vehicle-in-crop failure of §9.1 is the one real
   deployment lesson of the exercise.

### 9.5 Where step one leaves things

* Claim supported on the current arena: with identical live inputs and the same tracker, the
  sampled world-model route is cheaper than plain A* (−19 %), the route's predicted time and
  energy are borne out in Chrono (time exact, energy within 10 %), and the world model is worth
  7 % over a cheap scorer on the same candidates. Not shown, because the arena cannot show it:
  dynamic-feasibility failures of the rule-based planner (no rollover, stall or contact anywhere).
* Curse guard: the CEM picks still land 10 % under on energy. Refitting the floor on from-rest
  energies or penalising routes whose imagined energy sits far below the regression are the
  cheap next levers; a Chrono batch with a pessimistic ensemble of the two strong 17-D members
  is the expensive one.
* Reproducibility note: run both arms' map decoding on the same device — CPU vs GPU float noise
  in the predicted elevation moved 4 of 31 plain-A* geometries by up to 0.24 m.
* The terrain-stress study (steeper climbs, side slopes, crater rims) is what would earn the
  dynamic-feasibility half of the claim; it needs a new arena and hence re-collection and
  retraining of the encoder, map head and dynamics model.

### 9.6 Review of §9 (second opinion, 2026-09-05) — what checked out, and the revised next step

A reviewer qualified four statements in §9.3–9.5. Each was checked against the rows and the code:

| reviewer's point | check | verdict |
|---|---|---|
| "time exact, energy within 10 %" describes aggregate ratios; per-route errors are larger | pooled over the 159 world-model picks driven: time MAE **0.17 s** (p95 0.35–0.45 s); energy MAE **14 %** (11 % at the A* picks, 17 % at the CEM picks), **p95 37 %**, worst 81 %. The geometry regression: 24–25 % MAE, p95 45–57 % at its own picks | correct — §9.3's ratios are means over routes and hide a wide per-route spread; both are reported from now on |
| the gain decomposition is descriptive: a different scorer changes the CEM elites and later candidates | true by construction for the CEM picks. The round-0 banks were meant to be identical (same seed) but the RNG is shared across layouts and CEM consumes a different number of draws per arm, so only **14 / 32** round-0 banks coincided. On round 0 the world-model pick still beats the geometry pick 24/32, −1.95 ± 0.53 | correct — the 7 % is a pipeline comparison; a clean scorer comparison needs one candidate bank scored by every scorer (per-layout seeds, cross-scoring the union of banks) |
| the stalled imagination is rejected, not driven: acceptance requires `completed & ~failed & ~collided` (`traverse_wp5_sample_planner.py`, `Imaginer.__call__`) | confirmed; the first (unmasked) run showed "ok 0" on those layouts, i.e. the sampled arm would have collapsed to the plain-A* fallback | correct — §9.4 item 5 overstated the hazard. The vehicle-in-crop bug causes **false rejection** of feasible routes; the unsolved hazard is **false acceptance**: an imagined success that fails in Chrono. None occurred in 282 runs, but nothing in the pipeline would catch one |
| the remaining prediction error is a planning-reliability question, not guard tuning | agreed; whether the search selects the model's mistakes is exactly the curse measurement (§8.6, §9.3: 1.03 at round 0 → 1.10 after CEM). Reference the reviewer supplied: *Decision-Metric Alignment in Latent World Models* (arXiv 2608.18746) — title verified | correct — keep the curse measurement as a standing metric alongside the terrain work |

Two further dependences the reviewer flagged are real and matter for any "unfamiliar terrain" claim:
the crop projection (`map_crop.py`, `_terrain_height`) and the pose head's pixel-to-world inversion
(`traverse_wp4_train_posehead.py`, `pixel_to_world`) both read the **arena heightmap**. On a new
arena they need either the new heightmap (a prior map) or the map head's predicted elevation.

**What arena_v1 already offers for a pilot** (from `arena_meta.json`): 6 hills (1.5–3 m high,
σ 4–7.5 m) and 6 craters (1–2 m deep, σ 2–4 m, rimmed), slopes capped at 20° in generation
(measured max 22.8°, p99 19°, 10 % of the arena above 15°). The 282 routes of §9.3 crossed at most
18.6° along-track (mean of per-route maxima 10.6°) with 1.7 m of climb on average — the routes go
around or over the shoulders of the features, never through a crater. So the first feasibility map
can be made on the existing craters and hills without a new arena; steeper cases need
`terrain.ArenaSpec` with a raised slope cap.

**Revised research question** (reviewer's wording, adopted): *can the world-model planner identify
feasible, efficient route-and-speed combinations that a strong classical planner or a cheap learned
scorer misjudges?* Terrain is to be designed for meaningful choices — a direct crossing, a slower
crossing, a detour, with at least one demonstrably feasible option — not until A* fails. Failing to
hold 7 m/s is a performance limit, not infeasibility.

**Revised next step: a bounded pilot, before any recollection or retraining.**

1. *Feasibility map in Chrono* (newton). Parameterised challenges — first the existing craters and
   hills, then one climb, one side slope, one crater rim from `ArenaSpec` — each swept over cruise
   speed and approach heading with the same tracker, including slow profiles and detours. Record
   stall (speed under threshold with throttle on), wheel unloading (min tire normal force — already
   in the 15-D state), max roll / pitch, tracking loss, time, shaft work. Output: for each challenge,
   which route-and-speed combinations are feasible and what they cost.
2. *Shared candidate benchmark.* One candidate bank per challenge (per-layout RNG seeds so every
   scorer sees identical banks, union of all CEM children cross-scored), scored by (a) a classical
   planner with tuned speed profiles, (b) the geometry regression, (c) the world model, later (d) a
   state-only world model to isolate what vision contributes. Every scorer's pick is driven. Report
   energy at matched traversal time and safety, alongside the combined cost; report per-route
   prediction error (MAE, p95) and the curse ratio for every scorer.
3. *Sanity signals, measured not assumed.* Log the geometry-regression disagreement, completion,
   and predicted physical-limit violations for every pick; measure which of them actually flag the
   decisions that go wrong in Chrono. The regression is a disagreement signal, not a bound — it
   misestimates its own picks by 25 %.
4. *Scale only if the pilot shows decisions the cheap scorers get wrong.* Then: collect successful
   and failed executions, split terrain parameters into train / dev / test, test the frozen encoder
   first, fine-tune what the measured failures indicate, and resolve the heightmap dependences above.

Related work the reviewer pointed to, both titles verified: *Learning When to Jump for Off-road
Navigation* (arXiv 2602.00877) already treats motion-dependent traversability; the contribution to
aim for is that explicit physical-state prediction with the tracker inside the imagination
improves those decisions on terrain the planner has not seen.

## 10. Pilot (2026-09-05/06): terrain feasibility map and the shared candidate benchmark

Plan §24. Research question: can the world-model planner identify feasible, efficient route-and-speed
combinations that a strong classical planner or a cheap learned scorer misjudges? Everything below
uses the same tracker (`wp3_tracker_v1`) and, for the planners, live inputs only (§9.1).

### 10.1 Tooling

* `traverse_wp3_chrono_eval.py --tasks-file`: explicit runs (key, layout meta, route); `dump_frame0`
  saves the camera frame at t = 0, the 17-D rest state and the true pose; new row fields `stall_s`
  (throttle on, under 0.3 m/s, after the 2 s launch window), `stalled` (≥ 1 s), `unloaded_s`
  (any wheel under 500 N), `min_tire_fz_n`, `max_roll_deg`, `max_pitch_deg`.
* `traverse_wp6_challenges.py`: for every hill / crater in an arena and 8 approach headings (hills also
  get one-sigma "shoulder" lines = side slope), a layout with the start ≥ 18 m before the feature
  centre and the house ≥ 17 m beyond it, both on ground under 10° like the regular layouts (the first
  version spawned the vehicle on a 20° flank; it slid, three wheels unloaded, and both runs bogged
  down — a layout artefact, fixed). Routes per layout: straight through the centre at constant
  2 … 8 m/s (`direct_v*`, oracle launch ramp / taper), the same line with the oracle's rule-based
  slope-capped profile (`slope_aware`, the classical planner's answer), and two 5 m/s detours.
* `traverse_wp6_build_challenge_cache.py`: frame-0 dumps → planner cache (rest state tiled, vehicle
  masked out of its own frame at the camera-estimated pose, single-frame scene map) + start poses.
* `traverse_wp5_sample_planner.py --scorer both --challenge-cache`: the shared candidate benchmark —
  per-layout RNG seeds (identical banks whatever ran before), one CEM chain per scorer from the common
  round-0 bank, every candidate then scored by every scorer, picks per scorer from the union
  (`wm_*`, `geo_*`), and per pick the other scorer's opinion (`*_disagree`), acceptance, imagined
  roll / pitch / wheel-load extremes. `traverse_wp6_sanity_signals.py` ranks those signals by how
  well they identify picks that go wrong in Chrono.

### 10.2 Feasibility map on arena_v1 — `wp6_feasibility_v1` (33 challenges × 10 routes, true pose)

Arena_v1's 6 hills (1.5–3 m) and 6 craters (1–2 m) with a 20° generation cap: 33 layouts survive the
flat-start / flat-house requirement (11 crater, 22 hill), crossing lines up to 18.9° along-track.

* **328 / 330 routes feasible.** Every constant-speed crossing from 2 to 8 m/s completed without a
  stall or a roll / pitch abort (max roll and pitch 12–15° in every family). The two failures are
  detours: the tracker lost a curved 5 m/s detour on a side slope (cross-track 1.9 m, 19 s stall)
  and bogged in a crater rim on another (5 s stall).
* **The rule-based slope-aware profile is never the cheapest feasible route.** It costs 1.30× the
  best constant-speed crossing on average (1.04–1.64×): the same time (−0.3 s) but **+63 kJ**
  (163 vs 100 kJ, +63 %), because modulating speed on every slope spends energy re-accelerating.
  The best constant speed is 4 m/s on 18 of 33 layouts, 3 m/s on 7, 6–7 m/s on 8. Detours are
  always worse (1.5–3× the direct cost).
* **Wheel-unload metric unusable as implemented:** every run reports a wheel under 500 N for
  0.6–24 s (median 3 s) and a zero minimum, at 2 m/s on gentle ground as much as at 8 m/s. The
  reported tire force must be checked (contact reporting at the control rate) before it is used.
* Reading: on this arena the decision that matters is *speed*, not feasibility — everything is
  drivable, and the classical rule gets the speed wrong by a third of the cost. Feasibility failures
  need the steeper arena (§10.3).

### 10.3 Feasibility map on a steeper arena — `assets/traverse/arena_v2_steep`, `wp6_feasibility_steep_v1`

Same generator (`terrain.ArenaSpec`, seed 11) with hills of 3–5 m, craters of 2–3.5 m and the slope cap
raised from 20° to 32° (measured max 37°, p99 32°); Chrono's BMP orientation copied from arena_v1.
17 challenge layouts survive the flat-placement test (3 crater, 14 hill), lines up to 31.6° along-track.
Same 10 routes per layout, same tracker, true pose. **This arena has the decisions the pilot was
looking for:**

* **37 / 170 routes infeasible** (stall, off-route, timeout). Two layouts are infeasible at every speed
  and both detours (a 28° climb the tracker cannot hold the line on; a 12° shoulder line that stalls
  at every speed). Roll or pitch exceeds 25° on 6 of the 17 layouts even when the crossing succeeds.
* **The rule-based slope-aware profile is infeasible on 5 / 17 challenges** — on the steep climbs it
  slows down, as its rule says, and stalls — and where it is feasible it costs 1.38× the best feasible
  crossing (up to 2.79×). It is never the cheapest.
* **A momentum regime the rules get backwards.** On the 27–32° crater climbs the slow crossings stall
  (2 m/s always; 3–4 m/s on the steepest) while 6–8 m/s complete; on a 23° hill the *slow* crossing
  (2 m/s) is best and fast ones are 10 % dearer. Whether to carry speed into a climb or crawl it is a
  route-and-speed decision with a 2–3× cost spread, and neither a slope cap nor a geometry regression
  contains the answer.
* Detours are usually feasible but 1.3–3× the direct cost; on two layouts a detour is the only
  feasible option (`hill0_h225` at 2–3 m/s direct stalls, detour L completes).

The world model has never seen slopes above 20°; §10.5 asks whether it predicts any of this anyway.

### 10.4 Does the world model predict the feasibility maps? — `traverse_wp6_imagine_sweep.py`

The sweep routes of §10.2–10.3 were imagined exactly as the planner would (rest state at the camera
start pose, single self-masked frame, `wp2_mapv2_pt_dag_ro8_amd`, 30 s horizon) and compared route by
route with Chrono. On the steep arena the crop's prior height field is the steep arena's own; the model
has never seen slopes above 20°.

| | arena_v1 (training terrain), 330 routes | steep arena (unseen), 170 routes |
|---|---|---|
| Chrono infeasible | 2 | 37 |
| imagination rejects | 3 (1 of the 2 infeasible, 2 false rejects) | **0** — every stalled or off-route crossing is accepted |
| AUC of imagined time / pitch / energy for infeasibility | 0.96 / 0.96 / 0.99 (2 positives) | 0.67 / 0.59 / 0.54 |
| time, both feasible: Chrono / imagined (MAE, corr) | 1.00 (0.34 s, 0.99) | 1.04 (0.72 s, 0.94) |
| energy, both feasible: Chrono / imagined (MAE, corr) | 0.93 (22 %, 0.82) | **1.35** (33 %, 0.68) |
| imagined best = Chrono's best feasible | 19 / 33 | 1 / 17 |
| imagined best infeasible in Chrono | 0 | **3 / 17** |
| Chrono cost of imagined best / best feasible | 1.03 (max 1.35) | 1.08 (max 1.27) |

**Regret of each scorer's pick among the SAME 10 sweep routes** (Chrono cost of the pick / best feasible;
"infeasible" = the pick stalled or left the route in Chrono):

| picker | arena_v1: infeasible picks | regret mean / max | steep: infeasible picks | regret mean / max |
|---|---|---|---|---|
| rule (slope-aware profile) | 0 / 33 | 1.30 / 1.64 | 3 / 15 | 1.38 / 2.79 |
| world model (imagined cost) | 0 / 33 | **1.03** / 1.35 | **3 / 15** | 1.08 / 1.27 |
| geometry regression | 0 / 33 | 1.04 / 1.35 | 1 / 15 | 1.12 / 2.05 |
| always fastest (8 m/s) | 0 / 33 | 1.20 / 1.45 | 0 / 15 | 1.12 / 1.52 |
| always slowest (2 m/s) | 0 / 33 | 1.38 / 2.90 | 5 / 15 | 1.24 / 1.49 |

Reading:

1. **On its training terrain the world model reads the map well** — time exact, energy within 7 %
   in aggregate (22 % per route), the right crossing on 19 of 33 layouts and 3 % regret against the
   Chrono-optimal choice, where the classical rule pays 30 %. But **the geometry regression does the
   same on this bank (4 %)**: choosing a speed for a straight crossing of a gentle feature is a
   problem a length-and-speed regression solves. The world model's edge over the regression in §9 came
   from the free-form sampled routes, not from speed choice.
2. **On terrain it has not seen the world model does not know what it does not know.** It accepts all
   37 infeasible crossings, under-predicts energy by 35 %, and 3 of its 15 challenge-level picks stall
   in Chrono — false acceptances, the hazard §9.6 named. It is still no worse than the rule (3 / 15
   infeasible, 1.38 regret) and its feasible picks are cheaper than anyone's, but "carry speed into the
   climb" (always 8 m/s) is infeasible nowhere here, which shows how little any scorer understands the
   stall regime.
3. **Gate for step 4 (plan §24): partly open.** The steep arena contains exactly the decisions the
   research question asks about — stall vs momentum, detour vs direct, a 2–3× cost spread — and no
   current scorer resolves them. That justifies collecting there. It does *not* yet show a world-model
   advantage over a cheap scorer on speed choice; that advantage has to be demonstrated after training
   on the stall regime, on held-out steep features, against the regression refitted on the same data.

### 10.5 Shared candidate benchmark on the arena_v1 challenges — `wp6_plan_challenges_v1`, `wp6_chrono_challenges_v1`

33 challenge layouts (§10.2), live inputs from the frame-0 dumps (camera start pose error 0.065 m mean),
`--scorer both`: one common round-0 bank per layout (5000 samples → 500–1000 clear), one 3-round CEM chain
per scorer, every candidate scored by both, picks from the union; 227 distinct routes driven in Chrono with
camera localisation, 8 procs on newton. Cost = time + kJ/10; paired against plain A* where both completed.

| pick | completed | Chrono time | energy | cost | wins vs plain A* | Δcost ± SE | E / own pred | t / own pred |
|---|---|---|---|---|---|---|---|---|
| **plain A*** | **32 / 33** (stalled on `hill0_h000`, 7.9 s) | 9.19 s | 166 kJ | 25.79 | — | — | (0.99) | 1.00 |
| A* sweep, world-model choice | 33 | 11.25 | 138 | 25.05 | 22/32 | −2.12 ± 0.47 | 1.08 | 1.03 |
| A* sweep, geometry choice | **32 / 33** (stalled on `crater10_h000`) | 11.56 | 135 | 25.09 | 17/31 | −0.94 ± 0.44 | 1.33 | 1.07 |
| sampling + world model, round 0 | 33 | 11.66 | 112 | 22.82 | 26/32 | −3.48 ± 0.58 | 1.20 | 1.01 |
| **sampling + world model, CEM + clearance (`wm_deploy`)** | **33** | 11.31 | **113** | **22.65** | **31/32** | **−3.67 ± 0.54** | 1.34 | 1.01 |
| sampling + geometry, round 0 | 33 | 11.47 | 131 | 24.53 | 25/32 | −1.65 ± 0.79 | 1.35 | 1.07 |
| sampling + geometry, CEM + clearance (`geo_deploy`) | 33 | 11.35 | 125 | 23.89 | 26/32 | −2.39 ± 0.87 | 1.36 | 1.07 |

* **World model vs geometry regression on the shared bank: 18 / 33 layouts, −1.24 ± 0.71.** Smaller and
  less certain than §9's −1.84 ± 0.44 on the recorded layouts. On these feature-crossing layouts the search
  exploits *both* scorers about equally (Chrono / predicted energy 1.34 and 1.36 at the CEM picks; 1.20 at
  the world model's round-0 pick) — the curse is twice §9's 1.11 because every sampled route now crosses
  a hill or crater and the model is less accurate there (energy MAE 22 % per route on the straight
  crossings, §10.4).
* Against the sweep's best hand-made route (an oracle over 10 straight / detour crossings, §10.2) the
  world-model pick costs 1.12× (median 1.06) and beats it on 7 / 33 layouts; the geometry pick 1.19×
  (4 / 33); plain A* 1.31×. Free-form sampled routes do find crossings the straight sweep does not.
* **Feasibility.** Both classical picks stalled once (the plain-A* route over the top of `hill0`, the
  geometry-chosen A* variant into `crater10`): the rule-based speed profile is not always feasible even on
  arena_v1 once the goal lies just past a feature. All 165 world-model-chosen routes completed, and so did
  all sampled geometry picks. Roll and pitch stayed under 27° for every pick.

### 10.6 Which sanity signals identify the picks that go wrong? — `traverse_wp6_sanity_signals.py`

363 driven picks; 83 (23 %) "went wrong" = did not complete, stalled, contact, or Chrono cost more than
1.25× the planner's own prediction (81 of the 83 are mis-predictions, 2 are stalls).

| signal (higher = more suspicious) | AUC | top-decile precision (base 0.23) |
|---|---|---|
| \|energy disagreement between the two scorers\| | **0.67** | **0.50** |
| energy prediction close to the geometry floor | 0.62 | 0.50 |
| imagined time | 0.62 | 0.25 |
| imagined max pitch / roll | 0.57 / 0.54 | 0.26 / 0.16 |
| imagined min tire load | 0.44 | 0.05 |
| the other scorer rejects the route | 0.50 | — (it never does) |

The disagreement between the world model and the regression is the only useful flag: when the two differ
by a lot, half the time the pick is a mis-prediction. Imagined roll, pitch and wheel load carry nothing
here (the arena never gets near the limits), and neither scorer ever rejects a route the other accepts.
Used as a veto on the top decile of disagreement, it would have removed 50 % bad picks at the price of
50 % good ones — a signal worth logging, not yet a gate.

### 10.7 Where the pilot leaves things

1. **The decisions exist, and rules get them wrong.** On arena_v1's features the rule-based slope profile
   is never the cheapest crossing (+30 % cost, +63 % energy) and stalls once; on the steep arena it is
   infeasible on 5 / 17 challenges and up to 2.8× the best feasible cost, in a regime where carrying
   speed into a climb beats crawling. That is the substance the research question needs.
2. **On its training terrain the world model resolves them — but so does a regression.** Among matched
   straight crossings both pick within 3–4 % of the Chrono-optimal speed; on free-form sampled routes the
   world model is ahead by 1.2–1.8 cost (7 %) with a wide error bar on the harder layouts, and it is the
   only planner with zero infeasible picks (198 routes). Its per-route energy error on feature crossings
   is 22 %, and the CEM search exploits that to a 34 % under-estimate at the pick.
3. **On unseen steeper terrain it does not know that it does not know**: 0 of 37 infeasible crossings
   rejected, 3 of 15 picks would have stalled, energy under by 35 %. No current scorer or heuristic
   predicts the stall regime.
4. **Gate for recollection (plan §24 step 4): open for the steep arena, with the claim narrowed.** What
   retraining on steep terrain can be asked to show is (a) rejecting the infeasible crossings the rule
   accepts and (b) choosing the momentum-vs-crawl regime, on held-out steep features, against the
   regression refitted on the same data. Speed choice on gentle features is not where the world model
   earns its cost.
5. Housekeeping before that: fix the tire-load metric; make the geometry decoder device-independent;
   resolve the arena-heightmap dependence of the crop and pose head (the steep runs used the steep
   arena's own height field as the prior map).

### 10.8 Review of §10 and the corrected frozen-model replay

A reviewer checked `465dfc3` and found two implementation faults in the steep-arena test; both reproduced:

| finding | check | fix |
|---|---|---|
| the loaded model kept **arena_v1's crop height field** even when the steep arena was requested (the checkpoint's `cropper.heightmap` buffer overrode the one built for the arena; 5.21 m max difference) | reproduced: loading with `arena_v2_steep` gave arena_v1's heights | `load_map_model` drops the checkpoint's buffer and keeps the requested arena's; `MapCropper.heightmap` is now non-persistent |
| the encoder's **elevation channel was normalised with each arena's own height range** (3.90 m on v1, 8.45 m on the steep arena), so a metre of relief entered the frozen encoder at less than half its training scale | reproduced in `EpisodeMedian._elevation` | the cache builders take `--norm-arena` (default arena_v1): training normalisation whatever arena is planned on; the vehicle mask still uses the true ground |

Further corrections adopted: "feasible" now excludes contact (one steep detour completed with 10 kN against
the house → 38 infeasible, not 37); the wheel-unload metric fired everywhere because single wheels read
exactly zero for isolated 50 ms frames at 3–6 m/s (wheel hop on the 0.15 m roughness; the four loads still
sum to the 25 kN weight) — replaced by the longest consecutive unloaded stretch and an "airborne" count
(sum of loads under half the weight); the −1.24 of §10.5 has a feature-grouped bootstrap 95 % interval of
[−2.50, +0.05] over the 10 physical features (per-layout: [−2.73, +0.02]) — modest, not decisive; and the
disagreement flag of §10.6 has AUC **0.43** on the 33 `wm_deploy` picks alone (0.74 on `geo_deploy`): it
flags the regression's mistakes, not the world model's, and is not a guard for the deployed choice.

**Corrected replay of the 170 steep crossings** (same frames, same Chrono outcomes, frozen model):

| inputs | infeasible crossings accepted | energy Chrono / imagined (both feasible) | imagined best = Chrono best | imagined best infeasible |
|---|---|---|---|---|
| as reported (v1 crop, arena-own normalisation) | 37 / 37 | 1.35 | 1 / 17 | 3 |
| crop height field fixed | 37 / 38 | 1.34 | 1 / 17 | 2 |
| normalisation fixed | 38 / 38 | 1.22 | 2 / 17 | 2 |
| **both fixed** | **38 / 38** | **1.22** | 4 / 17 | 2 |

The preprocessing faults explain part of the energy error (1.35 → 1.22) and none of the feasibility
blindness: with correct inputs the frozen model still accepts every stalled or off-route crossing. The
steep-terrain failure is missing dynamics knowledge — the training data contains no stall regime — and
recollection there is justified. The reviewer's other conditions stand: the steep arena's decisions are
one-sided (8 m/s completes every challenge that has a feasible option; no challenge needs a detour), so the
benchmark must gain cases where speed is penalised (stability, tracking) and where only a detour works, and
splits must be by terrain instance, not by heading through the same feature. `assets/traverse/arena_v3_rough`
(30° cap, 0.35 m roughness, 2.5–4 m craters) is the first attempt at that; its sweep is §10.9.

Sharpened research question (reviewer's wording, adopted): **does imagining the vehicle's changing state help
choose successful action sequences beyond what terrain and entry speed alone can predict?** The distinguishing
case to build is *sequences* of features — carrying speed to clear a climb that leaves the vehicle badly placed
or too fast for the next turn or side slope — since a per-crossing entry-speed table already covers single
features (cf. *Learning When to Jump*, arXiv 2602.00877).

### 10.9 A two-sided decision range — `assets/traverse/arena_v3_rough`, `wp6_feasibility_rough_v1`

Generator seed 23, 30° cap, roughness 0.25 m at 2.5 m correlation (v1: 0.15 m at 2 m), hills 2.5–4.5 m,
craters 2.5–4 m; placement limit 14° (with 0.35 m roughness nothing was flat enough to spawn on). 27 challenge
layouts, 270 crossings, contact-aware feasibility, sustained-lift wheel metric.

* **113 / 270 infeasible; 5 layouts have no feasible route at all** (28–29° shoulder lines and one 20° climb
  the tracker cannot hold). The rule-based profile is infeasible on **16 / 27** challenges and cheapest on 1.
* **Speed is now penalised as well as required.** Momentum climbs remain (2–3 m/s stall, 5+ m/s cross on
  eight layouts); on five layouts the fast crossing is 10–17 % dearer than a mid speed (`hill3_h090`: 5 m/s
  40.9 vs 8 m/s 47.4; `hill1_h270`: 5 m/s 33.5 vs 7 m/s 37.9) and on `crater8_h225` every direct crossing
  stalls or times out and **only the detours are feasible**; on `crater7_h000` the detour is the cheapest
  feasible route. "Always 8 m/s" is infeasible on one challenge and 9 % over the optimum on average — no
  longer a free lunch, though still the best cheap heuristic.

**Pick among the identical 10 routes** (Chrono cost / best feasible; frozen world model with corrected inputs):

| picker | rough: infeasible picks / regret | steep (§10.3, corrected): infeasible / regret |
|---|---|---|
| rule (slope-aware profile) | 11 / 22, 1.43 (max 2.14) | 3 / 15, 1.38 (max 2.79) |
| always slowest (2 m/s) | 14 / 22, 1.44 | 5 / 15, 1.24 |
| always fastest (8 m/s) | 1 / 22, 1.09 (max 1.26) | 0 / 15, 1.12 (max 1.52) |
| geometry regression | 5 / 22, 1.26 (max 2.38) | 1 / 15, 1.12 (max 2.05) |
| world model, frozen, arena_v1 training | 5 / 22, 1.13 (max 1.79) | 2 / 15, 1.08 (max 1.27) |

The frozen model on the rough arena: rejects 10 routes (6 rightly, 4 wrongly) of 113 infeasible; imagined
time has AUC 0.81 for infeasibility (the imagined vehicle is slow where the real one stalls, without
predicting the stall itself); energy 1.36 under with correlation 0.41. It picks no worse than the regression
on feasibility and better on cost, but 5 of its 22 picks stall. Nothing trained on arena_v1 resolves this
arena; the fast heuristic is the one to beat.

**What the collection has to contain, from the three maps:** momentum climbs (v2/v3 stall, v5+ cross),
speed-penalised rough crossings, detour-only and detour-cheapest craters, contact cases, and the
no-solution layouts as negatives; successes and failures alike; several generator seeds so that terrain
instances (not headings through one feature) can be split train / dev / test; and sequences of features
(the sharpened question of §10.8), which none of the single-crossing challenges yet exercise.

## 11. Learning comparison (2026-09-06/07): pipeline, multi-arena collection, matched models — plan §28

### 11.1 Pipeline prerequisites (plan §28 step 0) — commit `890a46f`

* **Crop height field per episode.** `MapCropper.forward(maps, pose, heightmap=None)`: the world → image projection
  of the ego window reads an optional `(B, 1, H, W)` height field, so a batch may mix arenas; the trainer keeps a
  bank of arena heightmaps and indexes it per episode. Default path (module buffer) is numerically unchanged.
* **Cache schema v2.** `nrd_data.load_split` accepts variable-length episodes (files hold the recorded frames only,
  plus `arena` and `status`), pads them by repeating the last row and returns a `valid` mask, `arena_idx` /
  `arena_ids` and per-episode `status`; `concat_splits` merges caches of different lengths / arenas;
  `Normalizer.fit` uses recorded frames only.
* **Trainer** (`traverse_wp2_train_map.py`): `--caches ... --split-by arena --val-arenas ... --test-arenas ...`
  splits by terrain instance (test arenas are loaded by nothing); window sampling draws only windows inside the
  recorded frames (episodes weighted by the windows they offer, as before for uniform episodes); the rollout
  metrics average only episodes that were still recorded at the horizon (`n@h` reported) and per-arena `z1_mae`;
  `--extra-train-cache` still appends arena_v1 tracker episodes (sidecar applied when they are 15-D); `ckpt_last.pt`.
* **Collector = the Chrono runner** (`traverse_wp3_chrono_eval.py --tasks-file` with a `record` path per run):
  every frame from t = 0 until the run ends is written in the cache convention (state after `Synchronize` at the
  first substep, the action of that interval; 17-D powertrain preset), whichever way it ends — `completed`
  (+ `--park-s 1.5` of braking, the collector's parking convention), `stall` (`--stall-abort-s 3`: throttle on,
  no motion for 3 s after the launch window), `off_route`, `rollover`, `timeout` (30 s). `--skip-existing`
  resumes a batch (newton's start-up segfaults). One frame-0 camera dump per layout.
* **Pose head input.** The elevation channel is normalised with the head's training arena everywhere it is used
  (runner `--norm-arena`, challenge cache builder). The map head's elevation decode likewise uses its training
  range (`MapDecoder`; on the rough arena: bias 0.09 m, MAE 0.44 m, corr 0.89 against the true field).

Smoke on newton (3 runs, `arena_v3_rough` crater 8): direct 8 m/s completed (150 + 30 parking frames),
slope-aware stalled after 270 frames, 2 m/s timed out at 600 frames — all three recorded, labelled, and loaded by
the trainer with the mask.

### 11.2 Collection design (plan §28 step 1) — `traverse_wp7_arenas.py`, `traverse_wp7_collection_tasks.py`

Seven arena instances `assets/traverse/arena_f101 … f107` (one generator seed each; family ranges: slope cap
25–32°, roughness 0.15–0.28 m at 2–3 m correlation, 5–7 hills of 2.5–4.5 m, 5–7 craters of 2–4 m sigma;
orientation copied from arena_v1). **f101–f104 train, f105 validation / model selection, f106–f107 sealed** (their
runs are the evaluation's candidate banks and are looked at once, at the end).

Per arena, tracker-driven runs from rest, true localisation for the tracker, camera frame 0 dumped per layout:

| kind | layouts | runs per layout |
|---|---|---|
| crossing of one feature (8 headings; hills also on the shoulder = side slope) | 21–47 | `direct_v2…v9` constant speed, `slope_aware` (rule profile), `detour_L/R` at 5 m/s |
| sequence: climb through A, turn (±35° / ±60°) onto B's centre or shoulder, 14–28 m apart | 5–9 | `seq_v{v1}_{v2}` two-speed profiles (v1 through A, v2 from 8 m before B; curvature cap relaxed to 5 m/s² so fast turn entries are driven), `slope_aware` |
| free-form sampled routes between random flat start / goal pairs | 6–8 | 6 routes spread over cruise speed 2–9 m/s |

Runs recorded (`artifacts/traverse/wp7_cache_v1`, 2 471 episodes, 2026-09-07 02:28; the val arena and the sealed banks also
carry `detour_wide_L/R` and all 8 speed pairs per sequence):

| arena | runs | feasible (completed, no stall, no contact) | stall | timeout | off-route | rollover |
|---|---|---|---|---|---|---|
| arena_f101 | 441 | 340 (77 %) | 32 | 34 | 2 | 2 |
| arena_f102 | 402 | 343 (85 %) | 17 | 28 | 1 | 1 |
| arena_f103 | 314 | 274 (87 %) | 13 | 11 | 0 | 0 |
| arena_f104 | 592 | 451 (76 %) | 47 | 59 | 2 | 0 |
| arena_f105 | 722 | 451 (62 %) | 93 | 114 | 10 | 0 |

By layout kind: freeform 166/228 feasible, sequence 200/290 feasible, crossing 1493/1953 feasible. Sealed banks: f106 439 runs, f107 638 runs. Start / house placement needs ≤ 14° local slope, which skips
roughly half the heading variants on these rough surfaces. Chrono on newton, 12 processes, ≈ 6 runs / min.

### 11.3 First arena, frozen model (early read, 2026-09-07 00:55) — `wp7_collect_f101`

441 runs on `arena_f101` (47 layouts): 371 completed, 32 stall, 34 timeout, 2 rollover, 2 off-route; recorded
episodes 101–600 frames (mean 368). The camera start-pose estimate is 0.07 m off on average (max 0.18 m). The
sequence layouts are two-sided as intended: on `crater9_hill4_t+35` the slow pairs (3/3, 5/7) complete and the
fast ones (7/4, 8/3, 8/8) stall; on `crater6_crater9_t-60_sh` only 6/6 completes (slower pairs time out, faster
ones too); on `hill2_crater8_t+60` the cheapest pair is 7/4 (211 kJ) against 299 kJ for 8/8 and 771 kJ / 28 s
for the rule profile; one sequence layout has no feasible route at all.

The frozen arena_v1 model (`wp2_mapv2_pt_dag_ro8_amd`) imagined 104 of these runs on 16 layouts: it rejects 1 of
35 infeasible routes (AUC of "rejects" 0.51; imagined time AUC 0.69, imagined max roll 0.78), imagined energy is
1.36 under Chrono (corr 0.59), time 1.05 (corr 0.91); its pick among each layout's routes is infeasible on 1/13
layouts, regret 1.06 among feasible picks. This is the baseline the models trained on f101–f104 have to beat on
f105 and the sealed arenas.

### 11.4 Validation arena (`arena_f105`, 722 routes on 65 layouts, 271 infeasible; 13 layouts with no feasible route)

**Training (cluster, `slurm/wp7_finetune.sbatch`, split by arena, val = f105; selection = 5 s state error).**
Frozen arena_v1 model on f105: state MAE 1.00 at 5 s. Fine-tuning it at lr 1e-4 (12k steps; with the arena_v1 tracker
episodes mixed in, `wp7_ft_mix_amd`, or on the new arenas only, `wp7_ft_new_amd`) leaves the validation loss flat
(0.090 → 0.091) and the 5 s error at 0.95–1.05: the model does not move. Training from scratch on the same data
(`wp7_scratch_mix_amd`, lr 3e-4, 20k steps) fits the new arenas far better (val loss 0.068, one-step error 0.216 vs
0.270, 5 s error 0.78–0.85) — but, below, fits do not translate into decisions.

**Rejection of infeasible routes (imagination from rest at the camera start pose, `traverse_wp7_imagine_cache.py`):**

| model | rejects (rightly / wrongly) | accepts infeasible | AUC reject | AUC imagined time | time ratio / corr (feasible) | energy ratio / corr |
|---|---|---|---|---|---|---|
| frozen | 58 (35 / 23) | 236 / 271 | 0.54 | 0.64 | 1.00 / 0.72 | 1.16 / 0.37 |
| fine-tune mixed | 129 (77 / 52) | 194 | 0.58 | 0.62 | 1.05 / 0.72 | 1.14 / 0.38 |
| scratch mixed | 181 (84 / 97) | 187 | 0.55 | 0.61 | 1.00 / 0.54 | 1.09 / 0.30 |
| fine-tune new only | 148 (76 / 72) | 195 | 0.56 | 0.67 | 1.08 / 0.68 | 1.21 / 0.29 |
| cheap predictor, true elevation | — | — | **0.77** (sequences 0.98, crossings 0.78, free-form 0.49) | — | 0.97 / 0.91 | 0.96 / 0.68 |
| cheap predictor, predicted elevation | — | — | 0.70 | — | 0.99 / 0.91 | 0.95 / 0.64 |

Speed alone has AUC 0.43 (faster is more often feasible). The world models still accept most stalls and timeouts
(fine-tune mixed: 65 stalls, 88 timeouts accepted): training on 1 750 episodes with ~330 failures did not teach the
imagination to stall. The cheap predictor, given the terrain profile along the route and the speed profile, is a far
better feasibility classifier, especially on the sequences.

**Picks from the shared bank (`traverse_wp7_pick_table.py`, strict feasibility, cost = time + kJ/10):**

| method | picked / 52 | feasible | regret (mean, max) | feasible rejected | infeasible rejected | abstains on 13 no-solution layouts |
|---|---|---|---|---|---|---|
| rule-based profile | 47 | 31 | 1.44, 2.08 | – | – | 1 |
| **fastest commanded speed** | 52 | **44** | 1.15, 1.56 | – | – | 0 |
| slowest | 52 | 27 | 1.40, 2.40 | – | – | 0 |
| cheap, true elevation (τ 0.5) | 45 | 36 | 1.15, 1.71 | 118 / 451 | 80 / 130 | 8 |
| cheap, predicted elevation (τ 0.5) | 46 | 35 | 1.13, 1.40 | 101 / 451 | 50 / 130 | 5 |
| world model, frozen | 50 | 39 | 1.15, 2.18 | 23 / 451 | 6 / 130 | 2 |
| world model, fine-tune mixed | 50 | 39 | 1.13, 1.54 | 52 / 451 | 29 / 130 | 2 |
| world model, scratch mixed | 48 | 37 | 1.22, 3.68 | 97 / 451 | 24 / 130 | 2 |
| gate fine-tune + cost cheap-true | 50 | 42 | **1.11**, 1.71 | 52 / 451 | 29 / 130 | 2 |

By kind: on the **8 sequence layouts** the fine-tuned world model picks 8/8 feasible with regret 1.05 (frozen 8/8,
1.13; fastest 7/8, 1.08; rule 5/8; the cheap predictor abstains on 5 of 8 at τ 0.5 and is right on the 3 it picks). On
the **39 crossings** the fastest speed wins (33 feasible) against 27–28 for the world models and 28–29 for the cheap
predictors: the models pick slow crossings (`direct_v3/v4`) that stall or time out in Chrono — the momentum regime
the imagination still does not see. Where the fastest pick fails (8 layouts), the best route is usually a detour, and
no learned method finds it (one exception each). Lowering τ to 0.3 lets the cheap predictor pick 48/52 with 40
feasible, regret 1.12 — still below the fastest heuristic.

**Cost model alone (perfect feasibility gate, first choice among the Chrono-feasible routes):** cheap-true regret 1.11,
profile time 1.14, fastest 1.15, world models 1.22–1.26 (energy correlation 0.3–0.4 on this arena). The world
model's ranking of feasible routes is worse than the trivial heuristic here.

One layout (`x_hill1_h180`) has its camera heading estimate flipped by 177° (a symmetric vehicle seen from above;
1 of 248 layouts): every route on it fails in imagination with identical times, which is where the regret maxima
of 5.0 in the perfect-gate analysis come from. The cheap predictor does not use the start pose and is unaffected.

**Two more variants (val):** fine-tune mixed at lr 3e-4 (`wp7_ft_mix_lr3_amd`; 5 s error 0.93) rejects 170 (92 / 78),
picks 37 feasible of 47, regret 1.18; from scratch on the new arenas only (`wp7_scratch_new_amd`; the best fit of
all, 5 s error 0.68, one-step 0.195) rejects 151 (62 / 89), picks 36 of 49, regret 1.18, and on sequences 7/8 (1.07).
A better fit to the recorded states does not make a better selector. **Selected for the sealed evaluation:** the
fine-tuned mixed model (`wp7_ft_mix_amd`, best pick record among the trained models on f105), with the frozen model
as the baseline and the best-fitting model as a secondary; the cheap predictors at τ 0.5; the fastest-speed and
rule-based heuristics.

### 11.5 Sealed evaluation (`arena_f106` + `arena_f107`, run once, 2026-09-07 03:27) — `wp7_cache_sealed`, `wp7_pick_table_sealed_*.json`

1 077 bank routes on 98 layouts (93 with at least one feasible route, 5 with none); 234 routes infeasible. Every method
picks one route per layout from the same bank; the picked route's Chrono outcome is the score. Reported in the
pre-registered order — successful selections, cost among successes, rejection of feasible options.

| method | picked / 93 | **feasible** | regret mean (max) | feasible routes rejected / 843 | infeasible rejected / 193 | abstains on 5 no-solution layouts |
|---|---|---|---|---|---|---|
| rule-based profile | 79 | 53 | 1.53 (4.40) | – | – | 2 |
| **fastest commanded speed** | 93 | **86** | 1.15 (1.81) | – | – | 0 |
| slowest | 93 | 55 | 1.40 (2.51) | – | – | 0 |
| cheap predictor, true elevation | 86 | 78 | 1.15 (4.48) | 102 | 82 | 2 |
| cheap predictor, predicted elevation | 85 | 76 | 1.16 (4.48) | 120 | 71 | 4 |
| world model, frozen (arena_v1) | 93 | 80 | 1.15 (2.52) | 5 | 4 | 2 |
| world model, fine-tuned mixed (selected on f105) | 93 | 82 | 1.15 (2.20) | 83 | 55 | 1 |
| world model, scratch new-only (best fit) | 88 | 77 | 1.19 (4.48) | 133 | 59 | 1 |
| gate world model (fine-tuned) + cost cheap-true | 93 | 85 | 1.17 (4.48) | 83 | 55 | 1 |

Paired against the fastest heuristic on the 93 layouts: fine-tuned world model feasible where the heuristic is not
0 times, the reverse 4 times, cost difference where both feasible +0.74 ± 1.09 (n = 82); cheap-true +2 / −10,
+0.11 ± 1.39 (n = 76); frozen +2 / −8, +1.00 ± 1.14. By kind (both arenas): crossings (68) fastest 64, fine-tuned
60, cheap 60/65; sequences (11) fastest 8 (regret 1.07), fine-tuned 8 (1.26), cheap 5 of 8 picked (1.04) — the
validation arena's 8/8 did not carry over; free-form (14) fastest 14 (1.20), fine-tuned 14 (**1.11**), cheap 13
(1.06). Per arena: on f107 (the milder one, 55 layouts) the cheap predictor is best on cost (51 feasible, regret
1.09 vs 1.15 for the heuristic); on f106 (38 layouts) the heuristic leads (35 vs 31 / 27).

Per-route fidelity on the sealed routes: fine-tuned world model rejects 161 (78 rightly, 83 wrongly), accepts 156
of 234 infeasible (AUC of the reject flag 0.62, of imagined time 0.76); time 1.06 / corr 0.90, energy 1.15 /
corr 0.58 on feasible routes. Frozen: rejects 28, accepts 211 (0.55 / 0.76); energy 1.16 / 0.52. Cheap predictor
(true elevation): AUC 0.78 (crossings 0.73, sequences 0.80, free-form 0.82); time 0.98 / 0.88, energy 1.01 /
0.66; with the map head's elevation 0.74.

### 11.6 Where this leaves the thesis (2026-09-07)

* **The learning comparison is negative for selection.** On two sealed terrain instances, neither the world model
  fine-tuned on four new arenas nor a cheap crossing predictor given the same terrain profile beats "drive the
  fastest candidate" (86 / 93 feasible, regret 1.15): the world model is 82 / 93 at the same regret, the cheap
  predictor 78 of 86 picked. Training did what it could be expected to do — the fine-tuned imagination rejects
  78 infeasible routes where the frozen one rejected 23 — but at the price of 83 false rejections, and it still
  accepts two thirds of the stalls and timeouts. The imagination does not reproduce the stall regime, on this data
  and with this loss; the terrain-profile predictor classifies feasibility far better (AUC 0.78 vs 0.62) yet also
  does not convert that into better picks at the pre-registered threshold.
* **Where the world model does earn something:** the cost ranking of free-form routes (regret 1.11 vs 1.20 for the
  heuristic, n = 14) and, on the validation arena only, the momentum-then-braking sequences (8 / 8 at 1.05); the
  sealed sequences (8 / 11 at 1.26) do not confirm the latter.
* **Why the heuristic is so strong here:** on this family feasibility increases with speed (speed alone has AUC
  0.43–0.57 for infeasibility, the wrong direction for a "slow is safe" prior), the speed-penalised and detour-only
  cases are a minority, and where the fastest pick fails the answer is usually a detour that no learned method
  finds either. A benchmark on which the heuristic is beatable needs more layouts whose best route is slow or a
  detour — or a different cost that penalises speed (energy weight, ride severity).
* **Two live-input defects surfaced:** the camera heading flips by ~180° on 1 of 248 layouts (symmetric vehicle
  from above; every imagined route on that layout fails), and the map head's decoded elevation on the new arenas
  costs the cheap predictor 0.04 AUC against the true field.
* **Candidate next steps (not started):** (1) make the imagination fail where the vehicle stalls — weight failure
  episodes and their final seconds in the rollout loss, or add an explicit stall / progress head trained on the
  recorded outcomes; (2) rebalance the benchmark toward speed-penalised and detour-only layouts before spending
  more on models; (3) fix the heading ambiguity in the pose head (velocity-direction or two-frame cue) — cheap and
  independent of the thesis question.

## 12. Why the imagination accepts runs that Chrono stalls (2026-09-06, after the §11 review) — `traverse_wp7_stall_diagnosis.py`, `wp7_stall_diag/`

The §11 review asked for a diagnosis before any change to the benchmark or the loss: take the runs the world
model accepts and Chrono stalls, and find out whether the dynamics model is wrong or something else is (the
tracker, the camera map, the start-pose estimate, the evaluation's attitude limits, the tracker's action offsets).
Everything below uses the recorded schema-v2 caches (`wp7_cache_v1`, `wp7_cache_sealed`), the three dynamics
models (frozen `wp2_mapv2_pt_dag_ro8_amd`, selected fine-tuned `wp7_ft_mix_amd`, from-scratch `wp7_scratch_new_amd`)
and the tracker `wp3_tracker_v1`. Printouts: `wp7_stall_diag/classify.txt`, `analyze_{f105,f105cam,train,sealed}.txt`,
`predictable.txt`; per-run trajectories in `model_tests_*.json` (not committed, 130–190 MB each).

### 12.1 What stalls the vehicle in Chrono — `classify`

Every non-feasible run gets a class from its own 17-D trace and the true terrain: where it ended (progress along
the route), whether it ever launched, and what it was doing around the stop (throttle, engine torque, wheel speeds
→ slip, tire loads → wheel lift, pitch and the terrain slope 2 m ahead). 846 non-feasible runs on the seven arenas:

| class | n | what it is |
|---|---|---|
| contact | 140 | drove the whole route, touched an asset (infeasible by the strict rule only) |
| launch | 209 | never got 2 m from the start (126 stall aborts, 83 timeouts) |
| stop | 417 | moved, then stuck en route (161 stall aborts, 256 timeouts) |
| stall_recovered | 49 | ≥ 1 s cumulative stall, then finished |
| off_route / rollover | 21 / 10 | |

**One mechanism.** Stops happen at a median 15 m of a 39 m route, on a slope ahead of +10° (p25 +4°, p75 +16°),
nose up 10°, with a wheel unloaded on 76 % of the frames around the stop and a wheel-speed excess of 41 m/s over
the ground speed (a free wheel spinning at ~90 rad/s), throttle 0.52, engine torque 168 Nm: the vehicle loses a
wheel's contact on the 0.15–0.28 m roughness while climbing, and the HMMWV's open axle and central differentials
(`HMMWV_Full`, AWD shafts driveline, TMEASY tires, rigid heightmap) send the torque to the wheel in the air.
Tags: `stop|slope+lift+spin` 173, `stop|slope+spin` 94 (intermittent contact, no 50 % lift), `stop|lift+spin` 36,
`stop|spin` 34, the rest low-throttle variants. Launch failures are the same thing at rest: 78 % of them start
with a wheel already unloaded (the vehicle settles on the roughness with a diagonal wheel in the air, e.g.
`arena_f105__x_hill1_h000`: rear-left 0 N, front-right 3.6 kN at rest; both spin up to 100 rad/s under throttle,
the loaded pair never turns, 11 of 13 candidates fail, the two that pass steer left at launch). Tire-load
caveat: on healthy runs some wheel reads < 500 N on 49 % of frames (rigid-heightmap contact, 2.3 cm quantisation
steps), so the loads flicker; the wheel-speed excess is the robust traction signal.

**Momentum is the decision.** Over the direct-crossing speed ladders of 241 layouts: 132 pass at every speed,
65 fail slow and pass fast, 4 the reverse, 16 mixed, 24 fail at every speed. On 48 of the 65 momentum layouts
(where the slowest failing run stopped en route), the next-faster candidate passed the same station at 2.66 m/s
(median; dipping to 1.73 m/s in the next 2 s) where the failing run had 0.65 m/s before stopping; slope ahead
+8.5°. One step of commanded speed (1 m/s) separates the stall from the pass.

**What the imagination makes of it** (accepted / n with rows, frozen · fine-tuned · scratch): launch 111 · 96 · 93
of 146; stop 224 · 173 · 191 of 239; contact 71 · 46 · 62 of 76. On the 38 momentum layouts with imagination rows
the frozen model accepts every speed on 38 / 38; the fine-tuned model on 28 / 38 and reproduces Chrono's pattern
on 2 / 38. The imagined slow crossing simply takes longer (14.7 s at v2 against 7.1 s at v9) instead of stalling.

### 12.2 Where the prediction goes wrong — `model`, `analyze`

Per run, one imagination env per arena and model, the recorded episode's own scene map and route: (a) from rest
with the RECORDED controls (teacher forcing — no tracker, no camera pose); (b) the same with the tire loads set to
a healthy 6.25 kN pattern; (c) the same with another layout's scene map; (d) from rest with the tracker (the
planner's imagination; true start pose, and the camera estimate in `analyze_f105cam.txt` — identical within
0.05 m/s); (e) from the recorded 16-frame context 2 s before the stop with the recorded controls; (f) with another
layout's map; (g) with the tracker; (h) seeded 1 s INSIDE the stall (stationary, throttle on, wheel spinning);
(i) local k-step errors from the recorded context every 0.2 s from 2 s before to 1 s after the stop, k = 8
(0.4 s, the training rollout horizon) and k = 20 (1 s). Feasible runs on the same layouts are the controls, their
"stop" being the frame at which they passed the station where the layout's stalled runs stopped.

Validation arena `arena_f105` (105 launch failures, 102 stops, 265 feasible controls), speeds in m/s:

| test | Chrono | frozen | fine-tuned | scratch |
|---|---|---|---|---|
| launch, (a) rest + recorded controls, vx at 3 s | 0.31 | 2.03 (57 % > 1) | 2.29 (77 %) | 2.44 (78 %) |
| launch, (b) healthy loads | | 3.12 | 3.37 | 3.22 |
| launch, (c) wrong map | | 2.00 | 2.30 | 2.43 |
| launch, (d) rest + tracker, completes the route | 0 % | 75 % | 78 % | 67 % |
| stop, (e) context 2 s before + recorded controls, vx 2 s after the stop | 0.17 | 2.53 (25 % < 0.5) | 2.32 (26 %) | 2.44 (19 %) |
| stop, (f) wrong map | | 2.57 | 2.31 | 2.44 |
| stop, (g) tracker | | 1.90 (23 %) | 1.98 (23 %) | 2.01 (17 %) |
| stop, (h) seeded inside the stall, vx 4 s later | ≈ 0 | 1.69 (39 % < 0.5) | 1.95 (32 %) | 1.81 (31 %) |
| stop, (i) k = 8 error before / after the stop | | +0.14 / +0.16 | +0.12 / +0.16 | +0.03 / +0.04 |
| stop, (i) k = 20 error before / after | | +0.37 / +0.36 | +0.32 / +0.35 | +0.17 / +0.12 |
| feasible, (e) at the matched station, vx 2 s later | 3.30 | 4.61 | 4.37 | 4.32 |
| feasible, (i) k = 8 / k = 20 | | +0.01 / +0.13 | +0.01 / +0.09 | −0.03 / 0.00 |

Sealed arenas (41 launch, 137 stop, 450 controls) and the training arenas (63 / 178 / 463) are in
`analyze_sealed.txt` and `analyze_train.txt`; the stop-reproduction rates in (e) are 34 · 29 · 28 % sealed and
32 · 43 · 41 % on the training arenas; the launch-failure rates 24 · 49 · 51 % sealed, 33 · 56 · 62 % train.

1. **It is the dynamics model, not the controller, the map or the localisation.** Given the true state and the
   recorded controls, 2 s after Chrono's stop every model still predicts 2.3–2.5 m/s. Swapping the scene map for
   another layout's changes that by < 0.05 m/s; the tracker lowers it a little (1.9–2.0); the camera start pose
   changes nothing. The models do not even hold a stall they are placed inside: from 1 s into a stall they are
   back at 1.7–1.95 m/s four seconds later.
2. **Short-horizon accuracy is not what is missing.** At the training horizon (0.4 s) the signed speed error around
   the stop is +0.03 to +0.16 m/s; the scratch model has the smallest local errors (+0.03 / +0.17 at 1 s) and the
   worst stall reproduction (19 % of stops, 12 % of launch failures). The stall is a regime ("throttle on,
   stationary, free wheel spinning") that a small positive per-step drift leaves within a second or two, after
   which the imagined state is a moving vehicle and the throttle does what it always does.
3. **The models are optimistic about speed on the rough climbs even where Chrono passes:** teacher-forced from
   the matched station, +0.4 m/s at 2 s and +1.0 m/s at 3–4 s for every model. The closed-loop time is still
   within 3–7 % of Chrono's (ratio 1.03–1.07, corr 0.82–0.92) because the tracker absorbs the optimism; the
   momentum margin that decides a pass is exactly what it hides.
4. **Launch failures:** from rest with the recorded controls the models launch the vehicle (2.0–2.4 m/s at 3 s,
   57–78 % above 1 m/s) where Chrono never moved. They do read the tire loads — a healthy pattern raises the
   prediction by ~1 m/s — but not nearly enough; fine-tuning made this worse (57 → 77 % launched).
5. **The trained models learned some of it on their own arenas and little of it elsewhere:** 43 % / 41 % of the
   training-arena stops and 56 % / 62 % of the launch failures are reproduced from the recorded context, against
   29 % / 28 % and 49 % / 51 % on the sealed arenas; the closed-loop imagination from rest still completes
   67 % / 59 % of the training-arena failures.
6. **Local ranking vs. from rest.** With the true state 2 s before the stop, the model's own predicted speed
   separates stops from matched passes at AUC 0.74–0.84 (f105); from rest, the planner's setting, the same
   comparison is at chance (0.48–0.59): the imagined approach to the stall point is faster than Chrono's and the
   stall depends on the local state.

### 12.3 Is the stall predictable from what the model is given? — `predictable`

Small MLPs trained on the training arenas from the model's own inputs (16-frame 17-D context + controls, plus
the next 1–4 s of recorded controls), label "stuck N s later", judged on the sealed arenas (`predictable.txt`):

| lead | inputs | AUC val | AUC sealed | sensitivity at 5 % false positives |
|---|---|---|---|---|
| 1 s | context + future controls | 0.83 | 0.85 | 0.31 |
| 2 s | context + future controls | 0.84 | 0.86 | 0.32 |
| 2 s | context only | 0.83 | 0.85 | 0.33 |
| 2 s | last frame + controls | 0.83 | 0.86 | 0.24 |
| 2 s | last frame without tire loads / wheel speeds | 0.83 | 0.86 | 0.31 |
| 4 s | context + future controls | 0.84 | 0.86 | 0.31 |

From the state trajectory, only about a third of the stalls are foreseeable 1–4 s ahead at a 5 % false-alarm
rate, and the tire loads / wheel speeds add nothing to that. The world model's local prediction (25 % of stops
at 3 % false positives, 2 s ahead) is at that level. Launch failures from the rest state: tire loads + wheel
speeds alone AUC 0.78 val / 0.81 sealed (the perched signature is real); the full 20-D input overfits the 63
training positives (0.63 sealed).
**Terrain probe (`--terrain`, `predictable_terrain.txt`): the fine terrain does not carry the missing information
either.** Adding the TRUE height field as a 0.25 m ego patch (6 m ahead × 3 m wide, 325 samples, relative to the
vehicle's own height) to the same classifier at 2 s lead gives AUC 0.85 → 0.86 sealed and sensitivity 0.32 → 0.35;
the patch alone gives 0.76 / 0.17; the coarse 1.4 m grid the ego crop uses gives 0.85 / 0.30. Caveat: a small MLP
on raw heights with 1 700 training positives is a weak extractor, so this is a floor on the terrain's usefulness,
not a proof of chaos — but it is the same extractor that reaches 0.86 from the state, and the privileged terrain
adds three points of sensitivity to it. At a 2 s lead, about a third of the stalls at 5 % false alarms is what
everything observable (state trajectory, controls, true terrain ahead) supports.

### 12.4 The two audit items from the review

* **Attitude limits.** The imagination terminated rollouts at |pitch| > 0.4 rad (23°) or |roll| > 0.6 rad (34°)
  while Chrono's runner aborts at 60° and the feasibility rule caps nothing. Re-imagined with 60° limits
  (`traverse_wp7_imagine_cache.py --roll-limit-deg 60 --pitch-limit-deg 60`, `wp7_imagine_*_lim60`): the
  fine-tuned model's sealed rejections fall 161 → 77 (false 83 → 25, true 78 → 52), f105 129 → 74; the frozen
  model is unchanged (its rollouts never reach the limits); the pick outcome is unchanged (regret 1.14–1.15,
  11 infeasible picks). The limits inflated both kinds of rejection and are not the cause of the selection
  result. One definition for both places from here on: the imagination's termination = Chrono's abort (60°)
  unless the benchmark adopts an explicit attitude cap for both.
* **Tracker action offsets.** The tracker's action centre is the dynamics normaliser's action mean. It is
  identical for the frozen and every fine-tuned model ([−0.003, 0.20, 0.02]; fine-tuning keeps the checkpoint's
  normaliser) but not for the scratch models ([−0.13, 0.28, 0.11] new-only, [−0.06, 0.20, 0.22] mixed — a brake
  centre of 0.22 instead of 0.02). The scratch-model rows of §11.4 are confounded by this; the selected model is
  not. Fix: the tracker env should take its action centre from the tracker's own training normaliser, not from
  whichever dynamics model it is driving.

### 12.5 Reading

* The failure is in the dynamics model: given the true state, the recorded controls, no camera, no tracker and
  no localisation, it predicts motion where Chrono stalls, and it drifts out of a stall it is placed in. The
  0.4 s rollout loss on the 7 % of training frames that are stuck (2 % in the two seconds before a stop) leaves a
  small positive drift in exactly the regime where zero is the answer, and the models are optimistic about the
  speed on rough climbs generally.
* But what any predictor can do here is bounded by the event itself: the wheel lift that starts a stall is a
  contact event on roughness (0.16 m per pixel, 0.15–0.28 m bumps, quantised contact) and from the state
  trajectory a third of the stalls are foreseeable 1–4 s ahead at 5 % false alarms — and giving the classifier
  the TRUE terrain ahead adds three points (§12.3). The momentum decisions on this family are therefore
  statistical: a speed margin, which is what the fastest-candidate heuristic embodies without a model, and what
  no route-by-route imagination will call from rest whether or not it is a world model.
* What would still change the result: (1) a stall / progress head trained on the recorded outcomes (it can learn
  the statistic — "slow on a +10° rough climb stalls a third of the time" — which is what the pick needs, and what
  the one-step dynamics loss never sees); (2) a rollout loss that holds the stuck regime (failure-weighted, longer
  horizon) so the imagination at least stops where it is placed inside a stall; (3) the audit fixes (one attitude
  definition, the tracker's own action centre, the heading flip). Not: finer terrain input, and not longer
  fine-tuning of the same loss. Whether (1)–(2) beat the fastest-candidate heuristic on a benchmark that rewards
  speed is doubtful; they are worth it only on a benchmark with speed-penalised and detour-only layouts (§11.6).

## 13. Stall-reproduction ablation (2026-09-06, plan §30 review adopted) — `slurm/wp8_launch.sh`, `wp8_eval/`

The §12 review disagreed with reading the predictability probe as a ceiling and asked for one focused training
experiment on the existing data before anything else: balance the training around approaching a stall, staying
stuck, recovering, and matched successful crossings; train and judge over 2–4 s of recorded motion with progress
and stall persistence, not only the mean state error; first show the model reproduces known training examples,
then unseen terrain and route selection with the benchmark unchanged. The user's instruction: do a large ablation on
the cluster rather than detour around the problem. The model is a data-driven dynamics model; if the stall regime is
under-weighted (§12.5: 7 % of training frames stuck, 2 % in the two seconds before a stop; 61 % of the selected
model's windows from the gentle arena_v1 data; a 0.4 s rollout loss), training should be able to fix it.

### 13.1 What was added to the trainer (`traverse_wp2_train_map.py`)

* **Stall events** (`traverse_wp7_stall_diagnosis.py events` → `<cache>/events.json`): per episode the class,
  the stop frame, launch failure, the resume frames of recoveries (≥ 1 s cumulative stall with throttle on, then
  vx > 0.5 m/s), and for feasible episodes the frames at which they pass the stations where the layout's stalled
  siblings stopped. `wp7_cache_v1`: 280 stops, 168 launch failures, 204 recoveries, 1 091 matched frames.
* **Event-balanced sampling** (`--event-frac p`, `MapBatcher.event_table`): a fraction p of every batch is drawn
  from windows whose rollout span contains an event — `approach` (the stop inside the span), `stuck` (context and
  span after the stop), `launch` (the first frames of a launch failure), `recovery` (the resume inside the span),
  `matched` (a feasible sibling passing the stalled runs' station) — balanced over the kinds, uniform over windows
  within a kind; the rest of the batch as before (uniform over recorded windows). At K = 40 the training arenas
  offer 7 120 / 2 764 / 2 583 / 3 694 / 26 379 such windows.
* **Longer rollout horizons** (`--rollout-steps` 8 / 40 / 80 / 120 = 0.4 / 2 / 4 / 6 s of autoregressive
  prediction under the recorded controls, map re-cropped at the dead-reckoned pose).
* **Progress loss** (`--progress-weight`): Huber on the cumulative distance along the body axis (metres) at every
  rollout step — the quantity a stall zeroes and a drift inflates; `--vx-weight` scales the speed channel in the
  state losses; `--delta-scale` as before.
* **Stall validation metrics** (`stall_eval`, on the validation arena's event windows under the recorded controls,
  the §12.2 tests in-training): `stuck` (context 0.2 s after the stop, 3 s) → predicted |vx| at the end and the
  fraction under 0.5 m/s (`hold`); `approach` (context ends 2 s before the stop, 4 s) → predicted vx 2 s after the
  stop; `launch` (frames 0–15, 3 s); `recovery` (context ends 1 s before the resume, 3 s) and `matched` (a feasible
  sibling at the stalled runs' station, 4 s) → |predicted − recorded| vx, the guards against "always stop";
  `stall_score` = the mean of the five, in m/s; `--selection stall_score` picks `ckpt_best.pt` by it.

### 13.2 The grid (25 runs, `slurm/wp8_launch.sh`, mi3501x, one MI350 each)

All on the new data (`wp7_cache_v1`, train f101–f104, val f105), batch 256, fine-tuned from the frozen
`wp2_mapv2_pt_dag_ro8_amd` at lr 3e-4 unless noted:

| axis | runs |
|---|---|
| event fraction × horizon | p ∈ {0, 0.3, 0.6} × K ∈ {8, 40, 80} (`wp8_p{0,3,6}_k{8,40,80}`; 12 k steps, 10 k at K = 80) |
| loss, at (p .3, K 40) and (p .6, K 80) | progress weight 1 (`_prog`), delta-scale (`_ds`), progress + vx weight 5 (`_progvx`) |
| init / data, at the same two points with progress | from scratch on the new data (`_scratch`, 20 k / 14 k steps); fine-tuned on new + arena_v1 tracker episodes (`_mix`) |
| learning rate | lr 1e-4 (`_lr1`) at the same two points with progress |
| seeds | (p .6, K 80, progress) seeds 1, 2 |
| heavier / longer | p 0.9 at K 80 with progress; K 120 (6 s) at p 0.6 with progress (8 k steps) |

Throughput on MI350 at K = 8 was 2 880 samples/s (≈ 675 steps/min); the rollout part scales with K, so K = 80
runs are budgeted at 10 k steps inside the 3 h 50 walltime, with `ckpt_last.pt` every 1 000 steps as the fallback.

### 13.3 Evaluation (local, `traverse_wp8_eval_runs.sh` → `wp8_eval/<run>/`, `traverse_wp8_leaderboard.py`)

Every finished run (and the frozen and the §11 fine-tuned model as baselines) gets, with the audit fixes in place
(60° attitude termination = Chrono's abort; the tracker's own action centre whatever dynamics model it drives):
(1) the §12.2 stall tests on the validation arena f105 (`analyze_f105.txt`: from the recorded context, does it
predict the stop / hold a stall / keep a launch failure stationary, and does it still move the feasible controls);
(2) the same on the training arena f104 (`analyze_f104.txt`: the "reproduce known training examples" check);
(3) the imagination on every f105 route and the shared-bank pick table (`pick_f105.json`, `pick_f105.txt`): the
selection question with the benchmark unchanged. The sealed arenas are not touched by the ablation; one run of the
chosen configuration on them comes after, with the user.

### 13.4 Results, wave 1 (25 runs) and wave 2 (15 runs) — `wp8_eval/leaderboard_final.txt`

All 40 runs trained and were scored with one harness (`traverse_wp8_eval_runs.sh`). Two defects in that harness were
found afterwards by the independent audit (§13.6) — the "stop" class included timeouts that were still moving, and the
"stopped" fractions used the signed speed — so the leaderboard's stall columns overstate stall reproduction; the
corrected numbers for the key models are in §13.6. What survives from the leaderboard is the comparison *between*
runs, since every run was scored the same way:

* **Only three axes move the validation stall metrics beyond noise** (an OLS over the 40 runs on the best-checkpoint
  stall score, residual sd 0.12 m/s): the event fraction p (−0.29 m/s from 0 to 0.6), the rollout horizon (−0.36 from
  K 8 to 80, matched pairs −0.28 to −0.31 at every p) and the initialisation (from scratch +0.52). Lower learning
  rate (+0.23) and dropout (+0.20 per 0.1) hurt. The progress loss helps at (p .6, K 80) (0.97 → 0.75) but not at
  (p .3, K 40) (1.16 → 1.37); delta-scale, vx weight, weight decay, input noise, kind weighting, the data mix and the
  combined configuration are inside the noise. Seeds: 0.75 / 0.76 / 0.77 on the score, but the f105 local fractions
  vary by 0.11–0.20 across seeds, so single-pair differences below ~0.15 on those are not interpretable.
* **Every run over-fits early:** validation one-step loss is lowest at the first evaluation (step 1 000) in 28 of 40
  runs and rises by a median 15 % by the end; the stall score is best at ≤ 2 000 steps in 24 of 38 fine-tuned runs
  and worsens by a median 0.18 m/s afterwards while the training loss keeps falling. Longer training is not a lever;
  regularisation did not fix it (dropout runs start worse).
* **Training-arena reproduction saturates with horizon**, not with the event share: every K ≥ 40 run predicts 65–78 %
  of the f104 stops from context (frozen 37 %, K 8 runs 47–54 %) — and the f104 → f105 gap of 0.25–0.30 is the
  generalisation gap the ablation did not close.
* **What the imagination actually learned is to reject slow routes, not stalls:** in every run the closed-loop
  imagination still accepts ≥ 59 % of the f105 stall-abort routes (frozen 86 %, best family 73–86 %), while the
  acceptance of *timeout* routes fell from 64 / 64 to 40 / 64 (primary) or 16 / 64 (p .9, K 120) at the price of
  accepting only 392 / 451 or 252 / 451 feasible routes and imagining feasible routes 1.6 s / 6.9 s slower than Chrono.
* **No axis moves the pick:** own-pick feasible counts span 31–43 of 52 across the 40 runs (mean 38.8, frozen 39,
  binomial SE ≈ 3) and are uncorrelated with every stall metric (|ρ| ≤ 0.08); the only 2-SE effect is input noise,
  which lowers it (0.2 → 31 / 52).

### 13.5 The model as a gate in front of the heuristic; sealed second look (pre-registered) — negative

The models' own cost ranking is what loses, not the gate: keeping each model's accepted set but ranking by Chrono's
true cost gives a feasible pick on 50 / 50 layouts (frozen) and 51 / 51 (primary) — the accepted set almost always
contains a feasible route, and every own-pick failure is a slow candidate the imagination completes as a normal slow
crossing while Chrono stalls or times out (imagined 15.5–17 s where Chrono ends at 27–28 s). The cost time + energy /
10 is what drags the pick into the slow region: it is nearly speed-neutral in Chrono, but the imagined energy gradient
(slow = cheap) outweighs the imagined time penalty, so the own pick averages 3.4–4.3 m/s where the fastest-accepted
averages 5.0 m/s. The imagination's acceptance of *stalling* slow routes has a precision barely above the bank's base
rate (P(feasible | accepted) at v2: frozen 0.40, primary 0.46, base 0.43) and its rejection rate is nearly flat over
speed while Chrono infeasibility falls from 57 % (v2) to 14 % (v7–9).

Using a model only as a rejecter and driving the FASTEST accepted candidate gave 45–47 / 52 on f105 (heuristic 44,
frozen 44; best run `wp8_p6_k80_prog` at 47). That number is what the maximum of 40 exchangeable noisy runs looks like
(distribution over the runs 43:1, 44:13, 45:24, 46:3, 47:1; expected max under the per-layout null 46.5, P(max ≥ 47)
= 0.51), the metric is structurally protected (a wrongly rejected fastest candidate breaks the pick only when the
next-fastest accepted route is infeasible — 2 of 187 times — and rejecting everything falls back to the heuristic),
the individual "fixes" flip between seeds and between `ckpt_best` and `ckpt_last`, and the most-often fixed layout is
a *contact* failure of the heuristic rejected for an unrelated imagined timeout. Checked once on the sealed arenas with
the choice fixed in advance (primary `wp8_p6_k80_prog`, its two seeds, two exploratory runs; 93 layouts; 60° limits,
tracker's own action centre):

| sealed f106 + f107 | own pick | gate + fastest | fixes / breaks vs heuristic | rejected feasible / infeasible (of 843 / 193) |
|---|---|---|---|---|
| fastest heuristic | 86 / 93 (regret 1.15) | | | |
| frozen · earlier fine-tune | 80 · 82 | 86 · 86 | 0 / 0 · 0 / 0 | |
| **primary** | 76 | **85** | 0 / 1 | 113 / 68 |
| seed 1 · seed 2 | 77 · 79 | 85 · 87 | 0 / 1 · 1 / 0 | 147 / 112 · 68 / 52 |
| exploratory p .9 · p .3 | 81 · 80 | 86 · 87 | 0 / 0 · 1 / 0 | 129 / 85 · 159 / 111 |

The gate changes the heuristic's sealed result by −1 to +1 layout; the own picks are below the frozen model's. The
sealed arenas have now been looked at three times (§11.5, §12.4, here) and have no power for this question anyway: the
heuristic fails on 7 of 93 layouts, 3 of them contact-only, so a stall gate's ceiling there is +4 and a paired sign
test at zero breaks would need ≥ 6 fixes. **They must not be used again.** One correction to §13.5's earlier draft: the
7 sealed failure layouts *do* each have a feasible slower alternative, so a perfect gate could in principle help; the
issue is power and the model, not the benchmark's ceiling.

### 13.6 Audit corrections (independent review, 2026-09-07) and the corrected numbers

The multi-agent audit of the artifacts found, and I confirmed and fixed:

1. **The "stop" class was mis-defined.** `stuck_from_displacement` could never report a vehicle still moving over the
   final 2 s window (the tail without a full window counted as stuck), so `crawl` never fired and every timeout that
   moved ≥ 2 m became a "stop" at frame n − 40. 176 of the 280 training-cache "stops" were such timeouts, 161 of them
   with mean |vx| > 0.5 m/s over the "stuck" tail; at the frame the stall tests score, Chrono itself had |vx| < 0.5 in
   only 40 of the 102 f105 "stops". Fixed: the class now requires a stationary tail (or the stall abort); the
   corrected caches hold 221 stops (161 stall aborts + 60 stationary timeouts), 196 crawls, 209 launch failures.
2. **Signed speed** in the "predicts the stop" / "holds the stall" fractions (§12.2, §13.4–13.5): rolling backwards
   counted as stopped. Fixed to |vx|.
3. **The trainer's stall validation**: the "approach" term scored |predicted vx| against an implicit zero although the
   recorded speed at that frame averaged 1.2 m/s (because of defect 1); the "stuck" term had 17 windows; the
   "recovery" events fire at frame 40 for any vehicle that sat with throttle on during the launch window, so the guard
   is mostly a launch metric (genuine en-route recoveries: 24 train, 9 val). The events file is regenerated from the
   corrected classes; the trainer's stuck window is 2 s; the approach target and the recovery detector are still to
   be fixed before any retraining.
4. **Over-holding**: seeded inside a stall, the primary holds 34 / 42 windows where Chrono was stationary but also
   41 / 60 where Chrono moved on — per-episode agreement with Chrono 0.52 against the frozen model's 0.67 (f105).
5. **Reporting**: the leaderboard's in-stall regex dropped negative predicted speeds (35 of 40 runs blank in that
   column); the heading-flipped layout `x_hill1_h180` (§11.6) costs every model 13 false rejections and one own pick,
   so the frozen model's "23 rejected feasible" is really 10; the gate+fastest counts depend on whether the
   slope-aware profile is eligible (47 → 46 without it).
6. No leakage of the validation arena into training; the evaluation is deterministic (re-running reproduced every
   committed number); the 60° termination and the tracker's action centre were applied uniformly within wp8 (the
   action-centre fix only affects the two scratch runs).

**Corrected stall tests** (`wp8_eval/analyze_v2_{f105,f104,sealed}.txt`; true stops only; |vx| < 0.5 m/s):

| from the recorded context | frozen f105 / f104 / sealed | primary `wp8_p6_k80_prog` | `wp8_p9_k80_prog` |
|---|---|---|---|
| stop predicted 2 s after Chrono's stop (n 46 / 48 / 75) | 0.24 / 0.25 / 0.23 | 0.39 / 0.58 / 0.44 | 0.50 / 0.56 / 0.39 |
| mean predicted speed there, m/s (Chrono 0.28 / 0.19 / −0.05) | 2.38 / 1.51 / 1.65 | 0.45 / 0.21 / 0.50 | 0.23 / 0.18 / 0.52 |
| stall held when seeded 1 s inside, 4 s later | 0.22 / 0.27 / 0.32 | 0.54 / 0.67 / 0.55 | 0.61 / 0.67 / 0.61 |
| launch failure kept stationary (n 105 / 28 / 41) | 0.20 / 0.18 / 0.24 | 0.55 / 0.71 / 0.41 | 0.39 / 0.64 / 0.32 |
| mean predicted speed there (Chrono 0.0 / 0.4 / −0.1) | 1.02 / 1.40 / 1.65 | 0.01 / 0.42 / −0.33 | 0.11 / 0.49 / −0.35 |
| false stops on feasible controls (n 146 / 167 / 364) | 0.03 / 0.01 / 0.06 | 0.01 / 0.00 / 0.03 | 0.00 / 0.00 / 0.03 |

So the honest size of the training effect: on unseen terrain the stall-trained model calls two-fifths to a half of the
true stops (frozen a quarter), holds half to three-fifths of seeded stalls (a quarter to a third), keeps two-fifths to
half of the launch failures stationary (a fifth to a quarter), with no false stops — and predicts speeds near zero
where the frozen model predicts 1.5–2.4 m/s. On the training arena it reaches 0.58 / 0.67 / 0.71. Real, seed-stable
(seed 2: 0.37 / 0.57 / 0.41 on f105), sealed-confirmed, and smaller than §13.4's draft claimed.

### 13.7 Decision from the real pre-stall state (reviewer step 2), and what stands between context and closed loop

`traverse_wp7_stall_diagnosis.py decision` (`wp8_eval/decision_v2_f105.log`). For every f105 layout with a true stop or a
launch failure, the decision point is 5 m before the *earliest* stop on that layout (rest for launch failures), so every
run on the same path is still moving there; each run is seeded from its recorded context at that point and rolled 8 s
(a) with its recorded controls, (b) with the tracker; predicted stuck = |vx| < 0.5 m/s at the end or < 0.5 m of progress
in the last second. 105 windows on 33 layouts (33 stuck within 8 s, 73 infeasible runs). A cheap MLP trained on the
training arenas from the same context (and the same future controls) is the comparison.

| predictor | AUC stuck within 8 s | AUC run infeasible | sensitivity at 5 % false alarms | outcome accuracy per candidate |
|---|---|---|---|---|
| frozen, recorded controls | 0.76 | 0.56 | 0.55 | 0.44 |
| earlier fine-tune, recorded controls | 0.77 | 0.64 | 0.48 | 0.50 |
| **primary, recorded controls** | **0.89** | **0.84** | **0.76** | **0.61** |
| seed 2 · p .9, recorded controls | 0.86 · 0.87 | 0.80 · 0.83 | 0.73 · 0.64 | 0.53 · 0.57 |
| primary, tracker in the loop | 0.71 | 0.68 | 0.39 | 0.38 |
| seed 2 · p .9, tracker | 0.44 · 0.76 | 0.51 · 0.62 | 0.18 · 0.45 | 0.30 · 0.42 |
| cheap classifier, context + recorded controls | 0.67 | 0.91 | 0.09 | |
| cheap classifier, context only | 0.71 | 0.84 | 0.00 | |
| always-pass baseline | | | | 0.30 |

* **Given the real state and the real controls, the stall-trained model discriminates.** Runs that get stuck within
  8 s are ranked at AUC 0.89 (frozen 0.76) with 76 % caught at 5 % false alarms — the cheap classifier on identical
  inputs reaches 0.67 and 9 %. The classifier is *better* on the run-level label (0.91: it reads "slow commanded
  speed ⇒ infeasible" off the future controls) and blind to the local event; the world model is the reverse. This
  refutes §12.3's reading of the classifier probe as an information ceiling for the local prediction.
* **With the tracker generating the controls the discrimination collapses from the same state** (0.89 → 0.71,
  0.84 → 0.68; a seed to chance). It is not a throttle bias: shifting the recorded throttle by ±0.2 leaves the
  teacher-forced AUC at 0.85–0.89 (+0.4: 0.82; `decision_f105_throttle.log`). The tracker — trained inside the
  *frozen* model's imagination — issues a control sequence the stall-trained model has not been paired with, and the
  closed loop between the two drifts away from the recorded situation. This is the closed-loop gap of §13.4 located:
  not the state, not the horizon, not the throttle level — the controller–model pairing.
* **There is no decision to win here:** only 5 crossing ladders on f105 have a feasible candidate at the decision
  point and the fastest candidate passes on all 5, so no gate beats "drive fast" on this family (the primary as a
  gate matches it with the recorded controls, 5 / 5; the frozen model loses 1; every tracker-driven gate loses 2–3).

### 13.8 Reading and the next step

* **The reviewer's two open possibilities resolve as follows.** The imagination does *not* fail to distinguish the
  physics near the event: with the real controls it distinguishes well, on unseen terrain, and better than a cheap
  classifier from the same inputs. It fails in closed loop because the controller it is paired with was trained in a
  different imagination, and from rest because the whole approach is imagined. Neither more terrain sensing nor more
  sampling addresses that; retraining the tracker inside the stall-aware imagination does, and is the prerequisite for
  the replanning-with-state-updates design the review pointed to.
* **The selection endpoint on this family is not the right test bed.** The fastest candidate is feasible on 44 of 52
  validation and 86 of 93 sealed layouts; half of its failures are contact-only; the bank's slow candidates that stall
  are ones nobody needs to pick. A stall gate's ceiling is +4 layouts on the sealed set and the sealed set is spent.
* **Next round, in order** (revised in §13.9 after the fingerprint finding): (1) fix the remaining metric defects
  (approach target, recovery detector) and regenerate the events; (2) retrain with control-input augmentation so the
  stall must be read from the state (§13.9, wave 3) and select by a jitter-robust or tracker-in-the-loop test;
  (3) generate fresh sealed arenas (f108+) with a committed seed and a difficulty chosen model-free so that the
  fastest heuristic fails on ≥ 30 layouts that still have a feasible alternative (verified by running only the
  heuristic's bank in Chrono), commit a pre-registration (checkpoint by rule; decision rule; abstain handling;
  contact-only and no-solution layouts reported apart), and look once; (4) the tracker retraining and the replanning
  planner only if (2) makes closed-loop discrimination follow the teacher-forced one. Not: more training-axis sweeps
  of the same loss on the same events.

### 13.9 The stall fingerprint (audit, verified 2026-09-07) and wave 3

The audit's closed-loop analyst found, and an adversarial verifier reproduced, what the stall-trained models actually
read: **the constancy of the recorded throttle while the vehicle is stuck.** Chrono's tracker holds its output when
nothing moves (per-step |Δthrottle| 0.015 stuck against 0.031–0.037 moving, at the same level 0.37–0.49). Adding
zero-mean per-step throttle jitter of σ = 0.03 to the recorded controls — smaller than the imagined tracker's own
variability (0.026) — removes most of the reproduced stalls of `wp8_p6_k80_prog` (true stops predicted 2 s after the
stop 0.47 → 0.09, seeded stalls held 0.54 → 0.17, launch failures held under 1 m/s from rest 68 % → 7 %; p .9 / K 120
0.56 → 0.00) and leaves the frozen model unchanged (0.26 → 0.23, 0.41 → 0.42); the effect scales with the jitter
(σ = 0.01: 0.47 → 0.33) and steering jitter barely matters. Conversely, feeding the *tracker's* actions open-loop
reproduces the closed loop exactly, and feeding recorded steering with tracker throttle launches the launch failures
(vx at 3 s 1.83 m/s against 0.66 with all-recorded controls). Two further findings: the from-rest reproduction on the
training arena is route memorisation (a corrected wrong-map probe — a different layout's map for every env, since
rolling the maps by one had left most envs with their own layout — and the healthy-load probe change it by < 0.05),
and the "stops" that anchored the tests were mostly the last 2 s of a 30 s struggle (the vehicle first drops under
1 m/s at a median 6 s and rocks for a median 16 s before the detected stop), so the tests measured holding a
fifteen-second-old stall, not the loss of momentum a planner must foresee.

This re-reads §13.7: the teacher-forced discrimination (AUC 0.89) is largely the control fingerprint, the
tracker-in-the-loop number (0.71) is the honest one, and the closed-loop gap is not a controller–model pairing problem
(any live controller varies its throttle) but a shortcut in what was learned. Retraining the tracker would not fix it.

**Wave 3** (`slurm/wp8_launch3.sh`, 8 runs, 2026-09-07 01:00): the wave-1 primary configuration retrained with
control-input augmentation — per-step Gaussian jitter on the throttle and steering *inputs* (σ 0.02 / 0.03 / 0.05 /
0.10, physical units; targets unchanged) and, in two runs, a random 0.5 s smoothing of the inputs with probability
0.3 — with the checkpoint selected by the stall score computed under σ = 0.03 jittered recorded controls
(`--stall-eval-jitter`), plus an un-augmented control run selected the same way and a second seed. Events are the
corrected ones (§13.6). Evaluation adds the jitter probe (`pre_rec_jit`, `stuck_rec_jit`) to every analyze table and
uses the corrected wrong-map probe. The question is single: do the local stall metrics survive the jitter probe, and
does the tracker-in-the-loop decision test (§13.7) then move toward the teacher-forced one?

**Audit synthesis, corrections to §13.4–13.5 (16-agent workflow, `wp8_eval_summary/audit_workflow_result.json`).**
Five axes move the validation stall score beyond noise, not three (learning rate 1e-4 and dropout hurt); p and K act
as thresholds (p > 0 is a step of about −0.19 m/s that saturates by 0.3; K is a step between 40 and 80), the
progress-loss effect flips sign between the two grid points and is not identified, the data mix helps at (p .3, K 40,
progress) by −0.22, the combined configuration is worse because of its dropout, and the K 120 run diverged at step
4 000 (its row is the 2 000-step checkpoint). "Imagining feasible routes 1.6 s / 6.9 s slower" was an averaging
artefact over imagined timeouts: on *accepted* feasible routes the imagined time is unbiased (median ratio 0.98).
The own-pick loss is not "entirely the cost ranking": 1–2 layouts per model are all-rejected and won only by the
fallback, contacts are 2–4 of each model's failures, and the energy-gradient story holds only for the frozen model —
the fine-tuned models' imagined cost is as speed-neutral as Chrono's and the drag comes from a minority of slow rough
climbs imagined far too easy (78 kJ imagined against 248 kJ + stall). The gate metric is compressed, not one-sidedly
protected (a matched-rate random rejecter scores 43.9 ± 0.7; the fallback is worth +1.6 layouts); the plain fine-tune
already scores 45; the gate effect on unseen arenas is indistinguishable from zero. The primary's acceptance of
*stall* routes on f105 did not move (31 / 36, same as the frozen model): what the imagination learned is to time out
slow routes. A second all-feasible layout with a correct pose, `s_hill4_crater7_t+60_sh`, is rejected 7–9 / 9 by every
p ≥ 0.6 model and 0 / 9 by the frozen one — an imagination failure worth its own look. The "stuck" training kind is
nearly empty at the horizons used (62 / 22 / 9 / 4 episodes at K 40 / 80 / 120 / 160) yet drew 12–18 % of every batch,
and the "recovery" kind was launches from rest (genuine en-route recoveries: 28 train / 10 val after the fix).

**Jitter probe, corrected.** White σ 0.03 noise is about 1.6 × the imagined tracker's per-step change (0.041 vs
0.026) with the opposite autocorrelation; a tracker-matched AR(1) perturbation (per-step |Δ| 0.025, lag-1 +0.3)
removes about two thirds of the reproduced stalls, not 70–90 %; state-based discrimination survives it at the frozen
model's level; "the frozen model is unaffected" is a floor effect. The harness now carries both probes
(`pre_rec_jit`, `stuck_rec_jit`: white 0.03; `*_jit_ar`, `rest_rec_jit_ar`: AR(1) 0.0265, ρ 0.3).

**Pre-registration for waves 3 / 3b (written 2026-09-07 01:40, before any result).** Wave 3b adds four runs with the
tracker-matched AR(1) input noise (σ 0.0265, ρ 0.3; 0.05 in one), the recovery events fixed, the stuck kind weighted by
its episode count and the approach validation scored as |pred − rec|. Endpoints on f105, true stops (n 46) and launch
failures (n 105), |vx| < 0.5: (a) the augmented model's un-jittered true-stop prediction / in-stall hold / launch hold
≥ 0.30 / 0.45 / 0.45 (frozen 0.24 / 0.22 / 0.20; primary 0.39 / 0.54 / 0.55) AND a fall of < 0.10 under the AR(1)
probe; (b) the §13.7 decision test with the tracker in the loop ≥ AUC 0.80 for "stuck within 8 s" (primary 0.71,
frozen 0.76) with closed-loop feasible completion ≥ 0.90 (primary 0.92). Decision rule: (a) and (b) → the cue was the
fingerprint and closed-loop discrimination is attainable from the state; proceed to the tracker inside the augmented
imagination, then fresh sealed arenas (f108+, model-free difficulty, ≥ 30 heuristic-failure layouts with a feasible
alternative, one look, paired sign test, seed-median checkpoint rule). (a) fails → the stall reproduction was the
fingerprint; stop training this loss on these events and go to the terrain-reading failures (`s_hill4_crater7_t+60_sh`,
`x_hill1_h180`), a stall/progress head, or the replanning design. (a) holds, (b) fails → the gap is the controller
pairing; proceed with the tracker inside the augmented model. Checkpoint rule: lowest jitter-robust stall score at
step ≤ 5 000. One look at f105 per run; the sealed arenas are not touched.

### 13.10 Waves 3 / 3b, the verdict, and the controller in the loop (2026-09-07, automated follow-up)

Automation (`traverse_wp8_followup.sh`, `traverse_wp8_verdict.py`): the twelve augmentation runs were scored as they
finished, the pre-registered endpoints applied at 03:14, the branch launched without intervention.

**Verdict A_FAIL, informative.** Every augmented run's stall metrics survive tracker-like throttle noise (drops of
0.00–0.10 under the AR(1) probe) where the un-augmented control run collapses (stops 0.33 → 0.15, held stalls 0.59 →
0.17, launch failures 0.38 → 0.10), so the augmentation removed the fingerprint; but the robust levels are modest — best
run `wp8d_ar03` (AR(1) noise): true stops predicted 0.43 (AR 0.33), seeded stalls held 0.48 (0.39), launch failures
held 0.33 (0.29), no false stops, against 0.24 / 0.22 / 0.20 frozen — below the pre-registered 0.30 / 0.45 / 0.45 on the
launch level for every run (`wp8_eval/verdict.txt`). Smoothing augmentation lowers the levels further. Reading: once the
model cannot use the throttle, what it reads from the state on unseen terrain is a real but small stall signal.

**The controller in the loop, three ways** (`decision_*.log`, `pick_*.txt`, all on f105):

| imagination driven by | frozen: AUC stuck-within-8 s · pick | stall-trained `wp8_p6_k80_prog` | augmented `wp8d_ar03` |
|---|---|---|---|
| recorded controls (teacher-forced) | 0.76 | 0.89 | 0.85 |
| the WP3 tracker (trained in the frozen imagination) | 0.65 · 39 / 52 | 0.71 · 41 | 0.71 · 41 |
| a tracker retrained inside `wp8d_ar03` (2 seeds, 19 min each) | 0.70 | 0.33 | 0.33 · 26 / 30 |
| pure pursuit (fixed geometric controller) | 0.78 · 41 | 0.32 · 40 | 0.43 · 43 |
| fastest heuristic | 44 / 52 | | |

1. **A tracker retrained inside the stall-aware imagination exploits it.** It learns control patterns under which the
   model predicts motion: with it the model rejects 1 of 130 infeasible routes (17–19 of 451 feasible), the pre-stall
   discrimination inverts (AUC 0.33; 9 % of stuck runs predicted stuck, 50 % of passing ones), and the pick falls to
   26–30 / 52. The audit's warning that "a tracker trained inside an imagination whose stall signal is a control
   pattern learns to dither out of imagined stalls" is confirmed. Model-based RL's model-exploitation problem, in one
   table.
2. **A fixed controller is not the answer either:** pure pursuit holds a *perfectly* constant throttle at steady speed,
   a stronger version of the cue than the recorded stalls' 0.015 per step, and the stall-trained models read it in
   reverse — passing runs predicted stuck (49 %), stuck runs predicted moving (21 %); AUC 0.32–0.43, below chance. The
   augmentation reduced but did not remove the constancy cue (white and AR(1) noise never produce an exactly constant
   input). The frozen model, which never learned the cue, is unaffected (0.78).
3. **The original tracker** gives every model AUC 0.65–0.71 and picks of 39–43 against the heuristic's 44.

**Follow-ups running (cluster):** wave 4 (four runs on the momentum-loss events with the AR(1) augmentation, launched by
the orchestrator) and wave 5 (four runs adding a *hold* augmentation — random windows get a perfectly constant throttle
input — with and without the momentum events), both scored automatically with the jitter probes and the decision test.
The question for both is whether a stall signal read from the state alone can be made large enough to matter, and
whether any controller can be put in the loop without either exploiting the model or triggering its cues.
