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
