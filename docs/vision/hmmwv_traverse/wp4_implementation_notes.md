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

### Tracker margin (plan step 5): not the binding constraint

Reducing `tracker_p95_margin_m` from the interim 0.9 to 0.3 (Chrono p95 is
0.07–0.12 m) **raised** the oracle's no-path rate from 2 % to 18 %. A* then
threads closer to obstacles and the shortcut/Chaikin smoother pushes the path
into the uninflated footprint, so validation rejects it. The smoother, not the
margin, is what the inflation protects; the margin stays at 0.9 until the
smoother is made clearance-aware. Feasibility at 0.9 is already 98 %.

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
| full predicted map | _running_ | | | | | | | | |

**Plans built from the camera alone are safe to drive**: 145 of 145 completed
with no asset contact on 32 held-out layouts, tracked as tightly as the
oracle's plans (mean cross-track 0.034 vs 0.029 m). The imagined-vs-Chrono
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
- **Margin:** 0.9 m stays; shrinking it requires a clearance-aware smoother.

## Open

1. Chrono results for the full-predicted-map rung and for the 40-pass
   candidates (queued on newton).
2. Dynamics model retraining with tracker-driven Chrono episodes (DAgger) to
   remove the throttle-response bias behind the 10 % time and 15–20 % energy gaps.
3. Clearance-aware smoother so the tracker margin can drop to the measured
   0.1 m and layouts with narrow corridors become feasible.
4. Vehicle localisation from the camera (the tracker's pose in Chrono is still
   the simulator's) — WP1 showed 0.8 m / 3–4° from the spatial map.
5. Test split untouched throughout.

