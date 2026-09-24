# D1: why the soil drives fail from a moving start (K1 A5, 3 s approach)

Written 2026-09-22. Offline only, read-only use of the K1 artefacts. All numbers below are in `D1_anatomy.json` (same folder).
"Soil arms" = the five soil-competent arms Spcrm, H, Hmask, P, T; the rigid specialist Sprigid is kept apart except where
stated. Tables give median [25th, 75th percentile] unless marked as a share.

## How it was produced

```
cd /home/harry/NeDM-traverse_mppi
# 1. features without the per-wheel soil data (10 s) + the list of runs to fetch
PYTHONPATH=src:scripts OMP_NUM_THREADS=6 /home/harry/miniconda3/envs/nedm/bin/python \
    artifacts/traverse/crm_improve_20260922/scout/d1_cache/d1_anatomy.py features
# 2. per-wheel soil data of the pass-2 drives, read-only from the cluster (2,495 files, 77 MB, 8 s)
rsync -a --files-from=artifacts/traverse/crm_improve_20260922/scout/d1_cache/fetch_list.txt \
    amd:/work1/dannegrut/harry/experiments/crm_f104_20260916/generalist/a5/pass2/out/runs/ \
    artifacts/traverse/crm_improve_20260922/scout/d1_cache/extra/
# 3. analysis (40 s) -> scout/D1_anatomy.json
PYTHONPATH=src:scripts OMP_NUM_THREADS=6 /home/harry/miniconda3/envs/nedm/bin/python \
    artifacts/traverse/crm_improve_20260922/scout/d1_cache/d1_anatomy.py analyze
```

Data checks. 800 groups x 6 arms = 4,800 arm drives, 4,652 unique runs (arms that picked an identical route share a run).
888 unique runs failed (746 soil breakthrough, 196 prolonged blockage, 3 timeout counted over arm drives; 700 / 185 / 3
over unique runs); 597 of the failed runs belong to at least one soil arm, 291 only to the rigid specialist. For every
run: the branch pose equals the pass-1 frame-60 pose, the recorded vx at the branch equals the pass-1 terminal vx, the
route file's first speed equals `branch_route_v0_mps`, and the prefix frames 0-59 (state, action, pose) plus the
frame-60 state are bit-identical to the pass-1 approach in all 4,652 runs. The fetched per-wheel files (every failed run, every run of the 334 groups with any failure, and 600
randomly drawn runs from the other groups, seed 20260922) all match the sha256 recorded in the local
`episode_complete.json` (2,495 / 2,495).

Conventions used here. State pitch is negative when the nose points up (pitch = -1.06 x local grade, correlation
-0.99), so I report "nose-up pitch" = -pitch. Terrain from the v2 grid (`gb_crop.sample_height`), sampled every 0.5 m;
"steep" = grid cell slope > 15 deg (central difference over 0.625 m; 21 % of all arena cells). "Grade over the next
L m" = atan(height gain / L); "max uphill within L m" = steepest 1 m segment. Sinkage uses the collector's formula
(tyre radius minus spindle height above the undeformed map surface, maximum over the four wheels); on this soil it is
about -0.02 m for a vehicle rolling at 3 m/s (the wheels ride slightly above the map surface), so only changes are
meaningful. Wheel slip = per-frame median over the four wheels of |slip ratio| from the collector; "wheel spin" = that
median above 1. Stall onset = first of 20 consecutive frames with |vx| < 0.3 m/s and throttle > 0.3 after the branch.
Time is measured from the branch (frame 60, 3.0 s).

## Summary

1. **Where the vehicle is at the decision matters most, its condition hardly at all.** The 86 all-fail groups stand
   at the foot of a steep climb at frame 60: the first steep cell along the heading is 2.0 m ahead (all-success groups
   8.0 m), 69 % of the next 10 m straight ahead is steep (0 %), and the steepest 1 m of those 10 m is 23.6 deg (8.3).
   Sinkage at frame 60 does not separate the classes (AUC 0.53), speed only a little (2.75 vs 2.86 m/s, AUC 0.66).
   A logistic model of soil-arm failure on the position alone (steep share and max uphill ahead, nearest steep cell,
   approach grade) reaches AUC 0.855; on the vehicle's condition alone (speed, sinkage, approach slip, pitch) 0.60;
   both together 0.859.
2. **Failure sequence.** The route asks for more speed (median step +1.71 m/s in failures vs +0.68 in the successes of
   the same groups), the follower saturates the throttle within 50 ms (median throttle over frames 61-63: 1.00 vs 0.41),
   the wheels spin (median wheel slip above 1 at 1.2 s; 71 % of failures spin within 2 s vs 33 % of same-group
   successes), the vehicle stalls on a steep cell (stall onset 4.45 s [3.0, 7.3] and 9.0 m after the branch, 87 % on a
   cell steeper than 15 deg, median cell slope 22.8 deg), then digs in (sinkage +0.10 m at 5.7 s, +0.34 m at the end)
   and is terminated (soil breakthrough at 11.9 s, blockage at 31.0 s). Sinkage does not grow in the first 2 s
   (median -0.001 m in failures, -0.004 m in successes). 99.7 % of failures show the stall; 0.3 % of successes do.
3. **Handover-driven vs route-driven.** Only 85 of 888 failed runs stall within 2 s of the branch; they sit in 28
   groups (81 of the 85 in all-fail groups), 2.0 m past the branch, from a vehicle already slowed on the approach
   (vx 2.12 m/s) that the route asks to turn hard (|heading step| 37 deg) and speed up (+2.31 m/s). 544 more stall
   later but had wheel spin within the first 2 s (step +1.88 m/s, stall at 7.7 m); 256 stall later without early spin
   (step +0.89 m/s, stall at 17.0 m, i.e. a steep section further along the route). 3 are timeouts without a stall.
4. **The speed step is mostly a marker of hard situations.** Between groups the association is strong (fail 7.1 % at
   matched speed, 37.1 % at a step above +1.5 m/s; odds ratio 2.58 per m/s of positive step for the soil arms with
   vehicle speed, heading step and route grade in the model) and survives stratification by route grade and vehicle
   speed. But the planners ask for large steps where the terrain is hard (correlation of the step with the steepest
   metre of the next 10 m of the route +0.43), and within a group (identical frame-60 state, conditional logistic
   regression) the positive step is not significant: odds ratio 1.17 [0.91, 1.55] per m/s alone and 0.90 [0.65, 1.27]
   once the route's grade is in the model, where the route's grade over the first 10 m carries the effect (1.67 per
   deg [1.42, 2.21]). A route starting slower than the vehicle is harmful within a group (1.68 per m/s [1.02, 3.07]).
   Direct pairs in the same group: routes with a large step fail 5.7 points more often than routes with a matched
   start speed (19.5 vs 13.8 %, CI [+1.2, +10.1], 277 groups). In the all-fail groups, all 20 soil-arm routes that
   started within 0.5 m/s of the vehicle speed failed too.
5. **The 3 s approach hurts by committing the vehicle, not by damaging it.** The moving-start failure rate minus the
   standing-start failure rate (same groups, H arm) is 0.0 points when the first 12 m of the straight line are below
   5 deg, +3.4 at 5-10 deg, +12.4 at 12-17, +21.2 at 17-25 and +37.5 above 25 deg. From a standing start the soil
   planner turns off the straight line at once: in the 68 all-fail groups it completed from a standing start, it had
   moved 2.7 m sideways and was heading 35 deg away from the line after 6 m of travel, and its first 12 m crossed no
   steep cell (median), whereas the straight line crosses 36 % steep cells with a steepest metre of 16 deg. The
   approach drives those 6 m straight at the slope; 65 of the 82 groups that no arm completes were completed from a
   standing start.
6. **Recoverability.** 208 of the 597 failed soil-arm runs (35 %, all in the 105 choice-dependent groups) were
   completed by another soil arm from the identical state; 10 more (4 all-fail groups) only by the rigid specialist.
   The routes that succeeded did not have a smaller speed step (success minus failure +0.07 m/s [-0.14, +0.28]); they
   turned further off the approach line (|heading step| +3.9 deg [+1.2, +6.6], lateral offset at 10 m +0.55 m
   [+0.25, +0.87]) and met a lower grade over the first 10 m (-1.5 deg [-2.1, -0.9]). Picking, in each choice group,
   the route with the smallest positive speed step would have succeeded 49.5 % of the time against 57.7 % for a
   random soil-arm pick; picking the lowest 10 m grade would have succeeded 63.8 %.

What this says about the plan's hypotheses (diagnosis only; the closed-loop tests decide): the handover hypothesis H1
has a real but modest within-group signal (large steps: +5.7 points; slower-than-vehicle starts also hurt), so the
`cont` speed profiles of S1 should be expected to recover only part of the gap to the standing start (failure +9.9
points for H, +12.8 for the CRM specialist, same 800 groups); the
approach hypothesis H2 is strongly supported in its positional form, which is what the short-approach stage S2 tests.

## Q1. Group classes and the frame-60 situation

| quantity | all fail (86) | choice-dependent (105) | all success (609) |
|---|---|---|---|
| fresh / reused groups | 67 / 19 | 78 / 27 | 455 / 154 |
| soil-arm failure rate | 100 % | 42.1 % | 0 % |
| rigid specialist failure rate | 95.3 % | 65.7 % | 23.5 % |
| vx at frame 60 (m/s) | 2.75 [2.51, 2.89] | 2.83 [2.75, 2.94] | 2.86 [2.75, 2.96] |
| share with vx < 2.5 m/s | 24.4 % | 8.6 % | 7.4 % |
| nose-up pitch (deg) | 3.7 [-1.8, 8.4] | 2.5 [-0.7, 4.8] | 1.4 [-1.3, 4.9] |
| roll (deg) | -1.2 [-5.0, 2.6] | -0.1 [-2.2, 3.6] | -0.3 [-3.5, 2.3] |
| sinkage at the last approach frame (m) | -0.022 [-0.035, -0.011] | -0.022 [-0.038, -0.011] | -0.025 [-0.037, -0.014] |
| sinkage change over the approach (m) | -0.038 [-0.061, -0.025] | -0.047 [-0.063, -0.026] | -0.043 [-0.061, -0.025] |
| wheel slip, last 1 s of approach | 0.46 [0.33, 0.58] | 0.35 [0.28, 0.45] | 0.35 [0.28, 0.44] |
| throttle, last 1 s of approach | 0.28 [0.23, 0.37] | 0.25 [0.21, 0.29] | 0.24 [0.20, 0.29] |
| local grade under the vehicle, +-2 m (deg) | 3.6 [-2.1, 8.5] | 2.4 [-0.7, 4.7] | 1.5 [-1.2, 4.6] |
| **straight ahead along the frame-60 heading** | | | |
| height gain over 5 / 10 / 20 m (m) | 0.73 / 2.08 / 1.98 | 0.31 / 1.05 / 1.65 | 0.19 / 0.24 / 0.51 |
| grade over 5 m (deg) | 8.3 [-1.4, 15.9] | 3.6 [0.3, 7.5] | 2.1 [-0.7, 5.5] |
| grade over 10 m (deg) | 11.7 [6.3, 14.5] | 6.0 [-0.4, 12.4] | 1.4 [-1.0, 6.8] |
| grade over 20 m (deg) | 5.6 [3.1, 8.0] | 4.7 [1.4, 7.5] | 1.4 [-0.2, 4.9] |
| steepest metre within 10 m (deg) | 23.6 [18.8, 26.1] | 17.7 [7.7, 23.7] | 8.3 [4.8, 15.4] |
| share of the next 10 m on steep cells | 0.69 [0.52, 0.81] | 0.43 [0.19, 0.62] | 0.00 [0.00, 0.38] |
| first steep cell along the heading (m) | 2.0 [1.0, 3.5] | 5.5 [3.5, 8.4] | 8.0 [4.0, 13.5] (19 % none in 20 m) |
| share with a steep cell within 3 m ahead | 67.4 % | 22.9 % | 16.9 % |
| nearest steep cell in any direction (m) | 1.6 [0.5, 2.9] | 3.6 [2.2, 5.7] | 5.5 [2.8, 8.6] |
| approach line: max grade in the first 12 m (deg) | 21.3 [16.8, 24.8] | 11.7 [7.9, 19.0] | 8.2 [6.0, 13.6] |
| approach line: mean signed grade, 12 m (deg) | 3.6 [-0.4, 10.4] | 1.6 [-0.2, 4.4] | 0.8 [-0.3, 2.8] |
| **along the picked soil-arm routes** (unique routes) | 389 | 494 | 2,977 |
| height gain over 5 / 10 / 20 m (m) | 0.44 / 1.13 / 1.56 | 0.28 / 0.48 / 0.61 | 0.15 / 0.18 / 0.36 |
| grade over 10 m (deg) | 6.5 [1.6, 12.4] | 2.7 [-1.4, 7.4] | 1.0 [-1.0, 3.5] |
| grade over 20 m (deg) | 4.5 [0.9, 6.8] | 1.7 [0.1, 4.2] | 1.0 [-0.1, 2.8] |
| steepest metre within 10 m (deg) | 14.9 [9.3, 25.0] | 10.4 [7.8, 16.3] | 7.1 [4.6, 10.4] |
| steepest metre of the whole route (deg) | 20.8 [15.8, 25.3] | 19.9 [14.9, 24.6] | 14.5 [10.5, 19.9] |
| share of the whole route on steep cells | 0.49 [0.35, 0.66] | 0.41 [0.27, 0.54] | 0.18 [0.09, 0.28] |
| route start speed (m/s) | 4.79 [4.01, 5.63] | 3.89 [2.91, 4.93] | 3.10 [2.47, 3.84] |
| speed step, route start minus vx (m/s) | +2.30 [1.27, 3.02] | +1.13 [0.04, 2.23] | +0.26 [-0.40, 1.07] |
| abs. heading step (deg) | 29.5 [12.2, 39.8] | 25.3 [13.0, 37.0] | 19.9 [9.5, 31.1] |
| **same group from a standing start (A3)** | | | |
| CRM specialist S_crm completed | 79.1 % (68) | 87.6 % (92) | 99.5 % (606) |
| history model H completed | 74.4 % | 81.9 % | 98.5 % |
| oracle tag T completed | 76.7 % | 93.3 % | 99.3 % |

Separation of all-fail groups from the rest (AUC of each single frame-60 quantity, 0.5 = none; values below 0.5
flipped): steep share of the next 10 m 0.88, steepest metre within 10 m 0.86, approach max grade 0.85, first steep
cell distance 0.82, nearest steep cell 0.79, height gain over 10 m 0.75, vx 0.66, approach slip 0.66, approach
throttle 0.64, nose-up pitch 0.56, local grade 0.56, sinkage 0.53. Evaluation strata of the all-fail groups:
hill cross-slope 38, hill entry / cross exit 23, crater cross-slope 8, roughness transfer 7, crater entry 5, long
traverse 5.

## Q2. After the branch: failures vs successes

"Same-group successes" = successful runs in the groups that also have a failure (identical frame-60 state, any arm);
"other successes" = successful runs in groups with no failure (all runs for the trajectory rows, the 600 sampled runs
for the wheel rows).

| quantity | failures (888) | same-group successes (1,007) | other successes (2,757 / 600) |
|---|---|---|---|
| vx at frame 60 (m/s) | 2.80 [2.60, 2.91] | 2.84 [2.73, 2.94] | 2.87 [2.77, 2.96] |
| speed step (m/s) | +1.71 [0.77, 2.66] | +0.68 [-0.16, 1.67] | +0.39 [-0.27, 1.26] |
| abs. heading step (deg) | 22.3 [8.5, 35.6] | 22.2 [9.7, 34.0] | 19.4 [9.1, 30.6] |
| throttle, last approach frame | 0.26 [0.19, 0.38] | 0.24 [0.18, 0.30] | 0.22 [0.16, 0.28] |
| throttle, frames 61-63 | 1.00 [0.48, 1.00] | 0.41 [0.00, 1.00] | 0.24 [0.00, 0.75] |
| mean throttle, first 1 s / first 2 s | 0.91 / 0.91 | 0.39 / 0.42 | 0.27 / 0.29 |
| share with throttle >= 0.95 at some frame in the first 2 s | 71.2 % | 37.9 % | 23.8 % |
| share with any brake in the first 2 s | 13.2 % | 30.6 % | 35.3 % |
| vx after 1 s / 2 s (m/s) | 2.54 / 2.13 | 2.57 / 2.72 | 2.58 / 2.70 |
| lowest vx in the first 2 s (m/s) | 1.97 [1.10, 2.66] | 2.28 [1.76, 2.76] | 2.37 [1.87, 2.77] |
| wheel slip in the last 1 s before the branch | 0.38 [0.30, 0.48] | 0.33 [0.27, 0.43] | 0.33 [0.26, 0.41] |
| max wheel slip (median wheel), first 2 s | 1.45 [0.87, 2.89] | 0.79 [0.60, 1.14] | 0.68 [0.53, 0.93] |
| max slip of any wheel, first 2 s | 4.75 [1.80, 15.7] | 1.69 [1.17, 2.96] | 1.35 [1.02, 2.15] |
| time to wheel spin (s) | 1.2 [0.5, 2.25] | 2.9 [1.2, 5.35] | 4.05 [1.4, 7.1] |
| share spinning within 2 s | 71.1 % | 32.9 % | 23.7 % |
| sinkage growth after 1 s / 2 s (m) | 0.000 / -0.001 | -0.000 / -0.004 | -0.003 / -0.002 |
| max sinkage growth over the drive (m) | 0.335 [0.293, 0.369] | 0.058 [0.040, 0.083] | 0.050 [0.035, 0.067] |
| time to +0.10 m sinkage (s; share reaching it) | 5.7 [3.85, 8.1] (99.7 %) | 5.05 (16.4 %) | 5.98 (7.7 %) |
| stall onset (s; share stalling) | 4.45 [3.0, 7.3] (99.7 %) | 12.75 (0.3 %) | none |
| distance driven to the stall (m) | 9.0 [5.5, 16.9] | | |
| cell slope at the stall (deg); share on steep cells | 22.8 [17.5, 25.1]; 87.5 % | | |
| nose-up pitch at the stall (deg) | 16.9 [11.2, 21.1] | | |
| end of the recording after the branch (s) | 13.65 [9.65, 23.45] | 13.55 (goal) | 14.55 (goal) |

The recorded throttle of frame 60 itself is 0 in all 4,652 runs: the rebuilt follower starts from zero and the
recorded value is its first-substep input (the collector notes say the controllers are reset at the swap), so the
jump is read from frames 61-63. With the follower's speed gains (0.6, 0.05) a speed error of about 1.7 m/s saturates
the throttle by the proportional term alone. Failure rate by the share of the first 2 s spent at >= 0.95 throttle
(all unique runs): none 8.6 % (n 2,981), up to a quarter 24.4 % (270), 25-50 % 29.1 % (230), 50-75 % 30.1 % (269),
more than 75 % 46.3 % (902). Wheel spin within 2 s by speed step: among successes 16.5 % (step < -0.5, n 279),
11.8 % (-0.5..0.5, 493), 39.2 % (0.5..1.5, 431), 49.5 % (> 1.5, 404); among failures 50.0 / 40.0 / 66.0 / 82.2 %.

Failures split by status: soil breakthrough (700 runs) stall at 4.45 s and are terminated at 11.9 s [9.2, 15.7]
with +0.35 m sinkage; prolonged blockage (185) stall at 4.5 s and are terminated at 31.0 s with +0.22 m sinkage and
a gentler stall location (local grade 9.7 deg vs 17.6).

Onset classes (every failed unique run; the class columns count runs by group class, the "all success" column holds
rigid-specialist failures in groups the soil arms all completed):

| onset class | runs | groups | all fail / choice / all success | speed step (m/s) | throttle frames 61-63 | stall onset (s) | distance to stall (m) | cell slope at stall (deg); on steep | route grade over 10 m (deg) |
|---|---|---|---|---|---|---|---|---|---|
| stall within 2 s of the branch | 85 | 28 | 81 / 3 / 1 | +2.31 [1.60, 3.15] | 1.00 [1.00, 1.00] | 1.7 [1.35, 1.8] | 2.0 [1.3, 2.7] | 17.2; 84.7 % | 4.4 [-0.6, 11.8] |
| later stall, wheel spin within 2 s | 544 | 200 | 333 / 155 / 56 | +1.88 [1.05, 2.73] | 1.00 [0.63, 1.00] | 4.1 [3.0, 6.3] | 7.7 [5.4, 13.2] | 23.1; 89.2 % | 5.6 [1.3, 12.2] |
| later stall, no wheel spin in the first 2 s | 256 | 177 | 52 / 118 / 86 | +0.89 [0.12, 1.90] | 0.53 [0.08, 1.00] | 7.0 [4.9, 9.4] | 17.0 [10.8, 26.2] | 22.5; 84.8 % | 2.2 [-0.2, 6.3] |
| no stall (timeout) | 3 | 2 | 2 / 1 / 0 | +3.21 | 1.00 | - | - | - | 5.2 |

The 85 early stalls start from vx 2.12 m/s [1.77, 2.71] (59 % below 2.5 m/s), ask for a 37 deg [21, 44] heading
change, never lift the throttle (1.00 for the whole 2 s) and advance 2.0 m. Stall-onset histogram over all failures:
0-1 s 7, 1-2 s 70, 2-3 s 133 (incl. 8 at exactly 2.0 s), 3-5 s 279, 5-10 s 290, 10-20 s 102, > 20 s 4.

## Q3. The speed step, controlling for the terrain ahead

Unique runs of all six arms (4,652, 888 failures) unless marked "soil" (3,860 runs with at least one soil arm, 597
failures). "grade10" = the route's grade over its first 10 m; "maxup10" = steepest metre of the route's first 10 m.

Failure rate by speed step (all six arms): < -1.5 m/s 16.7 % (n 132); -1.5..-0.5 6.3 % (632); -0.5..0.5 7.1 %
(1,346); 0.5..1.5 19.7 % (1,223); > 1.5 37.1 % (1,319).

Stratified by the route's grade over the first 10 m (failure rate, n):

| route grade10 | step < -1.5 | -1.5..-0.5 | -0.5..0.5 | 0.5..1.5 | > 1.5 |
|---|---|---|---|---|---|
| < 0 deg | 16.4 % (55) | 3.2 % (248) | 5.4 % (498) | 16.2 % (376) | 18.8 % (346) |
| 0-5 deg | 14.9 % (74) | 7.5 % (348) | 7.7 % (660) | 19.8 % (524) | 33.0 % (394) |
| 5-10 deg | 50 % (2) | 9.4 % (32) | 5.8 % (172) | 15.5 % (252) | 28.8 % (288) |
| > 10 deg | 100 % (1) | 75 % (4) | 43.8 % (16) | 52.1 % (71) | 72.9 % (291) |

Stratified by the vehicle speed at frame 60:

| vx at frame 60 | step < -1.5 | -1.5..-0.5 | -0.5..0.5 | 0.5..1.5 | > 1.5 |
|---|---|---|---|---|---|
| < 2.5 m/s | - (0) | 33 % (3) | 4.3 % (23) | 14.6 % (89) | 44.1 % (329) |
| 2.5-2.8 | 9.1 % (11) | 11.0 % (100) | 7.6 % (291) | 19.1 % (408) | 39.9 % (446) |
| 2.8-3.0 | 18.1 % (83) | 4.3 % (416) | 6.7 % (864) | 22.3 % (584) | 30.0 % (416) |
| > 3.0 | 15.8 % (38) | 8.8 % (113) | 8.3 % (168) | 14.1 % (142) | 32.8 % (128) |

Stratified by the route's mean speed: the step effect is the same in every band (matched start 4.8 / 4.8 / 11.4 /
8.3 % vs step > 1.5: 34.5 / 35.8 / 39.0 / 37.0 % for mean speed < 2.5 / 2.5-3.2 / 3.2-4.0 / > 4.0 m/s).

Logistic regressions across runs (odds ratio per unit, 95 % interval from 400 group-bootstrap resamples; step+ =
max(step, 0), step- = max(-step, 0); head = abs. heading step in deg; vmean = route mean speed; maxup_all = steepest
metre of the whole route; hardness terms = steep share of the next 10 m along the frame-60 heading, approach max
grade, standing-start S_crm failure of the group):

| model | runs | AUC | step+ (per m/s) | step- (per m/s) | vx | head (per deg) | grade10 (per deg) | maxup10 | vmean | maxup_all | steep share ahead | standing-start fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| step only (linear step) | all 6 | 0.728 | 1.92 [1.75, 2.13] | | | | | | | | | |
| + vx, head, grade10 | all 6 | 0.761 | 2.01 [1.81, 2.26] | 1.32 [0.87, 1.75] | 1.15 [0.66, 2.12] | 1.02 [1.01, 1.03] | 1.12 [1.08, 1.15] | | | | | |
| + maxup10, vmean | all 6 | 0.793 | 1.47 [1.29, 1.70] | 1.38 [0.93, 1.86] | 0.75 [0.45, 1.38] | 1.03 [1.02, 1.04] | 1.05 [1.01, 1.09] | 1.11 [1.08, 1.14] | 1.64 [1.38, 1.93] | | | |
| + sinkage, pitch, maxup_all | all 6 | 0.806 | 1.49 [1.31, 1.70] | 1.22 [0.85, 1.62] | 0.69 [0.34, 1.51] | 1.04 [1.02, 1.05] | 1.09 [1.05, 1.12] | 1.05 [1.03, 1.09] | 1.36 [1.13, 1.62] | 1.09 [1.06, 1.13] | | |
| + group hardness | all 6 | 0.847 | 1.27 [1.11, 1.48] | 1.33 [0.93, 1.72] | 1.84 [0.84, 4.92] | 1.01 [1.00, 1.02] | 1.14 [1.09, 1.19] | 0.96 [0.92, 1.01] | 1.43 [1.18, 1.79] | 1.09 [1.06, 1.13] | 30 [14, 99] | 3.95 [2.04, 9.00] |
| step only (linear step) | soil | 0.765 | 2.15 [1.91, 2.44] | | | | | | | | | |
| + vx, head, grade10 | soil | 0.817 | 2.58 [2.26, 2.93] | 2.00 [1.32, 2.67] | 1.50 [0.78, 3.35] | 1.03 [1.02, 1.04] | 1.13 [1.09, 1.18] | | | | | |
| + maxup10, vmean | soil | 0.851 | 1.79 [1.51, 2.15] | 2.18 [1.44, 2.97] | 0.87 [0.45, 1.95] | 1.05 [1.04, 1.07] | 1.05 [1.01, 1.09] | 1.14 [1.09, 1.18] | 1.72 [1.39, 2.14] | | | |
| + sinkage, pitch, maxup_all | soil | 0.865 | 1.82 [1.54, 2.20] | 1.87 [1.26, 2.62] | 0.79 [0.33, 2.25] | 1.06 [1.04, 1.08] | 1.09 [1.04, 1.15] | 1.07 [1.03, 1.12] | 1.37 [1.08, 1.71] | 1.11 [1.06, 1.17] | | |
| + group hardness | soil | 0.905 | 1.50 [1.22, 1.83] | 2.23 [1.58, 3.17] | 2.58 [0.83, 9.29] | 1.03 [1.01, 1.05] | 1.15 [1.09, 1.22] | 0.96 [0.90, 1.02] | 1.44 [1.11, 1.90] | 1.12 [1.07, 1.19] | 68 [23, 347] | 4.09 [1.78, 10.4] |

(The linear-step intervals of the first rows are exp of the bootstrap percentiles of 0.652 [0.558, 0.757] and 0.765
[0.646, 0.891]. The sinkage and pitch terms are not significant in any model: sinkage per m 5.7 [-2.8, 14.1] on the
log-odds scale, pitch 0.016 [-0.012, 0.046] per deg, all six arms.)

Within a group (conditional logistic regression on the exact within-group likelihood: the frame-60 state is
identical, only groups with both outcomes contribute; 200 group-bootstrap resamples):

| model | runs | groups used | step+ (per m/s) | step- (per m/s) | head (per deg) | grade10 (per deg) | maxup10 | vmean | maxup_all |
|---|---|---|---|---|---|---|---|---|---|
| step+ / step- | all 6 | 252 | 1.29 [1.06, 1.54] | 0.67 [0.44, 1.00] | | | | | |
| + head, grade10, maxup10 | all 6 | 252 | 1.10 [0.90, 1.34] | 0.65 [0.42, 1.01] | 0.98 [0.96, 1.00] | 1.46 [1.32, 1.65] | 0.94 [0.87, 1.01] | | |
| + vmean, maxup_all | all 6 | 252 | 1.20 [0.96, 1.46] | 0.64 [0.42, 0.98] | 0.98 [0.96, 1.00] | 1.48 [1.33, 1.70] | 0.92 [0.81, 1.00] | 0.80 [0.57, 1.13] | 1.03 [0.98, 1.09] |
| step+ / step- | soil | 105 | 1.17 [0.91, 1.55] | 1.75 [1.06, 3.28] | | | | | |
| + head, grade10, maxup10 | soil | 105 | 0.90 [0.65, 1.27] | 1.68 [1.02, 3.07] | 0.98 [0.95, 1.02] | 1.67 [1.42, 2.21] | 0.86 [0.75, 0.95] | | |
| + vmean, maxup_all | soil | 105 | 0.96 [0.67, 1.43] | 1.60 [1.00, 3.02] | 0.98 [0.94, 1.02] | 1.74 [1.44, 2.39] | 0.80 [0.66, 0.92] | 0.85 [0.47, 1.54] | 1.07 [0.96, 1.18] |

The within-group test has spread to work with: the step varies within a group with a standard deviation of 0.83 m/s
(between group means 1.01, total 1.29); the within-group range is 2.0 m/s [1.4, 2.7]. The negative-step term changes
sign between the two row sets (all six arms: slower-than-vehicle starts look protective; soil arms only: harmful). The
rigid specialist's routes are the fastest starts and fail 36.8 %, so in "all 6" a slow start is mostly compared with
them; I read the soil-only estimate as the relevant one, but it rests on 105 groups and its lower bound is 1.02. Per arm: step median +0.46 (Spcrm), +1.32 (Sprigid), +0.25 (H), +0.37 (Hmask), +0.64 (P),
+0.68 (T) m/s; share of routes with a step above 1.5 m/s 27 / 42 / 25 / 27 / 31 / 29 %.

Direct within-group pairs (a route with |step| < 0.5 m/s and a route with step > 1.5 m/s in the same group, all six
arms): 277 groups; failure 13.8 % for the matched-start routes vs 19.5 % for the large-step routes, difference +5.7
points [+1.2, +10.1]; over all such route pairs 40 pairs fail only on the matched start, 92 only on the large step,
66 both, 600 neither. Restricted to pairs whose routes have similar terrain (grade10 within 2 deg and maxup10 within
3 deg): 32 vs 51 discordant pairs (27 both, 454 neither).

Correlations of the step with other quantities (all runs): vehicle speed -0.40, route start speed +0.98, route mean
speed +0.36, route grade10 +0.30, route maxup10 +0.43, grade over 10 m along the heading +0.36, heading step -0.01,
sinkage +0.02.

## Q4. Is the 3 s approach itself harmful?

Moving start (A5) vs standing start (A3), same 800 groups. Soil-arm failure is the mean over the five soil arms; the
paired differences use H (both protocols) and the CRM specialist (A5 same-row S'_crm vs A3 deployed S_crm), 95 %
group-bootstrap intervals.

| approach: max grade in the first 12 m | groups | vx at frame 60 | A5 soil-arm fail | A5 all-fail share | A3 S_crm fail | A3 H fail | H moving minus standing (points) | S_crm moving minus standing |
|---|---|---|---|---|---|---|---|---|
| 0-5 deg | 96 | 2.95 | 1.0 % | 0.0 % | 0.0 % | 1.0 % | +0.0 [-3.1, +3.1] | +2.1 [0.0, +5.2] |
| 5-10 | 323 | 2.88 | 6.7 % | 2.5 % | 1.5 % | 3.4 % | +3.4 [+0.3, +6.5] | +5.6 [+3.1, +8.4] |
| 10-12 | 80 | 2.79 | 11.0 % | 6.2 % | 5.0 % | 3.7 % | +8.7 [+1.3, +16.3] | +7.5 [0.0, +15.0] |
| 12-17 | 105 | 2.78 | 20.4 % | 11.4 % | 3.8 % | 8.6 % | +12.4 [+4.8, +20.0] | +15.2 [+8.6, +22.9] |
| 17-25 | 156 | 2.71 | 34.4 % | 26.3 % | 6.4 % | 12.2 % | +21.2 [+12.8, +28.8] | +28.8 [+21.8, +36.5] |
| > 25 | 40 | 2.82 | 59.5 % | 50.0 % | 27.5 % | 17.5 % | +37.5 [+22.5, +52.5] | +37.5 [+20.0, +55.0] |
| all | 800 | 2.85 | 16.3 % | 10.8 % | 4.3 % | 6.3 % | +9.9 [+7.3, +12.4] | +12.8 [+10.3, +15.1] |

| approach: mean signed grade, 12 m | groups | vx at frame 60 | A5 soil-arm fail | A3 S_crm fail | H moving minus standing |
|---|---|---|---|---|---|
| < -5 deg (downhill) | 28 | 3.09 | 35.7 % | 3.6 % | +39.3 [+21.4, +57.1] |
| -5..0 | 211 | 2.94 | 11.3 % | 1.4 % | +7.6 [+3.3, +12.3] |
| 0..5 | 397 | 2.86 | 11.9 % | 3.0 % | +6.8 [+3.5, +10.3] |
| 5..10 | 115 | 2.65 | 21.2 % | 7.8 % | +8.7 [+1.7, +16.5] |
| > 10 (uphill) | 49 | 2.49 | 50.2 % | 18.4 % | +30.6 [+18.4, +44.9] |

| vx at frame 60 | groups | A5 soil-arm fail | A5 all-fail share | A3 S_crm fail | H moving minus standing |
|---|---|---|---|---|---|
| < 1.5 m/s | 6 | 36.7 % | 33.3 % | 16.7 % | +16.7 [-33, +67] |
| 1.5-2.0 | 17 | 48.2 % | 47.1 % | 11.8 % | +47.1 [+23.5, +70.6] |
| 2.0-2.5 | 52 | 26.5 % | 21.2 % | 5.8 % | +21.2 [+9.6, +34.6] |
| 2.5-2.8 | 217 | 20.4 % | 13.8 % | 8.3 % | +6.9 [+2.3, +12.0] |
| 2.8-3.0 | 405 | 11.9 % | 6.4 % | 2.0 % | +8.1 [+4.9, +11.6] |
| > 3.0 | 103 | 13.4 % | 8.7 % | 1.9 % | +10.7 [+4.8, +17.5] |

Vehicle speed at frame 60 correlates -0.57 with the approach's mean signed grade and -0.27 with its max grade.

Condition vs position (group-level logistic, five soil-arm outcomes per group, 400 group-bootstrap resamples;
coefficient per standard deviation on the log-odds scale):

| model | AUC | vx | sinkage | approach slip | pitch (positive = nose down) | steep share ahead | steepest metre ahead | nearest steep cell | approach max grade |
|---|---|---|---|---|---|---|---|---|---|
| condition only | 0.601 | -0.78 [-1.10, -0.46] | -0.03 [-0.21, +0.17] | -0.31 [-0.56, -0.06] | +0.30 [+0.01, +0.54] | | | | |
| position only | 0.855 | | | | | +0.91 [+0.57, +1.29] | +0.83 [+0.51, +1.16] | +0.03 [-0.34, +0.37] | -0.06 [-0.44, +0.28] |
| both | 0.859 | -0.41 [-0.78, -0.10] | +0.14 [-0.06, +0.36] | -0.42 [-0.93, -0.04] | +0.16 [-0.05, +0.40] | +0.85 [+0.45, +1.29] | +0.84 [+0.52, +1.19] | +0.01 [-0.37, +0.36] | -0.00 [-0.37, +0.40] |

(Per-sd intervals are the per-unit bootstrap percentiles times the sd; the per-unit values are in the JSON. The AUC is
over the 4,000 soil-arm drives. The positive pitch term means nose-down arrivals, i.e. the downhill approaches below,
fail more at a given speed.)

Where the standing-start drives went in the first 12 m of travel (A3 S_crm; lateral offset and heading relative to
the straight start-goal line that the approach drives):

| groups | n | sideways after 6 m (m) | heading off the line after 6 m (deg) | share > 1 m sideways at 6 m | steepest metre in its first 12 m (deg) | steep share in its first 12 m | the straight line: steepest metre / steep share, 12 m |
|---|---|---|---|---|---|---|---|
| all fail, standing start completed | 68 | 2.66 [2.25, 2.93] | 34.6 [23.9, 42.3] | 91.2 % | 7.1 [5.1, 10.6] | 0.00 [0.00, 0.08] | 16.4 deg / 0.36 |
| all fail, standing start failed | 18 | 2.38 [1.69, 2.67] | 24.9 [12.1, 36.2] | 77.8 % | 9.8 [7.3, 13.5] | 0.20 [0.00, 0.32] | 23.8 deg / 0.30 |
| choice, standing start completed | 92 | 2.08 [1.19, 2.74] | 20.1 [14.4, 35.5] | 79.3 % | 7.3 [5.3, 10.4] | 0.00 [0.00, 0.08] | 7.6 deg / 0.00 |
| choice, standing start failed | 13 | 1.39 [0.42, 2.24] | 18.9 [4.0, 34.7] | 53.8 % | 15.7 [10.3, 18.9] | 0.20 [0.16, 0.29] | 18.6 deg / 0.16 |
| all success, standing start completed | 606 | 1.76 [0.83, 2.47] | 18.7 [8.9, 29.6] | 71.0 % | 5.9 [4.2, 8.4] | 0.00 [0.00, 0.00] | 6.7 deg / 0.00 |

In the all-fail groups the standing-start drives need 2.9 s [2.7, 3.05] to cover 6 m, the same time as the approach. Reading: the approach does
not damage the vehicle measurably (sinkage falls over the approach in every class, approach slip is only 0.1 higher in
all-fail groups and carries no weight once position is known); it hurts by spending the first 6 m on the straight line
into the slope, which the standing-start planner avoids by turning away at once, and by arriving slowed where the line
climbs (vx < 2.5 m/s in 24 % of all-fail groups vs 7 % of all-success groups). Downhill approaches are nearly as bad
as uphill ones (+39 points, 28 groups): the vehicle arrives fast at the bottom of a crater and has to climb out.

## Q5. Which failures look recoverable, and what the successful routes did differently

| count | value |
|---|---|
| failed unique runs, any arm / with a soil arm / rigid specialist only | 888 / 597 / 291 |
| groups with a soil-arm failure | 191 |
| soil-arm failures completed by another soil arm from the same state | 208 runs (34.8 %), 105 groups (all choice-dependent) |
| ... by any arm incl. the rigid specialist | 218 runs (36.5 %), 109 groups (4 of them all-fail groups) |
| soil-arm failures no arm completed | 379 runs in 82 groups; 65 of these groups completed from a standing start by S_crm |
| onset class of the 208 recoverable failures | later stall 204, stall within 2 s 3, timeout 1 |
| rescuing runs by arm (soil rescuers; a run can count for several arms) | Spcrm 55, H 62, Hmask 62, P 59, T 66, Sprigid 1 |

Successful minus failed route in the same group (soil rescuers; per-group mean over all fail x success pairs, then the
mean over 105 groups with a 95 % bootstrap interval; the last column is the share of groups where the successful route
is higher / lower):

| route quantity | failed route | successful route | success minus failure | higher / lower |
|---|---|---|---|---|
| start speed (m/s) | 4.04 [2.89, 5.14] | 3.85 [2.95, 4.68] | +0.07 [-0.14, +0.28] | 49 / 50 % |
| speed step (m/s) | +1.26 [0.05, 2.31] | +1.06 [0.05, 1.93] | +0.07 [-0.14, +0.28] | 49 / 50 % |
| lowest speed in the first 10 m (m/s) | 3.57 | 3.54 | +0.12 [-0.11, +0.35] | 56 / 44 % |
| mean speed (m/s) | 3.38 | 3.39 | -0.00 [-0.16, +0.16] | 43 / 57 % |
| top speed (m/s) | 5.08 | 4.93 | -0.07 [-0.20, +0.07] | 45 / 54 % |
| abs. heading step (deg) | 23.2 [8.8, 34.1] | 26.6 [15.9, 38.1] | +3.9 [+1.2, +6.6] | 66 / 34 % |
| abs. lateral offset at 10 m (m) | 3.59 | 3.81 | +0.55 [+0.25, +0.87] | 63 / 37 % |
| max lateral offset from the straight line (m) | 5.02 | 5.41 | +0.46 [+0.08, +0.83] | 60 / 40 % |
| length (m) | 35.7 | 36.4 | +0.31 [+0.08, +0.54] | 58 / 42 % |
| grade over the first 5 m (deg) | 3.3 | 3.0 | -1.1 [-1.8, -0.5] | 38 / 62 % |
| grade over the first 10 m (deg) | 2.8 [-1.4, 8.7] | 2.4 [-1.4, 6.2] | -1.5 [-2.1, -0.9] | 30 / 70 % |
| grade over the first 20 m (deg) | 2.2 | 1.4 | -0.4 [-0.8, -0.1] | 43 / 57 % |
| steepest metre in the first 10 m (deg) | 10.5 | 10.1 | -0.9 [-1.7, -0.1] | 50 / 50 % |
| steepest metre of the whole route (deg) | 21.2 | 19.3 | -0.5 [-1.4, +0.3] | 50 / 50 % |
| steep share of the whole route | 0.45 | 0.37 | -0.01 [-0.03, +0.01] | 38 / 59 % |
| planned time to go (s) | 12.7 | 13.4 | +0.70 [-0.22, +1.66] | 54 / 46 % |
| mean throttle, first 2 s | 0.69 | 0.63 | +0.02 [-0.02, +0.06] | 42 / 50 % |

Wheel data for the same runs: max wheel slip in the first 2 s 1.13 (failed) vs 0.87 (rescuer); spinning within 2 s
58 % vs 41 %; sinkage growth in the first 2 s -0.001 vs -0.005 m; over the drive +0.33 vs +0.06 m.

Would a simple rule over the five soil-arm picks have found the success? (success of the route the rule picks vs the
mean success of a random soil-arm pick)

| rule | choice groups (105) | all 800 groups |
|---|---|---|
| random soil-arm pick | 57.7 % | 83.7 % |
| smallest abs. speed step | 56.2 % | 83.5 % |
| smallest positive speed step | 49.5 % | 82.6 % |
| lowest start speed | 53.3 % | 83.1 % |
| lowest grade over the first 10 m | 63.8 % | 84.5 % |
| steepest metre in the first 10 m lowest | 56.2 % | 83.5 % |
| steepest metre of the whole route lowest | 57.1 % | 83.6 % |
| lowest steep share of the route | 63.8 % | 84.5 % |
| largest abs. heading step | 60.0 % | 84.0 % |

Reading: about a third of the soil-arm failures were avoidable with the routes the arms already proposed, and the
routes that avoided them did so by turning off the approach line toward lower ground in the first 10 m, not by
starting at the vehicle's speed. The differences are small (medians near zero for most quantities), so the outcome
in a choice-dependent group is close to a threshold. The other two thirds (82 groups) failed under every arm; most of
them (65) are reachable from a standing start, which points at the decision point, not the planner's choice among
its candidates.

## Caveats

- The standing-start comparison mixes two things: A3 used the deployed ensembles (trained on all rows) and A5 the
  same-row ensembles; the S_crm row pairs different training sets of the same specialist. H and T are the same arm
  names in both protocols but also different deploy ensembles. The trend with approach grade is the same for every
  pairing.
- The within-group analyses only see the routes the six planners chose; none of them forced a matched start in the
  all-fail groups (20 soil routes happened to start within 0.5 m/s there, all failed). A candidate family that always
  starts at the vehicle's speed (plan S1) is not in this data.
- Steep cells come from the undeformed map; soil deformation is not in the terrain metrics. Slope threshold 15 deg and
  the 2 s handover window are my choices; the stall-onset histogram shows the result does not hinge on 2 s.
- Wheel rows for successes in groups without any failure use a 600-run random sample (of 2,757); every failure and
  every run of a group with a failure has wheel data.
- Everything is one arena, one soil, one approach speed.

## Files

- `scout/D1_anatomy.json`: every number above (Q1_groups, Q2_time_course, Q2_onset, Q3_speed_step, Q4_approach,
  Q5_recoverability) plus the definitions.
- `scout/d1_cache/d1_anatomy.py`: the analysis script (stages `features`, `analyze`).
- `scout/d1_cache/features.json`: per-group and per-run features (no wheel data).
- `scout/d1_cache/fetch_list.txt`, `scout/d1_cache/extra/<run>/crm_extra.npz`: the 2,495 per-wheel files fetched
  read-only from the cluster (hash-checked).
