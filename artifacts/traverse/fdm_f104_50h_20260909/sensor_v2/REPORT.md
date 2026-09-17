# sensor_v2 — can direct depth match the height-based planner? (2026-09-15)

Starting point: the independent review (artifacts/reviews/sensor_input_20260915). Existing checkpoints, datasets and
results are untouched; this is a separate v2 path. Training on the AMD cluster, diagnostics and pilot as instructed:
saved-data first, then a small Chrono pilot.

## 1. What was wrong, and what the corrected pipeline does
v1 placed each depth pixel at `ray_direction * camera_height` (flat ground) instead of `ray_direction * axial_range`,
displacing points by up to 1.5 m on slopes, and the depth-side model arm received range relative to the route start.
`scripts/sensor_map_v2.py` back-projects every pixel with the camera intrinsics and rasterises a metric world grid
(0.15625 m, +-40 m) holding height, absolute range, ray secant, colour and pixels-per-cell; empty cells are invalid
with no authored-height fallback. `scripts/sensor_dataset_v2.py` samples 12-channel corridors from that grid.

## 2. Geometry: fixed, and verified independently
| | v1 flat sampling | v2 back-projection |
|---|---|---|
| height error vs the authored heightmap (all 262,144 cells x 6 arenas) | 0.027-0.044 m MAE | **0.0071-0.0087 m** |
| on the steepest 10% of cells | 0.080-0.116 m | 0.014-0.016 m |
| vs the terrain Chrono actually simulates (per pixel) | ~0.008 m | **0.0008 m** |
Orientation, ray sign, axial-vs-ray range, intrinsics, sub-cell registration, aggregation and edges each survive an
explicit adversarial test in two independent re-implementations; every wrong variant is 15-650x worse.

**Repo-level finding (not about sensors):** `TerrainMap`, the privileged height oracle used as "ground truth" across
this project, is offset from Chrono's own `RigidTerrain` by a pure 511/512 radial scale (samples at cell centres vs
patch edges), up to 0.078 m horizontally at the arena edge. Every previous "authored height" check inherits it.

## 3. Information: the review's algebra is right, its practical weight is not
With only (relative range, secant) height is formally undetermined - verified on real captures (two corridor points
with identical inputs, 5.22 m apart in true height). But the route-start secant is in the same tensor, so the
practical residual is ~4 mm RMS. The depth arm's real handicaps are **scale and conditioning** (the perspective term
is 1.9-2.5x the height signal), **locality** (the height offset lives in a cell the front end cannot reach before it
pools laterally) and **cross-arena shift**.

## 4. Replay of the shipped test with corrected corridors
Reproduced the sensor_v1 test 2 bit-for-bit (1,200 groups, 614,400 candidates). Under v2 corridors rankings barely
move (Spearman 0.96-0.98) but the driven route changes in 19.5-25.2% of groups (35.3-42.8% at fixed 2 m/s), almost
always to a near-tie 1-2 m away; flips track the flat-ground displacement, not slope. On routes with known outcomes
the corrected geometry orders unsafe-vs-safe pairs better for the height models (0.665 -> 0.738 at fixed 2 m/s), but
that test treated correlated pairs as independent and lacked a placebo, so treat it as suggestive, not significant.

## 5. Matched training (identical rows, labels, seeds, budget; split by whole arena)
Train f104 + g228 + g203 + g217; held out g216 + g231. Primary metric: route choice at matched speed (1,200 cells).
| Input | unsafe pick | vs height |
|---|---|---|
| height + slopes | 14.50% | - |
| height only | 14.50% | +0.00 [-0.75, +0.75] |
| **absolute range + ray secant** | 14.92% | +0.42 [-0.42, +1.25] |
| relative range + ray secant (v1 arm) | 15.58% | **+1.08 [+0.25, +2.00]** |
(random pick 29.72%, no safe alternative in 12.42% of cells.) So with corrected geometry and absolute range, direct
depth is statistically indistinguishable from height offline; the v1 representation remains measurably worse.

## 6. Chrono pilot (200 fresh held-out-arena start/goals, 1,066 drives, 0 failures)
| Arm | unsafe at 2 m/s | unsafe speed free | failures | median time | tilt > 30 deg |
|---|---|---|---|---|---|
| height | 6.0% | 0.5% | 0 | 12 s | 3.0% |
| absolute depth | 5.0% (-1.0 [-4.5, +2.5]) | 0.0% | 0 | 12 s | 5.5% |
| relative depth | 7.5% (+1.5 [-1.5, +5.0]) | 0.5% | 0 | 12 s | 7.0% |
| 6 m/s straight line | - | 6.0% | 3.5% | 8 s | 21.0% |
No difference is resolvable at this size, and travel time is identical. Geometric accuracy improved by 4-10x; a
planning gain has NOT been demonstrated.

## 7. What is unresolved
- Whether the +0.42-point offline gap (depth vs height) is real or zero: the pilot cannot resolve 1-1.5 points.
- Whether absolute range reintroduces arena memorisation (it is the strongest cross-arena shift measured, though the
  trained model did not suffer on the held-out arenas).
- Everything so far is on previously inspected arenas; genuinely unseen arenas are still reserved.

## 8. Smallest next experiment
One fixed-speed (2 m/s) Chrono test, ~1,200 fresh start/goals, arms height / absolute-depth / relative-depth from the
same candidate pools, on 3 genuinely new arenas (new generator seeds, captured once each) plus the two held-out
arenas. Fixed speed is where the choice matters and the base rate is ~6%, so ~1,200 paired groups resolve a 1.5-point
difference; speed-free rates (0-0.5%) cannot. Pre-declare non-inferiority at +2 points, report travel time alongside.
