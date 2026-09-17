# sensor_v1 — risk network on sensor channels instead of the height map (pre-registration, 2026-09-15)

Written before any sensor-channel model was trained or driven.

## What changes
The deployed night-2 model (`N2`) reads a corridor built from ONE overhead RGB-D capture per arena, but only through
a height channel computed from the depth image plus two finite-difference slopes (grade, cross-slope). Its colour
channels were never used. This milestone feeds the network the camera's own channels.

Captures: `sensor_v1/maps/<arena>` for f104 and the five sibling arenas, same camera/lights/renderer as f104's
original capture (`scripts/sensor_capture_map.py`). The f104 re-capture is byte-identical to `static_map_v1`
(rgb, depth, encoded map), so every arena's image comes from one reproducible sensor setup.

Tensors (`scripts/sensor_dataset.py`), same 96 x 32 corridor, ten channels: elev_rel, grade, cross (the current
input, reproduced to <= 0.002), speed, valid, depth_rel (raw ray depth minus depth at the route start), ray_sec1
(known per-pixel camera geometry), R, G, B. The sensor channels never pass through depth->height or slopes.

## Variants (identical network, loss, optimiser, schedule, rows, context as N2)
E (control: current input) | E0 (height only, no slopes) | D (raw depth + ray geometry) | RGBD (colour + raw depth
+ ray geometry) | RGB (colour + ray geometry, no depth).

## Offline sweep and selection rule
3 seeds per variant, fit on training groups outside the night-2 dev fold. Metric: dev-fold same-group same-speed
ranking AUC on unsafe (G_unsafe), as in night 2. Reported also: W/P, the f104 test split, and zero-shot ranking on
the 15,639 routes already driven on the five sibling arenas (captured-map corridors).
Selection: among D, RGBD, RGB, the variant with the highest mean dev G_unsafe; ties within 0.005 go to the variant
with fewer channels. It is deployed as `S` (5 seeds, exactly N2's deployment rows). E must land within 0.01 of N2's
night-2 dev numbers or the pipeline is investigated before anything is driven.

## Closed-loop test (Chrono, AMD cluster)
Fresh start/goals: 200 hill/crater groups per arena on f104 and the five sibling arenas (1,200), disjoint from every
earlier group. All arms of a group in one array task; identical 256-candidate pools per group. EVERY arm's corridor
comes from the arena's captured RGB-D image (no heightmap anywhere).
Arms: n2 (current input, speed free), s (sensor model, speed free), n2_fixed2, s_fixed2 (geometry-only pools at
2 m/s), straight6 (reference).
Primary (non-inferiority): pooled over the six arenas, fixed 2 m/s, unsafe rate of s_fixed2 minus n2_fixed2. The
sensor model "keeps the performance" if the upper end of the 95% CI (paired bootstrap over groups, resampled within
arena, 4,000 draws) is <= +2.0 percentage points.
Secondary: the same at speed free with margin +1.0 point; exact two-sided McNemar for either direction; f104-only and
new-arena-only; failure rate; tilt30; median time to goal.
If time allows (secondary, declared now): 100 f104 five-goal missions (gen_v1 mission files) with arms n2 and s in the
same job; mission success difference with exact McNemar.

## Result of the declared test (recorded before the follow-up below was designed)
Primary FAILED: fixed 2 m/s, pooled six arenas, sensor model (D, raw depth) 9.42% vs current 7.17% unsafe,
+2.25 points [+0.50, +4.00] (72 vs 45 discordant, p = 0.016). Speed free 2.17% vs 1.08% (+1.08 [+0.25, +2.00]).
f104 alone: identical (2.5 vs 2.5 at 2 m/s; 0.5 vs 0.5 speed free). The whole gap is on the new arenas, matching the
offline zero-shot drop (within-start/goal 0.932 vs 0.950). f104 missions: 96% vs 99% (1 vs 4, p = 0.375).

## Follow-up test (declared 2026-09-15 after the primary failed; exploratory relative to the original plan)
Hypothesis: raw ray depth mixes terrain height with the camera's viewing angle, and a model trained on one arena learns
that arena's view geometry. Unprojecting depth with the known camera intrinsics (z = camera height - depth/sec, the
standard pinhole step, no terrain knowledge) and dropping the hand-made slope channels should transfer.
Model E0 (channels: height from depth, speed, valid), deployed exactly like D (5 seeds, N2's rows).
Fresh start/goals: another 200 hill/crater groups per arena on the same six arenas, disjoint from all earlier groups.
Arms in one job per group: n2, e0, d (speed free), n2_fixed2, e0_fixed2, d_fixed2, straight6; all from captured images.
Primary: pooled, fixed 2 m/s, unsafe(e0_fixed2) - unsafe(n2_fixed2), non-inferior if the 95% CI upper end <= +2.0
points (same bootstrap). Secondary: speed free (margin +1.0); d vs n2 as a replication of the first test; f104 vs new.
