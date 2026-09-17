# Independent sensor-input review — 2026-09-15

## Judgment

The project has made measurable progress. Keep calibrated depth-to-height as the
working sensor interface. The next bounded task should correct depth-to-world
mapping and audit information retained in the raw-depth arm, using existing
observations and recorded routes before expanding training or physical tests.

This review inspected the current source, sensor/gen plans, reports, saved result
JSON, and the matching local worktree conversation. The supplied thread UUID
`b3766215-527a-4536-be7f-ca90e4a854c8` was not found in local Codex/Claude records.
The matching transcript is stored as Claude session
`00c16740-3609-430f-9620-dcad93459566`; its latest substantive response agrees with
the sensor_v1 report. The association with the supplied UUID is unverified.
No training, new renders, simulations, or cluster jobs were run for this review.

## What currently runs

Static overhead depth -> height/terrain corridor for each of 256 candidate
routes -> trained CNN/BiGRU risk ensemble -> minimum-risk route -> Chrono PID.
The BiGRU runs along route stations; this is not the earlier recurrent vehicle
world model. There is no vehicle state input, explicit time cost, or replanning
within a leg. Multi-goal missions replan at goal boundaries. The camera observes
the entire arena from 110 m above. Controller path heights still use simulator
terrain; the sensor-only boundary currently applies to scoring/proposals.

The f104 training input already came from depth converted to height. gen_v1 used
authored heightmaps for its cross-arena test. sensor_v1 uses captured depth for
all compared scorers, including the N2 baseline.

## Existing evidence

In sensor_v1 test 2, on the same 1,200 start/goal groups:

| Input | Unsafe at fixed 2 m/s | Unsafe with speed free |
|---|---:|---:|
| N2: sensor-derived height + slopes | 5.67% | 0.75% |
| E0: sensor-derived height, no slopes | 6.50% | 1.00% |
| D: relative raw depth + camera ray angle | 9.25% | 1.58% |

Unsafe means failure or backwards sliding under the experiment's label. E0's
fixed-speed difference is +0.83 percentage points, 95% CI [-0.58, +2.25]; it does
not meet the predeclared +2.0-point non-inferiority margin. Its speed-free result
meets the +1.0-point margin. Raw depth's worse transfer replicates test 1.
These tests reuse six terrains; fresh start/goals are not fresh terrain types.

The stronger gen_v1 result is at matched speed: new-arena unsafe rates 5.9%
for N2 versus 13.6% for the hand terrain/speed rule. With speed free, 1.3%
versus 1.8% did not pass the primary significance test, and N2 was slower.

## Independent code findings

1. **Elevation conversion does not produce a world-grid map.**
   `src/nedm/traverse/fdm_diverse_data.py:20` computes correct per-pixel height
   `z = H - depth/sec(theta)`, then resizes in image coordinates.
   `scripts/sensor_dataset.py:79` and `scripts/f104_n2_dataset.py:27` sample
   those pixels as if one pixel corresponded to a constant horizontal spacing
   at ground level. For a perspective camera, x/y also depend on measured depth.
   Correct coordinates are `x = ray_x * depth/sec(theta)` and
   `y = ray_y * depth/sec(theta)`, rather than `ray_x * H`, `ray_y * H`.
   Back-project points into 3D and rasterize them into a metric world grid.

   A read-only check of 10,201 saved raw pixels per arena finds mean height
   errors of 0.028-0.045 m under the current flat-grid assignment versus
   0.006-0.008 m using full back-projection. The 95th-percentile horizontal
   displacement is 0.53-0.78 m, with maximum 1.50 m across these samples.
   See `geometry_audit.json` and its runnable source. This is shared by N2 and
   E0; it is not proof of the cause of D's relative performance gap, nor proof
   that changing the map will improve an existing checkpoint.

2. **The raw-depth arm discards an input needed for exact height recovery.**
   `scripts/sensor_dataset.py:83` subtracts the route-start range `d0`.
   `scripts/sensor_train.py:19` passes the relative depth and ray angle, but
   does not pass `d0`. With `s_i = sec(theta_i)` and `delta_d_i = d_i-d0`,
   relative height is `z_i-z0 = d0/s0 - (d0+delta_d_i)/s_i`.
   In general this requires the omitted absolute range. The D comparison is
   therefore not simply the same information with a harder coordinate system.
   Preserve absolute depth or explicitly include d0 in a diagnostic variant.

The explanation that more arenas will fix raw depth, and that RGB fails because
of shading memorization, remains a hypothesis. The results establish the
performance differences, not those mechanisms.

## Suggested task for the coding agent

Implement a versioned depth -> 3D points -> metric elevation-grid adapter, with
coverage/validity masks and no authored-height fallback. Check world alignment
on saved captures (authored heights allowed only as evaluation references).
Audit the raw-depth arm with absolute range retained. Keep existing checkpoints
and preprocessing intact; first replay existing candidate routes to quantify
changed corridors, rankings, and disagreement cases. Do not claim deployment
gains from geometry error alone or silently replace trained preprocessing.

After that audit, run a small matched comparison of height+slopes, height-only,
and an information-preserving raw-depth arm, with identical rows, seeds,
candidate pools, and training budget. Split development by whole terrain, using
the existing multi-arena recordings. All previously inspected arenas are
development evidence; reserve genuinely new arenas for final confirmation.
Prioritize route-choice errors at matched speed over pooled AUC. Broader
training or physical reruns should follow a concrete result from this audit.
