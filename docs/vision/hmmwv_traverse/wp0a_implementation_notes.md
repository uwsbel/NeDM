# WP0a implementation notes — oracle vertical slice (G0a: PASSED)

**Date:** 2026-09-01 · **Machines:** newton (collection/sim), 5090 box (analysis)
**Gate artifacts:** `artifacts/traverse/wp0a_gate_v2/` on newton (`summary.json`, `episodes.jsonl`)

## G0a result

100-seed batch (`scripts/traverse_wp0a_gate.py`, seeds 20260901–20261000,
scripted ChPathFollowerDriver tracking the privileged oracle plan, no
rendering, no learning):

| Metric | v1 (pre-fix) | v2 (final) | Bar |
|---|---|---|---|
| Approach-pose reach (success@2 m, 0.5 s hold) | 92/100 | **100/100** (Wilson 95% CI [0.963, 1.0]) | ≥ 95% |
| Episodes with asset contact | 0 | **0** (max contact 0.0 N) | 0 |
| Cross-track mean / max | 0.46 / 27.98 m | **0.069 / 0.88 m** | — |
| Layout resamples needed | 27/100 episodes | same (deterministic) | — |
| Plan lengths | 34.3–61.0 m | same | — |

Wall: ~49 min at 12 procs on newton (~15–25× slower than realtime per episode).

## Bugs found and fixed on the way

1. **Orientation calibration read zeros** (`scene.calibrate_orientation`):
   `RigidTerrain.GetHeight` raycasts *downward from the query point*, and the
   Bullet collision system only sees the patch after `BindAll()`. Querying at
   z=0 before any step returned 0 everywhere, so all 8 dihedral orientations
   failed identically. Fix: bind + probe from above `height_max_m`.
   Calibration then froze decisively: `{rot90: 0, flipud: true}`,
   RMSE 8.1 mm (≈ BMP quantization floor), runner-up 0.82 m.
2. **Depth camera filter graph crash**: `ChDepthCamera` installs its own
   access filter internally (per the `demo_SEN_camera.py` note); pushing a
   second `ChFilterDepthAccess` fails AddSensor validation. Fix: don't.
3. **Oracle speed profile ignored the standing start** (all 8 gate failures):
   the forward accel pass ramped from `v[0]` = cruise-capped speed, so the
   driver full-throttled from rest. Failure taxonomy: launch wedge-in (tiny
   cross-track, no progress) and pure-P steering saturation → bang-bang
   divergence up to 28 m off-path, one rollover at 57° roll. Fix:
   `PlannerParams.v_launch_mps = 1.5` clamps `v[0]` before the forward pass.
4. **No steering rate limit on the scripted driver**: added the repo
   convention (0.1 per 20 Hz step = 2.0 full-scale/s) to the showcase and
   gate drive loops, matching the plan's "all drivers steering-rate-limited".

After fixes 3+4, the 10 worst v1 seeds (8 failures + 2 recovered 17–27 m
excursions) rerun 10/10 clean with max cross-track 0.64 m.

## Diagnosis workflow (for reference)

The batch gate is headless by design; episodes are seed-deterministic, so a
failing seed is reproduced exactly with rendering via
`scripts/traverse_showcase_episode.py --seed <seed> --episode-id <id> --out
artifacts/traverse/fail_<id>` (movie + 20 Hz `track.csv`). Failure movies for
v1 seeds 20260906/07/28 are under `artifacts/traverse/fail_gate_00{5,6}/`,
`fail_gate_027/` on newton.

## Showcase episode (WP0a demo)

`artifacts/traverse/showcase_000/` on newton: probe frame, mp4 (189 frames @
20 fps), `preview_obs_256.png` (collection-resolution observation — rocks,
trees, house roof, and vehicle marker all resolvable). Success in 9.4 s,
cross-track 0.05 m mean / 0.28 m max, zero contact.

## Known plan-text deviations (not yet reconciled)

- `LayoutParams` samples 6–10 rocks and 8–14 trees; plan §3.2 says rocks
  8–15, trees 5–10.

## Addendum (2026-09-02, from WP0c)

The "zero asset contact" result above was measured with the vehicle's chassis
collision type at Chrono's default `NONE` (and TMEASY tires never contact
rigid bodies), so the contact channel could not have fired: the claim was
vacuously true. Physical non-contact remains plausible (2 m plan inflation,
0.88 m max cross-track), but the gate must be re-run with
`vehicle.chassis_collision = "HULLS"` (now the traversal default) to make
G0a's no-collision claim real. See wp0c_implementation_notes.md finding 5.

**Re-run under HULLS (2026-09-02): G0a PASSES for real.** Two batches on
the same 100 seeds (4.4 min each on newton after the narrowphase fix, vs
the ~50 min pre-fix forecast):

- v3, interim `tracker_p95_margin_m = 0.6`: 100/100 reach the approach
  pose, but **one episode (seed 20260911) grazes an asset at 10.4 kN** —
  with its own cross-track max only 0.55 m. The planar margin misses
  roll/pitch-induced hull excursion; batch cross-track p99 is 0.76 m,
  max 0.85 m, so 0.6 was undersized on both counts.
- v3 with `tracker_p95_margin_m = 0.9`: **100/100 success, zero contact
  episodes, max contact 0.0 N**, layout resamples 36/100 (vs 33 at 0.6 —
  the tighter corridors cost almost nothing). Committed as the new
  default; still to be replaced by a measured held-out p95 at G6.

Gate artifacts: `artifacts/traverse/wp0a_gate_v3_hulls/` (0.6, fail) and
`wp0a_gate_v3_m090/` (0.9, pass) on newton.
