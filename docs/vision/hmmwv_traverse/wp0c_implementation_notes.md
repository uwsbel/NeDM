# WP0c implementation notes — storage schema + smoke-tier collection

**Date:** 2026-09-01 · **Modules:** `nedm/traverse/storage.py`, `nedm/traverse/drivers.py`
**Scripts:** `scripts/traverse_collect.py`, `scripts/traverse_storage_bench.py`
**Artifacts:** `artifacts/traverse/smoke_v1/` (collected on newton, mirrored locally; `storage_bench.json` inside)

## Schema v1 (plan §6.1, "episode-chunked zarr/zstd or equivalent")

One directory per episode: `rgb.bin` + `depth.bin` (temporal chunks of 20
frames = 1 s, each stored as keyframe + wraparound diffs, zstd-9), `states.npz`
(float32 table of the full `capture_row` set + powertrain torque/speed + the
applied 20 Hz actions), `meta.json` (layout manifest, driven route, contact
events, per-chunk byte index). Depth is uint16 mm above 80 m with a no-hit
sentinel — 0.5 mm quantization vs the 6–17 mm sensor error measured at WP0b.
Dataset root carries `manifest.json`: camera model (incl. `ray_scale=1.200`,
convention "ray"), heightmap sha256, mixture roster, `processed_caches:
"reference"` (§6.1 decision: caches point at this store, never duplicate it).

Every episode self-verifies at collection time (bit-exact RGB, ≤0.5 mm depth
roundtrip on random windows) before the worker reports success.

## Measured numbers (10 episodes × 400 frames, 256², newton)

| Check | Result | Consequence |
|---|---|---|
| Compression ratio (raw frames / disk) | **28.8×** (RGB 27×, depth 35×) | 4.3 MB per 20 s episode |
| Tier extrapolation | pilot ≈ **1.7 GiB**, full ≈ **17 GiB** | full tier fits everywhere; the §6.1 122–488 GiB raw risk is closed |
| Random-window loader (8 frames × batch 16, 1 thread) | **578 windows/s** = 4.6k frames/s = 1.44 GiB/s raw-equivalent | no loader bottleneck at 5090 training rates |
| Peak disk during collection | = final store size (writer streams compressed chunks) | no preprocessing spike |
| Collection wall, 3 procs on newton | 10 episodes / 20.7 min (~124 s/episode effective) | 3-way parallelism ≈ 90% efficient (physics-bound, GPU contention negligible) |
| Pilot forecast (200 eps × 20 s, 3 procs) | ≈ **7 h** on newton | ~3.5–4 h plausible at 6 procs — measure before relying on it |

## Findings

1. **Keyframe+diff is what makes the ratio:** the arena background is static,
   so within a 1 s chunk only the vehicle (~15×7 px) and its shadow change;
   zstd on wraparound diffs gives 28.8× where plain per-frame compression of
   noise-like content gives ~2×. Random access inside a chunk stays O(1)
   because diffs reference the chunk keyframe, not the previous frame.
2. **Routes must respect the episode duration** (fixed after smoke run 1): the
   first near-obstacle episode drove an 86 m route (est 41.8 s) and the 20 s
   window ended 3.4 m before the target — zero recorded contact. Spline routes
   now require est duration ≥ 1.05× episode; near-obstacle passes must arrive
   within 0.55× episode.
3. **Graze offsets must reference physical geometry, not planner footprints:**
   a rock's footprint radius is its circumscribed-corner radius and a tree's
   carries +0.4 m margin, so "footprint − 0.25 m" usually misses the actual
   body. Contact passes now aim at box-face / trunk radius ± small offset.
4. **Driver mixture roster is index-based** (`FAMILY_CYCLE`, 60/20/10/10 per
   any 10 episodes), so tiers of any size keep §6.2 proportions and the
   assignment is reproducible from the episode index alone.
5. **The vehicle was a ghost to obstacles** (found by the graze episode):
   `create_hmmwv` left Chrono's default `ChassisCollisionType = NONE`, and
   TMEASY tires only query the terrain — a pass with the rock buried 0.62 m
   inside the vehicle footprint recorded 0 N. Chassis collision is now
   config-driven (`vehicle.chassis_collision`, default NONE for legacy
   datasets); the traversal config sets `HULLS`. Consequence for WP0a: G0a's
   "zero asset contact" was vacuously true — re-run the gate under HULLS to
   make it real (see wp0a notes addendum).

## Smoke iteration log (runs 3–6, all under HULLS)

Enabling real contact exposed a chain of route/controller defects; each was
diagnosed from the recorded stores (states.npz trajectories), fixed, and
verified against the exact collection seeds locally before re-running:

| Run | Defect found | Fix |
|---|---|---|
| 3 | Oracle/spline grazes (2.0 m inflation == hull half-width + tracking); contact-pass pinned head-on 15 s at 118 kN | interim +0.6 m tracker margin in inflation (§7.4); sideswipe aims hull-side overlap, not centerline |
| 4 | Oracle backed into a tree at t=18 s (post-goal creep at terminal speed); Chaikin pulled a spline corner into a rock; sideswipe missed 6 m (U-turn geometry vs 8 m turn radius) | park at route end; re-validate smoothed splines; heading-aware pass geometry |
| 5 | Sideswipe stalled on an unvalidated 19.7° climb at the 2.0 m/s slope-modulation floor, rolled backward; follower pre-wound full steering lock during settle | slope caps (13° pass section / 19° route, splines require slope_ok); near-obstacle v_min 3.0; straight-wheel settle |
| 6 | **clean** — spline/oracle 0 N, sideswipe contact at t=10.1 s / 2.9 m/s / 54 kN peak | — (pilot launched) |

Final smoke audit (run 6): 10/10 complete; contact only in the two meander
episodes (69.9 / 139.6 kN — random driving legitimately hits things now) and
the intended sideswipe, which comes to rest against the rock after impact
(the honest outcome of a 0.4–0.65 m hull-overlap hit on an immovable
boulder: ~10 s driving + ~10 s contact-load dynamics).

## Pilot tier

Launched 2026-09-02 (newton): 200 episodes × 20 s, seeds 20261000+ (disjoint
from smoke), 3 procs, ~7 h, expected ~0.9 GB. Route-generation fallback to
oracle measured on pilot seeds: ~5% spline, ~50% near-obstacle (hilly arena +
slope/heading constraints; the meander families keep contact volume up).
`traverse_collect --indices` can re-run individual episodes into the store.

## Still owed for G0b

- Re-run the G0a gate under HULLS (CPU-only, ~50 min at 12 procs) so the
  no-collision claim is non-vacuous.
- Analytic class-mask rasterizer + one-shot `ChSegmentationCamera` validation
  (masks stay derivable-on-demand from the layout manifests in `meta.json` —
  §6.3 stores manifests, not masks).
