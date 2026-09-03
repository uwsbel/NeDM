# Study 3 — HMMWV terrain traversal + overhead RGB-D

**Status:** WP0 complete (G0a/G0b largely passed), **WP1 not started** ·
**Updated:** 2026-09-02 · **Branch:** `nrd_vision`

Plan: `docs/vision/hmmwv_traverse/NRD_hmmwv_traversal_study_plan.md` (v1.1,
revised after a written self-review in `..._review.md`).
Notes: `wp0a_`, `wp0b_`, `wp0c_implementation_notes.md`.

## Where we are

This is the first study where vision is **load-bearing**: per-episode obstacle
layout is available only through the camera, and the dynamics model can localize
only through the camera (`z1` carries no absolute pose by design). Start and
goal poses are privileged in v1 and the plan says so openly.

Architecture is hierarchical, not end-to-end:
overhead RGB-D → `z2` → costmap head → planner (once per episode); partial `z1`
→ 38-D tracker at 20 Hz.

The data pipeline is finished and de-risked. **The model does not exist yet.**

## What is done, with evidence

| WP | Gate | Result |
|---|---|---|
| WP0a oracle vertical slice | G0a | **FAILS under real collision.** See below |
| WP0b sensor smoke | G0b partial | alignment **0.97 px** median / 3.06 p95 (bar: 2 / 4); depth→elevation 6.3 mm median, **16.9 mm at image edges** |
| WP0c storage + collection | G0b partial | **28.8×** compression; pilot **200/200** episodes, 80 k frames, **873 MB**, 6.5 h at 3 procs |

**The storage risk is closed.** The plan's §6.1 projected 122–488 GiB for the
full tier. Episode-chunked keyframe + wraparound-diff zstd exploits the static
background for 28.8×, putting the full tier at ~17 GiB. Loader does 551
windows/s warm, 344 cold — no bottleneck at training rates.

**A real upstream Chrono bug was found.** This build's `ChDepthCamera` casts
rays 1.20× wider than the constructor HFOV implies, while the RGB
`ChCameraSensor` honors it. Uncorrected, depth→elevation error was 1.5 m median
with a radial signature. Fitted as a scalar `ray_scale = 1.200` in
`nedm/traverse/camera.py` and recorded in the dataset manifest. **Worth
reporting upstream.**

## Blocker: neither environment can render this study on `kyle-sbel`

**Found and resolved 2026-09-02.** `src/nedm/traverse/scene.py:410` calls
`manager.scene.AddDirectionalLight(...)`. `ChScene` in pychrono 9.0.0 has no
directional light at all: only `AddPointLight` and `AddAreaLight` exist, and
there is no `DirectionalLight` class. Both `kyle-sbel` and `kyle-N7-B650E` run
9.0.0, so **every rendering path in this study raises `AttributeError` on both
machines.** WP0b and WP0c ran on `newton`, which is unreachable, and nothing has
re-run them since.

`src/nedm/double_pendulum_data.py:199` uses `AddPointLight` and is unaffected,
so Study 1 renders and Study 3 does not.

**The environment fixes the API and breaks the renderer.** Both measured on
`kyle-sbel`, 2026-09-02:

| | `envs/chrono` (9.0.0) | `envs/nedm` (10.0.0) |
|---|---|---|
| `AddDirectionalLight` | **absent**, `AttributeError` | present |
| Chrono::Sensor / OptiX | renders (1246 frames) | **`OPTIX_ERROR_UNSUPPORTED_ABI_VERSION`** |

So 9.0.0 renders but cannot express this study's lighting, and 10.0.0 expresses
it but cannot start OptiX at all. **Moving this study to `nedm` trades an
`AttributeError` for an OptiX failure**; it does not unblock rendering.

The 10.0.0 failure is at `ChOptixEngine.cpp:86`, in the engine constructor, on
an RTX 3090 with driver 580.173.02 / CUDA 13.0 that the 9.0.0 build rendered on
an hour earlier. So it is the `projectchrono` build's OptiX ABI against this
driver, not the machine and not this repo's code. Whether it is the build or the
driver is **not yet known**: it needs testing on `kyle-N7-B650E`, which has a
different GPU. If both boxes fail it is the build.

A `hasattr` fallback to point lights in `scene.py` was considered and rejected:
point lights are not directional lights, and the shading change would silently
break comparability with the pilot tier already collected on `newton`.

**General lesson, which cost the wrong doc edit above:** a symbol being
importable does not mean its subsystem initialises. `pychrono.sensor` imported
cleanly under 10.0.0 and `AddDirectionalLight` was present; the engine still
refused to start. The same caution applies to the FSI/CRM finding in
[`future-case-studies.md`](future-case-studies.md), where presence has been
verified and a run has not.

## What is next — the one action

**Build the 4-channel RGB-D encoder and run the WP1 perception pilot.** Nothing
else can proceed; pilot data is collected and waiting.

Specifically, none of this exists yet:

- `src/nedm/nrd/vision.py` is RGB-only — line 41 hardcodes
  `_conv_block(3, channels[0])`, 128², `z2_dim=64`. Plan §5 needs 4 channels,
  256², `z2_dim=128`, one extra stride-2 stage.
- The four **mandatory** auxiliary warm-up heads (plan §5): occupancy/class
  mask, vehicle-center heatmap + yaw, foreground-weighted RGB, elevation with
  its own normalization. Labels are analytic and free.
- A traversal NRD config — `configs/nrd/` holds only `dpend_*`.
- The spatial-feature-map probe that quantifies what global pooling destroys,
  plus the pre-declared fallback if the single-latent probes miss their bars.

Plan §14 gates pilot-tier *training* on these probes passing.

## The zero-contact criterion barely discriminates

**Found 2026-09-02, and it outlives the bug that led to it.** G0a requires "zero
collisions". Reconstructing all 100 gate layouts from their seeds and measuring
the planned centreline's clearance to the nearest obstacle *edge*:

| | min | p05 | median | max |
|---|---|---|---|---|
| clearance to obstacle edge | **1.34 m** | 1.70 | 2.72 | 4.25 |

Against an HMMWV half-width of roughly 1.1 m, **0 of 100** episodes have a
*planned centreline* closer than half-width. But that is the wrong threshold: the
per-episode test subtracts that episode's own cross-track, and on that basis 4 of
100 fall inside half-width and **1 actually collided**.

So the criterion has **low discriminating power, not zero** — an earlier draft of
this section said zero and was too strong. The margin is thin enough that roughly
1 episode in 100 crosses it. The original G0a's zero was still vacuous, because
collision could not fire at all, but the arena does occasionally produce genuine
contact, so the gate is not purely measuring the planner.

Two ways to give the criterion content, and the study should do at least one:

1. **Record clearance.** `scripts/traverse_wp0a_gate.py` now writes
   `min_asset_clearance_m` per episode, so a pass reads "no contact **and**
   closest approach was X m" rather than an unfalsifiable zero. Runs predating
   this change cannot be audited from their own output.
2. **Keep a deliberate-graze case in the suite as a positive control** that must
   report non-zero. Without one, nothing proves the gate can fail.

Three separate mechanisms have now produced a vacuous zero here: chassis
collision at `NONE`, a hull mesh that failed to load, and a planner that simply
routes around everything. A fourth was hypothesised, that fixed asset bodies
might not accumulate contact force, and **disproved** by direct test: a fixed
`ChBodyEasyBox` under SMC/Bullet reported 11115.41 N, identical to the moving
body. The measurement path is sound.

## G0a FAILS once collision can actually fire

**Re-run 2026-09-02 on `kyle-N7-B650E`** under `HULLS`, `nedm` env, same 100
seeds from 20260901, 24 procs, 48 min 25 s. `gate_G0a_pass: false`.

| | Original (newton) | Re-run |
|---|---|---|
| Approach-pose reach | 100/100 | **100/100**, unchanged |
| Episodes with asset contact | 0, **vacuously** | **1** (episode 10, 10,266 N) |
| Cross-track mean / max | 0.069 / 0.88 m | 0.0755 / 0.7504 m |

The reach criterion still clears its 0.95 bar. The **zero-collision criterion
does not**, so the gate fails as written. Note reach did *not* move: contact
forces did not perturb the dynamics enough to cost a single approach pose, which
was an open question before this run.

**Episode 10 succeeded and collided.** Seed 20260911, reached the goal in 12.1 s,
hit a rock (footprint radius 1.17 m) sitting 1.42 m from the planned centreline
while running 0.55 m of cross-track. Success and no-contact are independent
criteria and this is the first episode to separate them.

The margin `planned clearance - that episode's own max cross-track` identifies
the risk band but does not determine contact: four episodes (10, 33, 64, 69) fall
below the 1.10 m half-width and exactly one collided. Episodes 64 and 33 are
tighter on paper and did not touch, because max cross-track is a bound over the
whole episode rather than a value attained at the closest obstacle. This is why
`min_asset_clearance_m`, measured from the driven trajectory, is strictly better
than anything reconstructable from the plan.

### Cause: smoothing destroys the planner's safety margin

**Located in code and measured, 2026-09-03.** The collision is not a property of
the arena. Min clearance to obstacle *edge*, all 100 seeds, densified to 0.1 m:

| Stage | min | p05 | median |
|---|---|---|---|
| Raw A* | **2.60** | 2.61 | 2.76 |
| After `_shortcut` | **2.60** | 2.60 | 2.80 |
| **Delivered** | **1.33** | 1.70 | 2.72 |

Raw A* sits exactly on `inflation_m 2.0 + tracker_p95_margin_m 0.6`, and
`_shortcut` preserves it because `_segment_valid` (`oracle.py:226-247`) re-checks
that same bound on every candidate segment. Then `_chaikin`, `_resample` and
`_repair_curvature` run with **no clearance check at all**.

The check behind them enforces a *different, weaker invariant*.
`validate_candidate` asserts `clearance > 0.0` against **uninflated** footprints,
which its own docstring states. With `footprint_width_m = 2.6`, that is exactly
"centreline ≥ 1.3 m". So the 1.33 m floor is not an accident, it is the
invariant being enforced. **41 of 100 episodes are delivered inside the budget
the search was built to guarantee**, median erosion 0.07 m, worst 2.22 m.

That is why episode 10 grazed. `oracle.py:41-44` says 2.0 m was chosen so
hulls-enabled episodes would not touch ("hull half-width 1.1 + gate cross-track
up to 0.88"). At the delivered floor of 1.3 m against a 1.1 m hull, 0.2 m remains
for tracking error while measured cross-track reaches 0.75 m. Episode 10 was
delivered at 1.42 m, ran 0.55 m of cross-track, and had 0.87 m against a 1.1 m
hull.

**What the study has to decide.** Plan §12.1 requires zero collisions, and the
honest framing is not "the arena occasionally produces contact" but "the stated
safety margin does not survive to the path the vehicle drives". Three shapes of
fix, and choosing between them is a judgement about what margin the study wants
to claim rather than a bug fix:

1. Make validation enforce what the search guarantees.
2. Re-shortcut after smoothing so the 2.6 m survives.
3. Lower `inflation_m`, on the view that 2.6 m was never really required.

`validate_candidate` now always reports `min_centreline_clearance_m`,
`search_inflation_m` and `inflation_preserved`, so any run is auditable against
this regardless of which fix is chosen. `PlannerParams.enforce_inflation_after_smoothing`
implements option 1 and is **off by default**, so prior results stay
reproducible and the two can be measured against each other.

### Still owed
2. Analytic class-mask rasterizer + one-shot `ChSegmentationCamera` validation.

## Open risks

- **Vehicle marker detection was 6/10 layouts** at 256² (~5×3 px). G1's
  vehicle-center probe depends on exactly this. Either enlarge the marker
  footprint or tune detection on collection-light frames.
- **Background collapse will be worse here than in Study 1.** The vehicle is
  ~15×7 px in a 256² frame — a smaller foreground fraction than the pendulum's
  3%. Foreground weighting is not optional.
- Terrain is one fixed authored map, so RQ2 is a **localization** claim, not a
  general terrain-from-depth claim. The plan reworded it and added an ablation
  grid plus a privileged-`(x,y,ψ)` upper bound to bound the claim; keep it that
  way when writing up.
- The `z2` prediction baselines are unusually strong here (static layout +
  constant-velocity vehicle). Aggregate latent metrics alone will not be
  informative; use the task-space metrics in plan §8.3.
