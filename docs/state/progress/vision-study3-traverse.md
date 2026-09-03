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
| WP0a oracle vertical slice | G0a | **100/100** approach-pose reach, cross-track 0.069 m mean / 0.88 m max |
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

## Resolved: the renderer needs the `nedm` environment

**Found and resolved 2026-09-02.** `src/nedm/traverse/scene.py:410` calls
`manager.scene.AddDirectionalLight(...)`. `ChScene` in pychrono 9.0.0 has no
directional light at all: only `AddPointLight` and `AddAreaLight` exist, and
there is no `DirectionalLight` class. Both `kyle-sbel` and `kyle-N7-B650E` run
9.0.0, so **every rendering path in this study raises `AttributeError` on both
machines.** WP0b and WP0c ran on `newton`, which is unreachable, and nothing has
re-run them since.

`src/nedm/double_pendulum_data.py:199` uses `AddPointLight` and is unaffected,
so Study 1 renders and Study 3 does not.

**The fix is the environment, not the code.** `conda env create -f
environment.nedm.yml` gives pychrono 10.0.0 from the `projectchrono` channel,
where `AddDirectionalLight` is present (verified on `kyle-sbel`). Run everything
in this study under `envs/nedm`, never under `envs/chrono`.

A `hasattr` fallback to point lights in `scene.py` was considered and rejected:
point lights are not directional lights, and the shading change would silently
break comparability with the pilot tier already collected on `newton`.

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

## Also owed before G0b is genuinely closed

1. **Re-run the G0a gate under `HULLS`.** The "zero asset contact" result was
   **vacuously true** — chassis collision was at Chrono's default `NONE` and
   TMEASY tires never contact rigid bodies, so the contact channel could not
   fire. Found in WP0c, flagged in the WP0a addendum. ~50 min at 12 procs,
   CPU-only.
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
