# NRD Study 1 — Implementation Notes (Double Pendulum + Chrono Camera)

**Status:** WP1–WP4 implemented and evaluated on the pilot dataset (2026-08-25).
**Plan:** `NRD_double_pendulum_study_plan.md` (this directory).

## Pilot results (200-episode dataset, 24 longest val episodes, horizon 3 s)

Open-loop tip error AT the given horizon (autonomous = only the 16 context
frames are ever encoded; z1 and z2 both recursively predicted afterwards):

| model | 0.5 s | 1 s | 2 s | 3 s |
|---|---:|---:|---:|---:|
| NRD autonomous | 18.7 mm | 37.0 mm | 36.7 mm | 24.0 mm |
| NRD frame-anchored (true z2 input each step) | 19.9 mm | 34.2 mm | 92.5 mm | 161.9 mm |
| NRD with z2 mean-blinded | 388.6 mm | 361.3 mm | 429.5 mm | 596.3 mm |
| State-only NeDM (matched backbone) | 3.7 mm | 3.5 mm | 5.8 mm | 8.7 mm |
| Persistence | 617.0 mm | 306.5 mm | 479.8 mm | 433.7 mm |

- **Cross-modal consistency (G4):** tip-blob position in DECODED predicted
  frames vs pinhole projection of PREDICTED z1: **median 1.0 px** (p95 17 px,
  719 checks, 1 miss) — the physical and visual heads describe the same future.
- **Camera stream:** z2 cosine (normalized space) 0.95 @0.5 s → 0.88 @3 s;
  decoded-frame PSNR ~29 dB throughout the 3 s autonomous rollout.
- **G3 is the open gap:** the joint model's z1 error is ~5× the state-only
  baseline. Candidate levers, in order: full 1000-episode dataset, multi-step
  curriculum (`training.rollout_horizon` > 1, already implemented), larger
  batch (state-only saw ~3.5× more windows), encoder fine-tuning.
- **Unexpected ablation result:** blinding z2 collapses z1 rollouts to
  persistence level — the joint model genuinely READS pose from the latent
  instead of ignoring it (the plan expected near-redundancy). If robustness to
  missing vision is wanted, train with z2 dropout. Frame-anchoring degrades
  long-horizon z1 (161.9 mm @3 s vs 24.0 autonomous): true latents paired with
  drifted z1 are an inconsistent input the model never saw in training.
- Checkpoints: `dpend_nrd_v1/checkpoints/best_val.pt` (epoch 29 of 60;
  epochs 31–60 with a cosine warm-restart did not improve rollout selection).
  Eval artifacts: `artifacts/nrd_eval/dpend_v1/` (summary.json, curves PNG,
  3 side-by-side rollout GIFs).
- **Pose-decoder honesty check (plan 10, baseline 5):** D(z1) trained with the
  same foreground-weighted loss reaches val weighted L1 0.0114 / foreground L1
  0.030 vs the image autoencoder's 0.0084 / 0.023 — the fixed-camera scene's
  pixels are (nearly) fully explained by z1, as expected for Study 1. Report
  the visual result as architecture validation, not added information.
- **Throughput (G6, RTX 4090, `throughput.json`):** Chrono alone 6.9 k
  transitions/s (one instance); Chrono + camera 2.3 k/s; batched NRD with the
  decoder OFF 293 k/s at batch 4096 (~129× the rendering pipeline, and only
  ~8% below the state-only model's 319 k/s — the 64-D latent is nearly free);
  decoder ON 22 k/s. The batched-surrogate scaling argument holds even in this
  cheap scene.

## What exists

| Piece | Location |
|---|---|
| Chrono mechanism + Chrono::Sensor camera collector | `src/nedm/double_pendulum_data.py` (`python -m nedm.double_pendulum_data`, needs `PYTHONPATH=src`, `nedm` conda env) |
| Validation gates G0–G2 | `scripts/collection/validate_dpend_dataset.py` (live checks + `--dataset-root` stored checks) |
| Frames-aware preprocessing | `nedm.training.preprocess --frames` (writes `{split}_frames.npy` in the rollout-offsets layout) |
| Frames-aware windows | `WindowedHMMWVDataset(..., load_frames=True)` → adds `frames` (L+1, H, W, 3) uint8 per window |
| State-only baseline | existing trainer + new `rollout_eval.pose: "pendulum_tip"` mode; config `configs/nrd/dpend_state_v1.json` |
| Camera encoder/decoder | `src/nedm/nrd/vision.py` (Conv 32→256, GroupNorm+SiLU, 64-D LayerNorm latent; mirrored decoder) |
| Joint NRD model | `src/nedm/nrd/model.py` — `[z1, z2, a]` tokens → shared `ContinuousTransformer` → physical head (Δz1, normalized-target space) + visual head (next z2) |
| Two-stage trainer | `src/nedm/nrd/trainer.py` — `--stage ae` (frame L1 warm-up), `--stage joint` (NRD loss, optional multi-step `training.rollout_horizon`); config `configs/nrd/dpend_nrd_v1.json` |
| Evaluation + demo | `scripts/evaluation/eval_nrd_dpend.py` — autonomous vs frame-anchored vs state-only vs persistence curves, latent cosine, decoded PSNR, cross-modal pixel consistency, side-by-side GIFs |

## Mechanism (versioned in each `dataset_index.json`)

Planar X–Z, revolute axes +Y. Passive shoulder (`ChLinkLockRevolute`), elbow
`ChLinkMotorRotationTorque` (a ∈ [−1,1] → τ = a·1.5 N·m), `ChLinkRSDA` viscous
damping 0.06 N·m·s/rad at BOTH joints (pilot-tuned: 0.02 let the motor pump most
10 s episodes past the 35 rad/s spin guard). Links 0.3 m / 0.5 kg each,
dt_sim 1e-3 s, control/record 50 Hz. z1 = [cos q1, sin q1, cos q2, sin q2, ω1, ω2];
q2 is RELATIVE to link 1. Rollout fields = Chrono-measured (tip_x, tip_z).
Camera: 128×128 RGB at every control boundary, fixed pose (0, −2, 0) looking +Y,
hFOV 0.72 rad → image right = +X, image up = +Z, pixel ≈ 5 mm of workspace.
Yellow tip marker + white elbow marker + gray pivot marker support automated
pixel-space checks.

## Hard-won Chrono::Sensor + reset gotchas (do not rediscover)

1. **Default sensor lag = one full period.** Frame data only becomes available
   once sim time passes `launch + lag`, so a reader that blocks at the boundary
   deadlocks. `SetLag(0)`; also `SetCollectionWindow(0)` for a true snapshot
   (no motion blur).
2. **The scheduler misses boundaries.** Launch times are float32 accumulations
   of k/rate; around some boundaries (first seen at t = 2.14 s) the launch slips
   one substep late → 1 ms content error, or a deadlock for a blocking reader.
   Bypass: nominal camera rate = 1/dt_sim so the schedule is always behind the
   clock, call `manager.Update()` ONLY at control boundaries — each call then
   fires exactly one render of the current state (verified: no catch-up loop).
   Associate frames by `LaunchedCount`, not timestamps.
3. **Teleporting bodies leaves stale state.** After `SetPos/SetRot/SetPosDt/
   SetAngVelParent`, the next step depends on the PREVIOUS episode's motion
   (measured up to 0.47 rad/s one-step deviation; identical across solver
   choices, so not solver noise). `system.Setup(); system.Update()` after the
   reset makes one-step replay bitwise deterministic. This is why episodes can
   share one system + one sensor manager (avoiding the OptiX/scene re-creation
   instability class) with clean boundaries.
4. Whole-trajectory replay comparison is meaningless for this system (chaotic,
   Lyapunov amplification of 1e-16); the determinism gate is ONE-step replay.

## Training-side findings (2026-08-25)

5. **Latent constant component.** The encoder's LayerNorm'd latents share a huge
   common direction: raw pairwise cosine between ARBITRARY frames is 0.9998,
   per-dim std across frames only 0.013. Latent losses/metrics/tokens must use
   the model's z2 normalization (z2_mean/z2_std fitted from the frozen warm-started
   encoder at joint-trainer init); raw-cosine metrics read 1.000 and mean nothing.
6. **Background collapse.** The pendulum covers ~3% of pixels; a plain L1
   autoencoder reaches val L1 ≈ 0.005 / 26 dB PSNR by reconstructing ONLY the
   static background and erasing the pendulum entirely (verified visually).
   This is the plan's section-7 "reconstruction emphasizes background pixels"
   risk in its extreme form. Fix: per-pixel foreground weighting (1 + 30 on
   pixels that differ from the static per-pixel-median background model, which
   is exact for a fixed camera). Applied to the AE warm-up AND the joint stage's
   decoded-frame loss; the background model is stored in `ae_best.pt`.

## Validation gate results (this box, 2026-08-25)

- State round-trip and FK vs Chrono markers: ~1e-16.
- One-step replay determinism: exactly 0.
- Timestep convergence 1e-3 vs 5e-4: tip 0.52 mm @ 0.5 s, 0.89 mm @ 2 s.
- Planarity: 1e-16 m. Constraint drift over 10 s: 2.6e-6 m.
- End-to-end frame↔state alignment (yellow-blob centroid vs pinhole projection
  of recorded state): median 0.70 px, max 1.18 px over the smoke dataset.

## Datasets on this box

- `artifacts/datasets/dpend_smoke` — 10 × 2 s, gate-checked.
- `artifacts/datasets/dpend_pilot_200` — 200 × 10 s (spin guard truncates 22%),
  86,486 rows, 4.0 GB, collected in 33 s (RTF ≈ 52× with rendering).
- Processed cache: `artifacts/training_datasets/dpend_pilot_seq16_v1`
  (73,981 train / 12,305 val transitions + frames arrays).
- **Full tier (2026-08-25):** `artifacts/datasets/dpend_full_1000` — 1,000 × 10 s
  (prefix `dpendf`, seed 20260826; ids disjoint from the pilot), 444,488 rows,
  21 GB, collected in 173 s (RTF 51×); stored gates pass at median 0.64 px over
  8,000 sampled rows. Merged cache `dpend_full_seq16_v1` (full + pilot, 1,200
  episodes): 460,276 train / 69,498 val transitions ≈ 2.9 h of simulated motion,
  25 GB. Script: `scripts/collection/collect_dpend_full.sh`.
- Data-scaling experiment: `configs/nrd/dpend_state_full_v1.json` and
  `configs/nrd/dpend_nrd_full_v1.json` keep the pilot training recipes
  UNCHANGED so the full-vs-pilot delta is attributable to data alone.

## Data-scaling result (2026-08-25, all four checkpoints on the SAME 24 full-cache val episodes)

Tip error AT horizon (mm); 29 min → 177 min of training data, recipes identical:

| model / data | 0.5 s | 1 s | 2 s | 3 s |
|---|---:|---:|---:|---:|
| NRD, pilot | 72.0 | 107.6 | 59.9 | 78.6 |
| NRD, full | **24.9** | **47.4** | 98.7 | 76.0 |
| state-only, pilot | 8.8 | 19.1 | 26.3 | 31.9 |
| state-only, full | 7.6 | 23.6 | 15.9 | 20.2 |

- **The z1 gap is primarily data-limited:** NRD improved 2.9× @0.5 s and 2.3× @1 s
  from 6.1× data (error ≈ data^-0.6 at 0.5 s) while the matched state-only model
  was already saturated (~flat). Gap narrowed 8.2× → 3.3× @0.5 s. Extrapolating the
  slope, another ~5–6× data (~15 h sim ≈ 15 min collection) would bring NRD near the
  state-only floor; residual factors are multi-task loss competition and
  latent-drift feedback.
- Past ~1.5 s single-horizon values are chaos-noisy; judge on 0.5–1 s.
- Full-model camera stream also improved: z2 cosine 0.93 @0.5 s (was 0.88 on this
  val set), and cross-modal p95 tightened 17 px → 5.1 px (median 1.5 px, 0 misses).
- NOTE these full-val-set numbers are NOT comparable to the pilot-val-set table
  above (different, on-average harder episodes); comparisons are within-table only.
- Runs: `dpend_state_full_v1` (best ep38/40), `dpend_nrd_full_v1` (best ep24/30);
  eval `artifacts/nrd_eval/dpend_full_v1/`; pilot-on-full-val eval in the session
  scratchpad only (regenerate with `eval_nrd_dpend.py --processed-dir ...`).

## Known deviations from the study plan

- Storage is the NeDM house format (CSV + per-episode `.npy` frames + npy
  caches), not Zarr/HDF5 — consistency with the existing pipeline won.
- Splits are train/val (hash-assigned, like the arm collector); the 15% test
  split is deferred until a result needs protecting.
- The AE warm-up currently runs 10 epochs (val L1 ≈ 0.005, PSNR ≈ 26 dB);
  encoder stays frozen in the joint stage (`vision.freeze_encoder`).
- RL swing-up (WP5) not started.
