# Vision-line datasets

**Updated:** 2026-09-02. Sourced from the Study 1 and WP0c implementation notes.
Locations were true when written; verify with `du -sh artifacts/*`.

## Study 1 — double pendulum (RGB, 128²)

| Dataset | Episodes / size | Location | Notes |
|---|---|---|---|
| `dpend_smoke` | 10 × 2 s | `artifacts/datasets/` | Gate-checked |
| `dpend_pilot_200` | 200 × 10 s, 86,486 rows, **4.0 GB** | `artifacts/datasets/` | Collected in 33 s (RTF ≈ 52× with rendering); spin guard truncates 22% |
| `dpend_full_1000` | 1,000 × 10 s, 444,488 rows, **21 GB** | `artifacts/datasets/` | Prefix `dpendf`, seed 20260826, ids disjoint from pilot. 173 s (RTF 51×) |
| `dpend_pilot_seq16_v1` | 73,981 train / 12,305 val | `artifacts/training_datasets/` | Processed cache |
| `dpend_full_seq16_v1` | 460,276 train / 69,498 val, **25 GB** | `artifacts/training_datasets/` | Merged full + pilot = 1,200 episodes ≈ 2.9 h of simulated motion |

Collection: `scripts/collection/collect_dpend_full.sh`. Validation gates:
`scripts/collection/validate_dpend_dataset.py`. Stored gates pass at median
0.64 px frame↔state alignment over 8,000 sampled rows.

Storage is the NeDM house format (CSV + per-episode `.npy` frames + npy caches),
**not** Zarr/HDF5 — a deliberate deviation from the study plan, for pipeline
consistency. Splits are train/val only; the 15% test split was deferred.

## Study 3 — HMMWV traversal (**RGB-D**, 256²)

| Dataset | Episodes / size | Location | Notes |
|---|---|---|---|
| `smoke_v1` | 10 × 20 s | `artifacts/traverse/smoke_v1` (newton, mirrored local) | Run 6 was the first clean one; runs 1–5 each found a route/controller defect |
| `pilot_v1` | **200/200**, 80 k frames, **873 MB** | newton `artifacts/traverse/pilot_v1`, mirrored to the 5090 box | Seeds 20261000+, 6.5 h at 3 procs. **WP1 training input** |

**Schema v1** — one directory per episode:

| File | Contents |
|---|---|
| `rgb.bin`, `depth.bin` | Temporal chunks of 20 frames (1 s), each a keyframe + wraparound diffs, zstd-9 |
| `states.npz` | float32 table: full `capture_row` set + powertrain torque/speed + applied 20 Hz actions |
| `meta.json` | Layout manifest, driven route, contact events, per-chunk byte index |

Dataset root carries `manifest.json`: camera model (including
`ray_scale = 1.200`, convention `"ray"`), heightmap sha256, mixture roster, and
`processed_caches: "reference"`. Depth is uint16 mm above 80 m with a no-hit
sentinel — 0.5 mm quantization against 6–17 mm sensor error. **Every episode
self-verifies at collection time** (bit-exact RGB, ≤0.5 mm depth roundtrip on
random windows) before the worker reports success.

### Measured storage numbers

| | |
|---|---|
| Compression | **28.8×** (RGB 27×, depth 35×) → 4.3 MB per 20 s episode |
| Tier forecast | pilot ~1.7 GiB, **full ~17 GiB** (vs the plan's 122–488 GiB raw) |
| Loader | 551 windows/s warm, 344 cold (8 frames × batch 16) = ~1.38 GiB/s raw-equivalent |
| Peak disk | equals final store size — the writer streams compressed chunks, no preprocessing spike |

Keyframe+diff is what makes the ratio: the arena background is static, so within
a 1 s chunk only the vehicle (~15×7 px) and its shadow change. Random access
inside a chunk stays O(1) because diffs reference the chunk keyframe, not the
previous frame.

### Driver mixture (plan §6.2, index-based `FAMILY_CYCLE`, 60/20/10/10 per 10)

Pilot actuals: 112 spline / 40 meander / 20 oracle / 10 near_obstacle /
18 oracle_fallback. Contact in 33 episodes (25 meander / 6 near_obstacle /
2 spline), peaks 8–153 kN, p50 73 kN. Of the 10 contact-intended slots, 4 fell
back and **all 6 that generated produced recorded contact**; of the 10
clear-pass slots, 6 fell back and **all 4 that generated passed at 0 N**. No
false grazes, no missed intents.
