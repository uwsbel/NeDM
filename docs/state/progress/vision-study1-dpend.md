# Study 1 — double pendulum + RGB camera

**Status:** Complete (WP1–WP5 + distillation) · **Updated:** 2026-09-02 ·
**Branch:** `nrd_vision`

Primary sources, richer than this summary:
`docs/vision/double_pen/implementation_notes.md`,
`rl_implementation_notes.md`, `distillation_implementation_notes.md`.

## Where we are

Done, and it validated the architecture. The purpose was never to show vision
*adds* information — on a fixed-camera pendulum the scene is nearly determined
by `z1`, and the notes verify this (a pose decoder `D(z1)` nearly matches the
autoencoder: weighted L1 0.0114 vs 0.0084). It was to show the joint `[z1, z2]`
model predicts both streams consistently and still supports policy transfer.

## What is done, with evidence

**Joint prediction.** Autonomous 3 s rollouts (only the 16 context frames are
ever encoded; both `z1` and `z2` recursively predicted afterwards) reach 24.0 mm
tip error. Cross-modal consistency — tip blob in the *decoded predicted* frame vs
pinhole projection of the *predicted* `z1` — **1.0 px median** (p95 17 px, 719
checks). Decoded PSNR ~29 dB throughout.

**Policy transfer, no gap.** A state-only policy trained entirely inside the
frozen decoder-free NRD: **87% NRD / 87% Chrono**, 84 of 87 successes shared,
per-pair closest-approach difference 3.3 mm median. Run
`dpend_nrd_reach_z1_armreward_lowerhalf_seed1_20260826`, 1500 iterations,
1 h 45 min.

**Camera-only student matches the privileged teacher.** Four 10 Hz latents plus
the goal (258-D, no `z1`, no tip, no goal error): 88 / 90 / 87 % across three
seeds vs the teacher's 87 %; action MAE 0.02 on a ±1 range, 1–2 % sign
disagreement.

**Throughput.** Batched NRD, decoder off, batch 4096: 293 k transitions/s vs
2.3 k/s for Chrono + camera (~129×), and only ~8 % below the state-only model's
319 k/s — the 64-D latent is nearly free. Decoder on: 22 k/s.

**The remaining `z1` gap is data-limited, not architectural.** 6.1× more data
closed the gap to state-only from 8.2× to 3.3× at 0.5 s, scaling as roughly
`data^-0.6`, while the matched state-only model was already saturated.
Extrapolating, another ~5–6× data (~15 min of collection) would bring NRD near
the state-only floor.

## What is next

Nothing pending. Two things were explicitly deferred by user direction and are
still open if ever wanted: Policy B (`--policy-obs z1z2`) under the same recipe,
and a Chrono transfer of the *distilled student* (camera frame → frozen encoder
→ 4-step history), which the distillation notes flag as the natural next check.

## Findings that generalize — read before any vision work

Both are in [`docs/state/lessons/`](../lessons/); they will recur at 256² where
the vehicle is ~15×7 px.

1. Latents share a huge constant component (raw pairwise cosine 0.9998).
2. A plain L1 autoencoder erases the moving object and reconstructs only the
   background.
