# Vision-line checkpoints

**Updated:** 2026-09-02. Sourced from the Study 1 implementation notes; **not**
verified against any filesystem in this doc's lifetime — see rule 2 in
[`README.md`](README.md).

## Study 1 — dynamics models

| Run | Path | Config | Selected | What it is for |
|---|---|---|---|---|
| `dpend_nrd_full_v1` | `artifacts/training_runs/dpend_nrd_full_v1/checkpoints/best_val.pt` | `configs/nrd/dpend_nrd_full_v1.json` | ep 24/30 | **The deployed joint NRD.** Everything downstream uses it |
| `dpend_state_full_v1` | `artifacts/training_runs/dpend_state_full_v1/…` | `configs/nrd/dpend_state_full_v1.json` | ep 38/40 | Matched state-only baseline (RQ comparator) |
| `dpend_nrd_v1` | `artifacts/training_runs/dpend_nrd_v1/…` | `configs/nrd/dpend_nrd_v1.json` | ep 29/60 | Pilot-tier joint model; kept for the data-scaling comparison |
| `dpend_state_v1` | — | `configs/nrd/dpend_state_v1.json` | — | Pilot-tier state-only |

The pilot and full configs keep training recipes **identical** so the
full-vs-pilot delta is attributable to data alone. Epochs 31–60 of `dpend_nrd_v1`
used a cosine warm restart and did not improve rollout selection.

## Study 1 — policies

| Run | Result | Notes |
|---|---|---|
| `dpend_nrd_reach_z1_armreward_lowerhalf_seed1_20260826` | **87% NRD / 87% Chrono** | The deployed teacher. `model_1499.pt` final; `model_400`/`model_800` also evaluated |
| `dpend_nrd_student_z2hist4_from_z1_armreward_lowerhalf_seed{1,2,3}_20260826` | 88 / 90 / 87 % | Camera-latent-only students distilled from the teacher |

Kept for the record, all failures: `*_plan_v1_*` (spin exploit, killed ~it 100),
`*_plan_v2_*` (failure charge, plateau 13–17%, killed ~it 250),
`*_lowerhalf_v2_*` (plan reward, killed ~it 150). See
[`../lessons/rl-in-nrd.md`](../lessons/rl-in-nrd.md).

## Context banks (`artifacts/rl_reference_sets/`, local-only)

| Bank | Use |
|---|---|
| `dpend_nrd_full_v1_train_contexts_16384_seed20260826.npz` | Env resets |
| `dpend_nrd_full_v1_val_contexts_512_seed20260826.npz` | Held-out evaluation |

Both encoded by `dpend_nrd_full_v1`. The env refuses a bank whose `z2_mean`
fingerprint does not match the model's.

## Study 3

**None.** No traversal model has been trained — see
[`../progress/vision-study3-traverse.md`](../progress/vision-study3-traverse.md).
