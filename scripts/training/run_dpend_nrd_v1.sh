#!/usr/bin/env bash
# Reproduce NRD Study 1 (double pendulum + Chrono camera) end to end on one box
# with a GPU (OptiX for Chrono::Sensor, CUDA for training). ~1.5 h total.
# Env: conda activate nedm; run from the repo root.
set -euo pipefail
export PYTHONPATH=src

# 1. Collect the pilot dataset (200 x 10 s episodes, ~4 GB, ~1 min).
python -m nedm.double_pendulum_data \
  --episodes 200 --max-steps 500 --seed 20260825 \
  --output-root artifacts/datasets/dpend_pilot_200 --dataset-name dpend_pilot_200

# 2. Validation gates G0-G2 (live mechanism/camera checks + stored-data checks).
python scripts/collection/validate_dpend_dataset.py --dataset-root artifacts/datasets/dpend_pilot_200

# 3. Processed cache with frames.
python -m nedm.training.preprocess \
  --dataset-root artifacts/datasets/dpend_pilot_200 \
  --output-dir artifacts/training_datasets/dpend_pilot_seq16_v1 \
  --state-fields cos_q1 sin_q1 cos_q2 sin_q2 omega1_radps omega2_radps \
  --action-fields action_elbow \
  --rollout-fields tip_x_m tip_z_m \
  --frames

# 4. State-only NeDM baseline (gate G3 reference).
python -m nedm.training.trainer --config configs/nrd/dpend_state_v1.json

# 5. NRD: autoencoder warm-up, then the joint [z1, z2, a] model.
python -m nedm.nrd.trainer --config configs/nrd/dpend_nrd_v1.json --stage ae
python -m nedm.nrd.trainer --config configs/nrd/dpend_nrd_v1.json --stage joint

# 6. Pose-conditioned decoder D(z1) honesty baseline (study plan section 10, #5).
python -m nedm.nrd.trainer --config configs/nrd/dpend_nrd_v1.json --stage posedec

# 7. Evaluation: curves, cross-modal consistency, side-by-side rollout GIFs.
python scripts/evaluation/eval_nrd_dpend.py \
  --nrd-checkpoint artifacts/training_runs/dpend_nrd_v1/checkpoints/best_val.pt \
  --state-checkpoint artifacts/training_runs/dpend_state_v1/checkpoints/best_val.pt

# 8. Throughput benchmark (gate G6).
python scripts/throughput/probe_nrd_dpend_throughput.py \
  --nrd-checkpoint artifacts/training_runs/dpend_nrd_v1/checkpoints/best_val.pt \
  --state-checkpoint artifacts/training_runs/dpend_state_v1/checkpoints/best_val.pt
