#!/usr/bin/env bash
# Train the 12-D [q, qd, qcmd] arm dynamics ROM (ee_base/Pee channel dropped).
#
# Identical recipe to arm_transformer_full_v1 (6L/8H/256, ctx16, seed 20260630,
# 80 epochs x 2000 steps, batch 256, rollout_sel checkpoint selection) -- the ONLY
# changes are the 12-D state (no ee_base) and, because ee_base is no longer a state
# channel, rollout_sel now scores EE via forward kinematics on the predicted joints
# (rollout_eval.pose = "ee_base_fk") against the Chrono-recorded ee_base.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=/home/harry/anaconda3/envs/nedm/bin/python
RUN_DIR=artifacts/training_runs/arm_transformer_noee_v1
mkdir -p "$RUN_DIR/logs"

PYTHONPATH=src "$PY" scripts/train_hmmwv_dynamics.py \
  --config configs/arm_transformer_noee_v1.json \
  --device cuda \
  2>&1 | tee "$RUN_DIR/logs/run.log"
