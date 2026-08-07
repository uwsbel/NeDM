#!/usr/bin/env bash
# Train the 8-D [q, qd] arm dynamics ROM with the ABSOLUTE joint command as the action.
#
# Layout history for f_arm (all on the same arm_dynamics_v3 raw data):
#   15-D  state [q, qd, qcmd, ee_base]   action = act (= dqcmd)   -> arm_transformer_full_v1
#   12-D  state [q, qd, qcmd]            action = act (= dqcmd)   -> arm_transformer_noee_v1
#    8-D  state [q, qd]                  action = qcmd_next       -> THIS RUN
#
# qcmd moves out of the state and becomes the action, so the model is a clean
# "joints in, commanded joints in, joint delta out" ROM: state 8-D, action 4-D,
# readout 8-D ([dq, dqd]). Both trivially-predicted channel groups are now gone --
# ee_base (retired in the 12-D run, FK on the predicted q beats it) and dqcmd
# (the identity channel the delta parameterization forced the head to emit).
# The qcmd *history* is not lost: the 16-step action window still carries it.
#
# Recipe follows arm_transformer_noee_v1 (8H/256, ctx16, seed 20260630, 80 epochs
# x 2000 steps, batch 256, rollout_sel selection via rollout_eval.pose =
# "ee_base_fk") with ONE deliberate capacity change: 5 transformer layers instead
# of 6, since the token is now 12-wide (8 state + 4 action) rather than 16-wide.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=/home/harry/anaconda3/envs/nedm/bin/python
RAW_ROOT=artifacts/datasets/arm_dynamics_v3_home_reset_fulltraj_shards
DATA_DIR=artifacts/training_datasets/arm_dyn_v3_8d_seq16_v1
RUN_DIR=artifacts/training_runs/arm_transformer_8d_v1
mkdir -p "$RUN_DIR/logs"

# ---------------------------------------------------------------------------
# 1. Processed cache. Same 15 raw shards / same episode split as the 12-D run --
#    only the column selection differs, so this is a re-slice, not new data.
#    state  = q_0..3, qd_0..3                (8-D; targets are the per-step deltas)
#    action = qcmd_next_0..3                 (4-D absolute command; see note below)
#    rollout= ee_base_{x,y,z}                (Chrono-recorded EE, the FK-eval ground truth)
#
#    The action MUST be qcmd_next, not qcmd. In arm_data.run_episode the row's
#    `qcmd` is the command that was in force over the PREVIOUS interval; the loop
#    then sets `actuator.qcmd = qcmd_next` and only THEN substeps, so qcmd_next is
#    the target the PD law actually drives toward across [t, t+1]. Feeding the
#    stale qcmd was measurably wrong: regressing the per-step joint acceleration
#    on the PD terms gives R^2 = [.24 .46 .08 .0005] with (qcmd - q) versus
#    [.24 .51 .20 .39] with (qcmd_next - q). It also makes this ROM informationally
#    equivalent to the 12-D one, whose token (q, qd, qcmd, dqcmd) reconstructs
#    qcmd_next exactly -- so the 8-D really is just the 12-D minus redundancy.
# ---------------------------------------------------------------------------
if [[ ! -f "$DATA_DIR/metadata.json" ]]; then
  echo "[1/2] building processed cache -> $DATA_DIR"
  PYTHONPATH=src "$PY" -m nedm.training.preprocess \
    --dataset-root "$RAW_ROOT"/shard_{000,001,002,003,004,005,006,007,008,009,010,011,012,013,014} \
    --output-dir "$DATA_DIR" \
    --state-fields q_0 q_1 q_2 q_3 qd_0 qd_1 qd_2 qd_3 \
    --action-fields qcmd_next_0 qcmd_next_1 qcmd_next_2 qcmd_next_3 \
    --rollout-fields ee_base_x ee_base_y ee_base_z \
    2>&1 | tee "$RUN_DIR/logs/preprocess.log"
else
  echo "[1/2] processed cache already present -> $DATA_DIR (skipping preprocess)"
fi

# ---------------------------------------------------------------------------
# 2. Train. Checkpoint selection is open-loop-rollout EE drift at 0.5 s
#    (rollout_sel), scored by FK on the predicted joints against the recorded
#    ee_base -- identical metric to the 12-D run, so best_val.pt is comparable.
# ---------------------------------------------------------------------------
echo "[2/2] training -> $RUN_DIR"
PYTHONPATH=src "$PY" scripts/training/train_hmmwv_dynamics.py \
  --config configs/arm_transformer_8d_v1.json \
  --device cuda \
  2>&1 | tee "$RUN_DIR/logs/run.log"

# Post-hoc open-loop EE eval (FK mode is auto-detected: no ee_base state channel):
#   PYTHONPATH=src $PY scripts/evaluation/eval_arm_rollout.py \
#     --checkpoint artifacts/training_runs/arm_transformer_8d_v1 --device cuda
