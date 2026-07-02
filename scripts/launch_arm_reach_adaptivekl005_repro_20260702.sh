#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/harry/miniconda3/envs/nedm/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/rl_runs_arm_goal_reach}"
RUN_NAME="${RUN_NAME:-arm_reach_adaptivekl005_lr1e4_tol005_ep150_bonus150_sigma015_luffy_repro_20260702}"
SESSION_NAME="${SESSION_NAME:-arm_reach_adaptivekl005_20260702}"
LAUNCH_LOG="$OUTPUT_ROOT/$RUN_NAME.launch.log"

mkdir -p "$OUTPUT_ROOT"

train_cmd=(
  "$PYTHON_BIN" scripts/train_arm_rl_reaching.py
  --output-root "$OUTPUT_ROOT"
  --run-name "$RUN_NAME"
  --exp-name arm-nn-reaching
  --device cuda
  --matmul-precision high
  --num-envs 4096
  --max-iterations 1500
  --num-steps-per-env 64
  --num-learning-epochs 3
  --num-mini-batches 16
  --learning-rate 0.0001
  --schedule adaptive
  --desired-kl 0.005
  --entropy-coef 0.001
  --seed 1
  --dynamics-checkpoint /home/harry/NeDM/artifacts/training_runs/arm_transformer_full_v1/checkpoints/best_val.pt
  --processed-dataset-dir artifacts/training_datasets/arm_dyn_v3_seq16_v1
  --geometry-path artifacts/arm_geometry/arm_geometry_v1.json
  --action-repeat 1
  --max-episode-steps 150
  --ee-error-scale-m 0.15
  --action-rate-weight 0.02
  --success-bonus 150.0
  --success-tolerance-m 0.05
  --success-steps 1
  --reach-reward-type exponential
  --init-noise-std 0.3
  --save-interval 100
  --logger tensorboard
)

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "training already running in tmux session $SESSION_NAME"
    echo "log: $LAUNCH_LOG"
    exit 0
  fi

  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" \
    "$(printf '%q ' "${train_cmd[@]}") > $(printf '%q' "$REPO_ROOT/$LAUNCH_LOG") 2>&1"
  echo "started arm reaching adaptive-KL training in tmux session $SESSION_NAME"
  echo "run: $OUTPUT_ROOT/$RUN_NAME"
  echo "log: $LAUNCH_LOG"
  echo "attach: tmux attach -t $SESSION_NAME"
  exit 0
fi

"${train_cmd[@]}" 2>&1 | tee "$LAUNCH_LOG"
