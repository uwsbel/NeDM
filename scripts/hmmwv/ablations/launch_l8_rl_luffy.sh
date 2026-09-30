#!/usr/bin/env bash
# Re-run the EXACT legacy RL tracking experiment
#   hmmwv_rl_15d_crm2000mix25_onehot_flat20crm20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010
# with the ONLY change being the frozen dynamics checkpoint: swap the mix25
# anchor (6L) for the OFAT Stage-A winner L8_H8_E256_ctx128 best_val.pt (ep51,
# S=0.0456). Every other env/train factor is held identical to that run's saved
# env_cfg.json / train_cfg.json (num_envs 2048, 64 steps/env, 2000 iters, seed 1,
# flat:1,crm:1 mix over the flat+crm 40-ref set, K=16 dynamics context,
# action_repeat 5, steering_rate_limit 0.10, pos_w 2.0, yaw_w 1.6, ar_w 0.2,
# state fields vx,vy,yawrate). Runs ON luffy (RTX 5090, ~/miniconda3) — the same
# box the legacy run used.
#   ssh luffy 'cd ~/NeDM && bash scripts/ablations/launch_l8_rl_luffy.sh'
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/harry/miniconda3/envs/nedm/bin/python}"
SESSION_NAME="${SESSION_NAME:-l8_rl_tracking}"
RUN_NAME="hmmwv_rl_15d_crm2000mix25_onehot_ofatL8_bestval51_flat20crm20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010"
CKPT="artifacts/training_runs/ablation_ofat/L8_H8_E256_ctx128/checkpoints/best_val.pt"
REF="artifacts/rl_reference_sets/hmmwv_tire_normal_force_omega_flat_crm_train_refs_40_1100_randwin_seed20260623.npz"
LOG_DIR="artifacts/rl_runs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${RUN_NAME}.launch.log"

if [[ ! -f "$CKPT" ]]; then echo "missing dynamics checkpoint: $CKPT" >&2; exit 1; fi
if [[ ! -f "$REF" ]]; then echo "missing reference set: $REF" >&2; exit 1; fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "RL run already live in tmux '$SESSION_NAME' (attach: tmux attach -t $SESSION_NAME)"; exit 0
fi

read -r -d '' CMD <<EOF || true
cd '$REPO_ROOT' && '$PYTHON' scripts/training/train_hmmwv_rl_tracking.py \
  --exp-name hmmwv-nn-tracking \
  --device cuda \
  --matmul-precision high \
  --num-envs 2048 \
  --num-steps-per-env 64 \
  --max-iterations 2000 \
  --num-learning-epochs 5 \
  --num-mini-batches 8 \
  --learning-rate 3e-4 \
  --entropy-coef 0.003 \
  --seed 1 \
  --dynamics-checkpoint '$CKPT' \
  --reference-path '$REF' \
  --terrain-mix flat:1,crm:1 \
  --dynamics-context-steps 16 \
  --action-repeat 5 \
  --steering-rate-limit 0.1 \
  --position-weight 2.0 \
  --yaw-weight 1.6 \
  --action-rate-weight 0.2 \
  --state-error-fields vel_body_x_mps,vel_body_y_mps,yaw_rate_radps \
  --obs-history-steps 10 \
  --reference-preview-steps 10 \
  --max-episode-steps 180 \
  --max-position-error-m 20.0 \
  --save-interval 100 \
  --logger tensorboard \
  --run-name '$RUN_NAME' 2>&1 | tee '$LOG'; echo '--- RL run exited ---'; exec bash
EOF

tmux new-session -d -s "$SESSION_NAME" "$CMD"
echo "launched L8 RL run in tmux '$SESSION_NAME'"
echo "run dir: $LOG_DIR/$RUN_NAME"
echo "attach:  tmux attach -t $SESSION_NAME"
echo "log:     $LOG"
