#!/usr/bin/env bash
# CLEAN BASELINE for the L8-vs-anchor RL comparison. Waits for the currently
# running L8 RL run to finish, then launches the SAME legacy RL recipe with the
# ONLY change being the dynamics checkpoint = the anchor's *best_val.pt* (ep69,
# rollout_sel 0.074) instead of the anchor's last.pt (ep80, rollout_sel 0.139)
# that the legacy run actually used. This isolates the checkpoint-selection
# effect from the architecture effect:
#   legacy       = anchor 6L  last.pt     (already exists)
#   THIS baseline= anchor 6L  best_val.pt (new)
#   L8 run       = L8    8L  best_val.pt (running now)
# => (L8 best_val) - (anchor best_val) = pure architecture; (anchor best_val) -
#    (legacy last) = pure checkpoint-selection.
# Runs ON luffy (same box as legacy + L8). Launch detached so it survives:
#   ssh luffy 'cd ~/NeDM && tmux new-session -d -s anchor_bestval_rl_chain \
#     "bash scripts/ablation_ofat/chain_anchor_bestval_rl_luffy.sh"'
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/harry/miniconda3/envs/nedm/bin/python}"
RUN_NAME="hmmwv_rl_15d_crm2000mix25_onehot_anchorL6_bestval69_flat20crm20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010"
CKPT="artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g_crm2000_mix25_rebal_rollout_onehot/checkpoints/best_val.pt"
REF="artifacts/rl_reference_sets/hmmwv_tire_normal_force_omega_flat_crm_train_refs_40_1100_randwin_seed20260623.npz"
LOG="artifacts/rl_runs/${RUN_NAME}.launch.log"

if [[ ! -f "$CKPT" ]]; then echo "missing anchor best_val checkpoint: $CKPT" >&2; exit 1; fi
if [[ ! -f "$REF" ]]; then echo "missing reference set: $REF" >&2; exit 1; fi

# 1) Wait for the L8 RL run to finish (its process disappears). The bracket in
#    the pattern keeps pgrep from matching this script or the grep itself; this
#    script's own name does not contain the pattern string.
echo "[chain $(date -Is)] waiting for the L8 RL training process to finish..."
while pgrep -f "[t]rain_hmmwv_rl_tracking" >/dev/null 2>&1; do sleep 60; done
echo "[chain $(date -Is)] L8 process gone; settling 20s then launching anchor best_val baseline"
sleep 20

# 2) Launch the anchor best_val.pt baseline — IDENTICAL args to the L8 run except
#    --dynamics-checkpoint and --run-name.
"$PYTHON" scripts/train_hmmwv_rl_tracking.py \
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
  --dynamics-checkpoint "$CKPT" \
  --reference-path "$REF" \
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
  --run-name "$RUN_NAME" 2>&1 | tee "$LOG"
echo "[chain $(date -Is)] --- anchor best_val baseline exited ---"
exec bash
