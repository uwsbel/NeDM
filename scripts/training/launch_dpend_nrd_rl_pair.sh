#!/usr/bin/env bash
# Train the paired NRD reaching policies (plan section 9): Policy A observes z1,
# Policy B observes [z1, z2]. Same frozen NRD, seed, reset bank, goals, reward.
# Both runs fit on one 24 GB GPU concurrently.
#
#   bash scripts/training/launch_dpend_nrd_rl_pair.sh [tag] [extra train args...]
#
# Env: conda activate nedm; run from the repo root.
set -euo pipefail
export PYTHONPATH=src

TAG="${1:-plan_v1}"
shift || true
SEED="${SEED:-1}"
ITERS="${ITERS:-800}"
NUM_ENVS="${NUM_ENVS:-4096}"
OUT_ROOT="artifacts/rl_runs"
DATE="$(date +%Y%m%d)"

for OBS in z1 z1z2; do
  RUN="dpend_nrd_reach_${OBS}_${TAG}_seed${SEED}_${DATE}"
  mkdir -p "${OUT_ROOT}"
  echo "launching ${RUN}"
  nohup python scripts/training/train_dpend_nrd_rl_reach.py \
    --policy-obs "${OBS}" \
    --seed "${SEED}" \
    --num-envs "${NUM_ENVS}" \
    --max-iterations "${ITERS}" \
    --run-name "${RUN}" \
    --output-root "${OUT_ROOT}" \
    "$@" \
    > "${OUT_ROOT}/${RUN}.log" 2>&1 &
  echo "  pid $!  log ${OUT_ROOT}/${RUN}.log"
done
wait
echo "both runs finished"
