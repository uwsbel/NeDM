#!/usr/bin/env bash
# Distill the z1 teacher into z2-history students for several seeds, ONE JOB AT A
# TIME (plan section 11 asks for >= 3 student seeds), then evaluate each student
# against the teacher on the identical held-out pairs.
#
#   bash scripts/training/launch_dpend_distill_seeds.sh [seeds...]   (default: 1 2 3)
#
# Env: conda activate nedm; run from the repo root.
set -euo pipefail
export PYTHONPATH=src
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1 2 3)
TEACHER="${TEACHER:-artifacts/rl_runs/dpend_nrd_reach_z1_armreward_lowerhalf_seed1_20260826}"
ITERS="${ITERS:-200}"
DATE="$(date +%Y%m%d)"

for SEED in "${SEEDS[@]}"; do
  RUN="dpend_nrd_student_z2hist4_from_z1_armreward_lowerhalf_seed${SEED}_${DATE}"
  echo "=== seed ${SEED}: ${RUN}"
  python scripts/training/distill_dpend_nrd_student.py \
    --teacher-run-dir "${TEACHER}" --seed "${SEED}" --iterations "${ITERS}" --run-name "${RUN}" \
    > "artifacts/rl_runs/${RUN}.log" 2>&1
  python scripts/evaluation/eval_dpend_nrd_student.py --student-run-dir "artifacts/rl_runs/${RUN}" \
    >> "artifacts/rl_runs/${RUN}.log" 2>&1
  tail -6 "artifacts/rl_runs/${RUN}.log"
done
echo "all seeds finished"
