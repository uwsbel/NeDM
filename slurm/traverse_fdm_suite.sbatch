#!/bin/bash
# Run one independent model on each of the eight AMD GPUs.
# CODE_ROOT must point to an immutable source snapshot, separate from live jobs.
#SBATCH -J traverse_fdm
#SBATCH -A dannegrut
#SBATCH -p mi3508x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -t 00:30:00
#SBATCH -o /work1/dannegrut/harry/experiments/traverse_mppi_20260908/logs/%x_%j.out
set -eo pipefail
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
: "${CODE_ROOT:?Set immutable CODE_ROOT}"
: "${DATA_ROOT:?Set prepared DATA_ROOT}"
: "${RUN_ROOT:?Set separate RUN_ROOT}"
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
STEPS=${STEPS:-100}
BATCH=${BATCH:-128}
EVAL_EVERY=${EVAL_EVERY:-100}
mkdir -p "$RUN_ROOT"
python3.12 -c 'import torch; print("Torch", torch.__version__, "GPUs", torch.cuda.device_count()); assert torch.cuda.device_count() == 8'
pids=()
gpu=0
for seed in 11 29; do
    for arm in profile no_history history no_terrain; do
        run="$RUN_ROOT/${arm}_s${seed}"
        mkdir -p "$run"
        (
            export HIP_VISIBLE_DEVICES=$gpu
            export MIOPEN_USER_DB_PATH="/tmp/fdm_${SLURM_JOB_ID}_${gpu}"
            export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
            mkdir -p "$MIOPEN_USER_DB_PATH"
            trap 'rm -rf "$MIOPEN_USER_DB_PATH"' EXIT
            python3.12 -u scripts/traverse_fdm_train.py \
                --data "$DATA_ROOT" --out "$run" --arm "$arm" --seed "$seed" \
                --steps "$STEPS" --batch "$BATCH" --eval-every "$EVAL_EVERY" --device cuda --save-predictions
        ) > "$run/job_${SLURM_JOB_ID}.log" 2>&1 &
        pids+=("$!")
        gpu=$((gpu+1))
    done
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
if [ "$failed" -ne 0 ]; then
    echo "FDM_SUITE_FAILED: inspect $RUN_ROOT/*/job_${SLURM_JOB_ID}.log"
    exit 1
fi
echo "FDM_SUITE_COMPLETE steps=$STEPS runs=$RUN_ROOT"
