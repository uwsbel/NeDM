#!/bin/bash
# RGB-D/blank x global/path-patch architecture x two seeds, all with rotation.
#SBATCH -J rgbd_focused
#SBATCH -A dannegrut
#SBATCH -p mi3508x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -t 00:45:00
#SBATCH -o /work1/dannegrut/harry/experiments/traverse_mppi_20260908/logs/%x_%j.out
set -eo pipefail
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
: "${CODE_ROOT:?Set immutable source snapshot}"
: "${DATA_ROOT:?Set focused RGB-D pack}"
: "${RUN_ROOT:?Set new output directory}"
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
STEPS=${STEPS:-100}
BATCH=${BATCH:-64}
EVAL_EVERY=${EVAL_EVERY:-100}
PROGRESS_TARGET=${PROGRESS_TARGET:-sustained_stall}
mkdir -p "$RUN_ROOT"
python3.12 -c 'import torch; print("Torch", torch.__version__, "GPUs", torch.cuda.device_count()); assert torch.cuda.device_count() == 8'
pids=()
gpu=0
for seed in 11 29; do
    for architecture in global patch; do
        for arm in rgbd blank; do
            run="$RUN_ROOT/${architecture}_${arm}_s${seed}"
            mkdir -p "$run"
            patch_arg=--no-candidate-patches
            if [ "$architecture" = patch ]; then patch_arg=--candidate-patches; fi
            (
                export HIP_VISIBLE_DEVICES=$gpu
                export MIOPEN_USER_DB_PATH="/tmp/fdm_focus_${SLURM_JOB_ID}_${gpu}"
                export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
                mkdir -p "$MIOPEN_USER_DB_PATH"
                trap 'rm -rf "$MIOPEN_USER_DB_PATH"' EXIT
                python3.12 -u scripts/traverse_fdm_rgbd_train.py \
                    --data "$DATA_ROOT" --out "$run" --arm "$arm" --seed "$seed" \
                    "$patch_arg" --rotation-augmentation --progress-target "$PROGRESS_TARGET" \
                    --steps "$STEPS" --batch "$BATCH" --eval-every "$EVAL_EVERY" \
                    --device cuda --save-predictions
            ) > "$run/job_${SLURM_JOB_ID}.log" 2>&1 &
            pids+=("$!")
            gpu=$((gpu+1))
        done
    done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [ "$failed" -ne 0 ]; then
    echo "RGBD_FOCUSED_SUITE_FAILED: inspect $RUN_ROOT/*/job_${SLURM_JOB_ID}.log"
    exit 1
fi
echo "RGBD_FOCUSED_SUITE_COMPLETE steps=$STEPS runs=$RUN_ROOT"
