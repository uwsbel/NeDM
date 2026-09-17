#!/bin/bash
#SBATCH -J fdm_diverse_online
#SBATCH -A dannegrut
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 01:00:00
#SBATCH -o /work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909/logs/%x_%j.out
set -eo pipefail
source /work1/dannegrut/harry/nrd/env.sh
nrd_pychrono
nrd_use_lavapipe
: "${CODE_ROOT:?Use an immutable source snapshot}"
: "${TASKS:?Use a predeclared task manifest}"
: "${OUT_ROOT:?Use a new output root}"
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
"$NRD_PYTHON" -u scripts/traverse_fdm_rgbd_diverse_online_batch.py \
    --tasks "$TASKS" --out "$OUT_ROOT" --workers "${WORKERS:-8}" \
    --python "$NRD_PYTHON" --chrono-data "$CHRONO_BUILD/data"
