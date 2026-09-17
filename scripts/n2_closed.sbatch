#!/bin/bash
#SBATCH -J f104_n2c
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -o /work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/night2_closed_v1/arr_%A_%a.out
set -eo pipefail
source /work1/dannegrut/harry/nrd/env.sh
nrd_pychrono
nrd_use_lavapipe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2
campaign=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909
export FDM_RUNTIME_FINGERPRINT=$campaign/pilot_runtime_412394.json
export PYTHONPATH="$campaign/source_v1/scripts:$campaign/source_v1/src:${PYTHONPATH:-}"
export F104_ARRAY_ID=${SLURM_ARRAY_TASK_ID} F104_WORKERS=4
"$NRD_PYTHON" -P -u "$campaign/night2_closed_v1/n2_runner.py"
