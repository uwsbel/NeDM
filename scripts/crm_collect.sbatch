#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
# submit with: -p <one partition> -c <all CPUs of that node type> -t <limit> [--array=0-(nodes-1)]
#   --export=ALL,CRM_TASKS=...,CRM_OUT=...[,CRM_CONFIG=configs/x.json][,CRM_BUDGET_S=...]
set -uo pipefail
source /work1/dannegrut/harry/nrd/env.sh
export CHRONO_BUILD=$NRD_ROOT/chrono-build-fsi   # FSI/HIP build (superset of the rigid build)
nrd_pychrono
export CRM_ROOT=/work1/dannegrut/harry/experiments/crm_f104_20260916
export CRM_CHRONO_DATA=$CHRONO_BUILD/data
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CRM_GPUS=$(rocminfo | grep -c -E '^\s+Name:\s+gfx')
export CRM_OMP=4
echo "host=$(hostname) job=${SLURM_JOB_ID} part=${SLURM_JOB_PARTITION} gpus=$CRM_GPUS tasks=$CRM_TASKS out=$CRM_OUT config=${CRM_CONFIG:-default}"
exec "$NRD_PYTHON" -P -u "$CRM_ROOT/source/scripts/crm_worker.py"
