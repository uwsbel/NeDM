#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
set -eo pipefail
source /work1/dannegrut/harry/nrd/env.sh
nrd_pychrono
nrd_use_lavapipe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
campaign=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909
export FDM_RUNTIME_FINGERPRINT=$campaign/pilot_runtime_412394.json
export GEN_ROOT=$campaign/gen_v1
export PYTHONPATH="$GEN_ROOT/source/scripts:$GEN_ROOT/source/src:${PYTHONPATH:-}"
export GEN_WORKERS=${GEN_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-16} - 2 ))}
"$NRD_PYTHON" -P -u "$GEN_ROOT/gen_runner.py"
