#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 04:00:00
set -eo pipefail
source /work1/dannegrut/harry/nrd/env.sh
nrd_pychrono
nrd_use_lavapipe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LP_NUM_THREADS=16
C=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909
export NAV_ROOT=$C/nav_v1
export NAV_SOURCE=$NAV_ROOT/source
export NAV_CHRONO_DATA=/work1/dannegrut/harry/nrd/chrono-build/data
export PYTHONPATH="$NAV_ROOT/code:$NAV_SOURCE/src:${PYTHONPATH:-}"
export GEN_SRC=$NAV_SOURCE/src
export NAV_MODELS="$NAV_ROOT/models/matched_Dabs_s*.pt"
export FDM_RUNTIME_FINGERPRINT=$C/pilot_runtime_412394.json
export NAV_WORKERS=${NAV_WORKERS:-1}
export NAV_TORCH_THREADS=8
"$NRD_PYTHON" -P -u "$NAV_ROOT/code/nav_array.py"
