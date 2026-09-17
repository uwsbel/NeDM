#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:00:00
set -eo pipefail
source /work1/dannegrut/harry/nrd/env.sh
nrd_pychrono
nrd_use_lavapipe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=8
C=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909
N=$C/nav_v1
export PYTHONPATH="$N/code:$N/source/src:${PYTHONPATH:-}"
export GEN_SRC=$N/source/src
export NAV_MODELS="$N/models/matched_Dabs_s*.pt"
export FDM_RUNTIME_FINGERPRINT=$C/pilot_runtime_412394.json
M=${NAV_MISSION:-g216_nav_000}
"$NRD_PYTHON" -P -u "$N/code/nav_runner.py" --mission "$N/missions/$M.json" \
  --mode ${NAV_MODE:-periodic} --period ${NAV_PERIOD:-2.0} \
  --out "$N/pilot/${M}__${NAV_TAG:-p2}" --source-root "$N/source" \
  --chrono-data /work1/dannegrut/harry/nrd/chrono-build/data ${NAV_EXTRA:-}
