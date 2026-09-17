#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
# One overhead RGB-D capture per arena (array index -> arena).
set -eo pipefail
source /work1/dannegrut/harry/nrd/env.sh
nrd_pychrono
nrd_use_lavapipe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=16
C=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909
S=$C/sensor_v1
ARENAS=(arena_f104_50h_v1 arena_g228 arena_g203 arena_g217 arena_g216 arena_g231)
A=${ARENAS[$SLURM_ARRAY_TASK_ID]}
cd $S
"$NRD_PYTHON" -P -u "$S/sensor_capture_map.py" --source-root "$C/gen_v1/source" --arena "$C/gen_v1/source/assets/traverse/$A" \
  --out "$S/maps/$A" --chrono-data "$CHRONO_BUILD/data"
