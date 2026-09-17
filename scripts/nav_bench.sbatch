#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:20:00
set -eo pipefail
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
export MIOPEN_USER_DB_PATH=/tmp/miopen_$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH
C=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909
N=$C/nav_v1
export PYTHONPATH="$N/code:$N/source/src:${PYTHONPATH:-}"
export GEN_SRC=$N/source/src
F=$C/sensor_v2/vehicle_frames/g216_v2_group_0000
for DEV in cuda cpu; do
  python3.12 -u $N/code/nav_latency_bench.py --models "$N/models/matched_Dabs_s*.pt" --frame $F \
    --device $DEV --threads 16 --out $N/bench_$DEV.json
done
