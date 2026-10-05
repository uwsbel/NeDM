#!/bin/bash
#SBATCH -J pool-certify
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 01:00:00
set -euo pipefail
set +u
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$POOL_CODE/src:${PYTHONPATH:-}"
cd "$POOL_CODE"
python3.12 -m nedm.pool_ball.certify --runs $POOL_RUNS --train-data "$POOL_DATA" --sealed-data "$POOL_SEALED" --output "$POOL_CERT"
