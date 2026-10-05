#!/bin/bash
#SBATCH -J ball-precision-select
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 00:25:00
set -euo pipefail
set +u
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
set -u
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$BALL_CODE/src:${PYTHONPATH:-}"
cd "$BALL_CODE"
python3.12 -m nedm.bouncing_ball.select_precision --root "$BALL_ROOT" --data /work1/dannegrut/harry/experiments/ball_span_v1_20260930/final_data
