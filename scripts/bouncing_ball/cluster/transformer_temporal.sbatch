#!/bin/bash
#SBATCH -J ball-temporal
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 04:00:00
set -euo pipefail
set +u
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$BALL_CODE/src:${PYTHONPATH:-}"
cd "$BALL_CODE"
if [[ "${BALL_TEST:-0}" == 1 ]]; then python3.12 tests/bouncing_ball/test_transformer_temporal.py; fi
python3.12 -m nedm.bouncing_ball.transformer_temporal_train --data "$BALL_DATA" --config "$BALL_CONFIG" --output-dir "$BALL_RUN" "$@"
