#!/bin/bash
#SBATCH -J ball-precision
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 03:30:00
set -euo pipefail
set +u
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
set -u
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=4
export PYTHONPATH="$BALL_CODE/src:${PYTHONPATH:-}"
cd "$BALL_CODE"
if [[ "${BALL_TEST:-0}" == 1 ]]; then
    python3.12 -m unittest discover -s tests/bouncing_ball -p 'test_precision.py' -v
    python3.12 -m unittest discover -s tests/bouncing_ball -p 'test_launch_optimization.py' -v
fi
python3.12 -m nedm.bouncing_ball.precision --data "$BALL_DATA" --output "$BALL_RUN" "$@"
