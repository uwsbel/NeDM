#!/bin/bash
#SBATCH -J ball-nrd
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 03:30:00
set -euo pipefail
set +u  # The site profile reads optional unset locale/environment variables.
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
set -u
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=4
export PYTHONPATH="$BALL_CODE/src:${PYTHONPATH:-}"
cd "$BALL_CODE"
python3.12 -m nedm.bouncing_ball.training --data "${BALL_DATA:-$BALL_ROOT/data}" --output-dir "$BALL_RUN" "$@"
