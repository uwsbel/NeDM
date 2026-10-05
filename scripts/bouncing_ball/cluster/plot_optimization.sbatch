#!/bin/bash
#SBATCH -J ball-opt-plot
#SBATCH -A dannegrut
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:05:00
set -euo pipefail
set +u
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
set -u
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$BALL_ROOT/plot_deps:$BALL_CODE/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="$BALL_ROOT/mpl_cache"
cd "$BALL_CODE"
python3.12 -m nedm.bouncing_ball.optimization_report --run "$BALL_RUN"
