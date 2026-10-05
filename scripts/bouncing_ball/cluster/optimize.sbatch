#!/bin/bash
#SBATCH -J ball-optimize
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 00:30:00
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
python3.12 -m unittest discover -s tests/bouncing_ball -p 'test_launch_optimization.py' -v
python3.12 -m nedm.bouncing_ball.optimize_launch --checkpoint "$BALL_CHECKPOINT" --data "$BALL_DATA" --output-dir "$BALL_RUN" "$@"
(
    source "$BALL_CODE/scripts/bouncing_ball/cluster/collection_env.sh"
    "$NRD_PYTHON" -m nedm.bouncing_ball.verify_launch --run "$BALL_RUN"
)
export MPLCONFIGDIR="$BALL_ROOT/mpl_cache"
export PYTHONPATH="${BALL_PLOT_DEPS:-$BALL_ROOT/plot_deps}:$PYTHONPATH"
python3.12 -m nedm.bouncing_ball.optimization_report --run "$BALL_RUN"
