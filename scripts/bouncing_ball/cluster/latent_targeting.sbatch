#!/bin/bash
#SBATCH -J ball-latent-target
#SBATCH -A dannegrut
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
set -euo pipefail
set +u
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$BALL_CODE/src:${PYTHONPATH:-}"
cd "$BALL_CODE"
python3.12 -m nedm.bouncing_ball.latent_targeting --checkpoint "$BALL_CHECKPOINT" --data "$BALL_DATA" --config configs/bouncing_ball/launch_precision_v3.json --output "$BALL_RUN"
(
    source "$BALL_CODE/scripts/bouncing_ball/cluster/collection_env.sh"
    "$NRD_PYTHON" -m nedm.bouncing_ball.verify_launch --run "$BALL_RUN"
)
export MPLCONFIGDIR="$BALL_ROOT/mpl_cache"
export PYTHONPATH="/work1/dannegrut/harry/experiments/ball_span_v1_20260930/plot_deps:$PYTHONPATH"
python3.12 -m nedm.bouncing_ball.optimization_report --run "$BALL_RUN"
