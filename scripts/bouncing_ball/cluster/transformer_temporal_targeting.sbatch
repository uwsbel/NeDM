#!/bin/bash
#SBATCH -J ball-temporal-target
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:00:00
set -euo pipefail
set +u
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$BALL_CODE/src:${PYTHONPATH:-}"
cd "$BALL_CODE"
python3.12 -m nedm.bouncing_ball.transformer_temporal_targeting --checkpoint "$BALL_CHECKPOINT" --data "$BALL_DATA" --config configs/bouncing_ball/launch_transformer_temporal.json --output "$BALL_RUN"
(
    source "$BALL_CODE/scripts/bouncing_ball/cluster/collection_env.sh"
    "$NRD_PYTHON" -m nedm.bouncing_ball.verify_launch --run "$BALL_RUN"
)
export MPLCONFIGDIR="$BALL_ROOT/mpl_cache"
export PYTHONPATH="/work1/dannegrut/harry/experiments/ball_span_v1_20260930/plot_deps:$PYTHONPATH"
python3.12 -m nedm.bouncing_ball.optimization_report --run "$BALL_RUN"
