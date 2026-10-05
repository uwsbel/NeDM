#!/bin/bash
#SBATCH -J ball-precision-plots
#SBATCH -A dannegrut
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:15:00
set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source "$BALL_CODE/scripts/bouncing_ball/cluster/collection_env.sh"
export PYTHONPATH="/work1/dannegrut/harry/experiments/ball_span_v1_20260930/plot_deps:$PYTHONPATH"
export MPLCONFIGDIR="$BALL_ROOT/mpl_cache"
"$NRD_PYTHON" -m nedm.bouncing_ball.optimization_report --run "$BALL_ROOT/runs/launch_precision_v1"
"$NRD_PYTHON" -m nedm.bouncing_ball.precision_summary --root "$BALL_ROOT" --baseline-run /work1/dannegrut/harry/experiments/ball_span_v1_20260930/runs/launch_opt_v3
