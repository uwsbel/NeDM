#!/bin/bash
#SBATCH -J pool-plots
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:40:00
# Figures/videos on a compute node (matplotlib from the ball study's plot_deps).
set -euo pipefail
source "$POOL_CODE/scripts/pool_ball/cluster/chrono_env.sh"
export PYTHONPATH="$PYTHONPATH:/work1/dannegrut/harry/experiments/ball_span_v1_20260930/plot_deps:/work1/dannegrut/harry/experiments/ball_span_v1_20260930/video_deps"
eval "$POOL_PLOT_CMD"
