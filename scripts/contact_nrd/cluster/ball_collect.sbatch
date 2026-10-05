#!/bin/bash
#SBATCH -J cn-ball-collect
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 01:00:00
set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source "$BALL_CODE/scripts/bouncing_ball/cluster/collection_env.sh"
"$NRD_PYTHON" -m nedm.bouncing_ball.campaign --campaign "$BALL_CAMPAIGN" --output-dir "$BALL_OUT" --workers "${SLURM_CPUS_PER_TASK:-16}"
