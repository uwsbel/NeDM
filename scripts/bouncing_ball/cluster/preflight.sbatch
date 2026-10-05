#!/bin/bash
#SBATCH -J ball-preflight
#SBATCH -A dannegrut
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:15:00
set -euo pipefail
source "$BALL_CODE/scripts/bouncing_ball/cluster/collection_env.sh"
"$NRD_PYTHON" -m nedm.bouncing_ball.preflight "$BALL_ROOT/reference" --report "$BALL_ROOT/preflight.json"
