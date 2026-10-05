#!/bin/bash
#SBATCH -J ball-collect
#SBATCH -A dannegrut
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -t 01:00:00
set -euo pipefail
source "$BALL_CODE/scripts/bouncing_ball/cluster/collection_env.sh"
"$NRD_PYTHON" -m nedm.bouncing_ball.campaign --campaign "${BALL_CAMPAIGN:-configs/bouncing_ball/campaign_v1.json}" --output-dir "${BALL_DATA:-$BALL_ROOT/data}" --workers 64
