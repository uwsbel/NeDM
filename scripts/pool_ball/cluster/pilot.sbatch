#!/bin/bash
#SBATCH -J pool-pilot
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 00:45:00
set -euo pipefail
source "$POOL_CODE/scripts/pool_ball/cluster/chrono_env.sh"
"$NRD_PYTHON" -m nedm.pool_ball.pilot --output-dir "$POOL_OUT" --workers 24 --stage "${POOL_STAGE:-all}" "$@"
