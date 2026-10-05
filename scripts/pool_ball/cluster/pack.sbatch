#!/bin/bash
#SBATCH -J pool-pack
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 01:00:00
set -euo pipefail
source "$POOL_CODE/scripts/pool_ball/cluster/chrono_env.sh"
"$NRD_PYTHON" -m nedm.pool_ball.campaign --campaign "$POOL_CAMPAIGN" --output-dir "$POOL_DATA" --pack-only
"$NRD_PYTHON" -m nedm.pool_ball.data_report --data "$POOL_DATA" || echo "data report failed"
