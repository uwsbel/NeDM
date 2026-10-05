#!/bin/bash
#SBATCH -J pool-collect
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 02:00:00
# Array task k collects shards [k*PER, (k+1)*PER); POOL_PER shards per task.
set -euo pipefail
source "$POOL_CODE/scripts/pool_ball/cluster/chrono_env.sh"
first=$(( SLURM_ARRAY_TASK_ID * POOL_PER ))
"$NRD_PYTHON" -m nedm.pool_ball.campaign --campaign "$POOL_CAMPAIGN" --output-dir "$POOL_DATA" --workers 24 --shard-range $first $(( first + POOL_PER ))
