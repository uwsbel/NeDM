#!/bin/bash
#SBATCH -J cn-pool-collect
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 02:00:00
# Pool shots with the pool study's collector. Array task k collects shards [k*POOL_PER, (k+1)*POOL_PER);
# Without an array it collects every shard and packs; POOL_PACK=1 only packs.
set -euo pipefail
source "$POOL_CODE/scripts/pool_ball/cluster/chrono_env.sh"
workers=${SLURM_CPUS_PER_TASK:-16}
if [[ "${POOL_PACK:-0}" == 1 ]]; then
  "$NRD_PYTHON" -m nedm.pool_ball.campaign --campaign "$POOL_CAMPAIGN" --output-dir "$POOL_DATA" --pack-only
elif [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  "$NRD_PYTHON" -m nedm.pool_ball.campaign --campaign "$POOL_CAMPAIGN" --output-dir "$POOL_DATA" --workers "$workers"
else
  first=$(( SLURM_ARRAY_TASK_ID * POOL_PER ))
  "$NRD_PYTHON" -m nedm.pool_ball.campaign --campaign "$POOL_CAMPAIGN" --output-dir "$POOL_DATA" --workers "$workers" --shard-range $first $(( first + POOL_PER ))
fi
