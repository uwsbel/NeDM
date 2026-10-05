#!/bin/bash
#SBATCH -J pool-target
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 02:00:00
# GPU optimisation through the frozen NRD, then Chrono replay on the same node.
set -euo pipefail
set +u
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$POOL_CODE/src:${PYTHONPATH:-}"
cd "$POOL_CODE"
python3.12 -m nedm.pool_ball.optimize --checkpoint "$POOL_CHECKPOINT" --train-data "$POOL_DATA" --target-data "$POOL_TARGETS" --config "$POOL_TCONFIG" --output "$POOL_TRUN" ${POOL_TARGET_ARGS:-}
source "$POOL_CODE/scripts/pool_ball/cluster/chrono_env.sh"
"$NRD_PYTHON" -m nedm.pool_ball.verify --run "$POOL_TRUN" --workers 24 --branches
