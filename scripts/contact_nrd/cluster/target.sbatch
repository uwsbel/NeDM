#!/bin/bash
#SBATCH -J cn-target
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 03:00:00
# Gradient targeting through the frozen unified NRD, then Chrono replay on the same node.
set -euo pipefail
set +u
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$CN_CODE/src:${PYTHONPATH:-}"
cd "$CN_CODE"
python3.12 -m nedm.contact_nrd.optimize --checkpoint "$CN_CHECKPOINT" --train-data "$CN_TRAIN" --target-data "$CN_TARGETS" --config "$CN_TCONFIG" --output "$CN_TRUN" ${CN_TARGET_ARGS:-}
if [[ "$CN_SYSTEM" == pool ]]; then
  export POOL_CODE="$CN_CODE"; source "$CN_CODE/scripts/pool_ball/cluster/chrono_env.sh"
  "$NRD_PYTHON" -m nedm.pool_ball.verify --run "$CN_TRUN" --workers 24 --branches
else
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" PYTHONPATH="${PYTHONPATH:-}" BALL_CODE="$CN_CODE"
  source "$CN_CODE/scripts/bouncing_ball/cluster/collection_env.sh"
  "$NRD_PYTHON" -m nedm.contact_nrd.ball_verify --run "$CN_TRUN" --workers 24
fi
