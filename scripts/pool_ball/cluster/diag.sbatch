#!/bin/bash
#SBATCH -J pool-diag
#SBATCH -A dannegrut
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 00:30:00
set -euo pipefail
set +u
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
set -u
export PYTHONPATH="$POOL_CODE/src:${PYTHONPATH:-}"
cd "$POOL_CODE"
python3.12 -m nedm.pool_ball.diagnose --checkpoint "$CK" --data "$POOL_DATA" --episodes 300
