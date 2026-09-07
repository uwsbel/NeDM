#!/bin/bash
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -t 03:50:00
#SBATCH -o /work1/dannegrut/harry/nedm/logs/%x_%j.out
# Plan 31 step 2: the route tracker retrained inside a stall-aware imagination on the multi-arena data.
#   sbatch -J wp8_trk slurm/wp8_tracker.sbatch OUT DYNAMICS_CKPT [trainer args]
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
cd /work1/dannegrut/harry/nedm
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p "${MIOPEN_USER_DB_PATH}"
OUT=$1; CKPT=$2; shift 2
PYTHONPATH=src:$PYTHONPATH python3.12 -u scripts/traverse_wp3_train_tracker.py --out artifacts/traverse/${OUT} --dynamics-checkpoint ${CKPT} \
  --wp7-cache artifacts/traverse/wp7_cache_v1 --train-arenas arena_f101 arena_f102 arena_f103 arena_f104 --arena assets/traverse/arena_v1 "$@"
STATUS=$?
echo "exit: $STATUS"
rm -rf "${MIOPEN_USER_DB_PATH}"
exit $STATUS
