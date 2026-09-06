#!/bin/bash
#SBATCH -p mi3501x
#SBATCH -N 1
#SBATCH -t 03:30:00
#SBATCH -o /work1/dannegrut/harry/nedm/logs/%x_%j.out
# Plan 28 step 2: dynamics model on the multi-arena collection (split by arena). Usage:
#   sbatch -J wp7_ft slurm/wp7_finetune.sbatch OUT [extra trainer args]
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
cd /work1/dannegrut/harry/nedm
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p "${MIOPEN_USER_DB_PATH}"
OUT=$1; shift 1
PYTHONPATH=src:$PYTHONPATH python3.12 scripts/traverse_wp2_train_map.py \
  --caches artifacts/traverse/wp7_cache_v1 --split-by arena --val-arenas arena_f105 \
  --map-mode index --map-key map_v2 --arena assets/traverse/arena_v1 \
  --out artifacts/traverse/${OUT} \
  --batch 256 --eval-every 1000 --eval-episodes 256 --horizons 10 20 40 100 --selection z1_mae_norm "$@"
STATUS=$?
echo "exit: $STATUS"
rm -rf "${MIOPEN_USER_DB_PATH}"
exit $STATUS
