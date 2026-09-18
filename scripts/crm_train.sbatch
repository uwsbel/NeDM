#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 02:00:00
# submit with: -p mi3501x --export=ALL,CRM_DS=<npz>,CRM_TRAIN_OUT=<dir>,CRM_MODE=deploy|holdout,CRM_SEEDS=5,CRM_TAG=CRM_N2
set -eo pipefail
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID} MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
mkdir -p /tmp/miopen_${SLURM_JOB_ID}
CRM=/work1/dannegrut/harry/experiments/crm_f104_20260916
cd $CRM/source/scripts
python3.12 -u crm_train.py --ds "$CRM_DS" --out "$CRM_TRAIN_OUT" --mode ${CRM_MODE:-deploy} --seeds ${CRM_SEEDS:-5} --tag ${CRM_TAG:-CRM_N2} ${CRM_EXTRA:-}
