#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -t 03:00:00
set -eo pipefail
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
export MIOPEN_USER_DB_PATH=/tmp/miopen_${SLURM_JOB_ID} MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${SLURM_JOB_ID}
mkdir -p /tmp/miopen_${SLURM_JOB_ID}
C=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909
N=$C/nav_v1
export PYTHONPATH="$N/code:$N/source/src:${PYTHONPATH:-}"
cd $N/code
python3.12 -u sensor_train_v2.py --files $N/ds_r20/ds_v2_f104_train.npz $N/ds_r20/ds_v2_gen_f104.npz \
  $N/ds_r20/ds_v2_gen_g228.npz $N/ds_r20/ds_v2_gen_g203.npz $N/ds_r20/ds_v2_gen_g217.npz \
  $N/ds_r20/ds_v2_gen_g216.npz $N/ds_r20/ds_v2_gen_g231.npz \
  --out $N/matched_r20 --variants Dabs,H --seeds 3 --tag r20 --save
