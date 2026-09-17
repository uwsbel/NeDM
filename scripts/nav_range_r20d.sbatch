#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:00:00
set -eo pipefail
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
export MIOPEN_USER_DB_PATH=/tmp/miopen_$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH
C=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909
N=$C/nav_v1
export PYTHONPATH="$N/code:$N/source/src:${PYTHONPATH:-}"
export GEN_SRC=$N/source/src
python3.12 -u $N/code/nav_eval_range.py --files $N/ds_r20/ds_v2_gen_f104.npz $N/ds_r20/ds_v2_gen_g228.npz $N/ds_r20/ds_v2_gen_g203.npz $N/ds_r20/ds_v2_gen_g217.npz $N/ds_r20/ds_v2_gen_g216.npz $N/ds_r20/ds_v2_gen_g231.npz \
  --ck-dir $N/matched_r20 --variants Dabs --tag r20 --radii 1000000000 --out $N/r20_choice.json
