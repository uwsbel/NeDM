#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:30:00
set -eo pipefail
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
C=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909
N=$C/nav_v1
export PYTHONPATH="$N/code:$N/source/src:${PYTHONPATH:-}"
python3.12 -u $N/code/nav_mask_datasets.py --files $C/sensor_v2/ds_v2_f104_train.npz \
  $C/sensor_v2/ds_v2_gen_f104.npz $C/sensor_v2/ds_v2_gen_g228.npz $C/sensor_v2/ds_v2_gen_g203.npz \
  $C/sensor_v2/ds_v2_gen_g217.npz $C/sensor_v2/ds_v2_gen_g216.npz $C/sensor_v2/ds_v2_gen_g231.npz \
  --out-dir $N/ds_r20 --radius-m 20
