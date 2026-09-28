#!/bin/bash
# v2 surrogates, seeds 6-9, one MI300X each (mi3001x, 0.125x). train.py smoke-tested on
# this partition as job 430502 (22 s for a 200-step epoch with the batched rollout).
# COST: ~2.5 h x 0.125 x 4 = ~1.25 node-hours; -t caps it at 2.0.
#SBATCH -J kyle-v2-surrogate
#SBATCH -p mi3001x
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH --array=6-9
#SBATCH -o /work1/dannegrut/kyle/logs/%x-%A_%a.out
#SBATCH -e /work1/dannegrut/kyle/logs/%x-%A_%a.err
set -uo pipefail
K=/work1/dannegrut/kyle
module purge
module load gnu12 rocm
source $K/nedm/env-scoring.sh
export PYTHONPATH=$K/qrun:$K/qrun/src:$PYTHONPATH
cd $K/qrun/quadruped
S=${SLURM_ARRAY_TASK_ID}
echo "seed $S  host $(hostname)  job ${SLURM_JOB_ID}  code $(cat $K/qrun/.source_commit)"
python -u train.py --corpus $K/qrun/data/go2_crm_v2 --out $K/qrun/nnrom_v2h_s${S} --preset crm_baseline \
  --epochs 80 --steps-per-epoch 2000 --batch-size 64 --rollout-episodes 32 --rollout-horizon-s 10.0 \
  --select-window 5 --seed ${S} --device cuda
echo "SEED_${S}_DONE"
