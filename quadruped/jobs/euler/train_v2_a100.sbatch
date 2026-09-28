#!/bin/bash
# v2 surrogates, seeds 2-5, one A100 each on euler19. The sbel partition, because it is
# the one tier that is not preempted: train.py writes its checkpoint only at the end, so
# a requeue on research/interactive would silently restart a run from zero.
# Code is the committed repo checkout; the corpus is a copy (md5-checked against hpcfund).
#SBATCH -J kyle-v2-surrogate
#SBATCH -p sbel
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH --array=2-5
#SBATCH -o /srv/home/kasha2/qrun/logs/%x-%A_%a.out
#SBATCH -e /srv/home/kasha2/qrun/logs/%x-%A_%a.err
set -uo pipefail
source ~/env.sh
S=${SLURM_ARRAY_TASK_ID}
R=/srv/home/kasha2/nedm/NeDM
cd $R/quadruped
echo "seed $S  host $(hostname)  job ${SLURM_JOB_ID}  code $(git -C $R rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader
~/venvs/nedm/bin/python -u train.py --corpus /srv/home/kasha2/qrun/data/go2_crm_v2 \
  --out /srv/home/kasha2/qrun/nnrom_v2e_s${S} --preset crm_baseline --epochs 80 \
  --steps-per-epoch 2000 --batch-size 64 --rollout-episodes 32 --rollout-horizon-s 10.0 \
  --select-window 5 --seed ${S} --device cuda
echo "SEED_${S}_DONE"
