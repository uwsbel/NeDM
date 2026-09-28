#!/bin/bash
# 100-step multi-step fine-tune of the four euler surrogates (seeds 2-5), one A100 each on the
# non-preemptible sbel partition. Same recipe as hpcfund job 430610, which took a surrogate
# from usable to ~2 s to usable across all 10 s. Committed code (523fa4d3).
#SBATCH -J kyle-v2-msft
#SBATCH -p sbel
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=14G
#SBATCH -t 05:00:00
#SBATCH --array=2-4
#SBATCH -o /srv/home/kasha2/qrun/logs/%x-%A_%a.out
#SBATCH -e /srv/home/kasha2/qrun/logs/%x-%A_%a.err
set -uo pipefail
source ~/env.sh
S=${SLURM_ARRAY_TASK_ID}
R=/srv/home/kasha2/nedm/NeDM
cd $R/quadruped
echo "seed $S  host $(hostname)  job ${SLURM_JOB_ID}  code $(git -C $R rev-parse --short HEAD)"
~/venvs/nedm/bin/python -u train.py --corpus /srv/home/kasha2/qrun/data/go2_crm_v2 \
  --out /srv/home/kasha2/qrun/nnrom_v2e_s${S}_ms100 --preset crm_baseline \
  --epochs 8 --steps-per-epoch 1000 --batch-size 64 --lr 1e-4 --min-lr 1e-5 --warmup-steps 200 \
  --rollout-episodes 32 --rollout-horizon-s 10.0 --select-window 3 --seed 0 --device cuda \
  --init-from /srv/home/kasha2/qrun/nnrom_v2e_s${S}/best.pt --rollout-loss-steps 100 --rollout-loss-batch 4
echo "SEED_${S}_MS100_DONE"
