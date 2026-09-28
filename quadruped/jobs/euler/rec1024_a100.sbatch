#!/bin/bash
# RECIPE REPLICATION: PPO with 2 s branches and 1024 parallel rollouts in each of euler's four
# 1 s rollout-trained surrogates (seeds 2-5). Fine-tune only; the policies are scored on
# hpcfund against the same base arms as every other result (paired_eval refuses mixed
# Chrono builds). Committed code.
#   sbatch --array=0-3 rec1024_a100.sbatch
#SBATCH -J kyle-rec1024
#SBATCH -p sbel
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=14G
#SBATCH -t 06:00:00
#SBATCH -o /srv/home/kasha2/qrun/logs/%x-%A_%a.out
#SBATCH -e /srv/home/kasha2/qrun/logs/%x-%A_%a.err
set -uo pipefail
source ~/env.sh
Q=/srv/home/kasha2/qrun
R=/srv/home/kasha2/nedm/NeDM
S=$((SLURM_ARRAY_TASK_ID + 2))
M=nnrom_v2e_s${S}_ms100
cd $R/quadruped
echo "host $(hostname) job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} model $M code $(git -C $R rev-parse --short HEAD)"
~/venvs/nedm/bin/python -u finetune.py --model $Q/$M/best.pt --policy $Q/policy/policy.pt \
  --corpus $Q/data/go2_crm_v2 --out $Q/ft_rec1024_$M --method ppo --device cuda --seed 0 \
  --steps 100 --branches 1024
echo "REC1024_${M}_DONE exit $?"
