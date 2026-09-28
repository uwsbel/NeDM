#!/bin/bash
# Does a ONE-stage surrogate transfer like a two-stage one? The recipe's PPO fine-tune
# (ft_data.sbatch settings, iteration 1000) in each one-stage surrogate from 70170.
#   sbatch -p research --gres=gpu:h100:1 --array=0-2 ft_onestage.sbatch            (16k)
#   sbatch -p research --gres=gpu:h100:1 --array=3-4 --dependency=... ft_onestage.sbatch (40k)
#SBATCH -J kyle-ftone
#SBATCH -c 6
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -o /srv/home/kasha2/qrun/logs/%x-%A_%a.out
#SBATCH -e /srv/home/kasha2/qrun/logs/%x-%A_%a.err
set -uo pipefail
source ~/env.sh
Q=/srv/home/kasha2/qrun
R=/srv/home/kasha2/nedm/NeDM
LABELS=(nnrom_one16k_s80 nnrom_one16k_s81 nnrom_one16k_s82 nnrom_one40k_s80 nnrom_one40k_s81)
L=${LABELS[$SLURM_ARRAY_TASK_ID]}
LOCAL=${TMPDIR:-/tmp}/kasha2_ftone_$SLURM_JOB_ID
mkdir -p $LOCAL && cp -r $Q/data/go2_crm_v2 $LOCAL/ || { echo "corpus copy failed"; exit 1; }
cd $R/quadruped
echo "$L host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) code $(git -C $R rev-parse --short HEAD) $(date +%T)"
~/venvs/nedm/bin/python -u finetune.py --model $Q/$L/best.pt --policy $Q/policy/policy.pt \
  --corpus $LOCAL/go2_crm_v2 --out $Q/ft_$L --method ppo --device cuda --seed 0 \
  --steps 100 --branches 1024 --iters 1000 --target-dw 1e9 --snapshot-every 500
echo "FTONE_${L}_DONE exit $?"
