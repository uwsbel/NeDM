#!/bin/bash
# The recipe at the new stop, in each scaling-study surrogate, so data and capacity are
# judged on TRANSFER and not on one-step accuracy. Branch starts and the OOD reference come
# from the full corpus in every arm, so the only thing that differs is the surrogate.
#   sbatch --array=0-N --export=ALL,LIST=<"label model" lines>,ITERS=1000
#SBATCH -J kyle-ftscale
#SBATCH -p sbel
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 10:00:00
#SBATCH -o /srv/home/kasha2/qrun/logs/%x-%A_%a.out
#SBATCH -e /srv/home/kasha2/qrun/logs/%x-%A_%a.err
set -uo pipefail
source ~/env.sh
Q=/srv/home/kasha2/qrun
R=/srv/home/kasha2/nedm/NeDM
cd $R/quadruped
read -r LABEL MODEL < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $LIST)
echo "host $(hostname) job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} $LABEL model $MODEL code $(git -C $R rev-parse --short HEAD)"
rm -rf $Q/ft_scale_$LABEL
~/venvs/nedm/bin/python -u finetune.py --model $Q/$MODEL/best.pt --policy $Q/policy/policy.pt \
  --corpus $Q/data/go2_crm_v2 --out $Q/ft_scale_$LABEL --method ppo --device cuda --seed 0 \
  --steps 100 --branches 1024 --iters ${ITERS:-1000} --target-dw 1e9 --snapshot-every 500
echo "FTSCALE_${LABEL}_DONE exit $?"
