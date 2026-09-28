#!/bin/bash
# 2048-env PPO, seed 1, in seed 7's 0.5 s rollout-trained surrogate at 2 s branches. Moved here
# from north (parked idle) after d33 proved ~3 min per iteration; whichever of d33 and this
# finishes first is kept. Committed code; evaluation happens on hpcfund.
#SBATCH -J kyle-env2048
#SBATCH -p sbel
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=14G
#SBATCH -t 14:00:00
#SBATCH -o /srv/home/kasha2/qrun/logs/%x-%j.out
#SBATCH -e /srv/home/kasha2/qrun/logs/%x-%j.err
set -uo pipefail
source ~/env.sh
R=/srv/home/kasha2/nedm/NeDM
cd $R/quadruped
echo "host $(hostname) job ${SLURM_JOB_ID} code $(git -C $R rev-parse --short HEAD)"
~/venvs/nedm/bin/python -u finetune.py --model /srv/home/kasha2/qrun/nnrom_v2h_s7_ms50/best.pt \
  --policy /srv/home/kasha2/qrun/policy/policy.pt --corpus /srv/home/kasha2/qrun/data/go2_crm_v2 \
  --out /srv/home/kasha2/qrun/ft_env2048_s1 --method ppo --device cuda --seed 1 --steps 100 --branches 2048
echo ENV2048_S1_DONE
