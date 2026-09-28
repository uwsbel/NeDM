#!/bin/bash
# BUDGET TEST, replication on euler: the recipe (1 s rollout-trained surrogate, 2 s branches,
# 1024 envs, PPO seed 0) in euler's four 1 s surrogates (seeds 2-5), run to 3000 iterations
# with a checkpoint every 500 iterations and at dw 1-10. Up to dw 4 each run repeats the
# recipe run already scored (66715). Checkpoints are scored on hpcfund/sbel if needed.
#   sbatch --array=0-3 long_a100.sbatch
#SBATCH -J kyle-long
#SBATCH -p sbel
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=14G
#SBATCH -t 22:00:00
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
rm -rf $Q/ft_long_$M
~/venvs/nedm/bin/python -u finetune.py --model $Q/$M/best.pt --policy $Q/policy/policy.pt \
  --corpus $Q/data/go2_crm_v2 --out $Q/ft_long_$M --method ppo --device cuda --seed 0 \
  --steps 100 --branches 1024 --iters 3000 --target-dw 1e9 --snapshot-every 500 \
  --snapshot-dw 1 2 3 4 5 6 7 8 9 10
echo "LONG_${M}_DONE exit $?"
