#!/bin/bash
# Rigid-data control fine-tunes, moved from the desktops to euler for speed (2026-09-25).
# The recipe exactly (ft_data.sbatch): 1024 branches x 100 steps, iteration 1000,
# snapshots every 500. Surrogates and corpus were built on sbel (build 3b0bd530); PPO is
# torch-only, so where it runs does not affect pairing. Scoring stays on sbel.
#   sbatch --array=0-2 ft_rigid_h100.sbatch      (seeds 60, 61, 62)
# research partition: the H100 nodes (euler29-30) are not in the sbel partition, so no
# higher tier can preempt them.
#SBATCH -J kyle-ftrigid
#SBATCH -p research
#SBATCH --gres=gpu:h100:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH -o /srv/home/kasha2/qrun/logs/%x-%A_%a.out
#SBATCH -e /srv/home/kasha2/qrun/logs/%x-%A_%a.err
set -uo pipefail
source ~/env.sh
Q=/srv/home/kasha2/qrun
R=/srv/home/kasha2/nedm/NeDM
SEEDS=(60 61 62)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
cd $R/quadruped
echo "host $(hostname) job ${SLURM_JOB_ID} seed $S gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) code $(git -C $R rev-parse --short HEAD)"
~/venvs/nedm/bin/python -u finetune.py --model $Q/nnrom_rig_s${S}_ms100/best.pt --policy $Q/policy/policy.pt \
  --corpus $Q/data/go2_rigid_v2 --out $Q/ft_rig_s$S --method ppo --device cuda --seed 0 \
  --steps 100 --branches 1024 --iters 1000 --target-dw 1e9 --snapshot-every 500
echo "FTRIGID_${S}_DONE exit $?"
