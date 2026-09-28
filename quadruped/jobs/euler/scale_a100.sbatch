#!/bin/bash
# DOES MORE DATA, OR A BIGGER SURROGATE, IMPROVE TRANSFER? The old pipeline asked this of
# one-step accuracy; this asks it of what the study actually cares about, the Chrono result
# after fine-tuning. Both stages of a surrogate run here: the one-step fit, then the 1 s
# rollout fine-tune that makes long branches possible. The 100% / 6x256 cell already exists
# (seeds 8 and 9), so only the other cells are trained.
#   sbatch --array=0-N --export=ALL,LIST=<"label layers width heads fraction subset seed" lines>
#SBATCH -J kyle-scale
#SBATCH -p sbel
#SBATCH --gres=gpu:a100:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o /srv/home/kasha2/qrun/logs/%x-%A_%a.out
#SBATCH -e /srv/home/kasha2/qrun/logs/%x-%A_%a.err
set -uo pipefail
source ~/env.sh
Q=/srv/home/kasha2/qrun
R=/srv/home/kasha2/nedm/NeDM
cd $R/quadruped
read -r LABEL NL NE NH FRAC SUB SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $LIST)
echo "host $(hostname) job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} $LABEL layers $NL width $NE heads $NH fraction $FRAC subset $SUB seed $SEED code $(git -C $R rev-parse --short HEAD)"
P=~/venvs/nedm/bin/python
# Stage 1: one-step. Same optimiser budget for every cell, so data and capacity vary and
# the amount of training does not.
$P -u train.py --corpus $Q/data/go2_crm_v2 --out $Q/nnrom_$LABEL \
  --n-layer $NL --n-embd $NE --n-head $NH --train-fraction $FRAC --subset-seed $SUB \
  --seed $SEED --device cuda
echo "STAGE1 $LABEL exit $?"
# Stage 2: 1 s rollouts, from that model.
$P -u train.py --corpus $Q/data/go2_crm_v2 --out $Q/nnrom_${LABEL}_ms100 \
  --n-layer $NL --n-embd $NE --n-head $NH --train-fraction $FRAC --subset-seed $SUB \
  --seed $SEED --device cuda --init-from $Q/nnrom_$LABEL/best.pt \
  --rollout-loss-steps 100 --rollout-loss-batch 4 --epochs 8 --steps-per-epoch 1000 \
  --lr 1e-4 --min-lr 1e-5 --warmup-steps 200 --select-window 3
echo "STAGE2 $LABEL exit $?"
echo SCALE_DONE_$LABEL
