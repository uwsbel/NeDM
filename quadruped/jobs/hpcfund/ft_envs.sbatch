#!/bin/bash
# ENV SCALING: the best recipe (seed 7's 50-step surrogate, 2 s branches) with many more
# parallel rollouts per PPO iteration. The weight budget and the 10 epochs x 4 minibatches
# per iteration are unchanged, so the policy moves the same distance through the same
# number of optimiser steps; only how many rollouts each update averages over changes.
# Massive parallelism is the thing the surrogate can do and Chrono cannot.
# Each GPU: fine-tune, then score on straight walking (eval_h/) and on 40 held-out paths
# (eval_paths/), both paired against this partition's base arms.
#   sbatch --array=0-N --export=ALL,LIST=<"label seed steps branches model" lines>
# COST: <=8 h x 0.4 per node.
#SBATCH -J kyle-ftenvs
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -o /work1/dannegrut/kyle/logs/%x-%A_%a.out
#SBATCH -e /work1/dannegrut/kyle/logs/%x-%A_%a.err
set -uo pipefail
K=/work1/dannegrut/kyle
module purge
module load gnu12 rocm
source $K/nedm/env-scoring.sh
export PYTHONPATH=$K/qrun:$K/qrun/src:$PYTHONPATH
cd $K/qrun/quadruped
A=${SLURM_ARRAY_TASK_ID}
U=$K/qrun/assets/data/robot/go2_irrvis/urdf/go2_description.urdf
C2=$K/qrun/data/go2_crm_v2
mkdir -p $K/qrun/eval_h $K/qrun/eval_paths $K/qrun/ft_h $K/logs/eval_h $K/logs/eval_paths
echo "task $A  host $(hostname)  job ${SLURM_JOB_ID}  list $LIST  code $(cat $K/qrun/.source_commit)"
G=0
while read -r LABEL SEED STEPS BR MODEL; do
  [ -z "$LABEL" ] && continue
  ( export ROCR_VISIBLE_DEVICES=$G HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
    rm -rf $K/qrun/ft_h/$LABEL
    python -u finetune.py --model $MODEL --policy $K/qrun/policy/policy.pt --corpus $C2 \
      --out $K/qrun/ft_h/$LABEL --method ppo --device cuda --seed $SEED --steps $STEPS \
      --branches $BR > $K/logs/eval_h/ft_$LABEL.log 2>&1
    echo "FT $LABEL exit $? on GPU $G"
    if [ -f $K/qrun/ft_h/$LABEL/policy_ft.pt ]; then
      python -u evaluate.py --policy $K/qrun/ft_h/$LABEL/policy_ft.pt --urdf $U --corpus $C2 \
        --out $K/qrun/eval_h/$LABEL --label $LABEL --terrain crm --episodes 16 --seconds 6.0 \
        --vx 0.5 --spawn-spread 1.0 > $K/logs/eval_h/$LABEL.log 2>&1
      echo "EVAL straight $LABEL exit $?"
      python -u evaluate.py --policy $K/qrun/ft_h/$LABEL/policy_ft.pt --urdf $U --corpus $C2 \
        --out $K/qrun/eval_paths/$LABEL --label $LABEL --terrain crm --paths 4 --seconds 15 \
        --spawn-spread 1.0 > $K/logs/eval_paths/$LABEL.log 2>&1
      echo "EVAL paths $LABEL exit $?"
    fi ) &
  G=$((G + 1))
done < <(sed -n "$((4 * A + 1)),$((4 * A + 4))p" $LIST)
wait
echo "TASK_${A}_DONE"
