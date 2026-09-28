#!/bin/bash
# 100-STEP multi-step fine-tuning of the remaining one-step surrogates. 100-step made the
# off mi3001x, where the scheduler put the start 11 h out. Same recipe as job 430610, whose
# 50-step fine-tune took seed 7 from usable to ~2 s to usable across all 10 s.
# Members for the next ensemble (50-step on five more surrogates), a second seed of the
# seed-7 50-step result (does it reproduce?), and 100-step runs on seed 7 and north s1.
# seed-1 surrogate trustworthy across 10 s (0.572) where 50-step failed past ~3 s.
# COST: 1 node x ~2.5 h x 0.4 = ~1 node-hour.
#SBATCH -J kyle-v2-ms100
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -t 05:00:00
#SBATCH --array=0
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
#        base            steps rb seed
SPECS=( "nnrom_v2h_s6   100 4 0"
        "nnrom_v2h_s8   100 4 0"
        "nnrom_v2h_s9   100 4 0"
        "nnrom_v2n      100 4 0" )
echo "task $A  host $(hostname)  job ${SLURM_JOB_ID}  code $(cat $K/qrun/.source_commit)"
for G in 0 1 2 3; do
  read -r BASE KS RB SD <<< "${SPECS[$((4 * A + G))]}"
  NAME=${BASE}_ms${KS}_s${SD}
  ( export ROCR_VISIBLE_DEVICES=$G HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
    python -u train.py --corpus $K/qrun/data/go2_crm_v2 --out $K/qrun/$NAME --preset crm_baseline \
      --epochs 8 --steps-per-epoch 1000 --batch-size 64 --lr 1e-4 --min-lr 1e-5 --warmup-steps 200 \
      --rollout-episodes 32 --rollout-horizon-s 10.0 --select-window 3 --seed $SD --device cuda \
      --init-from $K/qrun/$BASE/best.pt --rollout-loss-steps $KS --rollout-loss-batch $RB \
      > $K/logs/msft210_$NAME.log 2>&1
    echo "TRAIN $NAME exit $? on GPU $G" ) &
done
wait
echo "TASK_${A}_DONE"
