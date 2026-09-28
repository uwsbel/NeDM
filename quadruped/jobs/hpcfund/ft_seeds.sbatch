#!/bin/bash
# IS IT THE SURROGATE OR THE RUN? Seed 6's surrogate collapsed under PPO seed 0 (OOD spikes
# from the first hundred iterations, value loss 480,000 by iteration 1150) while three other
# surrogates ran clean past dw 6.8 on the same corpus and settings. This runs more PPO seeds
# in seed 6's surrogate and in seed 8's as a control, with the same checkpoints, so the
# instability can be attributed to the model or to the run.
#   sbatch --array=0-N --export=ALL,LIST=<file of "label model seed" lines>,ITERS=1200
#SBATCH -J kyle-ftseeds
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
ITERS=${ITERS:-1200}
mkdir -p $K/qrun/ft_h $K/logs/eval_h
echo "task $A host $(hostname) job ${SLURM_JOB_ID} iters $ITERS list $LIST code $(cat $K/qrun/.source_commit)"
G=0
while read -r LABEL MODEL SEED; do
  [ -z "$LABEL" ] && continue
  ( export ROCR_VISIBLE_DEVICES=$G HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
    rm -rf $K/qrun/ft_h/$LABEL
    python -u finetune.py --model $K/qrun/$MODEL/best.pt --policy $K/qrun/policy/policy.pt \
      --corpus $K/qrun/data/go2_crm_v2 --out $K/qrun/ft_h/$LABEL --method ppo --device cuda \
      --seed $SEED --steps 100 --branches 1024 --iters $ITERS --target-dw 1e9 \
      --snapshot-every 500 --snapshot-dw 4 5 6 7 > $K/logs/eval_h/ft_$LABEL.log 2>&1
    echo "FT $LABEL exit $? on GPU $G $(date +%H:%M)" ) &
  G=$((G + 1))
done < <(sed -n "$((4 * A + 1)),$((4 * A + 4))p" $LIST)
wait
echo FTSEEDS_DONE
