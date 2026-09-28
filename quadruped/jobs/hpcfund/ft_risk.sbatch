#!/bin/bash
# INDUCED FAILURES, to calibrate the guard on more than one collapse. The OOD penalty is
# documented way this pipeline fails. The guard RECORDS but does not stop them (--guard-spike
# 1.0), so thresholds can be replayed offline against failures that actually happened.
#   sbatch --array=0-N --export=ALL,LIST=<file of "label model ood" lines>,ITERS=800


#SBATCH -J kyle-ftrisk
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -t 06:00:00
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
ITERS=${ITERS:-800}
mkdir -p $K/qrun/ft_h $K/logs/eval_h
echo "task $A host $(hostname) job ${SLURM_JOB_ID} iters $ITERS list $LIST code $(cat $K/qrun/.source_commit)"
G=0
while read -r LABEL MODEL OOD; do
  [ -z "$LABEL" ] && continue
  ( export ROCR_VISIBLE_DEVICES=$G HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
    rm -rf $K/qrun/ft_h/$LABEL
    python -u finetune.py --model $K/qrun/$MODEL/best.pt --policy $K/qrun/policy/policy.pt \
      --corpus $K/qrun/data/go2_crm_v2 --out $K/qrun/ft_h/$LABEL --method ppo --device cuda \
      --seed 0 --steps 100 --branches 1024 --iters $ITERS --target-dw 1e9 \
      --ood-penalty $OOD --guard-spike 1.0 --monitor-ood \
      --snapshot-every 500 --snapshot-dw 4 5 6 7 > $K/logs/eval_h/ft_$LABEL.log 2>&1
    echo "FT $LABEL exit $? on GPU $G $(date +%H:%M)" ) &
  G=$((G + 1))
done < <(sed -n "$((4 * A + 1)),$((4 * A + 4))p" $LIST)
wait
echo FTSEEDS_DONE
