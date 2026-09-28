#!/bin/bash
# DOES MORE DATA HELP TRANSFER? The recipe at the new stop (1000 iterations, 1024 rollouts,
# 2 s branches) in the surrogates trained on the extended corpus. Two arms take their branch
# starts and OOD reference from the extended corpus and two from the original, so a change
# in the SURROGATE is separable from a change in the branch pool.
#   sbatch --array=0-N --export=ALL,LIST=<"label model corpus" lines>,ITERS=1000
#SBATCH -J kyle-ftdata
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
ITERS=${ITERS:-1000}
mkdir -p $K/qrun/ft_h $K/logs/eval_h
echo "task $A host $(hostname) job ${SLURM_JOB_ID} iters $ITERS list $LIST code $(cat $K/qrun/.source_commit)"
G=0
while read -r LABEL MODEL CORPUS; do
  [ -z "$LABEL" ] && continue
  ( export ROCR_VISIBLE_DEVICES=$G HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
    rm -rf $K/qrun/ft_h/$LABEL
    python -u finetune.py --model $K/qrun/$MODEL/best.pt --policy $K/qrun/policy/policy.pt \
      --corpus $K/qrun/data/$CORPUS --out $K/qrun/ft_h/$LABEL --method ppo --device cuda \
      --seed 0 --steps 100 --branches 1024 --iters $ITERS --target-dw 1e9 \
      --snapshot-every 500 > $K/logs/eval_h/ft_$LABEL.log 2>&1
    echo "FT $LABEL exit $? on GPU $G $(date +%H:%M)" ) &
  G=$((G + 1))
done < <(sed -n "$((4 * A + 1)),$((4 * A + 4))p" $LIST)
wait
echo FTDATA_DONE
