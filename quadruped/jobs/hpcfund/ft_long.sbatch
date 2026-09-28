#!/bin/bash
# BUDGET TEST: is stopping at dw 4.0 right? The recipe (1 s rollout-trained surrogate, 2 s
# branches, 1024 envs, PPO seed 0) run well past the budget, saving the policy every EVERY
# iterations (snapshots/policy_it<N>.pt) and each time dw passes a mark (policy_dw<m>.pt). Up to the dw-4 mark each run is the recipe run
# already scored (431232), so its dw-4 snapshot doubles as a determinism check. Snapshots are
# scored separately (straight, 40 paths on hpcfund; rigid on sbel) as they appear.
#   sbatch --export=ALL,MODELS="m1 m2 m3 m4",ITERS=3000 ft_long.sbatch
# COST: ~14 h x 0.4 = ~5.6 node-hours for 4 runs.
#SBATCH -J kyle-ftlong
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -t 16:00:00
#SBATCH -o /work1/dannegrut/kyle/logs/%x-%j.out
#SBATCH -e /work1/dannegrut/kyle/logs/%x-%j.err
set -uo pipefail
K=/work1/dannegrut/kyle
module purge
module load gnu12 rocm
source $K/nedm/env-scoring.sh
export PYTHONPATH=$K/qrun:$K/qrun/src:$PYTHONPATH
cd $K/qrun/quadruped
ITERS=${ITERS:-3000}
MARKS=${MARKS:-"1 2 3 4 5 6 7 8 9 10"}
EVERY=${EVERY:-500}
TAG=${TAG:-long}
mkdir -p $K/qrun/ft_h $K/logs/eval_h
echo "host $(hostname) job ${SLURM_JOB_ID} iters $ITERS every $EVERY marks $MARKS code $(cat $K/qrun/.source_commit)"
G=0
for M in $MODELS; do
  L=${TAG}_$M
  ( export ROCR_VISIBLE_DEVICES=$G HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
    rm -rf $K/qrun/ft_h/$L
    python -u finetune.py --model $K/qrun/$M/best.pt --policy $K/qrun/policy/policy.pt \
      --corpus $K/qrun/data/go2_crm_v2 --out $K/qrun/ft_h/$L --method ppo --device cuda \
      --seed 0 --steps 100 --branches 1024 --iters $ITERS --target-dw 1e9 \
      --snapshot-every $EVERY --snapshot-dw $MARKS > $K/logs/eval_h/ft_$L.log 2>&1
    echo "FT $L exit $? on GPU $G $(date +%H:%M)" ) &
  G=$((G + 1))
done
wait
echo FTLONG_DONE
