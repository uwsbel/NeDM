#!/bin/bash
# MORE DATA: four surrogates on the extended corpus (go2_crm_v23, 2,094 training episodes
# against go2_crm_v2's 920, same 230 held-out episodes), both stages, with the SAME
# optimiser budget the v2 surrogates had -- so what differs is how much data they saw and
# not how much training they got. Scored later by fine-tuning the recipe in each.
#   sbatch --array=0 --export=ALL,SEEDS="40 41 42 43"
#SBATCH -J kyle-v23train
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -t 09:00:00
#SBATCH -o /work1/dannegrut/kyle/logs/%x-%j.out
#SBATCH -e /work1/dannegrut/kyle/logs/%x-%j.err
set -uo pipefail
K=/work1/dannegrut/kyle
module purge
module load gnu12 rocm
source $K/nedm/env-scoring.sh
export PYTHONPATH=$K/qrun:$K/qrun/src:$PYTHONPATH
cd $K/qrun/quadruped
SEEDS=${SEEDS:-"40 41 42 43"}
echo "host $(hostname) job ${SLURM_JOB_ID} seeds $SEEDS corpus go2_crm_v23 code $(cat $K/qrun/.source_commit)"
G=0
for SEED in $SEEDS; do
  ( export ROCR_VISIBLE_DEVICES=$G HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
    L=nnrom_v23_s$SEED
    python -u train.py --corpus $K/qrun/data/go2_crm_v23 --out $K/qrun/$L --seed $SEED \
      --device cuda > $K/logs/train_$L.log 2>&1
    echo "STAGE1 $L exit $? on GPU $G $(date +%H:%M)"
    python -u train.py --corpus $K/qrun/data/go2_crm_v23 --out $K/qrun/${L}_ms100 --seed $SEED \
      --device cuda --init-from $K/qrun/$L/best.pt --rollout-loss-steps 100 \
      --rollout-loss-batch 4 --epochs 8 --steps-per-epoch 1000 --lr 1e-4 --min-lr 1e-5 \
      --warmup-steps 200 --select-window 3 > $K/logs/train_${L}_ms100.log 2>&1
    echo "STAGE2 $L exit $? on GPU $G $(date +%H:%M)" ) &
  G=$((G + 1))
done
wait
echo V23TRAIN_DONE
