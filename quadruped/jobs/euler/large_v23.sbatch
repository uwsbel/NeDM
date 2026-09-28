#!/bin/bash
# The untested scaling cell: a 12x512 surrogate on the DOUBLED corpus (go2_crm_v23: v2 plus
# 1,174 training-only episodes, same held-out set). Recipe's two stages. Tasks 0-1 train
# (seeds 90, 91); task 2 waits for them and sweeps rollout accuracy of all four cells
# (large/standard x v2/v23) on v2's held-out episodes, which v23 shares.
#   sbatch -p sbel --gres=gpu:a100:1 --array=0-2 large_v23.sbatch
#SBATCH -J kyle-largev23
#SBATCH -c 6
#SBATCH --mem=40G
#SBATCH -t 2-00:00:00
#SBATCH -o /srv/home/kasha2/qrun/logs/%x-%A_%a.out
#SBATCH -e /srv/home/kasha2/qrun/logs/%x-%A_%a.err
set -uo pipefail
source ~/env.sh
Q=/srv/home/kasha2/qrun
R=/srv/home/kasha2/nedm/NeDM
PY=~/venvs/nedm/bin/python
cd $R/quadruped
i=$SLURM_ARRAY_TASK_ID
echo "task $i host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) code $(git -C $R rev-parse --short HEAD) $(date +%T)"
if [ $i -lt 2 ]; then
  S=$((90 + i)); L=nnrom_capL_v23_s$S
  LOCAL=${TMPDIR:-/tmp}/kasha2_lv23_$SLURM_JOB_ID; mkdir -p $LOCAL && cp -r $Q/data/go2_crm_v23 $LOCAL/ || exit 1
  C=$LOCAL/go2_crm_v23
  if [ ! -f $Q/$L/.done ]; then
    $PY -u train.py --corpus $C --out $Q/$L --seed $S --device cuda --n-layer 12 --n-embd 512 --n-head 8 || exit 11
    touch $Q/$L/.done; fi
  echo "STAGE1 $L done $(date +%T)"
  if [ ! -f $Q/${L}_ms100/.done ]; then
    $PY -u train.py --corpus $C --out $Q/${L}_ms100 --seed $S --device cuda --n-layer 12 --n-embd 512 --n-head 8 \
      --init-from $Q/$L/best.pt --rollout-loss-steps 100 --rollout-loss-batch 4 --epochs 8 \
      --steps-per-epoch 1000 --lr 1e-4 --min-lr 1e-5 --warmup-steps 200 --select-window 3 || exit 12
    touch $Q/${L}_ms100/.done; fi
  echo "LV23_${L}_DONE $(date +%T)"; exit 0
fi
until [ -f $Q/nnrom_capL_v23_s90_ms100/.done ] && [ -f $Q/nnrom_capL_v23_s91_ms100/.done ]; do sleep 600; done
C=$Q/data/go2_crm_v2
for m in nnrom_capL_v23_s90_ms100 nnrom_capL_v23_s91_ms100 nnrom_capL_a_ms100 nnrom_capL_b_ms100 \
         nnrom_v23_s40_ms100 nnrom_v23_s41_ms100 nnrom_v23_s42_ms100 nnrom_v23_s43_ms100 \
         nnrom_v2e_s2_ms100 nnrom_v2e_s3_ms100 nnrom_v2e_s4_ms100 nnrom_v2e_s5_ms100; do
  [ -f $Q/$m/best.pt ] || { echo "missing $m"; continue; }
  echo "=== SWEEP $m"; $PY -u diagnostics/horizon_sweep.py --model $Q/$m/best.pt --corpus $C --device cuda 2>&1 | grep -E "^\s+[0-9.]+s"
done
echo LV23_SWEEP_DONE
