#!/bin/bash
# Is the two-stage surrogate training necessary? One-stage = one-step + 1 s rollout loss
# together FROM SCRATCH (no --init-from), recipe corpus and preset. Tasks 0-2: 16k steps
# (2x the rollout stage), tasks 3-4: 40k steps. Task 5: horizon sweep of the reference
# two-stage surrogates (euler seeds 2-5) and their one-step parents on the same episodes.
#   sbatch -p research --array=0-5 --gres=gpu:h100:1 onestage.sbatch
#SBATCH -J kyle-onestage
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o /srv/home/kasha2/qrun/logs/%x-%A_%a.out
#SBATCH -e /srv/home/kasha2/qrun/logs/%x-%A_%a.err
set -uo pipefail
source ~/env.sh
Q=/srv/home/kasha2/qrun
R=/srv/home/kasha2/nedm/NeDM
PY=~/venvs/nedm/bin/python
i=$SLURM_ARRAY_TASK_ID
LOCAL=${TMPDIR:-/tmp}/kasha2_one_$SLURM_JOB_ID
mkdir -p $LOCAL && cp -r $Q/data/go2_crm_v2 $LOCAL/ || { echo "corpus copy failed"; exit 1; }
C=$LOCAL/go2_crm_v2
cd $R/quadruped
echo "task $i host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) code $(git -C $R rev-parse --short HEAD) $(date +%T)"
if [ $i -eq 5 ]; then
  for m in nnrom_v2e_s2 nnrom_v2e_s3 nnrom_v2e_s4 nnrom_v2e_s5 nnrom_v2e_s2_ms100 nnrom_v2e_s3_ms100 nnrom_v2e_s4_ms100 nnrom_v2e_s5_ms100; do
    [ -f $Q/$m/best.pt ] || { echo "missing $m"; continue; }
    echo "=== SWEEP $m"; $PY -u diagnostics/horizon_sweep.py --model $Q/$m/best.pt --corpus $C --device cuda 2>&1 | tail -12
  done
  echo ONESTAGE_REF_DONE; exit 0
fi
SEEDS=(80 81 82 80 81); EP=(16 16 16 40 40)
S=${SEEDS[$i]}; E=${EP[$i]}; L=nnrom_one${E}k_s$S
t0=$(date +%s)
$PY -u train.py --corpus $C --out $Q/$L --seed $S --device cuda --epochs $E --steps-per-epoch 1000 \
  --rollout-loss-steps 100 --rollout-loss-batch 4 --select-window 3 || exit 11
echo "TRAIN $L wall_s $(( $(date +%s) - t0 ))"
echo "=== SWEEP $L"; $PY -u diagnostics/horizon_sweep.py --model $Q/$L/best.pt --corpus $C --device cuda 2>&1 | tail -12
echo "ONESTAGE_${L}_DONE $(date +%T)"
