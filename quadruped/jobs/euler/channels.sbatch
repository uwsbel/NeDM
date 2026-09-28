#!/bin/bash
# Channel study (2026-09-25): what should the reduced model carry? One task = one
# surrogate end to end: one-step training, 1 s rollout training, then the recipe's PPO
# fine-tune to iteration 1000 (ft_data.sbatch settings). Presets in params/presets.yaml.
#   sbatch -p research --array=0-5  --gres=gpu:h100:1 channels.sbatch
#   sbatch -p research --array=6-9  --gres=gpu:a100:1 channels.sbatch   (euler17)
#   sbatch -p sbel     --array=10-11 --gres=gpu:a100:1 channels.sbatch  (euler19, not preemptible)
# research CAN be requeued (rigid task 69921_0 was, 2026-09-25), so stages resume below. The corpus is copied to node-local disk first:
# three jobs reading it from CephFS at once stalled for ~8 min on 2026-09-25.
#SBATCH -J kyle-chan
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o /srv/home/kasha2/qrun/logs/%x-%A_%a.out
#SBATCH -e /srv/home/kasha2/qrun/logs/%x-%A_%a.err
set -uo pipefail
source ~/env.sh
Q=/srv/home/kasha2/qrun
R=/srv/home/kasha2/nedm/NeDM
PRESETS=(crm_forcez crm_sinkage crm_soil crm_force3d)
SEEDS=(70 71 72)
i=$SLURM_ARRAY_TASK_ID
P=${PRESETS[$((i % 4))]}; S=${SEEDS[$((i / 4))]}
L=nnrom_${P#crm_}_s$S
PY=~/venvs/nedm/bin/python
LOCAL=${TMPDIR:-/tmp}/kasha2_chan_$SLURM_JOB_ID
mkdir -p $LOCAL && cp -r $Q/data/go2_crm_v2 $LOCAL/ || { echo "corpus copy failed"; exit 1; }
C=$LOCAL/go2_crm_v2
cd $R/quadruped
echo "task $i preset $P seed $S host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) code $(git -C $R rev-parse --short HEAD) $(date +%T)"
# A requeue restarts the task; each stage skips if it already finished (a .done marker,
# since best.pt is written during training and its presence proves nothing).
if [ ! -f $Q/$L/.done ]; then
  $PY -u train.py --corpus $C --preset $P --out $Q/$L --seed $S --device cuda || exit 11
  touch $Q/$L/.done
fi
echo "STAGE1 $L done $(date +%T)"
if [ ! -f $Q/${L}_ms100/.done ]; then
  $PY -u train.py --corpus $C --preset $P --out $Q/${L}_ms100 --seed $S --device cuda \
    --init-from $Q/$L/best.pt --rollout-loss-steps 100 --rollout-loss-batch 4 --epochs 8 \
    --steps-per-epoch 1000 --lr 1e-4 --min-lr 1e-5 --warmup-steps 200 --select-window 3 || exit 12
  touch $Q/${L}_ms100/.done
fi
echo "STAGE2 $L done $(date +%T)"
$PY -u finetune.py --model $Q/${L}_ms100/best.pt --policy $Q/policy/policy.pt \
  --corpus $C --out $Q/ft_${L} --method ppo --device cuda --seed 0 \
  --steps 100 --branches 1024 --iters 1000 --target-dw 1e9 --snapshot-every 500 || exit 13
echo "CHAN_${L}_DONE $(date +%T)"
