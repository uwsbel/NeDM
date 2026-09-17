#!/bin/bash
#SBATCH -J kyle-soil-train
#SBATCH -p research
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH -t 06:00:00
#SBATCH -o /srv/home/kasha2/logs/%x-%j.out
#SBATCH -e /srv/home/kasha2/logs/%x-%j.err
#SBATCH --requeue
#
# Soil-range surrogate: does varying soil stiffness (0.40x-1.51x) move the 8-9%
# optimistic velocity bias, and does that reach the policy? This is the last of
# the "does a better model give a better policy" probes and the only one aimed at
# the bias rather than the variance.
#
# RESUME. This job was requeued once at epoch 45 of 80 (Restarts=1, Reason=Priority)
# and would have restarted from zero, overwriting the checkpoints it had already
# earned, because the trainer's --resume-from-checkpoint existed and nothing ever
# passed it. On a 6-hour limit against an 80-epoch fit that is not a rare edge
# case, it is the expected path. Every start now looks for work already done.
#
# The trainer restores model, optimiser, global_step, start_epoch and best_val_loss
# together, and no-ops if start_epoch already reached num_epochs, so resuming a
# finished run is safe and resuming repeatedly is idempotent.

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source /srv/home/kasha2/nedm/env-scoring.sh
export PYTHONPATH=/srv/home/kasha2/nedm/NeDM/src:$PYTHONPATH
VENVNV=/srv/home/kasha2/venvs/nedm/lib/python3.12/site-packages/nvidia
for d in "$VENVNV"/*/lib; do [ -d "$d" ] && export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"; done
cd /srv/home/kasha2/nedm/NeDM
PY=/srv/home/kasha2/venvs/nedm/bin/python

$PY -c "
import torch
assert torch.cuda.is_available(), 'no GPU'
a = torch.randn(256,256, device='cuda'); b = (a@a).sum().item()
print('GPU OK:', torch.cuda.get_device_name(0), 'matmul', round(b,2))
" || { echo "GPU CHECK FAILED"; exit 1; }

echo "restart count: ${SLURM_RESTART_COUNT:-0}"

# Where a given arm writes, according to its own config -- never hard-coded, so this
# cannot drift from the config the trainer actually reads.
out_dir_of() {
  $PY -c "import json,sys;print(json.load(open('configs/go2_crm_%s.json'%sys.argv[1]))['output_dir'])" "$1"
}

# A checkpoint is only resumable if it actually loads. A job killed mid-write leaves
# a truncated last.pt, and handing that to the trainer turns a lost hour into a lost
# run. Prefer last.pt, fall back to best_val.pt, and say clearly which was used.
resume_flag_for() {
  local out="$1" ck
  for ck in "$out/checkpoints/last.pt" "$out/checkpoints/best_val.pt"; do
    [ -f "$ck" ] || continue
    if $PY -c "
import sys, torch
c = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
assert 'model_state_dict' in c, 'no model_state_dict'
print('    %s: epoch %s, global_step %s' % (sys.argv[1].split('/')[-1],
      c.get('epoch','?'), c.get('global_step','?')), file=sys.stderr)
" "$ck" 2>>/dev/stderr; then
      echo "--resume-from-checkpoint $ck"
      return 0
    fi
    echo "    WARNING: $ck will not load (killed mid-write?), trying the next" >&2
  done
  echo "" # nothing resumable; a fresh start is correct
}

ARMS="soil17_w256"
i=0
for tag in $ARMS; do
  OUT=$(out_dir_of "$tag")
  echo "GPU $i -> $tag   out=$OUT"
  RESUME=$(resume_flag_for "$OUT")
  if [ -n "$RESUME" ]; then echo "    RESUMING: $RESUME"; else echo "    starting fresh"; fi

  # Append rather than truncate: the previous attempt's log is the only record of
  # why it stopped, and a requeue used to silently overwrite it.
  LOG=/srv/home/kasha2/logs/sat_$tag.log
  echo "=== attempt $(date -Is) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} ===" >> $LOG
  CUDA_VISIBLE_DEVICES=$i $PY scripts/training/train_hmmwv_dynamics.py \
      --config configs/go2_crm_$tag.json $RESUME >> $LOG 2>&1 &
  i=$((i+1))
done
wait
echo "SAT TRAIN BATCH COMPLETE"
