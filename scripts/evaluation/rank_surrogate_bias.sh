#!/bin/bash
# Rank surrogates by the ONE error the attribution named as binding: the optimistic
# velocity bias.
#
# The expensive way to score a surrogate ablation is fine-tune a policy in each and run a
# Chrono verdict: 6 arms x 90 min of GPU scoring, and it answers a compound question
# (model quality AND what the optimiser does with it). The attribution result makes a much
# cheaper measurement sufficient. The bias is present for the BASE policy too -- 0.3908
# predicted against 0.3565 delivered, 8-9% high -- so it can be read with no fine-tuning
# at all: drive the base policy in each candidate surrogate and compare its predicted
# velocity against the Chrono number already scored for that policy.
#
# Minutes per arm instead of hours, and it isolates the model.
set -u
REPO=$(for r in /home/kyle/Documents/sbel/NeDM /home/kyle/sbel/NeDM; do [ -d "$r/scripts" ] && echo "$r" && break; done)
PY=/home/kyle/miniconda3/envs/nedm/bin/python
S=/home/kyle/sbel-artifacts
cd "$REPO"; export PYTHONPATH=$PWD/src NEDM_REPO=$PWD
BASEPOL=$S/checkpoints/go2_cts_150k.pt
CHR=${CHR:-$S/crmtrack_BASE_$(hostname).json}
[ -f "$CHR" ] || CHR=$(ls -t $S/crmtrack_go2_cts_150k*.json $S/crmtrack_BASE*.json 2>/dev/null | head -1)
[ -f "$CHR" ] || { echo "FATAL: no scored base-policy Chrono file on this box"; exit 1; }
echo "chrono truth: $(basename "$CHR")"
for run in "$@"; do
  # WHICH CHECKPOINT, and why it is not best_val by default any more.
  #
  # best_val.pt is saved at the epoch with the lowest rollout_sel, and rollout_sel is
  # computed on 12 rollouts and swings by 2-7x between consecutive epochs. Across nine
  # runs the arm with the lowest MINIMUM had the worst MEDIAN of its last forty epochs,
  # so best_val.pt is the luckiest evaluation of eighty rather than the best model. Two
  # surrogates compared through their best_val checkpoints are being compared on which
  # run drew a better twelve-episode sample.
  #
  # last.pt is the final epoch for every run, chosen by no metric at all, so it is the
  # fair default for comparing MODELS. Set CKPT_NAME=best_val to reproduce the old
  # behaviour deliberately.
  ck=$S/training_runs/$run/checkpoints/${CKPT_NAME:-last}.pt
  [ -f "$ck" ] || { printf "  %-22s no checkpoint yet\n" "$run"; continue; }
  printf "  %-22s " "$run"
  $PY scripts/evaluation/attribute_tracking_error.py \
      --policy "$BASEPOL" --chrono "$CHR" --surrogate "$ck" \
      --steps 100 --episodes 24 2>&1 | grep -iE "surrogate|chrono|command" | paste -sd" | " -
done
