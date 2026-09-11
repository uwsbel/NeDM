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
  ck=$S/training_runs/$run/checkpoints/best_val.pt
  [ -f "$ck" ] || { printf "  %-22s no checkpoint yet\n" "$run"; continue; }
  printf "  %-22s " "$run"
  $PY scripts/evaluation/attribute_tracking_error.py \
      --policy "$BASEPOL" --chrono "$CHR" --surrogate "$ck" \
      --steps 100 --episodes 24 2>&1 | grep -iE "surrogate|chrono|command" | paste -sd" | " -
done
