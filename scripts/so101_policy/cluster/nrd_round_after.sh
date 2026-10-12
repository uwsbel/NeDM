#!/bin/bash
# NRD data round of the SO101 policy study: the 'after' numbers of the fine-tuned grasp NRDs on luffy (RTX 5090), next
# to the 'before' numbers of the final models (same script, same episodes). Run from the worktree root when both cluster
# runs have complete.json (nrd_round_ft.sbatch):
#   bash scripts/so101_policy/cluster/nrd_round_after.sh            (ROUND=noisy_v1, OUT=~/so101_policy/runs/nrd_round)
# Steps: (1) rsync the two run directories from AMD (no contact_state.pt) to $OUT/after/; (2) copy best.pt to
# ~/so101_policy/ckpt/nrd_<ROUND>_s6{1,2}.pt and last.pt to ..._s6{1,2}_last.pt (sha256 printed); (3) for best and for
# last (the fixed-budget row; best may be the starting model): eval_nrd_noisy.py with both models on noisy_v1 split 1
# and on the production validation episodes (val_only: the 7 task families, and the 5 probe families apart); (4) the
# before / after tables (--compare: per group 1 s and 4 s cube p95, lift wrong; paired lift fixed / broken with a sign
# test; the forgetting gate after <= 1.10 x before, RF sec. 7 G1) against $OUT/build2/before_*.json. Episodes:
# validation only (never the sealed test). About 15 min on the 5090.
set -euo pipefail
ROUND=${ROUND:-noisy_v1}
OUT=${OUT:-$HOME/so101_policy/runs/nrd_round}
BEFORE=${BEFORE:-$OUT/build2}
B=/work1/dannegrut/harry/experiments/so101_policy_20261011
NOISY=${NOISY:-$HOME/so101_policy/data/$ROUND}
VAL=${VAL:-$HOME/so101_nrd_fix/data/val_only}
A=$OUT/after
mkdir -p "$A" "$HOME/so101_policy/ckpt"
for s in 61 62; do
  rsync -a --exclude contact_state.pt --exclude 'contact_state.pt.tmp' "amd:$B/runs/nrd_ft_${ROUND}_s$s/" "$A/nrd_ft_${ROUND}_s$s/"
  [ -f "$A/nrd_ft_${ROUND}_s$s/complete.json" ] || { echo "s$s: no complete.json yet"; exit 1; }
  cp "$A/nrd_ft_${ROUND}_s$s/best.pt" "$HOME/so101_policy/ckpt/nrd_${ROUND}_s$s.pt"
  cp "$A/nrd_ft_${ROUND}_s$s/last.pt" "$HOME/so101_policy/ckpt/nrd_${ROUND}_s${s}_last.pt"
  sha256sum "$HOME/so101_policy/ckpt/nrd_${ROUND}_s$s.pt" "$HOME/so101_policy/ckpt/nrd_${ROUND}_s${s}_last.pt"
done
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nedm
export PYTHONPATH=src OMP_NUM_THREADS=2
for kind in best last; do   # best.pt (validation-selected, the init check included) and last.pt (fixed budget)
  sfx=$([ "$kind" = best ] && echo "" || echo "_last")
  M="s61=$HOME/so101_policy/ckpt/nrd_${ROUND}_s61$sfx.pt s62=$HOME/so101_policy/ckpt/nrd_${ROUND}_s62$sfx.pt"
  python -P scripts/so101_policy/eval_nrd_noisy.py --data "$NOISY" --model $M --out "$A/after${sfx}_${ROUND}_val.json" | tee "$A/after${sfx}_${ROUND}_val.log"
  python -P scripts/so101_policy/eval_nrd_noisy.py --data "$VAL" --model $M --out "$A/after${sfx}_val_only.json" | tee "$A/after${sfx}_val_only.log"
  python -P scripts/so101_policy/eval_nrd_noisy.py --data "$VAL" --families free_arm cube_flight cube_table jaw_fixed jaw_moving \
    --model $M --out "$A/after${sfx}_val_only_probes.json" | tee "$A/after${sfx}_val_only_probes.log"
  for t in "${ROUND}_val" val_only val_only_probes; do
    python -P scripts/so101_policy/eval_nrd_noisy.py --compare before="$BEFORE/before_$t.json" after$sfx="$A/after${sfx}_$t.json" \
      --out "$A/compare${sfx}_$t.json" | tee "$A/compare${sfx}_$t.txt"
  done
done
