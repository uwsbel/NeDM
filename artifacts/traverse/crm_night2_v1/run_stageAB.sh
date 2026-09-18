#!/bin/bash
# Stage A (architecture x world, hazard only) then stage B (energy head) on the RTX 5090, sequential, resumable (skips done tags).
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; K=artifacts/traverse/crm_night2_v1
run() {  # world arch lr extra...
  local w=$1 arch=$2 lr=$3; shift 3; local out=$K/stageA
  local tag="${w}_${arch}_geom_lr${lr}_holdout"
  [ -f "$out/$tag.json" ] && { echo "skip $tag"; return; }
  $PY scripts/n2_arch_train.py --ds $K/datasets/twin_$w.npz --out $out --world $w --arch $arch --lr $lr --seeds 5 "$@" 2>&1 | grep -v Warn | grep "ENSEMBLE\|Error\|error" | cut -c1-200
}
for w in crm rigid; do
  for arch in gru gru120 gru2 mlp; do run $w $arch 0.002; done
  for arch in tx96_2 tx128_4 patch4_128_4 tx_conv tx96_4 tx192_4 patch2_128_6; do run $w $arch 0.001; run $w $arch 0.002; done
done
echo STAGE_A_DONE
runB() {  # world arch lr lambda
  local w=$1 arch=$2 lr=$3 lam=$4; local out=$K/stageB
  local tag="${w}_${arch}_geom_E${lam}_lr${lr}_holdout"
  [ -f "$out/$tag.json" ] && { echo "skip $tag"; return; }
  $PY scripts/n2_arch_train.py --ds $K/datasets/twin_$w.npz --out $out --world $w --arch $arch --lr $lr --seeds 5 --energy $lam 2>&1 | grep -v Warn | grep "ENSEMBLE\|Error\|error" | cut -c1-200
}
for w in crm rigid; do for lam in 0.1 0.3 1; do runB $w gru 0.002 $lam; runB $w tx96_2 0.001 $lam; runB $w tx128_4 0.001 $lam; done; done
echo STAGE_B_DONE
