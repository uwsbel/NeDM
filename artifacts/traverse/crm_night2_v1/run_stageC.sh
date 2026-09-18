#!/bin/bash
# Stage C (velocity input on re-anchored samples) after stage B has finished on the 5090.
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; K=artifacts/traverse/crm_night2_v1; mkdir -p $K/stageC
# runs concurrently with stage A/B (the sweep is latency-bound; two processes share the 5090)
runC() {  # world arch ctx lr extra...
  local w=$1 arch=$2 ctx=$3 lr=$4; shift 4; local out=$K/stageC
  local tag="${w}_${arch}_${ctx}$( [[ " $* " == *"--vplane"* ]] && echo _vp )_lr${lr}_holdout"
  [ -f "$out/$tag.json" ] && { echo "skip $tag"; return; }
  $PY scripts/n2_arch_train.py --ds $K/datasets/reanchor_$w.npz --out $out --world $w --arch $arch --ctx $ctx --lr $lr --seeds 5 --save "$@" 2>&1 | grep -v Warn | grep "ENSEMBLE\|Error\|error" | cut -c1-200
}
for w in crm rigid; do
  runC $w gru geom 0.002; runC $w gru vel 0.002; runC $w gru chassis 0.002; runC $w gru geom 0.002 --vplane
  runC $w tx96_2 geom 0.001; runC $w tx96_2 chassis 0.001
done
echo STAGE_C_DONE
