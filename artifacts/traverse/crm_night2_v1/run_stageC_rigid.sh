#!/bin/bash
# Rigid velocity arms first (needed by the moving-start analysis), 3 seeds; the main stage C script skips finished tags.
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; K=artifacts/traverse/crm_night2_v1; mkdir -p $K/stageC
runC() { local w=$1 arch=$2 ctx=$3 lr=$4; shift 4; local tag="${w}_${arch}_${ctx}$( [[ " $* " == *"--vplane"* ]] && echo _vp )_lr${lr}_holdout"
  [ -f "$K/stageC/$tag.json" ] && { echo "skip $tag"; return; }
  $PY scripts/n2_arch_train.py --ds $K/datasets/reanchor_$w.npz --out $K/stageC --world $w --arch $arch --ctx $ctx --lr $lr --seeds 3 --save "$@" 2>&1 | grep -v Warn | grep "ENSEMBLE\|Error\|error" | cut -c1-200; }
runC rigid gru geom 0.002; runC rigid gru chassis 0.002; runC rigid gru geom 0.002 --vplane; runC rigid tx96_2 chassis 0.001; runC rigid gru vel 0.002
echo STAGE_C_RIGID_DONE
