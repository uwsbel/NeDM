#!/bin/bash
# CRM held-out paths on sbel for the rigid-data control: the base policy and CRM-surrogate
# comparators now, the rigid-surrogate arms later (same script, labels as arguments).
# One machine, one build, one code version (qrc 7cf70c43), so everything pairs.
# A GPU fault exits 90 with the partial record written; resume the arm.
cd ~/qrc/quadruped
export PYTHONPATH=$HOME/qrc:$HOME/qrc/src OMP_NUM_THREADS=8
P=~/miniconda3/envs/nedm/bin/python
U=~/.cache/nedm_cluster_stage/assets/data/robot/go2_irrvis/urdf/go2_description.urdf
O=~/qrc/eval_paths_sb; mkdir -p $O ~/qrc/logs/evp
for spec in "$@"; do
  L=${spec%%=*}; POL=${spec#*=}
  for attempt in 1 2 3; do
    if [ $attempt -eq 1 ]; then R=""; else R="--resume"; fi
    $P -u evaluate.py --policy $POL --urdf $U --corpus ~/qrun/quadruped/data/go2_crm_v2 \
      --out $O/$L --label $L --terrain crm --paths 4 --seconds 15 --spawn-spread 1.0 $R \
      >> ~/qrc/logs/evp/$L.log 2>&1
    rc=$?; [ $rc -ne 90 ] && break
    echo "  GPU fault in $L; resuming" 
  done
  echo "EVAL $L rc=$rc $(date +%H:%M)"
done
echo SCORE_BATCH_DONE
