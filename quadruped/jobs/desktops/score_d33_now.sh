#!/bin/bash
# Score the analytic arms ALONE on the GPU: evaluations aborted while the long-branch
# fine-tunes shared the 16 GB card. Wait for the grid to finish first. Fresh output dir;
# the aborted partial records in eval_paths_d33 are left where they are.
cd ~/qrc/quadruped
export PYTHONPATH=$HOME/qrc:$HOME/qrc/src OMP_NUM_THREADS=8
P=~/miniconda3/envs/nedm/bin/python
U=~/.cache/nedm_cluster_stage/assets/data/robot/go2_irrvis/urdf/go2_description.urdf
O=~/qrc/eval_paths_d33b; mkdir -p $O ~/qrc/logs/evp2
for L in an_h15_dw1 an_h15_dw05 an_h15_dw2 an_h15_dw4 an_h50_dw1 an_h100_dw1; do
  [ -s ~/qrc/$L/policy_ft.pt ] || { echo "no policy for $L"; continue; }
  for attempt in 1 2 3; do
    if [ $attempt -eq 1 ]; then R=""; else R="--resume"; fi
    $P -u evaluate.py --policy ~/qrc/$L/policy_ft.pt --urdf $U --corpus ~/qrun/quadruped/data/go2_crm_v2 \
      --out $O/$L --label $L --terrain crm --paths 4 --seconds 15 --spawn-spread 1.0 $R >> ~/qrc/logs/evp2/$L.log 2>&1
    rc=$?; [ $rc -ne 90 ] && break; done
  echo "EVAL $L rc=$rc $(date +%H:%M)"
done
echo SCORE_D33_DONE
