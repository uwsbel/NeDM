#!/bin/bash
# Rigid-ground held-out paths on sbel for the rigid-data control: base, CRM comparators and
# rigid-data arms, one machine, build and code version. Labels as label=policy arguments.
cd ~/qrc/quadruped
export PYTHONPATH=$HOME/qrc:$HOME/qrc/src OMP_NUM_THREADS=4
P=~/miniconda3/envs/nedm/bin/python
U=~/.cache/nedm_cluster_stage/assets/data/robot/go2_irrvis/urdf/go2_description.urdf
O=~/qrc/eval_rigid_sb; mkdir -p $O ~/qrc/logs/evr
for spec in "$@"; do L=${spec%%=*}; POL=${spec#*=}
  $P -u evaluate.py --policy $POL --urdf $U --corpus ~/qrun/quadruped/data/go2_crm_v2 --out $O/$L --label $L     --terrain rigid --paths 4 --seconds 15 --spawn-spread 1.0 > ~/qrc/logs/evr/$L.log 2>&1
  echo "EVAL $L rc=$? $(date +%H:%M)"; done
echo RIGID_BATCH_DONE
