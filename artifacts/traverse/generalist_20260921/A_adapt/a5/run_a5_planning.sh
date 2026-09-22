#!/bin/bash
# A5 pass-2 planning (PLAN v2 A5): every arm plans from the recorded frame-60 state of the pass-1 approach drive.
# usage: run_a5_planning.sh <world>
set -u
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts
K=artifacts/traverse/generalist_20260921/A_adapt; W=$1; D=$K/train/deploy_v1; A5=$K/a5
plan() { local arm=$1 model=$2 poses=$3
  [ -f $A5/picks_${W}_${arm}/summary.json ] && { echo "skip $arm"; return; }
  echo "$(date +%H:%M:%S) start $W $arm"
  $PY scripts/ga_planner.py --cases $K/suite/cases --map-root artifacts/traverse/crm_f104_v1/map_root --models "$D/${model}_deploy_s*.pt" \
     --world $W --domain $W --arms B --poses $poses --out $A5/picks_${W}_${arm} --task-root $K --verify 0 > $A5/picks_${W}_${arm}.log 2>&1
  echo "$(date +%H:%M:%S) done  $W $arm rc=$?"; }
plan H     H     $A5/poses_${W}.json
plan Hmask H     $A5/poses_${W}_masked.json
plan T     T     $A5/poses_${W}.json
plan P     P     $A5/poses_${W}.json
plan Spcrm Sp_crm $A5/poses_${W}.json
while [ ! -f $D/Sp_rigid_deploy.json ]; do sleep 60; done
plan Sprigid Sp_rigid $A5/poses_${W}.json
echo "A5 planning $W finished $(date)"
