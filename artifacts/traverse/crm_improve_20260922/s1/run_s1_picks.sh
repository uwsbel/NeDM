#!/bin/bash
# S1 pick sets (PLAN crm_improve S1): CEM 4x64 (arm B) from the recorded frame-60 states of K1 A5 pass 1, CRM world.
# usage: bash artifacts/traverse/crm_improve_20260922/s1/run_s1_picks.sh
set -u
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=6
K1=artifacts/traverse/generalist_20260921/A_adapt; D=$K1/train/deploy_v1; K2=artifacts/traverse/crm_improve_20260922; S1=$K2/s1
plan() { local arm=$1 model=$2 family=$3
  [ -f $S1/picks_crm_${arm}/summary.json ] && { echo "skip $arm"; return; }
  echo "$(date +%H:%M:%S) start crm $arm"
  $PY scripts/ci_planner.py --family $family --cases $K1/suite/cases --map-root artifacts/traverse/crm_f104_v1/map_root \
     --models "$D/${model}_deploy_s*.pt" --world crm --domain crm --arms B --poses $K1/a5/poses_crm.json \
     --out $S1/picks_crm_${arm} --task-root $K2 --verify 0 > $S1/picks_crm_${arm}.log 2>&1
  echo "$(date +%H:%M:%S) done  crm $arm rc=$?"; }
plan H_cont          H      cont
plan H_conthead      H      cont_head
plan Spcrm_conthead  Sp_crm cont_head
echo "S1 planning finished $(date)"
