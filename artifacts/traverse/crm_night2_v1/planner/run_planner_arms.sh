#!/bin/bash
# Study 2 pick pass (2026-09-18): four runs, one GPU process at a time. Logs in <out>/log.txt.
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python
P=artifacts/traverse/crm_night2_v1/planner
G=artifacts/traverse/fdm_f104_50h_20260909
mkdir -p $P/iter_crm $P/iter_rigid_f104_fixed2 $P/iter_rigid_g216 $P/iter_rigid_g231
$PY scripts/planner_arms.py --cases artifacts/traverse/crm_f104_v1/cases_eval/cases --map-root artifacts/traverse/crm_f104_v1/map_root \
  --models 'artifacts/traverse/crm_f104_v1/train_v1/deploy/CRM_N2_s*.pt' --out $P/iter_crm --world crm --arms A,B,C,D,E,F --verify 5 > $P/iter_crm/log.txt 2>&1
$PY scripts/planner_arms.py --cases $G/gen_v1/cases_test_f104/cases --rigid-arena assets/traverse/arena_f104_50h_v1 \
  --models "$G/night2_v1/final/N2_s*.pt" --out $P/iter_rigid_f104_fixed2 --world rigid --fixed2 --arms A,B,C,D,E --verify 5 > $P/iter_rigid_f104_fixed2/log.txt 2>&1
$PY scripts/planner_arms.py --cases $G/gen_v1/cases_test_g216/cases --rigid-arena assets/traverse/arena_g216 \
  --models "$G/night2_v1/final/N2_s*.pt" --out $P/iter_rigid_g216 --world rigid --arms A,B,C,D,E --verify 5 > $P/iter_rigid_g216/log.txt 2>&1
$PY scripts/planner_arms.py --cases $G/gen_v1/cases_test_g231/cases --rigid-arena assets/traverse/arena_g231 \
  --models "$G/night2_v1/final/N2_s*.pt" --out $P/iter_rigid_g231 --world rigid --arms A,B,C,D,E --verify 5 > $P/iter_rigid_g231/log.txt 2>&1
echo ALL_DONE
