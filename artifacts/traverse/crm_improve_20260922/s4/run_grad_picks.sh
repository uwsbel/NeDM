#!/bin/bash
set -u
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=6
K1=artifacts/traverse/generalist_20260921/A_adapt; K2=artifacts/traverse/crm_improve_20260922
for L in 0p5 1; do
  for M in Hn:deploy_a1_haux_gru X:deploy_a3_haux_txjoint; do
    arm=${M%%:*}; glob=${M#*:}
    out=$K2/s4/L$L/picks_crm_${arm}G
    [ -f $out/summary.json ] && { echo "skip $arm L$L"; continue; }
    echo "$(date +%H:%M:%S) start grad $arm L$L"
    $PY scripts/ci_grad.py --cases $K1/suite/cases --map-root artifacts/traverse/crm_f104_v1/map_root \
      --models "$K2/deploy_v1/${glob}_s*.pt" --world crm --domain crm \
      --poses $K2/a5data/decision_suite_L$L/poses_crm_all.json --task-root $K2 \
      --out $out --ref-b-picks $K2/s2/L$L/picks_crm_${arm}/picks > $out.log 2>&1
    echo "$(date +%H:%M:%S) done grad $arm L$L rc=$?"
  done
done
echo "gradient picks finished $(date)"
