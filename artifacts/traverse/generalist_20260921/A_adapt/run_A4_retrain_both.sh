#!/bin/bash
set -u
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts
K=artifacts/traverse/generalist_20260921/A_adapt; DS=$K/datasets/mixed_reanchor_plus_branch_both.npz; OUT=$K/train/branch_v2; mkdir -p $OUT/logs
run() { local cond=$1 filt=$2; local tag="ga_${cond}_${filt}_plusbranchboth_holdout_val"
  [ -f "$OUT/$tag.json" ] && { echo "skip $tag"; return; }
  echo "$(date +%H:%M:%S) start $tag"
  $PY scripts/ga_train.py --ds $DS --out $OUT --cond $cond --domain-filter $filt --split-eval val --mode holdout --seeds 5 --epochs 30 --x-half --tag $tag > $OUT/logs/$tag.log 2>&1
  echo "$(date +%H:%M:%S) done  $tag rc=$?"; }
( run hist_aux both; run none crm ) > $OUT/logs/lane1.txt 2>&1 &
( run none rigid; run tag both ) > $OUT/logs/lane2.txt 2>&1 &
wait; echo "branch-both retrain finished $(date)"
