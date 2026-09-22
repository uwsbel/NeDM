#!/bin/bash
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
K=artifacts/traverse/generalist_20260921/A_adapt; DS=$K/datasets/mixed_reanchor_plus_branch_both.npz; OUT=$K/train/branch_v2
while [ ! -f $OUT/ga_hist_aux_both_plusbranchboth_holdout_val.json ]; do sleep 20; done
for f in rigid crm; do tag="ga_none_${f}_plusbranchboth_holdout_val"
  [ -f "$OUT/$tag.json" ] && continue
  echo "$(date +%H:%M:%S) start $tag"
  $PY scripts/ga_train.py --ds $DS --out $OUT --cond none --domain-filter $f --split-eval val --mode holdout --seeds 5 --epochs 30 --x-half --tag $tag > $OUT/logs/$tag.log 2>&1
  echo "$(date +%H:%M:%S) done  $tag rc=$?"
done
echo "refs finished $(date)"
