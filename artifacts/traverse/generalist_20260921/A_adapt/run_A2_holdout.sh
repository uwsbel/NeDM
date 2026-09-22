#!/bin/bash
# A2 holdout arm matrix on the mixed re-anchored dataset (PLAN.md v2, stage A2). Two lanes on the shared 5090 with
# --x-half (~5 GB each). Selection split = val (test sealed). Run from the repo root after the substep audit finished.
set -u
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python
export PYTHONPATH=src:scripts
K=artifacts/traverse/generalist_20260921/A_adapt
DS=$K/datasets/mixed_reanchor.npz
OUT=$K/train/holdout_v1
mkdir -p $OUT/logs

run() {  # cond domain_filter [extra...]
  local cond=$1 filt=$2; shift 2
  local tag="ga_${cond}_${filt}_holdout_val$( [ "$*" != "" ] && echo "_$(echo $* | tr -d ' -')" )"
  if [ -f "$OUT/$tag.json" ]; then echo "skip $tag (done)"; return; fi
  echo "$(date +%H:%M:%S) start $tag"
  $PY scripts/ga_train.py --ds $DS --out $OUT --cond $cond --domain-filter $filt --split-eval val --mode holdout \
      --seeds 5 --epochs 30 --x-half --tag $tag "$@" > $OUT/logs/$tag.log 2>&1
  echo "$(date +%H:%M:%S) done  $tag rc=$?"
}

lane1() {
  run hist_aux both
  run none both
  run none crm
  run hist_rma both
  run tag both --startup-only
}
lane2() {
  run tag both
  run hist both
  run none rigid
  run hist_aux both --startup-only
  run none crm --startup-only
  run none rigid --startup-only
}
lane1 > $OUT/logs/lane1.txt 2>&1 &
lane2 > $OUT/logs/lane2.txt 2>&1 &
wait
echo "A2 holdout matrix finished $(date)"
