#!/bin/bash
# A3 deploy ensembles (PLAN v2 A3): T = tag (oracle), H = hist_aux (deployable), P = none (pooled); --mode deploy fits
# every split==train row (val/test never), 5 seeds each, two lanes with --x-half on the shared 5090.
set -u
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python
export PYTHONPATH=src:scripts
K=artifacts/traverse/generalist_20260921/A_adapt
DS=$K/datasets/mixed_reanchor.npz
OUT=$K/train/deploy_v1
mkdir -p $OUT/logs
run() { local cond=$1 name=$2; local tag="${name}_deploy"
  if [ -f "$OUT/$tag.json" ]; then echo "skip $tag"; return; fi
  echo "$(date +%H:%M:%S) start $tag"
  $PY scripts/ga_train.py --ds $DS --out $OUT --cond $cond --domain-filter both --split-eval val --mode deploy \
      --seeds 5 --epochs 30 --x-half --tag $tag > $OUT/logs/$tag.log 2>&1
  echo "$(date +%H:%M:%S) done  $tag rc=$?"; }
( run hist_aux H; run none P ) > $OUT/logs/lane1.txt 2>&1 &
( run tag T ) > $OUT/logs/lane2.txt 2>&1 &
wait; echo "A3 deploy ensembles finished $(date)"
