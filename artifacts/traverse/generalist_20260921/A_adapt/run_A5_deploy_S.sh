#!/bin/bash
# A5 same-row specialist references S'_crm / S'_rigid (PLAN v2 A5): --cond none --domain-filter crm|rigid, deploy mode.
# Waits for run_A3_deploy.sh to finish (shares the GPU), then runs both in one lane.
set -u
cd /home/harry/NeDM-traverse_mppi
while pgrep -f run_A3_deploy.sh > /dev/null; do sleep 60; done
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts
K=artifacts/traverse/generalist_20260921/A_adapt; DS=$K/datasets/mixed_reanchor.npz; OUT=$K/train/deploy_v1; mkdir -p $OUT/logs
for f in crm rigid; do tag="Sp_${f}_deploy"
  [ -f "$OUT/$tag.json" ] && { echo "skip $tag"; continue; }
  echo "$(date +%H:%M:%S) start $tag"
  $PY scripts/ga_train.py --ds $DS --out $OUT --cond none --domain-filter $f --split-eval val --mode deploy --seeds 5 --epochs 30 --x-half --tag $tag > $OUT/logs/$tag.log 2>&1
  echo "$(date +%H:%M:%S) done  $tag rc=$?"
done
echo "A5 specialist references finished $(date)"
