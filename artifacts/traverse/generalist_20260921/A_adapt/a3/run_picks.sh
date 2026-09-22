#!/bin/bash
# A3 picks for one deploy ensemble in both worlds: usage run_picks.sh <MODEL_TAG e.g. H> [groups-file]
set -u
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts
K=artifacts/traverse/generalist_20260921/A_adapt; M=$1; GFILE=${2:-}
for W in crm rigid; do
  extra=""; [ -n "$GFILE" ] && extra="--groups @$GFILE"
  $PY scripts/ga_planner.py --cases $K/suite/cases --map-root artifacts/traverse/crm_f104_v1/map_root \
     --models "$K/train/deploy_v1/${M}_deploy_s*.pt" --world $W --domain $W --arms B --out $K/a3/picks_${W}_${M} \
     --ref-picks $K/suite/picks_${W}_Scrm/picks --ref-arms B:B --task-root $K --verify 2 $extra > $K/a3/picks_${W}_${M}.log 2>&1
  echo "$(date +%H:%M:%S) $M $W rc=$?"
done
