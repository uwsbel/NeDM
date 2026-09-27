#!/bin/bash
# Soil analysis (arena_gator_20260925, S2 finish): copy only the small files the outcome index needs (outcome.json,
# trajectory.npz, episode_complete.json, case.json, vehicle_extra.npz) of every run id the five soil evaluation
# mappings point to (new evaluation rows + reused soil_v2 drives) from G3/soil_v1/runs into $K3/e6/runs_soil, then
# build the outcome index with scripts/ag_eval_index.py (unchanged).
#   bash scripts/ag_s2b_sync.sh [index name]      -> $K3/e6/index/<name>.json (default soil_eval_v1)
set -euo pipefail
cd "$(dirname "$0")/.."
K=artifacts/traverse/arena_gator_20260925
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
PY=/home/harry/miniconda3/envs/nedm/bin/python
export PYTHONPATH=src:scripts
NAME=${1:-soil_eval_v1}
RUNS=$K/e6/runs_soil; mkdir -p $RUNS
TMP=$K/e6/index/sync_$NAME; mkdir -p $TMP
MAPS=$(ls $K/e6/tasks/soil_eval_p[1-5].json.mapping.json)
python3 - $MAPS > $TMP/run_ids.txt <<'PY'
import json, sys
ids = sorted({e['run_id'] for f in sys.argv[1:] for g in json.load(open(f))['groups'].values() for e in g.values()})
print('\n'.join(ids))
PY
# only folders that are complete on the cluster
ssh amd "cd $G3/soil_v1/runs && while read d; do [ -e \$d/episode_complete.json ] && echo \$d; done" < $TMP/run_ids.txt > $TMP/run_ids_complete.txt
echo "$(wc -l < $TMP/run_ids_complete.txt) of $(wc -l < $TMP/run_ids.txt) run folders complete on G3"
awk '{for (i = 1; i <= 5; i++) print $1 "/" (i==1?"outcome.json":i==2?"trajectory.npz":i==3?"episode_complete.json":i==4?"case.json":"vehicle_extra.npz")}' \
  $TMP/run_ids_complete.txt > $TMP/files.txt
rsync -a --ignore-missing-args --files-from=$TMP/files.txt amd:$G3/soil_v1/runs/ $RUNS/
echo "local: $(ls $RUNS | wc -l) run folders, $(find $RUNS -name episode_complete.json | wc -l) complete markers"
$PY scripts/ag_eval_index.py --mapping $MAPS --runs $RUNS --out $K/e6/index/$NAME.json | tail -40
