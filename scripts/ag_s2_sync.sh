#!/bin/bash
# Soil track step 2 (arena_gator_20260925): copy the soil evaluation run folders (every run id the five soil build
# mappings point to: new evaluation rows + reused soil_v2 drives) from G3/soil_v1/runs to a local scratch folder,
# then build the outcome index with scripts/ag_eval_index.py (unchanged).
#   bash scripts/ag_s2_sync.sh [index name]      -> $K3/e6/index/<name>.json (default soil_eval_v1)
set -euo pipefail
cd "$(dirname "$0")/.."
K=artifacts/traverse/arena_gator_20260925
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
PY=/home/harry/miniconda3/envs/nedm/bin/python
export PYTHONPATH=src:scripts
NAME=${1:-soil_eval_v1}
RUNS=/tmp/ag_s2/soil_runs; mkdir -p $RUNS
LIST=/tmp/ag_s2/run_ids.txt
MAPS=$(ls $K/e6/tasks/soil_eval_p[1-5].json.mapping.json)
python3 - $MAPS > $LIST <<'PY'
import json, sys
ids = sorted({e['run_id'] for f in sys.argv[1:] for g in json.load(open(f))['groups'].values() for e in g.values()})
print('\n'.join(ids))
PY
# only folders that are complete on the cluster (a running episode's folder would be copied half-written)
ssh amd "cd $G3/soil_v1/runs && while read d; do [ -e \$d/episode_complete.json ] && echo \$d; done" < $LIST > /tmp/ag_s2/run_ids_complete.txt
echo "$(wc -l < /tmp/ag_s2/run_ids_complete.txt) of $(wc -l < $LIST) run folders complete on G3"
rsync -a -r --files-from=/tmp/ag_s2/run_ids_complete.txt amd:$G3/soil_v1/runs/ $RUNS/
$PY scripts/ag_eval_index.py --mapping $MAPS --runs $RUNS --out $K/e6/index/$NAME.json | tail -30
