#!/bin/bash
# arena_gator_20260925 task B stage 2 (Bfull): copy the small files the outcome index needs (outcome.json, trajectory.npz,
# episode_complete.json, case.json, vehicle_extra.npz) of every run folder the two Bfull build mappings point to (new
# drives + the reused stage-1 drives) from G3/soil_v1/runs to K3/e6/runs_soil (only folders complete on G3), then build
# the outcome index with scripts/ag_eval_index.py (unchanged) and, with --analyze, the analysis under the frozen spec.
#   bash scripts/ag_bf_sync.sh [--analyze]
set -euo pipefail
cd "$(dirname "$0")/.."
K=artifacts/traverse/arena_gator_20260925
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
PY=/home/harry/miniconda3/envs/nedm/bin/python
export PYTHONPATH=src:scripts
RUNS=$K/e6/runs_soil; mkdir -p $RUNS /tmp/ag_bf
MAPS="$K/e6/tasks/soil_eval_bf1.json.mapping.json $K/e6/tasks/soil_eval_bf2.json.mapping.json"
python3 - $MAPS > /tmp/ag_bf/bf_run_ids.txt <<'PY'
import json, sys
print('\n'.join(sorted({e['run_id'] for f in sys.argv[1:] for g in json.load(open(f))['groups'].values() for e in g.values()})))
PY
ssh amd "cd $G3/soil_v1/runs && while read d; do [ -e \$d/episode_complete.json ] && echo \$d; done" < /tmp/ag_bf/bf_run_ids.txt > /tmp/ag_bf/bf_run_ids_complete.txt
echo "$(wc -l < /tmp/ag_bf/bf_run_ids_complete.txt) of $(wc -l < /tmp/ag_bf/bf_run_ids.txt) run folders complete on G3"
awk '{print $0"/outcome.json\n"$0"/trajectory.npz\n"$0"/episode_complete.json\n"$0"/case.json\n"$0"/vehicle_extra.npz"}' /tmp/ag_bf/bf_run_ids_complete.txt > /tmp/ag_bf/bf_files.txt
rsync -a --ignore-missing-args --files-from=/tmp/ag_bf/bf_files.txt amd:$G3/soil_v1/runs/ $RUNS/
$PY scripts/ag_eval_index.py --mapping $MAPS --runs $RUNS --out $K/e6/index/soil_eval_bfull.json | tail -12
if [[ "${1:-}" == "--analyze" ]]; then
  sha256sum -c $K/e6/analysis/spec_soil_v1_Bfull.sha256
  $PY scripts/ag_analyze.py --index $K/e6/index/soil_eval_bfull.json --spec $K/e6/analysis/spec_soil_v1_Bfull.json \
      --out $K/e6/analysis/results_soil_v1_Bfull.json | grep -v '^  '
fi
