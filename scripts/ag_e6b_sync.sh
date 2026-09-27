#!/bin/bash
# E6b: sync the rigid evaluation outcomes (only the files the index, host check and contact read) and the pool records.
#   bash scripts/ag_e6b_sync.sh     -> /tmp/ag_e6b/rigid_eval_runs/<id>/{outcome.json,trajectory.npz,episode_complete.json,
#                                      simulation_provenance.json,rich_intervals.npz}, $K3/e6/pool/{done,logs}
set -euo pipefail
cd "$(dirname "$0")/.."
K3=artifacts/traverse/arena_gator_20260925
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
mkdir -p /tmp/ag_e6b/rigid_eval_runs $K3/e6/pool
rsync -a --include='*/' --include='outcome.json' --include='trajectory.npz' --include='episode_complete.json' \
  --include='simulation_provenance.json' --include='rich_intervals.npz' --exclude='*' amd:$G3/rigid_eval/runs/ /tmp/ag_e6b/rigid_eval_runs/
rsync -a amd:$G3/rigid_eval/pool/done amd:$G3/rigid_eval/TASKLIST.txt $K3/e6/pool/
rsync -a --include='step_*.out' --include='autostep.out' --exclude='*' amd:$G3/rigid_eval/pool/logs/ $K3/e6/pool/logs/
echo "runs: $(ls /tmp/ag_e6b/rigid_eval_runs | wc -l), complete: $(ls /tmp/ag_e6b/rigid_eval_runs/*/episode_complete.json 2>/dev/null | wc -l), shards done: $(ls $K3/e6/pool/done | wc -l)"
