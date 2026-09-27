#!/bin/bash
# E6b final analysis (arena_gator_20260925, 09-26): copy the small per-drive files of the rigid evaluation that the
# outcome index reads (outcome.json, trajectory.npz, episode_complete.json, case.json, vehicle_extra.npz if present)
# from G3/rigid_eval/runs to K3/e6/runs_rigid, plus the pool records (done/, TASKLIST.txt); host and chassis contact per
# run are extracted on the login node (scripts/ag_e6b_host_extract.py) and only that json is copied.
#   bash scripts/ag_e6b_final_sync.sh
set -euo pipefail
cd "$(dirname "$0")/.."
K3=artifacts/traverse/arena_gator_20260925
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
mkdir -p $K3/e6/runs_rigid $K3/e6/pool $K3/e6/index
rsync -a --include='*/' --include='outcome.json' --include='trajectory.npz' --include='episode_complete.json' \
  --include='case.json' --include='vehicle_extra.npz' --exclude='*' amd:$G3/rigid_eval/runs/ $K3/e6/runs_rigid/
rsync -a amd:$G3/rigid_eval/pool/done amd:$G3/rigid_eval/TASKLIST.txt $K3/e6/pool/
ssh amd "mkdir -p $G3/tools/e6b/final"
scp -q scripts/ag_e6b_host_extract.py amd:$G3/tools/e6b/final/
ssh amd "python3 $G3/tools/e6b/final/ag_e6b_host_extract.py --runs $G3/rigid_eval/runs --out $G3/tools/e6b/final/run_hosts.json"
rsync -a amd:$G3/tools/e6b/final/run_hosts.json $K3/e6/index/rigid_run_hosts.json
echo "runs: $(ls $K3/e6/runs_rigid | wc -l), complete: $(ls $K3/e6/runs_rigid/*/episode_complete.json 2>/dev/null | wc -l), shards done: $(ls $K3/e6/pool/done | wc -l)"
