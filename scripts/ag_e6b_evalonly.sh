#!/bin/bash
# arena_gator_20260925 E6b: evaluation-only rigid row files of the 8 unseen test arenas from their designed-route drives
# (rigid_v1/runs, kind test_designed in rigid_v2.json = near rows of rigid_hmmwv_v1 + spread rows), built ON THE CLUSTER
# LOGIN NODE with scripts/ag_build_evalonly.py from the tool tree G3/tools/e6b/tree (worktree copy, hashes = local).
# Output G3/e4/evalonly/<arena>_rigid_test (for offline scoring only; split evalonly). Usage: ag_e6b_evalonly.sh [arenas]
set -eo pipefail
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
T=$G3/tools/e6b/tree
source /work1/dannegrut/harry/nrd/env.sh >/dev/null 2>&1; nrd_pychrono >/dev/null 2>&1
unset NEDM_VEHICLE
cd $T; export PYTHONPATH=$T/src:$T/scripts:${PYTHONPATH:-}
ARENAS=${*:-g260 g271 g251 g247 g258 g268 g263 g241}
for a in $ARENAS; do
  o=$G3/e4/evalonly/${a}_rigid_test
  [[ -e $o/${a}_hmmwv_rigid_evalonly_record.json ]] && { echo "$(date +%T) $a done already"; continue; }
  echo "$(date +%T) $a start"
  nice -n 10 $NRD_PYTHON -u scripts/ag_build_evalonly.py --arena $a --world rigid --runs $G3/rigid_v1/runs --tasks $G3/tasks/rigid_v2.json \
    --map-root $G3/map_roots/$a --source-root $G3/r2/source --out $o --allow-suite-ids --workers 12
  echo "$(date +%T) $a exit $?"
done
