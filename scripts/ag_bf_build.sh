#!/bin/bash
# arena_gator_20260925 task B stage 2 ("Bfull", PLAN 7.10: retrain G/H on more tiers once the collection finished):
# the Gator f104 soil ci_train file on ALL tiers of the task file (0-12), with scripts/ag_build_ds.py (unchanged, the
# same file as S1/E4/E5a: 5e2eae03...) from the Bfull tool tree G3/tools/bf (a copy of this worktree's scripts/ + src/),
# on the cluster login node (numpy only). Same arguments as scripts/ag_s1_build.sh gator except --tiers 0-12 and the
# task file soil_v3.json (= soil_v2 unchanged as a prefix + evaluation rows of tier < 0, which the builder ignores).
#   bash ag_bf_build.sh gator
# Output G3/e4/soil_bf/f104_gator/ci_f104_gator_crm.npz + record.
set -eo pipefail
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
T=$G3/tools/bf
source /work1/dannegrut/harry/nrd/env.sh >/dev/null 2>&1; nrd_pychrono >/dev/null 2>&1
unset NEDM_VEHICLE
cd $T; export PYTHONPATH=$T/src:$T/scripts:${PYTHONPATH:-}
case $1 in
  gator) ARGS="--arena f104 --vehicle gator --map-root $G3/e4/map_roots/f104 --out $G3/e4/soil_bf/f104_gator";;
  *) echo "usage: $0 gator"; exit 2;;
esac
CMD="scripts/ag_build_ds.py --world crm --crm-runs $G3/soil_v1/runs --tasks-crm $G3/tasks/soil_v3.json --tiers 0-12 $ARGS --source-root $G3/source --workers ${AG_WORKERS:-16}"
echo "$(date +%T) $NRD_PYTHON $CMD"
nice -n 10 $NRD_PYTHON -u $CMD
echo "$(date +%T) build exit: $?"
