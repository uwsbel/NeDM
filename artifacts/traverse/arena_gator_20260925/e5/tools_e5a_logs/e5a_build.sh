#!/bin/bash
# arena_gator_20260925 E5a: one rigid ci_train file per arena and vehicle (scripts/ag_build_ds.py from the E5a tool tree),
# on the login node (numpy only). Usage: e5a_build.sh g203|g228|gator
set -eo pipefail
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
T=$G3/tools/e5a
source /work1/dannegrut/harry/nrd/env.sh >/dev/null 2>&1; nrd_pychrono >/dev/null 2>&1
unset NEDM_VEHICLE
cd $T; export PYTHONPATH=$T/src:$T/scripts:${PYTHONPATH:-}
case $1 in
  g203|g228) ARGS="--arena $1 --vehicle hmmwv --tasks-rigid $G3/tasks/rigid_hmmwv_v1.json --map-root $G3/map_roots/$1 --out $G3/e4/$1_hmmwv";;
  gator)     ARGS="--arena f104 --vehicle gator --tasks-rigid $G3/tasks/rigid_v2.json --map-root $G3/e4/map_roots/f104 --out $G3/e4/f104_gator";;
  *) echo "usage: $0 g203|g228|gator"; exit 2;;
esac
echo "$(date +%T) $NRD_PYTHON scripts/ag_build_ds.py --world rigid --rigid-runs $G3/rigid_v1/runs $ARGS --source-root $G3/source --workers 16"
nice -n 10 $NRD_PYTHON -u scripts/ag_build_ds.py --world rigid --rigid-runs $G3/rigid_v1/runs $ARGS --source-root $G3/source --workers 16
echo "$(date +%T) build exit: $?"
