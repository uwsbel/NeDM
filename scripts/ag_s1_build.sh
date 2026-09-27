#!/bin/bash
# arena_gator_20260925 soil stage 1 (S1, PLAN 7.10): one soil ci_train file per arena and vehicle, TIERS 0-6 ONLY,
# with scripts/ag_build_ds.py from the S1 tool tree (G3/tools/s1 = copy of the worktree scripts/src), on the login node
# (numpy only). Usage: ag_s1_build.sh g203|g228|gator
#   g203 / g228: HMMWV rows of tasks/soil_v2.json (soil_v1 rows are identical), map root G3/map_roots/<arena>
#   gator:       gator__<collect_v1 id> rows of tasks/soil_v2.json, --vehicle gator (vehicle block required on every
#                run), map root G3/e4/map_roots/f104 (the crm_f104_v1 capture behind every f104 file)
# Output G3/e4/soil_s1/<arena>_<vehicle>/ci_<arena>_<vehicle>_crm.npz + record.
set -eo pipefail
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
T=$G3/tools/s1
source /work1/dannegrut/harry/nrd/env.sh >/dev/null 2>&1; nrd_pychrono >/dev/null 2>&1
unset NEDM_VEHICLE
cd $T; export PYTHONPATH=$T/src:$T/scripts:${PYTHONPATH:-}
case $1 in
  g203|g228) ARGS="--arena $1 --vehicle hmmwv --map-root $G3/map_roots/$1 --out $G3/e4/soil_s1/$1_hmmwv";;
  gator)     ARGS="--arena f104 --vehicle gator --map-root $G3/e4/map_roots/f104 --out $G3/e4/soil_s1/f104_gator";;
  *) echo "usage: $0 g203|g228|gator"; exit 2;;
esac
CMD="scripts/ag_build_ds.py --world crm --crm-runs $G3/soil_v1/runs --tasks-crm $G3/tasks/soil_v2.json --tiers 0-6 $ARGS --source-root $G3/source --workers ${AG_WORKERS:-16}"
echo "$(date +%T) $NRD_PYTHON $CMD"
nice -n 10 $NRD_PYTHON -u $CMD
echo "$(date +%T) build exit: $?"
