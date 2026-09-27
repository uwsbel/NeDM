#!/bin/bash
# HMMWV bit-identity, rigid: original gen_collect_ext.py vs ag_gen_collect_ext.py (default and --vehicle hmmwv)
cd /home/harry/NeDM-traverse_mppi
E=artifacts/traverse/arena_gator_20260925/e2/bitid
CASE=$PWD/artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases/f104_v2_group_0000.json
ROUTE=$PWD/artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases/routes/f104_v2_group_0000/route_00.json
COMMON="--source-root . --case $CASE --route $ROUTE --chrono-data /home/harry/chrono/data --horizon-s 12 --local"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export PYTHONPATH=/home/harry/chrono/build/bin:src:scripts
PY="/usr/bin/python3.12 -P -u"
$PY scripts/gen_collect_ext.py $COMMON --out $E/rigid_orig > $E/rigid_orig.log 2>&1 &
$PY scripts/ag_gen_collect_ext.py $COMMON --out $E/rigid_ag_default > $E/rigid_ag_default.log 2>&1 &
NEDM_VEHICLE=gator $PY scripts/ag_gen_collect_ext.py $COMMON --vehicle hmmwv --out $E/rigid_ag_explicit > $E/rigid_ag_explicit.log 2>&1 &
wait
