#!/bin/bash
# VERIFY_E2: HMMWV bit-identity, rigid, on a route the builder did not use (crater cross-slope 0017 route_01, 4 m/s,
# 25 s horizon).  orig = unmodified gen_collect_ext.py; ag_explicit = ag_gen_collect_ext.py --vehicle hmmwv with
# NEDM_VEHICLE=gator in the environment (the argument must win); ag_default = ag_gen_collect_ext.py, no switch, no env.
cd /home/harry/NeDM-traverse_mppi
V=artifacts/traverse/arena_gator_20260925/verify_e2
C=$PWD/artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases
COMMON="--source-root . --case $C/f104_v2_group_0017.json --route $C/routes/f104_v2_group_0017/route_01.json --chrono-data /home/harry/chrono/data --horizon-s 25 --local"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export PYTHONPATH=/home/harry/chrono/build/bin:src:scripts
unset NEDM_VEHICLE
PY="/usr/bin/python3.12 -P -u"
rm -rf $V/bitid_rigid; mkdir -p $V/bitid_rigid
$PY scripts/gen_collect_ext.py $COMMON --out $V/bitid_rigid/orig > $V/bitid_rigid/orig.log 2>&1 &
NEDM_VEHICLE=gator $PY scripts/ag_gen_collect_ext.py $COMMON --vehicle hmmwv --out $V/bitid_rigid/ag_explicit > $V/bitid_rigid/ag_explicit.log 2>&1 &
$PY scripts/ag_gen_collect_ext.py $COMMON --out $V/bitid_rigid/ag_default > $V/bitid_rigid/ag_default.log 2>&1 &
wait
