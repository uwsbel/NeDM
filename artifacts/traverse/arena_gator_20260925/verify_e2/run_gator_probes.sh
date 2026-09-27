#!/bin/bash
# VERIFY_E2: what does Gator mode actually build?  Real wrapper collectors run under probe_gator.py (logs the built
# vehicle / soil geometry, changes nothing).  Routes the builder did not use.
#  rigid: NEDM_VEHICLE=gator and NO --vehicle argument (the environment path of the switch), hill cross-slope 0012
#         route_01 (4 m/s), 15 s, --local, single thread.
#  soil : --vehicle gator, long traverse 0021 route_05 (4 m/s, lateral offset -4 m), 12 s, production config, lock.
cd /home/harry/NeDM-traverse_mppi
V=$PWD/artifacts/traverse/arena_gator_20260925/verify_e2
C=$PWD/artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases
CFG=$PWD/artifacts/traverse/crm_f104_v1/configs/crm_main.json
export PYTHONPATH=/home/harry/chrono/build/bin:src:scripts
PY="/usr/bin/python3.12 -P -u"
mkdir -p $V/gator
rm -rf $V/gator/rigid_0012_r01 $V/gator/soil_0021_r05
( export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
  NEDM_VEHICLE=gator $PY $V/probe_gator.py rigid --source-root . --case $C/f104_v2_group_0012.json \
    --route $C/routes/f104_v2_group_0012/route_01.json --chrono-data /home/harry/chrono/data --horizon-s 15 --local \
    --out $V/gator/rigid_0012_r01 > $V/gator/rigid_0012_r01.log 2>&1 ) &
( unset NEDM_VEHICLE; export OMP_NUM_THREADS=4
  flock /tmp/luffy_crm.lock $PY $V/probe_gator.py soil --vehicle gator --source-root . --case $C/f104_v2_group_0021.json \
    --route $C/routes/f104_v2_group_0021/route_05.json --chrono-data /home/harry/chrono/data --crm-config $CFG \
    --horizon-s 12 --episode-seed 7 --out $V/gator/soil_0021_r05 > $V/gator/soil_0021_r05.log 2>&1 ) &
wait
