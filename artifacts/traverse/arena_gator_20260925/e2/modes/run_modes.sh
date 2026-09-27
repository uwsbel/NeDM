#!/bin/bash
# Gator mode through the non-native collector modes: rigid branch_auto (children re-launched through the wrapper),
# soil crm_collect_ext native and pid_held (short horizons, local CRM lock).
cd /home/harry/NeDM-traverse_mppi
E=artifacts/traverse/arena_gator_20260925/e2/modes
C=$PWD/artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases
CFG=$PWD/artifacts/traverse/crm_f104_v1/configs/crm_main.json
REC=$PWD/artifacts/traverse/arena_gator_20260925/e2/rigid_f104/gator/f104_v2_group_0000_route_01
export OMP_NUM_THREADS=1 PYTHONPATH=/home/harry/chrono/build/bin:src:scripts
PY="/usr/bin/python3.12 -P -u"
$PY scripts/ag_gen_collect_ext.py --vehicle gator --source-root . --case $C/f104_v2_group_0000.json --route $C/routes/f104_v2_group_0000/route_01.json \
  --chrono-data /home/harry/chrono/data --horizon-s 10 --local --mode branch_auto --recorded $REC --branch-frame 60 --n-cont 2 \
  --cont-seed 20260925 --cont-parallel 2 --out $E/rigid_branch_auto > $E/rigid_branch_auto.log 2>&1 &
S="--source-root . --case $C/f104_v2_group_0005.json --route $C/routes/f104_v2_group_0005/route_01.json --chrono-data /home/harry/chrono/data --crm-config $CFG --horizon-s 6 --episode-seed 7"
OMP_NUM_THREADS=4 flock /tmp/luffy_crm.lock $PY scripts/ag_crm_collect.py --base crm_collect_ext --vehicle gator $S --out $E/soil_ext_native > $E/soil_ext_native.log 2>&1
OMP_NUM_THREADS=4 flock /tmp/luffy_crm.lock $PY scripts/ag_crm_collect.py --base crm_collect_ext --vehicle gator $S --mode pid_held --out $E/soil_ext_pid_held > $E/soil_ext_pid_held.log 2>&1
OMP_NUM_THREADS=4 flock /tmp/luffy_crm.lock $PY scripts/ag_crm_collect.py --vehicle gator $S --out $E/soil_base_native > $E/soil_base_native.log 2>&1
wait
