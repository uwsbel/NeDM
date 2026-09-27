#!/bin/bash
# VERIFY_E2: HMMWV bit-identity, soil, on a route the builder did not use (hill cross-slope 0012 route_02, 6 m/s; the
# HMMWV reaches the goal at ~9 s, so the whole episode incl. goal/finalisation is compared), 15 s horizon, production
# crm_main.json, episode seed = md5-derived like the task builders.  Each run holds the local CRM lock.
cd /home/harry/NeDM-traverse_mppi
V=artifacts/traverse/arena_gator_20260925/verify_e2
C=$PWD/artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases
CFG=$PWD/artifacts/traverse/crm_f104_v1/configs/crm_main.json
SEED=$(python3 -c "import hashlib;print(int(hashlib.md5(b'f104_v2_group_0012_route_02').hexdigest()[:8],16))")
COMMON="--source-root . --case $C/f104_v2_group_0012.json --route $C/routes/f104_v2_group_0012/route_02.json --chrono-data /home/harry/chrono/data --crm-config $CFG --horizon-s 15 --episode-seed $SEED"
export PYTHONPATH=/home/harry/chrono/build/bin:src:scripts OMP_NUM_THREADS=4
unset NEDM_VEHICLE
PY="/usr/bin/python3.12 -P -u"
L="flock /tmp/luffy_crm.lock"
rm -rf $V/bitid_soil; mkdir -p $V/bitid_soil
$L $PY scripts/crm_collect.py $COMMON --out $V/bitid_soil/orig > $V/bitid_soil/orig.log 2>&1
NEDM_VEHICLE=gator $L $PY scripts/ag_crm_collect.py $COMMON --vehicle hmmwv --out $V/bitid_soil/ag_explicit > $V/bitid_soil/ag_explicit.log 2>&1
