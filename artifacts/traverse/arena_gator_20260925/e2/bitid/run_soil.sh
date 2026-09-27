#!/bin/bash
# HMMWV bit-identity, soil: crm_collect.py vs ag_crm_collect.py (default = hmmwv) and crm_collect_ext.py vs
# ag_crm_collect.py --base crm_collect_ext; 12 s horizon, production config; each run holds the local CRM lock.
cd /home/harry/NeDM-traverse_mppi
E=artifacts/traverse/arena_gator_20260925/e2/bitid
CASE=$PWD/artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases/f104_v2_group_0000.json
ROUTE=$PWD/artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases/routes/f104_v2_group_0000/route_00.json
CFG=$PWD/artifacts/traverse/crm_f104_v1/configs/crm_main.json
COMMON="--source-root . --case $CASE --route $ROUTE --chrono-data /home/harry/chrono/data --crm-config $CFG --horizon-s 12 --episode-seed 1"
export PYTHONPATH=/home/harry/chrono/build/bin:src:scripts OMP_NUM_THREADS=4
PY="/usr/bin/python3.12 -P -u"
L="flock /tmp/luffy_crm.lock"
$L $PY scripts/crm_collect.py $COMMON --out $E/soil_orig > $E/soil_orig.log 2>&1
$L $PY scripts/ag_crm_collect.py $COMMON --out $E/soil_ag_default > $E/soil_ag_default.log 2>&1
$L $PY scripts/crm_collect_ext.py $COMMON --out $E/soil_ext_orig > $E/soil_ext_orig.log 2>&1
$L $PY scripts/ag_crm_collect.py --base crm_collect_ext $COMMON --vehicle hmmwv --out $E/soil_ag_ext > $E/soil_ag_ext.log 2>&1
