#!/bin/bash
# VERIFY_E2: re-run the flat soil calibration once for the chosen Gator cylinders (defaults of ag_vehicle, no radius
# override) and the HMMWV reference, same protocol as the builder (0.8 s full-brake settle + 2 s full throttle, flat
# 16 m patch, production crm_main.json through crm_collect.build_crm).  Under the CRM lock.
cd /home/harry/NeDM-traverse_mppi
V=$PWD/artifacts/traverse/arena_gator_20260925/verify_e2
export PYTHONPATH=/home/harry/chrono/build/bin:src:scripts OMP_NUM_THREADS=4
unset NEDM_VEHICLE
PY="/usr/bin/python3.12 -P -u"
L="flock /tmp/luffy_crm.lock"
rm -f $V/calib.jsonl
$L $PY scripts/ag_soil_calibrate.py --vehicle gator --tag verify_gator_default --out $V/calib.jsonl > $V/calib_gator.log 2>&1
$L $PY scripts/ag_soil_calibrate.py --vehicle hmmwv --tag verify_hmmwv_ref --out $V/calib.jsonl > $V/calib_hmmwv.log 2>&1
