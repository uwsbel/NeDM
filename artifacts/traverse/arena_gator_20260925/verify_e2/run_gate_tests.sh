#!/bin/bash
# VERIFY_E2: refusal paths of the vehicle switch (all must fail before any Chrono work and create no run folder).
cd /home/harry/NeDM-traverse_mppi
V=$PWD/artifacts/traverse/arena_gator_20260925/verify_e2
C=$PWD/artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases
CFG=$PWD/artifacts/traverse/crm_f104_v1/configs/crm_main.json
GFP=$PWD/artifacts/traverse/arena_gator_20260925/e2/runtime/gator_runtime_fingerprint.json
mkdir -p $V/gates
# HMMWV-only fingerprint = the Gator fingerprint without its /vehicle/gator/ entries
python3 -c "
import json,sys; d=json.load(open('$GFP')); d['runtime_sha256']={k:v for k,v in d['runtime_sha256'].items() if '/vehicle/gator/' not in k}
json.dump(d,open('$V/gates/hmmwv_only_fp.json','w'))"
RIG="--source-root . --case $C/f104_v2_group_0003.json --route $C/routes/f104_v2_group_0003/route_00.json --chrono-data /home/harry/chrono/data --horizon-s 4"
SOIL="--source-root . --case $C/f104_v2_group_0003.json --route $C/routes/f104_v2_group_0003/route_00.json --chrono-data /home/harry/chrono/data --crm-config $CFG --horizon-s 4"
export PYTHONPATH=/home/harry/chrono/build/bin:src:scripts
unset NEDM_VEHICLE FDM_RUNTIME_FINGERPRINT
PY="/usr/bin/python3.12 -P -u"
t() { name=$1; shift; out=$V/gates/$name; rm -rf $out; echo "== $name: $*" | sed "s|$PWD/||g"; "$@" --out $out 2>&1 | tail -n 1; echo "   rc=${PIPESTATUS[0]} run_folder_created=$([ -e $out ] && echo yes || echo no)"; }
t T1_rigid_gator_no_fp_no_local $PY scripts/ag_gen_collect_ext.py --vehicle gator $RIG
t T2_rigid_gator_hmmwv_only_fp_arg $PY scripts/ag_gen_collect_ext.py --vehicle gator --runtime-fingerprint $V/gates/hmmwv_only_fp.json $RIG
t T3_rigid_gator_hmmwv_only_fp_env env FDM_RUNTIME_FINGERPRINT=$V/gates/hmmwv_only_fp.json $PY scripts/ag_gen_collect_ext.py --vehicle gator $RIG
t T4_rigid_gator_hmmwv_only_fp_env_local env FDM_RUNTIME_FINGERPRINT=$V/gates/hmmwv_only_fp.json $PY scripts/ag_gen_collect_ext.py --vehicle gator $RIG --local
t T5_rigid_hmmwv_with_fp_arg $PY scripts/ag_gen_collect_ext.py --vehicle hmmwv --runtime-fingerprint $GFP $RIG --local
t T6_soil_with_fp_arg $PY scripts/ag_crm_collect.py --vehicle gator --runtime-fingerprint $GFP $SOIL
t T7_soil_hmmwv_with_gator_radius $PY scripts/ag_crm_collect.py --gator-soil-radius-front 0.2 $SOIL
t T8_bad_env_vehicle env NEDM_VEHICLE=polaris $PY scripts/ag_crm_collect.py $SOIL
t T9_bad_arg_vehicle $PY scripts/ag_gen_collect_ext.py --vehicle polaris $RIG --local
t T10_soil_bad_base $PY scripts/ag_crm_collect.py --base gen_collect --vehicle gator $SOIL
t T11_implausible_radius $PY scripts/ag_crm_collect.py --vehicle gator --gator-soil-radius-rear 0.9 $SOIL
