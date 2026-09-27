#!/bin/bash
# Gate checks of ag_gen_collect_ext.py without --local (temporary source root /tmp/ag_gate_src = symlinks to this worktree
# + the crm_improve_20260922 source_manifest.json, whose 9 frozen-file hashes equal the worktree's).
cd /home/harry/NeDM-traverse_mppi
# /tmp/ag_gate_src = rsync copy of src/, scripts/, assets/traverse/arena_f104_50h_v1 + the crm_improve manifest
E=artifacts/traverse/arena_gator_20260925/e2/gates
C=$PWD/artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases
COMMON="--source-root /tmp/ag_gate_src --case $C/f104_v2_group_0000.json --route $C/routes/f104_v2_group_0000/route_02.json --chrono-data /home/harry/chrono/data --horizon-s 4"
export OMP_NUM_THREADS=1 PYTHONPATH=/home/harry/chrono/build/bin:src:scripts
PY="/usr/bin/python3.12 -P -u"
echo "== G1 gator, HMMWV-only fingerprint (expect refusal before any output)"
$PY scripts/ag_gen_collect_ext.py --vehicle gator --runtime-fingerprint $E/hmmwv_only_fingerprint_test.json $COMMON --out $E/G1 2>&1 | tail -n 1; ls $E/G1 2>/dev/null | head -3
echo "== G2 gator, no fingerprint, no --local (expect refusal)"
env -u FDM_RUNTIME_FINGERPRINT $PY scripts/ag_gen_collect_ext.py --vehicle gator $COMMON --out $E/G2 2>&1 | tail -n 1; ls $E/G2 2>/dev/null | head -3
echo "== G3 gator, Gator fingerprint, gates on (expect a complete 4 s episode)"
$PY scripts/ag_gen_collect_ext.py --vehicle gator --runtime-fingerprint $E/../runtime/gator_runtime_fingerprint.json $COMMON --out $E/G3 2>&1 | tail -n 1
echo "== G4 NEDM_VEHICLE=gator (env switch) with FDM_RUNTIME_FINGERPRINT (expect a complete 4 s episode, Gator)"
NEDM_VEHICLE=gator FDM_RUNTIME_FINGERPRINT=$E/../runtime/gator_runtime_fingerprint.json $PY scripts/ag_gen_collect_ext.py $COMMON --out $E/G4 2>&1 | tail -n 1
echo "== G5 hmmwv with --gator-soil-radius-front (expect refusal)"
$PY scripts/ag_gen_collect_ext.py $COMMON --gator-soil-radius-front 0.2 --out $E/G5 2>&1 | tail -n 1
