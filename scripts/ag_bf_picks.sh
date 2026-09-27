#!/bin/bash
# arena_gator_20260925 task B stage 2 (Bfull): soil picks of the all-tier models on the 800-group f104 suite, speed free
# (CEM 4 x 64 from the case pose at rest, ag_picks.py --mode free = ci_planner.py --family free --world crm --arms B),
# made locally on the 5090 exactly as scripts/ag_s2_picks.sh f104 made the stage-1 G / H picks (same suite cases, same
# default f104 map root, map check first), plus --rerun-check (every group planned twice, identical files required).
#   bash scripts/ag_bf_picks.sh            -> $K3/e6/picks/crm_bfull/f104/{G_full_free,H_full_free}
# Models: $K3/e5/deploy/{G_full_soil,H_full_soil}/*_deploy_s*.pt (sha256 checked against SHA256SUMS first).
# H_full's picks serve both H_full on the Gator and H_full on the HMMWV (the planner does not know the vehicle), as the
# stage-1 M1a_free picks served H on both vehicles. The set lock (scripts/ag_e6b_lock.py) is written before any drive.
set -euo pipefail
cd "$(dirname "$0")/.."
K3=artifacts/traverse/arena_gator_20260925
PY=/home/harry/miniconda3/envs/nedm/bin/python
export PYTHONPATH=src:scripts OMP_NUM_THREADS=4
unset NEDM_VEHICLE
SUITE800=artifacts/traverse/generalist_20260921/A_adapt/suite/cases
B=$K3/e6/picks/crm_bfull/f104
mkdir -p $B $K3/e6/picks/logs
for m in G_full H_full; do (cd $K3/e5/deploy/${m}_soil && sha256sum -c --quiet SHA256SUMS) || { echo "sha256 mismatch in ${m}_soil"; exit 3; }; done
pids=()
for m in G_full H_full; do
  o=$B/${m}_free
  [[ -e $o/ag_picks.json ]] && { echo "exists (not re-planned): $o"; continue; }
  $PY scripts/ag_picks.py --arena f104 --world crm --mode free --cases $SUITE800 --groups all \
      --models "$K3/e5/deploy/${m}_soil/${m}_soil_deploy_s*.pt" --model-tag $m --out $o --rerun-check \
      > $K3/e6/picks/logs/crm_bfull_f104_${m}_free.log 2>&1 && echo "OK $o" || echo "FAIL $o" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
echo "Bfull picks done"
