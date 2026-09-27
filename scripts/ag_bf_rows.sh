#!/bin/bash
# arena_gator_20260925 task B stage 2 (Bfull): soil drive rows of the all-tier planners (scripts/ag_eval_tasks.py build
# --world crm, unchanged), one build per declared priority group with its own tier, soil_v3 (and the earlier build) in
# --existing, so every stage-1 arm listed here resolves to its existing stage-1 drive by route content and any new pick
# equal to an already-driven route (same group, vehicle, case and route content) is not driven again:
#   bf1, tier -9 (Gator, spec F1-F4): G_full and H_full on the Gator (new), + stage-1 G, H (= M1a picks) and straight
#        6 m/s on the Gator (reused)
#   bf2, tier -8 (HMMWV, secondary): H_full on the HMMWV (new; the H_full picks), + stage-1 H (= M1a) and straight 6 m/s on
#        the HMMWV (reused)
# Then soil_v4.json = soil_v3 + both builds (scripts/ag_bf_soil_v4.py) and a 4-row check file; stage with --execute.
#   bash scripts/ag_bf_rows.sh
set -euo pipefail
cd "$(dirname "$0")/.."
K=artifacts/traverse/arena_gator_20260925
PY=/home/harry/miniconda3/envs/nedm/bin/python
export PYTHONPATH=src:scripts
unset NEDM_VEHICLE
P=$K/e6/picks/crm/f104; PB=$K/e6/picks/crm_bfull/f104; T=$K/e6/tasks; V3=$K/e3/tasks/soil_v3.json
b() { local n=$1; shift; $PY scripts/ag_eval_tasks.py build --world crm --out $T/soil_eval_$n.json "$@" 2>&1 | tee $T/soil_eval_$n.build.log | tail -3; }
b bf1 --tier -9 --arm Gfull_free_gator="$PB/G_full_free@gator" --arm Hfull_free_gator="$PB/H_full_free@gator" \
      --arm G_free_gator="$P/G_free@gator" --arm H_free_gator="$P/M1a_free@gator" --arm straight6_gator="$P/straight6@gator" --existing $V3
b bf2 --tier -8 --arm Hfull_free="$PB/H_full_free" --arm M1a_free="$P/M1a_free" --arm straight6="$P/straight6" \
      --existing $V3 $T/soil_eval_bf1.json
$PY scripts/ag_bf_soil_v4.py --v3 $V3 --builds $T/soil_eval_bf1.json $T/soil_eval_bf2.json --out $K/e3/tasks/soil_v4.json \
    --check-out $K/e3/tasks/soil_v4_check.json --check-n 2
for n in bf1 bf2; do $PY scripts/ag_eval_tasks.py stage --tasks $T/soil_eval_$n.json --execute 2>&1 | tee $T/soil_eval_$n.stage.log | tail -3; done
