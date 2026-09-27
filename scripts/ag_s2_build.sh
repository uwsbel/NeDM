#!/bin/bash
# Soil track step 2 (arena_gator_20260925): soil evaluation rows (scripts/ag_eval_tasks.py build --world crm), one build
# per declared priority group, each with its own tier and every earlier build in --existing (identical routes are one
# drive, in the earliest group; soil_v2 drives such as the spread-arena straight 6 m/s headroom rows are reused by
# content). Then soil_v3.json = soil_v2 + all evaluation rows (scripts/ag_s2_soil_v3.py) and a 6-row check file.
# Unseen pick dirs are matched with crm/g2[4-7]* (g241 g247 g251 g258 g260 g263 g268 g271; not g203 g228 g217),
# f104 + held-out with crm/[fg][12][02]* (f104 g203 g228), held-out alone with crm/g2[02]* (g203 g228).
set -euo pipefail
cd "$(dirname "$0")/.."
K=artifacts/traverse/arena_gator_20260925
PY=/home/harry/miniconda3/envs/nedm/bin/python
export PYTHONPATH=src:scripts
unset NEDM_VEHICLE
P=$K/e6/picks/crm; T=$K/e6/tasks; V2=$K/e3/tasks/soil_v2.json
U="$P/g2[4-7]*"
b() { local n=$1; shift; $PY scripts/ag_eval_tasks.py build --world crm --out $T/soil_eval_$n.json "$@" 2>&1 | tee $T/soil_eval_$n.build.log | tail -3; }
b p1 --tier -9 --arm M1a_free="$U/M1a_free" --arm M1b_free="$U/M1b_free" --arm M3a_free="$U/M3a_free" --arm A3_free="$U/A3_free" \
      --arm straight6="$U/straight6" --arm G_free_gator="$P/f104/G_free@gator" --arm H_free_gator="$P/f104/M1a_free@gator" --existing $V2
b p2 --tier -8 --arm M3b_free="$U/M3b_free" --arm M2_free="$U/M2_free" --existing $V2 $T/soil_eval_p1.json
b p3 --tier -7 --arm straight6_gator="$P/f104/straight6@gator" --arm M1a_free="$P/f104/M1a_free" --arm straight6="$P/f104/straight6" \
      --existing $V2 $T/soil_eval_p1.json $T/soil_eval_p2.json
b p4 --tier -6 --arm M3a_free="$P/[fg][12][02]*/M3a_free" --arm A3_free="$P/[fg][12][02]*/A3_free" --arm M1a_free="$P/g2[02]*/M1a_free" \
      --existing $V2 $T/soil_eval_p1.json $T/soil_eval_p2.json $T/soil_eval_p3.json
b p5 --tier -5 --arm straight6="$P/g2[02]*/straight6" \
      --existing $V2 $T/soil_eval_p1.json $T/soil_eval_p2.json $T/soil_eval_p3.json $T/soil_eval_p4.json
$PY scripts/ag_s2_soil_v3.py --v2 $V2 --builds $T/soil_eval_p1.json $T/soil_eval_p2.json $T/soil_eval_p3.json $T/soil_eval_p4.json \
    $T/soil_eval_p5.json --out $K/e3/tasks/soil_v3.json --check-out $K/e3/tasks/soil_v3_check.json --check-n 3
