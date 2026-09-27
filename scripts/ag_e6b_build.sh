#!/bin/bash
# E6b rigid evaluation rows for one suite set (arena_gator_20260925): lock the set's pick directories, build the rows
# (scripts/ag_eval_tasks.py build: every arm and vehicle of a group in ONE shard), stage routes/cases to G3, copy the
# task file read-only to G3/tasks and add it to G3/rigid_eval/TASKLIST.txt (read by the running eval pool steps).
#   bash scripts/ag_eval_... ; usage: bash scripts/ag_e6b_build.sh <unseen|f104|heldout|dev|f104B2> [--no-stage]
set -euo pipefail
cd "$(dirname "$0")/.."
SET=$1; STAGE=${2:-}
K3=artifacts/traverse/arena_gator_20260925
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
PY=/home/harry/miniconda3/envs/nedm/bin/python
export PYTHONPATH=src:scripts; unset NEDM_VEHICLE
P=$K3/e6/picks/rigid
T=$K3/e6/tasks; mkdir -p $T
EX="$K3/e3/tasks/rigid_v2.json $(ls $T/rigid_eval_*.json 2>/dev/null | grep -v "rigid_eval_${SET}.json" | grep -v '\.json\.' | tr '\n' ' ' || true)"
A=()
case $SET in
  unseen)  D="$P/g2[4-7]?"; BASE=3000; LOCKDIRS="$P/g2[4-7]?/*" ;;
  heldout) D="$P/g2[02][38]"; BASE=5000; LOCKDIRS="$P/g203/* $P/g228/*" ;;
  dev)     D="$P/g217"; BASE=5500; LOCKDIRS="$P/g217/*" ;;
  f104)    D="$P/f104"; BASE=4000; LOCKDIRS="$P/f104/*" ;;
  f104B2)  D="$K3/e6/picks/rigid_B2/f104"; BASE=6000; LOCKDIRS="$D/*" ;;
  *) echo "unknown set"; exit 2 ;;
esac
if [[ $SET == f104B2 ]]; then
  A+=(--arm "G_fx2_gator=$D/G_fixed2@gator" --arm "H_fx2_gator=$D/M1a_fixed2@gator" --arm "straight2_gator=$D/straight2@gator")
else
  for m in M1a M1b M2 M3a M3b A3; do A+=(--arm "${m}_fx2=$D/${m}_fixed2" --arm "${m}_free=$D/${m}_free"); done
  A+=(--arm "straight6=$D/straight6" --arm "straight2=$D/straight2")
  if [[ $SET == f104 ]]; then
    A+=(--arm "G_free_gator=$D/G_free@gator" --arm "H_free_gator=$D/M1a_free@gator" --arm "straight6_gator=$D/straight6@gator")
  fi
fi
$PY scripts/ag_e6b_lock.py --dirs $LOCKDIRS --out $K3/e6/picks/LOCK_${SET}
$PY scripts/ag_eval_tasks.py build --world rigid "${A[@]}" --existing $EX --out $T/rigid_eval_${SET}.json --shard-base $BASE \
  > $T/rigid_eval_${SET}.build.log
tail -n 5 $T/rigid_eval_${SET}.build.log
[[ $STAGE == --no-stage ]] && exit 0
$PY scripts/ag_eval_tasks.py stage --tasks $T/rigid_eval_${SET}.json --execute > $T/rigid_eval_${SET}.stage.log
cat $T/rigid_eval_${SET}.stage.log
rsync -a --ignore-existing $T/rigid_eval_${SET}.json $T/rigid_eval_${SET}.json.meta.json $T/rigid_eval_${SET}.json.mapping.json amd:$G3/tasks/
h=$(sha256sum $T/rigid_eval_${SET}.json | cut -d' ' -f1)
ssh amd "cd $G3/tasks && chmod a-w rigid_eval_${SET}.json && echo \"$h  rigid_eval_${SET}.json\" | sha256sum -c && mkdir -p $G3/rigid_eval/pool && touch $G3/rigid_eval/TASKLIST.txt && (grep -qx '$G3/tasks/rigid_eval_${SET}.json' $G3/rigid_eval/TASKLIST.txt || echo '$G3/tasks/rigid_eval_${SET}.json' >> $G3/rigid_eval/TASKLIST.txt) && cat $G3/rigid_eval/TASKLIST.txt"
echo "set $SET staged: rows $($PY -c "import json;print(len(json.load(open('$T/rigid_eval_${SET}.json'))))"), sha256 ${h:0:16}"
