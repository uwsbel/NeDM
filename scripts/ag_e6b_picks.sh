#!/bin/bash
# E6b rigid evaluation picks (arena_gator_20260925; PLAN 2.2-2.3, 3, 7.1-7.7; NOTES_E6a section 4 recipe).
# Writes one ag_picks.py command per (arena, arm) for a suite set and runs them in parallel on the local 5090.
#   bash scripts/ag_e6b_picks.sh <set> [parallel]      set = unseen | f104 | heldout | dev | f104B2 (task B fixed 2 m/s)
# Pick dirs: $K3/e6/picks/rigid/<arena>/<model>_<free|fixed2> and straight6 / straight2 (task B fixed 2 m/s arms:
# $K3/e6/picks/rigid_B2/f104/...). Every pick dir gets its own PICKS_LOCKED.sha256 (ag_picks.py); the set's lock of all
# dirs is written by scripts/ag_e6b_lock.py before any drive. Models: $K3/e5/deploy/<model>/<model>_rigid_deploy_s*.pt.
set -euo pipefail
cd "$(dirname "$0")/.."
SET=$1; PAR=${2:-8}
K3=artifacts/traverse/arena_gator_20260925
PY=/home/harry/miniconda3/envs/nedm/bin/python
export PYTHONPATH=src:scripts OMP_NUM_THREADS=2
unset NEDM_VEHICLE
SUITE800=artifacts/traverse/generalist_20260921/A_adapt/suite/cases
MODELS_A="M1a M1b M2 M3a M3b A3"
JOBS=$K3/e6/picks/jobs_${SET}.txt
mkdir -p $K3/e6/picks/logs
: > $JOBS
emit() {   # arena cases groups outdir mode [model]
  local a=$1 c=$2 g=$3 o=$4 mode=$5 m=${6:-}
  [[ -e $o/ag_picks.json ]] && { echo "skip (done): $o" >&2; return; }
  if [[ -n $m ]]; then
    echo "$PY scripts/ag_picks.py --arena $a --world rigid --mode $mode --cases $c --groups $g --models '$K3/e5/deploy/$m/${m}_rigid_deploy_s*.pt' --model-tag $m --out $o --force" >> $JOBS
  else
    echo "$PY scripts/ag_picks.py --arena $a --world rigid --mode $mode --cases $c --groups $g --out $o --force" >> $JOBS
  fi
}
armset() {  # arena cases groups base   (task A: 6 models x free/fixed2 + straight6 + straight2)
  local a=$1 c=$2 g=$3 b=$4
  for mode in fixed2 free; do for m in $MODELS_A; do emit $a $c $g $b/${m}_$mode $mode $m; done; done
  emit $a $c $g $b/straight6 straight6; emit $a $c $g $b/straight2 straight2
}
case $SET in
  unseen)
    for a in g260 g271 g251 g247 g258 g268 g263 g241; do armset $a $K3/cases/test_$a/cases all $K3/e6/picks/rigid/$a; done ;;
  f104)
    # task A in distribution: the 200 declared f104 hill/crater groups; task B: all 800 suite groups for M1a free (= H),
    # G free and straight 6 m/s (the 200 are a subset of the 800, planned once in the same dirs)
    b=$K3/e6/picks/rigid/f104; I=$K3/suites/f104_indist_200.json
    emit f104 $SUITE800 all $b/M1a_free free M1a; emit f104 $SUITE800 all $b/G_free free G; emit f104 $SUITE800 all $b/straight6 straight6
    for m in $MODELS_A; do emit f104 $SUITE800 $I $b/${m}_fixed2 fixed2 $m; done
    for m in M1b M2 M3a M3b A3; do emit f104 $SUITE800 $I $b/${m}_free free $m; done
    emit f104 $SUITE800 $I $b/straight2 straight2 ;;
  heldout)
    for a in g203 g228; do armset $a $K3/cases/heldout_$a/cases all $K3/e6/picks/rigid/$a; done ;;
  dev)
    armset g217 $K3/cases/dev_g217/cases all $K3/e6/picks/rigid/g217 ;;
  f104B2)
    b=$K3/e6/picks/rigid_B2/f104
    emit f104 $SUITE800 all $b/G_fixed2 fixed2 G; emit f104 $SUITE800 all $b/M1a_fixed2 fixed2 M1a; emit f104 $SUITE800 all $b/straight2 straight2 ;;
  *) echo "unknown set $SET"; exit 2 ;;
esac
echo "$(wc -l < $JOBS) pick runs for set $SET, $PAR in parallel; jobs $JOBS"
i=0
while IFS= read -r cmd; do
  i=$((i+1)); o=$(echo "$cmd" | sed -E 's/.* --out ([^ ]+) .*/\1/'); n=$(echo "$o" | sed "s#$K3/e6/##; s#/#_#g")
  echo "$cmd > $K3/e6/picks/logs/$n.log 2>&1 && echo OK $o || echo FAIL $o"
done < $JOBS | xargs -d '\n' -P "$PAR" -I{} bash -c '{}'
echo "set $SET done"
