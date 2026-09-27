#!/bin/bash
# Soil track step 2 (arena_gator_20260925; PLAN 2.3, 3, 7.1-7.4, 7.7, 7.10): soil evaluation picks, speed free (CEM 4 x 64,
# standing start), made locally on the 5090 with scripts/ag_picks.py (NOTES_E6a section 4 recipe, world crm).
#   bash scripts/ag_s2_picks.sh <set> [parallel]      set = unseen | f104 | heldout
# Pick dirs: $K3/e6/picks/crm/<arena>/<model>_free and .../straight6. Models: $K3/e5/deploy/<M>_soil/<M>_soil_deploy_s*.pt
# (H = M1a_soil, e5/deploy/H_soil/README.md, so the f104 dir M1a_free on all 800 suite groups serves H and the
# in-distribution M1a arm). Every pick dir gets its own PICKS_LOCKED.sha256; the set lock (scripts/ag_e6b_lock.py) is
# written before any drive.
set -euo pipefail
cd "$(dirname "$0")/.."
SET=$1; PAR=${2:-8}
K3=artifacts/traverse/arena_gator_20260925
PY=/home/harry/miniconda3/envs/nedm/bin/python
export PYTHONPATH=src:scripts OMP_NUM_THREADS=2
unset NEDM_VEHICLE
SUITE800=artifacts/traverse/generalist_20260921/A_adapt/suite/cases
SUB=$K3/suites/soil_unseen_subset.json
JOBS=$K3/e6/picks/jobs_crm_${SET}.txt
mkdir -p $K3/e6/picks/logs
: > $JOBS
emit() {   # arena cases groups outdir mode [model]
  local a=$1 c=$2 g=$3 o=$4 mode=$5 m=${6:-}
  [[ -e $o/ag_picks.json ]] && { echo "skip (done): $o" >&2; return; }
  if [[ -n $m ]]; then
    echo "$PY scripts/ag_picks.py --arena $a --world crm --mode $mode --cases $c --groups $g --models '$K3/e5/deploy/${m}_soil/${m}_soil_deploy_s*.pt' --model-tag $m --out $o --force" >> $JOBS
  else
    echo "$PY scripts/ag_picks.py --arena $a --world crm --mode $mode --cases $c --groups $g --out $o --force" >> $JOBS
  fi
}
case $SET in
  unseen)   # task A: 8 unseen arenas x the declared 125-group soil subset; primary M1a M3a A3 straight6, then M3b M2 M1b
    for a in g260 g271 g251 g247 g258 g268 g263 g241; do
      b=$K3/e6/picks/crm/$a
      for m in M1a M3a A3 M3b M2 M1b; do emit $a $K3/cases/test_$a/cases $SUB $b/${m}_free free $m; done
      emit $a $K3/cases/test_$a/cases $SUB $b/straight6 straight6
    done ;;
  f104)     # task B: all 800 suite groups for M1a (= H), G and straight 6 m/s; task A in distribution: M3a, A3 on the 200
    b=$K3/e6/picks/crm/f104; I=$K3/suites/f104_indist_200.json
    emit f104 $SUITE800 all $b/M1a_free free M1a; emit f104 $SUITE800 all $b/G_free free G
    emit f104 $SUITE800 all $b/straight6 straight6
    for m in M3a A3; do emit f104 $SUITE800 $I $b/${m}_free free $m; done ;;
  heldout)  # task A in distribution on the new training arenas: 150 held-out groups each (+ straight 6 m/s, last tier)
    for a in g203 g228; do
      b=$K3/e6/picks/crm/$a
      for m in M1a M3a A3; do emit $a $K3/cases/heldout_$a/cases all $b/${m}_free free $m; done
      emit $a $K3/cases/heldout_$a/cases all $b/straight6 straight6
    done ;;
  *) echo "unknown set $SET"; exit 2 ;;
esac
echo "$(wc -l < $JOBS) pick runs for set $SET, $PAR in parallel; jobs $JOBS"
while IFS= read -r cmd; do
  o=$(echo "$cmd" | sed -E 's/.* --out ([^ ]+) .*/\1/'); n=$(echo "$o" | sed "s#$K3/e6/##; s#/#_#g")
  echo "$cmd > $K3/e6/picks/logs/$n.log 2>&1 && echo OK $o || echo FAIL $o"
done < $JOBS | xargs -d '\n' -P "$PAR" -I{} bash -c '{}'
echo "set $SET done"
