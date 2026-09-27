#!/bin/bash
# E2 task 4, rigid: Gator (ag_gen_collect_ext.py --vehicle gator) and HMMWV (unmodified gen_collect_ext.py) on the same
# 12 designed f104 routes: 4 hill/crater groups x constant 2 / 4 / 6 m/s at lateral offset 0; 120 s horizon, --local
# (no manifest/fingerprint gates on the workstation), single-threaded, 8 at a time.
cd /home/harry/NeDM-traverse_mppi
E=artifacts/traverse/arena_gator_20260925/e2/rigid_f104
C=$PWD/artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export PYTHONPATH=/home/harry/chrono/build/bin:src:scripts
jobs=()
for g in 0000 0005 0012 0017; do
  for r in 00 01 02; do
    for v in gator hmmwv; do jobs+=("$g $r $v"); done
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P 8 -L 1 bash -c '
g=$0; r=$1; v=$2; id=f104_v2_group_${g}_route_${r}; out='$E'/${v}/${id}
[ -f $out/episode_complete.json ] && exit 0
mkdir -p '$E'/${v}
if [ $v = gator ]; then S="scripts/ag_gen_collect_ext.py --vehicle gator"; else S=scripts/gen_collect_ext.py; fi
/usr/bin/python3.12 -P -u $S --source-root . --case '$C'/f104_v2_group_${g}.json --route '$C'/routes/f104_v2_group_${g}/route_${r}.json \
  --chrono-data /home/harry/chrono/data --horizon-s 120 --local --out $out > '$E'/${v}/${id}.log 2>&1 || echo "FAIL $id $v"'
