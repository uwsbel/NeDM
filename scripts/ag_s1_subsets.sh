#!/bin/bash
# arena_gator_20260925 soil stage 1 (S1, PLAN 7.10): soil training subsets (scripts/ag_subset.py, lowest md5 of the
# group id; --tiers 0-6 --eval-rows filtered, so training AND val/test rows are tiers 0-6 on every arena) and the
# evaluation files of the offline read-out, on the login node. Usage: ag_s1_subsets.sh f104|new|gator
#   f104:  M1 and the f104 learning-curve files (LC272, LC545) from the existing f104 HMMWV file cut to tiers 0-6, and
#          the f104 evaluation file
#   new:   M2 (f104 545 + g203 545), M3 (363 x 3), A3 (f104 1,089 + every g203 + every g228 training group),
#          leave-one-arena-out holdout files (LOAO1_g203 545; LOAO2 273 + 272; g228 has only 520 training groups, so
#          no LOAO1_g228 at 545), evaluation files of g203 and g228
#   gator: validated Gator ids of tiers 0-6 (ag_e5a_ids.py on a tier 0-6 copy of the Gator task rows), G (Gator,
#          every validated training group), H (HMMWV f104 on exactly the validated ids, every row incl. val/test),
#          the Gator evaluation file, and the H-vs-M1 comparison
# Each subset writes <out>.npz, then <out>.manifest.json (the training lanes wait for the manifest).
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
T=$G3/tools/s1; S=$G3/e4/soil_s1/subsets; EV=$G3/e5/eval_soil; L=$T/logs; ID=$G3/e5/ids
source /work1/dannegrut/harry/nrd/env.sh >/dev/null 2>&1; nrd_pychrono >/dev/null 2>&1
unset NEDM_VEHICLE
cd $T; export PYTHONPATH=$T/src:$T/scripts:${PYTHONPATH:-}
mkdir -p $S $EV $L $ID
F104=$G3/e4/f104_hmmwv/ci_f104_hmmwv_both.npz
G203=$G3/e4/soil_s1/g203_hmmwv/ci_g203_hmmwv_crm.npz
G228=$G3/e4/soil_s1/g228_hmmwv/ci_g228_hmmwv_crm.npz
GAT=$G3/e4/soil_s1/f104_gator/ci_f104_gator_crm.npz
TC="--world crm --tiers 0-6 --eval-rows filtered"
sub() {  # name, then ag_subset arguments
  local n=$1; shift
  echo "$(date +%T) start $n: ag_subset.py $TC $*"
  nice -n 10 $NRD_PYTHON -u scripts/ag_subset.py $TC "$@" > $L/subset_$n.log 2>&1; local rc=$?
  echo "$(date +%T) end $n exit: $rc ($(grep -h '^wrote' $L/subset_$n.log))"
}
case $1 in
  f104)
    sub M1_soil --preset M1 --ds $F104 --out $S/M1_f104_hmmwv_soil.npz &
    sub LC545_soil --preset LC545 --ds $F104 --out $S/LC545_f104_hmmwv_soil.npz &
    sub LC272_soil --preset LC272 --ds $F104 --out $S/LC272_f104_hmmwv_soil.npz &
    sub EV_f104_soil --groups f104=0 --keep-dev-fold --ds $F104 --out $EV/EV_f104_hmmwv_crm.npz &
    wait;;
  new)
    sub M2_soil --preset M2 --ds $F104 $G203 --out $S/M2_hmmwv_soil.npz &
    sub M3_soil --preset M3 --ds $F104 $G203 $G228 --out $S/M3_hmmwv_soil.npz &
    sub A3_soil --preset A3 --ds $F104 $G203 $G228 --out $S/A3_hmmwv_soil.npz &
    sub LOAO1_g203_soil --preset LOAO1_g203 --ds $G203 --out $S/LOAO1_g203_hmmwv_soil.npz &
    sub LOAO2_f104_g203_soil --preset LOAO2_f104_g203 --ds $F104 $G203 --out $S/LOAO2_f104_g203_hmmwv_soil.npz &
    sub LOAO2_f104_g228_soil --preset LOAO2_f104_g228 --ds $F104 $G228 --out $S/LOAO2_f104_g228_hmmwv_soil.npz &
    sub LOAO2_g203_g228_soil --preset LOAO2_g203_g228 --ds $G203 $G228 --out $S/LOAO2_g203_g228_hmmwv_soil.npz &
    sub EV_g203_soil --groups g203=0 --keep-dev-fold --ds $G203 --out $EV/EV_g203_hmmwv_crm.npz &
    sub EV_g228_soil --groups g228=0 --keep-dev-fold --ds $G228 --out $EV/EV_g228_hmmwv_crm.npz &
    wait;;
  gator)
    python3 - <<PY
import json
rows = [r for r in json.load(open('$G3/tasks/soil_v2.json')) if r['id'].startswith('gator__') and 0 <= int(r.get('tier', -1)) <= 6]
json.dump(rows, open('$ID/soil_v2_gator_tiers0-6.json', 'w'))
print('gator task rows tiers 0-6:', len(rows))
PY
    nice -n 10 $NRD_PYTHON -u scripts/ag_e5a_ids.py --world crm --ci $GAT --record $G3/e4/soil_s1/f104_gator/f104_gator_crm_record.json \
      --tasks $ID/soil_v2_gator_tiers0-6.json --runs $G3/soil_v1/runs --out-ids $ID/gator_soil_validated_t0-6.txt \
      --out-json $ID/gator_soil_validated_t0-6.json > $L/gator_soil_ids.log 2>&1
    echo "$(date +%T) ids exit: $? ($(grep -h '"validated"' $L/gator_soil_ids.log))"
    sub G_soil --groups f104=all --ds $GAT --ids-file $ID/gator_soil_validated_t0-6.txt --out $S/G_f104_gator_soil.npz &
    sub H_soil --preset H --ds $F104 --ids-file $ID/gator_soil_validated_t0-6.txt --out $S/H_f104_hmmwv_soil.npz &
    sub EV_gator_soil --groups f104=0 --keep-dev-fold --ds $GAT --out $EV/EV_f104_gator_crm.npz &
    sub EV_f104H_soil --groups f104=0 --keep-dev-fold --ds $F104 --ids-file $ID/gator_soil_validated_t0-6.txt --out $EV/EV_f104_hmmwv_gatorids_crm.npz &
    wait
    nice -n 10 $NRD_PYTHON -u scripts/ag_e5a_ids.py --compare $S/M1_f104_hmmwv_soil.npz $S/H_f104_hmmwv_soil.npz --out-json $ID/H_vs_M1_soil.json > $L/H_vs_M1_soil.log 2>&1
    echo "$(date +%T) H vs M1: $(tr -d '\n' < $L/H_vs_M1_soil.log | cut -c1-600)";;
  *) echo "usage: $0 f104|new|gator"; exit 2;;
esac
echo "$(date +%T) stage $1 done"
