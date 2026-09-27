#!/bin/bash
# arena_gator_20260925 task B stage 2 (Bfull): the training and evaluation files of G_full / H_full on ALL tiers
# (0-12), on the cluster login node, with the unchanged S1 tools (ag_e5a_ids.py, ag_subset.py) from G3/tools/bf.
# Same calls as scripts/ag_s1_subsets.sh gator, except --tiers 0-12 (every tier of the task file) instead of 0-6:
#   ids      validated Gator ids = the episodes in the all-tier Gator ci file (ag_e5a_ids.py on the gator__ rows of
#            soil_v3.json with tier >= 0, i.e. the 15,235 HMMWV collect_v1 ids)
#   G_full   Gator, every validated training group (f104=all)
#   H_full   HMMWV f104 soil rows (E4 file) on exactly the validated ids (--preset H, applied to every row)
#   F104all  the full HMMWV f104 soil file with no id filter (f104=all, all tiers): H_full is compared with it
#   EV_*     dev fold + val/test rows of each vehicle (holdout-mode scoring; ag_offline_auc.py's fitted-group guard)
# Each subset writes <out>.npz, then <out>.manifest.json (the training lanes wait for the manifest).
#   bash ag_bf_subsets.sh
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
T=$G3/tools/bf; S=$G3/e4/soil_bf/subsets; EV=$G3/e5/eval_soil_bf; L=$T/logs; ID=$G3/e5/ids_bf
source /work1/dannegrut/harry/nrd/env.sh >/dev/null 2>&1; nrd_pychrono >/dev/null 2>&1
unset NEDM_VEHICLE
cd $T; export PYTHONPATH=$T/src:$T/scripts:${PYTHONPATH:-}
mkdir -p $S $EV $L $ID
F104=$G3/e4/f104_hmmwv/ci_f104_hmmwv_both.npz
GAT=$G3/e4/soil_bf/f104_gator/ci_f104_gator_crm.npz
TC="--world crm --tiers 0-12 --eval-rows filtered"
sub() {  # name, then ag_subset arguments
  local n=$1; shift
  echo "$(date +%T) start $n: ag_subset.py $TC $*"
  nice -n 10 $NRD_PYTHON -u scripts/ag_subset.py $TC "$@" > $L/subset_$n.log 2>&1; local rc=$?
  echo "$(date +%T) end $n exit: $rc ($(grep -h '^wrote' $L/subset_$n.log))"
}
nice -n 10 $NRD_PYTHON -u scripts/ag_e5a_ids.py --world crm --ci $GAT --record $G3/e4/soil_bf/f104_gator/f104_gator_crm_record.json \
  --tasks $G3/tasks/soil_v3.json --runs $G3/soil_v1/runs --out-ids $ID/gator_soil_validated_all.txt \
  --out-json $ID/gator_soil_validated_all.json > $L/gator_soil_ids_all.log 2>&1
echo "$(date +%T) ids exit: $? ($(grep -h '"validated"' $L/gator_soil_ids_all.log))"
sub G_full_soil --groups f104=all --ds $GAT --ids-file $ID/gator_soil_validated_all.txt --out $S/G_full_f104_gator_soil.npz &
sub H_full_soil --preset H --ds $F104 --ids-file $ID/gator_soil_validated_all.txt --out $S/H_full_f104_hmmwv_soil.npz &
sub F104all_soil --groups f104=all --ds $F104 --out $S/F104all_f104_hmmwv_soil.npz &
sub EV_gator_full --groups f104=0 --keep-dev-fold --ds $GAT --out $EV/EV_f104_gator_full_crm.npz &
sub EV_f104H_full --groups f104=0 --keep-dev-fold --ds $F104 --ids-file $ID/gator_soil_validated_all.txt --out $EV/EV_f104_hmmwv_gatorids_full_crm.npz &
wait
nice -n 10 $NRD_PYTHON -u scripts/ag_e5a_ids.py --compare $S/F104all_f104_hmmwv_soil.npz $S/H_full_f104_hmmwv_soil.npz --out-json $ID/H_full_vs_F104all_soil.json > $L/H_full_vs_F104all.log 2>&1
echo "$(date +%T) H_full vs F104all: $(tr -d '\n' < $L/H_full_vs_F104all.log | cut -c1-600)"
sha256sum $S/*.npz $EV/*.npz
echo "$(date +%T) Bfull subsets done"
