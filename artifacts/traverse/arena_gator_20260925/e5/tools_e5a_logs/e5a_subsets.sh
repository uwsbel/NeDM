#!/bin/bash
# arena_gator_20260925 E5a: rigid training subsets (scripts/ag_subset.py, lowest md5 of the group id, all 20 tiers) and the
# evaluation files of the offline read-out, on the login node. Usage: e5a_subsets.sh g203|g228|gator  (stage)
# The subsets of one stage are written in parallel; each writes <out>.npz, then <out>.manifest.json (the training
# lanes wait for the manifest).
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
T=$G3/tools/e5a; S=$G3/e4/subsets; EV=$G3/e5/eval; L=$T/logs
source /work1/dannegrut/harry/nrd/env.sh >/dev/null 2>&1; nrd_pychrono >/dev/null 2>&1
unset NEDM_VEHICLE
cd $T; export PYTHONPATH=$T/src:$T/scripts:${PYTHONPATH:-}
mkdir -p $S $EV $L
F104=$G3/e4/f104_hmmwv/ci_f104_hmmwv_both.npz
G203=$G3/e4/g203_hmmwv/ci_g203_hmmwv_rigid.npz
G228=$G3/e4/g228_hmmwv/ci_g228_hmmwv_rigid.npz
GAT=$G3/e4/f104_gator/ci_f104_gator_rigid.npz
sub() {  # name, then ag_subset arguments
  local n=$1; shift
  echo "$(date +%T) start $n: ag_subset.py $*"
  nice -n 10 $NRD_PYTHON -u scripts/ag_subset.py --world rigid "$@" > $L/subset_$n.log 2>&1; local rc=$?
  echo "$(date +%T) end $n exit: $rc ($(grep -h '^wrote' $L/subset_$n.log))"
}
case $1 in
  g203)
    sub M2 --preset M2 --ds $F104 $G203 --out $S/M2_hmmwv_rigid.npz &
    sub LOAO1_g203 --preset LOAO1_g203 --ds $G203 --out $S/LOAO1_g203_hmmwv_rigid.npz &
    sub LOAO2_f104_g203 --preset LOAO2_f104_g203 --ds $F104 $G203 --out $S/LOAO2_f104_g203_hmmwv_rigid.npz &
    sub EV_g203 --groups g203=0 --keep-dev-fold --ds $G203 --out $EV/EV_g203_hmmwv_rigid.npz &
    sub LOAO1_f104_list --preset LOAO1_f104 --ds $F104 --list-only --out $S/LOAO1_f104_hmmwv_rigid.npz &
    wait;;
  g228)
    sub M3 --preset M3 --ds $F104 $G203 $G228 --out $S/M3_hmmwv_rigid.npz &
    sub A3 --preset A3 --ds $F104 $G203 $G228 --out $S/A3_hmmwv_rigid.npz &
    sub LOAO1_g228 --preset LOAO1_g228 --ds $G228 --out $S/LOAO1_g228_hmmwv_rigid.npz &
    sub LOAO2_f104_g228 --preset LOAO2_f104_g228 --ds $F104 $G228 --out $S/LOAO2_f104_g228_hmmwv_rigid.npz &
    sub LOAO2_g203_g228 --preset LOAO2_g203_g228 --ds $G203 $G228 --out $S/LOAO2_g203_g228_hmmwv_rigid.npz &
    sub EV_g228 --groups g228=0 --keep-dev-fold --ds $G228 --out $EV/EV_g228_hmmwv_rigid.npz &
    wait;;
  gator)
    mkdir -p $G3/e5/ids
    nice -n 10 $NRD_PYTHON -u scripts/ag_e5a_ids.py --ci $GAT --record $G3/e4/f104_gator/f104_gator_rigid_record.json --tasks $G3/tasks/rigid_v2.json \
      --runs $G3/rigid_v1/runs --out-ids $G3/e5/ids/gator_rigid_validated.txt --out-json $G3/e5/ids/gator_rigid_validated.json > $L/gator_ids.log 2>&1
    echo "$(date +%T) ids exit: $? ($(grep -h '"validated"' $L/gator_ids.log))"
    sub G --groups f104=all --ds $GAT --out $S/G_f104_gator_rigid.npz &
    sub H --preset H --ds $F104 --ids-file $G3/e5/ids/gator_rigid_validated.txt --out $S/H_f104_hmmwv_rigid.npz &
    sub EV_gator --groups f104=0 --keep-dev-fold --ds $GAT --out $EV/EV_f104_gator_rigid.npz &
    wait
    nice -n 10 $NRD_PYTHON -u scripts/ag_e5a_ids.py --compare $S/M1_f104_hmmwv_rigid.npz $S/H_f104_hmmwv_rigid.npz --out-json $G3/e5/ids/H_vs_M1.json > $L/H_vs_M1.log 2>&1
    echo "$(date +%T) H vs M1: $(tr -d '\n' < $L/H_vs_M1.log | cut -c1-600)";;
  *) echo "usage: $0 g203|g228|gator"; exit 2;;
esac
echo "$(date +%T) stage $1 done"
