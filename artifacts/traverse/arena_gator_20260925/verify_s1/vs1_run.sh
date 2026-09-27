#!/bin/bash
# VERIFY_S1: extract light columns + sha256 of every soil stage-1 npz, run records of every soil run (login node, read only)
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925; T=$G3/tools/verify_s1
source /work1/dannegrut/harry/nrd/env.sh >/dev/null 2>&1; nrd_pychrono >/dev/null 2>&1
unset NEDM_VEHICLE
mkdir -p $T/out
F="$G3/e4/f104_hmmwv/ci_f104_hmmwv_both.npz $G3/e4/soil_s1/g203_hmmwv/ci_g203_hmmwv_crm.npz $G3/e4/soil_s1/g228_hmmwv/ci_g228_hmmwv_crm.npz $G3/e4/soil_s1/f104_gator/ci_f104_gator_crm.npz $(ls $G3/e4/soil_s1/subsets/*.npz) $(ls $G3/e5/eval_soil/*.npz)"
( nice -n 15 ionice -c3 $NRD_PYTHON -u $T/vs1_runs.py $T/out/runs_soil_v1.jsonl $G3/soil_v1/runs > $T/out_runs1.log 2>&1; echo "runs1 exit $?" >> $T/out_runs1.log ) &
( nice -n 15 ionice -c3 $NRD_PYTHON -u $T/vs1_runs.py $T/out/runs_collect_v1.jsonl /work1/dannegrut/harry/experiments/crm_f104_20260916/collect_v1/runs > $T/out_runs2.log 2>&1; echo "runs2 exit $?" >> $T/out_runs2.log ) &
nice -n 15 ionice -c3 $NRD_PYTHON -u $T/vs1_extract_light.py $T/out $F > $T/out_extract.log 2>&1
echo "extract exit $?" >> $T/out_extract.log
nice -n 15 ionice -c3 sha256sum $F > $T/out/sha256_npz.txt 2>&1
( cd $G3/e5/train/soil_s1 && nice -n 15 sha256sum */* ) > $T/out/sha256_train.txt 2>&1
echo "sha exit $?" >> $T/out_extract.log
wait
echo ALL DONE >> $T/out_extract.log
