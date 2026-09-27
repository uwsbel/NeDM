#!/bin/bash
# VERIFY_gator_full: read-only extraction on login1 (run records of every f104 suite drive, light columns + sha256 of the
# task B stage-2 npz files, sacct of the stage-2 jobs). Writes only G3/tools/verify_bf/.
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925; T=$G3/tools/verify_bf
source /work1/dannegrut/harry/nrd/env.sh >/dev/null 2>&1; nrd_pychrono >/dev/null 2>&1
unset NEDM_VEHICLE
mkdir -p $T/out
F="$G3/e4/soil_bf/subsets/G_full_f104_gator_soil.npz $G3/e4/soil_bf/subsets/H_full_f104_hmmwv_soil.npz $G3/e4/soil_bf/subsets/F104all_f104_hmmwv_soil.npz $G3/e4/soil_bf/f104_gator/ci_f104_gator_crm.npz $G3/e4/f104_hmmwv/ci_f104_hmmwv_both.npz $G3/e4/soil_s1/subsets/G_f104_gator_soil.npz"
( nice -n 15 ionice -c3 $NRD_PYTHON -u $T/ag_vbf_cluster.py runs $T/out $G3/soil_v1/runs > $T/out_runs.log 2>&1; echo "runs exit $?" >> $T/out_runs.log ) &
nice -n 15 ionice -c3 $NRD_PYTHON -u $T/ag_vbf_cluster.py light $T/out $F > $T/out_light.log 2>&1
echo "light exit $?" >> $T/out_light.log
nice -n 15 ionice -c3 sha256sum $F $G3/e5/eval_soil_bf/*.npz $G3/tasks/soil_v3.json $G3/tasks/soil_v4.json $G3/e5/ids_bf/gator_soil_validated_all.txt > $T/out/sha256.txt 2>&1
( cd $G3/e5/train/soil_bf && nice -n 15 sha256sum */*.pt ) > $T/out/sha256_train.txt 2>&1
sacct -j 439361,439362,439414,439415,439426 -X -P --format=JobID,JobName,Partition,NNodes,AllocTRES,Start,End,Elapsed,State,ExitCode > $T/out/sacct.txt 2>&1
wait
echo ALL DONE >> $T/out_light.log
