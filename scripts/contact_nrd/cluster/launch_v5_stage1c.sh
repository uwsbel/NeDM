#!/bin/bash
# Stage-1c ablations on the space-time design (both systems) + the MLP-control ball rerun. Run on the login node.
# Usage: launch_v5_stage1c.sh [dependency_job_id]
set -euo pipefail
R=/work1/dannegrut/harry/experiments/contact_nrd_20261002
C=$R/code_v5
O=$R/runs_v5
rm -f $O/v5s_mlp_k4_ball_s61/contact_done.json
J=""
g=0
for a in st_k1 st_k8 st_noise0 st_noise3 st_20ms st_s62; do
  cfg=$C/configs/contact_nrd/v5/v5s_${a}_s61.json; [[ $a == st_s62 ]] && cfg=$C/configs/contact_nrd/v5/v5s_st_s62.json
  for s in pool ball; do
    name=v5s_${a}_${s}; [[ $a != st_s62 ]] && name=${name}_s61
    J="$J$R/data/${s}_train|$cfg|$O/$name|nedm.contact_nrd.train_v5|$(( g % 8 ));"
    g=$((g + 1))
  done
done
J="$J$R/data/ball_train|$C/configs/contact_nrd/v5/v5s_mlp_k4_s61.json|$O/v5s_mlp_k4_ball_s61|nedm.contact_nrd.train_v5|$(( g % 8 ))"
dep=${1:+--dependency=afterany:$1}
CN_CODE=$C CN_JOBS="$J" sbatch --parsable --export=ALL $dep -o $O/pack8_s1c_%j.out -J cn-v5-s1c $C/scripts/contact_nrd/cluster/pack8_v5.sbatch
