#!/bin/bash
# Stage-1 screen: 4 mixers x 2 systems with the version-5 recipe, plus the old best state-only recipe (REF) on the
# same short schedule; all on one 8-GPU node. Run on the login node.
set -euo pipefail
R=/work1/dannegrut/harry/experiments/contact_nrd_20261002
C=$R/code_v5
O=$R/runs_v5
mkdir -p $O
J=""
g=0
for m in mlp_k4 time_k4 entity_k2 spacetime_k4; do
  for s in pool ball; do
    J="$J$R/data/${s}_train|$C/configs/contact_nrd/v5/v5s_${m}_s61.json|$O/v5s_${m}_${s}_s61|nedm.contact_nrd.train_v5|$g;"
    g=$((g + 1))
  done
done
J="$J$R/data/pool_train|$C/configs/contact_nrd/v5/v5s_ref_pool_s61.json|$O/v5s_ref_pool_s61|nedm.contact_nrd.train_v3|0;"
J="$J$R/data/ball_train|$C/configs/contact_nrd/v5/v5s_ref_ball_s61.json|$O/v5s_ref_ball_s61|nedm.contact_nrd.train_v3|1"
echo "$J" | tr ';' '\n'
CN_CODE=$C CN_JOBS="$J" sbatch --parsable --export=ALL -o $O/pack8_s1_%j.out -J cn-v5-s1 $C/scripts/contact_nrd/cluster/pack8_v5.sbatch
