#!/bin/bash
# Stage-1 screen, restart of the contact stage after the own-history fix (cores and collision networks kept),
# plus no-own-history controls (spacetime) and the REF arms. Run on the login node.
set -euo pipefail
R=/work1/dannegrut/harry/experiments/contact_nrd_20261002
C=$R/code_v5
O=$R/runs_v5
for d in $O/v5s_*_s61; do
  case $d in *ref*) rm -rf $d $d.stdout.log ;; *) rm -f $d/best.pt $d/last.pt $d/contact_scale.json ;; esac
done
for s in pool ball; do
  n=$O/v5s_spacetime_k4_nopush_${s}_s61
  mkdir -p $n
  cp $O/v5s_spacetime_k4_${s}_s61/{core.pt,collision.pt,run_config.json} $n/
done
J=""
g=0
for m in mlp_k4 time_k4 entity_k2 spacetime_k4; do
  for s in pool ball; do
    J="$J$R/data/${s}_train|$C/configs/contact_nrd/v5/v5s_${m}_s61.json|$O/v5s_${m}_${s}_s61|nedm.contact_nrd.train_v5|$g;"
    g=$((g + 1))
  done
done
J="$J$R/data/pool_train|$C/configs/contact_nrd/v5/v5s_spacetime_k4_nopush_s61.json|$O/v5s_spacetime_k4_nopush_pool_s61|nedm.contact_nrd.train_v5|0;"
J="$J$R/data/ball_train|$C/configs/contact_nrd/v5/v5s_spacetime_k4_nopush_s61.json|$O/v5s_spacetime_k4_nopush_ball_s61|nedm.contact_nrd.train_v5|1;"
J="$J$R/data/pool_train|$C/configs/contact_nrd/v5/v5s_ref_pool_s61.json|$O/v5s_ref_pool_s61|nedm.contact_nrd.train_v3|2;"
J="$J$R/data/ball_train|$C/configs/contact_nrd/v5/v5s_ref_ball_s61.json|$O/v5s_ref_ball_s61|nedm.contact_nrd.train_v3|3"
CN_CODE=$C CN_JOBS="$J" sbatch --parsable --export=ALL -o $O/pack8_s1b_%j.out -J cn-v5-s1b $C/scripts/contact_nrd/cluster/pack8_v5.sbatch
