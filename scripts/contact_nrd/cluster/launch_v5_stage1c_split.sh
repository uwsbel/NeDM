#!/bin/bash
# Stage-1c ablations spread over four single-GPU MI350 nodes (3-4 runs share each GPU). Run on the login node.
set -euo pipefail
R=/work1/dannegrut/harry/experiments/contact_nrd_20261002
C=$R/code_v5
O=$R/runs_v5
rm -f $O/v5s_mlp_k4_ball_s61/contact_done.json
entry() {  # arm system
  local a=$1 s=$2 cfg name
  cfg=$C/configs/contact_nrd/v5/v5s_${a}_s61.json; name=v5s_${a}_${s}_s61
  if [[ $a == st_s62 ]]; then cfg=$C/configs/contact_nrd/v5/v5s_st_s62.json; name=v5s_st_s62_${s}; fi
  if [[ $a == mlp_k4 ]]; then cfg=$C/configs/contact_nrd/v5/v5s_mlp_k4_s61.json; fi
  echo "$R/data/${s}_train|$cfg|$O/$name|nedm.contact_nrd.train_v5|0"
}
groups=("st_k1:pool st_k1:ball st_noise0:pool st_noise0:ball"
        "st_k8:pool st_k8:ball st_noise3:pool"
        "st_20ms:pool st_20ms:ball st_noise3:ball"
        "st_s62:pool st_s62:ball mlp_k4:ball")
n=0
for grp in "${groups[@]}"; do
  J=""
  for item in $grp; do J="$J$(entry ${item%%:*} ${item##*:});"; done
  n=$((n + 1))
  CN_CODE=$C CN_JOBS="${J%;}" sbatch --parsable --export=ALL -p mi3501x -c 24 -o $O/pack1_s1c_${n}_%j.out -J cn-v5-s1c$n $C/scripts/contact_nrd/cluster/pack8_v5.sbatch
done
