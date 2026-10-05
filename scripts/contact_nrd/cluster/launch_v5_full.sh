#!/bin/bash
# Stage-2 full runs: space-time design, 20 ms and 10 ms, seeds 61-63, both systems (12 runs), two runs per
# single-GPU MI350 node, each node with a resume job behind it (the trainer resumes after the last saved point).
set -euo pipefail
R=/work1/dannegrut/harry/experiments/contact_nrd_20261002
C=$R/code_v5
O=$R/runs_v5
for tag in 20ms 10ms; do
  for seed in 61 62 63; do
    J=""
    for s in pool ball; do
      J="$J$R/data/${s}_train|$C/configs/contact_nrd/v5/v5f_st_${tag}_s${seed}.json|$O/v5f_st_${tag}_${s}_s${seed}|nedm.contact_nrd.train_v5|0;"
    done
    J=${J%;}
    a=$(CN_CODE=$C CN_JOBS="$J" sbatch --parsable --export=ALL -p mi3501x -c 24 -o $O/pack_full_${tag}_s${seed}_%j.out -J cn-v5f-$tag-$seed $C/scripts/contact_nrd/cluster/pack8_v5.sbatch)
    b=$(CN_CODE=$C CN_JOBS="$J" sbatch --parsable --export=ALL -p mi3501x -c 24 --dependency=afterany:$a -o $O/pack_full_${tag}_s${seed}_resume_%j.out -J cn-v5r-$tag-$seed $C/scripts/contact_nrd/cluster/pack8_v5.sbatch)
    echo "$tag s$seed: $a resume $b"
  done
done
