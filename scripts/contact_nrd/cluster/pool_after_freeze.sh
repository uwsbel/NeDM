#!/bin/bash
# Pool: freeze, fresh cohort (4 array tasks + pack), convert, certification, targeting, exploratory mirrored shots.
# Run on the login node: bash pool_after_freeze.sh <deploy_run_name>
set -euo pipefail
R=/work1/dannegrut/harry/experiments/contact_nrd_20261002
E=/work1/dannegrut/harry/experiments
DEPLOY=$1
C=$R/code_eval_pool
[[ -d $C ]] || cp -a $R/code_v1 $C
RUNS=()
for r in v2_pool_centerline_s61 v2_pool_approach_s61 v2_pool_approach_exactdecay_s61 \
         v2_pool_centerline_exactdecay_s61 v2_pool_centerline_exactdecay_s62 v2_pool_centerline_exactdecay_s63; do RUNS+=("$R/runs_v2/$r"); done
RUNS+=("$R/runs/pool_world" "$R/runs/pool_pair")
P=$E/pool_ball_20261002/frozen
REFS=(per_cushion=$P/structured4_refcore_rr10/best.pt shared_cushion=$P/structured_pb_v4_refcore_rr10/best.pt
      literal_port=$P/scalar_joint_v4/best.pt transformer_only=$P/none_joint/best.pt)
cd $C
PYTHONPATH=src python3 -m nedm.contact_nrd.freeze --root $R --name pool --unified "${RUNS[@]}" --reference "${REFS[@]}" \
  --cohort pool=$C/configs/pool_ball/fresh_test_v2.json:4
F=$R/frozen_pool
FROZEN=(); for r in "${RUNS[@]}"; do FROZEN+=("$F/$(basename $r)"); done
mkdir -p $R/fresh $R/results
j1=$(POOL_CODE=$C POOL_CAMPAIGN=$F/cohort_pool.json POOL_DATA=$R/fresh/pool_raw POOL_PER=30 \
  sbatch --parsable --export=ALL --array=0-3 -c 24 -t 01:00:00 -J cn-fresh-pool $C/scripts/contact_nrd/cluster/pool_collect.sbatch)
j2=$(POOL_CODE=$C POOL_CAMPAIGN=$F/cohort_pool.json POOL_DATA=$R/fresh/pool_raw POOL_PACK=1 \
  sbatch --parsable --export=ALL --dependency=afterok:$j1 -c 8 -t 00:30:00 -J cn-pack-pool $C/scripts/contact_nrd/cluster/pool_collect.sbatch)
j3=$(CN_CODE=$C sbatch --parsable --export=ALL --dependency=afterok:$j2 -t 00:30:00 -J cn-conv-pool $C/scripts/contact_nrd/cluster/convert.sbatch \
  pool --source $R/fresh/pool_raw --output $R/fresh/pool_unified)
j4=$(CN_CODE=$C sbatch --parsable --export=ALL --dependency=afterok:$j2 -t 01:00:00 -J cn-cert-pool $C/scripts/contact_nrd/cluster/py.sbatch \
  nedm.contact_nrd.certify_shared --system pool --data $R/fresh/pool_raw --unified "${FROZEN[@]}" --reference "${REFS[@]}" \
  --headline $DEPLOY --comparison per_cushion --output $R/results/pool_certification.json)
j5=$(CN_CODE=$C CN_SYSTEM=pool CN_CHECKPOINT=$F/$DEPLOY/best.pt CN_TRAIN=$R/data/pool_train CN_TARGETS=$R/fresh/pool_unified \
  CN_TCONFIG=$C/configs/contact_nrd/target_pool.json CN_TRUN=$R/results/pool_targeting \
  sbatch --parsable --export=ALL --dependency=afterok:$j3 -J cn-target-pool $C/scripts/contact_nrd/cluster/target.sbatch)
j6=$(CN_CODE=$C sbatch --parsable --export=ALL -t 00:30:00 -J cn-ood-pool $C/scripts/contact_nrd/cluster/py.sbatch \
  nedm.contact_nrd.ood_eval --data $R/ood/pool_mirrored_unified --unified "${FROZEN[@]}" --reference "${REFS[@]}" --output $R/results/pool_ood_mirrored.json)
echo "collect $j1 pack $j2 convert $j3 certify $j4 target $j5 ood $j6"
