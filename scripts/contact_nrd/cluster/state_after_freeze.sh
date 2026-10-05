#!/bin/bash
# State-only study: freeze, fresh cohort (CPU nodes), certification against the deployed pair-feature model.
# Run on the login node: bash state_after_freeze.sh <ball|pool> <headline_run> <run_name>...   (run names under runs_v3)
set -euo pipefail
R=/work1/dannegrut/harry/experiments/contact_nrd_20261002
E=/work1/dannegrut/harry/experiments
SYS=$1; HEAD=$2; shift 2
C=$R/code_eval_state_$SYS
[[ -d $C ]] || cp -a $R/code_v1 $C
RUNS=(); for r in "$@"; do RUNS+=("$R/runs_v3/$r"); done
if [[ $SYS == ball ]]; then
  DEPLOYED=$R/frozen_ball/v2_ball_exact_s61; TEMPLATE=$C/configs/bouncing_ball/unified_certification.json:4
else
  DEPLOYED=$R/frozen_pool/v2_pool_centerline_exactdecay_s61; TEMPLATE=$C/configs/pool_ball/fresh_test_v2.json:4
fi
cd $C
PYTHONPATH=src python3 -m nedm.contact_nrd.freeze --root $R --name state_$SYS --unified "${RUNS[@]}" \
  --reference deployed_v2=$DEPLOYED/best.pt --cohort $SYS=$TEMPLATE
F=$R/frozen_state_$SYS
FROZEN=(deployed_v2=$DEPLOYED); for r in "${RUNS[@]}"; do FROZEN+=("$F/$(basename $r)"); done
mkdir -p $R/fresh_state $R/results_state
if [[ $SYS == ball ]]; then
  j1=$(BALL_CODE=$C BALL_CAMPAIGN=$F/cohort_ball.json BALL_OUT=$R/fresh_state/ball_raw sbatch --parsable --export=ALL -p mi2101x -c 16 -t 01:00:00 -J st-fresh-ball $C/scripts/contact_nrd/cluster/ball_collect.sbatch)
  j2=$(BALL_CODE=$C sbatch --parsable --export=ALL --dependency=afterok:$j1 -A dannegrut -p mi2101x -N 1 -n 1 -c 8 -t 00:20:00 -J st-merge-ball \
    --wrap 'source $BALL_CODE/scripts/bouncing_ball/cluster/collection_env.sh && $NRD_PYTHON -m nedm.contact_nrd.ball_merge --primary '"$E"'/ball_span_v1_20260930/data --heldout '"$R"'/fresh_state/ball_raw --output-dir '"$R"'/fresh_state/ball_merged --margin-s .02')
  DATA=$R/fresh_state/ball_merged
else
  j0=$(POOL_CODE=$C POOL_CAMPAIGN=$F/cohort_pool.json POOL_DATA=$R/fresh_state/pool_raw POOL_PER=15 \
    sbatch --parsable --export=ALL -p mi2101x --array=0-7 -c 16 -t 01:30:00 -J st-fresh-pool $C/scripts/contact_nrd/cluster/pool_collect.sbatch)
  j2=$(POOL_CODE=$C POOL_CAMPAIGN=$F/cohort_pool.json POOL_DATA=$R/fresh_state/pool_raw POOL_PACK=1 \
    sbatch --parsable --export=ALL --dependency=afterok:$j0 -p mi2101x -c 8 -t 00:30:00 -J st-pack-pool $C/scripts/contact_nrd/cluster/pool_collect.sbatch)
  DATA=$R/fresh_state/pool_raw
fi
j3=$(CN_CODE=$C sbatch --parsable --export=ALL --dependency=afterok:$j2 -t 01:30:00 -J st-cert-$SYS $C/scripts/contact_nrd/cluster/py.sbatch \
  nedm.contact_nrd.certify_shared --system $SYS --data $DATA --unified "${FROZEN[@]}" --headline $HEAD --comparison deployed_v2 \
  --output $R/results_state/${SYS}_certification.json)
echo "fresh+pack ${j0:-$j1} ${j2} certify $j3"
