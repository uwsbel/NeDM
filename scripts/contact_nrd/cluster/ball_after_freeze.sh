#!/bin/bash
# Ball: freeze, fresh cohort, merge/convert, certification, targeting, exploratory moved wall.
# Run on the login node: bash ball_after_freeze.sh <deploy_run_name> <comparison_reference_name>
set -euo pipefail
R=/work1/dannegrut/harry/experiments/contact_nrd_20261002
E=/work1/dannegrut/harry/experiments
DEPLOY=$1
COMPARISON=${2:-two_switch_k8}
C=$R/code_eval_ball
[[ -d $C ]] || cp -a $R/code_v1 $C
RUNS=()
for r in v2_ball_band_s61 v2_ball_gap2_s61 v2_ball_noband_s61 v2_ball_exact_s61 v2_ball_exact_s62 v2_ball_exact_s63 \
         v2_ball_exactdecay_s61 v2_ball_exactdecay_s62 v2_ball_exactdecay_s63; do RUNS+=("$R/runs_v2/$r"); done
RUNS+=("$R/runs/ball_world" "$R/runs/ball_pair")
REFS=(two_switch_k8=$E/ball_transformer_v2_20261001T214000Z/frozen/trained_ap_k8_w128_l4/best.pt
      one_switch_shared=$E/ball_transformer_v2_20261001T214000Z/frozen/direct_binary_mlp_shared5/best.pt
      transformer_analytic_timing=$E/ball_span_v1_20260930/runs/nrd_v2_certified/best.pt
      analytic_flight_mlp=$E/ball_precision_20261001/runs/certified_v2/best.pt)
cd $C
PYTHONPATH=src python3 -m nedm.contact_nrd.freeze --root $R --name ball --unified "${RUNS[@]}" --reference "${REFS[@]}" \
  --cohort ball=$C/configs/bouncing_ball/unified_certification.json:4
F=$R/frozen_ball
FROZEN=(); for r in "${RUNS[@]}"; do FROZEN+=("$F/$(basename $r)"); done
mkdir -p $R/fresh $R/results
j1=$(BALL_CODE=$C BALL_CAMPAIGN=$F/cohort_ball.json BALL_OUT=$R/fresh/ball_raw sbatch --parsable --export=ALL -c 24 -J cn-fresh-ball $C/scripts/contact_nrd/cluster/ball_collect.sbatch)
j2=$(BALL_CODE=$C sbatch --parsable --export=ALL --dependency=afterok:$j1 -A dannegrut -p mi3501x -N 1 -n 1 -c 8 -t 00:20:00 -J cn-merge-ball \
  --wrap 'source $BALL_CODE/scripts/bouncing_ball/cluster/collection_env.sh && $NRD_PYTHON -m nedm.bouncing_ball.merge --primary '"$E"'/ball_span_v1_20260930/data --heldout '"$R"'/fresh/ball_raw --output-dir '"$R"'/fresh/ball_merged --margin-s .02')
j3=$(CN_CODE=$C sbatch --parsable --export=ALL --dependency=afterok:$j1 -t 00:30:00 -J cn-conv-ball $C/scripts/contact_nrd/cluster/convert.sbatch \
  ball --source $E/ball_span_v1_20260930/data --heldout $R/fresh/ball_raw --output $R/fresh/ball_unified --workers 24)
j4=$(CN_CODE=$C sbatch --parsable --export=ALL --dependency=afterok:$j2 -t 01:00:00 -J cn-cert-ball $C/scripts/contact_nrd/cluster/py.sbatch \
  nedm.contact_nrd.certify_shared --system ball --data $R/fresh/ball_merged --unified "${FROZEN[@]}" --reference "${REFS[@]}" \
  --headline $DEPLOY --comparison $COMPARISON --output $R/results/ball_certification.json)
j5=$(CN_CODE=$C CN_SYSTEM=ball CN_CHECKPOINT=$F/$DEPLOY/best.pt CN_TRAIN=$R/data/ball_train CN_TARGETS=$R/fresh/ball_unified \
  CN_TCONFIG=$C/configs/contact_nrd/target_ball.json CN_TRUN=$R/results/ball_targeting \
  sbatch --parsable --export=ALL --dependency=afterok:$j3 -J cn-target-ball $C/scripts/contact_nrd/cluster/target.sbatch)
j6=$(CN_CODE=$C sbatch --parsable --export=ALL -t 00:30:00 -J cn-ood-ball $C/scripts/contact_nrd/cluster/py.sbatch \
  nedm.contact_nrd.ood_eval --data $R/ood/ball_wall45_unified --unified "${FROZEN[@]:3}" --reference "${REFS[@]}" --output $R/results/ball_ood_wall45.json)
echo "collect $j1 merge $j2 convert $j3 certify $j4 target $j5 ood $j6"
