#!/bin/bash
# Stage 3: certification on the new fresh cohorts (fresh_v5) and targeting in Chrono for the picked seeds.
# Usage: launch_v5_stage3.sh <pool_pick_seed> <ball_pick_seed>   (validation-picked seeds of the 20 ms runs)
set -euo pipefail
R=/work1/dannegrut/harry/experiments/contact_nrd_20261002
C=$R/code_v5
O=$R/runs_v5
PP=$1; BP=$2
mkdir -p $R/results_v5
# ---- pool certification -----------------------------------------------------------
U=(); for t in 20ms 10ms; do for s in 61 62 63; do U+=("v5_${t}_s$s=$O/v5f_st_${t}_pool_s$s"); done; done
U+=("p0_s61=$R/runs_v4/v4_pool_p0_state_s61" "p0_s62=$R/runs_v4/v4_pool_p0_state_s62" "p0_s63=$R/runs_v4/v4_pool_p0_state_s63"
    "pair_features=$R/frozen_pool/v2_pool_centerline_exactdecay_s61")
EN=("v5_20ms_ens=$O/v5f_st_20ms_pool_s61,$O/v5f_st_20ms_pool_s62,$O/v5f_st_20ms_pool_s63"
    "v5_10ms_ens=$O/v5f_st_10ms_pool_s61,$O/v5f_st_10ms_pool_s62,$O/v5f_st_10ms_pool_s63"
    "p0_ens=$R/runs_v4/v4_pool_p0_state_s61,$R/runs_v4/v4_pool_p0_state_s62,$R/runs_v4/v4_pool_p0_state_s63"
    "pair_features_ens=$R/frozen_pool/v2_pool_centerline_exactdecay_s61,$R/frozen_pool/v2_pool_centerline_exactdecay_s62,$R/frozen_pool/v2_pool_centerline_exactdecay_s63")
a=$(CN_CODE=$C sbatch --parsable --export=ALL -t 03:00:00 -J v5-cert-pool -o $R/results_v5/cert_pool_%j.out $C/scripts/contact_nrd/cluster/py.sbatch \
  nedm.contact_nrd.certify_shared --system pool --data $R/fresh_v5/pool_raw --unified "${U[@]}" --ensemble "${EN[@]}" \
  --headline v5_20ms_s$PP --comparison pair_features --output $R/results_v5/pool_fresh_v5_certification.json)
# ---- ball certification -----------------------------------------------------------
U=(); for t in 20ms 10ms; do for s in 61 62 63; do U+=("v5_${t}_s$s=$O/v5f_st_${t}_ball_s$s"); done; done
U+=("state_k8_tf=$R/frozen_state_ball/v3_ball_a3_routed_k8_tf_s61" "state_k1=$R/runs_v4/v4_ball_b0_state_s61"
    "pair_features=$R/frozen_ball/v2_ball_exact_s61")
EN=("v5_20ms_ens=$O/v5f_st_20ms_ball_s61,$O/v5f_st_20ms_ball_s62,$O/v5f_st_20ms_ball_s63"
    "v5_10ms_ens=$O/v5f_st_10ms_ball_s61,$O/v5f_st_10ms_ball_s62,$O/v5f_st_10ms_ball_s63")
b=$(CN_CODE=$C sbatch --parsable --export=ALL -t 02:00:00 -J v5-cert-ball -o $R/results_v5/cert_ball_%j.out $C/scripts/contact_nrd/cluster/py.sbatch \
  nedm.contact_nrd.certify_shared --system ball --data $R/fresh_v5/ball_merged --unified "${U[@]}" --ensemble "${EN[@]}" \
  --headline v5_20ms_s$BP --comparison state_k8_tf --output $R/results_v5/ball_fresh_v5_certification.json)
# ---- targeting (same targets as every earlier targeting run) --------------------------------
c=$(CN_CODE=$C CN_SYSTEM=pool CN_CHECKPOINT=$O/v5f_st_20ms_pool_s$PP/best.pt CN_TRAIN=$R/data/pool_train CN_TARGETS=$R/fresh/pool_unified \
  CN_TCONFIG=$C/configs/contact_nrd/target_pool.json CN_TRUN=$R/results_v5/pool_targeting_20ms_s$PP \
  sbatch --parsable --export=ALL -J v5-tgt-pool -o $R/results_v5/target_pool_%j.out $C/scripts/contact_nrd/cluster/target.sbatch)
d=$(CN_CODE=$C CN_SYSTEM=ball CN_CHECKPOINT=$O/v5f_st_20ms_ball_s$BP/best.pt CN_TRAIN=$R/data/ball_train CN_TARGETS=$R/fresh/ball_unified \
  CN_TCONFIG=$C/configs/contact_nrd/target_ball.json CN_TRUN=$R/results_v5/ball_targeting_20ms_s$BP \
  sbatch --parsable --export=ALL -J v5-tgt-ball -o $R/results_v5/target_ball_%j.out $C/scripts/contact_nrd/cluster/target.sbatch)
echo "cert pool $a ball $b target pool $c ball $d"
