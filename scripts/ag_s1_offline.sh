#!/bin/bash
# arena_gator_20260925 soil stage 1: offline within-group AUC of every soil ensemble on the soil evaluation files
# (dev fold + val rows, tiers 0-6) of the training arenas, LOCAL (RTX 5090, TF32 off via NVIDIA_TF32_OVERRIDE=0 so the
# scores are exact float32 as on the cluster), with scripts/ag_offline_auc.py (unchanged, fitted-group guard).
# Usage (from the worktree root): bash scripts/ag_s1_offline.sh A|B
#   A: task A models (deploy M1a M1b M2 M3a M3b A3; holdout M1 M2 M3 A3 LC545 LC272 LOAO*) on f104 / g203 / g228
#   B: task B models (G and H deploy + holdout; H = M1a / M1 holdout when its file equals M1's) on the Gator f104
#      rows and on the HMMWV f104 rows restricted to the Gator-validated ids (each model on both vehicles' rows)
set -eo pipefail
K3=artifacts/traverse/arena_gator_20260925; G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
PY=/home/harry/miniconda3/envs/nedm/bin/python
export PYTHONPATH=src:scripts NVIDIA_TF32_OVERRIDE=0
mkdir -p $K3/e5/eval_soil $K3/e5/train/soil_s1/offline_soil $K3/e5/offline
rsync -a amd:$G3/e5/eval_soil/ $K3/e5/eval_soil/
rsync -a amd:$G3/e5/train/soil_s1/offline_soil/ $K3/e5/train/soil_s1/offline_soil/
D=$K3/e5/deploy; O=$K3/e5/train/soil_s1/offline_soil; EV=$K3/e5/eval_soil
case $1 in
  A)
    M=()
    for m in M1a M1b M2 M3a M3b A3; do M+=(--model "${m}_deploy=$D/${m}_soil/${m}_soil_deploy_s*.pt"); done
    for m in M1 M2 M3 A3 LC545 LC272 LOAO1_g203 LOAO1_g228 LOAO2_f104_g203 LOAO2_f104_g228 LOAO2_g203_g228; do
      ls $O/${m}_soil_holdout_s*.pt >/dev/null 2>&1 && M+=(--model "${m}_holdout=$O/${m}_soil_holdout_s*.pt"); done
    $PY -u scripts/ag_offline_auc.py "${M[@]}" --eval f104=$EV/EV_f104_hmmwv_crm.npz --eval g203=$EV/EV_g203_hmmwv_crm.npz \
      --eval g228=$EV/EV_g228_hmmwv_crm.npz --check-trainer --out $K3/e5/offline/soil_s1_A_auc.json
    $PY scripts/ag_s1_auc_table.py $K3/e5/offline/soil_s1_A_auc.json --md $K3/e5/offline/soil_s1_A_auc.md;;
  B)
    # H = the M1a soil ensemble when the H file is byte-identical to the M1 file (e5/deploy/H_soil/README.md)
    HD=$D/H_soil; ls $HD/*_s*.pt >/dev/null 2>&1 || HD=$D/M1a_soil
    HH="$O/H_soil_holdout_s*.pt"; ls $O/H_soil_holdout_s*.pt >/dev/null 2>&1 || HH="$O/M1_soil_holdout_s*.pt"
    M=(--model "G_deploy=$D/G_soil/G_soil_deploy_s*.pt" --model "G_holdout=$O/G_soil_holdout_s*.pt"
       --model "H_deploy=$HD/*_deploy_s*.pt" --model "H_holdout=$HH")
    $PY -u scripts/ag_offline_auc.py "${M[@]}" --eval f104_gator_rows=$EV/EV_f104_gator_crm.npz --eval f104_hmmwv_rows=$EV/EV_f104_hmmwv_gatorids_crm.npz \
      --check-trainer --out $K3/e5/offline/soil_s1_B_auc.json
    $PY scripts/ag_s1_auc_table.py $K3/e5/offline/soil_s1_B_auc.json --md $K3/e5/offline/soil_s1_B_auc.md;;
  *) echo "usage: $0 A|B"; exit 2;;
esac
