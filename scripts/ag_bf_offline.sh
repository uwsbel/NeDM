#!/bin/bash
# arena_gator_20260925 task B stage 2 (Bfull): offline within-group AUC of the all-tier soil planners G_full / H_full next to
# the stage-1 (tiers 0-6) G / H, LOCAL (RTX 5090, TF32 off via NVIDIA_TF32_OVERRIDE=0, as scripts/ag_s1_offline.sh B), with
# scripts/ag_offline_auc.py (unchanged, fitted-group guard). Every model is scored on the same rows:
#   f104_gator_all / f104_hmmwv_all   dev fold + val (+ test, never read) rows of ALL tiers (0-12): the Gator drives and
#                                     the HMMWV drives of the same 15,235 ids (e5/eval_soil_bf, written by ag_bf_subsets.sh)
#   f104_gator_t06 / f104_hmmwv_t06   the stage-1 evaluation files (tiers 0-6 only; e5/eval_soil), for the S1 numbers
# Models: G_full / H_full deploy (e5/deploy/*_full_soil) and holdout (e5/train/soil_bf/offline_soil_bf); stage-1 G deploy
# (e5/deploy/G_soil), G holdout (e5/train/soil_s1/offline_soil), H = the M1a soil ensemble (deploy) and the M1 soil
# holdout run (H's data file is byte-identical to M1's, e5/deploy/H_soil/README.md).
#   bash scripts/ag_bf_offline.sh      -> e5/offline/soil_bf_B_auc.{json,md}
set -eo pipefail
cd "$(dirname "$0")/.."
K3=artifacts/traverse/arena_gator_20260925; G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
PY=/home/harry/miniconda3/envs/nedm/bin/python
export PYTHONPATH=src:scripts NVIDIA_TF32_OVERRIDE=0
mkdir -p $K3/e5/eval_soil_bf $K3/e5/train/soil_bf/offline_soil_bf $K3/e5/offline
rsync -a amd:$G3/e5/eval_soil_bf/ $K3/e5/eval_soil_bf/
rsync -a amd:$G3/e5/train/soil_bf/offline_soil_bf/ $K3/e5/train/soil_bf/offline_soil_bf/
( cd $K3/e5/eval_soil_bf && sha256sum *.npz > SHA256SUMS.local )
ssh amd "cd $G3/e5/eval_soil_bf && sha256sum *.npz" > /tmp/ag_bf_ev_remote.sha
diff -q $K3/e5/eval_soil_bf/SHA256SUMS.local /tmp/ag_bf_ev_remote.sha
D=$K3/e5/deploy; O=$K3/e5/train/soil_bf/offline_soil_bf; O1=$K3/e5/train/soil_s1/offline_soil
M=(--model "G_full_deploy=$D/G_full_soil/G_full_soil_deploy_s*.pt" --model "G_full_holdout=$O/G_full_soil_holdout_s*.pt"
   --model "H_full_deploy=$D/H_full_soil/H_full_soil_deploy_s*.pt" --model "H_full_holdout=$O/H_full_soil_holdout_s*.pt"
   --model "G_deploy=$D/G_soil/G_soil_deploy_s*.pt" --model "G_holdout=$O1/G_soil_holdout_s*.pt"
   --model "H_deploy=$D/M1a_soil/M1a_soil_deploy_s*.pt" --model "H_holdout=$O1/M1_soil_holdout_s*.pt")
$PY -u scripts/ag_offline_auc.py "${M[@]}" \
  --eval f104_gator_all=$K3/e5/eval_soil_bf/EV_f104_gator_full_crm.npz --eval f104_hmmwv_all=$K3/e5/eval_soil_bf/EV_f104_hmmwv_gatorids_full_crm.npz \
  --eval f104_gator_t06=$K3/e5/eval_soil/EV_f104_gator_crm.npz --eval f104_hmmwv_t06=$K3/e5/eval_soil/EV_f104_hmmwv_gatorids_crm.npz \
  --check-trainer --out $K3/e5/offline/soil_bf_B_auc.json
{ echo "## auto (holdout-mode models: dev fold + val; deploy-mode models: val)"; $PY scripts/ag_s1_auc_table.py $K3/e5/offline/soil_bf_B_auc.json;
  echo; echo "## val groups only (every model)"; $PY scripts/ag_s1_auc_table.py $K3/e5/offline/soil_bf_B_auc.json --set val; } > $K3/e5/offline/soil_bf_B_auc.md
cat $K3/e5/offline/soil_bf_B_auc.md
