#!/bin/bash
# E6b offline within-group AUC on the 8 unseen test arenas (evaluation-only designed-route rows, e4/evalonly/<a>_rigid_test)
# for every rigid model of task A (deploy ensembles) and the holdout-mode learning-curve / leave-one-arena-out runs.
# scripts/ag_score_offline.py, TF32 off. Output $K3/e6/offline/<tag>.json. Usage: bash scripts/ag_e6b_offline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
K3=artifacts/traverse/arena_gator_20260925
PY=/home/harry/miniconda3/envs/nedm/bin/python
export PYTHONPATH=src:scripts OMP_NUM_THREADS=2; unset NEDM_VEHICLE
O=$K3/e6/offline; mkdir -p $O/logs
DS=$(ls $K3/e4/evalonly/g2??_rigid_test/ci_g2??_hmmwv_rigid_evalonly.npz)
[[ $(echo "$DS" | wc -l) == 8 ]] || { echo "expected 8 evaluation-only files"; exit 2; }
run() { local tag=$1 glob=$2; [[ -e $O/$tag.json ]] && { echo "skip $tag"; return; }
  $PY scripts/ag_score_offline.py --models "$glob" --tag $tag --ds $DS --world rigid --save-logits --out $O/$tag.json > $O/logs/$tag.log 2>&1 && echo "OK $tag" || echo "FAIL $tag"; }
for m in M1a M1b M2 M3a M3b A3 G; do run ${m}_deploy "$K3/e5/deploy/$m/${m}_rigid_deploy_s*.pt"; done
for t in M1 M2 M3 A3 LC272 LC545 LOAO1_g203 LOAO1_g228 LOAO2_f104_g203 LOAO2_f104_g228 LOAO2_g203_g228; do
  run ${t}_holdout "$K3/e5/train/offline_rigid/${t}_rigid_holdout_s*.pt"; done
echo "offline done"
