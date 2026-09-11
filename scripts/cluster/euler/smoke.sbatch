#!/bin/bash
# One CRM episode with the base policy, on a GPU node. Run this after any change
# to the repo, the Chrono tree, or the environment, BEFORE spending a node on a
# full 80-episode pass.
#
#   sbatch scripts/cluster/euler/smoke.sbatch
#
# Prints the environment it actually got (which node, which GPU, which pychrono,
# which parsers module) before running, because every failure this catches is a
# failure of one of those to be what the script assumed.
#
#SBATCH -J kyle-crmsmoke
#SBATCH -p sbel
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -o /srv/home/kasha2/logs/%x-%j.out
#SBATCH -e /srv/home/kasha2/logs/%x-%j.err

set -o pipefail
module load gcc/12.2.0 nvidia/cuda/12.9.1
source /srv/home/kasha2/nedm/env-scoring.sh

echo "== host $(hostname)  start $(date) =="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo "== chrono   : $NEDM_CHRONO"
echo "== index    : $NEDM_INDEX"
python - <<'PY'
import os, sys
import pychrono, pychrono.parsers as p, pychrono.fsi, numpy, torch
from nedm.quadruped.gpu import gpu_backend
print("pychrono   :", pychrono.__file__)
print("parsers    :", p.__file__, "-> ChParserURDF", hasattr(p, "ChParserURDF"))
print("numpy      :", numpy.__version__, numpy.__file__)
print("torch      :", torch.__version__)
print("gpu_backend:", gpu_backend())
PY
rc=$?
[ $rc -ne 0 ] && { echo "FATAL: import check failed"; exit $rc; }

# One episode: shard 0 of 80 from the val index.
python "$NEDM_REPO/scripts/cluster/mkindex.py" \
  --index "$NEDM_INDEX" --out "$NEDM_ROOT/idx/smoke1.json" --limit 1

t0=$(date +%s)
cd "$NEDM_REPO"
python scripts/evaluation/score_crm_tracking.py \
  --policy "$NEDM_CKPTS/go2_cts_150k.pt" \
  --index  "$NEDM_ROOT/idx/smoke1.json" \
  --out    "$NEDM_OUT/smoke1_euler.json" \
  --concurrency 1
rc=$?
t1=$(date +%s)
echo "== rc=$rc  one episode in $((t1 - t0))s  end $(date) =="
cat "$NEDM_OUT/smoke1_euler.json" 2>/dev/null
exit $rc
