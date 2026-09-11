#!/bin/bash
# One CRM episode on devel (0.1x, 30 min cap). Run this after ANY change to the
# repo, the environment, or the Chrono build, before spending mi2101x time.
#
#SBATCH -J kyle-crm-smoke
#SBATCH -p devel
#SBATCH -N 1
#SBATCH -t 00:28:00
#SBATCH -o /work1/dannegrut/kyle/logs/%x-%j.out
#SBATCH -e /work1/dannegrut/kyle/logs/%x-%j.err

set -o pipefail
source /work1/dannegrut/kyle/env.sh
module purge
module load gnu12 rocm
source /work1/dannegrut/kyle/nedm/env-scoring.sh

echo "== host $(hostname)   start $(date) =="
echo "== chrono tree: $NEDM_CHRONO"
echo "-- GPU driver nodes (what the backend guard tests) --"
ls -l /dev/kfd /dev/nvidiactl 2>&1
rocm-smi --showproductname 2>&1 | grep -i "card series" | head -2

echo "-- python / pychrono / torch --"
python - <<'PY'
import sys
print("python", sys.version.split()[0])
import numpy; print("numpy", numpy.__version__)
import torch;  print("torch", torch.__version__)
import pychrono; print("pychrono", pychrono.__file__)
import pychrono.vehicle; print("pychrono.vehicle ok")
import pychrono.fsi;     print("pychrono.fsi ok")
import pychrono.parsers as P; print("pychrono.parsers ok")
print("  ChParserURDF present:", hasattr(P, "ChParserURDF"))
from nedm.quadruped.gpu import gpu_backend, require_gpu_backend
print("gpu_backend() ->", gpu_backend(), "| require ->", require_gpu_backend())
PY
rc=$?; echo "-- import stage rc=$rc --"; [ $rc -ne 0 ] && exit $rc

echo "== one-episode index =="
python $NEDM_ROOT/mkindex.py --index $NEDM_INDEX --out $NEDM_ROOT/idx/smoke1.json --limit 1

echo "== scoring 1 episode, base policy, concurrency 1 =="
cd $NEDM_REPO
t0=$(date +%s)
python scripts/evaluation/score_crm_tracking.py \
  --policy $NEDM_CKPTS/go2_cts_150k.pt \
  --index  $NEDM_ROOT/idx/smoke1.json \
  --out    $NEDM_OUT/smoke1.json \
  --concurrency 1
rc=$?
t1=$(date +%s)
# Per-episode wall at concurrency 1. This is the number that sizes -t for the
# array and anchors the concurrency sweep.
echo "== scorer rc=$rc  ONE EPISODE WALL: $((t1-t0)) s  end $(date) =="
cat $NEDM_OUT/smoke1.json 2>/dev/null
exit $rc
