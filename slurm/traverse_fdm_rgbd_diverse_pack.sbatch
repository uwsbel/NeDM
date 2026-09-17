#!/bin/bash
# CPU-only data packaging; no training, inference, rendering, or physics.
# This cluster reports placeholder RealMemory=1; do not request --mem.
# Expected peak below 6 GiB; one mi2101x instance has roughly 60 GiB physical RAM.
#SBATCH -J fdm_diverse_pack
#SBATCH -A dannegrut
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH -o /work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909/logs/%x_%j.out
set -eo pipefail
source /work1/dannegrut/harry/nrd/env.sh
nrd_pychrono
: "${PACK_CODE_ROOT:?Set immutable preparation snapshot}"
: "${COLLECTION_CODE_ROOT:?Set immutable collection snapshot}"
: "${MANIFEST:?Set frozen full cases manifest}"
: "${COLLECTED_ROOT:?Set completed full cohort root}"
: "${PACK_OUT:?Set fresh matched-pair output}"
cd "$PACK_CODE_ROOT"
export PYTHONPATH="$PACK_CODE_ROOT/src:$PACK_CODE_ROOT/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$NRD_PYTHON" -u scripts/traverse_fdm_rgbd_diverse_pack_full.py full \
    --manifest "$MANIFEST" --collected "$COLLECTED_ROOT" \
    --source-snapshot-root "$COLLECTION_CODE_ROOT" --out "$PACK_OUT" --workers 8
