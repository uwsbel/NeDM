#!/bin/bash
# Staged current-runtime observations/headless physics; no training.
#SBATCH -J fdm_diverse_collect
#SBATCH -A dannegrut
#SBATCH -p mi2104x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -t 00:30:00
#SBATCH -o /work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909/logs/%x_%j.out
set -eo pipefail
source /work1/dannegrut/harry/nrd/env.sh
nrd_pychrono
nrd_use_lavapipe
: "${CODE_ROOT:?Set immutable CODE_ROOT}"
: "${MANIFEST:?Set frozen cases manifest}"
: "${OUT_ROOT:?Set new isolated output root}"
cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Lavapipe has its own thread pool; cap it independently from physics workers.
export LP_NUM_THREADS=${LP_NUM_THREADS:-4}
STAGE=${STAGE:-all}
WORKERS=${WORKERS:-8}
OBSERVATION_WORKERS=${OBSERVATION_WORKERS:-2}
SPLITS=${SPLITS:-"train val"}
read -r -a split_args <<< "$SPLITS"
args=(--manifest "$MANIFEST" --out "$OUT_ROOT" --chrono-data "$CHRONO_BUILD/data"
      --stage "$STAGE" --workers "$WORKERS" --observation-workers "$OBSERVATION_WORKERS"
      --splits "${split_args[@]}")
if [ -n "${ROUTE_INDICES:-}" ]; then
    read -r -a route_args <<< "$ROUTE_INDICES"
    args+=(--route-indices "${route_args[@]}")
fi
if [ -n "${SCENE_IDS:-}" ]; then
    read -r -a scene_args <<< "$SCENE_IDS"
    args+=(--scene-ids "${scene_args[@]}")
fi
if [ -n "${HORIZON_S:-}" ]; then args+=(--horizon-s "$HORIZON_S"); fi
if [ "${RENDER_PARITY:-0}" = 1 ]; then args+=(--render-parity); fi
if [ "${RICH_TELEMETRY:-1}" = 0 ]; then args+=(--no-rich-telemetry); fi
if [ "${EVALUATION_ONLY:-0}" = 1 ]; then args+=(--evaluation-only); fi
"$NRD_PYTHON" -u scripts/traverse_fdm_rgbd_diverse_batch.py "${args[@]}"
