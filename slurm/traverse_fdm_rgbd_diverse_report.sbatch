#!/bin/bash
# CPU-only saved prediction summaries. No model inference or training.
#SBATCH -J fdm_diverse_report
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
: "${REPORT_CODE_ROOT:?Set immutable reporting snapshot}"
: "${PACK_PAIR:?Set complete matched pack pair}"
: "${RUNS_H60:?Set completed H60 four-arm suite}"
: "${RUNS_H20:?Set completed H20 four-arm suite}"
: "${RAW_ROOT:?Set train/validation raw collection root}"
: "${REPORT_OUT:?Set fresh report output}"
cd "$REPORT_CODE_ROOT"
export PYTHONPATH="$REPORT_CODE_ROOT/src:$REPORT_CODE_ROOT/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$NRD_PYTHON" -u scripts/traverse_fdm_rgbd_diverse_report_pair.py run \
    --pair "$PACK_PAIR" --runs-h60 "$RUNS_H60" --runs-h20 "$RUNS_H20" \
    --raw-root "$RAW_ROOT" --out "$REPORT_OUT"
