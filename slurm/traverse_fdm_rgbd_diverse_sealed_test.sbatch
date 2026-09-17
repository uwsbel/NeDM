#!/bin/bash
# Explicit final freeze required. No optimizer, no checkpoint selection.
#SBATCH -J fdm_test_offline
#SBATCH -A dannegrut
#SBATCH -p mi2101x
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o /work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909/logs/%x_%j.out
set -eo pipefail
source /etc/profile
module load pytorch/2.10.0
source /work1/dannegrut/harry/venvs/nedm/bin/activate
: "${EVAL_SNAPSHOT:?Set the frozen evaluator snapshot}"
: "${FINAL_FREEZE:?Set the explicitly authorized final freeze}"
: "${FINAL_FREEZE_SHA256:?Set the exact freeze SHA256}"
: "${OFFLINE_FREEZE:?Set the frozen offline analysis supplement}"
: "${OFFLINE_FREEZE_SHA256:?Set the exact offline supplement SHA256}"
: "${TEST_PACK_OUT:?Set a fresh test pack output}"
: "${TEST_REPORT_OUT:?Set a fresh test report output}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
export PYTHONPATH="$EVAL_SNAPSHOT/src:$EVAL_SNAPSHOT/scripts:${PYTHONPATH:-}"
cd "$EVAL_SNAPSHOT"
python3.12 -u scripts/evaluate_traverse_fdm_diverse_sealed_test.py \
    --freeze "$FINAL_FREEZE" --freeze-sha256 "$FINAL_FREEZE_SHA256" \
    --offline-freeze "$OFFLINE_FREEZE" --offline-freeze-sha256 "$OFFLINE_FREEZE_SHA256" \
    --pack-out "$TEST_PACK_OUT" --out "$TEST_REPORT_OUT" \
    --stage "${EVAL_STAGE:-all}" --device "${EVAL_DEVICE:-cpu}" \
    --threads 8 --workers 8 --batch 32
