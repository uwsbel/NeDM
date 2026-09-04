#!/bin/bash
# Build the two Go2 processed caches the training config expects.
#
# THE CONSTRAINTS BELOW ARE ENFORCED HERE RATHER THAN REMEMBERED.
#
# 1. ROOT ORDER MUST MATCH between the validator and preprocess. preprocess does
#    not sort or shuffle episodes -- it appends them per root, per index, in
#    order (preprocess.py:380-396), and the split partition preserves relative
#    order. _select_rollout_episodes then pops(0) within a family, so it is
#    ORDER-SENSITIVE, not just membership-sensitive. Root order is the only
#    ordering degree of freedom left, so the same order in both places makes the
#    gate's predicted selection the trainer's actual selection.
#
# 2. --max-episodes-per-split IS NEVER PASSED. It truncates each split with a
#    bare slice (preprocess.py:410) that the validator cannot see, which would
#    silently break the equality in (1). Its absence is enforced by this script
#    not offering it, because "remember not to pass a flag" is not a control.
#
# Root order is defined ONCE, below, and reused by both commands.
set -euo pipefail

PY="${NEDM_PY:-/home/kyle/miniconda3/envs/nedm-src/bin/python}"
REPO="${NEDM_REPO:-/home/kyle/Documents/sbel/NeDM}"
MERGED="${NEDM_MERGED_ROOT:-$HOME/sbel-artifacts/datasets/go2_merged}"
DORM="${NEDM_DORM_ROOT:-}"        # dorm-pc's crm root, if it has been synced here
OUT="${NEDM_CACHE_DIR:-$REPO/artifacts/training_datasets}"

# --- the single source of root order ---------------------------------------
FLAT_ROOTS=("$MERGED/flat")
CRM_ROOTS=("$MERGED/crm")
[ -n "$DORM" ] && CRM_ROOTS+=("$DORM")

cd "$REPO"
export PYTHONPATH=src

echo "=== GATE (same root order as preprocess) ==="
$PY scripts/collection/validate_go2_dataset.py \
    "${FLAT_ROOTS[@]/#/--dataset-root }" "${CRM_ROOTS[@]/#/--dataset-root }" \
    --rollout-episodes 12 || { echo "GATE FAILED -- not preprocessing"; exit 1; }

echo "=== PREPROCESS flat ==="
$PY -m nedm.training.preprocess \
    --dataset-root "${FLAT_ROOTS[@]}" \
    --output-dir "$OUT/go2_flat_seq_v1" \
    --state-field-preset quadruped_contact \
    --action-fields cmd_vx_mps cmd_vy_mps cmd_wz_radps

echo "=== PREPROCESS crm ==="
$PY -m nedm.training.preprocess \
    --dataset-root "${CRM_ROOTS[@]}" \
    --output-dir "$OUT/go2_crm_seq_v1" \
    --state-field-preset quadruped_contact \
    --action-fields cmd_vx_mps cmd_vy_mps cmd_wz_radps

echo "=== POST-PREPROCESS COVERAGE (measured, not predicted) ==="
# The gate validates the RAW index; the trainer selects from the PROCESSED
# cache. The prediction is exact under (1) and (2) above -- so a disagreement
# here MEANS something broke, rather than being shruggable as drift.
$PY scripts/collection/check_processed_coverage.py \
    "$OUT/go2_flat_seq_v1" "$OUT/go2_crm_seq_v1" --rollout-episodes 12
