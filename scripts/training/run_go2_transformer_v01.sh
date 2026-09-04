#!/bin/bash
# Train the Go2 NRD model against the anchor config.
#
# CONSTRAINTS ENFORCED HERE RATHER THAN REMEMBERED.
#
# 1. python -u. Without it stdout is BLOCK-BUFFERED when redirected, so the log
#    stays empty while the GPU sits at 93% -- the run is training invisibly. That
#    is the wrong state to be in when the first few hundred steps are the thing
#    you are meant to report, and it cost one epoch to notice and relaunch.
#
# 2. The config is NOT parameterised. It is the HMMWV anchor with dataset paths
#    changed and nothing else -- verified key by key, flattened, nested lists
#    included. Parity is the point of this run, so a --lr or --batch-size flag
#    here would be an invitation to break the one property it exists to test. If
#    a hyperparameter needs to change, change the config and say why in the
#    commit, so the departure is on the record rather than in a shell history.
#
# 3. Selection stays rollout_sel at the 10 s horizon. One-step loss and rollout
#    error rank checkpoints differently -- the paper says so -- and switching
#    after seeing results is how a metric gets chosen for its answer.
set -euo pipefail

REPO="${NEDM_REPO:-/home/kyle/Documents/sbel/NeDM}"
PY="${NEDM_PY:-/home/kyle/miniconda3/envs/nedm-src/bin/python}"
CONFIG="${CONFIG:-configs/go2_transformer_v01_contact_mix25_onehot.json}"
OUT="${OUTPUT_DIR:-artifacts/training_runs/go2_transformer_v01_contact_mix25_onehot}"

cd "$REPO"
export PYTHONPATH=src

# Fail early if the caches are missing rather than 90 seconds into a load.
for d in artifacts/training_datasets/go2_flat_seq_v1 artifacts/training_datasets/go2_crm_seq_v1; do
    [ -f "$d/metadata.json" ] || { echo "missing processed cache: $d"; exit 1; }
done

exec "$PY" -u scripts/training/train_hmmwv_dynamics.py \
    --config "$CONFIG" --device cuda --output-dir "$OUT"
