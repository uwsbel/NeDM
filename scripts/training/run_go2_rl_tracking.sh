#!/bin/bash
# PPO velocity-command tracking for the Go2 inside the frozen NRD model.
#
# CONSTRAINTS ENFORCED HERE RATHER THAN REMEMBERED.
#
# 1. python -u. Without it stdout is BLOCK-BUFFERED when redirected, so the log
#    stays empty while the GPU sits pinned -- the run trains invisibly and you
#    cannot tell a healthy start from a hang. That cost an epoch on the dynamics
#    run; it is not going to cost an RL run too.
#
# 2. --robot go2 IS NOT OPTIONAL AND IS NOT DEFAULTED HERE BY ACCIDENT. The
#    shared entry point defaults to hmmwv, and every robot-specific number --
#    action bounds, episode length, termination, reward sigmas -- comes from the
#    ROBOTS preset that flag selects. Omitting it silently trains an HMMWV
#    config against a Go2 checkpoint, which loads and runs.
#
# 3. THE REWARD AND TERMINATION SCALES LIVE IN go2_tracking_env.py, NOT HERE.
#    This script exposes no --position-weight or --yaw-weight, because those
#    were chosen from measured error distributions and a shell-history override
#    would leave no record of the departure. Change the env module and say why
#    in the commit.
#
# 4. Terrain-mixed by default. The checkpoint is terrain-conditioned
#    (terrains: flat, crm) and the 40-reference set carries per-reference
#    domains, so each env samples references only from its own terrain. A
#    single-terrain run is possible with TERRAIN=flat, but the default matches
#    the HMMWV anchor's generalist setup.
set -euo pipefail

REPO="${NEDM_REPO:-/home/kyle/Documents/sbel/NeDM}"
PY="${NEDM_PY:-/home/kyle/miniconda3/envs/nedm-src/bin/python}"
RUN_NAME="${RUN_NAME:-go2_nn_tracking_v01}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERS="${MAX_ITERS:-2000}"
SEED="${SEED:-1}"
TERRAIN="${TERRAIN:-}"
TERRAIN_MIX="${TERRAIN_MIX:-flat:1,crm:1}"

cd "$REPO"
export PYTHONPATH=src

CKPT="artifacts/training_runs/go2_transformer_v01_contact_mix25_onehot/checkpoints/best_val.pt"
REFS="artifacts/rl_references/go2_flat_crm_ref40.npz"
# Fail on a missing input rather than 60 seconds into a load.
for f in "$CKPT" "$REFS"; do
    [ -f "$f" ] || { echo "missing required input: $f"; exit 1; }
done

TERRAIN_ARGS=(--terrain-mix "$TERRAIN_MIX")
[ -n "$TERRAIN" ] && TERRAIN_ARGS=(--terrain "$TERRAIN")

exec "$PY" -u scripts/training/train_hmmwv_rl_tracking.py \
    --robot go2 \
    --run-name "$RUN_NAME" \
    --num-envs "$NUM_ENVS" \
    --max-iterations "$MAX_ITERS" \
    --seed "$SEED" \
    --dynamics-checkpoint "$CKPT" \
    --reference-path "$REFS" \
    "${TERRAIN_ARGS[@]}" \
    "$@"
