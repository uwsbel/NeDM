#!/usr/bin/env bash
# =====================================================================================================================
# scripts/contact_nrd/arch_ablation/reproduce.sh: train and score the architecture ablation of one contact NRD case
# (contact_nrd/arch_ablation/README.md). On a cluster: <case>/train.sbatch, one per case.
#
#   scripts/contact_nrd/arch_ablation/reproduce.sh <case> [step ...]
#
#   case   bouncing_ball | pool | so101_push_t | g1_locomotion
#   step   download | 3m | 1t | 1m | test      (default: all, in this order)
#
# The 3T is the released contact NRD checkpoint and is not trained. 3M, 1T and 1M are trained by
# python -m nedm.contact_nrd.arch_ablation.train with the 3T's config; a step whose run is complete is skipped, and an
# interrupted one continues from its last check. Everything runs from this repository alone (src/nedm); no simulator.
#
# Data, runs and results are kept in artifacts/ of this checkout (not in git), as the contact NRD's own:
#   ARTIFACTS     default <repo>/artifacts
#   DATA_ROOT     the released contact NRD files, <DATA_ROOT>/<case>/{train,test,models}/
#                 default $ARTIFACTS/contact_nrd (step "download" fills it from the Hub at the pinned
#                 revision; point it at an existing copy to skip the download)
#   runs          $ARTIFACTS/contact_nrd/arch_ablation/runs/<case>_<arch>_s<seed><RUN_TAG>/
#   scores        $ARTIFACTS/contact_nrd/arch_ablation/<case>/test_scores_nedm<RUN_TAG>.json
#
# Settings (environment variables)
#   SEED          61
#   PY            python (an environment with torch and numpy: environment.nedm.yml or requirements.txt)
#   DEVICE        cuda
#   TRAIN_DATA    <DATA_ROOT>/<case>/train     the training directory
#   MODEL_3T      <DATA_ROOT>/<case>/models/seed<SEED>.pt
#   ONE_UPDATES   co-training updates of 1T and 1M (default: the config's 120000)
#   ONE_ARGS      more arguments for the 1T / 1M runs, e.g. "--width 256 --parameters 4829696"
#   TRAIN_ARGS    more arguments for every training run, e.g. "--smoke" (a pipeline check of a few updates)
#   RUN_TAG       suffix of the run folders and the score file; give one whenever a setting above differs from an
#                 earlier run of the same case, since an existing run folder is continued, not started again
#                 (default: empty, or _smoke with --smoke)
#
# The original SO101 runs trained on the larger push-T corpus, compared with its 3T, and gave the one-network models
# the 3T's total of 150000 updates:
#   TRAIN_DATA=<...>/so101_push_rl/nrd_data/train MODEL_3T=<...>/so101_push_rl/nrd/rot_sc8_s61/best.pt \
#   ONE_UPDATES=150000 scripts/contact_nrd/arch_ablation/reproduce.sh so101_push_t 3m 1t 1m test
# =====================================================================================================================
set -euo pipefail

CASE=${1:?usage: reproduce.sh bouncing_ball|pool|so101_push_t|g1_locomotion [download|3m|1t|1m|test ...]}
shift
STEPS=("$@")
[ ${#STEPS[@]} -gt 0 ] || STEPS=(download 3m 1t 1m test)

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ARTIFACTS=${ARTIFACTS:-$REPO/artifacts}
DATA_ROOT=${DATA_ROOT:-$ARTIFACTS/contact_nrd}
SEED=${SEED:-61}
PY=${PY:-python}
DEVICE=${DEVICE:-cuda}
TRAIN_DATA=${TRAIN_DATA:-$DATA_ROOT/$CASE/train}
MODEL_3T=${MODEL_3T:-$DATA_ROOT/$CASE/models/seed$SEED.pt}
RUNS=$ARTIFACTS/contact_nrd/arch_ablation/runs
RUN_TAG=${RUN_TAG-$([[ "${TRAIN_ARGS:-}" == *--smoke* ]] && echo _smoke || true)}
CONFIG=$REPO/configs/contact_nrd/train.json
[ "$CASE" = g1_locomotion ] && CONFIG=$REPO/configs/contact_nrd/train_g1_locomotion.json

cd "$REPO"
export PYTHONPATH=$REPO/src${PYTHONPATH:+:$PYTHONPATH}

for step in "${STEPS[@]}"; do
  run=$RUNS/${CASE}_${step}_s$SEED$RUN_TAG
  case $step in
    download)
      [ -f "$TRAIN_DATA/unified_data.npz" ] && [ -f "$DATA_ROOT/$CASE/test/unified_data.npz" ] && [ -f "$MODEL_3T" ] \
        && { echo "download: files in place"; continue; }
      [ "$(basename "$DATA_ROOT")" = contact_nrd ] || { echo "download: the last folder of DATA_ROOT must be contact_nrd"; exit 2; }
      "$PY" -m nedm.contact_nrd.download --case "$CASE" --part train test models --dest "$(dirname "$DATA_ROOT")" ;;
    3m)
      # shellcheck disable=SC2086
      "$PY" -m nedm.contact_nrd.arch_ablation.train --arch 3m --data "$TRAIN_DATA" --output "$run" --config "$CONFIG" \
        --seed "$SEED" --device "$DEVICE" ${TRAIN_ARGS:-} ;;
    1t|1m)
      # shellcheck disable=SC2086
      "$PY" -m nedm.contact_nrd.arch_ablation.train --arch "$step" --data "$TRAIN_DATA" --output "$run" --config "$CONFIG" \
        --seed "$SEED" --device "$DEVICE" ${ONE_UPDATES:+--updates $ONE_UPDATES} ${ONE_ARGS:-} ${TRAIN_ARGS:-} ;;
    test)
      models=("3t=$MODEL_3T")
      for arch in 3m 1t 1m; do
        [ -f "$RUNS/${CASE}_${arch}_s$SEED$RUN_TAG/complete.json" ] && models+=("$arch=$RUNS/${CASE}_${arch}_s$SEED$RUN_TAG")
      done
      "$PY" -m nedm.contact_nrd.arch_ablation.test --data "$DATA_ROOT/$CASE/test" --model "${models[@]}" --device "$DEVICE" \
        --output "$ARTIFACTS/contact_nrd/arch_ablation/$CASE/test_scores_nedm$RUN_TAG.json" ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
done
