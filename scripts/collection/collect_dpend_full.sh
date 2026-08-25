#!/usr/bin/env bash
# Full-tier collection for NRD Study 1 (plan section 6.1): 1,000 x 10 s
# double-pendulum + camera episodes, then the merged processed cache.
#
# The new episodes use prefix "dpendf" and a fresh seed so their ids never
# collide with the 200-episode pilot (prefix "dpend", seed 20260825); the cache
# is built from BOTH roots, giving ~586k transitions (~3.2 h of simulated
# motion). Splits stay hash-assigned per episode id (~15% val).
#
# Cost on the RTX 4090 box: ~5 min collection (RTF ~50x incl. rendering),
# ~21 GB raw frames + ~28 GB cache. Run from the repo root in the nedm env:
#   bash scripts/collection/collect_dpend_full.sh
set -euo pipefail
export PYTHONPATH=src

FULL_ROOT=artifacts/datasets/dpend_full_1000
PILOT_ROOT=artifacts/datasets/dpend_pilot_200
CACHE_DIR=artifacts/training_datasets/dpend_full_seq16_v1

python -m nedm.double_pendulum_data \
  --episodes 1000 --max-steps 500 --seed 20260826 \
  --episode-prefix dpendf --dataset-name dpend_full_1000 \
  --output-root "$FULL_ROOT"

# Stored-data gates on the new episodes (frame counts, timestamps, and the
# tip-pixel projection test; the live mechanism gates already ran for the pilot).
python scripts/collection/validate_dpend_dataset.py \
  --dataset-root "$FULL_ROOT" --skip-live

# Merged cache: full + pilot. Frames are written disk-backed by default.
python -m nedm.training.preprocess \
  --dataset-root "$FULL_ROOT" "$PILOT_ROOT" \
  --output-dir "$CACHE_DIR" \
  --state-fields cos_q1 sin_q1 cos_q2 sin_q2 omega1_radps omega2_radps \
  --action-fields action_elbow \
  --rollout-fields tip_x_m tip_z_m \
  --frames

python - <<'EOF'
import json
meta = json.load(open("artifacts/training_datasets/dpend_full_seq16_v1/metadata.json"))
tr, va = meta["splits"]["train"]["transition_count"], meta["splits"]["val"]["transition_count"]
print(f"cache ready: {tr} train / {va} val transitions "
      f"({(tr + va) * meta['dt_s'] / 60:.1f} min of simulated motion)")
EOF
