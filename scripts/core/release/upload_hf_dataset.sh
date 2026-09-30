#!/usr/bin/env bash
# Upload the staged NeDM release to the Hugging Face Hub, one commit per top-level
# folder so a dropped connection costs at most one folder and re-runs are cheap
# (Xet-backed uploads skip chunks the Hub already has).
#
#   hf auth login                      # once, with a write token (or export HF_TOKEN)
#   bash scripts/release/upload_hf_dataset.sh                 # everything
#   bash scripts/release/upload_hf_dataset.sh raw/tracked raw/arm README.md   # a subset (pilot)
#
# Env: HF_REPO_ID (default harryzhang1018/NeDM), STAGING (default artifacts/hf_release/NeDM).
set -euo pipefail

REPO_ID="${HF_REPO_ID:-harryzhang1018/NeDM}"
STAGING="${STAGING:-artifacts/hf_release/NeDM}"
HF_BIN="${HF_BIN:-hf}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

if [[ ! -f "$STAGING/release_manifest.json" ]]; then
  echo "no release_manifest.json under $STAGING -- run scripts/release/export_hf_dataset.py first" >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  TARGETS=("$@")
else
  TARGETS=(README.md release_manifest.json assets)
  for d in "$STAGING"/raw/* "$STAGING"/processed/*; do
    [[ -d "$d" ]] && TARGETS+=("${d#"$STAGING"/}")
  done
fi

LOG_DIR="$(dirname "$STAGING")"
for target in "${TARGETS[@]}"; do
  local_path="$STAGING/$target"
  if [[ ! -e "$local_path" ]]; then
    echo "skip $target (not staged)"; continue
  fi
  echo "== upload $target -> $REPO_ID:$target  ($(du -sh "$local_path" | cut -f1))"
  "$HF_BIN" upload "$REPO_ID" "$local_path" "$target" \
    --repo-type dataset \
    --commit-message "Add $target" \
    2>&1 | tee -a "$LOG_DIR/upload.log"
done
echo "done: https://huggingface.co/datasets/$REPO_ID"
